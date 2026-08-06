use crate::{
    update::update_edge, AllEdges, EdgeVisitor, GlobalIndexBatch, PivotCandidateProposer, TreeTCI2,
};
use anyhow::{ensure, Result};
use tensor4all_core::CommonScalar;
use tensor4all_tcicore::{MatrixLuciScalar as Scalar, RrLUOptions};

/// MVP optimization options for TreeTCI.
///
/// Controls convergence criteria, iteration limits, and bond dimension caps
/// for the tree tensor cross interpolation optimization loop.
///
/// # Defaults
///
/// | Field              | Default       | Description                                         |
/// |--------------------|---------------|-----------------------------------------------------|
/// | `tolerance`        | `1e-8`        | Relative stopping tolerance on normalized bond error |
/// | `max_iter`         | `20`          | Maximum number of edge-order iterations              |
/// | `max_bond_dim`     | `usize::MAX`  | Maximum bond dimension (no cap by default)           |
/// | `normalize_error`  | `true`        | Normalize error by maximum sample magnitude          |
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::TreeTciOptions;
///
/// // Default options
/// let opts = TreeTciOptions::default();
/// assert!((opts.tolerance - 1e-8).abs() < 1e-15);
/// assert_eq!(opts.max_iter, 20);
/// assert_eq!(opts.max_bond_dim, usize::MAX);
/// assert!(opts.normalize_error);
///
/// // Custom options for high-precision work
/// let opts = TreeTciOptions {
///     tolerance: 1e-12,
///     max_iter: 50,
///     max_bond_dim: 100,
///     normalize_error: true,
/// };
/// assert!((opts.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.max_bond_dim, 100);
/// ```
#[derive(Clone, Debug)]
pub struct TreeTciOptions {
    /// Relative stopping tolerance on the normalized bond error.
    ///
    /// The optimization loop monitors the maximum bond error across all edges.
    /// When `normalize_error` is true, this error is divided by the maximum
    /// observed sample magnitude. Recommended range: `1e-6` to `1e-12`.
    /// Default: `1e-8`.
    pub tolerance: f64,

    /// Maximum number of edge-order iterations (outer sweeps).
    ///
    /// Each iteration visits all edges twice (two inner passes) and updates
    /// pivot sets. Typical values: 10--50. Default: `20`.
    pub max_iter: usize,

    /// Maximum bond dimension retained by the tcicore LUCI pivot substrate.
    ///
    /// Caps the number of pivots per edge bipartition. Use this to limit
    /// memory and computation for large problems. Default: `usize::MAX` (no cap).
    pub max_bond_dim: usize,

    /// Whether to normalize the bond error by the maximum observed sample magnitude.
    ///
    /// When `true`, the stopping criterion uses relative error
    /// `max_bond_error / max_sample_value`. When `false`, the raw absolute
    /// bond error is used. Default: `true`.
    pub normalize_error: bool,
}

impl Default for TreeTciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: usize::MAX,
            normalize_error: true,
        }
    }
}

/// Optimize a TreeTCI state with the MVP strategy choices:
/// `AllEdges` visitation and [`DefaultProposer`](crate::DefaultProposer).
///
/// Returns `(ranks_per_iter, normalized_errors_per_iter)`.
///
/// This is a convenience wrapper around [`optimize_with_proposer`] with the
/// default neighbor-product proposer.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_default, GlobalIndexBatch, TreeTCI2, TreeTciEdge,
///     TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let local_dims = vec![2, 2];
/// let mut state = TreeTCI2::<f64>::new(local_dims, graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.max_sample_value = 1.0;
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut vals = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         vals.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(vals)
/// };
///
/// let options = TreeTciOptions { tolerance: 1e-10, max_iter: 5, ..Default::default() };
/// let (ranks, errors) = optimize_default(&mut state, evaluate, &options).unwrap();
///
/// // One entry per sweep actually run; the loop stops early once converged,
/// // so this may be less than max_iter (5).
/// assert!(!ranks.is_empty() && ranks.len() <= 5);
/// assert_eq!(ranks.len(), errors.len());
/// assert!(errors.last().copied().unwrap_or(1.0) < 1e-8);
/// ```
pub fn optimize_default<T, F>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    optimize_with_proposer(state, evaluate, options, &crate::DefaultProposer)
}

/// Optimize a TreeTCI state with `AllEdges` visitation and a caller-supplied
/// pivot candidate proposer.
///
/// Returns `(ranks_per_iter, normalized_errors_per_iter)`.
///
/// Use this when you need a custom proposer (e.g., [`SimpleProposer`](crate::SimpleProposer)
/// or [`TruncatedDefaultProposer`](crate::TruncatedDefaultProposer)).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     optimize_with_proposer, GlobalIndexBatch, SimpleProposer,
///     TreeTCI2, TreeTciEdge, TreeTciGraph, TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let mut state = TreeTCI2::<f64>::new(vec![2, 2], graph).unwrap();
/// state.add_global_pivots(&[vec![0, 0]]).unwrap();
/// state.max_sample_value = 1.0;
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut vals = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         vals.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(vals)
/// };
///
/// let proposer = SimpleProposer::seeded(42);
/// let options = TreeTciOptions { tolerance: 1e-10, max_iter: 3, ..Default::default() };
/// let (ranks, errors) = optimize_with_proposer(
///     &mut state, evaluate, &options, &proposer,
/// ).unwrap();
///
/// // One entry per sweep actually run; the loop stops early once converged,
/// // so this may be less than max_iter (3).
/// assert!(!ranks.is_empty() && ranks.len() <= 3);
/// assert_eq!(ranks.len(), errors.len());
/// ```
pub fn optimize_with_proposer<T, F, P>(
    state: &mut TreeTCI2<T>,
    evaluate: F,
    options: &TreeTciOptions,
    proposer: &P,
) -> Result<(Vec<usize>, Vec<f64>)>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    ensure!(
        options.max_iter > 0,
        "TreeTCI optimization requires max_iter > 0"
    );
    ensure!(
        options.max_bond_dim > 0,
        "TreeTCI optimization requires max_bond_dim > 0"
    );

    let mut ranks = Vec::new();
    let mut errors = Vec::new();
    let visitor = AllEdges;
    const INNER_EDGE_PASSES: usize = 2;
    // Matches `tensor4all-tensorci`'s `TensorCI2Options::ncheck_history` default
    // (see `convergence_criterion` in tensorci2.rs, itself a port of Julia's
    // `convergencecriterion`). A single below-tolerance sweep is not reliable:
    // the bond-error estimate can dip before the pivot search has actually
    // stabilized. TreeTCI2 has no per-sweep "new global pivots" count to check
    // (unlike TensorCI2), so this only checks error-below-tolerance + rank
    // stability over the trailing window, not the third TensorCI2 criterion.
    const NCHECK_HISTORY: usize = 3;

    for _iter in 0..options.max_iter {
        for _pass in 0..INNER_EDGE_PASSES {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let kernel_options = RrLUOptions {
                rel_tol: 1e-14,
                abs_tol: options.tolerance * error_scale,
                max_rank: options.max_bond_dim,
                left_orthogonal: true,
            };

            state.ijset_history.push(state.ijset.clone());
            state.flush_pivot_errors();

            for edge in visitor.visit_order(state) {
                update_edge(state, edge, &evaluate, &kernel_options, proposer)?;
            }
        }

        ranks.push(state.max_rank());
        let normalized_error = if options.normalize_error && state.max_sample_value > 0.0 {
            state.max_bond_error() / state.max_sample_value
        } else {
            state.max_bond_error()
        };
        errors.push(normalized_error);

        // BUGFIX (local, not upstream): the sweep loop previously always ran
        // to `max_iter` with no convergence check, wasting O(max_iter /
        // actual_sweeps_needed) work on already-converged problems. See
        // ~/tensor4all-rust/treetci-optimize-no-early-stop-bug.md.
        //
        // Mirrors `TreeTCI.jl`'s `convergencecriterion` (as locally patched
        // for the scale-mismatch bug on branch `local-fix-convergence`,
        // commit 06563dd): error-below-tolerance + rank-stable over the
        // trailing window, OR the rank has saturated at `max_bond_dim` for
        // the whole window (further sweeps cannot reduce the error once the
        // bond dimension is capped, so waiting for it to also cross
        // `tolerance` would just burn the remaining `max_iter` sweeps).
        if errors.len() >= NCHECK_HISTORY {
            let n = errors.len();
            let last_errors = &errors[n - NCHECK_HISTORY..];
            let last_ranks = &ranks[n - NCHECK_HISTORY..];
            let errors_converged = last_errors.iter().all(|&e| e < options.tolerance);
            let rank_stable = last_ranks.iter().min().copied().unwrap_or(0)
                == last_ranks.last().copied().unwrap_or(0);
            let bond_dim_saturated = last_ranks.iter().all(|&r| r >= options.max_bond_dim);
            if (errors_converged && rank_stable) || bond_dim_saturated {
                break;
            }
        }
    }

    Ok((ranks, errors))
}

#[cfg(test)]
mod tests;
