use crate::error::Result as TreeTciResult;
use crate::{
    globalpivot::{find_global_pivots, ScalarParts},
    update::update_edge,
    AllEdges, EdgeVisitor, GlobalIndexBatch, PivotCandidateProposer, TreeTCI2,
};
use anyhow::Result;
use tensor4all_core::CommonScalar;
use tensor4all_tcicore::{MatrixLuciScalar as Scalar, RrLUOptions};
use tensor4all_tensorbackend::FullPivLuScalar;

/// MVP optimization options for TreeTCI.
///
/// Controls convergence criteria, iteration limits, and bond dimension caps
/// for the tree tensor cross interpolation optimization loop.
///
/// # Defaults
///
/// | Field                     | Default       | Description                                         |
/// |---------------------------|---------------|-----------------------------------------------------|
/// | `tolerance`               | `1e-8`        | Relative stopping tolerance on normalized bond error |
/// | `max_iter`                | `20`          | Maximum number of edge-order iterations              |
/// | `max_bond_dim`            | `None`        | Maximum bond dimension (no cap by default)           |
/// | `normalize_error`         | `true`        | Normalize error by maximum sample magnitude          |
/// | `enable_global_pivots`    | `false`       | Run automatic global pivot search after each sweep   |
/// | `nsearch`                 | `5`           | Random starting points for the global pivot search   |
/// | `max_nglobal_pivot`       | `5`           | Global pivots added per iteration                    |
/// | `tol_margin_global_search`| `10.0`        | Global pivot acceptance margin over `abs_tol`        |
/// | `seed`                    | `None`        | RNG seed for the global pivot search                 |
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
/// assert_eq!(opts.max_bond_dim, None);
/// assert!(opts.normalize_error);
/// assert!(!opts.enable_global_pivots);
///
/// // Custom options for high-precision work
/// let opts = TreeTciOptions {
///     tolerance: 1e-12,
///     max_iter: 50,
///     max_bond_dim: Some(100),
///     normalize_error: true,
///     enable_global_pivots: true,
///     nsearch: 20,
///     max_nglobal_pivot: 10,
///     tol_margin_global_search: 5.0,
///     seed: Some(42),
/// };
/// assert!((opts.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(opts.max_iter, 50);
/// assert_eq!(opts.max_bond_dim, Some(100));
/// assert!(opts.enable_global_pivots);
/// assert_eq!(opts.seed, Some(42));
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
    /// memory and computation for large problems. Default: `None` (no cap).
    pub max_bond_dim: Option<usize>,

    /// Whether to normalize the bond error by the maximum observed sample magnitude.
    ///
    /// When `true`, the stopping criterion uses relative error
    /// `max_bond_error / max_sample_value`. When `false`, the raw absolute
    /// bond error is used. Default: `true`.
    pub normalize_error: bool,

    /// Whether to run an automatic global pivot search after each sweep.
    ///
    /// When `true`, the optimizer materializes the current approximation
    /// after each iteration and searches for multi-indices where
    /// `|f(idx) - tt(idx)|` is large, injecting the best finds via
    /// [`TreeTCI2::add_global_pivots`](crate::TreeTCI2::add_global_pivots).
    /// This recovers separated features that local pivot updates miss when
    /// the initial pivots sit in a single basin. Default: `false` (opt-in,
    /// preserves existing behavior).
    pub enable_global_pivots: bool,

    /// Number of random starting points for the global pivot search.
    ///
    /// Each starting point is locally optimized over all site coordinates.
    /// Larger values explore the index space more thoroughly at the cost of
    /// more evaluations. Ignored when `enable_global_pivots` is `false`.
    /// Default: `5`.
    pub nsearch: usize,

    /// Maximum number of global pivots added per iteration.
    ///
    /// Ignored when `enable_global_pivots` is `false`. Default: `5`.
    pub max_nglobal_pivot: usize,

    /// Tolerance margin for the global pivot search.
    ///
    /// A candidate pivot is accepted when its interpolation error exceeds
    /// `abs_tol * tol_margin_global_search`, where `abs_tol` is the sweep's
    /// absolute tolerance (`tolerance * max_sample_value` when
    /// `normalize_error` is enabled). The value is always validated (it must
    /// be finite and nonnegative) but only consulted when `enable_global_pivots`
    /// is `true`. Default: `10.0`.
    pub tol_margin_global_search: f64,

    /// Random seed for the global pivot search.
    ///
    /// `None` seeds from OS entropy. Only used when `enable_global_pivots`
    /// is `true`. Default: `None`.
    pub seed: Option<u64>,
}

impl Default for TreeTciOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: None,
            normalize_error: true,
            enable_global_pivots: false,
            nsearch: 5,
            max_nglobal_pivot: 5,
            tol_margin_global_search: 10.0,
            seed: None,
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
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
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
) -> TreeTciResult<(Vec<usize>, Vec<f64>)>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
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
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
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
) -> TreeTciResult<(Vec<usize>, Vec<f64>)>
where
    T: Scalar + CommonScalar + FullPivLuScalar + tensor4all_core::TensorElement + ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    if !(options.max_iter > 0) {
        return Err(anyhow::anyhow!("TreeTCI optimization requires max_iter > 0").into());
    };
    if options.max_bond_dim == Some(0) {
        return Err(anyhow::anyhow!("TreeTCI optimization requires max_bond_dim > 0").into());
    };
    if !options.tol_margin_global_search.is_finite() || options.tol_margin_global_search < 0.0 {
        return Err(anyhow::anyhow!(
            "TreeTCI optimization requires a finite nonnegative tol_margin_global_search"
        )
        .into());
    };

    let mut ranks = Vec::new();
    let mut errors = Vec::new();
    let mut nglobal_pivots_history: Vec<usize> = Vec::new();
    let visitor = AllEdges;
    const INNER_EDGE_PASSES: usize = 2;
    // Mirrors `tensor4all-tensorci`'s `TensorCI2Options::ncheck_history` default
    // (see `convergence_criterion` in tensorci2.rs, itself a port of Julia's
    // `convergencecriterion`). A single below-tolerance sweep is not reliable:
    // the bond-error estimate can dip before the pivot search has actually
    // stabilized. Like TensorCI2, the window must also contain no newly added
    // global pivots: an iteration that injected pivots has not yet swept them,
    // so its error estimate is stale with respect to those pivots.
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
                max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
                left_orthogonal: true,
            };

            state.ijset_history.push(state.ijset.clone());
            state.flush_pivot_errors();

            for edge in visitor.visit_order(state) {
                update_edge(state, edge, &evaluate, &kernel_options, proposer)?;
            }
        }

        ranks.push(state.max_bond_dim());
        let normalized_error = if options.normalize_error && state.max_sample_value > 0.0 {
            state.max_bond_error() / state.max_sample_value
        } else {
            state.max_bond_error()
        };
        errors.push(normalized_error);

        // Global pivot search: after each sweep, materialize the current
        // approximation and inject pivots where |f - tt| is large, so
        // separated features that the local pivot updates miss are sampled
        // in the next sweep. Opt-in; see `TreeTciOptions::enable_global_pivots`.
        // The search is skipped on the final iteration: a pivot injected after
        // the last sweep would never be processed by a subsequent sweep, so the
        // recorded error and termination reason would not reflect it.
        if options.enable_global_pivots && _iter + 1 < options.max_iter {
            let error_scale = if options.normalize_error && state.max_sample_value > 0.0 {
                state.max_sample_value
            } else {
                1.0
            };
            let abs_tol = options.tolerance * error_scale;
            let seed = match options.seed {
                Some(base) => base.wrapping_add(_iter as u64),
                None => rand::random(),
            };
            let pivots = find_global_pivots(
                state,
                &evaluate,
                options.nsearch,
                options.max_nglobal_pivot,
                options.tol_margin_global_search,
                abs_tol,
                seed,
            )?;
            state.add_global_pivots(&pivots)?;
            nglobal_pivots_history.push(pivots.len());
        } else {
            nglobal_pivots_history.push(0);
        }

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
            let last_ngp = &nglobal_pivots_history[n - NCHECK_HISTORY..];
            let errors_converged = last_errors.iter().all(|&e| e < options.tolerance);
            let no_global_pivots = last_ngp.iter().all(|&n| n == 0);
            let rank_stable = last_ranks.iter().min().copied().unwrap_or(0)
                == last_ranks.last().copied().unwrap_or(0);
            let bond_dim_saturated = options
                .max_bond_dim
                .is_some_and(|cap| last_ranks.iter().all(|&r| r >= cap));
            if (errors_converged && no_global_pivots && rank_stable) || bond_dim_saturated {
                break;
            }
        }
    }

    Ok((ranks, errors))
}

#[cfg(test)]
mod tests;
