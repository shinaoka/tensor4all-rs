use crate::error::Result as TreeTciResult;
use crate::{
    materialize::to_treetn, optimize_with_proposer, GlobalIndexBatch, MultiIndex,
    PivotCandidateProposer, TreeTCI2, TreeTciGraph, TreeTciOptions,
};
use anyhow::Result;
use tensor4all_core::CommonScalar;
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetn::TreeTN;

/// High-level TreeTCI return type:
/// `(treetn, ranks_per_iter, normalized_errors_per_iter)`.
///
/// - `treetn`: The materialized tree tensor network.
/// - `ranks_per_iter`: Maximum bond dimension at each iteration.
/// - `normalized_errors_per_iter`: Normalized bond error at each iteration.
pub type TreeTciRunResult = (
    TreeTN<tensor4all_core::IdxTensor, usize>,
    Vec<usize>,
    Vec<f64>,
);

/// Cross interpolate a function on a tree graph and return a `TreeTN`.
///
/// This is the unified entry point for tree tensor cross interpolation.
/// The `evaluate` closure receives batches of multi-indices and must return
/// one scalar per point.
///
/// The `proposer` controls how pivot candidates are generated.
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{
///     crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph,
///     TreeTciOptions,
/// };
/// use anyhow::Result;
///
/// // Approximate the 2-site identity function f(i, j) = 1 if i==j else 0
/// let graph = TreeTciGraph::new(2, &[TreeTciEdge::new(0, 1)]).unwrap();
/// let local_dims = vec![2, 2];
///
/// let evaluate = |batch: GlobalIndexBatch<'_>| -> Result<Vec<f64>> {
///     let mut values = Vec::with_capacity(batch.n_points());
///     for p in 0..batch.n_points() {
///         let i = batch.get(0, p).unwrap();
///         let j = batch.get(1, p).unwrap();
///         values.push(if i == j { 1.0 } else { 0.0 });
///     }
///     Ok(values)
/// };
///
/// let options = TreeTciOptions {
///     tolerance: 1e-10,
///     max_iter: 10,
///     max_bond_dim: Some(10),
///     normalize_error: true,
///     ..Default::default()
/// };
///
/// let proposer = DefaultProposer;
/// let (treetn, ranks, errors) = crossinterpolate2::<f64, _, _>(
///     evaluate,
///     local_dims,
///     graph,
///     vec![],
///     options,
///     None,
///     &proposer,
/// ).unwrap();
///
/// // The identity on a 2x2 space has rank 2
/// assert!(ranks.last().copied().unwrap_or(0) <= 2);
/// // Error should converge to near zero
/// assert!(errors.last().copied().unwrap_or(1.0) < 1e-8);
/// ```
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns [`TreeTciError::InvalidConfiguration`] for invalid options. It
/// also returns an error when the operation fails (a shape or index mismatch,
/// or a backend failure).
///
pub fn crossinterpolate2<T, F, P>(
    evaluate: F,
    local_dims: Vec<usize>,
    graph: TreeTciGraph,
    initial_pivots: Vec<MultiIndex>,
    options: TreeTciOptions,
    center_site: Option<usize>,
    proposer: &P,
) -> TreeTciResult<TreeTciRunResult>
where
    T: FullPivLuScalar
        + CommonScalar
        + tensor4all_core::MatrixLuciScalar
        + tensor4all_core::TensorElement
        + crate::globalpivot::ScalarParts,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    options.validate()?;
    if !(local_dims.len() == graph.n_sites()) {
        return Err(anyhow::anyhow!(
            "local_dims length {} must match graph site count {}",
            local_dims.len(),
            graph.n_sites()
        )
        .into());
    };

    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = TreeTCI2::<T>::new(local_dims, graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let n_sites = tci.local_dims.len();
    let flat_len = n_sites
        .checked_mul(pivots.len())
        .ok_or_else(|| anyhow::anyhow!("initial pivot batch size overflowed usize"))?;
    let mut flat = Vec::with_capacity(flat_len);
    for pivot in &pivots {
        flat.extend_from_slice(pivot);
    }
    let batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = evaluate(batch)?;
    if init_vals.len() != pivots.len() {
        return Err(anyhow::anyhow!(
            "initial evaluator returned {} values for {} pivots",
            init_vals.len(),
            pivots.len()
        )
        .into());
    }
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| CommonScalar::abs_val(*v))
        .fold(0.0f64, f64::max);
    if !matches!(
        tci.max_sample_value.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    ) {
        return Err(anyhow::anyhow!("initial pivots must not all evaluate to zero").into());
    }

    let (ranks, errors) = optimize_with_proposer(&mut tci, &evaluate, &options, proposer)?;
    let treetn = to_treetn(&tci, &evaluate, center_site)?;

    Ok((treetn, ranks, errors))
}
