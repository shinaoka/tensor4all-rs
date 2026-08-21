//! Public native TreeTN elementwise ACI entry points.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::{
    global_guard::InputEvaluators,
    schedule::{run_directional_pass, run_local_sweeps, PassDirection},
    single_site::evaluate_single_site,
    state::TreeAciState,
    Result, TreeAciDiagnostics, TreeAciNode, TreeAciOptions, TreeAciResult, TreeAciScalar,
    TreeElementwiseBatch,
};

/// Approximates a batched pointwise operator directly as a tree tensor network.
///
/// All inputs must have the same labeled tree topology and identical full
/// physical indices at corresponding nodes. A node may own zero, one, or many
/// physical indices; no quantization is required. The callback receives an
/// `n_inputs × n_points` column-major batch.
///
/// # Arguments
///
/// * `operator` - Fallible batched pointwise operation.
/// * `inputs` - Nonempty, topology-compatible native TreeTNs.
/// * `options` - Sweep, guard, rank, traversal, and allocation controls.
///
/// # Returns
///
/// The interpolated TreeTN, pass histories, termination reason, and diagnostics.
///
/// # Errors
///
/// Returns [`crate::TreeAciError`] for invalid inputs/options, callback failure,
/// resource exhaustion, scalar mismatch, or a numerical/tree operation error.
///
/// # Panics
///
/// This function does not intentionally panic. A callback panic is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treeaci::{tree_elementwise_batched, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let result = tree_elementwise_batched::<f64, _, _>(
///     |batch, output| {
///         for (point, value) in output.iter_mut().enumerate() {
///             *value = batch.get(0, point)?.powi(2);
///         }
///         Ok(())
///     },
///     &[tree],
///     &TreeAciOptions::default(),
/// )?;
/// assert_eq!(result.tree.to_dense()?.to_vec::<f64>()?, vec![4.0, 9.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tree_elementwise_batched<T, V, F>(
    mut operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let mut state = TreeAciState::<T, V>::initialize(inputs, options)?;
    if state.problem.node_order.len() == 1 {
        return evaluate_single_site(inputs, options, &mut operator);
    }
    let mut input_evaluators = InputEvaluators::new_with_message_cache_max_bytes(
        inputs,
        &state.problem,
        options.message_cache_max_bytes,
    )?;
    let history = run_local_sweeps(&mut state, &mut input_evaluators, options, &mut operator)?;
    let mut evaluated_points = history.evaluated_points;
    if history
        .global_pivots_found
        .last()
        .is_some_and(|found| *found > 0)
    {
        let direction = if history.max_ranks.len() % 2 == 0 {
            PassDirection::Forward
        } else {
            PassDirection::Reverse
        };
        let cleanup = run_directional_pass(&mut state, options, direction, &mut operator)?;
        evaluated_points = evaluated_points
            .checked_add(cleanup.evaluated_points)
            .ok_or(crate::TreeAciError::SizeOverflow {
                context: "final evaluated point count",
            })?;
    }
    state.output.verify_internal_consistency()?;
    let edge_ranks = state
        .edge_ranks
        .iter()
        .copied()
        .enumerate()
        .map(|(edge, rank)| {
            let prepared = &state.problem.directed_edges[2 * edge];
            (prepared.from.clone(), prepared.to.clone(), rank)
        })
        .collect::<Vec<_>>();
    let saturated_edges = edge_ranks
        .iter()
        .zip(&state.algebraic_edge_bounds)
        .filter(|((_, _, rank), algebraic)| {
            let limit = options.max_bond_dim.unwrap_or(usize::MAX).min(**algebraic);
            *rank >= limit
        })
        .map(|((from, to, _), _)| (from.clone(), to.clone()))
        .collect();
    Ok(TreeAciResult {
        tree: state.output,
        max_ranks: history.max_ranks,
        max_errors: history.max_errors,
        global_pivots_found: history.global_pivots_found,
        termination: history.termination,
        diagnostics: TreeAciDiagnostics {
            edge_ranks,
            saturated_edges,
            evaluated_points,
            sample_arena_records: state.sample_arena.record_count(),
            sample_arena_retained_bytes: state.sample_arena.retained_bytes(),
            frame_records: state.input_frames.records(),
            frame_retained_bytes: state.input_frames.retained_bytes(),
            candidate_set_sizes: state
                .problem
                .directed_edges
                .iter()
                .map(|edge| {
                    (
                        edge.from.clone(),
                        edge.to.clone(),
                        state.candidates.ids[edge.id].len(),
                    )
                })
                .collect(),
        },
    })
}

/// Approximates a scalar pointwise operator directly as a tree tensor network.
///
/// This convenience wrapper has the same topology, index, convergence, and
/// quantization rules as [`tree_elementwise_batched`].
///
/// # Arguments
///
/// * `operator` - Pointwise function receiving one scalar from every input.
/// * `inputs` - Nonempty, topology-compatible native TreeTNs.
/// * `options` - Sweep, guard, rank, traversal, and allocation controls.
///
/// # Returns
///
/// The same result structure as [`tree_elementwise_batched`].
///
/// # Errors
///
/// Returns [`crate::TreeAciError`] under the same conditions as the batched API.
///
/// # Panics
///
/// This function does not intentionally panic. A callback panic is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_treeaci::{tree_elementwise, TreeAciOptions};
/// use tensor4all_treetn::TreeTN;
///
/// let site = DynIndex::new_dyn(2);
/// let tree = TreeTN::from_tensors(
///     vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0])?],
///     vec![0usize],
/// )?;
/// let result = tree_elementwise::<f64, _, _>(
///     |values| values[0] + 1.0,
///     &[tree],
///     &TreeAciOptions::default(),
/// )?;
/// assert_eq!(result.tree.to_dense()?.to_vec::<f64>()?, vec![3.0, 4.0]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn tree_elementwise<T, V, F>(
    mut operator: F,
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: FnMut(&[T]) -> T,
{
    let mut scratch = Vec::with_capacity(inputs.len());
    tree_elementwise_batched(
        |batch, output| {
            for (point, value) in output.iter_mut().enumerate() {
                scratch.clear();
                for input in 0..batch.n_inputs() {
                    scratch.push(batch.get(input, point)?);
                }
                *value = operator(&scratch);
            }
            Ok(())
        },
        inputs,
        options,
    )
}

#[cfg(test)]
mod tests;
