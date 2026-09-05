//! Exact direct evaluation for trees with one node and no bonds.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::{
    initialize::validate_initial_guess, problem::prepare_problem, Result, TreeAciDiagnostics,
    TreeAciError, TreeAciNode, TreeAciOptions, TreeAciResult, TreeAciScalar, TreeAciTermination,
    TreeElementwiseBatch,
};

pub(crate) fn evaluate_single_site<T, V, F>(
    inputs: &[TreeTN<IdxTensor, V>],
    options: &TreeAciOptions<V>,
    operator: &mut F,
) -> Result<TreeAciResult<V>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let problem = prepare_problem::<T, V>(inputs, options)?;
    if problem.node_order.len() != 1 {
        return Err(TreeAciError::InternalInvariant {
            message: "single-site evaluation received a multi-node tree",
        });
    }
    if let Some(guess) = &options.initial_guess {
        validate_initial_guess::<T, V>(guess, &inputs[0], &problem)?;
    }
    let node = &problem.node_order[0];
    let indices = problem.physical[0].indices.clone();
    let point_count = problem.physical[0].local_dim;
    // Checked and charged against the working budget. The exact single-node
    // path skips the sweep entirely, so it used to be the one public entry that
    // allocated without consulting either.
    let input_elements =
        inputs
            .len()
            .checked_mul(point_count)
            .ok_or(TreeAciError::SizeOverflow {
                context: "single-site input buffer",
            })?;
    let working_elements = inputs
        .len()
        .checked_add(1)
        .and_then(|buffers| buffers.checked_mul(point_count))
        .ok_or(TreeAciError::SizeOverflow {
            context: "single-site working elements",
        })?;
    let working_bytes = working_elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(TreeAciError::SizeOverflow {
            context: "single-site working bytes",
        })?;
    crate::problem::enforce_limit("working bytes", working_bytes, problem.max_working_bytes)?;
    let mut input_values = vec![T::default(); input_elements];
    for (input_number, input) in inputs.iter().enumerate() {
        let node_index = input
            .node_index(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "prepared single-site input is missing its node",
            })?;
        let tensor = input
            .tensor(node_index)
            .ok_or(TreeAciError::InternalInvariant {
                message: "prepared single-site input is missing its tensor",
            })?
            .permute_indices(&indices)
            .map_err(|error| TreeAciError::Numerical {
                message: error.to_string(),
            })?;
        let values = tensor
            .to_vec::<T>()
            .map_err(|error| TreeAciError::ScalarKind {
                message: error.to_string(),
            })?;
        for (point, value) in values.into_iter().enumerate() {
            input_values[input_number + inputs.len() * point] = value;
        }
    }
    let batch = TreeElementwiseBatch::new(&input_values, inputs.len(), point_count)?;
    let mut output_values = vec![T::default(); point_count];
    operator(batch, &mut output_values)?;
    let output_tensor =
        IdxTensor::from_dense(indices, output_values).map_err(|error| TreeAciError::Numerical {
            message: error.to_string(),
        })?;
    let tree = TreeTN::from_tensors(vec![output_tensor], vec![node.clone()])?;
    let evaluated_points = u64::try_from(point_count).map_err(|_| TreeAciError::SizeOverflow {
        context: "evaluated point count",
    })?;
    Ok(TreeAciResult {
        tree,
        max_ranks: Vec::new(),
        max_errors: Vec::new(),
        global_pivots_found: Vec::new(),
        termination: TreeAciTermination::Converged,
        diagnostics: TreeAciDiagnostics {
            edge_ranks: Vec::new(),
            saturated_edges: Vec::new(),
            evaluated_points,
            sample_arena_records: 0,
            sample_arena_retained_bytes: 0,
            // The exact path builds no samples and no frames.
            frame_records: 0,
            frame_retained_bytes: 0,
            // A single-node tree has no cuts, so there are no candidate sets.
            candidate_set_sizes: Vec::new(),
        },
    })
}

#[cfg(test)]
mod tests;
