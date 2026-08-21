//! Atomic proposal and commit of one directed edge update.

#![allow(dead_code)]

use tensor4all_core::{DynIndex, IdxTensor};

use crate::{
    local_update::{materialize_and_factor_edge, LocalUpdateResult},
    problem::DirectedEdgeId,
    state::TreeAciState,
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EdgeCommitReport {
    pub(crate) forward: DirectedEdgeId,
    pub(crate) new_rank: usize,
    pub(crate) pivot_error: f64,
    pub(crate) sampled_scale: f64,
    pub(crate) evaluated_points: usize,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_edge_transaction<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    left_orthogonal: bool,
    operator: &mut F,
) -> Result<EdgeCommitReport>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let proposal = materialize_and_factor_edge(
        state.inputs,
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.input_frames,
        forward,
        options,
        left_orthogonal,
        operator,
    )?;
    commit_edge_proposal(state, forward, proposal)
}

fn commit_edge_proposal<T: TreeAciScalar, V: TreeAciNode>(
    state: &mut TreeAciState<'_, T, V>,
    forward: DirectedEdgeId,
    proposal: LocalUpdateResult<T>,
) -> Result<EdgeCommitReport> {
    let directed =
        state
            .problem
            .directed_edges
            .get(forward)
            .ok_or(TreeAciError::InternalInvariant {
                message: "edge proposal references an unknown directed edge",
            })?;
    let reverse = directed.reverse;
    let new_rank = proposal.left.ncols();
    let evaluated_points =
        proposal
            .row_count
            .checked_mul(proposal.col_count)
            .ok_or(TreeAciError::SizeOverflow {
                context: "edge evaluated point count",
            })?;
    if proposal.right.nrows() != new_rank
        || proposal.row_samples.len() != new_rank
        || proposal.col_samples.len() != new_rank
    {
        return Err(TreeAciError::InternalInvariant {
            message: "edge proposal factors and selected samples disagree on rank",
        });
    }

    let next_generation = state
        .generation
        .checked_add(1)
        .ok_or(TreeAciError::SizeOverflow {
            context: "tree ACI state generation",
        })?;
    let mut proposed_output = state.output.clone();
    replace_edge_cores(
        &mut proposed_output,
        &state.problem,
        forward,
        proposal.left,
        proposal.right,
    )?;
    let checkpoint = state.sample_arena.checkpoint();
    let staged = (|| {
        let left_ids = proposal
            .row_samples
            .iter()
            .cloned()
            .map(|sample| {
                state
                    .sample_arena
                    .intern_component(&state.problem, forward, sample)
            })
            .collect::<Result<Vec<_>>>()?;
        let right_ids = proposal
            .col_samples
            .iter()
            .cloned()
            .map(|sample| {
                state
                    .sample_arena
                    .intern_component(&state.problem, reverse, sample)
            })
            .collect::<Result<Vec<_>>>()?;
        let frames =
            state
                .input_frames
                .extend(state.inputs, &state.problem, &state.sample_arena)?;
        Ok((left_ids, right_ids, frames))
    })();
    let (left_ids, right_ids, proposed_frames) = match staged {
        Ok(staged) => staged,
        Err(error) => {
            state.sample_arena.rollback(checkpoint)?;
            return Err(error);
        }
    };

    let edge_number = forward / 2;
    // The commit rule: pivot pairs set the bond rank and own `P_e`, while the
    // candidate sets are *replaced* by the same selections so that neighbouring
    // edges see them. Replacement rather than union is what keeps candidate
    // growth bounded without an eviction policy, mirroring train ACI rebuilding
    // its frames from the selected pivots.
    let mut proposed_candidates = state.candidates.clone();
    proposed_candidates.ids[forward] = left_ids.clone();
    proposed_candidates.ids[reverse] = right_ids.clone();
    proposed_candidates.generation = next_generation;

    let mut proposed_pivots = state.pivots.clone();
    let pivot_pairs = if forward.is_multiple_of(2) {
        left_ids.into_iter().zip(right_ids).collect()
    } else {
        right_ids.into_iter().zip(left_ids).collect()
    };
    proposed_pivots.set(edge_number, pivot_pairs);

    let pivot_error = proposal.pivot_errors.last().copied().unwrap_or(0.0);
    state.output = proposed_output;
    state.candidates = proposed_candidates;
    state.pivots = proposed_pivots;
    state.input_frames = proposed_frames;
    state.edge_ranks[edge_number] = state.pivots.rank(edge_number);
    state.edge_errors[edge_number] = pivot_error;
    state.edge_scales[edge_number] = proposal.sampled_scale;
    state.generation = next_generation;
    Ok(EdgeCommitReport {
        forward,
        new_rank,
        pivot_error,
        sampled_scale: proposal.sampled_scale,
        evaluated_points,
    })
}

fn replace_edge_cores<T: TreeAciScalar, V: TreeAciNode>(
    output: &mut tensor4all_treetn::TreeTN<IdxTensor, V>,
    problem: &crate::problem::PreparedTreeProblem<V>,
    forward: DirectedEdgeId,
    left: tensor4all_tensorbackend::Matrix<T>,
    right: tensor4all_tensorbackend::Matrix<T>,
) -> Result<()> {
    let directed = &problem.directed_edges[forward];
    let reverse = directed.reverse;
    let new_bond = DynIndex::new_dyn(left.ncols());
    let left_indices = factor_indices(output, problem, forward, Some(new_bond.clone()))?;
    let mut right_indices = vec![new_bond.clone()];
    right_indices.extend(factor_indices(output, problem, reverse, None)?);
    let left_tensor =
        IdxTensor::from_dense(left_indices, left.into_col_major_vec()).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?;
    let right_tensor =
        IdxTensor::from_dense(right_indices, right.into_col_major_vec()).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?;
    let graph_edge = output.edge_between(&directed.from, &directed.to).ok_or(
        TreeAciError::InternalInvariant {
            message: "output is missing the committed edge",
        },
    )?;
    let left_node = output
        .node_index(&directed.from)
        .ok_or(TreeAciError::InternalInvariant {
            message: "output is missing the committed source node",
        })?;
    let right_node = output
        .node_index(&directed.to)
        .ok_or(TreeAciError::InternalInvariant {
            message: "output is missing the committed target node",
        })?;
    output.replace_edge_bond(graph_edge, new_bond.clone())?;
    output.replace_tensor(left_node, left_tensor)?;
    output.replace_tensor(right_node, right_tensor)?;
    output.set_edge_ortho_towards(graph_edge, Some(directed.to.clone()))?;
    output.set_canonical_region([directed.to.clone()])?;
    Ok(())
}

fn factor_indices<V: TreeAciNode>(
    output: &tensor4all_treetn::TreeTN<IdxTensor, V>,
    problem: &crate::problem::PreparedTreeProblem<V>,
    directed_edge: DirectedEdgeId,
    outgoing: Option<DynIndex>,
) -> Result<Vec<DynIndex>> {
    let directed = &problem.directed_edges[directed_edge];
    let node =
        *problem
            .node_positions
            .get(&directed.from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "factor source has no prepared node position",
            })?;
    let mut indices = problem.physical[node].indices.clone();
    for incoming in &directed.incoming_to_from {
        let incoming = &problem.directed_edges[*incoming];
        let edge = output.edge_between(&incoming.from, &incoming.to).ok_or(
            TreeAciError::InternalInvariant {
                message: "output is missing an incoming factor edge",
            },
        )?;
        indices.push(
            output
                .bond_index(edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "output incoming factor edge has no bond index",
                })?
                .clone(),
        );
    }
    if let Some(outgoing) = outgoing {
        indices.push(outgoing);
    }
    Ok(indices)
}

#[cfg(test)]
mod tests;
