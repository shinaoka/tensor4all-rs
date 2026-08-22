//! Budgeted dense edge-local matrices and LUCI factors.

// INVARIANT: this conformance implementation materializes only one checked
// edge-local matrix and always uses the owned dense LUCI entry point.
#![allow(dead_code)]

use std::mem::size_of;

use tensor4all_core::{matrix_luci_factors_from_matrix_owned, RrLUOptions};
use tensor4all_core::{IdxTensor, IndexLike};
use tensor4all_tensorbackend::{mat_mul, transpose, Matrix};
use tensor4all_treetn::TreeTN;

use crate::{
    frames::InputFrameStore,
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{CandidateSets, ComponentSample, SampleArena},
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

#[derive(Clone, Debug)]
pub(crate) struct LocalUpdateResult<T> {
    pub(crate) row_samples: Vec<ComponentSample>,
    pub(crate) col_samples: Vec<ComponentSample>,
    pub(crate) left: Matrix<T>,
    pub(crate) right: Matrix<T>,
    pub(crate) pivot_errors: Vec<f64>,
    pub(crate) sampled_scale: f64,
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
    pub(crate) local_values: Vec<T>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_and_factor_edge<T, V, F>(
    inputs: &[TreeTN<IdxTensor, V>],
    problem: &PreparedTreeProblem<V>,
    _arena: &SampleArena,
    candidates: &CandidateSets,
    frames: &InputFrameStore<T>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    left_orthogonal: bool,
    operator: &mut F,
) -> Result<LocalUpdateResult<T>>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    if inputs.is_empty() {
        return Err(TreeAciError::NoInputs);
    }
    let edge = problem
        .directed_edges
        .get(forward)
        .ok_or(TreeAciError::InternalInvariant {
            message: "local update references an unknown directed edge",
        })?;
    let reverse = edge.reverse;
    let row_candidates = enumerate_candidates(
        problem,
        candidates,
        forward,
        "candidate rows",
        options.max_candidate_rows,
    )?;
    let col_candidates = enumerate_candidates(
        problem,
        candidates,
        reverse,
        "candidate columns",
        options.max_candidate_cols,
    )?;
    let row_count = row_candidates.len();
    let col_count = col_candidates.len();
    let point_count = row_count
        .checked_mul(col_count)
        .ok_or(TreeAciError::SizeOverflow {
            context: "local matrix elements",
        })?;
    enforce_limit(
        "local matrix elements",
        point_count,
        options.max_local_matrix_elements,
    )?;
    let matrix_working_elements = inputs
        .len()
        .checked_add(2)
        .and_then(|factor| factor.checked_mul(point_count))
        .ok_or(TreeAciError::SizeOverflow {
            context: "local matrix working elements",
        })?;
    let cut_rank_sum = inputs.iter().try_fold(0usize, |sum, input| {
        let graph_edge =
            input
                .edge_between(&edge.from, &edge.to)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "local update input is missing its cut edge",
                })?;
        let bond_dim = input
            .bond_index(graph_edge)
            .ok_or(TreeAciError::InternalInvariant {
                message: "local update input cut is missing its bond index",
            })?
            .dim();
        sum.checked_add(bond_dim).ok_or(TreeAciError::SizeOverflow {
            context: "input cut-rank sum",
        })
    })?;
    let candidate_frame_elements = row_count
        .checked_add(col_count)
        .and_then(|count| count.checked_mul(cut_rank_sum))
        .ok_or(TreeAciError::SizeOverflow {
            context: "candidate frame working elements",
        })?;
    let working_elements = matrix_working_elements
        .checked_add(candidate_frame_elements)
        .ok_or(TreeAciError::SizeOverflow {
            context: "local update working elements",
        })?;
    let working_bytes =
        working_elements
            .checked_mul(size_of::<T>())
            .ok_or(TreeAciError::SizeOverflow {
                context: "local matrix working bytes",
            })?;
    enforce_limit("working bytes", working_bytes, options.max_working_bytes)?;
    let factor_rank_bound = options
        .max_bond_dim
        .unwrap_or(usize::MAX)
        .min(row_count)
        .min(col_count)
        .max(1);
    let left_elements =
        row_count
            .checked_mul(factor_rank_bound)
            .ok_or(TreeAciError::SizeOverflow {
                context: "left local factor elements",
            })?;
    let right_elements =
        col_count
            .checked_mul(factor_rank_bound)
            .ok_or(TreeAciError::SizeOverflow {
                context: "right local factor elements",
            })?;
    enforce_limit("core elements", left_elements, options.max_core_elements)?;
    enforce_limit("core elements", right_elements, options.max_core_elements)?;

    let row_frames = candidate_frames(inputs, problem, frames, forward, &row_candidates)?;
    let col_frames = candidate_frames(inputs, problem, frames, reverse, &col_candidates)?;
    let mut input_values = vec![T::default(); inputs.len() * point_count];
    // Per input, `input_values[.., point] = row_frames[row] . col_frames[col]`
    // is one (row_count x chi) times (chi x col_count) matrix product, not a
    // per-point scalar dot product: pack each side's candidate frame vectors
    // into a dense matrix (they are already contiguous per candidate, so this
    // is a flatten, not a transpose) and let BLAS do the O(row*col*chi)
    // contraction in one `mat_mul` call. Only the O(row*col) scatter into the
    // batch's interleaved-by-input layout remains a plain loop.
    if point_count > 0 {
        for (input, (row_input_frames, col_input_frames)) in
            row_frames.iter().zip(col_frames.iter()).enumerate()
        {
            let bond_dim = row_input_frames.first().map_or(0, |frame| frame.len());
            for frame in row_input_frames.iter().chain(col_input_frames.iter()) {
                if frame.len() != bond_dim {
                    return Err(TreeAciError::InternalInvariant {
                        message: "opposite input frames have different cut bond dimensions",
                    });
                }
            }
            if bond_dim == 0 {
                continue;
            }
            let row_flat: Vec<T> = row_input_frames
                .iter()
                .flat_map(|frame| frame.iter().copied())
                .collect();
            let col_flat: Vec<T> = col_input_frames
                .iter()
                .flat_map(|frame| frame.iter().copied())
                .collect();
            let row_bond_matrix = Matrix::from_col_major_vec(bond_dim, row_count, row_flat);
            let col_bond_matrix = Matrix::from_col_major_vec(bond_dim, col_count, col_flat);
            let row_candidate_matrix = transpose(&row_bond_matrix);
            let product = mat_mul(&row_candidate_matrix, &col_bond_matrix).map_err(|error| {
                TreeAciError::Numerical {
                    message: error.to_string(),
                }
            })?;
            for col in 0..col_count {
                for row in 0..row_count {
                    let point = row + row_count * col;
                    input_values[input + inputs.len() * point] = product[[row, col]];
                }
            }
        }
    }
    let batch = TreeElementwiseBatch::new(&input_values, inputs.len(), point_count)?;
    let mut local_values = vec![T::default(); point_count];
    operator(batch, &mut local_values)?;
    let sampled_scale = local_values.iter().copied().fold(0.0_f64, |scale, value| {
        scale.max(tensor4all_core::Scalar::abs_val(value))
    });
    let matrix = Matrix::from_col_major_vec(row_count, col_count, local_values.clone());
    let factors = matrix_luci_factors_from_matrix_owned(
        matrix,
        Some(RrLUOptions {
            max_bond_dim: options.max_bond_dim.unwrap_or(usize::MAX),
            rel_tol: if options.scale_tolerance {
                options.tolerance
            } else {
                0.0
            },
            abs_tol: if options.scale_tolerance {
                0.0
            } else {
                options.tolerance
            },
            left_orthogonal,
        }),
    )
    .map_err(|error| TreeAciError::Numerical {
        message: error.to_string(),
    })?;
    let (left, right, row_indices, col_indices) = if factors.rank == 0 {
        (
            Matrix::zeros(row_count, 1),
            Matrix::zeros(1, col_count),
            vec![0],
            vec![0],
        )
    } else {
        (
            factors.left,
            factors.right,
            factors.row_indices,
            factors.col_indices,
        )
    };
    let row_samples = row_indices
        .into_iter()
        .map(|index| row_candidates[index].clone())
        .collect();
    let col_samples = col_indices
        .into_iter()
        .map(|index| col_candidates[index].clone())
        .collect();
    Ok(LocalUpdateResult {
        row_samples,
        col_samples,
        left,
        right,
        pivot_errors: factors.pivot_errors,
        sampled_scale,
        row_count,
        col_count,
        local_values,
    })
}

fn enumerate_candidates<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    candidate_sets: &CandidateSets,
    edge: DirectedEdgeId,
    resource: &'static str,
    limit: usize,
) -> Result<Vec<ComponentSample>> {
    let directed = &problem.directed_edges[edge];
    let node =
        *problem
            .node_positions
            .get(&directed.from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate source has no prepared node position",
            })?;
    let mut count = problem.physical[node].local_dim;
    for incoming in &directed.incoming_to_from {
        let ids = candidate_sets
            .ids
            .get(*incoming)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate incoming edge has no candidate set",
            })?;
        if ids.is_empty() {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate incoming edge has an empty candidate set",
            });
        }
        count = count
            .checked_mul(ids.len())
            .ok_or(TreeAciError::SizeOverflow {
                context: "candidate count",
            })?;
    }
    enforce_limit(resource, count, limit)?;
    let mut candidates = Vec::with_capacity(count);
    for encoded in 0..count {
        let mut quotient = encoded;
        let local_coordinate = quotient % problem.physical[node].local_dim;
        quotient /= problem.physical[node].local_dim;
        let mut incoming_samples = Vec::with_capacity(directed.incoming_to_from.len());
        for incoming in &directed.incoming_to_from {
            let ids = &candidate_sets.ids[*incoming];
            incoming_samples.push((*incoming, ids[quotient % ids.len()]));
            quotient /= ids.len();
        }
        candidates.push(ComponentSample {
            local_coordinate,
            incoming: incoming_samples,
        });
    }
    Ok(candidates)
}

fn candidate_frames<T: TreeAciScalar, V: TreeAciNode>(
    inputs: &[TreeTN<IdxTensor, V>],
    problem: &PreparedTreeProblem<V>,
    frames: &InputFrameStore<T>,
    edge: DirectedEdgeId,
    candidates: &[ComponentSample],
) -> Result<Vec<Vec<Vec<T>>>> {
    (0..inputs.len())
        .map(|input| frames.candidate_frames_for_edge(inputs, problem, input, edge, candidates))
        .collect()
}

fn enforce_limit(resource: &'static str, requested: usize, limit: usize) -> Result<()> {
    if requested > limit {
        return Err(TreeAciError::ResourceLimit {
            resource,
            requested,
            limit,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
