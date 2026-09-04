//! Budgeted dense edge-local matrices and LUCI factors.

use std::mem::size_of;

use tensor4all_core::IdxTensor;
use tensor4all_core::{matrix_luci_factors_from_matrix_owned, RrLUOptions};
#[cfg(test)]
use tensor4all_tensorbackend::mat_mul;
#[cfg(not(test))]
use tensor4all_tensorbackend::mat_mul_owned;
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use crate::{
    frames::InputFrameStore,
    problem::{enforce_limit, DirectedEdgeId, PreparedTreeProblem},
    samples::{CandidateSets, ComponentSample},
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
    #[cfg(test)]
    pub(crate) local_values: Vec<T>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn materialize_and_factor_edge<T, V, F>(
    inputs: &[TreeTN<IdxTensor, V>],
    problem: &PreparedTreeProblem<V>,
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
    #[cfg(test)]
    let preparation_started = std::time::Instant::now();
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
    let input_value_elements =
        inputs
            .len()
            .checked_mul(point_count)
            .ok_or(TreeAciError::SizeOverflow {
                context: "local input value elements",
            })?;
    let max_cut_rank = (0..inputs.len()).try_fold(0usize, |max_rank, input| {
        Ok::<usize, TreeAciError>(max_rank.max(frames.bond_dim(input, forward)?))
    })?;
    let candidate_frame_elements = row_count
        .checked_add(col_count)
        .and_then(|count| count.checked_mul(max_cut_rank))
        // Candidate vectors and their packed BLAS matrices coexist for one
        // input at a time. Other inputs are streamed and dropped.
        .and_then(|count| count.checked_mul(2))
        .ok_or(TreeAciError::SizeOverflow {
            context: "candidate frame working elements",
        })?;
    let candidate_frame_scratch =
        inputs
            .iter()
            .enumerate()
            .try_fold(0usize, |peak, (input, _)| {
                let row_scratch = frames.enumerated_candidate_frame_scratch_elements(
                    problem, input, forward, candidates,
                )?;
                let col_scratch = frames.enumerated_candidate_frame_scratch_elements(
                    problem, input, reverse, candidates,
                )?;
                Ok::<usize, TreeAciError>(peak.max(row_scratch).max(col_scratch))
            })?;
    let working_elements = input_value_elements
        // One point-sized output/product buffer coexists with input values.
        .checked_add(point_count)
        .and_then(|count| count.checked_add(candidate_frame_elements))
        .and_then(|count| count.checked_add(candidate_frame_scratch))
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

    let mut input_values = vec![T::default(); input_value_elements];
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.local_preparation += preparation_started.elapsed();
    });
    #[cfg(test)]
    let input_frames_started = std::time::Instant::now();
    // Per input, `input_values[.., point] = row_frames[row] . col_frames[col]`
    // is one (row_count x chi) times (chi x col_count) matrix product, not a
    // per-point scalar dot product: pack each side's candidate frame vectors
    // into packed dense matrices and let BLAS do the O(row*col*chi)
    // contraction in one `mat_mul_owned` call. The candidate-frame cache and
    // frame batching no longer create a Vec<Vec<T>> round trip; only the
    // row-side flat layout conversion and O(row*col) scatter remain plain
    // loops.
    if point_count > 0 {
        for input in 0..inputs.len() {
            #[cfg(test)]
            let row_frames_started = std::time::Instant::now();
            let row_input_frames = frames.candidate_frames_for_edge_rows(
                inputs,
                problem,
                input,
                forward,
                &row_candidates,
            )?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_row_frames += row_frames_started.elapsed();
            });
            #[cfg(test)]
            let col_frames_started = std::time::Instant::now();
            let col_input_frames = frames.candidate_frames_for_edge(
                inputs,
                problem,
                input,
                reverse,
                &col_candidates,
            )?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_col_frames += col_frames_started.elapsed();
            });
            #[cfg(test)]
            let pack_started = std::time::Instant::now();
            let bond_dim = row_input_frames.bond_dim();
            if col_input_frames.bond_dim() != bond_dim
                || row_input_frames.candidate_count() != row_count
                || col_input_frames.candidate_count() != col_count
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "packed input frames have inconsistent candidate dimensions",
                });
            }
            if bond_dim == 0 {
                continue;
            }
            #[cfg(test)]
            let use_legacy_frame_pack =
                std::env::var("T4A_TREEACI_USE_LEGACY_LOCAL_FRAME_PACK").as_deref() == Ok("1");
            #[cfg(test)]
            let (row_candidate_matrix, col_bond_matrix) = if use_legacy_frame_pack {
                // [AI Supplied] Diagnostic-only pre-#714 path. It exists only
                // for the paired release measurement and reproduces the old
                // per-candidate extraction plus two flat repacks.
                crate::state::profile_debug_stats::record(|stats| {
                    stats.local_legacy_frame_vectors += row_count + col_count;
                    stats.local_legacy_frame_values += (row_count + col_count) * bond_dim;
                });
                let row_frames = row_input_frames.to_candidate_vecs();
                let col_frames = col_input_frames.to_candidate_vecs();
                let mut row_flat = Vec::with_capacity(row_count * bond_dim);
                for bond in 0..bond_dim {
                    for frame in &row_frames {
                        row_flat.push(frame[bond]);
                    }
                }
                let col_flat = col_frames
                    .iter()
                    .flat_map(|frame| frame.iter().copied())
                    .collect::<Vec<_>>();
                (
                    Matrix::from_col_major_vec(row_count, bond_dim, row_flat),
                    Matrix::from_col_major_vec(bond_dim, col_count, col_flat),
                )
            } else {
                crate::state::profile_debug_stats::record(|stats| {
                    stats.local_packed_frame_batches += 2;
                    stats.local_packed_frame_values += (row_count + col_count) * bond_dim;
                });
                (
                    row_input_frames.into_candidate_by_bond_matrix(),
                    col_input_frames.into_bond_by_candidate_matrix(),
                )
            };
            #[cfg(not(test))]
            let row_candidate_matrix = row_input_frames.into_candidate_by_bond_matrix();
            #[cfg(not(test))]
            let col_bond_matrix = col_input_frames.into_bond_by_candidate_matrix();
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_pack += pack_started.elapsed();
            });
            #[cfg(test)]
            let matmul_started = std::time::Instant::now();
            // [AI Supplied] Keep the diagnostic A/B switch in tests while
            // production always consumes these short-lived matrices.
            #[cfg(test)]
            let product_result =
                if std::env::var("T4A_TREEACI_USE_OWNED_LOCAL_MATMUL").as_deref() == Ok("1") {
                    tensor4all_tensorbackend::mat_mul_owned(row_candidate_matrix, col_bond_matrix)
                } else {
                    mat_mul(&row_candidate_matrix, &col_bond_matrix)
                };
            #[cfg(not(test))]
            let product_result = mat_mul_owned(row_candidate_matrix, col_bond_matrix);
            let product = product_result.map_err(|error| TreeAciError::Numerical {
                message: error.to_string(),
            })?;
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_matmul += matmul_started.elapsed();
            });
            #[cfg(test)]
            let scatter_started = std::time::Instant::now();
            for col in 0..col_count {
                for row in 0..row_count {
                    let point = row + row_count * col;
                    input_values[input + inputs.len() * point] = product[[row, col]];
                }
            }
            #[cfg(test)]
            crate::state::profile_debug_stats::record(|stats| {
                stats.local_frame_scatter += scatter_started.elapsed();
            });
        }
    }
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.local_input_frames += input_frames_started.elapsed();
    });
    let batch = TreeElementwiseBatch::new(&input_values, inputs.len(), point_count)?;
    let mut local_values = vec![T::default(); point_count];
    #[cfg(test)]
    let operator_started = std::time::Instant::now();
    operator(batch, &mut local_values)?;
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.operator += operator_started.elapsed();
    });
    let sampled_scale = local_values.iter().copied().fold(0.0_f64, |scale, value| {
        scale.max(tensor4all_core::Scalar::abs_val(value))
    });
    #[cfg(test)]
    let retained_local_values = local_values.clone();
    let matrix = Matrix::from_col_major_vec(row_count, col_count, local_values);
    #[cfg(test)]
    let luci_started = std::time::Instant::now();
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
    #[cfg(test)]
    crate::state::profile_debug_stats::record(|stats| {
        stats.luci += luci_started.elapsed();
    });
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
    let row_samples = select_pivot_samples(row_indices, &row_candidates)?;
    let col_samples = select_pivot_samples(col_indices, &col_candidates)?;
    Ok(LocalUpdateResult {
        row_samples,
        col_samples,
        left,
        right,
        pivot_errors: factors.pivot_errors,
        sampled_scale,
        row_count,
        col_count,
        #[cfg(test)]
        local_values: retained_local_values,
    })
}

fn select_pivot_samples(
    indices: Vec<usize>,
    candidates: &[ComponentSample],
) -> Result<Vec<ComponentSample>> {
    indices
        .into_iter()
        .map(|index| {
            candidates
                .get(index)
                .cloned()
                .ok_or(TreeAciError::InternalInvariant {
                    message: "LUCI returned a pivot index outside the candidate matrix",
                })
        })
        .collect()
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

#[cfg(test)]
mod tests;
