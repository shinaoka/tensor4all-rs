use crate::{
    assemble::{assemble_points_column_major, MultiIndex},
    assemble_global_point,
    batch::GlobalIndexBatch,
    PivotCandidateProposer, TreeTCI2, TreeTciEdge,
};
use anyhow::{ensure, Result};
use tensor4all_core::{ColMajorArray, CommonScalar};
use tensor4all_tcicore::{
    matrix_luci_factors_from_matrix, MatrixLuciFactors, MatrixLuciScalar as Scalar, RrLUOptions,
};
use tensor4all_tensorbackend::Matrix;

#[cfg(test)]
use crate::DefaultProposer;

/// Update one edge bipartition using a batch evaluator and a pivot-candidate proposer.
///
/// Evaluates the function at candidate pivot points for one edge bipartition,
/// then selects pivots via LU decomposition. The selected pivots are stored
/// back into `state.ijset`.
///
/// This is a low-level building block; prefer [`optimize_default`](crate::optimize_default)
/// or [`crossinterpolate2`](crate::crossinterpolate2) for typical usage.
pub(crate) fn update_edge<T, F, P>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &RrLUOptions,
    proposer: &P,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
    P: PivotCandidateProposer,
{
    let (left_key, right_key) = state.graph.subregion_vertices(edge)?;
    let (left_candidates, right_candidates) = proposer.candidates(state, edge)?;
    if left_candidates.is_empty() || right_candidates.is_empty() {
        return Err(anyhow::anyhow!(
            "proposer returned empty candidate list for edge {edge:?}",
        ));
    }
    let values = evaluate_candidate_matrix(
        state.local_dims.len(),
        &left_key,
        &left_candidates,
        &right_key,
        &right_candidates,
        evaluate,
    )?;

    for value in &values {
        state.max_sample_value = state.max_sample_value.max(CommonScalar::abs_val(*value));
    }

    let mut matrix = Matrix::zeros(left_candidates.len(), right_candidates.len());
    for col in 0..right_candidates.len() {
        for row in 0..left_candidates.len() {
            matrix[[row, col]] = values[row + left_candidates.len() * col];
        }
    }
    let selection = matrix_luci_factors_from_matrix(&matrix, Some(options.clone()))?;

    // The LU can select zero pivots when the sampled submatrix is numerically
    // zero (the function underflows in a subdomain). Keep at least one index
    // so the stored pivot sets never become empty and the state stays
    // materializable; the resulting rank-1 zero core is the correct
    // approximation of a zero subdomain. Candidate lists are validated
    // non-empty above.
    let row_indices = if selection.row_indices.is_empty() {
        vec![0]
    } else {
        selection.row_indices.clone()
    };
    let col_indices = if selection.col_indices.is_empty() {
        vec![0]
    } else {
        selection.col_indices.clone()
    };

    // Build ColMajorArray from selected pivot indices
    let n_left_sites = left_key.as_slice().len();
    let n_right_sites = right_key.as_slice().len();

    let left_data: Vec<usize> = row_indices
        .iter()
        .flat_map(|&row| left_candidates[row].iter().copied())
        .collect();
    let left_arr = ColMajorArray::new(left_data, vec![n_left_sites, row_indices.len()])?;
    state.ijset.insert(left_key.clone(), left_arr);

    let right_data: Vec<usize> = col_indices
        .iter()
        .flat_map(|&col| right_candidates[col].iter().copied())
        .collect();
    let right_arr = ColMajorArray::new(right_data, vec![n_right_sites, col_indices.len()])?;
    state.ijset.insert(right_key.clone(), right_arr);

    let last_error = selection.pivot_errors.last().copied().unwrap_or(0.0);
    state.update_bond_error(edge, last_error);
    state.update_pivot_errors(&selection.pivot_errors);

    Ok(selection)
}

/// Update one edge using the default proposer.
///
/// Convenience wrapper around [`update_edge`] that uses [`DefaultProposer`].
#[cfg(test)]
pub(crate) fn update_edge_default<T, F>(
    state: &mut TreeTCI2<T>,
    edge: TreeTciEdge,
    evaluate: F,
    options: &RrLUOptions,
) -> Result<MatrixLuciFactors<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    update_edge(state, edge, evaluate, options, &DefaultProposer)
}

fn evaluate_candidate_matrix<T, F>(
    n_sites: usize,
    left_key: &crate::SubtreeKey,
    left_candidates: &[MultiIndex],
    right_key: &crate::SubtreeKey,
    right_candidates: &[MultiIndex],
    evaluate: F,
) -> Result<Vec<T>>
where
    T: Scalar + CommonScalar,
    F: Fn(GlobalIndexBatch<'_>) -> Result<Vec<T>>,
{
    let mut points = Vec::with_capacity(left_candidates.len() * right_candidates.len());
    for right in right_candidates {
        for left in left_candidates {
            points.push(assemble_global_point(
                n_sites,
                &[(left_key, left), (right_key, right)],
                &[],
            )?);
        }
    }
    let batch = assemble_points_column_major(&points)?;
    let values = evaluate(batch.as_view())?;
    ensure!(
        values.len() == left_candidates.len() * right_candidates.len(),
        "batch evaluator returned {} values for {} candidate-matrix entries",
        values.len(),
        left_candidates.len() * right_candidates.len()
    );
    Ok(values)
}

#[cfg(test)]
mod tests;
