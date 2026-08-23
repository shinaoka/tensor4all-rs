//! Lazy pivot-kernel implementations.

use crate::matrixluci::error::MatrixLuciError;
use crate::matrixluci::factors::{load_block, subtract_inplace};
use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::{PivotKernelOptions, PivotSelectionCore};
use crate::matrixluci::Result;
use num_complex::{Complex32, Complex64};
use tensor4all_tensorbackend::{mat_mul, solve_matrix, Matrix};

/// Lazy pivot kernel based on residual row/column rook search.
///
/// Selects pivots by computing residual blocks on demand, avoiding full
/// matrix materialization. Suitable for large matrices accessed via
/// [`LazyMatrixSource`](super::LazyMatrixSource).
#[derive(Default)]
pub struct LazyBlockRookKernel;

fn residual_block<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> Result<Matrix<T>> {
    let mut residual = load_block(source, rows, cols);
    if selected_rows.is_empty() {
        return Ok(residual);
    }

    let pivot = load_block(source, selected_rows, selected_cols);
    let a_rj = load_block(source, rows, selected_cols);
    let a_ic = load_block(source, selected_rows, cols);
    let solved = solve_matrix(&pivot, &a_ic).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual pivot solve failed: {err}"),
    })?;
    let approx = mat_mul(&a_rj, &solved).map_err(|err| MatrixLuciError::InvalidArgument {
        message: format!("residual multiplication failed: {err}"),
    })?;
    subtract_inplace(&mut residual, &approx);
    Ok(residual)
}

fn argmax_abs<T: MatrixLuciScalar>(matrix: &Matrix<T>) -> (usize, usize, f64) {
    let mut best_row = 0usize;
    let mut best_col = 0usize;
    let mut best_abs = -1.0f64;
    for col in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            let value = matrix[[row, col]].abs_val();
            if value > best_abs {
                best_row = row;
                best_col = col;
                best_abs = value;
            }
        }
    }
    (best_row, best_col, best_abs.max(0.0))
}

fn remaining_indices(total: usize, selected: &[usize]) -> Vec<usize> {
    let mut used = vec![false; total];
    for &idx in selected {
        used[idx] = true;
    }
    (0..total).filter(|&idx| !used[idx]).collect()
}

fn rook_pivot<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    remaining_rows: &[usize],
    remaining_cols: &[usize],
    selected_rows: &[usize],
    selected_cols: &[usize],
) -> Result<(usize, usize, f64)> {
    let mut current_col = remaining_cols[0];
    let mut current_row = remaining_rows[0];
    let max_steps = remaining_rows.len() + remaining_cols.len() + 1;

    for _ in 0..max_steps {
        let col_residual = residual_block(
            source,
            remaining_rows,
            &[current_col],
            selected_rows,
            selected_cols,
        )?;
        let (best_row_pos, _, _) = argmax_abs(&col_residual);
        current_row = remaining_rows[best_row_pos];

        let row_residual = residual_block(
            source,
            &[current_row],
            remaining_cols,
            selected_rows,
            selected_cols,
        )?;
        let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
        let next_col = remaining_cols[best_col_pos];

        if next_col == current_col {
            return Ok((current_row, current_col, best_abs));
        }
        current_col = next_col;
    }

    let row_residual = residual_block(
        source,
        &[current_row],
        remaining_cols,
        selected_rows,
        selected_cols,
    )?;
    let (_, best_col_pos, best_abs) = argmax_abs(&row_residual);
    Ok((current_row, remaining_cols[best_col_pos], best_abs))
}

fn factorize_lazy<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    options: &PivotKernelOptions,
) -> Result<PivotSelectionCore> {
    let nrows = source.nrows();
    let ncols = source.ncols();
    let full_rank = nrows.min(ncols);
    if full_rank == 0 {
        return Ok(PivotSelectionCore {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            pivot_errors: vec![0.0],
            rank: 0,
        });
    }

    let max_bond_dim = options.max_bond_dim.min(full_rank);
    let mut selected_rows = Vec::with_capacity(max_bond_dim);
    let mut selected_cols = Vec::with_capacity(max_bond_dim);
    let mut accepted = Vec::with_capacity(max_bond_dim + 1);
    let mut max_error = 0.0f64;
    let mut last_error = f64::NAN;

    while selected_rows.len() < max_bond_dim {
        let remaining_rows = remaining_indices(nrows, &selected_rows);
        let remaining_cols = remaining_indices(ncols, &selected_cols);
        if remaining_rows.is_empty() || remaining_cols.is_empty() {
            break;
        }

        let (pivot_row, pivot_col, pivot_abs) = rook_pivot(
            source,
            &remaining_rows,
            &remaining_cols,
            &selected_rows,
            &selected_cols,
        )?;
        last_error = pivot_abs;

        if !selected_rows.is_empty()
            && (pivot_abs < options.rel_tol * max_error || pivot_abs < options.abs_tol)
        {
            break;
        }

        if pivot_abs < T::epsilon() {
            if selected_rows.is_empty() {
                last_error = pivot_abs;
            }
            break;
        }

        max_error = max_error.max(pivot_abs);
        selected_rows.push(pivot_row);
        selected_cols.push(pivot_col);
        accepted.push(pivot_abs);
    }

    let rank = selected_rows.len();
    if rank >= full_rank {
        last_error = 0.0;
    } else if rank == max_bond_dim && rank > 0 {
        last_error = accepted[rank - 1];
    }
    accepted.push(last_error);

    Ok(PivotSelectionCore {
        row_indices: selected_rows,
        col_indices: selected_cols,
        pivot_errors: accepted,
        rank,
    })
}

macro_rules! impl_lazy_block_rook_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for LazyBlockRookKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                factorize_lazy(source, options)
            }
        }
    };
}

impl_lazy_block_rook_kernel!(f32);
impl_lazy_block_rook_kernel!(f64);
impl_lazy_block_rook_kernel!(Complex32);
impl_lazy_block_rook_kernel!(Complex64);

#[cfg(test)]
mod tests;
