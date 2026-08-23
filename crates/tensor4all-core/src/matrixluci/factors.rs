//! Cross-factor reconstruction helpers.
//!
//! [`CrossFactors`] gathers the pivot block, pivot columns, and pivot rows
//! from a [`CandidateMatrixSource`], and
//! provides methods to compute the left and right CI factors.

use crate::matrixluci::error::MatrixLuciError;
use crate::matrixluci::scalar::MatrixLuciScalar;
use crate::matrixluci::source::CandidateMatrixSource;
use crate::matrixluci::types::PivotSelectionCore;
use crate::matrixluci::Result;
use tensor4all_tensorbackend::{solve_matrix, transpose, Matrix};

/// Gather a dense column-major block from a source.
pub(crate) fn load_block<T: MatrixLuciScalar, S: CandidateMatrixSource<T>>(
    source: &S,
    rows: &[usize],
    cols: &[usize],
) -> Matrix<T> {
    let mut data = vec![T::zero(); rows.len() * cols.len()];
    source.get_block(rows, cols, &mut data);
    Matrix::from_col_major_vec(rows.len(), cols.len(), data)
}

/// Subtract one dense matrix from another in place.
pub(crate) fn subtract_inplace<T: MatrixLuciScalar>(lhs: &mut Matrix<T>, rhs: &Matrix<T>) {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    for j in 0..lhs.ncols() {
        for i in 0..lhs.nrows() {
            lhs[[i, j]] = lhs[[i, j]] - rhs[[i, j]];
        }
    }
}

/// Dense factors derived from a pivot selection.
///
/// Contains the pivot block `A[I, J]`, full pivot columns `A[:, J]`, and
/// full pivot rows `A[I, :]`. These are used to reconstruct the left and
/// right CI factors for the cross interpolation approximation
/// `A ~ A[:, J] * A[I, J]^{-1} * A[I, :]`.
#[derive(Debug, Clone)]
pub(crate) struct CrossFactors<T: MatrixLuciScalar> {
    /// Pivot block `A[I, J]`.
    pub pivot: Matrix<T>,
    /// Columns through selected pivot columns `A[:, J]`.
    pub pivot_cols: Matrix<T>,
    /// Rows through selected pivot rows `A[I, :]`.
    pub pivot_rows: Matrix<T>,
}

impl<T: MatrixLuciScalar> CrossFactors<T> {
    /// Gather a dense block from a source.
    #[cfg(test)]
    pub fn gather<S: CandidateMatrixSource<T>>(
        source: &S,
        rows: &[usize],
        cols: &[usize],
    ) -> Matrix<T> {
        load_block(source, rows, cols)
    }

    /// Reconstruct factors from a source and pivot-only selection.
    /// # Errors
    ///
    /// Returns an error when the matrix source is invalid (a shape mismatch) or
    /// /// the factor construction fails.
    ///
    pub fn from_source<S: CandidateMatrixSource<T>>(
        source: &S,
        selection: &PivotSelectionCore,
    ) -> Result<Self> {
        let all_rows: Vec<usize> = (0..source.nrows()).collect();
        let all_cols: Vec<usize> = (0..source.ncols()).collect();
        Ok(Self {
            pivot: load_block(source, &selection.row_indices, &selection.col_indices),
            pivot_cols: load_block(source, &all_rows, &selection.col_indices),
            pivot_rows: load_block(source, &selection.row_indices, &all_cols),
        })
    }

    /// Solve for `A[:, J] * A[I, J]^{-1}` without forming an explicit inverse.
    /// # Errors
    ///
    /// Returns an error when the pivot solve fails (a singular or backend
    /// /// failure).
    ///
    pub fn cols_solve_pivot(&self) -> Result<Matrix<T>> {
        let pivot_t = transpose(&self.pivot);
        let pivot_cols_t = transpose(&self.pivot_cols);
        let solved_t = solve_matrix(&pivot_t, &pivot_cols_t).map_err(|err| {
            MatrixLuciError::InvalidArgument {
                message: format!("left factor solve failed: {err}"),
            }
        })?;
        Ok(transpose(&solved_t))
    }

    /// Solve for `A[I, J]^{-1} * A[I, :]` without forming an explicit inverse.
    /// # Errors
    ///
    /// Returns an error when the pivot solve fails (a singular or backend
    /// /// failure).
    ///
    pub fn solve_pivot_rows(&self) -> Result<Matrix<T>> {
        solve_matrix(&self.pivot, &self.pivot_rows).map_err(|err| {
            MatrixLuciError::InvalidArgument {
                message: format!("right factor solve failed: {err}"),
            }
        })
    }
}

#[cfg(test)]
mod tests;
