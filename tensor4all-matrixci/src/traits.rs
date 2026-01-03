//! Abstract traits for matrix cross interpolation

use crate::error::Result;
use crate::util::{submatrix, zeros, Matrix, Scalar};

/// Abstract trait for matrix cross interpolation objects
///
/// This trait provides a common interface for different matrix CI implementations
/// including MatrixCI, MatrixACA, and MatrixLUCI.
pub trait AbstractMatrixCI<T: Scalar>: Sized {
    /// Number of rows in the approximated matrix
    fn nrows(&self) -> usize;

    /// Number of columns in the approximated matrix
    fn ncols(&self) -> usize;

    /// Current rank of the approximation (number of pivots)
    fn rank(&self) -> usize;

    /// Row indices selected as pivots (I set)
    fn row_indices(&self) -> &[usize];

    /// Column indices selected as pivots (J set)
    fn col_indices(&self) -> &[usize];

    /// Check if the approximation is empty (no pivots)
    fn is_empty(&self) -> bool {
        self.rank() == 0
    }

    /// Evaluate the approximation at position (i, j)
    fn evaluate(&self, i: usize, j: usize) -> T;

    /// Get a submatrix of the approximation
    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T>;

    /// Get a row of the approximation
    fn row(&self, i: usize) -> Vec<T> {
        let cols: Vec<usize> = (0..self.ncols()).collect();
        let sub = self.submatrix(&[i], &cols);
        (0..self.ncols()).map(|j| sub[[0, j]]).collect()
    }

    /// Get a column of the approximation
    fn col(&self, j: usize) -> Vec<T> {
        let rows: Vec<usize> = (0..self.nrows()).collect();
        let sub = self.submatrix(&rows, &[j]);
        (0..self.nrows()).map(|i| sub[[i, 0]]).collect()
    }

    /// Get the full approximated matrix
    fn to_matrix(&self) -> Matrix<T> {
        let rows: Vec<usize> = (0..self.nrows()).collect();
        let cols: Vec<usize> = (0..self.ncols()).collect();
        self.submatrix(&rows, &cols)
    }

    /// Get available row indices (rows without pivots)
    fn available_rows(&self) -> Vec<usize> {
        let pivot_rows: std::collections::HashSet<usize> =
            self.row_indices().iter().copied().collect();
        (0..self.nrows())
            .filter(|i| !pivot_rows.contains(i))
            .collect()
    }

    /// Get available column indices (columns without pivots)
    fn available_cols(&self) -> Vec<usize> {
        let pivot_cols: std::collections::HashSet<usize> =
            self.col_indices().iter().copied().collect();
        (0..self.ncols())
            .filter(|j| !pivot_cols.contains(j))
            .collect()
    }

    /// Compute local error |A - CI| for given indices
    fn local_error(&self, a: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T>
    where
        T: std::ops::Sub<Output = T>,
    {
        let sub_a = submatrix(a, rows, cols);
        let sub_ci = self.submatrix(rows, cols);

        let mut result = zeros(rows.len(), cols.len());
        for i in 0..rows.len() {
            for j in 0..cols.len() {
                let diff = sub_a[[i, j]] - sub_ci[[i, j]];
                result[[i, j]] = diff.abs();
            }
        }
        result
    }

    /// Find a new pivot that maximizes the local error
    fn find_new_pivot(&self, a: &Matrix<T>) -> Result<((usize, usize), T)>
    where
        T: std::ops::Sub<Output = T>,
    {
        let avail_rows = self.available_rows();
        let avail_cols = self.available_cols();

        self.find_new_pivot_in(a, &avail_rows, &avail_cols)
    }

    /// Find a new pivot in the given row/column subsets
    fn find_new_pivot_in(
        &self,
        a: &Matrix<T>,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<((usize, usize), T)>
    where
        T: std::ops::Sub<Output = T>,
    {
        use crate::error::MatrixCIError;

        if self.rank() == self.nrows().min(self.ncols()) {
            return Err(MatrixCIError::FullRank);
        }

        if rows.is_empty() {
            return Err(MatrixCIError::EmptyIndexSet {
                dimension: "rows".to_string(),
            });
        }

        if cols.is_empty() {
            return Err(MatrixCIError::EmptyIndexSet {
                dimension: "cols".to_string(),
            });
        }

        let errors = self.local_error(a, rows, cols);

        // Find maximum error position (comparing by abs_sq which returns f64)
        let mut max_val_sq: f64 = errors[[0, 0]].abs_sq();
        let mut max_i = 0;
        let mut max_j = 0;

        for i in 0..rows.len() {
            for j in 0..cols.len() {
                let val_sq: f64 = errors[[i, j]].abs_sq();
                if val_sq > max_val_sq {
                    max_val_sq = val_sq;
                    max_i = i;
                    max_j = j;
                }
            }
        }

        Ok(((rows[max_i], cols[max_j]), errors[[max_i, max_j]]))
    }
}
