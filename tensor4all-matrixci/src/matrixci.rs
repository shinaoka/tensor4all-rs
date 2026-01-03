//! Matrix Cross Interpolation (MatrixCI) implementation

use crate::error::{MatrixCIError, Result};
use crate::traits::AbstractMatrixCI;
use crate::util::{
    a_inv_times_b, a_times_b_inv, dot, from_vec2d, get_col, get_row, mat_mul, ncols, nrows,
    submatrix, submatrix_argmax, zeros, append_col, append_row, Matrix, Scalar,
};

/// Matrix Cross Interpolation representation
///
/// Represents a cross interpolation of a matrix A as:
/// A ≈ L * P^{-1} * R
///
/// where:
/// - L = A[:, J] (pivot columns)
/// - P = A[I, J] (pivot matrix)
/// - R = A[I, :] (pivot rows)
/// - I = row indices (rowindices)
/// - J = column indices (colindices)
#[derive(Debug, Clone)]
pub struct MatrixCI<T: Scalar> {
    /// Row indices (I set)
    row_indices: Vec<usize>,
    /// Column indices (J set)
    col_indices: Vec<usize>,
    /// Pivot columns: A[:, J] with shape (nrows, npivots)
    pivot_cols: Matrix<T>,
    /// Pivot rows: A[I, :] with shape (npivots, ncols)
    pivot_rows: Matrix<T>,
}

impl<T: Scalar> MatrixCI<T> {
    /// Create an empty MatrixCI for a matrix of given size
    pub fn new(nr: usize, nc: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            pivot_cols: zeros(nr, 0),
            pivot_rows: zeros(0, nc),
        }
    }

    /// Create a MatrixCI from existing parts
    ///
    /// # Arguments
    /// * `row_indices` - Row indices (I set)
    /// * `col_indices` - Column indices (J set)
    /// * `pivot_cols` - Pivot columns A[:, J]
    /// * `pivot_rows` - Pivot rows A[I, :]
    pub fn from_parts(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        pivot_cols: Matrix<T>,
        pivot_rows: Matrix<T>,
    ) -> Self {
        Self {
            row_indices,
            col_indices,
            pivot_cols,
            pivot_rows,
        }
    }

    /// Create a MatrixCI from a matrix with an initial pivot
    pub fn from_matrix_with_pivot(a: &Matrix<T>, first_pivot: (usize, usize)) -> Self {
        let (i, j) = first_pivot;
        let pivot_col = get_col(a, j);
        let pivot_row = get_row(a, i);

        let nr = nrows(a);
        let _nc = ncols(a);

        let pivot_cols = from_vec2d(
            (0..nr).map(|r| vec![pivot_col[r]]).collect()
        );

        let pivot_rows = from_vec2d(vec![pivot_row]);

        Self {
            row_indices: vec![i],
            col_indices: vec![j],
            pivot_cols,
            pivot_rows,
        }
    }

    /// Get I set (row indices)
    pub fn iset(&self) -> &[usize] {
        &self.row_indices
    }

    /// Get J set (column indices)
    pub fn jset(&self) -> &[usize] {
        &self.col_indices
    }

    /// Get the pivot matrix P = A[I, J]
    pub fn pivot_matrix(&self) -> Matrix<T> {
        let ranks: Vec<usize> = (0..self.rank()).collect();
        submatrix(&self.pivot_cols, &self.row_indices, &ranks)
    }

    /// Get the left matrix L = A[:, J] * P^{-1}
    pub fn left_matrix(&self) -> Matrix<T> {
        if self.is_empty() {
            return zeros(self.nrows(), 0);
        }
        a_times_b_inv(&self.pivot_cols, &self.pivot_matrix())
    }

    /// Get the right matrix R = P^{-1} * A[I, :]
    pub fn right_matrix(&self) -> Matrix<T> {
        if self.is_empty() {
            return zeros(0, self.ncols());
        }
        a_inv_times_b(&self.pivot_matrix(), &self.pivot_rows)
    }

    /// Get the value of the first pivot
    pub fn first_pivot_value(&self) -> T {
        if self.is_empty() {
            T::one()
        } else {
            self.pivot_cols[[self.row_indices[0], 0]]
        }
    }

    /// Add a pivot row
    pub fn add_pivot_row(&mut self, a: &Matrix<T>, row_index: usize) -> Result<()> {
        if nrows(a) != self.nrows() || ncols(a) != self.ncols() {
            return Err(MatrixCIError::DimensionMismatch {
                expected_rows: self.nrows(),
                expected_cols: self.ncols(),
                actual_rows: nrows(a),
                actual_cols: ncols(a),
            });
        }

        if row_index >= self.nrows() {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: row_index,
                col: 0,
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }

        if self.row_indices.contains(&row_index) {
            return Err(MatrixCIError::DuplicatePivotRow { row: row_index });
        }

        let row = get_row(a, row_index);
        self.pivot_rows = append_row(&self.pivot_rows, &row);
        self.row_indices.push(row_index);

        Ok(())
    }

    /// Add a pivot column
    pub fn add_pivot_col(&mut self, a: &Matrix<T>, col_index: usize) -> Result<()> {
        if nrows(a) != self.nrows() || ncols(a) != self.ncols() {
            return Err(MatrixCIError::DimensionMismatch {
                expected_rows: self.nrows(),
                expected_cols: self.ncols(),
                actual_rows: nrows(a),
                actual_cols: ncols(a),
            });
        }

        if col_index >= self.ncols() {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: 0,
                col: col_index,
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }

        if self.col_indices.contains(&col_index) {
            return Err(MatrixCIError::DuplicatePivotCol { col: col_index });
        }

        let col = get_col(a, col_index);
        self.pivot_cols = append_col(&self.pivot_cols, &col);
        self.col_indices.push(col_index);

        Ok(())
    }

    /// Add a pivot at the given position
    pub fn add_pivot(&mut self, a: &Matrix<T>, pivot: (usize, usize)) -> Result<()> {
        let (i, j) = pivot;

        if nrows(a) != self.nrows() || ncols(a) != self.ncols() {
            return Err(MatrixCIError::DimensionMismatch {
                expected_rows: self.nrows(),
                expected_cols: self.ncols(),
                actual_rows: nrows(a),
                actual_cols: ncols(a),
            });
        }

        if i >= self.nrows() || j >= self.ncols() {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: i,
                col: j,
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }

        if self.row_indices.contains(&i) {
            return Err(MatrixCIError::DuplicatePivotRow { row: i });
        }

        if self.col_indices.contains(&j) {
            return Err(MatrixCIError::DuplicatePivotCol { col: j });
        }

        self.add_pivot_row(a, i)?;
        self.add_pivot_col(a, j)?;

        Ok(())
    }

    /// Add a pivot that maximizes the error
    pub fn add_best_pivot(&mut self, a: &Matrix<T>) -> Result<(usize, usize)> {
        let ((i, j), _) = self.find_new_pivot(a)?;
        self.add_pivot(a, (i, j))?;
        Ok((i, j))
    }
}

impl<T: Scalar> AbstractMatrixCI<T> for MatrixCI<T> {
    fn nrows(&self) -> usize {
        nrows(&self.pivot_cols)
    }

    fn ncols(&self) -> usize {
        ncols(&self.pivot_rows)
    }

    fn rank(&self) -> usize {
        self.row_indices.len()
    }

    fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    fn evaluate(&self, i: usize, j: usize) -> T {
        if self.is_empty() {
            return T::zero();
        }

        let left = self.left_matrix();
        let left_row: Vec<T> = (0..ncols(&left)).map(|k| left[[i, k]]).collect();
        let pivot_col: Vec<T> = (0..nrows(&self.pivot_rows))
            .map(|k| self.pivot_rows[[k, j]])
            .collect();

        dot(&left_row, &pivot_col)
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {
        if self.is_empty() {
            return zeros(rows.len(), cols.len());
        }

        let left = self.left_matrix();
        let ranks: Vec<usize> = (0..self.rank()).collect();
        let left_sub = submatrix(&left, rows, &ranks);

        let pivot_sub = submatrix(&self.pivot_rows, &ranks, cols);

        mat_mul(&left_sub, &pivot_sub)
    }
}

/// Options for cross interpolation
#[derive(Debug, Clone)]
pub struct CrossInterpolateOptions {
    /// Tolerance for stopping criterion
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Default for CrossInterpolateOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            max_iter: 200,
        }
    }
}

/// Perform cross interpolation of a matrix
pub fn crossinterpolate<T: Scalar>(
    a: &Matrix<T>,
    options: Option<CrossInterpolateOptions>,
) -> MatrixCI<T> {
    let opts = options.unwrap_or_default();

    // Find initial pivot (maximum absolute value)
    let rows: Vec<usize> = (0..nrows(a)).collect();
    let cols: Vec<usize> = (0..ncols(a)).collect();
    let (first_i, first_j, _) = submatrix_argmax(a, &rows, &cols);

    let mut ci = MatrixCI::from_matrix_with_pivot(a, (first_i, first_j));

    for _ in 1..opts.max_iter {
        match ci.find_new_pivot(a) {
            Ok(((i, j), error)) => {
                let error_val: f64 = error.abs_sq();
                let tol_sq = opts.tolerance * opts.tolerance;
                if error_val < tol_sq {
                    break;
                }
                let _ = ci.add_pivot(a, (i, j));
            }
            Err(_) => break,
        }
    }

    ci
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrixci_new() {
        let ci = MatrixCI::<f64>::new(5, 5);
        assert_eq!(ci.nrows(), 5);
        assert_eq!(ci.ncols(), 5);
        assert_eq!(ci.rank(), 0);
        assert!(ci.is_empty());
    }

    #[test]
    fn test_matrixci_from_matrix() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let ci = MatrixCI::from_matrix_with_pivot(&m, (2, 2));
        assert_eq!(ci.nrows(), 3);
        assert_eq!(ci.ncols(), 3);
        assert_eq!(ci.rank(), 1);
        assert_eq!(ci.row_indices(), &[2]);
        assert_eq!(ci.col_indices(), &[2]);
    }

    #[test]
    fn test_matrixci_add_pivot() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let mut ci = MatrixCI::from_matrix_with_pivot(&m, (0, 0));
        assert!(ci.add_pivot(&m, (1, 1)).is_ok());
        assert_eq!(ci.rank(), 2);
    }

    #[test]
    fn test_crossinterpolate() {
        // Create a rank-1 matrix
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ]);

        let ci = crossinterpolate(&m, None);

        // For a rank-1 matrix, should need only 1 pivot
        assert!(ci.rank() >= 1);

        // Check approximation quality
        let approx = ci.to_matrix();
        for i in 0..3 {
            for j in 0..3 {
                let diff = (m[[i, j]] - approx[[i, j]]).abs();
                assert!(diff < 1e-10, "Approximation error too large at ({}, {})", i, j);
            }
        }
    }
}
