//! Abstract traits for matrix cross interpolation.
//!
//! The [`AbstractMatrixCI`] trait provides a common interface for all matrix
//! cross interpolation implementations ([`MatrixLUCI`](crate::MatrixLUCI),
//! [`MatrixACA`](crate::MatrixACA)).

use crate::error::Result;
use crate::scalar::Scalar;
use tensor4all_tensorbackend::{submatrix, Matrix};

/// Common interface for matrix cross interpolation objects.
///
/// Implementors provide low-rank approximations of matrices via
/// selected pivot rows and columns. This trait unifies the API for
/// [`MatrixLUCI`](crate::MatrixLUCI) and [`MatrixACA`](crate::MatrixACA).
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
///
/// // All AbstractMatrixCI methods are available:
/// assert_eq!(ci.nrows(), 2);
/// assert_eq!(ci.ncols(), 2);
/// assert!(ci.rank() >= 1);
/// assert!(!ci.is_empty());
///
/// // Full reconstruction
/// let full = ci.to_matrix();
/// for i in 0..2 {
///     for j in 0..2 {
///         assert!((full[[i, j]] - m[[i, j]]).abs() < 1e-10);
///     }
/// }
/// ```
pub trait AbstractMatrixCI<T: Scalar>: Sized {
    /// Number of rows in the approximated matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert_eq!(ci.nrows(), 3);
    /// ```
    fn nrows(&self) -> usize;

    /// Number of columns in the approximated matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert_eq!(ci.ncols(), 3);
    /// ```
    fn ncols(&self) -> usize;

    /// Current rank of the approximation (number of pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert!(ci.rank() >= 1);
    /// assert!(ci.rank() <= 2);
    /// ```
    fn rank(&self) -> usize;

    /// Row indices selected as pivots (I set)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let rows = ci.row_indices();
    /// assert_eq!(rows.len(), ci.rank());
    /// for &r in rows {
    ///     assert!(r < ci.nrows());
    /// }
    /// ```
    fn row_indices(&self) -> &[usize];

    /// Column indices selected as pivots (J set)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let cols = ci.col_indices();
    /// assert_eq!(cols.len(), ci.rank());
    /// for &c in cols {
    ///     assert!(c < ci.ncols());
    /// }
    /// ```
    fn col_indices(&self) -> &[usize];

    /// Check if the approximation is empty (no pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    ///
    /// let aca = MatrixACA::<f64>::new(2, 2);
    /// assert!(aca.is_empty());
    /// ```
    fn is_empty(&self) -> bool {
        self.rank() == 0
    }

    /// Evaluate the approximation at position (i, j)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// assert!((ci.evaluate(0, 0) - 1.0).abs() < 1e-10);
    /// assert!((ci.evaluate(1, 1) - 4.0).abs() < 1e-10);
    /// ```
    fn evaluate(&self, i: usize, j: usize) -> T;

    /// Get a submatrix of the approximation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    ///     vec![7.0, 8.0, 9.0],
    /// ]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let sub = ci.submatrix(&[0, 2], &[1]);
    /// assert_eq!(sub.nrows(), 2);
    /// assert_eq!(sub.ncols(), 1);
    /// assert!((sub[[0, 0]] - 2.0).abs() < 1e-10);
    /// assert!((sub[[1, 0]] - 8.0).abs() < 1e-10);
    /// ```
    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T>;

    /// Get a row of the approximation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let row0 = ci.row(0);
    /// assert_eq!(row0.len(), 2);
    /// assert!((row0[0] - 1.0).abs() < 1e-10);
    /// assert!((row0[1] - 2.0).abs() < 1e-10);
    /// ```
    fn row(&self, i: usize) -> Vec<T> {
        let cols: Vec<usize> = (0..self.ncols()).collect();
        let sub = self.submatrix(&[i], &cols);
        (0..self.ncols()).map(|j| sub[[0, j]]).collect()
    }

    /// Get a column of the approximation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let col1 = ci.col(1);
    /// assert_eq!(col1.len(), 2);
    /// assert!((col1[0] - 2.0).abs() < 1e-10);
    /// assert!((col1[1] - 4.0).abs() < 1e-10);
    /// ```
    fn col(&self, j: usize) -> Vec<T> {
        let rows: Vec<usize> = (0..self.nrows()).collect();
        let sub = self.submatrix(&rows, &[j]);
        (0..self.nrows()).map(|i| sub[[i, 0]]).collect()
    }

    /// Get the full approximated matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
    /// let full = ci.to_matrix();
    /// assert_eq!(full.nrows(), 2);
    /// assert_eq!(full.ncols(), 2);
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((full[[i, j]] - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    fn to_matrix(&self) -> Matrix<T> {
        let rows: Vec<usize> = (0..self.nrows()).collect();
        let cols: Vec<usize> = (0..self.ncols()).collect();
        self.submatrix(&rows, &cols)
    }

    /// Get available row indices (rows without pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (1, 0)).unwrap();
    /// let avail = aca.available_rows();
    /// // Row 1 was used as pivot, so 0 and 2 remain
    /// assert_eq!(avail, vec![0, 2]);
    /// ```
    fn available_rows(&self) -> Vec<usize> {
        let pivot_rows: std::collections::HashSet<usize> =
            self.row_indices().iter().copied().collect();
        (0..self.nrows())
            .filter(|i| !pivot_rows.contains(i))
            .collect()
    }

    /// Get available column indices (columns without pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 1)).unwrap();
    /// let avail = aca.available_cols();
    /// // Column 1 was used as pivot, so 0 and 2 remain
    /// assert_eq!(avail, vec![0, 2]);
    /// ```
    fn available_cols(&self) -> Vec<usize> {
        let pivot_cols: std::collections::HashSet<usize> =
            self.col_indices().iter().copied().collect();
        (0..self.ncols())
            .filter(|j| !pivot_cols.contains(j))
            .collect()
    }

    /// Compute local error |A - CI| for given indices
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    ///     vec![7.0, 8.0, 10.0],
    /// ]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// let err = aca.local_error(&m, &[1, 2], &[1, 2]);
    /// // Error at pivot position (0,0) would be zero; off-pivot may be non-zero
    /// assert_eq!(err.nrows(), 2);
    /// assert_eq!(err.ncols(), 2);
    /// ```
    fn local_error(&self, a: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T>
    where
        T: std::ops::Sub<Output = T>,
    {
        let sub_a = submatrix(a, rows, cols);
        let sub_ci = self.submatrix(rows, cols);

        let mut result = Matrix::zeros(rows.len(), cols.len());
        for i in 0..rows.len() {
            for j in 0..cols.len() {
                let diff = sub_a[[i, j]] - sub_ci[[i, j]];
                result[[i, j]] = diff.abs();
            }
        }
        result
    }

    /// Find a new pivot that maximizes the local error
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    ///     vec![7.0, 8.0, 10.0],
    /// ]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// let ((r, c), err_val) = aca.find_new_pivot(&m).unwrap();
    /// // New pivot must be in available rows/cols (not row 0 or col 0)
    /// assert_ne!(r, 0);
    /// assert_ne!(c, 0);
    /// ```
    fn find_new_pivot(&self, a: &Matrix<T>) -> Result<((usize, usize), T)>
    where
        T: std::ops::Sub<Output = T>,
    {
        let avail_rows = self.available_rows();
        let avail_cols = self.available_cols();

        self.find_new_pivot_in(a, &avail_rows, &avail_cols)
    }

    /// Find a new pivot in the given row/column subsets
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![
    ///     vec![1.0_f64, 2.0, 3.0],
    ///     vec![4.0, 5.0, 6.0],
    ///     vec![7.0, 8.0, 10.0],
    /// ]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// // Search only in rows [1,2] and cols [1,2]
    /// let ((r, c), _) = aca.find_new_pivot_in(&m, &[1, 2], &[1, 2]).unwrap();
    /// assert!(r == 1 || r == 2);
    /// assert!(c == 1 || c == 2);
    /// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MatrixCIError;
    use tensor4all_tensorbackend::from_vec2d;

    struct ExactCi {
        matrix: Matrix<f64>,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    }

    impl ExactCi {
        fn new(matrix: Matrix<f64>, row_indices: Vec<usize>, col_indices: Vec<usize>) -> Self {
            Self {
                matrix,
                row_indices,
                col_indices,
            }
        }
    }

    impl AbstractMatrixCI<f64> for ExactCi {
        fn nrows(&self) -> usize {
            self.matrix.nrows()
        }

        fn ncols(&self) -> usize {
            self.matrix.ncols()
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

        fn evaluate(&self, i: usize, j: usize) -> f64 {
            self.matrix[[i, j]]
        }

        fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<f64> {
            tensor4all_tensorbackend::submatrix(&self.matrix, rows, cols)
        }
    }

    fn sample_matrix() -> Matrix<f64> {
        from_vec2d(vec![
            vec![1.0_f64, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ])
    }

    #[test]
    fn default_helpers_return_expected_views_and_available_indices() {
        let matrix = sample_matrix();
        let ci = ExactCi::new(matrix.clone(), vec![1], vec![2]);

        assert!(!ci.is_empty());
        assert_eq!(ci.row(2), vec![7.0, 8.0, 10.0]);
        assert_eq!(ci.col(1), vec![2.0, 5.0, 8.0]);
        assert_eq!(ci.available_rows(), vec![0, 2]);
        assert_eq!(ci.available_cols(), vec![0, 1]);

        let full = ci.to_matrix();
        assert_eq!(full.nrows(), matrix.nrows());
        assert_eq!(full.ncols(), matrix.ncols());
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                assert_eq!(full[[i, j]], matrix[[i, j]]);
            }
        }

        let local_error = ci.local_error(&matrix, &[0, 2], &[0, 1]);
        for i in 0..local_error.nrows() {
            for j in 0..local_error.ncols() {
                assert_eq!(local_error[[i, j]], 0.0);
            }
        }
    }

    #[test]
    fn find_new_pivot_covers_success_and_error_paths() {
        let ci = ExactCi::new(sample_matrix(), vec![1], vec![2]);
        let mut perturbed = sample_matrix();
        perturbed[[2, 1]] += 10.0;

        let ((row, col), error) = ci.find_new_pivot(&perturbed).unwrap();

        assert_eq!((row, col), (2, 1));
        assert_eq!(error, 10.0);

        let ((row, col), error) = ci.find_new_pivot_in(&perturbed, &[0, 2], &[0, 1]).unwrap();

        assert_eq!((row, col), (2, 1));
        assert_eq!(error, 10.0);

        assert!(matches!(
            ci.find_new_pivot_in(&perturbed, &[], &[0]),
            Err(MatrixCIError::EmptyIndexSet { dimension }) if dimension == "rows"
        ));
        assert!(matches!(
            ci.find_new_pivot_in(&perturbed, &[0], &[]),
            Err(MatrixCIError::EmptyIndexSet { dimension }) if dimension == "cols"
        ));

        let full_rank = ExactCi::new(sample_matrix(), vec![0, 1, 2], vec![0, 1, 2]);
        assert!(matches!(
            full_rank.find_new_pivot_in(&perturbed, &[0], &[0]),
            Err(MatrixCIError::FullRank)
        ));
    }
}
