//! Matrix LU-based Cross Interpolation (MatrixLUCI) implementation

use crate::matrixlu::{rrlu, RrLU, RrLUOptions};
use crate::traits::AbstractMatrixCI;
use crate::util::{mat_mul, ncols, nrows, submatrix, zeros, Matrix, Scalar};

/// Matrix LU-based Cross Interpolation
///
/// A wrapper around RrLU that implements the AbstractMatrixCI trait.
#[derive(Debug, Clone)]
pub struct MatrixLUCI<T: Scalar> {
    /// Underlying rrLU decomposition
    lu: RrLU<T>,
}

impl<T: Scalar> MatrixLUCI<T> {
    /// Create a MatrixLUCI from a matrix
    pub fn from_matrix(a: &Matrix<T>, options: Option<RrLUOptions>) -> Self {
        Self {
            lu: rrlu(a, options),
        }
    }

    /// Create from an existing rrLU decomposition
    pub fn from_rrlu(lu: RrLU<T>) -> Self {
        Self { lu }
    }

    /// Get reference to underlying rrLU
    pub fn lu(&self) -> &RrLU<T> {
        &self.lu
    }

    /// Get column matrix: L * U[:, :npivots]
    pub fn col_matrix(&self) -> Matrix<T> {
        let l = self.lu.left(true);
        let u_sub = {
            let u = self.lu.right(false);
            let n = self.lu.npivots();
            let rows: Vec<usize> = (0..n).collect();
            let cols: Vec<usize> = (0..n).collect();
            submatrix(&u, &rows, &cols)
        };
        mat_mul(&l, &u_sub)
    }

    /// Get row matrix: L[:npivots, :] * U
    pub fn row_matrix(&self) -> Matrix<T> {
        let l_sub = {
            let l = self.lu.left(false);
            let n = self.lu.npivots();
            let rows: Vec<usize> = (0..n).collect();
            let cols: Vec<usize> = (0..n).collect();
            submatrix(&l, &rows, &cols)
        };
        let u = self.lu.right(true);
        mat_mul(&l_sub, &u)
    }

    /// Get cols times pivot inverse
    pub fn cols_times_pivot_inv(&self) -> Matrix<T> {
        let n = self.lu.npivots();
        let nr = self.nrows();

        let mut actual_result = zeros(nr, n);

        // Copy identity part
        for i in 0..nr.min(n) {
            actual_result[[i, i]] = T::one();
        }

        if n < nr {
            let l = self.lu.left(false);
            let l_sub = {
                let rows: Vec<usize> = (n..nrows(&l)).collect();
                let cols: Vec<usize> = (0..n).collect();
                submatrix(&l, &rows, &cols)
            };
            let l_pivot = {
                let rows: Vec<usize> = (0..n).collect();
                let cols: Vec<usize> = (0..n).collect();
                submatrix(&l, &rows, &cols)
            };

            // Solve: result[n:, :] = L[n:, :] * L[0:n, 0:n]^{-1}
            // This is forward substitution
            for i in 0..(nr - n) {
                for j in 0..n {
                    let mut val = l_sub[[i, j]];
                    for k in 0..j {
                        val = val - actual_result[[n + i, k]] * l_pivot[[k, j]];
                    }
                    let diag = l_pivot[[j, j]];
                    actual_result[[n + i, j]] = val / diag;
                }
            }
        }

        // Apply row permutation
        let perm = self.lu.row_permutation();
        let mut permuted = zeros(nr, n);
        for (new_i, &old_i) in perm.iter().enumerate() {
            for j in 0..n {
                permuted[[old_i, j]] = actual_result[[new_i, j]];
            }
        }

        permuted
    }

    /// Get pivot inverse times rows
    pub fn pivot_inv_times_rows(&self) -> Matrix<T> {
        let n = self.lu.npivots();
        let nc = self.ncols();

        let mut actual_result = zeros(n, nc);

        // Copy identity part
        for i in 0..n.min(nc) {
            actual_result[[i, i]] = T::one();
        }

        if n < nc {
            let u = self.lu.right(false);
            let u_sub = {
                let rows: Vec<usize> = (0..n).collect();
                let cols: Vec<usize> = (n..ncols(&u)).collect();
                submatrix(&u, &rows, &cols)
            };
            let u_pivot = {
                let rows: Vec<usize> = (0..n).collect();
                let cols: Vec<usize> = (0..n).collect();
                submatrix(&u, &rows, &cols)
            };

            // Solve: result[:, n:] = U[0:n, 0:n]^{-1} * U[:, n:]
            // This is back substitution
            for j in 0..(nc - n) {
                for i in (0..n).rev() {
                    let mut val = u_sub[[i, j]];
                    for k in (i + 1)..n {
                        val = val - u_pivot[[i, k]] * actual_result[[k, n + j]];
                    }
                    let diag = u_pivot[[i, i]];
                    actual_result[[i, n + j]] = val / diag;
                }
            }
        }

        // Apply column permutation
        let perm = self.lu.col_permutation();
        let mut permuted = zeros(n, nc);
        for i in 0..n {
            for (new_j, &old_j) in perm.iter().enumerate() {
                permuted[[i, old_j]] = actual_result[[i, new_j]];
            }
        }

        permuted
    }

    /// Get left matrix for CI representation
    pub fn left(&self) -> Matrix<T> {
        if self.lu.is_left_orthogonal() {
            self.cols_times_pivot_inv()
        } else {
            self.col_matrix()
        }
    }

    /// Get right matrix for CI representation
    pub fn right(&self) -> Matrix<T> {
        if self.lu.is_left_orthogonal() {
            self.row_matrix()
        } else {
            self.pivot_inv_times_rows()
        }
    }

    /// Get pivot errors
    pub fn pivot_errors(&self) -> Vec<f64> {
        self.lu.pivot_errors()
    }

    /// Get last pivot error
    pub fn last_pivot_error(&self) -> f64 {
        self.lu.last_pivot_error()
    }
}

impl<T: Scalar> AbstractMatrixCI<T> for MatrixLUCI<T> {
    fn nrows(&self) -> usize {
        self.lu.nrows()
    }

    fn ncols(&self) -> usize {
        self.lu.ncols()
    }

    fn rank(&self) -> usize {
        self.lu.npivots()
    }

    fn row_indices(&self) -> &[usize] {
        // Return slice of row permutation
        &self.lu.row_permutation()[0..self.rank()]
    }

    fn col_indices(&self) -> &[usize] {
        // Return slice of column permutation
        &self.lu.col_permutation()[0..self.rank()]
    }

    fn evaluate(&self, i: usize, j: usize) -> T {
        let left = self.left();
        let right = self.right();

        let mut sum = T::zero();
        for k in 0..self.rank() {
            sum = sum + left[[i, k]] * right[[k, j]];
        }
        sum
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {
        let left = self.left();
        let right = self.right();

        let r = self.rank();
        let left_sub = submatrix(&left, rows, &(0..r).collect::<Vec<_>>());
        let right_sub = submatrix(&right, &(0..r).collect::<Vec<_>>(), cols);

        mat_mul(&left_sub, &right_sub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::from_vec2d;

    #[test]
    fn test_matrixluci_from_matrix() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ]);

        let luci = MatrixLUCI::from_matrix(&m, None);
        assert_eq!(luci.nrows(), 3);
        assert_eq!(luci.ncols(), 3);
        assert_eq!(luci.rank(), 3);
    }

    #[test]
    fn test_matrixluci_reconstruct() {
        let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let luci = MatrixLUCI::from_matrix(&m, None);
        let approx = luci.to_matrix();

        for i in 0..2 {
            for j in 0..2 {
                let diff = (m[[i, j]] - approx[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Reconstruction error at ({}, {}): {}",
                    i,
                    j,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_matrixluci_rank_deficient() {
        // Rank-1 matrix
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ]);

        let luci = MatrixLUCI::from_matrix(&m, None);
        assert_eq!(luci.rank(), 1);
    }
}
