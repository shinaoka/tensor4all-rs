//! Rank-Revealing LU decomposition (rrLU) implementation

use crate::error::Result;
use crate::util::{
    ncols, nrows, submatrix_argmax, swap_cols, swap_rows, transpose, zeros, Matrix, Scalar,
};

/// Rank-Revealing LU decomposition
///
/// Represents a matrix A as:
/// P_row * A * P_col = L * U
///
/// where P_row and P_col are permutation matrices.
#[derive(Debug, Clone)]
pub struct RrLU<T: Scalar> {
    /// Row permutation
    row_permutation: Vec<usize>,
    /// Column permutation
    col_permutation: Vec<usize>,
    /// Lower triangular matrix L
    l: Matrix<T>,
    /// Upper triangular matrix U
    u: Matrix<T>,
    /// Whether L is left-orthogonal (L has 1s on diagonal) or U is (U has 1s on diagonal)
    left_orthogonal: bool,
    /// Number of pivots
    n_pivot: usize,
    /// Last pivot error
    error: f64,
}

impl<T: Scalar> RrLU<T> {
    /// Create an empty rrLU for a matrix of given size
    pub fn new(nr: usize, nc: usize, left_orthogonal: bool) -> Self {
        Self {
            row_permutation: (0..nr).collect(),
            col_permutation: (0..nc).collect(),
            l: zeros(nr, 0),
            u: zeros(0, nc),
            left_orthogonal,
            n_pivot: 0,
            error: f64::NAN,
        }
    }

    /// Number of rows
    pub fn nrows(&self) -> usize {
        nrows(&self.l)
    }

    /// Number of columns
    pub fn ncols(&self) -> usize {
        ncols(&self.u)
    }

    /// Number of pivots
    pub fn npivots(&self) -> usize {
        self.n_pivot
    }

    /// Row permutation
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_permutation
    }

    /// Column permutation
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_permutation
    }

    /// Get row indices (selected pivots)
    pub fn row_indices(&self) -> Vec<usize> {
        self.row_permutation[0..self.n_pivot].to_vec()
    }

    /// Get column indices (selected pivots)
    pub fn col_indices(&self) -> Vec<usize> {
        self.col_permutation[0..self.n_pivot].to_vec()
    }

    /// Get left matrix (optionally permuted)
    pub fn left(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = zeros(nrows(&self.l), ncols(&self.l));
            for (new_i, &old_i) in self.row_permutation.iter().enumerate() {
                for j in 0..ncols(&self.l) {
                    result[[old_i, j]] = self.l[[new_i, j]];
                }
            }
            result
        } else {
            self.l.clone()
        }
    }

    /// Get right matrix (optionally permuted)
    pub fn right(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = zeros(nrows(&self.u), ncols(&self.u));
            for i in 0..nrows(&self.u) {
                for (new_j, &old_j) in self.col_permutation.iter().enumerate() {
                    result[[i, old_j]] = self.u[[i, new_j]];
                }
            }
            result
        } else {
            self.u.clone()
        }
    }

    /// Get diagonal elements
    pub fn diag(&self) -> Vec<T> {
        let n = self.n_pivot;
        if self.left_orthogonal {
            (0..n).map(|i| self.u[[i, i]]).collect()
        } else {
            (0..n).map(|i| self.l[[i, i]]).collect()
        }
    }

    /// Get pivot errors
    pub fn pivot_errors(&self) -> Vec<f64> {
        let mut errors: Vec<f64> = self
            .diag()
            .iter()
            .map(|d| f64::sqrt(d.abs_sq()))
            .collect();
        errors.push(self.error);
        errors
    }

    /// Get last pivot error
    pub fn last_pivot_error(&self) -> f64 {
        self.error
    }

    /// Transpose the decomposition
    pub fn transpose(&self) -> RrLU<T> {
        RrLU {
            row_permutation: self.col_permutation.clone(),
            col_permutation: self.row_permutation.clone(),
            l: transpose(&self.u),
            u: transpose(&self.l),
            left_orthogonal: !self.left_orthogonal,
            n_pivot: self.n_pivot,
            error: self.error,
        }
    }

    /// Check if left-orthogonal (L has 1s on diagonal)
    pub fn is_left_orthogonal(&self) -> bool {
        self.left_orthogonal
    }
}

/// Options for rank-revealing LU decomposition
#[derive(Debug, Clone)]
pub struct RrLUOptions {
    /// Maximum rank
    pub max_rank: usize,
    /// Relative tolerance
    pub rel_tol: f64,
    /// Absolute tolerance
    pub abs_tol: f64,
    /// Left orthogonal (L has 1s on diagonal) or right orthogonal (U has 1s)
    pub left_orthogonal: bool,
}

impl Default for RrLUOptions {
    fn default() -> Self {
        Self {
            max_rank: usize::MAX,
            rel_tol: 1e-14,
            abs_tol: 0.0,
            left_orthogonal: true,
        }
    }
}

/// Perform in-place rank-revealing LU decomposition
pub fn rrlu_inplace<T: Scalar>(a: &mut Matrix<T>, options: Option<RrLUOptions>) -> RrLU<T> {
    let opts = options.unwrap_or_default();
    let nr = nrows(a);
    let nc = ncols(a);

    let mut lu = RrLU::new(nr, nc, opts.left_orthogonal);
    let max_rank = opts.max_rank.min(nr).min(nc);
    let mut max_error = 0.0f64;

    while lu.n_pivot < max_rank {
        let k = lu.n_pivot;
        let rows: Vec<usize> = (k..nr).collect();
        let cols: Vec<usize> = (k..nc).collect();

        if rows.is_empty() || cols.is_empty() {
            break;
        }

        // Find pivot with maximum absolute value in submatrix
        let (pivot_row, pivot_col, pivot_val) = submatrix_argmax(a, &rows, &cols);

        let pivot_abs = f64::sqrt(pivot_val.abs_sq());
        lu.error = pivot_abs;

        // Check stopping criteria (but add at least 1 pivot)
        if lu.n_pivot > 0 && (pivot_abs < opts.rel_tol * max_error || pivot_abs < opts.abs_tol) {
            break;
        }

        max_error = max_error.max(pivot_abs);

        // Swap rows and columns
        if pivot_row != k {
            *a = swap_rows(a, k, pivot_row);
            lu.row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            *a = swap_cols(a, k, pivot_col);
            lu.col_permutation.swap(k, pivot_col);
        }

        let pivot = a[[k, k]];

        // Eliminate
        if opts.left_orthogonal {
            // Scale column below pivot
            for i in (k + 1)..nr {
                let val = a[[i, k]] / pivot;
                a[[i, k]] = val;
            }
        } else {
            // Scale row to the right of pivot
            for j in (k + 1)..nc {
                let val = a[[k, j]] / pivot;
                a[[k, j]] = val;
            }
        }

        // Update submatrix: A[k+1:, k+1:] -= A[k+1:, k] * A[k, k+1:]
        for i in (k + 1)..nr {
            for j in (k + 1)..nc {
                let x = a[[i, k]];
                let y = a[[k, j]];
                let old = a[[i, j]];
                a[[i, j]] = old - x * y;
            }
        }

        lu.n_pivot += 1;
    }

    // Extract L and U
    let n = lu.n_pivot;

    // L is lower triangular part
    let mut l = zeros(nr, n);
    for i in 0..nr {
        for j in 0..n.min(i + 1) {
            l[[i, j]] = a[[i, j]];
        }
    }

    // U is upper triangular part
    let mut u = zeros(n, nc);
    for i in 0..n {
        for j in i..nc {
            u[[i, j]] = a[[i, j]];
        }
    }

    // Set diagonal to 1 for the orthogonal factor
    if opts.left_orthogonal {
        for i in 0..n {
            l[[i, i]] = T::one();
        }
    } else {
        for i in 0..n {
            u[[i, i]] = T::one();
        }
    }

    // Check for NaNs
    for i in 0..nrows(&l) {
        for j in 0..ncols(&l) {
            if l[[i, j]].is_nan() {
                panic!("NaN in L matrix");
            }
        }
    }
    for i in 0..nrows(&u) {
        for j in 0..ncols(&u) {
            if u[[i, j]].is_nan() {
                panic!("NaN in U matrix");
            }
        }
    }

    // Set error to 0 if full rank
    if n >= nr.min(nc) {
        lu.error = 0.0;
    }

    lu.l = l;
    lu.u = u;

    lu
}

/// Perform rank-revealing LU decomposition (non-destructive)
pub fn rrlu<T: Scalar>(a: &Matrix<T>, options: Option<RrLUOptions>) -> RrLU<T> {
    let mut a_copy = a.clone();
    rrlu_inplace(&mut a_copy, options)
}

/// Convert L matrix to solve L * X = B given pivot matrix P
pub fn cols_to_l_matrix<T: Scalar>(c: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = nrows(p);

    for k in 0..n {
        let pivot = p[[k, k]];
        // c[:, k] /= pivot
        for i in 0..nrows(c) {
            let val = c[[i, k]] / pivot;
            c[[i, k]] = val;
        }

        // c[:, k+1:] -= c[:, k] * p[k, k+1:]
        for j in (k + 1)..ncols(c) {
            let p_kj = p[[k, j]];
            for i in 0..nrows(c) {
                let c_ik = c[[i, k]];
                let old = c[[i, j]];
                c[[i, j]] = old - c_ik * p_kj;
            }
        }
    }
}

/// Convert R matrix to solve X * U = B given pivot matrix P
pub fn rows_to_u_matrix<T: Scalar>(r: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = nrows(p);

    for k in 0..n {
        let pivot = p[[k, k]];
        // r[k, :] /= pivot
        for j in 0..ncols(r) {
            let val = r[[k, j]] / pivot;
            r[[k, j]] = val;
        }

        // r[k+1:, :] -= p[k+1:, k] * r[k, :]
        for i in (k + 1)..nrows(r) {
            let p_ik = p[[i, k]];
            for j in 0..ncols(r) {
                let r_kj = r[[k, j]];
                let old = r[[i, j]];
                r[[i, j]] = old - p_ik * r_kj;
            }
        }
    }
}

/// Solve LU * x = b
pub fn solve_lu<T: Scalar>(l: &Matrix<T>, u: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    let _n1 = nrows(l);
    let n2 = ncols(l);
    let n3 = ncols(u);
    let m = ncols(b);

    // Solve L * y = b (forward substitution)
    let mut y: Matrix<T> = zeros(n2, m);
    for i in 0..n2 {
        for k in 0..m {
            let mut sum = b[[i, k]];
            for j in 0..i {
                sum = sum - l[[i, j]] * y[[j, k]];
            }
            let diag = l[[i, i]];
            y[[i, k]] = sum / diag;
        }
    }

    // Solve U * x = y (back substitution)
    let mut x: Matrix<T> = zeros(n3, m);
    for i in (0..n3).rev() {
        for k in 0..m {
            let mut sum = y[[i, k]];
            for j in (i + 1)..n3 {
                sum = sum - u[[i, j]] * x[[j, k]];
            }
            let diag = u[[i, i]];
            x[[i, k]] = sum / diag;
        }
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{eye, from_vec2d, mat_mul};

    #[test]
    fn test_rrlu_identity() {
        let m: Matrix<f64> = eye(3);
        let lu = rrlu(&m, None);

        assert_eq!(lu.npivots(), 3);
        assert!(lu.error.abs() < 1e-10 || lu.error == 0.0);
    }

    #[test]
    fn test_rrlu_rank_deficient() {
        // Rank-1 matrix
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ]);

        let lu = rrlu(&m, None);

        // Should detect rank-1
        assert_eq!(lu.npivots(), 1);
    }

    #[test]
    fn test_rrlu_full_rank() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0], // Not rank-deficient
        ]);

        let lu = rrlu(&m, None);

        assert_eq!(lu.npivots(), 3);
    }

    #[test]
    fn test_rrlu_reconstruct() {
        let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let lu = rrlu(&m, None);
        let l = lu.left(true);
        let u = lu.right(true);

        // Reconstruct: L * U should approximate original matrix
        let reconstructed = mat_mul(&l, &u);

        for i in 0..2 {
            for j in 0..2 {
                let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
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
}
