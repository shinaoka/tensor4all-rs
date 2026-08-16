//! Rank-Revealing LU decomposition (rrLU) implementation.
//!
//! Provides [`RrLU`], a full-pivoting LU decomposition that reveals the
//! numerical rank of a matrix. The decomposition is:
//!
//! ```text
//! P_row * A * P_col = L * U
//! ```
//!
//! where `P_row`, `P_col` are permutation matrices. The rank is determined
//! by the number of pivots exceeding the tolerance thresholds in
//! [`RrLUOptions`].
//!
//! # Examples
//!
//! ```
//! use tensor4all_core::matrixlu::rrlu;
//! use tensor4all_tensorbackend::from_vec2d;
//!
//! let m = from_vec2d(vec![
//!     vec![1.0_f64, 2.0],
//!     vec![3.0, 4.0],
//! ]);
//! let lu = rrlu(&m, None).unwrap();
//! assert_eq!(lu.npivots(), 2);
//! ```

use crate::error::{MatrixCIError, Result};
use crate::scalar::Scalar;
use tensor4all_tensorbackend::{transpose, Matrix};

/// Rank-Revealing LU decomposition.
///
/// Represents a matrix `A` as `P_row * A * P_col = L * U`, where `P_row`
/// and `P_col` are permutation matrices, `L` is lower-triangular, and `U`
/// is upper-triangular. One of `L` or `U` has unit diagonal, controlled by
/// the `left_orthogonal` option.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrixlu::rrlu;
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 10.0],
/// ]);
///
/// let lu = rrlu(&m, None).unwrap();
/// assert_eq!(lu.npivots(), 3);
///
/// // Verify L * U reconstructs the permuted matrix
/// let l = lu.left(false);
/// let u = lu.right(false);
/// let reconstructed = mat_mul(&l, &u).unwrap();
///
/// // Check reconstruction matches the permuted matrix
/// for i in 0..3 {
///     for j in 0..3 {
///         let orig_row = lu.row_permutation()[i];
///         let orig_col = lu.col_permutation()[j];
///         assert!((reconstructed[[i, j]] - m[[orig_row, orig_col]]).abs() < 1e-10);
///     }
/// }
/// ```
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
    /// Create an empty rrLU for a matrix of given size.
    ///
    /// Used internally. Most users should call [`rrlu`] or [`rrlu_mut`]
    /// instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::RrLU;
    ///
    /// let lu = RrLU::<f64>::new(3, 4, true);
    /// assert_eq!(lu.nrows(), 3);
    /// assert_eq!(lu.ncols(), 4);
    /// assert_eq!(lu.npivots(), 0);
    /// assert!(lu.is_left_orthogonal());
    /// ```
    pub fn new(nr: usize, nc: usize, left_orthogonal: bool) -> Self {
        Self {
            row_permutation: (0..nr).collect(),
            col_permutation: (0..nc).collect(),
            l: Matrix::zeros(nr, 0),
            u: Matrix::zeros(0, nc),
            left_orthogonal,
            n_pivot: 0,
            error: f64::NAN,
        }
    }

    /// Number of rows
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.nrows(), 3);
    /// ```
    pub fn nrows(&self) -> usize {
        self.l.nrows()
    }

    /// Number of columns
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.ncols(), 3);
    /// ```
    pub fn ncols(&self) -> usize {
        self.u.ncols()
    }

    /// Number of pivots
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// assert_eq!(lu.npivots(), 2);
    /// ```
    pub fn npivots(&self) -> usize {
        self.n_pivot
    }

    /// Row permutation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let perm = lu.row_permutation();
    /// assert_eq!(perm.len(), 2);
    /// // Permutation is a rearrangement of 0..nrows
    /// let mut sorted = perm.to_vec();
    /// sorted.sort();
    /// assert_eq!(sorted, vec![0, 1]);
    /// ```
    pub fn row_permutation(&self) -> &[usize] {
        &self.row_permutation
    }

    /// Column permutation
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let perm = lu.col_permutation();
    /// assert_eq!(perm.len(), 2);
    /// let mut sorted = perm.to_vec();
    /// sorted.sort();
    /// assert_eq!(sorted, vec![0, 1]);
    /// ```
    pub fn col_permutation(&self) -> &[usize] {
        &self.col_permutation
    }

    /// Get row indices (selected pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{matrixlu::rrlu, RrLUOptions};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, Some(RrLUOptions { max_bond_dim: 1, ..Default::default() })).unwrap();
    /// let rows = lu.row_indices();
    /// assert_eq!(rows.len(), 1);
    /// assert!(rows[0] < 2);
    /// ```
    pub fn row_indices(&self) -> Vec<usize> {
        self.row_permutation[0..self.n_pivot].to_vec()
    }

    /// Get column indices (selected pivots)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{matrixlu::rrlu, RrLUOptions};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, Some(RrLUOptions { max_bond_dim: 1, ..Default::default() })).unwrap();
    /// let cols = lu.col_indices();
    /// assert_eq!(cols.len(), 1);
    /// assert!(cols[0] < 2);
    /// ```
    pub fn col_indices(&self) -> Vec<usize> {
        self.col_permutation[0..self.n_pivot].to_vec()
    }

    /// Get left matrix (optionally permuted)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    ///
    /// // Unpermuted: L * U reconstructs the row/col-permuted matrix
    /// let l = lu.left(false);
    /// let u = lu.right(false);
    /// let prod = mat_mul(&l, &u).unwrap();
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         let ri = lu.row_permutation()[i];
    ///         let cj = lu.col_permutation()[j];
    ///         assert!((prod[[i, j]] - m[[ri, cj]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn left(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = Matrix::zeros(self.l.nrows(), self.l.ncols());
            for j in 0..self.l.ncols() {
                for (new_i, &old_i) in self.row_permutation.iter().enumerate() {
                    result[[old_i, j]] = self.l[[new_i, j]];
                }
            }
            result
        } else {
            self.l.clone()
        }
    }

    /// Get right matrix (optionally permuted)
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let l = lu.left(false);
    /// let u = lu.right(false);
    /// assert_eq!(u.nrows(), lu.npivots());
    /// assert_eq!(u.ncols(), lu.ncols());
    /// // L * U reconstructs the permuted matrix
    /// let prod = mat_mul(&l, &u).unwrap();
    /// assert!((prod[[0, 0]] - m[[lu.row_permutation()[0], lu.col_permutation()[0]]]).abs() < 1e-10);
    /// ```
    pub fn right(&self, permute: bool) -> Matrix<T> {
        if permute {
            let mut result = Matrix::zeros(self.u.nrows(), self.u.ncols());
            for (new_j, &old_j) in self.col_permutation.iter().enumerate() {
                for i in 0..self.u.nrows() {
                    result[[i, old_j]] = self.u[[i, new_j]];
                }
            }
            result
        } else {
            self.u.clone()
        }
    }

    pub(crate) fn left_unpermuted(&self) -> &Matrix<T> {
        &self.l
    }

    pub(crate) fn right_unpermuted(&self) -> &Matrix<T> {
        &self.u
    }

    /// Get diagonal elements
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let d = lu.diag();
    /// assert_eq!(d.len(), lu.npivots());
    /// // Diagonal elements are non-zero for full-rank matrices
    /// for &val in &d {
    ///     assert!(val.abs() > 1e-14);
    /// }
    /// ```
    pub fn diag(&self) -> Vec<T> {
        let n = self.n_pivot;
        if self.left_orthogonal {
            (0..n).map(|i| self.u[[i, i]]).collect()
        } else {
            (0..n).map(|i| self.l[[i, i]]).collect()
        }
    }

    /// Get pivot errors
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let errs = lu.pivot_errors();
    /// // One entry per pivot plus the final residual
    /// assert_eq!(errs.len(), lu.npivots() + 1);
    /// // Errors are non-negative
    /// for &e in &errs {
    ///     assert!(e >= 0.0);
    /// }
    /// ```
    pub fn pivot_errors(&self) -> Vec<f64> {
        let mut errors: Vec<f64> = self.diag().iter().map(|d| f64::sqrt(d.abs_sq())).collect();
        errors.push(self.error);
        errors
    }

    /// Get last pivot error
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// // Full-rank decomposition has zero residual
    /// assert_eq!(lu.last_pivot_error(), 0.0);
    /// ```
    pub fn last_pivot_error(&self) -> f64 {
        self.error
    }

    /// Transpose the decomposition
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::matrixlu::rrlu;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let lu = rrlu(&m, None).unwrap();
    /// let lu_t = lu.transpose();
    /// assert_eq!(lu_t.nrows(), lu.ncols());
    /// assert_eq!(lu_t.ncols(), lu.nrows());
    /// assert_eq!(lu_t.npivots(), lu.npivots());
    /// assert_eq!(lu_t.is_left_orthogonal(), !lu.is_left_orthogonal());
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{matrixlu::rrlu, RrLUOptions};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    ///
    /// let lu = rrlu(&m, None).unwrap();
    /// assert!(lu.is_left_orthogonal()); // default
    ///
    /// let lu2 = rrlu(&m, Some(RrLUOptions {
    ///     left_orthogonal: false, ..Default::default()
    /// })).unwrap();
    /// assert!(!lu2.is_left_orthogonal());
    /// ```
    pub fn is_left_orthogonal(&self) -> bool {
        self.left_orthogonal
    }
}

fn validate_col_major_matrix_len(
    nrows: usize,
    ncols: usize,
    actual_len: usize,
) -> crate::Result<()> {
    let expected = nrows
        .checked_mul(ncols)
        .ok_or_else(|| MatrixCIError::InvalidArgument {
            message: format!("matrix shape {nrows} x {ncols} overflows usize"),
        })?;
    if actual_len != expected {
        return Err(MatrixCIError::InvalidArgument {
            message: format!(
                "column-major matrix length mismatch: expected {expected}, got {actual_len}"
            ),
        });
    }
    Ok(())
}

#[inline]
fn col_major_offset(nrows: usize, row: usize, col: usize) -> usize {
    row + nrows * col
}

#[inline]
fn col_major_get<T: Copy>(data: &[T], nrows: usize, row: usize, col: usize) -> T {
    let offset = col_major_offset(nrows, row, col);
    debug_assert!(row < nrows);
    debug_assert!(offset < data.len());
    // SAFETY: callers pass matrix-derived dimensions and prechecked ranges.
    unsafe { *data.get_unchecked(offset) }
}

#[inline]
fn col_major_set<T>(data: &mut [T], nrows: usize, row: usize, col: usize, value: T) {
    let offset = col_major_offset(nrows, row, col);
    debug_assert!(row < nrows);
    debug_assert!(offset < data.len());
    // SAFETY: callers pass matrix-derived dimensions and prechecked ranges.
    unsafe {
        *data.get_unchecked_mut(offset) = value;
    }
}

fn submatrix_argmax_col_major<T: Scalar>(
    data: &[T],
    nrows: usize,
    ncols: usize,
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
) -> (usize, usize, T) {
    debug_assert!(row_start < row_end);
    debug_assert!(col_start < col_end);
    debug_assert!(row_end <= nrows);
    debug_assert!(col_end <= ncols);
    debug_assert_eq!(data.len(), nrows * ncols);

    let first = col_major_get(data, nrows, row_start, col_start);
    let mut max_val = first.abs_sq();
    let mut max_row = row_start;
    let mut max_col = col_start;

    for col in col_start..col_end {
        let col_start_offset = col_major_offset(nrows, row_start, col);
        for (offset, row) in (col_start_offset..).zip(row_start..row_end) {
            // SAFETY: row and column loops stay within the prechecked region.
            let value = unsafe { *data.get_unchecked(offset) };
            let value_abs = value.abs_sq();
            if value_abs > max_val {
                max_val = value_abs;
                max_row = row;
                max_col = col;
            }
        }
    }

    (
        max_row,
        max_col,
        col_major_get(data, nrows, max_row, max_col),
    )
}

fn swap_rows_col_major<T>(data: &mut [T], nrows: usize, ncols: usize, row_a: usize, row_b: usize) {
    debug_assert!(row_a < nrows);
    debug_assert!(row_b < nrows);
    debug_assert_eq!(data.len(), nrows * ncols);
    if row_a == row_b {
        return;
    }

    let ptr = data.as_mut_ptr();
    for col in 0..ncols {
        let offset_a = col_major_offset(nrows, row_a, col);
        let offset_b = col_major_offset(nrows, row_b, col);
        // SAFETY: row_a/row_b are in range, and each column offset is within
        // the matrix-sized backing slice.
        unsafe {
            std::ptr::swap(ptr.add(offset_a), ptr.add(offset_b));
        }
    }
}

fn swap_cols_col_major<T>(data: &mut [T], nrows: usize, ncols: usize, col_a: usize, col_b: usize) {
    debug_assert!(col_a < ncols);
    debug_assert!(col_b < ncols);
    debug_assert_eq!(data.len(), nrows * ncols);
    if col_a == col_b || nrows == 0 {
        return;
    }

    let start_a = col_major_offset(nrows, 0, col_a);
    let start_b = col_major_offset(nrows, 0, col_b);
    // SAFETY: distinct columns are non-overlapping contiguous ranges of
    // length nrows in column-major storage.
    unsafe {
        std::ptr::swap_nonoverlapping(
            data.as_mut_ptr().add(start_a),
            data.as_mut_ptr().add(start_b),
            nrows,
        );
    }
}

fn scale_column_tail<T: Scalar>(
    data: &mut [T],
    nrows: usize,
    col: usize,
    row_start: usize,
    pivot: T,
) {
    if row_start >= nrows {
        return;
    }
    let start = col_major_offset(nrows, row_start, col);
    let end = col_major_offset(nrows, nrows - 1, col) + 1;
    for value in &mut data[start..end] {
        *value = *value / pivot;
    }
}

fn scale_row_tail<T: Scalar>(
    data: &mut [T],
    nrows: usize,
    ncols: usize,
    row: usize,
    col_start: usize,
    pivot: T,
) {
    for col in col_start..ncols {
        let value = col_major_get(data, nrows, row, col) / pivot;
        col_major_set(data, nrows, row, col, value);
    }
}

fn update_trailing_submatrix<T: Scalar>(data: &mut [T], nrows: usize, ncols: usize, pivot: usize) {
    let tail_row_start = pivot + 1;
    let tail_col_start = pivot + 1;
    if tail_row_start >= nrows || tail_col_start >= ncols {
        return;
    }

    let tail_len = nrows - tail_row_start;
    let pivot_col_tail_start = col_major_offset(nrows, tail_row_start, pivot);
    for col in tail_col_start..ncols {
        let y = col_major_get(data, nrows, pivot, col);
        let target_start = col_major_offset(nrows, tail_row_start, col);
        let (before_target, target_and_after) = data.split_at_mut(target_start);
        let pivot_col_tail = &before_target[pivot_col_tail_start..pivot_col_tail_start + tail_len];
        let target_tail = &mut target_and_after[..tail_len];
        for (target, &x) in target_tail.iter_mut().zip(pivot_col_tail.iter()) {
            *target = *target - x * y;
        }
    }
}

fn extract_lu_from_factorized<T: Scalar>(
    data: &[T],
    nrows: usize,
    ncols: usize,
    rank: usize,
    left_orthogonal: bool,
) -> Result<(Matrix<T>, Matrix<T>)> {
    debug_assert!(rank <= nrows.min(ncols));
    debug_assert_eq!(data.len(), nrows * ncols);

    let mut l_data = vec![T::zero(); nrows * rank];
    for col in 0..rank {
        let src_start = col_major_offset(nrows, col, col);
        let src_end = col_major_offset(nrows, nrows - 1, col) + 1;
        let dst_start = col_major_offset(nrows, col, col);
        l_data[dst_start..dst_start + (nrows - col)].copy_from_slice(&data[src_start..src_end]);
    }

    let mut u_data = vec![T::zero(); rank * ncols];
    for col in 0..ncols {
        let rows_to_copy = rank.min(col + 1);
        if rows_to_copy > 0 {
            let src_start = col_major_offset(nrows, 0, col);
            let dst_start = col_major_offset(rank, 0, col);
            u_data[dst_start..dst_start + rows_to_copy]
                .copy_from_slice(&data[src_start..src_start + rows_to_copy]);
        }
    }

    if left_orthogonal {
        for i in 0..rank {
            l_data[col_major_offset(nrows, i, i)] = T::one();
        }
    } else {
        for i in 0..rank {
            u_data[col_major_offset(rank, i, i)] = T::one();
        }
    }

    if l_data.iter().any(|&value| value.is_nan()) {
        return Err(MatrixCIError::NaNEncountered {
            matrix: "L".to_string(),
        });
    }
    if u_data.iter().any(|&value| value.is_nan()) {
        return Err(MatrixCIError::NaNEncountered {
            matrix: "U".to_string(),
        });
    }

    Ok((
        Matrix::from_col_major_vec(nrows, rank, l_data),
        Matrix::from_col_major_vec(rank, ncols, u_data),
    ))
}

/// Options for rank-revealing LU decomposition.
///
/// # Examples
///
/// ```
/// use tensor4all_core::RrLUOptions;
///
/// // Default: rel_tol = 1e-14, no absolute tolerance, no rank limit
/// let opts = RrLUOptions::default();
/// assert_eq!(opts.rel_tol, 1e-14);
/// assert_eq!(opts.abs_tol, 0.0);
/// assert!(opts.left_orthogonal);
///
/// // Limit rank to 5
/// let opts = RrLUOptions { max_bond_dim: 5, ..Default::default() };
/// assert_eq!(opts.max_bond_dim, 5);
/// ```
#[derive(Debug, Clone)]
pub struct RrLUOptions {
    /// Maximum rank
    pub max_bond_dim: usize,
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
            max_bond_dim: usize::MAX,
            rel_tol: 1e-14,
            abs_tol: 0.0,
            left_orthogonal: true,
        }
    }
}

/// Perform in-place rank-revealing LU decomposition.
///
/// The input matrix `a` is modified in place. Use [`rrlu`] for a
/// non-destructive version.
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] if the matrix shape product
/// overflows `usize` or its backing storage length does not match the shape.
/// Returns [`MatrixCIError::NaNEncountered`] if NaN values appear in the L or U
/// factors.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{matrixlu::rrlu_mut, RrLUOptions};
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let mut m = from_vec2d(vec![
///     vec![1.0_f64, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// let lu = rrlu_mut(&mut m, Some(RrLUOptions { max_bond_dim: 1, ..Default::default() })).unwrap();
/// assert_eq!(lu.npivots(), 1);
/// ```
pub fn rrlu_mut<T: Scalar>(a: &mut Matrix<T>, options: Option<RrLUOptions>) -> Result<RrLU<T>> {
    let opts = options.unwrap_or_default();
    let nr = a.nrows();
    let nc = a.ncols();
    let data = a.as_col_major_mut_slice();
    validate_col_major_matrix_len(nr, nc, data.len())?;
    debug_assert_eq!(data.len(), nr * nc);

    let mut lu = RrLU::new(nr, nc, opts.left_orthogonal);
    let max_bond_dim = opts.max_bond_dim.min(nr).min(nc);
    let mut max_error = 0.0f64;

    while lu.n_pivot < max_bond_dim {
        let k = lu.n_pivot;

        if k >= nr || k >= nc {
            break;
        }

        let (pivot_row, pivot_col, pivot_val) =
            submatrix_argmax_col_major(data, nr, nc, k, nr, k, nc);

        let pivot_abs = f64::sqrt(pivot_val.abs_sq());
        lu.error = pivot_abs;

        // Check stopping criteria (but add at least 1 pivot)
        if lu.n_pivot > 0 && (pivot_abs < opts.rel_tol * max_error || pivot_abs < opts.abs_tol) {
            break;
        }

        // Guard against tiny pivots to prevent NaN from division. A caller that
        // sets both tolerances to zero is requesting a non-truncating
        // decomposition, so only an exactly zero pivot stops the factorization.
        let min_pivot_abs = if opts.rel_tol == 0.0 && opts.abs_tol == 0.0 {
            0.0
        } else {
            f64::EPSILON
        };
        if pivot_abs <= min_pivot_abs {
            if lu.n_pivot == 0 {
                // First pivot is near-zero: the matrix is effectively zero
                lu.error = pivot_abs;
            }
            break;
        }

        max_error = max_error.max(pivot_abs);

        // Swap rows and columns
        if pivot_row != k {
            swap_rows_col_major(data, nr, nc, k, pivot_row);
            lu.row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            swap_cols_col_major(data, nr, nc, k, pivot_col);
            lu.col_permutation.swap(k, pivot_col);
        }

        let pivot = col_major_get(data, nr, k, k);

        // Eliminate
        if opts.left_orthogonal {
            scale_column_tail(data, nr, k, k + 1, pivot);
        } else {
            scale_row_tail(data, nr, nc, k, k + 1, pivot);
        }

        update_trailing_submatrix(data, nr, nc, k);

        lu.n_pivot += 1;
    }

    let n = lu.n_pivot;
    let (l, u) = extract_lu_from_factorized(data, nr, nc, n, opts.left_orthogonal)?;

    // Set error to 0 if full rank
    if n >= nr.min(nc) {
        lu.error = 0.0;
    }

    lu.l = l;
    lu.u = u;

    Ok(lu)
}

/// Perform rank-revealing LU decomposition (non-destructive).
///
/// Clones the input matrix and calls [`rrlu_mut`].
///
/// # Errors
///
/// Returns [`MatrixCIError::InvalidArgument`] if the matrix shape product
/// overflows `usize` or its backing storage length does not match the shape.
/// Returns [`MatrixCIError::NaNEncountered`] if NaN values appear in the L or U
/// factors.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrixlu::rrlu;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 0.0],
///     vec![0.0, 2.0],
/// ]);
/// let lu = rrlu(&m, None).unwrap();
/// assert_eq!(lu.npivots(), 2);
/// assert_eq!(lu.nrows(), 2);
/// assert_eq!(lu.ncols(), 2);
/// ```
pub fn rrlu<T: Scalar>(a: &Matrix<T>, options: Option<RrLUOptions>) -> Result<RrLU<T>> {
    let mut a_copy = a.clone();
    rrlu_mut(&mut a_copy, options)
}

/// Convert L matrix to solve L * X = B given pivot matrix P
///
/// Modifies `c` in place so that the columns satisfy the triangular
/// system defined by `p`. The matrix `c` must have at least `p.nrows()`
/// columns.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrixlu::cols_to_l_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// // Upper-triangular P (2x2)
/// let p = from_vec2d(vec![
///     vec![2.0_f64, 1.0],
///     vec![0.0, 3.0],
/// ]);
/// // c has 3 rows, 2 columns (ncols >= p.nrows())
/// let mut c = from_vec2d(vec![
///     vec![4.0_f64, 5.0],
///     vec![6.0, 9.0],
///     vec![8.0, 7.0],
/// ]);
/// cols_to_l_matrix(&mut c, &p, true);
/// // After processing: c[:,0] was divided by p[0,0]=2
/// assert!((c[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 3.0).abs() < 1e-10);
/// assert!((c[[2, 0]] - 4.0).abs() < 1e-10);
/// ```
pub fn cols_to_l_matrix<T: Scalar>(c: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = p.nrows();

    for k in 0..n {
        let pivot = p[[k, k]];
        // c[:, k] /= pivot
        for i in 0..c.nrows() {
            let val = c[[i, k]] / pivot;
            c[[i, k]] = val;
        }

        // c[:, k+1:] -= c[:, k] * p[k, k+1:]
        for j in (k + 1)..c.ncols() {
            let p_kj = p[[k, j]];
            for i in 0..c.nrows() {
                let c_ik = c[[i, k]];
                let old = c[[i, j]];
                c[[i, j]] = old - c_ik * p_kj;
            }
        }
    }
}

/// Convert R matrix to solve X * U = B given pivot matrix P
///
/// Modifies `r` in place so that the rows satisfy the triangular
/// system defined by `p`. The matrix `r` must have at least `p.nrows()`
/// rows.
///
/// # Examples
///
/// ```
/// use tensor4all_core::matrixlu::rows_to_u_matrix;
/// use tensor4all_tensorbackend::from_vec2d;
///
/// // Lower-triangular P (2x2)
/// let p = from_vec2d(vec![
///     vec![2.0_f64, 0.0],
///     vec![1.0, 3.0],
/// ]);
/// // r has 2 rows (nrows >= p.nrows()), 3 columns
/// let mut r = from_vec2d(vec![
///     vec![4.0_f64, 6.0, 8.0],
///     vec![5.0, 9.0, 7.0],
/// ]);
/// rows_to_u_matrix(&mut r, &p, true);
/// // After processing: r[0,:] was divided by p[0,0]=2
/// assert!((r[[0, 0]] - 2.0).abs() < 1e-10);
/// assert!((r[[0, 1]] - 3.0).abs() < 1e-10);
/// assert!((r[[0, 2]] - 4.0).abs() < 1e-10);
/// ```
pub fn rows_to_u_matrix<T: Scalar>(r: &mut Matrix<T>, p: &Matrix<T>, _left_orthogonal: bool) {
    let n = p.nrows();

    for k in 0..n {
        let pivot = p[[k, k]];
        // r[k, :] /= pivot
        for j in 0..r.ncols() {
            let val = r[[k, j]] / pivot;
            r[[k, j]] = val;
        }

        // r[k+1:, :] -= p[k+1:, k] * r[k, :]
        for i in (k + 1)..r.nrows() {
            let p_ik = p[[i, k]];
            for j in 0..r.ncols() {
                let r_kj = r[[k, j]];
                let old = r[[i, j]];
                r[[i, j]] = old - p_ik * r_kj;
            }
        }
    }
}

#[cfg(test)]
mod tests;
