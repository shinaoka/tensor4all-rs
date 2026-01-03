//! Utility functions for matrix cross interpolation

use num_traits::{Float, Zero, One};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;
use std::ops::{Index, IndexMut};

/// Simple 2D matrix backed by Vec
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Clone> Matrix<T> {
    /// Create a new matrix from dimensions and initial value
    pub fn from_elem(nrows: usize, ncols: usize, elem: T) -> Self {
        Self {
            data: vec![elem; nrows * ncols],
            nrows,
            ncols,
        }
    }

    /// Number of rows
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns
    pub fn ncols(&self) -> usize {
        self.ncols
    }
}

impl<T: Clone + Zero> Matrix<T> {
    /// Create a zeros matrix
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![T::zero(); nrows * ncols],
            nrows,
            ncols,
        }
    }
}

impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        &self.data[idx[0] * self.ncols + idx[1]]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        &mut self.data[idx[0] * self.ncols + idx[1]]
    }
}

/// Create a zeros matrix with given dimensions
pub fn zeros<T: Clone + Zero>(nrows: usize, ncols: usize) -> Matrix<T> {
    Matrix::zeros(nrows, ncols)
}

/// Create an identity matrix
pub fn eye<T: Clone + Zero + One>(n: usize) -> Matrix<T> {
    let mut m = zeros(n, n);
    for i in 0..n {
        m[[i, i]] = T::one();
    }
    m
}

/// Create a matrix from a 2D vector (row-major)
pub fn from_vec2d<T: Clone + Zero>(data: Vec<Vec<T>>) -> Matrix<T> {
    let nrows = data.len();
    let ncols = if nrows > 0 { data[0].len() } else { 0 };
    let mut m = zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[[i, j]] = data[i][j].clone();
        }
    }
    m
}

/// Get number of rows
pub fn nrows<T>(m: &Matrix<T>) -> usize {
    m.nrows
}

/// Get number of columns
pub fn ncols<T>(m: &Matrix<T>) -> usize {
    m.ncols
}

/// Get a row as a vector
pub fn get_row<T: Clone>(m: &Matrix<T>, i: usize) -> Vec<T> {
    (0..m.ncols).map(|j| m[[i, j]].clone()).collect()
}

/// Get a column as a vector
pub fn get_col<T: Clone>(m: &Matrix<T>, j: usize) -> Vec<T> {
    (0..m.nrows).map(|i| m[[i, j]].clone()).collect()
}

/// Get a submatrix by selecting specific rows and columns
pub fn submatrix<T: Clone + Zero>(m: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T> {
    let mut result = zeros(rows.len(), cols.len());
    for (ri, &r) in rows.iter().enumerate() {
        for (ci, &c) in cols.iter().enumerate() {
            result[[ri, ci]] = m[[r, c]].clone();
        }
    }
    result
}

/// Append a column to the right of a matrix
pub fn append_col<T: Clone + Zero>(m: &Matrix<T>, col: &[T]) -> Matrix<T> {
    let nr = m.nrows;
    let nc = m.ncols;
    assert_eq!(col.len(), nr);

    let mut result = zeros(nr, nc + 1);
    for i in 0..nr {
        for j in 0..nc {
            result[[i, j]] = m[[i, j]].clone();
        }
        result[[i, nc]] = col[i].clone();
    }
    result
}

/// Append a row to the bottom of a matrix
pub fn append_row<T: Clone + Zero>(m: &Matrix<T>, row: &[T]) -> Matrix<T> {
    let nr = m.nrows;
    let nc = m.ncols;
    assert_eq!(row.len(), nc);

    let mut result = zeros(nr + 1, nc);
    for i in 0..nr {
        for j in 0..nc {
            result[[i, j]] = m[[i, j]].clone();
        }
    }
    for j in 0..nc {
        result[[nr, j]] = row[j].clone();
    }
    result
}

/// Swap two rows in a matrix (in-place style, returns new matrix)
pub fn swap_rows<T: Clone + Zero>(m: &Matrix<T>, a: usize, b: usize) -> Matrix<T> {
    if a == b {
        return m.clone();
    }
    let mut result = m.clone();
    for j in 0..m.ncols {
        let tmp = result[[a, j]].clone();
        result[[a, j]] = result[[b, j]].clone();
        result[[b, j]] = tmp;
    }
    result
}

/// Swap two columns in a matrix (in-place style, returns new matrix)
pub fn swap_cols<T: Clone + Zero>(m: &Matrix<T>, a: usize, b: usize) -> Matrix<T> {
    if a == b {
        return m.clone();
    }
    let mut result = m.clone();
    for i in 0..m.nrows {
        let tmp = result[[i, a]].clone();
        result[[i, a]] = result[[i, b]].clone();
        result[[i, b]] = tmp;
    }
    result
}

/// Transpose the matrix
pub fn transpose<T: Clone + Zero>(m: &Matrix<T>) -> Matrix<T> {
    let mut result = zeros(m.ncols, m.nrows);
    for i in 0..m.nrows {
        for j in 0..m.ncols {
            result[[j, i]] = m[[i, j]].clone();
        }
    }
    result
}

/// Trait for scalar types used in matrix operations
pub trait Scalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Send
    + Sync
    + 'static
{
    /// Absolute value
    fn abs(self) -> Self;

    /// Square of absolute value (for complex numbers, |z|^2)
    fn abs_sq(self) -> f64;

    /// Check if value is NaN
    fn is_nan(self) -> bool;

    /// Small epsilon value for numerical comparisons
    fn epsilon() -> f64 {
        1e-30
    }
}

impl Scalar for f64 {
    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn abs_sq(self) -> f64 {
        self * self
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for f32 {
    fn abs(self) -> Self {
        Float::abs(self)
    }

    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }

    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for num_complex::Complex64 {
    fn abs(self) -> Self {
        num_complex::Complex64::new(self.norm(), 0.0)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl Scalar for num_complex::Complex32 {
    fn abs(self) -> Self {
        num_complex::Complex32::new(self.norm(), 0.0)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }

    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// Calculates A * B^{-1} using Gaussian elimination for numerical stability.
pub fn a_times_b_inv<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let n = ncols(a);
    assert_eq!(nrows(b), n);
    assert_eq!(ncols(b), n);

    // Solve XB = A by solving B'X' = A'
    let bt = transpose(b);
    let at = transpose(a);
    let xt = solve_linear_system(&bt, &at);
    transpose(&xt)
}

/// Calculates A^{-1} * B using Gaussian elimination for numerical stability.
pub fn a_inv_times_b<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let bt = transpose(b);
    let at = transpose(a);
    let result = a_times_b_inv(&bt, &at);
    transpose(&result)
}

/// Solve linear system AX = B using Gaussian elimination with partial pivoting
#[allow(clippy::needless_range_loop)]
fn solve_linear_system<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let n = nrows(a);
    assert_eq!(ncols(a), n);
    assert_eq!(nrows(b), n);
    let m = ncols(b);

    // Create augmented matrix [A | B]
    let mut aug: Vec<Vec<T>> = (0..n).map(|i| {
        let mut row = Vec::with_capacity(n + m);
        for j in 0..n {
            row.push(a[[i, j]]);
        }
        for j in 0..m {
            row.push(b[[i, j]]);
        }
        row
    }).collect();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_idx = k;
        let mut max_val: f64 = aug[k][k].abs_sq();
        for i in (k + 1)..n {
            let val: f64 = aug[i][k].abs_sq();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        // Swap rows
        if max_idx != k {
            aug.swap(k, max_idx);
        }

        let pivot = aug[k][k];
        if pivot.abs_sq() < T::epsilon() {
            continue;
        }

        // Eliminate below
        for i in (k + 1)..n {
            let factor = aug[i][k] / pivot;
            for j in k..(n + m) {
                aug[i][j] = aug[i][j] - factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x: Vec<Vec<T>> = vec![vec![T::zero(); m]; n];
    for i in (0..n).rev() {
        for j in 0..m {
            let mut sum = aug[i][n + j];
            for k in (i + 1)..n {
                sum = sum - aug[i][k] * x[k][j];
            }
            let diag = aug[i][i];
            if diag.abs_sq() > T::epsilon() {
                x[i][j] = sum / diag;
            }
        }
    }

    from_vec2d(x)
}

/// Find the position of maximum absolute value in a submatrix
pub fn submatrix_argmax<T: Scalar>(
    a: &Matrix<T>,
    rows: &[usize],
    cols: &[usize],
) -> (usize, usize, T) {
    assert!(!rows.is_empty(), "rows must not be empty");
    assert!(!cols.is_empty(), "cols must not be empty");

    let mut max_val: f64 = a[[rows[0], cols[0]]].abs_sq();
    let mut max_row = rows[0];
    let mut max_col = cols[0];

    for &r in rows {
        for &c in cols {
            let val: f64 = a[[r, c]].abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    (max_row, max_col, a[[max_row, max_col]])
}

/// Select a random subset of elements from a set
pub fn random_subset<T: Clone, R: Rng>(set: &[T], n: usize, rng: &mut R) -> Vec<T> {
    let n = n.min(set.len());
    if n == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..set.len()).collect();
    indices.shuffle(rng);
    indices.truncate(n);
    indices.into_iter().map(|i| set[i].clone()).collect()
}

/// Set difference: elements in `set` that are not in `exclude`
pub fn set_diff(set: &[usize], exclude: &[usize]) -> Vec<usize> {
    let exclude_set: HashSet<usize> = exclude.iter().copied().collect();
    set.iter()
        .copied()
        .filter(|x| !exclude_set.contains(x))
        .collect()
}

/// Dot product of two vectors
pub fn dot<T: Scalar>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .fold(T::zero(), |acc, (&x, &y)| acc + x * y)
}

/// Matrix multiplication: A * B
pub fn mat_mul<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Matrix<T> {
    let m = nrows(a);
    let k = ncols(a);
    let n = ncols(b);
    assert_eq!(nrows(b), k);

    let mut result = zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::zero();
            for l in 0..k {
                sum = sum + a[[i, l]] * b[[l, j]];
            }
            result[[i, j]] = sum;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_basic() {
        let mut m = zeros::<f64>(3, 3);
        m[[0, 0]] = 1.0;
        m[[1, 1]] = 2.0;
        m[[2, 2]] = 3.0;

        assert_eq!(m[[0, 0]], 1.0);
        assert_eq!(m[[1, 1]], 2.0);
        assert_eq!(m[[2, 2]], 3.0);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
        let mt = transpose(&m);

        assert_eq!(nrows(&mt), 3);
        assert_eq!(ncols(&mt), 2);
        assert_eq!(mt[[0, 0]], 1.0);
        assert_eq!(mt[[0, 1]], 4.0);
        assert_eq!(mt[[2, 0]], 3.0);
    }

    #[test]
    fn test_submatrix_argmax() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let rows: Vec<usize> = vec![0, 1, 2];
        let cols: Vec<usize> = vec![0, 1, 2];
        let (r, c, _) = submatrix_argmax(&m, &rows, &cols);
        assert_eq!((r, c), (2, 2));
    }

    #[test]
    fn test_set_diff() {
        let set = vec![1, 2, 3, 4, 5];
        let exclude = vec![2, 4];
        let diff = set_diff(&set, &exclude);
        assert_eq!(diff, vec![1, 3, 5]);
    }

    #[test]
    fn test_mat_mul() {
        let a = from_vec2d(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let b = from_vec2d(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let c = mat_mul(&a, &b);

        assert_eq!(c[[0, 0]], 19.0);
        assert_eq!(c[[0, 1]], 22.0);
        assert_eq!(c[[1, 0]], 43.0);
        assert_eq!(c[[1, 1]], 50.0);
    }
}
