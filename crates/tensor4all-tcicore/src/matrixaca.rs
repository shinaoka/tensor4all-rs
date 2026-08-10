//! Adaptive Cross Approximation (ACA) implementation.
//!
//! [`MatrixACA`] provides the ACA algorithm for incrementally building a
//! low-rank approximation of a matrix. It stores the approximation in
//! factored form `A ~ U * diag(alpha) * V` and supports adding pivots
//! one at a time.

use crate::error::{MatrixCIError, Result};
use crate::scalar::Scalar;
use crate::traits::AbstractMatrixCI;
use tensor4all_tensorbackend::{submatrix_argmax, Matrix};

fn matrix_row<T: Scalar>(m: &Matrix<T>, i: usize) -> Vec<T> {
    (0..m.ncols()).map(|j| m[[i, j]]).collect()
}

fn matrix_col<T: Scalar>(m: &Matrix<T>, j: usize) -> Vec<T> {
    (0..m.nrows()).map(|i| m[[i, j]]).collect()
}

fn append_col<T: Scalar>(m: &Matrix<T>, col: &[T]) -> Matrix<T> {
    assert_eq!(col.len(), m.nrows());

    let mut result = Matrix::zeros(m.nrows(), m.ncols() + 1);
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            result[[i, j]] = m[[i, j]];
        }
        result[[i, m.ncols()]] = col[i];
    }
    result
}

fn append_row<T: Scalar>(m: &Matrix<T>, row: &[T]) -> Matrix<T> {
    assert_eq!(row.len(), m.ncols());

    let mut result = Matrix::zeros(m.nrows() + 1, m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            result[[i, j]] = m[[i, j]];
        }
    }
    for j in 0..m.ncols() {
        result[[m.nrows(), j]] = row[j];
    }
    result
}

/// Adaptive Cross Approximation representation.
///
/// Stores a low-rank approximation `A ~ U * diag(alpha) * V`, where
/// `alpha[k] = 1/delta[k]`. Implements [`AbstractMatrixCI`] for
/// element-wise evaluation and submatrix extraction.
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
///     vec![7.0, 8.0, 9.0],
/// ]);
///
/// // Build ACA from a starting pivot
/// let mut aca = MatrixACA::from_matrix_with_pivot(&m, (1, 1)).unwrap();
/// assert_eq!(aca.rank(), 1);
///
/// // Add more pivots
/// aca.add_pivot(&m, (0, 0)).unwrap();
/// assert_eq!(aca.rank(), 2);
///
/// // Evaluate the approximation
/// let approx_11 = aca.evaluate(1, 1);
/// assert!((approx_11 - 5.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct MatrixACA<T: Scalar> {
    /// Row indices (I set)
    row_indices: Vec<usize>,
    /// Column indices (J set)
    col_indices: Vec<usize>,
    /// U matrix: u_k(x) for all x, shape (nrows, npivots)
    u: Matrix<T>,
    /// V matrix: v_k(y) for all y, shape (npivots, ncols)
    v: Matrix<T>,
    /// Alpha values: 1/delta for each pivot
    alpha: Vec<T>,
}

impl<T: Scalar> MatrixACA<T> {
    /// Create an empty MatrixACA for a matrix of given size
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    ///
    /// let aca = MatrixACA::<f64>::new(3, 4);
    /// assert_eq!(aca.nrows(), 3);
    /// assert_eq!(aca.ncols(), 4);
    /// assert_eq!(aca.rank(), 0);
    /// assert!(aca.is_empty());
    /// ```
    pub fn new(nr: usize, nc: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            u: Matrix::zeros(nr, 0),
            v: Matrix::zeros(0, nc),
            alpha: Vec::new(),
        }
    }

    /// Create a MatrixACA from a matrix with an initial pivot
    ///
    /// Returns an error if the pivot value is zero or near-zero.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix dimensions are invalid (a shape mismatch)
    /// /// or the initial pivot is out of range (an out of bounds failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// assert_eq!(aca.rank(), 1);
    /// assert_eq!(aca.row_indices(), &[0]);
    /// assert_eq!(aca.col_indices(), &[0]);
    /// ```
    pub fn from_matrix_with_pivot(a: &Matrix<T>, first_pivot: (usize, usize)) -> Result<Self> {
        let (i, j) = first_pivot;
        let pivot_val = a[[i, j]];

        if pivot_val.abs_sq() < f64::EPSILON * f64::EPSILON {
            return Err(MatrixCIError::SingularMatrix);
        }

        // u = A[:, j]
        let mut u = Matrix::zeros(a.nrows(), 1);
        for r in 0..a.nrows() {
            u[[r, 0]] = a[[r, j]];
        }

        // v = A[i, :]
        let mut v = Matrix::zeros(1, a.ncols());
        for c in 0..a.ncols() {
            v[[0, c]] = a[[i, c]];
        }

        let alpha = vec![T::one() / pivot_val];

        Ok(Self {
            row_indices: vec![i],
            col_indices: vec![j],
            u,
            v,
            alpha,
        })
    }

    /// Number of pivots
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// assert_eq!(aca.npivots(), 1);
    /// aca.add_pivot(&m, (1, 1)).unwrap();
    /// assert_eq!(aca.npivots(), 2);
    /// ```
    pub fn npivots(&self) -> usize {
        self.alpha.len()
    }

    /// Get reference to U matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// let u = aca.u();
    /// assert_eq!(u.nrows(), 2); // nrows of original matrix
    /// assert_eq!(u.ncols(), 1); // one pivot
    /// ```
    pub fn u(&self) -> &Matrix<T> {
        &self.u
    }

    /// Get reference to V matrix
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// let v = aca.v();
    /// assert_eq!(v.nrows(), 1); // one pivot
    /// assert_eq!(v.ncols(), 2); // ncols of original matrix
    /// ```
    pub fn v(&self) -> &Matrix<T> {
        &self.v
    }

    /// Get reference to alpha values
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 3.0]]);
    /// let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// let alpha = aca.alpha();
    /// assert_eq!(alpha.len(), 1);
    /// // alpha = 1 / pivot_value = 1 / m[0,0] = 0.5
    /// assert!((alpha[0] - 0.5).abs() < 1e-10);
    /// ```
    pub fn alpha(&self) -> &[T] {
        &self.alpha
    }

    /// Compute u_k(x) for all x (the k-th column of U after adding a new pivot column)
    fn compute_uk(&self, a: &Matrix<T>) -> Result<Vec<T>> {
        let k = self.col_indices.len();
        let yk = self.col_indices[k - 1];

        // result = A[:, yk]
        let mut result: Vec<T> = matrix_col(a, yk);

        // Subtract contribution from previous pivots
        for l in 0..(k - 1) {
            let xl = self.row_indices[l];
            let v_l_yk = self.v[[l, yk]];
            let u_xl_l = self.u[[xl, l]];
            if u_xl_l.abs_sq() < f64::EPSILON * f64::EPSILON {
                return Err(MatrixCIError::SingularMatrix);
            }
            let factor = v_l_yk / u_xl_l;

            for (i, res) in result.iter_mut().enumerate() {
                *res = *res - factor * self.u[[i, l]];
            }
        }

        Ok(result)
    }

    /// Compute v_k(y) for all y (the k-th row of V after adding a new pivot row)
    fn compute_vk(&self, a: &Matrix<T>) -> Result<Vec<T>> {
        let k = self.row_indices.len();
        let xk = self.row_indices[k - 1];

        // result = A[xk, :]
        let mut result: Vec<T> = matrix_row(a, xk);

        // Subtract contribution from previous pivots
        for l in 0..(k - 1) {
            let xl = self.row_indices[l];
            let u_xk_l = self.u[[xk, l]];
            let u_xl_l = self.u[[xl, l]];
            if u_xl_l.abs_sq() < f64::EPSILON * f64::EPSILON {
                return Err(MatrixCIError::SingularMatrix);
            }
            let factor = u_xk_l / u_xl_l;

            for (j, res) in result.iter_mut().enumerate() {
                *res = *res - factor * self.v[[l, j]];
            }
        }

        Ok(result)
    }

    /// Add a pivot column
    ///
    /// # Errors
    ///
    /// Returns an error when the pivot column is out of range (an out of bounds
    /// /// failure) or the ACA update fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::<f64>::new(2, 2);
    /// aca.add_pivot_col(&m, 0).unwrap();
    /// assert_eq!(aca.u().ncols(), 1);
    /// ```
    pub fn add_pivot_col(&mut self, a: &Matrix<T>, col_index: usize) -> Result<()> {
        if col_index >= self.ncols() {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: 0,
                col: col_index,
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }

        self.col_indices.push(col_index);
        let uk = self.compute_uk(a)?;
        self.u = append_col(&self.u, &uk);

        Ok(())
    }

    /// Add a pivot row
    ///
    /// # Errors
    ///
    /// Returns an error when the pivot row is out of range (an out of bounds
    /// /// failure) or the ACA update fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::MatrixACA;
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::<f64>::new(2, 2);
    /// aca.add_pivot_col(&m, 0).unwrap();
    /// aca.add_pivot_row(&m, 0).unwrap();
    /// assert_eq!(aca.npivots(), 1);
    /// assert_eq!(aca.v().nrows(), 1);
    /// ```
    pub fn add_pivot_row(&mut self, a: &Matrix<T>, row_index: usize) -> Result<()> {
        if row_index >= self.nrows() {
            return Err(MatrixCIError::IndexOutOfBounds {
                row: row_index,
                col: 0,
                nrows: self.nrows(),
                ncols: self.ncols(),
            });
        }

        self.row_indices.push(row_index);
        let vk = self.compute_vk(a)?;
        self.v = append_row(&self.v, &vk);

        // Update alpha
        let xk = row_index;
        let u_xk_last = self.u[[xk, self.u.ncols() - 1]];
        if u_xk_last.abs_sq() < f64::EPSILON * f64::EPSILON {
            return Err(MatrixCIError::SingularMatrix);
        }
        self.alpha.push(T::one() / u_xk_last);

        Ok(())
    }

    /// Add a pivot at the given position
    ///
    /// # Errors
    ///
    /// Returns an error when the pivot is out of range (an out of bounds failure)
    /// /// or the ACA update fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// aca.add_pivot(&m, (1, 1)).unwrap();
    /// assert_eq!(aca.rank(), 2);
    /// // Exact reconstruction at all positions
    /// for i in 0..2 {
    ///     for j in 0..2 {
    ///         assert!((aca.evaluate(i, j) - m[[i, j]]).abs() < 1e-10);
    ///     }
    /// }
    /// ```
    pub fn add_pivot(&mut self, a: &Matrix<T>, pivot: (usize, usize)) -> Result<()> {
        self.add_pivot_col(a, pivot.1)?;
        self.add_pivot_row(a, pivot.0)?;
        Ok(())
    }

    /// Add a pivot that maximizes the error using ACA heuristic
    ///
    /// # Errors
    ///
    /// Returns an error when no valid pivot exists (a singular failure) or the
    /// /// ACA update fails.
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
    /// let mut aca = MatrixACA::<f64>::new(3, 3);
    /// let (r, c) = aca.add_best_pivot(&m).unwrap();
    /// assert!(r < 3);
    /// assert!(c < 3);
    /// assert_eq!(aca.rank(), 1);
    /// ```
    pub fn add_best_pivot(&mut self, a: &Matrix<T>) -> Result<(usize, usize)> {
        if self.is_empty() {
            // Find global maximum
            let (i, j, _) = submatrix_argmax(a, 0..self.nrows(), 0..self.ncols());
            self.add_pivot(a, (i, j))?;
            return Ok((i, j));
        }

        // ACA heuristic: find max in last row of V, then max in corresponding column of U
        let avail_cols = self.available_cols();
        if avail_cols.is_empty() {
            return Err(MatrixCIError::FullRank);
        }

        // Find column with max |v[last, j]| among available columns
        let last_row = self.v.nrows() - 1;
        let mut max_val = self.v[[last_row, avail_cols[0]]].abs_sq();
        let mut best_col = avail_cols[0];
        for &c in &avail_cols {
            let val = self.v[[last_row, c]].abs_sq();
            if val > max_val {
                max_val = val;
                best_col = c;
            }
        }

        self.add_pivot_col(a, best_col)?;

        // Find row with max |u[i, last]| among available rows
        let avail_rows = self.available_rows();
        if avail_rows.is_empty() {
            return Err(MatrixCIError::FullRank);
        }

        let last_col = self.u.ncols() - 1;
        let mut max_val = self.u[[avail_rows[0], last_col]].abs_sq();
        let mut best_row = avail_rows[0];
        for &r in &avail_rows {
            let val = self.u[[r, last_col]].abs_sq();
            if val > max_val {
                max_val = val;
                best_row = r;
            }
        }

        self.add_pivot_row(a, best_row)?;

        Ok((best_row, best_col))
    }

    /// Set columns with new pivot rows and permutation
    ///
    /// Updates the column indices and V matrix according to the given
    /// permutation and new pivot row data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// // Identity permutation keeps columns the same
    /// let pivot_rows = from_vec2d(vec![vec![1.0_f64, 2.0]]);
    /// aca.set_cols(&pivot_rows, &[0, 1]);
    /// assert_eq!(aca.col_indices(), &[0]);
    /// ```
    pub fn set_cols(&mut self, new_pivot_rows: &Matrix<T>, permutation: &[usize]) {
        // Permute column indices
        self.col_indices = self.col_indices.iter().map(|&c| permutation[c]).collect();

        // Permute V matrix columns
        let mut temp_v = Matrix::zeros(self.v.nrows(), new_pivot_rows.ncols());
        for i in 0..self.v.nrows() {
            for (new_j, &old_j) in permutation.iter().enumerate() {
                if old_j < self.v.ncols() {
                    temp_v[[i, new_j]] = self.v[[i, old_j]];
                }
            }
        }
        self.v = temp_v;

        // Insert new elements
        let new_indices: Vec<usize> = (0..new_pivot_rows.ncols())
            .filter(|j| !permutation.contains(j))
            .collect();

        for k in 0..new_pivot_rows.nrows() {
            for &j in &new_indices {
                let mut val = new_pivot_rows[[k, j]];
                for l in 0..k {
                    let factor = self.u[[self.row_indices[k], l]] * self.alpha[l];
                    val = val - self.v[[l, j]] * factor;
                }
                self.v[[k, j]] = val;
            }
        }
    }

    /// Set rows with new pivot columns and permutation
    ///
    /// Updates the row indices and U matrix according to the given
    /// permutation and new pivot column data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::{AbstractMatrixCI, MatrixACA};
    /// use tensor4all_tensorbackend::from_vec2d;
    ///
    /// let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    /// let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    /// // Identity permutation keeps rows the same
    /// let pivot_cols = from_vec2d(vec![vec![1.0_f64], vec![3.0]]);
    /// aca.set_rows(&pivot_cols, &[0, 1]);
    /// assert_eq!(aca.row_indices(), &[0]);
    /// ```
    pub fn set_rows(&mut self, new_pivot_cols: &Matrix<T>, permutation: &[usize]) {
        // Permute row indices
        self.row_indices = self.row_indices.iter().map(|&r| permutation[r]).collect();

        // Permute U matrix rows
        let mut temp_u = Matrix::zeros(new_pivot_cols.nrows(), self.u.ncols());
        for (new_i, &old_i) in permutation.iter().enumerate() {
            if old_i < self.u.nrows() {
                for j in 0..self.u.ncols() {
                    temp_u[[new_i, j]] = self.u[[old_i, j]];
                }
            }
        }
        self.u = temp_u;

        // Insert new elements
        let new_indices: Vec<usize> = (0..new_pivot_cols.nrows())
            .filter(|i| !permutation.contains(i))
            .collect();

        for k in 0..new_pivot_cols.ncols() {
            for &i in &new_indices {
                let mut val = new_pivot_cols[[i, k]];
                for l in 0..k {
                    let factor = self.v[[l, self.col_indices[k]]] * self.alpha[l];
                    val = val - self.u[[i, l]] * factor;
                }
                self.u[[i, k]] = val;
            }
        }
    }
}

impl<T: Scalar> AbstractMatrixCI<T> for MatrixACA<T> {
    fn nrows(&self) -> usize {
        self.u.nrows()
    }

    fn ncols(&self) -> usize {
        self.v.ncols()
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

        let mut sum = T::zero();
        for k in 0..self.rank() {
            sum = sum + self.u[[i, k]] * self.alpha[k] * self.v[[k, j]];
        }
        sum
    }

    fn submatrix(&self, rows: &[usize], cols: &[usize]) -> Matrix<T> {
        if self.is_empty() {
            return Matrix::zeros(rows.len(), cols.len());
        }

        let r = self.rank();
        let mut result = Matrix::zeros(rows.len(), cols.len());
        for (j_out, &col) in cols.iter().enumerate() {
            for (i_out, &row) in rows.iter().enumerate() {
                let mut sum = T::zero();
                for k in 0..r {
                    sum = sum + self.u[[row, k]] * self.alpha[k] * self.v[[k, col]];
                }
                result[[i_out, j_out]] = sum;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests;
