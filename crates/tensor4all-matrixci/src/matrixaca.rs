//! Adaptive Cross Approximation (MatrixACA) implementation

use crate::error::{MatrixCIError, Result};
use crate::traits::AbstractMatrixCI;
use crate::util::{
    append_col, append_row, from_vec2d, get_col, get_row, ncols, nrows, submatrix_argmax, zeros,
    Matrix, Scalar,
};

/// Adaptive Cross Approximation representation
///
/// Represents a matrix approximation using the ACA algorithm.
/// The approximation is stored as:
/// A ≈ U * diag(alpha) * V
///
/// where alpha[k] = 1/delta[k] for efficiency.
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
    pub fn new(nr: usize, nc: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            u: zeros(nr, 0),
            v: zeros(0, nc),
            alpha: Vec::new(),
        }
    }

    /// Create a MatrixACA from a matrix with an initial pivot
    pub fn from_matrix_with_pivot(a: &Matrix<T>, first_pivot: (usize, usize)) -> Self {
        let (i, j) = first_pivot;
        let pivot_val = a[[i, j]];

        // u = A[:, j]
        let u_col = get_col(a, j);
        let u = from_vec2d((0..nrows(a)).map(|r| vec![u_col[r]]).collect());

        // v = A[i, :]
        let v_row = get_row(a, i);
        let v = from_vec2d(vec![v_row]);

        let alpha = vec![T::one() / pivot_val];

        Self {
            row_indices: vec![i],
            col_indices: vec![j],
            u,
            v,
            alpha,
        }
    }

    /// Number of pivots
    pub fn npivots(&self) -> usize {
        self.alpha.len()
    }

    /// Get reference to U matrix
    pub fn u(&self) -> &Matrix<T> {
        &self.u
    }

    /// Get reference to V matrix
    pub fn v(&self) -> &Matrix<T> {
        &self.v
    }

    /// Get reference to alpha values
    pub fn alpha(&self) -> &[T] {
        &self.alpha
    }

    /// Compute u_k(x) for all x (the k-th column of U after adding a new pivot column)
    fn compute_uk(&self, a: &Matrix<T>) -> Vec<T> {
        let k = self.col_indices.len();
        let yk = self.col_indices[k - 1];

        // result = A[:, yk]
        let mut result: Vec<T> = get_col(a, yk);

        // Subtract contribution from previous pivots
        for l in 0..(k - 1) {
            let xl = self.row_indices[l];
            let v_l_yk = self.v[[l, yk]];
            let u_xl_l = self.u[[xl, l]];
            let factor = v_l_yk / u_xl_l;

            for (i, res) in result.iter_mut().enumerate() {
                *res = *res - factor * self.u[[i, l]];
            }
        }

        result
    }

    /// Compute v_k(y) for all y (the k-th row of V after adding a new pivot row)
    fn compute_vk(&self, a: &Matrix<T>) -> Vec<T> {
        let k = self.row_indices.len();
        let xk = self.row_indices[k - 1];

        // result = A[xk, :]
        let mut result: Vec<T> = get_row(a, xk);

        // Subtract contribution from previous pivots
        for l in 0..(k - 1) {
            let xl = self.row_indices[l];
            let u_xk_l = self.u[[xk, l]];
            let u_xl_l = self.u[[xl, l]];
            let factor = u_xk_l / u_xl_l;

            for (j, res) in result.iter_mut().enumerate() {
                *res = *res - factor * self.v[[l, j]];
            }
        }

        result
    }

    /// Add a pivot column
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
        let uk = self.compute_uk(a);
        self.u = append_col(&self.u, &uk);

        Ok(())
    }

    /// Add a pivot row
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
        let vk = self.compute_vk(a);
        self.v = append_row(&self.v, &vk);

        // Update alpha
        let xk = row_index;
        let u_xk_last = self.u[[xk, ncols(&self.u) - 1]];
        self.alpha.push(T::one() / u_xk_last);

        Ok(())
    }

    /// Add a pivot at the given position
    pub fn add_pivot(&mut self, a: &Matrix<T>, pivot: (usize, usize)) -> Result<()> {
        self.add_pivot_col(a, pivot.1)?;
        self.add_pivot_row(a, pivot.0)?;
        Ok(())
    }

    /// Add a pivot that maximizes the error using ACA heuristic
    pub fn add_best_pivot(&mut self, a: &Matrix<T>) -> Result<(usize, usize)> {
        if self.is_empty() {
            // Find global maximum
            let rows: Vec<usize> = (0..self.nrows()).collect();
            let cols: Vec<usize> = (0..self.ncols()).collect();
            let (i, j, _) = submatrix_argmax(a, &rows, &cols);
            self.add_pivot(a, (i, j))?;
            return Ok((i, j));
        }

        // ACA heuristic: find max in last row of V, then max in corresponding column of U
        let avail_cols = self.available_cols();
        if avail_cols.is_empty() {
            return Err(MatrixCIError::FullRank);
        }

        // Find column with max |v[last, j]| among available columns
        let last_row = nrows(&self.v) - 1;
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

        let last_col = ncols(&self.u) - 1;
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
    pub fn set_cols(&mut self, new_pivot_rows: &Matrix<T>, permutation: &[usize]) {
        // Permute column indices
        self.col_indices = self
            .col_indices
            .iter()
            .map(|&c| permutation[c])
            .collect();

        // Permute V matrix columns
        let mut temp_v = zeros(nrows(&self.v), ncols(new_pivot_rows));
        for i in 0..nrows(&self.v) {
            for (new_j, &old_j) in permutation.iter().enumerate() {
                if old_j < ncols(&self.v) {
                    temp_v[[i, new_j]] = self.v[[i, old_j]];
                }
            }
        }
        self.v = temp_v;

        // Insert new elements
        let new_indices: Vec<usize> = (0..ncols(new_pivot_rows))
            .filter(|j| !permutation.contains(j))
            .collect();

        for k in 0..nrows(new_pivot_rows) {
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
    pub fn set_rows(&mut self, new_pivot_cols: &Matrix<T>, permutation: &[usize]) {
        // Permute row indices
        self.row_indices = self
            .row_indices
            .iter()
            .map(|&r| permutation[r])
            .collect();

        // Permute U matrix rows
        let mut temp_u = zeros(nrows(new_pivot_cols), ncols(&self.u));
        for (new_i, &old_i) in permutation.iter().enumerate() {
            if old_i < nrows(&self.u) {
                for j in 0..ncols(&self.u) {
                    temp_u[[new_i, j]] = self.u[[old_i, j]];
                }
            }
        }
        self.u = temp_u;

        // Insert new elements
        let new_indices: Vec<usize> = (0..nrows(new_pivot_cols))
            .filter(|i| !permutation.contains(i))
            .collect();

        for k in 0..ncols(new_pivot_cols) {
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
        nrows(&self.u)
    }

    fn ncols(&self) -> usize {
        ncols(&self.v)
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
            return zeros(rows.len(), cols.len());
        }

        let r = self.rank();
        let mut result = zeros(rows.len(), cols.len());

        for (ri, &row) in rows.iter().enumerate() {
            for (ci, &col) in cols.iter().enumerate() {
                let mut sum = T::zero();
                for k in 0..r {
                    sum = sum + self.u[[row, k]] * self.alpha[k] * self.v[[k, col]];
                }
                result[[ri, ci]] = sum;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrixaca_new() {
        let aca = MatrixACA::<f64>::new(5, 5);
        assert_eq!(aca.nrows(), 5);
        assert_eq!(aca.ncols(), 5);
        assert_eq!(aca.rank(), 0);
        assert!(aca.is_empty());
    }

    #[test]
    fn test_matrixaca_from_matrix() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let aca = MatrixACA::from_matrix_with_pivot(&m, (1, 1));
        assert_eq!(aca.nrows(), 3);
        assert_eq!(aca.ncols(), 3);
        assert_eq!(aca.rank(), 1);

        // Check that evaluation at pivot is correct
        let val = aca.evaluate(1, 1);
        assert!((val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrixaca_add_pivot() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0));
        assert!(aca.add_pivot(&m, (1, 1)).is_ok());
        assert_eq!(aca.rank(), 2);
    }
}
