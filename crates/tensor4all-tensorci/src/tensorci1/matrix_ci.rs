use crate::error::{Result, TCIError};
use tensor4all_core::Scalar;
#[cfg(test)]
use tensor4all_tensorbackend::{solve_matrix, transpose};
use tensor4all_tensorbackend::{Matrix, MatrixSolveScalar};

#[derive(Debug, Clone)]
pub(super) struct MatrixCI<T: Scalar> {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
}

impl<T> MatrixCI<T>
where
    T: Scalar + MatrixSolveScalar,
{
    pub(super) fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        left: Matrix<T>,
        right: Matrix<T>,
    ) -> Result<Self> {
        if col_indices.len() != left.ncols() || row_indices.len() != right.nrows() {
            return Err(TCIError::DimensionMismatch {
                message: "MatrixCI left/right rank mismatch".to_string(),
            });
        }
        Ok(Self {
            row_indices,
            col_indices,
            left,
            right,
        })
    }

    #[cfg(test)]
    pub(super) fn rank(&self) -> usize {
        self.left.ncols()
    }

    #[cfg(test)]
    pub(super) fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    #[cfg(test)]
    pub(super) fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub(super) fn pivot_cols(&self) -> &Matrix<T> {
        &self.left
    }

    pub(super) fn pivot_rows(&self) -> &Matrix<T> {
        &self.right
    }

    pub(super) fn add_pivot_col(&mut self, matrix: &Matrix<T>, col_index: usize) -> Result<()> {
        if matrix.nrows() != self.left.nrows() || matrix.ncols() != self.right.ncols() {
            return Err(TCIError::DimensionMismatch {
                message: "MatrixCI source matrix shape mismatch".to_string(),
            });
        }
        if col_index >= matrix.ncols() {
            return Err(TCIError::IndexOutOfBounds {
                message: format!("column {col_index} is outside 0..{}", matrix.ncols()),
            });
        }
        if self.col_indices.contains(&col_index) {
            return Err(TCIError::InvalidPivot {
                message: format!("column {col_index} already has a pivot"),
            });
        }

        let mut next = Matrix::zeros(self.left.nrows(), self.left.ncols() + 1);
        for col in 0..self.left.ncols() {
            for row in 0..self.left.nrows() {
                next[[row, col]] = self.left[[row, col]];
            }
        }
        for row in 0..matrix.nrows() {
            next[[row, self.left.ncols()]] = matrix[[row, col_index]];
        }
        self.left = next;
        self.col_indices.push(col_index);
        Ok(())
    }

    pub(super) fn add_pivot_row(&mut self, matrix: &Matrix<T>, row_index: usize) -> Result<()> {
        if matrix.nrows() != self.left.nrows() || matrix.ncols() != self.right.ncols() {
            return Err(TCIError::DimensionMismatch {
                message: "MatrixCI source matrix shape mismatch".to_string(),
            });
        }
        if row_index >= matrix.nrows() {
            return Err(TCIError::IndexOutOfBounds {
                message: format!("row {row_index} is outside 0..{}", matrix.nrows()),
            });
        }
        if self.row_indices.contains(&row_index) {
            return Err(TCIError::InvalidPivot {
                message: format!("row {row_index} already has a pivot"),
            });
        }

        let mut next = Matrix::zeros(self.right.nrows() + 1, self.right.ncols());
        for col in 0..self.right.ncols() {
            for row in 0..self.right.nrows() {
                next[[row, col]] = self.right[[row, col]];
            }
            next[[self.right.nrows(), col]] = matrix[[row_index, col]];
        }
        self.right = next;
        self.row_indices.push(row_index);
        Ok(())
    }

    pub(super) fn pivot_matrix(&self) -> Matrix<T> {
        let mut pivot = Matrix::zeros(self.row_indices.len(), self.left.ncols());
        for (i_out, &row) in self.row_indices.iter().enumerate() {
            for j_out in 0..self.left.ncols() {
                pivot[[i_out, j_out]] = self.left[[row, j_out]];
            }
        }
        pivot
    }

    #[cfg(test)]
    fn left_matrix(&self) -> Result<Matrix<T>> {
        self.ensure_square_rank_for_evaluation()?;
        right_solve(&self.left, &self.pivot_matrix())
    }

    #[cfg(test)]
    pub(super) fn evaluate(&self, row: usize, col: usize) -> Result<T> {
        let left = self.left_matrix()?;
        let mut value = T::zero();
        for k in 0..self.rank() {
            value = value + left[[row, k]] * self.right[[k, col]];
        }
        Ok(value)
    }

    #[cfg(test)]
    fn ensure_square_rank_for_evaluation(&self) -> Result<()> {
        if self.left.ncols() != self.right.nrows() {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "MatrixCI evaluation requires matching left/right ranks, got {} and {}",
                    self.left.ncols(),
                    self.right.nrows()
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
fn right_solve<T>(a: &Matrix<T>, p: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Scalar + MatrixSolveScalar,
{
    let solution_t =
        solve_matrix(&transpose(p), &transpose(a)).map_err(|err| TCIError::InvalidOperation {
            message: format!("MatrixCI right solve failed: {err}"),
        })?;
    Ok(transpose(&solution_t))
}
