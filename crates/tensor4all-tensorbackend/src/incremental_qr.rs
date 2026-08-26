//! Incremental QR factorization for column-major dense matrices.

use anyhow::anyhow;
use num_complex::{Complex32, Complex64, ComplexFloat};

use crate::backend::{
    qr_backend, src_error_estimate, BackendLinalgError, BackendLinalgScalar,
    MatrixTriangularSolveScalar,
};
use crate::matrix::{mat_mul, Matrix, MatrixScalar};

/// Scalar operations required by [`IncrementalQr`].
///
/// The implementation is provided for the four scalar types supported by the
/// backend. The conjugation hook keeps the update correct for complex matrices
/// while using the same column-major algorithm for real matrices.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::IncrementalQrScalar;
/// assert_eq!(<f64 as IncrementalQrScalar>::conjugate(2.0), 2.0);
/// ```
pub trait IncrementalQrScalar:
    BackendLinalgScalar + MatrixScalar + MatrixTriangularSolveScalar + ComplexFloat
{
    /// Return the Hermitian conjugate of one scalar.
    fn conjugate(self) -> Self;
}

impl IncrementalQrScalar for f32 {
    fn conjugate(self) -> Self {
        self
    }
}

impl IncrementalQrScalar for f64 {
    fn conjugate(self) -> Self {
        self
    }
}

impl IncrementalQrScalar for Complex32 {
    fn conjugate(self) -> Self {
        self.conj()
    }
}

impl IncrementalQrScalar for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }
}

/// Thin QR state that can append columns without refactorizing the old block.
///
/// The state stores explicit thin `Q` and square `R` factors. Appending a
/// `m × k` block performs one projection `Q†B`, one residual GEMM, and a QR of
/// only the residual block; it never reruns QR on the complete accumulated
/// matrix. This is the update required by the adaptive SRC path.
///
/// The matrix layout is column-major throughout. The current state must have
/// at least as many rows as columns, and appends are accepted only while the
/// resulting factorization remains thin.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{IncrementalQr, Matrix, mat_mul};
///
/// let first = Matrix::from_col_major_vec(3, 1, vec![1.0_f64, 2.0, 3.0]);
/// let appended = Matrix::from_col_major_vec(3, 1, vec![2.0, 0.0, 1.0]);
/// let mut qr = IncrementalQr::new(first).unwrap();
/// qr.append(&appended).unwrap();
/// let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
/// assert!(reconstructed
///     .as_col_major_slice()
///     .iter()
///     .zip([1.0, 2.0, 3.0, 2.0, 0.0, 1.0])
///     .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
/// ```
#[derive(Debug, Clone)]
pub struct IncrementalQr<T> {
    q: Matrix<T>,
    r: Matrix<T>,
}

impl<T> IncrementalQr<T>
where
    T: IncrementalQrScalar,
{
    /// Factorize a non-empty tall-or-square matrix into thin `Q` and square `R`.
    ///
    /// # Arguments
    /// * `input` - Column-major `m × n` matrix with `m >= n` and `n > 0`.
    ///
    /// # Returns
    /// A state containing factors satisfying `input = Q R` up to backend
    /// floating-point error.
    ///
    /// # Errors
    /// Returns a backend error when the matrix is empty, wide, malformed, or
    /// the backend QR factorization fails.
    pub fn new(input: Matrix<T>) -> std::result::Result<Self, BackendLinalgError> {
        if input.nrows() == 0 || input.ncols() == 0 {
            return Err(anyhow!("incremental QR requires a non-empty matrix").into());
        }
        if input.nrows() < input.ncols() {
            return Err(anyhow!(
                "incremental QR requires a tall-or-square matrix, got {}x{}",
                input.nrows(),
                input.ncols()
            )
            .into());
        }

        let (q, r) = qr_matrix(&input)?;
        Ok(Self { q, r })
    }

    /// Append a column block using the existing QR state.
    ///
    /// # Arguments
    /// * `new_columns` - Column-major `m × k` block with the same row count as
    ///   the initial matrix and `k > 0`.
    ///
    /// # Returns
    /// Updates this state in place so that `Q R` represents the original
    /// matrix followed by `new_columns`.
    ///
    /// # Errors
    /// Returns a backend error when row counts differ, the append is empty, the
    /// resulting matrix would be wide, or a residual QR update fails.
    pub fn append(
        &mut self,
        new_columns: &Matrix<T>,
    ) -> std::result::Result<(), BackendLinalgError> {
        if new_columns.nrows() != self.q.nrows() {
            return Err(anyhow!(
                "incremental QR append row count {} does not match {}",
                new_columns.nrows(),
                self.q.nrows()
            )
            .into());
        }
        if new_columns.ncols() == 0 {
            return Err(anyhow!("incremental QR append requires at least one column").into());
        }
        let old_rank = self.r.ncols();
        let new_rank = old_rank
            .checked_add(new_columns.ncols())
            .ok_or_else(|| anyhow!("incremental QR rank overflow"))?;
        if new_rank > self.q.nrows() {
            return Err(anyhow!(
                "incremental QR append would produce a wide factorization: {} rows, {} columns",
                self.q.nrows(),
                new_rank
            )
            .into());
        }

        let coupling = conjugate_transpose_mat_mul(&self.q, new_columns)?;
        let projected = mat_mul(&self.q, &coupling)
            .map_err(|error| anyhow!("incremental QR projection failed: {error}"))?;
        let mut residual = new_columns.clone();
        for (value, projection) in residual
            .as_col_major_mut_slice()
            .iter_mut()
            .zip(projected.as_col_major_slice())
        {
            *value = *value - *projection;
        }

        let (q2, r2) = qr_matrix(&residual)?;
        let mut q_data = Vec::with_capacity(self.q.nrows() * new_rank);
        q_data.extend_from_slice(self.q.as_col_major_slice());
        q_data.extend_from_slice(q2.as_col_major_slice());
        let q = Matrix::from_col_major_vec(self.q.nrows(), new_rank, q_data);

        let mut r = Matrix::zeros(new_rank, new_rank);
        for col in 0..old_rank {
            for row in 0..old_rank {
                r[[row, col]] = self.r[[row, col]];
            }
        }
        for col in 0..new_columns.ncols() {
            for row in 0..old_rank {
                r[[row, old_rank + col]] = coupling[[row, col]];
            }
            for row in 0..new_columns.ncols() {
                r[[old_rank + row, old_rank + col]] = r2[[row, col]];
            }
        }

        self.q = q;
        self.r = r;
        Ok(())
    }

    /// Return a copy of the current thin `Q` factor.
    pub fn q(&self) -> Matrix<T> {
        self.q.clone()
    }

    /// Return a copy of the current square upper-triangular `R` factor.
    pub fn r(&self) -> Matrix<T> {
        self.r.clone()
    }

    /// Compute the Appendix C SRC estimate from the current `R` factor.
    ///
    /// # Returns
    /// The randomized residual and norm estimates associated with the current
    /// sketch width.
    ///
    /// # Errors
    /// Returns a backend error when the current factor is singular or contains
    /// invalid values.
    pub fn error_estimate(
        &self,
    ) -> std::result::Result<crate::SrcErrorEstimate, BackendLinalgError> {
        src_error_estimate(&self.r)
    }
}

fn qr_matrix<T>(
    input: &Matrix<T>,
) -> std::result::Result<(Matrix<T>, Matrix<T>), BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let (q, r) = qr_backend(&input.to_typed_tensor())?;
    let q = Matrix::try_from_typed_tensor(q)
        .map_err(|error| anyhow!("incremental QR Q conversion failed: {error}"))?;
    let r = Matrix::try_from_typed_tensor(r)
        .map_err(|error| anyhow!("incremental QR R conversion failed: {error}"))?;
    Ok((q, r))
}

fn conjugate_transpose_mat_mul<T>(
    q: &Matrix<T>,
    b: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if q.nrows() != b.nrows() {
        return Err(anyhow!("incremental QR projection row mismatch").into());
    }
    let mut result = Matrix::zeros(q.ncols(), b.ncols());
    for col in 0..b.ncols() {
        for q_col in 0..q.ncols() {
            let mut value = T::zero();
            for row in 0..q.nrows() {
                value = value + q[[row, q_col]].conjugate() * b[[row, col]];
            }
            result[[q_col, col]] = value;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::IncrementalQr;
    use crate::{mat_mul, Matrix};
    use num_complex::Complex64;

    fn reconstruction_error(qr: &IncrementalQr<f64>, original: &Matrix<f64>) -> f64 {
        let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
        reconstructed
            .as_col_major_slice()
            .iter()
            .zip(original.as_col_major_slice())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn incremental_qr_reconstructs_after_appending_columns() {
        let first = Matrix::from_col_major_vec(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.5, 3.0, 4.0],
        );
        let appended = Matrix::from_col_major_vec(5, 1, vec![0.5, -1.0, 2.0, 1.5, 3.5]);
        let mut original_data = first.as_col_major_slice().to_vec();
        original_data.extend_from_slice(appended.as_col_major_slice());
        let original = Matrix::from_col_major_vec(5, 3, original_data);

        let mut qr = IncrementalQr::new(first).unwrap();
        assert_eq!(qr.q().ncols(), 2);
        assert_eq!(qr.r().nrows(), 2);
        qr.append(&appended).unwrap();

        assert_eq!(qr.q().nrows(), 5);
        assert_eq!(qr.q().ncols(), 3);
        assert_eq!(qr.r().nrows(), 3);
        assert_eq!(qr.r().ncols(), 3);
        assert!(reconstruction_error(&qr, &original) < 1.0e-12);
    }

    #[test]
    fn incremental_qr_rejects_invalid_shapes() {
        let initial = Matrix::from_col_major_vec(3, 1, vec![1.0, 2.0, 3.0]);
        let mut qr = IncrementalQr::new(initial).unwrap();
        let wrong_rows = Matrix::from_col_major_vec(2, 1, vec![1.0, 2.0]);
        assert!(qr.append(&wrong_rows).is_err());

        let too_many = Matrix::from_col_major_vec(3, 3, vec![1.0; 9]);
        assert!(qr.append(&too_many).is_err());
    }

    #[test]
    fn incremental_qr_uses_the_hermitian_projection_for_complex_columns() {
        let first = Matrix::from_col_major_vec(
            4,
            1,
            vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(-1.0, 0.5),
                Complex64::new(0.25, 2.0),
            ],
        );
        let appended = Matrix::from_col_major_vec(
            4,
            1,
            vec![
                Complex64::new(0.5, -2.0),
                Complex64::new(-1.0, 1.5),
                Complex64::new(2.0, 0.25),
                Complex64::new(3.0, -0.5),
            ],
        );
        let mut original_data = first.as_col_major_slice().to_vec();
        original_data.extend_from_slice(appended.as_col_major_slice());
        let original = Matrix::from_col_major_vec(4, 2, original_data);

        let mut qr = IncrementalQr::new(first).unwrap();
        qr.append(&appended).unwrap();
        let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
        let error = reconstructed
            .as_col_major_slice()
            .iter()
            .zip(original.as_col_major_slice())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0, f64::max);
        assert!(error < 1.0e-12, "complex reconstruction error is {error}");
    }
}
