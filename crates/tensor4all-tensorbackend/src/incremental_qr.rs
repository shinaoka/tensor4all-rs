//! Backend-native incremental QR factorization for column-major dense matrices.
//!
//! The block update is independently derived from Appendix C.3 of
//! Camaño--Epperly--Tropp,
//! [arXiv:2504.06475](https://arxiv.org/abs/2504.06475). For an existing
//! factorization `Y = Q R` and appended columns `Y'`, it computes two
//! block Gram--Schmidt projection passes followed by a backend QR of the
//! residual. The second pass limits loss of orthogonality without introducing
//! scalar reflector kernels.

use anyhow::anyhow;
use num_complex::{Complex64, ComplexFloat};

use crate::backend::{
    qr_backend, src_error_estimate, src_error_estimate_from_inverse_adjoint, src_inverse_adjoint,
    BackendLinalgError, BackendLinalgScalar, MatrixTriangularSolveScalar,
};
use crate::matrix::{mat_mul, Matrix, MatrixScalar};

/// Scalar operations required by [`IncrementalQr`].
///
/// The implementation is provided for the two scalar types supported by the
/// backend, `f64` and `Complex64`. The conjugation hook keeps the update
/// correct for complex matrices while using the same column-major algorithm
/// for real matrices.
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

    /// Convert a non-negative real norm into this scalar type.
    fn from_real(value: f64) -> Self;
}

impl IncrementalQrScalar for f64 {
    fn conjugate(self) -> Self {
        self
    }

    fn from_real(value: f64) -> Self {
        value
    }
}

impl IncrementalQrScalar for Complex64 {
    fn conjugate(self) -> Self {
        self.conj()
    }

    fn from_real(value: f64) -> Self {
        Self::new(value, 0.0)
    }
}

/// Thin QR state that can append columns without refactorizing the old block.
///
/// The state stores an explicit thin `Q` factor and an upper-trapezoidal
/// `R` factor. Appending a full-rank block uses two backend matrix-product
/// projection passes, factorizes only the residual block through the configured
/// QR backend, and updates the block-triangular `R`.
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
    /// `R^{-†}` for the current square full-rank QR block. `None` denotes a
    /// rank-deficient or rectangular state for which the Appendix C estimate
    /// is not defined.
    inverse_adjoint: Option<Matrix<T>>,
}

impl<T> IncrementalQr<T>
where
    T: IncrementalQrScalar,
{
    /// Resume an incremental QR update from compatible thin factors.
    ///
    /// # Arguments
    /// * `q` - Existing column-major `m × p` thin factor.
    /// * `r` - Existing column-major `p × n` upper-trapezoidal factor, where
    ///   `n >= p`.
    ///
    /// # Returns
    /// An update state whose next append extends the represented factorization.
    ///
    /// # Errors
    /// Returns a backend error when the factors are empty, have incompatible
    /// dimensions, are not thin, or backend QR/multiplication fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let state = IncrementalQr::from_factors(
    ///     Matrix::from_col_major_vec(2, 1, vec![1.0_f64, 0.0]),
    ///     Matrix::from_col_major_vec(1, 1, vec![2.0]),
    /// )
    /// .unwrap();
    /// assert_eq!(state.q().ncols(), 1);
    /// assert_eq!(state.r().nrows(), 1);
    /// ```
    pub fn from_factors(
        q: Matrix<T>,
        r: Matrix<T>,
    ) -> std::result::Result<Self, BackendLinalgError> {
        if q.nrows() == 0 || q.ncols() == 0 {
            return Err(anyhow!("incremental QR factors must be non-empty").into());
        }
        if q.nrows() < q.ncols() {
            return Err(anyhow!(
                "incremental QR factors must be thin, got Q {}x{}",
                q.nrows(),
                q.ncols()
            )
            .into());
        }
        if r.nrows() != q.ncols() || r.ncols() < q.ncols() {
            return Err(anyhow!(
                "incremental QR factor dimensions are incompatible: Q {}x{}, R {}x{}",
                q.nrows(),
                q.ncols(),
                r.nrows(),
                r.ncols()
            )
            .into());
        }

        let (q, q_r) = factorize_backend(&q)?;
        let r = mat_mul(&q_r, &r)
            .map_err(|error| anyhow!("incremental QR factor conversion failed: {error}"))?;
        let inverse_adjoint = try_inverse_adjoint(&r);
        Ok(Self {
            q,
            r,
            inverse_adjoint,
        })
    }

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
    /// Returns a backend error when the input dimensions are invalid because
    /// the matrix is empty or wide, or when backend QR conversion or
    /// factorization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     2,
    ///     1,
    ///     vec![1.0_f64, 2.0],
    /// ))
    /// .unwrap();
    /// assert_eq!(qr.q().nrows(), 2);
    /// assert_eq!(qr.r().ncols(), 1);
    /// ```
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

        let (q, r) = factorize_backend(&input)?;
        let inverse_adjoint = try_inverse_adjoint(&r);
        Ok(Self {
            q,
            r,
            inverse_adjoint,
        })
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
    /// Returns a backend error when the input dimensions are invalid because
    /// row counts differ, the append is empty, or the resulting matrix would
    /// be wide; when a rank or column count overflows; or when backend matrix
    /// multiplication, conversion, or QR factorization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let mut qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     3,
    ///     1,
    ///     vec![1.0_f64, 2.0, 3.0],
    /// ))
    /// .unwrap();
    /// qr.append(&Matrix::from_col_major_vec(
    ///     3,
    ///     1,
    ///     vec![3.0_f64, 2.0, 1.0],
    /// ))
    /// .unwrap();
    /// assert_eq!(qr.r().ncols(), 2);
    /// ```
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
        let maximum_new_rank = self
            .q
            .ncols()
            .checked_add(new_columns.ncols())
            .ok_or_else(|| anyhow!("incremental QR rank overflow"))?;
        if maximum_new_rank > self.q.nrows() {
            return Err(anyhow!(
                "incremental QR append would produce a wide factorization: {} rows, {} columns",
                self.q.nrows(),
                maximum_new_rank
            )
            .into());
        }

        let new_columns_norm = frobenius_norm(new_columns)?;
        let residual_tolerance = 32.0
            * f64::EPSILON
            * (self.q.nrows().max(new_columns.ncols()) as f64)
            * new_columns_norm.max(1.0);

        let (projection, residual) = project_twice(&self.q, new_columns)?;
        let (appended_q, appended_r) = factorize_backend(&residual)?;
        if diagonal_is_full_rank(&appended_r, residual_tolerance) {
            return self.commit_full_rank_block(projection, appended_q, appended_r);
        }

        for column in 0..new_columns.ncols() {
            let column = matrix_column(new_columns, column)?;
            let (projection, residual) = project_twice(&self.q, &column)?;
            let residual_norm = frobenius_norm(&residual)?;
            if residual_norm <= residual_tolerance {
                self.commit_dependent_column(projection)?;
                continue;
            }
            let (appended_q, appended_r) = factorize_backend(&residual)?;
            self.commit_full_rank_block(projection, appended_q, appended_r)?;
        }
        Ok(())
    }

    fn commit_full_rank_block(
        &mut self,
        projection: Matrix<T>,
        appended_q: Matrix<T>,
        appended_r: Matrix<T>,
    ) -> std::result::Result<(), BackendLinalgError> {
        let old_rank = self.q.ncols();
        let old_column_count = self.r.ncols();
        let appended_rank = appended_q.ncols();
        if projection.nrows() != old_rank
            || projection.ncols() != appended_rank
            || appended_r.nrows() != appended_rank
            || appended_r.ncols() != appended_rank
        {
            return Err(anyhow!(
                "incremental QR backend update returned incompatible blocks: projection {}x{}, Q' {}x{}, R'' {}x{}",
                projection.nrows(),
                projection.ncols(),
                appended_q.nrows(),
                appended_q.ncols(),
                appended_r.nrows(),
                appended_r.ncols()
            )
            .into());
        }

        let r = assemble_r(&self.r, &projection, &appended_r)?;
        let new_rank = old_rank
            .checked_add(appended_rank)
            .ok_or_else(|| anyhow!("incremental QR rank overflow"))?;
        let new_column_count = r.ncols();
        let inverse_adjoint = if new_rank == new_column_count {
            if old_rank == old_column_count {
                if let Some(previous) = self.inverse_adjoint.as_ref() {
                    Some(update_inverse_adjoint(previous, &projection, &appended_r)?)
                } else {
                    try_inverse_adjoint(&r)
                }
            } else {
                try_inverse_adjoint(&r)
            }
        } else {
            None
        };

        self.q
            .append_columns(&appended_q)
            .map_err(|error| anyhow!("incremental QR Q append failed: {error}"))?;
        self.r = r;
        self.inverse_adjoint = inverse_adjoint;
        Ok(())
    }

    fn commit_dependent_column(
        &mut self,
        projection: Matrix<T>,
    ) -> std::result::Result<(), BackendLinalgError> {
        if projection.nrows() != self.q.ncols() || projection.ncols() != 1 {
            return Err(anyhow!(
                "incremental QR dependent-column projection has shape {}x{}, expected {}x1",
                projection.nrows(),
                projection.ncols(),
                self.q.ncols()
            )
            .into());
        }
        let new_column_count = self
            .r
            .ncols()
            .checked_add(1)
            .ok_or_else(|| anyhow!("incremental QR column count overflow"))?;
        let mut r = Matrix::try_zeros(self.r.nrows(), new_column_count)
            .map_err(|error| anyhow!("incremental QR R allocation failed: {error}"))?;
        for column in 0..self.r.ncols() {
            for row in 0..self.r.nrows() {
                r[[row, column]] = self.r[[row, column]];
            }
        }
        for row in 0..self.r.nrows() {
            r[[row, self.r.ncols()]] = projection[[row, 0]];
        }
        self.r = r;
        self.inverse_adjoint = None;
        Ok(())
    }

    /// Return a copy of the current thin `Q` factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     2,
    ///     1,
    ///     vec![1.0_f64, 0.0],
    /// ))
    /// .unwrap();
    /// assert_eq!(qr.q().ncols(), 1);
    /// ```
    pub fn q(&self) -> Matrix<T> {
        self.q.clone()
    }

    /// Return the current thin factor width.
    ///
    /// This is the number of columns in both `Q` and the row count of `R`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     2,
    ///     1,
    ///     vec![1.0_f64, 0.0],
    /// ))
    /// .unwrap();
    /// assert_eq!(qr.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        self.q.ncols()
    }

    /// Return a contiguous range of columns from the current thin `Q` factor.
    ///
    /// # Arguments
    /// * `start` - Zero-based column in the current `Q` factor.
    /// * `count` - Number of columns to materialize.
    ///
    /// # Returns
    /// The requested column-major `m × count` block of `Q`.
    ///
    /// # Errors
    /// Returns a backend error when the requested range overflows or is out of
    /// bounds for the current thin-factor width, or when the output shape is
    /// invalid because its element count overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     3,
    ///     2,
    ///     vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0],
    /// ))
    /// .unwrap();
    /// let second = qr.q_columns(1, 1).unwrap();
    /// assert_eq!(second.nrows(), 3);
    /// assert_eq!(second.ncols(), 1);
    /// assert!((second[[1, 0]].abs() - 1.0).abs() < 1.0e-12);
    /// ```
    pub fn q_columns(
        &self,
        start: usize,
        count: usize,
    ) -> std::result::Result<Matrix<T>, BackendLinalgError> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow!("incremental QR Q-column range overflows usize"))?;
        if end > self.q.ncols() {
            return Err(anyhow!(
                "incremental QR Q-column range {start}..{end} exceeds width {}",
                self.q.ncols()
            )
            .into());
        }
        let mut q = Matrix::try_zeros(self.q.nrows(), count)
            .map_err(|error| anyhow!("incremental QR Q-column allocation failed: {error}"))?;
        for column in 0..count {
            for row in 0..self.q.nrows() {
                q[[row, column]] = self.q[[row, start + column]];
            }
        }
        Ok(q)
    }

    /// Return a copy of the current upper-trapezoidal `R` factor.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     2,
    ///     1,
    ///     vec![1.0_f64, 0.0],
    /// ))
    /// .unwrap();
    /// assert_eq!(qr.r().nrows(), 1);
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::{IncrementalQr, Matrix};
    ///
    /// let qr = IncrementalQr::new(Matrix::from_col_major_vec(
    ///     2,
    ///     1,
    ///     vec![1.0_f64, 0.0],
    /// ))
    /// .unwrap();
    /// let estimate = qr.error_estimate().unwrap();
    /// assert!(estimate.error.is_finite());
    /// assert!(estimate.norm.is_finite());
    /// ```
    pub fn error_estimate(
        &self,
    ) -> std::result::Result<crate::SrcErrorEstimate, BackendLinalgError> {
        if let Some(inverse_adjoint) = self.inverse_adjoint.as_ref() {
            src_error_estimate_from_inverse_adjoint(&self.r, inverse_adjoint)
        } else {
            src_error_estimate(&self.r)
        }
    }
}

fn factorize_backend<T>(
    input: &Matrix<T>,
) -> std::result::Result<(Matrix<T>, Matrix<T>), BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let (q, r) = qr_backend(&input.to_typed_tensor())?;
    let q = Matrix::try_from_typed_tensor(q)
        .map_err(|error| anyhow!("incremental QR backend Q conversion failed: {error}"))?;
    let r = Matrix::try_from_typed_tensor(r)
        .map_err(|error| anyhow!("incremental QR backend R conversion failed: {error}"))?;
    Ok((q, r))
}

fn project_twice<T>(
    q: &Matrix<T>,
    columns: &Matrix<T>,
) -> std::result::Result<(Matrix<T>, Matrix<T>), BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let q_adjoint = matrix_adjoint(q)?;
    let first_projection = mat_mul(&q_adjoint, columns)
        .map_err(|error| anyhow!("incremental QR first projection failed: {error}"))?;
    let first_reconstruction = mat_mul(q, &first_projection)
        .map_err(|error| anyhow!("incremental QR first reconstruction failed: {error}"))?;
    let first_residual = matrix_subtract(columns, &first_reconstruction)?;

    let correction = mat_mul(&q_adjoint, &first_residual)
        .map_err(|error| anyhow!("incremental QR reorthogonalization failed: {error}"))?;
    let correction_reconstruction = mat_mul(q, &correction).map_err(|error| {
        anyhow!("incremental QR reorthogonalization reconstruction failed: {error}")
    })?;
    let residual = matrix_subtract(&first_residual, &correction_reconstruction)?;
    let projection = matrix_add(&first_projection, &correction)?;
    Ok((projection, residual))
}

fn matrix_adjoint<T>(matrix: &Matrix<T>) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let mut adjoint = Matrix::try_zeros(matrix.ncols(), matrix.nrows())
        .map_err(|error| anyhow!("incremental QR adjoint allocation failed: {error}"))?;
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            adjoint[[column, row]] = matrix[[row, column]].conjugate();
        }
    }
    Ok(adjoint)
}

fn matrix_add<T>(
    left: &Matrix<T>,
    right: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    ensure_same_shape("addition", left, right)?;
    let values = left
        .as_col_major_slice()
        .iter()
        .zip(right.as_col_major_slice())
        .map(|(left, right)| *left + *right)
        .collect();
    Matrix::try_from_col_major_vec(left.nrows(), left.ncols(), values)
        .map_err(|error| anyhow!("incremental QR addition result is invalid: {error}").into())
}

fn matrix_subtract<T>(
    left: &Matrix<T>,
    right: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    ensure_same_shape("subtraction", left, right)?;
    let values = left
        .as_col_major_slice()
        .iter()
        .zip(right.as_col_major_slice())
        .map(|(left, right)| *left - *right)
        .collect();
    Matrix::try_from_col_major_vec(left.nrows(), left.ncols(), values)
        .map_err(|error| anyhow!("incremental QR subtraction result is invalid: {error}").into())
}

fn ensure_same_shape<T>(
    operation: &str,
    left: &Matrix<T>,
    right: &Matrix<T>,
) -> std::result::Result<(), BackendLinalgError> {
    if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
        return Err(anyhow!(
            "incremental QR {operation} shape mismatch: {}x{} and {}x{}",
            left.nrows(),
            left.ncols(),
            right.nrows(),
            right.ncols()
        )
        .into());
    }
    Ok(())
}

fn matrix_column<T>(
    matrix: &Matrix<T>,
    column: usize,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if column >= matrix.ncols() {
        return Err(anyhow!(
            "incremental QR column {column} exceeds width {}",
            matrix.ncols()
        )
        .into());
    }
    let start = column
        .checked_mul(matrix.nrows())
        .ok_or_else(|| anyhow!("incremental QR column offset overflow"))?;
    let end = start
        .checked_add(matrix.nrows())
        .ok_or_else(|| anyhow!("incremental QR column range overflow"))?;
    Matrix::try_from_col_major_vec(
        matrix.nrows(),
        1,
        matrix.as_col_major_slice()[start..end].to_vec(),
    )
    .map_err(|error| anyhow!("incremental QR column extraction failed: {error}").into())
}

fn frobenius_norm<T>(matrix: &Matrix<T>) -> std::result::Result<f64, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let norm = matrix
        .as_col_major_slice()
        .iter()
        .map(|value| value.matrix_abs_sq())
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() {
        return Err(anyhow!("incremental QR produced a non-finite residual norm").into());
    }
    Ok(norm)
}

fn diagonal_is_full_rank<T>(r: &Matrix<T>, tolerance: f64) -> bool
where
    T: IncrementalQrScalar,
{
    r.nrows() == r.ncols()
        && (0..r.ncols()).all(|diagonal| {
            let magnitude = r[[diagonal, diagonal]].matrix_abs_sq().sqrt();
            magnitude.is_finite() && magnitude > tolerance
        })
}

fn concatenate_columns<T>(
    left: &Matrix<T>,
    right: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if left.nrows() != right.nrows() {
        return Err(anyhow!(
            "incremental QR Q block row mismatch: {} and {}",
            left.nrows(),
            right.nrows()
        )
        .into());
    }
    let ncols = left
        .ncols()
        .checked_add(right.ncols())
        .ok_or_else(|| anyhow!("incremental QR Q width overflow"))?;
    let mut values = Vec::with_capacity(
        left.as_col_major_slice()
            .len()
            .checked_add(right.as_col_major_slice().len())
            .ok_or_else(|| anyhow!("incremental QR Q element count overflow"))?,
    );
    values.extend_from_slice(left.as_col_major_slice());
    values.extend_from_slice(right.as_col_major_slice());
    Matrix::try_from_col_major_vec(left.nrows(), ncols, values)
        .map_err(|error| anyhow!("incremental QR Q assembly failed: {error}").into())
}

fn assemble_r<T>(
    old: &Matrix<T>,
    projection: &Matrix<T>,
    residual_r: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if projection.nrows() != old.nrows()
        || residual_r.nrows() != residual_r.ncols()
        || projection.ncols() != residual_r.ncols()
    {
        return Err(anyhow!(
            "incremental QR R blocks are incompatible: R {}x{}, projection {}x{}, residual R {}x{}",
            old.nrows(),
            old.ncols(),
            projection.nrows(),
            projection.ncols(),
            residual_r.nrows(),
            residual_r.ncols()
        )
        .into());
    }
    let new_rows = old
        .nrows()
        .checked_add(residual_r.nrows())
        .ok_or_else(|| anyhow!("incremental QR R row count overflow"))?;
    let new_columns = old
        .ncols()
        .checked_add(residual_r.ncols())
        .ok_or_else(|| anyhow!("incremental QR R column count overflow"))?;
    let mut result = Matrix::try_zeros(new_rows, new_columns)
        .map_err(|error| anyhow!("incremental QR R allocation failed: {error}"))?;

    for column in 0..old.ncols() {
        for row in 0..old.nrows() {
            result[[row, column]] = old[[row, column]];
        }
    }
    for column in 0..projection.ncols() {
        let target_column = old.ncols() + column;
        for row in 0..projection.nrows() {
            result[[row, target_column]] = projection[[row, column]];
        }
        for row in 0..residual_r.nrows() {
            result[[old.nrows() + row, target_column]] = residual_r[[row, column]];
        }
    }
    Ok(result)
}

fn try_inverse_adjoint<T>(r: &Matrix<T>) -> Option<Matrix<T>>
where
    T: IncrementalQrScalar,
{
    src_inverse_adjoint(r).ok()
}

fn update_inverse_adjoint<T>(
    previous: &Matrix<T>,
    projection: &Matrix<T>,
    residual_r: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    let old_rank = previous.nrows();
    let appended_rank = residual_r.nrows();
    if previous.ncols() != old_rank
        || projection.nrows() != old_rank
        || projection.ncols() != appended_rank
        || residual_r.ncols() != appended_rank
    {
        return Err(anyhow!(
            "incremental QR inverse-adjoint blocks are incompatible: G {}x{}, projection {}x{}, R'' {}x{}",
            previous.nrows(),
            previous.ncols(),
            projection.nrows(),
            projection.ncols(),
            residual_r.nrows(),
            residual_r.ncols()
        )
        .into());
    }

    let projection_adjoint = matrix_adjoint(projection)?;
    let residual_inverse_adjoint = src_inverse_adjoint(residual_r)?;
    let coupling = mat_mul(&projection_adjoint, previous)
        .map_err(|error| anyhow!("incremental QR inverse-adjoint coupling failed: {error}"))?;
    let lower = mat_mul(&residual_inverse_adjoint, &coupling)
        .map_err(|error| anyhow!("incremental QR inverse-adjoint update failed: {error}"))?;

    let new_rank = old_rank
        .checked_add(appended_rank)
        .ok_or_else(|| anyhow!("incremental QR inverse-adjoint rank overflow"))?;
    let mut updated = Matrix::try_zeros(new_rank, new_rank)
        .map_err(|error| anyhow!("incremental QR inverse-adjoint allocation failed: {error}"))?;
    for column in 0..old_rank {
        for row in 0..old_rank {
            updated[[row, column]] = previous[[row, column]];
        }
    }
    for column in 0..appended_rank {
        for row in 0..appended_rank {
            updated[[old_rank + row, old_rank + column]] = residual_inverse_adjoint[[row, column]];
        }
        for row in 0..old_rank {
            updated[[old_rank + column, row]] = -lower[[column, row]];
        }
    }
    Ok(updated)
}

#[cfg(test)]
mod tests;
