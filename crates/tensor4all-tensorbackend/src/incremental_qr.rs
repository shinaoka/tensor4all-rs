//! Incremental QR factorization for column-major dense matrices.
//!
//! Provenance: the append layout and block-triangular update are cross-checked
//! against `chriscamano/RandomMPOMPS/code/tensornetwork/incrementalqr.py`,
//! `IncrementalQR::_setup`/`append` (lines 90--151), and
//! `incrementalqr.cpp::setup`/`add_cols` (lines 21--88), following Appendix C
//! of Camaño--Epperly--Tropp, [arXiv:2504.06475](https://arxiv.org/abs/2504.06475).
//! The safe-Rust Householder arithmetic, actual-R storage, rank-deficiency
//! policy, and `q_columns` optimization are derived or engineering choices;
//! the audit labels choices without an external basis `[AI-Supplied]`.

use anyhow::anyhow;
use num_complex::{Complex32, Complex64, ComplexFloat};

use crate::backend::{
    src_error_estimate, src_error_estimate_from_inverse_adjoint, src_inverse_adjoint,
    BackendLinalgError, BackendLinalgScalar, MatrixTriangularSolveScalar,
};
use crate::matrix::{mat_mul, Matrix, MatrixScalar};

type HouseholderFactorization<T> = (Matrix<T>, Vec<T>, Matrix<T>);

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

    /// Convert a non-negative real norm into this scalar type.
    fn from_real(value: f64) -> Self;
}

impl IncrementalQrScalar for f32 {
    fn conjugate(self) -> Self {
        self
    }

    fn from_real(value: f64) -> Self {
        value as f32
    }
}

impl IncrementalQrScalar for f64 {
    fn conjugate(self) -> Self {
        self
    }

    fn from_real(value: f64) -> Self {
        value
    }
}

impl IncrementalQrScalar for Complex32 {
    fn conjugate(self) -> Self {
        self.conj()
    }

    fn from_real(value: f64) -> Self {
        Self::new(value as f32, 0.0)
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
/// The state stores Householder reflectors for the thin `Q` factor and an
/// upper-trapezoidal `R` factor. Appending a block applies the stored
/// reflectors to the new columns, factors only the residual rows, and updates
/// the block-triangular `R`. This is the same state layout used by the
/// reference implementation's incremental QR path, expressed in safe Rust.
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
    reflectors: Matrix<T>,
    tau: Vec<T>,
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
    /// * `q` - Existing column-major `m × p` thin orthonormal factor.
    /// * `r` - Existing column-major `p × n` upper-trapezoidal factor, where
    ///   `n >= p`.
    ///
    /// # Returns
    /// An update state whose next append extends the represented factorization.
    ///
    /// # Errors
    /// Returns a backend error when the factors are empty, have incompatible
    /// dimensions, or are not thin.
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
        let (reflectors, tau, r_factor) = householder_factor(&q)?;
        let r = mat_mul(&r_factor, &r)
            .map_err(|error| anyhow!("incremental QR factor conversion failed: {error}"))?;
        let inverse_adjoint = try_inverse_adjoint(&r);
        Ok(Self {
            reflectors,
            tau,
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
    /// Returns a backend error when the matrix is empty, wide, malformed, or
    /// the backend QR factorization fails.
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

        let (reflectors, tau, r) = householder_factor(&input)?;
        let inverse_adjoint = try_inverse_adjoint(&r);
        Ok(Self {
            reflectors,
            tau,
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
    /// Returns a backend error when row counts differ, the append is empty, the
    /// resulting matrix would be wide, or a residual QR update fails.
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
        if new_columns.nrows() != self.reflectors.nrows() {
            return Err(anyhow!(
                "incremental QR append row count {} does not match {}",
                new_columns.nrows(),
                self.reflectors.nrows()
            )
            .into());
        }
        if new_columns.ncols() == 0 {
            return Err(anyhow!("incremental QR append requires at least one column").into());
        }
        let old_rank = self.reflectors.ncols();
        let maximum_new_rank = old_rank
            .checked_add(new_columns.ncols())
            .ok_or_else(|| anyhow!("incremental QR rank overflow"))?;
        if maximum_new_rank > self.reflectors.nrows() {
            return Err(anyhow!(
                "incremental QR append would produce a wide factorization: {} rows, {} columns",
                self.reflectors.nrows(),
                maximum_new_rank
            )
            .into());
        }

        let mut transformed = new_columns.clone();
        apply_q_adjoint(&self.reflectors, &self.tau, &mut transformed);
        let new_columns_norm = new_columns
            .as_col_major_slice()
            .iter()
            .map(|value| value.matrix_abs_sq())
            .sum::<f64>()
            .sqrt();
        let residual_tolerance = 32.0
            * f64::EPSILON
            * (self.reflectors.nrows().max(new_columns.ncols()) as f64)
            * new_columns_norm.max(1.0);
        let mut reflectors = Matrix::zeros(self.reflectors.nrows(), maximum_new_rank);
        for col in 0..old_rank {
            for row in 0..self.reflectors.nrows() {
                reflectors[[row, col]] = self.reflectors[[row, col]];
            }
        }
        let mut tau = self.tau.clone();
        let mut rank = old_rank;
        for col in 0..new_columns.ncols() {
            let residual_norm = (rank..transformed.nrows())
                .map(|row| transformed[[row, col]].matrix_abs_sq())
                .sum::<f64>()
                .sqrt();
            if !residual_norm.is_finite() {
                return Err(
                    anyhow!("incremental QR append produced a non-finite residual norm").into(),
                );
            }
            if residual_norm <= residual_tolerance {
                continue;
            }

            let (reflector_tau, vector) = householder_vector(&transformed, rank, col)?;
            apply_reflector(
                &mut transformed,
                rank,
                col,
                new_columns.ncols(),
                &vector,
                reflector_tau,
            );
            for (offset, value) in vector.iter().enumerate() {
                reflectors[[rank + offset, rank]] = *value;
            }
            tau.push(reflector_tau);
            rank += 1;
        }

        let old_column_count = self.r.ncols();
        let new_column_count = old_column_count
            .checked_add(new_columns.ncols())
            .ok_or_else(|| anyhow!("incremental QR column count overflow"))?;
        let mut r = Matrix::zeros(rank, new_column_count);
        for col in 0..old_column_count {
            for row in 0..old_rank {
                r[[row, col]] = self.r[[row, col]];
            }
        }
        for col in 0..new_columns.ncols() {
            for row in 0..rank {
                r[[row, old_column_count + col]] = transformed[[row, col]];
            }
        }

        let inverse_adjoint = if rank == new_column_count {
            if old_rank == old_column_count {
                if let Some(previous_inverse_adjoint) = self.inverse_adjoint.as_ref() {
                    Some(update_inverse_adjoint(
                        previous_inverse_adjoint,
                        &transformed,
                        old_rank,
                        old_column_count,
                        new_columns.ncols(),
                    )?)
                } else {
                    try_inverse_adjoint(&r)
                }
            } else {
                try_inverse_adjoint(&r)
            }
        } else {
            None
        };

        let mut reflector_data = Vec::with_capacity(self.reflectors.nrows() * rank);
        for col in 0..rank {
            for row in 0..self.reflectors.nrows() {
                reflector_data.push(reflectors[[row, col]]);
            }
        }
        self.reflectors = Matrix::from_col_major_vec(self.reflectors.nrows(), rank, reflector_data);
        self.tau = tau;
        self.r = r;
        self.inverse_adjoint = inverse_adjoint;
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
        form_q(&self.reflectors, &self.tau)
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
        self.reflectors.ncols()
    }

    /// Return a contiguous range of columns from the current thin `Q` factor.
    ///
    /// This is useful when a caller already materialized an earlier prefix and
    /// only needs the newly appended orthonormal columns. The returned matrix
    /// has the same row count as `Q` and `count` columns.
    ///
    /// # Arguments
    /// * `start` - Zero-based column in the current `Q` factor.
    /// * `count` - Number of columns to materialize.
    ///
    /// # Returns
    /// The requested column-major `m × count` block of `Q`.
    ///
    /// # Errors
    /// Returns a backend error if `start + count` exceeds the current thin
    /// factor width.
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
        if end > self.reflectors.ncols() {
            return Err(anyhow!(
                "incremental QR Q-column range {start}..{end} exceeds width {}",
                self.reflectors.ncols()
            )
            .into());
        }
        let mut q = Matrix::zeros(self.reflectors.nrows(), count);
        for column in 0..count {
            q[[start + column, column]] = T::one();
        }
        for reflector in (0..self.tau.len()).rev() {
            let vector = (reflector..self.reflectors.nrows())
                .map(|row| self.reflectors[[row, reflector]])
                .collect::<Vec<_>>();
            apply_reflector(&mut q, reflector, 0, count, &vector, self.tau[reflector]);
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

fn try_inverse_adjoint<T>(r: &Matrix<T>) -> Option<Matrix<T>>
where
    T: IncrementalQrScalar,
{
    src_inverse_adjoint(r).ok()
}

fn update_inverse_adjoint<T>(
    previous: &Matrix<T>,
    transformed: &Matrix<T>,
    old_rank: usize,
    old_column_count: usize,
    appended_column_count: usize,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    debug_assert_eq!(old_rank, old_column_count);
    debug_assert_eq!(previous.nrows(), old_rank);
    debug_assert_eq!(previous.ncols(), old_rank);

    let mut b_adjoint = Matrix::zeros(appended_column_count, old_rank);
    for column in 0..appended_column_count {
        for row in 0..old_rank {
            b_adjoint[[column, row]] = transformed[[row, column]].conjugate();
        }
    }

    let mut c = Matrix::zeros(appended_column_count, appended_column_count);
    for column in 0..appended_column_count {
        for row in 0..appended_column_count {
            c[[row, column]] = transformed[[old_rank + row, column]];
        }
    }
    let c_inverse_adjoint = src_inverse_adjoint(&c)?;
    let coupling = mat_mul(&b_adjoint, previous)
        .map_err(|error| anyhow!("incremental QR inverse-adjoint coupling failed: {error}"))?;
    let lower = mat_mul(&c_inverse_adjoint, &coupling)
        .map_err(|error| anyhow!("incremental QR inverse-adjoint update failed: {error}"))?;

    let new_rank = old_rank + appended_column_count;
    let mut updated = Matrix::zeros(new_rank, new_rank);
    for column in 0..old_rank {
        for row in 0..old_rank {
            updated[[row, column]] = previous[[row, column]];
        }
    }
    for column in 0..appended_column_count {
        for row in 0..appended_column_count {
            updated[[old_rank + row, old_rank + column]] = c_inverse_adjoint[[row, column]];
        }
        for row in 0..old_rank {
            updated[[old_rank + column, row]] = -lower[[column, row]];
        }
    }
    Ok(updated)
}

fn householder_factor<T>(
    input: &Matrix<T>,
) -> std::result::Result<HouseholderFactorization<T>, BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if input.nrows() < input.ncols() {
        return Err(anyhow!(
            "incremental QR requires a tall-or-square matrix, got {}x{}",
            input.nrows(),
            input.ncols()
        )
        .into());
    }
    let mut data = input.clone();
    let mut tau = Vec::with_capacity(input.ncols());
    let mut diagonal = vec![T::zero(); input.ncols()];
    for column in 0..input.ncols() {
        let (reflector_tau, vector) = householder_vector(&data, column, column)?;
        apply_reflector(
            &mut data,
            column,
            column,
            input.ncols(),
            &vector,
            reflector_tau,
        );
        diagonal[column] = data[[column, column]];
        data[[column, column]] = T::one();
        for row in column + 1..input.nrows() {
            data[[row, column]] = vector[row - column];
        }
        tau.push(reflector_tau);
    }

    let mut r = Matrix::zeros(input.ncols(), input.ncols());
    for column in 0..input.ncols() {
        for row in 0..=column {
            r[[row, column]] = if row == column {
                diagonal[column]
            } else {
                data[[row, column]]
            };
        }
    }
    Ok((data, tau, r))
}

fn householder_vector<T>(
    data: &Matrix<T>,
    start: usize,
    column: usize,
) -> std::result::Result<(T, Vec<T>), BackendLinalgError>
where
    T: IncrementalQrScalar,
{
    if start >= data.nrows() || column >= data.ncols() {
        return Err(anyhow!(
            "incremental QR reflector ({start}, {column}) is outside {}x{} matrix",
            data.nrows(),
            data.ncols()
        )
        .into());
    }
    let tail_norm_sq = (start + 1..data.nrows())
        .map(|row| data[[row, column]].matrix_abs_sq())
        .sum::<f64>();
    let alpha = data[[start, column]];
    let alpha_norm_sq = alpha.matrix_abs_sq();
    let norm_sq = alpha_norm_sq + tail_norm_sq;
    if !norm_sq.is_finite() {
        return Err(anyhow!("incremental QR reflector norm is not finite").into());
    }
    let norm = norm_sq.sqrt();
    let mut vector = vec![T::zero(); data.nrows() - start];
    vector[0] = T::one();
    if norm == 0.0 {
        return Ok((T::zero(), vector));
    }

    let phase = if alpha_norm_sq == 0.0 {
        T::one()
    } else {
        alpha / T::from_real(alpha_norm_sq.sqrt())
    };
    let beta = -(phase * T::from_real(norm));
    let denominator = alpha - beta;
    if denominator.matrix_abs_sq() == 0.0 || !denominator.matrix_abs_sq().is_finite() {
        return Err(anyhow!("incremental QR reflector denominator is invalid").into());
    }
    for row in start + 1..data.nrows() {
        vector[row - start] = data[[row, column]] / denominator;
    }
    let tau = (beta - alpha) / beta;
    Ok((tau, vector))
}

fn apply_reflector<T>(
    data: &mut Matrix<T>,
    start: usize,
    first_column: usize,
    column_count: usize,
    vector: &[T],
    tau: T,
) where
    T: IncrementalQrScalar,
{
    if tau == T::zero() {
        return;
    }
    for column in first_column..column_count {
        let dot = (0..vector.len()).fold(T::zero(), |sum, offset| {
            sum + vector[offset].conjugate() * data[[start + offset, column]]
        });
        let scale = tau * dot;
        for (offset, value) in vector.iter().enumerate() {
            let row = start + offset;
            data[[row, column]] = data[[row, column]] - *value * scale;
        }
    }
}

fn apply_q_adjoint<T>(reflectors: &Matrix<T>, tau: &[T], data: &mut Matrix<T>)
where
    T: IncrementalQrScalar,
{
    for reflector in 0..tau.len() {
        let vector = (reflector..reflectors.nrows())
            .map(|row| reflectors[[row, reflector]])
            .collect::<Vec<_>>();
        apply_reflector(data, reflector, 0, data.ncols(), &vector, tau[reflector]);
    }
}

fn form_q<T>(reflectors: &Matrix<T>, tau: &[T]) -> Matrix<T>
where
    T: IncrementalQrScalar,
{
    let mut q = Matrix::zeros(reflectors.nrows(), reflectors.ncols());
    for column in 0..reflectors.ncols() {
        q[[column, column]] = T::one();
    }
    for reflector in (0..tau.len()).rev() {
        let vector = (reflector..reflectors.nrows())
            .map(|row| reflectors[[row, reflector]])
            .collect::<Vec<_>>();
        let column_count = q.ncols();
        apply_reflector(&mut q, reflector, 0, column_count, &vector, tau[reflector]);
    }
    q
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
        let appended_again = Matrix::from_col_major_vec(5, 1, vec![2.5, 1.0, -0.5, 4.5, -3.0]);
        let mut original_data = first.as_col_major_slice().to_vec();
        original_data.extend_from_slice(appended.as_col_major_slice());
        original_data.extend_from_slice(appended_again.as_col_major_slice());
        let original = Matrix::from_col_major_vec(5, 4, original_data);

        let mut qr = IncrementalQr::new(first.clone()).unwrap();
        let initial_original =
            Matrix::from_col_major_vec(5, 2, first.as_col_major_slice().to_vec());
        assert!(reconstruction_error(&qr, &initial_original) < 1.0e-12);
        assert_eq!(qr.q().ncols(), 2);
        assert_eq!(qr.r().nrows(), 2);
        qr.append(&appended).unwrap();
        let after_one = Matrix::from_col_major_vec(5, 3, {
            let mut values = first.as_col_major_slice().to_vec();
            values.extend_from_slice(appended.as_col_major_slice());
            values
        });
        assert!(reconstruction_error(&qr, &after_one) < 1.0e-12);
        qr.append(&appended_again).unwrap();

        assert_eq!(qr.q().nrows(), 5);
        assert_eq!(qr.q().ncols(), 4);
        assert_eq!(qr.r().nrows(), 4);
        assert_eq!(qr.r().ncols(), 4);
        let error = reconstruction_error(&qr, &original);
        assert!(error < 1.0e-12, "reconstruction error={error}");

        let q = qr.q();
        for left in 0..q.ncols() {
            for right in 0..q.ncols() {
                let inner = (0..q.nrows())
                    .map(|row| q[[row, left]] * q[[row, right]])
                    .sum::<f64>();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (inner - expected).abs() < 1.0e-12,
                    "Q is not orthonormal at ({left}, {right}): {inner}"
                );
            }
        }

        let full = IncrementalQr::new(original).unwrap();
        let incremental_estimate = qr.error_estimate().unwrap();
        let full_estimate = full.error_estimate().unwrap();
        assert!(
            (incremental_estimate.error - full_estimate.error).abs() < 1.0e-10,
            "incremental and full QR error estimates differ: {} vs {}",
            incremental_estimate.error,
            full_estimate.error
        );
        assert!(
            (incremental_estimate.norm - full_estimate.norm).abs() < 1.0e-10,
            "incremental and full QR norm estimates differ: {} vs {}",
            incremental_estimate.norm,
            full_estimate.norm
        );
    }

    #[test]
    fn incremental_qr_updates_the_inverse_adjoint_after_appending_columns() {
        let first = Matrix::from_col_major_vec(
            5,
            2,
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.5, 3.0, 4.0],
        );
        let appended = Matrix::from_col_major_vec(5, 1, vec![0.5, -1.0, 2.0, 1.5, 3.5]);
        let mut qr = IncrementalQr::new(first).unwrap();
        qr.append(&appended).unwrap();

        let r = qr.r();
        let g = qr
            .inverse_adjoint
            .as_ref()
            .expect("full-rank incremental QR must retain R^{-T}");
        let mut r_transpose = Matrix::zeros(r.ncols(), r.nrows());
        for col in 0..r.ncols() {
            for row in 0..r.nrows() {
                r_transpose[[row, col]] = r[[col, row]];
            }
        }
        let product = mat_mul(&r_transpose, g).unwrap();
        for row in 0..product.nrows() {
            for col in 0..product.ncols() {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (product[[row, col]] - expected).abs() < 1.0e-10,
                    "R^T G is not identity at ({row}, {col}): {}",
                    product[[row, col]]
                );
            }
        }
    }

    #[test]
    fn incremental_qr_preserves_rank_for_dependent_appended_columns() {
        let first = Matrix::from_col_major_vec(
            5,
            2,
            vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        );
        let appended = Matrix::from_col_major_vec(
            5,
            2,
            vec![1.0_f64, 2.0, 0.0, 0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 0.0],
        );
        let mut original_data = first.as_col_major_slice().to_vec();
        original_data.extend_from_slice(appended.as_col_major_slice());
        let original = Matrix::from_col_major_vec(5, 4, original_data);

        let mut qr = IncrementalQr::new(first).unwrap();
        qr.append(&appended).unwrap();

        assert_eq!(qr.q().ncols(), 2);
        assert_eq!(qr.r().nrows(), 2);
        assert_eq!(qr.r().ncols(), 4);
        let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
        assert!(reconstructed
            .as_col_major_slice()
            .iter()
            .zip(original.as_col_major_slice())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    }

    #[test]
    fn incremental_qr_from_factors_reconstructs_the_supplied_product() {
        let q = Matrix::from_col_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let r = Matrix::from_col_major_vec(2, 3, vec![2.0_f64, 0.0, 1.0, 3.0, -1.0, 0.5]);
        let expected = mat_mul(&q, &r).unwrap();
        let state = IncrementalQr::from_factors(q, r).unwrap();
        let actual = mat_mul(&state.q(), &state.r()).unwrap();
        assert!(actual
            .as_col_major_slice()
            .iter()
            .zip(expected.as_col_major_slice())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
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

        let r = qr.r();
        let g = qr
            .inverse_adjoint
            .as_ref()
            .expect("full-rank complex QR must retain R^{-dagger}");
        let mut r_adjoint = Matrix::zeros(r.ncols(), r.nrows());
        for col in 0..r.ncols() {
            for row in 0..r.nrows() {
                r_adjoint[[row, col]] = r[[col, row]].conj();
            }
        }
        let product = mat_mul(&r_adjoint, g).unwrap();
        for row in 0..product.nrows() {
            for col in 0..product.ncols() {
                let expected = if row == col {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                };
                assert!(
                    (product[[row, col]] - expected).norm() < 1.0e-10,
                    "R^dagger G is not identity at ({row}, {col}): {}",
                    product[[row, col]]
                );
            }
        }
    }
}
