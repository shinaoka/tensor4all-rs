//! Backend dispatch helpers for linear algebra operations.
//!
//! This module keeps tensor4all's typed factorization entry points thin while
//! routing the actual work through the shared tenferro CPU backend.

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64, ComplexFloat};
use tenferro::{DType, Tensor, TensorScalar, TensorSessionOpsExt, TypedTensor};
use tenferro_linalg::TensorLinalgExt;

use crate::context::with_default_session;
use crate::matrix::Matrix;

/// Result of SVD decomposition `A = U * diag(S) * Vt`.
/// The singular values are stored in a real-valued typed tensor, even when the
/// input matrix is complex.
/// # Examples
/// ```
/// use tensor4all_tensorbackend::svd_backend;
/// use tenferro::TypedTensor;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0])?;
/// let result = svd_backend(&a)?;
/// assert_eq!(result.u().shape(), &[2, 2]);
/// assert_eq!(result.s().shape(), &[2]);
/// assert_eq!(result.vt().shape(), &[2, 2]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct SvdResult<T: TensorScalar> {
    u: TypedTensor<T>,
    s: TypedTensor<T::Real>,
    vt: TypedTensor<T>,
}

/// Result of complete-pivoting LU decomposition `P A Q^T = L U`.
/// The parity output from tenferro is intentionally omitted because current
/// tensor4all callers only need the permutation matrices and the upper
/// triangular factor for pivot selection.
#[derive(Debug)]
pub struct FullPivLuResult<T: TensorScalar> {
    p: TypedTensor<T>,
    l: TypedTensor<T>,
    u: TypedTensor<T>,
    q: TypedTensor<T>,
}

fn clone_linalg_tensor<T: TensorScalar>(tensor: &TypedTensor<T>) -> TypedTensor<T> {
    crate::require_invariant(tensor.duplicate(), "linalg result duplication failed")
}

impl<T: TensorScalar> SvdResult<T> {
    /// Borrow the left singular vectors.
    pub fn u(&self) -> &TypedTensor<T> {
        &self.u
    }

    /// Borrow the singular values.
    pub fn s(&self) -> &TypedTensor<T::Real> {
        &self.s
    }

    /// Borrow the transposed right singular vectors.
    pub fn vt(&self) -> &TypedTensor<T> {
        &self.vt
    }

    /// Consume the decomposition into `(U, S, Vt)`.
    pub fn into_parts(self) -> (TypedTensor<T>, TypedTensor<T::Real>, TypedTensor<T>) {
        (self.u, self.s, self.vt)
    }
}

impl<T: TensorScalar> FullPivLuResult<T> {
    /// Borrow the left permutation matrix.
    pub fn p(&self) -> &TypedTensor<T> {
        &self.p
    }

    /// Borrow the lower triangular factor.
    pub fn l(&self) -> &TypedTensor<T> {
        &self.l
    }

    /// Borrow the upper triangular factor.
    pub fn u(&self) -> &TypedTensor<T> {
        &self.u
    }

    /// Borrow the right permutation matrix.
    pub fn q(&self) -> &TypedTensor<T> {
        &self.q
    }

    /// Consume the decomposition into `(P, L, U, Q)`.
    pub fn into_parts(
        self,
    ) -> (
        TypedTensor<T>,
        TypedTensor<T>,
        TypedTensor<T>,
        TypedTensor<T>,
    ) {
        (self.p, self.l, self.u, self.q)
    }
}

impl<T: TensorScalar> Clone for SvdResult<T> {
    fn clone(&self) -> Self {
        Self {
            u: clone_linalg_tensor(&self.u),
            s: clone_linalg_tensor(&self.s),
            vt: clone_linalg_tensor(&self.vt),
        }
    }
}

impl<T: TensorScalar> Clone for FullPivLuResult<T> {
    fn clone(&self) -> Self {
        Self {
            p: clone_linalg_tensor(&self.p),
            l: clone_linalg_tensor(&self.l),
            u: clone_linalg_tensor(&self.u),
            q: clone_linalg_tensor(&self.q),
        }
    }
}

/// Result of complete-pivoting LU decomposition on [`Matrix`] values.
/// This is the matrix-shaped counterpart of [`FullPivLuResult`]. It exists so
/// downstream crates can use backend linalg without hand-writing
/// `TypedTensor` conversion code.
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, full_piv_lu_matrix};
/// let matrix = from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);
/// let factors = full_piv_lu_matrix(&matrix).unwrap();
/// assert_eq!(factors.u.nrows(), 2);
/// assert_eq!(factors.u.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct FullPivLuMatrixResult<T> {
    /// Left permutation matrix.
    pub p: Matrix<T>,
    /// Lower triangular factor.
    pub l: Matrix<T>,
    /// Upper triangular factor.
    pub u: Matrix<T>,
    /// Right permutation matrix.
    pub q: Matrix<T>,
}

/// Scalar bound accepted by tensor4all's typed linalg wrappers.
pub trait BackendLinalgScalar: TensorScalar {}

/// Scalar types that can solve `T * P = Pi1` via a right full-pivoting LU
/// solve on the tenferro backend.
///
/// This is the foundational seam for matrix cross-interpolation (CI)
/// materialization: it lets an algorithm layer ask for the backend's
/// full-pivot LU solve without depending on any higher crate. The four
/// supported scalar types (f32, f64, Complex32, Complex64) are implemented
/// here; `f32`/`Complex32` inputs are solved in double precision by the
/// backend and converted back.
///
/// # Errors
///
/// Returns an error when the pivot matrix is not square, the shapes are
/// incompatible, or the backend solve fails.
pub trait FullPivLuScalar: BackendLinalgScalar {
    /// Solve `T * P = Pi1` for `T`, where `P` is the pivot matrix (column-major).
    ///
    /// # Errors
    ///
    /// Returns a [`BackendLinalgError`] when the pivot matrix is not square
    /// (a shape mismatch), when the shapes are incompatible (`lhs_cols !=
    /// pivot_rows`, a shape mismatch), or when the backend full-pivot LU solve
    /// fails (a backend failure).
    fn solve_right_full_piv_lu(
        lhs_values: &[Self],
        lhs_rows: usize,
        lhs_cols: usize,
        pivot_values: &[Self],
        pivot_rows: usize,
        pivot_cols: usize,
    ) -> std::result::Result<Vec<Self>, BackendLinalgError>;
}

macro_rules! impl_full_piv_lu_scalar {
    ($t:ty) => {
        impl FullPivLuScalar for $t {
            fn solve_right_full_piv_lu(
                lhs_values: &[Self],
                lhs_rows: usize,
                lhs_cols: usize,
                pivot_values: &[Self],
                pivot_rows: usize,
                pivot_cols: usize,
            ) -> std::result::Result<Vec<Self>, BackendLinalgError> {
                if pivot_rows != pivot_cols {
                    return Err(BackendLinalgError::from(anyhow::anyhow!(
                        "full-pivot solve requires a square pivot matrix, got {}x{}",
                        pivot_rows,
                        pivot_cols
                    )));
                }
                if lhs_cols != pivot_rows {
                    return Err(BackendLinalgError::from(anyhow::anyhow!(
                        "cannot solve T * P = Pi1 with Pi1 shape {}x{} and P shape {}x{}",
                        lhs_rows,
                        lhs_cols,
                        pivot_rows,
                        pivot_cols
                    )));
                }

                let lhs_t = transpose_column_major(lhs_values, lhs_rows, lhs_cols);
                let pivot_t = transpose_column_major(pivot_values, pivot_rows, pivot_cols);
                let pivot_tensor = tenferro_tensor::Tensor::from_vec_col_major(
                    vec![pivot_cols, pivot_rows],
                    pivot_t,
                )
                .map_err(|e| BackendLinalgError::from(anyhow::Error::new(e)))?;
                let lhs_tensor =
                    tenferro_tensor::Tensor::from_vec_col_major(vec![lhs_cols, lhs_rows], lhs_t)
                        .map_err(|e| BackendLinalgError::from(anyhow::Error::new(e)))?;
                let solved_t = with_default_session(|session| {
                    pivot_tensor.full_piv_lu_solve(&lhs_tensor, session)
                })
                .map_err(|e| {
                    BackendLinalgError::from(anyhow::anyhow!("full_piv_lu_solve failed: {e}"))
                })?;

                let solved_t_values = solved_t.as_slice::<Self>().map_err(|e| {
                    BackendLinalgError::from(anyhow::anyhow!(
                        "full_piv_lu_solve returned unexpected dtype: {e}"
                    ))
                })?;
                Ok(transpose_column_major(solved_t_values, lhs_cols, lhs_rows))
            }
        }
    };
}

impl_full_piv_lu_scalar!(f32);
impl_full_piv_lu_scalar!(f64);
impl_full_piv_lu_scalar!(num_complex::Complex32);
impl_full_piv_lu_scalar!(num_complex::Complex64);

/// Transpose a column-major flat buffer.
fn transpose_column_major<T: Copy + num_traits::Zero>(
    values: &[T],
    nrows: usize,
    ncols: usize,
) -> Vec<T> {
    let mut out = vec![T::zero(); nrows * ncols];
    for col in 0..ncols {
        for row in 0..nrows {
            out[col + ncols * row] = values[row + nrows * col];
        }
    }
    out
}

impl<T: TensorScalar> BackendLinalgScalar for T {}

/// Scalar types supported by [`solve_matrix`].
/// `f64` and `Complex64` are solved directly. `f32` and `Complex32` are
/// promoted to the corresponding 64-bit dtype for the backend solve and then
/// converted back, because the current tenferro CPU LU solve is double
/// precision only.
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, solve_matrix};
/// let a = from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
/// let b = from_vec2d(vec![vec![1.0_f32], vec![0.0]]);
/// let x = solve_matrix(&a, &b).unwrap();
/// assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
/// ```
pub trait MatrixSolveScalar: BackendLinalgScalar + crate::matrix::MatrixScalar {
    #[doc(hidden)]
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>>;

    #[doc(hidden)]
    fn solve_matrix_owned_impl(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>> {
        Self::solve_matrix_impl(&a, &b)
    }
}

/// Error returned by the CPU backend linear-algebra dispatch helpers.
///
/// Wraps the underlying tenferro or shape diagnostic, preserving its source
/// chain.
///
/// # Remedies
/// - Shape/dtype mismatch: verify matrix dims and scalar dtypes against the
///   operation contract before calling.
/// - Singular or invalid input: check matrix conditioning and finite values
///   before solve/factorization operations.
/// - Backend failure: the wrapped source chain identifies the failing stage.
#[derive(Debug, thiserror::Error)]
#[error("backend linear algebra failed: {source}")]
pub struct BackendLinalgError {
    /// Original backend or shape diagnostic.
    #[source]
    pub source: anyhow::Error,
}

impl From<anyhow::Error> for BackendLinalgError {
    fn from(source: anyhow::Error) -> Self {
        Self { source }
    }
}

/// Scalar types supported by [`triangular_solve_matrix`].
/// `f64` and `Complex64` are solved directly. `f32` and `Complex32` are
/// promoted to the corresponding 64-bit dtype for the backend solve and then
/// converted back, because the current tenferro CPU triangular solve is double
/// precision only.
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, triangular_solve_matrix};
/// let a = from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
/// let b = from_vec2d(vec![vec![2.0_f64], vec![7.0]]);
/// let x = triangular_solve_matrix(&a, &b, true, true, false, false).unwrap();
/// assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
/// assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
/// ```
pub trait MatrixTriangularSolveScalar: BackendLinalgScalar + crate::matrix::MatrixScalar {
    #[doc(hidden)]
    fn triangular_solve_matrix_impl(
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>>;

    #[doc(hidden)]
    fn triangular_solve_matrix_owned_impl(
        a: Matrix<Self>,
        b: Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        Self::triangular_solve_matrix_impl(&a, &b, left_side, lower, transpose_a, unit_diagonal)
    }
}

/// Small-matrix diagnostics produced by the successive randomized compression
/// stopping estimator.
///
/// `error` is the Appendix C randomized residual estimate and `norm` is the
/// corresponding Frobenius norm estimate. Both values use the sketch width as
/// their normalization factor.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{src_error_estimate, Matrix};
///
/// let r = Matrix::from_col_major_vec(1, 1, vec![2.0_f64]);
/// let estimate = src_error_estimate(&r).unwrap();
/// assert!((estimate.error - 2.0).abs() < 1.0e-12);
/// assert!((estimate.norm - 2.0).abs() < 1.0e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrcErrorEstimate {
    /// Estimated residual magnitude from the inverse-adjoint QR factor.
    pub error: f64,
    /// Estimated norm of the sketched tensor from the QR factor.
    pub norm: f64,
}

/// Compute the Appendix C SRC error and norm estimates from an upper-triangular
/// QR factor `R`.
///
/// Provenance: the formulas are Eq. (err-est) and Eq. (norm-est) in Appendix C
/// of Camaño--Epperly--Tropp, [arXiv:2504.06475](https://arxiv.org/abs/2504.06475),
/// cross-checked against `chriscamano/RandomMPOMPS/code/tensornetwork/incrementalqr.cpp::get_error_estimate`
/// (lines 106--119). The use of actual `R` plus an `R†` solve is an equivalent
/// representation derived in the audit; it is not a literal port of the
/// author's inverse-`R` storage.
///
/// The helper explicitly builds `R†` before solving `R† G = I`, so complex
/// inputs use the Hermitian adjoint rather than a plain transpose. The solve is
/// delegated to the configured tensor4all backend and is restricted to the
/// small sketch matrix; no general dense inverse routine is used.
///
/// # Errors
///
/// Returns [`BackendLinalgError`] when `r` is empty, non-square, singular, or
/// contains non-finite values, or when the backend triangular solve fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{src_error_estimate, Matrix};
///
/// let r = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 0.0, 1.0, 3.0]);
/// let estimate = src_error_estimate(&r).unwrap();
/// assert!(estimate.error.is_finite());
/// assert!(estimate.norm.is_finite());
/// ```
pub fn src_error_estimate<T>(
    r: &Matrix<T>,
) -> std::result::Result<SrcErrorEstimate, BackendLinalgError>
where
    T: MatrixTriangularSolveScalar + ComplexFloat,
{
    let inverse_adjoint = src_inverse_adjoint(r)?;
    src_error_estimate_from_inverse_adjoint(r, &inverse_adjoint)
}

/// Compute the inverse adjoint `R^{-†}` used by the Appendix C estimator.
///
/// This is crate-visible so incremental QR can initialize the stored
/// estimator state once and then update it with the block formula from
/// Appendix C.3 instead of solving the same triangular system after every
/// appended sketch block.
pub(crate) fn src_inverse_adjoint<T>(
    r: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: MatrixTriangularSolveScalar + ComplexFloat,
{
    let nrows = r.nrows();
    let ncols = r.ncols();
    if nrows != ncols {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator requires a square R, got {nrows}x{ncols}"
        )));
    }
    if nrows == 0 {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator requires a non-empty R"
        )));
    }

    // Only the diagonal is checked here; the full-R Frobenius-norm
    // finiteness check lives solely in `src_error_estimate_from_inverse_adjoint`,
    // which every `src_error_estimate` call already runs immediately after
    // this function returns. Duplicating that O(rank^2) accumulation here
    // would recompute the identical sum for nothing on SRC's adaptive
    // stopping-test hot path.
    for col in 0..ncols {
        let diagonal_sq = r[[col, col]].matrix_abs_sq();
        if !diagonal_sq.is_finite() || diagonal_sq == 0.0 {
            return Err(BackendLinalgError::from(anyhow!(
                "SRC estimator requires a finite, nonzero diagonal in R at ({col}, {col})"
            )));
        }
    }

    let mut adjoint = Matrix::zeros(nrows, ncols);
    for col in 0..ncols {
        for row in 0..nrows {
            adjoint[[row, col]] = r[[col, row]].conj();
        }
    }
    let mut identity = Matrix::zeros(nrows, ncols);
    for diagonal in 0..nrows {
        identity[[diagonal, diagonal]] = T::one();
    }

    let inverse_adjoint = triangular_solve_matrix(&adjoint, &identity, true, true, false, false)
        .map_err(|error| {
            BackendLinalgError::from(anyhow!(
                "SRC inverse-adjoint triangular solve failed: {error}"
            ))
        })?;
    Ok(inverse_adjoint)
}

/// Evaluate the Appendix C estimator from a previously computed `R^{-†}`.
///
/// The inverse-adjoint argument is intentionally separate from
/// [`src_error_estimate`] so incremental QR can reuse its updated triangular
/// solve state. This helper performs only norm accumulation and validation.
pub(crate) fn src_error_estimate_from_inverse_adjoint<T>(
    r: &Matrix<T>,
    inverse_adjoint: &Matrix<T>,
) -> std::result::Result<SrcErrorEstimate, BackendLinalgError>
where
    T: MatrixTriangularSolveScalar + ComplexFloat,
{
    let nrows = r.nrows();
    let ncols = r.ncols();
    if nrows != ncols || inverse_adjoint.nrows() != nrows || inverse_adjoint.ncols() != ncols {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator requires matching square R and inverse-adjoint factors"
        )));
    }
    if nrows == 0 {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator requires a non-empty R"
        )));
    }

    let mut norm_sq = 0.0_f64;
    for value in r.as_col_major_slice() {
        norm_sq += value.matrix_abs_sq();
    }
    if !norm_sq.is_finite() {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator requires finite entries in R"
        )));
    }
    let mut inverse_column_error_sq = 0.0_f64;
    for col in 0..ncols {
        let column_norm_sq = (0..nrows)
            .map(|row| inverse_adjoint[[row, col]].matrix_abs_sq())
            .sum::<f64>();
        if !column_norm_sq.is_finite() || column_norm_sq == 0.0 {
            return Err(BackendLinalgError::from(anyhow!(
                "SRC inverse-adjoint solve returned an invalid column norm at column {col}"
            )));
        }
        inverse_column_error_sq += 1.0 / column_norm_sq;
    }

    let sketch_width = ncols as f64;
    let error_sq = inverse_column_error_sq / sketch_width;
    let norm_estimate_sq = norm_sq / sketch_width;
    if !error_sq.is_finite() || !norm_estimate_sq.is_finite() {
        return Err(BackendLinalgError::from(anyhow!(
            "SRC estimator produced a non-finite estimate"
        )));
    }
    Ok(SrcErrorEstimate {
        error: error_sq.sqrt(),
        norm: norm_estimate_sq.sqrt(),
    })
}

fn solve_matrix_direct<T>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>>
where
    T: BackendLinalgScalar + Copy,
    Tensor: From<TypedTensor<T>>,
{
    solve_matrix_direct_owned(a.clone(), b.clone())
}

fn solve_matrix_direct_owned<T>(a: Matrix<T>, b: Matrix<T>) -> Result<Matrix<T>>
where
    T: BackendLinalgScalar + Copy,
    Tensor: From<TypedTensor<T>>,
{
    let a_tensor: Tensor = a.into_typed_tensor().into();
    let b_tensor: Tensor = b.into_typed_tensor().into();
    let result = with_default_session(|session| a_tensor.solve(&b_tensor, session))
        .map_err(|e| anyhow!("linear solve failed via tenferro-tensor: {e}"))?;
    let x = try_into_typed_result::<T>("solve", result)?;
    typed_tensor_to_matrix("solve", x)
}

fn triangular_solve_matrix_direct<T>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<Matrix<T>>
where
    T: BackendLinalgScalar + Copy,
    Tensor: From<TypedTensor<T>>,
{
    triangular_solve_matrix_direct_owned(
        a.clone(),
        b.clone(),
        left_side,
        lower,
        transpose_a,
        unit_diagonal,
    )
}

fn triangular_solve_matrix_direct_owned<T>(
    a: Matrix<T>,
    b: Matrix<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> Result<Matrix<T>>
where
    T: BackendLinalgScalar + Copy,
    Tensor: From<TypedTensor<T>>,
{
    let a_tensor: Tensor = a.into_typed_tensor().into();
    let b_tensor: Tensor = b.into_typed_tensor().into();
    let result = with_default_session(|session| {
        a_tensor.triangular_solve(
            &b_tensor,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            session,
        )
    })
    .map_err(|e| anyhow!("triangular solve failed via tenferro-tensor: {e}"))?;
    let x = try_into_typed_result::<T>("triangular_solve", result)?;
    typed_tensor_to_matrix("triangular_solve", x)
}

impl MatrixSolveScalar for f64 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct(a, b)
    }

    fn solve_matrix_owned_impl(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct_owned(a, b)
    }
}

impl MatrixTriangularSolveScalar for f64 {
    fn triangular_solve_matrix_impl(
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        triangular_solve_matrix_direct(a, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn triangular_solve_matrix_owned_impl(
        a: Matrix<Self>,
        b: Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        triangular_solve_matrix_direct_owned(a, b, left_side, lower, transpose_a, unit_diagonal)
    }
}

impl MatrixSolveScalar for Complex64 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct(a, b)
    }

    fn solve_matrix_owned_impl(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>> {
        solve_matrix_direct_owned(a, b)
    }
}

impl MatrixTriangularSolveScalar for Complex64 {
    fn triangular_solve_matrix_impl(
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        triangular_solve_matrix_direct(a, b, left_side, lower, transpose_a, unit_diagonal)
    }

    fn triangular_solve_matrix_owned_impl(
        a: Matrix<Self>,
        b: Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        triangular_solve_matrix_direct_owned(a, b, left_side, lower, transpose_a, unit_diagonal)
    }
}

impl MatrixSolveScalar for f32 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let x64 = solve_matrix_direct(&a64, &b64)?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| value as f32)
                .collect(),
        ))
    }
}

impl MatrixTriangularSolveScalar for f32 {
    fn triangular_solve_matrix_impl(
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| value as f64)
                .collect(),
        );
        let x64 = triangular_solve_matrix_direct(
            &a64,
            &b64,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| value as f32)
                .collect(),
        ))
    }
}

impl MatrixSolveScalar for Complex32 {
    fn solve_matrix_impl(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let x64 = solve_matrix_direct(&a64, &b64)?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| Complex32::new(value.re as f32, value.im as f32))
                .collect(),
        ))
    }
}

impl MatrixTriangularSolveScalar for Complex32 {
    fn triangular_solve_matrix_impl(
        a: &Matrix<Self>,
        b: &Matrix<Self>,
        left_side: bool,
        lower: bool,
        transpose_a: bool,
        unit_diagonal: bool,
    ) -> Result<Matrix<Self>> {
        let a64 = Matrix::from_col_major_vec(
            a.nrows(),
            a.ncols(),
            a.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let b64 = Matrix::from_col_major_vec(
            b.nrows(),
            b.ncols(),
            b.as_col_major_slice()
                .iter()
                .map(|&value| Complex64::new(value.re as f64, value.im as f64))
                .collect(),
        );
        let x64 = triangular_solve_matrix_direct(
            &a64,
            &b64,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )?;
        Ok(Matrix::from_col_major_vec(
            x64.nrows(),
            x64.ncols(),
            x64.as_col_major_slice()
                .iter()
                .map(|&value| Complex32::new(value.re as f32, value.im as f32))
                .collect(),
        ))
    }
}

fn tensor_scalar_dtype<T: TensorScalar>() -> DType {
    T::dtype()
}

fn try_into_typed_result<T: TensorScalar>(
    op: &'static str,
    tensor: Tensor,
) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::into_typed(tensor).map_err(|source| {
        anyhow!(
            "{op}: dtype mismatch lhs={actual:?} rhs={:?}: {source}",
            tensor_scalar_dtype::<T>()
        )
    })
}

fn convert_for_typed<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let expected = tensor_scalar_dtype::<T>();
    let tensor = if tensor.dtype() == expected {
        tensor
    } else {
        with_default_session(|session| tensor.convert(expected, session))
            .map_err(|e| anyhow!("{op}: dtype conversion to {expected:?} failed: {e}"))?
    };
    try_into_typed_result::<T>(op, tensor)
}

fn matrix_to_typed_tensor<T>(matrix: &Matrix<T>) -> TypedTensor<T>
where
    T: TensorScalar + Copy,
{
    crate::require_invariant(
        TypedTensor::from_vec_col_major(
            vec![matrix.nrows(), matrix.ncols()],
            matrix.as_col_major_slice().to_vec(),
        ),
        "validated matrix rejected by tenferro",
    )
}

fn typed_tensor_to_matrix<T>(op: &'static str, tensor: TypedTensor<T>) -> Result<Matrix<T>>
where
    T: TensorScalar + Copy,
{
    Matrix::try_from_typed_tensor(tensor).map_err(|err| anyhow!("{op}: {err}"))
}

fn require_host_linalg_tensor<T: TensorScalar>(
    op: &'static str,
    tensor: TypedTensor<T>,
) -> Result<TypedTensor<T>> {
    tensor
        .host_data()
        .map_err(|error| anyhow!("{op}: result must be host-backed: {error}"))?;
    Ok(tensor)
}

/// Compute a thin/economy SVD on a typed tensor.
/// # Errors
///
/// Returns an error when the SVD fails (a backend or non-convergence
/// /// failure).
///
pub fn svd_backend<T>(a: &TypedTensor<T>) -> std::result::Result<SvdResult<T>, BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let tensor = T::into_tensor(
        a.shape().to_vec(),
        a.host_data()
            .map_err(|e| anyhow!("SVD input host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("SVD input tensor construction failed: {e}"))?;
    let (u, s, vt) = with_default_session(|session| tensor.svd(session))
        .map_err(|e| anyhow!("SVD computation failed via tenferro-tensor: {e}"))?;
    Ok(SvdResult {
        u: require_host_linalg_tensor("svd", convert_for_typed::<T>("svd", u)?)?,
        s: require_host_linalg_tensor("svd", convert_for_typed::<T::Real>("svd", s)?)?,
        vt: require_host_linalg_tensor("svd", convert_for_typed::<T>("svd", vt)?)?,
    })
}

/// Compute a thin/economy QR decomposition, consuming the input tensor so its
/// column-major storage can be transferred to the backend without copying.
/// # Errors
///
/// Returns an error when the QR fails (a backend or non-convergence
/// /// failure).
///
pub fn qr_backend<T>(
    a: TypedTensor<T>,
) -> std::result::Result<(TypedTensor<T>, TypedTensor<T>), BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let (shape, data) = a
        .into_vec_col_major()
        .map_err(|e| anyhow!("QR input host access failed: {e}"))?;
    let tensor = T::into_tensor(shape, data)
        .map_err(|e| anyhow!("QR input tensor construction failed: {e}"))?;
    let (q, r) = with_default_session(|session| tensor.qr(session))
        .map_err(|e| anyhow!("QR computation failed via tenferro-tensor: {e}"))?;
    Ok((
        convert_for_typed::<T>("qr", q)?,
        convert_for_typed::<T>("qr", r)?,
    ))
}

/// Solve `A X = B` with the configured tenferro backend.
/// # Errors
/// Returns an error when the input shapes or scalar dtype are invalid (a
/// shape or dtype mismatch) or the coefficient matrix is singular (a singular
/// failure).
pub fn solve_backend<T>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
) -> std::result::Result<TypedTensor<T>, BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let a_tensor = T::into_tensor(
        a.shape().to_vec(),
        a.host_data()
            .map_err(|e| anyhow!("solve input host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("solve lhs tensor construction failed: {e}"))?;
    let b_tensor = T::into_tensor(
        b.shape().to_vec(),
        b.host_data()
            .map_err(|e| anyhow!("solve rhs host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("solve rhs tensor construction failed: {e}"))?;
    let result = with_default_session(|session| a_tensor.solve(&b_tensor, session))
        .map_err(|e| anyhow!("linear solve failed via tenferro-tensor: {e}"))?;
    try_into_typed_result::<T>("solve", result).map_err(BackendLinalgError::from)
}

/// Solve a triangular system with the configured tenferro backend.
/// If `left_side` is true, this solves `op(A) X = B`; otherwise it solves
/// `X op(A) = B`. `lower` selects the triangular half, `transpose_a` applies
/// a transpose to `A`, and `unit_diagonal` treats the diagonal of `A` as ones.
/// # Errors
///
/// Returns an error when the solve fails (a backend, singular, or shape
/// /// mismatch failure).
///
pub fn triangular_solve_backend<T>(
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> std::result::Result<TypedTensor<T>, BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let a_tensor = T::into_tensor(
        a.shape().to_vec(),
        a.host_data()
            .map_err(|e| anyhow!("triangular solve input host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("triangular solve lhs tensor construction failed: {e}"))?;
    let b_tensor = T::into_tensor(
        b.shape().to_vec(),
        b.host_data()
            .map_err(|e| anyhow!("triangular solve rhs host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("triangular solve rhs tensor construction failed: {e}"))?;
    let result = with_default_session(|session| {
        a_tensor.triangular_solve(
            &b_tensor,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
            session,
        )
    })
    .map_err(|e| anyhow!("triangular solve failed via tenferro-tensor: {e}"))?;
    try_into_typed_result::<T>("triangular_solve", result).map_err(BackendLinalgError::from)
}

/// Solve `A X = B` for column-major [`Matrix`] values.
/// This routes the operation through the configured tenferro backend and keeps
/// matrix-to-tensor conversion centralized in `tensor4all-tensorbackend`.
/// # Errors
///
/// Returns an error when the input shapes or scalar dtype are invalid (a
/// /// shape or dtype mismatch) or the solve fails (a backend or singular
/// /// failure).
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, solve_matrix};
/// let a = from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
/// let b = from_vec2d(vec![vec![1.0_f64], vec![0.0]]);
/// let x = solve_matrix(&a, &b).unwrap();
/// assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
/// assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
/// ```
pub fn solve_matrix<T>(
    a: &Matrix<T>,
    b: &Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: MatrixSolveScalar,
{
    T::solve_matrix_impl(a, b).map_err(BackendLinalgError::from)
}

/// Solve `A X = B` while consuming column-major [`Matrix`] values.
/// This routes the operation through the configured tenferro backend and reuses
/// the input buffers when constructing backend tensors for directly supported
/// scalar types.
/// # Errors
///
/// Returns an error when the input shapes or scalar dtype are invalid (a
/// /// shape or dtype mismatch) or the solve fails (a backend or singular
/// /// failure).
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, solve_matrix_owned};
/// let a = from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
/// let b = from_vec2d(vec![vec![1.0_f64], vec![0.0]]);
/// let x = solve_matrix_owned(a, b).unwrap();
/// assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
/// assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
/// ```
pub fn solve_matrix_owned<T>(
    a: Matrix<T>,
    b: Matrix<T>,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: MatrixSolveScalar,
{
    T::solve_matrix_owned_impl(a, b).map_err(BackendLinalgError::from)
}

/// Solve a triangular system for column-major [`Matrix`] values.
/// If `left_side` is true, this solves `op(A) X = B`; otherwise it solves
/// `X op(A) = B`. `lower` selects the triangular half, `transpose_a` applies
/// a transpose to `A`, and `unit_diagonal` treats the diagonal of `A` as ones.
/// # Errors
///
/// Returns an error when the input shapes or scalar dtype are invalid (a
/// /// shape or dtype mismatch), the triangular flags are invalid (an
/// /// invalid-configuration failure), or the solve fails (a backend or singular
/// /// failure).
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, triangular_solve_matrix};
/// let a = from_vec2d(vec![vec![2.0_f64, 1.0], vec![0.0, 3.0]]);
/// let b = from_vec2d(vec![vec![2.0_f64, 7.0]]);
/// let x = triangular_solve_matrix(&a, &b, false, false, false, false).unwrap();
/// assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
/// assert!((x[[0, 1]] - 2.0).abs() < 1.0e-12);
/// ```
pub fn triangular_solve_matrix<T>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: MatrixTriangularSolveScalar,
{
    T::triangular_solve_matrix_impl(a, b, left_side, lower, transpose_a, unit_diagonal)
        .map_err(BackendLinalgError::from)
}

/// Solve a triangular system while consuming column-major [`Matrix`] values.
/// If `left_side` is true, this solves `op(A) X = B`; otherwise it solves
/// `X op(A) = B`. `lower` selects the triangular half, `transpose_a` applies
/// a transpose to `A`, and `unit_diagonal` treats the diagonal of `A` as ones.
/// # Errors
///
/// Returns an error when the input shapes or scalar dtype are invalid (a
/// /// shape or dtype mismatch), the triangular flags are invalid (an
/// /// invalid-configuration failure), or the solve fails (a backend or singular
/// /// failure).
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, triangular_solve_matrix_owned};
/// let a = from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
/// let b = from_vec2d(vec![vec![2.0_f64], vec![7.0]]);
/// let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();
/// assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
/// assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
/// ```
pub fn triangular_solve_matrix_owned<T>(
    a: Matrix<T>,
    b: Matrix<T>,
    left_side: bool,
    lower: bool,
    transpose_a: bool,
    unit_diagonal: bool,
) -> std::result::Result<Matrix<T>, BackendLinalgError>
where
    T: MatrixTriangularSolveScalar,
{
    T::triangular_solve_matrix_owned_impl(a, b, left_side, lower, transpose_a, unit_diagonal)
        .map_err(BackendLinalgError::from)
}

/// Compute complete-pivoting LU with the configured tenferro backend.
/// # Errors
///
/// Returns an error when the LU factorization fails (a backend or
/// /// non-convergence failure).
///
pub fn full_piv_lu_backend<T>(
    a: &TypedTensor<T>,
) -> std::result::Result<FullPivLuResult<T>, BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let tensor = T::into_tensor(
        a.shape().to_vec(),
        a.host_data()
            .map_err(|e| anyhow!("LU input host access failed: {e}"))?
            .to_vec(),
    )
    .map_err(|e| anyhow!("LU input tensor construction failed: {e}"))?;
    let (p, l, u, q, _parity) = with_default_session(|session| tensor.full_piv_lu(session))
        .map_err(|e| anyhow!("complete-pivoting LU failed via tenferro-tensor: {e}"))?;
    Ok(FullPivLuResult {
        p: require_host_linalg_tensor("full_piv_lu", convert_for_typed::<T>("full_piv_lu", p)?)?,
        l: require_host_linalg_tensor("full_piv_lu", convert_for_typed::<T>("full_piv_lu", l)?)?,
        u: require_host_linalg_tensor("full_piv_lu", convert_for_typed::<T>("full_piv_lu", u)?)?,
        q: require_host_linalg_tensor("full_piv_lu", convert_for_typed::<T>("full_piv_lu", q)?)?,
    })
}

/// Compute complete-pivoting LU for a column-major [`Matrix`].
/// This is a convenience wrapper over [`full_piv_lu_backend`] for callers that
/// use [`Matrix`] as their dense boundary type.
/// # Errors
///
/// Returns an error when the backend does not support the input dtype (a
/// /// dtype mismatch) or the factorization fails (a backend or
/// /// non-convergence failure).
///
/// # Examples
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, full_piv_lu_matrix};
/// let matrix = from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);
/// let factors = full_piv_lu_matrix(&matrix).unwrap();
/// assert_eq!(factors.p.nrows(), 2);
/// assert_eq!(factors.q.ncols(), 2);
/// ```
pub fn full_piv_lu_matrix<T>(
    a: &Matrix<T>,
) -> std::result::Result<FullPivLuMatrixResult<T>, BackendLinalgError>
where
    T: BackendLinalgScalar + Copy,
{
    let tensor = matrix_to_typed_tensor(a);
    let decomp = full_piv_lu_backend(&tensor)?;
    Ok(FullPivLuMatrixResult {
        p: typed_tensor_to_matrix("full_piv_lu", decomp.p)?,
        l: typed_tensor_to_matrix("full_piv_lu", decomp.l)?,
        u: typed_tensor_to_matrix("full_piv_lu", decomp.u)?,
        q: typed_tensor_to_matrix("full_piv_lu", decomp.q)?,
    })
}

#[cfg(test)]
mod tests;
