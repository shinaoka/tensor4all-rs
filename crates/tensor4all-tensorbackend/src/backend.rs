//! Backend dispatch helpers for linear algebra operations.
//!
//! This module keeps tensor4all's typed factorization entry points thin while
//! routing the actual work through the shared tenferro CPU backend.

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use tenferro::{DType, Tensor, TensorScalar, TypedTensor};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::BackendSessionHost;

use crate::context::with_default_backend;
use crate::matrix::Matrix;

/// Result of SVD decomposition `A = U * diag(S) * Vt`.
/// The singular values are stored in a real-valued typed tensor, even when the
/// input matrix is complex.
/// # Examples
/// ```
/// use tensor4all_tensorbackend::svd_backend;
/// use tenferro::TypedTensor;
/// let a = TypedTensor::<f64>::from_vec_col_major(vec![2, 2], vec![1.0, 0.0, 0.0, 2.0]);
/// let result = svd_backend(&a).unwrap();
/// assert_eq!(result.u.shape(), &[2, 2]);
/// assert_eq!(result.s.shape(), &[2]);
/// assert_eq!(result.vt.shape(), &[2, 2]);
/// ```
#[derive(Debug, Clone)]
pub struct SvdResult<T: TensorScalar> {
    /// Left singular vectors.
    pub u: TypedTensor<T>,
    /// Singular values.
    pub s: TypedTensor<T::Real>,
    /// Right singular vectors transposed.
    pub vt: TypedTensor<T>,
}

/// Result of complete-pivoting LU decomposition `P A Q^T = L U`.
/// The parity output from tenferro is intentionally omitted because current
/// tensor4all callers only need the permutation matrices and the upper
/// triangular factor for pivot selection.
#[derive(Debug, Clone)]
pub struct FullPivLuResult<T: TensorScalar> {
    /// Left permutation matrix.
    pub p: TypedTensor<T>,
    /// Lower triangular factor.
    pub l: TypedTensor<T>,
    /// Upper triangular factor.
    pub u: TypedTensor<T>,
    /// Right permutation matrix.
    pub q: TypedTensor<T>,
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
    let result = with_default_backend(|backend| backend.solve(&a_tensor, &b_tensor))
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
    let result = with_default_backend(|backend| {
        backend.triangular_solve(
            &a_tensor,
            &b_tensor,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
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
    T::into_tensor(vec![0], Vec::new()).dtype()
}

fn try_into_typed_result<T: TensorScalar>(
    op: &'static str,
    tensor: Tensor,
) -> Result<TypedTensor<T>> {
    let actual = tensor.dtype();
    T::try_into_typed(tensor).ok_or_else(|| {
        anyhow!(
            "{op}: dtype mismatch lhs={actual:?} rhs={:?}",
            tensor_scalar_dtype::<T>()
        )
    })
}

fn convert_for_typed<T: TensorScalar>(op: &'static str, tensor: Tensor) -> Result<TypedTensor<T>> {
    let expected = tensor_scalar_dtype::<T>();
    let tensor = if tensor.dtype() == expected {
        tensor
    } else {
        with_default_backend(|backend| {
            backend.with_backend_session(|exec| exec.convert(&tensor, expected))
        })
        .map_err(|e| anyhow!("{op}: dtype conversion to {expected:?} failed: {e}"))?
    };
    try_into_typed_result::<T>(op, tensor)
}

fn matrix_to_typed_tensor<T>(matrix: &Matrix<T>) -> TypedTensor<T>
where
    T: TensorScalar + Copy,
{
    TypedTensor::from_vec_col_major(
        vec![matrix.nrows(), matrix.ncols()],
        matrix.as_col_major_slice().to_vec(),
    )
}

fn typed_tensor_to_matrix<T>(op: &'static str, tensor: TypedTensor<T>) -> Result<Matrix<T>>
where
    T: TensorScalar + Copy,
{
    Matrix::try_from_typed_tensor(tensor).map_err(|err| anyhow!("{op}: {err}"))
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
    let tensor = T::into_tensor(a.shape().to_vec(), a.host_data().to_vec());
    let outputs = with_default_backend(|backend| backend.svd(&tensor))
        .map_err(|e| anyhow!("SVD computation failed via tenferro-tensor: {e}"))?;
    let [u, s, vt]: [Tensor; 3] = outputs.try_into().map_err(|outputs: Vec<Tensor>| {
        anyhow!("svd: expected 3 outputs, got {}", outputs.len())
    })?;
    Ok(SvdResult {
        u: convert_for_typed::<T>("svd", u)?,
        s: convert_for_typed::<T::Real>("svd", s)?,
        vt: convert_for_typed::<T>("svd", vt)?,
    })
}

/// Compute a thin/economy QR decomposition on a typed tensor.
/// # Errors
///
/// Returns an error when the QR fails (a backend or non-convergence
/// /// failure).
///
pub fn qr_backend<T>(
    a: &TypedTensor<T>,
) -> std::result::Result<(TypedTensor<T>, TypedTensor<T>), BackendLinalgError>
where
    T: BackendLinalgScalar,
{
    let tensor = T::into_tensor(a.shape().to_vec(), a.host_data().to_vec());
    let outputs = with_default_backend(|backend| backend.qr(&tensor))
        .map_err(|e| anyhow!("QR computation failed via tenferro-tensor: {e}"))?;
    let [q, r]: [Tensor; 2] = outputs
        .try_into()
        .map_err(|outputs: Vec<Tensor>| anyhow!("qr: expected 2 outputs, got {}", outputs.len()))?;
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
    let a_tensor = T::into_tensor(a.shape().to_vec(), a.host_data().to_vec());
    let b_tensor = T::into_tensor(b.shape().to_vec(), b.host_data().to_vec());
    let result = with_default_backend(|backend| backend.solve(&a_tensor, &b_tensor))
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
) -> Result<TypedTensor<T>>
where
    T: BackendLinalgScalar,
{
    let a_tensor = T::into_tensor(a.shape().to_vec(), a.host_data().to_vec());
    let b_tensor = T::into_tensor(b.shape().to_vec(), b.host_data().to_vec());
    let result = with_default_backend(|backend| {
        backend.triangular_solve(
            &a_tensor,
            &b_tensor,
            left_side,
            lower,
            transpose_a,
            unit_diagonal,
        )
    })
    .map_err(|e| anyhow!("triangular solve failed via tenferro-tensor: {e}"))?;
    try_into_typed_result::<T>("triangular_solve", result)
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
    let tensor = T::into_tensor(a.shape().to_vec(), a.host_data().to_vec());
    let outputs = with_default_backend(|backend| backend.full_piv_lu(&tensor))
        .map_err(|e| anyhow!("complete-pivoting LU failed via tenferro-tensor: {e}"))?;
    let [p, l, u, q, _parity]: [Tensor; 5] =
        outputs.try_into().map_err(|outputs: Vec<Tensor>| {
            anyhow!("full_piv_lu: expected 5 outputs, got {}", outputs.len())
        })?;
    Ok(FullPivLuResult {
        p: convert_for_typed::<T>("full_piv_lu", p)?,
        l: convert_for_typed::<T>("full_piv_lu", l)?,
        u: convert_for_typed::<T>("full_piv_lu", u)?,
        q: convert_for_typed::<T>("full_piv_lu", q)?,
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
