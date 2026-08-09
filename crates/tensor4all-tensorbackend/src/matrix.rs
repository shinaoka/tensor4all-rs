//! Dense column-major matrix type and utility functions.
//!
//! [`Matrix<T>`] is a simple dense 2D matrix in column-major layout, indexed
//! by `m[[row, col]]`. It is the shared dense matrix boundary for tensor4all
//! crates that need flat buffers and backend-backed matrix multiplication.
//!
//! # Examples
//!
//! ```
//! use tensor4all_tensorbackend::{from_vec2d, Matrix};
//!
//! let m = from_vec2d(vec![
//!     vec![1.0_f64, 2.0],
//!     vec![3.0, 4.0],
//! ]);
//! assert_eq!(m.nrows(), 2);
//! assert_eq!(m.ncols(), 2);
//! assert_eq!(m[[0, 1]], 2.0);
//! assert_eq!(m[[1, 0]], 3.0);
//! ```

use anyhow::{ensure, Context, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use std::ops::{Index, IndexMut};
use tenferro::{Tensor, TensorScalar, TypedTensor};
use tenferro_ad::EagerTensor;
use tenferro_tensor::BackendSessionHost;

/// A dense 2D matrix in column-major layout.
///
/// Access elements with `m[[row, col]]` syntax. Data is stored contiguously
/// in column-major order, so flat buffers use `row + nrows * col`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::Matrix;
///
/// let mut m = Matrix::zeros(2, 3);
/// m[[0, 1]] = 5.0_f64;
/// assert_eq!(m[[0, 1]], 5.0);
/// assert_eq!(m[[0, 0]], 0.0);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 3);
/// ```
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    data: Vec<T>,
    nrows: usize,
    ncols: usize,
}

fn checked_matrix_len(nrows: usize, ncols: usize) -> Option<usize> {
    nrows.checked_mul(ncols)
}

/// Error returned when converting a [`TypedTensor`] into a [`Matrix`].
///
/// Use this when accepting dynamic tensor-shaped values at a dense-matrix
/// boundary. It reports whether conversion failed because the tensor was not a
/// rank-2 matrix or because its host buffer could not be consumed.
///
/// # Examples
///
/// ```
/// use tenferro::TypedTensor;
/// use tensor4all_tensorbackend::Matrix;
///
/// let tensor = TypedTensor::from_vec_col_major(vec![2, 1, 1], vec![1.0_f64, 2.0]);
/// let err = Matrix::try_from_typed_tensor(tensor).unwrap_err();
/// assert!(err.to_string().contains("rank-2 tensor"));
/// ```
#[derive(Debug, thiserror::Error)]
pub enum MatrixTensorConversionError {
    /// The input tensor rank was not two.
    #[error("expected a rank-2 tensor, got shape {shape:?}")]
    Rank {
        /// Tensor shape that failed the rank check.
        shape: Vec<usize>,
    },
    /// The tensor did not contain an owned host buffer that can be consumed.
    #[error("failed to consume typed tensor host buffer: {message}")]
    HostBuffer {
        /// Backend conversion error reported by tenferro.
        message: String,
    },
}

/// Error returned when row-shaped input cannot be converted into a [`Matrix`].
///
/// Use this when accepting user-provided row data. It reports the first row
/// whose length differs from the first row, so callers can reject malformed
/// input before any values are dropped or indexed out of bounds.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{try_from_vec2d, MatrixShapeError};
///
/// let err = try_from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0]]).unwrap_err();
/// assert!(matches!(
///     err,
///     MatrixShapeError::RaggedRows {
///         row: 1,
///         expected: 2,
///         actual: 1,
///     }
/// ));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MatrixShapeError {
    /// A later row had a different length than the first row.
    #[error("row {row} has length {actual}, expected {expected}")]
    RaggedRows {
        /// Zero-based row number with the mismatched length.
        row: usize,
        /// Column count established by the first row.
        expected: usize,
        /// Actual number of entries in `row`.
        actual: usize,
    },
}

/// Error returned by [`lowest_hermitian_eigenpair`].
///
/// The eigensolver is intended for small Rayleigh-Ritz projected matrices; it validates shape and Hermitian structure before calling the backend Hermitian
/// eigendecomposition, symmetrizing only roundoff that is within the requested
/// tolerance. Non-Hermitian effective operators are rejected explicitly instead
/// of silently taking a real part.
#[derive(Debug, thiserror::Error)]
pub enum HermitianEigenError {
    /// The matrix has zero rows and columns, so it has no eigenpair.
    #[error("Hermitian eigenpair requires a non-empty matrix")]
    Empty,
    /// The input is not square.
    #[error("Hermitian eigenpair requires a square matrix, got {nrows}x{ncols}")]
    NonSquare {
        /// Number of matrix rows.
        nrows: usize,
        /// Number of matrix columns.
        ncols: usize,
    },
    /// The Hermitian validation tolerance was negative or not finite.
    #[error("Hermitian tolerance must be finite and non-negative, got {tolerance}")]
    InvalidTolerance {
        /// Rejected tolerance value.
        tolerance: f64,
    },
    /// A matrix entry violates `A[i, j] = conj(A[j, i])` within tolerance.
    #[error(
        "matrix is not Hermitian at ({row}, {col}): difference {difference} exceeds tolerance {tolerance}"
    )]
    NonHermitian {
        /// Row of the first offending entry.
        row: usize,
        /// Column of the first offending entry.
        col: usize,
        /// Absolute Hermitian residual for the offending pair.
        difference: f64,
        /// Effective tolerance used for this entry pair.
        tolerance: f64,
    },
    /// The backend returned an output with an unexpected dtype.
    #[error("{output} output dtype mismatch: expected {expected}, got {actual}")]
    DType {
        /// Output tensor name.
        output: &'static str,
        /// Expected dtype string.
        expected: String,
        /// Actual dtype string.
        actual: String,
    },
    /// The backend returned an output with an unexpected shape.
    #[error("{output} output shape mismatch: expected {expected:?}, got {actual:?}")]
    Shape {
        /// Output tensor name.
        output: &'static str,
        /// Expected shape.
        expected: Vec<usize>,
        /// Actual shape.
        actual: Vec<usize>,
    },
    /// A Hermitian backend eigenvalue had a non-negligible imaginary part.
    #[error(
        "Hermitian eigenvalue {index} has imaginary part {imaginary}, exceeding tolerance {tolerance}"
    )]
    NonRealEigenvalue {
        /// Eigenvalue position in the backend output.
        index: usize,
        /// Absolute imaginary part.
        imaginary: f64,
        /// Tolerance used for validation.
        tolerance: f64,
    },
    /// The tenferro backend rejected or failed the eigendecomposition.
    #[error("Hermitian eigendecomposition failed: {source}")]
    Backend {
        /// Original backend diagnostic.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },
}

/// Small Hermitian eigenpair returned by [`lowest_hermitian_eigenpair`].
///
/// `eigenvector` stores the Ritz vector coefficients in ordinary vector order.
/// It has length equal to the input matrix dimension and is normalized according
/// to the backend eigendecomposition.
#[derive(Debug, Clone, PartialEq)]
pub struct HermitianEigenpair<T> {
    /// Smallest eigenvalue of the Hermitian matrix.
    pub eigenvalue: f64,
    /// Corresponding eigenvector coefficients.
    pub eigenvector: Vec<T>,
}

/// Full eigendecomposition of a small Hermitian projected matrix.
///
/// `eigenvectors` stores one normalized eigenvector per column in column-major
/// [`Matrix`] layout. Eigenvalues are returned in the backend's ascending
/// Hermitian eigensolver order.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{hermitian_eigendecomposition, Matrix};
///
/// let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
/// let decomp = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap();
/// assert_eq!(decomp.eigenvalues, vec![1.0, 2.0]);
/// assert_eq!(decomp.eigenvectors.nrows(), 2);
/// assert_eq!(decomp.eigenvectors.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct HermitianEigendecomposition<T> {
    /// Real eigenvalues of the Hermitian matrix.
    pub eigenvalues: Vec<f64>,
    /// Eigenvector matrix with one eigenvector in each column.
    pub eigenvectors: Matrix<T>,
}

/// Scalar types supported by [`lowest_hermitian_eigenpair`].
///
/// The current backend path is used for `f64` and `Complex64`, which are the
/// scalar types needed by tensor4all's Hermitian Krylov and DMRG algorithms.
pub trait HermitianEigenScalar: TensorScalar + MatrixScalar {
    #[doc(hidden)]
    fn hermitian_difference(a_ij: Self, a_ji: Self) -> f64;

    #[doc(hidden)]
    fn hermitian_scale(a_ij: Self, a_ji: Self) -> f64;

    #[doc(hidden)]
    fn symmetrized_hermitian_pair(a_ij: Self, a_ji: Self) -> (Self, Self);

    #[doc(hidden)]
    fn eigenvalues_from_tensor(
        tensor: Tensor,
        tolerance: f64,
    ) -> std::result::Result<(Vec<usize>, Vec<f64>), HermitianEigenError>;

    #[doc(hidden)]
    fn to_complex64(value: Self) -> Complex64;
}

impl HermitianEigenScalar for f64 {
    fn hermitian_difference(a_ij: Self, a_ji: Self) -> f64 {
        (a_ij - a_ji).abs()
    }

    fn hermitian_scale(a_ij: Self, a_ji: Self) -> f64 {
        a_ij.abs().max(a_ji.abs()).max(1.0)
    }

    fn symmetrized_hermitian_pair(a_ij: Self, a_ji: Self) -> (Self, Self) {
        let value = 0.5 * (a_ij + a_ji);
        (value, value)
    }

    fn eigenvalues_from_tensor(
        tensor: Tensor,
        _tolerance: f64,
    ) -> std::result::Result<(Vec<usize>, Vec<f64>), HermitianEigenError> {
        let values = typed_eigh_output::<f64>("eigenvalues", tensor)?;
        Ok((values.shape().to_vec(), values.as_slice().to_vec()))
    }

    fn to_complex64(value: Self) -> Complex64 {
        Complex64::new(value, 0.0)
    }
}

impl HermitianEigenScalar for Complex64 {
    fn hermitian_difference(a_ij: Self, a_ji: Self) -> f64 {
        (a_ij - a_ji.conj()).norm()
    }

    fn hermitian_scale(a_ij: Self, a_ji: Self) -> f64 {
        a_ij.norm().max(a_ji.norm()).max(1.0)
    }

    fn symmetrized_hermitian_pair(a_ij: Self, a_ji: Self) -> (Self, Self) {
        let value = 0.5 * (a_ij + a_ji.conj());
        (value, value.conj())
    }

    fn eigenvalues_from_tensor(
        tensor: Tensor,
        tolerance: f64,
    ) -> std::result::Result<(Vec<usize>, Vec<f64>), HermitianEigenError> {
        let values = typed_eigh_output::<Complex64>("eigenvalues", tensor)?;
        let mut real_values = Vec::with_capacity(values.as_slice().len());
        for (index, value) in values.as_slice().iter().copied().enumerate() {
            let imaginary = value.im.abs();
            let allowed = tolerance * value.norm().max(1.0);
            if imaginary > allowed {
                return Err(HermitianEigenError::NonRealEigenvalue {
                    index,
                    imaginary,
                    tolerance: allowed,
                });
            }
            real_values.push(value.re);
        }
        Ok((values.shape().to_vec(), real_values))
    }

    fn to_complex64(value: Self) -> Complex64 {
        value
    }
}

impl<T> Matrix<T> {
    /// Create a matrix from raw column-major data.
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows or if `data.len() != nrows * ncols`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
    /// assert_eq!(m[[0, 0]], 1.0);
    /// assert_eq!(m[[0, 1]], 2.0);
    /// assert_eq!(m[[1, 0]], 3.0);
    /// assert_eq!(m[[1, 1]], 4.0);
    /// ```
    pub fn from_col_major_vec(nrows: usize, ncols: usize, data: Vec<T>) -> Self {
        let expected = checked_matrix_len(nrows, ncols);
        assert!(
            expected.is_some(),
            "matrix shape product overflow: {nrows} rows * {ncols} columns"
        );
        let expected = expected.map_or(0, |expected| expected);
        assert_eq!(data.len(), expected);
        Self { data, nrows, ncols }
    }

    /// View the underlying column-major data as a contiguous slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    /// assert_eq!(m.as_col_major_slice(), &[1, 3, 2, 4]);
    /// ```
    pub fn as_col_major_slice(&self) -> &[T] {
        &self.data
    }

    /// View the underlying column-major data as a mutable contiguous slice.
    ///
    /// The slice uses `row + nrows * col` ordering. This is useful for kernels
    /// that validate dimensions once and then operate over contiguous columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let mut m = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    /// m.as_col_major_mut_slice()[1] = 30;
    /// assert_eq!(m[[1, 0]], 30);
    /// ```
    pub fn as_col_major_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consume the matrix and return its owned column-major buffer.
    ///
    /// The returned buffer uses `row + nrows * col` ordering. Use this when
    /// transferring matrix storage to another column-major dense container
    /// without cloning.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);
    /// let data = m.into_col_major_vec();
    /// assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn into_col_major_vec(self) -> Vec<T> {
        self.data
    }

    /// Borrow this matrix as an owned tenferro [`TypedTensor`].
    ///
    /// This clones the matrix buffer and preserves column-major layout. Use
    /// [`Matrix::into_typed_tensor`] when the matrix can be consumed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let tensor = m.to_typed_tensor();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn to_typed_tensor(&self) -> TypedTensor<T>
    where
        T: TensorScalar,
    {
        TypedTensor::from_vec_col_major(vec![self.nrows, self.ncols], self.data.clone())
    }

    /// Consume this matrix as a tenferro [`TypedTensor`] without cloning.
    ///
    /// The tensor shape is `[nrows, ncols]`, and the owned data remains in
    /// column-major layout.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let tensor = m.into_typed_tensor();
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
    /// ```
    pub fn into_typed_tensor(self) -> TypedTensor<T>
    where
        T: TensorScalar,
    {
        TypedTensor::from_vec_col_major(vec![self.nrows, self.ncols], self.data)
    }

    /// Consume a rank-2 tenferro [`TypedTensor`] as a [`Matrix`].
    ///
    /// The input tensor must have shape `[nrows, ncols]` and an owned host
    /// buffer. The buffer is reused without cloning and interpreted as
    /// column-major matrix storage.
    ///
    /// # Errors
    ///
    /// Returns [`MatrixTensorConversionError::Rank`] if the tensor is not
    /// rank-2, or [`MatrixTensorConversionError::HostBuffer`] if tenferro
    /// cannot export the tensor as an owned host buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tenferro::TypedTensor;
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]);
    /// let m = Matrix::try_from_typed_tensor(tensor).unwrap();
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 2);
    /// assert_eq!(m[[0, 1]], 2.0);
    /// ```
    pub fn try_from_typed_tensor(
        tensor: TypedTensor<T>,
    ) -> std::result::Result<Self, MatrixTensorConversionError>
    where
        T: Clone,
    {
        let (shape, data) = tensor.try_into_vec_col_major().map_err(|source| {
            MatrixTensorConversionError::HostBuffer {
                message: source.to_string(),
            }
        })?;
        if shape.len() != 2 {
            return Err(MatrixTensorConversionError::Rank { shape });
        }
        Ok(Self::from_col_major_vec(shape[0], shape[1], data))
    }

    fn offset(&self, row: usize, col: usize) -> usize {
        assert!(
            row < self.nrows,
            "matrix row index {row} out of bounds (bound: {})",
            self.nrows
        );
        assert!(
            col < self.ncols,
            "matrix column index {col} out of bounds (bound: {})",
            self.ncols
        );
        row + self.nrows * col
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

/// Compute the smallest eigenpair of a small Hermitian projected matrix.
///
/// This function validates that `matrix` is square, non-empty, and Hermitian
/// within `hermitian_tol`, symmetrizes accepted roundoff as `(A + A†) / 2`,
/// then calls tenferro's Hermitian eigendecomposition. It is intended for
/// Rayleigh-Ritz projected Krylov matrices, whose dimension is bounded by the
/// Krylov subspace size. It must not be used to materialize a full
/// tensor-network effective Hamiltonian.
///
/// # Arguments
/// * `matrix` - Small dense Hermitian matrix in column-major [`Matrix`] layout.
/// * `hermitian_tol` - Relative tolerance for `A[i, j] = conj(A[j, i])`,
///   applied as `hermitian_tol * max(1, |A[i,j]|, |A[j,i]|)`.
///   Typical values are `1e-12` for `f64`/`Complex64` projected matrices.
///
/// # Returns
/// The smallest real eigenvalue and the corresponding normalized eigenvector
/// coefficients.
///
/// # Errors
/// Returns [`HermitianEigenError`] if the matrix is empty, non-square,
/// non-Hermitian within `hermitian_tol`, or if the backend eigendecomposition
/// fails or returns an unexpected dtype/shape.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{lowest_hermitian_eigenpair, Matrix};
///
/// let matrix = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0]);
/// let pair = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap();
///
/// assert!((pair.eigenvalue - 1.0).abs() < 1.0e-12);
/// assert_eq!(pair.eigenvector.len(), 2);
/// ```
pub fn lowest_hermitian_eigenpair<T>(
    matrix: &Matrix<T>,
    hermitian_tol: f64,
) -> std::result::Result<HermitianEigenpair<T>, HermitianEigenError>
where
    T: HermitianEigenScalar,
{
    let decomp = hermitian_eigendecomposition(matrix, hermitian_tol)?;

    let (min_col, eigenvalue) = decomp
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .ok_or(HermitianEigenError::Empty)?;

    let n = decomp.eigenvalues.len();
    let vector_data = decomp.eigenvectors.as_col_major_slice();
    let start = n * min_col;
    let eigenvector = vector_data[start..start + n].to_vec();

    Ok(HermitianEigenpair {
        eigenvalue,
        eigenvector,
    })
}

/// Compute all eigenpairs of a small Hermitian projected matrix.
///
/// This validates Hermitian structure and symmetrizes accepted roundoff before
/// calling the backend. It is meant for bounded Krylov/Rayleigh-Ritz matrices,
/// not full tensor-network materialization.
///
/// # Arguments
///
/// * `matrix` - Square Hermitian matrix in column-major [`Matrix`] layout.
/// * `hermitian_tol` - Relative tolerance for checking `A = A†`, applied per
///   entry pair with scale `max(1, |A[i,j]|, |A[j,i]|)`.
///
/// # Returns
///
/// All real eigenvalues and all eigenvectors of `matrix`.
///
/// # Errors
///
/// Returns [`HermitianEigenError`] if `matrix` is not square, is not Hermitian
/// within `hermitian_tol`, or the backend eigensolver fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{hermitian_eigendecomposition, Matrix};
///
/// let matrix = Matrix::from_col_major_vec(2, 2, vec![3.0, 0.0, 0.0, 5.0]);
/// let decomp = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap();
/// assert_eq!(decomp.eigenvalues, vec![3.0, 5.0]);
/// assert_eq!(decomp.eigenvectors.as_col_major_slice().len(), 4);
/// ```
pub fn hermitian_eigendecomposition<T>(
    matrix: &Matrix<T>,
    hermitian_tol: f64,
) -> std::result::Result<HermitianEigendecomposition<T>, HermitianEigenError>
where
    T: HermitianEigenScalar,
{
    let matrix = validate_and_symmetrize_hermitian_matrix(matrix, hermitian_tol)?;

    let n = matrix.nrows();
    let input_tensor = T::into_tensor(vec![n, n], matrix.as_col_major_slice().to_vec());
    let eager_ctx = crate::default_eager_ctx().map_err(|source| HermitianEigenError::Backend {
        source: Box::new(source),
    })?;
    let input = EagerTensor::from_tensor_in(input_tensor, eager_ctx);
    let (values, vectors) = tenferro_linalg::eager_tensor::eigh(&input).map_err(|source| {
        HermitianEigenError::Backend {
            source: Box::new(source),
        }
    })?;

    let (values_shape, eigenvalues) =
        T::eigenvalues_from_tensor(values.data().clone(), hermitian_tol)?;
    ensure_eigh_shape("eigenvalues", &values_shape, &[n])?;
    let vectors = typed_eigh_output::<T>("eigenvectors", vectors.data().clone())?;
    ensure_eigh_shape("eigenvectors", vectors.shape(), &[n, n])?;

    Ok(HermitianEigendecomposition {
        eigenvalues,
        eigenvectors: Matrix::from_col_major_vec(n, n, vectors.as_slice().to_vec()),
    })
}

/// Compute the first column of `exp(exponent * A)` for a small Hermitian matrix.
///
/// Krylov exponential routines use this for the projected matrix action on the
/// first basis vector. The returned coefficients are complex even when `A` is
/// real because real-time evolution has complex phases.
///
/// # Arguments
///
/// * `matrix` - Square Hermitian matrix in column-major [`Matrix`] layout.
/// * `exponent` - Scalar multiplier in `exp(exponent * A)`.
/// * `hermitian_tol` - Relative tolerance for checking `A = A†`; accepted
///   roundoff is symmetrized before eigensolving.
///
/// # Returns
///
/// The first column of the matrix exponential.
///
/// # Errors
///
/// Returns [`HermitianEigenError`] if Hermitian validation or eigensolving
/// fails.
///
/// # Examples
///
/// ```
/// use num_complex::Complex64;
/// use tensor4all_tensorbackend::{hermitian_exponential_first_column, Matrix};
///
/// let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
/// let column = hermitian_exponential_first_column(
///     &matrix,
///     Complex64::new(0.0, -0.5),
///     1.0e-12,
/// ).unwrap();
/// let expected = Complex64::new(0.5_f64.cos(), -0.5_f64.sin());
/// assert!((column[0] - expected).norm() < 1.0e-12);
/// assert!(column[1].norm() < 1.0e-12);
/// ```
pub fn hermitian_exponential_first_column<T>(
    matrix: &Matrix<T>,
    exponent: Complex64,
    hermitian_tol: f64,
) -> std::result::Result<Vec<Complex64>, HermitianEigenError>
where
    T: HermitianEigenScalar,
{
    let decomp = hermitian_eigendecomposition(matrix, hermitian_tol)?;
    let n = decomp.eigenvalues.len();
    let vectors = decomp.eigenvectors.as_col_major_slice();
    let mut result = vec![Complex64::new(0.0, 0.0); n];

    for col in 0..n {
        let lambda = decomp.eigenvalues[col];
        let phase = (exponent * lambda).exp();
        let first_component = T::to_complex64(vectors[col * n]).conj();
        for row in 0..n {
            result[row] += T::to_complex64(vectors[row + col * n]) * phase * first_component;
        }
    }

    Ok(result)
}

fn validate_and_symmetrize_hermitian_matrix<T>(
    matrix: &Matrix<T>,
    hermitian_tol: f64,
) -> std::result::Result<Matrix<T>, HermitianEigenError>
where
    T: HermitianEigenScalar,
{
    if !hermitian_tol.is_finite() || hermitian_tol < 0.0 {
        return Err(HermitianEigenError::InvalidTolerance {
            tolerance: hermitian_tol,
        });
    }
    if matrix.nrows() != matrix.ncols() {
        return Err(HermitianEigenError::NonSquare {
            nrows: matrix.nrows(),
            ncols: matrix.ncols(),
        });
    }
    if matrix.nrows() == 0 {
        return Err(HermitianEigenError::Empty);
    }

    let n = matrix.nrows();
    let mut data = matrix.as_col_major_slice().to_vec();
    for col in 0..matrix.ncols() {
        for row in 0..=col {
            let row_col = matrix[[row, col]];
            let col_row = matrix[[col, row]];
            let difference = T::hermitian_difference(row_col, col_row);
            let tolerance = hermitian_tol * T::hermitian_scale(row_col, col_row);
            if difference > tolerance {
                return Err(HermitianEigenError::NonHermitian {
                    row,
                    col,
                    difference,
                    tolerance,
                });
            }
            let (row_col, col_row) = T::symmetrized_hermitian_pair(row_col, col_row);
            data[row + n * col] = row_col;
            data[col + n * row] = col_row;
        }
    }
    Ok(Matrix::from_col_major_vec(n, n, data))
}

fn typed_eigh_output<T>(
    output: &'static str,
    tensor: Tensor,
) -> std::result::Result<TypedTensor<T>, HermitianEigenError>
where
    T: TensorScalar,
{
    let actual = tensor.dtype();
    T::try_into_typed(tensor).ok_or_else(|| HermitianEigenError::DType {
        output,
        expected: format!("{:?}", T::dtype()),
        actual: format!("{actual:?}"),
    })
}

fn ensure_eigh_shape(
    output: &'static str,
    actual: &[usize],
    expected: &[usize],
) -> std::result::Result<(), HermitianEigenError> {
    if actual != expected {
        return Err(HermitianEigenError::Shape {
            output,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        });
    }
    Ok(())
}

impl<T: Clone> Matrix<T> {
    /// Create a new matrix filled with a constant value.
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::from_elem(2, 3, 7.0);
    /// assert_eq!(m[[0, 0]], 7.0);
    /// assert_eq!(m[[1, 2]], 7.0);
    /// ```
    pub fn from_elem(nrows: usize, ncols: usize, elem: T) -> Self {
        let len = checked_matrix_len(nrows, ncols);
        assert!(
            len.is_some(),
            "matrix shape product overflow: {nrows} rows * {ncols} columns"
        );
        let len = len.map_or(0, |len| len);
        Self {
            data: vec![elem; len],
            nrows,
            ncols,
        }
    }
}

impl<T: Clone + Zero> Matrix<T> {
    /// Create a zeros matrix
    ///
    /// # Panics
    ///
    /// Panics if `nrows * ncols` overflows.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::Matrix;
    ///
    /// let m = Matrix::<f64>::zeros(2, 3);
    /// assert_eq!(m.nrows(), 2);
    /// assert_eq!(m.ncols(), 3);
    /// assert_eq!(m[[0, 0]], 0.0);
    /// assert_eq!(m[[1, 2]], 0.0);
    /// ```
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        let len = checked_matrix_len(nrows, ncols);
        assert!(
            len.is_some(),
            "matrix shape product overflow: {nrows} rows * {ncols} columns"
        );
        let len = len.map_or(0, |len| len);
        Self {
            data: vec![T::zero(); len],
            nrows,
            ncols,
        }
    }
}

impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &Self::Output {
        &self.data[self.offset(idx[0], idx[1])]
    }
}

impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output {
        let offset = self.offset(idx[0], idx[1]);
        &mut self.data[offset]
    }
}

/// Create a matrix from a 2D vector, returning an error for ragged rows.
///
/// Each inner `Vec` is one row. The resulting matrix is stored internally in
/// column-major order.
///
/// # Errors
///
/// Returns [`MatrixShapeError::RaggedRows`] when any row has a different length
/// than the first row.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::try_from_vec2d;
///
/// let m = try_from_vec2d(vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
/// ])?;
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 2);
/// assert_eq!(m[[0, 1]], 2.0);
/// assert_eq!(m[[1, 0]], 3.0);
/// # Ok::<(), tensor4all_tensorbackend::MatrixShapeError>(())
/// ```
pub fn try_from_vec2d<T: Clone + Zero>(
    data: Vec<Vec<T>>,
) -> std::result::Result<Matrix<T>, MatrixShapeError> {
    let nrows = data.len();
    let ncols = data.first().map_or(0, Vec::len);
    for (row, values) in data.iter().enumerate() {
        let actual = values.len();
        if actual != ncols {
            return Err(MatrixShapeError::RaggedRows {
                row,
                expected: ncols,
                actual,
            });
        }
    }
    let mut m = Matrix::zeros(nrows, ncols);
    for i in 0..nrows {
        for j in 0..ncols {
            m[[i, j]] = data[i][j].clone();
        }
    }
    Ok(m)
}

/// Create a matrix from a rectangular 2D vector.
///
/// Each inner `Vec` is one row. The resulting matrix is stored internally in
/// column-major order.
///
/// # Panics
///
/// Panics if the row lengths are not all equal or if the rectangular shape's
/// element count overflows `usize`. Use [`try_from_vec2d`] when row-shaped
/// input comes from users, files, or other fallible boundaries to receive a
/// typed error for ragged rows.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::from_vec2d;
///
/// let m = from_vec2d(vec![
///     vec![1.0, 2.0],
///     vec![3.0, 4.0],
/// ]);
/// assert_eq!(m.nrows(), 2);
/// assert_eq!(m.ncols(), 2);
/// assert_eq!(m[[0, 1]], 2.0);
/// assert_eq!(m[[1, 0]], 3.0);
/// ```
pub fn from_vec2d<T: Clone + Zero>(data: Vec<Vec<T>>) -> Matrix<T> {
    let result = try_from_vec2d(data);
    let error_message = match &result {
        Ok(_) => String::new(),
        Err(error) => error.to_string(),
    };
    assert!(result.is_ok(), "{error_message}");
    match result {
        Ok(matrix) => matrix,
        Err(_) => Matrix {
            data: Vec::new(),
            nrows: 0,
            ncols: 0,
        },
    }
}

/// Get a submatrix by selecting specific rows and columns.
///
/// # Panics
///
/// Panics if any row is not less than `m.nrows()` or any column is not less
/// than `m.ncols()`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, submatrix};
///
/// let m = from_vec2d(vec![
///     vec![1.0, 2.0, 3.0],
///     vec![4.0, 5.0, 6.0],
///     vec![7.0, 8.0, 9.0],
/// ]);
/// let sub = submatrix(&m, &[0, 2], &[1, 2]);
/// assert_eq!(sub.nrows(), 2);
/// assert_eq!(sub.ncols(), 2);
/// assert_eq!(sub[[0, 0]], 2.0); // m[0, 1]
/// assert_eq!(sub[[1, 1]], 9.0); // m[2, 2]
/// ```
pub fn submatrix<T: Clone + Zero>(m: &Matrix<T>, rows: &[usize], cols: &[usize]) -> Matrix<T> {
    assert!(
        rows.iter().all(|&row| row < m.nrows),
        "submatrix row index out of bounds"
    );
    assert!(
        cols.iter().all(|&col| col < m.ncols),
        "submatrix column index out of bounds"
    );

    let mut data = Vec::with_capacity(rows.len() * cols.len());
    let source = m.as_col_major_slice();
    for &col in cols {
        let col_start = col * m.nrows;
        for &row in rows {
            let offset = col_start + row;
            // SAFETY: rows and cols are range-checked above, and Matrix stores
            // exactly nrows * ncols values in column-major order.
            data.push(unsafe { source.get_unchecked(offset).clone() });
        }
    }
    Matrix::from_col_major_vec(rows.len(), cols.len(), data)
}

/// Swap two rows in a matrix in-place.
///
/// No-op if `a == b`.
///
/// # Panics
///
/// Panics if `a` or `b` is not less than `m.nrows()` when they differ.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, swap_rows};
///
/// let mut m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// swap_rows(&mut m, 0, 1);
/// assert_eq!(m[[0, 0]], 3.0);
/// assert_eq!(m[[1, 0]], 1.0);
/// ```
pub fn swap_rows<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    assert!(a < m.nrows, "row index out of bounds");
    assert!(b < m.nrows, "row index out of bounds");
    let nrows = m.nrows;
    let ncols = m.ncols;
    let data = m.as_col_major_mut_slice();
    for j in 0..ncols {
        data.swap(a + nrows * j, b + nrows * j);
    }
}

/// Swap two columns in a matrix in-place.
///
/// No-op if `a == b`.
///
/// # Panics
///
/// Panics if `a` or `b` is not less than `m.ncols()` when they differ.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, swap_cols};
///
/// let mut m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
/// swap_cols(&mut m, 0, 1);
/// assert_eq!(m[[0, 0]], 2.0);
/// assert_eq!(m[[0, 1]], 1.0);
/// ```
pub fn swap_cols<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    assert!(a < m.ncols, "column index out of bounds");
    assert!(b < m.ncols, "column index out of bounds");
    let nrows = m.nrows;
    let start_a = nrows * a;
    let start_b = nrows * b;
    let data = m.as_col_major_mut_slice();
    if start_a < start_b {
        let (left, right) = data.split_at_mut(start_b);
        left[start_a..start_a + nrows].swap_with_slice(&mut right[..nrows]);
    } else {
        let (left, right) = data.split_at_mut(start_a);
        right[..nrows].swap_with_slice(&mut left[start_b..start_b + nrows]);
    }
}

/// Transpose the matrix.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, transpose};
///
/// let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
/// let mt = transpose(&m);
/// assert_eq!(mt.nrows(), 3);
/// assert_eq!(mt.ncols(), 2);
/// assert_eq!(mt[[0, 0]], 1.0);
/// assert_eq!(mt[[2, 1]], 6.0);
/// ```
pub fn transpose<T: Clone + Zero>(m: &Matrix<T>) -> Matrix<T> {
    let mut result = Matrix::zeros(m.ncols, m.nrows);
    for j in 0..m.ncols {
        for i in 0..m.nrows {
            result[[j, i]] = m[[i, j]].clone();
        }
    }
    result
}

/// Find the position and value of the maximum absolute value in a submatrix.
///
/// Searches within the rectangular region defined by `rows x cols` ranges.
/// Returns `(row, col, value)` of the element with the largest `|value|^2`.
///
/// # Panics
///
/// Panics if either range is empty or either end is out of bounds (`rows.end > a.nrows()` or `cols.end > a.ncols()`).
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, submatrix_argmax};
///
/// let m = from_vec2d(vec![
///     vec![1.0_f64, 2.0, 3.0],
///     vec![4.0, 9.0, 6.0],
///     vec![7.0, 8.0, 5.0],
/// ]);
/// let (row, col, val) = submatrix_argmax(&m, 0..3, 0..3);
/// assert_eq!(row, 1);
/// assert_eq!(col, 1);
/// assert_eq!(val, 9.0);
/// ```
pub fn submatrix_argmax<T: MatrixScalar>(
    a: &Matrix<T>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> (usize, usize, T) {
    assert!(!rows.is_empty(), "rows must not be empty");
    assert!(!cols.is_empty(), "cols must not be empty");
    assert!(rows.end <= a.nrows, "row range out of bounds");
    assert!(cols.end <= a.ncols, "column range out of bounds");

    let data = a.as_col_major_slice();
    let first_offset = rows.start + a.nrows * cols.start;
    // SAFETY: the non-empty ranges are checked against the matrix shape above.
    let first = unsafe { *data.get_unchecked(first_offset) };
    let mut max_val: f64 = first.matrix_abs_sq();
    let mut max_row = rows.start;
    let mut max_col = cols.start;
    let row_start = rows.start;
    let row_end = rows.end;
    let col_start = cols.start;
    let col_end = cols.end;

    for c in col_start..col_end {
        let col_start_offset = row_start + a.nrows * c;
        for (offset, r) in (col_start_offset..).zip(row_start..row_end) {
            // SAFETY: row and column loops stay within the checked ranges.
            let value = unsafe { *data.get_unchecked(offset) };
            let val: f64 = value.matrix_abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    let max_offset = max_row + a.nrows * max_col;
    // SAFETY: max_row/max_col were selected from the checked ranges.
    (max_row, max_col, unsafe { *data.get_unchecked(max_offset) })
}

/// BLAS-backed matrix multiplication dispatch.
///
/// Implemented for all scalar types supported by tenferro einsum
/// (f64, f32, Complex64, Complex32). This trait is sealed — external
/// types cannot implement it.
pub trait BlasMul: Sized {
    #[doc(hidden)]
    fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>>;

    #[doc(hidden)]
    fn blas_mat_mul_owned(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>>;
}

fn dot_general_matrices<T>(
    a_tensor: Tensor,
    b_tensor: Tensor,
    m: usize,
    n: usize,
    expected_len: usize,
) -> Result<Matrix<T>>
where
    T: TensorScalar,
{
    use crate::with_default_backend;
    use tenferro::DotGeneralConfig;

    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![],
        rhs_batch_dims: vec![],
    };
    let c = with_default_backend(|backend| {
        backend.with_backend_session(|exec| exec.dot_general(&a_tensor, &b_tensor, &config))
    })
    .context("matrix multiplication failed")?;
    let c = T::try_into_typed(c)
        .ok_or_else(|| anyhow::anyhow!("matrix multiplication returned wrong dtype"))?;
    let result = Matrix::try_from_typed_tensor(c)?;
    ensure!(
        result.nrows() == m && result.ncols() == n,
        "matrix multiplication returned shape {}x{} for expected shape {}x{}",
        result.nrows(),
        result.ncols(),
        m,
        n
    );
    ensure!(
        result.as_col_major_slice().len() == expected_len,
        "matrix multiplication returned {} values for expected shape {}x{}",
        result.as_col_major_slice().len(),
        m,
        n
    );
    Ok(result)
}

macro_rules! impl_blas_mul {
    ($($t:ty),*) => {
        $(
        impl BlasMul for $t {
            fn blas_mat_mul(a: &Matrix<Self>, b: &Matrix<Self>) -> Result<Matrix<Self>> {
                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                ensure!(
                    b.nrows() == k,
                    "matrix dimensions must agree for multiplication: left is {}x{}, right is {}x{}",
                    m,
                    k,
                    b.nrows(),
                    n
                );
                // Reject an overflowing output element count before any tensor
                // conversion or backend call, matching the constructor contract.
                let expected_len = m.checked_mul(n).ok_or_else(|| {
                    anyhow::anyhow!(
                        "matrix multiplication output shape {m}x{n} overflows usize"
                    )
                })?;

                let a_tensor: Tensor = a.to_typed_tensor().into();
                let b_tensor: Tensor = b.to_typed_tensor().into();
                dot_general_matrices::<$t>(a_tensor, b_tensor, m, n, expected_len)
            }

            fn blas_mat_mul_owned(a: Matrix<Self>, b: Matrix<Self>) -> Result<Matrix<Self>> {
                let m = a.nrows();
                let k = a.ncols();
                let n = b.ncols();
                ensure!(
                    b.nrows() == k,
                    "matrix dimensions must agree for multiplication: left is {}x{}, right is {}x{}",
                    m,
                    k,
                    b.nrows(),
                    n
                );
                let expected_len = m.checked_mul(n).ok_or_else(|| {
                    anyhow::anyhow!(
                        "matrix multiplication output shape {m}x{n} overflows usize"
                    )
                })?;

                let a_tensor: Tensor = a.into_typed_tensor().into();
                let b_tensor: Tensor = b.into_typed_tensor().into();
                dot_general_matrices::<$t>(a_tensor, b_tensor, m, n, expected_len)
            }
        }
        )*
    };
}

impl_blas_mul!(f64, f32, num_complex::Complex64, num_complex::Complex32);

/// Scalar bound for dense backend matrix utilities.
///
/// This is the storage/linalg-layer scalar trait. Higher-level crates may
/// extend it with domain-specific methods, but matrix utilities only rely on
/// these algebraic operations and absolute-value comparisons.
pub trait MatrixScalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + BlasMul
    + 'static
{
    /// Squared absolute value as `f64`.
    fn matrix_abs_sq(self) -> f64;
}

impl MatrixScalar for f64 {
    fn matrix_abs_sq(self) -> f64 {
        self * self
    }
}

impl MatrixScalar for f32 {
    fn matrix_abs_sq(self) -> f64 {
        (self * self) as f64
    }
}

impl MatrixScalar for Complex64 {
    fn matrix_abs_sq(self) -> f64 {
        self.norm_sqr()
    }
}

impl MatrixScalar for Complex32 {
    fn matrix_abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }
}

/// Matrix multiplication: A * B.
///
/// Uses BLAS-backed einsum via tenferro for high performance.
///
/// # Errors
///
/// Returns an error if `a.ncols() != b.nrows()` or the backend einsum fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul(&a, &b).unwrap();
/// assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
/// assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
/// assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
/// assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
/// ```
pub fn mat_mul<T: BlasMul>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    T::blas_mat_mul(a, b)
}

/// Matrix multiplication: consume `A` and `B`, returning `A * B`.
///
/// Uses BLAS-backed einsum via tenferro. Compared with [`mat_mul`], this
/// reuses the input matrix buffers when building tenferro tensors.
///
/// # Errors
///
/// Returns an error if `a.ncols() != b.nrows()` or the backend einsum fails.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::{from_vec2d, mat_mul_owned};
///
/// let a = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
/// let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
/// let c = mat_mul_owned(a, b).unwrap();
/// assert_eq!(c.as_col_major_slice(), &[19.0, 43.0, 22.0, 50.0]);
/// ```
pub fn mat_mul_owned<T: BlasMul>(a: Matrix<T>, b: Matrix<T>) -> Result<Matrix<T>> {
    T::blas_mat_mul_owned(a, b)
}

/// Batched matrix multiplication for column-major matrices with one shared shape.
///
/// Computes `C[p] = A[p] * B[p]` for `batch` matrices. Each `A[p]` is an
/// `m x k` column-major matrix and each `B[p]` is a `k x n` column-major
/// matrix. The input buffers store complete matrices consecutively, and the
/// returned buffer stores `batch` consecutive `m x n` column-major outputs.
///
/// # Errors
///
/// Returns an error if the input buffer lengths do not match the declared
/// shapes or if the backend rejects the batched GEMM.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::batched_mat_mul_same_shape;
///
/// let a = vec![1.0_f64, 3.0, 2.0, 4.0];
/// let b = vec![5.0_f64, 7.0, 6.0, 8.0];
/// let out = batched_mat_mul_same_shape(1, 2, 2, 2, &a, &b).unwrap();
/// assert_eq!(out, vec![19.0, 43.0, 22.0, 50.0]);
/// ```
pub fn batched_mat_mul_same_shape<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &[T],
    b: &[T],
) -> Result<Vec<T>>
where
    T: tenferro::TensorScalar + Copy,
{
    batched_mat_mul_same_shape_owned(batch, m, k, n, a.to_vec(), b.to_vec())
}

/// Batched matrix multiplication while consuming column-major input buffers.
///
/// This is the owned-buffer counterpart of [`batched_mat_mul_same_shape`].
/// It avoids cloning the two input batches when callers have just built the
/// contiguous buffers for a backend call.
///
/// # Errors
///
/// Returns an error if the input buffer lengths do not match the declared
/// shapes or if the backend rejects the batched GEMM.
pub fn batched_mat_mul_same_shape_owned<T>(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a: Vec<T>,
    b: Vec<T>,
) -> Result<Vec<T>>
where
    T: tenferro::TensorScalar + Copy,
{
    use crate::with_default_backend;
    use tenferro::DotGeneralConfig;

    validate_batched_mat_mul_inputs(batch, m, k, n, a.len(), b.len())?;

    let a_tensor = T::into_tensor(vec![m, k, batch], a);
    let b_tensor = T::into_tensor(vec![k, n, batch], b);
    let config = DotGeneralConfig {
        lhs_contracting_dims: vec![1],
        rhs_contracting_dims: vec![0],
        lhs_batch_dims: vec![2],
        rhs_batch_dims: vec![2],
    };
    let c = with_default_backend(|backend| {
        backend.with_backend_session(|exec| exec.dot_general(&a_tensor, &b_tensor, &config))
    })
    .context("batched matrix multiplication failed")?;
    let c = T::try_into_typed(c)
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication returned wrong dtype"))?;
    let (_shape, data) = c.try_into_vec_col_major()?;
    let expected_len = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication output shape overflows"))?;
    ensure!(
        data.len() == expected_len,
        "batched matrix multiplication returned {} values for expected shape {}x{}x{}",
        data.len(),
        m,
        n,
        batch
    );
    Ok(data)
}

fn validate_batched_mat_mul_inputs(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    a_len: usize,
    b_len: usize,
) -> Result<()> {
    let expected_a_len = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(k))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication left shape overflows"))?;
    let expected_b_len = batch
        .checked_mul(k)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| anyhow::anyhow!("batched matrix multiplication right shape overflows"))?;
    ensure!(
        a_len == expected_a_len,
        "batched matrix multiplication left buffer has length {}, expected {}",
        a_len,
        expected_a_len
    );
    ensure!(
        b_len == expected_b_len,
        "batched matrix multiplication right buffer has length {}, expected {}",
        b_len,
        expected_b_len
    );
    Ok(())
}

#[cfg(test)]
mod tests;
