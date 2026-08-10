//! Factorization methods for MPO tensors
//!
//! This module provides various factorization methods (SVD, RSVD, LU, CI)
//! for compressing and reshaping MPO tensors.

use super::error::{MPOError, Result};
use super::Matrix2;
use crate::einsum_helper::tensor_to_col_major_vec;
use num_complex::{Complex64, ComplexFloat};
use tenferro_tensor::{TensorScalar, TypedTensor};
use tensor4all_tensorbackend::{svd_backend, BackendLinalgScalar};

/// Factorization method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FactorizeMethod {
    /// Singular Value Decomposition
    #[default]
    SVD,
    /// Randomized SVD (faster for large matrices)
    RSVD,
    /// LU decomposition with rank-revealing pivoting
    LU,
    /// Cross Interpolation
    CI,
}

/// Options for factorization
#[derive(Debug, Clone)]
pub struct FactorizeOptions {
    /// Factorization method to use
    pub method: FactorizeMethod,
    /// Relative truncation tolerance.
    ///
    /// Singular values (or pivots) smaller than `tolerance * sigma_max` are
    /// discarded, where `sigma_max` is the largest singular value (or pivot)
    /// of the matrix being factorized. The threshold is therefore invariant
    /// under a global rescaling of the input, matching
    /// [`crate::CompressionOptions::tolerance`] and
    /// `tensor4all_core::SvdTruncationPolicy::new`.
    ///
    /// Smaller values preserve more accuracy but produce larger ranks.
    pub tolerance: f64,
    /// Maximum rank (bond dimension) after factorization
    pub max_rank: usize,
    /// Whether to return the left factor as left-orthogonal
    pub left_orthogonal: bool,
    /// Number of random projections for RSVD (parameter q)
    pub rsvd_q: usize,
    /// Oversampling parameter for RSVD (parameter p)
    pub rsvd_p: usize,
}

impl Default for FactorizeOptions {
    fn default() -> Self {
        Self {
            method: FactorizeMethod::SVD,
            tolerance: 1e-12,
            max_rank: usize::MAX,
            left_orthogonal: true,
            rsvd_q: 2,
            rsvd_p: 10,
        }
    }
}

/// Result of factorization
#[derive(Debug, Clone)]
pub struct FactorizeResult<T: TensorScalar> {
    /// Left factor matrix (m x rank)
    pub left: Matrix2<T>,
    /// Right factor matrix (rank x n)
    pub right: Matrix2<T>,
    /// New rank (number of columns in left / rows in right)
    pub rank: usize,
    /// Discarded weight (for error estimation)
    pub discarded: f64,
}

/// Trait bounds for SVD-compatible scalars
pub trait SVDScalar:
    crate::traits::TTScalar + ComplexFloat + Default + Copy + BackendLinalgScalar + 'static
{
    /// Convert a backend singular value into `f64` for truncation logic.
    fn linalg_real_to_f64(real: <Self as TensorScalar>::Real) -> f64;
    /// Promote a backend singular value into the matrix scalar type.
    fn from_linalg_real(real: <Self as TensorScalar>::Real) -> Self;
}

impl SVDScalar for f64 {
    fn linalg_real_to_f64(real: <Self as TensorScalar>::Real) -> f64 {
        real
    }

    fn from_linalg_real(real: <Self as TensorScalar>::Real) -> Self {
        real
    }
}

impl SVDScalar for Complex64 {
    fn linalg_real_to_f64(real: <Self as TensorScalar>::Real) -> f64 {
        real
    }

    fn from_linalg_real(real: <Self as TensorScalar>::Real) -> Self {
        Complex64::new(real, 0.0)
    }
}

/// Factorize a matrix into left and right factors
///
/// Returns (L, R, rank, discarded) where:
/// - L: left factor matrix (rows x rank)
/// - R: right factor matrix (rank x cols)
/// - rank: the resulting rank after truncation
/// - discarded: the discarded weight (for error estimation)
///
/// The original matrix M ≈ L @ R
///
/// Note: Only SVD method is fully supported. LU and CI require additional
/// traits and should use `factorize_lu` directly.
/// # Errors
///
/// Returns an error when the factorization fails (a non-convergence, singular,
/// /// or backend failure).
///
pub fn factorize<T: SVDScalar>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>> {
    match options.method {
        FactorizeMethod::SVD => factorize_svd(matrix, options),
        FactorizeMethod::RSVD => factorize_rsvd(matrix, options),
        FactorizeMethod::LU | FactorizeMethod::CI => {
            // For LU/CI, fall back to SVD for now
            // Full LU/CI support requires tensor4all_tcicore::Scalar trait
            factorize_svd(matrix, options)
        }
    }
}

// Use the shared matrix2_zeros from the parent module
use super::matrix2_zeros;

fn matrix2_to_typed_tensor<T>(matrix: &Matrix2<T>) -> TypedTensor<T>
where
    T: TensorScalar,
{
    matrix.as_inner().clone()
}

fn typed_tensor_to_matrix2<T>(tensor: &TypedTensor<T>, op: &'static str) -> Result<Matrix2<T>>
where
    T: crate::traits::TTScalar + Default,
{
    if tensor.shape().len() != 2 {
        return Err(MPOError::FactorizationError {
            message: format!(
                "{op} returned rank-{} tensor, expected matrix",
                tensor.shape().len()
            ),
        });
    }

    Ok(Matrix2::from_tenferro_unchecked(tensor.clone()))
}

fn typed_col_major_values<T>(tensor: &TypedTensor<T>, op: &'static str) -> Result<Vec<T>>
where
    T: TensorScalar,
{
    let _ = op;
    Ok(tensor_to_col_major_vec(tensor))
}

/// Factorize using SVD
fn factorize_svd<T: SVDScalar>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>> {
    let m = matrix.dim(0);
    let n = matrix.dim(1);

    if m == 0 || n == 0 {
        return Err(MPOError::FactorizationError {
            message: "Cannot factorize empty matrix".to_string(),
        });
    }

    // Compute SVD using tensorbackend (tenferro-backed implementation)
    let a_tensor = matrix2_to_typed_tensor(matrix);
    let svd_result = svd_backend(&a_tensor).map_err(|e| MPOError::FactorizationError {
        message: format!("SVD computation failed: {:?}", e),
    })?;

    let u = typed_tensor_to_matrix2(&svd_result.u, "svd.u")?;
    let vt = typed_tensor_to_matrix2(&svd_result.vt, "svd.vt")?;
    let singular_values = typed_col_major_values(&svd_result.s, "svd.s")?;

    // Determine rank based on tolerance and max_rank
    let min_dim = m.min(n);
    let mut rank = 0;
    let mut total_weight: f64 = 0.0;

    // Sum all squared singular values for total weight
    // Singular values are stored in first row: s[[0, i]] (LAPACK-style convention)
    for &singular_value in singular_values.iter().take(min_dim) {
        let sv = T::linalg_real_to_f64(singular_value);
        total_weight += sv * sv;
    }

    // Truncation is relative to the largest singular value, so the retained
    // rank does not change when the input matrix is globally rescaled. An
    // absolute cutoff would keep near-noise singular values for large-norm
    // inputs and over-truncate small-norm ones.
    // `svd_backend` returns singular values in non-increasing order, so the
    // first entry is the largest; fall back to a scan to stay robust.
    let s_max = singular_values
        .iter()
        .take(min_dim)
        .map(|&sv| T::linalg_real_to_f64(sv))
        .fold(0.0_f64, f64::max);

    // Find rank by keeping singular values above the relative cutoff.
    // A zero matrix (or an empty spectrum) has no meaningful reference scale;
    // it falls through to the rank-1 floor below.
    let mut kept_weight: f64 = 0.0;
    if s_max > 0.0 {
        let cutoff = options.tolerance * s_max;
        for &singular_value in singular_values.iter().take(min_dim) {
            if rank >= options.max_rank {
                break;
            }
            let sv = T::linalg_real_to_f64(singular_value);
            if sv < cutoff {
                break;
            }
            kept_weight += sv * sv;
            rank += 1;
        }
    }

    // Ensure at least rank 1
    rank = rank.max(1);

    // Calculate discarded weight
    let discarded: f64 = if total_weight > 0.0 {
        1.0 - kept_weight / total_weight
    } else {
        0.0
    };

    // Build result matrices
    let mut left: Matrix2<T> = matrix2_zeros(m, rank);
    let mut right: Matrix2<T> = matrix2_zeros(rank, n);

    if options.left_orthogonal {
        // Left = U[:, :rank], Right = diag(S[:rank]) * Vt[:rank, :]
        //
        // `svd_backend` returns `vt` in backend convention
        // (V^T for real and V^H for complex), which is used directly here.
        for i in 0..m {
            for j in 0..rank {
                left[[i, j]] = u[[i, j]];
            }
        }
        for i in 0..rank {
            // Singular values are stored in first row: s[[0, i]] (LAPACK-style convention)
            let sv = T::from_linalg_real(singular_values[i]);
            for j in 0..n {
                right[[i, j]] = sv * vt[[i, j]];
            }
        }
    } else {
        // Left = U[:, :rank] * diag(S[:rank]), Right = Vt[:rank, :]
        for i in 0..m {
            for j in 0..rank {
                // Singular values are stored in first row: s[[0, j]] (LAPACK-style convention)
                let sv = T::from_linalg_real(singular_values[j]);
                left[[i, j]] = u[[i, j]] * sv;
            }
        }
        for i in 0..rank {
            for j in 0..n {
                right[[i, j]] = vt[[i, j]];
            }
        }
    }

    Ok(FactorizeResult {
        left,
        right,
        rank,
        discarded,
    })
}

/// Factorize using randomized SVD
fn factorize_rsvd<T: SVDScalar>(
    _matrix: &Matrix2<T>,
    _options: &FactorizeOptions,
) -> Result<FactorizeResult<T>> {
    // TODO: Implement RSVD-based factorization
    Err(MPOError::FactorizationError {
        message: "RSVD factorization not yet implemented".to_string(),
    })
}

/// Factorize using LU decomposition
///
/// This function requires the tensor4all_tcicore::Scalar trait.
/// Use this directly when you need LU-based factorization.
///
/// [`FactorizeOptions::tolerance`] is passed as `RrLUOptions::rel_tol` with
/// `abs_tol` set to zero, so the cutoff is relative to the largest pivot,
/// matching the SVD path.
/// # Errors
///
/// Returns an error when the LU factorization fails (a singular or backend
/// /// failure).
///
pub fn factorize_lu<T>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    T: SVDScalar + tensor4all_tcicore::Scalar + tensor4all_tcicore::MatrixLuciScalar,
{
    use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI, RrLUOptions};
    use tensor4all_tensorbackend::Matrix;

    let m = matrix.dim(0);
    let n = matrix.dim(1);

    // Convert Matrix2 to the shared backend Matrix for LU/CI factorization.
    let mut mat_ci = Matrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            mat_ci[[i, j]] = matrix[[i, j]];
        }
    }

    let lu_options = RrLUOptions {
        max_rank: options.max_rank,
        rel_tol: options.tolerance,
        abs_tol: 0.0,
        left_orthogonal: options.left_orthogonal,
    };

    let luci = MatrixLUCI::from_matrix(&mat_ci, Some(lu_options))?;
    let left_ci = luci.left();
    let right_ci = luci.right();
    let rank = luci.rank().max(1);

    // Convert back to Matrix2
    let left_m = left_ci.nrows();
    let left_n = left_ci.ncols();
    let mut left: Matrix2<T> = matrix2_zeros(left_m, left_n);
    for i in 0..left_m {
        for j in 0..left_n {
            left[[i, j]] = left_ci[[i, j]];
        }
    }

    let right_m = right_ci.nrows();
    let right_n = right_ci.ncols();
    let mut right: Matrix2<T> = matrix2_zeros(right_m, right_n);
    for i in 0..right_m {
        for j in 0..right_n {
            right[[i, j]] = right_ci[[i, j]];
        }
    }

    Ok(FactorizeResult {
        left,
        right,
        rank,
        discarded: 0.0,
    })
}

/// Factorize using Cross Interpolation
///
/// This function requires the tensor4all_tcicore::Scalar trait.
/// Use this directly when you need CI-based factorization.
/// # Errors
///
/// Returns an error when the cross-interpolation factorization fails (a
/// /// non-convergence or backend failure).
///
pub fn factorize_ci<T>(
    matrix: &Matrix2<T>,
    options: &FactorizeOptions,
) -> Result<FactorizeResult<T>>
where
    T: SVDScalar + tensor4all_tcicore::Scalar + tensor4all_tcicore::MatrixLuciScalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    // CI uses the same LUCI implementation as LU
    factorize_lu(matrix, options)
}

#[cfg(test)]
mod tests;
