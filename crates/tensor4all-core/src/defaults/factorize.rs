//! Unified tensor factorization module.
//!
//! This module provides a unified `factorize()` function that dispatches to
//! SVD, QR, LU, or CI (Cross Interpolation) algorithms based on options.
//!
//! # Note
//!
//! This module works with concrete types (`DynIndex`, `TensorDynLen`) only.
//! Generic tensor types are not supported.
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{factorize, Canonical, DynIndex, FactorizeOptions, TensorDynLen, TensorLike};
//!
//! # fn main() -> anyhow::Result<()> {
//! let i = DynIndex::new_dyn(2);
//! let j = DynIndex::new_dyn(2);
//! let tensor = TensorDynLen::from_dense(
//!     vec![i.clone(), j.clone()],
//!     vec![1.0, 0.0, 0.0, 1.0],
//! )?;
//! let result = factorize(
//!     &tensor,
//!     std::slice::from_ref(&i),
//!     &FactorizeOptions::svd().with_canonical(Canonical::Left),
//! )?;
//!
//! assert_eq!(result.rank, 2);
//! assert_eq!(result.left.dims(), vec![2, 2]);
//! # Ok(())
//! # }
//! ```

use crate::defaults::tensordynlen::unfold_split_inner;
use crate::defaults::DynIndex;
use crate::{contract_pair, unfold_split, TensorDynLen};
use anyhow::Result as AnyhowResult;
use num_complex::{Complex64, ComplexFloat};
use tenferro_ad::EagerTensor;
use tenferro_linalg::eager_tensor::full_piv_lu_solve as eager_full_piv_lu_solve;
use tensor4all_tcicore::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};
use tensor4all_tensorbackend::{Matrix, TensorElement};

use crate::defaults::svd::svd_for_factorize;
use crate::qr::{qr_with, QrOptions};
use crate::svd::SvdOptions;

// Re-export types from tensor_like for backwards compatibility
pub use crate::tensor_like::{
    Canonical, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
};

/// Factorize a tensor into left and right factors.
///
/// This function dispatches to the appropriate algorithm based on `options.alg`:
/// - `SVD`: Singular Value Decomposition
/// - `QR`: QR decomposition
/// - `LU`: Rank-revealing LU decomposition
/// - `CI`: Cross Interpolation
///
/// The `canonical` option controls which factor is "canonical":
/// - `Canonical::Left`: Left factor is orthogonal (SVD/QR) or unit-diagonal (LU/CI)
/// - `Canonical::Right`: Right factor is orthogonal (SVD) or unit-diagonal (LU/CI)
///
/// # Arguments
/// * `t` - Input tensor
/// * `left_inds` - Indices to place on the left side
/// * `options` - Factorization options
///
/// # Returns
/// A `FactorizeResult` containing the left and right factors, bond index,
/// singular values (for SVD), and rank.
///
/// # Errors
/// Returns `FactorizeError` if:
/// - The storage type is not supported (only DenseF64 and DenseC64)
/// - QR is used with `Canonical::Right`
/// - The underlying algorithm fails
pub fn factorize(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    options.validate()?;

    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }

    if t.is_f64() {
        factorize_impl_f64(t, left_inds, options)
    } else if t.is_complex() {
        factorize_impl_c64(t, left_inds, options)
    } else {
        Err(FactorizeError::UnsupportedStorage(
            "factorize currently supports only f64 and Complex64 tensors",
        ))
    }
}

/// Factorize a tensor without applying algorithm-specific truncation options.
///
/// This path is intended for canonicalization and other exact tensor-network
/// rewrites where the decomposition must preserve the represented tensor rather
/// than obey global SVD/QR/LU rank-dropping defaults.
///
/// # Arguments
/// * `t` - Input tensor.
/// * `left_inds` - Indices to place on the left side.
/// * `alg` - Decomposition algorithm to use.
/// * `canonical` - Which factor should carry the canonical form.
///
/// # Returns
/// A factorization whose contracted factors reconstruct `t` up to numerical
/// roundoff, with no tolerance-based or maximum-rank truncation applied.
///
/// # Errors
/// Returns [`FactorizeError`] if the storage type is unsupported, the canonical
/// direction is unsupported for the selected algorithm, or the underlying
/// decomposition fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize_full_rank, Canonical, DynIndex, FactorizeAlg, TensorContractionLike, TensorDynLen,
/// };
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let tensor = TensorDynLen::from_dense(
///     vec![i.clone(), j.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0e-16],
/// )?;
///
/// let result = factorize_full_rank(
///     &tensor,
///     std::slice::from_ref(&i),
///     FactorizeAlg::QR,
///     Canonical::Left,
/// )?;
/// let reconstructed = result.left.contract_pair(&result.right).unwrap();
/// assert!(tensor.sub(&reconstructed)?.maxabs()? < 1.0e-18);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn factorize_full_rank(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }

    if t.is_f64() {
        factorize_impl_f64_full_rank(t, left_inds, alg, canonical)
    } else if t.is_complex() {
        factorize_impl_c64_full_rank(t, left_inds, alg, canonical)
    } else {
        Err(FactorizeError::UnsupportedStorage(
            "factorize currently supports only f64 and Complex64 tensors",
        ))
    }
}

fn factorize_impl_f64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<f64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<f64>(t, left_inds, options),
    }
}

fn factorize_impl_f64_full_rank(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match alg {
        FactorizeAlg::SVD => factorize_svd_full_rank(t, left_inds, canonical),
        FactorizeAlg::QR => factorize_qr_full_rank(t, left_inds, canonical),
        FactorizeAlg::LU => factorize_lu_full_rank::<f64>(t, left_inds, canonical),
        FactorizeAlg::CI => factorize_ci_full_rank::<f64>(t, left_inds, canonical),
    }
}

fn factorize_impl_c64(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<Complex64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<Complex64>(t, left_inds, options),
    }
}

fn factorize_impl_c64_full_rank(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    match alg {
        FactorizeAlg::SVD => factorize_svd_full_rank(t, left_inds, canonical),
        FactorizeAlg::QR => factorize_qr_full_rank(t, left_inds, canonical),
        FactorizeAlg::LU => factorize_lu_full_rank::<Complex64>(t, left_inds, canonical),
        FactorizeAlg::CI => factorize_ci_full_rank::<Complex64>(t, left_inds, canonical),
    }
}

/// SVD factorization implementation.
fn factorize_svd(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    let mut svd_options = SvdOptions::new();
    if let Some(policy) = options.svd_policy {
        svd_options = svd_options.with_policy(policy);
    }
    if let Some(max_rank) = options.max_rank {
        svd_options = svd_options.with_max_rank(max_rank);
    }

    factorize_svd_with_options(t, left_inds, options.canonical, &svd_options)
}

fn factorize_svd_full_rank(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    let svd_options = SvdOptions::full_rank();
    factorize_svd_with_options(t, left_inds, canonical, &svd_options)
}

fn factorize_svd_with_options(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
    svd_options: &SvdOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    let result = svd_for_factorize(t, left_inds, svd_options)?;
    let u = result.u;
    let s = result.s;
    let vh = result.vh;
    let bond_index = result.bond_index;
    let singular_values = result.singular_values;
    let rank = result.rank;
    let sim_bond_index = s.indices[1].clone();

    match canonical {
        Canonical::Left => {
            // L = U (orthogonal), R = S * V^H
            let right_contracted = contract_pair(&s, &vh)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            let right = right_contracted.replaceind(&sim_bond_index, &bond_index)?;
            Ok(FactorizeResult {
                left: u,
                right,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
        Canonical::Right => {
            // L = U * S, R = V^H
            let left_contracted = contract_pair(&u, &s)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            let left = left_contracted.replaceind(&sim_bond_index, &bond_index)?;
            Ok(FactorizeResult {
                left,
                right: vh,
                bond_index,
                singular_values: Some(singular_values),
                rank,
            })
        }
    }
}

/// QR factorization implementation.
fn factorize_qr(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    if options.canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    let qr_options = if let Some(rtol) = options.qr_rtol {
        QrOptions::new().with_rtol(rtol)
    } else {
        QrOptions::new()
    };

    factorize_qr_with_options(t, left_inds, &qr_options)
}

fn factorize_qr_full_rank(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    if canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    factorize_qr_with_options(t, left_inds, &QrOptions::full_rank())
}

fn factorize_qr_with_options(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    qr_options: &QrOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError> {
    let (q, r) = qr_with::<f64>(t, left_inds, qr_options)?;

    // Get bond index from Q tensor (last index)
    let bond_index = q
        .indices
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("QR factorization returned a rank-0 Q tensor"))?;
    // Rank is the last dimension of Q
    let q_dims = q.dims();
    let rank = *q_dims
        .last()
        .ok_or_else(|| anyhow::anyhow!("QR factorization returned Q with no dimensions"))?;

    Ok(FactorizeResult {
        left: q,
        right: r,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// LU factorization implementation.
fn factorize_lu<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_lu_with_options::<T>(
        t,
        left_inds,
        options.canonical,
        options.max_rank.unwrap_or(usize::MAX),
        1e-14,
    )
}

fn factorize_lu_full_rank<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_lu_with_options::<T>(t, left_inds, canonical, usize::MAX, 0.0)
}

fn factorize_lu_with_options<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
    max_rank: usize,
    rel_tol: f64,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix
    let (a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for rrlu
    let a_matrix = native_tensor_to_matrix::<T>(&a_tensor, m, n)?;

    // Set up LU options
    let left_orthogonal = canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank,
        rel_tol,
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform LU decomposition
    let lu = rrlu(&a_matrix, Some(lu_options))?;
    let rank = lu.npivots();

    // Extract L and U matrices (permuted)
    let l_matrix = lu.left(true);
    let u_matrix = lu.right(true);

    // Create bond index
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

    // Convert L matrix back to tensor
    let l_vec = matrix_to_vec(&l_matrix);
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let left =
        TensorDynLen::from_dense(l_indices, l_vec).map_err(FactorizeError::ComputationError)?;

    // Convert U matrix back to tensor
    let u_vec = matrix_to_vec(&u_matrix);
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let right =
        TensorDynLen::from_dense(r_indices, u_vec).map_err(FactorizeError::ComputationError)?;

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

/// CI (Cross Interpolation) factorization implementation.
fn factorize_ci<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_ci_with_options::<T>(
        t,
        left_inds,
        options.canonical,
        options.max_rank.unwrap_or(usize::MAX),
        1e-14,
    )
}

fn factorize_ci_full_rank<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_ci_with_options::<T>(t, left_inds, canonical, usize::MAX, 0.0)
}

fn factorize_ci_with_options<T>(
    t: &TensorDynLen,
    left_inds: &[DynIndex],
    canonical: Canonical,
    max_rank: usize,
    rel_tol: f64,
) -> Result<FactorizeResult<TensorDynLen>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into an eager matrix. Pivot selection is primal-only, but
    // fixed-pivot CI factors are rebuilt from this eager value below.
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for MatrixLUCI
    let a_matrix = native_tensor_to_matrix::<T>(matrix_inner.data(), m, n)?;

    // Set up LU options for CI
    let left_orthogonal = canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_rank,
        rel_tol,
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform CI decomposition
    let ci = MatrixLUCI::from_matrix(&a_matrix, Some(lu_options))?;
    let rank = ci.rank();

    let row_indices = ci.row_indices().to_vec();
    let col_indices = ci.col_indices().to_vec();
    let cols_inner = matrix_inner.take_cols(&col_indices).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!(
            "fixed-pivot CI column gather failed: {e}"
        ))
    })?;
    let rows_inner = matrix_inner.take_rows(&row_indices).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!("fixed-pivot CI row gather failed: {e}"))
    })?;
    let pivot_inner = matrix_inner
        .take_block(&row_indices, &col_indices)
        .map_err(|e| {
            FactorizeError::ComputationError(anyhow::anyhow!(
                "fixed-pivot CI pivot block gather failed: {e}"
            ))
        })?;
    let (left_inner, right_inner) = match canonical {
        Canonical::Left => {
            let left = eager_right_solve(&pivot_inner, &cols_inner).map_err(|e| {
                FactorizeError::ComputationError(anyhow::anyhow!(
                    "fixed-pivot CI right solve failed: {e}"
                ))
            })?;
            (left, rows_inner)
        }
        Canonical::Right => {
            let right = eager_full_piv_lu_solve(&pivot_inner, &rows_inner).map_err(|e| {
                FactorizeError::ComputationError(anyhow::anyhow!(
                    "fixed-pivot CI solve failed: {e}"
                ))
            })?;
            (cols_inner, right)
        }
    };

    // Create bond index
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let l_dims: Vec<usize> = l_indices.iter().map(|idx| idx.dim).collect();
    let left_inner = left_inner.reshape(&l_dims).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!("fixed-pivot CI left reshape failed: {e}"))
    })?;
    let left = TensorDynLen::from_inner(l_indices, left_inner)
        .map_err(FactorizeError::ComputationError)?;

    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_dims: Vec<usize> = r_indices.iter().map(|idx| idx.dim).collect();
    let right_inner = right_inner.reshape(&r_dims).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!(
            "fixed-pivot CI right reshape failed: {e}"
        ))
    })?;
    let right = TensorDynLen::from_inner(r_indices, right_inner)
        .map_err(FactorizeError::ComputationError)?;

    Ok(FactorizeResult {
        left,
        right,
        bond_index,
        singular_values: None,
        rank,
    })
}

fn eager_right_solve(a: &EagerTensor, rhs: &EagerTensor) -> AnyhowResult<EagerTensor> {
    let a_t = a.transpose(&[1, 0])?;
    let rhs_t = rhs.transpose(&[1, 0])?;
    Ok(eager_full_piv_lu_solve(&a_t, &rhs_t)?.transpose(&[1, 0])?)
}

/// Convert a native rank-2 tensor into a backend [`Matrix`].
fn native_tensor_to_matrix<T>(
    tensor: &tenferro::Tensor,
    m: usize,
    n: usize,
) -> Result<Matrix<T>, FactorizeError>
where
    T: TensorElement + MatrixScalar + Copy,
{
    let data = T::dense_values_from_native_col_major(tensor).map_err(|e| {
        FactorizeError::ComputationError(anyhow::anyhow!(
            "failed to extract dense matrix entries from native tensor: {e}"
        ))
    })?;
    if data.len() != m * n {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "native matrix materialization produced {} entries for shape ({m}, {n})",
            data.len()
        )));
    }

    let mut matrix = Matrix::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            matrix[[i, j]] = data[j * m + i];
        }
    }
    Ok(matrix)
}

/// Convert Matrix to Vec for storage.
fn matrix_to_vec<T>(matrix: &Matrix<T>) -> Vec<T>
where
    T: Clone,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    let mut vec = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            vec.push(matrix[[i, j]].clone());
        }
    }
    vec
}

#[cfg(test)]
mod tests;
