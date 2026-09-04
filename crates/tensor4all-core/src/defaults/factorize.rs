//! Unified tensor factorization module.
//!
//! This module provides a unified `factorize()` function that dispatches to
//! SVD, QR, LU, or CI (Cross Interpolation) algorithms based on options.
//!
//! # Note
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.
//! Generic tensor types are not supported.
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{factorize, Canonical, DynIndex, FactorizeOptions, IdxTensor, TensorLike};
//!
//! # fn main() -> anyhow::Result<()> {
//! let i = DynIndex::new_dyn(2);
//! let j = DynIndex::new_dyn(2);
//! let tensor = IdxTensor::from_dense(
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

use crate::defaults::idx_tensor::unfold_split_inner;
use crate::defaults::DynIndex;
use crate::{contract_pair, unfold_split, AnyScalar, IdxTensor};
use crate::{
    matrix_luci_factors_from_matrix, rrlu, MatrixLuciFactors, RrLUOptions, Scalar as MatrixScalar,
};
use num_complex::{Complex64, ComplexFloat};
use tenferro_ad::EagerTensor;
use tensor4all_tensorbackend::{Matrix, TensorElement};

use crate::defaults::svd::{
    compute_retained_rank, svd_for_factorize, svd_for_factorize_in, SvdFactorizeResult,
};
use crate::qr::{qr_with, qr_with_in, QrOptions};
use crate::svd::SvdOptions;
use crate::truncation::{
    validate_svd_truncation_policy, SingularValueMeasure, SvdTruncationPolicy, ThresholdScale,
    TruncationRule,
};
use crate::ExecutionContext;

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
/// - LU or CI is requested for a tracked tensor (those paths do not yet
///   preserve reverse-mode AD metadata)
/// - The underlying algorithm fails
pub fn factorize(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    options.validate()?;

    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }
    if t.tracks_grad() && matches!(options.alg, FactorizeAlg::LU | FactorizeAlg::CI) {
        return Err(FactorizeError::UnsupportedStorage(
            "LU and CI factorization do not support tracked tensors yet",
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

/// Select Gram eigendecomposition for safe untracked SVD truncations.
///
/// This is intentionally crate-visible: the public entry point is the
/// [`TensorFactorizationLike::factorize_auto`] method.
pub(crate) fn factorize_auto(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    if options.alg != FactorizeAlg::SVD {
        return Err(FactorizeError::InvalidOptions(
            "automatic factorization only supports SVD options",
        ));
    }
    options.validate()?;

    let Some(policy) = options.svd_policy else {
        return factorize(t, left_inds, options);
    };
    validate_svd_truncation_policy(policy).map_err(|error| FactorizeError::InvalidRtol(error.0))?;

    let effective_cutoff = match (policy.scale, policy.measure, policy.rule) {
        (ThresholdScale::Relative, SingularValueMeasure::SquaredValue, _) => policy.threshold,
        (ThresholdScale::Relative, SingularValueMeasure::Value, TruncationRule::PerValue) => {
            policy.threshold * policy.threshold
        }
        _ => 0.0,
    };
    if effective_cutoff <= 1.0e-12 || t.tracks_grad() || t.is_diag() {
        return factorize(t, left_inds, options);
    }
    if !(t.is_f64() || t.is_c64()) {
        return factorize(t, left_inds, options);
    }

    factorize_gram(t, left_inds, options, policy).or_else(|_| factorize(t, left_inds, options))
}

fn factorize_gram(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
    policy: SvdTruncationPolicy,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|error| FactorizeError::ComputationError(anyhow::anyhow!(error)))?;
    if m == 0 || n == 0 {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "cannot factorize a matrix with an empty dimension"
        )));
    }

    let row = DynIndex::new_dyn(m);
    let column = DynIndex::new_dyn(n);
    let matrix = IdxTensor::from_inner(vec![row.clone(), column.clone()], matrix_inner)
        .map_err(FactorizeError::ComputationError)?;

    // Form the smaller Gram tensor directly through the native contraction
    // path. No rectangular host Matrix or Matrix↔backend round-trip is needed.
    let eigenvectors_left = m <= n;
    let gram = if eigenvectors_left {
        let sim_row = DynIndex::new_dyn(m);
        let adjoint = matrix
            .conj()
            .replaceind(&row, &sim_row)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        contract_pair(&matrix, &adjoint)
    } else {
        let sim_column = DynIndex::new_dyn(n);
        let adjoint = matrix
            .conj()
            .replaceind(&column, &sim_column)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        contract_pair(&adjoint, &matrix)
    }
    .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    let decomposition = gram
        .hermitian_eigendecomposition(1.0e-12)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;

    let lambda_scale = decomposition
        .eigenvalues
        .iter()
        .map(|lambda| lambda.abs())
        .fold(0.0_f64, f64::max);
    if lambda_scale == 0.0 {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "zero Gram matrix uses the SVD fallback"
        )));
    }
    let negative_tolerance = 1.0e-12 * lambda_scale;
    let mut eigenpairs: Vec<(f64, usize)> = decomposition
        .eigenvalues
        .iter()
        .copied()
        .enumerate()
        .map(|(column, mut lambda)| {
            if !lambda.is_finite() {
                return Err(FactorizeError::ComputationError(anyhow::anyhow!(
                    "Gram eigendecomposition returned a non-finite eigenvalue"
                )));
            }
            if lambda < -negative_tolerance {
                return Err(FactorizeError::ComputationError(anyhow::anyhow!(
                    "Gram eigendecomposition returned a materially negative eigenvalue"
                )));
            }
            if lambda < 0.0 {
                lambda = 0.0;
            }
            Ok((lambda, column))
        })
        .collect::<Result<_, _>>()?;
    eigenpairs.sort_by(|(left, _), (right, _)| right.total_cmp(left));

    let all_singular_values: Vec<f64> =
        eigenpairs.iter().map(|(lambda, _)| lambda.sqrt()).collect();
    let mut rank = compute_retained_rank(&all_singular_values, &policy);
    if let Some(max_bond_dim) = options.max_bond_dim {
        rank = rank.min(max_bond_dim);
    }
    rank = rank.max(1).min(m.min(n));
    let singular_values = all_singular_values[..rank].to_vec();
    if singular_values.contains(&0.0) {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "zero retained singular value uses the SVD fallback"
        )));
    }

    let retained_columns: Vec<usize> = eigenpairs
        .iter()
        .take(rank)
        .map(|(_, column)| *column)
        .collect();
    let basis_inner = decomposition
        .eigenvectors
        .as_inner()
        .map_err(FactorizeError::ComputationError)?
        .take_cols(&retained_columns)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    let bond_index = DynIndex::new_bond(rank)
        .map_err(|error| FactorizeError::ComputationError(anyhow::anyhow!(error)))?;
    let basis_index = if eigenvectors_left {
        row.clone()
    } else {
        column.clone()
    };
    let basis = IdxTensor::from_inner(vec![basis_index, bond_index.clone()], basis_inner)
        .map_err(FactorizeError::ComputationError)?;

    let (left_matrix, right_matrix) = if eigenvectors_left {
        let sigma_vh = contract_pair(&basis.conj(), &matrix)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?
            .permute_indices(&[bond_index.clone(), column.clone()])
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        match options.canonical {
            Canonical::Left => (basis, sigma_vh),
            Canonical::Right => {
                let inverse: Vec<f64> = singular_values.iter().map(|sigma| 1.0 / sigma).collect();
                (
                    scale_bond(&basis, &bond_index, &singular_values)?,
                    scale_bond(&sigma_vh, &bond_index, &inverse)?,
                )
            }
        }
    } else {
        let u_sigma = contract_pair(&matrix, &basis)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?
            .permute_indices(&[row.clone(), bond_index.clone()])
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        let vh = basis
            .conj()
            .permute_indices(&[bond_index.clone(), column.clone()])
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
        match options.canonical {
            Canonical::Right => (u_sigma, vh),
            Canonical::Left => {
                let inverse: Vec<f64> = singular_values.iter().map(|sigma| 1.0 / sigma).collect();
                (
                    scale_bond(&u_sigma, &bond_index, &inverse)?,
                    scale_bond(&vh, &bond_index, &singular_values)?,
                )
            }
        }
    };

    let mut output_left_indices = left_indices;
    output_left_indices.push(bond_index.clone());
    let left = reshape_factor(left_matrix, output_left_indices)?;
    let mut output_right_indices = vec![bond_index.clone()];
    output_right_indices.extend(right_indices);
    let right = reshape_factor(right_matrix, output_right_indices)?;

    Ok(FactorizeResult::new(
        left,
        right,
        bond_index,
        Some(singular_values),
        rank,
    ))
}

fn scale_bond(
    tensor: &IdxTensor,
    bond: &DynIndex,
    values: &[f64],
) -> Result<IdxTensor, FactorizeError> {
    let temporary = DynIndex::new_bond(values.len())
        .map_err(|error| FactorizeError::ComputationError(anyhow::anyhow!(error)))?;
    let diagonal_values = values
        .iter()
        .map(|value| {
            if tensor.is_complex() {
                AnyScalar::new_complex(*value, 0.0)
            } else {
                AnyScalar::new_real(*value)
            }
        })
        .collect();
    let diagonal = IdxTensor::from_diag_any(vec![bond.clone(), temporary.clone()], diagonal_values)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    contract_pair(tensor, &diagonal)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?
        .replaceind(&temporary, bond)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?
        .permute_indices(tensor.indices())
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))
}

fn reshape_factor(tensor: IdxTensor, indices: Vec<DynIndex>) -> Result<IdxTensor, FactorizeError> {
    let dims: Vec<usize> = indices.iter().map(|index| index.dim).collect();
    let inner = tensor
        .as_inner()
        .map_err(FactorizeError::ComputationError)?
        .reshape(&dims)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    IdxTensor::from_inner(indices, inner).map_err(FactorizeError::ComputationError)
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
/// Returns [`FactorizeError`] if the storage type is unsupported, LU or CI is
/// requested for a tracked tensor, the canonical direction is unsupported for
/// the selected algorithm, or the underlying decomposition fails.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{
///     factorize_full_rank, Canonical, DynIndex, FactorizeAlg, TensorContractionLike, IdxTensor,
/// };
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(2);
/// let tensor = IdxTensor::from_dense(
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
    t: &IdxTensor,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }
    if t.tracks_grad() && matches!(alg, FactorizeAlg::LU | FactorizeAlg::CI) {
        return Err(FactorizeError::UnsupportedStorage(
            "LU and CI factorization do not support tracked tensors yet",
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

/// Factorize a tensor in a caller-owned execution context.
///
/// Context-scoped counterpart of [`factorize`]: the input must belong to
/// `context`, and QR/SVD factors, truncation, and results stay in `context`
/// with only bounded decision payloads crossing the explicit readback
/// boundary on CUDA. LU and CI have no context-scoped path and return a typed
/// error; run them through [`factorize`] on host inputs instead.
///
/// # Arguments
/// * `t` - Input tensor, which must belong to `context`.
/// * `left_inds` - Indices to place on the left side.
/// * `options` - Factorization options.
/// * `context` - Caller-owned execution context owning the input and results.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use tensor4all_core::{
///     DynIndex, ExecutionContext, FactorizeOptions, TensorContractionLike,
///     factorize_in, Canonical, FactorizeAlg,
/// };
/// use tensor4all_tensorbackend::CpuExecutionContext;
/// use tenferro_cpu::CpuBackend;
///
/// let context = ExecutionContext::Cpu(Arc::new(
///     CpuExecutionContext::from_backend(CpuBackend::new()),
/// ));
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
/// let tensor =
///     tensor4all_core::IdxTensor::from_dense_in(&context, vec![i.clone(), j.clone()], data)?;
/// let options = FactorizeOptions::qr();
/// let result = factorize_in(&tensor, &[i.clone()], &options, &context)?;
/// let recovered = result.left.contract_pair(&result.right)?;
/// assert_eq!(recovered.dims(), vec![4, 3]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
/// Returns `FactorizeError` when the tensor does not belong to `context`,
/// when the storage, algorithm, or options are unsupported, or when the
/// factorization or explicit decision readback fails.
pub fn factorize_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    options.validate()?;
    t.validate_context(context)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;

    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }

    if !(t.is_f64() || t.is_complex()) {
        return Err(FactorizeError::UnsupportedStorage(
            "factorize currently supports only f64 and Complex64 tensors",
        ));
    }

    match options.alg {
        FactorizeAlg::SVD => factorize_svd_with_options_in(t, left_inds, options, context),
        FactorizeAlg::QR => factorize_qr_with_options_in(t, left_inds, options, context),
        FactorizeAlg::LU | FactorizeAlg::CI => Err(FactorizeError::UnsupportedStorage(
            "LU and CI factorization have no context-scoped path; use factorize() on host inputs",
        )),
    }
}

/// Full-rank factorization in a caller-owned execution context.
///
/// Context-scoped counterpart of [`factorize_full_rank`] with the same
/// algorithm coverage: QR and SVD execute in `context`; LU and CI return a
/// typed error.
///
/// # Errors
/// Returns `FactorizeError` when the tensor does not belong to `context`,
/// when the storage or algorithm is unsupported, or when the factorization
/// fails.
pub fn factorize_full_rank_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    t.validate_context(context)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;

    if t.is_diag() {
        return Err(FactorizeError::UnsupportedStorage(
            "Diagonal storage not supported for factorize",
        ));
    }

    if !(t.is_f64() || t.is_complex()) {
        return Err(FactorizeError::UnsupportedStorage(
            "factorize currently supports only f64 and Complex64 tensors",
        ));
    }

    match alg {
        FactorizeAlg::SVD => {
            factorize_svd_with_options_in_full_rank(t, left_inds, canonical, context)
        }
        FactorizeAlg::QR => factorize_qr_full_rank_in(t, left_inds, canonical, context),
        FactorizeAlg::LU | FactorizeAlg::CI => Err(FactorizeError::UnsupportedStorage(
            "LU and CI factorization have no context-scoped path; use factorize_full_rank() on host inputs",
        )),
    }
}

fn factorize_svd_with_options_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let mut svd_options = SvdOptions::new();
    if let Some(policy) = options.svd_policy {
        svd_options = svd_options.with_policy(policy);
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        svd_options = svd_options.with_max_bond_dim(max_bond_dim);
    }

    factorize_svd_with_eager_options_in(t, left_inds, options.canonical, &svd_options, context)
}

fn factorize_svd_with_options_in_full_rank(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let svd_options = SvdOptions::full_rank();
    factorize_svd_with_eager_options_in(t, left_inds, canonical, &svd_options, context)
}

fn factorize_svd_with_eager_options_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    svd_options: &SvdOptions,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let result = svd_for_factorize_in(t, left_inds, svd_options, context)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    assemble_svd_factors(result, canonical)
}

fn factorize_qr_with_options_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
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

    factorize_qr_with_eager_options_in(t, left_inds, &qr_options, context)
}

fn factorize_qr_full_rank_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    if canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    factorize_qr_with_eager_options_in(t, left_inds, &QrOptions::full_rank(), context)
}

fn factorize_qr_with_eager_options_in(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    qr_options: &QrOptions,
    context: &ExecutionContext,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let (q, r) = qr_with_in::<f64>(t, left_inds, qr_options, context)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;

    let bond_index = q
        .indices
        .last()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("QR factorization returned a rank-0 Q tensor"))?;
    let q_dims = q.dims();
    let rank = *q_dims
        .last()
        .ok_or_else(|| anyhow::anyhow!("QR factorization returned Q with no dimensions"))?;

    Ok(FactorizeResult::new(q, r, bond_index, None, rank))
}

fn factorize_impl_f64(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<f64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<f64>(t, left_inds, options),
    }
}

fn factorize_impl_f64_full_rank(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    match alg {
        FactorizeAlg::SVD => factorize_svd_full_rank(t, left_inds, canonical),
        FactorizeAlg::QR => factorize_qr_full_rank(t, left_inds, canonical),
        FactorizeAlg::LU => factorize_lu_full_rank::<f64>(t, left_inds, canonical),
        FactorizeAlg::CI => factorize_ci_full_rank::<f64>(t, left_inds, canonical),
    }
}

fn factorize_impl_c64(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    match options.alg {
        FactorizeAlg::SVD => factorize_svd(t, left_inds, options),
        FactorizeAlg::QR => factorize_qr(t, left_inds, options),
        FactorizeAlg::LU => factorize_lu::<Complex64>(t, left_inds, options),
        FactorizeAlg::CI => factorize_ci::<Complex64>(t, left_inds, options),
    }
}

fn factorize_impl_c64_full_rank(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    alg: FactorizeAlg,
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    match alg {
        FactorizeAlg::SVD => factorize_svd_full_rank(t, left_inds, canonical),
        FactorizeAlg::QR => factorize_qr_full_rank(t, left_inds, canonical),
        FactorizeAlg::LU => factorize_lu_full_rank::<Complex64>(t, left_inds, canonical),
        FactorizeAlg::CI => factorize_ci_full_rank::<Complex64>(t, left_inds, canonical),
    }
}

/// SVD factorization implementation.
fn factorize_svd(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let mut svd_options = SvdOptions::new();
    if let Some(policy) = options.svd_policy {
        svd_options = svd_options.with_policy(policy);
    }
    if let Some(max_bond_dim) = options.max_bond_dim {
        svd_options = svd_options.with_max_bond_dim(max_bond_dim);
    }

    factorize_svd_with_options(t, left_inds, options.canonical, &svd_options)
}

fn factorize_svd_full_rank(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let svd_options = SvdOptions::full_rank();
    factorize_svd_with_options(t, left_inds, canonical, &svd_options)
}

fn factorize_svd_with_options(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    svd_options: &SvdOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let result = svd_for_factorize(t, left_inds, svd_options)?;
    assemble_svd_factors(result, canonical)
}

/// Absorb the SVD factors into left/right canonical form.
///
/// Shared by the host and context-scoped SVD factorization paths: contracts
/// `S` into one side and renames the leftover `sim` leg back to `bond`.
fn assemble_svd_factors(
    result: SvdFactorizeResult,
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    let u = result.u;
    let s = result.s;
    let vh = result.vh;
    let bond_index = result.bond_index;
    let singular_values = result.singular_values;
    let rank = result.rank;
    // Internal SVD plumbing: svd_for_factorize keeps the legacy convention
    // (vh shares the original `bond` leg), so S's leftover `sim` leg is
    // renamed back to `bond` after each contraction. The public svd_with API
    // instead returns V with the `sim` leg and needs no compensation.
    let sim_bond_index = s.indices[1].clone();

    match canonical {
        Canonical::Left => {
            // L = U (orthogonal), R = S * V^H
            let right_contracted = contract_pair(&s, &vh)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            let right = right_contracted
                .replaceind(&sim_bond_index, &bond_index)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            Ok(FactorizeResult::new(
                u,
                right,
                bond_index,
                Some(singular_values),
                rank,
            ))
        }
        Canonical::Right => {
            // L = U * S, R = V^H
            let left_contracted = contract_pair(&u, &s)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            let left = left_contracted
                .replaceind(&sim_bond_index, &bond_index)
                .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;
            Ok(FactorizeResult::new(
                left,
                vh,
                bond_index,
                Some(singular_values),
                rank,
            ))
        }
    }
}

/// QR factorization implementation.
fn factorize_qr(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
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
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
    if canonical == Canonical::Right {
        return Err(FactorizeError::UnsupportedCanonical(
            "QR only supports Canonical::Left (would need LQ for right)",
        ));
    }

    factorize_qr_with_options(t, left_inds, &QrOptions::full_rank())
}

fn factorize_qr_with_options(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    qr_options: &QrOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError> {
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

    Ok(FactorizeResult::new(q, r, bond_index, None, rank))
}

/// LU factorization implementation.
fn factorize_lu<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_lu_with_options::<T>(
        t,
        left_inds,
        options.canonical,
        options.max_bond_dim.unwrap_or(usize::MAX),
        1e-14,
    )
}

fn factorize_lu_full_rank<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_lu_with_options::<T>(t, left_inds, canonical, usize::MAX, 0.0)
}

fn factorize_lu_with_options<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    max_bond_dim: usize,
    rel_tol: f64,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
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
        max_bond_dim,
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
    let l_vec = matrix_to_vec(&l_matrix)?;
    let mut l_indices = left_indices.clone();
    l_indices.push(bond_index.clone());
    let left = IdxTensor::from_dense(l_indices, l_vec)
        .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;

    // Convert U matrix back to tensor
    let u_vec = matrix_to_vec(&u_matrix)?;
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let right = IdxTensor::from_dense(r_indices, u_vec)
        .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;

    Ok(FactorizeResult::new(left, right, bond_index, None, rank))
}

/// CI (Cross Interpolation) factorization implementation.
fn factorize_ci<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &FactorizeOptions,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_ci_with_options::<T>(
        t,
        left_inds,
        options.canonical,
        options.max_bond_dim.unwrap_or(usize::MAX),
        1e-14,
    )
}

fn factorize_ci_full_rank<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    factorize_ci_with_options::<T>(t, left_inds, canonical, usize::MAX, 0.0)
}

// CI-factorization default seam.
//
// The default LU/CI factorization implementations in this module consume the
// TCI substrate (rrlu, MatrixLUCI, MatrixLuciScalar) that was absorbed from
// tensor4all-tcicore when that crate was dissolved into tensor4all-core
// (#639); the substrate now lives in this crate and the inverted
// core -> tcicore dependency no longer exists. IdxTensor data is unfolded
// into a column-major eager matrix at this boundary, and fixed-pivot CI
// factors are rebuilt from
// that primal value.
fn factorize_ci_with_options<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    canonical: Canonical,
    max_bond_dim: usize,
    rel_tol: f64,
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    T: TensorElement
        + ComplexFloat
        + Default
        + From<<T as ComplexFloat>::Real>
        + MatrixScalar
        + crate::MatrixLuciScalar
        + 'static,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into an eager matrix. Pivot selection is primal-only, but
    // fixed-pivot CI factors are rebuilt from this eager value below.
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))?;

    // Convert to Matrix type for MatrixLUCI.
    let a_matrix = eager_tensor_to_matrix::<T>(&matrix_inner, m, n)?;

    // Set up LU options for CI
    let left_orthogonal = canonical == Canonical::Left;
    let lu_options = RrLUOptions {
        max_bond_dim,
        rel_tol,
        abs_tol: 0.0,
        left_orthogonal,
    };

    // Perform CI decomposition and reuse its backend factors directly. The
    // previous path gathered pivot blocks into eager tensors and solved them
    // element-wise at this boundary, duplicating work already done by the
    // backend factorization.
    let factors = matrix_luci_factors_from_matrix(&a_matrix, Some(lu_options))?;
    let rank = factors.rank;
    let (left, right, bond_index) =
        matrix_luci_factors_to_idx_tensors(factors, &left_indices, &right_indices)?;

    Ok(FactorizeResult::new(left, right, bond_index, None, rank))
}

fn matrix_luci_factors_to_idx_tensors<T>(
    factors: MatrixLuciFactors<T>,
    left_indices: &[DynIndex],
    right_indices: &[DynIndex],
) -> Result<(IdxTensor, IdxTensor, DynIndex), FactorizeError>
where
    T: TensorElement + Clone,
{
    let bond_index = DynIndex::new_bond(factors.rank)
        .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

    let mut l_indices = left_indices.to_vec();
    l_indices.push(bond_index.clone());
    let left = IdxTensor::from_dense(l_indices, matrix_to_vec(&factors.left)?)
        .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;

    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(right_indices);
    let right = IdxTensor::from_dense(r_indices, matrix_to_vec(&factors.right)?)
        .map_err(|e| FactorizeError::ComputationError(anyhow::Error::new(e)))?;

    Ok((left, right, bond_index))
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
    matrix_from_col_major_values(data, m, n, "native")
}

fn eager_tensor_to_matrix<T>(
    tensor: &EagerTensor,
    m: usize,
    n: usize,
) -> Result<Matrix<T>, FactorizeError>
where
    T: TensorElement + MatrixScalar + Copy,
{
    let values = tensor
        .value()
        .map_err(|source| FactorizeError::ComputationError(anyhow::Error::new(source)))?
        .as_slice::<T>()
        .map_err(|source| FactorizeError::ComputationError(anyhow::Error::new(source)))?
        .to_vec();
    matrix_from_col_major_values(values, m, n, "eager")
}

fn matrix_from_col_major_values<T>(
    data: Vec<T>,
    m: usize,
    n: usize,
    source: &str,
) -> Result<Matrix<T>, FactorizeError>
where
    T: MatrixScalar + Copy,
{
    let expected_len = checked_matrix_len(m, n, source)?;
    if data.len() != expected_len {
        return Err(FactorizeError::ComputationError(anyhow::anyhow!(
            "{source} matrix materialization produced {} entries for shape ({m}, {n})",
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
fn matrix_to_vec<T>(matrix: &Matrix<T>) -> Result<Vec<T>, FactorizeError>
where
    T: Clone,
{
    let m = matrix.nrows();
    let n = matrix.ncols();
    let len = checked_matrix_len(m, n, "factorize output")?;
    let mut vec = Vec::with_capacity(len);
    for j in 0..n {
        for i in 0..m {
            vec.push(matrix[[i, j]].clone());
        }
    }
    Ok(vec)
}

fn checked_matrix_len(m: usize, n: usize, source: &str) -> Result<usize, FactorizeError> {
    m.checked_mul(n).ok_or_else(|| {
        FactorizeError::ComputationError(anyhow::anyhow!(
            "{source} matrix shape ({m}, {n}) overflows usize"
        ))
    })
}

#[cfg(test)]
mod tests;
