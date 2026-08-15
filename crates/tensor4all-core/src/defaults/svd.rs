//! SVD decomposition for tensors.
//!
//! Provides [`svd`] and [`svd_with`] for computing truncated SVD of
//! [`IdxTensor`] values. The tensor is unfolded into a matrix by
//! splitting its indices into left and right groups, then the standard
//! matrix SVD is computed and truncated according to [`SvdOptions`].
//!
//! This module works with concrete types (`DynIndex`, `IdxTensor`) only.

use crate::defaults::idx_tensor::unfold_split_inner;
use crate::defaults::DynIndex;
use crate::index_like::IndexLike;
use crate::truncation::{
    validate_svd_truncation_policy, SingularValueMeasure, SvdTruncationPolicy, ThresholdScale,
    TruncationRule,
};
use crate::IdxTensor;
use num_complex::Complex64;
use std::sync::Mutex;
use tenferro::DType;
use tenferro_ad::EagerTensor;
use tenferro_linalg::EagerTensorLinalgExt;
#[cfg(test)]
use tensor4all_tensorbackend::native_tensor_primal_to_dense_col_major;
use thiserror::Error;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    /// SVD computation failed.
    #[error("SVD computation failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    /// Invalid truncation threshold value (must be finite and non-negative).
    #[error("Invalid SVD truncation threshold: {0}. Threshold must be finite and non-negative.")]
    InvalidThreshold(f64),
}

/// Options for SVD decomposition with truncation control.
///
/// # Examples
///
/// ```
/// use tensor4all_core::svd::{SvdOptions, svd_with};
/// use tensor4all_core::{DynIndex, SvdTruncationPolicy, IdxTensor};
///
/// let i = DynIndex::new_dyn(3);
/// let j = DynIndex::new_dyn(3);
/// let data: Vec<f64> = (0..9).map(|x| x as f64).collect();
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
///
/// // U has left index + bond, S is diagonal bond x bond, V has right index + bond
/// assert_eq!(u.dims()[0], 3);
/// assert_eq!(s.dims().len(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SvdOptions {
    /// Maximum retained rank after policy-based truncation.
    pub max_bond_dim: Option<usize>,
    /// Per-call SVD truncation policy.
    /// If `None`, the global default policy is used.
    pub policy: Option<SvdTruncationPolicy>,
    truncate: bool,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SvdOptions {
    /// Create new SVD options with no overrides.
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_bond_dim: None,
            policy: None,
            truncate: true,
        }
    }

    /// Set the maximum retained rank.
    #[must_use]
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy override.
    #[must_use]
    pub fn with_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    pub(crate) fn full_rank() -> Self {
        Self {
            max_bond_dim: None,
            policy: None,
            truncate: false,
        }
    }
}

fn default_policy_guard() -> std::sync::MutexGuard<'static, SvdTruncationPolicy> {
    match DEFAULT_SVD_TRUNCATION_POLICY.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

// Default value: relative per-value threshold 1e-12.
static DEFAULT_SVD_TRUNCATION_POLICY: Mutex<SvdTruncationPolicy> =
    Mutex::new(SvdTruncationPolicy::new(1e-12));

/// Get the global default truncation policy for SVD.
///
/// The default policy is `SvdTruncationPolicy::new(1e-12)`.
#[must_use]
pub fn default_svd_truncation_policy() -> SvdTruncationPolicy {
    *default_policy_guard()
}

/// Set the global default truncation policy for SVD.
///
/// # Arguments
/// * `policy` - SVD truncation policy to use when `SvdOptions::policy` is `None`
///
/// # Errors
/// Returns `SvdError::InvalidThreshold` if `policy.threshold` is invalid.
pub fn set_default_svd_truncation_policy(policy: SvdTruncationPolicy) -> Result<(), SvdError> {
    validate_svd_truncation_policy(policy).map_err(|e| SvdError::InvalidThreshold(e.0))?;
    *default_policy_guard() = policy;
    Ok(())
}

fn singular_value_measure(value: f64, measure: SingularValueMeasure) -> f64 {
    match measure {
        SingularValueMeasure::Value => value,
        SingularValueMeasure::SquaredValue => value * value,
    }
}

/// Compute the retained rank based on an explicit SVD truncation policy.
fn compute_retained_rank(s_vec: &[f64], policy: &SvdTruncationPolicy) -> usize {
    if s_vec.is_empty() {
        return 1;
    }

    let measured: Vec<f64> = s_vec
        .iter()
        .map(|&value| singular_value_measure(value, policy.measure))
        .collect();
    if measured.iter().all(|&value| value == 0.0) {
        return 1;
    }

    let retained = match (policy.scale, policy.rule) {
        (ThresholdScale::Relative, TruncationRule::PerValue) => {
            let reference = measured.iter().copied().fold(0.0_f64, f64::max);
            measured
                .iter()
                .take_while(|&&value| reference > 0.0 && value / reference > policy.threshold)
                .count()
        }
        (ThresholdScale::Absolute, TruncationRule::PerValue) => measured
            .iter()
            .take_while(|&&value| value > policy.threshold)
            .count(),
        (ThresholdScale::Relative, TruncationRule::DiscardedTailSum) => {
            let total: f64 = measured.iter().sum();
            if total == 0.0 {
                1
            } else {
                let mut discarded = 0.0;
                let mut keep = measured.len();
                for (i, value) in measured.iter().enumerate().rev() {
                    if (discarded + value) / total <= policy.threshold {
                        discarded += value;
                        keep = i;
                    } else {
                        break;
                    }
                }
                keep
            }
        }
        (ThresholdScale::Absolute, TruncationRule::DiscardedTailSum) => {
            let mut discarded = 0.0;
            let mut keep = measured.len();
            for (i, value) in measured.iter().enumerate().rev() {
                if discarded + value <= policy.threshold {
                    discarded += value;
                    keep = i;
                } else {
                    break;
                }
            }
            keep
        }
    };

    retained.max(1)
}

#[cfg(test)]
fn singular_values_from_native(tensor: &tenferro::Tensor) -> Result<Vec<f64>, SvdError> {
    match tensor.dtype() {
        DType::F64 => native_tensor_primal_to_dense_col_major::<f64>(tensor)
            .map_err(|e| SvdError::ComputationError(e.source)),
        DType::C64 => native_tensor_primal_to_dense_col_major::<Complex64>(tensor)
            .map(|values| values.into_iter().map(|value| value.re).collect())
            .map_err(|e| SvdError::ComputationError(e.source)),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "native SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
}

fn singular_values_from_eager(tensor: &EagerTensor) -> Result<Vec<f64>, SvdError> {
    let value = tensor
        .value()
        .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source)))?;
    match tensor.dtype() {
        DType::F64 => value
            .as_slice::<f64>()
            .map(|values| values.to_vec())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        DType::C64 => value
            .as_slice::<Complex64>()
            .map(|values| values.iter().map(|value| value.re).collect())
            .map_err(|source| SvdError::ComputationError(anyhow::Error::new(source))),
        other => Err(SvdError::ComputationError(anyhow::anyhow!(
            "eager SVD returned unsupported singular-value scalar type {other:?}"
        ))),
    }
}

type SvdTruncatedEagerResult = (
    EagerTensor,
    EagerTensor,
    EagerTensor,
    Vec<f64>,
    DynIndex,
    Vec<DynIndex>,
    Vec<DynIndex>,
);

fn svd_truncated_inner(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdTruncatedEagerResult, SvdError> {
    let (matrix_inner, _, m, n, left_indices, right_indices) = unfold_split_inner(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    let (mut u_inner, mut s_inner, mut vt_inner) = matrix_inner
        .svd()
        .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
    let s_full = singular_values_from_eager(&s_inner)?;
    let mut r = if options.truncate {
        let policy = options.policy.unwrap_or_else(default_svd_truncation_policy);
        validate_svd_truncation_policy(policy).map_err(|e| SvdError::InvalidThreshold(e.0))?;

        let mut retained = compute_retained_rank(&s_full, &policy);
        if let Some(max_bond_dim) = options.max_bond_dim {
            retained = retained.min(max_bond_dim);
        }
        retained.max(1)
    } else {
        k.max(1)
    };
    r = r.min(s_full.len());
    if r < k {
        let keep: Vec<usize> = (0..r).collect();
        u_inner = u_inner
            .take_axis(1, &keep)
            .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
        s_inner = s_inner
            .take_axis(0, &keep)
            .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
        vt_inner = vt_inner
            .take_axis(0, &keep)
            .map_err(|e| SvdError::ComputationError(anyhow::anyhow!("{e}")))?;
    }

    let bond_index = DynIndex::new_bond(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;
    let singular_values = s_full[..r].to_vec();

    Ok((
        u_inner,
        s_inner,
        vt_inner,
        singular_values,
        bond_index,
        left_indices,
        right_indices,
    ))
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IdxTensor, DynIndex, svd};
///
/// // Create a 2x3 matrix (rank-1 outer product: all-ones)
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let data = vec![1.0_f64; 6]; // all-ones 2x3 matrix
/// let t = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// let (u, s, v) = svd::<f64>(&t, &[i.clone()]).unwrap();
///
/// // U: shape (left_dim, bond) = (2, bond)
/// assert_eq!(u.dims()[0], 2);
/// // V: shape (right_dim, bond) = (3, bond)
/// assert_eq!(v.dims()[0], 3);
/// // S is a diagonal matrix (bond × bond)
/// assert_eq!(s.dims().len(), 2);
/// ```
pub fn svd<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    svd_with::<T>(t, left_inds, &SvdOptions::default())
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function allows per-call control of the truncation policy via `SvdOptions`.
/// If `options.policy` is `None`, it uses the global default policy.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IdxTensor};
/// use tensor4all_core::svd::{SvdOptions, svd_with};
///
/// let i = DynIndex::new_dyn(4);
/// let j = DynIndex::new_dyn(4);
/// // Rank-1 matrix
/// let mut data = vec![0.0_f64; 16];
/// data[0] = 1.0;
/// let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
///
/// use tensor4all_core::SvdTruncationPolicy;
///
/// // Truncate with a relative per-value threshold => rank 1
/// let opts = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
/// let (u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert_eq!(s.dims()[0], 1);  // rank-1
///
/// // Truncate with max_bond_dim => capped
/// let opts = SvdOptions::new().with_max_bond_dim(2);
/// let (_u, s, _v) = svd_with::<f64>(&tensor, &[i.clone()], &opts).unwrap();
/// assert!(s.dims()[0] <= 2);
/// ```
pub fn svd_with<T>(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<(IdxTensor, IdxTensor, IdxTensor), SvdError> {
    let (u_inner, s_inner, vt_inner, _singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner(t, left_inds, options)?;

    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = u_inner.reshape(&u_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD U reshape failed: {e}"))
    })?;
    let u = IdxTensor::from_inner(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s = IdxTensor::from_diag_inner(s_indices, s_inner).map_err(SvdError::ComputationError)?;

    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = vt_inner.reshape(&vh_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD V^T reshape failed: {e}"))
    })?;
    let vh = IdxTensor::from_inner(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;
    let perm: Vec<usize> = (1..vh.indices.len()).chain(std::iter::once(0)).collect();
    let v = vh
        .conj()
        .permute(&perm)
        .map_err(|e| SvdError::ComputationError(anyhow::Error::new(e)))?;

    Ok((u, s, v))
}

/// SVD result for factorization, returning `V^H` directly.
pub(crate) struct SvdFactorizeResult {
    pub u: IdxTensor,
    pub s: IdxTensor,
    pub vh: IdxTensor,
    pub bond_index: DynIndex,
    pub singular_values: Vec<f64>,
    pub rank: usize,
}

/// Compute truncated SVD for factorization, returning `V^H` instead of `V`.
pub(crate) fn svd_for_factorize(
    t: &IdxTensor,
    left_inds: &[DynIndex],
    options: &SvdOptions,
) -> Result<SvdFactorizeResult, SvdError> {
    let (u_inner, s_inner, vt_inner, singular_values, bond_index, left_indices, right_indices) =
        svd_truncated_inner(t, left_inds, options)?;
    let rank = singular_values.len();

    let mut u_indices = left_indices;
    u_indices.push(bond_index.clone());
    let u_dims: Vec<usize> = u_indices.iter().map(|idx| idx.dim).collect();
    let u_reshaped = u_inner.reshape(&u_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD U reshape failed: {e}"))
    })?;
    let u = IdxTensor::from_inner(u_indices, u_reshaped).map_err(SvdError::ComputationError)?;

    let s_indices = vec![bond_index.clone(), bond_index.sim()];
    let s = IdxTensor::from_diag_inner(s_indices, s_inner).map_err(SvdError::ComputationError)?;

    let mut vh_indices = vec![bond_index.clone()];
    vh_indices.extend(right_indices);
    let vh_dims: Vec<usize> = vh_indices.iter().map(|idx| idx.dim).collect();
    let vt_reshaped = vt_inner.reshape(&vh_dims).map_err(|e| {
        SvdError::ComputationError(anyhow::anyhow!("eager SVD V^T reshape failed: {e}"))
    })?;
    let vh = IdxTensor::from_inner(vh_indices, vt_reshaped).map_err(SvdError::ComputationError)?;

    Ok(SvdFactorizeResult {
        u,
        s,
        vh,
        bond_index,
        singular_values,
        rank,
    })
}

#[cfg(test)]
mod tests;
