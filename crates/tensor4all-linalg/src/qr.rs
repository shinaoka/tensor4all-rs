use mdarray::{DSlice, DTensor};
use num_complex::{Complex64, ComplexFloat};
use std::sync::atomic::{AtomicU64, Ordering};
use tensor4all_index::index::{DynId, Index, NoSymmSpace, Symmetry};
use tensor4all_index::tagset::DefaultTagSet;
use tensor4all_tensor::{unfold_split, StorageScalar, TensorDynLen};
use thiserror::Error;

use crate::backend::qr_backend;
use faer_traits::ComplexField;

/// Error type for QR operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum QrError {
    #[error("QR computation failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
}

/// Options for QR decomposition with truncation control.
#[derive(Debug, Clone, Copy)]
pub struct QrOptions {
    /// Relative tolerance for truncation based on R's diagonal elements.
    /// If `None`, uses the global default rtol.
    /// Columns of R with |R[i, i]| < rtol are truncated.
    pub rtol: Option<f64>,
}

impl Default for QrOptions {
    fn default() -> Self {
        Self {
            rtol: None, // Use global default
        }
    }
}

impl QrOptions {
    /// Create new QR options with the specified rtol.
    pub fn with_rtol(rtol: f64) -> Self {
        Self { rtol: Some(rtol) }
    }
}

// Global default rtol stored as AtomicU64 (f64::to_bits())
// Default value: 1e-15 (very strict, near machine precision)
static DEFAULT_QR_RTOL: AtomicU64 = AtomicU64::new(1e-15_f64.to_bits());

/// Get the global default rtol for QR truncation.
///
/// The default value is 1e-15 (very strict, near machine precision).
pub fn default_qr_rtol() -> f64 {
    f64::from_bits(DEFAULT_QR_RTOL.load(Ordering::Relaxed))
}

/// Set the global default rtol for QR truncation.
///
/// # Arguments
/// * `rtol` - Relative tolerance (must be finite and non-negative)
///
/// # Errors
/// Returns `QrError::InvalidRtol` if rtol is not finite or is negative.
pub fn set_default_qr_rtol(rtol: f64) -> Result<(), QrError> {
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(QrError::InvalidRtol(rtol));
    }
    DEFAULT_QR_RTOL.store(rtol.to_bits(), Ordering::Relaxed);
    Ok(())
}

/// Compute the retained rank based on rtol truncation for QR.
///
/// This checks R's diagonal elements and truncates columns where |R[i, i]| < rtol.
///
/// # Arguments
/// * `r_full` - Full R matrix (k×n, upper triangular)
/// * `k` - Number of rows in R (min(m, n))
/// * `n` - Number of columns in R
/// * `rtol` - Relative tolerance for diagonal elements
///
/// # Returns
/// The retained rank `r` (at least 1, at most k)
fn compute_retained_rank_qr<T>(r_full: &DTensor<T, 2>, k: usize, n: usize, rtol: f64) -> usize
where
    T: ComplexFloat,
    <T as ComplexFloat>::Real: Into<f64>,
{
    if k == 0 || n == 0 {
        return 1;
    }

    // Check diagonal elements of R (R is k×n, upper triangular)
    // Diagonal elements are at R[0,0], R[1,1], ..., R[min(k,n)-1, min(k,n)-1]
    let max_diag = k.min(n);
    let mut r = max_diag;

    for i in 0..max_diag {
        let r_ii = r_full[[i, i]];
        let abs_r_ii: f64 = r_ii.abs().into();
        if abs_r_ii < rtol {
            r = i;
            break;
        }
    }

    // Ensure at least rank 1 is kept
    r.max(1)
}

/// Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R).
///
/// This function uses the global default rtol for truncation.
/// See `qr_with` for per-call rtol control.
///
/// This function computes the thin QR decomposition, where for an unfolded matrix A (m×n),
/// we return Q (m×k) and R (k×n) with k = min(m, n).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on R's diagonal elements: columns with |R[i, i]| < rtol are truncated.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is orthogonal (or unitary for complex) and R is upper triangular.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
///
/// # Returns
/// A tuple `(Q, R)` where:
/// - `Q` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `R` is a tensor with indices `[bond_index, right_inds...]` and dimensions `[r, right_dims...]`
/// where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// # Errors
/// Returns `QrError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The QR computation fails
#[allow(private_bounds)]
pub fn qr<Id, Symm, T>(
    t: &TensorDynLen<Id, T, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<(TensorDynLen<Id, T, Symm>, TensorDynLen<Id, T, Symm>), QrError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    qr_with(t, left_inds, &QrOptions::default())
}

/// Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R).
///
/// This function allows per-call control of the truncation tolerance via `QrOptions`.
/// If `options.rtol` is `None`, uses the global default rtol.
///
/// This function computes the thin QR decomposition, where for an unfolded matrix A (m×n),
/// we return Q (m×k) and R (k×n) with k = min(m, n).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on R's diagonal elements: columns with |R[i, i]| < rtol are truncated.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is orthogonal (or unitary for complex) and R is upper triangular.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
/// * `options` - QR options including rtol for truncation control
///
/// # Returns
/// A tuple `(Q, R)` where:
/// - `Q` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `R` is a tensor with indices `[bond_index, right_inds...]` and dimensions `[r, right_dims...]`
/// where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// # Errors
/// Returns `QrError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The QR computation fails
/// - `options.rtol` is invalid (not finite or negative)
#[allow(private_bounds)]
pub fn qr_with<Id, Symm, T>(
    t: &TensorDynLen<Id, T, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &QrOptions,
) -> Result<(TensorDynLen<Id, T, Symm>, TensorDynLen<Id, T, Symm>), QrError>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Determine rtol to use
    let rtol = options.rtol.unwrap_or_else(default_qr_rtol);
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(QrError::InvalidRtol(rtol));
    }

    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;
    let k = m.min(n);

    // Call QR using selected backend
    // DTensor can be converted to DSlice via as_mut()
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let (q_full, r_full) = qr_backend(a_slice);

    // Compute retained rank based on rtol truncation
    let r = compute_retained_rank_qr(&r_full, k, n, rtol);

    // Extract truncated QR: keep first r columns of Q and first r rows of R
    // Q_thin: m×r (first r columns of Q)
    let mut q_vec = Vec::with_capacity(m * r);
    for i in 0..m {
        for j in 0..r {
            q_vec.push(q_full[[i, j]]);
        }
    }

    // R_thin: r×n (first r rows of R)
    let mut r_vec = Vec::with_capacity(r * n);
    for i in 0..r {
        for j in 0..n {
            r_vec.push(r_full[[i, j]]);
        }
    }

    // Create bond index with "Link" tag (dimension r, not k)
    let bond_index: Index<Id, Symm, DefaultTagSet> = Index::new_link(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(QrError::ComputationError)?;

    // Create Q tensor: [left_inds..., bond_index]
    let mut q_indices = left_indices.clone();
    q_indices.push(bond_index.clone());
    let q_storage = T::dense_storage(q_vec);
    let q = TensorDynLen::from_indices(q_indices, q_storage);

    // Create R tensor: [bond_index, right_inds...]
    let mut r_indices = vec![bond_index.clone()];
    r_indices.extend_from_slice(&right_indices);
    let r_storage = T::dense_storage(r_vec);
    let r = TensorDynLen::from_indices(r_indices, r_storage);

    Ok((q, r))
}

/// Compute QR decomposition of a complex tensor with arbitrary rank, returning (Q, R).
///
/// This is a convenience wrapper around the generic `qr` function for `Complex64` tensors.
///
/// For the mathematical convention:
/// \[ A = Q * R \]
/// where Q is unitary and R is upper triangular.
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
#[inline]
pub fn qr_c64<Id, Symm>(
    t: &TensorDynLen<Id, Complex64, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<
    (
        TensorDynLen<Id, Complex64, Symm>,
        TensorDynLen<Id, Complex64, Symm>,
    ),
    QrError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    qr(t, left_inds)
}
