use mdarray::DSlice;
use mdarray_linalg::svd::SVDDecomp;
use num_complex::{Complex64, ComplexFloat};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tensor4all_core::index::{DynId, Index, NoSymmSpace, Symmetry};
use tensor4all_core::index_ops::sim;
use tensor4all_core::tagset::DefaultTagSet;
use tensor4all_tensor::{unfold_split, Storage, StorageScalar, TensorDynLen};
use thiserror::Error;

use crate::backend::svd_backend;
use faer_traits::ComplexField;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    #[error("SVD computation failed: {0}")]
    ComputationError(#[from] anyhow::Error),
    #[error("Invalid rtol value: {0}. rtol must be finite and non-negative.")]
    InvalidRtol(f64),
}

/// Options for SVD decomposition with truncation control.
#[derive(Debug, Clone, Copy)]
pub struct SvdOptions {
    /// Relative Frobenius error tolerance for truncation.
    /// If `None`, uses the global default rtol.
    /// The truncation guarantees: ||A - A_approx||_F / ||A||_F <= rtol.
    pub rtol: Option<f64>,
}

impl Default for SvdOptions {
    fn default() -> Self {
        Self {
            rtol: None, // Use global default
        }
    }
}

impl SvdOptions {
    /// Create new SVD options with the specified rtol.
    pub fn with_rtol(rtol: f64) -> Self {
        Self { rtol: Some(rtol) }
    }
}

// Global default rtol stored as AtomicU64 (f64::to_bits())
// Default value: 1e-12 (near machine precision)
static DEFAULT_SVD_RTOL: AtomicU64 = AtomicU64::new(1e-12_f64.to_bits());

/// Get the global default rtol for SVD truncation.
///
/// The default value is 1e-12 (near machine precision).
pub fn default_svd_rtol() -> f64 {
    f64::from_bits(DEFAULT_SVD_RTOL.load(Ordering::Relaxed))
}

/// Set the global default rtol for SVD truncation.
///
/// # Arguments
/// * `rtol` - Relative Frobenius error tolerance (must be finite and non-negative)
///
/// # Errors
/// Returns `SvdError::InvalidRtol` if rtol is not finite or is negative.
pub fn set_default_svd_rtol(rtol: f64) -> Result<(), SvdError> {
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(SvdError::InvalidRtol(rtol));
    }
    DEFAULT_SVD_RTOL.store(rtol.to_bits(), Ordering::Relaxed);
    Ok(())
}

/// Compute the retained rank based on rtol (TSVD truncation).
///
/// This implements the truncation criterion:
///   sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
///
/// # Arguments
/// * `s_vec` - Singular values in descending order (non-negative)
/// * `rtol` - Relative Frobenius error tolerance
///
/// # Returns
/// The retained rank `r` (at least 1, at most s_vec.len())
fn compute_retained_rank(s_vec: &[f64], rtol: f64) -> usize {
    if s_vec.is_empty() {
        return 1;
    }

    // Compute total squared norm: sum_i σ_i²
    let total_sq_norm: f64 = s_vec.iter().map(|&s| s * s).sum();

    // Edge case: if total norm is zero, keep rank 1
    if total_sq_norm == 0.0 {
        return 1;
    }

    // Compute cumulative discarded weight from the end
    // We want: sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
    // So: sum_{i>r} σ_i² <= rtol² * sum_i σ_i²
    let rtol_sq = rtol * rtol;
    let threshold = rtol_sq * total_sq_norm;

    // Start from the end and accumulate discarded weight
    let mut discarded_sq_norm = 0.0;
    let mut r = s_vec.len();

    // Iterate backwards, adding singular values until threshold is exceeded
    for i in (0..s_vec.len()).rev() {
        let s_sq = s_vec[i] * s_vec[i];
        if discarded_sq_norm + s_sq <= threshold {
            discarded_sq_norm += s_sq;
            r = i;
        } else {
            break;
        }
    }

    // Ensure at least rank 1 is kept
    r.max(1)
}

/// Extract U, S, V from mdarray-linalg's SVDDecomp (which returns U, S, Vt).
///
/// This helper function converts the backend's SVD result to our desired format:
/// - Extracts singular values from the diagonal view (first row)
/// - Converts U from m×m to m×k (takes first k columns)
/// - Converts Vt to V (takes first k rows and (conjugate-)transposes)
///
/// # Arguments
/// * `decomp` - SVD decomposition from mdarray-linalg
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `k` - Bond dimension (min(m, n))
///
/// # Returns
/// A tuple `(u_vec, s_vec, v_vec)` where:
/// - `u_vec` is a vector of length `m * k` containing U matrix data
/// - `s_vec` is a vector of length `k` containing singular values (real, f64)
/// - `v_vec` is a vector of length `n * k` containing V matrix data
fn extract_usv_from_svd_decomp<T>(
    decomp: SVDDecomp<T>,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<T>, Vec<f64>, Vec<T>)
where
    T: ComplexFloat + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    let SVDDecomp { s, u, vt } = decomp;

    // Extract singular values and convert to real type.
    //
    // NOTE:
    // `mdarray-linalg-faer` writes singular values into a diagonal view created by
    // `into_faer_diag_mut`, which (by design) treats the **first row** as the
    // singular-value buffer (LAPACK-style convention). Therefore, the values live at
    // `s[0, i]`, not necessarily at `s[i, i]`.
    //
    // Singular values are always real (f64), even for complex matrices.
    let mut s_vec: Vec<f64> = Vec::with_capacity(k);
    for i in 0..k {
        let s_val = s[[0, i]];
        // <T as ComplexFloat>::Real is f64 for both f64 and Complex64
        let real_val: <T as ComplexFloat>::Real = s_val.re();
        // Convert to f64 using Into trait
        s_vec.push(real_val.into());
    }

    // Convert U from m×m to m×k (take first k columns)
    let mut u_vec = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            u_vec.push(u[[i, j]]);
        }
    }

    // Convert backend `vt` (V^T / V^H) to V (n×k).
    //
    // `mdarray-linalg` returns `vt` as (conceptually) V^T for real types or V^H for complex types.
    // We want V (not Vt), so we take the first k rows of V^T/V^H and (conjugate-)transpose.
    let mut vt_vec = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            vt_vec.push(vt[[i, j]]);
        }
    }

    let mut v_vec = Vec::with_capacity(n * k);
    for j in 0..n {
        for i in 0..k {
            // `ComplexFloat::conj` is a no-op for real types.
            v_vec.push(vt_vec[i * n + j].conj());
        }
    }

    (u_vec, s_vec, v_vec)
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function uses the global default rtol for truncation.
/// See `svd_with` for per-call rtol control.
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on the relative Frobenius error tolerance (rtol):
/// The truncation guarantees: ||A - A_approx||_F / ||A||_F <= rtol.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by (conjugate-)transposing the leading k rows.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `S` is a r×r diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is a tensor with indices `[right_inds..., bond_index]` and dimensions `[right_dims..., r]`
/// where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// Note: Singular values `S` are always real, even for complex input tensors.
///
/// # Errors
/// Returns `SvdError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The SVD computation fails
#[allow(private_bounds)]
pub fn svd<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<
    (
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    svd_with::<Id, Symm, T>(t, left_inds, &SvdOptions::default())
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function allows per-call control of the truncation tolerance via `SvdOptions`.
/// If `options.rtol` is `None`, uses the global default rtol.
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// Truncation is performed based on the relative Frobenius error tolerance (rtol):
/// The truncation guarantees: ||A - A_approx||_F / ||A||_F <= rtol.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by (conjugate-)transposing the leading k rows.
///
/// # Arguments
/// * `t` - Input tensor with DenseF64 or DenseC64 storage
/// * `left_inds` - Indices to place on the left (row) side of the unfolded matrix
/// * `options` - SVD options including rtol for truncation control
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., r]`
/// - `S` is a r×r diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is a tensor with indices `[right_inds..., bond_index]` and dimensions `[right_dims..., r]`
/// where `r` is the retained rank (≤ min(m, n)) determined by rtol truncation.
///
/// Note: Singular values `S` are always real, even for complex input tensors.
///
/// # Errors
/// Returns `SvdError` if:
/// - The tensor rank is < 2
/// - Storage is not DenseF64 or DenseC64
/// - `left_inds` is empty or contains all indices
/// - `left_inds` contains indices not in the tensor or duplicates
/// - The SVD computation fails
/// - `options.rtol` is invalid (not finite or negative)
#[allow(private_bounds)]
pub fn svd_with<Id, Symm, T>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
    options: &SvdOptions,
) -> Result<
    (
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Determine rtol to use
    let rtol = options.rtol.unwrap_or_else(default_svd_rtol);
    if !rtol.is_finite() || rtol < 0.0 {
        return Err(SvdError::InvalidRtol(rtol));
    }

    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split::<Id, T, Symm>(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(SvdError::ComputationError)?;
    let k = m.min(n);

    // Call SVD using selected backend
    // DTensor can be converted to DSlice via as_mut()
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let decomp = svd_backend(a_slice).map_err(SvdError::ComputationError)?;

    // Extract U, S, V from the decomposition (full rank k)
    let (u_vec_full, s_vec_full, v_vec_full) = extract_usv_from_svd_decomp(decomp, m, n, k);

    // Compute retained rank based on rtol truncation
    let r = compute_retained_rank(&s_vec_full, rtol);

    // Truncate to retained rank r
    let s_vec: Vec<f64> = s_vec_full[..r].to_vec();

    // Truncate U: keep first r columns (m×r)
    let mut u_vec = Vec::with_capacity(m * r);
    for i in 0..m {
        for j in 0..r {
            u_vec.push(u_vec_full[i * k + j]);
        }
    }

    // Truncate V: keep first r columns (n×r)
    let mut v_vec = Vec::with_capacity(n * r);
    for i in 0..n {
        for j in 0..r {
            v_vec.push(v_vec_full[i * k + j]);
        }
    }

    // Create bond index with "Link" tag (dimension r, not k)
    let bond_index: Index<Id, Symm, DefaultTagSet> = Index::new_link(r)
        .map_err(|e| anyhow::anyhow!("Failed to create Link index: {:?}", e))
        .map_err(SvdError::ComputationError)?;

    // Create U tensor: [left_inds..., bond_index]
    let mut u_indices = left_indices.clone();
    u_indices.push(bond_index.clone());
    let u_storage = T::dense_storage(u_vec);
    let u = TensorDynLen::from_indices(u_indices, u_storage);

    // Create S tensor: [bond_index, sim(bond_index)] (diagonal)
    // Singular values are always real (f64), even for complex input
    // Use sim() to create a similar index with a new ID to avoid duplicate index IDs
    let s_indices = vec![bond_index.clone(), sim(&bond_index)];
    let s_storage = Arc::new(Storage::new_diag_f64(s_vec));
    let s = TensorDynLen::from_indices(s_indices, s_storage);

    // Create V tensor: [right_inds..., bond_index]
    let mut v_indices = right_indices.clone();
    v_indices.push(bond_index.clone());
    let v_storage = T::dense_storage(v_vec);
    let v = TensorDynLen::from_indices(v_indices, v_storage);

    Ok((u, s, v))
}

/// Compute SVD decomposition of a complex tensor with arbitrary rank, returning (U, S, V).
///
/// This is a convenience wrapper around the generic `svd` function for `Complex64` tensors.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by conjugate-transposing the leading k rows.
///
/// Note: Singular values `S` are always real (f64), even for complex input tensors.
#[inline]
pub fn svd_c64<Id, Symm>(
    t: &TensorDynLen<Id, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<
    (
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
        TensorDynLen<Id, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    svd::<Id, Symm, Complex64>(t, left_inds)
}
