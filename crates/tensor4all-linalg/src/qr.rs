use mdarray::{DSlice, DTensor};
use num_complex::{Complex64, ComplexFloat};
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
}

/// Extract thin QR from full QR decomposition.
///
/// For a full QR decomposition where Q is m×m and R is m×n, this function
/// extracts the thin QR where Q_thin is m×k and R_thin is k×n, with k = min(m, n).
///
/// # Arguments
/// * `q_full` - Full Q matrix (m×m)
/// * `r_full` - Full R matrix (m×n)
/// * `m` - Number of rows
/// * `n` - Number of columns
/// * `k` - Bond dimension (min(m, n))
///
/// # Returns
/// A tuple `(q_vec, r_vec)` where:
/// - `q_vec` is a vector of length `m * k` containing Q_thin matrix data (row-major)
/// - `r_vec` is a vector of length `k * n` containing R_thin matrix data (row-major)
fn extract_thin_qr<T>(
    q_full: &DTensor<T, 2>,
    r_full: &DTensor<T, 2>,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<T>, Vec<T>)
where
    T: ComplexFloat + Default + Copy,
{
    // Extract Q_thin: first k columns of Q (m×k)
    let mut q_vec = Vec::with_capacity(m * k);
    for i in 0..m {
        for j in 0..k {
            q_vec.push(q_full[[i, j]]);
        }
    }

    // Extract R_thin: first k rows of R (k×n)
    let mut r_vec = Vec::with_capacity(k * n);
    for i in 0..k {
        for j in 0..n {
            r_vec.push(r_full[[i, j]]);
        }
    }

    (q_vec, r_vec)
}

/// Compute QR decomposition of a tensor with arbitrary rank, returning (Q, R).
///
/// This function computes the thin QR decomposition, where for an unfolded matrix A (m×n),
/// we return Q (m×k) and R (k×n) with k = min(m, n).
///
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
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
/// - `Q` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., k]`
/// - `R` is a tensor with indices `[bond_index, right_inds...]` and dimensions `[k, right_dims...]`
/// where `k = min(m, n)` is the bond dimension, `m = ∏left_dims`, and `n = ∏right_dims`.
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
    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(|e| anyhow::anyhow!("Failed to unfold tensor: {}", e))
        .map_err(QrError::ComputationError)?;
    let k = m.min(n);

    // Call QR using selected backend
    // DTensor can be converted to DSlice via as_mut()
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let (q_full, r_full) = qr_backend(a_slice);

    // Extract thin QR from full QR
    let (q_vec, r_vec) = extract_thin_qr(&q_full, &r_full, m, n, k);

    // Create bond index with "Link" tag
    let bond_index: Index<Id, Symm, DefaultTagSet> = Index::new_link(k)
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
