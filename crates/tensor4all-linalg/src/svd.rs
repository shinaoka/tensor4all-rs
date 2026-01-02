use std::sync::Arc;
use tensor4all_index::index::{Index, DynId, NoSymmSpace, Symmetry};
use tensor4all_index::tagset::DefaultTagSet;
use tensor4all_tensor::{Storage, StorageScalar, TensorDynLen, unfold_split};
use mdarray::DSlice;
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError as MdarraySvdError};
use mdarray_linalg_faer::Faer;
use num_complex::{Complex64, ComplexFloat};
use faer_traits::ComplexField;
use thiserror::Error;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    #[error("Tensor storage must be DenseF64 or DenseC64, got {0:?}")]
    UnsupportedStorage(String),

    #[error("mdarray-linalg SVD error: {0}")]
    BackendError(#[from] MdarraySvdError),

    #[error("Unfold error: {0}")]
    UnfoldError(#[from] anyhow::Error),

    #[error("Storage extraction error: {0}")]
    StorageExtractionError(String),
}

/// Compute SVD decomposition of a tensor with arbitrary rank, returning (U, S, V).
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor can have any rank >= 2, and indices are split into left and right groups.
/// The tensor is unfolded into a matrix by grouping left indices as rows and right indices as columns.
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
/// - `U` is a tensor with indices `[left_inds..., bond_index]` and dimensions `[left_dims..., k]`
/// - `S` is a k×k diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is a tensor with indices `[right_inds..., bond_index]` and dimensions `[right_dims..., k]`
/// where `k = min(m, n)` is the bond dimension, `m = ∏left_dims`, and `n = ∏right_dims`.
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
    t: &TensorDynLen<Id, T, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<
    (
        TensorDynLen<Id, T, Symm>,
        TensorDynLen<Id, <T as ComplexFloat>::Real, Symm>,
        TensorDynLen<Id, T, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    T: StorageScalar + ComplexFloat + ComplexField + Default + From<<T as ComplexFloat>::Real>,
    <T as ComplexFloat>::Real: Into<f64> + 'static,
{
    // Unfold tensor into matrix (returns DTensor<T, 2>)
    let (mut a_tensor, _, m, n, left_indices, right_indices) = unfold_split(t, left_inds)
        .map_err(SvdError::UnfoldError)?;
    let k = m.min(n);

    // Call SVD using faer backend
    // DTensor can be converted to DSlice via as_mut()
    let bd = Faer;
    let a_slice: &mut DSlice<T, 2> = a_tensor.as_mut();
    let SVDDecomp { s, u, vt } = bd.svd(a_slice)?;

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

    // Create bond index with "Link" tag
    // Convert from DynId to Id type
    let bond_dyn_id = DynId(tensor4all_index::index::generate_id());
    let bond_id: Id = bond_dyn_id.into();
    let bond_symm: Symm = NoSymmSpace::new(k).into();
    let mut bond_index: Index<Id, Symm, DefaultTagSet> = Index::new(bond_id, bond_symm);
    bond_index.tags_mut().add_tag("Link").map_err(|_| {
        SvdError::UnsupportedStorage("Failed to add Link tag".to_string())
    })?;

    // Create U tensor: [left_inds..., bond_index]
    let mut u_indices = left_indices.clone();
    u_indices.push(bond_index.clone());
    let mut u_dims: Vec<usize> = left_indices.iter().map(|idx| idx.size()).collect();
    u_dims.push(k);
    let u_storage = T::dense_storage(u_vec);
    let u = TensorDynLen::new(u_indices, u_dims, u_storage);

    // Create S tensor: [bond_index, bond_index] (diagonal)
    // Singular values are always real (f64), even for complex input
    let s_indices = vec![bond_index.clone(), bond_index.clone()];
    let s_storage = Arc::new(Storage::new_diag_f64(s_vec));
    let s = TensorDynLen::new(s_indices, vec![k, k], s_storage);

    // Create V tensor: [right_inds..., bond_index]
    let mut v_indices = right_indices.clone();
    v_indices.push(bond_index.clone());
    let mut v_dims: Vec<usize> = right_indices.iter().map(|idx| idx.size()).collect();
    v_dims.push(k);
    let v_storage = T::dense_storage(v_vec);
    let v = TensorDynLen::new(v_indices, v_dims, v_storage);

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
    t: &TensorDynLen<Id, Complex64, Symm>,
    left_inds: &[Index<Id, Symm>],
) -> Result<
    (
        TensorDynLen<Id, Complex64, Symm>,
        TensorDynLen<Id, f64, Symm>,
        TensorDynLen<Id, Complex64, Symm>,
    ),
    SvdError,
>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId>,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
{
    svd(t, left_inds)
}
