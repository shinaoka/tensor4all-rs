use std::sync::Arc;
use tensor4all_index::index::{Index, DynId, NoSymmSpace, Symmetry};
use tensor4all_index::tagset::DefaultTagSet;
use tensor4all_tensor::{Storage, TensorDynLen};
use tensor4all_tensor::storage::{DenseStorageF64, DenseStorageC64};
use mdarray::{Dense, Slice, tensor};
use mdarray_linalg::svd::{SVD, SVDDecomp, SVDError as MdarraySvdError};
use mdarray_linalg_faer::Faer;
use num_complex::{Complex64, ComplexFloat};
use faer_traits::ComplexField;
use thiserror::Error;

/// Error type for SVD operations in tensor4all-linalg.
#[derive(Debug, Error)]
pub enum SvdError {
    #[error("Tensor must have rank 2, got rank {0}")]
    InvalidRank(usize),

    #[error("Tensor storage must be DenseF64 or DenseC64, got {0:?}")]
    UnsupportedStorage(String),

    #[error("mdarray-linalg SVD error: {0}")]
    BackendError(#[from] MdarraySvdError),
}

/// Scalar types supported by this SVD wrapper.
///
/// We keep `f64` hardcoding (due to `Storage` being specialized to `f64`/`Complex64`)
/// sealed inside these impls, so the generic `svd` implementation can work with
/// `T` and `T::Real` without mentioning `f64` directly.
pub(crate) trait SvdScalar:
    Copy
    + ComplexFloat
    + ComplexField
    + Default
    + From<<Self as ComplexFloat>::Real>
    + 'static
{
    fn extract_dense(storage: &Storage) -> Result<Vec<Self>, SvdError>;
    fn dense_storage(data: Vec<Self>) -> Arc<Storage>;
    fn diag_real_storage(data: Vec<<Self as ComplexFloat>::Real>) -> Arc<Storage>;
}

impl SvdScalar for f64 {
    fn extract_dense(storage: &Storage) -> Result<Vec<Self>, SvdError> {
        match storage {
            Storage::DenseF64(ds) => Ok(ds.as_slice().to_vec()),
            _ => Err(SvdError::UnsupportedStorage(format!("{:?}", storage))),
        }
    }

    fn dense_storage(data: Vec<Self>) -> Arc<Storage> {
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)))
    }

    fn diag_real_storage(data: Vec<<Self as ComplexFloat>::Real>) -> Arc<Storage> {
        // Here `Self::Real = f64`.
        Arc::new(Storage::new_diag_f64(data))
    }
}

impl SvdScalar for Complex64 {
    fn extract_dense(storage: &Storage) -> Result<Vec<Self>, SvdError> {
        match storage {
            Storage::DenseC64(ds) => Ok(ds.as_slice().to_vec()),
            _ => Err(SvdError::UnsupportedStorage(format!("{:?}", storage))),
        }
    }

    fn dense_storage(data: Vec<Self>) -> Arc<Storage> {
        Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)))
    }

    fn diag_real_storage(data: Vec<<Self as ComplexFloat>::Real>) -> Arc<Storage> {
        // Here `Self::Real = f64`, and our `Storage` only supports `DiagF64` for real diagonals.
        Arc::new(Storage::new_diag_f64(data))
    }
}

/// Compute SVD decomposition of a rank-2 tensor, returning (U, S, V).
///
/// This function mimics ITensor's SVD API, returning U, S, and V (not Vt).
/// The input tensor must be rank-2 with DenseF64 or DenseC64 storage.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by (conjugate-)transposing the leading k rows.
///
/// # Arguments
/// * `t` - Input tensor of rank 2 with DenseF64 or DenseC64 storage
///
/// # Returns
/// A tuple `(U, S, V)` where:
/// - `U` is an m×k tensor with indices `[left_index, bond_index]`
/// - `S` is a k×k diagonal tensor with indices `[bond_index, bond_index]` (singular values are real)
/// - `V` is an n×k tensor with indices `[right_index, bond_index]`
/// where `k = min(m, n)` is the bond dimension.
///
/// Note: Singular values `S` are always real, even for complex input tensors.
///
/// # Errors
/// Returns `SvdError` if the tensor is not rank-2, storage is not DenseF64 or DenseC64,
/// or if the SVD computation fails.
#[allow(private_bounds)]
pub fn svd<Id, Symm, T>(
    t: &TensorDynLen<Id, T, Symm>,
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
    T: SvdScalar,
    <T as ComplexFloat>::Real: ComplexFloat + Default + 'static,
{
    // Validate rank
    if t.dims.len() != 2 {
        return Err(SvdError::InvalidRank(t.dims.len()));
    }

    let m = t.dims[0];
    let n = t.dims[1];
    let k = m.min(n);

    // Get original indices
    let left_index = t.indices[0].clone();
    let right_index = t.indices[1].clone();

    // Extract data and create mdarray tensor
    // SVD destroys the input, so we need to clone the data
    let a_data = T::extract_dense(t.storage.as_ref())?;

    // Create mdarray tensor
    let mut a_tensor = tensor![[T::default(); n]; m];
    for i in 0..m {
        for j in 0..n {
            a_tensor[[i, j]] = a_data[i * n + j];
        }
    }

    // Call SVD using faer backend
    let bd = Faer;
    let a_slice: &mut Slice<T, (usize, usize), Dense> = a_tensor.as_mut();
    let SVDDecomp { s, u, vt } = bd.svd(a_slice)?;

    // Extract singular values and convert to real type.
    //
    // NOTE:
    // `mdarray-linalg-faer` writes singular values into a diagonal view created by
    // `into_faer_diag_mut`, which (by design) treats the **first row** as the
    // singular-value buffer (LAPACK-style convention). Therefore, the values live at
    // `s[0, i]`, not necessarily at `s[i, i]`.
    //
    // Singular values are always real, even for complex matrices.
    let mut s_vec: Vec<<T as ComplexFloat>::Real> = Vec::with_capacity(k);
    for i in 0..k {
        let s_val = s[[0, i]];
        s_vec.push(s_val.re());
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

    // Create U, S, V tensors
    let u_indices = vec![left_index.clone(), bond_index.clone()];
    let s_indices = vec![bond_index.clone(), bond_index.clone()];
    let v_indices = vec![right_index.clone(), bond_index.clone()];

    // Create storage using trait methods.
    // Note: S uses `T::Real` (singular values), not `T`.
    let u_storage = T::dense_storage(u_vec);
    let s_storage = T::diag_real_storage(s_vec);
    let v_storage = T::dense_storage(v_vec);

    let u = TensorDynLen::new(u_indices, vec![m, k], u_storage);
    let s = TensorDynLen::new(s_indices, vec![k, k], s_storage);
    let v = TensorDynLen::new(v_indices, vec![n, k], v_storage);

    Ok((u, s, v))
}

/// Compute SVD decomposition of a rank-2 complex tensor, returning (U, S, V).
///
/// This is a convenience wrapper around the generic `svd` function for `Complex64` tensors.
///
/// For complex-valued matrices, the mathematical convention is:
/// \[ A = U * Σ * V^H \]
/// where \(V^H\) is the conjugate-transpose of \(V\).
///
/// `mdarray-linalg` returns `vt` (conceptually \(V^T\) / \(V^H\) depending on scalar type),
/// and we return **V** (not Vt), so we build V by conjugate-transposing the leading k rows.
///
/// Note: Singular values `S` are always real (f64), even for complex input tensors.
#[inline]
pub fn svd_c64<Id, Symm>(
    t: &TensorDynLen<Id, Complex64, Symm>,
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
    svd(t)
}

