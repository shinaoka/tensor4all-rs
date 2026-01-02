use std::sync::Arc;
use std::borrow::{Borrow, Cow};
use num_complex::Complex64;
use mdarray::{DenseMapping, View, DynRank, Shape, Dense, Slice};
use mdarray_linalg::{matmul::{MatMul, ContractBuilder}, Naive};

/// Dense storage for f64 elements.
#[derive(Debug, Clone)]
pub struct DenseStorageF64(Vec<f64>);

impl DenseStorageF64 {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn from_vec(vec: Vec<f64>) -> Self {
        Self(vec)
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }

    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn push(&mut self, val: f64) {
        self.0.push(val);
    }

    pub fn extend_from_slice(&mut self, other: &[f64]) {
        self.0.extend_from_slice(other);
    }

    pub fn extend<I: IntoIterator<Item = f64>>(&mut self, iter: I) {
        self.0.extend(iter);
    }

    pub fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    pub fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }

    /// Permute the dense storage data according to the given permutation.
    pub fn permute(&self, dims: &[usize], perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            dims.len(),
            "permutation length must match dimensions length"
        );

        // Create mdarray shape from dimensions
        let shape = DynRank::from_dims(dims);
        let mapping = DenseMapping::new(shape);

        // Create a view over the vector data
        let view: View<'_, f64, DynRank, Dense> = unsafe {
            View::new_unchecked(self.0.as_ptr(), mapping)
        };

        // Permute the view
        let permuted_view = view.into_permuted(perm);

        // Convert to tensor and extract vector
        let permuted_vec = permuted_view.to_tensor().into_vec();

        Self::from_vec(permuted_vec)
    }

    /// Contract this dense storage with another dense storage.
    pub fn contract(
        &self,
        dims: &[usize],
        axes: &[usize],
        other: &Self,
        other_dims: &[usize],
        other_axes: &[usize],
    ) -> Self {
        // Create mdarray views (which can be used as slices)
        let shape = DynRank::from_dims(dims);
        let mapping = DenseMapping::new(shape);
        let view: View<'_, f64, DynRank, Dense> = unsafe {
            View::new_unchecked(self.0.as_ptr(), mapping)
        };

        let other_shape = DynRank::from_dims(other_dims);
        let other_mapping = DenseMapping::new(other_shape);
        let other_view: View<'_, f64, DynRank, Dense> = unsafe {
            View::new_unchecked(other.0.as_ptr(), other_mapping)
        };

        // Contract using mdarray-linalg
        // View implements Borrow<Slice>, so we can use it directly
        let slice: &Slice<f64, DynRank, Dense> = view.borrow();
        let other_slice: &Slice<f64, DynRank, Dense> = other_view.borrow();

        let result = Naive
            .contract(
                slice,
                other_slice,
                axes.to_vec(),
                other_axes.to_vec(),
            )
            .eval();

        Self::from_vec(result.into_vec())
    }
}

/// Dense storage for Complex64 elements.
#[derive(Debug, Clone)]
pub struct DenseStorageC64(Vec<Complex64>);

impl DenseStorageC64 {
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub fn from_vec(vec: Vec<Complex64>) -> Self {
        Self(vec)
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.0
    }

    pub fn into_vec(self) -> Vec<Complex64> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn push(&mut self, val: Complex64) {
        self.0.push(val);
    }

    pub fn extend_from_slice(&mut self, other: &[Complex64]) {
        self.0.extend_from_slice(other);
    }

    pub fn extend<I: IntoIterator<Item = Complex64>>(&mut self, iter: I) {
        self.0.extend(iter);
    }

    pub fn get(&self, i: usize) -> Complex64 {
        self.0[i]
    }

    pub fn set(&mut self, i: usize, val: Complex64) {
        self.0[i] = val;
    }

    /// Permute the dense storage data according to the given permutation.
    pub fn permute(&self, dims: &[usize], perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            dims.len(),
            "permutation length must match dimensions length"
        );

        // Create mdarray shape from dimensions
        let shape = DynRank::from_dims(dims);
        let mapping = DenseMapping::new(shape);

        // Create a view over the vector data
        let view: View<'_, Complex64, DynRank, Dense> = unsafe {
            View::new_unchecked(self.0.as_ptr(), mapping)
        };

        // Permute the view
        let permuted_view = view.into_permuted(perm);

        // Convert to tensor and extract vector
        let permuted_vec = permuted_view.to_tensor().into_vec();

        Self::from_vec(permuted_vec)
    }

    /// Contract this dense storage with another dense storage.
    pub fn contract(
        &self,
        dims: &[usize],
        axes: &[usize],
        other: &Self,
        other_dims: &[usize],
        other_axes: &[usize],
    ) -> Self {
        // Create mdarray views (which can be used as slices)
        let shape = DynRank::from_dims(dims);
        let mapping = DenseMapping::new(shape);
        let view: View<'_, Complex64, DynRank, Dense> = unsafe {
            View::new_unchecked(self.0.as_ptr(), mapping)
        };

        let other_shape = DynRank::from_dims(other_dims);
        let other_mapping = DenseMapping::new(other_shape);
        let other_view: View<'_, Complex64, DynRank, Dense> = unsafe {
            View::new_unchecked(other.0.as_ptr(), other_mapping)
        };

        // Contract using mdarray-linalg
        // View implements Borrow<Slice>, so we can use it directly
        let slice: &Slice<Complex64, DynRank, Dense> = view.borrow();
        let other_slice: &Slice<Complex64, DynRank, Dense> = other_view.borrow();

        let result = Naive
            .contract(
                slice,
                other_slice,
                axes.to_vec(),
                other_axes.to_vec(),
            )
            .eval();

        Self::from_vec(result.into_vec())
    }
}

/// Diagonal storage for f64 elements.
#[derive(Debug, Clone)]
pub struct DiagStorageF64(Vec<f64>);

impl DiagStorageF64 {
    pub fn from_vec(vec: Vec<f64>) -> Self {
        Self(vec)
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.0
    }

    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: usize) -> f64 {
        self.0[i]
    }

    pub fn set(&mut self, i: usize, val: f64) {
        self.0[i] = val;
    }

    /// Convert diagonal storage to a dense vector representation.
    /// Creates a dense vector with diagonal elements set and off-diagonal elements as zero.
    pub fn to_dense_vec(&self, dims: &[usize]) -> Vec<f64> {
        let total_size: usize = dims.iter().product();
        let mut dense_vec = vec![0.0; total_size];
        let mindim_val = mindim(dims);
        
        // Set diagonal elements
        // For a tensor with indices [i, j, k, ...] where all have dimension d,
        // the diagonal elements are at positions where i == j == k == ...
        // The linear index for position (i, i, i, ...) is computed as:
        // i * (1 + stride_1 + stride_2 + ...)
        // For a tensor with dimensions [d, d, d, ...], the stride for dimension k is d^(rank - k - 1)
        let rank = dims.len();
        if rank == 0 {
            return vec![self.0[0]];
        }
        
        // Compute stride for each dimension
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        // Total stride for diagonal: sum of all strides
        let diag_stride: usize = strides.iter().sum();
        
        // Set diagonal elements
        for i in 0..mindim_val.min(self.0.len()) {
            let linear_idx = i * diag_stride;
            if linear_idx < total_size {
                dense_vec[linear_idx] = self.0[i];
            }
        }
        
        dense_vec
    }

    /// Contract this diagonal storage with another diagonal storage.
    /// Returns either a scalar (DenseStorageF64 with one element) or a diagonal storage.
    pub fn contract_diag_diag(
        &self,
        dims: &[usize],
        other: &Self,
        other_dims: &[usize],
        result_dims: &[usize],
    ) -> Storage {
        let mindim_a = mindim(dims);
        let mindim_b = mindim(other_dims);
        let min_len = mindim_a.min(mindim_b).min(self.0.len()).min(other.0.len());
        
        if result_dims.is_empty() {
            // All indices contracted: compute inner product (scalar result)
            let scalar: f64 = (0..min_len)
                .map(|i| self.0[i] * other.0[i])
                .sum();
            Storage::DenseF64(DenseStorageF64::from_vec(vec![scalar]))
        } else {
            // Some indices remain: element-wise product (DiagTensor result)
            let result_diag: Vec<f64> = (0..min_len)
                .map(|i| self.0[i] * other.0[i])
                .collect();
            Storage::DiagF64(DiagStorageF64::from_vec(result_diag))
        }
    }
}

/// Diagonal storage for Complex64 elements.
#[derive(Debug, Clone)]
pub struct DiagStorageC64(Vec<Complex64>);

impl DiagStorageC64 {
    pub fn from_vec(vec: Vec<Complex64>) -> Self {
        Self(vec)
    }

    pub fn as_slice(&self) -> &[Complex64] {
        &self.0
    }

    pub fn as_mut_slice(&mut self) -> &mut [Complex64] {
        &mut self.0
    }

    pub fn into_vec(self) -> Vec<Complex64> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn get(&self, i: usize) -> Complex64 {
        self.0[i]
    }

    pub fn set(&mut self, i: usize, val: Complex64) {
        self.0[i] = val;
    }

    /// Convert diagonal storage to a dense vector representation.
    /// Creates a dense vector with diagonal elements set and off-diagonal elements as zero.
    pub fn to_dense_vec(&self, dims: &[usize]) -> Vec<Complex64> {
        let total_size: usize = dims.iter().product();
        let mut dense_vec = vec![Complex64::new(0.0, 0.0); total_size];
        let mindim_val = mindim(dims);
        
        // Same logic as DiagStorageF64 but for complex
        let rank = dims.len();
        if rank == 0 {
            return vec![self.0[0]];
        }
        
        let mut strides = vec![1; rank];
        for i in (0..rank - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        
        let diag_stride: usize = strides.iter().sum();
        
        for i in 0..mindim_val.min(self.0.len()) {
            let linear_idx = i * diag_stride;
            if linear_idx < total_size {
                dense_vec[linear_idx] = self.0[i];
            }
        }
        
        dense_vec
    }

    /// Contract this diagonal storage with another diagonal storage.
    /// Returns either a scalar (DenseStorageC64 with one element) or a diagonal storage.
    pub fn contract_diag_diag(
        &self,
        dims: &[usize],
        other: &Self,
        other_dims: &[usize],
        result_dims: &[usize],
    ) -> Storage {
        let mindim_a = mindim(dims);
        let mindim_b = mindim(other_dims);
        let min_len = mindim_a.min(mindim_b).min(self.0.len()).min(other.0.len());
        
        if result_dims.is_empty() {
            // All indices contracted: compute inner product (scalar result)
            let scalar: Complex64 = (0..min_len)
                .map(|i| self.0[i] * other.0[i])
                .sum();
            Storage::DenseC64(DenseStorageC64::from_vec(vec![scalar]))
        } else {
            // Some indices remain: element-wise product (DiagTensor result)
            let result_diag: Vec<Complex64> = (0..min_len)
                .map(|i| self.0[i] * other.0[i])
                .collect();
            Storage::DiagC64(DiagStorageC64::from_vec(result_diag))
        }
    }
}

/// Storage backend for tensor data.
/// Supports Dense and Diag storage for f64 and Complex64 element types.
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(DenseStorageF64),
    DenseC64(DenseStorageC64),
    DiagF64(DiagStorageF64),
    DiagC64(DiagStorageC64),
}

/// Type-driven constructor for `Storage`.
///
/// This enables `<T as DenseStorageFactory>::new_dense(capacity)` which is
/// effectively `T::new_dense(capacity)` for scalar types `T` we support.
pub trait DenseStorageFactory {
    fn new_dense(capacity: usize) -> Storage;
}

impl DenseStorageFactory for f64 {
    fn new_dense(capacity: usize) -> Storage {
        Storage::DenseF64(DenseStorageF64::with_capacity(capacity))
    }
}

impl DenseStorageFactory for Complex64 {
    fn new_dense(capacity: usize) -> Storage {
        Storage::DenseC64(DenseStorageC64::with_capacity(capacity))
    }
}

/// Types that can be computed as the result of a reduction over `Storage`.
///
/// This lets callers write `let s: T = tensor.sum();` without matching on storage.
pub trait SumFromStorage: Sized {
    fn sum_from_storage(storage: &Storage) -> Self;
}

impl SumFromStorage for f64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DenseC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
            Storage::DiagF64(v) => v.as_slice().iter().copied().sum(),
            Storage::DiagC64(v) => v.as_slice().iter().map(|z| z.re).sum(),
        }
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            Storage::DenseC64(v) => v.as_slice().iter().copied().sum(),
            Storage::DiagF64(v) => Complex64::new(v.as_slice().iter().copied().sum(), 0.0),
            Storage::DiagC64(v) => v.as_slice().iter().copied().sum(),
        }
    }
}

/// Dynamic scalar value (for dynamic element type tensors).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyScalar {
    F64(f64),
    C64(Complex64),
}

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(_) | Storage::DiagF64(_) => AnyScalar::F64(f64::sum_from_storage(storage)),
            Storage::DenseC64(_) | Storage::DiagC64(_) => AnyScalar::C64(Complex64::sum_from_storage(storage)),
        }
    }
}

impl Storage {
    /// Create a new DenseF64 storage with the given capacity.
    pub fn new_dense_f64(capacity: usize) -> Self {
        Self::DenseF64(DenseStorageF64::with_capacity(capacity))
    }

    /// Create a new DenseC64 storage with the given capacity.
    pub fn new_dense_c64(capacity: usize) -> Self {
        Self::DenseC64(DenseStorageC64::with_capacity(capacity))
    }

    /// Create a new DiagF64 storage with the given diagonal data.
    pub fn new_diag_f64(diag_data: Vec<f64>) -> Self {
        Self::DiagF64(DiagStorageF64::from_vec(diag_data))
    }

    /// Create a new DiagC64 storage with the given diagonal data.
    pub fn new_diag_c64(diag_data: Vec<Complex64>) -> Self {
        Self::DiagC64(DiagStorageC64::from_vec(diag_data))
    }

    /// Check if this storage is a Diag storage type.
    pub fn is_diag(&self) -> bool {
        matches!(self, Self::DiagF64(_) | Self::DiagC64(_))
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match self {
            Self::DenseF64(v) => v.len(),
            Self::DenseC64(v) => v.len(),
            Self::DiagF64(v) => v.len(),
            Self::DiagC64(v) => v.len(),
        }
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(self)
    }

    /// Sum all elements as Complex64.
    pub fn sum_c64(&self) -> Complex64 {
        Complex64::sum_from_storage(self)
    }

    /// Convert this storage to dense storage.
    /// For Diag storage, creates a Dense storage with diagonal elements set
    /// and off-diagonal elements as zero.
    /// For Dense storage, returns a copy.
    pub fn to_dense_storage(&self, dims: &[usize]) -> Storage {
        match self {
            Storage::DenseF64(v) => Storage::DenseF64(DenseStorageF64::from_vec(v.as_slice().to_vec())),
            Storage::DenseC64(v) => Storage::DenseC64(DenseStorageC64::from_vec(v.as_slice().to_vec())),
            Storage::DiagF64(d) => Storage::DenseF64(DenseStorageF64::from_vec(d.to_dense_vec(dims))),
            Storage::DiagC64(d) => Storage::DenseC64(DenseStorageC64::from_vec(d.to_dense_vec(dims))),
        }
    }

    /// Permute the storage data according to the given permutation.
    pub fn permute_storage(&self, dims: &[usize], perm: &[usize]) -> Storage {
        match self {
            Storage::DenseF64(v) => Storage::DenseF64(v.permute(dims, perm)),
            Storage::DenseC64(v) => Storage::DenseC64(v.permute(dims, perm)),
            // For Diag storage, permute is trivial: data doesn't change, only index order changes
            Storage::DiagF64(v) => Storage::DiagF64(v.clone()),
            Storage::DiagC64(v) => Storage::DiagC64(v.clone()),
        }
    }
}

/// Helper to get a mutable reference to storage, cloning if needed (COW).
pub fn make_mut_storage(arc: &mut Arc<Storage>) -> &mut Storage {
    Arc::make_mut(arc)
}

/// Get the minimum dimension from a slice of dimensions.
/// This is used for DiagTensor where all indices must have the same dimension.
pub fn mindim(dims: &[usize]) -> usize {
    dims.iter().copied().min().unwrap_or(1)
}

/// Contract two storage tensors along specified axes.
///
/// This is an internal helper function that contracts two `Storage` tensors.
/// For Dense tensors, uses mdarray-linalg's contract method.
/// For Diag tensors, implements specialized diagonal contraction.
///
/// # Arguments
/// * `storage_a` - First tensor storage
/// * `dims_a` - Dimensions of the first tensor
/// * `axes_a` - Axes of the first tensor to contract
/// * `storage_b` - Second tensor storage
/// * `dims_b` - Dimensions of the second tensor
/// * `axes_b` - Axes of the second tensor to contract
/// * `result_dims` - Dimensions of the result tensor (empty for scalar result)
///
/// # Returns
/// A new `Storage` containing the contracted result.
///
/// # Panics
/// Panics if the contracted dimensions don't match, or if the storage types
/// are incompatible.
pub fn contract_storage(
    storage_a: &Storage,
    dims_a: &[usize],
    axes_a: &[usize],
    storage_b: &Storage,
    dims_b: &[usize],
    axes_b: &[usize],
    result_dims: &[usize],
) -> Storage {
    // Verify that contracted dimensions match
    for (a_axis, b_axis) in axes_a.iter().zip(axes_b.iter()) {
        assert_eq!(
            dims_a[*a_axis],
            dims_b[*b_axis],
            "Contracted dimensions must match: dims_a[{}] = {} != dims_b[{}] = {}",
            a_axis,
            dims_a[*a_axis],
            b_axis,
            dims_b[*b_axis]
        );
    }

    match (storage_a, storage_b) {
        (Storage::DenseF64(a), Storage::DenseF64(b)) => {
            Storage::DenseF64(a.contract(dims_a, axes_a, b, dims_b, axes_b))
        }
        (Storage::DenseC64(a), Storage::DenseC64(b)) => {
            Storage::DenseC64(a.contract(dims_a, axes_a, b, dims_b, axes_b))
        }
        // DiagTensor × DiagTensor contraction
        (Storage::DiagF64(a), Storage::DiagF64(b)) => {
            a.contract_diag_diag(dims_a, b, dims_b, result_dims)
        }
        (Storage::DiagC64(a), Storage::DiagC64(b)) => {
            a.contract_diag_diag(dims_a, b, dims_b, result_dims)
        }
        // DiagTensor × DenseTensor: convert Diag to Dense first
        (Storage::DiagF64(_), Storage::DenseF64(_)) | (Storage::DiagC64(_), Storage::DenseC64(_)) => {
            let dense_a = storage_a.to_dense_storage(dims_a);
            contract_storage(&dense_a, dims_a, axes_a, storage_b, dims_b, axes_b, result_dims)
        }
        (Storage::DenseF64(_), Storage::DiagF64(_)) | (Storage::DenseC64(_), Storage::DiagC64(_)) => {
            let dense_b = storage_b.to_dense_storage(dims_b);
            contract_storage(storage_a, dims_a, axes_a, &dense_b, dims_b, axes_b, result_dims)
        }
        _ => panic!("Storage types must be compatible for contraction"),
    }
}

/// Scalar types that can be extracted from and stored in `Storage`.
///
/// This trait provides conversion methods between scalar types and `Storage`,
/// supporting both view-based (borrowed) and owned operations.
///
/// # View-based operations
/// - `extract_dense_view`: Returns a borrowed slice (no copy)
/// - `extract_dense_cow`: Returns `Cow` (borrowed if possible, owned if needed)
///
/// # Owned operations
/// - `extract_dense`: Returns owned `Vec` (always copies)
/// - `dense_storage`: Creates `Storage` from owned `Vec`
pub trait StorageScalar: Copy + 'static {
    /// Extract a borrowed view of dense storage data (no copy).
    ///
    /// Returns an error if the storage is not the matching dense type.
    fn extract_dense_view<'a>(storage: &'a Storage) -> Result<&'a [Self], String>;

    /// Extract dense storage data as `Cow` (borrowed if possible, owned if needed).
    ///
    /// For dense storage, returns `Cow::Borrowed` (no copy).
    /// For other storage types, may need to convert to dense first (copy).
    fn extract_dense_cow<'a>(storage: &'a Storage) -> Result<Cow<'a, [Self]>, String> {
        Self::extract_dense_view(storage).map(Cow::Borrowed)
    }

    /// Extract dense storage data as owned `Vec` (always copies).
    ///
    /// This is a convenience method that calls `extract_dense_cow` and converts to owned.
    fn extract_dense(storage: &Storage) -> Result<Vec<Self>, String> {
        Ok(Self::extract_dense_cow(storage)?.into_owned())
    }

    /// Create `Storage` from owned dense data.
    fn dense_storage(data: Vec<Self>) -> Arc<Storage>;
}

impl StorageScalar for f64 {
    fn extract_dense_view<'a>(storage: &'a Storage) -> Result<&'a [Self], String> {
        match storage {
            Storage::DenseF64(ds) => Ok(ds.as_slice()),
            _ => Err(format!("Expected DenseF64 storage, got {:?}", storage)),
        }
    }

    fn dense_storage(data: Vec<Self>) -> Arc<Storage> {
        Arc::new(Storage::DenseF64(DenseStorageF64::from_vec(data)))
    }
}

impl StorageScalar for Complex64 {
    fn extract_dense_view<'a>(storage: &'a Storage) -> Result<&'a [Self], String> {
        match storage {
            Storage::DenseC64(ds) => Ok(ds.as_slice()),
            _ => Err(format!("Expected DenseC64 storage, got {:?}", storage)),
        }
    }

    fn dense_storage(data: Vec<Self>) -> Arc<Storage> {
        Arc::new(Storage::DenseC64(DenseStorageC64::from_vec(data)))
    }
}

