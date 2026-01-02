use std::sync::Arc;
use num_complex::Complex64;
use mdarray::{DenseMapping, View, DynRank, Shape, Dense};

/// Storage backend for tensor data.
/// Currently only DenseF64 and DenseC64 are supported.
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(Vec<f64>),
    DenseC64(Vec<Complex64>),
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
        Storage::DenseF64(Vec::with_capacity(capacity))
    }
}

impl DenseStorageFactory for Complex64 {
    fn new_dense(capacity: usize) -> Storage {
        Storage::DenseC64(Vec::with_capacity(capacity))
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
            Storage::DenseF64(v) => v.iter().copied().sum(),
            Storage::DenseC64(v) => v.iter().map(|z| z.re).sum(),
        }
    }
}

impl SumFromStorage for Complex64 {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(v) => Complex64::new(v.iter().copied().sum(), 0.0),
            Storage::DenseC64(v) => v.iter().copied().sum(),
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
            Storage::DenseF64(_) => AnyScalar::F64(f64::sum_from_storage(storage)),
            Storage::DenseC64(_) => AnyScalar::C64(Complex64::sum_from_storage(storage)),
        }
    }
}

impl Storage {
    /// Create a new DenseF64 storage with the given capacity.
    pub fn new_dense_f64(capacity: usize) -> Self {
        Self::DenseF64(Vec::with_capacity(capacity))
    }

    /// Create a new DenseC64 storage with the given capacity.
    pub fn new_dense_c64(capacity: usize) -> Self {
        Self::DenseC64(Vec::with_capacity(capacity))
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match self {
            Self::DenseF64(v) => v.len(),
            Self::DenseC64(v) => v.len(),
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
}

/// Helper to get a mutable reference to storage, cloning if needed (COW).
pub fn make_mut_storage(arc: &mut Arc<Storage>) -> &mut Storage {
    Arc::make_mut(arc)
}

/// Permute the dense storage data according to the given permutation.
///
/// This is an internal helper function that permutes the data in a `Storage`
/// according to the given dimensions and permutation.
///
/// # Arguments
/// * `storage` - The storage to permute
/// * `dims` - The original dimensions of the tensor
/// * `perm` - The permutation: new axis i corresponds to old axis `perm[i]`
///
/// # Panics
/// Panics if `perm.len() != dims.len()` or if the permutation is invalid.
pub fn permute_storage(storage: &Storage, dims: &[usize], perm: &[usize]) -> Storage {
    assert_eq!(
        perm.len(),
        dims.len(),
        "permutation length must match dimensions length"
    );

    match storage {
        Storage::DenseF64(vec) => {
            // Create mdarray shape from dimensions
            let shape = DynRank::from_dims(dims);
            let mapping = DenseMapping::new(shape);

            // Create a view over the vector data
            let view: View<'_, f64, DynRank, Dense> = unsafe {
                View::new_unchecked(vec.as_ptr(), mapping)
            };

            // Permute the view
            let permuted_view = view.into_permuted(perm);

            // Convert to tensor and extract vector
            let permuted_vec = permuted_view.to_tensor().into_vec();

            Storage::DenseF64(permuted_vec)
        }
        Storage::DenseC64(vec) => {
            // Create mdarray shape from dimensions
            let shape = DynRank::from_dims(dims);
            let mapping = DenseMapping::new(shape);

            // Create a view over the vector data
            let view: View<'_, Complex64, DynRank, Dense> = unsafe {
                View::new_unchecked(vec.as_ptr(), mapping)
            };

            // Permute the view
            let permuted_view = view.into_permuted(perm);

            // Convert to tensor and extract vector
            let permuted_vec = permuted_view.to_tensor().into_vec();

            Storage::DenseC64(permuted_vec)
        }
    }
}

