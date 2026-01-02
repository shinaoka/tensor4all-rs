use std::sync::Arc;
use crate::index::{Index, NoSymmSpace};
use crate::storage::{AnyScalar, Storage, SumFromStorage};

/// Tensor with dynamic rank (number of indices).
pub struct TensorDynLen<Id, T, Symm = NoSymmSpace> {
    pub indices: Vec<Index<Id, Symm>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<Id, T, Symm> TensorDynLen<Id, T, Symm> {
    /// Create a new tensor with dynamic rank.
    ///
    /// # Panics
    /// Panics if `indices.len() != dims.len()`.
    pub fn new(indices: Vec<Index<Id, Symm>>, dims: Vec<usize>, storage: Arc<Storage>) -> Self {
        assert_eq!(
            indices.len(),
            dims.len(),
            "indices and dims must have the same length"
        );
        Self {
            indices,
            dims,
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `T`.
    ///
    /// For dynamic element tensors, use `T = AnyScalar`.
    pub fn sum(&self) -> T
    where
        T: SumFromStorage,
    {
        T::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
    }
}

/// Tensor with static rank `N` (number of indices).
pub struct TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace> {
    pub indices: [Index<Id, Symm>; N],
    pub dims: [usize; N],
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<const N: usize, Id, T, Symm> TensorStaticLen<N, Id, T, Symm> {
    /// Create a new tensor with static rank `N`.
    pub fn new(indices: [Index<Id, Symm>; N], dims: [usize; N], storage: Arc<Storage>) -> Self {
        Self {
            indices,
            dims,
            storage,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get a mutable reference to storage (COW: clones if shared).
    pub fn storage_mut(&mut self) -> &mut Storage {
        Arc::make_mut(&mut self.storage)
    }

    /// Sum all elements, returning `T`.
    ///
    /// For dynamic element tensors, use `T = AnyScalar`.
    pub fn sum(&self) -> T
    where
        T: SumFromStorage,
    {
        T::sum_from_storage(&self.storage)
    }

    /// Sum all elements as f64.
    pub fn sum_f64(&self) -> f64 {
        f64::sum_from_storage(&self.storage)
    }
}

/// Convenience alias for dynamic element type tensors with dynamic rank.
pub type AnyTensorDynLen<Id, Symm = NoSymmSpace> = TensorDynLen<Id, AnyScalar, Symm>;

/// Convenience alias for dynamic element type tensors with static rank.
pub type AnyTensorStaticLen<const N: usize, Id, Symm = NoSymmSpace> = TensorStaticLen<N, Id, AnyScalar, Symm>;

