use std::sync::Arc;
use crate::index::Index;
use crate::storage::Storage;

/// Marker type: element type is dynamic (stored in `Storage`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnyScalar;

/// Tensor with dynamic rank (number of indices).
pub struct TensorDynLen<Id, T> {
    pub indices: Vec<Index<Id>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<Id, T> TensorDynLen<Id, T> {
    /// Create a new tensor with dynamic rank.
    ///
    /// # Panics
    /// Panics if `indices.len() != dims.len()`.
    pub fn new(indices: Vec<Index<Id>>, dims: Vec<usize>, storage: Arc<Storage>) -> Self {
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
}

/// Tensor with static rank `N` (number of indices).
pub struct TensorStaticLen<const N: usize, Id, T> {
    pub indices: [Index<Id>; N],
    pub dims: [usize; N],
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

impl<const N: usize, Id, T> TensorStaticLen<N, Id, T> {
    /// Create a new tensor with static rank `N`.
    pub fn new(indices: [Index<Id>; N], dims: [usize; N], storage: Arc<Storage>) -> Self {
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
}

/// Convenience alias for dynamic element type tensors with dynamic rank.
pub type AnyTensorDynLen<Id> = TensorDynLen<Id, AnyScalar>;

/// Convenience alias for dynamic element type tensors with static rank.
pub type AnyTensorStaticLen<const N: usize, Id> = TensorStaticLen<N, Id, AnyScalar>;

