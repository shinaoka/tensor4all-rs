use std::sync::Arc;

/// Storage backend for tensor data.
/// Currently only DenseF64 is supported.
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(Vec<f64>),
}

impl Storage {
    /// Create a new DenseF64 storage with the given capacity.
    pub fn new_dense_f64(capacity: usize) -> Self {
        Self::DenseF64(Vec::with_capacity(capacity))
    }

    /// Get the length of the storage (number of elements).
    pub fn len(&self) -> usize {
        match self {
            Self::DenseF64(v) => v.len(),
        }
    }
}

/// Helper to get a mutable reference to storage, cloning if needed (COW).
pub fn make_mut_storage(arc: &mut Arc<Storage>) -> &mut Storage {
    Arc::make_mut(arc)
}

