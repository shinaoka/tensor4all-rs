//! TensorIndex trait for index operations on tensor-like objects.
//!
//! This trait provides a minimal interface for objects that have external indices
//! and support index replacement operations. It is a subset of `TensorLike` that
//! can be implemented by both dense tensors and tensor networks (like TreeTN).

use crate::IndexLike;
use std::fmt::Debug;

/// Trait for objects that have external indices and support index operations.
///
/// This is a minimal trait that can be implemented by:
/// - Dense tensors (`IdxTensor`)
/// - Tensor networks (`TreeTN`)
/// - Any other structure that organizes tensors with indices
///
/// # Design
///
/// This trait is separate from `TensorLike` to allow tensor networks to implement
/// index operations without needing to implement contraction/factorization operations.
pub trait TensorIndex: Sized + Clone + Debug + Send + Sync {
    /// The index type used by this object.
    type Index: IndexLike;

    /// Error type for fallible index operations.
    ///
    /// Implementations map their internal diagnostics into a typed,
    /// source-preserving error; callers propagating into `anyhow::Result`
    /// keep working through `From`.
    type Error: std::error::Error + Send + Sync + 'static + From<anyhow::Error>;

    /// Return flattened external indices for this object.
    ///
    /// # Ordering
    ///
    /// The ordering MUST be stable (deterministic). Implementations should:
    /// - Sort indices by their full index identity, or
    /// - Use insertion-ordered storage
    ///
    /// This ensures consistent behavior for hashing, serialization, and comparison.
    fn external_indices(&self) -> Vec<Self::Index>;

    /// Number of external indices.
    ///
    /// Default implementation calls `external_indices().len()`, but implementations
    /// SHOULD override this for efficiency when the count can be computed without
    /// allocating the full index list.
    fn num_external_indices(&self) -> usize {
        self.external_indices().len()
    }

    /// Replace an index in this object.
    ///
    /// This replaces the index equal to `old_index` (including dimension,
    /// prime level, tags, and other identity metadata) with `new_index`.
    /// The storage data is not modified, only the index metadata is changed.
    ///
    /// # Arguments
    ///
    /// * `old_index` - The full index identity to replace
    /// * `new_index` - The new index to use
    ///
    /// # Returns
    ///
    /// A new object with the index replaced.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when `old_index` is not present (a
    /// missing-index failure) or the replacement violates an index identity
    /// invariant (an invalid replacement).
    fn replaceind(
        &self,
        old_index: &Self::Index,
        new_index: &Self::Index,
    ) -> std::result::Result<Self, Self::Error>;

    /// Replace multiple indices in this object.
    ///
    /// This replaces each full index identity in `old_indices` with the
    /// corresponding index in `new_indices`. The storage data is not modified.
    ///
    /// # Arguments
    ///
    /// * `old_indices` - The full index identities to replace
    /// * `new_indices` - The new indices to use
    ///
    /// # Returns
    ///
    /// A new object with the indices replaced.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when an `old_indices` entry is not present
    /// (a missing-index failure), when `old_indices` and `new_indices` differ
    /// in length (a length mismatch), or when the replacement violates an
    /// index identity invariant (an invalid replacement).
    fn replace_indices(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> std::result::Result<Self, Self::Error>;

    /// Replace indices using pairs of (old, new).
    ///
    /// This is a convenience method that wraps `replace_indices`.
    ///
    /// # Arguments
    ///
    /// * `pairs` - Pairs of (old_index, new_index) to replace
    ///
    /// # Returns
    ///
    /// A new object with the indices replaced.
    ///
    /// # Errors
    ///
    /// Returns `Self::Error` when an `old` entry is not present (a
    /// missing-index failure) or when the replacement violates an index
    /// identity invariant (an invalid replacement); propagates failures from
    /// [`Self::replace_indices`].
    fn replace_indices_pairs(
        &self,
        pairs: &[(Self::Index, Self::Index)],
    ) -> std::result::Result<Self, Self::Error> {
        let (old, new): (Vec<_>, Vec<_>) = pairs.iter().cloned().unzip();
        self.replace_indices(&old, &new)
    }
}
