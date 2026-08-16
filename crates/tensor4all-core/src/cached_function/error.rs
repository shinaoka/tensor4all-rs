//! Error types for cache key operations.
//!
//! [`CacheKeyError`] is returned when constructing a [`CachedFunction`](crate::CachedFunction)
//! fails due to an index space that exceeds the key type's capacity.

use thiserror::Error;

/// Errors that can occur during cache key computation.
#[derive(Debug, Error)]
pub enum CacheKeyError {
    /// The index space requires more bits than the key type supports.
    #[error(
        "Cache key overflow: {total_bits} bits required, but {key_type} supports only \
         {max_bits} bits. Use CachedFunction::with_key_type::<LargerType>() to specify \
         a larger key type."
    )]
    Overflow {
        /// Total bits required by the index space.
        total_bits: u32,
        /// Maximum bits the key type supports.
        max_bits: u32,
        /// Name of the key type.
        key_type: &'static str,
    },

    /// Tensor has wrong number of dimensions for batch evaluation.
    #[error("Expected 2D tensor for batch evaluation, got {ndim}D")]
    InvalidTensorDim {
        /// Actual number of dimensions.
        ndim: usize,
    },
}
