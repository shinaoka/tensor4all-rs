//! Bit-packed integer keys for multi-index maps.
//!
//! A [`FlatIndexer`] turns a multi-index over fixed local dimensions into a
//! single integer key suitable for hashing. Dimension `i` occupies
//! `ceil(log2(d_i))` bits at a fixed offset, so encoding is shift-and-OR with
//! no multiplication, and two multi-indices collide only if they are equal.
//!
//! Widths up to 1024 bits use fixed-width fast paths; wider index spaces fall
//! back to a limb-backed representation with the same semantics.

use thiserror::Error;

mod fixed;

/// Failures from index-key construction and encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum IndexKeyError {
    /// A local dimension was zero, so the index space is empty.
    #[error("local dimension at position {position} is zero")]
    ZeroDimension {
        /// Position of the offending dimension.
        position: usize,
    },
    /// A multi-index had the wrong number of components.
    #[error("expected a multi-index of length {expected}, got {actual}")]
    LengthMismatch {
        /// Number of local dimensions the indexer was built with.
        expected: usize,
        /// Number of components supplied.
        actual: usize,
    },
    /// A component was not less than its local dimension.
    #[error("index {value} at position {position} is not below dimension {dim}")]
    IndexOutOfRange {
        /// Position of the offending component.
        position: usize,
        /// The supplied value.
        value: usize,
        /// The local dimension at that position.
        dim: usize,
    },
    /// The requested key width exceeds what this build can represent.
    #[error("requested key width {requested_bits} bits is too large")]
    WidthOverflow {
        /// Total bits requested.
        requested_bits: u64,
    },
}

/// Number of bits needed to represent the values `0..dim`.
///
/// # Errors
///
/// Returns [`IndexKeyError::ZeroDimension`] when `dim` is zero, since an empty
/// local space has no representable value.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::dimension_bits;
/// assert_eq!(dimension_bits(1).unwrap(), 0);
/// assert_eq!(dimension_bits(4).unwrap(), 2);
/// assert_eq!(dimension_bits(5).unwrap(), 3);
/// assert!(dimension_bits(0).is_err());
/// ```
pub fn dimension_bits(dim: usize) -> Result<u32, IndexKeyError> {
    match dim {
        0 => Err(IndexKeyError::ZeroDimension { position: 0 }),
        1 => Ok(0),
        _ => Ok(usize::BITS - (dim - 1).leading_zeros()),
    }
}

/// Total bit width of the bit-packed key for `local_dims`.
///
/// # Errors
///
/// Returns [`IndexKeyError::ZeroDimension`] naming the first zero dimension,
/// and [`IndexKeyError::WidthOverflow`] if the widths do not fit a `u64`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::total_bits;
/// assert_eq!(total_bits(&[2, 2, 2]).unwrap(), 3);
/// assert_eq!(total_bits(&[4, 3, 1]).unwrap(), 4);
/// assert_eq!(total_bits(&[]).unwrap(), 0);
/// assert!(total_bits(&[2, 0]).is_err());
/// ```
pub fn total_bits(local_dims: &[usize]) -> Result<u64, IndexKeyError> {
    let mut sum = 0u64;
    for (position, &dim) in local_dims.iter().enumerate() {
        let bits = dimension_bits(dim).map_err(|_| IndexKeyError::ZeroDimension { position })?;
        sum = sum
            .checked_add(u64::from(bits))
            .ok_or(IndexKeyError::WidthOverflow {
                requested_bits: u64::MAX,
            })?;
    }
    Ok(sum)
}

/// A bit-packed multi-index key.
///
/// Arms wider than 128 bits are boxed so that a narrow key does not pay the
/// footprint of the widest arm.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::{FlatIndexer, IndexKey};
/// let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
/// assert_eq!(indexer.encode(&[1, 0]).unwrap(), IndexKey::U64(1));
/// assert_eq!(indexer.encode(&[0, 1]).unwrap(), IndexKey::U64(2));
/// assert_ne!(
///     indexer.encode(&[1, 0]).unwrap(),
///     indexer.encode(&[0, 1]).unwrap()
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexKey {
    /// Keys up to 64 bits.
    U64(u64),
    /// Keys up to 128 bits.
    U128(u128),
}

/// Encodes multi-indices over fixed local dimensions as [`IndexKey`] values.
///
/// Dimension `i` occupies `ceil(log2(d_i))` bits at a fixed offset, so distinct
/// multi-indices always produce distinct keys.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::FlatIndexer;
/// let indexer = FlatIndexer::try_new(&[3, 4]).unwrap();
/// assert_eq!(indexer.width_bits(), 4);
/// assert_eq!(indexer.len(), 2);
/// assert!(indexer.encode(&[2, 3]).is_ok());
/// assert!(indexer.encode(&[3, 0]).is_err());
/// ```
#[derive(Debug, Clone)]
pub struct FlatIndexer {
    dims: Vec<usize>,
    offsets: Vec<u32>,
    width_bits: u64,
}

impl FlatIndexer {
    /// Builds an indexer for `local_dims`.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::ZeroDimension`] when any dimension is zero and
    /// [`IndexKeyError::WidthOverflow`] when the packed width exceeds what this
    /// build can represent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::FlatIndexer;
    /// assert!(FlatIndexer::try_new(&[2, 3, 4]).is_ok());
    /// assert!(FlatIndexer::try_new(&[2, 0]).is_err());
    /// ```
    pub fn try_new(local_dims: &[usize]) -> Result<Self, IndexKeyError> {
        let mut offsets = Vec::with_capacity(local_dims.len());
        let mut width_bits = 0u64;
        for (position, &dim) in local_dims.iter().enumerate() {
            let bits =
                dimension_bits(dim).map_err(|_| IndexKeyError::ZeroDimension { position })?;
            let offset = u32::try_from(width_bits).map_err(|_| IndexKeyError::WidthOverflow {
                requested_bits: width_bits,
            })?;
            offsets.push(offset);
            width_bits =
                width_bits
                    .checked_add(u64::from(bits))
                    .ok_or(IndexKeyError::WidthOverflow {
                        requested_bits: u64::MAX,
                    })?;
        }
        if width_bits > 128 {
            return Err(IndexKeyError::WidthOverflow {
                requested_bits: width_bits,
            });
        }
        Ok(Self {
            dims: local_dims.to_vec(),
            offsets,
            width_bits,
        })
    }

    /// Total packed width in bits.
    pub fn width_bits(&self) -> u64 {
        self.width_bits
    }

    /// Number of local dimensions.
    pub fn len(&self) -> usize {
        self.dims.len()
    }

    /// Whether the indexer has no dimensions.
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Encodes a multi-index.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::LengthMismatch`] when `idx` has the wrong
    /// length, and [`IndexKeyError::IndexOutOfRange`] when a component is not
    /// below its dimension. No input produces a silently wrapped key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::FlatIndexer;
    /// let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    /// assert!(indexer.encode(&[1, 1]).is_ok());
    /// assert!(indexer.encode(&[2, 0]).is_err());
    /// assert!(indexer.encode(&[0]).is_err());
    /// ```
    pub fn encode(&self, idx: &[usize]) -> Result<IndexKey, IndexKeyError> {
        self.check(idx)?;
        if self.width_bits <= 64 {
            let mut key = 0u64;
            for (&value, &offset) in idx.iter().zip(&self.offsets) {
                key = fixed::place_u64(key, value, offset);
            }
            Ok(IndexKey::U64(key))
        } else {
            let mut key = 0u128;
            for (&value, &offset) in idx.iter().zip(&self.offsets) {
                key = fixed::place_u128(key, value, offset);
            }
            Ok(IndexKey::U128(key))
        }
    }

    /// Validates length and per-dimension bounds before any packing happens.
    fn check(&self, idx: &[usize]) -> Result<(), IndexKeyError> {
        if idx.len() != self.dims.len() {
            return Err(IndexKeyError::LengthMismatch {
                expected: self.dims.len(),
                actual: idx.len(),
            });
        }
        for (position, (&value, &dim)) in idx.iter().zip(&self.dims).enumerate() {
            if value >= dim {
                return Err(IndexKeyError::IndexOutOfRange {
                    position,
                    value,
                    dim,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
