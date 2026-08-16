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

mod dynamic;
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
    /// Keys up to 256 bits.
    U256(Box<bnum::types::U256>),
    /// Keys up to 512 bits.
    U512(Box<bnum::types::U512>),
    /// Keys up to 1024 bits.
    U1024(Box<bnum::types::U1024>),
    /// Keys wider than 1024 bits, as little-endian 64-bit limbs.
    Limbs(dynamic::Limbs),
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
    offsets: Vec<u64>,
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
            offsets.push(width_bits);
            width_bits =
                width_bits
                    .checked_add(u64::from(bits))
                    .ok_or(IndexKeyError::WidthOverflow {
                        requested_bits: u64::MAX,
                    })?;
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
        macro_rules! pack {
            ($zero:expr, $place:path, $arm:expr) => {{
                let mut key = $zero;
                for (&value, &offset) in idx.iter().zip(&self.offsets) {
                    // Offsets are below the arm's width here, so this cannot
                    // truncate; the conversion is checked rather than cast so
                    // that a future width change cannot silently wrap.
                    let offset =
                        u32::try_from(offset).map_err(|_| IndexKeyError::WidthOverflow {
                            requested_bits: self.width_bits,
                        })?;
                    key = $place(key, value, offset);
                }
                Ok($arm(key))
            }};
        }
        match self.width_bits {
            0..=64 => pack!(0u64, fixed::place_u64, IndexKey::U64),
            65..=128 => pack!(0u128, fixed::place_u128, IndexKey::U128),
            129..=256 => pack!(bnum::types::U256::ZERO, fixed::place_u256, |k| {
                IndexKey::U256(Box::new(k))
            }),
            257..=512 => pack!(bnum::types::U512::ZERO, fixed::place_u512, |k| {
                IndexKey::U512(Box::new(k))
            }),
            513..=1024 => pack!(bnum::types::U1024::ZERO, fixed::place_u1024, |k| {
                IndexKey::U1024(Box::new(k))
            }),
            _ => {
                let mut limbs = dynamic::zeroed(self.width_bits);
                for (&value, &offset) in idx.iter().zip(&self.offsets) {
                    dynamic::place(&mut limbs, value, offset);
                }
                Ok(IndexKey::Limbs(limbs))
            }
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

/// Assembles one key by appending sub-keys at successive bit offsets.
///
/// This is the composition operation a tree needs: a node's key is its own
/// local key followed by each child's key, so
/// `key(node) = local ++ key(c1) ++ key(c2) ++ ...`. Appending at a known bit
/// offset costs one pass over the pushed key's limbs, so composing a tree is
/// linear in total width rather than quadratic as a multiply-based mixed-radix
/// composition would be.
///
/// The result equals what [`FlatIndexer::encode`] produces for the
/// concatenated multi-index, so a composed key and a directly encoded one are
/// interchangeable as map keys.
///
/// # Examples
///
/// ```
/// use tensor4all_core::index_key::{FlatIndexer, KeyBuilder};
/// let local = FlatIndexer::try_new(&[3, 2]).unwrap();
/// let child = FlatIndexer::try_new(&[4]).unwrap();
/// let whole = FlatIndexer::try_new(&[3, 2, 4]).unwrap();
///
/// let mut builder = KeyBuilder::with_capacity_bits(whole.width_bits()).unwrap();
/// builder.push(&local.encode(&[2, 1]).unwrap(), local.width_bits()).unwrap();
/// builder.push(&child.encode(&[3]).unwrap(), child.width_bits()).unwrap();
///
/// assert_eq!(builder.finish(), whole.encode(&[2, 1, 3]).unwrap());
/// ```
#[derive(Debug, Clone)]
pub struct KeyBuilder {
    limbs: dynamic::Limbs,
    offset: u64,
    capacity_bits: u64,
}

impl KeyBuilder {
    /// Creates a builder able to hold `width_bits` of appended sub-keys.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::WidthOverflow`] when the width cannot be
    /// allocated on this platform.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::KeyBuilder;
    /// assert!(KeyBuilder::with_capacity_bits(0).is_ok());
    /// assert!(KeyBuilder::with_capacity_bits(4096).is_ok());
    /// ```
    pub fn with_capacity_bits(width_bits: u64) -> Result<Self, IndexKeyError> {
        usize::try_from(width_bits.div_ceil(64)).map_err(|_| IndexKeyError::WidthOverflow {
            requested_bits: width_bits,
        })?;
        Ok(Self {
            limbs: dynamic::zeroed(width_bits),
            offset: 0,
            capacity_bits: width_bits,
        })
    }

    /// Total bits appended so far.
    pub fn width_bits(&self) -> u64 {
        self.offset
    }

    /// Appends `key`, which occupies `key_width_bits`, at the current offset.
    ///
    /// # Errors
    ///
    /// Returns [`IndexKeyError::WidthOverflow`] when the append would exceed
    /// the declared capacity, rather than truncating the key.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::{FlatIndexer, KeyBuilder};
    /// let indexer = FlatIndexer::try_new(&[2, 2]).unwrap();
    /// let key = indexer.encode(&[1, 1]).unwrap();
    /// let mut builder = KeyBuilder::with_capacity_bits(2).unwrap();
    /// assert!(builder.push(&key, 2).is_ok());
    /// assert!(builder.push(&key, 2).is_err());
    /// ```
    pub fn push(&mut self, key: &IndexKey, key_width_bits: u64) -> Result<(), IndexKeyError> {
        let end = self
            .offset
            .checked_add(key_width_bits)
            .ok_or(IndexKeyError::WidthOverflow {
                requested_bits: u64::MAX,
            })?;
        if end > self.capacity_bits {
            return Err(IndexKeyError::WidthOverflow {
                requested_bits: end,
            });
        }
        let offset = self.offset;
        match key {
            IndexKey::U64(value) => {
                dynamic::place_limbs(&mut self.limbs, &[*value], key_width_bits, offset);
            }
            IndexKey::U128(value) => {
                let source = [*value as u64, (*value >> 64) as u64];
                dynamic::place_limbs(&mut self.limbs, &source, key_width_bits, offset);
            }
            IndexKey::U256(value) => {
                dynamic::place_limbs(&mut self.limbs, value.digits(), key_width_bits, offset);
            }
            IndexKey::U512(value) => {
                dynamic::place_limbs(&mut self.limbs, value.digits(), key_width_bits, offset);
            }
            IndexKey::U1024(value) => {
                dynamic::place_limbs(&mut self.limbs, value.digits(), key_width_bits, offset);
            }
            IndexKey::Limbs(value) => {
                dynamic::place_limbs(&mut self.limbs, value, key_width_bits, offset);
            }
        }
        self.offset = end;
        Ok(())
    }

    /// Narrows the accumulated limbs to the smallest arm holding the capacity.
    ///
    /// Narrowing is what makes a composed key compare equal to the same value
    /// produced by [`FlatIndexer::encode`], which also picks its arm from the
    /// declared width.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::index_key::{IndexKey, KeyBuilder};
    /// let builder = KeyBuilder::with_capacity_bits(8).unwrap();
    /// assert_eq!(builder.finish(), IndexKey::U64(0));
    /// ```
    pub fn finish(self) -> IndexKey {
        fn digit(limbs: &[u64], position: usize) -> u64 {
            limbs.get(position).copied().unwrap_or(0)
        }
        fn digits<const N: usize>(limbs: &[u64]) -> [u64; N] {
            let mut out = [0u64; N];
            for (position, slot) in out.iter_mut().enumerate() {
                *slot = digit(limbs, position);
            }
            out
        }
        match self.capacity_bits {
            0..=64 => IndexKey::U64(digit(&self.limbs, 0)),
            65..=128 => IndexKey::U128(
                u128::from(digit(&self.limbs, 0)) | (u128::from(digit(&self.limbs, 1)) << 64),
            ),
            129..=256 => IndexKey::U256(Box::new(bnum::types::U256::from_digits(digits::<4>(
                &self.limbs,
            )))),
            257..=512 => IndexKey::U512(Box::new(bnum::types::U512::from_digits(digits::<8>(
                &self.limbs,
            )))),
            513..=1024 => IndexKey::U1024(Box::new(bnum::types::U1024::from_digits(digits::<16>(
                &self.limbs,
            )))),
            _ => IndexKey::Limbs(self.limbs),
        }
    }
}

#[cfg(test)]
mod tests;
