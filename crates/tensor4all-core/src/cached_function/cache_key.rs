//! Cache key trait for flat-index computation.
//!
//! Built-in implementations: `u64`, `u128`, `U256`, `U512`, `U1024`.
//!
//! # Custom key types
//!
//! To support index spaces larger than 1024 bits, implement `CacheKey` for a
//! wider integer type and pass it via `CachedFunction::with_key_type`:
//!
//! ```
//! use bnum::types::U2048;
//! use tensor4all_core::{CacheKey, CachedFunction};
//!
//! #[derive(Clone, Hash, PartialEq, Eq)]
//! struct U2048Key(U2048);
//!
//! impl CacheKey for U2048Key {
//!     const BITS_COUNT: u32 = 2048;
//!     const ZERO: Self = Self(U2048::ZERO);
//!     const ONE: Self = Self(U2048::ONE);
//!     fn from_usize(v: usize) -> Self { Self(U2048::from(v as u64)) }
//!     fn checked_mul(self, rhs: Self) -> Option<Self> {
//!         self.0.checked_mul(rhs.0).map(Self)
//!     }
//!     fn wrapping_add(self, rhs: Self) -> Self {
//!         Self(self.0.wrapping_add(rhs.0))
//!     }
//! }
//!
//! let local_dims = vec![2usize; 1025];
//! let cf = CachedFunction::with_key_type::<U2048Key>(
//!     |idx: &[usize]| idx.iter().sum::<usize>(),
//!     &local_dims,
//! ).unwrap();
//! let zeros = vec![0usize; 1025];
//!
//! assert_eq!(cf.eval(&zeros), 0);
//! assert_eq!(cf.key_type(), "custom");
//! ```

use std::hash::Hash;

use bnum::types::{U1024, U256, U512};

/// Trait for cache key types used in flat-index computation.
///
/// Built-in implementations are provided for `u64`, `u128`, `U256`, `U512`,
/// and `U1024`. For index spaces larger than 1024 bits, implement this trait
/// for a wider integer type and use
/// [`CachedFunction::with_key_type`](crate::CachedFunction::with_key_type).
///
/// See module documentation for a complete custom key example.
pub trait CacheKey: Hash + Eq + Clone + Send + Sync + 'static {
    /// Number of bits this key type can represent.
    const BITS_COUNT: u32;
    /// The zero value.
    const ZERO: Self;
    /// The one value.
    const ONE: Self;

    /// Convert a `usize` to this key type.
    fn from_usize(v: usize) -> Self;

    /// Checked multiplication. Returns `None` on overflow.
    fn checked_mul(self, rhs: Self) -> Option<Self>;

    /// Wrapping addition (overflow wraps around).
    fn wrapping_add(self, rhs: Self) -> Self;
}

impl CacheKey for u64 {
    const BITS_COUNT: u32 = 64;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn from_usize(v: usize) -> Self {
        v as u64
    }

    fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.checked_mul(rhs)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl CacheKey for u128 {
    const BITS_COUNT: u32 = 128;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    fn from_usize(v: usize) -> Self {
        v as u128
    }

    fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.checked_mul(rhs)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

macro_rules! impl_cache_key_bnum {
    ($ty:ty, $bits:expr) => {
        impl CacheKey for $ty {
            const BITS_COUNT: u32 = $bits;
            const ZERO: Self = <$ty>::ZERO;
            const ONE: Self = <$ty>::ONE;

            fn from_usize(v: usize) -> Self {
                <$ty>::from(v as u64)
            }

            fn checked_mul(self, rhs: Self) -> Option<Self> {
                self.checked_mul(rhs)
            }

            fn wrapping_add(self, rhs: Self) -> Self {
                self.wrapping_add(rhs)
            }
        }
    };
}

impl_cache_key_bnum!(U256, 256);
impl_cache_key_bnum!(U512, 512);
impl_cache_key_bnum!(U1024, 1024);
