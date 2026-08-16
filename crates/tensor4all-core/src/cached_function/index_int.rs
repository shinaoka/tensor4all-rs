//! Index integer trait for multi-index elements.
//!
//! Use smaller types (e.g., `u8`) for quantics TCI where `local_dim = 2`.

/// Trait for index element types.
///
/// All indices within a single `CachedFunction` must use the same integer type.
/// Implemented for `u8`, `u16`, `u32`, and `usize`. Use `u8` for quantics TCI
/// where `local_dim = 2`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{IndexInt, CachedFunction};
///
/// // Using u8 indices for quantics (local_dim = 2)
/// let cf = CachedFunction::new(
///     |idx: &[u8]| idx.iter().map(|&x| x as usize).sum::<usize>(),
///     &[2, 2, 2],
/// ).unwrap();
/// assert_eq!(cf.eval(&[1u8, 0, 1]), 2);
/// ```
pub trait IndexInt: Copy + Send + Sync + 'static {
    /// Convert this index element to `usize`.
    fn to_usize(self) -> usize;
}

impl IndexInt for u8 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for u16 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for u32 {
    fn to_usize(self) -> usize {
        self as usize
    }
}

impl IndexInt for usize {
    fn to_usize(self) -> usize {
        self
    }
}
