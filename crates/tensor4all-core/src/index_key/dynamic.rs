//! Limb-backed keys for index spaces wider than the fixed-width arms.
//!
//! Limbs are little-endian 64-bit words. Placing a component writes into at
//! most two adjacent limbs, because a single local dimension never needs more
//! than 64 bits, so encoding stays linear in the number of components and the
//! key stays linear in the number of limbs.

use smallvec::SmallVec;

/// Little-endian `u64` limbs.
pub(super) type Limbs = SmallVec<[u64; 2]>;

/// Number of limbs needed to hold `width_bits`.
pub(super) fn limb_count(width_bits: u64) -> usize {
    width_bits.div_ceil(64) as usize
}

/// Allocates zeroed limbs for `width_bits`.
pub(super) fn zeroed(width_bits: u64) -> Limbs {
    smallvec::smallvec![0u64; limb_count(width_bits)]
}

/// Writes `value` into `limbs` starting at bit `offset`.
///
/// The caller has checked that `value` is below its local dimension and that
/// the field fits within `limbs`, so no bits are lost.
pub(super) fn place(limbs: &mut Limbs, value: usize, offset: u64) {
    let word = (offset / 64) as usize;
    let shift = (offset % 64) as u32;
    let value = value as u64;
    limbs[word] |= value << shift;
    if shift != 0 {
        let carry_bits = 64 - shift;
        let carry = value >> carry_bits;
        if carry != 0 {
            limbs[word + 1] |= carry;
        }
    }
}
