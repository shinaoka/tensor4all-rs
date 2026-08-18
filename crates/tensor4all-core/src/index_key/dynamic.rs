//! Limb-backed keys for index spaces wider than the fixed-width arms.
//!
//! Limbs are little-endian 64-bit words. Placing a component writes into at
//! most two adjacent limbs, because a single local dimension never needs more
//! than 64 bits, so encoding stays linear in the number of components and the
//! key stays linear in the number of limbs.

use smallvec::SmallVec;

use super::IndexKeyError;

/// Little-endian `u64` limbs.
pub(super) type Limbs = SmallVec<[u64; 2]>;

/// Number of limbs needed to hold `width_bits`.
///
/// Saturates rather than truncating: a width whose limb count does not fit
/// `usize` cannot be allocated anyway, and [`try_limb_count`] is the checked
/// form every allocation path goes through.
#[inline]
pub(super) fn limb_count(width_bits: u64) -> usize {
    usize::try_from(width_bits.div_ceil(64)).unwrap_or(usize::MAX)
}

/// Number of limbs needed to hold `width_bits`, or an error if that count does
/// not fit `usize`.
///
/// On a 32-bit target `u64 -> usize` can truncate, which would allocate too few
/// limbs and turn a width the API should reject into a later index-out-of-bounds
/// panic. Every allocation path goes through this.
#[inline]
pub(super) fn try_limb_count(width_bits: u64) -> Result<usize, IndexKeyError> {
    usize::try_from(width_bits.div_ceil(64)).map_err(|_| IndexKeyError::WidthOverflow {
        requested_bits: width_bits,
    })
}

/// Allocates zeroed limbs for `width_bits`.
///
/// # Errors
///
/// Returns [`IndexKeyError::WidthOverflow`] when the limb count does not fit
/// `usize` on this target.
pub(super) fn zeroed(width_bits: u64) -> Result<Limbs, IndexKeyError> {
    Ok(smallvec::smallvec![0u64; try_limb_count(width_bits)?])
}

/// Copies `source`'s low `width_bits` bits into `limbs` starting at `offset`.
///
/// The caller has checked that `offset + width_bits` fits `limbs`, so no bits
/// are dropped. Cost is linear in the number of limbs the source occupies,
/// which is what makes append-style tree composition linear rather than
/// quadratic.
///
/// The final partial limb is masked to `width_bits`. A key produced by this
/// module never carries bits above its own width, so the mask is redundant
/// today; it is what keeps a stray high bit from silently landing in the next
/// field rather than corrupting the composition.
pub(super) fn place_limbs(limbs: &mut Limbs, source: &[u64], width_bits: u64, offset: u64) {
    let word = (offset / 64) as usize;
    let shift = (offset % 64) as u32;
    let used = limb_count(width_bits).min(source.len());
    let tail_bits = (width_bits % 64) as u32;
    for (position, &digit) in source.iter().take(used).enumerate() {
        let digit = if tail_bits != 0 && position + 1 == used {
            digit & ((1u64 << tail_bits) - 1)
        } else {
            digit
        };
        if digit == 0 {
            continue;
        }
        limbs[word + position] |= digit << shift;
        if shift != 0 {
            let carry = digit >> (64 - shift);
            if carry != 0 {
                limbs[word + position + 1] |= carry;
            }
        }
    }
}

/// Writes `value` into `limbs` starting at bit `offset`.
///
/// The caller has checked that `value` is below its local dimension and that
/// the field fits within `limbs`, so no bits are lost.
#[inline]
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
