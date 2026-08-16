//! Fixed-width bit-packed key arms.
//!
//! Each helper places one already-validated component into its reserved bit
//! field. The caller has checked that the value is below its local dimension,
//! so the shift cannot carry out of the field and no bits are lost.
//!
//! Every helper is `#[inline]`, and that is load-bearing rather than
//! decorative. These are called once per component in the encode loop, and
//! without the attribute they are only inlined when the compiler happens to
//! place them in the same codegen unit as the caller. Dissolving
//! `tensor4all-tcicore` into this crate (#642) grew it from roughly 25k to 33k
//! lines, which changed that partitioning and cost the `bnum` arms a factor of
//! about three — 2.13 to 6.04 ns per dimension at `U256` — with no change to
//! this code at all. The primitive arms survived only because a one-instruction
//! body hides the call overhead; that is luck, not a guarantee, so they carry
//! the attribute too.

/// Places `value` into `key` at `offset` bits.
#[inline]
pub(super) fn place_u64(key: u64, value: usize, offset: u32) -> u64 {
    key | ((value as u64) << offset)
}

/// Places `value` into `key` at `offset` bits.
#[inline]
pub(super) fn place_u128(key: u128, value: usize, offset: u32) -> u128 {
    key | ((value as u128) << offset)
}

macro_rules! place_bnum {
    ($name:ident, $ty:ty) => {
        /// Places `value` into `key` at `offset` bits.
        #[inline]
        pub(super) fn $name(key: $ty, value: usize, offset: u32) -> $ty {
            key | (<$ty>::from(value as u64) << offset)
        }
    };
}

place_bnum!(place_u256, bnum::types::U256);
place_bnum!(place_u512, bnum::types::U512);
place_bnum!(place_u1024, bnum::types::U1024);
