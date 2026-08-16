//! Fixed-width bit-packed key arms.
//!
//! Each helper places one already-validated component into its reserved bit
//! field. The caller has checked that the value is below its local dimension,
//! so the shift cannot carry out of the field and no bits are lost.

/// Places `value` into `key` at `offset` bits.
pub(super) fn place_u64(key: u64, value: usize, offset: u32) -> u64 {
    key | ((value as u64) << offset)
}

/// Places `value` into `key` at `offset` bits.
pub(super) fn place_u128(key: u128, value: usize, offset: u32) -> u128 {
    key | ((value as u128) << offset)
}
