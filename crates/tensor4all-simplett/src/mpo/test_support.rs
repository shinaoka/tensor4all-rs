//! Shared fixtures for the MPO unit tests.

use super::mpo::MPO;
use super::types::tensor4_from_data;
use crate::einsum_helper::EinsumScalar;
use crate::mpo::factorize::SVDScalar;

/// Deterministic pseudo-random MPO with the given bond dimensions.
///
/// `bonds` lists the bond dimensions including the two trivial boundaries, so
/// `[1, 3, 1]` builds a two-site MPO. The site tensors are filled from a linear
/// congruential sequence seeded by `seed`, which makes the MPO generic: no site
/// tensor is orthogonal in either direction, which is exactly what the
/// canonicalization and truncation tests need.
pub(crate) fn random_mpo<T>(
    bonds: &[usize],
    s1: usize,
    s2: usize,
    seed: u64,
    from_f64: impl Fn(f64) -> T,
) -> MPO<T>
where
    T: SVDScalar + EinsumScalar,
{
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
    };

    let tensors = bonds
        .windows(2)
        .map(|w| {
            let (left, right) = (w[0], w[1]);
            let data: Vec<T> = (0..left * s1 * s2 * right)
                .map(|_| from_f64(next()))
                .collect();
            tensor4_from_data(data, left, s1, s2, right).expect("valid site tensor shape")
        })
        .collect();
    MPO::from_tensors_unchecked(tensors)
}
