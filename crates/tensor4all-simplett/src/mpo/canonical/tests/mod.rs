use super::*;

use crate::mpo::test_support::random_mpo;
use num_complex::{Complex64, ComplexFloat};

const SEED: u64 = 0x2545_F491_4F6C_DD1D;

fn assert_same_operator<T>(before: &MPO<T>, after: &MPO<T>, tol: f64)
where
    T: SVDScalar + EinsumScalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    let (a, shape_a) = before.full_tensor().unwrap();
    let (b, shape_b) = after.full_tensor().unwrap();
    assert_eq!(shape_a, shape_b);
    let max_diff = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| ComplexFloat::abs(*x - *y).into())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < tol,
        "canonicalization changed the operator: max abs difference {max_diff:e}"
    );
}

/// Check `sum_{s1, s2, r} B[l, s1, s2, r] * conj(B[l', s1, s2, r]) = delta`.
fn assert_right_orthogonal<T>(tensor: &Tensor4<T>, tol: f64)
where
    T: SVDScalar + EinsumScalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    let left = tensor.left_dim();
    for l in 0..left {
        for lp in 0..left {
            let mut sum = T::zero();
            for s1 in 0..tensor.site_dim_1() {
                for s2 in 0..tensor.site_dim_2() {
                    for r in 0..tensor.right_dim() {
                        sum = sum
                            + *tensor.get4(l, s1, s2, r)
                                * ComplexFloat::conj(*tensor.get4(lp, s1, s2, r));
                    }
                }
            }
            let expected = if l == lp { T::one() } else { T::zero() };
            let diff: f64 = ComplexFloat::abs(sum - expected).into();
            assert!(diff < tol, "gram entry ({l}, {lp}) is off by {diff:e}");
        }
    }
}

fn right_canonicalize_generic<T>(from_f64: impl Fn(f64) -> T)
where
    T: SVDScalar + EinsumScalar,
    <T as ComplexFloat>::Real: Into<f64>,
{
    let bonds = [1, 3, 5, 4, 1];
    let mpo = random_mpo(&bonds, 2, 2, SEED, from_f64);
    let mut canonical = mpo.clone();
    right_canonicalize(&mut canonical).unwrap();

    assert_same_operator(&mpo, &canonical, 1e-10);
    for i in 1..canonical.len() {
        assert_right_orthogonal(canonical.site_tensor(i), 1e-10);
    }
}

#[test]
fn right_canonicalize_preserves_operator_and_orthogonalizes_f64() {
    right_canonicalize_generic::<f64>(|x| x);
}

#[test]
fn right_canonicalize_preserves_operator_and_orthogonalizes_c64() {
    right_canonicalize_generic::<Complex64>(|x| Complex64::new(x, 0.5 - x));
}

#[test]
fn right_canonicalize_reduces_oversized_bonds_to_the_exact_rank() {
    // The last bond is declared as 7 but the site-2 matricization is only
    // 7 x 4, so the exact rank at that bond is 4. A right-to-left sweep sees
    // rank deficiency coming from the right, so this bond must shrink while
    // the first bond, whose deficiency would only be visible from the left,
    // is left alone.
    let bonds = [1, 3, 7, 1];
    let mpo = random_mpo(&bonds, 2, 2, SEED, |x: f64| x);
    let mut canonical = mpo.clone();
    right_canonicalize(&mut canonical).unwrap();

    assert_same_operator(&mpo, &canonical, 1e-10);
    assert_eq!(canonical.link_dims(), vec![3, 4]);
}

#[test]
fn right_canonicalize_is_a_noop_for_short_mpos() {
    let mut mpo = MPO::<f64>::constant(&[(2, 2)], 3.0);
    let before = mpo.clone();
    right_canonicalize(&mut mpo).unwrap();
    assert_same_operator(&before, &mpo, 1e-12);
}
