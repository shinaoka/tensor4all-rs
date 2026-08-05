use super::*;
use crate::mpo::factorize::FactorizeMethod;

#[test]
fn test_contract_zipup_identity() {
    // Identity * Identity = Identity
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let options = ContractionOptions {
        tolerance: 1e-12,
        max_bond_dim: 10,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    assert_eq!(result.len(), 2);

    // The result should be equivalent to identity
    assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
    assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_contract_zipup_constant() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

    let options = ContractionOptions::default();

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Each element of C = sum over k of A[i, k] * B[k, j]
    // = sum over k of 2 * 3 = 6 * 2 = 12
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn test_contract_zipup_compresses() {
    // Create MPOs with higher bond dimensions
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let options = ContractionOptions {
        tolerance: 1e-12,
        max_bond_dim: 2,
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();

    // Bond dimension should be limited
    assert!(result.rank() <= 2);
}

/// Without truncation the zip-up pass must reproduce the exact product, which
/// pins the index conventions of the two pairwise contractions and of the
/// reshapes around the local factorization.
#[test]
fn test_contract_zipup_untruncated_matches_naive() {
    use crate::mpo::contract_naive;
    use crate::mpo::test_support::random_mpo;

    let mpo_a = random_mpo(&[1, 3, 4, 1], 2, 3, 0x1234_5678_9ABC_DEF0, |x: f64| x);
    let mpo_b = random_mpo(&[1, 2, 5, 1], 3, 2, 0x0FED_CBA9_8765_4321, |x: f64| x);

    let options = ContractionOptions {
        tolerance: 0.0,
        max_bond_dim: usize::MAX,
        factorize_method: FactorizeMethod::SVD,
    };

    let zipped = contract_zipup(&mpo_a, &mpo_b, &options).unwrap();
    let exact = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    let (zipped_dense, zipped_shape) = zipped.fulltensor();
    let (exact_dense, exact_shape) = exact.fulltensor();
    assert_eq!(zipped_shape, exact_shape);
    assert_eq!(zipped_shape, vec![2, 2, 2, 2, 2, 2]);

    let max_diff = zipped_dense
        .iter()
        .zip(exact_dense.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max);
    let scale = exact_dense.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    assert!(
        max_diff < 1e-10 * scale,
        "zip-up deviates from the exact product: max abs difference {max_diff:e} \
         against scale {scale:e}"
    );
}
