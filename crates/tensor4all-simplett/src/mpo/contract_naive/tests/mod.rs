use super::*;
use crate::mpo::factorize::FactorizeMethod;
use crate::mpo::test_support::random_mpo;

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

#[test]
fn test_contract_naive_identity() {
    // Identity * Identity = Identity
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    assert_eq!(result.len(), 2);

    // The result should be equivalent to identity
    // Check some evaluations
    assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
    assert!((result.evaluate(&[0, 1, 0, 0]).unwrap()).abs() < 1e-10);
    assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_contract_naive_constant() {
    // Constant * Constant
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

    let result = contract_naive(&mpo_a, &mpo_b, None).unwrap();

    // Each element of C = sum over k of A[i, k] * B[k, j]
    // = sum over k of 2 * 3 = 6 * (number of k values = 2) = 12
    assert_eq!(result.len(), 1);
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 12.0).abs() < 1e-10);
}

#[test]
fn test_contract_naive_dimension_mismatch() {
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0); // s2 = 3
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0); // s1 = 2 ≠ 3

    let result = contract_naive(&mpo_a, &mpo_b, None);
    assert!(result.is_err());
}

#[test]
fn test_contract_naive_with_compression() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let options = ContractionOptions {
        tolerance: 1e-10,
        max_bond_dim: Some(2),
        factorize_method: FactorizeMethod::SVD,
    };

    let result = contract_naive(&mpo_a, &mpo_b, Some(options)).unwrap();

    // Bond dimension should be compressed
    assert!(result.rank() <= 2);
}

fn frobenius_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

/// A truncating sweep is only optimal when the sites it has not reached yet
/// are orthonormal. This pins that property: on a two-site product there is a
/// single bond, so the best rank-`k` MPO is exactly the best rank-`k` matrix
/// approximation of the dense matricization, and the compressed contraction
/// must attain it.
///
/// Without the right-canonicalization pass in `compress_mpo` the sweep
/// truncates against the raw gauge of site 1 and keeps the wrong subspace, so
/// this assertion fails.
#[test]
fn compression_at_a_binding_rank_cap_attains_the_optimal_truncation() {
    use tensor4all_tensorbackend::svd_backend;

    let mpo_a = random_mpo(&[1, 3, 1], 2, 2, SEED, |x: f64| x);
    let mpo_b = random_mpo(&[1, 3, 1], 2, 2, SEED ^ 0xFF, |x: f64| x);

    let exact = contract_naive(&mpo_a, &mpo_b, None).unwrap();
    let (exact_dense, shape) = exact.full_tensor();
    assert_eq!(shape, vec![2, 2, 2, 2]);

    // Column-major reshape to (site 0 pair) x (site 1 pair): the matricization
    // at the only bond of the two-site train.
    let matrix =
        tenferro_tensor::TypedTensor::from_vec_col_major(vec![4, 4], exact_dense.clone()).unwrap();
    let svd = svd_backend(&matrix).unwrap();
    let sigma = svd.s().host_data().unwrap().to_vec();
    let keep = 2;
    let optimal_error = sigma[keep..].iter().map(|s| s * s).sum::<f64>().sqrt();
    assert!(
        optimal_error > 1e-3,
        "the rank cap must actually bind for this test to mean anything, \
         discarded weight was {optimal_error:e}"
    );

    let options = ContractionOptions {
        tolerance: 0.0,
        max_bond_dim: Some(keep),
        factorize_method: FactorizeMethod::SVD,
    };
    let truncated = contract_naive(&mpo_a, &mpo_b, Some(options)).unwrap();
    assert_eq!(truncated.rank(), keep);

    let (truncated_dense, _) = truncated.full_tensor();
    let achieved_error = frobenius_distance(&exact_dense, &truncated_dense);
    assert!(
        achieved_error <= optimal_error * (1.0 + 1e-9),
        "compression is not optimal at the rank cap: achieved {achieved_error:e}, \
         best possible {optimal_error:e}"
    );
}
