use super::*;
use crate::mpo::types::tensor4_zeros;

#[test]
fn test_contract_site_tensors() {
    // Create two simple 4D tensors
    let mut a: Tensor4<f64> = tensor4_zeros(1, 2, 3, 1);
    let mut b: Tensor4<f64> = tensor4_zeros(1, 3, 2, 1);

    // Fill with some values
    for s1 in 0..2 {
        for s2 in 0..3 {
            a.set4(0, s1, s2, 0, (s1 * 3 + s2 + 1) as f64);
        }
    }
    for s1 in 0..3 {
        for s2 in 0..2 {
            b.set4(0, s1, s2, 0, (s1 * 2 + s2 + 1) as f64);
        }
    }

    let result = contract_site_tensors(&a, &b).unwrap();

    // Result should have shape (1, 2, 2, 1)
    assert_eq!(result.left_dim(), 1);
    assert_eq!(result.site_dim_1(), 2);
    assert_eq!(result.site_dim_2(), 2);
    assert_eq!(result.right_dim(), 1);
}

#[test]
fn test_contract_site_tensors_rejects_shared_dimension_mismatch() {
    let a: Tensor4<f64> = tensor4_zeros(1, 2, 3, 1);
    let b: Tensor4<f64> = tensor4_zeros(1, 4, 2, 1);

    let err = contract_site_tensors(&a, &b).unwrap_err();
    assert!(err.to_string().contains("Shared shape mismatch"));
    assert!(err.to_string().contains("3"));
    assert!(err.to_string().contains("4"));
}

#[test]
fn test_contract_site_tensors_preserves_combined_bond_order() {
    let mut a: Tensor4<f64> = tensor4_zeros(2, 1, 2, 2);
    let mut b: Tensor4<f64> = tensor4_zeros(2, 2, 1, 2);

    for la in 0..2 {
        for k in 0..2 {
            for ra in 0..2 {
                a.set4(la, 0, k, ra, (100 * la + 10 * k + ra + 1) as f64);
            }
        }
    }
    for lb in 0..2 {
        for k in 0..2 {
            for rb in 0..2 {
                b.set4(lb, k, 0, rb, (1000 * lb + 100 * k + 10 * rb + 1) as f64);
            }
        }
    }

    let result = contract_site_tensors(&a, &b).unwrap();

    assert_eq!(result.left_dim(), 4);
    assert_eq!(result.right_dim(), 4);

    for la in 0..2 {
        for lb in 0..2 {
            for ra in 0..2 {
                for rb in 0..2 {
                    let mut expected = 0.0;
                    for k in 0..2 {
                        expected += *a.get4(la, 0, k, ra) * *b.get4(lb, k, 0, rb);
                    }
                    let actual = *result.get4(la * 2 + lb, 0, 0, ra * 2 + rb);
                    assert!(
                        (actual - expected).abs() < 1e-10,
                        "Mismatch at (la={la}, lb={lb}, ra={ra}, rb={rb}): got {actual}, expected {expected}",
                    );
                }
            }
        }
    }
}

#[test]
fn test_left_environment() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let mut cache = Vec::new();

    // Left environment at site 0 should be [[1]]
    let env0 = left_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
    assert_eq!(env0[[0, 0]], 1.0);

    // Left environment at site 1
    let env1 = left_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
    // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
    assert!((env1[[0, 0]] - 4.0).abs() < 1e-10);

    let cached = left_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
    assert_eq!(cached[[0, 0]], env1[[0, 0]]);
}

#[test]
fn test_right_environment() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let mut cache = Vec::new();

    // Right environment at last site should be [[1]]
    let env_last = right_environment(&mpo_a, &mpo_b, 1, &mut cache).unwrap();
    assert_eq!(env_last[[0, 0]], 1.0);

    // Right environment at site 0
    let env0 = right_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
    // Each element contributes 1*1 = 1, sum over 2*2=4 physical indices
    assert!((env0[[0, 0]] - 4.0).abs() < 1e-10);

    let cached = right_environment(&mpo_a, &mpo_b, 0, &mut cache).unwrap();
    assert_eq!(cached[[0, 0]], env0[[0, 0]]);
}

#[test]
fn test_environment_reports_length_and_dimension_errors() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_short = MPO::<f64>::constant(&[(2, 2)], 1.0);
    let mut cache = Vec::new();

    let err = left_environment(&mpo_a, &mpo_short, 1, &mut cache).unwrap_err();
    assert!(err.to_string().contains("length mismatch"));

    let mut cache = Vec::new();
    let err = right_environment(&mpo_a, &mpo_short, 0, &mut cache).unwrap_err();
    assert!(err.to_string().contains("length mismatch"));

    let mpo_mismatch = MPO::<f64>::constant(&[(2, 3), (2, 2)], 1.0);
    let mut cache = Vec::new();
    let err = left_environment(&mpo_a, &mpo_mismatch, 1, &mut cache).unwrap_err();
    assert!(err.to_string().contains("Shared shape mismatch"));

    let mpo_mismatch_right = MPO::<f64>::constant(&[(2, 2), (2, 3)], 1.0);
    let mut cache = Vec::new();
    let err = right_environment(&mpo_a, &mpo_mismatch_right, 0, &mut cache).unwrap_err();
    assert!(err.to_string().contains("Shared shape mismatch"));
}

#[test]
fn test_mpo_helper_error_formats_context() {
    let err = mpo_helper_error("while building environment", "invalid rank");
    let message = err.to_string();
    assert!(message.contains("while building environment"));
    assert!(message.contains("invalid rank"));
}
