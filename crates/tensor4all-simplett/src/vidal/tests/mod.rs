use super::*;

#[test]
fn test_vidal_creation() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();

    assert_eq!(vidal.len(), 3);
    assert_eq!(vidal.all_singular_values().len(), 2);
}

#[test]
fn test_vidal_to_tensor_train_preserves_sum() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();
    let tt_back = vidal.to_tensor_train();

    let original_sum = tt.sum();
    let converted_sum = tt_back.sum();

    assert!(
        (original_sum - converted_sum).abs() < 1e-8,
        "Sum mismatch: {} vs {}",
        original_sum,
        converted_sum
    );
}

#[test]
fn test_inverse_creation() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let inverse = InverseTensorTrain::from_tensor_train(&tt).unwrap();

    assert_eq!(inverse.len(), 3);
    assert_eq!(inverse.all_inverse_singular_values().len(), 2);
}

#[test]
fn test_inverse_to_tensor_train_preserves_sum() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let inverse = InverseTensorTrain::from_tensor_train(&tt).unwrap();
    let tt_back = inverse.to_tensor_train();

    let original_sum = tt.sum();
    let converted_sum = tt_back.sum();

    assert!(
        (original_sum - converted_sum).abs() < 1e-8,
        "Sum mismatch: {} vs {}",
        original_sum,
        converted_sum
    );
}

#[test]
fn test_vidal_singular_values_positive() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 3.0);
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();

    // Check that singular values are positive
    for sv in vidal.all_singular_values() {
        for &v in sv {
            assert!(v > 0.0, "Singular value should be positive");
        }
    }
}

fn two_site_matrix_tt() -> SimpleTensorTrain<f64> {
    let mut left = tensor3_zeros(1, 2, 2);
    left.set3(0, 0, 0, 1.0);
    left.set3(0, 1, 1, 1.0);

    let mut right = tensor3_zeros(2, 2, 1);
    right.set3(0, 0, 0, 2.0);
    right.set3(1, 0, 0, 1.0);
    right.set3(0, 1, 0, 1.0);
    right.set3(1, 1, 0, 2.0);

    SimpleTensorTrain::new(vec![left, right]).unwrap()
}

fn assert_diag_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected.iter()) {
        assert!(
            (a - e).abs() < 1e-10,
            "expected diagonal {expected:?}, got {actual:?}"
        );
    }
}

#[test]
fn test_vidal_reports_true_two_site_singular_values() {
    let tt = two_site_matrix_tt();
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();

    assert_diag_close(vidal.singular_values(0), &[3.0, 1.0]);
}

#[test]
fn test_inverse_reports_true_two_site_inverse_singular_values() {
    let tt = two_site_matrix_tt();
    let inverse = InverseTensorTrain::from_tensor_train(&tt).unwrap();

    assert_diag_close(inverse.inverse_singular_values(0), &[1.0 / 3.0, 1.0]);
}

#[test]
fn test_vidal_empty() {
    let tt = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();
    assert_eq!(vidal.len(), 0);
    assert!(vidal.all_singular_values().is_empty());
    assert_eq!(*vidal.partition(), 0..0);

    let tt_back = vidal.to_tensor_train();
    assert_eq!(tt_back.len(), 0);
}

#[test]
fn test_vidal_new_wrong_sv_count() {
    let t: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    // 2 tensors need 1 singular value vector, but we give 0
    let err = VidalTensorTrain::new(vec![t.clone(), t.clone()], Vec::new());
    assert!(err.is_err());
}

#[test]
fn test_vidal_new_empty() {
    let vidal = VidalTensorTrain::<f64>::new(Vec::new(), Vec::new()).unwrap();
    assert_eq!(vidal.len(), 0);
}

#[test]
fn test_vidal_partition_exceeds_length() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 1.0);
    let err = VidalTensorTrain::from_tensor_train_with_partition(&tt, 0..5);
    assert!(err.is_err());
}

#[test]
fn test_vidal_from_tensor_train_reports_qr_failure() {
    let mut left = tensor3_zeros(1, 2, 1);
    left.set3(0, 0, 0, f64::NAN);
    left.set3(0, 1, 0, 1.0);
    let right = tensor3_zeros(1, 2, 1);
    let tt = SimpleTensorTrain::new(vec![left, right]).unwrap();

    let err = VidalTensorTrain::from_tensor_train(&tt).unwrap_err();

    assert!(err.to_string().contains("QR decomposition failed"));
}

#[test]
fn test_vidal_round_trip_evaluations() {
    // Test that Vidal round-trip preserves individual element values
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 0.3);
    t0.set3(0, 1, 1, 0.7);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 2);
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(l, s, r, (l + s * 2 + r * 3 + 1) as f64 * 0.1);
            }
        }
    }

    let mut t2: Tensor3<f64> = tensor3_zeros(2, 2, 1);
    t2.set3(0, 0, 0, 1.0);
    t2.set3(0, 1, 0, 2.0);
    t2.set3(1, 0, 0, 3.0);
    t2.set3(1, 1, 0, 4.0);

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();
    let tt_back = vidal.to_tensor_train();

    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                let orig = tt.evaluate(&[i0, i1, i2]).unwrap();
                let conv = tt_back.evaluate(&[i0, i1, i2]).unwrap();
                assert!(
                    (orig - conv).abs() < 1e-8,
                    "Mismatch at [{}, {}, {}]: {} vs {}",
                    i0,
                    i1,
                    i2,
                    orig,
                    conv
                );
            }
        }
    }
}

#[test]
fn test_vidal_accessors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let mut vidal = VidalTensorTrain::from_tensor_train(&tt).unwrap();
    assert_eq!(
        vidal.singular_values(0).len(),
        vidal.site_tensor(0).right_dim()
    );
    // Mutable accessors
    vidal.site_tensors_mut()[0].set3(0, 0, 0, 42.0);
    assert_eq!(*vidal.site_tensor(0).get3(0, 0, 0), 42.0);
    vidal.singular_values_mut(0)[0] = 99.0;
    assert_eq!(vidal.singular_values(0)[0], 99.0);
}

#[test]
fn test_inverse_empty() {
    let tt = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let inv = InverseTensorTrain::from_tensor_train(&tt).unwrap();
    assert_eq!(inv.len(), 0);
    assert!(inv.all_inverse_singular_values().is_empty());
    assert_eq!(*inv.partition(), 0..0);

    let tt_back = inv.to_tensor_train();
    assert_eq!(tt_back.len(), 0);
}

#[test]
fn test_inverse_round_trip_evaluations() {
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 0.3);
    t0.set3(0, 1, 1, 0.7);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 2);
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(l, s, r, (l + s * 2 + r * 3 + 1) as f64 * 0.1);
            }
        }
    }

    let mut t2: Tensor3<f64> = tensor3_zeros(2, 2, 1);
    t2.set3(0, 0, 0, 1.0);
    t2.set3(0, 1, 0, 2.0);
    t2.set3(1, 0, 0, 3.0);
    t2.set3(1, 1, 0, 4.0);

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let inv = InverseTensorTrain::from_tensor_train(&tt).unwrap();
    let tt_back = inv.to_tensor_train();

    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                let orig = tt.evaluate(&[i0, i1, i2]).unwrap();
                let conv = tt_back.evaluate(&[i0, i1, i2]).unwrap();
                assert!(
                    (orig - conv).abs() < 1e-8,
                    "Mismatch at [{}, {}, {}]: {} vs {}",
                    i0,
                    i1,
                    i2,
                    orig,
                    conv
                );
            }
        }
    }
}

#[test]
fn test_inverse_set_two_site_tensors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut inv = InverseTensorTrain::from_tensor_train(&tt).unwrap();

    let t0: Tensor3<f64> = tensor3_zeros(1, 2, 3);
    let t1: Tensor3<f64> = tensor3_zeros(3, 2, 1);
    let inv_sv = vec![1.0, 1.0, 1.0];

    inv.set_two_site_tensors(0, t0, inv_sv, t1).unwrap();
    assert_eq!(inv.site_tensor(0).right_dim(), 3);
    assert_eq!(inv.site_tensor(1).left_dim(), 3);
}

#[test]
fn test_inverse_set_two_site_tensors_out_of_bounds() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut inv = InverseTensorTrain::from_tensor_train(&tt).unwrap();

    let t0: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    let t1: Tensor3<f64> = tensor3_zeros(1, 2, 1);

    // site 2 is the last site (len=3), so i=2 should fail (i >= len-1)
    let err = inv.set_two_site_tensors(2, t0, vec![1.0], t1);
    assert!(err.is_err());
}

#[test]
fn test_inverse_accessors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let mut inv = InverseTensorTrain::from_tensor_train(&tt).unwrap();

    assert_eq!(
        inv.inverse_singular_values(0).len(),
        inv.site_tensor(0).right_dim()
    );
    // Mutable accessor
    inv.site_tensors_mut()[0].set3(0, 0, 0, 42.0);
    assert_eq!(*inv.site_tensor(0).get3(0, 0, 0), 42.0);
}
