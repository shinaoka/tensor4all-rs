use super::*;
use crate::types::Tensor3;

#[test]
fn test_add_constant_tensors() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);

    let result = tt1.add(&tt2).unwrap();

    // Sum should be (1.0 + 2.0) * 2 * 3 = 18.0
    assert!((result.sum() - 18.0).abs() < 1e-10);
}

#[test]
fn test_sub_constant_tensors() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 5.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);

    let result = tt1.sub(&tt2).unwrap();

    // Sum should be (5.0 - 2.0) * 2 * 3 = 18.0
    assert!((result.sum() - 18.0).abs() < 1e-10);
}

#[test]
fn test_add_single_site_constant_tensors() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[3], 2.0);

    let result = tt1.add(&tt2).unwrap();

    for i in 0..3 {
        assert!((result.evaluate(&[i]).unwrap() - 3.0).abs() < 1e-10);
    }
    assert!((result.sum() - 9.0).abs() < 1e-10);
}

#[test]
fn test_sub_single_site_constant_tensors() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[3], 5.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[3], 2.0);

    let result = tt1.sub(&tt2).unwrap();

    for i in 0..3 {
        assert!((result.evaluate(&[i]).unwrap() - 3.0).abs() < 1e-10);
    }
    assert!((result.sum() - 9.0).abs() < 1e-10);
}

#[test]
fn test_negate() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 3.0);
    let neg_tt = tt.negate();

    assert!((neg_tt.sum() + tt.sum()).abs() < 1e-10);
}

#[test]
fn test_add_operator() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 2], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 2], 1.0);

    let result = (&tt1 + &tt2).unwrap();

    // Sum should be 2.0 * 2 * 2 = 8.0
    assert!((result.sum() - 8.0).abs() < 1e-10);
}

#[test]
fn test_add_preserves_evaluation() {
    // Create two simple tensor trains
    let mut t0_a: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t0_a.set3(0, 0, 0, 1.0);
    t0_a.set3(0, 1, 0, 2.0);

    let mut t1_a: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t1_a.set3(0, 0, 0, 3.0);
    t1_a.set3(0, 1, 0, 4.0);

    let tt_a = SimpleTensorTrain::new(vec![t0_a, t1_a]).unwrap();

    let mut t0_b: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t0_b.set3(0, 0, 0, 0.5);
    t0_b.set3(0, 1, 0, 1.5);

    let mut t1_b: Tensor3<f64> = tensor3_zeros(1, 2, 1);
    t1_b.set3(0, 0, 0, 2.5);
    t1_b.set3(0, 1, 0, 3.5);

    let tt_b = SimpleTensorTrain::new(vec![t0_b, t1_b]).unwrap();

    let result = tt_a.add(&tt_b).unwrap();

    // Test some evaluations
    // result([0, 0]) = a([0,0]) + b([0,0]) = 1*3 + 0.5*2.5 = 3 + 1.25 = 4.25
    let val_00 = result.evaluate(&[0, 0]).unwrap();
    let expected_00 = tt_a.evaluate(&[0, 0]).unwrap() + tt_b.evaluate(&[0, 0]).unwrap();
    assert!((val_00 - expected_00).abs() < 1e-10);

    // result([1, 1]) = a([1,1]) + b([1,1]) = 2*4 + 1.5*3.5 = 8 + 5.25 = 13.25
    let val_11 = result.evaluate(&[1, 1]).unwrap();
    let expected_11 = tt_a.evaluate(&[1, 1]).unwrap() + tt_b.evaluate(&[1, 1]).unwrap();
    assert!((val_11 - expected_11).abs() < 1e-10);
}

#[test]
fn test_add_length_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    assert!(tt1.add(&tt2).is_err());
}

#[test]
fn test_add_site_dim_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 4], 1.0);
    assert!(tt1.add(&tt2).is_err());
}

#[test]
fn test_add_empty() {
    let tt1 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let tt2 = SimpleTensorTrain::<f64>::from_tensors_unchecked(Vec::new());
    let result = tt1.add(&tt2).unwrap();
    assert_eq!(result.len(), 0);
}

#[test]
fn test_sub_length_mismatch() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    assert!(tt1.sub(&tt2).is_err());
}

#[test]
fn test_add_three_sites() {
    // 3 sites to exercise the middle-tensor block-diagonal path
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let result = tt1.add(&tt2).unwrap();
    // Sum should be (1+2)*2*3*2 = 36
    assert!((result.sum() - 36.0).abs() < 1e-10);
}

#[test]
fn test_sub_by_value_operator() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 5.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let result = (tt1 - tt2).unwrap();
    // (5-2)*2*3 = 18
    assert!((result.sum() - 18.0).abs() < 1e-10);
}

#[test]
fn test_add_by_value_operator() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let result = (tt1 + tt2).unwrap();
    assert!((result.sum() - 18.0).abs() < 1e-10);
}

#[test]
fn test_neg_by_value_operator() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 3.0);
    let neg_tt = -tt;
    assert!((neg_tt.sum() + 12.0).abs() < 1e-10);
}

#[test]
fn test_neg_by_ref_operator() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2], 3.0);
    let neg_tt = -&tt;
    assert!((neg_tt.sum() + 12.0).abs() < 1e-10);
    // Original should still be accessible
    assert!((tt.sum() - 12.0).abs() < 1e-10);
}

#[test]
fn test_mul_by_value_operator() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let result = tt * 3.0;
    // 2*3 * 2*3 = 36
    assert!((result.sum() - 36.0).abs() < 1e-10);
}

#[test]
fn test_mul_by_ref_operator() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let result = &tt * 3.0;
    assert!((result.sum() - 36.0).abs() < 1e-10);
    // Original accessible
    assert!((tt.sum() - 12.0).abs() < 1e-10);
}

#[test]
fn test_sub_by_ref_operator() {
    let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 5.0);
    let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    let result = (&tt1 - &tt2).unwrap();
    assert!((result.sum() - 18.0).abs() < 1e-10);
}

fn test_arithmetic_generic<T: TTScalar>() {
    let tt1 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(1.0));
    let tt2 = SimpleTensorTrain::<T>::constant(&[2, 3], T::from_f64(2.0));

    let add_result = tt1.add(&tt2).unwrap();
    let add_sum = add_result.sum();
    assert!(
        (add_sum - T::from_f64(18.0)).abs_sq().sqrt() < 1e-10,
        "add sum mismatch"
    );

    let sub_result = tt1.sub(&tt2).unwrap();
    let sub_sum = sub_result.sum();
    assert!(
        (sub_sum - T::from_f64(-6.0)).abs_sq().sqrt() < 1e-10,
        "sub sum mismatch"
    );

    let neg_result = tt1.negate();
    let neg_sum = neg_result.sum();
    assert!(
        (neg_sum - T::from_f64(-6.0)).abs_sq().sqrt() < 1e-10,
        "negate sum mismatch"
    );
}

#[test]
fn test_arithmetic_f64() {
    test_arithmetic_generic::<f64>();
}

#[test]
fn test_arithmetic_c64() {
    test_arithmetic_generic::<num_complex::Complex64>();
}
