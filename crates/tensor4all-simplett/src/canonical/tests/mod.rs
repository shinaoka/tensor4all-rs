use super::*;
use num_complex::Complex64;

// Generic test functions for f64 and Complex64

fn test_site_tensor_train_creation_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
    let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();

    assert_eq!(stt.len(), 3);
    assert_eq!(stt.center(), 1);
}

fn test_site_tensor_train_preserves_values_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(2.0));
    let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();

    // Check that evaluation is preserved
    let original_sum = tt.sum();
    let stt_sum = stt.sum();
    assert!(
        (original_sum - stt_sum).abs_sq().sqrt() < 1e-10,
        "Sum mismatch"
    );
}

fn test_move_center_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 4, 2], T::from_f64(1.0));
    let mut stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

    assert_eq!(stt.center(), 0);

    stt.move_center_right().unwrap();
    assert_eq!(stt.center(), 1);

    stt.move_center_right().unwrap();
    assert_eq!(stt.center(), 2);

    stt.move_center_left().unwrap();
    assert_eq!(stt.center(), 1);
}

fn test_set_center_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 4, 2], T::from_f64(1.0));
    let mut stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

    stt.set_center(3).unwrap();
    assert_eq!(stt.center(), 3);

    stt.set_center(1).unwrap();
    assert_eq!(stt.center(), 1);

    // Sum should still be preserved
    let original_sum = tt.sum();
    let stt_sum = stt.sum();
    assert!(
        (original_sum - stt_sum).abs_sq().sqrt() < 1e-10,
        "Sum mismatch after moving center"
    );
}

fn test_to_tensor_train_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(3.0));
    let stt = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap();
    let tt_back = stt.to_tensor_train();

    let original_sum = tt.sum();
    let converted_sum = tt_back.sum();
    assert!(
        (original_sum - converted_sum).abs_sq().sqrt() < 1e-10,
        "Sum mismatch after round-trip"
    );
}

fn test_center_canonicalize_function_generic<T: TTScalar + Scalar + Default>() {
    let tt = SimpleTensorTrain::<T>::constant(&[2, 3, 2], T::from_f64(1.0));
    let mut tensors: Vec<Tensor3<T>> = tt.site_tensors().to_vec();

    let original_sum = tt.sum();

    center_canonicalize(&mut tensors, 1).unwrap();

    // Reconstruct and verify sum
    let tt_new = SimpleTensorTrain::from_tensors_unchecked(tensors);
    let new_sum = tt_new.sum();
    assert!(
        (original_sum - new_sum).abs_sq().sqrt() < 1e-10,
        "Sum mismatch"
    );
}

fn test_evaluate_matches_original_generic<T: TTScalar + Scalar + Default>() {
    let mut t0: Tensor3<T> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, T::from_f64(1.0));
    t0.set3(0, 0, 1, T::from_f64(0.5));
    t0.set3(0, 1, 0, T::from_f64(2.0));
    t0.set3(0, 1, 1, T::from_f64(1.0));

    let mut t1: Tensor3<T> = tensor3_zeros(2, 3, 1);
    for l in 0..2 {
        for s in 0..3 {
            t1.set3(l, s, 0, T::from_f64((l + s + 1) as f64));
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let stt = SiteTensorTrain::from_tensor_train(&tt, 0).unwrap();

    // Check multiple evaluations
    for i in 0..2 {
        for j in 0..3 {
            let original = tt.evaluate(&[i, j]).unwrap();
            let canonical = stt.evaluate(&[i, j]).unwrap();
            assert!(
                (original - canonical).abs_sq().sqrt() < 1e-10,
                "Evaluation mismatch at [{}, {}]",
                i,
                j
            );
        }
    }
}

#[test]
fn test_site_tensor_train_from_tensor_train_reports_qr_failure() {
    let mut left = tensor3_zeros(1, 2, 1);
    left.set3(0, 0, 0, f64::NAN);
    left.set3(0, 1, 0, 1.0);
    let right = tensor3_zeros(1, 2, 1);
    let tt = SimpleTensorTrain::new(vec![left, right]).unwrap();

    let err = SiteTensorTrain::from_tensor_train(&tt, 1).unwrap_err();

    assert!(err.to_string().contains("QR decomposition failed"));
}

#[test]
fn test_center_canonicalize_reports_qr_failure() {
    let mut left = tensor3_zeros(1, 2, 1);
    left.set3(0, 0, 0, f64::NAN);
    left.set3(0, 1, 0, 1.0);
    let right = tensor3_zeros(1, 2, 1);
    let mut tensors = vec![left, right];

    let err = center_canonicalize(&mut tensors, 1).unwrap_err();

    assert!(err.to_string().contains("QR decomposition failed"));
}

// f64 tests
#[test]
fn test_site_tensor_train_creation_f64() {
    test_site_tensor_train_creation_generic::<f64>();
}

#[test]
fn test_site_tensor_train_preserves_values_f64() {
    test_site_tensor_train_preserves_values_generic::<f64>();
}

#[test]
fn test_move_center_f64() {
    test_move_center_generic::<f64>();
}

#[test]
fn test_set_center_f64() {
    test_set_center_generic::<f64>();
}

#[test]
fn test_to_tensor_train_f64() {
    test_to_tensor_train_generic::<f64>();
}

#[test]
fn test_center_canonicalize_function_f64() {
    test_center_canonicalize_function_generic::<f64>();
}

#[test]
fn test_evaluate_matches_original_f64() {
    test_evaluate_matches_original_generic::<f64>();
}

// Complex64 tests
#[test]
fn test_site_tensor_train_creation_c64() {
    test_site_tensor_train_creation_generic::<Complex64>();
}

#[test]
fn test_site_tensor_train_preserves_values_c64() {
    test_site_tensor_train_preserves_values_generic::<Complex64>();
}

#[test]
fn test_move_center_c64() {
    test_move_center_generic::<Complex64>();
}

#[test]
fn test_set_center_c64() {
    test_set_center_generic::<Complex64>();
}

#[test]
fn test_to_tensor_train_c64() {
    test_to_tensor_train_generic::<Complex64>();
}

#[test]
fn test_center_canonicalize_function_c64() {
    test_center_canonicalize_function_generic::<Complex64>();
}

#[test]
fn test_evaluate_matches_original_c64() {
    test_evaluate_matches_original_generic::<Complex64>();
}
