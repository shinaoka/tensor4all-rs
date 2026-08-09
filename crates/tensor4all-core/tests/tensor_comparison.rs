use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{diag_tensor_dyn_len, TensorDynLen};

#[test]
fn test_sub_identical_tensors_is_zero() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let diff = a.sub(&a).unwrap();
    assert!(diff.norm().unwrap() < 1e-14);
    assert!(diff.maxabs().unwrap() < 1e-14);
}

#[test]
fn test_sub_different_tensors() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![3.0, 5.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();

    let diff = a.sub(&b).unwrap();
    let data = diff.to_vec::<f64>().unwrap();
    assert!((data[0] - 2.0).abs() < 1e-14);
    assert!((data[1] - 3.0).abs() < 1e-14);
}

#[test]
fn test_sub_permuted_indices() {
    // a[i,j] - b[j,i] should auto-permute
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b = TensorDynLen::from_dense(
        vec![j.clone(), i.clone()],
        vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], // transposed in column-major order
    )
    .unwrap();

    let diff = a.sub(&b).unwrap();
    assert!(diff.maxabs().unwrap() < 1e-14);
}

#[test]
fn test_neg() {
    let i = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -2.0, 3.0]).unwrap();

    let neg_a = a.neg().unwrap();
    let data = neg_a.to_vec::<f64>().unwrap();
    assert!((data[0] - (-1.0)).abs() < 1e-14);
    assert!((data[1] - 2.0).abs() < 1e-14);
    assert!((data[2] - (-3.0)).abs() < 1e-14);
}

#[test]
fn test_maxabs() {
    let i = Index::new_dyn(4);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, -5.0, 3.0, -2.0]).unwrap();
    assert!((a.maxabs().unwrap() - 5.0).abs() < 1e-14);
}

#[test]
fn test_maxabs_scalar() {
    let s = TensorDynLen::scalar(-7.0).unwrap();
    assert!((s.maxabs().unwrap() - 7.0).abs() < 1e-14);
}

#[test]
fn test_maxabs_diag_f64() {
    let i = Index::new_dyn(4);
    let j = Index::new_dyn(4);
    let d = diag_tensor_dyn_len(vec![i, j], vec![1.0, -5.0, 3.0, -2.0]).unwrap();
    assert!((d.maxabs().unwrap() - 5.0).abs() < 1e-14);
}

#[test]
fn test_maxabs_diag_c64() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let d = TensorDynLen::from_diag(
        vec![i, j],
        vec![
            Complex64::new(3.0, 4.0),  // |z| = 5
            Complex64::new(-1.0, 1.0), // |z| = sqrt(2)
            Complex64::new(0.0, -2.0), // |z| = 2
        ],
    )
    .unwrap();
    assert!((d.maxabs().unwrap() - 5.0).abs() < 1e-14);
}

#[test]
fn test_isapprox_identical() {
    let i = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    assert!(a.isapprox(&a, 0.0, 0.0).unwrap());
}

#[test]
fn test_isapprox_atol() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.01]).unwrap();

    // ||a - b|| = 0.01
    assert!(a.isapprox(&b, 0.1, 0.0).unwrap()); // atol=0.1 > 0.01
    assert!(!a.isapprox(&b, 0.001, 0.0).unwrap()); // atol=0.001 < 0.01
}

#[test]
fn test_isapprox_rtol() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![100.0, 200.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![100.0, 201.0]).unwrap();

    // ||a - b|| = 1.0, max(||a||, ||b||) ≈ 224
    // rtol * max_norm ≈ 0.01 * 224 ≈ 2.24 > 1.0
    assert!(a.isapprox(&b, 0.0, 0.01).unwrap());
    // rtol * max_norm ≈ 0.001 * 224 ≈ 0.224 < 1.0
    assert!(!a.isapprox(&b, 0.0, 0.001).unwrap());
}

#[test]
fn test_isapprox_index_mismatch_returns_error() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j.clone()], vec![1.0, 2.0, 3.0]).unwrap();

    // Different indices → sub fails → isapprox returns an error.
    assert!(a.isapprox(&b, 1e10, 1e10).is_err());
}

#[test]
fn test_sub_operator_owned() {
    let i = Index::new_dyn(2);
    let a = TensorDynLen::from_dense(vec![i.clone()], vec![5.0, 10.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 3.0]).unwrap();

    // owned - owned
    let diff = a.sub(&b).unwrap();
    let data = diff.to_vec::<f64>().unwrap();
    assert!((data[0] - 4.0).abs() < 1e-14);
    assert!((data[1] - 7.0).abs() < 1e-14);

    // owned - ref
    let diff2 = a.sub(&b).unwrap();
    assert!(diff2.isapprox(&diff, 1e-14, 0.0).unwrap());

    // ref - owned
    let diff3 = a.sub(&b).unwrap();
    assert!(diff3.isapprox(&diff, 1e-14, 0.0).unwrap());
}
