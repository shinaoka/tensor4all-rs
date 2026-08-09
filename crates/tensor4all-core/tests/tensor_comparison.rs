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
fn test_isapprox_aligns_permuted_axes_without_reordering_payloads() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let lhs =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let rhs = TensorDynLen::from_dense(vec![j, i], vec![1.0, 3.0, 2.0, 4.0]).unwrap();

    assert!(lhs.isapprox(&rhs, 0.0, 0.0).unwrap());
}

#[test]
fn test_isapprox_exact_comparison_does_not_underflow() {
    let lhs = TensorDynLen::scalar(1.0e-300).unwrap();
    let rhs = TensorDynLen::scalar(2.0e-300).unwrap();

    assert!(!lhs.isapprox(&rhs, 0.0, 0.0).unwrap());
}

#[test]
fn test_isapprox_relative_tolerance_does_not_underflow() {
    let lhs = TensorDynLen::scalar(1.0e-300).unwrap();
    let rhs = TensorDynLen::scalar(2.0e-300).unwrap();

    assert!(!lhs.isapprox(&rhs, 0.0, 0.1).unwrap());
    assert!(lhs.isapprox(&rhs, 0.0, 0.6).unwrap());
}

#[test]
fn test_isapprox_relative_tolerance_does_not_overflow() {
    let lhs = TensorDynLen::scalar(f64::MAX).unwrap();
    let rhs = TensorDynLen::scalar(-f64::MAX).unwrap();

    assert!(!lhs.isapprox(&rhs, 0.0, 1.0).unwrap());
    assert!(lhs.isapprox(&rhs, 0.0, 2.0).unwrap());
}

#[test]
fn test_isapprox_structured_support_matches_permuted_and_dense_layouts() {
    let left = Index::new_dyn(2);
    let site = Index::new_dyn(3);
    let right = Index::new_dyn(2);
    let structured =
        TensorDynLen::from_copy_selector(left.clone(), site.clone(), right.clone(), 1, 2.0_f64)
            .unwrap();
    let permuted = structured
        .permute_indices(&[site.clone(), right.clone(), left.clone()])
        .unwrap();
    let dense = TensorDynLen::from_dense(
        structured.indices().to_vec(),
        structured.to_vec::<f64>().unwrap(),
    )
    .unwrap();

    assert!(structured.isapprox(&permuted, 0.0, 0.0).unwrap());
    assert!(structured.isapprox(&dense, 0.0, 0.0).unwrap());
    // The same compact-support mapping must hold in non-exact tolerance mode.
    assert!(structured.isapprox(&permuted, 1.0e-12, 1.0e-12).unwrap());
    assert!(structured.isapprox(&dense, 1.0e-12, 1.0e-12).unwrap());
}

#[test]
fn test_isapprox_diagonal_vs_dense_collapsed_support() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, 2.0]).unwrap();
    // Dense expansion keeps the diagonal support; off-diagonal structural
    // zeros are still part of the logical tensor and must be compared.
    let same =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    let different = TensorDynLen::from_dense(vec![i.clone(), j], vec![1.0, 0.0, 0.0, 3.0]).unwrap();
    assert!(diag.isapprox(&same, 1.0e-12, 1.0e-12).unwrap());
    assert!(!diag.isapprox(&different, 0.0, 0.1).unwrap());
}

#[test]
fn test_isapprox_rejects_nan_input() {
    let lhs = TensorDynLen::scalar(f64::NAN).unwrap();
    let rhs = TensorDynLen::scalar(1.0).unwrap();
    assert!(lhs.isapprox(&rhs, 1.0e-12, 1.0e-12).is_err());
    // The NaN preflight must also apply in exact comparison mode.
    assert!(lhs.isapprox(&rhs, 0.0, 0.0).is_err());

    let index = Index::new_dyn(2);
    let nan_tensor = TensorDynLen::from_dense(vec![index.clone()], vec![1.0, f64::NAN]).unwrap();
    let finite = TensorDynLen::from_dense(vec![index], vec![1.0, 1.0]).unwrap();
    assert!(nan_tensor.isapprox(&finite, 1.0e-12, 1.0e-12).is_err());
}

#[test]
fn test_isapprox_rejects_nan_in_structured_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let nan_diag = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, f64::NAN]).unwrap();
    let dense = TensorDynLen::from_dense(vec![i.clone(), j], vec![1.0, 0.0, 0.0, 2.0]).unwrap();
    assert!(nan_diag.isapprox(&dense, 1.0e-12, 1.0e-12).is_err());
}

#[test]
fn test_isapprox_unmatched_support_is_compared_in_both_orders() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, 2.0]).unwrap();
    // The off-diagonal entry (1,0) lies outside the diagonal compact support;
    // it must be compared against the structural zero in both operand orders.
    let off_diag =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0, 10.0, 0.0, 2.0]).unwrap();
    assert!(!diag.isapprox(&off_diag, 0.0, 0.5).unwrap());
    assert!(!off_diag.isapprox(&diag, 0.0, 0.5).unwrap());
}

#[test]
fn test_isapprox_zero_vs_nonzero_tolerant() {
    let zero = TensorDynLen::scalar(0.0).unwrap();
    let small = TensorDynLen::scalar(1.0e-8).unwrap();
    // One side has a zero reference norm; the unit ratio must only pass at
    // rtol >= 1 and never in exact mode.
    assert!(!zero.isapprox(&small, 0.0, 0.5).unwrap());
    assert!(zero.isapprox(&small, 0.0, 1.0).unwrap());
}

#[test]
fn test_isapprox_large_structured_support_does_not_visit_logical_domain() {
    let bond_dim = 100_000;
    let site = Index::new_dyn(3);
    let tensor = TensorDynLen::from_copy_selector(
        Index::new_dyn(bond_dim),
        site,
        Index::new_dyn(bond_dim),
        1,
        2.0_f64,
    )
    .unwrap();

    assert!(tensor.isapprox(&tensor, 0.0, 0.0).unwrap());
    assert_eq!(tensor.maxabs().unwrap(), 2.0);
    assert!((tensor.norm_squared().unwrap() - 4.0 * bond_dim as f64).abs() < 1.0e-8);
}

#[test]
fn test_isapprox_matching_infinities_are_not_reference_norms() {
    let index = Index::new_dyn(2);
    let lhs = TensorDynLen::from_dense(vec![index.clone()], vec![f64::INFINITY, 0.0]).unwrap();
    let rhs = TensorDynLen::from_dense(vec![index], vec![f64::INFINITY, 1.0]).unwrap();

    assert!(!lhs.isapprox(&rhs, 0.0, 0.5).unwrap());
    assert!(lhs.isapprox(&rhs, 0.0, 1.0).unwrap());
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
