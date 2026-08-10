use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{
    default_svd_truncation_policy, set_default_svd_truncation_policy, svd, svd_with, SvdOptions,
    SvdTruncationPolicy,
};
use tensor4all_core::{DynIndex, TensorContractionLike, TensorDynLen};

fn dense_f64(indices: Vec<DynIndex>, data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn dense_c64(indices: Vec<DynIndex>, data: Vec<Complex64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn vh_from_v(v: &TensorDynLen) -> TensorDynLen {
    assert!(
        !v.indices.is_empty(),
        "V tensor must have at least one (bond) index"
    );
    let v_conj = v.conj();
    let ndim = v_conj.indices.len();
    let mut perm = vec![ndim - 1];
    perm.extend(0..(ndim - 1));
    v_conj.permute(&perm).unwrap()
}

fn reconstruct_from_svd(u: &TensorDynLen, s: &TensorDynLen, v: &TensorDynLen) -> TensorDynLen {
    let vh = vh_from_v(v);
    let svh = s.contract_pair(&vh).unwrap();
    let sim_bond = s.indices[1].clone();
    let bond = v.indices[v.indices.len() - 1].clone();
    let svh = svh.replaceind(&sim_bond, &bond).unwrap();
    u.contract_pair(&svh).unwrap()
}

#[test]
fn test_svd_identity() {
    // Test SVD of a 2×2 identity matrix
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create identity matrix: [[1, 0], [0, 1]]
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0]
    data[3] = 1.0; // [1, 1]

    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // Check dimensions
    assert_eq!(u.dims(), vec![2, 2]);
    assert_eq!(s.dims(), vec![2, 2]);
    assert_eq!(v.dims(), vec![2, 2]);

    // Check indices
    assert_eq!(u.indices.len(), 2);
    assert_eq!(s.indices.len(), 2);
    assert_eq!(v.indices.len(), 2);

    // For identity matrix, singular values should be [1, 1].
    // We avoid direct storage slice inspection and verify through tensor invariants.
    assert!((s.sum().unwrap().real() - 2.0).abs() < 1e-10);
    assert!((s.norm_squared().unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_svd_simple_matrix() {
    // Test SVD of a simple 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // Check dimensions: m=2, n=3, k=min(2,3)=2
    assert_eq!(u.dims(), vec![2, 2]);
    assert_eq!(s.dims(), vec![2, 2]);
    assert_eq!(v.dims(), vec![3, 2]);

    // Check indices
    assert_eq!(u.indices.len(), 2);
    assert_eq!(s.indices.len(), 2);
    assert_eq!(v.indices.len(), 2);

    // Check that U and V share the bond index
    assert_eq!(u.indices[1].id, s.indices[0].id);
    // S tensor has two indices with same dimension and tags but different IDs (to avoid duplicate IDs)
    assert_eq!(s.indices[0].size(), s.indices[1].size());
    assert_eq!(s.indices[0].tags(), s.indices[1].tags());
    assert_eq!(s.indices[0].id, v.indices[1].id);

    // Check that bond index has "Link" tag
    assert!(u.indices[1].tags().has_tag("Link"));
    assert!(s.indices[0].tags().has_tag("Link"));
    assert!(v.indices[1].tags().has_tag("Link"));
}

#[test]
fn test_svd_reconstruction() {
    // Test that U * S * V^T reconstructs the original matrix
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);

    // Create a random-ish matrix
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = dense_f64(vec![i.clone(), j.clone()], data.clone());

    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // Reconstruct: A = U * S * V^H
    let reconstructed = reconstruct_from_svd(&u, &s, &v);

    // Check reconstruction accuracy
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "SVD reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_svd_invalid_rank() {
    // Test that SVD fails for rank-1 tensors
    let i = Index::new_dyn(2);

    let tensor = TensorDynLen::zeros::<f64>(vec![i.clone()]).unwrap();

    let result = svd::<f64>(&tensor, std::slice::from_ref(&i));
    assert!(result.is_err());
    // Expected: unfold_split returns an error for rank < 2
    if result.is_ok() {
        panic!("Expected error but got Ok");
    }
}

#[test]
fn test_svd_invalid_split() {
    // Test that SVD fails when left_inds is empty or contains all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let tensor = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    // Empty left_inds should fail
    let result = svd::<f64>(&tensor, &[]);
    assert!(result.is_err(), "Expected error for empty left_inds");

    // All indices in left_inds should fail
    let result = svd::<f64>(&tensor, &[i, j]);
    assert!(
        result.is_err(),
        "Expected error for all indices in left_inds"
    );
}

#[test]
fn test_svd_rank3() {
    // Test SVD of a rank-3 tensor: split first index vs remaining two
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create a 2×3×4 tensor with some data
    let data = (0..24).map(|x| x as f64).collect::<Vec<_>>();
    let tensor = dense_f64(vec![i.clone(), j.clone(), k.clone()], data);

    // Split: left = [i], right = [j, k]
    // This unfolds to a 2×12 matrix
    let (u, s, v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // Check dimensions:
    // U should be [i, bond] = [2, min(2, 12)] = [2, 2]
    // S should be [bond, bond] = [2, 2]
    // V should be [j, k, bond] = [3, 4, 2]
    assert_eq!(u.dims(), vec![2, 2]);
    assert_eq!(s.dims(), vec![2, 2]);
    assert_eq!(v.dims(), vec![3, 4, 2]);

    // Check indices
    assert_eq!(u.indices.len(), 2);
    assert_eq!(s.indices.len(), 2);
    assert_eq!(v.indices.len(), 3);

    // Check that U has left index first, then bond
    assert_eq!(u.indices[0].id, i.id);

    // Check that V has right indices first, then bond
    assert_eq!(v.indices[0].id, j.id);
    assert_eq!(v.indices[1].id, k.id);

    // Check that U and V share the bond index
    assert_eq!(u.indices[1].id, s.indices[0].id);
    // S tensor has two indices with same dimension and tags but different IDs (to avoid duplicate IDs)
    assert_eq!(s.indices[0].size(), s.indices[1].size());
    assert_eq!(s.indices[0].tags(), s.indices[1].tags());
    assert_eq!(s.indices[0].id, v.indices[2].id);

    // Check that bond index has "Link" tag
    assert!(u.indices[1].tags().has_tag("Link"));
    assert!(s.indices[0].tags().has_tag("Link"));
    assert!(v.indices[2].tags().has_tag("Link"));
}

#[test]
fn test_svd_nontrivial_split_reconstruction() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);

    let data = (1..=24).map(|x| x as f64).collect::<Vec<_>>();
    let tensor = dense_f64(vec![i.clone(), j.clone(), k.clone(), l.clone()], data);

    let (u, s, v) = svd::<f64>(&tensor, &[i.clone(), k.clone()]).expect("SVD should succeed");
    let reconstructed = reconstruct_from_svd(&u, &s, &v);

    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "SVD nontrivial split reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_svd_complex_reconstruction() {
    // Complex diagonal-ish matrix where conjugation matters in principle:
    // A = [[i, 0], [0, 2]]
    let i_idx = Index::new_dyn(2);
    let j_idx = Index::new_dyn(2);

    let data = vec![
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let tensor = dense_c64(vec![i_idx.clone(), j_idx.clone()], data.clone());

    let (u, s, v) = svd::<Complex64>(&tensor, std::slice::from_ref(&i_idx))
        .expect("Complex SVD should succeed");

    let reconstructed = reconstruct_from_svd(&u, &s, &v);
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "Complex SVD reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_svd_complex_rectangular_reconstruction() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data = vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, -1.0),
        Complex64::new(0.0, 1.0),
        Complex64::new(4.0, 0.0),
        Complex64::new(-1.0, 3.0),
        Complex64::new(2.0, 2.0),
    ];
    let tensor = dense_c64(vec![i.clone(), j.clone()], data);

    let (u, s, v) = svd::<Complex64>(&tensor, std::slice::from_ref(&i))
        .expect("Complex rectangular SVD should succeed");
    let reconstructed = reconstruct_from_svd(&u, &s, &v);

    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "Complex rectangular SVD reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_svd_truncation() {
    // Test that truncation works correctly with a matrix that has a tiny singular value
    // Create a 2×2 diagonal matrix with singular values [1.0, 1e-14]
    // With rtol=1e-10, the second singular value should be truncated
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create diagonal matrix: [[1, 0], [0, 1e-14]]
    // This matrix has singular values [1.0, 1e-14]
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0] = 1.0
    data[3] = 1e-14; // [1, 1] = 1e-14

    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    // Use a more lenient rtol to ensure truncation happens
    let options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-10));
    let (u, s, v) =
        svd_with::<f64>(&tensor, std::slice::from_ref(&i), &options).expect("SVD should succeed");

    // With rtol=1e-10, the discarded weight ratio is (1e-14)^2 / (1^2 + (1e-14)^2) ≈ 1e-28
    // This is much less than (1e-10)^2 = 1e-20, so truncation should occur
    // However, we need to be careful: the actual criterion is sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
    // For σ = [1, 1e-14], we have:
    //   sum_i σ_i² = 1 + 1e-28 ≈ 1
    //   sum_{i>1} σ_i² = 1e-28
    //   ratio = 1e-28 / 1 = 1e-28 < (1e-10)^2 = 1e-20
    // So rank 1 should be retained

    // Check that truncation occurred: bond dimension should be 1
    assert_eq!(u.dims(), vec![2, 1], "U should be truncated to rank 1");
    assert_eq!(s.dims(), vec![1, 1], "S should be truncated to rank 1");
    assert_eq!(v.dims(), vec![2, 1], "V should be truncated to rank 1");

    // Retained singular value should be approximately 1.0
    assert!((s.sum().unwrap().real() - 1.0).abs() < 1e-10);
    assert!((s.norm_squared().unwrap() - 1.0).abs() < 1e-10);
}

#[test]
fn test_svd_with_policy_override() {
    // Test that svd_with can override the global default truncation policy
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create a matrix with singular values that will be truncated with a lenient rtol
    // but not with a strict rtol
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0] = 1.0
    data[3] = 1e-6; // [1, 1] = 1e-6

    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    // Save original default
    let original_policy = default_svd_truncation_policy();

    // Test with lenient threshold (should truncate)
    let lenient_options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-4));
    let (u1, s1, _v1) = svd_with::<f64>(&tensor, std::slice::from_ref(&i), &lenient_options)
        .expect("SVD should succeed");

    // Test with strict threshold (should not truncate)
    let strict_options = SvdOptions::new().with_policy(SvdTruncationPolicy::new(1e-12));
    let (u2, s2, _v2) = svd_with::<f64>(&tensor, std::slice::from_ref(&i), &strict_options)
        .expect("SVD should succeed");

    // With rtol=1e-4, the ratio is (1e-6)^2 / (1^2 + (1e-6)^2) ≈ 1e-12 < (1e-4)^2 = 1e-8
    // So truncation should occur
    let u1_dims = u1.dims();
    let s1_dims = s1.dims();
    assert_eq!(u1_dims[1], 1, "Lenient rtol should truncate to rank 1");
    assert_eq!(s1_dims[0], 1, "Lenient rtol should truncate to rank 1");

    // With rtol=1e-12, the ratio is 1e-12 < (1e-12)^2 = 1e-24 (not satisfied)
    // Actually, let's recalculate: (1e-6)^2 / (1^2 + (1e-6)^2) ≈ 1e-12
    // For rtol=1e-12, we need 1e-12 <= (1e-12)^2 = 1e-24, which is false
    // So with rtol=1e-12, we should NOT truncate (keep rank 2)
    let u2_dims = u2.dims();
    let s2_dims = s2.dims();
    assert_eq!(u2_dims[1], 2, "Strict rtol should keep full rank");
    assert_eq!(s2_dims[0], 2, "Strict rtol should keep full rank");

    // Restore original default
    set_default_svd_truncation_policy(original_policy).expect("Should restore original policy");
}

#[test]
fn test_default_svd_truncation_policy() {
    let original_policy = default_svd_truncation_policy();

    let expected_default = SvdTruncationPolicy::new(1e-12);
    set_default_svd_truncation_policy(expected_default).expect("Should set default policy");
    assert_eq!(default_svd_truncation_policy(), expected_default);

    let new_policy = SvdTruncationPolicy::new(1e-8).with_absolute();
    set_default_svd_truncation_policy(new_policy).expect("Should set policy");
    assert_eq!(default_svd_truncation_policy(), new_policy);

    assert!(
        set_default_svd_truncation_policy(SvdTruncationPolicy::new(-1.0)).is_err(),
        "Negative threshold should be rejected"
    );
    assert!(
        set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::NAN)).is_err(),
        "NaN threshold should be rejected"
    );
    assert!(
        set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::INFINITY)).is_err(),
        "Infinite threshold should be rejected"
    );

    set_default_svd_truncation_policy(original_policy).expect("Should restore original policy");
    assert_eq!(default_svd_truncation_policy(), original_policy);
}

#[test]
fn test_svd_uses_global_default() {
    // Test that svd() uses the global default truncation policy
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create a simple matrix
    let data = vec![1.0, 0.0, 0.0, 1.0];
    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    // Save original default
    let original_policy = default_svd_truncation_policy();

    // Change global default
    set_default_svd_truncation_policy(SvdTruncationPolicy::new(1e-6)).expect("Should set policy");

    // svd() should use the new global default
    let (u, s, _v) = svd::<f64>(&tensor, std::slice::from_ref(&i)).expect("SVD should succeed");

    // With threshold=1e-6, identity matrix should keep full rank (both singular values are 1.0)
    let u_dims = u.dims();
    let s_dims = s.dims();
    assert_eq!(u_dims[1], 2, "Identity matrix should keep full rank");
    assert_eq!(s_dims[0], 2, "Identity matrix should keep full rank");

    // Restore original
    set_default_svd_truncation_policy(original_policy).expect("Should restore original policy");
}

/// Helper: compute SVD reconstruction error for a given tensor and split.
fn svd_reconstruction_error_f64(t: &TensorDynLen, left_inds: &[DynIndex]) -> f64 {
    let (u, s, v) = svd::<f64>(t, left_inds).expect("SVD should succeed");
    let recon = reconstruct_from_svd(&u, &s, &v);
    let neg = recon
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = t.add(&neg).unwrap();
    diff.norm().unwrap()
}

/// Regression: SVD roundtrip with dim-1 axes.
///
/// tensor4all and tenferro both use column-major linearization semantics.
#[test]
fn test_svd_reconstruction_with_unit_dim_axis() {
    // [d=1, d=2, d=2] factorized with left_inds=[d=2, d=2]
    // → matrix shape (4, 1): column vector triggers the bug.
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(2);
    let i3 = Index::new_dyn(2);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = dense_f64(vec![i1.clone(), i2.clone(), i3.clone()], data);

    // left=[i2, i3], right=[i1]: 4×1 tall matrix
    let err = svd_reconstruction_error_f64(&tensor, &[i2.clone(), i3.clone()]);
    assert!(
        err < 1e-10,
        "SVD roundtrip with left=[d=2,d=2], right=[d=1] error: {err:.3e}"
    );

    // left=[i1], right=[i2, i3]: 1×4 wide matrix
    let err = svd_reconstruction_error_f64(&tensor, std::slice::from_ref(&i1));
    assert!(
        err < 1e-10,
        "SVD roundtrip with left=[d=1], right=[d=2,d=2] error: {err:.3e}"
    );
}

#[test]
fn test_svd_tall_matrix_preserves_left_axis_order() {
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(2);
    let i3 = Index::new_dyn(2);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = dense_f64(vec![i1, i2.clone(), i3.clone()], data);

    let (u, _s, _v) = svd::<f64>(&tensor, &[i2.clone(), i3.clone()]).expect("SVD should succeed");

    let norm = 30.0_f64.sqrt();
    let expected = dense_f64(
        vec![i2, i3, u.indices[2].clone()],
        vec![1.0 / norm, 2.0 / norm, 3.0 / norm, 4.0 / norm],
    );

    assert!(
        u.isapprox(&expected, 1e-10, 0.0).unwrap(),
        "left-axis order changed in tall-matrix SVD: maxabs diff = {}",
        u.sub(&expected).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_svd_reconstruction_with_multiple_unit_dims() {
    // [d=1, d=3, d=1, d=2] — multiple unit dimensions.
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(3);
    let i3 = Index::new_dyn(1);
    let i4 = Index::new_dyn(2);
    let data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    let tensor = dense_f64(vec![i1.clone(), i2.clone(), i3.clone(), i4.clone()], data);

    let err = svd_reconstruction_error_f64(&tensor, &[i1.clone(), i2.clone()]);
    assert!(
        err < 1e-10,
        "SVD multi-unit-dim [i1,i2] vs [i3,i4] error: {err:.3e}"
    );

    let err = svd_reconstruction_error_f64(&tensor, &[i2.clone(), i4.clone()]);
    assert!(
        err < 1e-10,
        "SVD multi-unit-dim [i2,i4] vs [i1,i3] error: {err:.3e}"
    );
}
