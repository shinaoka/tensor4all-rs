use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{qr, DynIndex};
use tensor4all_core::{TensorContractionLike, TensorDynLen};

fn dense_f64(indices: Vec<DynIndex>, data: Vec<f64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

fn dense_c64(indices: Vec<DynIndex>, data: Vec<Complex64>) -> TensorDynLen {
    TensorDynLen::from_dense(indices, data).unwrap()
}

#[test]
fn test_qr_identity() {
    // Test QR of a 2×2 identity matrix
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create identity matrix: [[1, 0], [0, 1]]
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0]
    data[3] = 1.0; // [1, 1]

    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 2]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 2);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_simple_matrix() {
    // Test QR of a simple 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = dense_f64(vec![i.clone(), j.clone()], data);

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions: m=2, n=3, k=min(2,3)=2
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 3]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 2);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_reconstruction() {
    // Test that Q * R reconstructs the original matrix
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);

    // Create a random-ish matrix
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let tensor = dense_f64(vec![i.clone(), j.clone()], data.clone());

    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Reconstruct: A = Q * R
    let reconstructed = q.contract_pair(&r).unwrap();

    // Check reconstruction accuracy
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "QR reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_qr_invalid_rank() {
    // Test that QR fails for rank-1 tensors
    let i = Index::new_dyn(2);

    let tensor = TensorDynLen::zeros::<f64>(vec![i.clone()]).unwrap();

    let result = qr::<f64>(&tensor, std::slice::from_ref(&i));
    assert!(result.is_err());
    // Expected: unfold_split returns an error for rank < 2
    if result.is_ok() {
        panic!("Expected error but got Ok");
    }
}

#[test]
fn test_qr_invalid_split() {
    // Test that QR fails when left_inds is empty or contains all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let tensor = TensorDynLen::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

    // Empty left_inds should fail
    let result = qr::<f64>(&tensor, &[]);
    assert!(result.is_err(), "Expected error for empty left_inds");

    // All indices in left_inds should fail
    let result = qr::<f64>(&tensor, &[i, j]);
    assert!(
        result.is_err(),
        "Expected error for all indices in left_inds"
    );
}

#[test]
fn test_qr_rank3() {
    // Test QR of a rank-3 tensor: split first index vs remaining two
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create a 2×3×4 tensor with some data
    let data = (0..24).map(|x| x as f64).collect::<Vec<_>>();
    let tensor = dense_f64(vec![i.clone(), j.clone(), k.clone()], data);

    // Split: left = [i], right = [j, k]
    // This unfolds to a 2×12 matrix
    let (q, r) = qr::<f64>(&tensor, std::slice::from_ref(&i)).expect("QR should succeed");

    // Check dimensions:
    // Q should be [i, bond] = [2, min(2, 12)] = [2, 2]
    // R should be [bond, j, k] = [2, 3, 4]
    assert_eq!(q.dims(), vec![2, 2]);
    assert_eq!(r.dims(), vec![2, 3, 4]);

    // Check indices
    assert_eq!(q.indices.len(), 2);
    assert_eq!(r.indices.len(), 3);

    // Check that Q has left index first, then bond
    assert_eq!(q.indices[0].id, i.id);

    // Check that R has bond first, then right indices
    assert_eq!(r.indices[0].id, q.indices[1].id); // bond index
    assert_eq!(r.indices[1].id, j.id);
    assert_eq!(r.indices[2].id, k.id);

    // Check that Q and R share the bond index
    assert_eq!(q.indices[1].id, r.indices[0].id);

    // Check that bond index has "Link" tag
    assert!(q.indices[1].tags().has_tag("Link"));
    assert!(r.indices[0].tags().has_tag("Link"));
}

#[test]
fn test_qr_nontrivial_split_reconstruction() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);

    let data = (1..=24).map(|x| x as f64).collect::<Vec<_>>();
    let tensor = dense_f64(vec![i.clone(), j.clone(), k.clone(), l.clone()], data);

    let (q, r) = qr::<f64>(&tensor, &[i.clone(), k.clone()]).expect("QR should succeed");
    let reconstructed = q.contract_pair(&r).unwrap();

    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "QR nontrivial split reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

#[test]
fn test_qr_complex_reconstruction() {
    // Complex matrix: [[i, 0], [0, 2]]
    let i_idx = Index::new_dyn(2);
    let j_idx = Index::new_dyn(2);

    let data = vec![
        Complex64::new(0.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let tensor = dense_c64(vec![i_idx.clone(), j_idx.clone()], data.clone());

    let (q, r) =
        qr::<Complex64>(&tensor, std::slice::from_ref(&i_idx)).expect("Complex QR should succeed");

    let reconstructed = q.contract_pair(&r).unwrap();
    assert!(
        tensor.isapprox(&reconstructed, 1e-8, 0.0).unwrap(),
        "Complex QR reconstruction failed: maxabs diff = {}",
        tensor.sub(&reconstructed).unwrap().maxabs().unwrap()
    );
}

/// Helper: compute ||Q*R - T|| via tensor contraction for any tensor shape.
fn qr_reconstruction_error_f64(t: &TensorDynLen, left_inds: &[DynIndex]) -> f64 {
    let (q, r) = qr::<f64>(t, left_inds).expect("QR should succeed");
    let recon = q.contract_pair(&r).unwrap();
    let neg = recon
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = t.add(&neg).unwrap();
    diff.norm().unwrap()
}

fn qr_reconstruction_error_c64(t: &TensorDynLen, left_inds: &[DynIndex]) -> f64 {
    let (q, r) = qr::<Complex64>(t, left_inds).expect("QR should succeed");
    let recon = q.contract_pair(&r).unwrap();
    let neg = recon
        .scale(tensor4all_core::AnyScalar::new_real(-1.0))
        .unwrap();
    let diff = t.add(&neg).unwrap();
    diff.norm().unwrap()
}

#[test]
fn test_qr_rank3_reconstruction_via_contract() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);
    let k = Index::new_dyn(5);

    let data: Vec<f64> = (0..60).map(|x| (x as f64) * 0.1 + 0.01).collect();
    let tensor = dense_f64(vec![i.clone(), j.clone(), k.clone()], data);

    // Split [i] vs [j, k]: 3×20 matrix
    let err = qr_reconstruction_error_f64(&tensor, std::slice::from_ref(&i));
    assert!(err < 1e-10, "rank-3 QR reconstruction error: {err:.3e}");

    // Split [i, j] vs [k]: 12×5 matrix
    let err = qr_reconstruction_error_f64(&tensor, &[i, j]);
    assert!(err < 1e-10, "rank-3 QR reconstruction error: {err:.3e}");
}

#[test]
fn test_qr_rank4_reconstruction_via_contract() {
    let a = Index::new_dyn(3);
    let b = Index::new_dyn(2);
    let c = Index::new_dyn(4);
    let d = Index::new_dyn(2);

    let data: Vec<f64> = (0..48).map(|x| (x as f64).sin()).collect();
    let tensor = dense_f64(vec![a.clone(), b.clone(), c.clone(), d.clone()], data);

    // Split [a, b] vs [c, d]: 6×8 matrix
    let err = qr_reconstruction_error_f64(&tensor, &[a.clone(), b.clone()]);
    assert!(err < 1e-10, "rank-4 QR [a,b] vs [c,d] error: {err:.3e}");

    // Split [a] vs [b, c, d]: 3×16 matrix
    let err = qr_reconstruction_error_f64(&tensor, std::slice::from_ref(&a));
    assert!(err < 1e-10, "rank-4 QR [a] vs [b,c,d] error: {err:.3e}");

    // Non-contiguous split [a, c] vs [b, d]: 12×4 matrix
    let err = qr_reconstruction_error_f64(&tensor, &[a, c]);
    assert!(
        err < 1e-10,
        "rank-4 QR non-contiguous [a,c] vs [b,d] error: {err:.3e}"
    );
}

#[test]
fn test_qr_complex_rank3_reconstruction() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(4);
    let k = Index::new_dyn(2);

    let data: Vec<Complex64> = (0..24)
        .map(|x| Complex64::new((x as f64).sin(), (x as f64).cos()))
        .collect();
    let tensor = dense_c64(vec![i.clone(), j.clone(), k.clone()], data);

    let err = qr_reconstruction_error_c64(&tensor, &[i.clone(), j.clone()]);
    assert!(
        err < 1e-10,
        "complex rank-3 QR [i,j] vs [k] error: {err:.3e}"
    );

    let err = qr_reconstruction_error_c64(&tensor, &[i]);
    assert!(
        err < 1e-10,
        "complex rank-3 QR [i] vs [j,k] error: {err:.3e}"
    );
}

/// Regression: QR roundtrip with dim-1 axes.
///
/// tensor4all and tenferro both use column-major linearization semantics.
#[test]
fn test_qr_reconstruction_with_unit_dim_axis() {
    // [d=1, d=2, d=2] factorized with left_inds=[d=2, d=2]
    // → matrix shape (4, 1): column vector triggers the bug.
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(2);
    let i3 = Index::new_dyn(2);
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let tensor = dense_f64(vec![i1.clone(), i2.clone(), i3.clone()], data);

    // left=[i2, i3], right=[i1]: 4×1 tall matrix
    let err = qr_reconstruction_error_f64(&tensor, &[i2.clone(), i3.clone()]);
    assert!(
        err < 1e-10,
        "QR roundtrip with left=[d=2,d=2], right=[d=1] error: {err:.3e}"
    );

    // left=[i1], right=[i2, i3]: 1×4 wide matrix
    let err = qr_reconstruction_error_f64(&tensor, std::slice::from_ref(&i1));
    assert!(
        err < 1e-10,
        "QR roundtrip with left=[d=1], right=[d=2,d=2] error: {err:.3e}"
    );
}

#[test]
fn test_qr_reconstruction_with_multiple_unit_dims() {
    // [d=1, d=3, d=1, d=2] — multiple unit dimensions.
    let i1 = Index::new_dyn(1);
    let i2 = Index::new_dyn(3);
    let i3 = Index::new_dyn(1);
    let i4 = Index::new_dyn(2);
    let data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
    let tensor = dense_f64(vec![i1.clone(), i2.clone(), i3.clone(), i4.clone()], data);

    // Various splits all involving unit-dim axes
    let err = qr_reconstruction_error_f64(&tensor, &[i1.clone(), i2.clone()]);
    assert!(
        err < 1e-10,
        "QR multi-unit-dim [i1,i2] vs [i3,i4] error: {err:.3e}"
    );

    let err = qr_reconstruction_error_f64(&tensor, &[i2.clone(), i4.clone()]);
    assert!(
        err < 1e-10,
        "QR multi-unit-dim [i2,i4] vs [i1,i3] error: {err:.3e}"
    );

    let err = qr_reconstruction_error_f64(&tensor, std::slice::from_ref(&i1));
    assert!(
        err < 1e-10,
        "QR multi-unit-dim [i1] vs [i2,i3,i4] error: {err:.3e}"
    );
}
