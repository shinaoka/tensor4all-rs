use num_complex::Complex64;
use std::sync::Arc;
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_linalg::{qr, qr_c64};
use tensor4all_tensor::{Storage, TensorDynLen};

#[test]
fn test_qr_identity() {
    // Test QR of a 2×2 identity matrix
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create identity matrix: [[1, 0], [0, 1]]
    let mut data = vec![0.0; 4];
    data[0] = 1.0; // [0, 0]
    data[3] = 1.0; // [1, 1]

    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 2], storage);

    let (q, r) = qr(&tensor, &[i.clone()]).expect("QR should succeed");

    // Check dimensions
    assert_eq!(q.dims, vec![2, 2]);
    assert_eq!(r.dims, vec![2, 2]);

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
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage);

    let (q, r) = qr(&tensor, &[i.clone()]).expect("QR should succeed");

    // Check dimensions: m=2, n=3, k=min(2,3)=2
    assert_eq!(q.dims, vec![2, 2]);
    assert_eq!(r.dims, vec![2, 3]);

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
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data.clone()),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![3, 4], storage);

    let (q, r) = qr(&tensor, &[i.clone()]).expect("QR should succeed");

    // Reconstruct: A = Q * R
    // Extract Q and R data
    let q_data = match q.storage.as_ref() {
        Storage::DenseF64(dense) => dense.as_slice(),
        _ => panic!("Q should be dense"),
    };
    let r_data = match r.storage.as_ref() {
        Storage::DenseF64(dense) => dense.as_slice(),
        _ => panic!("R should be dense"),
    };

    // Reconstruct: A[i,j] = sum_r Q[i,r] * R[r,j]
    let m = q.dims[0];
    let n = r.dims[1];
    let k = q.dims[1];
    let mut reconstructed_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for r in 0..k {
                sum += q_data[i * k + r] * r_data[r * n + j];
            }
            reconstructed_data[i * n + j] = sum;
        }
    }

    // Create reconstructed tensor for comparison
    let reconstructed_storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(reconstructed_data),
    ));
    let reconstructed: TensorDynLen<DynId, f64> = TensorDynLen::new(
        vec![i.clone(), j.clone()],
        vec![m, n],
        reconstructed_storage,
    );

    // Check dimensions match
    assert_eq!(reconstructed.dims, vec![3, 4]);

    // Extract reconstructed data
    let reconstructed_data_vec = match reconstructed.storage.as_ref() {
        Storage::DenseF64(dense) => dense.as_slice().to_vec(),
        _ => panic!("Reconstructed should be dense"),
    };

    // Check reconstruction accuracy (within tolerance)
    for (idx, (orig, recon)) in data.iter().zip(reconstructed_data_vec.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-8,
            "Element {}: original={}, reconstructed={}, diff={}",
            idx,
            orig,
            recon,
            (orig - recon).abs()
        );
    }
}

#[test]
fn test_qr_invalid_rank() {
    // Test that QR fails for rank-1 tensors
    let i = Index::new_dyn(2);

    let storage = Arc::new(Storage::new_dense_f64(2));
    let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(vec![i.clone()], vec![2], storage);

    let result = qr(&tensor, &[i.clone()]);
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

    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage);

    // Empty left_inds should fail
    let result = qr(&tensor, &[]);
    assert!(result.is_err(), "Expected error for empty left_inds");

    // All indices in left_inds should fail
    let result = qr(&tensor, &[i.clone(), j.clone()]);
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
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(
        vec![i.clone(), j.clone(), k.clone()],
        vec![2, 3, 4],
        storage,
    );

    // Split: left = [i], right = [j, k]
    // This unfolds to a 2×12 matrix
    let (q, r) = qr(&tensor, &[i.clone()]).expect("QR should succeed");

    // Check dimensions:
    // Q should be [i, bond] = [2, min(2, 12)] = [2, 2]
    // R should be [bond, j, k] = [2, 3, 4]
    assert_eq!(q.dims, vec![2, 2]);
    assert_eq!(r.dims, vec![2, 3, 4]);

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
    let storage = Arc::new(Storage::DenseC64(
        tensor4all_tensor::storage::DenseStorageC64::from_vec(data.clone()),
    ));
    let tensor: TensorDynLen<DynId, Complex64> =
        TensorDynLen::new(vec![i_idx.clone(), j_idx.clone()], vec![2, 2], storage);

    let (q, r) = qr_c64(&tensor, &[i_idx.clone()]).expect("Complex QR should succeed");

    let q_data = match q.storage.as_ref() {
        Storage::DenseC64(dense) => dense.as_slice(),
        _ => panic!("Q should be dense complex"),
    };
    let r_data = match r.storage.as_ref() {
        Storage::DenseC64(dense) => dense.as_slice(),
        _ => panic!("R should be dense complex"),
    };

    let m = q.dims[0];
    let n = r.dims[1];
    let k = q.dims[1];

    // A[i,j] = Σ_r Q[i,r] * R[r,j] (no conjugation needed for QR)
    let mut reconstructed = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for r in 0..k {
                sum += q_data[i * k + r] * r_data[r * n + j];
            }
            reconstructed[i * n + j] = sum;
        }
    }

    for (idx, (orig, recon)) in data.iter().zip(reconstructed.iter()).enumerate() {
        let diff = *orig - *recon;
        assert!(
            diff.norm() < 1e-8,
            "Element {}: original={:?}, reconstructed={:?}, diff={:?}",
            idx,
            orig,
            recon,
            diff
        );
    }
}
