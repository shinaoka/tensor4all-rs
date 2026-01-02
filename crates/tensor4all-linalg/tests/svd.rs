use num_complex::Complex64;
use std::sync::Arc;
use tensor4all_index::index::{DefaultIndex as Index, DynId};
use tensor4all_linalg::{
    default_svd_rtol, set_default_svd_rtol, svd, svd_c64, svd_with, SvdOptions,
};
use tensor4all_tensor::{Storage, TensorDynLen};

#[test]
fn test_svd_identity() {
    // Test SVD of a 2×2 identity matrix
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

    let (u, s, v) = svd(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Check dimensions
    assert_eq!(u.dims, vec![2, 2]);
    assert_eq!(s.dims, vec![2, 2]);
    assert_eq!(v.dims, vec![2, 2]);

    // Check indices
    assert_eq!(u.indices.len(), 2);
    assert_eq!(s.indices.len(), 2);
    assert_eq!(v.indices.len(), 2);

    // For identity matrix, singular values should be [1, 1] (in some order)
    // Extract singular values from diagonal storage
    match s.storage.as_ref() {
        Storage::DiagF64(diag) => {
            let s_vals = diag.as_slice();
            assert_eq!(s_vals.len(), 2);
            // Singular values should be 1.0 (may be in any order)
            let s0_ok = (s_vals[0] - 1.0).abs() < 1e-10;
            let s1_ok = (s_vals[1] - 1.0).abs() < 1e-10;
            assert!(
                s0_ok && s1_ok,
                "Singular values should be [1, 1], got [{}, {}]",
                s_vals[0],
                s_vals[1]
            );
        }
        _ => panic!("S should be diagonal storage"),
    }
}

#[test]
fn test_svd_simple_matrix() {
    // Test SVD of a simple 2×3 matrix: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage);

    let (u, s, v) = svd(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Check dimensions: m=2, n=3, k=min(2,3)=2
    assert_eq!(u.dims, vec![2, 2]);
    assert_eq!(s.dims, vec![2, 2]);
    assert_eq!(v.dims, vec![3, 2]);

    // Check indices
    assert_eq!(u.indices.len(), 2);
    assert_eq!(s.indices.len(), 2);
    assert_eq!(v.indices.len(), 2);

    // Check that U and V share the bond index
    assert_eq!(u.indices[1].id, s.indices[0].id);
    assert_eq!(s.indices[0].id, s.indices[1].id);
    assert_eq!(s.indices[1].id, v.indices[1].id);

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
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data.clone()),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![3, 4], storage);

    let (u, s, v) = svd(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Reconstruct: A = U * S * V^T
    // Note: Our SVD returns V (not V^T), so we need to compute U * S * V^T
    // First: S * V^T
    // V is n×k, V^T is k×n
    // We need to permute V's indices to get V^T shape, but the data needs to be transposed
    // Actually, let's contract S with V first, then with U
    // S is k×k (diagonal), V is n×k
    // To compute S * V^T, we need V^T which is k×n
    // But we have V which is n×k, so we need to transpose it

    // For now, let's use a simpler approach: manually reconstruct
    // Extract U, S, V data
    let u_data = match u.storage.as_ref() {
        Storage::DenseF64(dense) => dense.as_slice(),
        _ => panic!("U should be dense"),
    };
    let s_data = match s.storage.as_ref() {
        Storage::DiagF64(diag) => diag.as_slice(),
        _ => panic!("S should be diagonal"),
    };
    let v_data = match v.storage.as_ref() {
        Storage::DenseF64(dense) => dense.as_slice(),
        _ => panic!("V should be dense"),
    };

    // Reconstruct: A[i,j] = sum_r U[i,r] * S[r] * V[j,r]
    let m = u.dims[0];
    let n = v.dims[0];
    let k = u.dims[1];
    let mut reconstructed_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for r in 0..k {
                sum += u_data[i * k + r] * s_data[r] * v_data[j * k + r];
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
    for (i, (orig, recon)) in data.iter().zip(reconstructed_data_vec.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-8,
            "Element {}: original={}, reconstructed={}, diff={}",
            i,
            orig,
            recon,
            (orig - recon).abs()
        );
    }
}

#[test]
fn test_svd_invalid_rank() {
    // Test that SVD fails for rank-1 tensors
    let i = Index::new_dyn(2);

    let storage = Arc::new(Storage::new_dense_f64(2));
    let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(vec![i.clone()], vec![2], storage);

    let result = svd(&tensor, &[i.clone()]);
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

    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 3], storage);

    // Empty left_inds should fail
    let result = svd(&tensor, &[]);
    assert!(result.is_err(), "Expected error for empty left_inds");

    // All indices in left_inds should fail
    let result = svd(&tensor, &[i.clone(), j.clone()]);
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
    let (u, s, v) = svd(&tensor, &[i.clone()]).expect("SVD should succeed");

    // Check dimensions:
    // U should be [i, bond] = [2, min(2, 12)] = [2, 2]
    // S should be [bond, bond] = [2, 2]
    // V should be [j, k, bond] = [3, 4, 2]
    assert_eq!(u.dims, vec![2, 2]);
    assert_eq!(s.dims, vec![2, 2]);
    assert_eq!(v.dims, vec![3, 4, 2]);

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
    assert_eq!(s.indices[0].id, s.indices[1].id);
    assert_eq!(s.indices[1].id, v.indices[2].id);

    // Check that bond index has "Link" tag
    assert!(u.indices[1].tags().has_tag("Link"));
    assert!(s.indices[0].tags().has_tag("Link"));
    assert!(v.indices[2].tags().has_tag("Link"));
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
    let storage = Arc::new(Storage::DenseC64(
        tensor4all_tensor::storage::DenseStorageC64::from_vec(data.clone()),
    ));
    let tensor: TensorDynLen<DynId, Complex64> =
        TensorDynLen::new(vec![i_idx.clone(), j_idx.clone()], vec![2, 2], storage);

    let (u, s, v) = svd_c64(&tensor, &[i_idx.clone()]).expect("Complex SVD should succeed");

    let u_data = match u.storage.as_ref() {
        Storage::DenseC64(dense) => dense.as_slice(),
        _ => panic!("U should be dense complex"),
    };
    let s_data = match s.storage.as_ref() {
        Storage::DiagF64(diag) => diag.as_slice(),
        _ => panic!("S should be diagonal real (f64)"),
    };
    let v_data = match v.storage.as_ref() {
        Storage::DenseC64(dense) => dense.as_slice(),
        _ => panic!("V should be dense complex"),
    };

    let m = u.dims[0];
    let n = v.dims[0];
    let k = u.dims[1];

    // A[i,j] = Σ_r U[i,r] * S[r] * conj(V[j,r]), where S[r] is real.
    let mut reconstructed = vec![Complex64::new(0.0, 0.0); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for r in 0..k {
                sum += u_data[i * k + r] * s_data[r] * v_data[j * k + r].conj();
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
    data[0] = 1.0;      // [0, 0] = 1.0
    data[3] = 1e-14;    // [1, 1] = 1e-14

    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 2], storage);

    // Use a more lenient rtol to ensure truncation happens
    let options = SvdOptions::with_rtol(1e-10);
    let (u, s, v) = svd_with(&tensor, &[i.clone()], &options).expect("SVD should succeed");

    // With rtol=1e-10, the discarded weight ratio is (1e-14)^2 / (1^2 + (1e-14)^2) ≈ 1e-28
    // This is much less than (1e-10)^2 = 1e-20, so truncation should occur
    // However, we need to be careful: the actual criterion is sum_{i>r} σ_i² / sum_i σ_i² <= rtol²
    // For σ = [1, 1e-14], we have:
    //   sum_i σ_i² = 1 + 1e-28 ≈ 1
    //   sum_{i>1} σ_i² = 1e-28
    //   ratio = 1e-28 / 1 = 1e-28 < (1e-10)^2 = 1e-20
    // So rank 1 should be retained

    // Check that truncation occurred: bond dimension should be 1
    assert_eq!(u.dims, vec![2, 1], "U should be truncated to rank 1");
    assert_eq!(s.dims, vec![1, 1], "S should be truncated to rank 1");
    assert_eq!(v.dims, vec![2, 1], "V should be truncated to rank 1");

    // Check singular value
    match s.storage.as_ref() {
        Storage::DiagF64(diag) => {
            let s_vals = diag.as_slice();
            assert_eq!(s_vals.len(), 1);
            // The retained singular value should be approximately 1.0
            assert!(
                (s_vals[0] - 1.0).abs() < 1e-10,
                "Retained singular value should be ~1.0, got {}",
                s_vals[0]
            );
        }
        _ => panic!("S should be diagonal storage"),
    }
}

#[test]
fn test_svd_with_override() {
    // Test that svd_with can override the global default rtol
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create a matrix with singular values that will be truncated with a lenient rtol
    // but not with a strict rtol
    let mut data = vec![0.0; 4];
    data[0] = 1.0;      // [0, 0] = 1.0
    data[3] = 1e-6;     // [1, 1] = 1e-6

    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 2], storage);

    // Save original default
    let original_rtol = default_svd_rtol();

    // Test with lenient rtol (should truncate)
    let lenient_options = SvdOptions::with_rtol(1e-4);
    let (u1, s1, _v1) = svd_with(&tensor, &[i.clone()], &lenient_options)
        .expect("SVD should succeed");

    // Test with strict rtol (should not truncate)
    let strict_options = SvdOptions::with_rtol(1e-12);
    let (u2, s2, _v2) = svd_with(&tensor, &[i.clone()], &strict_options)
        .expect("SVD should succeed");

    // With rtol=1e-4, the ratio is (1e-6)^2 / (1^2 + (1e-6)^2) ≈ 1e-12 < (1e-4)^2 = 1e-8
    // So truncation should occur
    assert_eq!(u1.dims[1], 1, "Lenient rtol should truncate to rank 1");
    assert_eq!(s1.dims[0], 1, "Lenient rtol should truncate to rank 1");

    // With rtol=1e-12, the ratio is 1e-12 < (1e-12)^2 = 1e-24 (not satisfied)
    // Actually, let's recalculate: (1e-6)^2 / (1^2 + (1e-6)^2) ≈ 1e-12
    // For rtol=1e-12, we need 1e-12 <= (1e-12)^2 = 1e-24, which is false
    // So with rtol=1e-12, we should NOT truncate (keep rank 2)
    assert_eq!(u2.dims[1], 2, "Strict rtol should keep full rank");
    assert_eq!(s2.dims[0], 2, "Strict rtol should keep full rank");

    // Restore original default
    set_default_svd_rtol(original_rtol).expect("Should restore original rtol");
}

#[test]
fn test_default_svd_rtol() {
    // Test global default rtol getter and setter
    // Note: Other tests may have changed the global default, so we restore it first
    let original_rtol = default_svd_rtol();
    
    // Restore to expected default (1e-12) for this test
    set_default_svd_rtol(1e-12).expect("Should set default rtol");
    let current_rtol = default_svd_rtol();

    // Default should be 1e-12
    assert!(
        (current_rtol - 1e-12).abs() < 1e-15,
        "Default rtol should be 1e-12, got {}",
        current_rtol
    );

    // Test setting a new value
    let new_rtol = 1e-8;
    set_default_svd_rtol(new_rtol).expect("Should set rtol");
    assert!(
        (default_svd_rtol() - new_rtol).abs() < 1e-15,
        "Should retrieve the set rtol value"
    );

    // Test invalid values
    assert!(
        set_default_svd_rtol(-1.0).is_err(),
        "Negative rtol should be rejected"
    );
    assert!(
        set_default_svd_rtol(f64::NAN).is_err(),
        "NaN rtol should be rejected"
    );
    assert!(
        set_default_svd_rtol(f64::INFINITY).is_err(),
        "Infinite rtol should be rejected"
    );

    // Restore original
    set_default_svd_rtol(original_rtol).expect("Should restore original rtol");
    assert!(
        (default_svd_rtol() - original_rtol).abs() < 1e-15,
        "Should restore original rtol"
    );
}

#[test]
fn test_svd_uses_global_default() {
    // Test that svd() uses the global default rtol
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    // Create a simple matrix
    let data = vec![1.0, 0.0, 0.0, 1.0];
    let storage = Arc::new(Storage::DenseF64(
        tensor4all_tensor::storage::DenseStorageF64::from_vec(data),
    ));
    let tensor: TensorDynLen<DynId, f64> =
        TensorDynLen::new(vec![i.clone(), j.clone()], vec![2, 2], storage);

    // Save original default
    let original_rtol = default_svd_rtol();

    // Change global default
    set_default_svd_rtol(1e-6).expect("Should set rtol");

    // svd() should use the new global default
    let (u, s, _v) = svd(&tensor, &[i.clone()]).expect("SVD should succeed");

    // With rtol=1e-6, identity matrix should keep full rank (both singular values are 1.0)
    assert_eq!(u.dims[1], 2, "Identity matrix should keep full rank");
    assert_eq!(s.dims[0], 2, "Identity matrix should keep full rank");

    // Restore original
    set_default_svd_rtol(original_rtol).expect("Should restore original rtol");
}
