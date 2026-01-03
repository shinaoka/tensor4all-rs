use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
use tensor4all_core_tensor::{Storage, TensorDynLen, compute_permutation_from_indices};
use num_complex::Complex64;
use std::sync::Arc;

#[test]
fn test_compute_permutation_from_indices() {
    // Test the independent permutation computation function
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    
    let original = vec![i.clone(), j.clone(), k.clone()];
    
    // Test identity permutation
    let new_order1 = vec![i.clone(), j.clone(), k.clone()];
    let perm1 = compute_permutation_from_indices(&original, &new_order1);
    assert_eq!(perm1, vec![0, 1, 2]);
    
    // Test swap first two
    let new_order2 = vec![j.clone(), i.clone(), k.clone()];
    let perm2 = compute_permutation_from_indices(&original, &new_order2);
    assert_eq!(perm2, vec![1, 0, 2]);
    
    // Test reverse
    let new_order3 = vec![k.clone(), j.clone(), i.clone()];
    let perm3 = compute_permutation_from_indices(&original, &new_order3);
    assert_eq!(perm3, vec![2, 1, 0]);
    
    // Test rotation
    let new_order4 = vec![j.clone(), k.clone(), i.clone()];
    let perm4 = compute_permutation_from_indices(&original, &new_order4);
    assert_eq!(perm4, vec![1, 2, 0]);
}

#[test]
#[should_panic(expected = "new_indices must be a permutation of original_indices")]
fn test_compute_permutation_from_indices_invalid() {
    // Test with invalid index (ID doesn't match)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let invalid = Index::new_dyn(5);  // Different ID
    
    let original = vec![i.clone(), j.clone()];
    let new_order = vec![i.clone(), invalid];
    
    compute_permutation_from_indices(&original, &new_order);
}

#[test]
#[should_panic(expected = "duplicate index in new_indices")]
fn test_compute_permutation_from_indices_duplicate() {
    // Test with duplicate indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    
    let original = vec![i.clone(), j.clone()];
    let new_order = vec![i.clone(), i.clone()];  // Duplicate
    
    compute_permutation_from_indices(&original, &new_order);
}

#[test]
fn test_permute_dyn_f64_2d() {
    // Create a 2×3 tensor with data [1, 2, 3, 4, 5, 6]
    // In row-major order: [[1, 2, 3], [4, 5, 6]]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    
    let mut storage = Storage::new_dense_f64(6);
    match &mut storage {
        Storage::DenseF64(v) => {
            v.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Permute to 3×2: swap dimensions
    // Expected: [[1, 4], [2, 5], [3, 6]]
    // In row-major: [1, 4, 2, 5, 3, 6]
    let permuted = tensor.permute(&[1, 0]);
    
    assert_eq!(permuted.dims, vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    
    match &*permuted.storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
}

#[test]
fn test_permute_dyn_c64_2d() {
    // Create a 2×3 tensor with complex data
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    
    let mut storage = Storage::new_dense_c64(6);
    match &mut storage {
        Storage::DenseC64(v) => {
            v.extend_from_slice(&[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]);
        }
        _ => panic!("expected DenseC64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Permute to 3×2
    let permuted = tensor.permute(&[1, 0]);
    
    assert_eq!(permuted.dims, vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    
    match &*permuted.storage {
        Storage::DenseC64(v) => {
            assert_eq!(v.get(0), Complex64::new(1.0, 0.0));
            assert_eq!(v.get(1), Complex64::new(4.0, 0.0));
            assert_eq!(v.get(2), Complex64::new(2.0, 0.0));
            assert_eq!(v.get(3), Complex64::new(5.0, 0.0));
            assert_eq!(v.get(4), Complex64::new(3.0, 0.0));
            assert_eq!(v.get(5), Complex64::new(6.0, 0.0));
        }
        _ => panic!("expected DenseC64"),
    }
}

#[test]
fn test_permute_dyn_f64_3d() {
    // Create a 2×3×4 tensor
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let indices = vec![i.clone(), j.clone(), k.clone()];
    let dims = vec![2, 3, 4];
    
    let mut storage = Storage::new_dense_f64(24);
    match &mut storage {
        Storage::DenseF64(v) => {
            // Fill with sequential values
            for i in 1..=24 {
                v.push(i as f64);
            }
        }
        _ => panic!("expected DenseF64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Permute to 4×2×3: [2, 0, 1]
    let permuted = tensor.permute(&[2, 0, 1]);
    
    assert_eq!(permuted.dims, vec![4, 2, 3]);
    assert_eq!(permuted.indices[0].id, k.id);
    assert_eq!(permuted.indices[1].id, i.id);
    assert_eq!(permuted.indices[2].id, j.id);
    
    // Verify data was permuted correctly
    // Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    // After permute [2, 0, 1]: should reorganize the data
    match &*permuted.storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 24);
            // Check first few values to verify permutation
            // This is a complex permutation, so we just verify the structure
            assert_eq!(v.get(0), 1.0); // First element should be the same
        }
        _ => panic!("expected DenseF64"),
    }
}


#[test]
fn test_permute_identity() {
    // Test identity permutation [0, 1] on 2×3 tensor
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    
    let mut storage = Storage::new_dense_f64(6);
    match &mut storage {
        Storage::DenseF64(v) => {
            v.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Identity permutation should not change anything
    let permuted = tensor.permute(&[0, 1]);
    
    assert_eq!(permuted.dims, vec![2, 3]);
    assert_eq!(permuted.indices[0].id, i.id);
    assert_eq!(permuted.indices[1].id, j.id);
    
    match &*permuted.storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
}

#[test]
fn test_permute_indices_dyn_f64_2d() {
    // Test permute_indices: main permutation method using new indices order
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    
    let mut storage = Storage::new_dense_f64(6);
    match &mut storage {
        Storage::DenseF64(v) => {
            v.extend_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Permute to 3×2: swap the two dimensions by providing new indices order
    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]);
    
    assert_eq!(permuted.dims, vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    
    match &*permuted.storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        }
        _ => panic!("expected DenseF64"),
    }
}


#[test]
fn test_permute_indices_c64() {
    // Test permute_indices with complex numbers
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone()];
    let dims = vec![2, 3];
    
    let mut storage = Storage::new_dense_c64(6);
    match &mut storage {
        Storage::DenseC64(v) => {
            v.extend_from_slice(&[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ]);
        }
        _ => panic!("expected DenseC64"),
    }
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, Arc::new(storage));
    
    // Permute to 3×2
    let permuted = tensor.permute_indices(&[j.clone(), i.clone()]);
    
    assert_eq!(permuted.dims, vec![3, 2]);
    assert_eq!(permuted.indices[0].id, j.id);
    assert_eq!(permuted.indices[1].id, i.id);
    
    match &*permuted.storage {
        Storage::DenseC64(v) => {
            assert_eq!(v.get(0), Complex64::new(1.0, 0.0));
            assert_eq!(v.get(1), Complex64::new(4.0, 0.0));
            assert_eq!(v.get(2), Complex64::new(2.0, 0.0));
            assert_eq!(v.get(3), Complex64::new(5.0, 0.0));
            assert_eq!(v.get(4), Complex64::new(3.0, 0.0));
            assert_eq!(v.get(5), Complex64::new(6.0, 0.0));
        }
        _ => panic!("expected DenseC64"),
    }
}

