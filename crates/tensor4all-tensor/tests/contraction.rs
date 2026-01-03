use tensor4all_tensor::{Storage, TensorDynLen};
use tensor4all_tensor::storage::{DenseStorageF64, DenseStorageC64};
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_core::index_ops::common_inds;
use num_complex::Complex64;
use std::sync::Arc;

#[test]
fn test_common_inds() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let indices_b = vec![j.clone(), k.clone()];

    let common = common_inds(&indices_a, &indices_b);
    assert_eq!(common.len(), 1);
    assert_eq!(common[0].id, j.id);
}

#[test]
fn test_contract_dyn_len_matrix_multiplication() {
    // Create two matrices: A[i, j] and B[j, k]
    // Result should be C[i, k] = A[i, j] * B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create tensor A[i, j] with all ones
    let indices_a = vec![i.clone(), j.clone()];
    let dims_a = vec![2, 3];
    let storage_a = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k] with all ones
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![3, 4];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

    // Contract along j: result should be C[i, k] with all 3.0 (since each element is sum of 3 ones)
    let result = tensor_a.contract(&tensor_b);
    assert_eq!(result.dims, vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, k.id);

    // Check that all elements are 3.0
    if let Storage::DenseF64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 8); // 2 * 4 = 8
        for &val in vec.iter() {
            assert_eq!(val, 3.0);
        }
    } else {
        panic!("Expected DenseF64 storage");
    }
}


#[test]
#[should_panic(expected = "No common indices found")]
fn test_contract_no_common_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    let indices_a = vec![i.clone(), j.clone()];
    let dims_a = vec![2, 3];
    let storage_a = Arc::new(Storage::new_dense_f64(6));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, storage_a);

    let indices_b = vec![k.clone()];
    let dims_b = vec![4];
    let storage_b = Arc::new(Storage::new_dense_f64(4));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, storage_b);

    // This should panic because there are no common indices
    let _result = tensor_a.contract(&tensor_b);
}

#[test]
fn test_contract_three_indices() {
    // Create A[i, j, k] and B[j, k, l]
    // Contract along j and k: result should be C[i, l]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);
    let l = Index::new_dyn(5);

    // Create tensor A[i, j, k] with all ones
    let indices_a = vec![i.clone(), j.clone(), k.clone()];
    let dims_a = vec![2, 3, 4];
    let storage_a = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 24]));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k, l] with all ones
    let indices_b = vec![j.clone(), k.clone(), l.clone()];
    let dims_b = vec![3, 4, 5];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 60]));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

    // Contract along j and k: result should be C[i, l] with all 12.0 (3 * 4 = 12)
    let result = tensor_a.contract(&tensor_b);
    assert_eq!(result.dims, vec![2, 5]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, l.id);

    // Check that all elements are 12.0
    if let Storage::DenseF64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 10); // 2 * 5 = 10
        for &val in vec.as_slice().iter() {
            assert_eq!(val, 12.0);
        }
    } else {
        panic!("Expected DenseF64 storage");
    }
}

#[test]
fn test_contract_mixed_f64_c64() {
    // Test contraction between f64 and Complex64 tensors
    // A[i, j] (f64) × B[j, k] (Complex64) = C[i, k] (Complex64)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // Create tensor A[i, j] with all 1.0 (f64)
    let indices_a = vec![i.clone(), j.clone()];
    let dims_a = vec![2, 2];
    let storage_a = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 4]));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k] with complex values: [[1+2i, 3+4i], [5+6i, 7+8i]]
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![2, 2];
    let storage_b = Storage::DenseC64(DenseStorageC64::from_vec(vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ]));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

    // Contract along j: result should be C[i, k] (Complex64)
    // Expected result: [[1+2i + 5+6i, 3+4i + 7+8i], [1+2i + 5+6i, 3+4i + 7+8i]]
    //                  = [[6+8i, 10+12i], [6+8i, 10+12i]]
    let result = tensor_a.contract(&tensor_b);
    assert_eq!(result.dims, vec![2, 2]);
    assert_eq!(result.indices.len(), 2);
    assert_eq!(result.indices[0].id, i.id);
    assert_eq!(result.indices[1].id, k.id);

    // Check result storage type and values
    if let Storage::DenseC64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 4);
        // All elements should be the same: sum of first column and sum of second column
        // First row: [6+8i, 10+12i]
        assert!((vec.get(0) - Complex64::new(6.0, 8.0)).norm() < 1e-10);
        assert!((vec.get(1) - Complex64::new(10.0, 12.0)).norm() < 1e-10);
        // Second row: [6+8i, 10+12i]
        assert!((vec.get(2) - Complex64::new(6.0, 8.0)).norm() < 1e-10);
        assert!((vec.get(3) - Complex64::new(10.0, 12.0)).norm() < 1e-10);
    } else {
        panic!("Expected DenseC64 storage for mixed-type contraction");
    }
}

#[test]
fn test_contract_mixed_c64_f64() {
    // Test contraction between Complex64 and f64 tensors (reverse order)
    // A[i, j] (Complex64) × B[j, k] (f64) = C[i, k] (Complex64)
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);

    // Create tensor A[i, j] with complex values
    let indices_a = vec![i.clone(), j.clone()];
    let dims_a = vec![2, 2];
    let storage_a = Storage::DenseC64(DenseStorageC64::from_vec(vec![
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ]));
    let tensor_a: TensorDynLen<DynId> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k] with all 1.0 (f64)
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![2, 2];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 4]));
    let tensor_b: TensorDynLen<DynId> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

    // Contract along j: result should be C[i, k] (Complex64)
    // For A[i,j] * B[j,k] where A is complex and B is real:
    // C[i,k] = sum_j A[i,j] * B[j,k]
    // With A = [[1+2i, 3+4i], [5+6i, 7+8i]] and B = [[1, 1], [1, 1]]
    // C[0,0] = (1+2i)*1 + (3+4i)*1 = 4+6i
    // C[0,1] = (1+2i)*1 + (3+4i)*1 = 4+6i
    // C[1,0] = (5+6i)*1 + (7+8i)*1 = 12+14i
    // C[1,1] = (5+6i)*1 + (7+8i)*1 = 12+14i
    let result = tensor_a.contract(&tensor_b);
    assert_eq!(result.dims, vec![2, 2]);
    
    // Check result storage type
    if let Storage::DenseC64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 4);
        // Check actual computed values
        assert!((vec.get(0) - Complex64::new(4.0, 6.0)).norm() < 1e-10);
        assert!((vec.get(1) - Complex64::new(4.0, 6.0)).norm() < 1e-10);
        assert!((vec.get(2) - Complex64::new(12.0, 14.0)).norm() < 1e-10);
        assert!((vec.get(3) - Complex64::new(12.0, 14.0)).norm() < 1e-10);
    } else {
        panic!("Expected DenseC64 storage for mixed-type contraction");
    }
}

