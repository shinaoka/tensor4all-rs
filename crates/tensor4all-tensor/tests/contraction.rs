use tensor4all_tensor::{Storage, TensorDynLen, TensorStaticLen};
use tensor4all_tensor::storage::DenseStorageF64;
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_core::index_ops::common_inds;
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
    let tensor_a: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k] with all ones
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![3, 4];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]));
    let tensor_b: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

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
fn test_contract_static_len_matrix_multiplication() {
    // Create two matrices: A[i, j] and B[j, k]
    // Result should be C[i, k] = A[i, j] * B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(4);

    // Create tensor A[i, j] with all ones
    let indices_a = [i.clone(), j.clone()];
    let dims_a = [2, 3];
    let storage_a = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]));
    let tensor_a: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k] with all ones
    let indices_b = [j.clone(), k.clone()];
    let dims_b = [3, 4];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]));
    let tensor_b: TensorStaticLen<2, DynId, f64> = TensorStaticLen::new(indices_b, dims_b, Arc::new(storage_b));

    // Contract along j: result should be C[i, k] with all 3.0
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
    let tensor_a: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_a, dims_a, storage_a);

    let indices_b = vec![k.clone()];
    let dims_b = vec![4];
    let storage_b = Arc::new(Storage::new_dense_f64(4));
    let tensor_b: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_b, dims_b, storage_b);

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
    let tensor_a: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_a, dims_a, Arc::new(storage_a));

    // Create tensor B[j, k, l] with all ones
    let indices_b = vec![j.clone(), k.clone(), l.clone()];
    let dims_b = vec![3, 4, 5];
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 60]));
    let tensor_b: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));

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

