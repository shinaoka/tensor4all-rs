use tensor4all_tensor::{Storage, TensorDynLen, diag_tensor_dyn_len, diag_tensor_dyn_len_c64, diag_tensor_static_len, is_diag_tensor, is_diag_tensor_static};
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use num_complex::Complex64;
use std::sync::Arc;

#[test]
fn test_diag_tensor_creation() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];
    
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data.clone());
    assert_eq!(tensor.dims, vec![3, 3]);
    assert!(is_diag_tensor(&tensor));
}

#[test]
fn test_diag_tensor_creation_static() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];
    
    let tensor = diag_tensor_static_len([i.clone(), j.clone()], diag_data.clone());
    assert_eq!(tensor.dims, [3, 3]);
    assert!(is_diag_tensor_static(&tensor));
}

#[test]
#[should_panic(expected = "DiagTensor requires all indices to have the same dimension")]
fn test_diag_tensor_validation_different_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0];
    
    let _tensor = diag_tensor_dyn_len(vec![i, j], diag_data);
}

#[test]
fn test_diag_tensor_sum() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];
    
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data);
    let sum: f64 = tensor.sum();
    assert_eq!(sum, 6.0);
}

#[test]
fn test_diag_tensor_permute() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];
    
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone());
    
    // Permute: data should not change for DiagTensor
    let permuted = tensor.permute(&[2, 0, 1]);
    assert_eq!(permuted.dims, vec![3, 3, 3]);
    
    // Verify diagonal data is unchanged
    if let Storage::DiagF64(ref diag) = *permuted.storage {
        assert_eq!(diag.as_slice(), &diag_data);
    } else {
        panic!("Expected DiagF64 storage");
    }
}

#[test]
fn test_diag_tensor_contract_diag_diag_all_contracted() {
    // Create two 2x2 DiagTensors and contract all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];
    
    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_b);
    
    // Contract all indices: result should be scalar (inner product)
    let result = tensor_a.contract(&tensor_b);
    
    // Result should be scalar: 1*3 + 2*4 = 11
    assert_eq!(result.dims.len(), 0);
    if let Storage::DenseF64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.as_slice()[0], 11.0);
    } else {
        panic!("Expected DenseF64 storage for scalar result");
    }
}

#[test]
fn test_diag_tensor_contract_diag_diag_partial() {
    // Create A[i, j] and B[j, k], contract along j
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_a = vec![1.0, 2.0, 3.0];
    let diag_b = vec![4.0, 5.0, 6.0];
    
    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![j.clone(), k.clone()], diag_b);
    
    // Contract along j: result should be DiagTensor[i, k]
    let result = tensor_a.contract(&tensor_b);
    
    assert_eq!(result.dims, vec![3, 3]);
    assert!(is_diag_tensor(&result));
    
    // Result diagonal should be element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
    if let Storage::DiagF64(ref diag) = *result.storage {
        assert_eq!(diag.as_slice(), &vec![4.0, 10.0, 18.0]);
    } else {
        panic!("Expected DiagF64 storage");
    }
}

#[test]
fn test_diag_tensor_contract_diag_dense() {
    // Create DiagTensor A[i, j] and DenseTensor B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    
    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a);
    
    // Create DenseTensor B[j, k] with all ones
    let indices_b = vec![j.clone(), k.clone()];
    let dims_b = vec![2, 2];
    use tensor4all_tensor::storage::DenseStorageF64;
    let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 4]));
    let tensor_b: TensorDynLen<DynId, f64> = TensorDynLen::new(indices_b, dims_b, Arc::new(storage_b));
    
    // Contract along j: result should be DenseTensor[i, k]
    let result = tensor_a.contract(&tensor_b);
    
    assert_eq!(result.dims, vec![2, 2]);
    // Result should be DenseTensor (Diag×Dense converts Diag to Dense first)
    if let Storage::DenseF64(ref vec) = *result.storage {
        assert_eq!(vec.len(), 4);
        // A is diagonal [1, 2], B is all ones, so result[i, k] = A[i, i] * B[i, k] = diag[i] * 1
        // For i=0: result[0, k] = 1 * 1 = 1 for all k
        // For i=1: result[1, k] = 2 * 1 = 2 for all k
        // But wait, this is not quite right. Let me think...
        // Actually, when we convert Diag to Dense, A becomes:
        // A[0,0]=1, A[0,1]=0, A[1,0]=0, A[1,1]=2
        // Contracting with B (all ones) along j:
        // result[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 0*1 = 1
        // result[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*1 + 0*1 = 1
        // result[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 0*1 + 2*1 = 2
        // result[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 0*1 + 2*1 = 2
        // So result should be [[1, 1], [2, 2]]
        let vec_slice = vec.as_slice();
        assert_eq!(vec_slice[0], 1.0); // result[0,0]
        assert_eq!(vec_slice[1], 1.0); // result[0,1]
        assert_eq!(vec_slice[2], 2.0); // result[1,0]
        assert_eq!(vec_slice[3], 2.0); // result[1,1]
    } else {
        panic!("Expected DenseF64 storage");
    }
}

#[test]
fn test_diag_tensor_convert_to_dense() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];
    
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data);
    let dense_storage = tensor.storage.to_dense_storage(&tensor.dims);
    
    if let Storage::DenseF64(ref vec) = dense_storage {
        assert_eq!(vec.len(), 9); // 3x3 = 9
        let vec_slice = vec.as_slice();
        // Diagonal elements should be set
        assert_eq!(vec_slice[0], 1.0); // [0,0]
        assert_eq!(vec_slice[4], 2.0); // [1,1]
        assert_eq!(vec_slice[8], 3.0); // [2,2]
        // Off-diagonal should be zero
        assert_eq!(vec_slice[1], 0.0); // [0,1]
        assert_eq!(vec_slice[2], 0.0); // [0,2]
        assert_eq!(vec_slice[3], 0.0); // [1,0]
        assert_eq!(vec_slice[5], 0.0); // [1,2]
        assert_eq!(vec_slice[6], 0.0); // [2,0]
        assert_eq!(vec_slice[7], 0.0); // [2,1]
    } else {
        panic!("Expected DenseF64 storage");
    }
}

#[test]
fn test_diag_tensor_rank3() {
    // Test DiagTensor with rank 3
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_data = vec![1.0, 2.0];
    
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone());
    assert_eq!(tensor.dims, vec![2, 2, 2]);
    assert!(is_diag_tensor(&tensor));
    
    // Sum should work
    let sum: f64 = tensor.sum();
    assert_eq!(sum, 3.0);
}

#[test]
fn test_diag_tensor_complex() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_data = vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, 1.0)];
    
    let tensor = diag_tensor_dyn_len_c64(vec![i.clone(), j.clone()], diag_data.clone());
    assert_eq!(tensor.dims, vec![2, 2]);
    assert!(is_diag_tensor(&tensor));
    
    // Sum should work
    let sum: Complex64 = tensor.sum();
    assert_eq!(sum, Complex64::new(3.0, 1.5));
}

#[test]
fn test_diag_tensor_contract_rank3() {
    // Test contraction of rank-3 DiagTensors
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];
    
    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_a);
    let tensor_b = diag_tensor_dyn_len(vec![k.clone(), l.clone()], diag_b);
    
    // Contract along k: result should be DiagTensor[i, j, l]
    let result = tensor_a.contract(&tensor_b);
    
    assert_eq!(result.dims, vec![2, 2, 2]);
    assert!(is_diag_tensor(&result));
    
    // Result diagonal should be element-wise product: [1*3, 2*4] = [3, 8]
    if let Storage::DiagF64(ref diag) = *result.storage {
        assert_eq!(diag.as_slice(), &vec![3.0, 8.0]);
    } else {
        panic!("Expected DiagF64 storage");
    }
}

