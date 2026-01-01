use tensor4all_core::index::{Index, DynId, generate_id};
use tensor4all_core::storage::{Storage, make_mut_storage};
use tensor4all_core::tensor::{TensorDynLen, TensorStaticLen, AnyScalar};
use std::sync::Arc;

#[test]
fn test_id_generation() {
    let id1 = generate_id();
    let id2 = generate_id();
    let id3 = generate_id();
    
    // IDs should be unique and monotonically increasing
    assert!(id1 < id2);
    assert!(id2 < id3);
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
}

#[test]
fn test_index_dyn() {
    let idx = Index::new_dyn(8);
    assert_eq!(idx.size, 8);
    assert!(idx.id.0 > 0);
}

#[test]
fn test_index_with_custom_id() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct StaticId;
    
    let idx = Index::new(StaticId, 16);
    assert_eq!(idx.size, 16);
    assert_eq!(idx.id, StaticId);
}

#[test]
fn test_storage_dense_f64() {
    let storage = Storage::new_dense_f64(10);
    assert_eq!(storage.len(), 0);
    
    match storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.capacity(), 10);
        }
    }
}

#[test]
fn test_cow_storage() {
    let mut storage1 = Arc::new(Storage::new_dense_f64(0));
    let storage2 = Arc::clone(&storage1);
    
    // Initially, both point to the same storage
    assert!(Arc::ptr_eq(&storage1, &storage2));
    
    // Mutate storage1 (should clone due to COW)
    {
        let mut_storage = make_mut_storage(&mut storage1);
        match mut_storage {
            Storage::DenseF64(v) => {
                v.push(1.0);
                v.push(2.0);
            }
        }
    }
    
    // After mutation, they should be different
    assert!(!Arc::ptr_eq(&storage1, &storage2));
    
    // storage2 should be unchanged
    match storage2.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
    }
    
    // storage1 should have the new data
    match storage1.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[1], 2.0);
        }
    }
}

#[test]
fn test_tensor_dyn_len_creation() {
    let indices = vec![
        Index::new_dyn(2),
        Index::new_dyn(3),
    ];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    
    let tensor: TensorDynLen<DynId, AnyScalar> = TensorDynLen::new(indices, dims, storage);
    assert_eq!(tensor.indices.len(), 2);
    assert_eq!(tensor.dims.len(), 2);
    assert_eq!(tensor.dims[0], 2);
    assert_eq!(tensor.dims[1], 3);
}

#[test]
#[should_panic(expected = "indices and dims must have the same length")]
fn test_tensor_dyn_len_mismatch() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2, 3]; // mismatch
    let storage = Arc::new(Storage::new_dense_f64(6));
    
    let _tensor: TensorDynLen<DynId, AnyScalar> = TensorDynLen::new(indices, dims, storage);
}

#[test]
fn test_tensor_static_len_creation() {
    let indices = [
        Index::new_dyn(2),
        Index::new_dyn(3),
    ];
    let dims = [2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    
    let tensor = TensorStaticLen::<2, DynId, AnyScalar>::new(indices, dims, storage);
    assert_eq!(tensor.indices.len(), 2);
    assert_eq!(tensor.dims.len(), 2);
    assert_eq!(tensor.dims[0], 2);
    assert_eq!(tensor.dims[1], 3);
}

#[test]
fn test_tensor_cow() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let storage = Arc::new(Storage::new_dense_f64(2));
    
    let mut tensor1 = TensorDynLen::<DynId, AnyScalar>::new(indices.clone(), dims.clone(), Arc::clone(&storage));
    let tensor2 = TensorDynLen::<DynId, AnyScalar>::new(indices, dims, storage);
    
    // Initially, both tensors share the same storage
    assert!(Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // Mutate tensor1's storage (should clone due to COW)
    {
        let mut_storage = tensor1.storage_mut();
        match mut_storage {
            Storage::DenseF64(v) => {
                v.push(42.0);
            }
        }
    }
    
    // After mutation, they should have different storage
    assert!(!Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // tensor2's storage should be unchanged
    match tensor2.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
    }
    
    // tensor1's storage should have the new data
    match tensor1.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0], 42.0);
        }
    }
}

