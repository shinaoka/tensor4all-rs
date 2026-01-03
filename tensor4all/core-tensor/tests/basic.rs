use tensor4all_core_common::index::{DefaultIndex as Index, DynId};
use tensor4all_core_tensor::{AnyScalar, DenseStorageFactory, Storage, TensorDynLen, make_mut_storage};
use std::sync::Arc;
use num_complex::Complex64;

#[test]
fn test_storage_dense_f64() {
    let storage = Storage::new_dense_f64(10);
    assert_eq!(storage.len(), 0);
    assert_eq!(storage.sum_f64(), 0.0);
    
    match storage {
        Storage::DenseF64(v) => {
            assert_eq!(v.capacity(), 10);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
        Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_storage_dense_c64() {
    let storage = Storage::new_dense_c64(10);
    assert_eq!(storage.len(), 0);
    assert_eq!(storage.sum_c64(), Complex64::new(0.0, 0.0));

    match storage {
        Storage::DenseC64(v) => {
            assert_eq!(v.capacity(), 10);
        }
        Storage::DenseF64(_) => panic!("expected DenseC64"),
        Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
    }
}

#[test]
fn test_storage_factory_f64() {
    let storage = <f64 as DenseStorageFactory>::new_dense(7);
    match storage {
            Storage::DenseF64(v) => assert_eq!(v.capacity(), 7),
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_storage_factory_c64() {
    let storage = <Complex64 as DenseStorageFactory>::new_dense(9);
    match storage {
            Storage::DenseC64(v) => assert_eq!(v.capacity(), 9),
            Storage::DenseF64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
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
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should be different
    assert!(!Arc::ptr_eq(&storage1, &storage2));
    
    // storage2 should be unchanged
    match storage2.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
    
    // storage1 should have the new data
    match storage1.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v.get(0), 1.0);
            assert_eq!(v.get(1), 2.0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
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
    
    let tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
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
    
    let _tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
}


#[test]
fn test_tensor_cow() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let storage = Arc::new(Storage::new_dense_f64(2));
    
    let mut tensor1 = TensorDynLen::<DynId>::new(indices.clone(), dims.clone(), Arc::clone(&storage));
    let tensor2 = TensorDynLen::<DynId>::new(indices, dims, storage);
    
    // Initially, both tensors share the same storage
    assert!(Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // Mutate tensor1's storage (should clone due to COW)
    {
        let mut_storage = tensor1.storage_mut();
        match mut_storage {
            Storage::DenseF64(v) => {
                v.push(42.0);
            }
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should have different storage
    assert!(!Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // tensor2's storage should be unchanged
    match tensor2.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
    
    // tensor1's storage should have the new data
    match tensor1.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v.get(0), 42.0);
        }
        Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_tensor_sum_f64_no_match() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let mut storage = Arc::new(Storage::new_dense_f64(0));
    {
        let s = make_mut_storage(&mut storage);
        match s {
            Storage::DenseF64(v) => v.extend([1.0, 2.0, 3.0].iter().copied()),
            Storage::DenseC64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseF64"),
        }
    }

    let t: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    let sum_f64 = t.sum_f64();
    assert_eq!(sum_f64, 6.0);

    let sum_any: AnyScalar = t.sum();
    assert_eq!(sum_any, AnyScalar::F64(6.0));
}

#[test]
fn test_tensor_sum_c64() {
    let indices = vec![Index::new_dyn(2)];
    let dims = vec![2];
    let mut storage = Arc::new(Storage::new_dense_c64(0));
    {
        let s = make_mut_storage(&mut storage);
        match s {
            Storage::DenseC64(v) => v.extend([Complex64::new(1.0, 2.0), Complex64::new(3.0, -1.0)]),
            Storage::DenseF64(_) | Storage::DiagF64(_) | Storage::DiagC64(_) => panic!("expected DenseC64"),
        }
    }

    // Now always returns AnyScalar
    let t: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
    let sum_any: AnyScalar = t.sum();
    assert_eq!(sum_any, AnyScalar::C64(Complex64::new(4.0, 1.0)));
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_new() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let dims = vec![2, 3, 2];
    let storage = Arc::new(Storage::new_dense_f64(12));
    
    let _tensor: TensorDynLen<DynId> = TensorDynLen::new(indices, dims, storage);
}

#[test]
#[should_panic(expected = "Tensor indices must all be unique")]
fn test_tensor_duplicate_indices_from_indices() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i.clone(), j.clone(), i.clone()]; // duplicate i
    let storage = Arc::new(Storage::new_dense_f64(12));
    
    let _tensor: TensorDynLen<DynId> = TensorDynLen::from_indices(indices, storage);
}

