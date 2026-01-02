use tensor4all_core::index::{DefaultIndex as Index, DynId, generate_id};
use tensor4all_core::storage::{AnyScalar, DenseStorageFactory, Storage, make_mut_storage};
use tensor4all_core::tensor::{TensorDynLen, TensorStaticLen};
use std::sync::Arc;
use std::thread;
use std::collections::HashSet;
use num_complex::Complex64;

#[test]
fn test_id_generation() {
    let id1 = generate_id();
    let id2 = generate_id();
    let id3 = generate_id();
    
    // IDs should be unique (random generation, not sequential)
    assert_ne!(id1, id2);
    assert_ne!(id2, id3);
    assert_ne!(id1, id3);
    
    // IDs should be non-zero (very high probability with u128)
    assert_ne!(id1, 0);
    assert_ne!(id2, 0);
    assert_ne!(id3, 0);
}

#[test]
fn test_thread_local_rng_different_seeds() {
    // Test that different threads produce different ID sequences
    // This verifies that each thread gets a different seed
    const NUM_THREADS: usize = 4;
    const IDS_PER_THREAD: usize = 100;
    
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            thread::spawn(|| {
                let mut thread_ids = Vec::new();
                for _ in 0..IDS_PER_THREAD {
                    thread_ids.push(generate_id());
                }
                thread_ids
            })
        })
        .collect();
    
    let mut all_ids = HashSet::new();
    let mut thread_sets = Vec::new();
    
    for handle in handles {
        let thread_ids = handle.join().unwrap();
        let thread_set: HashSet<_> = thread_ids.iter().cloned().collect();
        thread_sets.push(thread_set.clone());
        all_ids.extend(thread_ids);
    }
    
    // All IDs should be unique across all threads
    assert_eq!(all_ids.len(), NUM_THREADS * IDS_PER_THREAD, 
               "All IDs should be unique across threads");
    
    // Check that different threads produce different sequences
    // (with extremely high probability if seeds are different)
    for i in 0..thread_sets.len() {
        for j in (i+1)..thread_sets.len() {
            let intersection: Vec<_> = thread_sets[i].intersection(&thread_sets[j]).collect();
            assert_eq!(intersection.len(), 0,
                       "Thread {} and {} should produce different ID sequences (different seeds)",
                       i, j);
        }
    }
}

#[test]
fn test_thread_count_changes() {
    // Test behavior when thread count changes dynamically
    // This simulates scenarios like thread pool resizing or new threads being spawned
    
    // Phase 1: Generate IDs in initial threads
    const INITIAL_THREADS: usize = 2;
    const IDS_PER_PHASE: usize = 50;
    
    let mut all_ids = HashSet::new();
    let mut phase1_ids = HashSet::new();
    
    // Phase 1: Initial threads
    let handles1: Vec<_> = (0..INITIAL_THREADS)
        .map(|_| {
            thread::spawn(|| {
                let mut thread_ids = Vec::new();
                for _ in 0..IDS_PER_PHASE {
                    thread_ids.push(generate_id());
                }
                thread_ids
            })
        })
        .collect();
    
    for handle in handles1 {
        let thread_ids = handle.join().unwrap();
        for id in &thread_ids {
            all_ids.insert(*id);
            phase1_ids.insert(*id);
        }
    }
    
    assert_eq!(phase1_ids.len(), INITIAL_THREADS * IDS_PER_PHASE);
    
    // Phase 2: Spawn new threads (simulating thread count increase)
    const NEW_THREADS: usize = 3;
    let handles2: Vec<_> = (0..NEW_THREADS)
        .map(|_| {
            thread::spawn(|| {
                let mut thread_ids = Vec::new();
                for _ in 0..IDS_PER_PHASE {
                    thread_ids.push(generate_id());
                }
                thread_ids
            })
        })
        .collect();
    
    let mut phase2_ids = HashSet::new();
    for handle in handles2 {
        let thread_ids = handle.join().unwrap();
        for id in &thread_ids {
            all_ids.insert(*id);
            phase2_ids.insert(*id);
        }
    }
    
    assert_eq!(phase2_ids.len(), NEW_THREADS * IDS_PER_PHASE);
    
    // All IDs should still be unique across both phases
    assert_eq!(all_ids.len(), (INITIAL_THREADS + NEW_THREADS) * IDS_PER_PHASE,
               "All IDs should be unique even when thread count changes");
    
    // Phase 1 and Phase 2 IDs should not overlap
    let intersection: Vec<_> = phase1_ids.intersection(&phase2_ids).collect();
    assert_eq!(intersection.len(), 0,
               "IDs from different thread phases should not overlap");
    
    // Phase 3: Reuse a thread (simulating thread pool reuse)
    // When a thread is reused, it continues using its existing RNG
    let handle3 = thread::spawn(|| {
        let mut thread_ids = Vec::new();
        for _ in 0..IDS_PER_PHASE {
            thread_ids.push(generate_id());
        }
        thread_ids
    });
    
    let phase3_ids: HashSet<_> = handle3.join().unwrap().into_iter().collect();
    
    // Phase 3 IDs should be unique (new thread = new seed)
    assert_eq!(phase3_ids.len(), IDS_PER_PHASE);
    
    // Phase 3 should not overlap with previous phases
    let intersection_13: Vec<_> = phase1_ids.intersection(&phase3_ids).collect();
    let intersection_23: Vec<_> = phase2_ids.intersection(&phase3_ids).collect();
    assert_eq!(intersection_13.len(), 0, "Phase 1 and Phase 3 should not overlap");
    assert_eq!(intersection_23.len(), 0, "Phase 2 and Phase 3 should not overlap");
}

#[test]
fn test_index_dyn() {
    let idx = Index::new_dyn(8);
    assert_eq!(idx.size(), 8);
    assert!(idx.id.0 > 0);
}

#[test]
fn test_index_with_custom_id() {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct StaticId;
    
    let idx = Index::new_with_size(StaticId, 16);
    assert_eq!(idx.size(), 16);
    assert_eq!(idx.id, StaticId);
}

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
    }
}

#[test]
fn test_storage_factory_f64() {
    let storage = <f64 as DenseStorageFactory>::new_dense(7);
    match storage {
        Storage::DenseF64(v) => assert_eq!(v.capacity(), 7),
        Storage::DenseC64(_) => panic!("expected DenseF64"),
    }
}

#[test]
fn test_storage_factory_c64() {
    let storage = <Complex64 as DenseStorageFactory>::new_dense(9);
    match storage {
        Storage::DenseC64(v) => assert_eq!(v.capacity(), 9),
        Storage::DenseF64(_) => panic!("expected DenseC64"),
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
            Storage::DenseC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should be different
    assert!(!Arc::ptr_eq(&storage1, &storage2));
    
    // storage2 should be unchanged
    match storage2.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
    }
    
    // storage1 should have the new data
    match storage1.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[1], 2.0);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
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
            Storage::DenseC64(_) => panic!("expected DenseF64"),
        }
    }
    
    // After mutation, they should have different storage
    assert!(!Arc::ptr_eq(&tensor1.storage, &tensor2.storage));
    
    // tensor2's storage should be unchanged
    match tensor2.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 0);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
    }
    
    // tensor1's storage should have the new data
    match tensor1.storage.as_ref() {
        Storage::DenseF64(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0], 42.0);
        }
        Storage::DenseC64(_) => panic!("expected DenseF64"),
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
            Storage::DenseF64(v) => v.extend([1.0, 2.0, 3.0]),
            Storage::DenseC64(_) => panic!("expected DenseF64"),
        }
    }

    let t: TensorDynLen<DynId, AnyScalar> = TensorDynLen::new(indices, dims, storage);
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
            Storage::DenseF64(_) => panic!("expected DenseC64"),
        }
    }

    // Static element type: returns Complex64
    let t_c64: TensorDynLen<DynId, Complex64> =
        TensorDynLen::new(indices.clone(), dims.clone(), Arc::clone(&storage));
    let sum_c64: Complex64 = t_c64.sum();
    assert_eq!(sum_c64, Complex64::new(4.0, 1.0));

    // Dynamic element type: returns AnyScalar
    let t_any: TensorDynLen<DynId, AnyScalar> = TensorDynLen::new(indices, dims, storage);
    let sum_any: AnyScalar = t_any.sum();
    assert_eq!(sum_any, AnyScalar::C64(Complex64::new(4.0, 1.0)));
}

