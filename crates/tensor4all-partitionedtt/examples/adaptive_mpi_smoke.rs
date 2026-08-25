use mpi::environment::Threading;
use mpi::traits::*;
use num_complex::Complex64;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tensor4all_core::contract;
use tensor4all_partitionedtt::{
    adaptiveinterpolate_mpi, AdaptiveInterpolateOptions, DynIndex, MultiIndex, PartitionedTTError,
    TCI2Options,
};

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(Threading::Funneled)
        .expect("MPI must not already be initialized or finalized");
    assert!(provided >= Threading::Funneled);
    let world = universe.world();
    let rank = world.rank();
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(2);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap(),
    );
    let domain =
        hataori::Domain::external(Arc::clone(&pool), (0..workers).collect(), workers).unwrap();

    let sites: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(4)).collect();
    let function = |index: &MultiIndex| {
        if index.iter().all(|value| *value == index[0]) {
            2.0
        } else {
            0.5
        }
    };
    let options = AdaptiveInterpolateOptions {
        tci_options: TCI2Options {
            tolerance: 1.0e-14,
            max_bond_dim: Some(1),
            max_iter: 4,
            ncheck_history: 1,
            nsearch: 0,
            max_nglobal_pivot: 0,
            seed: Some(17),
            ..TCI2Options::default()
        },
        patch_order: sites.clone(),
        recycle_pivots: true,
        ..AdaptiveInterpolateOptions::default()
    };
    let calls = Arc::new(AtomicI64::new(0));
    let calls_for_callback = Arc::clone(&calls);
    let delayed_function = move |index: &MultiIndex| {
        calls_for_callback.fetch_add(1, Ordering::Relaxed);
        if rank == 0 {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        function(index)
    };
    let result = adaptiveinterpolate_mpi::<_, f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &world,
        &domain,
        0,
        delayed_function,
        None,
        sites.clone(),
        vec![vec![0, 0, 0], vec![3, 3, 3]],
        options.clone(),
    )
    .unwrap();
    let mut remote_calls = 0_i64;
    let local_remote_calls = if rank == 0 {
        0
    } else {
        calls.load(Ordering::Relaxed)
    };
    world.all_reduce_into(
        &local_remote_calls,
        &mut remote_calls,
        mpi::collective::SystemOperation::sum(),
    );
    if world.size() > 1 {
        assert!(
            remote_calls > 0,
            "at least one remote rank must evaluate a patch"
        );
    }
    assert_eq!(result.is_some(), rank == 0);
    let first_dense = result.map(|result| {
        assert_eq!(result.patch_caches().len(), result.partitioned_tt().len());
        let tt = result.partitioned_tt().to_tensor_train().unwrap();
        let tensors: Vec<_> = (0..tt.len()).map(|site| tt.tensor(site).unwrap()).collect();
        contract(&tensors).unwrap().to_vec::<f64>().unwrap()
    });
    if let Some(dense) = &first_dense {
        let expected: Vec<_> = (0..4)
            .flat_map(|k| {
                (0..4).flat_map(move |j| {
                    (0..4).map(move |i| if i == j && j == k { 2.0 } else { 0.5 })
                })
            })
            .collect();
        assert_eq!(dense, &expected);
    }

    let repeated = adaptiveinterpolate_mpi::<_, f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &world,
        &domain,
        0,
        function,
        None,
        sites,
        vec![vec![0, 0, 0], vec![3, 3, 3]],
        options,
    )
    .unwrap();
    let repeated_dense = repeated.map(|result| {
        let tt = result.partitioned_tt().to_tensor_train().unwrap();
        let tensors: Vec<_> = (0..tt.len()).map(|site| tt.tensor(site).unwrap()).collect();
        contract(&tensors).unwrap().to_vec::<f64>().unwrap()
    });
    assert_eq!(repeated_dense, first_dense);

    let complex_sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let complex = adaptiveinterpolate_mpi::<_, Complex64, _, fn(&[MultiIndex]) -> Vec<Complex64>>(
        &world,
        &domain,
        0,
        |index| Complex64::new((index[0] + 1) as f64, index[1] as f64),
        None,
        complex_sites,
        vec![vec![1, 1]],
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap();
    assert_eq!(complex.is_some(), rank == 0);

    let invalid = adaptiveinterpolate_mpi::<_, f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &world,
        &domain,
        0,
        |_| 1.0,
        None,
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        vec![vec![0]],
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        invalid,
        PartitionedTTError::DistributedAdaptiveInterpolation(_)
    ));
}
