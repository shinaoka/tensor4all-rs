use super::*;
use num_complex::Complex64;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
#[cfg(feature = "adaptive-hataori-rayon")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "adaptive-hataori-rayon")]
use std::sync::Arc;
use tensor4all_core::contract;
use tensor4all_tensorbackend::StorageKind;

fn dense_f64(result: &AdaptiveInterpolationResult<f64>) -> Vec<f64> {
    let tt = result.partitioned_tt().to_tensor_train().unwrap();
    let tensors: Vec<_> = (0..tt.len()).map(|site| tt.tensor(site).unwrap()).collect();
    contract(&tensors).unwrap().to_vec::<f64>().unwrap()
}

fn binary_sites(nsites: usize) -> Vec<DynIndex> {
    (0..nsites).map(|_| DynIndex::new_dyn(2)).collect()
}

#[test]
fn accepts_only_full_tci_convergence() {
    assert!(patch_is_accepted(
        TCI2Termination::Converged,
        1.0e-10,
        1.0e-8
    ));
    assert!(!patch_is_accepted(
        TCI2Termination::MaxIterations,
        1.0e-10,
        1.0e-8,
    ));
    assert!(!patch_is_accepted(
        TCI2Termination::MaxBondDimension,
        1.0e-10,
        1.0e-8,
    ));
    assert!(!patch_is_accepted(
        TCI2Termination::Converged,
        1.0e-6,
        1.0e-8,
    ));
}

#[test]
fn interpolates_low_rank_function_without_splitting() {
    let sites = binary_sites(3);
    let function =
        |index: &MultiIndex| (index[0] + 1) as f64 * (index[1] + 2) as f64 * (index[2] + 3) as f64;
    let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        function,
        None,
        sites,
        vec![vec![1, 1, 1]],
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap();

    assert_eq!(result.partitioned_tt().len(), 1);
    assert_eq!(
        dense_f64(&result),
        vec![6.0, 12.0, 9.0, 18.0, 8.0, 16.0, 12.0, 24.0]
    );
}

#[test]
fn evaluates_single_active_site_exactly() {
    let site = DynIndex::new_dyn(4);
    let function = |index: &MultiIndex| {
        if index[0] == 3 {
            10.0
        } else {
            (index[0] * index[0] + 1) as f64
        }
    };
    let options = AdaptiveInterpolateOptions {
        n_initial_pivots: 1,
        ..AdaptiveInterpolateOptions::default()
    };
    let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        function,
        None,
        vec![site],
        Vec::new(),
        options,
    )
    .unwrap();

    assert_eq!(result.partitioned_tt().len(), 1);
    assert_eq!(dense_f64(&result), vec![1.0, 2.0, 5.0, 10.0]);
}

#[test]
fn rank_cap_forces_disjoint_exact_child_patches() {
    let sites = binary_sites(3);
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
            ..TCI2Options::default()
        },
        patch_order: sites.clone(),
        recycle_pivots: true,
        ..AdaptiveInterpolateOptions::default()
    };

    let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        function,
        None,
        sites.clone(),
        vec![vec![0, 0, 0], vec![1, 1, 1]],
        options,
    )
    .unwrap();

    assert!(result.partitioned_tt().len() >= 2);
    assert!(Projector::are_disjoint(
        &result
            .partitioned_tt()
            .projectors()
            .cloned()
            .collect::<Vec<_>>()
    ));
    assert_eq!(
        dense_f64(&result),
        vec![2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0]
    );
    assert_eq!(result.patch_caches().len(), result.partitioned_tt().len());
    for cache in result.patch_caches() {
        assert!(result.partitioned_tt().contains(cache.projector()));
        for local_index in cache.cache.entries.keys() {
            let full = expand_pivot(
                local_index,
                cache.active_positions(),
                cache.projector(),
                &sites,
            );
            assert!(is_compatible_pivot(&full, &sites, cache.projector()));
        }
    }
    let mut returned = result.patch_caches().to_vec();
    assert!(returned.iter().map(AcceptedPatchCache::len).sum::<usize>() > 0);
    let projector = returned[0].projector().clone();
    returned[0].clear();
    assert!(returned[0].is_empty());
    assert_eq!(returned[0].projector(), &projector);
}

#[test]
fn uses_batched_callback_on_tci_patches() {
    let sites = binary_sites(2);
    let batch_calls = Rc::new(Cell::new(0));
    let batch_calls_for_callback = Rc::clone(&batch_calls);
    let function = |index: &MultiIndex| (index[0] + index[1] + 1) as f64;
    let batched = move |indices: &[MultiIndex]| {
        batch_calls_for_callback.set(batch_calls_for_callback.get() + 1);
        indices
            .iter()
            .map(|index| (index[0] + index[1] + 1) as f64)
            .collect()
    };
    let result = adaptiveinterpolate(
        function,
        Some(batched),
        sites,
        vec![vec![1, 1]],
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap();

    assert!(batch_calls.get() > 0);
    assert_eq!(dense_f64(&result), vec![1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn supports_complex_values() {
    let sites = binary_sites(2);
    let function = |index: &MultiIndex| {
        Complex64::new((index[0] + 1) as f64, index[1] as f64)
            * Complex64::new((index[1] + 2) as f64, 0.0)
    };
    let result = adaptiveinterpolate::<Complex64, _, fn(&[MultiIndex]) -> Vec<Complex64>>(
        function,
        None,
        sites,
        vec![vec![1, 1]],
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap();
    let tt = result.partitioned_tt().to_tensor_train().unwrap();
    let tensors: Vec<_> = (0..tt.len()).map(|site| tt.tensor(site).unwrap()).collect();
    let dense = contract(&tensors).unwrap().to_vec::<Complex64>().unwrap();

    assert_eq!(
        dense,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(3.0, 3.0),
            Complex64::new(6.0, 3.0),
        ]
    );
}

#[test]
fn sampled_zero_patch_is_represented_as_zero() {
    let sites = binary_sites(2);
    let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        |_| 0.0,
        None,
        sites,
        Vec::new(),
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap();

    assert_eq!(result.partitioned_tt().len(), 1);
    assert_eq!(dense_f64(&result), vec![0.0; 4]);
}

#[test]
fn extracts_full_diagonal_pivots_for_recycling() {
    let function = |index: &MultiIndex| (index[0] + 2 * index[1] + 3 * index[2] + 1) as f64;
    let tensor4all_tensorci::TCI2OptimizationResult { tci, .. } =
        crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
            function,
            None,
            vec![2, 2, 2],
            vec![vec![0, 0, 0], vec![1, 1, 1]],
            TCI2Options {
                seed: Some(3),
                ..TCI2Options::default()
            },
        )
        .unwrap();
    let sites = binary_sites(3);

    let pivots = global_diagonal_pivots(&tci, &[0, 1, 2], &Projector::new(), &sites);

    assert!(!pivots.is_empty());
    assert!(pivots
        .iter()
        .all(|pivot| pivot.len() == 3 && pivot.iter().all(|value| *value < 2)));
}

#[test]
fn incompatible_recycled_pivots_are_replenished_for_nonzero_child() {
    let sites = binary_sites(3);
    let projector = Projector::from_pairs([(sites[0].clone(), 1)]).unwrap();
    let active = active_positions(&sites, &projector);
    let recycled = vec![vec![0, 0, 0], vec![0, 1, 1]];
    let mut rng = StdRng::seed_from_u64(7);

    let candidates =
        patch_candidates(&sites, &active, &projector, &[], &recycled, 3, &mut rng).unwrap();
    let values: Vec<_> = candidates
        .iter()
        .map(|local| {
            let full = expand_pivot(local, &active, &projector, &sites);
            if full[0] == 1 {
                1.0
            } else {
                0.0
            }
        })
        .collect();

    assert_eq!(candidates.len(), 3);
    assert!(values.iter().any(|value| *value != 0.0));
}

#[test]
fn projected_middle_sites_use_compact_structured_storage() {
    let sites = binary_sites(3);
    let active = vec![0, 2];
    let projector = Projector::from_pairs([(sites[1].clone(), 1)]).unwrap();
    let active_tt = tensor4all_simplett::SimpleTensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();

    let (active_tree, _) = tensor_train_to_treetn(&active_tt).unwrap();
    let tt = embed_active_tt::<f64>(active_tree, &sites, &active, &projector).unwrap();
    let middle = tt.tensor(1).unwrap();

    assert_eq!(
        middle.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(middle.storage().unwrap().payload_len(), 4);
    assert_eq!(middle.storage().unwrap().axis_classes(), &[0, 1, 0]);

    let tensors: Vec<_> = (0..tt.len())
        .map(|position| tt.tensor(position).unwrap())
        .collect();
    let dense = contract(&tensors).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(dense, vec![0.0, 0.0, 23.0, 34.0, 0.0, 0.0, 31.0, 46.0]);
}

#[test]
fn cache_split_moves_entries_to_one_child_for_every_split_position() {
    let dims = vec![2, 3, 2];
    let mut parent = PatchCache::new(dims.clone());
    for i in 0..dims[0] {
        for j in 0..dims[1] {
            for k in 0..dims[2] {
                parent.entries.insert(vec![i, j, k], 100 * i + 10 * j + k);
            }
        }
    }

    for (split_pos, &child_count) in dims.iter().enumerate() {
        let children = parent.clone().split(split_pos, child_count).unwrap();
        assert_eq!(children.len(), child_count);
        assert_eq!(
            children
                .iter()
                .map(|cache| cache.entries.len())
                .sum::<usize>(),
            parent.entries.len()
        );
        for (child, cache) in children.iter().enumerate() {
            for (local, &value) in &cache.entries {
                let mut full = local.clone();
                full.insert(split_pos, child);
                assert_eq!(parent.entries.get(&full), Some(&value));
            }
        }
    }
}

#[test]
fn cache_split_keeps_sibling_maps_independent() {
    let mut parent = PatchCache::new(vec![2, 2]);
    parent.entries.insert(vec![0, 0], 1);
    parent.entries.insert(vec![1, 1], 2);
    let mut children = parent.split(0, 2).unwrap();

    children[0].entries.clear();

    assert!(children[0].entries.is_empty());
    assert_eq!(children[1].entries.get(&vec![1]), Some(&2));
}

#[test]
fn child_cache_hits_inherited_samples_but_not_sibling_samples() {
    let sites = binary_sites(2);
    let mut parent = PatchCache::new(vec![2, 2]);
    parent.entries.insert(vec![0, 1], 7.0_f64);
    parent.entries.insert(vec![1, 0], 8.0_f64);
    let mut children = parent.split(0, 2).unwrap();
    let calls = Cell::new(0);
    let function = |full: &MultiIndex| {
        calls.set(calls.get() + 1);
        (10 * full[0] + full[1]) as f64
    };
    let projector = Projector::from_pairs([(sites[0].clone(), 0)]).unwrap();
    let evaluator = PatchEvaluator::<_, _, fn(&[MultiIndex]) -> Vec<f64>>::new(
        &function,
        None,
        &[1],
        &projector,
        &sites,
        children.remove(0),
    )
    .unwrap();

    assert_eq!(evaluator.eval(&vec![1]), 7.0);
    assert_eq!(
        calls.get(),
        0,
        "the matching parent sample must be inherited"
    );
    assert_eq!(evaluator.eval(&vec![0]), 0.0);
    assert_eq!(calls.get(), 1, "a sibling-only sample must be re-evaluated");
}

#[test]
fn batch_cache_deduplicates_misses_and_restores_order() {
    let sites = binary_sites(2);
    let calls = Rc::new(RefCell::new(Vec::<Vec<MultiIndex>>::new()));
    let recorded = Rc::clone(&calls);
    let scalar = |index: &MultiIndex| (10 * index[0] + index[1]) as f64;
    let batch = move |indices: &[MultiIndex]| {
        recorded.borrow_mut().push(indices.to_vec());
        indices
            .iter()
            .map(|index| (10 * index[0] + index[1]) as f64)
            .collect()
    };
    let projector = Projector::new();
    let evaluator = PatchEvaluator::new(
        &scalar,
        Some(&batch),
        &[0, 1],
        &projector,
        &sites,
        PatchCache::new(vec![2, 2]),
    )
    .unwrap();
    let input = vec![vec![0, 1], vec![0, 1], vec![1, 0]];

    assert_eq!(evaluator.eval_many(&input), vec![1.0, 1.0, 10.0]);
    assert_eq!(calls.borrow().as_slice(), &[vec![vec![0, 1], vec![1, 0]]]);
    assert_eq!(evaluator.eval_many(&input), vec![1.0, 1.0, 10.0]);
    assert_eq!(calls.borrow().len(), 1);
    assert_eq!(evaluator.into_cache().entries.len(), 2);
}

#[test]
fn batch_cache_length_mismatch_is_reported_without_inserting() {
    let sites = binary_sites(2);
    let scalar = |_: &MultiIndex| 1.0_f64;
    let batch = |_: &[MultiIndex]| vec![1.0_f64];
    let projector = Projector::new();
    let evaluator = PatchEvaluator::new(
        &scalar,
        Some(&batch),
        &[0, 1],
        &projector,
        &sites,
        PatchCache::new(vec![2, 2]),
    )
    .unwrap();

    assert_eq!(
        evaluator.eval_many(&[vec![0, 0], vec![1, 1]]),
        vec![0.0, 0.0]
    );
    assert!(matches!(
        evaluator.take_error(),
        Some(PartitionedTTError::InvalidAdaptiveInterpolationInput(message))
            if message.contains("cache misses")
    ));
    assert!(evaluator.into_cache().entries.is_empty());
}

#[cfg(feature = "adaptive-hataori-mpi")]
#[test]
fn malformed_wire_cores_are_rejected_before_reconstruction() {
    let valid = || WireCore {
        dims: [1, 2, 1],
        data: vec![1.0_f64, 2.0],
    };
    let cases = vec![
        vec![valid()],
        vec![
            WireCore {
                dims: [usize::MAX, 2, 1],
                data: Vec::new(),
            },
            valid(),
        ],
        vec![
            WireCore {
                dims: [1, 2, 1],
                data: vec![1.0],
            },
            valid(),
        ],
        vec![
            WireCore {
                dims: [1, 3, 1],
                data: vec![1.0, 2.0, 3.0],
            },
            valid(),
        ],
    ];
    for cores in cases {
        let outcome = WirePatchOutcome::Accepted(WireAcceptedPatch {
            path: Vec::new(),
            active_positions: vec![0, 1],
            cache: PatchCache::new(vec![2, 2]),
            data: WireAcceptedData::Active(cores),
        });
        let sites = binary_sites(2);
        let error = patch_outcome_from_wire(outcome, &sites, &sites).unwrap_err();
        assert!(matches!(
            error,
            PartitionedTTError::DistributedAdaptiveInterpolation(_)
        ));
    }
}

#[test]
fn patch_candidate_count_overflow_is_rejected() {
    let sites = vec![DynIndex::new_dyn(usize::MAX), DynIndex::new_dyn(2)];
    let mut rng = StdRng::seed_from_u64(1);

    let error =
        patch_candidates(&sites, &[0, 1], &Projector::new(), &[], &[], 1, &mut rng).unwrap_err();

    assert!(matches!(
        error,
        PartitionedTTError::InvalidAdaptiveInterpolationInput(message)
            if message.contains("point count")
    ));
}

#[test]
fn projected_site_tensor_rejects_unequal_carried_bonds() {
    let left = DynIndex::new_dyn(2);
    let site = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(3);

    let error = projected_site_tensor::<f64>(Some(&left), &site, Some(&right), 0, 1.0).unwrap_err();

    assert!(error.to_string().contains("bond dimensions differ"));
}

#[test]
fn rejects_invalid_scalar_options_and_site_lists() {
    let make_sites = || binary_sites(2);
    let valid_pivots = vec![vec![0, 0]];

    let mut cases = Vec::new();
    cases.push((
        make_sites(),
        AdaptiveInterpolateOptions {
            n_initial_pivots: 0,
            ..AdaptiveInterpolateOptions::default()
        },
    ));
    for tci_options in [
        TCI2Options {
            tolerance: -1.0,
            ..TCI2Options::default()
        },
        TCI2Options {
            max_iter: 0,
            ..TCI2Options::default()
        },
        TCI2Options {
            max_bond_dim: Some(0),
            ..TCI2Options::default()
        },
        TCI2Options {
            ncheck_history: 0,
            ..TCI2Options::default()
        },
        TCI2Options {
            tol_margin_global_search: f64::NAN,
            ..TCI2Options::default()
        },
    ] {
        cases.push((
            make_sites(),
            AdaptiveInterpolateOptions {
                tci_options,
                ..AdaptiveInterpolateOptions::default()
            },
        ));
    }

    for (sites, options) in cases {
        let error = validate_inputs(&sites, &valid_pivots, &options).unwrap_err();
        assert!(matches!(
            error,
            PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
        ));
    }

    let zero_dim_error = validate_inputs(
        &[DynIndex::new_dyn(0)],
        &[],
        &AdaptiveInterpolateOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        zero_dim_error,
        PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
    ));

    let duplicate = DynIndex::new_dyn(2);
    let duplicate_error = validate_inputs(
        &[duplicate.clone(), duplicate],
        &valid_pivots,
        &AdaptiveInterpolateOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        duplicate_error,
        PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
    ));
    let empty_error =
        validate_inputs(&[], &[], &AdaptiveInterpolateOptions::default()).unwrap_err();
    assert!(matches!(
        empty_error,
        PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
    ));
}

#[test]
fn rejects_invalid_patch_order_and_pivots() {
    let sites = binary_sites(2);
    let options = AdaptiveInterpolateOptions {
        patch_order: vec![sites[0].clone(), DynIndex::new_dyn(2)],
        ..AdaptiveInterpolateOptions::default()
    };
    let order_error = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        |_| 1.0,
        None,
        sites.clone(),
        vec![vec![0, 0]],
        options,
    )
    .unwrap_err();
    assert!(matches!(
        order_error,
        PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
    ));

    for pivots in [vec![vec![0]], vec![vec![0, 2]]] {
        let pivot_error = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
            |_| 1.0,
            None,
            sites.clone(),
            pivots,
            AdaptiveInterpolateOptions::default(),
        )
        .unwrap_err();
        assert!(matches!(
            pivot_error,
            PartitionedTTError::InvalidAdaptiveInterpolationInput(_)
        ));
    }
}

#[cfg(feature = "adaptive-hataori-rayon")]
#[test]
fn hataori_outer_matches_sequential_and_allows_nested_rayon() {
    let sites = binary_sites(1);
    let function = |index: &MultiIndex| (index[0] + 1) as f64;
    let options = AdaptiveInterpolateOptions {
        tci_options: TCI2Options {
            seed: Some(11),
            ..TCI2Options::default()
        },
        ..AdaptiveInterpolateOptions::default()
    };
    let sequential = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        function,
        None,
        sites.clone(),
        Vec::new(),
        options.clone(),
    )
    .unwrap();

    // One explicit worker keeps this nested-pool contract test independent of
    // workspace-wide nextest process parallelism and host thread limits.
    let workers = 1;
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap(),
    );
    let domain =
        hataori::Domain::external(Arc::clone(&pool), (0..workers).collect(), workers).unwrap();
    let nested = Arc::new(AtomicBool::new(false));
    let nested_for_callback = Arc::clone(&nested);
    let parallel_function = move |index: &MultiIndex| {
        let (left, right) = rayon::join(
            || rayon::current_thread_index().is_some(),
            || rayon::current_thread_index().is_some(),
        );
        nested_for_callback.fetch_or(left && right, Ordering::Relaxed);
        function(index)
    };
    let parallel = adaptiveinterpolate_in::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &domain,
        parallel_function,
        None,
        sites,
        Vec::new(),
        options,
    )
    .unwrap();

    assert!(nested.load(Ordering::Relaxed));
    assert_eq!(parallel.patch_caches()[0].get(&[0]), Some(&1.0));
    assert_eq!(parallel.patch_caches()[0].get(&[1]), Some(&2.0));
    assert_eq!(
        parallel.patch_caches()[0].len(),
        sequential.patch_caches()[0].len()
    );
    let sequential_projectors: HashSet<_> = sequential
        .patch_caches()
        .iter()
        .map(|cache| cache.projector().clone())
        .collect();
    let parallel_projectors: HashSet<_> = parallel
        .patch_caches()
        .iter()
        .map(|cache| cache.projector().clone())
        .collect();
    assert_eq!(parallel_projectors, sequential_projectors);
}

#[cfg(feature = "adaptive-hataori-rayon")]
#[test]
fn hataori_entry_rejects_a_sequential_domain() {
    let error = adaptiveinterpolate_in::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        &hataori::Domain::sequential(),
        |_| 1.0,
        None,
        binary_sites(2),
        Vec::new(),
        AdaptiveInterpolateOptions::default(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        PartitionedTTError::HataoriLocal {
            source: hataori::MapInError::MissingPool,
            ..
        }
    ));
}

#[test]
fn numerically_zero_child_patch_is_accepted_not_crash_issue598() {
    const WEIGHTS: [f64; 3] = [1.3, 0.9, 0.9];
    const ALPHAS: [f64; 3] = [2.8, 5.4, 0.7];
    const CENTERS: [(f64, f64); 3] = [(0.4, 0.1), (3.8, -0.8), (-5.5, -2.1)];
    const BOX_L: f64 = 12.0;
    const R: usize = 10;

    let eval_mixture = |index: &MultiIndex| -> f64 {
        let mut ix = 0u64;
        let mut iy = 0u64;
        for (n, &fused) in index.iter().enumerate() {
            let shift = R - 1 - n;
            ix |= ((fused & 1) as u64) << shift;
            iy |= (((fused >> 1) & 1) as u64) << shift;
        }
        let step = 2.0 * BOX_L / (1u64 << R) as f64;
        let (x, y) = (-BOX_L + ix as f64 * step, -BOX_L + iy as f64 * step);
        (0..3)
            .map(|i| {
                let (cx, cy) = CENTERS[i];
                WEIGHTS[i] * (-ALPHAS[i] * ((x - cx).powi(2) + (y - cy).powi(2))).exp()
            })
            .sum()
    };

    let sites: Vec<DynIndex> = (0..R).map(|_| DynIndex::new_dyn(4)).collect();
    let options = AdaptiveInterpolateOptions {
        tci_options: TCI2Options {
            tolerance: 1e-8,
            max_bond_dim: Some(64),
            max_iter: 20,
            normalize_error: false,
            seed: Some(1),
            ..TCI2Options::default()
        },
        ..AdaptiveInterpolateOptions::default()
    };
    let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        eval_mixture,
        None,
        sites,
        Vec::new(),
        options,
    )
    .expect("adaptiveinterpolate must accept every patch, including near-zero ones");

    let tt = result
        .partitioned_tt()
        .to_tensor_train()
        .expect("valid combined tensor train");
    let _ = tt.bond_dims();
}
