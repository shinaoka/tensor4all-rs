use super::*;
use num_complex::Complex64;
use std::cell::Cell;
use std::rc::Rc;
use tensor4all_core::contract;
use tensor4all_tensorbackend::StorageKind;

fn dense_f64(result: &PartitionedTT) -> Vec<f64> {
    let tt = result.to_tensor_train().unwrap();
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

    assert_eq!(result.len(), 1);
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

    assert_eq!(result.len(), 1);
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
            max_bond_dim: 1,
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
        sites,
        vec![vec![0, 0, 0], vec![1, 1, 1]],
        options,
    )
    .unwrap();

    assert!(result.len() >= 2);
    assert!(Projector::are_disjoint(
        &result.projectors().cloned().collect::<Vec<_>>()
    ));
    assert_eq!(
        dense_f64(&result),
        vec![2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0]
    );
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
    let tt = result.to_tensor_train().unwrap();
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

    assert_eq!(result.len(), 1);
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
    let projector = Projector::from_pairs([(sites[0].clone(), 1)]);
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
    let projector = Projector::from_pairs([(sites[1].clone(), 1)]);
    let active_tt = tensor4all_simplett::TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();

    let tt = embed_active_tt(active_tt, &sites, &active, &projector).unwrap();
    let middle = tt.tensor(1).unwrap();

    assert_eq!(middle.storage().storage_kind(), StorageKind::Structured);
    assert_eq!(middle.storage().payload_len(), 4);
    assert_eq!(middle.storage().axis_classes(), &[0, 1, 0]);

    let tensors: Vec<_> = (0..tt.len())
        .map(|position| tt.tensor(position).unwrap())
        .collect();
    let dense = contract(&tensors).unwrap().to_vec::<f64>().unwrap();
    assert_eq!(dense, vec![0.0, 0.0, 23.0, 34.0, 0.0, 0.0, 31.0, 46.0]);
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
            max_bond_dim: 0,
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
