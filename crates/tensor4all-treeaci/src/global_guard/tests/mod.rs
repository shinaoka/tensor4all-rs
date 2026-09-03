use num_complex::{Complex32, Complex64};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{
    find_global_pivots, inject_global_pivots, per_evaluator_message_cache_budget,
    sole_varying_site, GuardOutputEvaluator, InputEvaluators,
};
use crate::{
    schedule::{run_directional_pass, PassDirection},
    state::TreeAciState,
    transaction::update_edge_transaction,
    TreeAciOptions,
};

fn delta_tree() -> (TreeTN<IdxTensor, usize>, DynIndex, DynIndex) {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![left_site.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let right =
        IdxTensor::from_dense(vec![bond, right_site.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    (
        TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap(),
        left_site,
        right_site,
    )
}

fn zero_tree(left_site: DynIndex, right_site: DynIndex) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(1);
    let left = IdxTensor::from_dense(vec![left_site, bond.clone()], vec![0.0, 0.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![1.0, 1.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn typed_delta_tree<T: crate::TreeAciScalar>() -> (TreeTN<IdxTensor, usize>, DynIndex, DynIndex) {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![left_site.clone(), bond.clone()],
        vec![
            <T as tensor4all_core::Scalar>::from_f64(1.0),
            <T as tensor4all_core::Scalar>::from_f64(0.0),
            <T as tensor4all_core::Scalar>::from_f64(0.0),
            <T as tensor4all_core::Scalar>::from_f64(1.0),
        ],
    )
    .unwrap();
    let right = IdxTensor::from_dense(
        vec![bond, right_site.clone()],
        vec![
            <T as tensor4all_core::Scalar>::from_f64(1.0),
            <T as tensor4all_core::Scalar>::from_f64(0.0),
            <T as tensor4all_core::Scalar>::from_f64(0.0),
            <T as tensor4all_core::Scalar>::from_f64(1.0),
        ],
    )
    .unwrap();
    (
        TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap(),
        left_site,
        right_site,
    )
}

trait GuardTestScalar: crate::TreeAciScalar {
    const TOLERANCE: f64;
}

impl GuardTestScalar for f32 {
    const TOLERANCE: f64 = 1.0e-5;
}

impl GuardTestScalar for f64 {
    const TOLERANCE: f64 = 1.0e-12;
}

impl GuardTestScalar for Complex32 {
    const TOLERANCE: f64 = 1.0e-5;
}

impl GuardTestScalar for Complex64 {
    const TOLERANCE: f64 = 1.0e-12;
}

fn assert_guard_typed_evaluation<T: GuardTestScalar>() {
    let (input, _, _) = typed_delta_tree::<T>();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let state = TreeAciState::<T, usize>::initialize(&inputs, &options).unwrap();
    let input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut output_evaluator = GuardOutputEvaluator::new(
        &state.output,
        &state.problem,
        options.message_cache_max_bytes,
    )
    .unwrap();
    let points = vec![vec![0usize, 0], vec![1, 0], vec![0, 1], vec![1, 1]];

    let actual = output_evaluator
        .evaluate::<T>(&input_evaluators, &points)
        .unwrap();
    let coordinates = input_evaluators.expand_points(&points).unwrap();
    let indices = state
        .problem
        .physical
        .iter()
        .flat_map(|physical| physical.indices.iter().cloned())
        .collect::<Vec<_>>();
    let expected = state
        .output
        .evaluate(
            &indices,
            ColMajorArrayRef::new(&coordinates, &[indices.len(), points.len()]).unwrap(),
        )
        .unwrap();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        let expected = T::from_evaluated_scalar(expected).unwrap();
        let residual = tensor4all_core::Scalar::abs_val(actual - expected);
        let scale = tensor4all_core::Scalar::abs_val(expected).max(1.0);
        assert!(
            residual <= T::TOLERANCE * scale,
            "guard result residual {residual} exceeds tolerance for {}",
            std::any::type_name::<T>()
        );
    }
}

#[test]
fn message_cache_budget_is_shared_by_all_guard_evaluators() {
    assert_eq!(per_evaluator_message_cache_budget(256, 3).unwrap(), 64);
    assert_eq!(per_evaluator_message_cache_budget(3, 1).unwrap(), 1);
    assert_eq!(per_evaluator_message_cache_budget(0, 2).unwrap(), 0);
}

#[test]
fn guard_working_budget_counts_all_coexisting_evaluation_buffers() {
    let (input, _, _) = delta_tree();
    let inputs = vec![input];
    let options = TreeAciOptions::default();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let point_count = 2;
    let expected = point_count * std::mem::size_of::<Vec<usize>>()
        + 2 * point_count * std::mem::size_of::<usize>()
        + 2 * point_count * std::mem::size_of::<usize>()
        + 3 * point_count * std::mem::size_of::<f64>()
        + point_count
            * std::mem::size_of::<tensor4all_core::AnyScalar>().max(std::mem::size_of::<f64>());

    evaluators.max_working_bytes = expected;
    evaluators
        .enforce_guard_batch_budget::<f64>(point_count)
        .unwrap();
    evaluators.max_working_bytes = expected - 1;
    let error = evaluators
        .enforce_guard_batch_budget::<f64>(point_count)
        .unwrap_err();
    assert!(matches!(
        error,
        crate::TreeAciError::ResourceLimit {
            resource: "working bytes",
            requested,
            limit,
        } if requested == expected && limit == expected - 1
    ));
}

#[test]
fn guard_rejects_out_of_range_local_coordinates_instead_of_wrapping_them() {
    let (input, _, _) = delta_tree();
    let inputs = vec![input];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    assert!(matches!(
        evaluators.evaluate::<f64>(&[vec![2, 0]]),
        Err(crate::TreeAciError::PhysicalCoordinateOutOfBounds {
            node: 0,
            coordinate: 2,
            local_dim: 2,
        })
    ));
}

#[test]
fn global_search_rejects_the_start_batch_before_calling_the_operator() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 4,
        max_working_bytes: 64,
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut operator_called = false;
    let mut operator = |_: crate::TreeElementwiseBatch<'_, f64>, _: &mut [f64]| {
        operator_called = true;
        Ok(())
    };

    let error = find_global_pivots(&state, &mut evaluators, &options, 0, &mut operator)
        .expect_err("the start vectors must be budgeted before allocation/evaluation");

    assert!(matches!(
        error,
        crate::TreeAciError::ResourceLimit {
            resource: "working bytes",
            limit: 64,
            ..
        }
    ));
    assert!(!operator_called);
}

fn identity(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)?;
    }
    Ok(())
}

#[test]
fn floating_zone_finds_a_feature_missing_from_the_output() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 4,
        max_nglobal_pivots: 2,
        nsweeps_global_search: 4,
        global_tolerance_margin: 1.0,
        message_cache_max_bytes: 1,
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left_site, right_site);
    let mut input_evaluators = InputEvaluators::new_with_message_cache_max_bytes(
        state.inputs,
        &state.problem,
        options.message_cache_max_bytes,
    )
    .unwrap();

    let report =
        find_global_pivots(&state, &mut input_evaluators, &options, 9, &mut identity).unwrap();

    assert!(!report.pivots.is_empty());
    assert!(report
        .pivots
        .iter()
        .all(|point| point == &[0, 0] || point == &[1, 1]));
    assert!(report.evaluated_points > options.nsearch_global_pivots as u64);
}

#[test]
fn exact_output_has_no_global_pivot_and_injection_updates_every_cut() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 4,
        max_nglobal_pivots: 2,
        nsweeps_global_search: 4,
        global_tolerance_margin: 1.0,
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    run_directional_pass(&mut state, &options, PassDirection::Forward, &mut identity).unwrap();
    let exact =
        find_global_pivots(&state, &mut input_evaluators, &options, 3, &mut identity).unwrap();
    assert!(exact.pivots.is_empty());

    let (input, _, _) = delta_tree();
    let injection_options = TreeAciOptions {
        max_bond_dim: Some(1),
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut injection_state =
        TreeAciState::<f64, usize>::initialize(&inputs, &injection_options).unwrap();
    let lengths_before = injection_state
        .candidates
        .ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let generation_before = injection_state.generation;
    let injected = inject_global_pivots(&mut injection_state, &[vec![1, 1]], &[1]).unwrap();
    assert_eq!(injected, 1);
    assert!(injection_state
        .candidates
        .ids
        .iter()
        .zip(lengths_before)
        .all(|(ids, before)| ids.len() == before + 1));
    assert_eq!(injection_state.generation, generation_before + 1);
}

/// Injection must never reach the pivot pairs.
///
/// This is the defect the candidate/pivot split exists to prevent: the old
/// single `ActivePivotSets` let an injection push a repeated id onto the
/// already-represented side of a cut, which put two identical rows in `P_e`.
/// Candidate sets may carry that repeat harmlessly — the rank-revealing step
/// simply declines to select it twice — but the pivot pairs must not.
#[test]
fn injection_leaves_pivot_pairs_untouched_and_keeps_bonds_in_step() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let pivots_before = state.pivots.clone();

    inject_global_pivots(&mut state, &[vec![1, 1]], &[1]).unwrap();
    let records_after_first = state.sample_arena.record_count();
    inject_global_pivots(&mut state, &[vec![1, 1]], &[1]).unwrap();
    assert_eq!(state.sample_arena.record_count(), records_after_first);

    assert_eq!(state.pivots, pivots_before);
    for edge_number in 0..state.pivots.per_edge.len() {
        let mut rows = state.pivots.forward_ids(edge_number);
        let before = rows.len();
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(
            rows.len(),
            before,
            "pivot rows repeated on edge {edge_number}"
        );
    }

    // INVARIANT: in the moving-centre representation an output core's incoming
    // axes are reshaped over the candidate row space, so each bond dimension
    // must equal both directed candidate-set sizes. This is what forces
    // injection to pad the bond, and to grow both orientations together.
    for edge_number in 0..state.edge_ranks.len() {
        let forward = state.candidates.ids[2 * edge_number].len();
        let reverse = state.candidates.ids[2 * edge_number + 1].len();
        assert_eq!(forward, reverse, "cut {edge_number} grew asymmetrically");
        assert_eq!(state.output.link_dims()[edge_number], forward);
    }
}

/// Gate 1 of the three growth gates inherited from train ACI: the guard offers
/// at most `max_nglobal_pivots` points per run.
#[test]
fn max_nglobal_pivots_caps_what_the_guard_offers() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 8,
        max_nglobal_pivots: 1,
        nsweeps_global_search: 4,
        global_tolerance_margin: 1.0,
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left_site, right_site);
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    let report =
        find_global_pivots(&state, &mut input_evaluators, &options, 9, &mut identity).unwrap();

    assert_eq!(report.pivots.len(), 1);
}

/// Gate 3: a point already represented in every candidate set adds nothing.
///
/// Gate 2, algebraic saturation, is covered by
/// `injection_skips_saturated_cuts_but_retains_recursive_records`.
#[test]
fn an_already_represented_point_adds_nothing() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let seed_point = vec![0, 0];
    let generation_before = state.generation;

    let injected = inject_global_pivots(&mut state, &[seed_point], &[1]).unwrap();

    assert_eq!(injected, 0);
    assert_eq!(state.generation, generation_before);
}

#[test]
fn injection_skips_saturated_cuts_but_retains_recursive_records() {
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let site2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(1);
    let bond12 = DynIndex::new_dyn(1);
    let input = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![site0, bond01.clone()], vec![1.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![bond01, site1, bond12.clone()], vec![1.0, 1.0]).unwrap(),
            IdxTensor::from_dense(vec![bond12, site2], vec![1.0, 1.0]).unwrap(),
        ],
        vec![0, 1, 2],
    )
    .unwrap();
    let options = TreeAciOptions {
        max_bond_dim: Some(2),
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let before = state
        .candidates
        .ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let saturated_edge = state.output.edge_between(&0, &1).unwrap();
    let saturated_bond = state.output.bond_index(saturated_edge).unwrap().clone();
    let records_before = state.sample_arena.record_count();

    let injected = inject_global_pivots(&mut state, &[vec![1, 1, 1]], &[0, 1]).unwrap();

    assert_eq!(injected, 1);
    assert_eq!(state.candidates.ids[0].len(), before[0]);
    assert_eq!(state.candidates.ids[1].len(), before[1]);
    assert_eq!(state.candidates.ids[2].len(), before[2] + 1);
    assert_eq!(state.candidates.ids[3].len(), before[3] + 1);
    assert_eq!(state.edge_ranks, vec![1, 2]);
    assert_eq!(state.output.link_dims(), vec![1, 2]);
    assert_eq!(
        state.output.bond_index(saturated_edge),
        Some(&saturated_bond),
        "padding another cut must not replace an inactive bond"
    );
    assert!(state.sample_arena.record_count() > records_before);
    run_directional_pass(&mut state, &options, PassDirection::Forward, &mut identity).unwrap();
    let edge_one_pivots_before = state.pivots.per_edge[1].clone();
    update_edge_transaction(&mut state, 0, &options, true, &mut identity).unwrap();
    assert_eq!(state.pivots.per_edge[1], edge_one_pivots_before);
    state.output.verify_internal_consistency().unwrap();
}

#[test]
fn injection_never_exceeds_the_remaining_cut_capacity() {
    let (_, left_site, right_site) = delta_tree();
    let input = zero_tree(left_site, right_site);
    let options = TreeAciOptions {
        max_bond_dim: Some(2),
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let candidate_sizes_before = state
        .candidates
        .ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();

    let injected = inject_global_pivots(&mut state, &[vec![1, 1], vec![0, 1]], &[1]).unwrap();

    assert_eq!(injected, 1);
    assert_eq!(state.edge_ranks, vec![2]);
    assert_eq!(state.output.link_dims(), vec![2]);
    assert!(state
        .candidates
        .ids
        .iter()
        .zip(candidate_sizes_before)
        .all(|(ids, before)| ids.len() == before + 1));
}

#[test]
fn empty_global_pivot_injection_validates_capacity_without_mutating_state() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let output_before = state.output.to_dense().unwrap();
    let candidates_before = state.candidates.clone();
    let pivots_before = state.pivots.clone();
    let records_before = state.sample_arena.record_count();
    let generation_before = state.generation;
    let ranks_before = state.edge_ranks.clone();

    assert_eq!(inject_global_pivots(&mut state, &[], &[1]).unwrap(), 0);
    assert_eq!(state.candidates, candidates_before);
    assert_eq!(state.pivots, pivots_before);
    assert_eq!(state.sample_arena.record_count(), records_before);
    assert_eq!(state.generation, generation_before);
    assert_eq!(state.edge_ranks, ranks_before);
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&output_before, 0.0, 0.0)
        .unwrap());

    assert!(matches!(
        inject_global_pivots(&mut state, &[], &[]),
        Err(crate::TreeAciError::InternalInvariant { .. })
    ));
}

#[test]
fn injection_rank_overflow_rolls_back_every_staged_state_change() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions {
        max_bond_dim: Some(1),
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let output_before = state.output.to_dense().unwrap();
    let candidates_before = state.candidates.clone();
    let pivots_before = state.pivots.clone();
    let frames_before = state.input_frames.clone();
    let arena_records_before = state.sample_arena.record_count();
    let arena_bytes_before = state.sample_arena.retained_bytes();
    let frame_records_before = state.input_frames.records();
    let frame_bytes_before = state.input_frames.retained_bytes();
    let generation_before = state.generation;
    let ranks_before = state.edge_ranks.clone();
    state.edge_ranks[0] = usize::MAX;
    let ranks_before_injected_overflow = state.edge_ranks.clone();

    let error = inject_global_pivots(&mut state, &[vec![1, 1]], &[1])
        .expect_err("rank overflow must be reported before publication");

    assert!(matches!(
        error,
        crate::TreeAciError::SizeOverflow {
            context: "global-pivot output rank"
        }
    ));
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&output_before, 0.0, 0.0)
        .unwrap());
    assert_eq!(state.candidates, candidates_before);
    assert_eq!(state.pivots, pivots_before);
    assert_eq!(state.sample_arena.record_count(), arena_records_before);
    assert_eq!(state.sample_arena.retained_bytes(), arena_bytes_before);
    for (input_index, edges) in frames_before.frames.iter().enumerate() {
        for (edge, frame) in edges.iter().enumerate() {
            assert_eq!(
                state.input_frames.frames[input_index][edge].sample_count,
                frame.sample_count
            );
            assert_eq!(
                state.input_frames.frames[input_index][edge].bond_dim,
                frame.bond_dim
            );
            for sample in 0..frame.sample_count {
                assert_eq!(
                    state
                        .input_frames
                        .frame_values(input_index, edge, sample)
                        .unwrap(),
                    frames_before
                        .frame_values(input_index, edge, sample)
                        .unwrap()
                );
            }
        }
    }
    assert_eq!(state.input_frames.records(), frame_records_before);
    assert_eq!(state.input_frames.retained_bytes(), frame_bytes_before);
    assert_eq!(state.generation, generation_before);
    assert_eq!(state.edge_ranks, ranks_before_injected_overflow);
    assert_ne!(state.edge_ranks, ranks_before);
}

#[test]
fn output_guard_evaluator_matches_exact_values_across_scan_centers() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut output_evaluator = GuardOutputEvaluator::new(
        &state.output,
        &state.problem,
        options.message_cache_max_bytes,
    )
    .unwrap();

    // Warm the reusable output evaluator on a scan through site 0.
    let first_points = [vec![0, 0], vec![1, 0]];
    let _: Vec<f64> = output_evaluator
        .evaluate(&input_evaluators, &first_points)
        .unwrap();

    // The next guard scan varies site 1. Re-rooting around that site changes
    // contraction order only; exact values must remain unchanged.
    let second_points = [vec![0, 0], vec![0, 1]];
    let actual: Vec<f64> = output_evaluator
        .evaluate(&input_evaluators, &second_points)
        .unwrap();

    let values = [0usize, 0, 0, 1];
    let expected = state
        .output
        .evaluate(
            &[left_site, right_site],
            ColMajorArrayRef::new(&values, &[2, 2]).unwrap(),
        )
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected.real()).abs() < 1.0e-12);
    }
}

#[test]
fn shared_guard_hint_preserves_input_and_output_values() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let mut output_evaluator = GuardOutputEvaluator::new(
        &state.output,
        &state.problem,
        options.message_cache_max_bytes,
    )
    .unwrap();
    let points = vec![vec![0, 0], vec![1, 0]];
    let coordinates = input_evaluators.expand_points(&points).unwrap();
    let hint = input_evaluators.evaluation_hint(&points);
    let input_values = input_evaluators
        .evaluate_expanded::<f64>(&points, &coordinates, hint.clone())
        .unwrap();
    let output_values: Vec<f64> = output_evaluator
        .evaluate_expanded(&input_evaluators, &points, &coordinates, hint)
        .unwrap();

    assert_eq!(input_values, vec![1.0, 0.0]);
    let expected = state
        .output
        .evaluate(
            &[left_site, right_site],
            ColMajorArrayRef::new(&coordinates, &[2, 2]).unwrap(),
        )
        .unwrap();
    for (actual, expected) in output_values.iter().zip(expected) {
        assert!((actual - expected.real()).abs() < 1.0e-12);
    }
}

/// [AI Supplied] Guard-facing dtype gate: the cached output evaluator and its
/// `TreeAciScalar` conversion must agree with an ordinary dense evaluation for
/// all four supported scalar kinds.
#[test]
fn guard_cached_evaluator_preserves_all_scalar_kinds() {
    assert_guard_typed_evaluation::<f32>();
    assert_guard_typed_evaluation::<f64>();
    assert_guard_typed_evaluation::<Complex32>();
    assert_guard_typed_evaluation::<Complex64>();
}

#[test]
fn varying_site_detection_rejects_non_scan_batches() {
    assert_eq!(
        sole_varying_site(&[vec![0, 0, 0], vec![0, 1, 0], vec![0, 2, 0]]),
        Some(1)
    );
    assert_eq!(sole_varying_site(&[vec![0, 0], vec![1, 1]]), None);
    assert_eq!(sole_varying_site(&[vec![0, 0]]), None);
    assert_eq!(sole_varying_site(&[vec![0, 0], vec![0]]), None);
}

/// Padding rejects an over-budget request before allocating any padded core.
///
/// Sizing and allocation used to share one loop, with the aggregate
/// `max_working_bytes` comparison after it, so every core was already allocated
/// and retained by the time the caller was told the request was refused. The
/// per-core `max_core_elements` bound did not help: N nodes each just under it
/// still peak at N times that, whatever the working ceiling says.
///
/// The check now runs on the planned total, so a one-byte ceiling refuses the
/// request outright.
#[test]
fn padding_refuses_an_over_budget_request() {
    let (input, _, _) = delta_tree();
    let options = TreeAciOptions {
        max_bond_dim: Some(1),
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    // Tighten after preparation so this exercises the padding check rather than
    // the preparation-time one, which would reject the run before injection.
    state.problem.max_working_bytes = 1;
    let error = inject_global_pivots(&mut state, &[vec![1, 1]], &[1])
        .expect_err("a one-byte working ceiling must refuse padding");
    assert!(
        matches!(
            error,
            crate::TreeAciError::ResourceLimit {
                resource: "working bytes",
                ..
            }
        ),
        "unexpected error: {error}"
    );
}
