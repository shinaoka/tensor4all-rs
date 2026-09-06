use num_complex::{Complex32, Complex64};
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{
    find_global_pivots, inject_global_pivots, per_evaluator_message_cache_budget,
    GuardOutputEvaluator, InputEvaluators,
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
        input_evaluators.plan(),
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

/// Dense oracle for a whole batch: materializes the tree's values once
/// through the ordinary evaluator instead of re-contracting per element.
fn dense_guard_oracle<T: GuardTestScalar>(
    tree: &TreeTN<IdxTensor, usize>,
    problem: &crate::problem::PreparedTreeProblem<usize>,
    coordinates: &[usize],
    n_points: usize,
) -> Vec<T> {
    let indices = problem
        .physical
        .iter()
        .flat_map(|physical| physical.indices.iter().cloned())
        .collect::<Vec<_>>();
    tree.evaluate(
        &indices,
        ColMajorArrayRef::new(coordinates, &[indices.len(), n_points]).unwrap(),
    )
    .unwrap()
    .into_iter()
    .map(|value| T::from_evaluated_scalar(value).unwrap())
    .collect()
}

/// [AI Supplied] #709: the typed Guard input route must reproduce the
/// `AnyScalar` route's values exactly, for every supported scalar kind and
/// for the scan-shaped batches the floating-zone walk actually issues.
fn assert_guard_input_typed_matches_wrapper<T: GuardTestScalar>() {
    let (input, _, _) = typed_delta_tree::<T>();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let state = TreeAciState::<T, usize>::initialize(&inputs, &options).unwrap();
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    // A scan batch (one varying site) and a general batch take different
    // evaluator routes, so both are compared.
    for points in [
        vec![vec![0usize, 0], vec![1, 0]],
        vec![
            vec![0usize, 0],
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
            vec![1, 0],
        ],
    ] {
        let coordinates = input_evaluators.expand_points(&points).unwrap();
        let expected =
            dense_guard_oracle::<T>(&state.inputs[0], &state.problem, &coordinates, points.len());
        let actual = input_evaluators.evaluate::<T>(&points).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(&expected) {
            let residual = tensor4all_core::Scalar::abs_val(*actual - *expected);
            let scale = tensor4all_core::Scalar::abs_val(*expected).max(1.0);
            assert!(
                residual <= T::TOLERANCE * scale,
                "typed guard input residual {residual} exceeds tolerance for {}",
                std::any::type_name::<T>()
            );
        }
    }
}

/// [AI Supplied] #709 differential gate for the Guard's typed input route.
#[test]
fn guard_typed_input_evaluation_matches_the_dense_oracle_for_all_scalar_kinds() {
    assert_guard_input_typed_matches_wrapper::<f32>();
    assert_guard_input_typed_matches_wrapper::<f64>();
    assert_guard_input_typed_matches_wrapper::<Complex32>();
    assert_guard_input_typed_matches_wrapper::<Complex64>();
}

/// [AI Supplied] #709 invalidation gate. The Guard rebuilds its output
/// evaluator whenever the output tensors change, but the immutable plan must
/// survive that rebuild -- including a change of bond dimension -- while the
/// numerical messages must not: the second evaluator has to answer with the
/// new output's values.
#[test]
fn guard_output_evaluator_reuses_the_plan_when_output_tensors_change() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let points = vec![vec![0usize, 0], vec![1, 0], vec![0, 1], vec![1, 1]];
    let coordinates = input_evaluators.expand_points(&points).unwrap();

    let first_values: Vec<f64> = {
        let mut first = GuardOutputEvaluator::new(
            &state.output,
            input_evaluators.plan(),
            options.message_cache_max_bytes,
        )
        .unwrap();
        assert!(
            first.evaluator.plan().is_same_as(input_evaluators.plan()),
            "the output evaluator must share the retained input plan"
        );
        let values = first.evaluate::<f64>(&input_evaluators, &points).unwrap();
        let expected =
            dense_guard_oracle::<f64>(&state.output, &state.problem, &coordinates, points.len());
        assert_eq!(values, expected);
        values
    };

    // Replace the output with a same-topology tree that has different values
    // and a different bond dimension, as pivot injection does.
    state.output = zero_tree(left_site, right_site);
    let mut second = GuardOutputEvaluator::new(
        &state.output,
        input_evaluators.plan(),
        options.message_cache_max_bytes,
    )
    .unwrap();
    assert!(
        second.evaluator.plan().is_same_as(input_evaluators.plan()),
        "a changed output must reuse the plan rather than rebuild it"
    );
    let second_values: Vec<f64> = second.evaluate(&input_evaluators, &points).unwrap();
    let second_expected =
        dense_guard_oracle::<f64>(&state.output, &state.problem, &coordinates, points.len());
    assert_eq!(second_values, second_expected);
    assert_ne!(
        first_values, second_values,
        "the fixture must actually change the output values"
    );
}

/// [AI Supplied] #709: typed evaluation must keep reporting a dtype rejection
/// as `TreeAciError::ScalarKind`, the class the `AnyScalar` route reported,
/// rather than degrading it into an opaque evaluator failure.
#[test]
fn typed_guard_evaluation_still_reports_a_scalar_kind_error_for_complex_inputs() {
    let (input, _, _) = typed_delta_tree::<Complex64>();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let state = TreeAciState::<Complex64, usize>::initialize(&inputs, &options).unwrap();
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    let error = input_evaluators
        .evaluate::<f64>(&[vec![0usize, 0], vec![1, 0]])
        .expect_err("a complex input tree cannot answer an f64 request");

    match error {
        crate::TreeAciError::ScalarKind { message } => {
            assert!(
                message.contains("f64"),
                "the rejection must name the requested dtype: {message}"
            );
        }
        other => panic!("expected a scalar-kind rejection, got {other:?}"),
    }

    // The same evaluator still answers a complex request.
    let complex = input_evaluators
        .evaluate::<Complex64>(&[vec![0usize, 0], vec![1, 0]])
        .unwrap();
    assert_eq!(complex.len(), 2);
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
        // Pin the element ceilings so the 64-byte budget exercises only the
        // guard's start-batch charge. Left unset they would follow the budget
        // down to two elements and preparation would refuse the tree first.
        max_local_matrix_elements: Some(1 << 24),
        max_core_elements: Some(1 << 24),
        max_frame_elements: Some(1 << 24),
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
        input_evaluators.plan(),
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
        input_evaluators.plan(),
        options.message_cache_max_bytes,
    )
    .unwrap();
    let points = vec![vec![0, 0], vec![1, 0]];
    let coordinates = input_evaluators.expand_points(&points).unwrap();
    let hint = input_evaluators.hint_for_scan_site(Some(0));
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

/// The scan site the walk declares is what selects the centre, and the seed
/// evaluation declares none.
#[test]
fn scan_site_selects_the_hinted_center() {
    let (input, _, _) = delta_tree();
    let inputs = vec![input];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();
    let evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    assert_eq!(evaluators.hint_for_scan_site(Some(1)).center, Some(1));
    assert_eq!(evaluators.hint_for_scan_site(None).center, None);
    // A site outside the tree cannot be hinted, and must not be invented.
    assert_eq!(evaluators.hint_for_scan_site(Some(99)).center, None);
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

/// Opt-in cost attribution for the Guard's evaluation shape on the 32-site
/// high-rank chain (issue #728).
///
/// The Guard walks `nsearch_global_pivots` independent floating-zone
/// trajectories, and each walk step asks for one site's `local_dim` points.
/// That is 2 points per evaluator call, which is what makes the reported 2,218
/// calls for 8,844 scalars. The trajectories are independent, so the same
/// points could be requested in one call per (sweep, site) across all starts
/// instead -- "option 1" of the issue, which changes only how the search is
/// executed and not what it searches.
///
/// This measures whether that is actually cheaper, using the same points in
/// both shapes, before any search behaviour is changed. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn diagnostic_guard_batch_shape_on_the_high_rank_chain() {
    use std::time::Instant;

    const N_SITES: usize = 32;
    const NSEARCH: usize = 5;
    const SWEEPS: usize = 3;

    for chi in [64, 256] {
        let guess = crate::state::tests::full_rank_chain_guess(N_SITES, chi);
        let inputs = vec![guess.clone(), guess.clone()];
        let options = TreeAciOptions {
            tolerance: 1.0e-8,
            max_bond_dim: Some(4096),
            max_sweeps: 2,
            min_sweeps: 2,
            initial_guess: Some(guess),
            enable_global_guard: false,
            ..TreeAciOptions::default()
        };
        let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let budget =
            per_evaluator_message_cache_budget(options.message_cache_max_bytes, inputs.len())
                .unwrap();

        // The trajectories the Guard would walk, materialized up front so both
        // shapes evaluate exactly the same points in the same order.
        let mut starts: Vec<Vec<usize>> = Vec::with_capacity(NSEARCH);
        for start in 0..NSEARCH {
            starts.push(
                (0..N_SITES)
                    .map(|site| (start * 7 + site * 3) % 2)
                    .collect(),
            );
        }
        let mut steps: Vec<Vec<Vec<usize>>> = Vec::new();
        for _sweep in 0..SWEEPS {
            for site in 0..N_SITES {
                for start in &starts {
                    let mut points = Vec::with_capacity(2);
                    for value in 0..2 {
                        let mut point = start.clone();
                        point[site] = value;
                        points.push(point);
                    }
                    steps.push(points);
                }
            }
        }

        // Shape A: one call per (start, sweep, site), which is what the walk
        // does today.
        let mut per_start =
            InputEvaluators::new_with_message_cache_max_bytes(state.inputs, &state.problem, budget)
                .unwrap();
        per_start.evaluate::<f64>(&starts[..1]).unwrap();
        let started = Instant::now();
        let mut per_start_values = Vec::new();
        for points in &steps {
            per_start_values.extend(per_start.evaluate::<f64>(points).unwrap());
        }
        let per_start_elapsed = started.elapsed();

        // Shape B: one call per (sweep, site) carrying every start's points.
        // Identical points, identical order, `NSEARCH` times fewer calls.
        let mut lockstep =
            InputEvaluators::new_with_message_cache_max_bytes(state.inputs, &state.problem, budget)
                .unwrap();
        lockstep.evaluate::<f64>(&starts[..1]).unwrap();
        let started = Instant::now();
        let mut lockstep_values = Vec::new();
        for chunk in steps.chunks(NSEARCH) {
            let points = chunk.iter().flatten().cloned().collect::<Vec<_>>();
            lockstep_values.extend(lockstep.evaluate::<f64>(&points).unwrap());
        }
        let lockstep_elapsed = started.elapsed();

        assert_eq!(per_start_values.len(), lockstep_values.len());
        let residual = per_start_values
            .iter()
            .zip(&lockstep_values)
            .fold(0.0f64, |residual, (left, right)| {
                residual.max((left - right).abs())
            });
        let scale = per_start_values
            .iter()
            .fold(0.0f64, |scale, value| scale.max(value.abs()));
        assert!(
            residual <= 1.0e-12 * scale.max(1.0),
            "batching changed the values: {residual:.3e}"
        );

        eprintln!(
            "guard batch shape chi={chi}: per_start={per_start_elapsed:?} over {} calls of 2 points ({:.1} us/call), lockstep={lockstep_elapsed:?} over {} calls of {} points ({:.1} us/call), speedup={:.2}x",
            steps.len(),
            per_start_elapsed.as_secs_f64() * 1.0e6 / steps.len() as f64,
            steps.len() / NSEARCH,
            2 * NSEARCH,
            lockstep_elapsed.as_secs_f64() * 1.0e6 * NSEARCH as f64 / steps.len() as f64,
            per_start_elapsed.as_secs_f64() / lockstep_elapsed.as_secs_f64(),
        );
    }
}

/// Opt-in comparison of the Guard's per-call evaluation cost on a chain and on
/// a branched tree of the same walk shape (issues #727 and #728).
///
/// The #718 gate found an unequal incident-bond layout costing ~24x more per
/// evaluated point than an equal one at the same hub bond product. The phase
/// profile attributes that to the Guard running at all on the unequal layouts
/// (the equal one is rank-limited and skips it), so what matters is what one
/// Guard evaluation costs on a branched tree. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn diagnostic_guard_call_cost_on_a_branched_tree() {
    use std::time::Instant;

    const NSEARCH: usize = 5;
    const SWEEPS: usize = 3;
    const ARM_LENGTH: usize = 3;
    const ARM_CHI: usize = 4;

    let cases: [(&str, Vec<usize>); 5] = [
        ("chain_13_chi8", vec![]),
        // A hub of 32 elements: smaller than any chain core in the case above,
        // so a cost difference here cannot be the hub's arithmetic.
        ("spider_tiny_2x2x2x2", vec![2, 2, 2, 2]),
        ("spider_equal_4x4x4x4", vec![4, 4, 4, 4]),
        ("spider_unequal_2x4x8x4", vec![2, 4, 8, 4]),
        ("spider_unequal_8x4x2x4", vec![8, 4, 2, 4]),
    ];
    for (name, bonds) in cases {
        let n_sites = if bonds.is_empty() {
            13
        } else {
            bonds.len() * ARM_LENGTH + 1
        };
        let sites: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
        let inputs: Vec<_> = if bonds.is_empty() {
            let chain = crate::state::tests::full_rank_chain_guess(n_sites, 8);
            vec![chain.clone(), chain]
        } else {
            (0..2)
                .map(|input| {
                    crate::state::tests::spider_tree(&bonds, ARM_LENGTH, ARM_CHI, input, &sites)
                })
                .collect()
        };
        let options = TreeAciOptions {
            tolerance: 1.0e-8,
            max_sweeps: 2,
            min_sweeps: 2,
            enable_global_guard: false,
            ..TreeAciOptions::default()
        };
        let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let budget =
            per_evaluator_message_cache_budget(options.message_cache_max_bytes, inputs.len())
                .unwrap();
        let mut evaluators =
            InputEvaluators::new_with_message_cache_max_bytes(state.inputs, &state.problem, budget)
                .unwrap();

        let starts: Vec<Vec<usize>> = (0..NSEARCH)
            .map(|start| {
                (0..n_sites)
                    .map(|site| (start * 7 + site * 3) % 2)
                    .collect()
            })
            .collect();
        evaluators.evaluate::<f64>(&starts[..1]).unwrap();
        let mut calls = 0usize;
        let started = Instant::now();
        for _sweep in 0..SWEEPS {
            for site in 0..n_sites {
                for start in &starts {
                    let points: Vec<Vec<usize>> = (0..2)
                        .map(|value| {
                            let mut point = start.clone();
                            point[site] = value;
                            point
                        })
                        .collect();
                    evaluators.evaluate::<f64>(&points).unwrap();
                    calls += 1;
                }
            }
        }
        let elapsed = started.elapsed();

        // The same walk with the coordinate the pivot already holds dropped
        // from every batch: one point per call instead of two, with the
        // varying site still declared explicitly so the batch keeps the
        // centre hint a one-point batch cannot be asked to infer.
        let mut single_calls = 0usize;
        let single_started = Instant::now();
        for _sweep in 0..SWEEPS {
            for site in 0..n_sites {
                for start in &starts {
                    let mut point = start.clone();
                    point[site] = 1 - point[site];
                    let points = vec![point];
                    let coordinates = evaluators.expand_points(&points).unwrap();
                    let hint = tensor4all_treetn::EvaluationHint::around(site);
                    evaluators
                        .evaluate_expanded::<f64>(&points, &coordinates, hint)
                        .unwrap();
                    single_calls += 1;
                }
            }
        }
        let single_elapsed = single_started.elapsed();
        eprintln!(
            "guard call cost {name} ({n_sites} sites): two_point={elapsed:?} over {calls} calls = {:.1} us/call, one_point_hinted={single_elapsed:?} over {single_calls} calls = {:.1} us/call, per-call saving={:.0}%",
            elapsed.as_secs_f64() * 1.0e6 / calls as f64,
            single_elapsed.as_secs_f64() * 1.0e6 / single_calls as f64,
            100.0 * (1.0 - single_elapsed.as_secs_f64() / elapsed.as_secs_f64()),
        );
    }
}
