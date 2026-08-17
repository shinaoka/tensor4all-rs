use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{find_global_pivots, inject_global_pivots, InputEvaluators};
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
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left_site, right_site);
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

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
    let mut input_evaluators =
        InputEvaluators::new(injection_state.inputs, &injection_state.problem).unwrap();
    let lengths_before = injection_state
        .candidates
        .ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let generation_before = injection_state.generation;
    let injected = inject_global_pivots(
        &mut injection_state,
        &mut input_evaluators,
        &[vec![1, 1]],
        &[true],
    )
    .unwrap();
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
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let pivots_before = state.pivots.clone();

    inject_global_pivots(&mut state, &mut input_evaluators, &[vec![1, 1]], &[true]).unwrap();
    inject_global_pivots(&mut state, &mut input_evaluators, &[vec![1, 1]], &[true]).unwrap();

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
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let seed_point = vec![0, 0];
    let generation_before = state.generation;

    let injected =
        inject_global_pivots(&mut state, &mut input_evaluators, &[seed_point], &[true]).unwrap();

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
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();
    let before = state
        .candidates
        .ids
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let records_before = state.sample_arena.record_count();

    let injected = inject_global_pivots(
        &mut state,
        &mut input_evaluators,
        &[vec![1, 1, 1]],
        &[false, true],
    )
    .unwrap();

    assert_eq!(injected, 1);
    assert_eq!(state.candidates.ids[0].len(), before[0]);
    assert_eq!(state.candidates.ids[1].len(), before[1]);
    assert_eq!(state.candidates.ids[2].len(), before[2] + 1);
    assert_eq!(state.candidates.ids[3].len(), before[3] + 1);
    assert_eq!(state.edge_ranks, vec![1, 2]);
    assert_eq!(state.output.link_dims(), vec![1, 2]);
    assert!(state.sample_arena.record_count() > records_before);
    run_directional_pass(&mut state, &options, PassDirection::Forward, &mut identity).unwrap();
    let edge_one_pivots_before = state.pivots.per_edge[1].clone();
    update_edge_transaction(&mut state, 0, &options, true, &mut identity).unwrap();
    assert_eq!(state.pivots.per_edge[1], edge_one_pivots_before);
    state.output.verify_internal_consistency().unwrap();
}

#[test]
fn guard_reuses_input_evaluators_across_invocations() {
    let (input, left_site, right_site) = delta_tree();
    let options = TreeAciOptions {
        nsearch_global_pivots: 4,
        max_nglobal_pivots: 2,
        nsweeps_global_search: 4,
        global_tolerance_margin: 1.0,
        ..TreeAciOptions::default()
    };
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    state.output = zero_tree(left_site, right_site);
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    let before_first = input_evaluators.evaluations();
    find_global_pivots(&state, &mut input_evaluators, &options, 9, &mut identity).unwrap();
    let first = input_evaluators.evaluations() - before_first;

    let before_second = input_evaluators.evaluations();
    find_global_pivots(&state, &mut input_evaluators, &options, 10, &mut identity).unwrap();
    let second = input_evaluators.evaluations() - before_second;

    assert!(first > 0);
    assert!(
        second < first,
        "the second guard run must reuse warm evaluators: {second} vs {first}"
    );
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
    let mut input_evaluators = InputEvaluators::new(state.inputs, &state.problem).unwrap();

    // Tighten after preparation so this exercises the padding check rather than
    // the preparation-time one, which would reject the run before injection.
    state.problem.max_working_bytes = 1;
    let error = inject_global_pivots(&mut state, &mut input_evaluators, &[vec![1, 1]], &[true])
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
