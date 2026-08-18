use std::{cell::Cell, collections::HashSet};

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CanonicalForm, TreeTN};

use super::{
    convergence_criterion, current_state_is_rank_limited, edge_error_metric,
    global_injection_edges, run_directional_pass, run_local_sweeps, PassDirection,
};
use crate::global_guard::InputEvaluators;
use crate::transaction::update_edge_transaction;
use crate::{state::TreeAciState, TreeAciError, TreeAciOptions, TreeAciTermination};

fn product_tree(edges: &[(usize, usize)], node_count: usize) -> TreeTN<IdxTensor, usize> {
    let physical = (0..node_count)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = (0..node_count)
        .map(|node| {
            let mut indices = vec![physical[node].clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            let values = (0..len)
                .map(|linear| (node + 1) as f64 + linear as f64)
                .collect();
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..node_count).collect()).unwrap()
}

fn identity(batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)?;
    }
    Ok(())
}

fn rank_two_delta_tree() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left =
        IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn product_tree_with_physical(
    edges: &[(usize, usize)],
    physical: &[DynIndex],
    offset: f64,
) -> TreeTN<IdxTensor, usize> {
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = physical
        .iter()
        .enumerate()
        .map(|(node, site)| {
            let mut indices = vec![site.clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let len = indices.iter().map(IndexLike::dim).product();
            IdxTensor::from_dense(
                indices,
                (0..len)
                    .map(|coordinate| offset + node as f64 + coordinate as f64)
                    .collect(),
            )
            .unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..physical.len()).collect()).unwrap()
}

#[test]
fn path_pass_matches_train_endpoint_order_and_exact_reverse() {
    let options = TreeAciOptions::default();
    let inputs = vec![product_tree(&[(0, 1), (1, 2), (2, 3)], 4)];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();

    let forward =
        run_directional_pass(&mut state, &options, PassDirection::Forward, &mut identity).unwrap();
    let reverse =
        run_directional_pass(&mut state, &options, PassDirection::Reverse, &mut identity).unwrap();

    assert_eq!(forward.updated_edges, vec![5, 3, 1]);
    assert_eq!(reverse.updated_edges, vec![0, 2, 4]);
    assert_eq!(forward.update_count(), 3);
    assert_eq!(reverse.update_count(), 3);
    assert_eq!(state.output.canonical_region(), &HashSet::from([3]));
    assert_eq!(state.output.canonical_form(), Some(CanonicalForm::CI));
}

#[test]
fn branched_topologies_cover_every_edge_with_optimal_retracing() {
    for (edges, expected_updates) in [
        (vec![(0, 1), (0, 2), (0, 3)], 4),
        (vec![(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 8),
        (vec![(0, 1), (1, 2), (1, 3), (2, 4), (2, 5)], 7),
    ] {
        let options = TreeAciOptions::default();
        let inputs = vec![product_tree(&edges, edges.len() + 1)];
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let forward =
            run_directional_pass(&mut state, &options, PassDirection::Forward, &mut identity)
                .unwrap();
        let reverse =
            run_directional_pass(&mut state, &options, PassDirection::Reverse, &mut identity)
                .unwrap();
        let forward_edges = forward
            .updated_edges
            .iter()
            .map(|directed| directed / 2)
            .collect::<HashSet<_>>();
        let reverse_edges = reverse
            .updated_edges
            .iter()
            .map(|directed| directed / 2)
            .collect::<HashSet<_>>();

        assert_eq!(forward.update_count(), expected_updates);
        assert_eq!(reverse.update_count(), expected_updates);
        assert_eq!(forward_edges, (0..edges.len()).collect());
        assert_eq!(reverse_edges, (0..edges.len()).collect());
        let inverse = forward
            .updated_edges
            .iter()
            .rev()
            .map(|directed| directed ^ 1)
            .collect::<Vec<_>>();
        assert_eq!(reverse.updated_edges, inverse);
        state.output.verify_internal_consistency().unwrap();
    }
}

#[test]
fn failed_update_preserves_all_commits_before_the_failing_edge() {
    let options = TreeAciOptions::default();
    let inputs = vec![product_tree(&[(0, 1), (1, 2), (2, 3)], 4)];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let generation_before = state.generation;
    let calls = Cell::new(0usize);
    let mut fail_after_one = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        if calls.get() > 1 {
            return Err(TreeAciError::Callback {
                message: "stop after the first committed edge".into(),
            });
        }
        identity(batch, output)
    };
    let result = run_directional_pass(
        &mut state,
        &options,
        PassDirection::Forward,
        &mut fail_after_one,
    );

    assert!(matches!(result, Err(TreeAciError::Callback { .. })));
    assert_eq!(state.generation, generation_before + 1);
    assert_eq!(state.output.canonical_region(), &HashSet::from([2]));
    assert_eq!(state.output.canonical_form(), Some(CanonicalForm::CI));
    state.output.verify_internal_consistency().unwrap();
}

#[test]
fn convergence_requires_stable_max_rank_and_minimum_dwell() {
    // Network max is 2 at both sweeps (2 and 2), even though this scenario's
    // real-world equivalent is an individual edge fluctuating underneath
    // that max -- exactly the scenario `crates/tensor4all-treeaci/src/schedule.rs`'s
    // old per-edge-vector check used to reject. `max_ranks` is what
    // `run_local_sweeps` already computes from `state.edge_ranks.iter().max()`
    // every pass; this test operates directly on that scalar sequence, the
    // same shape `tensor4all-aci`'s `convergence_criterion_like_julia` takes.
    let max_ranks = vec![2usize, 2];
    let errors = vec![0.0, 0.0];
    let global = vec![0, 0];

    // Below the minimum dwell (only 1 completed sweep): never converges.
    assert!(!convergence_criterion(
        1,
        &max_ranks[..1],
        &errors[..1],
        &global[..1],
        2,
        1.0e-12
    ));

    // Two sweeps, stable max rank, error at tolerance, no global pivots
    // found in the window: must converge.
    assert!(convergence_criterion(
        2, &max_ranks, &errors, &global, 2, 1.0e-12
    ));

    // A max rank that actually increases within the window must still
    // block convergence (this part of the rule is unchanged).
    let growing = vec![2usize, 3];
    assert!(!convergence_criterion(
        2, &growing, &errors, &global, 2, 1.0e-12
    ));

    // Error above tolerance still blocks convergence, independent of rank.
    assert!(!convergence_criterion(
        2,
        &max_ranks,
        &[0.0, 1.0e-6],
        &global,
        2,
        1.0e-12
    ));

    // A global pivot found within the window still blocks convergence
    // (its error estimate is stale with respect to the injected pivot).
    assert!(!convergence_criterion(
        2,
        &max_ranks,
        &errors,
        &[1, 0],
        2,
        1.0e-12
    ));
}

#[test]
fn relative_and_absolute_error_metrics_match_train_aci() {
    assert_eq!(edge_error_metric(2.0, 10.0, true), 0.2);
    assert_eq!(edge_error_metric(2.0, 10.0, false), 2.0);
    assert_eq!(edge_error_metric(2.0, 0.0, true), 2.0);
}

#[test]
fn algebraically_saturated_cut_cannot_receive_global_rank_growth() {
    let options = TreeAciOptions::default();
    let inputs = vec![rank_two_delta_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    assert_eq!(state.edge_ranks, vec![2]);
    assert_eq!(state.algebraic_edge_bounds, vec![2]);
    assert_eq!(global_injection_edges(&state, &options), vec![false]);

    state.edge_errors[0] = 1.0;
    state.edge_scales[0] = 1.0;
    assert!(current_state_is_rank_limited(&state, &options));
}

#[test]
fn local_sweeps_honor_convergence_and_rank_limit_dwell() {
    let converged_options = TreeAciOptions {
        enable_global_guard: false,
        min_sweeps: 2,
        max_sweeps: 5,
        ..TreeAciOptions::default()
    };
    let converged_inputs = vec![product_tree(&[(0, 1), (1, 2)], 3)];
    let mut converged_state =
        TreeAciState::<f64, usize>::initialize(&converged_inputs, &converged_options).unwrap();
    let mut converged_evaluators =
        InputEvaluators::new(converged_state.inputs, &converged_state.problem).unwrap();
    let converged = run_local_sweeps(
        &mut converged_state,
        &mut converged_evaluators,
        &converged_options,
        &mut identity,
    )
    .unwrap();
    assert_eq!(converged.termination, TreeAciTermination::Converged);
    assert_eq!(converged.max_ranks.len(), 2);
    assert_eq!(converged.max_errors.len(), 2);

    let limited_options = TreeAciOptions {
        enable_global_guard: false,
        max_bond_dim: Some(1),
        min_sweeps: 2,
        max_sweeps: 5,
        ..TreeAciOptions::default()
    };
    let limited_inputs = vec![rank_two_delta_tree()];
    let mut limited_state =
        TreeAciState::<f64, usize>::initialize(&limited_inputs, &limited_options).unwrap();
    let mut limited_evaluators =
        InputEvaluators::new(limited_state.inputs, &limited_state.problem).unwrap();
    let limited = run_local_sweeps(
        &mut limited_state,
        &mut limited_evaluators,
        &limited_options,
        &mut identity,
    )
    .unwrap();
    assert_eq!(limited.termination, TreeAciTermination::RankLimited);
    assert_eq!(limited.max_ranks, vec![1, 1]);
    assert!(limited.max_errors.iter().all(|error| *error > 1.0e-12));
}

#[test]
fn continuous_minimum_retracing_walk_preserves_pivot_gauges_on_binary_tree() {
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
    let physical = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let left = product_tree_with_physical(&edges, &physical, 1.0);
    let right = product_tree_with_physical(&edges, &physical, 2.0);
    let expected_left = left.to_dense().unwrap();
    let expected_right = right
        .to_dense()
        .unwrap()
        .permute_indices(expected_left.indices())
        .unwrap();
    let expected = expected_left
        .to_vec::<f64>()
        .unwrap()
        .into_iter()
        .zip(expected_right.to_vec::<f64>().unwrap())
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let options = TreeAciOptions {
        root: Some(3),
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let inputs = vec![left, right];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut add = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)? + batch.get(1, point)?;
        }
        Ok(())
    };
    let forward = [5, 6, 7, 1, 2, 10, 11, 8];
    for directed in forward {
        update_edge_transaction(&mut state, directed, &options, true, &mut add).unwrap();
    }
    for directed in forward.into_iter().rev().map(|directed| directed ^ 1) {
        update_edge_transaction(&mut state, directed, &options, true, &mut add).unwrap();
    }
    let actual = state
        .output
        .to_dense()
        .unwrap()
        .permute_indices(expected_left.indices())
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-9));
}
