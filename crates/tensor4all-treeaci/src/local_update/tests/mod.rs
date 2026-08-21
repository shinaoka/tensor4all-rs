use std::cell::Cell;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{enumerate_candidates, materialize_and_factor_edge};
use crate::{
    frames::InputFrameStore, problem::prepare_problem, samples::SampleArena, TreeAciError,
    TreeAciOptions,
};

fn two_node_tree(scale: f64) -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    two_node_tree_with_indices(scale, s0, s1, bond)
}

fn two_node_tree_with_indices(
    scale: f64,
    s0: DynIndex,
    s1: DynIndex,
    bond: DynIndex,
) -> TreeTN<IdxTensor, usize> {
    let left = IdxTensor::from_dense(
        vec![s0, bond.clone()],
        vec![scale, 2.0 * scale, 10.0 * scale, 20.0 * scale],
    )
    .unwrap();
    let right = IdxTensor::from_dense(vec![bond, s1], vec![3.0, 4.0, 30.0, 40.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

#[test]
fn local_entries_equal_direct_values_and_callback_layout_is_column_major() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let inputs = vec![
        two_node_tree_with_indices(1.0, s0.clone(), s1.clone(), DynIndex::new_dyn(2)),
        two_node_tree_with_indices(2.0, s0, s1, DynIndex::new_dyn(2)),
    ];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let calls = Cell::new(0);
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        assert_eq!(batch.n_inputs(), 2);
        assert_eq!(batch.n_points(), 4);
        assert_eq!(
            batch.as_col_major_slice(),
            &[43.0, 86.0, 86.0, 172.0, 430.0, 860.0, 860.0, 1720.0]
        );
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)? * batch.get(1, point)?;
        }
        Ok(())
    };

    let update = materialize_and_factor_edge(
        &inputs,
        &problem,
        &arena,
        &active,
        &frames,
        0,
        &options,
        true,
        &mut operator,
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        update.local_values,
        vec![3698.0, 14792.0, 369800.0, 1479200.0]
    );
    assert_eq!(update.sampled_scale, 1479200.0);
}

#[test]
fn luci_factors_reconstruct_rank_one_and_zero_targets() {
    for zero in [false, true] {
        let inputs = vec![two_node_tree(1.0)];
        let options = TreeAciOptions::default();
        let problem = prepare_problem(&inputs, &options).unwrap();
        let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = if zero { 0.0 } else { batch.get(0, point)? };
            }
            Ok(())
        };
        let update = materialize_and_factor_edge(
            &inputs,
            &problem,
            &arena,
            &active,
            &frames,
            0,
            &options,
            true,
            &mut operator,
        )
        .unwrap();

        assert_eq!(update.left.ncols(), 1);
        assert_eq!(update.right.nrows(), 1);
        for col in 0..update.col_count {
            for row in 0..update.row_count {
                let reconstructed = update.left[[row, 0]] * update.right[[0, col]];
                assert!(
                    (reconstructed - update.local_values[row + update.row_count * col]).abs()
                        < 1.0e-10
                );
            }
        }
    }
}

#[test]
fn callback_error_and_matrix_budget_stop_before_factorization() {
    let inputs = vec![two_node_tree(1.0)];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let mut failing = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "sentinel".into(),
        })
    };
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs, &problem, &arena, &active, &frames, 0, &options, true, &mut failing,
        ),
        Err(TreeAciError::Callback { message }) if message == "sentinel"
    ));

    let limited = TreeAciOptions {
        max_local_matrix_elements: 3,
        ..TreeAciOptions::default()
    };
    let mut unused = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| Ok(());
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs,
            &problem,
            &arena,
            &active,
            &frames,
            0,
            &limited,
            true,
            &mut unused,
        ),
        Err(TreeAciError::ResourceLimit {
            resource: "local matrix elements",
            requested: 4,
            limit: 3
        })
    ));
}

/// Builds a 3-node chain `0 -- 1 -- 2` whose middle node (`1`) has a
/// non-trivial (dim-3) physical leg and asymmetric bond dimensions, so the
/// single-incoming-edge batched path in
/// `InputFrameStore::candidate_frames_for_edge` (dispatched from
/// `candidate_frames`) exercises a genuinely rectangular core matrix rather
/// than a degenerate square one. Directed edge `1 -> 2` has exactly one
/// incoming edge (`0 -> 1`), so it is the edge under test.
fn three_node_chain_for_batched_dispatch() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let bond12 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);

    let node0 = IdxTensor::from_dense(vec![s0, bond01.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let node1 = IdxTensor::from_dense(
        vec![bond01, s1, bond12.clone()],
        (1..=12).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node2 = IdxTensor::from_dense(vec![bond12, s2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2], vec![0, 1, 2]).unwrap()
}

#[test]
fn candidate_frames_batched_path_matches_scalar_path_on_a_chain() {
    let inputs = vec![three_node_chain_for_batched_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    // Seed both physical values of node 0 so directed edge 0 -> 1's
    // candidate set has two distinct incoming samples; combined with node
    // 1's dim-3 physical leg, `enumerate_candidates` below produces 6
    // candidates for edge 1 -> 2, whose `local_coordinate` values are
    // strided (not contiguous) across the candidate slice -- see
    // `local_update::enumerate_candidates`'s mixed-radix encoding.
    let seeds = vec![vec![0, 0, 0], vec![1, 0, 0]];
    let (_arena, candidates) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    // Directed edge id 2 is node 1 -> node 2 (forward edge of the second
    // undirected edge (1, 2)); it has exactly one incoming edge (0 -> 1),
    // making it the single-incoming-edge fast path under test.
    let edge = 2;
    assert_eq!(problem.directed_edges[edge].from, 1);
    assert_eq!(problem.directed_edges[edge].to, 2);
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 1);

    let row_candidates =
        enumerate_candidates(&problem, &candidates, edge, "candidate rows", 1_000).unwrap();
    assert_eq!(row_candidates.len(), 6);
    let distinct_local_coordinates = row_candidates
        .iter()
        .map(|candidate| candidate.local_coordinate)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(distinct_local_coordinates.len(), 3);

    // The batched-dispatch path, run first against a freshly built store so
    // its own (empty) candidate cache cannot be pre-populated by the scalar
    // oracle below.
    let (arena_for_batched, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames_batched =
        InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena_for_batched).unwrap();
    let batched = frames_batched
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &row_candidates)
        .unwrap();

    // The scalar oracle: a second, independently built store, walked one
    // candidate at a time through the pre-existing per-candidate
    // `candidate_frame` method that `candidate_frames_for_edge` falls back
    // to for non-single-incoming edges. This is the code path that shipped
    // before this task and is untouched by it, so a mismatch here is a real
    // regression in the new batched math, not a comparison of the new path
    // to itself.
    let (arena_for_scalar, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames_scalar =
        InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena_for_scalar).unwrap();
    let scalar = row_candidates
        .iter()
        .map(|candidate| {
            frames_scalar
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(batched.len(), scalar.len());
    assert_eq!(batched, scalar);
    // Sanity: the frame vectors are exactly `bond12`'s dimension long, and
    // not all zero/identical, so this is a meaningful numeric comparison and
    // not a vacuous shape check.
    for frame in &batched {
        assert_eq!(frame.len(), 2);
    }
    assert!(batched
        .iter()
        .any(|frame| frame.iter().any(|&value| value != batched[0][0])));
}
