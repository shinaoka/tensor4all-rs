use std::cell::Cell;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::{enumerate_candidates, materialize_and_factor_edge, select_pivot_samples};
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
fn pivot_indices_are_checked_before_selecting_samples() {
    let candidate = crate::samples::ComponentSample {
        local_coordinate: 1,
        incoming: Vec::new(),
    };
    assert_eq!(
        select_pivot_samples(vec![0], std::slice::from_ref(&candidate)).unwrap(),
        vec![candidate]
    );
    assert!(matches!(
        select_pivot_samples(vec![1], &[]),
        Err(TreeAciError::InternalInvariant { .. })
    ));
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
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
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
        let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
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
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let mut failing = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "sentinel".into(),
        })
    };
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs, &problem, &active, &frames, 0, &options, true, &mut failing,
        ),
        Err(TreeAciError::Callback { message }) if message == "sentinel"
    ));

    // Ceilings are resolved once, at preparation, so the ceiling this update
    // enforces lives on the prepared problem. A whole-run configuration cannot
    // reach this update with a ceiling below four elements -- preparation
    // refuses the same two-by-two minimum first, which is the point of that
    // earlier check -- so the boundary is pinned on the prepared problem the
    // way the working-byte boundaries below and in `frames` are.
    let limited = TreeAciOptions {
        max_local_matrix_elements: Some(4),
        ..TreeAciOptions::default()
    };
    let mut limited_problem = prepare_problem::<f64, _>(&inputs, &limited).unwrap();
    assert_eq!(limited_problem.max_local_matrix_elements, 4);
    limited_problem.max_local_matrix_elements = 3;
    let mut unused = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| Ok(());
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs,
            &limited_problem,
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

    // For this 2x2 two-node case the exact live element contract is:
    // input values 4 + operator output 4 + two packed candidate sides
    // (2 + 2) * bond 2 * coexistence factor 2 = 24 elements = 192 bytes.
    let working_limited = TreeAciOptions {
        max_working_bytes: 191,
        ..TreeAciOptions::default()
    };
    let working_limited_problem = prepare_problem::<f64, _>(&inputs, &working_limited).unwrap();
    let mut working_unused =
        |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| Ok(());
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs,
            &working_limited_problem,
            &active,
            &frames,
            0,
            &working_limited,
            true,
            &mut working_unused,
        ),
        Err(TreeAciError::ResourceLimit {
            resource: "working bytes",
            requested: 192,
            limit: 191
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

fn local_update_measurement_chain(n_sites: usize, bond_dim: usize) -> TreeTN<IdxTensor, usize> {
    assert!(n_sites >= 3);
    let sites = (0..n_sites)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(n_sites);
    tensors.push(
        IdxTensor::from_dense(
            vec![sites[0].clone(), bonds[0].clone()],
            (0..2 * bond_dim)
                .map(|value| 0.25 + (value % 13) as f64 / 13.0)
                .collect(),
        )
        .unwrap(),
    );
    for site in 1..n_sites - 1 {
        tensors.push(
            IdxTensor::from_dense(
                vec![
                    bonds[site - 1].clone(),
                    sites[site].clone(),
                    bonds[site].clone(),
                ],
                (0..bond_dim * 2 * bond_dim)
                    .map(|value| 0.25 + (value % 17) as f64 / 17.0)
                    .collect(),
            )
            .unwrap(),
        );
    }
    tensors.push(
        IdxTensor::from_dense(
            vec![bonds[n_sites - 2].clone(), sites[n_sites - 1].clone()],
            (0..bond_dim * 2)
                .map(|value| 0.25 + (value % 19) as f64 / 19.0)
                .collect(),
        )
        .unwrap(),
    );
    TreeTN::from_tensors(tensors, (0..n_sites).collect()).unwrap()
}

fn local_update_measurement_branch(bond_dim: usize) -> TreeTN<IdxTensor, usize> {
    let center_bonds = (0..3)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let center = IdxTensor::from_dense(
        center_bonds.clone(),
        (0..bond_dim * bond_dim * bond_dim)
            .map(|value| 0.25 + (value % 23) as f64 / 23.0)
            .collect(),
    )
    .unwrap();
    let mut tensors = vec![center];
    for bond in center_bonds {
        let site = DynIndex::new_dyn(2);
        tensors.push(
            IdxTensor::from_dense(
                vec![bond, site],
                (0..bond_dim * 2)
                    .map(|value| 0.25 + (value % 29) as f64 / 29.0)
                    .collect(),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn run_local_update_measurement(
    input: &TreeTN<IdxTensor, usize>,
    legacy_frame_pack: bool,
    samples: &[Vec<usize>],
    repetitions: usize,
) -> (
    std::time::Duration,
    Vec<f64>,
    crate::state::profile_debug_stats::Snapshot,
) {
    let inputs = vec![input.clone()];
    let options = crate::TreeAciOptions::default();
    let problem = crate::problem::prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, samples).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let forward = problem
        .directed_edges
        .iter()
        .position(|edge| edge.from == 1 && edge.to == 2)
        .or_else(|| {
            problem
                .directed_edges
                .iter()
                .position(|edge| edge.from == 0 && edge.to == 1)
        })
        .unwrap();

    if legacy_frame_pack {
        std::env::set_var("T4A_TREEACI_USE_LEGACY_LOCAL_FRAME_PACK", "1");
    } else {
        std::env::remove_var("T4A_TREEACI_USE_LEGACY_LOCAL_FRAME_PACK");
    }
    // Make the A/B isolate the packed-batch boundary. Both sides consume the
    // dense matrices through the same owned GEMM path.
    std::env::set_var("T4A_TREEACI_USE_OWNED_LOCAL_MATMUL", "1");
    crate::state::profile_debug_stats::reset();
    let started = std::time::Instant::now();
    let mut checksum = Vec::new();
    for _ in 0..repetitions {
        let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = batch.get(0, point)?;
            }
            Ok(())
        };
        let update = materialize_and_factor_edge(
            &inputs,
            &problem,
            &active,
            &frames,
            forward,
            &options,
            true,
            &mut operator,
        )
        .unwrap();
        checksum.extend(update.local_values);
    }
    (
        started.elapsed(),
        checksum,
        crate::state::profile_debug_stats::snapshot(),
    )
}

/// [AI Supplied] Paired release measurement for #714's removed nested frame
/// extraction. It is intentionally ignored: the complete crate matrix must be
/// green before this diagnostic is admitted as an efficiency gate.
#[test]
#[ignore]
fn packed_local_update_release_measurement_for_chain_and_branch() {
    let chain_seeds = (0..16)
        .map(|seed| (0..8).map(|site| (seed >> site) & 1).collect())
        .collect::<Vec<Vec<_>>>();
    let branch_seeds = (0..16)
        .map(|seed| vec![0, seed & 1, (seed >> 1) & 1, (seed >> 2) & 1])
        .collect::<Vec<Vec<_>>>();
    let cases = [
        (
            "chain-8x16",
            local_update_measurement_chain(8, 16),
            chain_seeds,
        ),
        (
            "branch-chi32",
            local_update_measurement_branch(32),
            branch_seeds,
        ),
    ];
    let repetitions = 16;
    let samples = 5;
    for (name, input, seeds) in cases {
        let mut legacy_times = Vec::with_capacity(samples);
        let mut packed_times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let (legacy_elapsed, legacy_values, legacy_profile) =
                run_local_update_measurement(&input, true, &seeds, repetitions);
            let (packed_elapsed, packed_values, packed_profile) =
                run_local_update_measurement(&input, false, &seeds, repetitions);
            assert_eq!(packed_values, legacy_values);
            legacy_times.push(legacy_elapsed.as_secs_f64() * 1.0e3);
            packed_times.push(packed_elapsed.as_secs_f64() * 1.0e3);
            if legacy_times.len() == samples {
                eprintln!(
                    "#714 packed counters: case={name}, legacy_vectors={}, legacy_values={}, packed_batches={}, packed_values={}",
                    legacy_profile.local_legacy_frame_vectors,
                    legacy_profile.local_legacy_frame_values,
                    packed_profile.local_packed_frame_batches,
                    packed_profile.local_packed_frame_values,
                );
            }
        }
        legacy_times.sort_by(f64::total_cmp);
        packed_times.sort_by(f64::total_cmp);
        let legacy_median = legacy_times[samples / 2];
        let packed_median = packed_times[samples / 2];
        eprintln!(
            "#714 paired release measurement: case={name}, repetitions={repetitions}, samples={samples}, legacy_median_ms={legacy_median:.3}, packed_median_ms={packed_median:.3}, reduction_pct={:.1}, legacy_all_ms={legacy_times:?}, packed_all_ms={packed_times:?}",
            (legacy_median - packed_median) / legacy_median * 100.0,
        );
    }
    std::env::remove_var("T4A_TREEACI_USE_LEGACY_LOCAL_FRAME_PACK");
    std::env::remove_var("T4A_TREEACI_USE_OWNED_LOCAL_MATMUL");
}

#[test]
fn candidate_frames_batched_path_matches_scalar_path_on_a_chain() {
    let inputs = vec![three_node_chain_for_batched_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();

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
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &row_candidates, 0)
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
