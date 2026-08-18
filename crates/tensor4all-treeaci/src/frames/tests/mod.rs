use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::InputFrameStore;
use crate::{
    problem::prepare_problem, samples::ComponentSample, samples::SampleArena, TreeAciOptions,
    TreeAciScalar,
};

fn two_node_tree<T: TreeAciScalar + From<f64>>() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![s0, bond.clone()],
        [1.0, 2.0, 10.0, 20.0].into_iter().map(T::from).collect(),
    )
    .unwrap();
    let right = IdxTensor::from_dense(
        vec![bond, s1],
        [3.0, 4.0, 30.0, 40.0].into_iter().map(T::from).collect(),
    )
    .unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn assert_two_node_frames<T: TreeAciScalar + From<f64> + PartialEq + std::fmt::Debug>() {
    let input = two_node_tree::<T>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1]]).unwrap();
    let frames = InputFrameStore::<T>::from_samples(&[input], &problem, &arena).unwrap();

    assert_eq!(
        frames.frame_values(0, 0, 0).unwrap(),
        vec![T::from(1.0), T::from(10.0)]
    );
    assert_eq!(
        frames.frame_values(0, 0, 1).unwrap(),
        vec![T::from(2.0), T::from(20.0)]
    );
    assert_eq!(
        frames.frame_values(0, 1, 0).unwrap(),
        vec![T::from(3.0), T::from(4.0)]
    );
    assert_eq!(
        frames.frame_values(0, 1, 1).unwrap(),
        vec![T::from(30.0), T::from(40.0)]
    );
}

#[test]
fn two_node_frames_are_exact_for_real_and_complex_inputs() {
    assert_two_node_frames::<f64>();
    assert_two_node_frames::<Complex64>();
}

fn y_tree<T: TreeAciScalar + From<f64>>() -> TreeTN<IdxTensor, usize> {
    let sites = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let bonds = (0..3).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let center = IdxTensor::from_dense(bonds.clone(), vec![T::from(1.0); 8]).unwrap();
    let mut tensors = vec![center];
    for leaf in 0..3 {
        tensors.push(
            IdxTensor::from_dense(
                vec![sites[leaf].clone(), bonds[leaf].clone()],
                [1.0, 3.0, 2.0, 4.0].into_iter().map(T::from).collect(),
            )
            .unwrap(),
        );
    }
    TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap()
}

fn assert_y_frames<T: TreeAciScalar + From<f64>>() {
    let input = y_tree::<T>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let seeds = [vec![0, 0, 0, 0], vec![0, 1, 1, 1]];
    let (arena, active) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<T>::from_samples(&[input], &problem, &arena).unwrap();

    for forward in (0..problem.directed_edges.len()).step_by(2) {
        let reverse = problem.directed_edges[forward].reverse;
        for (seed, expected) in [T::from(27.0), T::from(343.0)].into_iter().enumerate() {
            let left = frames
                .frame_values(0, forward, active.ids[forward][seed])
                .unwrap();
            let right = frames
                .frame_values(0, reverse, active.ids[reverse][seed])
                .unwrap();
            let contracted = left
                .into_iter()
                .zip(right)
                .fold(T::default(), |sum, (lhs, rhs)| sum + lhs * rhs);
            assert!((tensor4all_core::Scalar::abs_val(contracted - expected)) < 1.0e-12);
        }
    }
}

#[test]
fn y_frames_glue_to_the_exact_global_value_for_real_and_complex_inputs() {
    assert_y_frames::<f64>();
    assert_y_frames::<Complex64>();
}

#[test]
fn multiple_physical_axes_use_first_axis_fast_flattening() {
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![a, b, bond.clone()],
        (0..12).map(|value| value as f64).collect(),
    )
    .unwrap();
    let right = IdxTensor::from_dense(vec![bond, DynIndex::new_dyn(1)], vec![1.0, 1.0]).unwrap();
    let input = TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&problem, &[vec![4, 0]]).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&[input], &problem, &arena).unwrap();

    assert_eq!(frames.frame_values(0, 0, 0).unwrap(), vec![4.0, 10.0]);
}

#[test]
fn frames_remain_addressable_after_active_set_replacement() {
    let input = two_node_tree::<f64>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, mut active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0], vec![1, 1]]).unwrap();
    let old_id = active.ids[0][0];
    active.ids[0] = vec![active.ids[0][1]];
    active.generation += 1;
    let frames = InputFrameStore::<f64>::from_samples(&[input], &problem, &arena).unwrap();

    assert_eq!(frames.frame_values(0, 0, old_id).unwrap(), vec![1.0, 10.0]);
}

/// `extend` must reuse every already-computed frame: growing the arena and
/// calling `extend` must produce exactly the same frame values, for every
/// sample (old and new), as discarding the store and rebuilding from scratch
/// on the grown arena.
#[test]
fn extend_matches_a_full_rebuild_on_the_grown_arena() {
    let input = y_tree::<f64>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let seed0 = vec![0, 0, 0, 0];
    let seed1 = vec![0, 1, 1, 1];
    let (mut arena, mut candidates) =
        SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");

    arena
        .inject_global_point(&mut candidates, &problem, &seed1)
        .expect("grow the arena with a second global point");

    let extended = initial
        .extend(std::slice::from_ref(&input), &problem, &arena)
        .expect("extend the store to the grown arena");
    let rebuilt =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("rebuild from scratch on the grown arena");

    for edge in 0..problem.directed_edges.len() {
        let sample_count = arena.directed_record_count(edge).unwrap();
        for sample in 0..sample_count {
            assert_eq!(
                extended.frame_values(0, edge, sample).unwrap(),
                rebuilt.frame_values(0, edge, sample).unwrap(),
                "edge {edge} sample {sample} disagrees between extend and full rebuild"
            );
        }
    }
}

/// `extend` must not repeat work: computing frames for samples the store
/// already covers is the exact bug `commit_edge_proposal` had (see
/// `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`). Growing
/// the arena and calling `extend` must issue strictly fewer
/// `contract_prepared_core` calls than discarding the store and rebuilding
/// everything from scratch on the same grown arena.
#[test]
fn extend_recomputes_only_the_newly_interned_samples() {
    let input = y_tree::<f64>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let seed0 = vec![0, 0, 0, 0];
    let seed1 = vec![0, 1, 1, 1];
    let (mut arena, mut candidates) =
        SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    arena
        .inject_global_point(&mut candidates, &problem, &seed1)
        .expect("grow the arena with a second global point");

    super::debug_stats::reset();
    let _extended = initial
        .extend(std::slice::from_ref(&input), &problem, &arena)
        .expect("extend the store to the grown arena");
    let extend_calls = super::debug_stats::compute_calls();

    super::debug_stats::reset();
    let _rebuilt =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("rebuild from scratch on the grown arena");
    let rebuild_calls = super::debug_stats::compute_calls();

    assert!(
        extend_calls < rebuild_calls,
        "extend should recompute only the new samples: extend={extend_calls} rebuild={rebuild_calls}"
    );
    assert!(extend_calls > 0, "the new seed must still be computed");
}

/// `candidate_frame` must cache: an identical candidate looked up twice
/// should compute `contract_prepared_core` once and return the same,
/// correct value both times.
#[test]
fn candidate_frame_hits_the_cache_on_a_repeated_lookup() {
    let input = two_node_tree::<f64>();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&problem, &[vec![0, 0]]).unwrap();
    let frames =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .unwrap();
    let candidate = ComponentSample {
        local_coordinate: 0,
        incoming: vec![],
    };

    super::candidate_debug_stats::reset();
    let first = frames
        .candidate_frame(std::slice::from_ref(&input), &problem, 0, 0, &candidate)
        .unwrap();
    let second = frames
        .candidate_frame(std::slice::from_ref(&input), &problem, 0, 0, &candidate)
        .unwrap();

    assert_eq!(first, vec![1.0, 10.0]);
    assert_eq!(second, first);
    assert_eq!(
        super::candidate_debug_stats::misses(),
        1,
        "the first lookup must compute"
    );
    assert_eq!(
        super::candidate_debug_stats::hits(),
        1,
        "the second, identical lookup must hit the cache"
    );
}

/// When the shared frame budget has no headroom left for the candidate
/// cache, `candidate_frame` must keep returning the correct value -- just
/// without caching it -- rather than erroring or corrupting the result.
#[test]
fn candidate_frame_stays_correct_when_the_shared_budget_has_no_headroom_for_caching() {
    let input = two_node_tree::<f64>();
    let seeds = [vec![0, 0], vec![1, 1]];
    // Exactly the persistent frame cache's own cost (see
    // `the_frame_cache_is_bounded_in_aggregate`: 64 bytes for this fixture),
    // leaving zero headroom for the candidate cache.
    let tight = prepare_problem(
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_frame_bytes: 64,
            ..TreeAciOptions::default()
        },
    )
    .unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&tight, &seeds).unwrap();
    let frames =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &tight, &arena).unwrap();
    let candidate = ComponentSample {
        local_coordinate: 0,
        incoming: vec![],
    };

    super::candidate_debug_stats::reset();
    let first = frames
        .candidate_frame(std::slice::from_ref(&input), &tight, 0, 0, &candidate)
        .unwrap();
    let second = frames
        .candidate_frame(std::slice::from_ref(&input), &tight, 0, 0, &candidate)
        .unwrap();

    assert_eq!(first, vec![1.0, 10.0]);
    assert_eq!(second, first);
    assert_eq!(
        super::candidate_debug_stats::misses(),
        2,
        "with no budget headroom, every lookup must recompute rather than corrupt or error"
    );
    assert_eq!(super::candidate_debug_stats::hits(), 0);
}

/// The frame cache is bounded in aggregate, not only per frame.
///
/// `max_frame_elements` bounds one directed frame, but the cache retains one
/// per input per directed edge, so the retained total used to grow as
/// `inputs * directed_edges * max_frame_elements` with no ceiling a caller
/// could see or set. `max_frame_bytes` bounds the whole cache, and the check
/// runs before each allocation rather than after the fact.
#[test]
fn the_frame_cache_is_bounded_in_aggregate() {
    let input = two_node_tree::<f64>();
    let seeds = [vec![0, 0], vec![1, 1]];

    let generous =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&generous, &seeds).unwrap();
    let frames =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &generous, &arena)
            .unwrap();

    // One edge means two directed frames, each two samples by a bond of two:
    // 2 * (2 * 2) * 8 = 64 bytes.
    assert_eq!(frames.records(), 2);
    assert_eq!(frames.retained_bytes(), 64);

    let tight = prepare_problem(
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_frame_bytes: 63,
            ..TreeAciOptions::default()
        },
    )
    .unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&tight, &seeds).unwrap();
    let error = InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &tight, &arena)
        .expect_err("a 63-byte ceiling must refuse a 64-byte frame cache");
    assert!(
        matches!(
            error,
            crate::TreeAciError::ResourceLimit {
                resource: "frame bytes",
                limit: 63,
                ..
            }
        ),
        "unexpected error: {error}"
    );

    // The two bounds are independent: a per-frame ceiling sized exactly to one
    // frame still admits every frame, which is why it cannot stand in for the
    // aggregate.
    let per_frame_only = prepare_problem(
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_frame_elements: 4,
            ..TreeAciOptions::default()
        },
    )
    .unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&per_frame_only, &seeds).unwrap();
    assert!(InputFrameStore::<f64>::from_samples(
        std::slice::from_ref(&input),
        &per_frame_only,
        &arena
    )
    .is_ok());
}
