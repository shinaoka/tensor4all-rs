use std::rc::Rc;

use num_complex::Complex64;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::InputFrameStore;
use crate::{
    problem::prepare_problem, problem::PreparedTreeProblem, samples::ComponentSample,
    samples::SampleArena, TreeAciOptions, TreeAciScalar,
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

#[test]
fn extend_new_samples_matches_full_rebuild_on_chain_and_branch() {
    let cases = [
        (
            chain_tree_for_batched_compute(),
            vec![0, 0, 0],
            vec![1, 0, 0],
        ),
        (y_tree::<f64>(), vec![0, 0, 0, 0], vec![0, 1, 1, 1]),
    ];

    for (input, seed0, seed1) in cases {
        let problem =
            prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
        let (mut arena, mut candidates) =
            SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
        let initial =
            InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
                .expect("initial store");
        let previous_counts = initial.frames[0]
            .iter()
            .map(|frame| frame.sample_count)
            .collect::<Vec<_>>();

        arena
            .inject_global_point(&mut candidates, &problem, &seed1)
            .expect("grow the arena");
        let extended = initial
            .extend_new_samples(
                std::slice::from_ref(&input),
                &problem,
                &arena,
                std::slice::from_ref(&previous_counts),
            )
            .expect("extend only new samples");
        let rebuilt =
            InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
                .expect("rebuild from scratch");

        for edge in 0..problem.directed_edges.len() {
            let sample_count = arena.directed_record_count(edge).unwrap();
            for sample in 0..sample_count {
                assert_eq!(
                    extended.frame_values(0, edge, sample).unwrap(),
                    rebuilt.frame_values(0, edge, sample).unwrap(),
                    "edge {edge} sample {sample} disagrees between cut-local extension and rebuild"
                );
            }
        }

        let unchanged = (0..problem.directed_edges.len())
            .filter(|&edge| arena.directed_record_count(edge).unwrap() == previous_counts[edge]);
        for edge in unchanged {
            assert!(
                Rc::ptr_eq(&initial.frames[0][edge], &extended.frames[0][edge]),
                "unchanged edge {edge} must retain its frame allocation"
            );
        }
    }
}

#[test]
fn extend_new_samples_computes_only_new_ranges() {
    let input = chain5_tree_for_dedup_regression();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (mut arena, _candidates) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0, 0]]).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    let previous_counts = initial.frames[0]
        .iter()
        .map(|frame| frame.sample_count)
        .collect::<Vec<_>>();
    let grown_edge = problem
        .directed_edges
        .iter()
        .position(|edge| edge.from == 1 && edge.to == 0)
        .expect("chain must have the selected directed edge");
    arena
        .project_point_onto_edge(&problem, grown_edge, &[0, 0, 0, 0, 1])
        .expect("project a new point onto one cut");

    super::debug_stats::reset();
    crate::state::profile_debug_stats::reset();
    let extended = initial
        .extend_new_samples(
            std::slice::from_ref(&input),
            &problem,
            &arena,
            std::slice::from_ref(&previous_counts),
        )
        .expect("extend only new samples");
    let expected_new_values = (0..problem.directed_edges.len())
        .map(|edge| {
            let current = arena.directed_record_count(edge).unwrap();
            (current - previous_counts[edge]) * initial.frames[0][edge].bond_dim
        })
        .sum::<usize>();

    assert!(
        super::debug_stats::compute_calls() > 0,
        "the new ranges must still be contracted"
    );
    assert_eq!(
        super::debug_stats::old_values_copied(),
        0,
        "cut-local extension must not copy old frame rows into a new allocation"
    );
    assert_eq!(super::debug_stats::new_values_copied(), expected_new_values);
    assert!(
        extended.frames[0][grown_edge]
            .base
            .as_ref()
            .is_some_and(|base| { Rc::ptr_eq(base, &initial.frames[0][grown_edge]) }),
        "grown frame must retain the old prefix by Rc"
    );
}

#[test]
#[ignore = "paired release measurement; correctness is covered by the non-ignored full matrix"]
fn paired_release_measurement_for_cut_local_extension() {
    use std::time::{Duration, Instant};

    let input = chain5_tree_for_dedup_regression();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (mut arena, _candidates) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0, 0]]).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    let previous_counts = initial.frames[0]
        .iter()
        .map(|frame| frame.sample_count)
        .collect::<Vec<_>>();
    let grown_edge = problem
        .directed_edges
        .iter()
        .position(|edge| edge.from == 1 && edge.to == 0)
        .expect("chain must have the selected directed edge");
    arena
        .project_point_onto_edge(&problem, grown_edge, &[0, 0, 0, 0, 1])
        .expect("project a new point onto one cut");

    for _ in 0..8 {
        let extended = initial
            .extend_new_samples(
                std::slice::from_ref(&input),
                &problem,
                &arena,
                std::slice::from_ref(&previous_counts),
            )
            .expect("cut-local extension");
        let rebuilt =
            InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
                .expect("full rebuild");
        assert_eq!(
            extended.frame_values(0, grown_edge, 1).unwrap(),
            rebuilt.frame_values(0, grown_edge, 1).unwrap()
        );
    }

    super::debug_stats::reset();
    crate::state::profile_debug_stats::reset();
    let extended = initial
        .extend_new_samples(
            std::slice::from_ref(&input),
            &problem,
            &arena,
            std::slice::from_ref(&previous_counts),
        )
        .expect("cut-local extension");
    let candidate_compute_calls = super::debug_stats::compute_calls();
    let candidate_profile = crate::state::profile_debug_stats::snapshot();
    let extended_bytes = extended.retained_bytes();

    super::debug_stats::reset();
    let rebuilt =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("full rebuild");
    let full_rebuild_compute_calls = super::debug_stats::compute_calls();
    assert_eq!(
        extended.frame_values(0, grown_edge, 1).unwrap(),
        rebuilt.frame_values(0, grown_edge, 1).unwrap()
    );
    eprintln!(
        "#715 paired release resources: chain_nodes={}, max_degree=2, directed_edges={}, candidate_compute_calls={candidate_compute_calls}, full_rebuild_compute_calls={full_rebuild_compute_calls}, old_values_copied={}, new_values_copied={}, extension_calls={}, reused_edges={}, grown_edges={}, retained_bytes={extended_bytes}",
        problem.node_order.len(),
        problem.directed_edges.len(),
        candidate_profile.frame_extension_old_values_copied,
        candidate_profile.frame_extension_new_values_copied,
        candidate_profile.frame_extension_calls,
        candidate_profile.frame_extension_reused_edges,
        candidate_profile.frame_extension_grown_edges,
    );

    const REPETITIONS: usize = 128;
    const SAMPLES: usize = 7;
    let mut cut_local = Vec::with_capacity(SAMPLES);
    let mut full_rebuild = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let started = Instant::now();
        for _ in 0..REPETITIONS {
            let extended = initial
                .extend_new_samples(
                    std::slice::from_ref(&input),
                    &problem,
                    &arena,
                    std::slice::from_ref(&previous_counts),
                )
                .expect("cut-local extension");
            std::hint::black_box(extended.records());
        }
        cut_local.push(started.elapsed());

        let started = Instant::now();
        for _ in 0..REPETITIONS {
            let rebuilt = InputFrameStore::<f64>::from_samples(
                std::slice::from_ref(&input),
                &problem,
                &arena,
            )
            .expect("full rebuild");
            std::hint::black_box(rebuilt.records());
        }
        full_rebuild.push(started.elapsed());
    }
    cut_local.sort_unstable();
    full_rebuild.sort_unstable();
    let median = |samples: &[Duration]| samples[samples.len() / 2];
    eprintln!(
        "#715 paired release measurement: repetitions={REPETITIONS}, samples={SAMPLES}, cut_local_median={:?}, full_rebuild_median={:?}, cut_local_all={cut_local:?}, full_rebuild_all={full_rebuild:?}",
        median(&cut_local),
        median(&full_rebuild),
    );
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

/// The prepared `cores` for one input never change across a run -- they are
/// derived once from the fixed input operand -- so `extend`'s resulting
/// store must share the SAME underlying `Rc` allocation as the initial
/// store's, not a fresh clone of it. A regression back to `.clone()`ing
/// `cores` on every `build_or_extend` call would still pass
/// `extend_matches_a_full_rebuild_on_the_grown_arena` above (values would
/// still be equal), so that test alone cannot catch it; this test checks
/// allocation identity instead of value equality.
#[test]
fn extend_reuses_the_same_cores_allocation_instead_of_cloning_it() {
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

    assert!(
        Rc::ptr_eq(&initial.cores[0], &extended.cores[0]),
        "extend must share the initial store's prepared cores allocation via Rc, not clone it"
    );
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

/// Growing mandatory directed frames must reclaim an older optional
/// candidate cache before publishing a store over `max_frame_bytes`.
#[test]
fn extend_reclaims_candidate_cache_when_base_frames_consume_its_budget() {
    let input = two_node_tree::<f64>();
    let candidate_entry_bytes =
        std::mem::size_of::<super::CandidateCacheKey>() + 2 * std::mem::size_of::<f64>();
    let initial_frame_bytes = 32;
    let frame_budget = initial_frame_bytes + candidate_entry_bytes;
    let problem = prepare_problem(
        std::slice::from_ref(&input),
        &TreeAciOptions {
            max_frame_bytes: frame_budget,
            ..TreeAciOptions::default()
        },
    )
    .unwrap();
    let seed0 = vec![0, 0];
    let seed1 = vec![1, 1];
    let (mut arena, mut candidates) =
        SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
    let frames =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .unwrap();
    let candidate = ComponentSample {
        local_coordinate: 0,
        incoming: vec![],
    };

    frames
        .candidate_frame(std::slice::from_ref(&input), &problem, 0, 0, &candidate)
        .unwrap();
    assert_eq!(frames.retained_bytes(), frame_budget);

    arena
        .inject_global_point(&mut candidates, &problem, &seed1)
        .unwrap();
    let extended = frames
        .extend(std::slice::from_ref(&input), &problem, &arena)
        .unwrap();

    assert_eq!(extended.retained_bytes(), 64);
    assert_eq!(extended.candidate_cache_bytes.get(), 0);
    assert!(extended.candidate_cache.borrow().is_empty());

    super::candidate_debug_stats::reset();
    extended
        .candidate_frame(std::slice::from_ref(&input), &problem, 0, 0, &candidate)
        .unwrap();
    assert_eq!(super::candidate_debug_stats::misses(), 1);
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

#[test]
fn batched_single_incoming_contraction_matches_scalar_path() {
    // A 2 (physical) x 3 (incoming) x 4 (outgoing) core, values chosen so the
    // contraction result is easy to hand-check.
    let dims = vec![2usize, 3, 4];
    let mut strides = Vec::with_capacity(3);
    let mut stride = 1usize;
    for dim in &dims {
        strides.push(stride);
        stride *= dim;
    }
    let values: Vec<f64> = (0..24).map(|value| value as f64).collect();
    let core = super::PreparedCore {
        indices: Vec::new(), // unused by the functions under test
        dims,
        strides,
        values,
    };
    let physical_axis = 0usize;
    let incoming_axis = 1usize;
    let outgoing_axis = 2usize;
    let local_coordinate = 1usize; // fixes the physical axis to value 1
    let base_offset = local_coordinate * core.strides[physical_axis];

    let core_matrix = super::single_incoming_core_matrix(
        &core,
        outgoing_axis,
        incoming_axis,
        base_offset,
        core.dims[outgoing_axis],
        core.dims[incoming_axis],
    );

    // Two candidate incoming frame vectors, as columns.
    let frame_matrix = tensor4all_tensorbackend::Matrix::from_col_major_vec(
        core.dims[incoming_axis],
        2,
        vec![
            1.0, 0.0, 0.0, // candidate 0: e_0
            0.0, 1.0, 0.0, // candidate 1: e_1
        ],
    );

    let result = super::contract_prepared_core_batched(&core_matrix, &frame_matrix).unwrap();

    // candidate 0 selects incoming index 0: result column 0 must equal the
    // core's [physical=1, incoming=0, outgoing=*] slice exactly (e_0 dot
    // product with a matrix picks out its column verbatim).
    for outgoing_value in 0..core.dims[outgoing_axis] {
        let offset = base_offset + outgoing_value * core.strides[outgoing_axis];
        assert_eq!(result[[outgoing_value, 0]], core.values[offset]);
    }
    // candidate 1 selects incoming index 1.
    for outgoing_value in 0..core.dims[outgoing_axis] {
        let offset = base_offset
            + core.strides[incoming_axis]
            + outgoing_value * core.strides[outgoing_axis];
        assert_eq!(result[[outgoing_value, 1]], core.values[offset]);
    }

    // Genuine cross-check against the scalar path: `accumulate_incoming` is
    // exactly the per-candidate scalar accumulator `contract_prepared_core`
    // calls once per outgoing value; comparing the batched result column
    // against it (not just the hand-computed slice above) is what actually
    // catches a batched/scalar mismatch the hand-picked one-hot vectors
    // above might miss.
    let candidate_frames: [Vec<f64>; 2] = [vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
    for (candidate, frame) in candidate_frames.iter().enumerate() {
        for outgoing_value in 0..core.dims[outgoing_axis] {
            let outgoing_offset = base_offset + outgoing_value * core.strides[outgoing_axis];
            let expected =
                super::accumulate_incoming(&core, &[(incoming_axis, frame)], 0, outgoing_offset);
            assert_eq!(result[[outgoing_value, candidate]], expected);
        }
    }
}

#[test]
fn batched_path_matches_scalar_path_on_random_core() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    let outgoing_dim = 5usize;
    let incoming_dim = 7usize;
    let physical_dim = 2usize;
    let dims = vec![physical_dim, incoming_dim, outgoing_dim];
    let mut strides = Vec::with_capacity(3);
    let mut stride = 1usize;
    for dim in &dims {
        strides.push(stride);
        stride *= dim;
    }
    let values: Vec<f64> = (0..(physical_dim * incoming_dim * outgoing_dim))
        .map(|_| rng.random_range(-1.0..1.0))
        .collect();
    let core = super::PreparedCore {
        indices: Vec::new(),
        dims,
        strides,
        values,
    };

    for local_coordinate in 0..physical_dim {
        let base_offset = local_coordinate * core.strides[0];
        let core_matrix = super::single_incoming_core_matrix(
            &core,
            2,
            1,
            base_offset,
            outgoing_dim,
            incoming_dim,
        );

        let n_candidates = 4;
        let mut frame_data = Vec::with_capacity(incoming_dim * n_candidates);
        let mut per_candidate_frames = Vec::with_capacity(n_candidates);
        for _ in 0..n_candidates {
            let frame: Vec<f64> = (0..incoming_dim)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            frame_data.extend_from_slice(&frame);
            per_candidate_frames.push(frame);
        }
        let frame_matrix = tensor4all_tensorbackend::Matrix::from_col_major_vec(
            incoming_dim,
            n_candidates,
            frame_data,
        );
        let batched = super::contract_prepared_core_batched(&core_matrix, &frame_matrix).unwrap();

        for (candidate, frame) in per_candidate_frames.iter().enumerate() {
            for outgoing_value in 0..outgoing_dim {
                let outgoing_offset = base_offset + outgoing_value * core.strides[2];
                let mut expected = 0.0f64;
                for (incoming_value, &weight) in frame.iter().enumerate() {
                    expected +=
                        weight * core.values[outgoing_offset + incoming_value * core.strides[1]];
                }
                let actual = batched[[outgoing_value, candidate]];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "mismatch at outgoing={outgoing_value}, candidate={candidate}: \
                     actual={actual}, expected={expected}"
                );

                // Genuine cross-check against the scalar path itself
                // (`accumulate_incoming`, the exact accumulator
                // `contract_prepared_core` calls once per outgoing value),
                // not just the hand-rolled dot product above -- this is what
                // would actually catch a mistake shared between this test's
                // hand-computed `expected` and the production scalar path.
                let scalar_path_expected =
                    super::accumulate_incoming(&core, &[(1, frame)], 0, outgoing_offset);
                assert!(
                    (actual - scalar_path_expected).abs() < 1e-10,
                    "batched result disagrees with the scalar accumulate_incoming path at \
                     outgoing={outgoing_value}, candidate={candidate}: \
                     actual={actual}, scalar_path_expected={scalar_path_expected}"
                );
            }
        }
    }
}

fn assert_all_physical_single_incoming_matrix<T>()
where
    T: TreeAciScalar + From<f64> + PartialEq + std::fmt::Debug,
{
    // Core axes are [incoming, physical_1, outgoing, physical_0], while the
    // local physical flattening order is [physical_0, physical_1]. This makes
    // the test sensitive to both the physical-axis map and its strides.
    let core_dims = vec![2usize, 3, 5, 4];
    let mut core_strides = Vec::with_capacity(core_dims.len());
    let mut stride = 1usize;
    for &dimension in &core_dims {
        core_strides.push(stride);
        stride *= dimension;
    }
    let core = super::PreparedCore {
        indices: Vec::new(),
        dims: core_dims,
        strides: core_strides,
        values: (0..stride)
            .map(|value| T::from(value as f64 + 0.25))
            .collect(),
    };
    let physical = super::LocalPhysicalPlan {
        indices: Vec::new(),
        dims: vec![4, 3],
        strides: vec![1, 4],
        local_dim: 12,
    };
    let matrix =
        super::single_incoming_all_physical_core_matrix(&core, 2, 0, &physical, &[3, 1], 5, 2);

    for incoming in 0..2 {
        for local_coordinate in 0..12 {
            let physical_0 = local_coordinate % 4;
            let physical_1 = (local_coordinate / 4) % 3;
            for outgoing in 0..5 {
                let offset = physical_0 * core.strides[3]
                    + physical_1 * core.strides[1]
                    + incoming * core.strides[0]
                    + outgoing * core.strides[2];
                assert_eq!(
                    matrix[[outgoing + 5 * local_coordinate, incoming]],
                    core.values[offset]
                );
            }
        }
    }
}

#[test]
fn all_physical_single_incoming_matrix_respects_nontrivial_axis_order() {
    assert_all_physical_single_incoming_matrix::<f64>();
    assert_all_physical_single_incoming_matrix::<Complex64>();
}

/// Builds a 4-node star `1 -- 0 -- 2`, `0 -- 3`, whose center (node `0`) has
/// three neighbors. Directed edge `1 -> 0` has zero incoming edges (node `1`
/// is a leaf) -- `InputFrameStore::candidate_frames_for_edge`'s scalar
/// fallback branch -- and directed edge `0 -> 1` has two incoming edges
/// (`2 -> 0` and `3 -> 0`, node `0`'s other two neighbors) -- its batched
/// two-incoming-edge branch. See
/// `candidate_frames_for_edge_falls_back_on_a_leaf_edge_with_zero_incoming_edges`
/// and
/// `candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges`.
fn star_tree_for_fallback_dispatch() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond02 = DynIndex::new_dyn(2);
    let bond03 = DynIndex::new_dyn(2);

    let node0 = IdxTensor::from_dense(
        vec![s0, bond01.clone(), bond02.clone(), bond03.clone()],
        (1..=16).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(vec![bond01, s1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let node2 = IdxTensor::from_dense(vec![bond02, s2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let node3 = IdxTensor::from_dense(vec![bond03, s3], vec![9.0, 10.0, 11.0, 12.0]).unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2, node3], vec![0, 1, 2, 3]).unwrap()
}

#[test]
fn two_incoming_core_matrix_batched_matches_scalar_contraction_on_every_pair() {
    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let cores = super::prepare_cores::<f64, usize>(&inputs[0], &problem).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);
    let incoming_edge_1 = directed.incoming_to_from[0];
    let incoming_edge_2 = directed.incoming_to_from[1];

    let node = *problem.node_positions.get(&directed.from).unwrap();
    let core = &cores[node];
    let outgoing = super::outgoing_bond(&inputs[0], &problem, edge).unwrap();
    let outgoing_axis = super::axis_of(&core.indices, outgoing).unwrap();
    let incoming_bond_1 = super::outgoing_bond(&inputs[0], &problem, incoming_edge_1).unwrap();
    let incoming_bond_2 = super::outgoing_bond(&inputs[0], &problem, incoming_edge_2).unwrap();
    let incoming_axis_1 = super::axis_of(&core.indices, incoming_bond_1).unwrap();
    let incoming_axis_2 = super::axis_of(&core.indices, incoming_bond_2).unwrap();
    let outgoing_dim = core.dims[outgoing_axis];
    let incoming_dim_1 = core.dims[incoming_axis_1];
    let incoming_dim_2 = core.dims[incoming_axis_2];

    // Two arbitrary, non-trivial frame vectors per incoming edge (n1=2, n2=2)
    // so the test exercises real cross-combination behavior, not a
    // degenerate 1-candidate case.
    let v1_cols = [vec![1.0, 0.5], vec![0.25, 2.0]];
    let v2_cols = [vec![3.0, -1.0], vec![0.1, 4.0]];
    let mut v1_data = Vec::new();
    for col in &v1_cols {
        v1_data.extend(col.iter().copied());
    }
    let v1 = tensor4all_tensorbackend::Matrix::from_col_major_vec(incoming_dim_1, 2, v1_data);
    let mut v2_data = Vec::new();
    for col in &v2_cols {
        v2_data.extend(col.iter().copied());
    }
    let v2 = tensor4all_tensorbackend::Matrix::from_col_major_vec(incoming_dim_2, 2, v2_data);

    let batched = super::two_incoming_core_matrix_batched(
        core,
        outgoing_axis,
        incoming_axis_1,
        incoming_axis_2,
        0,
        outgoing_dim,
        incoming_dim_1,
        incoming_dim_2,
        &v1,
        &v2,
    )
    .unwrap();

    for (n1, v1_vec) in v1_cols.iter().enumerate() {
        for (n2, v2_vec) in v2_cols.iter().enumerate() {
            let incoming_frames = vec![
                (incoming_edge_1, v1_vec.clone()),
                (incoming_edge_2, v2_vec.clone()),
            ];
            let expected = super::contract_prepared_core(
                &inputs[0],
                &problem,
                &cores,
                edge,
                0,
                &incoming_frames,
            )
            .unwrap();
            let actual: Vec<f64> = (0..outgoing_dim)
                .map(|out| batched[[out + outgoing_dim * n1, n2]])
                .collect();
            assert_eq!(actual, expected, "mismatch at (n1={n1}, n2={n2})");
        }
    }
}

#[test]
fn two_incoming_core_matrix_batched_matches_complex_reference_for_every_axis_order() {
    use num_complex::Complex64;
    use tensor4all_tensorbackend::Matrix;

    const PHYSICAL_DIM: usize = 2;
    const OUTGOING_DIM: usize = 3;
    const INCOMING_DIM_1: usize = 4;
    const INCOMING_DIM_2: usize = 5;
    const N1: usize = 2;
    const N2: usize = 3;

    let v1 = Matrix::from_col_major_vec(
        INCOMING_DIM_1,
        N1,
        (0..INCOMING_DIM_1 * N1)
            .map(|i| Complex64::new((i + 1) as f64 / 7.0, (i % 3) as f64 / 11.0))
            .collect(),
    );
    let v2 = Matrix::from_col_major_vec(
        INCOMING_DIM_2,
        N2,
        (0..INCOMING_DIM_2 * N2)
            .map(|i| Complex64::new((i + 2) as f64 / 13.0, -((i % 4) as f64) / 17.0))
            .collect(),
    );

    for physical_axis in 0..4 {
        for outgoing_axis in 0..4 {
            if outgoing_axis == physical_axis {
                continue;
            }
            for incoming_axis_1 in 0..4 {
                if incoming_axis_1 == physical_axis || incoming_axis_1 == outgoing_axis {
                    continue;
                }
                let incoming_axis_2 = (0..4)
                    .find(|axis| {
                        *axis != physical_axis && *axis != outgoing_axis && *axis != incoming_axis_1
                    })
                    .unwrap();
                let mut dims = vec![0; 4];
                dims[physical_axis] = PHYSICAL_DIM;
                dims[outgoing_axis] = OUTGOING_DIM;
                dims[incoming_axis_1] = INCOMING_DIM_1;
                dims[incoming_axis_2] = INCOMING_DIM_2;
                let mut strides = Vec::with_capacity(4);
                let mut stride = 1;
                for &dim in &dims {
                    strides.push(stride);
                    stride *= dim;
                }
                let core = super::PreparedCore {
                    indices: dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect(),
                    dims,
                    strides,
                    values: (0..stride)
                        .map(|i| Complex64::new((i + 1) as f64 / 19.0, (i % 7) as f64 / 23.0))
                        .collect(),
                };
                let physical_base_offset = core.strides[physical_axis];
                let actual = super::two_incoming_core_matrix_batched(
                    &core,
                    outgoing_axis,
                    incoming_axis_1,
                    incoming_axis_2,
                    physical_base_offset,
                    OUTGOING_DIM,
                    INCOMING_DIM_1,
                    INCOMING_DIM_2,
                    &v1,
                    &v2,
                )
                .unwrap();

                for candidate_2 in 0..N2 {
                    for candidate_1 in 0..N1 {
                        for outgoing in 0..OUTGOING_DIM {
                            let mut expected = Complex64::new(0.0, 0.0);
                            for incoming_2 in 0..INCOMING_DIM_2 {
                                for incoming_1 in 0..INCOMING_DIM_1 {
                                    let offset = physical_base_offset
                                        + outgoing * core.strides[outgoing_axis]
                                        + incoming_1 * core.strides[incoming_axis_1]
                                        + incoming_2 * core.strides[incoming_axis_2];
                                    expected += core.values[offset]
                                        * v1[[incoming_1, candidate_1]]
                                        * v2[[incoming_2, candidate_2]];
                                }
                            }
                            let got = actual[[outgoing + OUTGOING_DIM * candidate_1, candidate_2]];
                            assert!(
                                (got - expected).norm() <= 1.0e-11,
                                "axis order ({physical_axis}, {outgoing_axis}, {incoming_axis_1}, {incoming_axis_2}) mismatch at ({outgoing}, {candidate_1}, {candidate_2}): got {got}, expected {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn candidate_frames_for_edge_falls_back_on_a_leaf_edge_with_zero_incoming_edges() {
    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 0)
        .expect("star tree must have a directed edge 1 -> 0");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 0);

    let (arena, _) = SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0]]).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    // Node 1's only physical leg has dimension 2, so directed edge 1 -> 0
    // (zero incoming edges) has exactly two candidates, one per physical
    // value, and no incoming samples at all.
    let candidates = [
        ComponentSample {
            local_coordinate: 0,
            incoming: Vec::new(),
        },
        ComponentSample {
            local_coordinate: 1,
            incoming: Vec::new(),
        },
    ];

    let dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let scalar = candidates
        .iter()
        .map(|candidate| {
            frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(dispatched, scalar);
    // Sanity: this is a meaningful comparison, not a vacuous shape check --
    // the two candidates' frames actually differ.
    assert_ne!(dispatched[0], dispatched[1]);
}

#[test]
fn candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges() {
    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);

    // Seed two distinct physical points so each incoming edge (2 -> 0 and
    // 3 -> 0) has two candidate samples, giving this branch-point fallback a
    // genuine multi-candidate cartesian product to walk, not just a single
    // trivial candidate.
    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let incoming_edge_a = directed.incoming_to_from[0];
    let incoming_edge_b = directed.incoming_to_from[1];
    let ids_a = &candidate_sets.ids[incoming_edge_a];
    let ids_b = &candidate_sets.ids[incoming_edge_b];
    assert_eq!(ids_a.len(), 2);
    assert_eq!(ids_b.len(), 2);

    let mut candidates = Vec::new();
    for local_coordinate in 0..2 {
        for &id_a in ids_a {
            for &id_b in ids_b {
                candidates.push(ComponentSample {
                    local_coordinate,
                    incoming: vec![(incoming_edge_a, id_a), (incoming_edge_b, id_b)],
                });
            }
        }
    }
    assert_eq!(candidates.len(), 8);

    let dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let scalar = candidates
        .iter()
        .map(|candidate| {
            frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(dispatched, scalar);
    // Sanity: candidates are not all identical, so this exercises real,
    // differing per-candidate contractions rather than a degenerate case.
    assert!(dispatched.iter().any(|frame| frame != &dispatched[0]));
}

#[cfg(feature = "diagnostics")]
#[test]
fn candidate_frames_for_edge_records_frame_diagnostics_with_hub_coordination_number_three() {
    use crate::branch_diagnostics;

    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);

    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let incoming_edge_a = directed.incoming_to_from[0];
    let incoming_edge_b = directed.incoming_to_from[1];
    let ids_a = &candidate_sets.ids[incoming_edge_a];
    let ids_b = &candidate_sets.ids[incoming_edge_b];

    let mut candidates = Vec::new();
    for local_coordinate in 0..2 {
        for &id_a in ids_a {
            for &id_b in ids_b {
                candidates.push(ComponentSample {
                    local_coordinate,
                    incoming: vec![(incoming_edge_a, id_a), (incoming_edge_b, id_b)],
                });
            }
        }
    }

    branch_diagnostics::reset();
    let _dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();

    let snapshot = branch_diagnostics::snapshot();
    let hub_record = snapshot
        .iter()
        .find(|record| record.node == "0:0")
        .expect("hub node (0) of input 0 recorded in branch diagnostics");
    assert_eq!(hub_record.coordination_number, 3);
    assert_eq!(hub_record.bond_dims.len(), 3);
    assert!(hub_record.frame_cache_hits + hub_record.frame_cache_misses > 0);
}

/// The same hub node of two different input trees must produce two separate
/// registry entries: the diagnostics key is namespaced by the operand index.
#[cfg(feature = "diagnostics")]
#[test]
fn frame_diagnostics_keys_are_namespaced_per_input_operand() {
    use crate::branch_diagnostics;

    // Two operands sharing the same physical indices, as a product's two
    // inputs do; the clone gives input 1 the same node labels as input 0.
    let tree = star_tree_for_fallback_dispatch();
    let inputs = vec![tree.clone(), tree];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];

    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let incoming_edge_a = directed.incoming_to_from[0];
    let incoming_edge_b = directed.incoming_to_from[1];
    let id_a = candidate_sets.ids[incoming_edge_a][0];
    let id_b = candidate_sets.ids[incoming_edge_b][0];
    let candidate = ComponentSample {
        local_coordinate: 0,
        incoming: vec![(incoming_edge_a, id_a), (incoming_edge_b, id_b)],
    };

    branch_diagnostics::reset();
    let from_input_0 = frames
        .candidate_frame(&inputs, &problem, 0, edge, &candidate)
        .unwrap();
    let from_input_1 = frames
        .candidate_frame(&inputs, &problem, 1, edge, &candidate)
        .unwrap();
    assert_eq!(from_input_0.len(), from_input_1.len());

    let mut snapshot = branch_diagnostics::snapshot();
    snapshot.sort_by(|a, b| a.node.cmp(&b.node));
    let nodes: Vec<&str> = snapshot.iter().map(|record| record.node.as_str()).collect();
    assert_eq!(
        nodes,
        vec!["0:0", "1:0"],
        "the same hub node of two operands must not merge into one entry"
    );
    for record in &snapshot {
        assert_eq!(record.coordination_number, 3);
        assert_eq!(record.bond_dims.len(), record.coordination_number);
        assert_eq!(record.frame_cache_hits + record.frame_cache_misses, 1);
    }
}

#[test]
fn two_incoming_candidate_batch_obeys_the_working_byte_limit() {
    let inputs = vec![star_tree_for_fallback_dispatch()];
    let mut problem = prepare_problem(&inputs, &TreeAciOptions::default()).unwrap();
    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .unwrap();
    let directed = &problem.directed_edges[edge];
    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();
    let mut candidates = Vec::new();
    for &sample_1 in &candidate_sets.ids[directed.incoming_to_from[0]] {
        for &sample_2 in &candidate_sets.ids[directed.incoming_to_from[1]] {
            candidates.push(ComponentSample {
                local_coordinate: 0,
                incoming: vec![
                    (directed.incoming_to_from[0], sample_1),
                    (directed.incoming_to_from[1], sample_2),
                ],
            });
        }
    }
    let scratch_bytes = frames
        .enumerated_candidate_frame_scratch_elements(&problem, 0, edge, &candidate_sets)
        .unwrap()
        * std::mem::size_of::<f64>();
    assert!(scratch_bytes > 0);
    problem.max_working_bytes = scratch_bytes - 1;

    let error = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap_err();

    assert!(matches!(
        error,
        crate::TreeAciError::ResourceLimit {
            resource: "working bytes",
            requested,
            limit,
        } if requested == scratch_bytes && limit == scratch_bytes - 1
    ));
}

/// 4-arm star: hub node 0 with three leaves plus a fourth arm, so directed
/// edge `0 -> 1` has exactly three incoming edges (`2 -> 0`, `3 -> 0`,
/// `4 -> 0`). Used to pin that 3+-incoming-edge dispatch still uses the
/// scalar fallback once the two-incoming case gets a batched path.
fn four_arm_star_tree_for_three_incoming_fallback() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(1);
    let s1 = DynIndex::new_dyn(1);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let s4 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond02 = DynIndex::new_dyn(2);
    let bond03 = DynIndex::new_dyn(2);
    let bond04 = DynIndex::new_dyn(2);

    let node0 = IdxTensor::from_dense(
        vec![
            s0,
            bond01.clone(),
            bond02.clone(),
            bond03.clone(),
            bond04.clone(),
        ],
        (1..=16).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(vec![bond01, s1], vec![1.0, 2.0]).unwrap();
    let node2 = IdxTensor::from_dense(vec![bond02, s2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let node3 = IdxTensor::from_dense(vec![bond03, s3], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let node4 = IdxTensor::from_dense(vec![bond04, s4], vec![9.0, 10.0, 11.0, 12.0]).unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2, node3, node4], vec![0, 1, 2, 3, 4]).unwrap()
}

#[test]
fn candidate_frames_for_edge_still_falls_back_on_three_incoming_edges() {
    let inputs = vec![four_arm_star_tree_for_three_incoming_fallback()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("4-arm star must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 3);

    let seeds = vec![vec![0, 0, 0, 0, 0], vec![0, 0, 1, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let mut candidates = Vec::new();
    let incoming: Vec<_> = directed.incoming_to_from.clone();
    for &id_a in &candidate_sets.ids[incoming[0]] {
        for &id_b in &candidate_sets.ids[incoming[1]] {
            for &id_c in &candidate_sets.ids[incoming[2]] {
                candidates.push(ComponentSample {
                    local_coordinate: 0,
                    incoming: vec![
                        (incoming[0], id_a),
                        (incoming[1], id_b),
                        (incoming[2], id_c),
                    ],
                });
            }
        }
    }

    let dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let scalar = candidates
        .iter()
        .map(|candidate| {
            frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(dispatched, scalar);
}

/// Builds a `FrameBuilder` the same way `InputFrameStore::build_or_extend`
/// builds its initial (non-`extend`) builder at `frames.rs:229-235`, so tests
/// can drive `FrameBuilder::compute`/`compute_batch` directly against a fresh,
/// unpolluted memo. Two independent builders over the same
/// input/problem/arena (one per call) let a test compare the scalar and
/// batched paths without either polluting the other's memo.
fn build_frame_builder<'a>(
    input: &'a TreeTN<IdxTensor, usize>,
    problem: &'a PreparedTreeProblem<usize>,
    arena: &'a SampleArena,
) -> super::FrameBuilder<'a, f64, usize> {
    let cores = Rc::new(super::prepare_cores::<f64, usize>(input, problem).unwrap());
    let memo = problem
        .directed_edges
        .iter()
        .enumerate()
        .map(|(edge, _)| {
            let count = arena.directed_record_count(edge).unwrap();
            vec![None; count]
        })
        .collect::<Vec<_>>();
    super::FrameBuilder {
        input,
        input_index: 0,
        problem,
        arena,
        cores,
        memo,
        existing_frames: None,
    }
}

/// 3-node chain `0 -- 1 -- 2`. Directed edge `1 -> 2` has exactly one
/// incoming edge (`0 -> 1`, node 1's only other neighbor) -- the shape
/// `FrameBuilder::compute_batch`'s batched BLAS path requires, and which
/// none of this module's other fixtures (`y_tree`, `star_tree_for_fallback_dispatch`)
/// provide, since both have a degree-3 center and therefore only 0- or
/// 2-incoming directed edges. Values are arbitrary; the tests using this
/// fixture only compare `compute_batch` against the pre-existing scalar
/// `compute` path, not against a hand-computed answer.
fn chain_tree_for_batched_compute() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(2);

    let node0 = IdxTensor::from_dense(vec![s0, bond01.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let node1 = IdxTensor::from_dense(
        vec![bond01, s1, bond12.clone()],
        (1..=8).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node2 = IdxTensor::from_dense(vec![bond12, s2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2], vec![0, 1, 2]).unwrap()
}

#[test]
fn batched_duplicate_candidates_are_counted_once_in_the_cache_budget() {
    let input = chain_tree_for_batched_compute();
    let inputs = vec![input];
    let problem = prepare_problem(&inputs, &TreeAciOptions::default()).unwrap();
    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 2)
        .unwrap();
    let incoming_edge = problem.directed_edges[edge].incoming_to_from[0];
    let (arena, candidate_sets) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0]]).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();
    let candidate = ComponentSample {
        local_coordinate: 0,
        incoming: vec![(incoming_edge, candidate_sets.ids[incoming_edge][0])],
    };
    let retained_before = frames.retained_bytes();

    let result = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &[candidate.clone(), candidate])
        .unwrap();

    assert_eq!(result[0], result[1]);
    assert_eq!(
        frames.retained_bytes() - retained_before,
        result[0].len() * std::mem::size_of::<f64>()
            + std::mem::size_of::<super::CandidateCacheKey>()
    );
    assert_eq!(frames.candidate_cache.borrow().len(), 1);
}

/// `compute_batch` on a single-incoming-edge cut must produce exactly the
/// same memoized values as calling `compute` once per sample -- both paths
/// perform the same floating-point operations per sample in the same order
/// (see the prior BLAS plan's Task 3/4, which established this exact-equality
/// property for the analogous candidate-frame batching).
#[test]
fn compute_batch_matches_scalar_compute_on_a_chain_edge() {
    let input = chain_tree_for_batched_compute();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 2)
        .expect("chain must have a directed edge 1 -> 2");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 1);

    // Three distinct seeds, varying both node 0's and node 1's physical
    // values, so directed edge 1 -> 2 (local_coordinate = node 1's physical
    // value, incoming = node 0's sample) ends up with three distinct
    // samples.
    let seeds = vec![vec![0, 0, 0], vec![1, 0, 0], vec![0, 1, 0]];
    let (arena, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let sample_count = arena.directed_record_count(edge).unwrap();
    assert!(
        sample_count >= 3,
        "expected at least 3 distinct samples on the single-incoming edge, got {sample_count}"
    );

    let mut builder_a = build_frame_builder(&input, &problem, &arena);
    let mut builder_b = build_frame_builder(&input, &problem, &arena);

    let mut scalar_results = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        scalar_results.push(builder_a.compute(edge, sample).unwrap());
    }

    builder_b.compute_batch(edge, 0..sample_count).unwrap();
    for (sample, expected) in scalar_results.iter().enumerate() {
        let batched = builder_b.memo[edge][sample]
            .clone()
            .expect("compute_batch must memoize every sample in its range");
        assert_eq!(
            &batched, expected,
            "sample {sample} disagrees between compute and compute_batch"
        );
    }
    // Sanity: this is a meaningful comparison, not a vacuous shape check --
    // the three samples' frames actually differ.
    assert!(scalar_results
        .iter()
        .any(|frame| frame != &scalar_results[0]));
}

/// Combines the two properties `extend_matches_a_full_rebuild_on_the_grown_arena`
/// and `compute_batch_matches_scalar_compute_on_a_chain_edge` each cover
/// separately, and neither covers together: `y_tree`'s directed edges are
/// always 0- or 2-incoming (its center has degree 3, its leaves degree 1),
/// so `extend_matches_a_full_rebuild_on_the_grown_arena` only ever drives
/// `compute_batch`'s scalar-fallback branch; `compute_batch_matches_scalar_compute_on_a_chain_edge`
/// drives the batched branch but always from a fresh builder with range
/// `0..sample_count`, never a nonzero `known`. This test uses the chain
/// fixture (so directed edge `1 -> 2` takes the batched BLAS branch) and a
/// genuine `extend` call after growing the arena (so `build_or_extend`'s
/// loop invokes `compute_batch` with `known > 0` on that same edge) --
/// exactly the interaction this task was commissioned to scrutinize.
#[test]
fn extend_matches_a_full_rebuild_on_a_chain_with_a_batched_edge() {
    let input = chain_tree_for_batched_compute();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 2)
        .expect("chain must have a directed edge 1 -> 2");
    assert_eq!(
        problem.directed_edges[edge].incoming_to_from.len(),
        1,
        "edge 1 -> 2 must be the single-incoming-edge cut compute_batch batches"
    );

    let seed0 = vec![0, 0, 0];
    let (mut arena, mut candidates) =
        SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    let known = initial.frames[0][edge].sample_count;
    assert!(
        known > 0,
        "the seed must already produce a sample on the batched edge, or `known` is vacuously 0"
    );

    // Two more seeds, varying node 0's and node 1's physical values -- the
    // same shape `compute_batch_matches_scalar_compute_on_a_chain_edge` uses
    // to guarantee multiple distinct samples on edge 1 -> 2.
    for seed in [vec![1, 0, 0], vec![0, 1, 0]] {
        arena
            .inject_global_point(&mut candidates, &problem, &seed)
            .expect("grow the arena with a new global point");
    }

    let grown_count = arena.directed_record_count(edge).unwrap();
    assert!(
        grown_count > known,
        "growth must add new samples on the batched edge: known={known} grown={grown_count}"
    );

    let extended = initial
        .extend(std::slice::from_ref(&input), &problem, &arena)
        .expect("extend the store to the grown arena");
    let rebuilt =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("rebuild from scratch on the grown arena");

    for e in 0..problem.directed_edges.len() {
        let sample_count = arena.directed_record_count(e).unwrap();
        for sample in 0..sample_count {
            assert_eq!(
                extended.frame_values(0, e, sample).unwrap(),
                rebuilt.frame_values(0, e, sample).unwrap(),
                "edge {e} sample {sample} disagrees between extend and full rebuild"
            );
        }
    }
}

/// 5-node chain `0 -- 1 -- 2 -- 3 -- 4`. Used only by
/// `from_samples_issues_exactly_one_compute_call_per_memo_slot_on_a_five_node_chain`
/// below. Before the dependency-order fix, numeric edge ids put directed edge
/// id 1 (`1 -> 0`) before ids 3 (`2 -> 1`) and 5 (`3 -> 2`); id 1's
/// `compute_batch` priming recursion then scalar-computed samples on those
/// later single-incoming edges before their own batch calls. A 3-node chain
/// (`chain_tree_for_batched_compute`) is too short to reproduce this: it needs
/// at least one directed edge whose own incoming edge is itself
/// single-incoming (not a leaf), which first appears at 5 nodes.
fn chain5_tree_for_dedup_regression() -> TreeTN<IdxTensor, usize> {
    let s: Vec<DynIndex> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..4).map(|_| DynIndex::new_dyn(2)).collect();

    let node0 = IdxTensor::from_dense(
        vec![s[0].clone(), bonds[0].clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(
        vec![bonds[0].clone(), s[1].clone(), bonds[1].clone()],
        (1..=8).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node2 = IdxTensor::from_dense(
        vec![bonds[1].clone(), s[2].clone(), bonds[2].clone()],
        (1..=8).map(|value| value as f64 * 0.5).collect(),
    )
    .unwrap();
    let node3 = IdxTensor::from_dense(
        vec![bonds[2].clone(), s[3].clone(), bonds[3].clone()],
        (1..=8).map(|value| value as f64 * 0.25).collect(),
    )
    .unwrap();
    let node4 = IdxTensor::from_dense(
        vec![bonds[3].clone(), s[4].clone()],
        vec![5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2, node3, node4], vec![0, 1, 2, 3, 4]).unwrap()
}

/// Regression guard for the memo-check fix in `compute_batch`
/// (`docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s Update 7
/// addendum). `debug_stats::compute_calls()` increments exactly once per
/// "materialize a value" operation -- once per cache-miss in `compute`'s
/// scalar branch, once per column in `compute_batch`'s batched write loop
/// -- so if every memo slot is filled exactly once (no duplicate work), the
/// total call count must equal the total number of memo slots filled across
/// every directed edge. Before the fix this equality did not hold: index-
/// order edge processing let an earlier edge's priming recursion (plain
/// scalar `compute`) materialize samples on a later single-incoming edge,
/// and `compute_batch`'s batched branch had no check for already-memoized
/// samples before grouping and re-contracting them through a second,
/// wasted `mat_mul` -- unlike `candidate_frames_for_edge`, which already
/// checked its cache before grouping. The pre-existing
/// `extend_recomputes_only_the_newly_interned_samples` test only asserts a
/// loose `< rebuild_calls` / `> 0` bound and does not catch this, since the
/// duplication happens within a single `from_samples` call, not across an
/// `extend`.
#[test]
fn from_samples_issues_exactly_one_compute_call_per_memo_slot_on_a_five_node_chain() {
    let input = chain5_tree_for_dedup_regression();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    // Four distinct global seeds, varying node 0/1/2's physical values, so
    // every directed edge accumulates multiple distinct samples rather than
    // the trivially-deduplicated single-seed case.
    let seeds = vec![
        vec![0, 0, 0, 0, 0],
        vec![1, 0, 0, 0, 0],
        vec![0, 1, 0, 0, 0],
        vec![0, 0, 1, 0, 0],
    ];
    let (arena, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    super::debug_stats::reset();
    let store =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("build the frame store from scratch");
    let calls = super::debug_stats::compute_calls();

    let total_memo_slots: usize = store.frames[0].iter().map(|frame| frame.sample_count).sum();

    assert_eq!(
        calls, total_memo_slots as u64,
        "compute_batch must not redundantly recompute samples already memoized \
         by an earlier edge's priming recursion: calls={calls} slots={total_memo_slots}"
    );
    // Sanity: this chain has genuinely multiple samples on multiple edges,
    // not a vacuous comparison where every edge has exactly one sample.
    assert!(
        total_memo_slots > seeds.len() * problem.directed_edges.len() / 2,
        "expected substantially more than one sample per edge on average, got \
         {total_memo_slots} slots across {} directed edges",
        problem.directed_edges.len()
    );
}

#[test]
fn priming_reuses_memoized_incoming_without_copying_again() {
    let input = chain_tree_for_batched_compute();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let (arena, _) = SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0]]).unwrap();
    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 2)
        .expect("chain must have a single-incoming edge 1 -> 2");
    let incoming = problem.directed_edges[edge].incoming_to_from[0];
    let mut builder = build_frame_builder(&input, &problem, &arena);

    super::debug_stats::reset();
    builder.ensure_computed(incoming, 0).unwrap();
    builder.ensure_computed(incoming, 0).unwrap();

    assert_eq!(super::debug_stats::compute_calls(), 1);
    assert_eq!(super::debug_stats::memo_hit_copies(), 0);
}

/// A chain's non-leaf directed edges have exactly one incoming edge and must
/// therefore be materialized by `compute_batch`, not scalar-primed by an
/// earlier edge whose dependency has not been processed yet. Before the
/// dependency-order fix, the numeric edge-id order caused the first
/// single-incoming edge to recurse through later single-incoming edges and
/// materialize their samples through the scalar path.
#[test]
fn from_samples_uses_scalar_only_for_non_batch_edges_on_a_chain() {
    let input = chain5_tree_for_dedup_regression();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let seeds = vec![
        vec![0, 0, 0, 0, 0],
        vec![1, 0, 0, 0, 0],
        vec![0, 1, 0, 0, 0],
        vec![0, 0, 1, 0, 0],
    ];
    let (arena, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    super::debug_stats::reset();
    InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
        .expect("build the frame store from scratch");

    let expected_scalar_calls: u64 = problem
        .directed_edges
        .iter()
        .enumerate()
        .filter(|(_, edge)| edge.incoming_to_from.len() != 1)
        .map(|(edge, _)| arena.directed_record_count(edge).unwrap() as u64)
        .sum();
    assert_eq!(
        super::debug_stats::scalar_compute_calls(),
        expected_scalar_calls,
        "only non-batchable edges should use scalar contraction"
    );
    assert!(
        super::debug_stats::batched_compute_calls() > 0,
        "the chain must exercise the batched single-incoming path"
    );
}

#[test]
fn dependency_order_places_incoming_frames_before_dependents_on_a_star() {
    let input = star_tree_for_fallback_dispatch();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let order = crate::problem::dependency_order(&problem.directed_edges).unwrap();
    assert_eq!(order.len(), problem.directed_edges.len());

    let mut positions = vec![0usize; order.len()];
    for (position, &edge) in order.iter().enumerate() {
        positions[edge] = position;
    }
    for (edge, directed) in problem.directed_edges.iter().enumerate() {
        for &dependency in &directed.incoming_to_from {
            assert!(
                positions[dependency] < positions[edge],
                "dependency edge {dependency} must precede dependent edge {edge}"
            );
        }
    }
}

/// `compute_batch` on a directed edge with zero incoming edges (a leaf
/// source) must fall back to `compute` per sample and produce identical
/// results, mirroring
/// `candidate_frames_for_edge_falls_back_on_a_leaf_edge_with_zero_incoming_edges`'s
/// coverage of the analogous candidate-frame dispatcher. Reuses
/// `star_tree_for_fallback_dispatch` (added by the prior BLAS plan's fix
/// round, commit `4b06c3c2`) rather than a new branch-point fixture.
#[test]
fn compute_batch_falls_back_correctly_on_a_leaf_edge() {
    let input = star_tree_for_fallback_dispatch();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 0)
        .expect("star tree must have a directed edge 1 -> 0");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 0);

    // Directed edge 1 -> 0's only component is node 1's own physical value
    // (incoming is empty), so the seeds must vary node 1's (position index 1
    // in `node_order`, which is sorted `[0, 1, 2, 3]`) physical value, not
    // node 0's.
    let seeds = vec![vec![0, 0, 0, 0], vec![0, 1, 0, 0]];
    let (arena, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let sample_count = arena.directed_record_count(edge).unwrap();
    assert!(
        sample_count >= 2,
        "expected at least 2 distinct samples on the leaf edge, got {sample_count}"
    );

    let mut builder_a = build_frame_builder(&input, &problem, &arena);
    let mut builder_b = build_frame_builder(&input, &problem, &arena);

    let mut scalar_results = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        scalar_results.push(builder_a.compute(edge, sample).unwrap());
    }

    builder_b.compute_batch(edge, 0..sample_count).unwrap();
    for (sample, expected) in scalar_results.iter().enumerate() {
        let batched = builder_b.memo[edge][sample]
            .clone()
            .expect("compute_batch must memoize every sample in its range");
        assert_eq!(
            &batched, expected,
            "sample {sample} disagrees between compute and compute_batch"
        );
    }
    // Sanity: this is a meaningful comparison, not a vacuous shape check --
    // the two samples' frames actually differ.
    assert_ne!(scalar_results[0], scalar_results[1]);
}

/// `compute_batch` on a directed edge with two incoming edges (a branch
/// point) must batch via `compute_batch_two_incoming` and produce results
/// identical to `compute` per sample, mirroring
/// `candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges`'s
/// coverage of the analogous candidate-frame dispatcher. Reuses
/// `star_tree_for_fallback_dispatch` rather than a new branch-point fixture.
#[test]
fn compute_batch_batches_a_branch_edge_with_two_incoming_edges() {
    let input = star_tree_for_fallback_dispatch();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 2);

    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, _) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let sample_count = arena.directed_record_count(edge).unwrap();
    assert!(
        sample_count >= 2,
        "expected at least 2 distinct samples on the branch edge, got {sample_count}"
    );

    let mut builder_a = build_frame_builder(&input, &problem, &arena);
    let mut builder_b = build_frame_builder(&input, &problem, &arena);

    let mut scalar_results = Vec::with_capacity(sample_count);
    for sample in 0..sample_count {
        scalar_results.push(builder_a.compute(edge, sample).unwrap());
    }

    builder_b.compute_batch(edge, 0..sample_count).unwrap();
    for (sample, expected) in scalar_results.iter().enumerate() {
        let batched = builder_b.memo[edge][sample]
            .clone()
            .expect("compute_batch must memoize every sample in its range");
        assert_eq!(
            &batched, expected,
            "sample {sample} disagrees between compute and compute_batch"
        );
    }
    // Sanity: this is a meaningful comparison, not a vacuous shape check --
    // the two samples' frames actually differ.
    assert_ne!(scalar_results[0], scalar_results[1]);
}

#[test]
fn compute_batch_still_falls_back_on_three_incoming_edges() {
    let input = four_arm_star_tree_for_three_incoming_fallback();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("4-arm star must have a directed edge 0 -> 1");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 3);

    let seeds = vec![vec![0, 0, 0, 0, 0], vec![0, 0, 1, 1, 1]];
    let (arena, _candidates) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    let mut scalar_builder = build_frame_builder(&input, &problem, &arena);
    let sample_count = arena.directed_record_count(edge).unwrap();
    for sample in 0..sample_count {
        scalar_builder.compute(edge, sample).unwrap();
    }
    let scalar_values: Vec<Vec<f64>> = (0..sample_count)
        .map(|sample| scalar_builder.memo[edge][sample].clone().unwrap())
        .collect();

    let mut batched_builder = build_frame_builder(&input, &problem, &arena);
    batched_builder
        .compute_batch(edge, 0..sample_count)
        .unwrap();
    let batched_values: Vec<Vec<f64>> = (0..sample_count)
        .map(|sample| batched_builder.memo[edge][sample].clone().unwrap())
        .collect();

    assert_eq!(batched_values, scalar_values);
}

/// `FrameBuilder::compute`'s memo-miss path must check `existing_frames`
/// (the previous store's frames for this same input) before falling through
/// to a genuine `contract_prepared_core` computation -- this is the lazy-pull
/// mechanism this task adds. Not yet wired into `build_or_extend` (that is a
/// later task): this test constructs a `FrameBuilder` directly, mirroring
/// `build_frame_builder`, but with `existing_frames` set and the grown
/// edge's memo left as `None` for the old sample indices (not eagerly
/// seeded, unlike `build_or_extend`'s current per-edge loop).
///
/// Reuses the exact fixture and seed sequence
/// `extend_matches_a_full_rebuild_on_a_chain_with_a_batched_edge` uses (chain
/// fixture, seed `[0,0,0]` then grown by `[1,0,0]` and `[0,1,0]`), so the
/// growth shape here is already known to produce both an old sample and a
/// genuinely new sample on directed edge `1 -> 2` whose own incoming sample
/// (on edge `0 -> 1`) is itself already known -- letting a single
/// `compute` call on the new sample cleanly isolate exactly one genuine
/// `contract_prepared_core` invocation (the outer edge-1->2 record) from one
/// lazy pull (the inner edge-0->1 incoming record), rather than conflating
/// them.
#[test]
fn compute_pulls_already_known_samples_from_the_previous_store_without_recomputing() {
    let input = chain_tree_for_batched_compute();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 2)
        .expect("chain must have a directed edge 1 -> 2");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 1);

    let seed0 = vec![0, 0, 0];
    let (mut arena, mut candidates) =
        SampleArena::from_global_seeds(&problem, std::slice::from_ref(&seed0)).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    let known = initial.frames[0][edge].sample_count;
    assert!(
        known > 0,
        "the seed must already produce a sample on the batched edge, or `known` is vacuously 0"
    );

    for seed in [vec![1, 0, 0], vec![0, 1, 0]] {
        arena
            .inject_global_point(&mut candidates, &problem, &seed)
            .expect("grow the arena with a new global point");
    }
    let grown_count = arena.directed_record_count(edge).unwrap();
    assert!(
        grown_count > known,
        "growth must add new samples on the batched edge: known={known} grown={grown_count}"
    );

    // Build a second `FrameBuilder` directly, mirroring `build_frame_builder`
    // / `build_or_extend`'s construction, but wiring `existing_frames` to the
    // initial store's frames for input 0 and leaving the grown edge's memo
    // as `None` for the old sample indices instead of eagerly seeding them.
    let cores = Rc::new(super::prepare_cores::<f64, usize>(&input, &problem).unwrap());
    let memo = problem
        .directed_edges
        .iter()
        .enumerate()
        .map(|(e, _)| {
            let count = arena.directed_record_count(e).unwrap();
            vec![None; count]
        })
        .collect::<Vec<_>>();
    let mut builder = super::FrameBuilder {
        input: &input,
        input_index: 0,
        problem: &problem,
        arena: &arena,
        cores,
        memo,
        existing_frames: Some(initial.frames[0].as_slice()),
    };

    super::debug_stats::reset();
    let old_sample: usize = 0;
    assert!(
        old_sample < known,
        "sample 0 on the batched edge must already have been known to the initial store"
    );
    let pulled = builder.compute(edge, old_sample).unwrap();
    assert_eq!(
        pulled,
        initial.frame_values(0, edge, old_sample).unwrap(),
        "a lazily-pulled sample must match the previous store's value for the same sample"
    );
    assert_eq!(
        super::debug_stats::compute_calls(),
        0,
        "pulling an already-known sample from `existing_frames` must not invoke \
         `contract_prepared_core`"
    );

    // A repeat read of the same sample must hit `self.memo` (populated as a
    // side effect of the lazy pull above), not pull from `existing_frames`
    // again -- either way it must not increment `compute_calls`.
    let pulled_again = builder.compute(edge, old_sample).unwrap();
    assert_eq!(pulled_again, pulled);
    assert_eq!(super::debug_stats::compute_calls(), 0);

    // A genuinely new sample -- one the initial store never saw -- must
    // still be computed via `contract_prepared_core`, not silently skipped
    // or treated as a lazy-pull miss.
    let new_sample = grown_count - 1;
    assert!(
        new_sample >= known,
        "test setup must pick a genuinely new sample: new_sample={new_sample} known={known}"
    );
    let computed = builder.compute(edge, new_sample).unwrap();
    assert_eq!(
        super::debug_stats::compute_calls(),
        1,
        "a genuinely new sample must increment compute_calls by exactly 1"
    );
    // Sanity: this is a meaningful comparison, not a vacuous shape check --
    // the new sample's frame actually differs from the old one's.
    assert_ne!(computed, pulled);
}

/// A directed edge whose sample count did not change since the previous store
/// was built must come out of `extend` as the SAME `Rc<DirectedFrame<_>>`
/// allocation, not a freshly rebuilt copy of an identical frame.
///
/// `extend_matches_a_full_rebuild_on_the_grown_arena` above pins the values
/// and would still pass if every edge were rebuilt from scratch on every
/// call, which is exactly the redundancy this checks against: at chi=256 the
/// eager seed copy plus the reconstruction copy for unchanged edges measured
/// 36.6% of total ACI wall time, all of it pure data movement.
///
/// Growth primitive: `SampleArena::project_point_onto_edge`, the narrowest
/// one this crate has. Unlike `from_global_seeds` / `inject_global_point`
/// (which project onto every directed edge and so touch every edge on the
/// point's ancestor chains), it projects onto one directed edge and, through
/// `project_component`'s recursion, only that edge's own ancestor chain. On
/// the 5-node chain fixture, projecting onto `1 -> 0` walks
/// `1->0, 2->1, 3->2, 4->3` and provably leaves the four opposite-direction
/// edges `0->1, 1->2, 2->3, 3->4` untouched.
#[test]
fn extend_reuses_unchanged_edges_via_rc_instead_of_rebuilding_them() {
    let input = chain5_tree_for_dedup_regression();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();
    let edge_count = problem.directed_edges.len();

    let (mut arena, _candidates) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0, 0]]).unwrap();
    let initial =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("initial store");
    let counts_before: Vec<usize> = (0..edge_count)
        .map(|edge| initial.frames[0][edge].sample_count)
        .collect();

    let leftward = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 1 && arc.to == 0)
        .expect("chain must have a directed edge 1 -> 0");
    // Differs from the seed only at node 4, the far end of `1 -> 0`'s
    // ancestor chain, so every edge on that chain gets a genuinely new
    // component sample rather than deduplicating back onto the seed's.
    arena
        .project_point_onto_edge(&problem, leftward, &[0, 0, 0, 0, 1])
        .expect("project a new point onto one directed edge's own ancestor chain");

    let counts_after: Vec<usize> = (0..edge_count)
        .map(|edge| arena.directed_record_count(edge).unwrap())
        .collect();
    let unchanged: Vec<usize> = (0..edge_count)
        .filter(|&edge| counts_after[edge] == counts_before[edge])
        .collect();
    let grown: Vec<usize> = (0..edge_count)
        .filter(|&edge| counts_after[edge] > counts_before[edge])
        .collect();
    assert_eq!(
        grown.len(),
        4,
        "expected the four edges on `1 -> 0`'s ancestor chain to grow, got {grown:?}"
    );
    assert_eq!(
        unchanged.len(),
        4,
        "expected the four opposite-direction edges to be untouched, got {unchanged:?}"
    );
    for &edge in &unchanged {
        let arc = &problem.directed_edges[edge];
        assert!(
            arc.from < arc.to,
            "the untouched edges must be exactly the rightward ones, but {} -> {} is untouched",
            arc.from,
            arc.to
        );
    }

    let extended = initial
        .extend(std::slice::from_ref(&input), &problem, &arena)
        .expect("extend the store to the grown arena");
    let rebuilt =
        InputFrameStore::<f64>::from_samples(std::slice::from_ref(&input), &problem, &arena)
            .expect("rebuild from scratch on the grown arena");

    for &edge in &unchanged {
        assert!(
            Rc::ptr_eq(&initial.frames[0][edge], &extended.frames[0][edge]),
            "edge {edge} did not change, so extend must share the initial store's frame \
             allocation via Rc instead of rebuilding it"
        );
    }
    for &edge in &grown {
        assert!(
            !Rc::ptr_eq(&initial.frames[0][edge], &extended.frames[0][edge]),
            "edge {edge} grew, so extend must produce a new frame covering the new samples"
        );
        assert_eq!(
            extended.frames[0][edge].sample_count, counts_after[edge],
            "a grown edge's rebuilt frame must cover every sample the arena now holds"
        );
    }

    // Sharing must not cost correctness: every edge, old and new, still has
    // to agree with a from-scratch rebuild on the grown arena.
    for (edge, &count) in counts_after.iter().enumerate() {
        for sample in 0..count {
            assert_eq!(
                extended.frame_values(0, edge, sample).unwrap(),
                rebuilt.frame_values(0, edge, sample).unwrap(),
                "edge {edge} sample {sample} disagrees between extend and full rebuild"
            );
        }
    }
    // ... and the accounting must still cover the shared edges, which are
    // real retained memory this store's `frames` references.
    assert_eq!(extended.records(), rebuilt.records());
    assert_eq!(extended.retained_bytes(), rebuilt.retained_bytes());
}

/// Reproduces #671's core question directly inside the crate: at a
/// 2-incoming-edge branch point, how much faster is the batched path
/// (`candidate_frames_for_edge`, since the two-incoming dispatch fix) than
/// looping the scalar `candidate_frame` path per candidate (the code path
/// every branch point used before this fix, and what 3+-incoming nodes
/// still use today)? Not run by default (`#[ignore]`): it is a timing
/// report, not a correctness gate. Run explicitly with
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn branch_point_batched_speedup_vs_scalar_at_realistic_scale() {
    use std::time::Instant;

    let s0 = DynIndex::new_dyn(1);
    let s1 = DynIndex::new_dyn(1);
    let m = 40usize;
    let d = 32usize;
    let s2 = DynIndex::new_dyn(m);
    let s3 = DynIndex::new_dyn(m);
    let bond01 = DynIndex::new_dyn(4);
    let bond02 = DynIndex::new_dyn(d);
    let bond03 = DynIndex::new_dyn(d);
    let node0 = IdxTensor::from_dense(
        vec![s0, bond01.clone(), bond02.clone(), bond03.clone()],
        (0..4 * d * d).map(|v| v as f64).collect(),
    )
    .unwrap();
    let node1 =
        IdxTensor::from_dense(vec![bond01, s1], (0..4).map(|v| v as f64).collect()).unwrap();
    let node2 =
        IdxTensor::from_dense(vec![bond02, s2], (0..d * m).map(|v| v as f64).collect()).unwrap();
    let node3 =
        IdxTensor::from_dense(vec![bond03, s3], (0..d * m).map(|v| v as f64).collect()).unwrap();
    let star = TreeTN::from_tensors(vec![node0, node1, node2, node3], vec![0, 1, 2, 3]).unwrap();

    let inputs = vec![star];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .unwrap();
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);

    let seeds: Vec<Vec<usize>> = (0..m).map(|i| vec![0, 0, i, i]).collect();
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    let incoming_a = directed.incoming_to_from[0];
    let incoming_b = directed.incoming_to_from[1];
    let ids_a = &candidate_sets.ids[incoming_a];
    let ids_b = &candidate_sets.ids[incoming_b];
    let mut candidates = Vec::new();
    for &id_a in ids_a {
        for &id_b in ids_b {
            candidates.push(ComponentSample {
                local_coordinate: 0,
                incoming: vec![(incoming_a, id_a), (incoming_b, id_b)],
            });
        }
    }

    // Independent `InputFrameStore`s, each with its own fresh
    // `candidate_cache`: sharing one store across both timed sections would
    // let whichever path runs first populate the cache and make the second
    // path measure cache hits instead of real computation.
    let batched_frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();
    let batched_start = Instant::now();
    let batched = batched_frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let batched_elapsed = batched_start.elapsed();

    let scalar_frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();
    let scalar_start = Instant::now();
    let scalar: Vec<Vec<f64>> = candidates
        .iter()
        .map(|candidate| {
            scalar_frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect();
    let scalar_elapsed = scalar_start.elapsed();

    assert_eq!(batched, scalar);
    eprintln!(
        "batched (two-incoming fix): {batched_elapsed:?}\nscalar (old fallback, still used for 3+ incoming): {scalar_elapsed:?}\nspeedup: {:.2}x",
        scalar_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64()
    );
}

/// [AI Supplied] Diagnostic-only A/B for the existing tensorbackend batch API.
///
/// Unlike the previously rejected two-GEMM experiment, this preserves the
/// current first-stage contraction decomposition: every fixed-second-bond
/// core slice is still multiplied by `v1` independently.  The only change is
/// whether those same-shaped products are dispatched one at a time or through
/// `batched_mat_mul_same_shape_owned`.  It therefore isolates backend dispatch
/// consolidation from a change in the mathematical reduction tree.
#[test]
#[ignore]
fn diagnostic_two_incoming_sequential_vs_upstream_same_shape_batch() {
    use std::time::{Duration, Instant};

    use tensor4all_tensorbackend::{batched_mat_mul_same_shape_owned, mat_mul, Matrix};

    const OUTGOING_DIM: usize = 32;
    const INCOMING_DIM_1: usize = 256;
    const INCOMING_DIM_2: usize = 256;
    const N1: usize = 40;
    const N2: usize = 40;
    const REPEATS: usize = 9;

    let dims = vec![1, OUTGOING_DIM, INCOMING_DIM_1, INCOMING_DIM_2];
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in &dims {
        strides.push(stride);
        stride *= dim;
    }
    let core = super::PreparedCore {
        indices: dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect(),
        dims,
        strides,
        values: (0..stride)
            .map(|i| ((i * 17 + 3) % 257) as f64 / 257.0)
            .collect(),
    };
    let v1 = Matrix::from_col_major_vec(
        INCOMING_DIM_1,
        N1,
        (0..INCOMING_DIM_1 * N1)
            .map(|i| ((i * 13 + 5) % 251) as f64 / 251.0)
            .collect(),
    );
    let v2 = Matrix::from_col_major_vec(
        INCOMING_DIM_2,
        N2,
        (0..INCOMING_DIM_2 * N2)
            .map(|i| ((i * 11 + 7) % 241) as f64 / 241.0)
            .collect(),
    );

    let upstream = || {
        let mut a = Vec::with_capacity(INCOMING_DIM_2 * OUTGOING_DIM * INCOMING_DIM_1);
        let mut b = Vec::with_capacity(INCOMING_DIM_2 * INCOMING_DIM_1 * N1);
        for i2 in 0..INCOMING_DIM_2 {
            let core_matrix = super::single_incoming_core_matrix(
                &core,
                1,
                2,
                i2 * core.strides[3],
                OUTGOING_DIM,
                INCOMING_DIM_1,
            );
            a.extend(core_matrix.into_col_major_vec());
            b.extend_from_slice(v1.as_col_major_slice());
        }
        let stage1 = batched_mat_mul_same_shape_owned(
            INCOMING_DIM_2,
            OUTGOING_DIM,
            INCOMING_DIM_1,
            N1,
            a,
            b,
        )
        .unwrap();
        let stage1 = Matrix::from_col_major_vec(OUTGOING_DIM * N1, INCOMING_DIM_2, stage1);
        mat_mul(&stage1, &v2).unwrap()
    };

    let sequential = || {
        super::two_incoming_core_matrix_batched(
            &core,
            1,
            2,
            3,
            0,
            OUTGOING_DIM,
            INCOMING_DIM_1,
            INCOMING_DIM_2,
            &v1,
            &v2,
        )
        .unwrap()
    };

    let expected = sequential();
    let actual = upstream();
    let max_abs = expected
        .as_col_major_slice()
        .iter()
        .zip(actual.as_col_major_slice())
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .as_col_major_slice()
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 2.0e-14 * scale.max(1.0),
        "max_abs={max_abs}, scale={scale}"
    );

    let mut sequential_times = Vec::with_capacity(REPEATS);
    let mut upstream_times = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        if repeat % 2 == 0 {
            let start = Instant::now();
            std::hint::black_box(sequential());
            sequential_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(upstream());
            upstream_times.push(start.elapsed());
        } else {
            let start = Instant::now();
            std::hint::black_box(upstream());
            upstream_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(sequential());
            sequential_times.push(start.elapsed());
        }
    }
    sequential_times.sort_unstable();
    upstream_times.sort_unstable();
    let sequential_median: Duration = sequential_times[REPEATS / 2];
    let upstream_median: Duration = upstream_times[REPEATS / 2];
    let repeated_rhs_bytes = INCOMING_DIM_2 * INCOMING_DIM_1 * N1 * std::mem::size_of::<f64>();
    eprintln!(
        "sequential median: {sequential_median:?}\nupstream same-shape batch median: {upstream_median:?}\nupstream/sequential: {:.3}x\nrepeated RHS payload: {repeated_rhs_bytes} bytes\nmax relative disagreement: {:.3e}",
        upstream_median.as_secs_f64() / sequential_median.as_secs_f64(),
        max_abs / scale.max(1.0),
    );
}

/// [AI Supplied] Diagnostic for the chain path's per-input dispatch loop.
///
/// The dimensions model a binary-physical chain plateau with two same-shaped
/// operands. SimpleTT ACI already batches this dimension-compatible case, but
/// TreeACI currently calls the backend separately for each input.
#[test]
#[ignore]
fn diagnostic_chain_two_inputs_sequential_vs_upstream_same_shape_batch() {
    use std::time::{Duration, Instant};

    use tensor4all_tensorbackend::{
        batched_mat_mul_same_shape_owned, mat_mul, mat_mul_owned, Matrix,
    };

    const BATCH: usize = 2;
    const M: usize = 512;
    const K: usize = 256;
    const N: usize = 256;
    const REPEATS: usize = 9;

    let a_items = M * K;
    let b_items = K * N;
    let a = (0..BATCH * a_items)
        .map(|i| ((i * 17 + 3) % 257) as f64 / 257.0)
        .collect::<Vec<_>>();
    let b = (0..BATCH * b_items)
        .map(|i| ((i * 13 + 5) % 251) as f64 / 251.0)
        .collect::<Vec<_>>();

    let sequential = || {
        let mut outputs = Vec::with_capacity(BATCH * M * N);
        for input in 0..BATCH {
            let left = Matrix::from_col_major_vec(
                M,
                K,
                a[input * a_items..(input + 1) * a_items].to_vec(),
            );
            let right = Matrix::from_col_major_vec(
                K,
                N,
                b[input * b_items..(input + 1) * b_items].to_vec(),
            );
            outputs.extend(mat_mul(&left, &right).unwrap().into_col_major_vec());
        }
        outputs
    };
    let sequential_owned = || {
        let mut outputs = Vec::with_capacity(BATCH * M * N);
        for input in 0..BATCH {
            let left = Matrix::from_col_major_vec(
                M,
                K,
                a[input * a_items..(input + 1) * a_items].to_vec(),
            );
            let right = Matrix::from_col_major_vec(
                K,
                N,
                b[input * b_items..(input + 1) * b_items].to_vec(),
            );
            outputs.extend(mat_mul_owned(left, right).unwrap().into_col_major_vec());
        }
        outputs
    };
    let upstream =
        || batched_mat_mul_same_shape_owned(BATCH, M, K, N, a.clone(), b.clone()).unwrap();

    let expected = sequential();
    assert_eq!(expected, sequential_owned());
    let actual = upstream();
    let max_abs = expected
        .iter()
        .zip(&actual)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs <= 2.0e-14 * scale.max(1.0),
        "max_abs={max_abs}, scale={scale}"
    );

    let mut sequential_times = Vec::with_capacity(REPEATS);
    let mut sequential_owned_times = Vec::with_capacity(REPEATS);
    let mut upstream_times = Vec::with_capacity(REPEATS);
    for repeat in 0..REPEATS {
        if repeat % 2 == 0 {
            let start = Instant::now();
            std::hint::black_box(sequential());
            sequential_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(sequential_owned());
            sequential_owned_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(upstream());
            upstream_times.push(start.elapsed());
        } else {
            let start = Instant::now();
            std::hint::black_box(upstream());
            upstream_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(sequential_owned());
            sequential_owned_times.push(start.elapsed());
            let start = Instant::now();
            std::hint::black_box(sequential());
            sequential_times.push(start.elapsed());
        }
    }
    sequential_times.sort_unstable();
    sequential_owned_times.sort_unstable();
    upstream_times.sort_unstable();
    let sequential_median: Duration = sequential_times[REPEATS / 2];
    let sequential_owned_median: Duration = sequential_owned_times[REPEATS / 2];
    let upstream_median: Duration = upstream_times[REPEATS / 2];
    eprintln!(
        "two sequential borrowed inputs median: {sequential_median:?}\ntwo sequential owned inputs median: {sequential_owned_median:?}\nupstream same-shape batch median: {upstream_median:?}\nowned/borrowed: {:.3}x\nbatch/borrowed: {:.3}x\nmax relative disagreement: {:.3e}",
        sequential_owned_median.as_secs_f64() / sequential_median.as_secs_f64(),
        upstream_median.as_secs_f64() / sequential_median.as_secs_f64(),
        max_abs / scale.max(1.0),
    );
}
