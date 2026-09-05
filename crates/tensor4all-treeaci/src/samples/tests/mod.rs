use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::{projection_debug_stats, CandidateSets, PivotPairs, SampleArena};
use crate::{problem::prepare_problem, TreeAciError, TreeAciOptions};

fn make_tree(edges: &[(usize, usize)], node_count: usize) -> TreeTN<IdxTensor, usize> {
    let physical = (0..node_count)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(2))
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
            IdxTensor::from_dense(indices, vec![1.0; len]).unwrap()
        })
        .collect();
    TreeTN::from_tensors(tensors, (0..node_count).collect()).unwrap()
}

fn prepare(
    edges: &[(usize, usize)],
    node_count: usize,
) -> crate::problem::PreparedTreeProblem<usize> {
    prepare_problem::<f64, _>(&[make_tree(edges, node_count)], &TreeAciOptions::default()).unwrap()
}

#[test]
fn every_seed_materializes_across_every_cut() {
    for (edges, node_count) in [
        (vec![(0, 1), (1, 2), (2, 3)], 4),
        (vec![(0, 1), (0, 2), (0, 3)], 4),
        (vec![(0, 1), (0, 2), (0, 3), (0, 4)], 5),
    ] {
        let problem = prepare(&edges, node_count);
        let seeds = vec![vec![0; node_count], vec![1; node_count]];
        let (arena, active) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

        for forward in (0..problem.directed_edges.len()).step_by(2) {
            let reverse = problem.directed_edges[forward].reverse;
            for (seed, expected) in seeds.iter().enumerate() {
                assert_eq!(
                    arena
                        .materialize_global_point(
                            &problem,
                            forward,
                            active.ids[forward][seed],
                            active.ids[reverse][seed],
                        )
                        .unwrap(),
                    *expected
                );
            }
        }
    }
}

#[test]
fn all_cut_seed_projection_visits_each_directed_edge_once() {
    let node_count = 64;
    let edges = (0..node_count - 1)
        .map(|node| (node, node + 1))
        .collect::<Vec<_>>();
    let problem = prepare(&edges, node_count);
    projection_debug_stats::reset();

    SampleArena::from_global_seeds(&problem, &[vec![1; node_count]]).unwrap();

    assert_eq!(
        projection_debug_stats::projected_edges(),
        problem.directed_edges.len() as u64,
        "one global point must be lowered by one iterative dependency pass"
    );
}

#[test]
fn empty_and_duplicate_seeds_create_one_deterministic_active_sample() {
    let problem = prepare(&[(0, 1), (1, 2)], 3);
    let (empty_arena, empty_active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let duplicate = vec![vec![0, 0, 0], vec![0, 0, 0]];
    let (duplicate_arena, duplicate_active) =
        SampleArena::from_global_seeds(&problem, &duplicate).unwrap();

    assert!(empty_active.ids.iter().all(|ids| ids.len() == 1));
    assert!(duplicate_active.ids.iter().all(|ids| ids.len() == 1));
    assert_eq!(empty_arena.record_count(), duplicate_arena.record_count());
    assert_eq!(
        empty_arena.retained_bytes(),
        duplicate_arena.retained_bytes()
    );
}

#[test]
fn injection_projects_one_point_to_all_directed_cuts_and_deduplicates() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let (mut arena, mut active) =
        SampleArena::from_global_seeds(&problem, &[vec![0, 0, 0, 0]]).unwrap();
    let report = arena
        .inject_global_point(&mut active, &problem, &[1, 1, 1, 1])
        .unwrap();

    assert_eq!(report.total_added, problem.directed_edges.len());
    assert!(report.added_by_edge.iter().all(|added| *added == 1));
    assert_eq!(active.generation, 1);
    for forward in (0..problem.directed_edges.len()).step_by(2) {
        let reverse = problem.directed_edges[forward].reverse;
        assert_eq!(
            arena
                .materialize_global_point(
                    &problem,
                    forward,
                    *active.ids[forward].last().unwrap(),
                    *active.ids[reverse].last().unwrap(),
                )
                .unwrap(),
            vec![1, 1, 1, 1]
        );
    }

    let generation = active.generation;
    let duplicate = arena
        .inject_global_point(&mut active, &problem, &[1, 1, 1, 1])
        .unwrap();
    assert_eq!(duplicate.total_added, 0);
    assert_eq!(active.generation, generation);
}

/// `project_point_onto_edge` must touch only the requested edge's ancestor
/// chain, unlike `inject_global_point` (which projects onto every directed
/// edge). This is the whole point of the method: `bootstrap_samples` calls
/// it up to `chi` times per edge, and projecting onto every other edge on
/// every call was the dominant cost at high bond dimension (see
/// `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s third
/// #646 continuation).
#[test]
fn project_point_onto_edge_touches_only_the_requested_edges_ancestor_chain() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let mut arena = SampleArena::from_global_seeds(&problem, &[]).unwrap().0;
    let mut candidates = CandidateSets::new(problem.directed_edges.len());

    // Edge 0 is directed 1 -> 0 or 0 -> 1 depending on orientation; whichever
    // it is, its own component's ancestor chain does not cover every node in
    // this star topology, so at least one other edge's candidate set must
    // stay untouched by a single-edge projection.
    let target_edge = 0;
    projection_debug_stats::reset();
    let id = arena
        .project_point_onto_edge(&problem, target_edge, &[1, 1, 1, 1])
        .unwrap();
    candidates.push_unique(target_edge, id);

    assert_eq!(candidates.ids[target_edge], vec![id]);
    let untouched_edges = (0..problem.directed_edges.len())
        .filter(|&edge| edge != target_edge)
        .filter(|&edge| candidates.ids[edge].is_empty())
        .count();
    assert!(
        untouched_edges > 0,
        "a single-edge projection must not populate every directed edge's candidate set"
    );
    let mut expected_edges = 0usize;
    let mut stack = vec![target_edge];
    while let Some(edge) = stack.pop() {
        expected_edges += 1;
        stack.extend(
            problem.directed_edges[edge]
                .incoming_to_from
                .iter()
                .copied(),
        );
    }
    assert_eq!(
        projection_debug_stats::projected_edges(),
        expected_edges as u64
    );
    assert!(expected_edges < problem.directed_edges.len());
}

#[test]
fn masked_injection_projects_only_active_cuts_and_their_dependencies() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let point = [1, 1, 1, 1];
    let forward = 0;
    let reverse = problem.directed_edges[forward].reverse;

    let mut expected = SampleArena::from_global_seeds(&problem, &[]).unwrap().0;
    expected
        .project_point_onto_edge(&problem, forward, &point)
        .unwrap();
    expected
        .project_point_onto_edge(&problem, reverse, &point)
        .unwrap();

    let (mut actual, mut candidates) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let candidates_before = candidates.clone();
    let mut active = vec![false; problem.directed_edges.len()];
    active[forward] = true;
    active[reverse] = true;
    let report = actual
        .inject_global_point_masked(&mut candidates, &problem, &point, &active)
        .unwrap();

    assert_eq!(report.total_added, 2);
    assert_eq!(actual.record_count(), expected.record_count());
    assert!(candidates
        .ids
        .iter()
        .zip(&candidates_before.ids)
        .enumerate()
        .all(|(edge, (ids, before))| { (edge == forward || edge == reverse) || ids == before }));
}

/// The projected sample must still materialize back to the correct point,
/// matching what `inject_global_point` would have produced for the same
/// edge -- the cheaper path must not change the result, only the cost.
#[test]
fn project_point_onto_edge_materializes_to_the_same_point_as_inject_global_point() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let point = [1, 1, 1, 1];

    let mut direct_arena = SampleArena::from_global_seeds(&problem, &[]).unwrap().0;
    let forward = 0;
    let reverse = problem.directed_edges[forward].reverse;
    let direct_forward_id = direct_arena
        .project_point_onto_edge(&problem, forward, &point)
        .unwrap();
    let direct_reverse_id = direct_arena
        .project_point_onto_edge(&problem, reverse, &point)
        .unwrap();

    let (mut injected_arena, mut active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    injected_arena
        .inject_global_point(&mut active, &problem, &point)
        .unwrap();
    let injected_forward_id = *active.ids[forward].last().unwrap();
    let injected_reverse_id = *active.ids[reverse].last().unwrap();

    assert_eq!(
        direct_arena
            .materialize_global_point(&problem, forward, direct_forward_id, direct_reverse_id)
            .unwrap(),
        injected_arena
            .materialize_global_point(&problem, forward, injected_forward_id, injected_reverse_id)
            .unwrap()
    );
    assert_eq!(
        direct_arena
            .materialize_global_point(&problem, forward, direct_forward_id, direct_reverse_id)
            .unwrap(),
        point.to_vec()
    );
}

#[test]
fn checkpoint_rollback_removes_appended_records_and_dedup_entries() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let mut arena = SampleArena::from_global_seeds(&problem, &[]).unwrap().0;
    let before_records = arena.record_count();
    let before_bytes = arena.retained_bytes();
    let checkpoint = arena.checkpoint();

    let first_id = arena
        .project_point_onto_edge(&problem, 0, &[1, 1, 1, 1])
        .unwrap();
    assert!(arena.record_count() > before_records);
    assert!(arena.retained_bytes() > before_bytes);

    arena.rollback(checkpoint).unwrap();
    assert_eq!(arena.record_count(), before_records);
    assert_eq!(arena.retained_bytes(), before_bytes);

    let repeated_id = arena
        .project_point_onto_edge(&problem, 0, &[1, 1, 1, 1])
        .unwrap();
    assert_eq!(repeated_id, first_id);
    assert!(arena.record(0, repeated_id).is_ok());
}

#[test]
fn replacing_active_sets_does_not_invalidate_old_recursive_ids() {
    let problem = prepare(&[(0, 1), (0, 2), (0, 3)], 4);
    let seeds = vec![vec![0, 0, 0, 0], vec![1, 1, 1, 1]];
    let (arena, mut active) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let forward = 0;
    let reverse = problem.directed_edges[forward].reverse;
    let old_left = active.ids[forward][0];
    let old_right = active.ids[reverse][0];

    for ids in &mut active.ids {
        *ids = vec![ids[1]];
    }
    active.generation += 1;

    assert_eq!(
        arena
            .materialize_global_point(&problem, forward, old_left, old_right)
            .unwrap(),
        seeds[0]
    );
}

#[test]
fn invalid_points_and_arena_budget_fail_before_active_commit() {
    let problem = prepare(&[(0, 1)], 2);
    assert!(matches!(
        SampleArena::from_global_seeds(&problem, &[vec![0]]),
        Err(TreeAciError::PointLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        SampleArena::from_global_seeds(&problem, &[vec![0, 2]]),
        Err(TreeAciError::PhysicalCoordinateOutOfBounds {
            node: 1,
            coordinate: 2,
            local_dim: 2
        })
    ));

    let options = TreeAciOptions {
        max_sample_arena_bytes: 1,
        ..TreeAciOptions::default()
    };
    let limited_problem = prepare_problem::<f64, _>(&[make_tree(&[(0, 1)], 2)], &options).unwrap();
    assert!(matches!(
        SampleArena::from_global_seeds(&limited_problem, &[vec![0, 0]]),
        Err(TreeAciError::ResourceLimit {
            resource: "sample arena bytes",
            limit: 1,
            ..
        })
    ));
}

#[test]
fn candidate_sets_require_one_vector_per_directed_edge() {
    let problem = prepare(&[(0, 1)], 2);
    assert!(
        CandidateSets::new(problem.directed_edges.len()).ids.len() == problem.directed_edges.len()
    );
}

#[test]
fn candidate_sets_append_only_unique_ids() {
    let mut candidates = CandidateSets::new(4);
    assert!(candidates.push_unique(0, 7));
    assert!(!candidates.push_unique(0, 7));
    assert!(candidates.push_unique(0, 9));
    assert_eq!(candidates.ids[0], vec![7, 9]);
    assert!(candidates.ids[1].is_empty());
}

#[test]
fn pivot_pairs_report_rank_and_projections() {
    let mut pairs = PivotPairs::new(2);
    assert_eq!(pairs.rank(0), 0);
    pairs.set(0, vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(pairs.rank(0), 3);
    assert_eq!(pairs.forward_ids(0), vec![1, 2, 3]);
    assert_eq!(pairs.reverse_ids(0), vec![10, 20, 30]);
    assert_eq!(pairs.rank(1), 0);
}
