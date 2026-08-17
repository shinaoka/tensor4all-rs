use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::*;
use crate::{
    problem::{prepare_problem, PreparedTreeProblem},
    samples::{CandidateSets, PivotPairs, SampleArena},
    TreeAciOptions,
};

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
            let elements = indices.iter().map(IndexLike::dim).product();
            IdxTensor::from_dense(indices, vec![1.0; elements]).expect("fixture tensor")
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..node_count).collect()).expect("fixture tree")
}

fn all_points(node_count: usize) -> Vec<Vec<usize>> {
    (0..(1usize << node_count))
        .map(|encoded| {
            (0..node_count)
                .map(|node| (encoded >> node) & 1)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn prepare(edges: &[(usize, usize)], node_count: usize) -> PreparedTreeProblem<usize> {
    prepare_problem(&[make_tree(edges, node_count)], &TreeAciOptions::default())
        .expect("fixture problem")
}

fn pivots_from_candidates(
    problem: &PreparedTreeProblem<usize>,
    candidates: &CandidateSets,
) -> PivotPairs {
    let mut pivots = PivotPairs::new(problem.directed_edges.len() / 2);
    for edge_number in 0..problem.directed_edges.len() / 2 {
        let forward = &candidates.ids[2 * edge_number];
        let reverse = &candidates.ids[2 * edge_number + 1];
        let rank = forward.len().min(reverse.len()).min(2);
        pivots.set(
            edge_number,
            (0..rank).map(|k| (forward[k], reverse[k])).collect(),
        );
    }
    pivots
}

fn two_node_full_rank_fixture() -> (PreparedTreeProblem<usize>, SampleArena, PivotPairs) {
    let problem = prepare(&[(0, 1)], 2);
    let seeds = all_points(2);
    let (arena, candidates) = SampleArena::from_global_seeds(&problem, &seeds).expect("samples");
    (
        problem.clone(),
        arena,
        pivots_from_candidates(&problem, &candidates),
    )
}

fn y_tree_full_rank_fixture() -> (PreparedTreeProblem<usize>, SampleArena, PivotPairs) {
    let problem = prepare(&[(0, 1), (0, 2)], 3);
    let seeds = all_points(3);
    let (arena, candidates) = SampleArena::from_global_seeds(&problem, &seeds).expect("samples");
    (
        problem.clone(),
        arena,
        pivots_from_candidates(&problem, &candidates),
    )
}

/// A full-rank cross must reproduce a nonsingular two-site oracle exactly.
#[test]
fn full_rank_skeleton_reproduces_the_oracle_exactly() {
    let (problem, arena, pivots) = two_node_full_rank_fixture();
    let mut oracle = |sigma: &[usize]| -> crate::Result<f64> {
        let a = sigma[0] as f64;
        let b = sigma[1] as f64;
        Ok(1.0 + 2.0 * a + 3.0 * b + 5.0 * a * b)
    };
    let tensors =
        skeleton_tensors(&problem, &arena, &pivots, &mut oracle).expect("skeleton must build");

    let mut worst = 0.0_f64;
    for sigma in all_points(2) {
        let expected = oracle(&sigma).expect("oracle");
        let actual = skeleton_evaluate(&tensors, &problem, &sigma).expect("evaluate");
        worst = worst.max((expected - actual).abs());
    }
    assert!(worst < 1e-12, "skeleton deviated by {worst}");
}

#[test]
fn skeleton_matches_a_dense_contraction_on_a_y_tree() {
    let (problem, arena, pivots) = y_tree_full_rank_fixture();
    let mut oracle = |sigma: &[usize]| -> crate::Result<f64> {
        let a = sigma[0] as f64;
        let b = sigma[1] as f64;
        let c = sigma[2] as f64;
        Ok(1.0
            + 0.2 * a
            + 0.3 * b
            + 0.5 * c
            + 0.7 * a * b
            + 1.1 * a * c
            + 1.3 * b * c
            + 1.7 * a * b * c)
    };
    let tensors = skeleton_tensors(&problem, &arena, &pivots, &mut oracle).expect("skeleton");
    let mut worst = 0.0_f64;
    for sigma in all_points(3) {
        let expected = oracle(&sigma).expect("oracle");
        let actual = skeleton_evaluate(&tensors, &problem, &sigma).expect("evaluate");
        worst = worst.max((expected - actual).abs());
    }
    assert!(worst < 1e-12, "Y-tree skeleton deviated by {worst}");
}
