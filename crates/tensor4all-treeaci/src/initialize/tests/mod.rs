use num_complex::{Complex32, Complex64};
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::{algebraic_edge_bounds, bootstrap_samples, validate_initial_guess_scalar_kind};
use crate::{problem::prepare_problem, TreeAciOptions};

fn make_tree(edges: &[(usize, usize)], node_count: usize) -> TreeTN<IdxTensor, usize> {
    make_tree_with_dims(edges, &vec![2; node_count])
}

fn make_tree_with_dims(
    edges: &[(usize, usize)],
    physical_dims: &[usize],
) -> TreeTN<IdxTensor, usize> {
    let physical = physical_dims
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let tensors = (0..physical_dims.len())
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
    TreeTN::from_tensors(tensors, (0..physical_dims.len()).collect()).unwrap()
}

#[test]
fn initial_guess_scalar_validation_checks_kind_without_reading_values() {
    let index = DynIndex::new_dyn(1);
    let real = IdxTensor::from_dense(vec![index.clone()], vec![1.0_f32]).unwrap();
    let complex = IdxTensor::from_dense(vec![index], vec![Complex64::new(1.0, -2.0)]).unwrap();

    validate_initial_guess_scalar_kind::<f32>(&real).unwrap();
    validate_initial_guess_scalar_kind::<f64>(&real).unwrap();
    validate_initial_guess_scalar_kind::<Complex32>(&real).unwrap();
    validate_initial_guess_scalar_kind::<Complex64>(&real).unwrap();
    assert!(validate_initial_guess_scalar_kind::<f64>(&complex).is_err());
    validate_initial_guess_scalar_kind::<Complex32>(&complex).unwrap();
    validate_initial_guess_scalar_kind::<Complex64>(&complex).unwrap();
}

/// `bootstrap_samples` must reach exactly the requested rank on every edge
/// (bounded by the algebraic maximum), with distinct, valid samples -- the
/// behaviour that `project_point_onto_edge` (see `samples::tests`) must
/// preserve now that it replaces the clone-and-project-every-edge path.
#[test]
fn bootstrap_samples_reaches_the_requested_rank_on_every_edge() {
    for (edges, node_count) in [
        (vec![(0, 1), (1, 2), (2, 3)], 4),
        (vec![(0, 1), (0, 2), (0, 3)], 4),
    ] {
        let tree = make_tree(&edges, node_count);
        let problem =
            prepare_problem::<f64, _>(std::slice::from_ref(&tree), &TreeAciOptions::default())
                .expect("prepared problem");
        let algebraic = algebraic_edge_bounds(&problem).expect("algebraic bounds");
        // Ask for less than the algebraic maximum everywhere it is more than
        // 1, so the loop must actually enumerate multiple points per edge
        // rather than exhausting the whole component space trivially.
        let targets = algebraic
            .iter()
            .map(|&bound| bound.clamp(1, 2))
            .collect::<Vec<_>>();

        let (arena, candidates, pivots) =
            bootstrap_samples(&problem, &targets).expect("bootstrap must reach every target");

        for (edge_number, &target) in targets.iter().enumerate() {
            let forward = 2 * edge_number;
            let reverse = forward + 1;
            assert_eq!(
                candidates.ids[forward].len(),
                target,
                "edge {edge_number} forward candidate count must equal its target rank"
            );
            assert_eq!(
                candidates.ids[reverse].len(),
                target,
                "edge {edge_number} reverse candidate count must equal its target rank"
            );
            assert_eq!(pivots.rank(edge_number), target);

            // Every selected pivot pair must materialize to a valid,
            // in-range point through the arena.
            for &(left, right) in &pivots.per_edge[edge_number] {
                let point = arena
                    .materialize_global_point(&problem, forward, left, right)
                    .expect("pivot pair must materialize to a full point");
                assert_eq!(point.len(), node_count);
            }
        }
    }
}

#[test]
fn long_chain_algebraic_bounds_saturate_without_rejecting_small_active_ranks() {
    let node_count = 130;
    let edges = (0..node_count - 1)
        .map(|node| (node, node + 1))
        .collect::<Vec<_>>();
    let tree = make_tree(&edges, node_count);
    let problem =
        prepare_problem::<f64, _>(std::slice::from_ref(&tree), &TreeAciOptions::default())
            .expect("long rank-two chain must prepare");

    let bounds = algebraic_edge_bounds(&problem)
        .expect("an unrepresentably large physical space is still a valid rank ceiling");
    assert_eq!(bounds.len(), edges.len());
    assert_eq!(bounds[edges.len() / 2], usize::MAX);

    let targets = vec![1; edges.len()];
    let (_, candidates, pivots) = bootstrap_samples(&problem, &targets)
        .expect("rank-one bootstrap must not materialize the full physical space");
    assert!(candidates.ids.iter().all(|ids| ids.len() == 1));
    assert!(pivots.per_edge.iter().all(|pairs| pairs.len() == 1));
}

fn middle_cut_bootstrap_points() -> Vec<Vec<usize>> {
    let edges = [(0, 1), (1, 2), (2, 3)];
    let tree = make_tree_with_dims(&edges, &[2, 3, 2, 3]);
    let problem =
        prepare_problem::<f64, _>(std::slice::from_ref(&tree), &TreeAciOptions::default())
            .expect("prepared problem");
    let (arena, _, pivots) = bootstrap_samples(&problem, &[1, 4, 1]).expect("bootstrap samples");

    let mut points = Vec::new();
    for &(left, right) in &pivots.per_edge[1] {
        points.push(
            arena
                .materialize_global_point(&problem, 2, left, right)
                .expect("pivot pair must materialize"),
        );
    }
    points
}

#[test]
fn bootstrap_samples_follow_generalized_digit_reversal() {
    assert_eq!(
        middle_cut_bootstrap_points(),
        vec![
            vec![0, 0, 0, 0],
            vec![0, 1, 0, 1],
            vec![0, 2, 0, 2],
            vec![1, 0, 1, 0],
        ]
    );
}
