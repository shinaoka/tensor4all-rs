use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise, TreeAciOptions};
use tensor4all_treetn::TreeTN;

fn rank_one_tree(
    edges: &[(usize, usize)],
    node_count: usize,
    physical: &[DynIndex],
    reverse_physical_values: bool,
) -> TreeTN<IdxTensor, usize> {
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
            let values = if reverse_physical_values {
                vec![2.0, 1.0]
            } else {
                vec![1.0, 2.0]
            };
            IdxTensor::from_dense(indices, values).expect("rank-one fixture tensor")
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..node_count).collect()).expect("rank-one fixture tree")
}

#[test]
fn binary_tree_rank_scaling_stays_bounded() {
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
    let physical = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let first = rank_one_tree(&edges, 7, &physical, false);
    let second = rank_one_tree(&edges, 7, &physical, true);
    let options = TreeAciOptions {
        enable_global_guard: true,
        max_sweeps: 6,
        min_sweeps: 1,
        nsearch_global_pivots: 2,
        max_nglobal_pivots: 2,
        nsweeps_global_search: 4,
        ..TreeAciOptions::default()
    };

    let result =
        tree_elementwise::<f64, _, _>(|values| values[0] + values[1], &[first, second], &options)
            .expect("binary-tree rank-scaling run");

    let maximum_rank = result
        .diagnostics
        .edge_ranks
        .iter()
        .map(|(_, _, rank)| *rank)
        .max()
        .unwrap_or(0);
    assert!(
        maximum_rank <= 2,
        "maximum rank {maximum_rank} exceeded bound"
    );
    // Guard-on observed 122 points with two starts and four coordinate walks;
    // the earlier guard-off rank-only baseline was 76 points. Keep modest
    // headroom while catching accidental dense or repeated evaluation.
    const GUARD_EVALUATION_BUDGET: u64 = 140;
    assert!(
        result.diagnostics.evaluated_points < GUARD_EVALUATION_BUDGET,
        "guard evaluated {} points, budget {GUARD_EVALUATION_BUDGET}",
        result.diagnostics.evaluated_points,
    );
}
