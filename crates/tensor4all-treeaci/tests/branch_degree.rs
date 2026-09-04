//! End-to-end coverage for TreeACI on a tree whose hub has coordination
//! number four, so every hub-outward directed edge carries three incoming
//! components and the run can only complete through the arbitrary-degree
//! candidate-frame route added for issue #713.

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treeaci::{tree_elementwise, TreeAciOptions};
use tensor4all_treetn::TreeTN;

/// 4-arm star: hub node `0` carries one physical leg plus four bonds, so its
/// tree coordination number is four and directed edge `0 -> k` has three
/// incoming components, because an outward arc excludes its own target.
/// Bond dimensions are deliberately unequal.
fn four_arm_star(offset: usize, physical: &[DynIndex]) -> TreeTN<IdxTensor, usize> {
    let b01 = DynIndex::new_dyn(2);
    let b02 = DynIndex::new_dyn(2);
    let b03 = DynIndex::new_dyn(3);
    let b04 = DynIndex::new_dyn(2);

    let value = |index: usize| ((index + offset) % 13) as f64 / 4.0 - 1.0;
    let hub = IdxTensor::from_dense(
        vec![
            physical[0].clone(),
            b01.clone(),
            b02.clone(),
            b03.clone(),
            b04.clone(),
        ],
        (0..2 * 2 * 2 * 3 * 2).map(value).collect(),
    )
    .expect("hub tensor");
    let arms = [
        IdxTensor::from_dense(
            vec![b01, physical[1].clone()],
            (0..4).map(|index| value(index + 101)).collect(),
        )
        .expect("arm 1"),
        IdxTensor::from_dense(
            vec![b02, physical[2].clone()],
            (0..4).map(|index| value(index + 211)).collect(),
        )
        .expect("arm 2"),
        IdxTensor::from_dense(
            vec![b03, physical[3].clone()],
            (0..6).map(|index| value(index + 331)).collect(),
        )
        .expect("arm 3"),
        IdxTensor::from_dense(
            vec![b04, physical[4].clone()],
            (0..4).map(|index| value(index + 457)).collect(),
        )
        .expect("arm 4"),
    ];

    let mut tensors = vec![hub];
    tensors.extend(arms);
    TreeTN::from_tensors(tensors, (0..5).collect()).expect("four-arm star fixture")
}

#[test]
fn degree_four_hub_elementwise_difference_matches_the_dense_reference() {
    let physical = (0..5).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let left = four_arm_star(0, &physical);
    let right = four_arm_star(7, &physical);

    // One dense materialization per side, then one whole-result residual --
    // no per-point re-contraction of the interpolated tree.
    let expected = left
        .to_dense()
        .expect("dense left")
        .sub(&right.to_dense().expect("dense right"))
        .expect("dense reference difference");

    let options = TreeAciOptions {
        tolerance: 1.0e-12,
        max_sweeps: 12,
        ..TreeAciOptions::default()
    };
    let result =
        tree_elementwise::<f64, _, _>(|values| values[0] - values[1], &[left, right], &options)
            .expect("degree-four hub elementwise run");

    let actual = result.tree.to_dense().expect("dense interpolated result");
    let scale = expected.maxabs().expect("reference scale");
    assert!(scale > 1.0, "fixture must be non-degenerate, got {scale}");
    let residual = actual
        .sub(&expected)
        .expect("residual tensor")
        .maxabs()
        .expect("residual magnitude");
    assert!(
        residual <= 1.0e-9 * scale,
        "degree-four hub result differs from the dense reference: \
         residual {residual:.3e}, scale {scale:.3e}"
    );

    // The fixture really is the arbitrary-degree case: a 5-node star has four
    // edges, and the hub's outward directed edges each carry three incoming
    // components.
    assert_eq!(result.diagnostics.edge_ranks.len(), 4);
    let maximum_rank = result
        .diagnostics
        .edge_ranks
        .iter()
        .map(|(_, _, rank)| *rank)
        .max()
        .unwrap_or(0);
    assert!(
        (2..=8).contains(&maximum_rank),
        "unexpected maximum edge rank {maximum_rank}: a rank-one result would not \
         exercise a nontrivial candidate cross at the degree-four hub"
    );
}
