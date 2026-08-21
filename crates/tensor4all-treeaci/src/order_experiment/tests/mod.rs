use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::*;
use crate::{state::TreeAciState, TreeAciOptions, TreeElementwiseBatch};

type Fixture = (Vec<TreeTN<IdxTensor, usize>>, TreeAciOptions<usize>);

fn product_tree(
    physical: &[DynIndex; 2],
    left_values: [f64; 2],
    right_values: [f64; 2],
) -> TreeTN<IdxTensor, usize> {
    let left_site = physical[0].clone();
    let right_site = physical[1].clone();
    let bond = DynIndex::new_dyn(1);
    let left = IdxTensor::from_dense(vec![left_site, bond.clone()], left_values.to_vec())
        .expect("left fixture tensor");
    let right = IdxTensor::from_dense(vec![bond, right_site], right_values.to_vec())
        .expect("right fixture tensor");
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).expect("fixture tree")
}

fn two_node_product_fixture() -> Fixture {
    let physical = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let first = product_tree(&physical, [1.0, 2.0], [1.0, 1.0]);
    let second = product_tree(&physical, [1.0, 1.0], [1.0, 2.0]);
    let options = TreeAciOptions::default();
    (vec![first, second], options)
}

fn sum_operator(batch: TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)? + batch.get(1, point)?;
    }
    Ok(())
}

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

fn tree_fixture(edges: &[(usize, usize)], node_count: usize) -> Fixture {
    let physical = (0..node_count)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let first = rank_one_tree(edges, node_count, &physical, false);
    let second = rank_one_tree(edges, node_count, &physical, true);
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    (vec![first, second], options)
}

fn path_fixture() -> Fixture {
    tree_fixture(&[(0, 1), (1, 2), (2, 3)], 4)
}

fn y_tree_fixture() -> Fixture {
    tree_fixture(&[(0, 1), (0, 2), (0, 3)], 4)
}

fn comb_fixture() -> Fixture {
    tree_fixture(&[(0, 1), (1, 2), (2, 3), (1, 4)], 5)
}

fn binary_tree_height_three_fixture() -> Fixture {
    tree_fixture(&[(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)], 7)
}

fn degree_four_star_fixture() -> Fixture {
    tree_fixture(&[(0, 1), (0, 2), (0, 3), (0, 4)], 5)
}

fn assert_arms_agree(fixture: fn() -> Fixture) {
    let sweeps = 6;
    let mut outcomes = Vec::new();
    for order in [
        EdgeOrder::ContinuousWalk,
        EdgeOrder::RandomPermutation,
        EdgeOrder::EdgeIndex,
    ] {
        let (inputs, options) = fixture();
        let mut state =
            TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
        let mut operator = sum_operator;
        outcomes.push(
            run_arm(&mut state, &options, order, sweeps, &mut operator)
                .expect("edge-order arm must run"),
        );
    }
    let baseline = &outcomes[0];
    for (index, arm) in outcomes.iter().enumerate().skip(1) {
        assert!(
            arm.pivot_error < 1e-10,
            "arm {index} lost interpolation at pivots: {}",
            arm.pivot_error
        );
        assert!(
            arm.held_out_error <= baseline.held_out_error.max(1e-8) * 10.0,
            "arm {index} held-out error {} vs baseline {}",
            arm.held_out_error,
            baseline.held_out_error
        );
        for (edge, (rank, base)) in arm.ranks.iter().zip(&baseline.ranks).enumerate() {
            assert!(
                *rank <= base + 1,
                "arm {index} inflated rank on edge {edge}: {rank} vs {base}"
            );
        }
    }
}

#[test]
fn pivot_only_update_changes_pivots_but_not_the_stored_output() {
    let (inputs, options) = two_node_product_fixture();
    let mut state =
        TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
    let output_before = state.output.to_dense().expect("output dense tensor");
    let pivots_before = state.pivots.clone();
    let mut operator = sum_operator;

    pivot_only_update(&mut state, 0, &options, &mut operator).expect("update must run");

    assert_ne!(state.pivots, pivots_before, "pivots must change");
    assert!(state
        .output
        .to_dense()
        .expect("output dense tensor")
        .isapprox(&output_before, 0.0, 0.0)
        .expect("dense comparison"));
}

#[test]
fn continuous_arm_reports_interpolation_and_held_out_errors() {
    let (inputs, options) = two_node_product_fixture();
    let mut state =
        TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
    let mut operator = sum_operator;

    let outcome = run_arm(
        &mut state,
        &options,
        EdgeOrder::ContinuousWalk,
        1,
        &mut operator,
    )
    .expect("arm must run");

    assert_eq!(outcome.edge_updates, 1);
    assert_eq!(outcome.ranks, vec![2]);
    assert!(outcome.pivot_error < 1e-10);
    assert!(outcome.held_out_error.is_finite());
}

#[test]
fn edge_order_does_not_change_the_result_on_a_path() {
    assert_arms_agree(path_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_y_tree() {
    assert_arms_agree(y_tree_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_comb() {
    assert_arms_agree(comb_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_binary_tree() {
    assert_arms_agree(binary_tree_height_three_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_degree_four_star() {
    assert_arms_agree(degree_four_star_fixture);
}
