use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::*;
use crate::{
    schedule::{run_directional_pass, PassDirection},
    state::TreeAciState,
    TreeAciOptions, TreeElementwiseBatch,
};

fn binary_tree_height_three(
    physical: &[DynIndex],
    reverse_physical_values: bool,
) -> TreeTN<IdxTensor, usize> {
    let edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)];
    let bonds = edges
        .iter()
        .map(|_| DynIndex::new_dyn(1))
        .collect::<Vec<_>>();
    let tensors = (0..7)
        .map(|node| {
            let mut indices = vec![physical[node].clone()];
            for (edge, &(left, right)) in edges.iter().enumerate() {
                if left == node || right == node {
                    indices.push(bonds[edge].clone());
                }
            }
            let elements: usize = indices.iter().map(IndexLike::dim).product();
            let values = if reverse_physical_values {
                vec![2.0, 1.0]
            } else {
                vec![1.0, 2.0]
            };
            assert_eq!(elements, values.len());
            IdxTensor::from_dense(indices, values).expect("fixture tensor")
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..7).collect()).expect("fixture tree")
}

fn binary_tree_height_three_fixture() -> (Vec<TreeTN<IdxTensor, usize>>, TreeAciOptions<usize>) {
    let options = TreeAciOptions {
        enable_global_guard: false,
        ..TreeAciOptions::default()
    };
    let physical = (0..7).map(|_| DynIndex::new_dyn(2)).collect::<Vec<_>>();
    let first = binary_tree_height_three(&physical, false);
    let second = binary_tree_height_three(&physical, true);
    (vec![first, second], options)
}

fn sum_operator(batch: TreeElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate() {
        *value = batch.get(0, point)? + batch.get(1, point)?;
    }
    Ok(())
}

fn run_one_directional_pass(
    state: &mut TreeAciState<'_, f64, usize>,
    options: &TreeAciOptions<usize>,
) -> crate::Result<()> {
    let mut operator = sum_operator;
    run_directional_pass(state, options, PassDirection::Forward, &mut operator).map(|_| ())
}

#[test]
fn seed_initialization_is_fully_nested() {
    let (inputs, _options) = binary_tree_height_three_fixture();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &_options).expect("fixture state");
    let report = check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )
    .expect("nesting check must run");
    assert!(
        (report.fraction() - 1.0).abs() < f64::EPSILON,
        "seed projections must be nested everywhere, got {}",
        report.fraction()
    );
}

#[test]
fn continuous_walk_interpolates_at_pivots_to_machine_precision() {
    let (inputs, options) = binary_tree_height_three_fixture();
    let mut state =
        TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
    run_one_directional_pass(&mut state, &options).expect("pass must run");
    let mut operator = sum_operator;
    let error = check_interpolation_for_state(&state, &mut operator).expect("check must run");
    assert!(
        error < 1e-10,
        "interpolation error {error} is not machine precision"
    );
}

#[test]
fn stored_output_is_a_gauge_of_the_skeleton_after_one_pass() {
    let (inputs, options) = binary_tree_height_three_fixture();
    let mut state =
        TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
    run_one_directional_pass(&mut state, &options).expect("pass must run");
    let mut operator = sum_operator;
    let deviation = check_gauge_equivalence(&state, &mut operator).expect("check must run");
    assert!(
        deviation < 1e-8,
        "stored output deviated from the skeleton by {deviation}"
    );
}

#[test]
fn nesting_report_detects_a_stale_incoming_sample() {
    let (inputs, options) = binary_tree_height_three_fixture();
    let mut state =
        TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("fixture state");
    run_one_directional_pass(&mut state, &options).expect("pass must run");

    let mut stale = None;
    for directed in 0..state.problem.directed_edges.len() {
        let edge_number = directed / 2;
        let ids = if directed == 2 * edge_number {
            state.pivots.forward_ids(edge_number)
        } else {
            state.pivots.reverse_ids(edge_number)
        };
        for sample in ids {
            if let Some(&(incoming, _)) = state
                .sample_arena
                .record(directed, sample)
                .expect("pivot record")
                .incoming
                .first()
            {
                stale = Some((directed, incoming));
                break;
            }
        }
        if stale.is_some() {
            break;
        }
    }
    let (directed, incoming) = stale.expect("binary tree must have a recursive pivot record");
    state.candidates.ids[incoming].clear();

    let report = check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )
    .expect("nesting check must run");
    assert!(
        !report.nested[directed],
        "validator failed to detect the removed incoming candidate"
    );
}
