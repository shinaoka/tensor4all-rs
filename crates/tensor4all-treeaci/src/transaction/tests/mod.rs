use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::update_edge_transaction;
use crate::{state::TreeAciState, TreeAciError, TreeAciOptions};

fn input_tree() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left =
        IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0, 2.0, 10.0, 20.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![3.0, 4.0, 30.0, 40.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn matrix_tree(dimension: usize, values: Vec<f64>) -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(dimension);
    let right_site = DynIndex::new_dyn(dimension);
    let bond = DynIndex::new_dyn(dimension);
    let mut identity = vec![0.0; dimension * dimension];
    for diagonal in 0..dimension {
        identity[diagonal + dimension * diagonal] = 1.0;
    }
    let left = IdxTensor::from_dense(vec![left_site, bond.clone()], identity).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], values).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

#[test]
fn one_edge_commit_materializes_the_exact_operator_tree() {
    let input = input_tree();
    let expected = input.to_dense().unwrap();
    let options = TreeAciOptions::default();
    let inputs = vec![input];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut identity = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };
    let report = update_edge_transaction(&mut state, 0, &options, true, &mut identity).unwrap();

    assert_eq!(report.forward, 0);
    assert_eq!(report.new_rank, 1);
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&expected, 1.0e-10, 0.0)
        .unwrap());
    assert_eq!(
        state.output.canonical_region(),
        &std::collections::HashSet::from([1])
    );
    state.output.verify_internal_consistency().unwrap();
}

#[test]
fn commit_sets_pivot_pairs_and_replaces_candidate_sets() {
    let options = TreeAciOptions::default();
    let inputs = vec![input_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut identity = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };
    let report = update_edge_transaction(&mut state, 0, &options, true, &mut identity).unwrap();

    let edge_number = 0;
    assert_eq!(state.pivots.rank(edge_number), report.new_rank);
    assert_eq!(
        state.pivots.forward_ids(edge_number),
        state.candidates.ids[0]
    );
    assert_eq!(
        state.pivots.reverse_ids(edge_number),
        state.candidates.ids[1]
    );

    let mut forward = state.pivots.forward_ids(edge_number);
    let before = forward.len();
    forward.sort_unstable();
    forward.dedup();
    assert_eq!(forward.len(), before, "pivot rows must not repeat");
}

#[test]
fn reverse_direction_commit_keeps_pivot_pairs_in_canonical_orientation() {
    let options = TreeAciOptions::default();
    let inputs = vec![matrix_tree(
        3,
        vec![1.0, 4.0, 2.0, 5.0, 2.0, 6.0, 3.0, 1.0, 7.0],
    )];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut identity = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };

    update_edge_transaction(&mut state, 1, &options, false, &mut identity).unwrap();

    assert_eq!(
        state.pivots.forward_ids(0),
        state.candidates.ids[0],
        "pivot rows must use the canonical forward cut direction"
    );
    assert_eq!(
        state.pivots.reverse_ids(0),
        state.candidates.ids[1],
        "pivot columns must use the canonical reverse cut direction"
    );
}

#[test]
fn proposal_error_leaves_output_samples_and_generation_unchanged() {
    let options = TreeAciOptions::default();
    let inputs = vec![input_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let before = state.output.to_dense().unwrap();
    let candidates_before = state.candidates.clone();
    let pivots_before = state.pivots.clone();
    let records_before = state.sample_arena.record_count();
    let generation_before = state.generation;

    let mut failing = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "stop".into(),
        })
    };
    let result = update_edge_transaction(&mut state, 0, &options, true, &mut failing);
    assert!(matches!(result, Err(TreeAciError::Callback { .. })));
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&before, 0.0, 0.0)
        .unwrap());
    assert_eq!(state.candidates, candidates_before);
    assert_eq!(state.pivots, pivots_before);
    assert_eq!(state.sample_arena.record_count(), records_before);
    assert_eq!(state.generation, generation_before);
}
