use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use super::{commit_edge_proposal, update_edge_transaction};
use crate::{
    local_update::LocalUpdateResult, samples::ComponentSample, state::TreeAciState, TreeAciError,
    TreeAciOptions,
};

fn input_tree() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let left =
        IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0, 2.0, 10.0, 20.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![3.0, 4.0, 30.0, 40.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn rank_one_tree_with_nonzero_maximum() -> TreeTN<IdxTensor, usize> {
    let left_site = DynIndex::new_dyn(2);
    let right_site = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    let left = IdxTensor::from_dense(vec![left_site, bond.clone()], vec![1.0, 10.0]).unwrap();
    let right = IdxTensor::from_dense(vec![bond, right_site], vec![1.0, 10.0]).unwrap();
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

#[test]
fn frame_extension_error_rolls_back_interned_samples() {
    let options = TreeAciOptions::default();
    let inputs = vec![rank_one_tree_with_nonzero_maximum()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let output_before = state.output.to_dense().unwrap();
    let candidates_before = state.candidates.clone();
    let pivots_before = state.pivots.clone();
    let records_before = state.sample_arena.record_count();
    let bytes_before = state.sample_arena.retained_bytes();
    let generation_before = state.generation;
    state.problem.max_frame_bytes = state.input_frames.retained_bytes();
    let mut identity = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };

    let result = update_edge_transaction(&mut state, 0, &options, true, &mut identity);

    assert!(matches!(
        result,
        Err(TreeAciError::ResourceLimit {
            resource: "frame bytes",
            ..
        })
    ));
    assert_eq!(state.sample_arena.record_count(), records_before);
    assert_eq!(state.sample_arena.retained_bytes(), bytes_before);
    assert_eq!(state.candidates, candidates_before);
    assert_eq!(state.pivots, pivots_before);
    assert_eq!(state.generation, generation_before);
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&output_before, 0.0, 0.0)
        .unwrap());
}

#[test]
fn invalid_factor_shape_is_rejected_before_any_commit_mutation() {
    let options = TreeAciOptions::default();
    let inputs = vec![input_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let output_before = state.output.to_dense().unwrap();
    let candidates_before = state.candidates.clone();
    let pivots_before = state.pivots.clone();
    let records_before = state.sample_arena.record_count();
    let generation_before = state.generation;
    let sample = ComponentSample {
        local_coordinate: 0,
        incoming: Vec::new(),
    };
    let proposal = LocalUpdateResult {
        row_samples: vec![sample.clone(), sample.clone()],
        col_samples: vec![sample.clone(), sample],
        left: Matrix::from_col_major_vec(1, 2, vec![1.0, 2.0]),
        right: Matrix::from_col_major_vec(1, 1, vec![1.0]),
        pivot_errors: vec![0.0],
        sampled_scale: 1.0,
        row_count: 1,
        col_count: 1,
        local_values: Vec::new(),
    };

    let error = commit_edge_proposal(&mut state, 0, proposal)
        .expect_err("incompatible factor shapes must be rejected");

    assert!(matches!(
        error,
        TreeAciError::InternalInvariant {
            message: "edge proposal factors and selected samples disagree on rank"
        }
    ));
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&output_before, 0.0, 0.0)
        .unwrap());
    assert_eq!(state.candidates, candidates_before);
    assert_eq!(state.pivots, pivots_before);
    assert_eq!(state.sample_arena.record_count(), records_before);
    assert_eq!(state.generation, generation_before);
}

#[test]
fn incomplete_commit_metadata_is_rejected_before_staging() {
    let options = TreeAciOptions::default();
    let inputs = vec![input_tree()];
    let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
    let mut identity = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)?;
        }
        Ok(())
    };
    let proposal = crate::local_update::materialize_and_factor_edge(
        state.inputs,
        &state.problem,
        &state.candidates,
        &state.input_frames,
        0,
        &options,
        true,
        &mut identity,
    )
    .unwrap();
    let output_before = state.output.to_dense().unwrap();
    let records_before = state.sample_arena.record_count();
    let generation_before = state.generation;
    state.candidates.ids.pop();

    let error = commit_edge_proposal(&mut state, 0, proposal)
        .expect_err("incomplete candidate metadata must be rejected");

    assert!(matches!(
        error,
        TreeAciError::InternalInvariant {
            message: "edge proposal state metadata is incomplete"
        }
    ));
    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&output_before, 0.0, 0.0)
        .unwrap());
    assert_eq!(state.sample_arena.record_count(), records_before);
    assert_eq!(state.generation, generation_before);
}
