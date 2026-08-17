use std::cell::Cell;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::materialize_and_factor_edge;
use crate::{
    frames::InputFrameStore, problem::prepare_problem, samples::SampleArena, TreeAciError,
    TreeAciOptions,
};

fn two_node_tree(scale: f64) -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    two_node_tree_with_indices(scale, s0, s1, bond)
}

fn two_node_tree_with_indices(
    scale: f64,
    s0: DynIndex,
    s1: DynIndex,
    bond: DynIndex,
) -> TreeTN<IdxTensor, usize> {
    let left = IdxTensor::from_dense(
        vec![s0, bond.clone()],
        vec![scale, 2.0 * scale, 10.0 * scale, 20.0 * scale],
    )
    .unwrap();
    let right = IdxTensor::from_dense(vec![bond, s1], vec![3.0, 4.0, 30.0, 40.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

#[test]
fn local_entries_equal_direct_values_and_callback_layout_is_column_major() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let inputs = vec![
        two_node_tree_with_indices(1.0, s0.clone(), s1.clone(), DynIndex::new_dyn(2)),
        two_node_tree_with_indices(2.0, s0, s1, DynIndex::new_dyn(2)),
    ];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let calls = Cell::new(0);
    let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        assert_eq!(batch.n_inputs(), 2);
        assert_eq!(batch.n_points(), 4);
        assert_eq!(
            batch.as_col_major_slice(),
            &[43.0, 86.0, 86.0, 172.0, 430.0, 860.0, 860.0, 1720.0]
        );
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)? * batch.get(1, point)?;
        }
        Ok(())
    };

    let update = materialize_and_factor_edge(
        &inputs,
        &problem,
        &arena,
        &active,
        &frames,
        0,
        &options,
        true,
        &mut operator,
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
    assert_eq!(
        update.local_values,
        vec![3698.0, 14792.0, 369800.0, 1479200.0]
    );
    assert_eq!(update.sampled_scale, 1479200.0);
}

#[test]
fn luci_factors_reconstruct_rank_one_and_zero_targets() {
    for zero in [false, true] {
        let inputs = vec![two_node_tree(1.0)];
        let options = TreeAciOptions::default();
        let problem = prepare_problem(&inputs, &options).unwrap();
        let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
        let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
        let mut operator = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = if zero { 0.0 } else { batch.get(0, point)? };
            }
            Ok(())
        };
        let update = materialize_and_factor_edge(
            &inputs,
            &problem,
            &arena,
            &active,
            &frames,
            0,
            &options,
            true,
            &mut operator,
        )
        .unwrap();

        assert_eq!(update.left.ncols(), 1);
        assert_eq!(update.right.nrows(), 1);
        for col in 0..update.col_count {
            for row in 0..update.row_count {
                let reconstructed = update.left[[row, 0]] * update.right[[0, col]];
                assert!(
                    (reconstructed - update.local_values[row + update.row_count * col]).abs()
                        < 1.0e-10
                );
            }
        }
    }
}

#[test]
fn callback_error_and_matrix_budget_stop_before_factorization() {
    let inputs = vec![two_node_tree(1.0)];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let (arena, active) = SampleArena::from_global_seeds(&problem, &[]).unwrap();
    let frames = InputFrameStore::from_samples(&inputs, &problem, &arena).unwrap();
    let mut failing = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(TreeAciError::Callback {
            message: "sentinel".into(),
        })
    };
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs, &problem, &arena, &active, &frames, 0, &options, true, &mut failing,
        ),
        Err(TreeAciError::Callback { message }) if message == "sentinel"
    ));

    let limited = TreeAciOptions {
        max_local_matrix_elements: 3,
        ..TreeAciOptions::default()
    };
    let mut unused = |_batch: crate::TreeElementwiseBatch<'_, f64>, _output: &mut [f64]| Ok(());
    assert!(matches!(
        materialize_and_factor_edge(
            &inputs,
            &problem,
            &arena,
            &active,
            &frames,
            0,
            &limited,
            true,
            &mut unused,
        ),
        Err(TreeAciError::ResourceLimit {
            resource: "local matrix elements",
            requested: 4,
            limit: 3
        })
    ));
}
