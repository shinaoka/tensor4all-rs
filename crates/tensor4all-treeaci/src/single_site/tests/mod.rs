use std::cell::Cell;

use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use super::evaluate_single_site;
use crate::{TreeAciOptions, TreeAciTermination};

#[test]
fn multiple_axes_and_different_input_axis_orders_are_evaluated_exactly_once() {
    let a = DynIndex::new_dyn(2);
    let b = DynIndex::new_dyn(3);
    let first = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![a.clone(), b.clone()],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap()],
        vec![7usize],
    )
    .unwrap();
    let second = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(
            vec![b.clone(), a.clone()],
            vec![0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        )
        .unwrap()],
        vec![7usize],
    )
    .unwrap();
    let calls = Cell::new(0);
    let mut multiply = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + 1);
        for (point, value) in output.iter_mut().enumerate() {
            *value = batch.get(0, point)? * batch.get(1, point)?;
        }
        Ok(())
    };

    let result =
        evaluate_single_site(&[first, second], &TreeAciOptions::default(), &mut multiply).unwrap();

    assert_eq!(calls.get(), 1);
    assert_eq!(result.tree.node_names(), vec![7]);
    assert_eq!(result.tree.to_dense().unwrap().indices(), &[a, b]);
    assert_eq!(
        result.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![0.0, 20.0, 3.0, 44.0, 10.0, 72.0]
    );
    assert!(result.max_ranks.is_empty());
    assert!(result.max_errors.is_empty());
    assert_eq!(result.termination, TreeAciTermination::Converged);
    assert_eq!(result.diagnostics.evaluated_points, 6);
}

#[test]
fn scalar_node_without_physical_indices_is_one_point() {
    let input = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(Vec::new(), vec![3.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let mut square = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = batch.get(0, 0)?.powi(2);
        Ok(())
    };
    let result = evaluate_single_site(&[input], &TreeAciOptions::default(), &mut square).unwrap();
    assert_eq!(
        result.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![9.0]
    );
    assert_eq!(result.diagnostics.evaluated_points, 1);
}

#[test]
fn public_single_site_entry_skips_general_tree_state_construction() {
    let site = DynIndex::new_dyn(2);
    let input = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![2.0_f64, 3.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    crate::state::profile_debug_stats::reset();

    let result = crate::tree_elementwise_batched::<f64, _, _>(
        |batch, output| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = batch.get(0, point)?.powi(2);
            }
            Ok(())
        },
        &[input],
        &TreeAciOptions::default(),
    )
    .unwrap();

    assert_eq!(
        result.tree.to_dense().unwrap().to_vec::<f64>().unwrap(),
        vec![4.0, 9.0]
    );
    let stats = crate::state::profile_debug_stats::snapshot();
    assert_eq!(stats.preparation, std::time::Duration::ZERO);
    assert_eq!(stats.output, std::time::Duration::ZERO);
    assert_eq!(stats.bootstrap, std::time::Duration::ZERO);
    assert_eq!(stats.frames, std::time::Duration::ZERO);
}

#[test]
fn public_single_site_entry_still_validates_an_explicit_guess() {
    let input = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![2.0_f64, 3.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let incompatible_guess = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f64, 1.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let options = TreeAciOptions {
        initial_guess: Some(incompatible_guess),
        ..TreeAciOptions::default()
    };

    assert!(matches!(
        crate::tree_elementwise_batched::<f64, _, _>(
            |batch, output| {
                output[0] = batch.get(0, 0)?;
                Ok(())
            },
            &[input],
            &options,
        ),
        Err(crate::TreeAciError::InvalidInitialGuess { .. })
    ));
}

/// The exact single-node path is charged against `max_working_bytes` like every
/// other allocation site.
///
/// It bypasses the sweep entirely, so it used to allocate `inputs * local_dim`
/// with a bare multiplication and no budget check at all — a caller could set a
/// hard ceiling and this one public entry would ignore it.
#[test]
fn the_exact_path_respects_the_working_budget() {
    let site = DynIndex::new_dyn(4);
    let tree = TreeTN::from_tensors(
        vec![IdxTensor::from_dense(vec![site], vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
        vec![0usize],
    )
    .unwrap();
    let mut square = |batch: crate::TreeElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate() {
            let x = batch.get(0, point)?;
            *value = x * x;
        }
        Ok(())
    };

    // The four-point input and output buffers coexist: 2 * 4 * 8 = 64 bytes.
    // The element ceilings are pinned so that this budget charges only those
    // buffers; derived ceilings would follow the budget down and refuse the
    // four-element core during preparation instead.
    let generous = TreeAciOptions::<usize> {
        max_working_bytes: 64,
        max_local_matrix_elements: Some(1 << 24),
        max_core_elements: Some(1 << 24),
        max_frame_elements: Some(1 << 24),
        ..TreeAciOptions::default()
    };
    assert!(evaluate_single_site(std::slice::from_ref(&tree), &generous, &mut square).is_ok());

    let tight = TreeAciOptions::<usize> {
        max_working_bytes: 63,
        ..generous.clone()
    };
    let error = evaluate_single_site(std::slice::from_ref(&tree), &tight, &mut square)
        .expect_err("a 63-byte ceiling must reject two 32-byte buffers");
    assert!(
        matches!(
            error,
            crate::TreeAciError::ResourceLimit {
                resource: "working bytes",
                requested: 64,
                limit: 63,
            }
        ),
        "unexpected error: {error}"
    );
}
