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
