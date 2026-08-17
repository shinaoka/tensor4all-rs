use tensor4all_core::{DynIndex, IdxTensor};

#[test]
fn stack_along_new_index_uses_trailing_column_major_batch_axis() {
    let batch = DynIndex::new_dyn(2);
    let i = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![i.clone()], vec![3.0_f64, 4.0]).unwrap();

    let stacked = IdxTensor::stack_along_new_index(&[&a, &b], batch.clone(), -1).unwrap();

    assert_eq!(stacked.indices(), &[i, batch]);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn index_select_replaces_trailing_index_and_allows_repeated_positions() {
    let source_batch = DynIndex::new_dyn(3);
    let target_batch = DynIndex::new_dyn(4);
    let i = DynIndex::new_dyn(2);
    let source = IdxTensor::from_dense(
        vec![i.clone(), source_batch.clone()],
        vec![10.0_f64, 11.0, 20.0, 21.0, 30.0, 31.0],
    )
    .unwrap();

    let selected = source
        .index_select(&source_batch, target_batch.clone(), &[2, 0, 2, 1])
        .unwrap();

    assert_eq!(selected.indices(), &[i, target_batch]);
    assert_eq!(
        selected.to_vec::<f64>().unwrap(),
        vec![30.0, 31.0, 10.0, 11.0, 30.0, 31.0, 20.0, 21.0]
    );
}

#[test]
fn index_select_backward_scatter_adds_repeated_positions() {
    let source = DynIndex::new_dyn(3);
    let target = DynIndex::new_dyn(3);
    let x = IdxTensor::from_dense(vec![source.clone()], vec![1.0_f64, 2.0, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let weights = IdxTensor::from_dense(vec![target.clone()], vec![10.0_f64, 20.0, 30.0]).unwrap();

    let y = x.index_select(&source, target, &[1, 1, 2]).unwrap();
    let loss = y.inner_product(&weights).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.indices(), &[source]);
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![0.0, 30.0, 30.0]);
}

#[test]
fn stack_along_new_index_backward_splits_cotangent_to_inputs() {
    let batch = DynIndex::new_dyn(2);
    let x0 = IdxTensor::scalar(2.0).unwrap().enable_grad().unwrap();
    let x1 = IdxTensor::scalar(3.0).unwrap().enable_grad().unwrap();
    let weights = IdxTensor::from_dense(vec![batch.clone()], vec![10.0_f64, 20.0]).unwrap();

    let stacked = IdxTensor::stack_along_new_index(&[&x0, &x1], batch, -1).unwrap();
    let loss = stacked.inner_product(&weights).unwrap();
    loss.backward().unwrap();

    let grad0 = x0.grad().unwrap().unwrap();
    let grad1 = x1.grad().unwrap().unwrap();
    assert!((grad0.only().unwrap().real() - 10.0).abs() < 1.0e-12);
    assert!((grad1.only().unwrap().real() - 20.0).abs() < 1.0e-12);
}

#[test]
fn stack_along_new_index_rejects_tracked_compact_storage() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(1);
    let diag = IdxTensor::from_diag(vec![i, j], vec![1.0_f64, 2.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let err = IdxTensor::stack_along_new_index(&[&diag], batch, -1).unwrap_err();

    assert!(err.to_string().contains("structured AD"));
}

#[test]
fn select_indices_rejects_tracked_compact_storage() {
    let source = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let diag = IdxTensor::from_diag(vec![source.clone(), j], vec![1.0_f64, 2.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let err = diag.select_indices(&[source], &[0]).unwrap_err();
    assert!(err.to_string().contains("structured AD"));
}

#[test]
fn direct_sum_rejects_tracked_inputs_before_detaching() {
    use tensor4all_core::TensorContractionLike;

    let a_index = DynIndex::new_dyn(2);
    let b_index = DynIndex::new_dyn(2);
    let common_a = DynIndex::new_dyn(2);
    let common_b = DynIndex::new_dyn(2);
    let a = IdxTensor::from_dense(vec![a_index, common_a], vec![1.0_f64; 4])
        .unwrap()
        .enable_grad()
        .unwrap();
    let b = IdxTensor::from_dense(vec![b_index, common_b], vec![2.0_f64; 4]).unwrap();

    let err = a
        .direct_sum(&b, &[(a.indices()[1].clone(), b.indices()[1].clone())])
        .unwrap_err();
    assert!(err.to_string().contains("tracked"));
}

#[test]
fn index_select_rejects_tracked_compact_storage() {
    let source = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let target = DynIndex::new_dyn(1);
    let diag = IdxTensor::from_diag(vec![source.clone(), j], vec![1.0_f64, 2.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let err = diag.index_select(&source, target, &[0]).unwrap_err();

    assert!(err.to_string().contains("structured AD"));
}
