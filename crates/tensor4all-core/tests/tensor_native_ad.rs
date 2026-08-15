use num_complex::Complex64;
use tensor4all_core::{
    contract, contract_with_options, AnyScalar, ContractionOptions, IdxTensor, Index,
    TensorContractionLike,
};
use tensor4all_tensorbackend::{Storage, StorageKind};

#[test]
fn plain_dense_storage_auto_seeds_native_payload() {
    let i = Index::new_dyn(2);
    let tensor = IdxTensor::from_storage(
        vec![i],
        Storage::from_dense_col_major(vec![1.0, 2.0], &[2])
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn plain_diag_storage_preserves_diag_metadata() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = IdxTensor::from_storage(
        vec![i, j],
        Storage::from_diag_col_major(vec![1.0, 2.0, 3.0], 2)
            .map(std::sync::Arc::new)
            .unwrap(),
    )
    .unwrap();

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
    assert!(tensor.is_diag());
}

#[test]
fn contraction_without_grad_returns_rank_zero_scalar() {
    let i = Index::new_dyn(3);
    let a = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let ones = IdxTensor::from_dense(vec![i], vec![1.0, 1.0, 1.0]).unwrap();

    let result = contract(&[&a, &ones]).unwrap();

    assert!(result.indices().is_empty());
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![6.0]);
}

#[test]
fn tracked_complex_conjugation_preserves_values_and_gradient_path() {
    let i = Index::new_dyn(2);
    let x = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
    )
    .unwrap()
    .enable_grad()
    .unwrap();

    let conjugated = x.conj();
    assert!(conjugated.tracks_grad());
    assert_eq!(
        conjugated.to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(1.0, -2.0), Complex64::new(-3.0, -4.0)]
    );

    let loss = conjugated.sum().unwrap();
    assert!(loss.tracks_grad());
    loss.backward().unwrap();
    assert!(x.grad().unwrap().is_some());
}

#[test]
fn tracked_complex_axpby_with_real_coefficients_preserves_gradients() {
    let i = Index::new_dyn(2);
    let x = IdxTensor::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)],
    )
    .unwrap()
    .enable_grad()
    .unwrap();
    let y = IdxTensor::from_dense(
        vec![i],
        vec![Complex64::new(5.0, -1.0), Complex64::new(2.0, 3.0)],
    )
    .unwrap()
    .enable_grad()
    .unwrap();

    let combined = x
        .axpby(AnyScalar::new_real(0.5), &y, AnyScalar::new_real(-0.25))
        .unwrap();
    assert!(combined.tracks_grad());
    combined.sum().unwrap().backward().unwrap();

    assert_eq!(
        x.grad().unwrap().unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(0.5, 0.0); 2]
    );
    assert_eq!(
        y.grad().unwrap().unwrap().to_vec::<Complex64>().unwrap(),
        vec![Complex64::new(-0.25, 0.0); 2]
    );
}

#[test]
fn backward_accumulates_until_clear_grad() {
    let i = Index::new_dyn(3);
    let x = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0, 3.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let ones = IdxTensor::from_dense(vec![i], vec![1.0, 1.0, 1.0]).unwrap();

    let loss = contract(&[&x, &ones]).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![1.0, 1.0, 1.0]);

    let loss = contract(&[&x, &ones]).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![2.0, 2.0, 2.0]);

    x.clear_grad().unwrap();
    assert!(x.grad().unwrap().is_none());
}

#[test]
fn general_structured_grad_preserves_input_axis_classes() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .map(std::sync::Arc::new)
    .unwrap();
    let x = IdxTensor::from_storage(vec![i.clone(), j.clone(), k.clone()], storage)
        .unwrap()
        .enable_grad()
        .unwrap();
    let ones = IdxTensor::from_dense(vec![i, j, k], vec![1.0; 12]).unwrap();

    let loss = contract(&[&x, &ones]).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.storage().unwrap().axis_classes(), &[0, 1, 0]);
    assert_eq!(
        grad.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(
        grad.storage().unwrap().payload_f64_col_major_vec().unwrap(),
        vec![1.0; 6]
    );
}

#[test]
fn tracks_grad_and_detach_report_leaf_state() {
    let scalar = IdxTensor::scalar(2.0).unwrap();
    assert!(!scalar.tracks_grad());

    let tracked = scalar.enable_grad();
    let tracked = tracked.unwrap();
    assert!(tracked.tracks_grad());

    let detached = tracked.detach().unwrap();
    assert!(!detached.tracks_grad());
    assert!(tracked.tracks_grad());
}

#[test]
fn clone_shares_tracked_leaf_gradient_slot() {
    let x = IdxTensor::scalar(2.0).unwrap().enable_grad().unwrap();
    let alias = x.clone();

    let loss = x.contract_pair(&alias).unwrap();
    loss.backward().unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_alias = alias.grad().unwrap().unwrap();
    assert!((grad_x.only().unwrap().real() - 4.0).abs() < 1e-12);
    assert!((grad_alias.only().unwrap().real() - 4.0).abs() < 1e-12);
}

#[test]
fn retained_multi_contraction_preserves_grad_path() {
    let batch = Index::new_dyn(2);
    let i = Index::new_dyn(2);
    let k = Index::new_dyn(3);
    let j = Index::new_dyn(2);

    let x = IdxTensor::from_dense(
        vec![batch.clone(), i.clone(), k.clone()],
        (1..=12).map(|value| value as f64).collect(),
    )
    .unwrap()
    .enable_grad()
    .unwrap();
    let y =
        IdxTensor::from_dense(vec![batch.clone(), k.clone(), j.clone()], vec![1.0; 12]).unwrap();
    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);

    let result = contract_with_options(&[&x, &y], options).unwrap();
    assert_eq!(result.dims(), vec![2, 2, 2]);
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![15.0, 18.0, 21.0, 24.0, 15.0, 18.0, 21.0, 24.0]
    );

    let ones = IdxTensor::from_dense(result.indices().to_vec(), vec![1.0; 8]).unwrap();
    let loss = contract(&[&result, &ones]).unwrap();
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert_eq!(grad.dims(), vec![2, 2, 3]);
    assert_eq!(grad.to_vec::<f64>().unwrap(), vec![2.0; 12]);
}

#[test]
fn mixed_nary_copy_selector_contraction_stays_compact() {
    let bond = 128;
    let site = Index::new_dyn(3);
    let a = IdxTensor::from_copy_selector(
        Index::new_dyn(bond),
        site.clone(),
        Index::new_dyn(bond),
        1,
        2.0_f32,
    )
    .unwrap();
    let b = IdxTensor::from_copy_selector(
        a.indices()[2].clone(),
        site.clone(),
        Index::new_dyn(bond),
        1,
        3.0_f64,
    )
    .unwrap();
    let c = IdxTensor::from_copy_selector(
        b.indices()[2].clone(),
        site,
        Index::new_dyn(bond),
        1,
        Complex64::new(5.0, 0.0),
    )
    .unwrap();

    let result = contract_with_options(
        &[&a, &b, &c],
        ContractionOptions::new().with_retain_indices(&[a.indices()[1].clone()]),
    )
    .unwrap();
    let storage = result.storage().unwrap();
    assert_eq!(result.storage_kind(), StorageKind::Structured);
    assert_eq!(storage.axis_classes(), &[0, 1, 0]);
    assert_eq!(storage.payload_dims(), &[bond, 3]);
    assert_eq!(storage.payload_len(), bond * 3);
    assert_eq!(
        storage.scalar_at(&[0, 1]).unwrap().as_c64(),
        Some(Complex64::new(30.0, 0.0))
    );
}

#[test]
fn tracked_nary_copy_selector_contraction_preserves_compact_gradient() {
    let bond = 4;
    let site = Index::new_dyn(3);
    let a = IdxTensor::from_copy_selector(
        Index::new_dyn(bond),
        site.clone(),
        Index::new_dyn(bond),
        1,
        2.0_f64,
    )
    .unwrap()
    .enable_grad()
    .unwrap();
    let b = IdxTensor::from_copy_selector(
        a.indices()[2].clone(),
        site.clone(),
        Index::new_dyn(bond),
        1,
        3.0_f64,
    )
    .unwrap();
    let c = IdxTensor::from_copy_selector(
        b.indices()[2].clone(),
        site.clone(),
        Index::new_dyn(bond),
        1,
        5.0_f64,
    )
    .unwrap();

    let result = contract_with_options(
        &[&a, &b, &c],
        ContractionOptions::new().with_retain_indices(&[a.indices()[1].clone()]),
    )
    .unwrap();
    let ones =
        IdxTensor::from_dense(result.indices().to_vec(), vec![1.0_f64; bond * 3 * bond]).unwrap();
    contract(&[&result, &ones]).unwrap().backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    let grad_storage = grad.storage().unwrap();
    assert_eq!(grad_storage.axis_classes(), &[0, 1, 0]);
    assert_eq!(grad_storage.payload_dims(), &[bond, 3]);
    assert_eq!(grad_storage.payload_len(), bond * 3);
    assert_eq!(grad_storage.scalar_at(&[0, 1]).unwrap().real(), 15.0);
}

#[test]
fn structured_retained_multi_contraction_preserves_grad_path() {
    let batch = Index::new_dyn(2);
    let i = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let storage = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .map(std::sync::Arc::new)
    .unwrap();
    let x = IdxTensor::from_storage(vec![batch.clone(), i.clone(), k.clone()], storage)
        .unwrap()
        .enable_grad()
        .unwrap();
    let y = IdxTensor::from_dense(vec![batch.clone(), k.clone(), j.clone()], vec![1.0; 8]).unwrap();
    let retain_indices = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retain_indices);

    let result = contract_with_options(&[&x, &y], options).unwrap();
    assert_eq!(result.dims(), vec![2, 3, 2]);
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let ones = IdxTensor::from_dense(result.indices().to_vec(), vec![1.0; 12]).unwrap();
    let loss = contract(&[&result, &ones]).unwrap();
    loss.backward().unwrap();
    assert_eq!(
        x.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0],
    );
}
