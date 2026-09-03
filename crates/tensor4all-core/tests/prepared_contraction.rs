use tensor4all_core::{
    contract, ContractionOptions, DynId, DynIndex, IdxTensor, IdxTensorError, Index,
    PreparedContraction, TagSet,
};
use tensor4all_tensorbackend::StorageKind;

fn dense(indices: Vec<DynIndex>, values: Vec<f64>) -> IdxTensor {
    IdxTensor::from_dense(indices, values).unwrap()
}

fn three_operand_fixture() -> (Vec<DynIndex>, Vec<IdxTensor>) {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let a = dense(vec![i.clone(), k.clone()], vec![1.0, 2.0, 3.0, 4.0]);
    let b = dense(vec![k.clone(), j.clone()], vec![5.0, 6.0, 7.0, 8.0]);
    let c = dense(vec![j.clone()], vec![1.0, 2.0]);
    (vec![i, j, k], vec![a, b, c])
}

#[test]
fn prepared_dense_nary_matches_ordinary_for_new_values_and_mixed_dtype() {
    let (indices, tensors) = three_operand_fixture();
    let plan = PreparedContraction::new(
        &[&tensors[0], &tensors[1], &tensors[2]],
        ContractionOptions::new(),
    )
    .unwrap();
    let a = IdxTensor::from_dense(
        vec![indices[0].clone(), indices[2].clone()],
        vec![1.0_f32, 3.0, 2.0, 4.0],
    )
    .unwrap();
    let b = dense(
        vec![indices[2].clone(), indices[1].clone()],
        vec![2.0, 0.0, 1.0, 3.0],
    );
    let c = dense(vec![indices[1].clone()], vec![4.0, 5.0]);

    let prepared = plan.execute(&[&a, &b, &c]).unwrap();
    let ordinary = contract(&[&a, &b, &c]).unwrap();
    assert!(prepared.isapprox(&ordinary, 1.0e-12, 0.0).unwrap());
}

#[test]
fn prepared_retained_contraction_preserves_output_order_and_values() {
    let batch = DynIndex::new_dyn(2);
    let left = DynIndex::new_dyn(2);
    let contracted = DynIndex::new_dyn(3);
    let right = DynIndex::new_dyn(2);
    let a = dense(
        vec![batch.clone(), left.clone(), contracted.clone()],
        (1..=12).map(f64::from).collect(),
    );
    let b = dense(
        vec![batch.clone(), contracted, right.clone()],
        (1..=12).map(|value| f64::from(value) / 2.0).collect(),
    );
    let retained = [batch.clone()];
    let options = ContractionOptions::new().with_retain_indices(&retained);
    let plan = PreparedContraction::new(&[&a, &b], options).unwrap();

    let prepared = plan.execute(&[&a, &b]).unwrap();
    let ordinary = tensor4all_core::contract_with_options(&[&a, &b], options).unwrap();
    assert_eq!(prepared.indices(), &[batch, left, right]);
    assert_eq!(
        prepared.to_vec::<f64>().unwrap(),
        ordinary.to_vec::<f64>().unwrap()
    );
}

#[test]
fn prepared_structured_contraction_stays_compact() {
    let i = DynIndex::new_dyn(3);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(3);
    let output = DynIndex::new_dyn(3);
    let a = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let b = IdxTensor::from_diag(vec![k, j.clone()], vec![4.0, 5.0, 6.0]).unwrap();
    let c = IdxTensor::from_diag(vec![j, output.clone()], vec![1.0; 3]).unwrap();
    let plan = PreparedContraction::new(&[&a, &b, &c], ContractionOptions::new()).unwrap();

    let result = plan.execute(&[&a, &b, &c]).unwrap();
    assert_eq!(result.indices(), &[i, output]);
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![4.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 18.0]
    );
}

#[test]
fn prepared_structured_contraction_supports_mixed_dtype() {
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let prototype_a = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![0.0_f64; 2]).unwrap();
    let b = IdxTensor::from_diag(vec![k.clone(), j.clone()], vec![3.0_f64, 4.0]).unwrap();
    let plan = PreparedContraction::new(&[&prototype_a, &b], ContractionOptions::new()).unwrap();
    let a = IdxTensor::from_diag(vec![i, k], vec![1.0_f32, 2.0]).unwrap();

    let result = plan.execute(&[&a, &b]).unwrap();
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(result.to_vec::<f64>().unwrap(), vec![3.0, 0.0, 0.0, 8.0]);
}

#[test]
fn prepared_execution_preserves_gradients() {
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let prototype_a = dense(vec![i.clone(), k.clone()], vec![0.0; 4]);
    let prototype_b = dense(vec![k.clone(), j.clone()], vec![0.0; 4]);
    let prototype_c = dense(vec![j.clone()], vec![0.0; 2]);
    let plan = PreparedContraction::new(
        &[&prototype_a, &prototype_b, &prototype_c],
        ContractionOptions::new(),
    )
    .unwrap();
    let a = dense(vec![i, k.clone()], vec![1.0, 2.0, 3.0, 4.0])
        .enable_grad()
        .unwrap();
    let b = dense(vec![k, j.clone()], vec![1.0; 4]);
    let c = dense(vec![j], vec![1.0; 2]);

    plan.execute(&[&a, &b, &c])
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(
        a.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![2.0; 4]
    );
}

#[test]
fn prepared_structured_contraction_preserves_gradients() {
    let i = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let output = DynIndex::new_dyn(2);
    let prototype_a = IdxTensor::from_diag(vec![i.clone(), k.clone()], vec![0.0_f64; 2]).unwrap();
    let b = IdxTensor::from_diag(vec![k.clone(), j.clone()], vec![3.0_f64, 4.0]).unwrap();
    let c = IdxTensor::from_diag(vec![j, output], vec![5.0_f64, 6.0]).unwrap();
    let plan =
        PreparedContraction::new(&[&prototype_a, &b, &c], ContractionOptions::new()).unwrap();
    let a = IdxTensor::from_diag(vec![i, k], vec![1.0_f64, 2.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    plan.execute(&[&a, &b, &c])
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let gradient = a.grad().unwrap().unwrap();
    assert_eq!(
        gradient.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(
        gradient.to_vec::<f64>().unwrap(),
        vec![15.0, 0.0, 0.0, 24.0]
    );
}

#[test]
fn prepared_execution_rejects_count_dimension_index_and_layout_mismatches() {
    let shared = Index::new(DynId(700), 2);
    let left = dense(vec![shared.clone()], vec![1.0, 2.0]);
    let right = dense(vec![shared.clone()], vec![3.0, 4.0]);
    let plan = PreparedContraction::new(&[&left, &right], ContractionOptions::new()).unwrap();

    assert!(matches!(
        plan.execute(&[&left]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        plan.execute(&[]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));

    let different_dim = Index::new(DynId(700), 3);
    let dimension_mismatch = dense(vec![different_dim], vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        plan.execute(&[&dimension_mismatch, &right]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));

    let primed = dense(vec![shared.prime()], vec![1.0, 2.0]);
    assert!(matches!(
        plan.execute(&[&primed, &right]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));

    let tagged_index = Index::new_with_tags(
        DynId(700),
        2,
        TagSet::from_str("prepared-mismatch").unwrap(),
    );
    let tagged = dense(vec![tagged_index], vec![1.0, 2.0]);
    assert!(matches!(
        plan.execute(&[&tagged, &right]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));

    let diagonal =
        IdxTensor::from_diag(vec![shared.clone(), shared.prime()], vec![1.0, 2.0]).unwrap();
    let dense_layout = dense(diagonal.indices().to_vec(), vec![1.0, 0.0, 0.0, 2.0]);
    let layout_plan = PreparedContraction::new(&[&diagonal], ContractionOptions::new()).unwrap();
    assert!(matches!(
        layout_plan.execute(&[&dense_layout]),
        Err(IdxTensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn prepared_contraction_rejects_empty_disconnected_and_missing_retained_inputs() {
    assert!(PreparedContraction::new(&[], ContractionOptions::new()).is_err());

    let a = dense(vec![DynIndex::new_dyn(2)], vec![1.0, 2.0]);
    let b = dense(vec![DynIndex::new_dyn(2)], vec![3.0, 4.0]);
    assert!(PreparedContraction::new(&[&a, &b], ContractionOptions::new()).is_err());

    let missing = [DynIndex::new_dyn(2)];
    assert!(PreparedContraction::new(
        &[&a],
        ContractionOptions::new().with_retain_indices(&missing),
    )
    .is_err());
}
