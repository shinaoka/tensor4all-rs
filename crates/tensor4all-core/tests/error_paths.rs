use std::sync::Arc;

use tensor4all_core::block_tensor::BlockTensor;
use tensor4all_core::col_major_array::{ColMajorArrayError, ColMajorArrayRef};
use tensor4all_core::index_like::IndexLike;
use tensor4all_core::{
    compute_permutation_from_indices, diag_idx_tensor, DynIndex, IdxTensor, TensorContractionLike,
};
use tensor4all_tensorbackend::Storage;

#[test]
fn permutation_helper_rejects_non_permutation() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let k = DynIndex::new_dyn(2);

    let err = compute_permutation_from_indices(&[i, j.clone()], &[j, k]).unwrap_err();

    assert!(err.to_string().contains("permutation"));
}

#[test]
fn tensor_new_rejects_duplicate_indices() {
    let i = DynIndex::new_dyn(2);
    let storage = Arc::new(Storage::from_dense_col_major(vec![1.0_f64, 2.0], &[2]).unwrap());

    let err = IdxTensor::new(vec![i.clone(), i], storage).unwrap_err();

    assert!(err.to_string().contains("unique"));
}

#[test]
fn tensor_permute_rejects_invalid_axis_order() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let err = tensor.permute_indices(&[j, i.clone(), i]).unwrap_err();

    assert!(err.to_string().contains("length"));
}

#[test]
fn tensor_contract_rejects_mismatched_common_dimension() {
    let shared_left = DynIndex::new_dyn(2);
    let shared_right = DynIndex::new_with_tags(*shared_left.id(), 3, shared_left.tags().clone());
    let a = IdxTensor::from_dense(vec![shared_left], vec![1.0_f64, 2.0]).unwrap();
    let b = IdxTensor::from_dense(vec![shared_right], vec![1.0_f64, 2.0, 3.0]).unwrap();

    let err = a.contract_pair(&b).unwrap_err();

    let message = err.to_string();
    assert!(
        message.contains("contraction")
            || message.contains("Dimension mismatch")
            || message.contains("unique")
    );
}

#[test]
fn random_rejects_duplicate_indices() {
    let i = DynIndex::new_dyn(2);
    let mut rng = rand::rng();

    let err = IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), i]).unwrap_err();

    assert!(err.to_string().contains("unique"));
}

#[test]
fn block_tensor_new_rejects_shape_mismatch() {
    let i = DynIndex::new_dyn(2);
    let block = IdxTensor::from_dense(vec![i], vec![1.0_f64, 2.0]).unwrap();

    let err = BlockTensor::new(vec![block], (2, 1)).unwrap_err();

    assert!(err.to_string().contains("Block count mismatch"));
}

#[test]
fn col_major_ref_new_rejects_overflow_shape() {
    let data = [0usize];
    let err = ColMajorArrayRef::new(&data, &[usize::MAX, 2]).unwrap_err();

    assert_eq!(
        err,
        ColMajorArrayError::ShapeOverflow {
            shape: vec![usize::MAX, 2]
        }
    );
}

#[test]
fn diag_tensor_helper_rejects_dimension_mismatch() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let err = diag_idx_tensor(vec![i, j], vec![1.0, 2.0]).unwrap_err();

    assert!(err.to_string().contains("same dimension"));
}
