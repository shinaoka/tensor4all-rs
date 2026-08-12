//! Tests for TensorTrain inner product operations

use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::IdxTensor;
use tensor4all_itensorlike::TensorTrain;

/// Test inner product of empty tensor trains
#[test]
fn test_inner_empty_tensor_trains() {
    let tt1 = TensorTrain::new(vec![]).unwrap();
    let tt2 = TensorTrain::new(vec![]).unwrap();

    let result = tt1.inner(&tt2).unwrap();
    assert_eq!(result.real(), 0.0);
}

/// Test inner product of single-site tensor trains
#[test]
fn test_inner_single_site() {
    let site = Index::new_dyn(2);
    let link_left = Index::new_dyn(1);
    let link_right = Index::new_dyn(1);

    // Create [1, 0] as tensor with shape [1, 2, 1]
    let t1 = IdxTensor::from_dense(
        vec![link_left.clone(), site.clone(), link_right.clone()],
        vec![1.0, 0.0],
    )
    .unwrap();

    // Create [0, 1] as tensor with shape [1, 2, 1]
    let t2 = IdxTensor::from_dense(
        vec![link_left.clone(), site.clone(), link_right.clone()],
        vec![0.0, 1.0],
    )
    .unwrap();

    let tt1 = TensorTrain::new(vec![t1.clone()]).unwrap();
    let tt2 = TensorTrain::new(vec![t2.clone()]).unwrap();

    // Orthogonal vectors: inner product should be 0
    let result = tt1.inner(&tt2).unwrap();
    assert!((result.real() - 0.0).abs() < 1e-10);

    // Same vector: inner product should be 1
    let tt3 = TensorTrain::new(vec![t1.clone()]).unwrap();
    let result2 = tt1.inner(&tt3).unwrap();
    assert!((result2.real() - 1.0).abs() < 1e-10);
}
