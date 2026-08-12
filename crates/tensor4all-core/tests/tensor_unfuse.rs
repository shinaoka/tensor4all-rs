use num_complex::Complex64;
use tensor4all_core::{
    DynIndex, IdxTensor, IndexLike, LinearizationOrder, TensorContractionLike, TensorIndex,
};

#[test]
fn unfuse_index_column_major_preserves_column_major_layout() {
    let fused = DynIndex::new_dyn(4);
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let unfused = tensor
        .unfuse_index(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    let expected = IdxTensor::from_dense(vec![left, right], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(unfused.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn unfuse_index_row_major_reorders_fused_axis_meaning() {
    let fused = DynIndex::new_dyn(4);
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let unfused = tensor
        .unfuse_index(
            &fused,
            &[left.clone(), right.clone()],
            LinearizationOrder::RowMajor,
        )
        .unwrap();

    let expected = IdxTensor::from_dense(vec![left, right], vec![1.0, 3.0, 2.0, 4.0]).unwrap();
    assert!(unfused.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn unfuse_index_rejects_dimension_mismatch() {
    let fused = DynIndex::new_dyn(4);
    let tensor = IdxTensor::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let err = tensor
        .unfuse_index(
            &fused,
            &[DynIndex::new_dyn(2), DynIndex::new_dyn(3)],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("product of new index dimensions must match the replaced index dimension"));
}

#[test]
fn fuse_indices_column_major_roundtrips_unfuse_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, j], LinearizationOrder::ColumnMajor)
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0).unwrap());
}

#[test]
fn fuse_indices_backward_preserves_source_gradient() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone(), k.clone()],
        (0..12).map(|x| x as f64).collect(),
    )
    .unwrap()
    .enable_grad()
    .unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert!(fused_tensor.tracks_grad());

    let weights = IdxTensor::from_dense(
        vec![fused.clone(), k.clone()],
        (1..=12).map(|x| x as f64).collect(),
    )
    .unwrap();
    let loss = fused_tensor.inner_product(&weights).unwrap();
    loss.backward().unwrap();

    let grad = tensor.grad().unwrap().unwrap();
    let expected =
        IdxTensor::from_dense(vec![i, j, k], (1..=12).map(|x| x as f64).collect()).unwrap();
    assert!(grad.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn fuse_indices_trait_dispatch_on_idx_tensor_uses_old_index_order() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = <IdxTensor as TensorContractionLike>::fuse_indices(
        &tensor,
        &[j.clone(), i.clone()],
        fused.clone(),
        LinearizationOrder::ColumnMajor,
    )
    .unwrap();

    assert_eq!(fused_tensor.indices(), &[fused.clone(), k.clone()]);
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[j, i], LinearizationOrder::ColumnMajor)
        .unwrap()
        .permuteinds(tensor.indices())
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0).unwrap());
}

#[test]
fn unfuse_index_backward_preserves_source_gradient() {
    let fused = DynIndex::new_link(6).unwrap();
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(
        vec![fused.clone(), k.clone()],
        (0..12).map(|x| x as f64).collect(),
    )
    .unwrap()
    .enable_grad()
    .unwrap();

    let unfused = tensor
        .unfuse_index(
            &fused,
            &[i.clone(), j.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert!(unfused.tracks_grad());

    let weights =
        IdxTensor::from_dense(vec![i, j, k.clone()], (1..=12).map(|x| x as f64).collect()).unwrap();
    let loss = unfused.inner_product(&weights).unwrap();
    loss.backward().unwrap();

    let grad = tensor.grad().unwrap().unwrap();
    let expected =
        IdxTensor::from_dense(vec![fused, k], (1..=12).map(|x| x as f64).collect()).unwrap();
    assert!(grad.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn fuse_indices_row_major_roundtrips_unfuse_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::RowMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.dims(), vec![6, 2]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, j], LinearizationOrder::RowMajor)
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0).unwrap());
}

#[test]
fn fuse_indices_preserves_complex_payload_roundtrip() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let fused = DynIndex::new_link(6).unwrap();
    let data: Vec<Complex64> = (0..6)
        .map(|x| Complex64::new(x as f64, -(x as f64)))
        .collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], data.clone()).unwrap();

    let roundtrip = tensor
        .fuse_indices(
            &[i.clone(), j.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap()
        .unfuse_index(&fused, &[i, j], LinearizationOrder::ColumnMajor)
        .unwrap();

    assert!(roundtrip.is_complex());
    assert_eq!(roundtrip.to_vec::<Complex64>().unwrap(), data);
}

#[test]
fn fuse_indices_supports_non_adjacent_axes() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(4).unwrap();
    let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], data).unwrap();

    let fused_tensor = tensor
        .fuse_indices(
            &[i.clone(), k.clone()],
            fused.clone(),
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();
    assert_eq!(fused_tensor.indices(), &[fused.clone(), j.clone()]);

    let roundtrip = fused_tensor
        .unfuse_index(&fused, &[i, k], LinearizationOrder::ColumnMajor)
        .unwrap()
        .permuteinds(tensor.indices())
        .unwrap();
    assert!(roundtrip.isapprox(&tensor, 1e-12, 0.0).unwrap());
}

#[test]
fn fuse_indices_rejects_empty_old_indices() {
    let i = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(2).unwrap();
    let tensor = IdxTensor::from_dense(vec![i], vec![1.0, 2.0]).unwrap();

    let err = tensor
        .fuse_indices(&[], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("fuse_indices requires at least one index to fuse"));
}

#[test]
fn fuse_indices_rejects_duplicate_old_indices() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let fused = DynIndex::new_link(4).unwrap();
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i.clone(), i], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err.to_string().contains("duplicate index in old_indices"));
}

#[test]
fn fuse_indices_rejects_missing_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let missing = DynIndex::new_dyn(2);
    let fused = DynIndex::new_link(6).unwrap();
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i, missing], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err.to_string().contains("not found in tensor"));
}

#[test]
fn fuse_indices_rejects_dimension_mismatch() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let fused = DynIndex::new_link(5).unwrap();
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let err = tensor
        .fuse_indices(&[i, j], fused, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("product of old index dimensions must match the replacement index dimension"));
}

#[test]
fn fuse_indices_rejects_same_id_with_wrong_dimension() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let same_id_wrong_dim = DynIndex::new(*i.id(), 3);
    let fused = DynIndex::new_link(9).unwrap();
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let err = tensor
        .fuse_indices(
            &[same_id_wrong_dim, j],
            fused,
            LinearizationOrder::ColumnMajor,
        )
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("old index dimension does not match tensor axis dimension"));
}

#[test]
fn fuse_indices_rejects_duplicate_result_index() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let err = tensor
        .fuse_indices(&[i], j, LinearizationOrder::ColumnMajor)
        .unwrap_err();

    assert!(err
        .to_string()
        .contains("fuse_indices result would contain duplicate index"));
}

#[test]
fn fuse_indices_allows_same_id_prime_pair_in_result() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let result = tensor
        .fuse_indices(&[i], j.prime(), LinearizationOrder::ColumnMajor)
        .unwrap();

    assert_eq!(result.external_indices(), vec![j.prime(), j]);
}

#[test]
fn unfuse_index_selects_exact_same_id_prime_index() {
    let fused = DynIndex::new_dyn(4);
    let fused_prime = fused.prime();
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let data: Vec<f64> = (0..16).map(|value| value as f64).collect();
    let tensor = IdxTensor::from_dense(vec![fused.clone(), fused_prime.clone()], data).unwrap();

    let result = tensor
        .unfuse_index(
            &fused_prime,
            &[i.clone(), j.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    assert_eq!(result.external_indices(), vec![fused, i, j]);
}
