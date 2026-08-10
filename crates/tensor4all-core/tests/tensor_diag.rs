use num_complex::Complex64;
use tensor4all_core::index::DefaultIndex as Index;
use tensor4all_core::{
    diag_tensor_dyn_len, outer_product, tensordot, AnyScalar, TensorConstructionLike,
    TensorContractionLike, TensorDynLen,
};
use tensor4all_tensorbackend::StorageKind;

#[test]
fn test_diag_tensor_creation() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data.clone()).unwrap();
    assert_eq!(tensor.dims(), vec![3, 3]);
    assert!(tensor.is_diag());
    assert_eq!(
        tensor.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ]
    );
}

#[test]
fn test_diag_tensor_validation_different_dims() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0];

    let err = diag_tensor_dyn_len(vec![i, j], diag_data).unwrap_err();
    assert!(err.to_string().contains("same dimension"));
}

#[test]
fn test_diag_tensor_sum() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data).unwrap();
    let sum: AnyScalar = tensor.sum().unwrap();
    assert!(!sum.is_complex());
    assert!((sum.real() - 6.0).abs() < 1e-10);
}

#[test]
fn test_diag_tensor_scale_preserves_diagonal_values() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, -2.0, 4.0]).unwrap();

    let scaled = tensor.scale(AnyScalar::new_real(-0.5)).unwrap();

    assert!(scaled.is_diag());
    assert_eq!(
        scaled.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    let expected = diag_tensor_dyn_len(vec![i, j], vec![-0.5, 1.0, -2.0]).unwrap();
    assert!(scaled.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_diag_tensor_permute() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor =
        diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone()).unwrap();

    // Permute: data should not change for DiagTensor
    let permuted = tensor.permute(&[2, 0, 1]).unwrap();
    assert_eq!(permuted.dims(), vec![3, 3, 3]);
    let expected = diag_tensor_dyn_len(vec![k, i, j], diag_data).unwrap();
    assert!(permuted.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn diag_tensor_select_index_returns_dense_slice_from_payload() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![2.0, 5.0, 7.0]).unwrap();

    let selected = tensor.select_indices(&[j], &[1]).unwrap();

    assert_eq!(selected.dims(), vec![3]);
    assert_eq!(
        selected.storage().unwrap().storage_kind(),
        StorageKind::Dense
    );
    assert_eq!(selected.to_vec::<f64>().unwrap(), vec![0.0, 5.0, 0.0]);
}

#[test]
fn diag_tensor_select_multiple_indices_requires_matching_coordinates() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let tensor =
        diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], vec![2.0, 5.0, 7.0]).unwrap();

    let matching = tensor
        .select_indices(&[i.clone(), k.clone()], &[2, 2])
        .unwrap();
    assert_eq!(matching.dims(), vec![3]);
    assert_eq!(matching.to_vec::<f64>().unwrap(), vec![0.0, 0.0, 7.0]);

    let mismatched = tensor.select_indices(&[i, k], &[1, 2]).unwrap();
    assert_eq!(mismatched.dims(), vec![3]);
    assert_eq!(mismatched.to_vec::<f64>().unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn test_diag_tensor_contract_diag_diag_all_contracted() {
    // Create two 2x2 DiagTensors and contract all indices
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a).unwrap();
    let tensor_b = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_b).unwrap();

    // Contract all indices: result should be scalar (inner product)
    let result = tensor_a.contract_pair(&tensor_b).unwrap();

    // Result should be scalar: 1*3 + 2*4 = 11
    assert_eq!(result.dims().len(), 0);
    assert!((result.only().unwrap().real() - 11.0).abs() < 1e-12);
}

#[test]
fn test_diag_tensor_contract_diag_diag_partial() {
    // Create A[i, j] and B[j, k], contract along j
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let diag_a = vec![1.0, 2.0, 3.0];
    let diag_b = vec![4.0, 5.0, 6.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a).unwrap();
    let tensor_b = diag_tensor_dyn_len(vec![j.clone(), k.clone()], diag_b).unwrap();

    // Contract along j: result should be DiagTensor[i, k]
    let result = tensor_a.contract_pair(&tensor_b).unwrap();

    assert_eq!(result.dims(), vec![3, 3]);
    assert!(result.is_diag());
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );

    // Result diagonal should be element-wise product: [1*4, 2*5, 3*6] = [4, 10, 18]
    let expected = diag_tensor_dyn_len(vec![i, k], vec![4.0, 10.0, 18.0]).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn tracked_diag_partial_contraction_preserves_diag_result_and_grad() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![2.0, 3.0, 5.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let b = diag_tensor_dyn_len(vec![j, k.clone()], vec![7.0, 11.0, 13.0]).unwrap();

    let c = a.contract_pair(&b).unwrap();
    assert_eq!(c.storage().unwrap().storage_kind(), StorageKind::Diagonal);

    let ones = diag_tensor_dyn_len(vec![i, k], vec![1.0, 1.0, 1.0]).unwrap();
    let loss = c.contract_pair(&ones).unwrap();
    loss.backward().unwrap();

    let grad = a.grad().unwrap().unwrap();
    assert_eq!(
        grad.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(
        grad.storage().unwrap().payload_f64_col_major_vec().unwrap(),
        vec![7.0, 11.0, 13.0]
    );
}

#[test]
fn test_diag_tensor_tensordot_diag_diag_partial_preserves_diagonal_storage() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let l = Index::new_dyn(3);

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let tensor_b = diag_tensor_dyn_len(vec![k.clone(), l.clone()], vec![4.0, 5.0, 6.0]).unwrap();

    let result =
        tensordot(&tensor_a, &tensor_b, &[(j, k)]).expect("diag-diag tensordot should succeed");

    assert_eq!(result.dims(), vec![3, 3]);
    assert!(result.is_diag());
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );

    let expected = diag_tensor_dyn_len(vec![i, l], vec![4.0, 10.0, 18.0]).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_diag_tensor_contract_diag_dense() {
    // Create DiagTensor A[i, j] and DenseTensor B[j, k]
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_a).unwrap();

    // Create DenseTensor B[j, k] with all ones
    let tensor_b = TensorDynLen::from_dense(vec![j.clone(), k.clone()], vec![1.0; 4]).unwrap();

    // Contract along j: result should be DenseTensor[i, k]
    let result = tensor_a.contract_pair(&tensor_b).unwrap();

    assert_eq!(result.dims(), vec![2, 2]);
    let expected = TensorDynLen::from_dense(vec![i, k], vec![1.0, 2.0, 1.0, 2.0]).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_diag_tensor_convert_to_dense() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let diag_data = vec![1.0, 2.0, 3.0];

    let tensor = diag_tensor_dyn_len(vec![i.clone(), j.clone()], diag_data).unwrap();
    let dense_tensor =
        TensorDynLen::from_dense(vec![i.clone(), j.clone()], tensor.to_vec::<f64>().unwrap())
            .unwrap();
    let expected = TensorDynLen::from_dense(
        vec![i, j],
        vec![
            1.0, 0.0, 0.0, //
            0.0, 2.0, 0.0, //
            0.0, 0.0, 3.0,
        ],
    )
    .unwrap();
    assert!(dense_tensor.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn from_diag_storage_roundtrip_uses_payload_not_dense_logical_values() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_diag(vec![i, j], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let storage = tensor.storage().unwrap();

    assert_eq!(storage.storage_kind(), StorageKind::Diagonal);
    assert_eq!(storage.payload_dims(), &[3]);
    assert_eq!(storage.axis_classes(), &[0, 0]);
    assert_eq!(
        storage.payload_f64_col_major_vec().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );
}

#[test]
fn tensorlike_diagonal_uses_compact_diagonal_storage() {
    let i = Index::new_dyn(4);
    let o = Index::new_dyn(4);

    let delta = <TensorDynLen as TensorConstructionLike>::diagonal(&i, &o).unwrap();

    assert!(delta.is_diag());
    assert_eq!(
        delta.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(delta.storage().unwrap().payload_dims(), &[4]);
    assert_eq!(delta.storage().unwrap().axis_classes(), &[0, 0]);
    assert_eq!(
        delta
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![1.0, 1.0, 1.0, 1.0],
    );
}

#[test]
fn tensorlike_delta_two_pairs_preserves_independent_copy_structure() {
    let i1 = Index::new_dyn(2);
    let o1 = Index::new_dyn(2);
    let i2 = Index::new_dyn(3);
    let o2 = Index::new_dyn(3);

    let delta = <TensorDynLen as TensorConstructionLike>::delta(
        &[i1.clone(), i2.clone()],
        &[o1.clone(), o2.clone()],
    )
    .unwrap();

    assert_eq!(delta.dims(), vec![2, 2, 3, 3]);
    assert_eq!(
        delta.storage().unwrap().storage_kind(),
        StorageKind::Structured
    );
    assert_eq!(delta.storage().unwrap().payload_dims(), &[2, 3]);
    assert_eq!(delta.storage().unwrap().axis_classes(), &[0, 0, 1, 1]);

    let expected = outer_product(
        &TensorDynLen::from_diag(vec![i1, o1], vec![1.0_f64, 1.0]).unwrap(),
        &TensorDynLen::from_diag(vec![i2, o2], vec![1.0_f64, 1.0, 1.0]).unwrap(),
    )
    .unwrap();
    assert!(delta.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn diag_permute_scale_conj_and_replaceind_preserve_payload_metadata() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let tensor = TensorDynLen::from_diag(
        vec![i.clone(), j.clone(), k.clone()],
        vec![1.0_f64, -2.0, 4.0],
    )
    .unwrap();

    let permuted = tensor.permute(&[2, 0, 1]).unwrap();
    assert!(permuted.is_diag());
    assert_eq!(permuted.storage().unwrap().axis_classes(), &[0, 0, 0]);
    assert_eq!(
        permuted
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![1.0, -2.0, 4.0]
    );

    let scaled = permuted.scale(AnyScalar::new_real(2.0)).unwrap();
    assert!(scaled.is_diag());
    assert_eq!(
        scaled
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![2.0, -4.0, 8.0]
    );

    let replaced = scaled.replaceind(&k, &Index::new_dyn(3)).unwrap();
    assert!(replaced.is_diag());
    assert_eq!(
        replaced
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![2.0, -4.0, 8.0]
    );

    let conjugated = replaced.conj();
    assert!(conjugated.is_diag());
    assert_eq!(
        conjugated
            .storage()
            .unwrap()
            .payload_f64_col_major_vec()
            .unwrap(),
        vec![2.0, -4.0, 8.0]
    );
}

#[test]
fn test_diag_tensor_rank3() {
    // Test DiagTensor with rank 3
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag_data = vec![1.0, 2.0];

    let tensor =
        diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_data.clone()).unwrap();
    assert_eq!(tensor.dims(), vec![2, 2, 2]);
    assert!(tensor.is_diag());
    assert_eq!(
        tensor.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );

    // Sum should work
    let sum: AnyScalar = tensor.sum().unwrap();
    assert!(!sum.is_complex());
    assert!((sum.real() - 3.0).abs() < 1e-10);
}

#[test]
fn test_copy_tensor_rank3() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);

    let tensor = TensorDynLen::copy_tensor(
        vec![i.clone(), j.clone(), k.clone()],
        AnyScalar::new_real(1.0),
    )
    .unwrap();

    let expected = TensorDynLen::from_diag(vec![i, j, k], vec![1.0, 1.0, 1.0]).unwrap();
    assert!(tensor.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_copy_tensor_rejects_mismatched_dimensions() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);

    let err = TensorDynLen::copy_tensor(vec![i, j], AnyScalar::new_real(1.0)).unwrap_err();
    assert!(err
        .to_string()
        .contains("DiagTensor requires all indices to have the same dimension"));
}

#[test]
fn test_from_diag_any_real() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    let tensor = TensorDynLen::from_diag_any(
        vec![i.clone(), j.clone()],
        vec![AnyScalar::new_real(1.0), AnyScalar::new_real(2.0)],
    )
    .unwrap();

    let expected = TensorDynLen::from_diag(vec![i, j], vec![1.0, 2.0]).unwrap();
    assert!(tensor.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_from_diag_any_complex_promotes_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);

    let tensor = TensorDynLen::from_diag_any(
        vec![i.clone(), j.clone()],
        vec![AnyScalar::new_real(1.0), AnyScalar::new_complex(2.0, -0.5)],
    )
    .unwrap();

    let expected = TensorDynLen::from_diag(
        vec![i, j],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, -0.5)],
    )
    .unwrap();
    assert!(tensor.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_diag_tensor_complex() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_data = vec![Complex64::new(1.0, 0.5), Complex64::new(2.0, 1.0)];

    let tensor = TensorDynLen::from_diag(vec![i.clone(), j.clone()], diag_data.clone()).unwrap();
    assert_eq!(tensor.dims(), vec![2, 2]);
    assert!(tensor.is_diag());
    assert_eq!(
        tensor.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    assert_eq!(
        tensor.to_vec::<Complex64>().unwrap(),
        vec![
            Complex64::new(1.0, 0.5),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 1.0),
        ]
    );

    // Sum should work
    let sum: AnyScalar = tensor.sum().unwrap();
    assert!(sum.is_complex());
    let z: Complex64 = sum.into();
    assert!((z.re - 3.0).abs() < 1e-10);
    assert!((z.im - 1.5).abs() < 1e-10);
}

#[test]
fn test_diag_tensor_complex_axpby_preserves_diagonal_values() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let diag_a = vec![Complex64::new(1.0, 0.5), Complex64::new(-2.0, 1.0)];
    let diag_b = vec![Complex64::new(0.5, -1.0), Complex64::new(3.0, 0.25)];

    let tensor_a = TensorDynLen::from_diag(vec![i.clone(), j.clone()], diag_a.clone()).unwrap();
    let tensor_b = TensorDynLen::from_diag(vec![i.clone(), j.clone()], diag_b.clone()).unwrap();

    let a = AnyScalar::new_real(2.0);
    let b = AnyScalar::new_complex(-0.5, 1.0);
    let result = tensor_a.axpby(a, &tensor_b, b).unwrap();

    assert!(result.is_diag());
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );
    let b_c = Complex64::new(-0.5, 1.0);
    let expected_diag: Vec<Complex64> = diag_a
        .iter()
        .zip(diag_b.iter())
        .map(|(&x, &y)| Complex64::new(2.0, 0.0) * x + b_c * y)
        .collect();
    let expected = TensorDynLen::from_diag(vec![i, j], expected_diag).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_diag_tensor_contract_rank3() {
    // Test contraction of rank-3 DiagTensors
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let l = Index::new_dyn(2);
    let diag_a = vec![1.0, 2.0];
    let diag_b = vec![3.0, 4.0];

    let tensor_a = diag_tensor_dyn_len(vec![i.clone(), j.clone(), k.clone()], diag_a).unwrap();
    let tensor_b = diag_tensor_dyn_len(vec![k.clone(), l.clone()], diag_b).unwrap();

    // Contract along k: result should be DiagTensor[i, j, l]
    let result = tensor_a.contract_pair(&tensor_b).unwrap();

    assert_eq!(result.dims(), vec![2, 2, 2]);
    assert!(result.is_diag());
    assert_eq!(
        result.storage().unwrap().storage_kind(),
        StorageKind::Diagonal
    );

    // Result diagonal should be element-wise product: [1*3, 2*4] = [3, 8]
    let expected = diag_tensor_dyn_len(vec![i, j, l], vec![3.0, 8.0]).unwrap();
    assert!(result.isapprox(&expected, 1e-12, 0.0).unwrap());
}
