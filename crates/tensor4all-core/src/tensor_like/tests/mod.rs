use super::*;
use crate::{DynIndex, IdxTensor};

// Compile-time check that TensorLike requires Sized (no dyn TensorLike)
fn _assert_sized<T: TensorLike>() {
    // This confirms T: Sized is required
}

#[test]
fn factorize_options_builders_and_validation_accept_supported_fields() {
    let svd = FactorizeOptions::svd()
        .with_canonical(Canonical::Right)
        .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
        .with_max_bond_dim(4);
    assert_eq!(svd.alg, FactorizeAlg::SVD);
    assert_eq!(svd.canonical, Canonical::Right);
    assert_eq!(svd.max_bond_dim, Some(4));
    assert_eq!(svd.svd_policy, Some(SvdTruncationPolicy::new(1.0e-8)));
    svd.validate().unwrap();

    let qr = FactorizeOptions::qr()
        .with_qr_rtol(0.0)
        .with_max_bond_dim(3);
    assert_eq!(qr.alg, FactorizeAlg::QR);
    assert_eq!(qr.qr_rtol, Some(0.0));
    assert_eq!(qr.max_bond_dim, Some(3));
    qr.validate().unwrap();

    let lu = FactorizeOptions::lu();
    assert_eq!(lu.alg, FactorizeAlg::LU);
    lu.validate().unwrap();

    let ci = FactorizeOptions::ci();
    assert_eq!(ci.alg, FactorizeAlg::CI);
    ci.validate().unwrap();
}

#[test]
fn factorize_options_validation_rejects_algorithm_specific_mismatches() {
    assert!(matches!(
        FactorizeOptions::svd().with_qr_rtol(1.0e-8).validate(),
        Err(FactorizeError::InvalidOptions(
            "SVD factorization does not accept qr_rtol"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::qr()
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "QR factorization does not accept svd_policy"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::lu()
            .with_svd_policy(SvdTruncationPolicy::new(1.0e-8))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "LU/CI factorization does not accept svd_policy"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::ci().with_qr_rtol(1.0e-8).validate(),
        Err(FactorizeError::InvalidOptions(
            "LU/CI factorization does not accept qr_rtol"
        ))
    ));
}

#[test]
fn factorize_options_validation_rejects_zero_cap_and_invalid_svd_thresholds() {
    // The shared `validate_svd_truncation_options` seam is delegated from
    // `FactorizeOptions::validate`; both typed error kinds it maps must be
    // exercised here (they are validation facts, not algorithm mismatches).
    assert!(matches!(
        FactorizeOptions::svd().with_max_bond_dim(0).validate(),
        Err(FactorizeError::InvalidOptions(
            "max_bond_dim must be at least 1"
        ))
    ));
    assert!(matches!(
        FactorizeOptions::svd()
            .with_svd_policy(SvdTruncationPolicy::new(f64::NAN))
            .validate(),
        Err(FactorizeError::InvalidOptions(
            "SVD truncation threshold must be finite and non-negative"
        ))
    ));
}

#[test]
fn linearization_order_labels_are_stable() {
    assert_eq!(LinearizationOrder::ColumnMajor.as_str(), "column-major");
    assert_eq!(LinearizationOrder::RowMajor.as_str(), "row-major");
}

#[test]
fn tensor_like_default_neg_and_delta_helpers_work() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(2);
    let l = DynIndex::new_dyn(3);

    let tensor = IdxTensor::from_dense(vec![i.clone()], vec![2.0, -3.0]).unwrap();
    let negated = tensor.neg().unwrap();
    assert_eq!(negated.to_vec::<f64>().unwrap(), vec![-2.0, 3.0]);

    let delta = IdxTensor::delta(&[i.clone(), j.clone()], &[k, l]).unwrap();
    assert_eq!(delta.dims(), vec![2, 2, 3, 3]);
    assert!((delta.sum().unwrap().real() - 6.0).abs() < 1.0e-12);

    let err = IdxTensor::delta(&[i], &[]).unwrap_err();
    assert!(err.to_string().contains("Number of input indices"));
}

#[test]
fn tensor_construction_supports_column_major_dense_payloads() {
    fn construct<T: TensorConstructionLike + TensorVectorSpace>(
        indices: Vec<T::Index>,
        data: Vec<AnyScalar>,
    ) -> std::result::Result<T, T::Error> {
        T::from_dense_any(indices, data)
    }

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = construct::<IdxTensor>(
        vec![i, j],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .into_iter()
            .map(AnyScalar::new_real)
            .collect(),
    )
    .unwrap();

    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tensor_construction_supports_stacking_a_batch_axis() {
    fn stack<T: TensorConstructionLike + TensorVectorSpace>(
        tensors: &[&T],
        new_index: T::Index,
    ) -> std::result::Result<T, T::Error> {
        T::stack_along_new_index(tensors, new_index, -1)
    }

    let i = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(2);
    let first = IdxTensor::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap();
    let second = IdxTensor::from_dense(vec![i], vec![3.0, 4.0]).unwrap();
    let stacked = stack(&[&first, &second], batch).unwrap();

    assert_eq!(stacked.external_indices().len(), 2);
    assert_eq!(stacked.to_vec::<f64>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn tensor_construction_concatenates_existing_batch_blocks() {
    fn concatenate<T: TensorConstructionLike + TensorVectorSpace>(
        tensors: &[&T],
        source_indices: &[T::Index],
        new_index: T::Index,
    ) -> std::result::Result<T, T::Error> {
        T::concatenate_along_new_index(tensors, source_indices, new_index)
    }

    let row = DynIndex::new_dyn(2);
    let first_batch = DynIndex::new_link(1).unwrap();
    let second_batch = DynIndex::new_link(2).unwrap();
    let combined = DynIndex::new_link(3).unwrap();
    let first =
        IdxTensor::from_dense(vec![row.clone(), first_batch.clone()], vec![1.0_f64, 2.0]).unwrap();
    let second = IdxTensor::from_dense(
        vec![row.clone(), second_batch.clone()],
        vec![3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let result = concatenate(
        &[&first, &second],
        &[first_batch, second_batch],
        combined.clone(),
    )
    .unwrap();

    assert_eq!(result.external_indices(), vec![row, combined]);
    assert_eq!(
        result.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn tensor_factorization_supports_incremental_probe_prefixes() {
    let row = DynIndex::new_dyn(5);
    let first = IdxTensor::from_dense(vec![row.clone()], vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let second = IdxTensor::from_dense(vec![row.clone()], vec![-2.0, 1.0, 0.0, 3.0, -1.0]).unwrap();
    let third = IdxTensor::from_dense(vec![row.clone()], vec![0.5, -1.0, 2.0, 1.0, 4.0]).unwrap();
    let fourth =
        IdxTensor::from_dense(vec![row.clone()], vec![3.0, -2.0, 1.5, 0.25, -3.0]).unwrap();

    let first_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            None,
            &[&first, &second],
            &[&first, &second],
            std::slice::from_ref(&row),
        )
        .unwrap();
    let second_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&first_factorization),
            &[&first, &second, &third],
            &[&third],
            std::slice::from_ref(&row),
        )
        .unwrap();
    let third_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&second_factorization),
            &[&first, &second, &third, &fourth],
            &[&fourth],
            std::slice::from_ref(&row),
        )
        .unwrap();

    assert_eq!(first_factorization.rank, 2);
    assert_eq!(second_factorization.rank, 3);
    assert_eq!(third_factorization.rank, 4);
    let reconstructed = third_factorization
        .left
        .contract_pair(&third_factorization.right)
        .unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.0, 3.0, -1.0, 0.5, -1.0, 2.0, 1.0, 4.0, 3.0, -2.0,
        1.5, 0.25, -3.0,
    ];
    let actual = reconstructed.to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12),
        "reconstructed columns differ: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn tensor_factorization_preserves_multi_axis_probe_row_order() {
    let first_row = DynIndex::new_dyn(2);
    let second_row = DynIndex::new_dyn(3);
    let rows = vec![first_row.clone(), second_row.clone()];
    let first = IdxTensor::from_dense(rows.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let second = IdxTensor::from_dense(rows.clone(), vec![-1.0, 0.5, 2.0, -2.0, 3.0, 4.0]).unwrap();
    let third =
        IdxTensor::from_dense(rows.clone(), vec![0.25, -3.0, 1.5, 2.5, -0.75, 5.0]).unwrap();
    let fourth =
        IdxTensor::from_dense(rows.clone(), vec![4.0, -1.0, 0.5, 3.5, 2.25, -2.5]).unwrap();

    let first_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            None,
            &[&first, &second],
            &[&first, &second],
            &rows,
        )
        .unwrap();
    let second_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&first_factorization),
            &[&first, &second, &third],
            &[&third],
            &rows,
        )
        .unwrap();
    let final_factorization =
        <IdxTensor as TensorFactorizationLike>::factorize_probe_columns_incremental(
            Some(&second_factorization),
            &[&first, &second, &third, &fourth],
            &[&fourth],
            &rows,
        )
        .unwrap();

    let reconstructed = final_factorization
        .left
        .contract_pair(&final_factorization.right)
        .unwrap();
    let expected = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.0, -2.0, 3.0, 4.0, 0.25, -3.0, 1.5, 2.5, -0.75,
        5.0, 4.0, -1.0, 0.5, 3.5, 2.25, -2.5,
    ];
    let actual = reconstructed.to_vec::<f64>().unwrap();
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected.iter())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12),
        "reconstructed multi-axis sketch differs: actual={actual:?}, expected={expected:?}"
    );
}
