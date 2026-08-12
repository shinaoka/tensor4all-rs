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
