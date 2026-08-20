use super::*;
use crate::index::DefaultIndex as Index;
use crate::{SingularValueMeasure, SvdTruncationPolicy, ThresholdScale, TruncationRule};
use num_complex::Complex64;
use tenferro::Tensor as NativeTensor;

#[test]
fn compute_retained_rank_handles_edge_cases() {
    let per_value = SvdTruncationPolicy::new(1.0e-6);
    assert_eq!(compute_retained_rank(&[], &per_value), 1);
    assert_eq!(compute_retained_rank(&[0.0, 0.0], &per_value), 1);
    assert_eq!(compute_retained_rank(&[5.0, 1.0e-9], &per_value), 1);
    assert_eq!(
        compute_retained_rank(&[5.0, 1.0], &SvdTruncationPolicy::new(1.0e-12)),
        2
    );
}

#[test]
fn compute_retained_rank_supports_all_policy_axes() {
    let absolute_per_value = SvdTruncationPolicy::new(1.5).with_absolute();
    assert_eq!(compute_retained_rank(&[5.0, 1.0], &absolute_per_value), 1);

    let relative_squared_tail = SvdTruncationPolicy::new(0.05)
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(
        compute_retained_rank(&[10.0, 1.0, 0.1], &relative_squared_tail),
        1
    );

    let absolute_squared_tail = SvdTruncationPolicy::new(0.02)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    assert_eq!(
        compute_retained_rank(&[1.0, 0.1, 0.1], &absolute_squared_tail),
        2
    );
}

#[test]
fn singular_values_from_native_accepts_real_and_complex_dense() {
    let dense = NativeTensor::from_vec_col_major(vec![2], vec![3.0_f64, 1.5]).unwrap();
    assert_eq!(singular_values_from_native(&dense).unwrap(), vec![3.0, 1.5]);

    let complex = NativeTensor::from_vec_col_major(
        vec![2],
        vec![Complex64::new(1.0, 2.0), Complex64::new(0.5, -4.0)],
    )
    .unwrap();
    assert_eq!(
        singular_values_from_native(&complex).unwrap(),
        vec![1.0, 0.5]
    );
}

#[test]
fn set_default_svd_truncation_policy_rejects_invalid_values() {
    let original = default_svd_truncation_policy();
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::NAN)).is_err());
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(-1.0)).is_err());
    assert!(set_default_svd_truncation_policy(SvdTruncationPolicy::new(f64::INFINITY)).is_err());
    set_default_svd_truncation_policy(original).unwrap();
}

#[test]
fn svd_options_accessors_roundtrip() {
    let by_policy = SvdOptions::new().with_policy(
        SvdTruncationPolicy::new(1.0e-5)
            .with_absolute()
            .with_squared_values()
            .with_discarded_tail_sum(),
    );
    assert_eq!(
        by_policy.policy,
        Some(SvdTruncationPolicy {
            threshold: 1.0e-5,
            scale: ThresholdScale::Absolute,
            measure: SingularValueMeasure::SquaredValue,
            rule: TruncationRule::DiscardedTailSum,
        })
    );
    assert_eq!(by_policy.max_bond_dim, None);

    let by_rank = SvdOptions::new().with_max_bond_dim(7);
    assert_eq!(by_rank.policy, None);
    assert_eq!(by_rank.max_bond_dim, Some(7));
}

#[test]
fn singular_values_from_native_rejects_unsupported_scalar_types() {
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let err = singular_values_from_native(&tensor).unwrap_err();
    assert!(err
        .to_string()
        .contains("unsupported singular-value scalar type"));
}

#[test]
fn svd_with_max_bond_dim_truncates_native_outputs() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![3.0, 0.0, 0.0, 1.0]).unwrap();

    let (u, s, v) = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_max_bond_dim(1),
    )
    .unwrap();

    assert_eq!(u.indices.last().unwrap().dim, 1);
    assert_eq!(s.indices[0].dim, 1);
    assert_eq!(s.indices[1].dim, 1);
    assert_eq!(v.indices.last().unwrap().dim, 1);
}

#[test]
fn svd_with_invalid_rtol_is_rejected_before_linalg() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let tensor = IdxTensor::from_dense(vec![i.clone(), j.clone()], vec![0.0; 4]).unwrap();

    let nan = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_policy(SvdTruncationPolicy::new(f64::NAN)),
    );
    assert!(matches!(nan, Err(SvdError::InvalidThreshold(v)) if v.is_nan()));

    let negative = svd_with::<f64>(
        &tensor,
        std::slice::from_ref(&i),
        &SvdOptions::new().with_policy(SvdTruncationPolicy::new(-1.0)),
    );
    assert!(matches!(negative, Err(SvdError::InvalidThreshold(v)) if v == -1.0));
}

#[test]
fn svd_with_rejects_zero_max_bond_dim_and_infinite_thresholds() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        (0..9).map(|x| x as f64).collect(),
    )
    .unwrap();

    // `max_bond_dim == 0` is a typed validation error, never a silent clamp to
    // rank one.
    assert!(matches!(
        svd_with::<f64>(
            &tensor,
            std::slice::from_ref(&i),
            &SvdOptions::new().with_max_bond_dim(0),
        ),
        Err(SvdError::InvalidMaxBondDim)
    ));

    // Positive/negative infinity and NaN thresholds are rejected before any
    // truncation decision.
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
        assert!(
            matches!(
                svd_with::<f64>(
                    &tensor,
                    std::slice::from_ref(&i),
                    &SvdOptions::new().with_policy(SvdTruncationPolicy::new(threshold)),
                ),
                Err(SvdError::InvalidThreshold(_))
            ),
            "svd_with must reject threshold {threshold}"
        );
    }
}
