use super::*;
use tensor4all_core::{
    FactorizeAlg, SingularValueMeasure, SvdTruncationPolicy, ThresholdScale, TruncationRule,
};
use tensor4all_treetn::contraction::ContractionMethod;
use tensor4all_treetn::CanonicalForm;

#[test]
fn test_canonical_form_roundtrip() {
    for form in [CanonicalForm::Unitary, CanonicalForm::LU, CanonicalForm::CI] {
        let ffi = t4a_canonical_form::from(form);
        let roundtrip = CanonicalForm::from(ffi);
        assert_eq!(roundtrip, form);
    }
}

#[test]
fn test_contract_method_roundtrip() {
    for method in [
        ContractionMethod::Zipup,
        ContractionMethod::Fit,
        ContractionMethod::Naive,
    ] {
        let ffi = t4a_contract_method::from(method);
        let roundtrip = ContractionMethod::from(ffi);
        assert_eq!(roundtrip, method);
    }
}

#[test]
fn test_factorize_alg_roundtrip() {
    for alg in [
        FactorizeAlg::SVD,
        FactorizeAlg::QR,
        FactorizeAlg::LU,
        FactorizeAlg::CI,
    ] {
        let ffi = t4a_factorize_alg::from(alg);
        let roundtrip = FactorizeAlg::from(ffi);
        assert_eq!(roundtrip, alg);
    }
}

#[test]
fn test_threshold_scale_roundtrip() {
    for scale in [ThresholdScale::Relative, ThresholdScale::Absolute] {
        let ffi = t4a_threshold_scale::from(scale);
        let roundtrip = ThresholdScale::from(ffi);
        assert_eq!(roundtrip, scale);
    }
}

#[test]
fn test_singular_value_measure_roundtrip() {
    for measure in [
        SingularValueMeasure::Value,
        SingularValueMeasure::SquaredValue,
    ] {
        let ffi = t4a_singular_value_measure::from(measure);
        let roundtrip = SingularValueMeasure::from(ffi);
        assert_eq!(roundtrip, measure);
    }
}

#[test]
fn test_truncation_rule_roundtrip() {
    for rule in [TruncationRule::PerValue, TruncationRule::DiscardedTailSum] {
        let ffi = t4a_truncation_rule::from(rule);
        let roundtrip = TruncationRule::from(ffi);
        assert_eq!(roundtrip, rule);
    }
}

#[test]
fn test_svd_truncation_policy_roundtrip() {
    let policy = SvdTruncationPolicy::new(1e-8)
        .with_absolute()
        .with_squared_values()
        .with_discarded_tail_sum();
    let ffi = t4a_svd_truncation_policy::from(policy);
    let roundtrip = SvdTruncationPolicy::from(ffi);
    assert_eq!(roundtrip, policy);
}

#[test]
fn test_boundary_condition_roundtrip() {
    for bc in [
        tensor4all_quanticstransform::BoundaryCondition::Periodic,
        tensor4all_quanticstransform::BoundaryCondition::AntiPeriodic,
        tensor4all_quanticstransform::BoundaryCondition::Open,
    ] {
        let ffi = t4a_boundary_condition::from(bc);
        let roundtrip = tensor4all_quanticstransform::BoundaryCondition::from(ffi);
        assert_eq!(roundtrip, bc);
    }
}

#[test]
fn test_qtt_layout_validation() {
    let interleaved = InternalQttLayout::new(t4a_qtt_layout_kind::Interleaved, vec![4, 4]).unwrap();
    assert_eq!(interleaved.nsites(), 8);

    let fused = InternalQttLayout::new(t4a_qtt_layout_kind::Fused, vec![4, 4, 4]).unwrap();
    assert_eq!(fused.nsites(), 4);

    assert!(InternalQttLayout::new(t4a_qtt_layout_kind::Interleaved, vec![3, 2]).is_err());
    assert!(InternalQttLayout::new(t4a_qtt_layout_kind::Fused, vec![3, 2]).is_err());
}
