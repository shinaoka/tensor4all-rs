use super::*;
use crate::SwapOptions;
use tensor4all_core::{SingularValueMeasure, SvdTruncationPolicy, TruncationRule};

#[test]
fn test_canonicalization_options_default() {
    let opts = CanonicalizationOptions::default();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.force);
}

#[test]
fn test_canonicalization_options_new() {
    let opts = CanonicalizationOptions::new();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.force);
}

#[test]
fn test_canonicalization_options_forced() {
    let opts = CanonicalizationOptions::forced();
    assert!(opts.force);
}

#[test]
fn test_canonicalization_options_builder() {
    let opts = CanonicalizationOptions::new()
        .with_form(CanonicalForm::LU)
        .force();
    assert_eq!(opts.form, CanonicalForm::LU);
    assert!(opts.force);

    let opts = opts.smart();
    assert!(!opts.force);
}

#[test]
fn test_truncation_options_default() {
    let opts = TruncationOptions::default();
    assert_eq!(opts.max_bond_dim(), None);
    assert_eq!(opts.svd_policy(), None);
}

#[test]
fn test_truncation_options_new() {
    let opts = TruncationOptions::new();
    assert_eq!(opts.max_bond_dim(), None);
    assert_eq!(opts.svd_policy(), None);
}

#[test]
fn test_truncation_options_do_not_expose_form() {
    let opts = TruncationOptions::new().with_max_bond_dim(8);
    assert_eq!(opts.max_bond_dim(), Some(8));
}

#[test]
fn test_tree_truncate_uses_svd_policy() {
    let opts = TruncationOptions::new().with_svd_policy(SvdTruncationPolicy::new(1e-10));
    assert_eq!(opts.svd_policy().unwrap().rule, TruncationRule::PerValue);
}

#[test]
fn test_truncation_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = TruncationOptions::new()
        .with_max_bond_dim(50)
        .with_svd_policy(policy);
    assert_eq!(opts.max_bond_dim(), Some(50));
    assert_eq!(opts.svd_policy(), Some(policy));
}

#[test]
fn test_split_options_default() {
    let opts = SplitOptions::default();
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(!opts.final_sweep);
    assert_eq!(opts.max_bond_dim(), None);
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.qr_rtol(), None);
}

#[test]
fn test_split_options_new() {
    let opts = SplitOptions::new();
    assert_eq!(opts.form, CanonicalForm::Unitary);
}

#[test]
fn test_split_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-12).with_squared_values();
    let opts = SplitOptions::new()
        .with_max_bond_dim(100)
        .with_svd_policy(policy)
        .with_qr_rtol(1e-12)
        .with_form(CanonicalForm::Unitary)
        .with_final_sweep(true);
    assert_eq!(opts.max_bond_dim(), Some(100));
    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.qr_rtol(), Some(1e-12));
    assert_eq!(opts.form, CanonicalForm::Unitary);
    assert!(opts.final_sweep);
}

#[test]
fn test_split_options_policy_builders_are_independent() {
    let opts = SplitOptions::new()
        .with_svd_policy(SvdTruncationPolicy::new(1e-8).with_squared_values())
        .with_qr_rtol(1e-6);
    assert_eq!(
        opts.svd_policy().map(|policy| policy.measure),
        Some(SingularValueMeasure::SquaredValue)
    );
    assert_eq!(opts.qr_rtol(), Some(1e-6));
}

#[test]
fn test_restructure_options_default_and_builder() {
    let opts = RestructureOptions::default();
    assert_eq!(opts.split.form, CanonicalForm::Unitary);
    assert!(!opts.split.final_sweep);
    assert_eq!(opts.swap.max_bond_dim, None);
    assert_eq!(opts.swap.rtol, None);
    assert!(opts.final_truncation.is_none());

    let final_policy = SvdTruncationPolicy::new(1e-10).with_discarded_tail_sum();
    let opts = RestructureOptions::new()
        .with_split(
            SplitOptions::new()
                .with_max_bond_dim(16)
                .with_final_sweep(true),
        )
        .with_swap(SwapOptions {
            max_bond_dim: Some(8),
            rtol: Some(1e-9),
        })
        .with_final_truncation(TruncationOptions::new().with_svd_policy(final_policy));
    assert_eq!(opts.split.max_bond_dim(), Some(16));
    assert!(opts.split.final_sweep);
    assert_eq!(opts.swap.max_bond_dim, Some(8));
    assert_eq!(opts.swap.rtol, Some(1e-9));
    assert_eq!(
        opts.final_truncation
            .as_ref()
            .and_then(TruncationOptions::svd_policy),
        Some(final_policy)
    );
}
