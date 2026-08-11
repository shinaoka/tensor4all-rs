use super::*;
use num_complex::Complex64;
use tensor4all_core::SvdTruncationPolicy;

#[test]
fn test_default_options() {
    let opts = LinsolveOptions::default();
    assert_eq!(opts.nfullsweeps, 5);
    assert_eq!(opts.a0, AnyScalar::new_real(0.0));
    assert_eq!(opts.a1, AnyScalar::new_real(1.0));
    assert_eq!(opts.gmres_tolerance_mode, GmresToleranceMode::Relative);
    assert!(opts.convergence_tol.is_none());
    assert!(opts.check_residual);
}

#[test]
fn test_builder_pattern() {
    let policy = SvdTruncationPolicy::new(1e-9)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = LinsolveOptions::new(5)
        .with_max_bond_dim(100)
        .with_svd_policy(policy)
        .with_gmres_tol(1e-8)
        .with_gmres_absolute_tolerance()
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6)
        .with_residual_check(false);

    assert_eq!(opts.nfullsweeps, 5);
    assert_eq!(opts.truncation.max_bond_dim(), Some(100));
    assert_eq!(opts.truncation.svd_policy(), Some(policy));
    assert_eq!(opts.gmres_tol, 1e-8);
    assert_eq!(opts.gmres_tolerance_mode, GmresToleranceMode::Absolute);
    assert_eq!(opts.a0, AnyScalar::new_real(1.0));
    assert_eq!(opts.a1, AnyScalar::new_real(-1.0));
    assert_eq!(opts.convergence_tol, Some(1e-6));
    assert!(!opts.check_residual);
}

#[test]
fn test_complex_coefficients() {
    let opts = LinsolveOptions::default().with_coefficients(
        Complex64::new(0.5, -0.25),
        AnyScalar::new_complex(-1.0, 2.0),
    );

    assert_eq!(opts.a0, AnyScalar::new_complex(0.5, -0.25));
    assert_eq!(opts.a1, AnyScalar::new_complex(-1.0, 2.0));
}
