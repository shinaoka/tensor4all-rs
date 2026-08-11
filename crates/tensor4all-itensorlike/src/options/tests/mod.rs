use super::*;
use num_complex::Complex64;
use tensor4all_core::AnyScalar;
use tensor4all_core::SvdTruncationPolicy;

#[test]
fn test_validate_svd_truncation_options_accepts_default_settings() {
    assert!(validate_svd_truncation_options(None, None).is_ok());
}

#[test]
fn test_validate_svd_truncation_options_rejects_negative_threshold() {
    let err =
        validate_svd_truncation_options(None, Some(SvdTruncationPolicy::new(-1.0))).unwrap_err();
    assert!(err.to_string().contains("svd_policy.threshold"));
}

#[test]
fn test_validate_svd_truncation_options_rejects_non_finite_threshold() {
    let err = validate_svd_truncation_options(None, Some(SvdTruncationPolicy::new(f64::NAN)))
        .unwrap_err();
    assert!(err.to_string().contains("svd_policy.threshold"));
}

#[test]
fn test_validate_svd_truncation_options_rejects_zero_max_bond_dim() {
    let err =
        validate_svd_truncation_options(Some(0), Some(SvdTruncationPolicy::new(1e-8))).unwrap_err();
    assert!(err.to_string().contains("max_bond_dim/maxdim"));
}

#[test]
fn test_truncate_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = TruncateOptions::svd()
        .with_svd_policy(policy)
        .with_max_bond_dim(50)
        .with_site_range(0..5);

    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.max_bond_dim(), Some(50));
    assert_eq!(opts.site_range(), Some(0..5));
}

#[test]
fn test_truncate_options_default() {
    let opts = TruncateOptions::default();
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_bond_dim(), None);
    assert_eq!(opts.site_range(), None);
}

#[test]
fn test_contract_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-12);
    let opts = ContractOptions::zipup()
        .with_max_bond_dim(100)
        .with_svd_policy(policy)
        .with_nhalfsweeps(6)
        .with_dense_reference_limit(256);

    assert_eq!(opts.method(), ContractMethod::Zipup);
    assert_eq!(opts.max_bond_dim(), Some(100));
    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.nhalfsweeps(), 6);
    assert_eq!(opts.dense_reference_limit(), Some(256));
}

#[test]
fn test_contract_options_methods() {
    assert_eq!(ContractOptions::zipup().method(), ContractMethod::Zipup);
    assert_eq!(ContractOptions::fit().method(), ContractMethod::Fit);
    assert_eq!(ContractOptions::naive().method(), ContractMethod::Naive);
}

#[test]
fn test_contract_options_default() {
    let opts = ContractOptions::default();
    assert_eq!(opts.method(), ContractMethod::Zipup);
    assert_eq!(opts.nhalfsweeps(), 2);
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_bond_dim(), None);
    assert_eq!(opts.dense_reference_limit(), None);
}

#[test]
fn test_contract_options_nsweeps() {
    let opts = ContractOptions::fit().with_nsweeps(5);
    assert_eq!(opts.nhalfsweeps(), 10);
}

#[test]
fn test_contract_options_odd_nhalfsweeps_allowed_in_builder() {
    let opts = ContractOptions::fit().with_nhalfsweeps(1);
    assert_eq!(opts.nhalfsweeps(), 1);
}

#[test]
fn test_contract_method_default() {
    let method: ContractMethod = Default::default();
    assert_eq!(method, ContractMethod::Zipup);
}

#[test]
fn test_linsolve_options_default() {
    let opts = LinsolveOptions::default();
    assert_eq!(opts.nhalfsweeps(), 10);
    assert_eq!(
        opts.coefficients(),
        (AnyScalar::new_real(0.0), AnyScalar::new_real(1.0))
    );
    assert_eq!(opts.gmres_tol(), 1e-10);
    assert_eq!(opts.gmres_max_restarts(), 100);
    assert_eq!(opts.gmres_restart_dim(), 30);
    assert_eq!(opts.convergence_tol(), None);
    assert!(opts.check_residual());
    assert_eq!(opts.svd_policy(), None);
    assert_eq!(opts.max_bond_dim(), None);
}

#[test]
fn test_linsolve_options_new() {
    let opts = LinsolveOptions::new(5);
    assert_eq!(opts.nhalfsweeps(), 10);
}

#[test]
fn test_linsolve_options_builder() {
    let policy = SvdTruncationPolicy::new(1e-10)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = LinsolveOptions::default()
        .with_nsweeps(10)
        .with_svd_policy(policy)
        .with_max_bond_dim(100)
        .with_gmres_tol(1e-8)
        .with_gmres_max_restarts(200)
        .with_gmres_restart_dim(50)
        .with_coefficients(1.0, -1.0)
        .with_convergence_tol(1e-6)
        .with_residual_check(false);

    assert_eq!(opts.nhalfsweeps(), 20);
    assert_eq!(opts.svd_policy(), Some(policy));
    assert_eq!(opts.max_bond_dim(), Some(100));
    assert_eq!(opts.gmres_tol(), 1e-8);
    assert_eq!(opts.gmres_max_restarts(), 200);
    assert_eq!(opts.gmres_restart_dim(), 50);
    assert_eq!(
        opts.coefficients(),
        (AnyScalar::new_real(1.0), AnyScalar::new_real(-1.0))
    );
    assert_eq!(opts.convergence_tol(), Some(1e-6));
    assert!(!opts.check_residual());
}

#[test]
fn test_linsolve_options_complex_coefficients() {
    let opts = LinsolveOptions::default()
        .with_coefficients(Complex64::new(0.25, -0.5), AnyScalar::new_complex(1.5, 2.0));

    assert_eq!(
        opts.coefficients(),
        (
            AnyScalar::new_complex(0.25, -0.5),
            AnyScalar::new_complex(1.5, 2.0)
        )
    );
}

#[test]
fn test_linsolve_options_nsweeps() {
    let opts = LinsolveOptions::default().with_nsweeps(3);
    assert_eq!(opts.nhalfsweeps(), 6);
}

#[test]
fn test_linsolve_options_odd_nhalfsweeps_allowed_in_builder() {
    let opts = LinsolveOptions::default().with_nhalfsweeps(3);
    assert_eq!(opts.nhalfsweeps(), 3);
}
