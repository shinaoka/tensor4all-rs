use super::super::site_mpo::SiteMPO;
use super::super::types::tensor4_from_data;
use super::*;

fn scalar_tensor4(value: f64) -> Tensor4<f64> {
    tensor4_from_data(vec![value], 1, 1, 1, 1).unwrap()
}

#[test]
fn test_fit_options_default_values() {
    let options = FitOptions::default();
    assert_eq!(options.tolerance, 1e-12);
    assert_eq!(options.max_bond_dim, Some(100));
    assert_eq!(options.max_sweeps, 10);
    assert_eq!(options.convergence_tol, 1e-10);
    assert_eq!(options.factorize_method, FactorizeMethod::SVD);
}

#[test]
fn test_contract_fit_empty_returns_empty_mpo() {
    let mpo_a = MPO::<f64>::constant(&[], 1.0);
    let mpo_b = MPO::<f64>::constant(&[], 2.0);

    let result = contract_fit(&mpo_a, &mpo_b, &FitOptions::default(), None).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_contract_fit_length_mismatch_errors() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    assert!(matches!(
        contract_fit(&mpo_a, &mpo_b, &FitOptions::default(), None),
        Err(MPOError::LengthMismatch {
            expected: 1,
            got: 2
        })
    ));
}

#[test]
fn test_contract_fit_shared_dimension_mismatch_errors() {
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(4, 2)], 1.0);

    assert!(matches!(
        contract_fit(&mpo_a, &mpo_b, &FitOptions::default(), None),
        Err(MPOError::SharedDimensionMismatch {
            site: 0,
            dim_a: 3,
            dim_b: 4,
        })
    ));
}

#[test]
fn test_environment_identity_get_and_set() {
    let mut env = Environment::<f64>::identity(2, 2, 2);
    assert_eq!(env.get(0, 0, 0), 1.0);
    assert_eq!(env.get(1, 1, 1), 1.0);
    assert_eq!(env.get(0, 1, 1), 0.0);

    env.set(0, 1, 1, 3.5);
    assert_eq!(env.get(0, 1, 1), 3.5);

    let scalar_env = Environment::<f64>::identity(1, 1, 1);
    assert_eq!(scalar_env.get(0, 0, 0), 1.0);
}

#[test]
fn missing_environment_returns_typed_error() {
    let envs: Vec<Option<Environment<f64>>> = vec![None];
    let err = require_environment(&envs, 0, "left").unwrap_err();

    assert!(matches!(err, MPOError::InvalidOperation { .. }));
    assert!(err
        .to_string()
        .contains("missing left environment at site 0"));
}

#[test]
fn test_build_left_and_right_environment_for_scalar_tensors() {
    let tensor_a = scalar_tensor4(2.0);
    let tensor_b = scalar_tensor4(3.0);
    let tensor_result = scalar_tensor4(5.0);
    let prev_env = Environment::<f64>::identity(1, 1, 1);

    let left = build_left_environment(&tensor_a, &tensor_b, &tensor_result, &prev_env).unwrap();
    assert_eq!(left.get(0, 0, 0), 30.0);

    let right = build_right_environment(&tensor_a, &tensor_b, &tensor_result, &prev_env).unwrap();
    assert_eq!(right.get(0, 0, 0), 30.0);
}

#[test]
fn test_update_two_site_core_placeholder_returns_true() {
    let mpo_a = MPO::<f64>::constant(&[(1, 1), (1, 1)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(1, 1), (1, 1)], 3.0);
    let mut result = SiteMPO::from_mpo(MPO::<f64>::constant(&[(1, 1), (1, 1)], 1.0), 0).unwrap();
    let left_envs = vec![Some(Environment::<f64>::identity(1, 1, 1)), None];
    let right_envs = vec![None, Some(Environment::<f64>::identity(1, 1, 1))];

    let updated = update_two_site_core(
        &mpo_a,
        &mpo_b,
        &mut result,
        0,
        &left_envs,
        &right_envs,
        &FitOptions::default(),
    )
    .unwrap();
    assert!(updated);
}

#[test]
fn test_contract_fit_identity() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let options = FitOptions {
        factorize_method: FactorizeMethod::SVD,
        ..Default::default()
    };
    let result = contract_fit(&mpo_a, &mpo_b, &options, None).unwrap();

    assert_eq!(result.len(), 2);

    // The result should be equivalent to identity
    assert!((result.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-8);
    assert!((result.evaluate(&[1, 1, 1, 1]).unwrap() - 1.0).abs() < 1e-8);
}

#[test]
fn test_contract_fit_constant() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 2.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 3.0);

    let options = FitOptions {
        factorize_method: FactorizeMethod::SVD,
        ..Default::default()
    };
    let result = contract_fit(&mpo_a, &mpo_b, &options, None).unwrap();

    // Each element of C = sum over k of A[i, k] * B[k, j]
    // = sum over k of 2 * 3 = 6 * 2 = 12
    let val = result.evaluate(&[0, 0]).unwrap();
    assert!((val - 12.0).abs() < 1e-8);
}
