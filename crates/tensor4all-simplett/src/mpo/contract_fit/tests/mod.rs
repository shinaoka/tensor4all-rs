use super::*;

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
fn test_contract_fit_empty_returns_unsupported() {
    let mpo_a = MPO::<f64>::constant(&[], 1.0);
    let mpo_b = MPO::<f64>::constant(&[], 2.0);

    let err = contract_fit(&mpo_a, &mpo_b, &FitOptions::default(), None).unwrap_err();
    assert!(matches!(err, MPOError::Unsupported { .. }));
    assert!(err.to_string().contains("not implemented"));
}

#[test]
fn test_contract_fit_valid_input_returns_unsupported() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0);

    let err = contract_fit(&mpo_a, &mpo_b, &FitOptions::default(), None).unwrap_err();
    assert!(matches!(err, MPOError::Unsupported { .. }));
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
