use super::*;

#[test]
fn test_contraction_new() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    assert_eq!(contraction.len(), 2);
    assert_eq!(contraction.result_site_dims(), vec![(2, 2), (2, 2)]);
}

#[test]
fn test_contraction_evaluate() {
    // Identity * Identity = Identity
    let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    // C[0, 0] = sum_k I[0, k] * I[k, 0] = I[0, 0] * I[0, 0] + I[0, 1] * I[1, 0]
    //         = 1 * 1 + 0 * 0 = 1
    let val_00 = contraction.evaluate(&[(0, 0)]).unwrap();
    assert!((val_00 - 1.0).abs() < 1e-10);

    // C[0, 1] = sum_k I[0, k] * I[k, 1] = I[0, 0] * I[0, 1] + I[0, 1] * I[1, 1]
    //         = 1 * 0 + 0 * 1 = 0
    let val_01 = contraction.evaluate(&[(0, 1)]).unwrap();
    assert!(val_01.abs() < 1e-10);

    // C[1, 1] = sum_k I[1, k] * I[k, 1] = I[1, 0] * I[0, 1] + I[1, 1] * I[1, 1]
    //         = 0 * 0 + 1 * 1 = 1
    let val_11 = contraction.evaluate(&[(1, 1)]).unwrap();
    assert!((val_11 - 1.0).abs() < 1e-10);
}

#[test]
fn test_contraction_with_transform() {
    let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

    let mut contraction = Contraction::with_transform(mpo_a, mpo_b, |x| x * 2.0).unwrap();

    let val = contraction.evaluate(&[(0, 0)]).unwrap();
    assert!((val - 2.0).abs() < 1e-10);
}

#[test]
fn test_contraction_length_mismatch() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2, 2]).unwrap();

    let result = Contraction::new(mpo_a, mpo_b);
    assert!(result.is_err());
    assert!(matches!(
        result.as_ref(),
        Err(MPOError::LengthMismatch {
            expected: 2,
            got: 3
        })
    ));
}

#[test]
fn test_contraction_shared_dimension_mismatch() {
    // A has site_dim (2, 3), B has site_dim (2, 2) => s2_a=3 != s1_b=2
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2)], 1.0);

    let result = Contraction::new(mpo_a, mpo_b);
    assert!(result.is_err());
    assert!(matches!(
        result.as_ref(),
        Err(MPOError::SharedDimensionMismatch {
            site: 0,
            dim_a: 3,
            dim_b: 2
        })
    ));
}

#[test]
fn test_contraction_is_empty() {
    let mpo_a = MPO::<f64>::identity(&[2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2]).unwrap();

    let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    assert!(!contraction.is_empty());
}

#[test]
fn test_contraction_result_site_dims() {
    // A: site_dim (2, 3), B: site_dim (3, 4)
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(3, 4)], 1.0);

    let contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    assert_eq!(contraction.result_site_dims(), vec![(2, 4)]);
}

#[test]
fn test_contraction_evaluate_wrong_indices() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    let result = contraction.evaluate(&[(0, 0)]);
    assert!(result.is_err());

    let result = contraction.evaluate(&[(0, 0), (0, 0), (0, 0)]);
    assert!(result.is_err());
}

#[test]
fn test_contraction_clear_cache() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    let _ = contraction.evaluate_left(2, &[(0, 0), (1, 1)]).unwrap();
    let _ = contraction.evaluate_right(0, &[(0, 0), (1, 1)]).unwrap();

    contraction.clear_cache();

    // After clearing, cache should be empty, but evaluation should still work
    let val = contraction.evaluate(&[(0, 0), (1, 1)]).unwrap();
    assert!((val - 1.0).abs() < 1e-10);
}

#[test]
fn test_evaluate_left_boundary() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    let indices = vec![(0, 0), (1, 1)];

    // n=0 returns 1x1 identity
    let env0 = contraction.evaluate_left(0, &indices).unwrap();
    assert_eq!(env0.dim(0), 1);
    assert_eq!(env0.dim(1), 1);
    assert!((env0[[0, 0]] - 1.0).abs() < 1e-10);

    // n=len returns the full left environment
    let env_full = contraction.evaluate_left(2, &indices).unwrap();
    assert_eq!(env_full.dim(0), 1);
    assert_eq!(env_full.dim(1), 1);

    // Out of range
    let result = contraction.evaluate_left(3, &indices);
    assert!(result.is_err());
}

#[test]
fn test_evaluate_right_boundary() {
    let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    let indices = vec![(0, 0), (1, 1)];

    // n=len returns 1x1 identity
    let env_end = contraction.evaluate_right(2, &indices).unwrap();
    assert_eq!(env_end.dim(0), 1);
    assert_eq!(env_end.dim(1), 1);
    assert!((env_end[[0, 0]] - 1.0).abs() < 1e-10);

    // n=0 returns the full right environment
    let env_full = contraction.evaluate_right(0, &indices).unwrap();
    assert_eq!(env_full.dim(0), 1);
    assert_eq!(env_full.dim(1), 1);

    // Out of range
    let result = contraction.evaluate_right(3, &indices);
    assert!(result.is_err());
}

#[test]
fn test_contraction_two_sites() {
    // Two-site identity contraction: I * I = I at each configuration
    let mpo_a = MPO::<f64>::identity(&[2, 3]).unwrap();
    let mpo_b = MPO::<f64>::identity(&[2, 3]).unwrap();

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();
    assert_eq!(contraction.len(), 2);
    assert_eq!(contraction.result_site_dims(), vec![(2, 2), (3, 3)]);

    // (0,0), (0,0) -> delta(0,0)*delta(0,0) = 1
    let val = contraction.evaluate(&[(0, 0), (0, 0)]).unwrap();
    assert!((val - 1.0).abs() < 1e-10);

    // (0,1), (0,0) -> delta(0,k)*delta(k,1) at site 0 = delta(0,1) = 0
    let val = contraction.evaluate(&[(0, 1), (0, 0)]).unwrap();
    assert!(val.abs() < 1e-10);

    // (1,1), (2,2) -> delta(1,1)*delta(2,2) = 1
    let val = contraction.evaluate(&[(1, 1), (2, 2)]).unwrap();
    assert!((val - 1.0).abs() < 1e-10);

    // Verify left/right environments are consistent
    let indices = vec![(1, 1), (2, 2)];
    let left = contraction.evaluate_left(1, &indices).unwrap();
    let right = contraction.evaluate_right(1, &indices).unwrap();

    // Left env after site 0 with (1,1): should be 1x1 with value 1 (diagonal)
    assert!((left[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((right[[0, 0]] - 1.0).abs() < 1e-10);
}

#[test]
fn test_contraction_constant_two_sites() {
    // Constant(2,2) * Constant(2,2) at each site
    // A[i,k] = 1 for all i,k; B[k,j] = 1 for all k,j
    // C[i,j] = sum_k A[i,k]*B[k,j] = sum_k 1 = 2 (shared dim is 2)
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);

    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    // At any (i,j) pair at each site, the contraction sums over the shared dim
    // Site 0: sum_k A0[0,i,k,0] * B0[0,k,j,0] = sum_k 1*1 = 2
    // Site 1: sum_k A1[0,i,k,0] * B1[0,k,j,0] = sum_k 1*1 = 2
    // Total = 2 * 2 = 4
    let val = contraction.evaluate(&[(0, 0), (0, 0)]).unwrap();
    assert!((val - 4.0).abs() < 1e-10);

    let val = contraction.evaluate(&[(1, 1), (1, 0)]).unwrap();
    assert!((val - 4.0).abs() < 1e-10);
}

#[test]
fn test_contraction_evaluate_rejects_out_of_range_values() {
    // Asymmetric physical dims: mpo_a sites are (2, 3), mpo_b sites are (3, 4)
    // -> i_k must be < 2, j_k must be < 4.
    let mpo_a = MPO::<f64>::constant(&[(2, 3)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(3, 4)], 1.0);
    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    // Valid pair still works.
    assert!(contraction.evaluate(&[(1, 3)]).is_ok());

    // i out of range.
    let err = contraction.evaluate(&[(5, 0)]).unwrap_err();
    assert!(
        matches!(
            err,
            MPOError::IndexOutOfBounds {
                site: 0,
                index: 5,
                max: 4
            }
        ),
        "got {err:?}"
    );

    // j out of range.
    let err = contraction.evaluate(&[(0, 9)]).unwrap_err();
    assert!(
        matches!(
            err,
            MPOError::IndexOutOfBounds {
                site: 0,
                index: 9,
                max: 4
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn test_contraction_evaluate_left_rejects_short_slice() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    // indices[..2] on a 1-element slice used to panic.
    let err = contraction.evaluate_left(2, &[(0, 0)]).unwrap_err();
    assert!(
        matches!(err, MPOError::InvalidOperation { .. }),
        "got {err:?}"
    );

    // Out-of-range value at a covered site is also rejected.
    let err = contraction.evaluate_left(2, &[(5, 0), (0, 0)]).unwrap_err();
    assert!(
        matches!(
            err,
            MPOError::IndexOutOfBounds {
                site: 0,
                index: 5,
                max: 2
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn test_contraction_evaluate_right_rejects_short_slice() {
    let mpo_a = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mpo_b = MPO::<f64>::constant(&[(2, 2), (2, 2)], 1.0);
    let mut contraction = Contraction::new(mpo_a, mpo_b).unwrap();

    // Right evaluation reads indices[n] .. indices[len-1]; a short slice that
    // used to panic (len - n would under-require) must be a typed error.
    let err = contraction.evaluate_right(1, &[(0, 0)]).unwrap_err();
    assert!(
        matches!(err, MPOError::InvalidOperation { .. }),
        "got {err:?}"
    );

    // Out-of-range value at a covered site is also rejected.
    let err = contraction
        .evaluate_right(1, &[(0, 0), (5, 5)])
        .unwrap_err();
    assert!(
        matches!(
            err,
            MPOError::IndexOutOfBounds {
                site: 1,
                index: 5,
                max: 2
            }
        ),
        "got {err:?}"
    );
}
