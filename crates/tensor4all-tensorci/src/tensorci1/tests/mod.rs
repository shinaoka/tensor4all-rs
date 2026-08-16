use super::*;
use num_complex::Complex64;
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorbackend::from_vec2d;

#[test]
fn test_matrix_ci_reconstructs_rank1_outer_product() {
    let pivot_cols = from_vec2d(vec![vec![14.0_f64], vec![21.0], vec![35.0]]);
    let pivot_rows = from_vec2d(vec![vec![14.0_f64, 22.0]]);
    let ci = matrix_ci::MatrixCI::new(vec![0], vec![0], pivot_cols, pivot_rows).unwrap();

    assert_eq!(ci.rank(), 1);
    assert_eq!(ci.row_indices(), &[0]);
    assert_eq!(ci.col_indices(), &[0]);
    assert!((ci.evaluate(0, 0).unwrap() - 14.0).abs() < 1e-12);
    assert!((ci.evaluate(2, 1).unwrap() - 55.0).abs() < 1e-12);
}

#[test]
fn test_matrix_ci_evaluate_uses_pivot_block_solve() {
    let pivot_cols = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 5.0]]);
    let pivot_rows = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 5.0]]);
    let ci = matrix_ci::MatrixCI::new(vec![0, 1], vec![0, 1], pivot_cols, pivot_rows).unwrap();

    assert!((ci.evaluate(0, 0).unwrap() - 1.0).abs() < 1e-12);
    assert!((ci.evaluate(0, 1).unwrap() - 2.0).abs() < 1e-12);
    assert!((ci.evaluate(1, 0).unwrap() - 3.0).abs() < 1e-12);
    assert!((ci.evaluate(1, 1).unwrap() - 5.0).abs() < 1e-12);
}

#[test]
fn test_matrix_ci_rejects_factor_rank_mismatch() {
    let left = from_vec2d(vec![vec![1.0_f64, 2.0]]);
    let right = from_vec2d(vec![vec![3.0_f64]]);
    let err = matrix_ci::MatrixCI::new(vec![0], vec![0], left, right).unwrap_err();

    assert!(matches!(err, TCIError::DimensionMismatch { .. }));
    assert!(err.to_string().contains("rank mismatch"));
}

#[test]
fn test_matrix_ci_evaluate_rejects_left_right_rank_mismatch() {
    let left = from_vec2d(vec![vec![1.0_f64, 2.0]]);
    let right = from_vec2d(vec![vec![3.0_f64, 4.0]]);
    let ci = matrix_ci::MatrixCI::new(vec![0], vec![0, 1], left, right).unwrap();
    let err = ci.evaluate(0, 0).unwrap_err();

    assert!(matches!(err, TCIError::DimensionMismatch { .. }));
    assert!(err
        .to_string()
        .contains("requires matching left/right ranks"));
}

#[test]
fn test_crossinterpolate1_rank2_function() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![4, 4],
        vec![3, 3],
        TCI1Options {
            tolerance: 1e-12,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!((tci.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
    let tt = tci.to_tensor_train().unwrap();
    assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn test_crossinterpolate1_three_site_product() {
    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 2) * (idx[2] + 3)) as f64;
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![3, 3, 3],
        vec![2, 2, 2],
        TCI1Options {
            tolerance: 1e-12,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    let tt = tci.to_tensor_train().unwrap();
    assert!((tt.evaluate(&[1, 2, 0]).unwrap() - 24.0).abs() < 1e-10);
}

#[test]
fn test_tensorci1_lorentz_local_pivot_sweep_grows_rank() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };
    let mut tci = TensorCI1::from_function(&f, local_dims, vec![0; 5]).unwrap();

    assert_eq!(tci.link_dims(), vec![1, 1, 1, 1]);
    assert_eq!(tci.rank(), 1);

    for bond in 0..4 {
        tci.add_pivot(bond, &f, 1e-8).unwrap();
    }

    assert_eq!(tci.link_dims(), vec![2, 2, 2, 2]);
    assert_eq!(tci.rank(), 2);
}

#[test]
fn test_tensorci1_global_pivot_is_inserted_and_deduplicated() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };
    let mut tci = TensorCI1::from_function(&f, local_dims, vec![0; 5]).unwrap();
    for bond in 0..4 {
        tci.add_pivot(bond, &f, 1e-8).unwrap();
    }

    let global_pivot = vec![1, 8, 9, 4, 6];
    tci.add_global_pivot(&f, global_pivot.clone(), 1e-12)
        .unwrap();

    assert_eq!(tci.link_dims(), vec![3, 3, 3, 3]);
    assert_eq!(tci.rank(), 3);
    assert!((tci.evaluate(&global_pivot).unwrap() - f(&global_pivot)).abs() < 1e-10);

    tci.add_global_pivot(&f, global_pivot.clone(), 1e-12)
        .unwrap();
    assert_eq!(tci.link_dims(), vec![3, 3, 3, 3]);
    assert_eq!(tci.rank(), 3);
    assert!((tci.evaluate(&global_pivot).unwrap() - f(&global_pivot)).abs() < 1e-10);
}

#[test]
fn test_crossinterpolate1_forward_sweep_matches_manual_local_pivots() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };

    let mut manual = TensorCI1::from_function(&f, local_dims.clone(), vec![0; 5]).unwrap();
    for expected_rank in 2..=8 {
        for bond in 0..manual.len() - 1 {
            manual.add_pivot(bond, &f, 1e-8).unwrap();
        }
        assert_eq!(manual.link_dims(), vec![expected_rank; manual.len() - 1]);
    }

    let (automatic, ranks, _errors) = crossinterpolate1::<f64, _>(
        f,
        local_dims,
        vec![0; 5],
        TCI1Options {
            tolerance: 0.0,
            max_iter: 8,
            pivot_tolerance: 1e-8,
            sweep_strategy: TCI1SweepStrategy::Forward,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert_eq!(automatic.link_dims(), manual.link_dims());
    assert_eq!(ranks, (2..=8).collect::<Vec<_>>());
}

#[test]
fn test_crossinterpolate1_backward_sweep_matches_manual_local_pivots() {
    let local_dims = vec![6; 4];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };

    let mut manual = TensorCI1::from_function(&f, local_dims.clone(), vec![0; 4]).unwrap();
    for expected_rank in 2..=5 {
        for bond in (0..manual.len() - 1).rev() {
            manual.add_pivot(bond, &f, 1e-8).unwrap();
        }
        assert_eq!(manual.link_dims(), vec![expected_rank; manual.len() - 1]);
    }

    let (automatic, ranks, _errors) = crossinterpolate1::<f64, _>(
        f,
        local_dims,
        vec![0; 4],
        TCI1Options {
            tolerance: 0.0,
            max_iter: 5,
            pivot_tolerance: 1e-8,
            sweep_strategy: TCI1SweepStrategy::Backward,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert_eq!(automatic.link_dims(), manual.link_dims());
    assert_eq!(ranks, (2..=5).collect::<Vec<_>>());
}

#[test]
fn test_crossinterpolate1_lorentz_converges_and_matches_tensor_train() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        local_dims,
        vec![0; 5],
        TCI1Options {
            tolerance: 1e-12,
            max_iter: 200,
            ..TCI1Options::default()
        },
    )
    .unwrap();
    let tt = tci.to_tensor_train().unwrap();

    assert!(tci.pivot_errors().iter().all(|&err| err <= 1e-12));
    assert!(tci.link_dims().iter().all(|&dim| dim <= 200));
    assert!(tci.rank() <= 200);
    assert_eq!(ranks.last().copied(), Some(tci.rank()));
    assert!(!errors.is_empty());
    assert!(errors.last().copied().unwrap().is_finite());

    for i0 in 0..3 {
        for i1 in 0..3 {
            for i2 in 0..3 {
                for i3 in 0..3 {
                    for i4 in 0..3 {
                        let idx = vec![i0, i1, i2, i3, i4];
                        let expected = f(&idx);
                        assert!((tci.evaluate(&idx).unwrap() - expected).abs() < 1e-9);
                        assert!((tt.evaluate(&idx).unwrap() - expected).abs() < 1e-9);
                    }
                }
            }
        }
    }
}

#[test]
fn test_crossinterpolate1_complex_lorentz_converges() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        Complex64::new(0.0, 1.0 / denom)
    };
    let (tci, _ranks, errors) = crossinterpolate1::<Complex64, _>(
        f,
        local_dims,
        vec![0; 5],
        TCI1Options {
            tolerance: 1e-12,
            max_iter: 200,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(tci.pivot_errors().iter().all(|&err| err <= 1e-12));
    assert!(!errors.is_empty());
    assert!(errors.last().copied().unwrap().is_finite());
    let idx = vec![2, 1, 0, 2, 1];
    assert!((tci.evaluate(&idx).unwrap() - f(&idx)).norm() < 1e-9);
}

#[test]
fn test_crossinterpolate1_additional_pivots_converges_with_duplicates() {
    let local_dims = vec![10; 5];
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        1.0 / denom
    };
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        local_dims,
        vec![0; 5],
        TCI1Options {
            tolerance: 1e-12,
            max_iter: 200,
            additional_pivots: vec![
                vec![9, 7, 9, 3, 3],
                vec![4, 3, 7, 8, 2],
                vec![6, 6, 9, 4, 8],
                vec![6, 6, 9, 4, 8],
            ],
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(tci.pivot_errors().iter().all(|&err| err <= 1e-12));
    assert!(tci.link_dims().iter().all(|&dim| dim <= 200));
    assert!(tci.rank() <= 200);
    assert_eq!(ranks.last().copied(), Some(tci.rank()));
    assert!(!errors.is_empty());
    assert!((tci.evaluate(&[2, 1, 0, 2, 1]).unwrap() - f(&vec![2, 1, 0, 2, 1])).abs() < 1e-9);
}

#[test]
fn test_crossinterpolate1_rejects_invalid_options_before_callback() {
    use std::cell::Cell;

    for tolerance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let calls = Cell::new(0);
        let options = TCI1Options {
            tolerance,
            ..TCI1Options::default()
        };
        let error = crossinterpolate1::<f64, _>(
            |_| {
                calls.set(calls.get() + 1);
                1.0
            },
            vec![2, 2],
            vec![0, 0],
            options,
        )
        .unwrap_err();
        assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
        assert_eq!(calls.get(), 0);
    }

    for pivot_tolerance in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let options = TCI1Options {
            pivot_tolerance,
            ..TCI1Options::default()
        };
        let error =
            crossinterpolate1::<f64, _>(|_| 1.0, vec![2, 2], vec![0, 0], options).unwrap_err();
        assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
    }

    let error = crossinterpolate1::<f64, _>(
        |_| 1.0,
        vec![2, 2],
        vec![0, 0],
        TCI1Options {
            max_iter: 0,
            ..TCI1Options::default()
        },
    )
    .unwrap_err();
    assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
}

#[test]
fn test_tensorci1_raw_tolerances_validate_before_callback_and_accept_zero() {
    use std::cell::Cell;

    let calls = Cell::new(0);
    let f = |idx: &MultiIndex| {
        calls.set(calls.get() + 1);
        (idx[0] + idx[1] + 1) as f64
    };
    let mut tci = TensorCI1::<f64>::from_function(&f, vec![2, 2], vec![0, 0]).unwrap();

    calls.set(0);
    let error = tci.add_pivot(0, &f, f64::NAN).unwrap_err();
    assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
    assert_eq!(calls.get(), 0);

    calls.set(0);
    let error = tci.add_global_pivot(&f, vec![1, 1], -1.0).unwrap_err();
    assert!(matches!(error, TCIError::InvalidConfiguration { .. }));
    assert_eq!(calls.get(), 0);

    tci.add_pivot(0, &f, 0.0).unwrap();
    tci.add_global_pivot(&f, vec![1, 1], 0.0).unwrap();
}

#[test]
fn test_crossinterpolate1_rejects_invalid_first_pivots() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;

    let err =
        crossinterpolate1::<f64, _>(f, vec![2, 2], vec![0], TCI1Options::default()).unwrap_err();
    assert!(matches!(err, TCIError::DimensionMismatch { .. }));

    let err =
        crossinterpolate1::<f64, _>(f, vec![2, 2], vec![0, 2], TCI1Options::default()).unwrap_err();
    assert!(matches!(err, TCIError::IndexOutOfBounds { .. }));

    let zero = |_idx: &MultiIndex| 0.0_f64;
    let err = crossinterpolate1::<f64, _>(zero, vec![2, 2], vec![0, 0], TCI1Options::default())
        .unwrap_err();
    assert!(matches!(err, TCIError::InvalidPivot { .. }));
}
