use crate::{crossinterpolate2, TCI2Options, TensorCI2, TensorCI2FromTensorTrainOptions};
use num_complex::Complex64;
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
use tensor4all_tcicore::MultiIndex;

#[test]
fn test_tensorci2_from_tensor_train_preserves_values() {
    let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.5);
    let tci = TensorCI2::from_tensor_train(tt, TensorCI2FromTensorTrainOptions::default()).unwrap();
    let roundtrip = tci.to_tensor_train().unwrap();

    assert!((roundtrip.evaluate(&[1, 2, 1]).unwrap() - 2.5).abs() < 1e-12);
}

#[test]
fn test_tensorci2_from_tensor_train_respects_max_bond_dim() {
    let tt = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let options = TensorCI2FromTensorTrainOptions {
        max_bond_dim: 1,
        ..TensorCI2FromTensorTrainOptions::default()
    };
    let tci = TensorCI2::from_tensor_train(tt, options).unwrap();

    assert!(tci.link_dims().iter().all(|&dim| dim <= 1));
}

#[test]
fn test_tensorci2_from_tensor_train_complex_constant() {
    let value = Complex64::new(1.25, -0.5);
    let tt = TensorTrain::<Complex64>::constant(&[2, 2], value);
    let tci = TensorCI2::from_tensor_train(tt, TensorCI2FromTensorTrainOptions::default()).unwrap();
    let roundtrip = tci.to_tensor_train().unwrap();
    let actual = roundtrip.evaluate(&[1, 1]).unwrap();

    assert!((actual - value).norm() < 1e-12);
}

#[test]
fn test_tensorci2_from_tensor_train_matches_complex_lorentz_full_grid() {
    let coeff = Complex64::new(1.0, 2.0);
    let f = |idx: &MultiIndex| {
        let denom = idx
            .iter()
            .map(|&i| {
                let x = (i + 1) as f64;
                x * x
            })
            .sum::<f64>()
            + 1.0;
        coeff / Complex64::new(denom, 0.0)
    };
    let crate::TCI2OptimizationResult {
        tci: source,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<Complex64, _, fn(&[MultiIndex]) -> Vec<Complex64>>(
        f,
        None,
        vec![4; 4],
        vec![vec![0; 4]],
        TCI2Options {
            tolerance: 1e-12,
            max_iter: 20,
            max_bond_dim: 5,
            ..TCI2Options::default()
        },
    )
    .unwrap();
    let source_tt = source.to_tensor_train().unwrap();
    let (expected_data, expected_shape) = source_tt.fulltensor();
    let converted = TensorCI2::from_tensor_train(
        source_tt,
        TensorCI2FromTensorTrainOptions {
            tolerance: 1e-12,
            max_bond_dim: 5,
            ..TensorCI2FromTensorTrainOptions::default()
        },
    )
    .unwrap();
    let (actual_data, actual_shape) = converted.to_tensor_train().unwrap().fulltensor();

    assert_eq!(actual_shape, expected_shape);
    assert_eq!(converted.link_dims(), source.link_dims());
    for (&actual, &expected) in actual_data.iter().zip(expected_data.iter()) {
        assert!((actual - expected).norm() < 1e-10);
    }
}

#[test]
fn test_tensorci2_from_tensor_train_preserves_nontrivial_tensor() {
    let f = |idx: &MultiIndex| {
        let x = (idx[0] + 1) as f64;
        let y = (idx[1] + 2) as f64;
        let z = (idx[2] + 3) as f64;
        x * y + z
    };
    let crate::TCI2OptimizationResult {
        tci: source,
        ranks: _ranks,
        errors: _errors,
        ..
    } = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
        f,
        None,
        vec![3, 3, 3],
        vec![vec![2, 2, 2]],
        TCI2Options {
            tolerance: 1e-12,
            max_iter: 10,
            ..TCI2Options::default()
        },
    )
    .unwrap();
    let source_tt = source.to_tensor_train().unwrap();
    let (expected_data, expected_shape) = source_tt.fulltensor();
    let converted =
        TensorCI2::from_tensor_train(source_tt, TensorCI2FromTensorTrainOptions::default())
            .unwrap();
    let (actual_data, actual_shape) = converted.to_tensor_train().unwrap().fulltensor();

    assert_eq!(actual_shape, expected_shape);
    assert_eq!(converted.link_dims(), source.link_dims());
    for (&actual, &expected) in actual_data.iter().zip(expected_data.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn test_tensorci2_from_index_sets_rejects_wrong_lengths() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let err = TensorCI2::from_index_sets(vec![2, 2], vec![vec![vec![]]], vec![], &f).unwrap_err();

    assert!(err.to_string().contains("I/J set length"));
}

#[test]
fn test_tensorci2_from_index_sets_preserves_explicit_sets_and_samples() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let tci = TensorCI2::from_index_sets(
        vec![4, 4],
        vec![vec![vec![]], vec![vec![0], vec![1]]],
        vec![vec![vec![0], vec![1]], vec![vec![]]],
        &f,
    )
    .unwrap();
    let tt = tci.to_tensor_train().unwrap();

    assert_eq!(tci.i_set(0), &[Vec::<usize>::new()]);
    assert_eq!(tci.i_set(1), &[vec![0], vec![1]]);
    assert_eq!(tci.j_set(0), &[vec![0], vec![1]]);
    assert_eq!(tci.j_set(1), &[Vec::<usize>::new()]);
    assert!((tci.max_sample_value() - 5.0).abs() < 1e-12);
    assert_eq!(tci.link_dims(), vec![2]);
    assert!((tt.evaluate(&[0, 0]).unwrap() - 1.0).abs() < 1e-10);
    assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
}

#[test]
fn test_tensorci2_from_index_sets_rejects_out_of_range_indices() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let err = TensorCI2::from_index_sets(
        vec![2, 2],
        vec![vec![vec![]], vec![vec![2]]],
        vec![vec![vec![0]], vec![vec![]]],
        &f,
    )
    .unwrap_err();

    assert!(matches!(err, crate::TCIError::IndexOutOfBounds { .. }));
}

#[test]
fn test_tensorci2_from_index_sets_rejects_rank_mismatch() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let err = TensorCI2::from_index_sets(
        vec![3, 3],
        vec![vec![vec![]], vec![vec![0], vec![1]]],
        vec![vec![vec![0]], vec![vec![]]],
        &f,
    )
    .unwrap_err();

    assert!(err.to_string().contains("rank mismatch"));
}

#[test]
fn test_tensorci2_from_index_sets_rejects_duplicate_indices() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let err = TensorCI2::from_index_sets(
        vec![3, 3],
        vec![vec![vec![]], vec![vec![0], vec![0]]],
        vec![vec![vec![0], vec![1]], vec![vec![]]],
        &f,
    )
    .unwrap_err();

    assert!(err.to_string().contains("duplicate"));
}

#[test]
fn test_tensorci2_from_index_sets_rejects_zero_samples() {
    let zero = |_idx: &MultiIndex| 0.0_f64;
    let err = TensorCI2::from_index_sets(
        vec![2, 2],
        vec![vec![vec![]], vec![vec![0]]],
        vec![vec![vec![0]], vec![vec![]]],
        &zero,
    )
    .unwrap_err();

    assert!(matches!(err, crate::TCIError::InvalidPivot { .. }));
}
