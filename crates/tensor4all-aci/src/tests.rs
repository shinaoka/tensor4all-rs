use crate::local::LocalInputSetupTiming;
use crate::validation::{validate_inputs, validate_options};
use crate::{
    elementwise,
    elementwise::{
        convergence_criterion_like_julia, error_metric, max_error_metric, ranks_are_stable,
    },
    elementwise_batched, initial_guess,
    random_tt::{
        initial_guess_core_entry_count, initial_guess_existing_entry_count,
        initial_guess_total_entry_count, MAX_INITIAL_GUESS_CORE_ENTRIES,
    },
    AciError, AciOptions, ElementwiseBatch, ElementwiseProblem, LocalBlockEvaluator,
};
use num_complex::Complex64;
use std::cell::RefCell;
use std::time::{Duration, Instant};
use tensor4all_simplett::{
    tensor3_from_data, tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain,
};
use tensor4all_tcicore::{matrix_luci_factors_from_matrix_owned, RrLUOptions};
use tensor4all_tensorbackend::Matrix;

fn tensor_train_with_link_dims(site_dims: &[usize], link_dims: &[usize]) -> TensorTrain<f64> {
    assert_eq!(link_dims.len(), site_dims.len().saturating_sub(1));
    let cores = site_dims
        .iter()
        .enumerate()
        .map(|(site, &site_dim)| {
            let left_dim = if site == 0 { 1 } else { link_dims[site - 1] };
            let right_dim = link_dims.get(site).copied().unwrap_or(1);
            tensor3_zeros(left_dim, site_dim, right_dim)
        })
        .collect();
    TensorTrain::new(cores).unwrap()
}

fn local_test_problem() -> ElementwiseProblem<f64> {
    let input_a = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
            2,
            2,
            3,
        )
        .unwrap(),
        tensor3_from_data(
            vec![
                5.0, 6.0, 7.0, 50.0, 60.0, 70.0, 500.0, 600.0, 700.0, 8.0, 9.0, 10.0,
            ],
            3,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let input_b = TensorTrain::new(vec![
        tensor3_from_data(vec![2.0, 3.0, 20.0, 30.0], 1, 2, 2).unwrap(),
        tensor3_from_data(
            vec![
                2.0, 3.0, 4.0, 5.0, 20.0, 30.0, 40.0, 50.0, 200.0, 300.0, 400.0, 500.0,
            ],
            2,
            2,
            3,
        )
        .unwrap(),
        tensor3_from_data(
            vec![
                6.0, 7.0, 8.0, 60.0, 70.0, 80.0, 600.0, 700.0, 800.0, 9.0, 10.0, 11.0,
            ],
            3,
            2,
            2,
        )
        .unwrap(),
        tensor3_from_data(vec![2.0, 3.0, 4.0, 5.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem =
        ElementwiseProblem::new(vec![input_a, input_b], AciOptions::default()).unwrap();
    for input in 0..problem.n_inputs() {
        problem.left_frames[input][1] =
            Some(Matrix::from_col_major_vec(2, 2, vec![1.0, 2.0, 10.0, 20.0]));
        problem.right_frames[input][3] =
            Some(Matrix::from_col_major_vec(2, 2, vec![3.0, 4.0, 30.0, 40.0]));
    }
    problem
}

fn explicit_local_value(
    problem: &ElementwiseProblem<f64>,
    input: usize,
    bond: usize,
    row: usize,
    col: usize,
) -> f64 {
    let left_frame = problem.left_frames[input][bond].as_ref().unwrap();
    let right_frame = problem.right_frames[input][bond + 2].as_ref().unwrap();
    let left_core = problem.inputs[input].site_tensor(bond);
    let right_core = problem.inputs[input].site_tensor(bond + 1);
    let left_pivot = row % left_frame.nrows();
    let site_left = row / left_frame.nrows();
    let site_right = col % right_core.site_dim();
    let right_pivot = col / right_core.site_dim();
    let mut sum = 0.0;
    for a in 0..left_core.left_dim() {
        for m in 0..left_core.right_dim() {
            for b in 0..right_core.right_dim() {
                sum += left_frame[[left_pivot, a]]
                    * *left_core.get3(a, site_left, m)
                    * *right_core.get3(m, site_right, b)
                    * right_frame[[b, right_pivot]];
            }
        }
    }
    sum
}

fn multiply_batch(batch: ElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    assert_eq!(output.len(), batch.n_points());
    for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
        *value = batch.get(0, point).unwrap() * batch.get(1, point).unwrap();
    }
    Ok(())
}

fn zero_batch(batch: ElementwiseBatch<'_, f64>, output: &mut [f64]) -> crate::Result<()> {
    assert_eq!(output.len(), batch.n_points());
    output.fill(0.0);
    Ok(())
}

fn assert_solution_is_zero_on_binary_three_site_grid(problem: &ElementwiseProblem<f64>) {
    for indices in [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ] {
        assert_eq!(problem.solution.evaluate(&indices).unwrap(), 0.0);
    }
}

#[test]
fn elementwise_multiplies_constant_tensor_trains() {
    let input_a = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 3, 2], 4.0);

    let result = elementwise(
        |values| values[0] * values[1],
        &[input_a, input_b],
        &AciOptions::default(),
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                let value = result.tensor_train.evaluate(&[i, j, k]).unwrap();
                assert!((value - 8.0).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn scalar_and_batched_paths_match() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 5.0);
    let inputs = [input_a, input_b];
    let options = AciOptions::default();

    let scalar_result = elementwise(|values| values[0] + values[1], &inputs, &options).unwrap();
    let batched_result = elementwise_batched(
        |batch, output| {
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                *value = batch.get(0, point)? + batch.get(1, point)?;
            }
            Ok(())
        },
        &inputs,
        &options,
    )
    .unwrap();

    for i in 0..2 {
        for j in 0..2 {
            let scalar_value = scalar_result.tensor_train.evaluate(&[i, j]).unwrap();
            let batched_value = batched_result.tensor_train.evaluate(&[i, j]).unwrap();
            assert!((scalar_value - batched_value).abs() < 1e-12);
            assert!((scalar_value - 7.0).abs() < 1e-12);
        }
    }
}

#[test]
fn elementwise_batched_propagates_operator_error() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 4.0);
    let options = AciOptions {
        max_iters: 1,
        min_iters: 1,
        ..AciOptions::default()
    };

    let err = elementwise_batched(
        |_batch, _output| {
            Err(AciError::Operator {
                message: "public operator failed".to_string(),
            })
        },
        &[input_a, input_b],
        &options,
    )
    .unwrap_err();

    assert!(matches!(err, AciError::Operator { .. }));
    assert!(err.to_string().contains("public operator failed"));
}

#[test]
fn elementwise_single_site_scalar_evaluates_operator() {
    let input_a = TensorTrain::<f64>::constant(&[3], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[3], 4.0);

    let result = elementwise(
        |values| values[0] * values[1] + 1.0,
        &[input_a, input_b],
        &AciOptions::default(),
    )
    .unwrap();

    assert_eq!(result.tensor_train.rank(), 1);
    assert!(result.ranks.is_empty());
    assert!(result.errors.is_empty());
    for i in 0..3 {
        let value = result.tensor_train.evaluate(&[i]).unwrap();
        assert!((value - 9.0).abs() < 1e-12);
    }
}

#[test]
fn elementwise_batched_single_site_uses_column_major_batch_layout() {
    let input_a = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0], 1, 3, 1).unwrap()
    ])
    .unwrap();
    let input_b = TensorTrain::new(vec![
        tensor3_from_data(vec![10.0, 20.0, 30.0], 1, 3, 1).unwrap()
    ])
    .unwrap();
    let observed_batches = std::cell::RefCell::new(Vec::new());

    let result = elementwise_batched(
        |batch, output| {
            observed_batches
                .borrow_mut()
                .push(batch.as_col_major_slice().to_vec());
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                *value = batch.get(0, point)? + batch.get(1, point)?;
            }
            Ok(())
        },
        &[input_a, input_b],
        &AciOptions::default(),
    )
    .unwrap();

    assert_eq!(
        observed_batches.into_inner(),
        vec![vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]]
    );
    assert_eq!(result.tensor_train.evaluate(&[0]).unwrap(), 11.0);
    assert_eq!(result.tensor_train.evaluate(&[1]).unwrap(), 22.0);
    assert_eq!(result.tensor_train.evaluate(&[2]).unwrap(), 33.0);
}

#[test]
fn elementwise_batched_single_site_propagates_operator_error() {
    let input_a = TensorTrain::<f64>::constant(&[3], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[3], 4.0);

    let err = elementwise_batched(
        |_batch, _output| {
            Err(AciError::Operator {
                message: "single-site operator failed".to_string(),
            })
        },
        &[input_a, input_b],
        &AciOptions::default(),
    )
    .unwrap_err();

    assert!(matches!(err, AciError::Operator { .. }));
    assert!(err.to_string().contains("single-site operator failed"));
}

#[test]
fn relative_error_metric_normalizes_by_sampled_scale() {
    assert_eq!(error_metric(2.0, 100.0, true), 0.02);
    assert_eq!(error_metric(2.0, 100.0, false), 2.0);
    assert_eq!(error_metric(2.0, 0.0, true), 2.0);
}

#[test]
fn relative_error_metric_pairs_each_bond_with_its_scale() {
    let errors = [1.0, 10.0];
    let scales = [1.0, 10_000.0];

    assert_eq!(max_error_metric(&errors, &scales, true), 1.0);
    assert_eq!(max_error_metric(&errors, &scales, false), 10.0);
    assert_eq!(max_error_metric(&errors, &[], true), 10.0);
}

#[test]
fn rank_stability_matches_julia_min_iter_window() {
    assert!(ranks_are_stable(&[10, 12, 12], 2));
    assert!(!ranks_are_stable(&[10, 12, 13], 2));
    assert!(!ranks_are_stable(&[10, 12], 2));
    assert!(ranks_are_stable(&[10, 10], 2));
}

#[test]
fn convergence_criterion_matches_julia_algorithm() {
    let ranks = [10, 12, 12];
    let errors = [1.0e-8, 2.0e-11, 3.0e-11];

    assert!(!convergence_criterion_like_julia(
        1,
        &ranks[..1],
        &errors[..1],
        2,
        1.0e-10
    ));
    assert!(!convergence_criterion_like_julia(
        2,
        &ranks[..2],
        &errors[..2],
        2,
        1.0e-10
    ));
    assert!(convergence_criterion_like_julia(
        3, &ranks, &errors, 2, 1.0e-10
    ));

    let increasing_ranks = [10, 12, 13];
    assert!(!convergence_criterion_like_julia(
        3,
        &increasing_ranks,
        &errors,
        2,
        1.0e-10
    ));
}

#[test]
fn default_options_are_conservative() {
    let options = AciOptions::<f64>::default();
    assert_eq!(options.max_iters, 20);
    assert_eq!(options.min_iters, 2);
    assert_eq!(options.max_bond_dim, usize::MAX);
    assert!((options.tolerance - 1e-12).abs() < 1e-15);
    assert!(!options.scale_tolerance);
    assert!(options.initial_guess.is_none());
}

#[test]
fn elementwise_batch_uses_column_major_input_point_layout() {
    let values = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
    let batch = ElementwiseBatch::new(&values, 2, 3).unwrap();
    assert_eq!(batch.n_inputs(), 2);
    assert_eq!(batch.n_points(), 3);
    assert_eq!(batch.get(0, 0).unwrap(), 10.0);
    assert_eq!(batch.get(1, 0).unwrap(), 20.0);
    assert_eq!(batch.get(0, 2).unwrap(), 12.0);
    assert_eq!(batch.get(1, 2).unwrap(), 22.0);
    assert_eq!(batch.as_col_major_slice(), values.as_slice());
}

#[test]
fn elementwise_batch_rejects_bad_length() {
    let err = ElementwiseBatch::<f64>::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err.to_string().contains("length"));
}

#[test]
fn elementwise_batch_rejects_zero_inputs() {
    let err = ElementwiseBatch::<f64>::new(&[], 0, 1).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("at least one input"));
}

#[test]
fn elementwise_batch_rejects_zero_points() {
    let err = ElementwiseBatch::<f64>::new(&[], 1, 0).unwrap_err();
    assert!(matches!(err, AciError::EmptyInputs));
    assert!(err.to_string().contains("interpolation point"));
}

#[test]
fn elementwise_batch_rejects_shape_overflow() {
    let err = ElementwiseBatch::<f64>::new(&[], usize::MAX, 2).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("overflows usize"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_input_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(2, 0).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "input",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("input index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}

#[test]
fn elementwise_batch_rejects_out_of_range_point_index() {
    let values = [10.0, 20.0, 11.0, 21.0];
    let batch = ElementwiseBatch::new(&values, 2, 2).unwrap();

    let err = batch.get(0, 2).unwrap_err();
    assert!(matches!(
        err,
        AciError::BatchIndexOutOfBounds {
            axis: "point",
            index: 2,
            len: 2,
        }
    ));
    let message = err.to_string();
    assert!(message.contains("point index"));
    assert!(message.contains("out of bounds"));
    assert!(message.contains("len 2"));
}

#[test]
fn elementwise_problem_initializes_boundary_frames() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();
    assert_eq!(problem.len(), 3);
    assert_eq!(problem.n_inputs(), 2);
    assert_eq!(problem.left_frame_shape(0, 0), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 3), Some((1, 1)));
    assert_eq!(problem.pivot_errors, vec![0.0, 0.0]);
}

#[test]
fn elementwise_problem_initializes_all_rank_one_right_frames() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();

    for input in 0..problem.n_inputs() {
        assert_eq!(
            problem.right_frame_shape(input, problem.len()),
            Some((1, 1))
        );
        for site in 1..problem.len() {
            assert_eq!(problem.right_frame_shape(input, site), Some((1, 1)));
        }
        assert_eq!(problem.right_frame_shape(input, 0), None);
    }
}

#[test]
fn elementwise_problem_handles_one_site_input() {
    let input = TensorTrain::<f64>::constant(&[2], 3.0);
    let problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();
    assert_eq!(problem.len(), 1);
    assert_eq!(problem.left_frame_shape(0, 0), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 0), None);
}

#[test]
fn elementwise_problem_preserves_initial_guess_values() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let tt = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 3.0, 4.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        initial_guess: Some(tt.clone()),
        ..AciOptions::default()
    };

    let problem = ElementwiseProblem::new(vec![input], options).unwrap();

    for indices in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        assert!(
            (problem.solution.evaluate(&indices).unwrap() - tt.evaluate(&indices).unwrap()).abs()
                < 1e-12
        );
    }
}

#[test]
fn elementwise_problem_updates_left_frame_for_selected_rows() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_left_frame(0, 0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 1)));
}

#[test]
fn elementwise_problem_updates_left_frame_values() {
    let input = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![3.0, 4.0, 5.0, 6.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_left_frame(0, 0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 2)));
    assert_eq!(problem.left_frame_value(0, 1, 0, 0), Some(1.0));
    assert_eq!(problem.left_frame_value(0, 1, 1, 0), Some(2.0));
    assert_eq!(problem.left_frame_value(0, 1, 0, 1), Some(10.0));
    assert_eq!(problem.left_frame_value(0, 1, 1, 1), Some(20.0));
    assert_eq!(problem.left_frame_value(1, 1, 0, 0), None);
    assert_eq!(problem.left_frame_value(0, 2, 0, 0), None);
    assert_eq!(problem.left_frame_value(0, 1, 2, 0), None);
    assert_eq!(problem.left_frame_value(0, 1, 0, 2), None);
}

#[test]
fn elementwise_problem_updates_all_left_frames_for_selected_rows() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let mut problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();

    problem.update_left_frames(0, &[0, 1]).unwrap();

    assert_eq!(problem.left_frame_shape(0, 1), Some((2, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((2, 1)));
}

#[test]
fn elementwise_problem_updates_right_frame_for_selected_columns() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_right_frame(0, 2, &[0, 1]).unwrap();

    assert_eq!(problem.right_frame_shape(0, 2), Some((1, 2)));
}

#[test]
fn elementwise_problem_updates_right_frame_values() {
    let input = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0, 2.0, 10.0, 20.0], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![3.0, 30.0, 4.0, 40.0], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    problem.update_right_frame(0, 1, &[0, 1]).unwrap();

    assert_eq!(problem.right_frame_shape(0, 1), Some((2, 2)));
    assert_eq!(problem.right_frame_value(0, 1, 0, 0), Some(3.0));
    assert_eq!(problem.right_frame_value(0, 1, 1, 0), Some(30.0));
    assert_eq!(problem.right_frame_value(0, 1, 0, 1), Some(4.0));
    assert_eq!(problem.right_frame_value(0, 1, 1, 1), Some(40.0));
    assert_eq!(problem.right_frame_value(1, 1, 0, 0), None);
    assert_eq!(problem.right_frame_value(0, 3, 0, 0), None);
    assert_eq!(problem.right_frame_value(0, 1, 2, 0), None);
    assert_eq!(problem.right_frame_value(0, 1, 0, 2), None);
}

#[test]
fn elementwise_problem_rejects_invalid_frame_selection_indices() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let mut problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    let left_err = problem.update_left_frame(0, 0, &[2]).unwrap_err();
    assert!(matches!(left_err, AciError::InvalidInitialGuess { .. }));
    assert!(left_err.to_string().contains("row index"));

    let right_err = problem.update_right_frame(0, 2, &[2]).unwrap_err();
    assert!(matches!(right_err, AciError::InvalidInitialGuess { .. }));
    assert!(right_err.to_string().contains("column index"));
}

#[test]
fn elementwise_problem_one_bond_local_update_matches_dense_product_left_orthogonal() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = multiply_batch;

    problem
        .local_update(0, true, &options, &mut operator)
        .unwrap();

    for indices in [[0, 0], [1, 0], [0, 1], [1, 1]] {
        assert!((problem.solution.evaluate(&indices).unwrap() - 6.0).abs() < 1e-12);
    }
    assert_eq!(problem.left_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((1, 1)));
    assert_eq!(problem.pivot_errors.len(), 1);
    assert!(problem.pivot_errors[0] <= 1e-12);
}

#[test]
fn elementwise_problem_one_bond_local_update_matches_dense_product_right_orthogonal() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    for input in 0..problem.n_inputs() {
        problem.right_frames[input][1] = None;
    }
    let mut operator = multiply_batch;

    problem
        .local_update(0, false, &options, &mut operator)
        .unwrap();

    for indices in [[0, 0], [1, 0], [0, 1], [1, 1]] {
        assert!((problem.solution.evaluate(&indices).unwrap() - 6.0).abs() < 1e-12);
    }
    assert_eq!(problem.right_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(1, 1), Some((1, 1)));
    assert_eq!(problem.pivot_errors.len(), 1);
    assert!(problem.pivot_errors[0] <= 1e-12);
}

#[test]
fn elementwise_problem_one_bond_local_update_returns_operator_error_side_channel() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = |_batch: ElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        Err(AciError::Operator {
            message: "side-channel operator failure".to_string(),
        })
    };

    let err = problem
        .local_update(0, true, &options, &mut operator)
        .unwrap_err();

    assert!(matches!(err, AciError::Operator { .. }));
    assert!(err.to_string().contains("side-channel operator failure"));
}

#[test]
fn elementwise_problem_one_bond_zero_update_keeps_left_frames_nonzero_dimensional() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    let mut operator = zero_batch;

    problem
        .local_update(0, true, &options, &mut operator)
        .unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    assert_eq!(problem.left_frame_shape(0, 1), Some((1, 1)));
    assert_eq!(problem.left_frame_shape(1, 1), Some((1, 1)));
    assert!(problem.local_input_value(0, 1, 0, 0).is_ok());

    problem
        .local_update(1, true, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn elementwise_problem_one_bond_zero_update_keeps_right_frames_nonzero_dimensional() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let options = AciOptions::<f64>::default();
    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();
    problem.update_left_frames(0, &[0]).unwrap();
    for input in 0..problem.n_inputs() {
        problem.right_frames[input][2] = None;
    }
    let mut operator = zero_batch;

    problem
        .local_update(1, false, &options, &mut operator)
        .unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    assert_eq!(problem.right_frame_shape(0, 2), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(1, 2), Some((1, 1)));
    assert!(problem.local_input_value(0, 0, 0, 0).is_ok());

    problem
        .local_update(0, false, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn local_input_value_matches_explicit_two_site_contraction() {
    let problem = local_test_problem();

    let value = problem.local_input_value(0, 1, 3, 2).unwrap();
    let expected = explicit_local_value(&problem, 0, 1, 3, 2);

    assert_eq!(value, expected);
    assert_eq!(value, 265_133_700.0);
}

#[test]
fn local_input_factors_match_explicit_two_site_contraction_for_all_inputs() {
    let problem = local_test_problem();
    let factors = crate::local::local_input_factors_for_problem(&problem, 1).unwrap();

    assert_eq!(factors.len(), problem.n_inputs());
    for (input, factors) in factors.iter().enumerate().take(problem.n_inputs()) {
        let (nrows, ncols) = problem.local_input_shape(input, 1).unwrap();
        for row in 0..nrows {
            for col in 0..ncols {
                let actual = factors.value(row, col).unwrap();
                let expected = explicit_local_value(&problem, input, 1, row, col);
                assert_eq!(actual, expected);
            }
        }
    }

    assert_ne!(
        factors[0].value(3, 2).unwrap(),
        factors[1].value(3, 2).unwrap()
    );
}

#[test]
fn local_input_factors_report_bounds_errors_and_record_timing() {
    let problem = local_test_problem();
    let factors = crate::local::local_input_factors_for_problem(&problem, 1).unwrap();
    let (nrows, ncols) = problem.local_input_shape(0, 1).unwrap();

    let row_err = match factors[0].value(nrows, 0) {
        Ok(_) => panic!("expected row bounds error"),
        Err(err) => err,
    };
    assert!(row_err.to_string().contains("local row index"));

    let col_err = match factors[0].value(0, ncols) {
        Ok(_) => panic!("expected column bounds error"),
        Err(err) => err,
    };
    assert!(col_err.to_string().contains("local column index"));

    let mut timing = LocalInputSetupTiming::default();
    let timed = crate::local::local_input_factors_for_problem_with_timing(&problem, 1, &mut timing)
        .unwrap();

    assert_eq!(timed.len(), factors.len());
}

#[test]
fn tensor3_reshape_helpers_accept_and_reject_shapes() {
    let left_factor = Matrix::from_col_major_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);

    let core = crate::state::matrix_to_tensor3(&left_factor, 1, 2, 2).unwrap();
    assert_eq!(core.left_dim(), 1);
    assert_eq!(core.site_dim(), 2);
    assert_eq!(core.right_dim(), 2);

    let core = crate::state::matrix_into_tensor3(left_factor.clone(), 1, 2, 2).unwrap();
    assert_eq!(core.left_dim(), 1);
    assert_eq!(core.site_dim(), 2);
    assert_eq!(core.right_dim(), 2);

    let err = match crate::state::matrix_to_tensor3(&left_factor, 2, 2, 1) {
        Ok(_) => panic!("expected left factor shape error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("cannot reshape matrix"));

    let err = match crate::state::matrix_into_tensor3(left_factor, 2, 2, 1) {
        Ok(_) => panic!("expected owned left factor shape error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("cannot reshape matrix"));

    let right_factor =
        Matrix::from_col_major_vec(2, 6, (0..12).map(|value| value as f64).collect());
    let core = crate::state::right_factor_to_tensor3(&right_factor, 2, 3, 2).unwrap();
    assert_eq!(core.left_dim(), 2);
    assert_eq!(core.site_dim(), 3);
    assert_eq!(core.right_dim(), 2);

    let core = crate::state::right_factor_into_tensor3(right_factor.clone(), 2, 3, 2).unwrap();
    assert_eq!(core.left_dim(), 2);
    assert_eq!(core.site_dim(), 3);
    assert_eq!(core.right_dim(), 2);

    let err = match crate::state::right_factor_to_tensor3(&right_factor, 3, 2, 2) {
        Ok(_) => panic!("expected right factor shape error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("cannot reshape right factor"));

    let err = match crate::state::right_factor_into_tensor3(right_factor, 3, 2, 2) {
        Ok(_) => panic!("expected owned right factor shape error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("cannot reshape right factor"));
}

#[test]
fn local_block_evaluator_uses_matrix_luci_point_order_and_batch_layout() {
    let problem = local_test_problem();
    let rows = [0, 3];
    let cols = [1, 2];
    let mut out = vec![0.0; rows.len() * cols.len()];
    let observed_batches = std::cell::RefCell::new(Vec::new());

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        assert_eq!(batch.n_inputs(), problem.n_inputs());
        assert_eq!(batch.n_points(), rows.len() * cols.len());
        observed_batches
            .borrow_mut()
            .push(batch.as_col_major_slice().to_vec());
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point).unwrap() + 10.0 * batch.get(1, point).unwrap();
        }
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();

    let mut expected_values = Vec::new();
    for &col in &cols {
        for &row in &rows {
            for input in 0..problem.n_inputs() {
                expected_values.push(problem.local_input_value(input, 1, row, col).unwrap());
            }
        }
    }
    assert_eq!(observed_batches.into_inner(), vec![expected_values]);

    let mut expected_out = Vec::new();
    for &col in &cols {
        for &row in &rows {
            expected_out.push(
                problem.local_input_value(0, 1, row, col).unwrap()
                    + 10.0 * problem.local_input_value(1, 1, row, col).unwrap(),
            );
        }
    }
    assert_eq!(out, expected_out);
}

#[test]
fn local_block_evaluator_materializes_full_matrix_in_column_major_order() {
    let problem = local_test_problem();
    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point).unwrap() + 10.0 * batch.get(1, point).unwrap();
        }
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();
    let matrix = evaluator.materialize_local_matrix().unwrap();
    let rows = (0..evaluator.nrows()).collect::<Vec<_>>();
    let cols = (0..evaluator.ncols()).collect::<Vec<_>>();
    let mut expected = vec![0.0; rows.len() * cols.len()];

    evaluator
        .fill_local_block(&rows, &cols, &mut expected)
        .unwrap();

    assert_eq!(matrix.nrows(), evaluator.nrows());
    assert_eq!(matrix.ncols(), evaluator.ncols());
    assert_eq!(matrix.as_col_major_slice(), expected);
}

#[test]
fn local_block_evaluator_serves_duplicate_entries_from_cache() {
    let problem = local_test_problem();
    let rows = [0, 0, 1];
    let cols = [0];
    let mut first = vec![0.0; rows.len() * cols.len()];
    let mut second = vec![0.0; rows.len() * cols.len()];
    let calls = std::cell::Cell::new(0usize);

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        calls.set(calls.get() + batch.n_points());
        for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
            *value = batch.get(0, point).unwrap() * batch.get(1, point).unwrap();
        }
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator
        .fill_local_block(&rows, &cols, &mut first)
        .unwrap();
    evaluator
        .fill_local_block(&rows, &cols, &mut second)
        .unwrap();

    assert_eq!(calls.get(), 2);
    assert_eq!(first[0], first[1]);
    assert_eq!(first, second);
}

#[test]
fn local_block_evaluator_cache_can_be_cleared() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut out = vec![0.0];
    let observed_points = std::cell::RefCell::new(Vec::new());

    let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        observed_points.borrow_mut().push(batch.n_points());
        output[0] = batch.get(0, 0).unwrap();
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();
    evaluator.clear_cache();
    evaluator.fill_local_block(&rows, &cols, &mut out).unwrap();

    assert_eq!(observed_points.into_inner(), vec![1, 1]);
}

#[test]
fn local_block_evaluators_with_different_operators_have_separate_caches() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut first = vec![0.0];
    let mut second = vec![0.0];

    let first_operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = batch.get(0, 0).unwrap();
        Ok(())
    };
    let second_operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = batch.get(0, 0).unwrap() + 1.0;
        Ok(())
    };

    let first_evaluator = LocalBlockEvaluator::new(&problem, 1, &first_operator).unwrap();
    let second_evaluator = LocalBlockEvaluator::new(&problem, 1, &second_operator).unwrap();

    first_evaluator
        .fill_local_block(&rows, &cols, &mut first)
        .unwrap();
    second_evaluator
        .fill_local_block(&rows, &cols, &mut second)
        .unwrap();

    assert_eq!(second, vec![first[0] + 1.0]);
}

#[test]
fn local_block_evaluator_or_zero_records_operator_error_once() {
    let problem = local_test_problem();
    let rows = [0];
    let cols = [0];
    let mut out = vec![123.0];
    let calls = std::cell::Cell::new(0);
    let operator = |_batch: ElementwiseBatch<'_, f64>, _output: &mut [f64]| {
        calls.set(calls.get() + 1);
        Err(AciError::Operator {
            message: format!("operator failed on call {}", calls.get()),
        })
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);
    assert_eq!(out, vec![0.0]);
    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);
    assert_eq!(out, vec![0.0]);

    let err = evaluator.take_error().unwrap();
    assert!(err.to_string().contains("call 1"));
    assert!(evaluator.take_error().is_none());
}

#[test]
fn local_block_evaluator_or_zero_records_local_error_once() {
    let problem = local_test_problem();
    let rows = [4];
    let cols = [0];
    let mut out = vec![123.0];
    let operator = |_batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        output[0] = 1.0;
        Ok(())
    };
    let evaluator = LocalBlockEvaluator::new(&problem, 1, &operator).unwrap();

    evaluator.fill_local_block_or_zero(&rows, &cols, &mut out);

    assert_eq!(out, vec![0.0]);
    let err = evaluator.take_error().unwrap();
    assert!(err.to_string().contains("row index"));
}

#[test]
fn local_input_value_rejects_out_of_range_indices() {
    let problem = local_test_problem();

    let input_err = problem.local_input_value(2, 1, 0, 0).unwrap_err();
    assert!(matches!(input_err, AciError::InvalidInitialGuess { .. }));
    assert!(input_err.to_string().contains("input index"));

    let bond_err = problem.local_input_value(0, 3, 0, 0).unwrap_err();
    assert!(matches!(bond_err, AciError::InvalidInitialGuess { .. }));
    assert!(bond_err.to_string().contains("bond index"));

    let row_err = problem.local_input_value(0, 1, 4, 0).unwrap_err();
    assert!(matches!(row_err, AciError::InvalidInitialGuess { .. }));
    assert!(row_err.to_string().contains("row index"));

    let col_err = problem.local_input_value(0, 1, 0, 4).unwrap_err();
    assert!(matches!(col_err, AciError::InvalidInitialGuess { .. }));
    assert!(col_err.to_string().contains("column index"));
}

#[test]
fn local_input_value_rejects_missing_frames() {
    let input = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let problem = ElementwiseProblem::new(vec![input], AciOptions::default()).unwrap();

    let err = problem.local_input_value(0, 1, 0, 0).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    assert!(err.to_string().contains("left frame"));
}

#[test]
fn validate_inputs_rejects_empty_inputs() {
    let err = validate_inputs::<f64>(&[]).unwrap_err();
    assert!(err.to_string().contains("at least one"));
}

#[test]
fn validate_inputs_rejects_zero_site_tensor_train() {
    let input = TensorTrain::<f64>::new(vec![]).unwrap();
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("at least one site"));
}

#[test]
fn validate_inputs_rejects_zero_physical_dim_in_first_input() {
    let input = tensor_train_with_link_dims(&[0, 3], &[1]);
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("site 0"));
    assert!(message.contains("dimension must be positive"));
}

#[test]
fn validate_inputs_rejects_zero_physical_dim_in_later_input() {
    let first = tensor_train_with_link_dims(&[2, 3], &[1]);
    let later = tensor_train_with_link_dims(&[2, 0], &[1]);
    let err = validate_inputs(&[first, later]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("site 1"));
    assert!(message.contains("dimension must be positive"));
}

#[test]
fn validate_inputs_rejects_zero_internal_bond_dim_in_first_input() {
    let input = tensor_train_with_link_dims(&[2, 3], &[0]);
    let err = validate_inputs(&[input]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn validate_inputs_rejects_zero_internal_bond_dim_in_later_input() {
    let first = tensor_train_with_link_dims(&[2, 3], &[1]);
    let later = tensor_train_with_link_dims(&[2, 3], &[0]);
    let err = validate_inputs(&[first, later]).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn validate_inputs_rejects_length_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("length mismatch"));
}

#[test]
fn validate_inputs_rejects_site_dim_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 3], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 4], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("site shape mismatch"));
}

#[test]
fn validate_options_rejects_zero_max_iters() {
    let options = AciOptions::<f64> {
        max_iters: 0,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("max_iters"));
}

#[test]
fn validate_options_rejects_zero_max_bond_dim() {
    let options = AciOptions::<f64> {
        max_bond_dim: 0,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("max_bond_dim"));
}

#[test]
fn validate_options_rejects_min_iters_above_max_iters() {
    let options = AciOptions::<f64> {
        min_iters: 3,
        max_iters: 2,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("min_iters"));
}

#[test]
fn validate_options_rejects_zero_min_iters() {
    let options = AciOptions::<f64> {
        min_iters: 0,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("min_iters"));
}

#[test]
fn validate_options_rejects_negative_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: -1e-12,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn validate_options_rejects_nan_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: f64::NAN,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn validate_options_rejects_infinite_tolerance() {
    let options = AciOptions::<f64> {
        tolerance: f64::INFINITY,
        ..AciOptions::default()
    };
    let err = validate_options(&options).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("tolerance"));
}

#[test]
fn default_initial_guess_matches_input_site_dims() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 4], 2.0);
    let guess = initial_guess(&[a, b], &AciOptions::default()).unwrap();
    assert_eq!(guess.site_dims(), vec![2, 3, 4]);
}

#[test]
fn initial_guess_link_dims_are_empty_for_one_site_input() {
    let input = TensorTrain::<f64>::constant(&[5], 1.0);
    let guess = initial_guess(&[input], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), Vec::<usize>::new());
}

#[test]
fn initial_guess_link_dims_are_limited_by_max_bond_dim() {
    let input = tensor_train_with_link_dims(&[8, 8, 8], &[8, 8]);
    let options = AciOptions::<f64> {
        max_bond_dim: 3,
        ..AciOptions::default()
    };
    let guess = initial_guess(&[input], &options).unwrap();
    assert_eq!(guess.link_dims(), vec![3, 3]);
}

#[test]
fn initial_guess_link_dims_are_limited_by_physical_left_right_products() {
    let input = tensor_train_with_link_dims(&[2, 3, 5, 7], &[20, 20, 20]);
    let guess = initial_guess(&[input], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), vec![2, 6, 7]);
}

#[test]
fn initial_guess_link_dims_are_limited_by_minimum_input_link_dim() {
    let wide = tensor_train_with_link_dims(&[8, 8, 8], &[8, 8]);
    let narrow = tensor_train_with_link_dims(&[8, 8, 8], &[5, 2]);
    let guess = initial_guess(&[wide, narrow], &AciOptions::default()).unwrap();
    assert_eq!(guess.link_dims(), vec![5, 2]);
}

#[test]
fn initial_guess_is_deterministic_for_same_seed() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let options = AciOptions::<f64> {
        max_bond_dim: 2,
        rng_seed: 1234,
        ..AciOptions::default()
    };

    let guess_a = initial_guess(&[a.clone(), b.clone()], &options).unwrap();
    let guess_b = initial_guess(&[a, b], &options).unwrap();

    for indices in [[0, 0, 0], [1, 2, 1], [0, 1, 1]] {
        assert_eq!(
            guess_a.evaluate(&indices).unwrap(),
            guess_b.evaluate(&indices).unwrap()
        );
    }
}

#[test]
fn initial_guess_accepts_compatible_explicit_guess() {
    let input = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let explicit = TensorTrain::<f64>::constant(&[2, 3], 7.0);
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let guess = initial_guess(&[input], &options).unwrap();

    assert_eq!(guess.site_dims(), vec![2, 3]);
    assert_eq!(guess.site_tensor(0).site_dim(), 2);
    assert_eq!(guess.site_tensor(1).site_dim(), 3);
    assert!((guess.evaluate(&[0, 0]).unwrap() - 7.0).abs() < 1e-12);
    assert!((guess.evaluate(&[1, 2]).unwrap() - 7.0).abs() < 1e-12);
}

#[test]
fn initial_guess_zero_initializes_nonzero_dimensional_right_frames() {
    let input_a = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let input_b = TensorTrain::<f64>::constant(&[2, 2, 2], 3.0);
    let zero_guess = tensor_train_with_link_dims(&[2, 2, 2], &[1, 1]);
    let options = AciOptions {
        initial_guess: Some(zero_guess),
        ..AciOptions::default()
    };

    let mut problem = ElementwiseProblem::new(vec![input_a, input_b], options.clone()).unwrap();

    assert_solution_is_zero_on_binary_three_site_grid(&problem);
    for input in 0..problem.n_inputs() {
        assert_eq!(problem.right_frame_shape(input, 1), Some((1, 1)));
        assert_eq!(problem.right_frame_shape(input, 2), Some((1, 1)));
        assert_eq!(problem.right_frame_shape(input, 3), Some((1, 1)));
    }
    problem.update_left_frames(0, &[0]).unwrap();
    assert!(problem.local_input_value(0, 1, 0, 0).is_ok());

    let mut operator = zero_batch;
    problem
        .local_update(1, false, &options, &mut operator)
        .unwrap();
    assert_solution_is_zero_on_binary_three_site_grid(&problem);
}

#[test]
fn initial_guess_rejects_incompatible_explicit_guess_site_dims() {
    let input = TensorTrain::<f64>::constant(&[2, 3], 2.0);
    let explicit = TensorTrain::<f64>::constant(&[2, 4], 7.0);
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    assert!(err.to_string().contains("site dimensions"));
}

#[test]
fn explicit_initial_guess_rejects_rank_above_max_bond_dim() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let explicit = TensorTrain::new(vec![
        tensor3_from_data(vec![1.0; 4], 1, 2, 2).unwrap(),
        tensor3_from_data(vec![1.0; 4], 2, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        max_bond_dim: 1,
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    let message = err.to_string();
    assert!(message.contains("bond dimension"));
    assert!(message.contains("max_bond_dim"));
}

#[test]
fn explicit_initial_guess_rejects_zero_bond_dimension() {
    let input = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let explicit = TensorTrain::new(vec![
        tensor3_from_data(Vec::<f64>::new(), 1, 2, 0).unwrap(),
        tensor3_from_data(Vec::<f64>::new(), 0, 2, 1).unwrap(),
    ])
    .unwrap();
    let options = AciOptions {
        initial_guess: Some(explicit),
        ..AciOptions::default()
    };

    let err = initial_guess(&[input], &options).unwrap_err();

    assert!(matches!(err, AciError::InvalidInitialGuess { .. }));
    let message = err.to_string();
    assert!(message.contains("dimension"));
    assert!(message.contains("positive"));
}

#[test]
fn complex_initial_guess_is_deterministic() {
    let a = TensorTrain::<Complex64>::constant(&[2, 3, 2], Complex64::new(1.0, 0.0));
    let b = TensorTrain::<Complex64>::constant(&[2, 3, 2], Complex64::new(2.0, 0.0));
    let options = AciOptions::<Complex64> {
        max_bond_dim: 2,
        rng_seed: 4321,
        ..AciOptions::default()
    };

    let guess_a = initial_guess(&[a.clone(), b.clone()], &options).unwrap();
    let guess_b = initial_guess(&[a, b], &options).unwrap();

    assert_eq!(guess_a.site_dims(), vec![2, 3, 2]);
    assert_eq!(guess_b.site_dims(), vec![2, 3, 2]);
    assert_eq!(
        guess_a.evaluate(&[1, 2, 1]).unwrap(),
        guess_b.evaluate(&[1, 2, 1]).unwrap()
    );
}

#[test]
fn initial_guess_existing_entry_count_matches_explicit_guess_cores() {
    let guess = tensor_train_with_link_dims(&[2, 3, 5], &[4, 6]);
    let entry_count = initial_guess_existing_entry_count(&guess).unwrap();
    assert_eq!(entry_count, 8 + 72 + 30);
}

#[test]
fn initial_guess_rejects_huge_non_overflowing_core_size() {
    let err = initial_guess_core_entry_count(MAX_INITIAL_GUESS_CORE_ENTRIES + 1, 1, 1).unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("initial guess core size"));
}

#[test]
fn initial_guess_rejects_oversized_total_entries() {
    let err = initial_guess_total_entry_count(&[(MAX_INITIAL_GUESS_CORE_ENTRIES, 1, 1), (1, 1, 1)])
        .unwrap_err();
    assert!(matches!(err, AciError::InvalidOptions { .. }));
    assert!(err.to_string().contains("initial guess total size"));
}

const STEP_TIMING_DEFAULT_N_SITES: usize = 12;
const STEP_TIMING_LOCAL_DIM: usize = 2;
const STEP_TIMING_N_INPUTS: usize = 2;
const STEP_TIMING_TOLERANCE: f64 = 1e-10;

#[derive(Clone, Copy, Default)]
struct LocalStepTiming {
    setup: Duration,
    setup_shape_validation: Duration,
    setup_dims: Duration,
    setup_left_factor: Duration,
    setup_right_factor: Duration,
    input_values: Duration,
    operator: Duration,
    matrix_luci: Duration,
    core_update: Duration,
    frame_update: Duration,
    updates: usize,
    sweeps: usize,
    final_rank: usize,
    final_error: f64,
}

impl LocalStepTiming {
    fn total(self) -> Duration {
        self.setup
            + self.input_values
            + self.operator
            + self.matrix_luci
            + self.core_update
            + self.frame_update
    }

    fn setup_other(self) -> Duration {
        self.setup
            .checked_sub(
                self.setup_shape_validation
                    + self.setup_dims
                    + self.setup_left_factor
                    + self.setup_right_factor,
            )
            .unwrap_or_default()
    }
}

fn step_timing_link_dims(n_sites: usize, local_dim: usize, chi: usize) -> Vec<usize> {
    (0..n_sites.saturating_sub(1))
        .map(|bond| {
            let left_sites = bond + 1;
            let right_sites = n_sites - left_sites;
            let max_exact_rank = local_dim.pow(left_sites.min(right_sites) as u32);
            chi.min(max_exact_rank).max(1)
        })
        .collect()
}

fn step_timing_core_value(
    input_index: usize,
    site: usize,
    physical: usize,
    left: usize,
    right: usize,
    left_dim: usize,
    right_dim: usize,
) -> f64 {
    let input = input_index as f64 + 1.0;
    let site = site as f64 + 1.0;
    let physical = physical as f64 + 1.0;
    let left = left as f64 + 1.0;
    let right = right as f64 + 1.0;
    let left_coord = left / (left_dim as f64 + 1.0);
    let right_coord = right / (right_dim as f64 + 1.0);
    let phase = 0.173 * input * site
        + 0.193 * physical
        + 0.071 * left * right
        + 0.109 * input * left
        + 0.131 * site * right;
    let bond_mix = 0.29 * phase.sin()
        + 0.23 * (0.157 * input * physical * right + 0.211 * site * left).cos()
        + 0.17 * (left_coord - right_coord) * physical;
    let site_value = 0.31 + bond_mix;
    let scale = ((left_dim * right_dim) as f64).powf(0.25);
    site_value / scale
}

fn step_timing_deterministic_tt(
    input_index: usize,
    n_sites: usize,
    chi: usize,
) -> TensorTrain<f64> {
    let links = step_timing_link_dims(n_sites, STEP_TIMING_LOCAL_DIM, chi);
    let cores = (0..n_sites)
        .map(|site| {
            let left_dim = if site == 0 { 1 } else { links[site - 1] };
            let right_dim = links.get(site).copied().unwrap_or(1);
            let mut data = vec![0.0; left_dim * STEP_TIMING_LOCAL_DIM * right_dim];
            for right in 0..right_dim {
                for physical in 0..STEP_TIMING_LOCAL_DIM {
                    for left in 0..left_dim {
                        data[left + left_dim * (physical + STEP_TIMING_LOCAL_DIM * right)] =
                            step_timing_core_value(
                                input_index,
                                site,
                                physical,
                                left,
                                right,
                                left_dim,
                                right_dim,
                            );
                    }
                }
            }
            tensor3_from_data(data, left_dim, STEP_TIMING_LOCAL_DIM, right_dim)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    TensorTrain::new(cores).unwrap()
}

fn step_timing_inputs(n_sites: usize, chi: usize) -> Vec<TensorTrain<f64>> {
    (0..STEP_TIMING_N_INPUTS)
        .map(|input| step_timing_deterministic_tt(input, n_sites, chi))
        .collect()
}

fn step_timing_multiply_batch(
    batch: ElementwiseBatch<'_, f64>,
    output: &mut [f64],
) -> crate::Result<()> {
    for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
        let mut product = 1.0;
        for input in 0..batch.n_inputs() {
            product *= batch.get(input, point)?;
        }
        *value = product;
    }
    Ok(())
}

fn timed_local_update<F>(
    problem: &mut ElementwiseProblem<f64>,
    bond: usize,
    left_orthogonal: bool,
    options: &AciOptions<f64>,
    op: &mut F,
    timing: &mut LocalStepTiming,
) -> crate::Result<()>
where
    F: for<'batch> FnMut(ElementwiseBatch<'batch, f64>, &mut [f64]) -> crate::Result<()>,
{
    let left_core = problem.solution.site_tensor(bond);
    let right_core = problem.solution.site_tensor(bond + 1);
    let left_solution_rank = left_core.left_dim();
    let site_dim_left = left_core.site_dim();
    let site_dim_right = right_core.site_dim();
    let right_solution_rank = right_core.right_dim();

    let (factors, sampled_scale) = {
        let op_cell = RefCell::new(op);
        let operator = |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
            let mut op_ref = op_cell.borrow_mut();
            (*op_ref)(batch, output)
        };

        let start = Instant::now();
        let mut setup_timing = LocalInputSetupTiming::default();
        let evaluator = LocalBlockEvaluator::new_with_setup_timing(
            problem,
            bond,
            &operator,
            &mut setup_timing,
        )?;
        timing.setup += start.elapsed();
        timing.setup_shape_validation += setup_timing.shape_validation;
        timing.setup_dims += setup_timing.dims;
        timing.setup_left_factor += setup_timing.left_factor;
        timing.setup_right_factor += setup_timing.right_factor;

        let start = Instant::now();
        let input_values = evaluator.materialize_input_values()?;
        timing.input_values += start.elapsed();

        let start = Instant::now();
        let local_matrix = evaluator.apply_operator_to_input_values(&input_values)?;
        timing.operator += start.elapsed();
        let sampled_scale = evaluator.max_output_abs();

        let start = Instant::now();
        let factors = matrix_luci_factors_from_matrix_owned(
            local_matrix,
            Some(RrLUOptions {
                max_rank: options.max_bond_dim,
                rel_tol: if options.scale_tolerance {
                    options.tolerance
                } else {
                    0.0
                },
                abs_tol: if options.scale_tolerance {
                    0.0
                } else {
                    options.tolerance
                },
                left_orthogonal,
            }),
        )?;
        timing.matrix_luci += start.elapsed();
        (factors, sampled_scale)
    };

    let pivot_error = factors.pivot_errors.last().copied().unwrap_or(0.0);
    let new_rank = factors.rank.max(1);
    let left_factor = if factors.rank == 0 {
        Matrix::zeros(left_solution_rank * site_dim_left, 1)
    } else {
        factors.left
    };
    let right_factor = if factors.rank == 0 {
        Matrix::zeros(1, site_dim_right * right_solution_rank)
    } else {
        factors.right
    };
    let row_indices = if factors.rank == 0 {
        vec![0]
    } else {
        factors.row_indices
    };
    let col_indices = if factors.rank == 0 {
        vec![0]
    } else {
        factors.col_indices
    };

    let start = Instant::now();
    let new_left_core = crate::state::matrix_into_tensor3(
        left_factor,
        left_solution_rank,
        site_dim_left,
        new_rank,
    )?;
    let new_right_core = crate::state::right_factor_into_tensor3(
        right_factor,
        new_rank,
        site_dim_right,
        right_solution_rank,
    )?;
    let solution_cores = problem.solution.site_tensors_mut();
    solution_cores[bond] = new_left_core;
    solution_cores[bond + 1] = new_right_core;
    timing.core_update += start.elapsed();

    let start = Instant::now();
    if left_orthogonal {
        problem.update_left_frames(bond, &row_indices)?;
    } else {
        problem.update_right_frames(bond + 1, &col_indices)?;
    }
    problem.pivot_errors[bond] = pivot_error;
    problem.pivot_scales[bond] = sampled_scale;
    timing.frame_update += start.elapsed();
    timing.updates += 1;
    Ok(())
}

fn timed_aci_run(
    n_sites: usize,
    chi: usize,
    min_iters: usize,
    fixed_sweeps: Option<usize>,
) -> LocalStepTiming {
    let inputs = step_timing_inputs(n_sites, chi);
    let max_iters = fixed_sweeps.unwrap_or(20).max(20);
    let options = AciOptions {
        max_iters,
        min_iters,
        tolerance: STEP_TIMING_TOLERANCE,
        initial_guess: Some(step_timing_deterministic_tt(
            STEP_TIMING_N_INPUTS,
            n_sites,
            chi,
        )),
        ..AciOptions::default()
    };
    let mut problem = ElementwiseProblem::new(inputs, options.clone()).unwrap();
    let mut timing = LocalStepTiming::default();
    let mut op = step_timing_multiply_batch;
    let mut ranks = Vec::new();
    let mut errors = Vec::new();

    for iteration in 0..options.max_iters {
        let forward = iteration % 2 == 0;
        if forward {
            for bond in 0..problem.len() - 1 {
                timed_local_update(&mut problem, bond, true, &options, &mut op, &mut timing)
                    .unwrap();
            }
        } else {
            for bond in (0..problem.len() - 1).rev() {
                timed_local_update(&mut problem, bond, false, &options, &mut op, &mut timing)
                    .unwrap();
            }
        }

        let max_error = max_error_metric(
            &problem.pivot_errors,
            &problem.pivot_scales,
            options.scale_tolerance,
        );
        ranks.push(problem.solution.rank());
        errors.push(max_error);
        timing.sweeps += 1;
        timing.final_rank = problem.solution.rank();
        timing.final_error = max_error;

        if fixed_sweeps.is_some_and(|target| timing.sweeps >= target) {
            break;
        }

        if fixed_sweeps.is_none()
            && convergence_criterion_like_julia(
                iteration + 1,
                &ranks,
                &errors,
                options.min_iters,
                options.tolerance,
            )
        {
            break;
        }
    }
    timing
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn median_ms(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

#[test]
#[ignore]
fn local_update_step_timing() {
    let repeats = std::env::var("T4A_STEP_TIMING_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    let n_sites = std::env::var("T4A_STEP_TIMING_N_SITES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(STEP_TIMING_DEFAULT_N_SITES);
    assert!(n_sites >= 2, "T4A_STEP_TIMING_N_SITES must be at least 2");
    let min_iters = std::env::var("T4A_STEP_TIMING_MIN_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(AciOptions::<f64>::default().min_iters);
    assert!(
        min_iters >= 1,
        "T4A_STEP_TIMING_MIN_ITERS must be at least 1"
    );
    let fixed_sweeps = std::env::var("T4A_STEP_TIMING_FIXED_SWEEPS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let chis = std::env::var("T4A_STEP_TIMING_CHIS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![2, 4, 8, 16]);
    println!(
        "impl,chi,n_sites,repeats,min_iters,fixed_sweeps,n_sweeps,n_updates,setup_ms,setup_shape_ms,setup_dims_ms,setup_left_factor_ms,setup_right_factor_ms,setup_other_ms,input_values_ms,operator_ms,matrix_luci_ms,core_update_ms,frame_update_ms,total_ms,final_rank,final_error"
    );
    for chi in chis {
        let runs = (0..repeats)
            .map(|_| timed_aci_run(n_sites, chi, min_iters, fixed_sweeps))
            .collect::<Vec<_>>();
        let sweeps = runs[0].sweeps;
        let updates = runs[0].updates;
        let final_rank = runs[0].final_rank;
        let final_error = runs[0].final_error;
        let fixed_sweeps_value = fixed_sweeps.unwrap_or(0);
        println!(
            "rust,{chi},{n_sites},{repeats},{min_iters},{fixed_sweeps_value},{sweeps},{updates},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{final_rank},{final_error:.6e}",
            median_ms(runs.iter().map(|run| duration_ms(run.setup)).collect()),
            median_ms(
                runs.iter()
                    .map(|run| duration_ms(run.setup_shape_validation))
                    .collect()
            ),
            median_ms(runs.iter().map(|run| duration_ms(run.setup_dims)).collect()),
            median_ms(
                runs.iter()
                    .map(|run| duration_ms(run.setup_left_factor))
                    .collect()
            ),
            median_ms(
                runs.iter()
                    .map(|run| duration_ms(run.setup_right_factor))
                    .collect()
            ),
            median_ms(runs.iter().map(|run| duration_ms(run.setup_other())).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.input_values)).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.operator)).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.matrix_luci)).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.core_update)).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.frame_update)).collect()),
            median_ms(runs.iter().map(|run| duration_ms(run.total())).collect()),
        );
    }
}
