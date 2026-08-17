use approx::assert_abs_diff_eq;
use tensor4all_aci::{elementwise, elementwise_batched, AciOptions, ElementwiseBatch};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};

type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

fn exact_options() -> AciOptions<f64> {
    AciOptions {
        max_iters: 8,
        min_iters: 2,
        max_bond_dim: Some(8),
        tolerance: 1e-12,
        ..AciOptions::default()
    }
}

fn assert_dense_close(actual: &SimpleTensorTrain<f64>, expected_shape: &[usize], expected: &[f64]) {
    let (actual_data, actual_shape) = actual.full_tensor().unwrap();
    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual_data.len(), expected.len());
    for (actual_value, expected_value) in actual_data.iter().zip(expected) {
        assert_abs_diff_eq!(actual_value, expected_value, epsilon = 1e-9);
    }
}

fn dense_oracle2<F>(
    left: &SimpleTensorTrain<f64>,
    right: &SimpleTensorTrain<f64>,
    mut op: F,
) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let (left_data, left_shape) = left.full_tensor().unwrap();
    let (right_data, right_shape) = right.full_tensor().unwrap();
    assert_eq!(left_shape, right_shape);
    left_data
        .into_iter()
        .zip(right_data)
        .map(|(left_value, right_value)| op(left_value, right_value))
        .collect()
}

fn col_major_index3(
    left: usize,
    site: usize,
    right: usize,
    left_dim: usize,
    site_dim: usize,
) -> usize {
    left + left_dim * (site + site_dim * right)
}

fn two_site_tt_from_col_major(
    data: &[f64],
    shape: [usize; 2],
) -> TestResult<SimpleTensorTrain<f64>> {
    assert_eq!(data.len(), shape[0] * shape[1]);
    let left_site_dim = shape[0];
    let right_site_dim = shape[1];
    let bond_dim = left_site_dim;

    let mut left_core = vec![0.0; left_site_dim * bond_dim];
    for site in 0..left_site_dim {
        left_core[col_major_index3(0, site, site, 1, left_site_dim)] = 1.0;
    }

    let mut right_core = vec![0.0; bond_dim * right_site_dim];
    for right_site in 0..right_site_dim {
        for bond in 0..bond_dim {
            let dense_index = bond + left_site_dim * right_site;
            right_core[col_major_index3(bond, right_site, 0, bond_dim, right_site_dim)] =
                data[dense_index];
        }
    }

    Ok(SimpleTensorTrain::new(vec![
        tensor3_from_data(left_core, 1, left_site_dim, bond_dim)?,
        tensor3_from_data(right_core, bond_dim, right_site_dim, 1)?,
    ])?)
}

fn three_site_tt_from_col_major(
    data: &[f64],
    shape: [usize; 3],
) -> TestResult<SimpleTensorTrain<f64>> {
    assert_eq!(data.len(), shape.iter().product::<usize>());
    let first_site_dim = shape[0];
    let second_site_dim = shape[1];
    let third_site_dim = shape[2];
    let first_bond_dim = first_site_dim;
    let second_bond_dim = first_site_dim * second_site_dim;

    let mut first_core = vec![0.0; first_site_dim * first_bond_dim];
    for first_site in 0..first_site_dim {
        first_core[col_major_index3(0, first_site, first_site, 1, first_site_dim)] = 1.0;
    }

    let mut second_core = vec![0.0; first_bond_dim * second_site_dim * second_bond_dim];
    for first_site in 0..first_site_dim {
        for second_site in 0..second_site_dim {
            let prefix = first_site + first_site_dim * second_site;
            second_core[col_major_index3(
                first_site,
                second_site,
                prefix,
                first_bond_dim,
                second_site_dim,
            )] = 1.0;
        }
    }

    let mut third_core = vec![0.0; second_bond_dim * third_site_dim];
    for third_site in 0..third_site_dim {
        for prefix in 0..second_bond_dim {
            let dense_index = prefix + second_bond_dim * third_site;
            third_core[col_major_index3(prefix, third_site, 0, second_bond_dim, third_site_dim)] =
                data[dense_index];
        }
    }

    let tensor_train = SimpleTensorTrain::new(vec![
        tensor3_from_data(first_core, 1, first_site_dim, first_bond_dim)?,
        tensor3_from_data(
            second_core,
            first_bond_dim,
            second_site_dim,
            second_bond_dim,
        )?,
        tensor3_from_data(third_core, second_bond_dim, third_site_dim, 1)?,
    ])?;
    assert_dense_close(&tensor_train, &shape, data);
    Ok(tensor_train)
}

#[test]
fn dense_oracle_matches_rank_one_constant_multiplication() -> TestResult {
    let left = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.5);
    let right = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], -4.0);
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value * right_value
    });

    let result = elementwise_batched(
        |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                *value = batch.get(0, point)? * batch.get(1, point)?;
            }
            Ok(())
        },
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3, 2]);
    assert_dense_close(&result.tensor_train, &[2, 3, 2], &expected);
    Ok(())
}

#[test]
fn dense_oracle_matches_nonconstant_three_site_batched_operation() -> TestResult {
    let left =
        three_site_tt_from_col_major(&[1.0, -2.0, 3.5, 4.25, -5.5, 6.75, 7.125, -8.25], [2, 2, 2])?;
    let right =
        three_site_tt_from_col_major(&[0.25, 1.5, -2.25, 3.0, 4.5, -5.75, 6.25, -7.5], [2, 2, 2])?;
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value * right_value + 0.5 * left_value - right_value
    });

    let result = elementwise_batched(
        |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                let left_value = batch.get(0, point)?;
                let right_value = batch.get(1, point)?;
                *value = left_value * right_value + 0.5 * left_value - right_value;
            }
            Ok(())
        },
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 2, 2]);
    assert_dense_close(&result.tensor_train, &[2, 2, 2], &expected);
    Ok(())
}

#[test]
fn dense_oracle_matches_nonconstant_addition() -> TestResult {
    let left = two_site_tt_from_col_major(&[1.0, -2.0, 3.0, 4.0, -5.0, 6.0], [2, 3])?;
    let right = two_site_tt_from_col_major(&[0.5, 1.5, -3.0, 2.0, 8.0, -7.0], [2, 3])?;
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value + right_value
    });

    let result = elementwise(
        |values| values[0] + values[1],
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
    assert_dense_close(&result.tensor_train, &[2, 3], &expected);
    Ok(())
}

#[test]
fn dense_oracle_matches_nonlinear_scalar_operation() -> TestResult {
    let left = two_site_tt_from_col_major(&[0.0, 0.25, 0.5, 1.0, -0.75, 1.25], [2, 3])?;
    let right = two_site_tt_from_col_major(&[2.0, -1.0, 0.5, -0.25, 1.5, -2.0], [2, 3])?;
    let expected = dense_oracle2(&left, &right, |left_value, right_value| {
        left_value.sin() + right_value * right_value
    });

    let result = elementwise(
        |values| values[0].sin() + values[1] * values[1],
        &[left, right],
        &exact_options(),
    )?;

    assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
    assert_dense_close(&result.tensor_train, &[2, 3], &expected);
    Ok(())
}
