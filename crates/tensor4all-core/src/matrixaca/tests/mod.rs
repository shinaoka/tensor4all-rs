use super::*;
use tensor4all_tensorbackend::from_vec2d;

#[test]
fn test_matrixaca_new() {
    let aca = MatrixACA::<f64>::new(5, 5);
    assert_eq!(aca.nrows(), 5);
    assert_eq!(aca.ncols(), 5);
    assert_eq!(aca.rank(), 0);
    assert!(aca.is_empty());
}

#[test]
fn test_matrixaca_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let aca = MatrixACA::from_matrix_with_pivot(&m, (1, 1)).unwrap();
    assert_eq!(aca.nrows(), 3);
    assert_eq!(aca.ncols(), 3);
    assert_eq!(aca.rank(), 1);

    // Check that evaluation at pivot is correct
    let val = aca.evaluate(1, 1);
    assert!((val - 5.0).abs() < 1e-10);
}

#[test]
fn test_matrixaca_add_pivot() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    assert!(aca.add_pivot(&m, (1, 1)).is_ok());
    assert_eq!(aca.rank(), 2);
}

#[test]
fn test_matrixaca_zero_pivot_returns_error() {
    let m = from_vec2d(vec![vec![0.0, 1.0], vec![1.0, 1.0]]);
    // Pivot at (0,0) which is 0.0 should fail
    let result = MatrixACA::from_matrix_with_pivot(&m, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_matrixaca_near_zero_pivot_returns_error() {
    let m = from_vec2d(vec![vec![1e-200, 1.0], vec![1.0, 1.0]]);
    let result = MatrixACA::from_matrix_with_pivot(&m, (0, 0));
    assert!(result.is_err());
}

#[test]
fn test_matrixaca_add_pivot_row_zero_diagonal() {
    // Matrix where second pivot would cause division by zero
    let m = from_vec2d(vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ]);
    let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();
    // Adding pivot at (1,1) where value is 0 should fail during add_pivot_row
    let result = aca.add_pivot(&m, (1, 1));
    assert!(result.is_err());
}

#[test]
fn test_matrixaca_accessors_and_empty_approximation_paths() {
    let empty = MatrixACA::<f64>::new(2, 3);

    assert_eq!(empty.evaluate(1, 2), 0.0);

    let empty_block = empty.submatrix(&[0, 1], &[0, 2]);
    assert_eq!(empty_block.nrows(), 2);
    assert_eq!(empty_block.ncols(), 2);
    for i in 0..empty_block.nrows() {
        for j in 0..empty_block.ncols() {
            assert_eq!(empty_block[[i, j]], 0.0);
        }
    }

    let m = from_vec2d(vec![vec![2.0_f64, 1.0], vec![3.0, 4.0]]);
    let aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();

    assert_eq!(aca.u().nrows(), 2);
    assert_eq!(aca.u().ncols(), 1);
    assert_eq!(aca.v().nrows(), 1);
    assert_eq!(aca.v().ncols(), 2);
    assert_eq!(aca.alpha(), &[0.5]);
}

#[test]
fn test_matrixaca_add_pivot_row_and_col_bounds_errors() {
    let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    let mut aca = MatrixACA::<f64>::new(2, 2);

    assert!(matches!(
        aca.add_pivot_col(&m, 2),
        Err(MatrixCIError::IndexOutOfBounds {
            row: 0,
            col: 2,
            nrows: 2,
            ncols: 2
        })
    ));
    assert!(matches!(
        aca.add_pivot_row(&m, 2),
        Err(MatrixCIError::IndexOutOfBounds {
            row: 2,
            col: 0,
            nrows: 2,
            ncols: 2
        })
    ));
}

#[test]
fn test_matrixaca_add_best_pivot_covers_empty_incremental_and_full_rank_paths() {
    let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 5.0]]);
    let mut aca = MatrixACA::<f64>::new(2, 2);

    assert_eq!(aca.add_best_pivot(&m).unwrap(), (1, 1));
    assert_eq!(aca.rank(), 1);

    let second = aca.add_best_pivot(&m).unwrap();
    assert_eq!(second, (0, 0));
    assert_eq!(aca.rank(), 2);

    assert!(matches!(
        aca.add_best_pivot(&m),
        Err(MatrixCIError::FullRank)
    ));

    let one_row = from_vec2d(vec![vec![1.0_f64, 2.0]]);
    let mut row_full = MatrixACA::from_matrix_with_pivot(&one_row, (0, 0)).unwrap();

    assert!(matches!(
        row_full.add_best_pivot(&one_row),
        Err(MatrixCIError::FullRank)
    ));
}

#[test]
fn test_matrixaca_set_rows_and_cols_permute_and_fill_new_entries() {
    let m = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0]]);
    let mut aca = MatrixACA::from_matrix_with_pivot(&m, (0, 0)).unwrap();

    let new_pivot_rows = from_vec2d(vec![vec![10.0_f64, 20.0, 30.0]]);
    aca.set_cols(&new_pivot_rows, &[1, 0]);

    assert_eq!(aca.col_indices(), &[1]);
    assert_eq!(aca.v().ncols(), 3);
    assert_eq!(aca.v()[[0, 2]], 30.0);

    let new_pivot_cols = from_vec2d(vec![vec![10.0_f64], vec![20.0], vec![30.0]]);
    aca.set_rows(&new_pivot_cols, &[1, 0]);

    assert_eq!(aca.row_indices(), &[1]);
    assert_eq!(aca.u().nrows(), 3);
    assert_eq!(aca.u()[[2, 0]], 30.0);
}
