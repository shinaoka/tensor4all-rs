use crate::matrixluci::{
    CandidateMatrixSource, CrossFactors, DenseLuKernel, DenseMatrixSource, PivotKernel,
    PivotKernelOptions,
};
use approx::assert_abs_diff_eq;
use tensor4all_tensorbackend::Matrix;

fn test_matrix_data() -> Vec<f64> {
    vec![
        1.0, 5.0, 2.0, 0.5, //
        2.0, 6.0, 1.0, 0.25, //
        3.0, 7.0, 4.0, 0.75, //
        4.0, 8.0, 3.0, 9.0, //
    ]
}

fn assert_dense_eq(lhs: &Matrix<f64>, rhs: &Matrix<f64>) {
    assert_eq!(lhs.nrows(), rhs.nrows());
    assert_eq!(lhs.ncols(), rhs.ncols());
    for j in 0..lhs.ncols() {
        for i in 0..lhs.nrows() {
            assert_abs_diff_eq!(lhs[[i, j]], rhs[[i, j]], epsilon = 1e-12);
        }
    }
}

#[test]
fn reconstruct_cross_factors_matches_selected_submatrices() {
    let data = test_matrix_data();
    let src = DenseMatrixSource::from_column_major(&data, 4, 4);
    let selection = DenseLuKernel
        .factorize(&src, &PivotKernelOptions::default())
        .unwrap();

    let factors = CrossFactors::from_source(&src, &selection).unwrap();

    assert_eq!(factors.pivot.nrows(), selection.rank);
    assert_eq!(factors.pivot.ncols(), selection.rank);
    assert_eq!(factors.pivot_cols.nrows(), 4);
    assert_eq!(factors.pivot_cols.ncols(), selection.rank);
    assert_eq!(factors.pivot_rows.nrows(), selection.rank);
    assert_eq!(factors.pivot_rows.ncols(), 4);

    let expected_pivot = CrossFactors::gather(&src, &selection.row_indices, &selection.col_indices);
    let all_rows: Vec<usize> = (0..src.nrows()).collect();
    let all_cols: Vec<usize> = (0..src.ncols()).collect();
    let expected_cols = CrossFactors::gather(&src, &all_rows, &selection.col_indices);
    let expected_rows = CrossFactors::gather(&src, &selection.row_indices, &all_cols);

    assert_dense_eq(&factors.pivot, &expected_pivot);
    assert_dense_eq(&factors.pivot_cols, &expected_cols);
    assert_dense_eq(&factors.pivot_rows, &expected_rows);
}

#[test]
fn reconstruct_cross_factors_can_form_inverse_scaled_factors() {
    let data = test_matrix_data();
    let src = DenseMatrixSource::from_column_major(&data, 4, 4);
    let selection = DenseLuKernel
        .factorize(&src, &PivotKernelOptions::default())
        .unwrap();

    let factors = CrossFactors::from_source(&src, &selection).unwrap();
    let left = factors.cols_solve_pivot().unwrap();
    let right = factors.solve_pivot_rows().unwrap();

    assert_eq!(left.nrows(), 4);
    assert_eq!(left.ncols(), selection.rank);
    assert_eq!(right.nrows(), selection.rank);
    assert_eq!(right.ncols(), 4);
}
