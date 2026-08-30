use super::*;
use num_complex::Complex64;
use tenferro::TypedTensor;

#[test]
fn fallible_matrix_constructors_reject_overflow_and_length_mismatch() {
    assert!(matches!(
        Matrix::<f64>::try_from_col_major_vec(usize::MAX, 2, Vec::new()),
        Err(MatrixShapeError::ShapeOverflow { .. })
    ));
    assert!(matches!(
        Matrix::<f64>::try_from_col_major_vec(2, 2, vec![0.0; 3]),
        Err(MatrixShapeError::DataLengthMismatch { .. })
    ));
    assert!(matches!(
        Matrix::<f64>::try_zeros(usize::MAX, 2),
        Err(MatrixShapeError::ShapeOverflow { .. })
    ));
}

#[test]
fn hermitian_eigendecomposition_maps_eager_context_error_with_source_chain() {
    let matrix = Matrix::from_col_major_vec(1, 1, vec![2.0_f64]);
    let error = crate::context::with_forced_eager_context_failure(|| {
        hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap_err()
    });

    let HermitianEigenError::Backend { source } = error else {
        panic!("expected backend error from forced eager context failure");
    };
    let context_error = source
        .downcast_ref::<crate::context::EagerContextError>()
        .expect("backend source should retain EagerContextError");
    let registration_source = std::error::Error::source(context_error).unwrap();
    assert_eq!(
        registration_source.to_string(),
        "forced default eager context registration failure"
    );
    assert!(registration_source.source().is_none());
}

#[test]
fn hermitian_backend_error_preserves_source_chain() {
    let source = std::io::Error::other("forced eigensolver failure");
    let error = HermitianEigenError::Backend {
        source: Box::new(source),
    };

    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(
        std::error::Error::source(&error).unwrap().to_string(),
        "forced eigensolver failure"
    );
}

fn real_eigen_residual_norm(matrix: &Matrix<f64>, eigenvalue: f64, vector: &[f64]) -> f64 {
    let mut max_abs = 0.0_f64;
    for row in 0..matrix.nrows() {
        let mut av = 0.0;
        for col in 0..matrix.ncols() {
            av += matrix[[row, col]] * vector[col];
        }
        max_abs = max_abs.max((av - eigenvalue * vector[row]).abs());
    }
    max_abs
}

fn complex_eigen_residual_norm(
    matrix: &Matrix<Complex64>,
    eigenvalue: f64,
    vector: &[Complex64],
) -> f64 {
    let mut max_abs = 0.0_f64;
    for row in 0..matrix.nrows() {
        let mut av = Complex64::new(0.0, 0.0);
        for col in 0..matrix.ncols() {
            av += matrix[[row, col]] * vector[col];
        }
        max_abs = max_abs.max((av - vector[row] * eigenvalue).norm());
    }
    max_abs
}

#[test]
fn projected_hermitian_lowest_eigenpair_works_for_one_by_one() {
    let matrix = Matrix::from_col_major_vec(1, 1, vec![3.5_f64]);

    let pair = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap();

    assert!((pair.eigenvalue - 3.5).abs() < 1.0e-12);
    assert_eq!(pair.eigenvector.len(), 1);
    assert!((pair.eigenvector[0].abs() - 1.0).abs() < 1.0e-12);
}

#[test]
fn projected_hermitian_lowest_eigenpair_works_for_real_symmetric_matrix() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0]);

    let pair = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap();

    assert!((pair.eigenvalue - 1.0).abs() < 1.0e-12);
    assert!(real_eigen_residual_norm(&matrix, pair.eigenvalue, &pair.eigenvector) < 1.0e-10);
}

#[test]
fn hermitian_eigendecomposition_returns_all_real_symmetric_pairs() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![2.0_f64, 1.0, 1.0, 2.0]);

    let decomp = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap();

    assert_eq!(decomp.eigenvalues.len(), 2);
    assert_eq!(decomp.eigenvectors.nrows(), 2);
    assert_eq!(decomp.eigenvectors.ncols(), 2);
    assert!((decomp.eigenvalues[0] - 1.0).abs() < 1.0e-12);
    assert!((decomp.eigenvalues[1] - 3.0).abs() < 1.0e-12);
    for col in 0..2 {
        let vector = [decomp.eigenvectors[[0, col]], decomp.eigenvectors[[1, col]]];
        assert!(real_eigen_residual_norm(&matrix, decomp.eigenvalues[col], &vector) < 1.0e-10);
    }
}

#[test]
fn projected_hermitian_lowest_eigenpair_works_for_complex_hermitian_matrix() {
    let i = Complex64::new(0.0, 1.0);
    let matrix = Matrix::from_col_major_vec(
        2,
        2,
        vec![Complex64::new(0.0, 0.0), i, -i, Complex64::new(0.0, 0.0)],
    );

    let pair = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap();

    assert!((pair.eigenvalue + 1.0).abs() < 1.0e-12);
    assert!(complex_eigen_residual_norm(&matrix, pair.eigenvalue, &pair.eigenvector) < 1.0e-10);
}

#[test]
fn hermitian_eigendecomposition_returns_all_complex_pairs() {
    let i = Complex64::new(0.0, 1.0);
    let matrix = Matrix::from_col_major_vec(
        2,
        2,
        vec![Complex64::new(0.0, 0.0), i, -i, Complex64::new(0.0, 0.0)],
    );

    let decomp = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap();

    assert!((decomp.eigenvalues[0] + 1.0).abs() < 1.0e-12);
    assert!((decomp.eigenvalues[1] - 1.0).abs() < 1.0e-12);
    for col in 0..2 {
        let vector = [decomp.eigenvectors[[0, col]], decomp.eigenvectors[[1, col]]];
        assert!(complex_eigen_residual_norm(&matrix, decomp.eigenvalues[col], &vector) < 1.0e-10);
    }
}

#[test]
fn hermitian_exponential_first_column_matches_diagonal_action() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 3.0]);

    let coeffs =
        hermitian_exponential_first_column(&matrix, Complex64::new(0.0, -0.5), 1.0e-12).unwrap();

    assert_eq!(coeffs.len(), 2);
    assert!((coeffs[0] - Complex64::new(0.5_f64.cos(), -0.5_f64.sin())).norm() < 1.0e-12);
    assert!(coeffs[1].norm() < 1.0e-12);
}

#[test]
fn projected_hermitian_lowest_eigenpair_accepts_degenerate_smallest_eigenvalues() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 0.0, 0.0, 1.0]);

    let pair = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap();

    assert!((pair.eigenvalue - 1.0).abs() < 1.0e-12);
    assert!(real_eigen_residual_norm(&matrix, pair.eigenvalue, &pair.eigenvector) < 1.0e-10);
}

#[test]
fn projected_hermitian_lowest_eigenpair_rejects_non_hermitian_diagonal() {
    let matrix = Matrix::from_col_major_vec(1, 1, vec![Complex64::new(1.0, 1.0e-3)]);

    let err = lowest_hermitian_eigenpair(&matrix, 1.0e-12).unwrap_err();

    assert!(matches!(err, HermitianEigenError::NonHermitian { .. }));
    assert!(err.to_string().contains("not Hermitian"));
}

#[test]
fn hermitian_eigendecomposition_accepts_relative_roundoff_and_symmetrizes() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0e8_f64, 2.0e8, 2.0e8 + 1.0e-5, 3.0e8]);

    let decomp = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap();

    let symmetrized =
        Matrix::from_col_major_vec(2, 2, vec![1.0e8, 2.0e8 + 0.5e-5, 2.0e8 + 0.5e-5, 3.0e8]);
    for col in 0..2 {
        let vector = [decomp.eigenvectors[[0, col]], decomp.eigenvectors[[1, col]]];
        let residual = real_eigen_residual_norm(&symmetrized, decomp.eigenvalues[col], &vector);
        assert!(
            residual < 1.0e-6,
            "residual {residual:.3e} exceeds tolerance"
        );
    }
}

#[test]
fn hermitian_eigendecomposition_rejects_relative_asymmetry_above_tolerance() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0e8_f64, 2.0e8, 2.0e8 + 1.0e-2, 3.0e8]);

    let err = hermitian_eigendecomposition(&matrix, 1.0e-12).unwrap_err();

    assert!(matches!(err, HermitianEigenError::NonHermitian { .. }));
}

#[test]
fn test_matrix_basic() {
    let mut m = Matrix::<f64>::zeros(3, 3);
    m[[0, 0]] = 1.0;
    m[[1, 1]] = 2.0;
    m[[2, 2]] = 3.0;

    assert_eq!(m[[0, 0]], 1.0);
    assert_eq!(m[[1, 1]], 2.0);
    assert_eq!(m[[2, 2]], 3.0);
}

#[test]
fn test_matrix_column_major_storage() {
    let m = Matrix::from_col_major_vec(2, 3, vec![1, 4, 2, 5, 3, 6]);

    assert_eq!(m[[0, 0]], 1);
    assert_eq!(m[[0, 1]], 2);
    assert_eq!(m[[0, 2]], 3);
    assert_eq!(m[[1, 0]], 4);
    assert_eq!(m[[1, 1]], 5);
    assert_eq!(m[[1, 2]], 6);
    assert_eq!(m.as_col_major_slice(), &[1, 4, 2, 5, 3, 6]);
}

#[test]
fn matrix_constructors_reject_shape_product_overflow() {
    let panic =
        std::panic::catch_unwind(|| Matrix::from_col_major_vec(usize::MAX, 2, Vec::<u8>::new()))
            .unwrap_err();
    assert_eq!(
        *panic
            .downcast::<String>()
            .expect("shape overflow should panic with a String message"),
        format!(
            "matrix shape product overflow: {} rows * 2 columns",
            usize::MAX
        )
    );

    let panic = std::panic::catch_unwind(|| Matrix::from_elem(usize::MAX, 2, 0_u8)).unwrap_err();
    assert_eq!(
        *panic
            .downcast::<String>()
            .expect("shape overflow should panic with a String message"),
        format!(
            "matrix shape product overflow: {} rows * 2 columns",
            usize::MAX
        )
    );

    let panic = std::panic::catch_unwind(|| Matrix::<u8>::zeros(usize::MAX, 2)).unwrap_err();
    assert_eq!(
        *panic
            .downcast::<String>()
            .expect("shape overflow should panic with a String message"),
        format!(
            "matrix shape product overflow: {} rows * 2 columns",
            usize::MAX
        )
    );

    let zero = Matrix::<u8>::zeros(0, usize::MAX);
    assert_eq!(zero.nrows(), 0);
    assert_eq!(zero.ncols(), usize::MAX);
    assert!(zero.as_col_major_slice().is_empty());
}

#[test]
fn matrix_indexing_rejects_each_axis_before_linearization() {
    let mut matrix = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    assert!(std::panic::catch_unwind(|| matrix[[2, 0]]).is_err());
    assert!(std::panic::catch_unwind(|| matrix[[0, 2]]).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        matrix[[2, 0]] = 99;
    }))
    .is_err());
    assert_eq!(matrix.as_col_major_slice(), &[1, 3, 2, 4]);
    matrix[[1, 1]] = 8;
    assert_eq!(matrix[[1, 1]], 8);
}

#[test]
fn matrix_multiplication_rejects_output_shape_overflow_before_backend() {
    // A zero-column matrix may legitimately have an arbitrary row count (the
    // checked shape product is 0). Multiplying it by a 0 x 2 matrix would
    // produce an overflowing usize::MAX x 2 output; the multiplication must
    // reject the output element count before any backend call.
    let left = Matrix::<f64>::from_col_major_vec(usize::MAX, 0, Vec::new());
    let right = Matrix::<f64>::from_col_major_vec(0, 2, Vec::new());
    let error = mat_mul(&left, &right).unwrap_err().to_string();
    assert!(error.contains("overflows"), "got: {error}");

    let error = mat_mul_owned(left, right).unwrap_err().to_string();
    assert!(error.contains("overflows"), "got: {error}");
}

#[test]
fn matrix_complex64_construction_preserves_column_major_indexing() {
    let i = Complex64::new(0.0, 1.0);
    let matrix = Matrix::from_col_major_vec(
        2,
        2,
        vec![Complex64::new(1.0, 0.0), i, Complex64::new(2.0, 0.0), -i],
    );

    assert_eq!(matrix[[0, 0]], Complex64::new(1.0, 0.0));
    assert_eq!(matrix[[1, 0]], i);
    assert_eq!(matrix[[0, 1]], Complex64::new(2.0, 0.0));
    assert_eq!(matrix[[1, 1]], -i);
    assert_eq!(
        matrix.as_col_major_slice(),
        &[Complex64::new(1.0, 0.0), i, Complex64::new(2.0, 0.0), -i]
    );
}

#[test]
fn matrix_into_col_major_vec_consumes_storage() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let data = m.into_col_major_vec();

    assert_eq!(data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_to_typed_tensor_preserves_column_major_layout() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let tensor = m.to_typed_tensor();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(
        tensor
            .as_slice()
            .expect("matrix test tensor should expose host values"),
        &[1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_into_typed_tensor_consumes_column_major_layout() {
    let m = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let tensor = m.into_typed_tensor();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(
        tensor
            .as_slice()
            .expect("matrix test tensor should expose host values"),
        &[1.0, 3.0, 2.0, 4.0]
    );
}

#[test]
fn matrix_from_typed_tensor_consumes_column_major_layout() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 3.0, 2.0, 4.0])
        .expect("valid matrix conversion test tensor");

    let m = Matrix::try_from_typed_tensor(tensor).unwrap();

    assert_eq!(m.nrows(), 2);
    assert_eq!(m.ncols(), 2);
    assert_eq!(m.as_col_major_slice(), &[1.0, 3.0, 2.0, 4.0]);
    assert_eq!(m[[0, 1]], 2.0);
}

#[test]
fn matrix_from_typed_tensor_rejects_non_matrix_rank() {
    let tensor = TypedTensor::from_vec_col_major(vec![2, 1, 1], vec![1.0, 2.0])
        .expect("valid non-matrix test tensor");

    let err = Matrix::try_from_typed_tensor(tensor).unwrap_err();

    assert!(err.to_string().contains("rank-2 tensor"));
}

#[test]
fn try_from_vec2d_rejects_longer_rows() {
    let err = try_from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0, 5.0]]).unwrap_err();

    assert!(matches!(
        err,
        MatrixShapeError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 3,
        }
    ));
    assert_eq!(err.to_string(), "row 1 has length 3, expected 2");
}

#[test]
fn try_from_vec2d_rejects_shorter_rows() {
    let err = try_from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0]]).unwrap_err();

    assert!(matches!(
        err,
        MatrixShapeError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 1,
        }
    ));
    assert_eq!(err.to_string(), "row 1 has length 1, expected 2");
}

#[test]
#[should_panic(expected = "row 1 has length 3, expected 2")]
fn from_vec2d_panics_with_shape_error_for_ragged_rows() {
    let _ = from_vec2d(vec![vec![1.0_f64, 2.0], vec![3.0, 4.0, 5.0]]);
}

#[test]
fn matrix_public_precondition_assertions_reject_invalid_axes() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);

    assert!(std::panic::catch_unwind(|| submatrix(&matrix, &[2], &[0])).is_err());
    assert!(std::panic::catch_unwind(|| submatrix(&matrix, &[0], &[2])).is_err());
    assert!(std::panic::catch_unwind(|| submatrix_argmax(&matrix, 0..0, 0..2)).is_err());
    assert!(std::panic::catch_unwind(|| submatrix_argmax(&matrix, 0..2, 0..0)).is_err());
    assert!(std::panic::catch_unwind(|| submatrix_argmax(&matrix, 0..3, 0..2)).is_err());
    assert!(std::panic::catch_unwind(|| submatrix_argmax(&matrix, 0..2, 0..3)).is_err());
}

#[test]
fn matrix_swaps_are_fallible_and_never_mutate_on_invalid_indices() {
    let matrix = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 3.0, 2.0, 4.0]);

    let mut rows = matrix.clone();
    swap_rows(&mut rows, 0, 1).unwrap();
    assert_eq!(rows.as_col_major_slice(), &[3.0, 1.0, 4.0, 2.0]);
    swap_rows(&mut rows, 1, 1).unwrap();
    assert_eq!(rows.as_col_major_slice(), &[3.0, 1.0, 4.0, 2.0]);
    for (a, b, rejected) in [(2, 0, 2), (0, 2, 2), (2, 2, 2)] {
        let mut invalid = matrix.clone();
        assert_eq!(
            swap_rows(&mut invalid, a, b),
            Err(MatrixShapeError::RowIndexOutOfBounds {
                index: rejected,
                nrows: 2,
            })
        );
        assert_eq!(invalid.as_col_major_slice(), matrix.as_col_major_slice());
    }

    let mut cols = matrix.clone();
    swap_cols(&mut cols, 0, 1).unwrap();
    assert_eq!(cols.as_col_major_slice(), &[2.0, 4.0, 1.0, 3.0]);
    swap_cols(&mut cols, 1, 1).unwrap();
    assert_eq!(cols.as_col_major_slice(), &[2.0, 4.0, 1.0, 3.0]);
    for (a, b, rejected) in [(2, 0, 2), (0, 2, 2), (2, 2, 2)] {
        let mut invalid = matrix.clone();
        assert_eq!(
            swap_cols(&mut invalid, a, b),
            Err(MatrixShapeError::ColumnIndexOutOfBounds {
                index: rejected,
                ncols: 2,
            })
        );
        assert_eq!(invalid.as_col_major_slice(), matrix.as_col_major_slice());
    }
}

#[test]
fn test_matrix_transpose() {
    let m = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let mt = transpose(&m);

    assert_eq!(mt.nrows(), 3);
    assert_eq!(mt.ncols(), 2);
    assert_eq!(mt[[0, 0]], 1.0);
    assert_eq!(mt[[0, 1]], 4.0);
    assert_eq!(mt[[2, 0]], 3.0);
}

#[test]
fn test_submatrix_argmax() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    let (r, c, _) = submatrix_argmax(&m, 0..3, 0..3);
    assert_eq!((r, c), (2, 2));
}

#[test]
fn test_mat_mul() {
    let a = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = mat_mul(&a, &b).unwrap();

    assert_eq!(c[[0, 0]], 19.0);
    assert_eq!(c[[0, 1]], 22.0);
    assert_eq!(c[[1, 0]], 43.0);
    assert_eq!(c[[1, 1]], 50.0);
}

#[test]
fn mat_mul_owned_matches_borrowed_multiplication() {
    let a = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = from_vec2d(vec![vec![5.0, 6.0], vec![7.0, 8.0]]);

    let c = mat_mul_owned(a, b).unwrap();

    assert_eq!(c.as_col_major_slice(), &[19.0, 43.0, 22.0, 50.0]);
}

#[test]
fn test_mat_mul_rectangular_preserves_column_major_layout() {
    let a = from_vec2d(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    let b = from_vec2d(vec![vec![7.0, 8.0], vec![9.0, 10.0], vec![11.0, 12.0]]);
    let c = mat_mul(&a, &b).unwrap();

    assert_eq!(c.nrows(), 2);
    assert_eq!(c.ncols(), 2);
    assert_eq!(c[[0, 0]], 58.0);
    assert_eq!(c[[0, 1]], 64.0);
    assert_eq!(c[[1, 0]], 139.0);
    assert_eq!(c[[1, 1]], 154.0);
    assert_eq!(c.as_col_major_slice(), &[58.0, 139.0, 64.0, 154.0]);
}

#[test]
fn batched_mat_mul_same_shape_preserves_column_major_batches() {
    let a0 = vec![1.0, 3.0, 2.0, 4.0];
    let a1 = vec![2.0, 0.0, 0.0, 3.0];
    let b0 = vec![5.0, 7.0, 6.0, 8.0];
    let b1 = vec![1.0, 4.0, 2.0, 5.0];
    let mut a = a0;
    a.extend(a1);
    let mut b = b0;
    b.extend(b1);

    let out = batched_mat_mul_same_shape(2, 2, 2, 2, &a, &b).unwrap();

    assert_eq!(out.len(), 8);
    assert_eq!(&out[0..4], &[19.0, 43.0, 22.0, 50.0]);
    assert_eq!(&out[4..8], &[2.0, 12.0, 4.0, 15.0]);
}

#[test]
fn batched_mat_mul_same_shape_owned_matches_borrowed() {
    let a = vec![1.0, 3.0, 2.0, 4.0, 2.0, 0.0, 0.0, 3.0];
    let b = vec![5.0, 7.0, 6.0, 8.0, 1.0, 4.0, 2.0, 5.0];

    let borrowed = batched_mat_mul_same_shape(2, 2, 2, 2, &a, &b).unwrap();
    let owned = batched_mat_mul_same_shape_owned(2, 2, 2, 2, a, b).unwrap();

    assert_eq!(owned, borrowed);
}

#[test]
fn mat_mul_reports_dimension_mismatch() {
    let a = Matrix::from_col_major_vec(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    let b = Matrix::from_col_major_vec(2, 2, vec![1.0, 3.0, 2.0, 4.0]);

    let err = mat_mul(&a, &b).unwrap_err();

    assert!(err.to_string().contains("matrix dimensions"));
}

#[test]
fn append_columns_reuses_existing_data_and_grows_ncols() {
    let mut left = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);
    let right = Matrix::from_col_major_vec(2, 1, vec![5.0_f64, 6.0]);
    left.append_columns(&right).unwrap();
    assert_eq!(left.nrows(), 2);
    assert_eq!(left.ncols(), 3);
    assert_eq!(left.as_col_major_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn append_columns_matches_building_a_fresh_concatenated_matrix() {
    let left = Matrix::from_col_major_vec(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let right = Matrix::from_col_major_vec(3, 1, vec![7.0_f64, 8.0, 9.0]);

    let mut grown = left.clone();
    grown.append_columns(&right).unwrap();

    let mut expected_data = left.as_col_major_slice().to_vec();
    expected_data.extend_from_slice(right.as_col_major_slice());
    let expected = Matrix::try_from_col_major_vec(3, 3, expected_data).unwrap();

    assert_eq!(grown.nrows(), expected.nrows());
    assert_eq!(grown.ncols(), expected.ncols());
    assert_eq!(grown.as_col_major_slice(), expected.as_col_major_slice());
}

#[test]
fn append_columns_reuses_existing_data_and_grows_ncols_complex() {
    let mut left = Matrix::from_col_major_vec(
        2,
        2,
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 4.0),
        ],
    );
    let right = Matrix::from_col_major_vec(
        2,
        1,
        vec![Complex64::new(5.0, -1.0), Complex64::new(6.0, 0.5)],
    );
    left.append_columns(&right).unwrap();
    assert_eq!(left.nrows(), 2);
    assert_eq!(left.ncols(), 3);
    assert_eq!(
        left.as_col_major_slice(),
        &[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 4.0),
            Complex64::new(5.0, -1.0),
            Complex64::new(6.0, 0.5),
        ]
    );
}

#[test]
fn append_columns_reuses_spare_capacity_without_reallocating() {
    let mut left = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);
    // Reserve enough spare capacity up front so the upcoming append cannot
    // possibly need to grow the underlying `Vec`.
    left.data.reserve(8);
    let capacity_before = left.data.capacity();

    let right = Matrix::from_col_major_vec(2, 1, vec![5.0_f64, 6.0]);
    left.append_columns(&right).unwrap();

    assert_eq!(
        left.data.capacity(),
        capacity_before,
        "append_columns should reuse existing spare capacity instead of reallocating"
    );
}
