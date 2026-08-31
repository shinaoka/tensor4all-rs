use super::IncrementalQr;
use crate::{mat_mul, qr_backend, src_error_estimate, Matrix};
use num_complex::Complex64;

fn reconstruction_error(qr: &IncrementalQr<f64>, original: &Matrix<f64>) -> f64 {
    let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
    reconstructed
        .as_col_major_slice()
        .iter()
        .zip(original.as_col_major_slice())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max)
}

fn transpose(matrix: &Matrix<f64>) -> Matrix<f64> {
    let mut result = Matrix::zeros(matrix.ncols(), matrix.nrows());
    for column in 0..matrix.ncols() {
        for row in 0..matrix.nrows() {
            result[[column, row]] = matrix[[row, column]];
        }
    }
    result
}

fn max_abs_difference(left: &Matrix<f64>, right: &Matrix<f64>) -> f64 {
    assert_eq!(left.nrows(), right.nrows());
    assert_eq!(left.ncols(), right.ncols());
    left.as_col_major_slice()
        .iter()
        .zip(right.as_col_major_slice())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0, f64::max)
}

#[test]
fn incremental_qr_matches_backend_qr_projector_and_estimate() {
    let full = Matrix::from_col_major_vec(
        7,
        5,
        vec![
            1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 4.0, -2.0, 1.0, 0.25, 3.0, -1.5, 2.0, 0.75, 0.5, -1.0,
            2.0, 1.5, -0.25, 3.5, -2.5, 2.0, 1.25, -0.5, 4.0, 0.75, -1.0, 2.25, 3.0, -0.75, 1.0,
            2.5, -2.0, 0.5, 1.75,
        ],
    );
    let first = Matrix::from_col_major_vec(7, 2, full.as_col_major_slice()[..14].to_vec());
    let second = Matrix::from_col_major_vec(7, 2, full.as_col_major_slice()[14..28].to_vec());
    let third = Matrix::from_col_major_vec(7, 1, full.as_col_major_slice()[28..].to_vec());

    let mut incremental = IncrementalQr::new(first).unwrap();
    incremental.append(&second).unwrap();
    incremental.append(&third).unwrap();

    let (direct_q, direct_r) = qr_backend(full.clone().into_typed_tensor()).unwrap();
    let direct_q = Matrix::try_from_typed_tensor(direct_q).unwrap();
    let direct_r = Matrix::try_from_typed_tensor(direct_r).unwrap();
    let direct_reconstruction = mat_mul(&direct_q, &direct_r).unwrap();
    let incremental_q = incremental.q();
    let incremental_reconstruction = mat_mul(&incremental_q, &incremental.r()).unwrap();
    assert!(max_abs_difference(&direct_reconstruction, &full) < 1.0e-12);
    assert!(max_abs_difference(&incremental_reconstruction, &full) < 1.0e-12);

    let direct_projector = mat_mul(&direct_q, &transpose(&direct_q)).unwrap();
    let incremental_projector = mat_mul(&incremental_q, &transpose(&incremental_q)).unwrap();
    let projector_error = max_abs_difference(&incremental_projector, &direct_projector);
    assert!(
        projector_error < 1.0e-11,
        "incremental and backend QR projectors differ by {projector_error}"
    );

    let direct_estimate = src_error_estimate(&direct_r).unwrap();
    let incremental_estimate = incremental.error_estimate().unwrap();
    assert!((incremental_estimate.error - direct_estimate.error).abs() < 1.0e-10);
    assert!((incremental_estimate.norm - direct_estimate.norm).abs() < 1.0e-10);
}

#[test]
fn incremental_qr_delegates_factorization_to_backend() {
    let source = include_str!("../incremental_qr.rs");
    assert!(source.contains("qr_backend"));
    for scalar_kernel in [
        "fn householder_factor",
        "fn householder_vector",
        "fn apply_reflector",
        "fn apply_q_adjoint",
        "fn form_q",
    ] {
        assert!(
            !source.contains(scalar_kernel),
            "incremental QR must not define scalar kernel {scalar_kernel}"
        );
    }
}

#[test]
fn incremental_qr_reorthogonalizes_nearly_dependent_blocks() {
    let first = Matrix::from_col_major_vec(
        12,
        3,
        vec![
            1.0, 0.5, -0.25, 2.0, 1.5, -1.0, 0.75, 3.0, -2.0, 0.25, 1.25, -0.5, -1.0, 2.0, 0.5,
            -0.75, 1.25, 2.5, -1.5, 0.25, 3.0, 1.0, -0.5, 0.75, -2.0, 1.5, 0.25, -1.5, 2.0, 0.5,
            -0.25, 1.0, 2.75, -0.75, 1.5, -2.5,
        ],
    );
    let mut qr = IncrementalQr::new(first.clone()).unwrap();
    let basis = qr.q();
    let mut appended = Matrix::zeros(12, 2);
    for row in 0..12 {
        appended[[row, 0]] = basis[[row, 0]] - 0.75 * basis[[row, 1]];
        appended[[row, 1]] = 0.5 * basis[[row, 1]] + 1.25 * basis[[row, 2]];
    }
    appended[[7, 0]] += 1.0e-7;
    appended[[9, 0]] -= 2.0e-8;
    appended[[8, 1]] -= 1.5e-7;
    appended[[10, 1]] += 3.0e-8;

    let mut original_values = first.as_col_major_slice().to_vec();
    original_values.extend_from_slice(appended.as_col_major_slice());
    let original = Matrix::from_col_major_vec(12, 5, original_values);
    qr.append(&appended).unwrap();

    let reconstruction = mat_mul(&qr.q(), &qr.r()).unwrap();
    let reconstruction_error = max_abs_difference(&reconstruction, &original);
    assert!(
        reconstruction_error < 1.0e-12,
        "near-dependent reconstruction error is {reconstruction_error}"
    );
    let q = qr.q();
    let gram = mat_mul(&transpose(&q), &q).unwrap();
    for column in 0..gram.ncols() {
        for row in 0..gram.nrows() {
            let expected = if row == column { 1.0 } else { 0.0 };
            let error = (gram[[row, column]] - expected).abs();
            assert!(
                error < 1.0e-11,
                "near-dependent Q†Q error at ({row}, {column}) is {error}"
            );
        }
    }
}

#[test]
fn incremental_qr_reconstructs_after_appending_columns() {
    let first = Matrix::from_col_major_vec(
        5,
        2,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.5, 3.0, 4.0],
    );
    let appended = Matrix::from_col_major_vec(5, 1, vec![0.5, -1.0, 2.0, 1.5, 3.5]);
    let appended_again = Matrix::from_col_major_vec(5, 1, vec![2.5, 1.0, -0.5, 4.5, -3.0]);
    let mut original_data = first.as_col_major_slice().to_vec();
    original_data.extend_from_slice(appended.as_col_major_slice());
    original_data.extend_from_slice(appended_again.as_col_major_slice());
    let original = Matrix::from_col_major_vec(5, 4, original_data);

    let mut qr = IncrementalQr::new(first.clone()).unwrap();
    let initial_original = Matrix::from_col_major_vec(5, 2, first.as_col_major_slice().to_vec());
    assert!(reconstruction_error(&qr, &initial_original) < 1.0e-12);
    assert_eq!(qr.q().ncols(), 2);
    assert_eq!(qr.r().nrows(), 2);
    qr.append(&appended).unwrap();
    let after_one = Matrix::from_col_major_vec(5, 3, {
        let mut values = first.as_col_major_slice().to_vec();
        values.extend_from_slice(appended.as_col_major_slice());
        values
    });
    assert!(reconstruction_error(&qr, &after_one) < 1.0e-12);
    qr.append(&appended_again).unwrap();

    assert_eq!(qr.q().nrows(), 5);
    assert_eq!(qr.q().ncols(), 4);
    assert_eq!(qr.r().nrows(), 4);
    assert_eq!(qr.r().ncols(), 4);
    let error = reconstruction_error(&qr, &original);
    assert!(error < 1.0e-12, "reconstruction error={error}");

    let q = qr.q();
    for left in 0..q.ncols() {
        for right in 0..q.ncols() {
            let inner = (0..q.nrows())
                .map(|row| q[[row, left]] * q[[row, right]])
                .sum::<f64>();
            let expected = if left == right { 1.0 } else { 0.0 };
            assert!(
                (inner - expected).abs() < 1.0e-12,
                "Q is not orthonormal at ({left}, {right}): {inner}"
            );
        }
    }

    let full = IncrementalQr::new(original).unwrap();
    let incremental_estimate = qr.error_estimate().unwrap();
    let full_estimate = full.error_estimate().unwrap();
    assert!(
        (incremental_estimate.error - full_estimate.error).abs() < 1.0e-10,
        "incremental and full QR error estimates differ: {} vs {}",
        incremental_estimate.error,
        full_estimate.error
    );
    assert!(
        (incremental_estimate.norm - full_estimate.norm).abs() < 1.0e-10,
        "incremental and full QR norm estimates differ: {} vs {}",
        incremental_estimate.norm,
        full_estimate.norm
    );
}

#[test]
fn incremental_qr_updates_the_inverse_adjoint_after_appending_columns() {
    let first = Matrix::from_col_major_vec(
        5,
        2,
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, -2.0, 1.0, 0.5, 3.0, 4.0],
    );
    let appended = Matrix::from_col_major_vec(5, 1, vec![0.5, -1.0, 2.0, 1.5, 3.5]);
    let mut qr = IncrementalQr::new(first).unwrap();
    qr.append(&appended).unwrap();

    let r = qr.r();
    let g = qr
        .inverse_adjoint
        .as_ref()
        .expect("full-rank incremental QR must retain R^{-T}");
    let mut r_transpose = Matrix::zeros(r.ncols(), r.nrows());
    for col in 0..r.ncols() {
        for row in 0..r.nrows() {
            r_transpose[[row, col]] = r[[col, row]];
        }
    }
    let product = mat_mul(&r_transpose, g).unwrap();
    for row in 0..product.nrows() {
        for col in 0..product.ncols() {
            let expected = if row == col { 1.0 } else { 0.0 };
            assert!(
                (product[[row, col]] - expected).abs() < 1.0e-10,
                "R^T G is not identity at ({row}, {col}): {}",
                product[[row, col]]
            );
        }
    }
}

#[test]
fn incremental_qr_preserves_rank_for_dependent_appended_columns() {
    let first = Matrix::from_col_major_vec(
        5,
        2,
        vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    let appended = Matrix::from_col_major_vec(
        5,
        2,
        vec![1.0_f64, 2.0, 0.0, 0.0, 0.0, -3.0, 1.0, 0.0, 0.0, 0.0],
    );
    let mut original_data = first.as_col_major_slice().to_vec();
    original_data.extend_from_slice(appended.as_col_major_slice());
    let original = Matrix::from_col_major_vec(5, 4, original_data);

    let mut qr = IncrementalQr::new(first).unwrap();
    qr.append(&appended).unwrap();

    assert_eq!(qr.q().ncols(), 2);
    assert_eq!(qr.r().nrows(), 2);
    assert_eq!(qr.r().ncols(), 4);
    let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
    assert!(reconstructed
        .as_col_major_slice()
        .iter()
        .zip(original.as_col_major_slice())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
}

#[test]
fn incremental_qr_from_factors_reconstructs_the_supplied_product() {
    let q = Matrix::from_col_major_vec(4, 2, vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let r = Matrix::from_col_major_vec(2, 3, vec![2.0_f64, 0.0, 1.0, 3.0, -1.0, 0.5]);
    let expected = mat_mul(&q, &r).unwrap();
    let state = IncrementalQr::from_factors(q, r).unwrap();
    let actual = mat_mul(&state.q(), &state.r()).unwrap();
    assert!(actual
        .as_col_major_slice()
        .iter()
        .zip(expected.as_col_major_slice())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
}

#[test]
fn incremental_qr_rejects_invalid_shapes() {
    let initial = Matrix::from_col_major_vec(3, 1, vec![1.0, 2.0, 3.0]);
    let mut qr = IncrementalQr::new(initial).unwrap();
    let wrong_rows = Matrix::from_col_major_vec(2, 1, vec![1.0, 2.0]);
    assert!(qr.append(&wrong_rows).is_err());

    let too_many = Matrix::from_col_major_vec(3, 3, vec![1.0; 9]);
    assert!(qr.append(&too_many).is_err());
}

#[test]
fn incremental_qr_new_rejects_empty_matrix() {
    let empty_rows: Matrix<f64> = Matrix::from_col_major_vec(0, 1, vec![]);
    assert!(IncrementalQr::new(empty_rows).is_err());

    let empty_cols: Matrix<f64> = Matrix::from_col_major_vec(3, 0, vec![]);
    assert!(IncrementalQr::new(empty_cols).is_err());
}

#[test]
fn incremental_qr_new_rejects_wide_matrix() {
    let wide = Matrix::from_col_major_vec(2, 3, vec![1.0; 6]);
    assert!(IncrementalQr::new(wide).is_err());
}

#[test]
fn incremental_qr_from_factors_rejects_empty_q() {
    let empty_q: Matrix<f64> = Matrix::from_col_major_vec(0, 0, vec![]);
    let r = Matrix::from_col_major_vec(1, 1, vec![1.0]);
    assert!(IncrementalQr::from_factors(empty_q, r).is_err());
}

#[test]
fn incremental_qr_from_factors_rejects_non_thin_q() {
    // Q must be tall-or-square: nrows >= ncols. 2x3 violates that.
    let wide_q = Matrix::from_col_major_vec(2, 3, vec![1.0; 6]);
    let r = Matrix::from_col_major_vec(3, 3, vec![1.0; 9]);
    assert!(IncrementalQr::from_factors(wide_q, r).is_err());
}

#[test]
fn incremental_qr_from_factors_rejects_incompatible_r() {
    let q = Matrix::from_col_major_vec(3, 2, vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    // R must have r.nrows() == q.ncols() == 2, and r.ncols() >= 2.
    let bad_r = Matrix::from_col_major_vec(1, 2, vec![1.0, 1.0]);
    assert!(IncrementalQr::from_factors(q, bad_r).is_err());
}

#[test]
fn incremental_qr_uses_the_hermitian_projection_for_complex_columns() {
    let first = Matrix::from_col_major_vec(
        4,
        1,
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(-1.0, 0.5),
            Complex64::new(0.25, 2.0),
        ],
    );
    let appended = Matrix::from_col_major_vec(
        4,
        1,
        vec![
            Complex64::new(0.5, -2.0),
            Complex64::new(-1.0, 1.5),
            Complex64::new(2.0, 0.25),
            Complex64::new(3.0, -0.5),
        ],
    );
    let mut original_data = first.as_col_major_slice().to_vec();
    original_data.extend_from_slice(appended.as_col_major_slice());
    let original = Matrix::from_col_major_vec(4, 2, original_data);

    let mut qr = IncrementalQr::new(first).unwrap();
    qr.append(&appended).unwrap();
    let reconstructed = mat_mul(&qr.q(), &qr.r()).unwrap();
    let error = reconstructed
        .as_col_major_slice()
        .iter()
        .zip(original.as_col_major_slice())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0, f64::max);
    assert!(error < 1.0e-12, "complex reconstruction error is {error}");

    let r = qr.r();
    let g = qr
        .inverse_adjoint
        .as_ref()
        .expect("full-rank complex QR must retain R^{-dagger}");
    let mut r_adjoint = Matrix::zeros(r.ncols(), r.nrows());
    for col in 0..r.ncols() {
        for row in 0..r.nrows() {
            r_adjoint[[row, col]] = r[[col, row]].conj();
        }
    }
    let product = mat_mul(&r_adjoint, g).unwrap();
    for row in 0..product.nrows() {
        for col in 0..product.ncols() {
            let expected = if row == col {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            };
            assert!(
                (product[[row, col]] - expected).norm() < 1.0e-10,
                "R^dagger G is not identity at ({row}, {col}): {}",
                product[[row, col]]
            );
        }
    }
}
