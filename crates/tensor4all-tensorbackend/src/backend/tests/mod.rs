use super::*;
use num_complex::{Complex32, Complex64};
use num_traits::Zero;
use tenferro::TensorScalar;

fn row_major_values<T>(tensor: &TypedTensor<T>) -> Vec<T>
where
    T: TensorScalar + Copy,
{
    assert_eq!(tensor.shape().len(), 2, "test helper expects a matrix");
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let values = tensor
        .as_slice()
        .expect("typed test tensor should expose host values");
    let mut out = Vec::with_capacity(values.len());
    for row in 0..rows {
        for col in 0..cols {
            out.push(values[row + col * rows]);
        }
    }
    out
}

fn matmul_row_major<T>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T>
where
    T: Copy + Zero + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let mut out = vec![T::zero(); m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                out[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    out
}

fn scale_columns_complex(
    data: &[Complex64],
    rows: usize,
    cols: usize,
    scales: &[f64],
) -> Vec<Complex64> {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(scales.len(), cols);
    let mut out = vec![Complex64::zero(); data.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[i * cols + j] = data[i * cols + j] * Complex64::new(scales[j], 0.0);
        }
    }
    out
}

#[test]
fn qr_backend_reconstructs_real_matrix() {
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0_f64, 3.0, 2.0, 4.0])
        .expect("valid QR test input");

    let (q, r) = qr_backend(&input).unwrap();
    assert_eq!(q.shape(), &[2, 2]);
    assert_eq!(r.shape(), &[2, 2]);

    let q_values = row_major_values(&q);
    let r_values = row_major_values(&r);
    let reconstructed = matmul_row_major(&q_values, 2, 2, &r_values, 2);
    let input_values = row_major_values(&input);

    for (actual, expected) in reconstructed.iter().zip(input_values.iter()) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "QR reconstruction mismatch: {actual} vs {expected}"
        );
    }
}

#[test]
fn svd_backend_reconstructs_complex_matrix() {
    let input = TypedTensor::from_vec_col_major(
        vec![2, 2],
        vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-3.0, 0.25),
            Complex64::new(2.0, 1.5),
            Complex64::new(4.0, -2.0),
        ],
    )
    .expect("valid SVD test input");

    let decomp = svd_backend(&input).unwrap();
    assert_eq!(decomp.u.shape(), &[2, 2]);
    assert_eq!(decomp.s.shape(), &[2]);
    assert_eq!(decomp.vt.shape(), &[2, 2]);
    let cloned = decomp.clone();
    assert_eq!(cloned.s.shape(), &[2]);

    let u = row_major_values(&decomp.u);
    let s = decomp
        .s
        .as_slice()
        .expect("SVD singular values should expose host values")
        .to_vec();
    let vt = row_major_values(&decomp.vt);
    let us = scale_columns_complex(&u, 2, 2, &s);
    let reconstructed = matmul_row_major(&us, 2, 2, &vt, 2);
    let input_values = row_major_values(&input);

    for (actual, expected) in reconstructed.iter().zip(input_values.iter()) {
        assert!(
            (*actual - *expected).norm() < 1.0e-10,
            "SVD reconstruction mismatch: {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn solve_backend_solves_real_system() {
    let a = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 1.0, 2.0])
        .expect("valid solve lhs test input");
    let b = TypedTensor::from_vec_col_major(vec![2, 1], vec![1.0_f64, 0.0])
        .expect("valid solve rhs test input");

    let x = solve_backend(&a, &b).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    let x_values = x
        .as_slice()
        .expect("solve result should expose host values");
    assert!((x_values[0] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x_values[1] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_solves_real_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f64], vec![0.0]]);

    let x = solve_matrix(&a, &b).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_owned_solves_real_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f64], vec![0.0]]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn solve_matrix_owned_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f32], vec![0.0]]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-6);
}

#[test]
fn solve_matrix_owned_solves_complex64_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex64::new(1.0, 0.0)],
        vec![Complex64::new(0.0, 0.0)],
    ]);

    let x = solve_matrix_owned(a, b).unwrap();

    assert!((x[[0, 0]].re - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((x[[1, 0]].re + 1.0 / 3.0).abs() < 1.0e-12);
    assert!(x[[0, 0]].im.abs() < 1.0e-12);
    assert!(x[[1, 0]].im.abs() < 1.0e-12);
}

#[test]
fn solve_matrix_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 1.0], vec![1.0, 2.0]]);
    let b = crate::from_vec2d(vec![vec![1.0_f32], vec![0.0]]);

    let x = solve_matrix(&a, &b).unwrap();

    assert!((x[[0, 0]] - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] + 1.0 / 3.0).abs() < 1.0e-6);
}

#[test]
fn solve_matrix_promotes_complex32_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0), Complex32::new(1.0, 0.0)],
        vec![Complex32::new(1.0, 0.0), Complex32::new(2.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex32::new(1.0, 0.0)],
        vec![Complex32::new(0.0, 0.0)],
    ]);

    let x = solve_matrix(&a, &b).unwrap();

    assert!((x[[0, 0]].re - 2.0 / 3.0).abs() < 1.0e-6);
    assert!((x[[1, 0]].re + 1.0 / 3.0).abs() < 1.0e-6);
    assert!(x[[0, 0]].im.abs() < 1.0e-6);
    assert!(x[[1, 0]].im.abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_solves_left_lower_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64], vec![7.0]]);

    let x = triangular_solve_matrix(&a, &b, true, true, false, false).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_owned_solves_left_lower_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64], vec![7.0]]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert_eq!(x.nrows(), 2);
    assert_eq!(x.ncols(), 1);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_solves_right_upper_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f64, 1.0], vec![0.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f64, 7.0]]);

    let x = triangular_solve_matrix(&a, &b, false, false, false, false).unwrap();

    assert_eq!(x.nrows(), 1);
    assert_eq!(x.ncols(), 2);
    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-12);
    assert!((x[[0, 1]] - 2.0).abs() < 1.0e-12);
}

#[test]
fn triangular_solve_matrix_owned_promotes_f32_system() {
    let a = crate::from_vec2d(vec![vec![2.0_f32, 0.0], vec![1.0, 3.0]]);
    let b = crate::from_vec2d(vec![vec![2.0_f32], vec![7.0]]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert!((x[[0, 0]] - 1.0).abs() < 1.0e-6);
    assert!((x[[1, 0]] - 2.0).abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_promotes_complex32_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0)],
        vec![Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex32::new(2.0, 0.0)],
        vec![Complex32::new(7.0, 0.0)],
    ]);

    let x = triangular_solve_matrix(&a, &b, true, true, false, false).unwrap();

    assert!((x[[0, 0]].re - 1.0).abs() < 1.0e-6);
    assert!((x[[1, 0]].re - 2.0).abs() < 1.0e-6);
    assert!(x[[0, 0]].im.abs() < 1.0e-6);
    assert!(x[[1, 0]].im.abs() < 1.0e-6);
}

#[test]
fn triangular_solve_matrix_owned_solves_complex64_system() {
    let a = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
    ]);
    let b = crate::from_vec2d(vec![
        vec![Complex64::new(2.0, 0.0)],
        vec![Complex64::new(7.0, 0.0)],
    ]);

    let x = triangular_solve_matrix_owned(a, b, true, true, false, false).unwrap();

    assert!((x[[0, 0]].re - 1.0).abs() < 1.0e-12);
    assert!((x[[1, 0]].re - 2.0).abs() < 1.0e-12);
    assert!(x[[0, 0]].im.abs() < 1.0e-12);
    assert!(x[[1, 0]].im.abs() < 1.0e-12);
}

#[test]
fn triangular_solve_backend_solves_typed_tensor_system() {
    let a = TypedTensor::from_vec_col_major(vec![2, 2], vec![2.0_f64, 1.0, 0.0, 3.0])
        .expect("valid triangular solve lhs test input");
    let b = TypedTensor::from_vec_col_major(vec![2, 1], vec![2.0_f64, 7.0])
        .expect("valid triangular solve rhs test input");

    let x = triangular_solve_backend(&a, &b, true, true, false, false).unwrap();

    assert_eq!(x.shape(), &[2, 1]);
    let x_values = x
        .as_slice()
        .expect("triangular solve result should expose host values");
    assert!((x_values[0] - 1.0).abs() < 1.0e-12);
    assert!((x_values[1] - 2.0).abs() < 1.0e-12);
}

#[test]
fn try_into_typed_result_reports_dtype_mismatch() {
    let tensor = Complex64::into_tensor(vec![1], vec![Complex64::new(1.0, 0.0)])
        .expect("valid dtype-mismatch test tensor");

    let err = try_into_typed_result::<f64>("test_op", tensor).unwrap_err();

    assert!(err.to_string().contains("test_op: dtype mismatch"));
}

#[test]
fn full_piv_lu_backend_returns_square_factors() {
    let input = TypedTensor::from_vec_col_major(vec![2, 2], vec![0.0_f64, 2.0, 1.0, 3.0])
        .expect("valid LU test input");

    let decomp = full_piv_lu_backend(&input).unwrap();

    assert_eq!(decomp.p.shape(), &[2, 2]);
    assert_eq!(decomp.l.shape(), &[2, 2]);
    assert_eq!(decomp.u.shape(), &[2, 2]);
    assert_eq!(decomp.q.shape(), &[2, 2]);
    let cloned = decomp.clone();
    assert_eq!(cloned.u.shape(), &[2, 2]);
}

#[test]
fn full_piv_lu_matrix_returns_square_factors() {
    let input = crate::from_vec2d(vec![vec![0.0_f64, 1.0], vec![2.0, 3.0]]);

    let decomp = full_piv_lu_matrix(&input).unwrap();

    assert_eq!(decomp.p.nrows(), 2);
    assert_eq!(decomp.p.ncols(), 2);
    assert_eq!(decomp.l.nrows(), 2);
    assert_eq!(decomp.l.ncols(), 2);
    assert_eq!(decomp.u.nrows(), 2);
    assert_eq!(decomp.u.ncols(), 2);
    assert_eq!(decomp.q.nrows(), 2);
    assert_eq!(decomp.q.ncols(), 2);
}
