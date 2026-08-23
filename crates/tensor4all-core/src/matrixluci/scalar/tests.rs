use super::*;
use crate::matrix_luci::MatrixLuciFactors;
use crate::Scalar;
use tensor4all_tensorbackend::Matrix;

fn identity2<T: Scalar>() -> Matrix<T> {
    Matrix::from_col_major_vec(
        2,
        2,
        vec![
            <T as Scalar>::from_f64(1.0),
            <T as Scalar>::from_f64(0.0),
            <T as Scalar>::from_f64(0.0),
            <T as Scalar>::from_f64(1.0),
        ],
    )
}

fn test_scalar_generic<T: Scalar>() {
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);

    let sum = one + one;
    assert!((sum.abs_sq() - 4.0).abs() < 1e-10);

    let conj_two = two.conj();
    assert!((conj_two.abs_sq() - 4.0).abs() < 1e-10);

    assert!(!one.is_nan());
    assert!(T::epsilon() > 0.0);
}

fn options() -> crate::matrixlu::RrLUOptions {
    crate::matrixlu::RrLUOptions {
        max_bond_dim: 2,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    }
}

fn assert_identity2_factor_shapes<T>(factors: &MatrixLuciFactors<T>) {
    assert_eq!(factors.rank, 2);
    assert_eq!(factors.row_indices.len(), factors.rank);
    assert_eq!(factors.col_indices.len(), factors.rank);
    assert!(factors.pivot_errors.len() >= factors.rank);
    assert_eq!(factors.left.nrows(), 2);
    assert_eq!(factors.left.ncols(), factors.rank);
    assert_eq!(factors.right.nrows(), factors.rank);
    assert_eq!(factors.right.ncols(), 2);
}

fn test_matrix_luci_factor_dispatch_supported<T>()
where
    T: MatrixLuciScalar,
{
    let matrix = identity2::<T>();
    let dense =
        <T as MatrixLuciScalar>::matrix_luci_factors_from_matrix(&matrix, options()).unwrap();
    assert_identity2_factor_shapes(&dense);

    let lazy = <T as MatrixLuciScalar>::matrix_luci_factors_from_blocks(
        matrix.nrows(),
        matrix.ncols(),
        |rows, cols, out| {
            for (j, &col) in cols.iter().enumerate() {
                for (i, &row) in rows.iter().enumerate() {
                    out[i + rows.len() * j] = matrix[[row, col]];
                }
            }
        },
        options(),
    )
    .unwrap();
    assert_identity2_factor_shapes(&lazy);
}

#[test]
fn test_scalar_f64() {
    test_scalar_generic::<f64>();

    let x: f64 = -3.5;
    assert!((Scalar::abs(x) - 3.5).abs() < 1e-10);
    assert!((x.abs_val() - 3.5).abs() < 1e-10);
    assert!(f64::from_f64(f64::NAN).is_nan());
}

#[test]
fn test_matrix_luci_factor_dispatch_f64() {
    test_matrix_luci_factor_dispatch_supported::<f64>();
}

#[test]
fn test_scalar_f32() {
    test_scalar_generic::<f32>();

    let x: f32 = -2.5;
    assert!((Scalar::abs(x) - 2.5).abs() < 1e-6);
    assert!((x.abs_val() - 2.5).abs() < 1e-6);
    assert!(f32::from_f64(f64::NAN).is_nan());
}

#[test]
fn test_matrix_luci_factor_dispatch_f32() {
    test_matrix_luci_factor_dispatch_supported::<f32>();
}

#[test]
fn test_scalar_c64() {
    test_scalar_generic::<Complex64>();

    let z = Complex64::new(3.0, 4.0);
    assert!((z.abs_sq() - 25.0).abs() < 1e-10);
    assert!((z.abs_val() - 5.0).abs() < 1e-10);

    let z_abs = Scalar::abs(z);
    assert!((z_abs.re - 5.0).abs() < 1e-10);
    assert!(z_abs.im.abs() < 1e-10);

    let z_conj = z.conj();
    assert!((z_conj.re - 3.0).abs() < 1e-10);
    assert!((z_conj.im + 4.0).abs() < 1e-10);

    assert!(Complex64::from_f64(f64::NAN).is_nan());
}

#[test]
fn test_matrix_luci_factor_dispatch_c64() {
    test_matrix_luci_factor_dispatch_supported::<Complex64>();
}

#[test]
fn test_scalar_c32() {
    test_scalar_generic::<Complex32>();

    let z = Complex32::new(3.0, 4.0);
    assert!((z.abs_sq() - 25.0).abs() < 1e-5);
    assert!((z.abs_val() - 5.0).abs() < 1e-5);

    let z_abs = Scalar::abs(z);
    assert!((z_abs.re - 5.0).abs() < 1e-5);
    assert!(z_abs.im.abs() < 1e-5);

    let z_conj = z.conj();
    assert!((z_conj.re - 3.0).abs() < 1e-5);
    assert!((z_conj.im + 4.0).abs() < 1e-5);

    assert!(Complex32::from_f64(f64::NAN).is_nan());
}

#[test]
fn test_matrix_luci_factor_dispatch_c32() {
    test_matrix_luci_factor_dispatch_supported::<Complex32>();
}
