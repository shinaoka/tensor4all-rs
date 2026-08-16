use super::*;

fn test_scalar_generic<T: Scalar>() {
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);

    // Basic arithmetic
    let sum = one + one;
    assert!((sum.abs_sq() - 4.0).abs() < 1e-10);

    // Conjugate (for real, conj is identity)
    let conj_two = two.conj();
    assert!((conj_two.abs_sq() - 4.0).abs() < 1e-10);

    // NaN check
    assert!(!one.is_nan());
}

#[test]
fn test_scalar_f64() {
    test_scalar_generic::<f64>();
}

#[test]
fn test_scalar_f32() {
    test_scalar_generic::<f32>();
}

#[test]
fn test_scalar_c64() {
    test_scalar_generic::<Complex64>();

    // Complex-specific test
    let z = Complex64::new(3.0, 4.0);
    assert!((z.abs_sq() - 25.0).abs() < 1e-10);
    assert!((z.abs_val() - 5.0).abs() < 1e-10);

    let z_conj = z.conj();
    assert!((z_conj.re - 3.0).abs() < 1e-10);
    assert!((z_conj.im - (-4.0)).abs() < 1e-10);
}

#[test]
fn test_scalar_c32() {
    test_scalar_generic::<Complex32>();
}

#[test]
fn test_abs_and_abs_val_f64() {
    let x: f64 = -3.5;
    // abs returns the absolute value
    assert!((Scalar::abs(x) - 3.5).abs() < 1e-10);
    // abs_val returns the same as abs for real types
    assert!((x.abs_val() - 3.5).abs() < 1e-10);
}

#[test]
fn test_abs_and_abs_val_f32() {
    let x: f32 = -2.5;
    // abs returns the absolute value
    assert!((Scalar::abs(x) - 2.5).abs() < 1e-6);
    // abs_val returns f64
    assert!((x.abs_val() - 2.5).abs() < 1e-6);
}

#[test]
fn test_abs_and_abs_val_c64() {
    let z = Complex64::new(3.0, 4.0);
    // abs returns Complex64 with re=|z|, im=0
    let z_abs = Scalar::abs(z);
    assert!((z_abs.re - 5.0).abs() < 1e-10);
    assert!(z_abs.im.abs() < 1e-10);
    // abs_val returns f64
    assert!((z.abs_val() - 5.0).abs() < 1e-10);
}

#[test]
fn test_abs_and_abs_val_c32() {
    let z = Complex32::new(3.0, 4.0);
    // abs returns Complex32 with re=|z|, im=0
    let z_abs = Scalar::abs(z);
    assert!((z_abs.re - 5.0).abs() < 1e-5);
    assert!(z_abs.im.abs() < 1e-5);
    // abs_val returns f64
    assert!((z.abs_val() - 5.0).abs() < 1e-5);
}
