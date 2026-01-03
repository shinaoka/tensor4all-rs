use tensor4all_tensor::AnyScalar;
use num_complex::Complex64;
use num_traits::{Zero, One};

#[test]
fn test_is_complex() {
    assert!(!AnyScalar::F64(1.0).is_complex());
    assert!(AnyScalar::C64(Complex64::new(1.0, 0.0)).is_complex());
}

#[test]
fn test_real() {
    assert_eq!(AnyScalar::F64(3.5).real(), 3.5);
    assert_eq!(AnyScalar::C64(Complex64::new(2.0, 3.0)).real(), 2.0);
}

#[test]
fn test_abs() {
    assert_eq!(AnyScalar::F64(-5.0).abs(), 5.0);
    assert_eq!(AnyScalar::C64(Complex64::new(3.0, 4.0)).abs(), 5.0);
}

#[test]
fn test_sqrt() {
    // Positive real
    let result = AnyScalar::F64(4.0).sqrt();
    match result {
        AnyScalar::F64(x) => assert!((x - 2.0).abs() < 1e-10),
        _ => panic!("Expected F64"),
    }

    // Negative real -> complex
    let result = AnyScalar::F64(-4.0).sqrt();
    match result {
        AnyScalar::C64(z) => {
            assert!((z.re - 0.0).abs() < 1e-10);
            assert!((z.im - 2.0).abs() < 1e-10);
        }
        _ => panic!("Expected C64"),
    }

    // Complex
    let result = AnyScalar::C64(Complex64::new(-1.0, 0.0)).sqrt();
    match result {
        AnyScalar::C64(z) => {
            assert!((z.re - 0.0).abs() < 1e-10);
            assert!((z.im - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_powf() {
    // Positive real
    let result = AnyScalar::F64(4.0).powf(0.5);
    match result {
        AnyScalar::F64(x) => assert!((x - 2.0).abs() < 1e-10),
        _ => panic!("Expected F64"),
    }

    // Negative real -> complex
    let result = AnyScalar::F64(-4.0).powf(0.5);
    match result {
        AnyScalar::C64(_) => {} // Should be complex
        _ => panic!("Expected C64"),
    }

    // Complex
    let result = AnyScalar::C64(Complex64::new(1.0, 1.0)).powf(2.0);
    match result {
        AnyScalar::C64(_) => {} // Should be complex
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_powi() {
    // Real
    assert_eq!(AnyScalar::F64(2.0).powi(3), AnyScalar::F64(8.0));
    assert_eq!(AnyScalar::F64(2.0).powi(-2), AnyScalar::F64(0.25));

    // Complex
    let result = AnyScalar::C64(Complex64::new(1.0, 1.0)).powi(2);
    match result {
        AnyScalar::C64(z) => {
            assert!((z.re - 0.0).abs() < 1e-10);
            assert!((z.im - 2.0).abs() < 1e-10);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_add() {
    // F64 + F64
    assert_eq!(AnyScalar::F64(1.0) + AnyScalar::F64(2.0), AnyScalar::F64(3.0));

    // F64 + C64
    let result = AnyScalar::F64(1.0) + AnyScalar::C64(Complex64::new(2.0, 3.0));
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 3.0);
            assert_eq!(z.im, 3.0);
        }
        _ => panic!("Expected C64"),
    }

    // C64 + F64
    let result = AnyScalar::C64(Complex64::new(1.0, 2.0)) + AnyScalar::F64(3.0);
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 4.0);
            assert_eq!(z.im, 2.0);
        }
        _ => panic!("Expected C64"),
    }

    // C64 + C64
    let result = AnyScalar::C64(Complex64::new(1.0, 2.0)) + AnyScalar::C64(Complex64::new(3.0, 4.0));
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 4.0);
            assert_eq!(z.im, 6.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_sub() {
    assert_eq!(AnyScalar::F64(5.0) - AnyScalar::F64(2.0), AnyScalar::F64(3.0));

    let result = AnyScalar::C64(Complex64::new(5.0, 6.0)) - AnyScalar::F64(2.0);
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 3.0);
            assert_eq!(z.im, 6.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_mul() {
    assert_eq!(AnyScalar::F64(3.0) * AnyScalar::F64(4.0), AnyScalar::F64(12.0));

    let result = AnyScalar::C64(Complex64::new(1.0, 2.0)) * AnyScalar::F64(2.0);
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 2.0);
            assert_eq!(z.im, 4.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_div() {
    assert_eq!(AnyScalar::F64(12.0) / AnyScalar::F64(4.0), AnyScalar::F64(3.0));

    let result = AnyScalar::C64(Complex64::new(6.0, 8.0)) / AnyScalar::F64(2.0);
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 3.0);
            assert_eq!(z.im, 4.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_neg() {
    assert_eq!(-AnyScalar::F64(5.0), AnyScalar::F64(-5.0));

    let result = -AnyScalar::C64(Complex64::new(1.0, 2.0));
    match result {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, -1.0);
            assert_eq!(z.im, -2.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_from_f64() {
    let s: AnyScalar = 3.5.into();
    assert_eq!(s, AnyScalar::F64(3.5));
}

#[test]
fn test_new_real() {
    let s = AnyScalar::new_real(3.5);
    assert_eq!(s, AnyScalar::F64(3.5));
    assert_eq!(s.real(), 3.5);
}

#[test]
fn test_new_complex() {
    let s = AnyScalar::new_complex(1.0, 2.0);
    match s {
        AnyScalar::C64(z) => {
            assert_eq!(z.re, 1.0);
            assert_eq!(z.im, 2.0);
        }
        _ => panic!("Expected C64"),
    }
    assert_eq!(s.real(), 1.0);
    assert!(s.is_complex());
}

#[test]
fn test_from_complex64() {
    let z = Complex64::new(1.0, 2.0);
    let s: AnyScalar = z.into();
    match s {
        AnyScalar::C64(c) => {
            assert_eq!(c.re, 1.0);
            assert_eq!(c.im, 2.0);
        }
        _ => panic!("Expected C64"),
    }
}

#[test]
fn test_try_from_anyscalar_to_f64() {
    // Success case
    let s = AnyScalar::F64(3.5);
    let x: Result<f64, _> = s.try_into();
    assert_eq!(x.unwrap(), 3.5);

    // Error case (complex)
    let s = AnyScalar::C64(Complex64::new(1.0, 2.0));
    let x: Result<f64, _> = s.try_into();
    assert!(x.is_err());
}

#[test]
fn test_from_anyscalar_to_complex64() {
    // Real -> Complex
    let s = AnyScalar::F64(3.5);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 3.5);
    assert_eq!(z.im, 0.0);

    // Complex -> Complex
    let s = AnyScalar::C64(Complex64::new(1.0, 2.0));
    let z: Complex64 = s.into();
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, 2.0);
}

#[test]
fn test_default() {
    assert_eq!(AnyScalar::default(), AnyScalar::F64(0.0));
}

#[test]
fn test_zero() {
    assert_eq!(AnyScalar::zero(), AnyScalar::F64(0.0));
    assert!(AnyScalar::F64(0.0).is_zero());
    assert!(!AnyScalar::F64(1.0).is_zero());
    assert!(AnyScalar::C64(Complex64::new(0.0, 0.0)).is_zero());
    assert!(!AnyScalar::C64(Complex64::new(1.0, 0.0)).is_zero());
}

#[test]
fn test_one() {
    assert_eq!(AnyScalar::one(), AnyScalar::F64(1.0));
}

#[test]
fn test_partial_ord() {
    // Real numbers can be compared
    assert!(AnyScalar::F64(1.0) < AnyScalar::F64(2.0));
    assert!(AnyScalar::F64(2.0) > AnyScalar::F64(1.0));
    assert!(AnyScalar::F64(1.0) <= AnyScalar::F64(1.0));

    // Complex numbers cannot be compared
    let c1 = AnyScalar::C64(Complex64::new(1.0, 0.0));
    let c2 = AnyScalar::C64(Complex64::new(2.0, 0.0));
    assert!(c1.partial_cmp(&c2).is_none());

    // Mixed types cannot be compared
    assert!(AnyScalar::F64(1.0).partial_cmp(&AnyScalar::C64(Complex64::new(1.0, 0.0))).is_none());
}

#[test]
fn test_display() {
    let s1 = AnyScalar::F64(3.14);
    assert_eq!(format!("{}", s1), "3.14");

    let s2 = AnyScalar::C64(Complex64::new(1.0, 2.0));
    let display_str = format!("{}", s2);
    assert!(display_str.contains("1") && display_str.contains("2"));
}

