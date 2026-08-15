use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tensor4all_core::AnyScalar;

#[test]
fn test_is_complex() {
    assert!(!AnyScalar::new_real(1.0).is_complex());
    assert!(AnyScalar::new_complex(1.0, 0.0).is_complex());
}

#[test]
fn test_from_f32_and_complex32_scalar_paths() {
    let real = AnyScalar::from_value(1.25_f32);
    assert_eq!(real.as_f64(), Some(1.25));
    assert_eq!(real.real(), 1.25);
    assert_eq!(real.imag(), 0.0);
    assert_eq!(real.abs(), 1.25);
    assert!(!real.is_zero());
    assert_eq!(Complex64::from(real), Complex64::new(1.25, 0.0));

    let zero = AnyScalar::from_value(0.0_f32);
    assert!(zero.is_zero());

    let complex = AnyScalar::from_value(Complex32::new(3.0, -4.0));
    assert!(complex.is_complex());
    assert_eq!(complex.real(), 3.0);
    assert_eq!(complex.imag(), -4.0);
    assert_eq!(complex.abs(), 5.0);
    assert_eq!(complex.as_c64(), Some(Complex64::new(3.0, -4.0)));
    assert_eq!(Complex64::from(complex), Complex64::new(3.0, -4.0));

    let complex_zero = AnyScalar::from_value(Complex32::new(0.0, 0.0));
    assert!(complex_zero.is_zero());
}

#[test]
fn test_real() {
    assert_eq!(AnyScalar::new_real(3.5).real(), 3.5);
    assert_eq!(AnyScalar::new_complex(2.0, 3.0).real(), 2.0);
}

#[test]
fn test_abs() {
    assert_eq!(AnyScalar::new_real(-5.0).abs(), 5.0);
    assert_eq!(AnyScalar::new_complex(3.0, 4.0).abs(), 5.0);
}

#[test]
fn test_sqrt() {
    // Positive real
    let result = AnyScalar::new_real(4.0).sqrt();
    assert!(!result.is_complex());
    assert!((result.real() - 2.0).abs() < 1e-10);

    // Negative real -> complex
    let result = AnyScalar::new_real(-4.0).sqrt();
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 2.0).abs() < 1e-10);

    // Complex
    let result = AnyScalar::new_complex(-1.0, 0.0).sqrt();
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 1.0).abs() < 1e-10);
}

#[test]
fn test_powf() {
    // Positive real
    let result = AnyScalar::new_real(4.0).powf(0.5);
    assert!(!result.is_complex());
    assert!((result.real() - 2.0).abs() < 1e-10);

    // Negative real -> complex
    let result = AnyScalar::new_real(-4.0).powf(0.5);
    assert!(result.is_complex());

    // Complex
    let result = AnyScalar::new_complex(1.0, 1.0).powf(2.0);
    assert!(result.is_complex());
}

#[test]
fn test_powi() {
    // Real
    let result = AnyScalar::new_real(2.0).powi(3);
    assert!(!result.is_complex());
    assert!((result.real() - 8.0).abs() < 1e-10);

    let result = AnyScalar::new_real(2.0).powi(-2);
    assert!(!result.is_complex());
    assert!((result.real() - 0.25).abs() < 1e-10);

    // Complex
    let result = AnyScalar::new_complex(1.0, 1.0).powi(2);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert!((z.re - 0.0).abs() < 1e-10);
    assert!((z.im - 2.0).abs() < 1e-10);
}

#[test]
fn test_add() {
    // F64 + F64
    let result = AnyScalar::new_real(1.0) + AnyScalar::new_real(2.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    // F64 + C64
    let result = AnyScalar::new_real(1.0) + AnyScalar::new_complex(2.0, 3.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 3.0);

    // C64 + F64
    let result = AnyScalar::new_complex(1.0, 2.0) + AnyScalar::new_real(3.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 4.0);
    assert_eq!(z.im, 2.0);

    // C64 + C64
    let result = AnyScalar::new_complex(1.0, 2.0) + AnyScalar::new_complex(3.0, 4.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 4.0);
    assert_eq!(z.im, 6.0);
}

#[test]
fn test_sub() {
    let result = AnyScalar::new_real(5.0) - AnyScalar::new_real(2.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(5.0, 6.0) - AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 6.0);
}

#[test]
fn test_mul() {
    let result = AnyScalar::new_real(3.0) * AnyScalar::new_real(4.0);
    assert!(!result.is_complex());
    assert!((result.real() - 12.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(1.0, 2.0) * AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 2.0);
    assert_eq!(z.im, 4.0);
}

#[test]
fn test_div() {
    let result = AnyScalar::new_real(12.0) / AnyScalar::new_real(4.0);
    assert!(!result.is_complex());
    assert!((result.real() - 3.0).abs() < 1e-10);

    let result = AnyScalar::new_complex(6.0, 8.0) / AnyScalar::new_real(2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, 3.0);
    assert_eq!(z.im, 4.0);
}

#[test]
fn test_neg() {
    let result = -AnyScalar::new_real(5.0);
    assert!(!result.is_complex());
    assert!((result.real() - (-5.0)).abs() < 1e-10);

    let result = -AnyScalar::new_complex(1.0, 2.0);
    assert!(result.is_complex());
    let z: Complex64 = result.into();
    assert_eq!(z.re, -1.0);
    assert_eq!(z.im, -2.0);
}

#[test]
fn test_from_f64() {
    let s: AnyScalar = 3.5.into();
    assert!(!s.is_complex());
    assert_eq!(s.real(), 3.5);
}

#[test]
fn test_new_real() {
    let s = AnyScalar::new_real(3.5);
    assert!(!s.is_complex());
    assert_eq!(s.real(), 3.5);
}

#[test]
fn test_new_complex() {
    let s = AnyScalar::new_complex(1.0, 2.0);
    assert!(s.is_complex());
    assert_eq!(s.real(), 1.0);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, 2.0);
}

#[test]
fn test_from_complex64() {
    let z = Complex64::new(1.0, 2.0);
    let s: AnyScalar = z.into();
    assert!(s.is_complex());
    let c: Complex64 = s.into();
    assert_eq!(c.re, 1.0);
    assert_eq!(c.im, 2.0);
}

#[test]
fn test_try_from_anyscalar_to_f64() {
    // Success case
    let s = AnyScalar::new_real(3.5);
    let x: Result<f64, _> = s.try_into();
    assert_eq!(x.unwrap(), 3.5);

    // Error case (complex)
    let s = AnyScalar::new_complex(1.0, 2.0);
    let x: Result<f64, _> = s.try_into();
    assert!(x.is_err());
}

#[test]
fn test_from_anyscalar_to_complex64() {
    // Real -> Complex
    let s = AnyScalar::new_real(3.5);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 3.5);
    assert_eq!(z.im, 0.0);

    // Complex -> Complex
    let s = AnyScalar::new_complex(1.0, 2.0);
    let z: Complex64 = s.into();
    assert_eq!(z.re, 1.0);
    assert_eq!(z.im, 2.0);
}

#[test]
fn test_default() {
    let s = AnyScalar::default();
    assert!(s.is_zero());
    assert!(!s.is_complex());
}

#[test]
fn test_zero() {
    let z = AnyScalar::zero();
    assert!(z.is_zero());
    assert!(!AnyScalar::new_real(0.0).is_complex());
    assert!(AnyScalar::new_real(0.0).is_zero());
    assert!(!AnyScalar::new_real(1.0).is_zero());
    assert!(AnyScalar::new_complex(0.0, 0.0).is_zero());
    assert!(!AnyScalar::new_complex(1.0, 0.0).is_zero());
}

#[test]
fn test_one() {
    let one = AnyScalar::one();
    assert!(!one.is_complex());
    assert!((one.real() - 1.0).abs() < 1e-10);
}

#[test]
fn test_partial_ord() {
    // Real numbers can be compared
    assert!(AnyScalar::new_real(1.0) < AnyScalar::new_real(2.0));
    assert!(AnyScalar::new_real(2.0) > AnyScalar::new_real(1.0));
    assert!(AnyScalar::new_real(1.0) <= AnyScalar::new_real(1.0));

    // Complex numbers cannot be compared
    let c1 = AnyScalar::new_complex(1.0, 0.0);
    let c2 = AnyScalar::new_complex(2.0, 0.0);
    assert!(c1.partial_cmp(&c2).is_none());

    // Mixed types cannot be compared
    assert!(AnyScalar::new_real(1.0)
        .partial_cmp(&AnyScalar::new_complex(1.0, 0.0))
        .is_none());
}

#[test]
fn test_display() {
    #[allow(clippy::approx_constant)]
    let s1 = AnyScalar::new_real(3.14);
    assert_eq!(format!("{}", s1), "3.14");

    let s2 = AnyScalar::new_complex(1.0, 2.0);
    let display_str = format!("{}", s2);
    assert!(display_str.contains("1") && display_str.contains("2"));
}

#[test]
fn test_is_real() {
    assert!(AnyScalar::new_real(3.0).is_real());
    assert!(!AnyScalar::new_complex(3.0, 1.0).is_real());
}

#[test]
fn test_real_imag_part_preserve_scalar_kind() {
    let x = AnyScalar::new_complex(2.5, -1.25);
    let xr = x.real_part();
    let xi = x.imag_part();
    assert!(xr.is_real());
    assert!(xi.is_real());
    assert!((xr.real() - 2.5).abs() < 1e-12);
    assert!((xi.real() - (-1.25)).abs() < 1e-12);

    let y = AnyScalar::new_real(-3.0);
    let yr = y.real_part();
    let yi = y.imag_part();
    assert!(yr.is_real());
    assert!(yi.is_real());
    assert!((yr.real() - (-3.0)).abs() < 1e-12);
    assert!(yi.is_zero());
}

#[test]
fn test_compose_complex_from_real_parts() {
    let re = AnyScalar::new_real(1.5);
    let im = AnyScalar::new_real(-2.0);
    let z = AnyScalar::compose_complex(re, im).unwrap();
    assert!(z.is_complex());
    let c: Complex64 = z.into();
    assert!((c.re - 1.5).abs() < 1e-12);
    assert!((c.im + 2.0).abs() < 1e-12);
}

#[test]
fn test_compose_complex_rejects_non_real_inputs() {
    let re = AnyScalar::new_complex(1.0, 0.0);
    let im = AnyScalar::new_real(2.0);
    assert!(AnyScalar::compose_complex(re, im).is_err());
}

#[test]
fn test_scalar_lhs_rhs_mixed_ops() {
    let x = AnyScalar::new_complex(1.0, -2.0);
    let y = 2.0_f64 * x.clone();
    let z: Complex64 = y.into();
    assert!((z.re - 2.0).abs() < 1e-12);
    assert!((z.im + 4.0).abs() < 1e-12);

    let x = AnyScalar::new_real(2.0);
    let y = Complex64::new(4.0, -2.0) / x;
    let z: Complex64 = y.into();
    assert!((z.re - 2.0).abs() < 1e-12);
    assert!((z.im + 1.0).abs() < 1e-12);
}

#[test]
fn test_enable_grad_and_borrow_arithmetic_backward() {
    let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(3.0).enable_grad().unwrap();

    let product = &x * &y;
    let loss = &product + &x;

    assert!(loss.tracks_grad());
    loss.backward().unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_y = y.grad().unwrap().unwrap();
    assert!((grad_x.real() - 4.0).abs() < 1e-12);
    assert!((grad_y.real() - 2.0).abs() < 1e-12);
}

#[test]
fn test_clone_shares_tracked_leaf_gradient_slot() {
    let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let alias = x.clone();

    let loss = &x * &alias;
    loss.backward().unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_alias = alias.grad().unwrap().unwrap();
    assert!((grad_x.real() - 4.0).abs() < 1e-12);
    assert!((grad_alias.real() - 4.0).abs() < 1e-12);
}

#[test]
fn test_owned_arithmetic_backward_with_moved_operands() {
    let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(3.0).enable_grad().unwrap();

    let x_alias = x.clone();
    let y_alias = y.clone();
    let x_grad = x_alias.clone();
    let y_grad = y_alias.clone();

    let product = x * y;
    let loss = product + x_alias;
    loss.backward().unwrap();

    let grad_x = x_grad.grad().unwrap().unwrap();
    let grad_y = y_grad.grad().unwrap().unwrap();
    assert!((grad_x.real() - 4.0).abs() < 1e-12);
    assert!((grad_y.real() - 2.0).abs() < 1e-12);
}

#[test]
fn test_clear_grad_resets_anyscalar_accumulation() {
    let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(3.0).enable_grad().unwrap();

    let loss = &x * &y;
    loss.backward().unwrap();

    let loss = &x * &y;
    loss.backward().unwrap();

    let grad_x = x.grad().unwrap().unwrap();
    let grad_y = y.grad().unwrap().unwrap();
    assert!((grad_x.real() - 6.0).abs() < 1e-12);
    assert!((grad_y.real() - 4.0).abs() < 1e-12);

    x.clear_grad().unwrap();
    y.clear_grad().unwrap();
    assert!(x.grad().unwrap().is_none());
    assert!(y.grad().unwrap().is_none());
}

#[test]
fn test_detach_breaks_reverse_graph_for_anyscalar() {
    let x = AnyScalar::new_real(3.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let detached = x.detach().unwrap();

    assert!(!detached.tracks_grad());
    assert!(x.tracks_grad());
    assert!(y.tracks_grad());

    let loss = &detached * &y;
    assert!(loss.tracks_grad());
    loss.backward().unwrap();

    assert!(x.grad().unwrap().is_none());
    let grad_y = y.grad().unwrap().unwrap();
    assert!((grad_y.real() - 3.0).abs() < 1e-12);
}

#[test]
fn test_primal_matches_detach_semantics_for_anyscalar() {
    let x = AnyScalar::new_real(3.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let primal = x.primal().unwrap();

    assert!(!primal.tracks_grad());
    assert_eq!(primal.real(), 3.0);

    let loss = &primal * &y;
    assert!(loss.tracks_grad());
    loss.backward().unwrap();

    assert!(x.grad().unwrap().is_none());
    let grad_y = y.grad().unwrap().unwrap();
    assert!((grad_y.real() - 3.0).abs() < 1e-12);
}

#[test]
fn test_function_api_takes_anyscalar_ref() {
    fn quadratic_plus_one(x: &AnyScalar) -> AnyScalar {
        let square = x * x;
        &square + &AnyScalar::one()
    }

    let x = AnyScalar::new_real(3.0).enable_grad().unwrap();
    let loss = quadratic_plus_one(&x);
    loss.backward().unwrap();

    let grad = x.grad().unwrap().unwrap();
    assert!((grad.real() - 6.0).abs() < 1e-12);
}

#[test]
fn tracked_negative_real_analytic_ops_match_principal_complex_values_and_backward() {
    let x = AnyScalar::new_real(-4.0).enable_grad().unwrap();

    let sqrt = x.sqrt();
    let sqrt_value: Complex64 = sqrt.clone().into();
    assert!((sqrt_value.re).abs() < 1e-12);
    assert!((sqrt_value.im - 2.0).abs() < 1e-12);
    assert!(sqrt.tracks_grad());
    sqrt.backward().unwrap();
    let sqrt_grad: Complex64 = x.grad().unwrap().unwrap().into();
    // The real input's cotangent is the real projection of the complex VJP.
    assert!(sqrt_grad.norm() < 1.0e-12);
    x.clear_grad().unwrap();

    sqrt.imag_part().backward().unwrap();
    let sqrt_imag_grad = x.grad().unwrap().unwrap();
    assert!((sqrt_imag_grad.real() + 0.25).abs() < 1.0e-12);
    x.clear_grad().unwrap();

    let pow = x.powf(0.5);
    let pow_value: Complex64 = pow.clone().into();
    assert!((pow_value.re).abs() < 1e-12);
    assert!((pow_value.im - 2.0).abs() < 1e-12);
    assert!(pow.tracks_grad());
    pow.backward().unwrap();
    let pow_grad: Complex64 = x.grad().unwrap().unwrap().into();
    assert!(pow_grad.norm() < 1.0e-12);
    x.clear_grad().unwrap();

    pow.imag_part().backward().unwrap();
    let pow_imag_grad = x.grad().unwrap().unwrap();
    assert!((pow_imag_grad.real() + 0.25).abs() < 1.0e-12);
}

#[test]
fn tracked_real_and_imag_parts_keep_graph_and_values() {
    let x = AnyScalar::new_complex(2.0, 3.0).enable_grad().unwrap();

    let real = x.real_part();
    assert!(real.is_real());
    assert!(real.tracks_grad());
    assert!((real.real() - 2.0).abs() < 1e-12);
    real.backward().unwrap();
    let real_grad: Complex64 = x.grad().unwrap().unwrap().into();
    assert!((real_grad.re - 1.0).abs() < 1e-12);
    assert!(real_grad.im.abs() < 1e-12);
    x.clear_grad().unwrap();

    let imag = x.imag_part();
    assert!(imag.is_real());
    assert!(imag.tracks_grad());
    assert!((imag.real() - 3.0).abs() < 1e-12);
    imag.backward().unwrap();
    let imag_grad: Complex64 = x.grad().unwrap().unwrap().into();
    assert!(imag_grad.re.abs() < 1e-12);
    assert!((imag_grad.im - 1.0).abs() < 1e-12);
}

#[test]
fn tracked_powi_zero_is_graph_connected_with_zero_gradient() {
    for value in [3.0, 0.0] {
        let x = AnyScalar::new_real(value).enable_grad().unwrap();
        let result = x.powi(0);

        assert_eq!(result.real(), 1.0);
        assert!(result.tracks_grad());
        result.backward().unwrap();
        let gradient = x.grad().unwrap().unwrap();
        assert!(gradient.real().abs() < 1e-12);
    }
}

#[test]
fn tracked_scalar_subtraction_preserves_autodiff() {
    let x = AnyScalar::new_real(5.0).enable_grad().unwrap();
    let y = AnyScalar::new_real(2.0).enable_grad().unwrap();
    let result = &x - &y;

    assert_eq!(result.real(), 3.0);
    result.backward().unwrap();

    assert!((x.grad().unwrap().unwrap().real() - 1.0).abs() < 1.0e-12);
    assert!((y.grad().unwrap().unwrap().real() + 1.0).abs() < 1.0e-12);
}
