use super::*;
use tenferro::DType;

fn assert_scalar_close(actual: &BackendScalar, expected: &BackendScalar) {
    match (actual.as_f64(), expected.as_f64()) {
        (Some(a), Some(e)) => assert!((a - e).abs() < 1e-6),
        _ => {
            let a = actual
                .as_c64()
                .unwrap_or_else(|| Complex64::new(actual.real(), actual.imag()));
            let e = expected
                .as_c64()
                .unwrap_or_else(|| Complex64::new(expected.real(), expected.imag()));
            assert!((a.re - e.re).abs() < 1e-6);
            assert!((a.im - e.im).abs() < 1e-6);
        }
    }
}

#[test]
fn scalar_from_value_supports_all_supported_element_types() {
    let f32_scalar = BackendScalar::from_value(1.25_f32);
    let f64_scalar = BackendScalar::from_value(-2.5_f64);
    let c32_scalar = BackendScalar::from_value(Complex32::new(3.0, -0.5));
    let c64_scalar = BackendScalar::from_value(Complex64::new(-1.0, 2.0));

    assert_eq!(f32_scalar.real(), 1.25);
    assert_eq!(f64_scalar.real(), -2.5);
    assert_eq!(c32_scalar.real(), 3.0);
    assert_eq!(c32_scalar.imag(), -0.5);
    assert_eq!(Complex64::from(c64_scalar), Complex64::new(-1.0, 2.0));
}

#[test]
fn any_scalar_sum_from_real_storage_stays_real() {
    let dense = Storage::from_dense_col_major(vec![1.0, -2.5], &[2]).unwrap();
    let diag = Storage::from_diag_col_major(vec![3.0, 4.5], 2).unwrap();

    let dense_sum = BackendScalar::sum_from_storage(&dense);
    let diag_sum = BackendScalar::sum_from_storage(&diag);

    assert!(dense_sum.is_real());
    assert_eq!(dense_sum.real(), -1.5);
    assert!(diag_sum.is_real());
    assert_eq!(diag_sum.real(), 7.5);
}

#[test]
fn any_scalar_sum_from_complex_storage_stays_complex() {
    let dense = Storage::from_dense_col_major(
        vec![Complex64::new(1.0, -1.0), Complex64::new(-0.5, 2.0)],
        &[2],
    )
    .unwrap();

    let sum = BackendScalar::sum_from_storage(&dense);
    let sum_c64: Complex64 = sum.into();
    assert_eq!(sum_c64, Complex64::new(0.5, 1.0));
}

#[test]
fn complex_abs_rejects_nonfinite_components_before_norm() {
    for value in [
        Complex64::new(f64::INFINITY, f64::NAN),
        Complex64::new(f64::NAN, f64::INFINITY),
    ] {
        assert!(BackendScalar::from_value(value).abs().is_nan());
    }
    for value in [
        Complex32::new(f32::INFINITY, f32::NAN),
        Complex32::new(f32::NAN, f32::INFINITY),
    ] {
        assert!(BackendScalar::from_value(value).abs().is_nan());
    }
}

#[test]
fn scalar_arithmetic_uses_runtime_bridge() {
    let sum = BackendScalar::from_real(1.5) + BackendScalar::from_real(2.0);
    let diff = BackendScalar::from_complex(3.0, -1.0) - BackendScalar::from_real(1.0);
    let prod = BackendScalar::from_real(2.0) * BackendScalar::from_complex(0.0, 1.0);

    assert_eq!(sum.as_f64(), Some(3.5));
    assert_eq!(Complex64::from(diff), Complex64::new(2.0, -1.0));
    assert_eq!(Complex64::from(prod), Complex64::new(0.0, 2.0));
}

#[test]
fn promote_scalar_native_covers_all_scalar_type_pairs() {
    let cases = vec![
        (
            BackendScalar::from_value(1.25_f32),
            DType::F32,
            BackendScalar::from_value(1.25_f32),
        ),
        (
            BackendScalar::from_value(1.25_f32),
            DType::F64,
            BackendScalar::from_value(1.25_f64),
        ),
        (
            BackendScalar::from_value(1.25_f32),
            DType::C32,
            BackendScalar::from_value(Complex32::new(1.25, 0.0)),
        ),
        (
            BackendScalar::from_value(1.25_f32),
            DType::C64,
            BackendScalar::from_value(Complex64::new(1.25, 0.0)),
        ),
        (
            BackendScalar::from_value(-2.5_f64),
            DType::F32,
            BackendScalar::from_value(-2.5_f32),
        ),
        (
            BackendScalar::from_value(-2.5_f64),
            DType::F64,
            BackendScalar::from_value(-2.5_f64),
        ),
        (
            BackendScalar::from_value(-2.5_f64),
            DType::C32,
            BackendScalar::from_value(Complex32::new(-2.5, 0.0)),
        ),
        (
            BackendScalar::from_value(-2.5_f64),
            DType::C64,
            BackendScalar::from_value(Complex64::new(-2.5, 0.0)),
        ),
        (
            BackendScalar::from_value(Complex32::new(3.0, -0.5)),
            DType::F32,
            BackendScalar::from_value(3.0_f32),
        ),
        (
            BackendScalar::from_value(Complex32::new(3.0, -0.5)),
            DType::F64,
            BackendScalar::from_value(3.0_f64),
        ),
        (
            BackendScalar::from_value(Complex32::new(3.0, -0.5)),
            DType::C32,
            BackendScalar::from_value(Complex32::new(3.0, -0.5)),
        ),
        (
            BackendScalar::from_value(Complex32::new(3.0, -0.5)),
            DType::C64,
            BackendScalar::from_value(Complex64::new(3.0, -0.5)),
        ),
        (
            BackendScalar::from_value(Complex64::new(-1.0, 2.0)),
            DType::F32,
            BackendScalar::from_value(-1.0_f32),
        ),
        (
            BackendScalar::from_value(Complex64::new(-1.0, 2.0)),
            DType::F64,
            BackendScalar::from_value(-1.0_f64),
        ),
        (
            BackendScalar::from_value(Complex64::new(-1.0, 2.0)),
            DType::C32,
            BackendScalar::from_value(Complex32::new(-1.0, 2.0)),
        ),
        (
            BackendScalar::from_value(Complex64::new(-1.0, 2.0)),
            DType::C64,
            BackendScalar::from_value(Complex64::new(-1.0, 2.0)),
        ),
    ];

    for (source, target, expected) in cases {
        let promoted =
            BackendScalar::from_native(promote_scalar_native(source.as_native(), target).unwrap())
                .expect("promoted scalar");
        assert_eq!(promoted.native.dtype(), expected.native.dtype());
        assert_scalar_close(&promoted, &expected);
    }
}

#[test]
fn i64_native_scalar_is_supported_without_public_tensor_element() {
    let scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![-3_i64])
            .expect("valid i64 scalar test tensor"),
    )
    .expect("i64 native scalar");
    let zero = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![0_i64]).expect("valid i64 zero test tensor"),
    )
    .expect("i64 zero scalar");

    assert_eq!(scalar.native.dtype(), DType::I64);
    assert_eq!(scalar.real(), -3.0);
    assert_eq!(scalar.imag(), 0.0);
    assert_eq!(scalar.abs(), 3.0);
    assert!(!scalar.is_complex());
    assert!(scalar.is_real());
    assert!(!scalar.is_zero());
    assert!(zero.is_zero());
    assert_eq!(scalar.as_f64(), Some(-3.0));
    assert_eq!(scalar.as_c64(), None);
    assert_eq!(Complex64::from(scalar.clone()), Complex64::new(-3.0, 0.0));
    assert_eq!((-scalar).as_f64(), Some(3.0));
    assert_eq!(format!("{}", zero), "0");
}

#[test]
fn i32_native_scalar_is_supported_without_public_tensor_element() {
    let scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![-4_i32])
            .expect("valid i32 scalar test tensor"),
    )
    .expect("i32 native scalar");
    let zero = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![0_i32]).expect("valid i32 zero test tensor"),
    )
    .expect("i32 zero scalar");

    assert_eq!(scalar.native.dtype(), DType::I32);
    assert_eq!(scalar.real(), -4.0);
    assert_eq!(scalar.imag(), 0.0);
    assert_eq!(scalar.abs(), 4.0);
    assert!(!scalar.is_complex());
    assert!(scalar.is_real());
    assert!(!scalar.is_zero());
    assert!(zero.is_zero());
    assert_eq!(scalar.as_f64(), Some(-4.0));
    assert_eq!(scalar.as_c64(), None);
    assert_eq!(Complex64::from(scalar.clone()), Complex64::new(-4.0, 0.0));
    assert_eq!(scalar.conj(), scalar);
    assert_eq!((-scalar).as_f64(), Some(4.0));
    assert_eq!(format!("{}", zero), "0");
}

#[test]
fn bool_native_scalar_is_supported_without_public_tensor_element() {
    let true_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![true])
            .expect("valid bool true scalar test tensor"),
    )
    .expect("bool true scalar");
    let false_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![false])
            .expect("valid bool false scalar test tensor"),
    )
    .expect("bool false scalar");

    assert_eq!(true_scalar.native.dtype(), DType::Bool);
    assert_eq!(true_scalar.real(), 1.0);
    assert_eq!(true_scalar.imag(), 0.0);
    assert_eq!(true_scalar.abs(), 1.0);
    assert!(true_scalar.is_real());
    assert!(!true_scalar.is_zero());
    assert_eq!(true_scalar.as_f64(), Some(1.0));
    assert_eq!(true_scalar.as_c64(), None);
    assert_eq!(
        Complex64::from(true_scalar.clone()),
        Complex64::new(1.0, 0.0)
    );
    assert_eq!(true_scalar.conj(), true_scalar);
    assert_eq!((-true_scalar).as_f64(), Some(-1.0));
    assert_eq!(format!("{}", false_scalar), "false");

    assert_eq!(false_scalar.real(), 0.0);
    assert_eq!(false_scalar.abs(), 0.0);
    assert!(false_scalar.is_zero());
    assert_eq!(false_scalar.as_f64(), Some(0.0));
    assert_eq!((-false_scalar).as_f64(), Some(0.0));
}

#[test]
fn promote_i64_native_scalar_covers_supported_targets_and_rejections() {
    let i64_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![7_i64])
            .expect("valid i64 scalar test tensor"),
    )
    .expect("i64 scalar");

    let promoted_f32 = BackendScalar::from_native(
        promote_scalar_native(i64_scalar.as_native(), DType::F32).unwrap(),
    )
    .expect("promoted f32");
    let promoted_f64 = BackendScalar::from_native(
        promote_scalar_native(i64_scalar.as_native(), DType::F64).unwrap(),
    )
    .expect("promoted f64");
    let promoted_i64 = BackendScalar::from_native(
        promote_scalar_native(i64_scalar.as_native(), DType::I64).unwrap(),
    )
    .expect("promoted i64");
    let promoted_c32 = BackendScalar::from_native(
        promote_scalar_native(i64_scalar.as_native(), DType::C32).unwrap(),
    )
    .expect("promoted c32");
    let promoted_c64 = BackendScalar::from_native(
        promote_scalar_native(i64_scalar.as_native(), DType::C64).unwrap(),
    )
    .expect("promoted c64");

    assert_eq!(promoted_f32.native.dtype(), DType::F32);
    assert_eq!(promoted_f32.as_f64(), Some(7.0));
    assert_eq!(promoted_f64.native.dtype(), DType::F64);
    assert_eq!(promoted_f64.as_f64(), Some(7.0));
    assert_eq!(promoted_i64.native.dtype(), DType::I64);
    assert_eq!(promoted_i64.as_f64(), Some(7.0));
    assert_eq!(promoted_c32.native.dtype(), DType::C32);
    assert_eq!(promoted_c32.as_c64(), Some(Complex64::new(7.0, 0.0)));
    assert_eq!(promoted_c64.native.dtype(), DType::C64);
    assert_eq!(promoted_c64.as_c64(), Some(Complex64::new(7.0, 0.0)));

    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f32).as_native(), DType::I64).is_err()
    );
    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f64).as_native(), DType::I64).is_err()
    );
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex32::new(1.0, 0.0)).as_native(),
        DType::I64
    )
    .is_err());
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex64::new(1.0, 0.0)).as_native(),
        DType::I64
    )
    .is_err());
}

#[test]
fn promote_i32_native_scalar_covers_supported_targets_and_rejections() {
    let i32_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![5_i32])
            .expect("valid i32 scalar test tensor"),
    )
    .expect("i32 scalar");

    let promoted_f32 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::F32).unwrap(),
    )
    .expect("promoted f32");
    let promoted_f64 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::F64).unwrap(),
    )
    .expect("promoted f64");
    let promoted_i32 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::I32).unwrap(),
    )
    .expect("promoted i32");
    let promoted_i64 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::I64).unwrap(),
    )
    .expect("promoted i64");
    let promoted_c32 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::C32).unwrap(),
    )
    .expect("promoted c32");
    let promoted_c64 = BackendScalar::from_native(
        promote_scalar_native(i32_scalar.as_native(), DType::C64).unwrap(),
    )
    .expect("promoted c64");

    assert_eq!(promoted_f32.native.dtype(), DType::F32);
    assert_eq!(promoted_f32.as_f64(), Some(5.0));
    assert_eq!(promoted_f64.native.dtype(), DType::F64);
    assert_eq!(promoted_f64.as_f64(), Some(5.0));
    assert_eq!(promoted_i32.native.dtype(), DType::I32);
    assert_eq!(promoted_i32.as_f64(), Some(5.0));
    assert_eq!(promoted_i64.native.dtype(), DType::I64);
    assert_eq!(promoted_i64.as_f64(), Some(5.0));
    assert_eq!(promoted_c32.native.dtype(), DType::C32);
    assert_eq!(promoted_c32.as_c64(), Some(Complex64::new(5.0, 0.0)));
    assert_eq!(promoted_c64.native.dtype(), DType::C64);
    assert_eq!(promoted_c64.as_c64(), Some(Complex64::new(5.0, 0.0)));

    assert!(promote_scalar_native(i32_scalar.as_native(), DType::Bool).is_err());
    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f32).as_native(), DType::I32).is_err()
    );
    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f64).as_native(), DType::I32).is_err()
    );
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex32::new(1.0, 0.0)).as_native(),
        DType::I32
    )
    .is_err());
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex64::new(1.0, 0.0)).as_native(),
        DType::I32
    )
    .is_err());
}

#[test]
fn promote_bool_native_scalar_covers_supported_targets_and_rejections() {
    let true_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![true])
            .expect("valid bool scalar test tensor"),
    )
    .expect("bool scalar");

    let promoted_f32 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::F32).unwrap(),
    )
    .expect("promoted f32");
    let promoted_f64 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::F64).unwrap(),
    )
    .expect("promoted f64");
    let promoted_i32 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::I32).unwrap(),
    )
    .expect("promoted i32");
    let promoted_i64 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::I64).unwrap(),
    )
    .expect("promoted i64");
    let promoted_bool = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::Bool).unwrap(),
    )
    .expect("promoted bool");
    let promoted_c32 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::C32).unwrap(),
    )
    .expect("promoted c32");
    let promoted_c64 = BackendScalar::from_native(
        promote_scalar_native(true_scalar.as_native(), DType::C64).unwrap(),
    )
    .expect("promoted c64");

    assert_eq!(promoted_f32.native.dtype(), DType::F32);
    assert_eq!(promoted_f32.as_f64(), Some(1.0));
    assert_eq!(promoted_f64.native.dtype(), DType::F64);
    assert_eq!(promoted_f64.as_f64(), Some(1.0));
    assert_eq!(promoted_i32.native.dtype(), DType::I32);
    assert_eq!(promoted_i32.as_f64(), Some(1.0));
    assert_eq!(promoted_i64.native.dtype(), DType::I64);
    assert_eq!(promoted_i64.as_f64(), Some(1.0));
    assert_eq!(promoted_bool.native.dtype(), DType::Bool);
    assert_eq!(promoted_bool.as_f64(), Some(1.0));
    assert_eq!(promoted_c32.native.dtype(), DType::C32);
    assert_eq!(promoted_c32.as_c64(), Some(Complex64::new(1.0, 0.0)));
    assert_eq!(promoted_c64.native.dtype(), DType::C64);
    assert_eq!(promoted_c64.as_c64(), Some(Complex64::new(1.0, 0.0)));

    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f32).as_native(), DType::Bool)
            .is_err()
    );
    assert!(
        promote_scalar_native(BackendScalar::from_value(1.25_f64).as_native(), DType::Bool)
            .is_err()
    );
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex32::new(1.0, 0.0)).as_native(),
        DType::Bool
    )
    .is_err());
    assert!(promote_scalar_native(
        BackendScalar::from_value(Complex64::new(1.0, 0.0)).as_native(),
        DType::Bool
    )
    .is_err());
}

#[test]
fn promote_scalar_native_rejects_non_scalar_tensor() {
    let tensor = NativeTensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])
        .expect("valid non-scalar test tensor");

    let err = promote_scalar_native(&tensor, DType::F64).unwrap_err();

    assert!(err.to_string().contains("rank-0 scalar"));
}

#[test]
fn i64_native_scalar_participates_in_real_ordering() {
    let i64_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![3_i64])
            .expect("valid i64 scalar test tensor"),
    )
    .expect("i64 scalar");

    assert_eq!(
        i64_scalar.partial_cmp(&BackendScalar::from_value(2.5_f32)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        i64_scalar.partial_cmp(&BackendScalar::from_value(3.5_f64)),
        Some(Ordering::Less)
    );
    assert_eq!(
        i64_scalar.partial_cmp(
            &BackendScalar::from_native(
                NativeTensor::from_vec_col_major(vec![], vec![3_i64])
                    .expect("valid comparison scalar test tensor"),
            )
            .unwrap()
        ),
        Some(Ordering::Equal)
    );
}

#[test]
fn i32_and_bool_native_scalars_participate_in_real_ordering() {
    let i32_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![2_i32])
            .expect("valid i32 scalar test tensor"),
    )
    .expect("i32 scalar");
    let true_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![true])
            .expect("valid bool true scalar test tensor"),
    )
    .expect("bool true scalar");
    let false_scalar = BackendScalar::from_native(
        NativeTensor::from_vec_col_major(vec![], vec![false])
            .expect("valid bool false scalar test tensor"),
    )
    .expect("bool false scalar");

    assert_eq!(
        i32_scalar.partial_cmp(&BackendScalar::from_value(1.5_f32)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        i32_scalar.partial_cmp(&BackendScalar::from_value(2.5_f64)),
        Some(Ordering::Less)
    );
    assert_eq!(
        i32_scalar.partial_cmp(
            &BackendScalar::from_native(
                NativeTensor::from_vec_col_major(vec![], vec![2_i32])
                    .expect("valid comparison scalar test tensor"),
            )
            .unwrap()
        ),
        Some(Ordering::Equal)
    );
    assert_eq!(
        i32_scalar.partial_cmp(&true_scalar),
        Some(Ordering::Greater)
    );
    assert_eq!(
        true_scalar.partial_cmp(&BackendScalar::from_value(0.5_f32)),
        Some(Ordering::Greater)
    );
    assert_eq!(
        true_scalar.partial_cmp(&BackendScalar::from_value(1.5_f64)),
        Some(Ordering::Less)
    );
    assert_eq!(true_scalar.partial_cmp(&i32_scalar), Some(Ordering::Less));
    assert_eq!(
        true_scalar.partial_cmp(
            &BackendScalar::from_native(
                NativeTensor::from_vec_col_major(vec![], vec![1_i64])
                    .expect("valid comparison scalar test tensor"),
            )
            .unwrap()
        ),
        Some(Ordering::Equal)
    );
    assert_eq!(
        false_scalar.partial_cmp(
            &BackendScalar::from_native(
                NativeTensor::from_vec_col_major(vec![], vec![0_i32])
                    .expect("valid comparison scalar test tensor"),
            )
            .unwrap()
        ),
        Some(Ordering::Equal)
    );
    assert_eq!(
        false_scalar.partial_cmp(
            &BackendScalar::from_native(
                NativeTensor::from_vec_col_major(vec![], vec![false])
                    .expect("valid comparison scalar test tensor"),
            )
            .unwrap()
        ),
        Some(Ordering::Equal)
    );
}

#[test]
fn scalar_utility_methods_cover_real_and_complex_cases() {
    let zero = BackendScalar::zero();
    let real = BackendScalar::from_real(-4.0);
    let complex = BackendScalar::from_complex(3.0, -4.0);

    assert!(zero.is_zero());
    assert_eq!(zero.as_f64(), Some(0.0));
    assert!(real.is_real());
    assert!(!real.is_complex());
    assert_eq!(real.abs(), 4.0);
    assert_eq!(real.as_f64(), Some(-4.0));
    assert_eq!(real.as_c64(), None);

    assert!(complex.is_complex());
    assert!(!complex.is_real());
    assert_eq!(complex.abs(), 5.0);
    assert_eq!(complex.as_f64(), None);
    assert_eq!(complex.as_c64(), Some(Complex64::new(3.0, -4.0)));
    assert_eq!(complex.conj(), BackendScalar::from_complex(3.0, 4.0));
    assert_eq!(complex.real_part(), BackendScalar::from_real(3.0));
    assert_eq!(complex.imag_part(), BackendScalar::from_real(-4.0));

    let recomposed = BackendScalar::compose_complex(
        BackendScalar::from_real(1.5),
        BackendScalar::from_real(-2.0),
    )
    .expect("compose complex");
    assert_eq!(recomposed, BackendScalar::from_complex(1.5, -2.0));
    assert!(BackendScalar::compose_complex(
        BackendScalar::from_complex(1.0, 1.0),
        BackendScalar::from_real(0.0)
    )
    .is_err());

    assert_eq!(
        BackendScalar::from_real(9.0).sqrt(),
        BackendScalar::from_real(3.0)
    );
    assert_eq!(
        BackendScalar::from_real(-4.0).sqrt(),
        BackendScalar::from_complex(0.0, 2.0)
    );
    assert_scalar_close(
        &BackendScalar::from_complex(3.0, -4.0).sqrt().powi(2),
        &BackendScalar::from_complex(3.0, -4.0),
    );
    assert_eq!(
        BackendScalar::from_real(-2.0).powi(3),
        BackendScalar::from_real(-8.0)
    );
    assert_scalar_close(
        &BackendScalar::from_real(-4.0).powf(0.5),
        &BackendScalar::from_complex(0.0, 2.0),
    );
}

#[test]
fn scalar_trait_helpers_cover_ordering_and_conversions() {
    assert_eq!(BackendScalar::default(), BackendScalar::zero());
    assert_eq!(BackendScalar::one(), BackendScalar::from_real(1.0));
    assert!(BackendScalar::from_value(1.0_f32) < BackendScalar::from_value(2.0_f32));
    assert!(BackendScalar::from_value(1.0_f32) < BackendScalar::from_value(2.0_f64));
    assert!(BackendScalar::from_complex(1.0, 1.0)
        .partial_cmp(&BackendScalar::from_complex(1.0, 0.0))
        .is_none());

    assert_eq!(f64::try_from(BackendScalar::from_real(2.5)).unwrap(), 2.5);
    assert_eq!(
        f64::try_from(BackendScalar::from_value(1.25_f32)).unwrap(),
        1.25
    );
    assert!(f64::try_from(BackendScalar::from_complex(1.0, 0.5)).is_err());

    let debug = format!("{:?}", BackendScalar::from_real(1.0));
    assert!(debug.contains("BackendScalar"));
    assert!(debug.contains("dtype"));
}
