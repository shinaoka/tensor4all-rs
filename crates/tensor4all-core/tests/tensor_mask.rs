use num_traits::Zero;
use tensor4all_core::{DynIndex, TensorDynLen};

fn compact_expected<T: Copy + Zero>(value: T) -> Vec<T> {
    let mut expected = vec![T::zero(); 12];
    expected[2] = value;
    expected[9] = value;
    expected
}

#[test]
fn mask_index_preserves_values_indices_and_reverse_mode_graph() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let source = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])
        .unwrap()
        .enable_grad()
        .unwrap();

    let masked = source.mask_index(&i, 1).unwrap();
    assert_eq!(masked.indices(), &[i.clone(), j.clone()]);
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 2.0, 0.0, 4.0]);
    assert!(masked.tracks_grad());

    let loss = masked.sum().unwrap();
    loss.backward().unwrap();
    assert_eq!(
        source.grad().unwrap().unwrap().to_vec::<f64>().unwrap(),
        vec![0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn mask_index_rejects_missing_or_out_of_range_indices() {
    let index = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![index.clone()], vec![1.0_f64, 2.0]).unwrap();

    assert!(tensor.mask_index(&DynIndex::new_dyn(2), 0).is_err());
    assert!(tensor.mask_index(&index, 2).is_err());
}

#[test]
fn mask_index_preserves_compact_diagonal_storage() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let source = TensorDynLen::from_diag(vec![i.clone(), j], vec![3.0_f64, 4.0]).unwrap();

    let masked = source.mask_index(&i, 1).unwrap();
    assert!(masked.is_diag());
    assert_eq!(masked.to_vec::<f64>().unwrap(), vec![0.0, 0.0, 0.0, 4.0]);
}

macro_rules! assert_mask_index_dtype_and_grad {
    ($name:ident, $ty:ty, $values:expr, $expected:expr, $grad:expr, $wrong:ty) => {
        #[test]
        fn $name() {
            let index = DynIndex::new_dyn(2);
            let source = TensorDynLen::from_dense(vec![index.clone()], $values)
                .unwrap()
                .enable_grad()
                .unwrap();
            let source_alias = source.clone();

            let masked = source.mask_index(&index, 1).unwrap();
            assert_eq!(masked.to_vec::<$ty>().unwrap(), $expected);
            assert!(masked.to_vec::<$wrong>().is_err());
            assert!(masked.tracks_grad());

            masked.sum().unwrap().backward().unwrap();
            assert_eq!(
                source_alias
                    .grad()
                    .unwrap()
                    .unwrap()
                    .to_vec::<$ty>()
                    .unwrap(),
                $grad,
            );
        }
    };
}

assert_mask_index_dtype_and_grad!(
    mask_index_preserves_f32_dtype_and_gradient,
    f32,
    vec![1.0_f32, 2.0_f32],
    vec![0.0_f32, 2.0_f32],
    vec![0.0_f32, 1.0_f32],
    f64
);
assert_mask_index_dtype_and_grad!(
    mask_index_preserves_f64_dtype_and_gradient,
    f64,
    vec![1.0_f64, 2.0_f64],
    vec![0.0_f64, 2.0_f64],
    vec![0.0_f64, 1.0_f64],
    f32
);
assert_mask_index_dtype_and_grad!(
    mask_index_preserves_c32_dtype_and_gradient,
    num_complex::Complex32,
    vec![
        num_complex::Complex32::new(1.0, 0.5),
        num_complex::Complex32::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
    ],
    num_complex::Complex64
);
assert_mask_index_dtype_and_grad!(
    mask_index_preserves_c64_dtype_and_gradient,
    num_complex::Complex64,
    vec![
        num_complex::Complex64::new(1.0, 0.5),
        num_complex::Complex64::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
    ],
    num_complex::Complex32
);

macro_rules! assert_structured_mask_index_dtype_and_grad {
    ($name:ident, $ty:ty, $values:expr, $expected:expr, $expected_gradient:expr) => {
        #[test]
        fn $name() {
            let i = DynIndex::new_dyn(2);
            let j = DynIndex::new_dyn(2);
            let source = TensorDynLen::from_diag(vec![i.clone(), j], $values)
                .unwrap()
                .enable_grad()
                .unwrap();
            let masked = source.mask_index(&i, 1).unwrap();
            assert_eq!(masked.to_vec::<$ty>().unwrap(), $expected);
            assert!(masked.tracks_grad());

            masked.sum().unwrap().backward().unwrap();
            let gradient = source.grad().unwrap().unwrap().to_vec::<$ty>().unwrap();
            assert_eq!(gradient, $expected_gradient);
        }
    };
}

assert_structured_mask_index_dtype_and_grad!(
    mask_index_preserves_structured_f32_gradient,
    f32,
    vec![1.0_f32, 2.0_f32],
    vec![0.0_f32, 0.0, 0.0, 2.0],
    vec![0.0_f32, 0.0, 0.0, 1.0]
);
assert_structured_mask_index_dtype_and_grad!(
    mask_index_preserves_structured_f64_gradient,
    f64,
    vec![1.0_f64, 2.0_f64],
    vec![0.0_f64, 0.0, 0.0, 2.0],
    vec![0.0_f64, 0.0, 0.0, 1.0]
);
assert_structured_mask_index_dtype_and_grad!(
    mask_index_preserves_structured_c32_gradient,
    num_complex::Complex32,
    vec![
        num_complex::Complex32::new(1.0, 0.5),
        num_complex::Complex32::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
    ]
);
assert_structured_mask_index_dtype_and_grad!(
    mask_index_preserves_structured_c64_gradient,
    num_complex::Complex64,
    vec![
        num_complex::Complex64::new(1.0, 0.5),
        num_complex::Complex64::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(2.0, -0.25),
    ],
    vec![
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(0.0, 0.0),
        num_complex::Complex64::new(1.0, 0.0),
    ]
);

macro_rules! assert_compact_mask_index_dtype_and_grad {
    ($name:ident, $ty:ty, $scale:expr, $expected:expr, $expected_gradient:expr) => {
        #[test]
        fn $name() {
            let left = DynIndex::new_dyn(2);
            let site = DynIndex::new_dyn(3);
            let right = DynIndex::new_dyn(2);
            let source =
                TensorDynLen::from_copy_selector(left.clone(), site.clone(), right, 1, $scale)
                    .unwrap()
                    .enable_grad()
                    .unwrap();
            let masked = source.mask_index(&site, 1).unwrap();
            assert_eq!(masked.to_vec::<$ty>().unwrap(), $expected);
            assert!(masked.tracks_grad());

            masked.sum().unwrap().backward().unwrap();
            assert_eq!(
                source.grad().unwrap().unwrap().to_vec::<$ty>().unwrap(),
                $expected_gradient,
            );
        }
    };
}

assert_compact_mask_index_dtype_and_grad!(
    mask_index_preserves_compact_f32_gradient,
    f32,
    2.0_f32,
    compact_expected(2.0_f32),
    compact_expected(1.0_f32)
);
assert_compact_mask_index_dtype_and_grad!(
    mask_index_preserves_compact_f64_gradient,
    f64,
    2.0_f64,
    compact_expected(2.0_f64),
    compact_expected(1.0_f64)
);
assert_compact_mask_index_dtype_and_grad!(
    mask_index_preserves_compact_c32_gradient,
    num_complex::Complex32,
    num_complex::Complex32::new(2.0, -0.5),
    compact_expected(num_complex::Complex32::new(2.0, -0.5)),
    compact_expected(num_complex::Complex32::new(1.0, 0.0))
);
assert_compact_mask_index_dtype_and_grad!(
    mask_index_preserves_compact_c64_gradient,
    num_complex::Complex64,
    num_complex::Complex64::new(2.0, -0.5),
    compact_expected(num_complex::Complex64::new(2.0, -0.5)),
    compact_expected(num_complex::Complex64::new(1.0, 0.0))
);
