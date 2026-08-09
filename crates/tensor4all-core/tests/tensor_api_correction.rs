use num_complex::{Complex32, Complex64};
use std::error::Error;
use std::sync::Arc;
use tensor4all_core::{DynIndex, TensorDynLen, TensorDynLenError, TensorStorageError};

#[test]
fn tensor_dyn_len_norm_and_comparison_operations_are_fallible() {
    let index = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![index.clone()], vec![3.0_f64, 4.0]).unwrap();

    assert!((tensor.norm_squared().unwrap() - 25.0).abs() < 1.0e-12);
    assert!((tensor.norm().unwrap() - 5.0).abs() < 1.0e-12);
    assert!((tensor.maxabs().unwrap() - 4.0).abs() < 1.0e-12);
    assert!(tensor.isapprox(&tensor, 0.0, 0.0).unwrap());

    let other_index = DynIndex::new_dyn(2);
    let incompatible = TensorDynLen::from_dense(vec![other_index], vec![3.0_f64, 4.0]).unwrap();
    let error = tensor.isapprox(&incompatible, 1.0e-12, 0.0).unwrap_err();
    assert!(error.to_string().contains("Index"));
}

#[test]
fn tensor_dyn_len_norm_and_maxabs_preserve_complex_values() {
    let index = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(
        vec![index],
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, -2.0)],
    )
    .unwrap();

    assert!((tensor.norm_squared().unwrap() - 29.0).abs() < 1.0e-12);
    assert!((tensor.norm().unwrap() - 29.0_f64.sqrt()).abs() < 1.0e-12);
    assert!((tensor.maxabs().unwrap() - 5.0).abs() < 1.0e-12);
}

#[test]
fn tensor_dyn_len_operations_cover_f32_and_c32() {
    let index = DynIndex::new_dyn(2);
    let f32_tensor = TensorDynLen::from_dense(vec![index.clone()], vec![3.0_f32, 4.0]).unwrap();
    assert!((f32_tensor.norm_squared().unwrap() - 25.0).abs() < 1.0e-6);
    assert_eq!(f32_tensor.to_vec::<f32>().unwrap(), vec![3.0_f32, 4.0]);

    let c32_tensor = TensorDynLen::from_dense(
        vec![index],
        vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, -2.0)],
    )
    .unwrap();
    assert!((c32_tensor.norm_squared().unwrap() - 29.0).abs() < 1.0e-5);
    assert_eq!(
        c32_tensor.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, -2.0)]
    );
}

#[test]
fn arithmetic_does_not_promote_32_bit_tensors() {
    let index = DynIndex::new_dyn(2);
    let f32_tensor = TensorDynLen::from_dense(vec![index.clone()], vec![1.0_f32, 2.0]).unwrap();
    let scaled_f32 = f32_tensor
        .scale(tensor4all_core::AnyScalar::new_real(2.0))
        .unwrap();
    assert_eq!(scaled_f32.to_vec::<f32>().unwrap(), vec![2.0, 4.0]);
    assert!(scaled_f32.to_vec::<f64>().is_err());

    let complex_scaled_f32 = f32_tensor
        .scale(tensor4all_core::AnyScalar::new_complex(0.0, 1.0))
        .unwrap();
    assert_eq!(
        complex_scaled_f32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(0.0, 1.0), Complex32::new(0.0, 2.0)]
    );
    assert!(complex_scaled_f32.to_vec::<Complex64>().is_err());

    let combined_f32 = f32_tensor
        .axpby(
            tensor4all_core::AnyScalar::new_real(2.0),
            &f32_tensor,
            tensor4all_core::AnyScalar::new_real(3.0),
        )
        .unwrap();
    assert_eq!(combined_f32.to_vec::<f32>().unwrap(), vec![5.0, 10.0]);
    assert!(combined_f32.to_vec::<f64>().is_err());

    let c32_tensor = TensorDynLen::from_dense(
        vec![index],
        vec![Complex32::new(1.0, 0.5), Complex32::new(2.0, -0.25)],
    )
    .unwrap();
    let scaled_c32 = c32_tensor
        .scale(tensor4all_core::AnyScalar::new_real(2.0))
        .unwrap();
    assert_eq!(
        scaled_c32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(2.0, 1.0), Complex32::new(4.0, -0.5)]
    );
    assert!(scaled_c32.to_vec::<Complex64>().is_err());

    let combined_c32 = c32_tensor
        .axpby(
            tensor4all_core::AnyScalar::new_real(2.0),
            &c32_tensor,
            tensor4all_core::AnyScalar::new_real(3.0),
        )
        .unwrap();
    assert_eq!(
        combined_c32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(5.0, 2.5), Complex32::new(10.0, -1.25)]
    );
    assert!(combined_c32.to_vec::<Complex64>().is_err());
}

#[test]
fn tensor_dyn_len_norm_api_has_crate_local_error_and_source_chain() {
    fn assert_typed_result<T, E: Error + Send + Sync + 'static>(_: Result<T, E>) {}

    let index = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![index], vec![3.0_f64, 4.0]).unwrap();
    assert_typed_result(tensor.norm_squared());
    assert_typed_result(tensor.norm());
    assert_typed_result(tensor.maxabs());
    assert_typed_result(tensor.isapprox(&tensor, 0.0, 0.0));

    let error = TensorDynLenError::Storage {
        source: TensorStorageError::Materialization {
            source: Arc::new(std::io::Error::other("backend unavailable")),
        },
    };
    assert!(error.source().is_some());
    assert!(error.source().and_then(Error::source).is_some());
}

#[test]
fn tensor_dyn_len_norm_rejects_nan_and_preserves_infinity() {
    for tensor in [
        TensorDynLen::scalar(f64::NAN).unwrap(),
        TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0, f64::NAN]).unwrap(),
    ] {
        assert!(matches!(
            tensor.norm_squared(),
            Err(TensorDynLenError::NaNInput { .. })
        ));
        assert!(matches!(
            tensor.norm(),
            Err(TensorDynLenError::NaNInput { .. })
        ));
        assert!(matches!(
            tensor.maxabs(),
            Err(TensorDynLenError::NaNInput { .. })
        ));
    }

    let tensor = TensorDynLen::scalar(f64::INFINITY).unwrap();
    assert!(tensor.norm_squared().unwrap().is_infinite());
    assert!(tensor.norm().unwrap().is_infinite());
    assert!(tensor.maxabs().unwrap().is_infinite());

    for tensor in [
        TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f32, f32::NAN]).unwrap(),
        TensorDynLen::from_dense(
            vec![DynIndex::new_dyn(2)],
            vec![Complex32::new(1.0, 0.0), Complex32::new(f32::NAN, 0.0)],
        )
        .unwrap(),
    ] {
        assert!(matches!(
            tensor.norm_squared(),
            Err(TensorDynLenError::NaNInput { .. })
        ));
        assert!(matches!(
            tensor.maxabs(),
            Err(TensorDynLenError::NaNInput { .. })
        ));
    }
}

#[test]
fn tensor_dyn_len_distance_uses_typed_metric_error() {
    let tensor = TensorDynLen::scalar(f64::NAN).unwrap();
    let result: Result<f64, TensorDynLenError> = tensor.distance(&tensor);
    assert!(matches!(
        result,
        Err(TensorDynLenError::NaNInput {
            operation: "norm_squared"
        })
    ));
}

#[test]
fn structured_32_bit_selection_preserves_dtype() {
    let f32_i = DynIndex::new_dyn(2);
    let f32_j = DynIndex::new_dyn(2);
    let f32_tensor =
        TensorDynLen::from_diag(vec![f32_i.clone(), f32_j], vec![1.0_f32, 2.0]).unwrap();
    let selected_f32 = f32_tensor.select_indices(&[f32_i], &[1]).unwrap();
    assert_eq!(selected_f32.to_vec::<f32>().unwrap(), vec![0.0, 2.0]);
    assert!(selected_f32.to_vec::<f64>().is_err());

    let c32_i = DynIndex::new_dyn(2);
    let c32_j = DynIndex::new_dyn(2);
    let c32_tensor = TensorDynLen::from_diag(
        vec![c32_i.clone(), c32_j],
        vec![Complex32::new(1.0, 0.5), Complex32::new(2.0, -0.25)],
    )
    .unwrap();
    let selected_c32 = c32_tensor.select_indices(&[c32_i], &[1]).unwrap();
    assert_eq!(
        selected_c32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(0.0, 0.0), Complex32::new(2.0, -0.25)]
    );
    assert!(selected_c32.to_vec::<Complex64>().is_err());

    let selector_f32 = TensorDynLen::from_copy_selector(
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
        1,
        2.0_f32,
    )
    .unwrap();
    let selected_selector_f32 = selector_f32
        .select_indices(&[selector_f32.indices()[1].clone()], &[1])
        .unwrap();
    assert_eq!(
        selected_selector_f32.to_vec::<f32>().unwrap(),
        vec![2.0, 0.0, 0.0, 2.0]
    );
    assert!(selected_selector_f32.to_vec::<f64>().is_err());

    let selector_c32 = TensorDynLen::from_copy_selector(
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(3),
        DynIndex::new_dyn(2),
        1,
        Complex32::new(2.0, -0.5),
    )
    .unwrap();
    let selected_selector_c32 = selector_c32
        .select_indices(&[selector_c32.indices()[1].clone()], &[1])
        .unwrap();
    assert_eq!(
        selected_selector_c32.to_vec::<Complex32>().unwrap(),
        vec![
            Complex32::new(2.0, -0.5),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(2.0, -0.5),
        ]
    );
    assert!(selected_selector_c32.to_vec::<Complex64>().is_err());
}

#[test]
fn detach_preserves_32_bit_dtype_after_enabling_grad() {
    let f32_tensor = TensorDynLen::from_dense(vec![DynIndex::new_dyn(2)], vec![1.0_f32, 2.0])
        .unwrap()
        .enable_grad()
        .unwrap();
    let detached_f32 = f32_tensor.detach().unwrap();
    assert_eq!(detached_f32.to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    assert!(detached_f32.to_vec::<f64>().is_err());

    let c32_tensor = TensorDynLen::from_dense(
        vec![DynIndex::new_dyn(2)],
        vec![Complex32::new(1.0, 0.5), Complex32::new(2.0, -0.25)],
    )
    .unwrap()
    .enable_grad()
    .unwrap();
    let detached_c32 = c32_tensor.detach().unwrap();
    assert_eq!(
        detached_c32.to_vec::<Complex32>().unwrap(),
        vec![Complex32::new(1.0, 0.5), Complex32::new(2.0, -0.25)]
    );
    assert!(detached_c32.to_vec::<Complex64>().is_err());

    let structured_f32 = TensorDynLen::from_diag(
        vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        vec![1.0_f32, 2.0],
    )
    .unwrap()
    .enable_grad()
    .unwrap();
    let detached_structured_f32 = structured_f32.detach().unwrap();
    assert_eq!(
        detached_structured_f32.to_vec::<f32>().unwrap(),
        vec![1.0, 0.0, 0.0, 2.0]
    );
    assert!(detached_structured_f32.to_vec::<f64>().is_err());
}

#[test]
fn structured_constructors_preserve_f32_and_c32_dtype() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let f32_diag =
        TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![1.0_f32, 2.0_f32]).unwrap();
    assert!(f32_diag.is_diag());
    assert_eq!(f32_diag.to_vec::<f32>().unwrap(), vec![1.0, 0.0, 0.0, 2.0]);
    assert!(f32_diag.to_vec::<f64>().is_err());

    let c32_diag = TensorDynLen::from_diag(
        vec![i, j],
        vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
    )
    .unwrap();
    assert!(c32_diag.is_diag());
    assert_eq!(
        c32_diag.to_vec::<Complex32>().unwrap(),
        vec![
            Complex32::new(1.0, 2.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
            Complex32::new(3.0, 4.0),
        ]
    );
    assert!(c32_diag.to_vec::<Complex64>().is_err());

    let left = DynIndex::new_dyn(2);
    let site = DynIndex::new_dyn(3);
    let right = DynIndex::new_dyn(2);
    let selector = TensorDynLen::from_copy_selector(left, site, right, 1, 2.5_f32).unwrap();
    assert_eq!(
        selector
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .filter(|v| **v != 0.0)
            .count(),
        2
    );
    assert!(selector.to_vec::<f64>().is_err());
}
