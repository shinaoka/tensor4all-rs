use num_complex::{Complex32, Complex64};
use tensor4all_core::{DynIndex, TensorDynLen};

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
