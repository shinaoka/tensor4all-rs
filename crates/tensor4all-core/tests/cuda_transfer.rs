#![cfg(feature = "tenferro-cuda")]

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IdxTensorCudaError};

#[test]
fn dense_upload_download_preserves_indices_and_values() {
    let context = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let source = IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let resident = source.upload_cuda(&context).unwrap();
    resident.validate_cuda_residency(&context).unwrap();
    assert!(resident.to_vec::<f64>().is_err());

    let restored = resident.download(&context).unwrap();
    assert_eq!(restored.indices(), &[i, j]);
    assert_eq!(
        restored.to_vec::<f64>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn foreign_context_is_typed() {
    let first = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let second = CudaExecutionContext::new().expect("CUDA ordinal 0 must be available");
    let source = IdxTensor::from_dense(vec![DynIndex::new_dyn(1)], vec![2.0_f64]).unwrap();
    let resident = source.upload_cuda(&first).unwrap();

    assert!(matches!(
        resident.validate_cuda_residency(&second),
        Err(IdxTensorCudaError::ForeignContext { .. })
    ));
    assert!(matches!(
        resident.download(&second),
        Err(IdxTensorCudaError::ForeignContext { .. })
    ));
}
