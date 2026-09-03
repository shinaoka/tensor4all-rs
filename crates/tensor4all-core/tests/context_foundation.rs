use std::sync::Arc;

use tenferro_cpu::CpuBackend;
use tensor4all_core::{DynIndex, ExecutionContext, IdxTensor};
use tensor4all_tensorbackend::CpuExecutionContext;

fn cpu_context() -> ExecutionContext {
    ExecutionContext::Cpu(Arc::new(CpuExecutionContext::from_backend(
        CpuBackend::new(),
    )))
}

#[test]
fn context_scoped_construction_uses_the_supplied_cpu_runtime() {
    let context = cpu_context();
    let index = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense_in(&context, vec![index], vec![1.0_f64, 2.0]).unwrap();

    tensor.validate_context(&context).unwrap();
    assert_eq!(tensor.to_vec::<f64>().unwrap(), vec![1.0, 2.0]);

    let ones = IdxTensor::ones_in(&context, tensor.indices()).unwrap();
    assert_eq!(ones.to_vec::<f64>().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn context_validation_rejects_a_different_cpu_runtime() {
    let first = cpu_context();
    let second = cpu_context();
    let tensor =
        IdxTensor::from_dense_in(&first, vec![DynIndex::new_dyn(1)], vec![3.0_f64]).unwrap();

    assert!(tensor.validate_context(&second).is_err());
}

#[cfg(feature = "tenferro-cuda")]
#[test]
#[ignore]
fn context_scoped_cuda_construction_stays_in_the_selected_runtime() {
    let cuda = tensor4all_tensorbackend::CudaExecutionContext::new().unwrap();
    let context = ExecutionContext::Cuda(Arc::new(cuda));
    let tensor =
        IdxTensor::from_dense_in(&context, vec![DynIndex::new_dyn(2)], vec![1.0_f64, 2.0]).unwrap();

    tensor.validate_context(&context).unwrap();
}
