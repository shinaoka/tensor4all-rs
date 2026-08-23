use std::error::Error;

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

fn main() -> Result<(), Box<dyn Error>> {
    let left = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let tree = TreeTN::from_tensors(
        vec![
            IdxTensor::from_dense(vec![left, bond.clone()], vec![1.0_f64, 2.0, 3.0, 4.0])?,
            IdxTensor::from_dense(vec![bond, right], vec![5.0_f64, 6.0, 7.0, 8.0])?,
        ],
        vec![0, 1],
    )?;

    let cpu = tree.contract_to_tensor()?;
    let context = CudaExecutionContext::new()?;
    let resident_tree = tree.upload_cuda(&context)?;
    let resident_result = resident_tree.contract_to_tensor_cuda(&context)?;
    resident_result.validate_cuda_residency(&context)?;
    assert!(resident_result.to_vec::<f64>().is_err());

    let result = resident_result.download(&context)?;
    let residual = result.sub(&cpu)?.maxabs()?;
    assert!(residual <= 1.0e-10, "CUDA/CPU residual: {residual}");
    println!(
        "device={:?} residual_max_abs={residual:.3e}",
        context.device_name()
    );
    Ok(())
}
