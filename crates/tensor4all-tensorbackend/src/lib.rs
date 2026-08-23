#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! [`CpuExecutionContext`] is the canonical CPU integration path. It requires a
//! caller-supplied backend and owns plain, graph, and eager-AD runtime state.
//!
//! ## Feature flags
//!
//! - `explicit-context`: explicit CPU execution and logical tensor transfer.
//! - `global-defaults`: legacy process-global tensor operations.
//! - `backend-tenferro` (default): compatibility alias for `global-defaults`.

#[cfg(feature = "global-defaults")]
/// Dynamic scalar types supporting f32, f64, Complex32, and Complex64.
mod any_scalar;
#[cfg(feature = "global-defaults")]
/// Backend dispatch for dense linear algebra operations.
mod backend;
#[cfg(feature = "explicit-context")]
/// Explicit and optional process-global tenferro execution helpers.
mod context;
#[cfg(feature = "tenferro-cuda")]
/// Explicit visible-ordinal-0 CUDA execution and transfer boundaries.
mod cuda;
#[cfg(feature = "explicit-context")]
/// Backend-free tensor snapshots for execution-domain transfer.
mod logical_tensor;
#[cfg(feature = "global-defaults")]
/// Dense column-major matrix type and backend-backed matrix utilities.
mod matrix;
#[cfg(feature = "global-defaults")]
/// Process-level memory pressure helpers.
mod memory;
#[cfg(feature = "global-defaults")]
/// Tensor snapshot storage types and low-level dense/diagonal kernels.
mod storage;
#[cfg(feature = "global-defaults")]
pub(crate) mod tenferro_bridge;
#[cfg(feature = "global-defaults")]
/// Supported public tensor element types and native constructor hooks.
mod tensor_element;

#[cfg(feature = "global-defaults")]
pub use any_scalar::AnyScalar;
#[cfg(feature = "global-defaults")]
pub use backend::{
    full_piv_lu_backend, full_piv_lu_matrix, qr_backend, solve_backend, solve_matrix,
    solve_matrix_owned, svd_backend, triangular_solve_backend, triangular_solve_matrix,
    triangular_solve_matrix_owned, BackendLinalgError, BackendLinalgScalar, FullPivLuMatrixResult,
    FullPivLuResult, FullPivLuScalar, MatrixSolveScalar, MatrixTriangularSolveScalar, SvdResult,
};
#[cfg(feature = "global-defaults")]
pub use context::{default_eager_ctx, with_default_backend, EagerContextError};
#[cfg(feature = "explicit-context")]
pub use context::{CpuExecutionContext, CpuExecutionContextError};
#[cfg(feature = "tenferro-cuda")]
pub use cuda::{CudaExecutionContext, CudaExecutionContextError, CUDA_ORDINAL};
#[cfg(feature = "explicit-context")]
pub use logical_tensor::{LogicalTensor, LogicalTensorData, LogicalTensorError};
#[cfg(feature = "global-defaults")]
pub use matrix::{
    batched_mat_mul_same_shape, batched_mat_mul_same_shape_owned, from_vec2d,
    hermitian_eigendecomposition, hermitian_exponential_first_column, lowest_hermitian_eigenpair,
    mat_mul, mat_mul_owned, submatrix, submatrix_argmax, swap_cols, swap_rows, transpose,
    try_from_vec2d, BlasMul, HermitianEigenError, HermitianEigenScalar,
    HermitianEigendecomposition, HermitianEigenpair, Matrix, MatrixScalar, MatrixShapeError,
    MatrixTensorConversionError,
};
#[cfg(feature = "global-defaults")]
pub use memory::{release_process_allocator_cached_memory, AllocatorPressureRelief};
#[cfg(feature = "global-defaults")]
pub use storage::{
    contract_storage, make_mut_storage, min_dim, Storage, StorageError, StorageKind, StorageResult,
    StorageScalar, StructuredStorage, SumFromStorage,
};
#[cfg(feature = "global-defaults")]
pub use tenferro_bridge::{
    axpby_native_tensor, axpby_storage_native, conj_native_tensor, contract_native_tensor,
    contract_storage_native, dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    einsum_native_tensor_reads, einsum_native_tensors, einsum_native_tensors_owned,
    native_tensor_primal_to_dense_col_major, native_tensor_primal_to_diag,
    native_tensor_primal_to_storage, outer_product_native_tensor, outer_product_storage_native,
    permute_native_tensor, permute_storage_native, print_and_reset_native_einsum_profile,
    qr_native_tensor, reset_native_einsum_profile, reshape_col_major_native_tensor,
    scale_native_tensor, scale_storage_native, storage_payload_native_read_input,
    storage_to_native_tensor, sum_native_tensor, svd_native_tensor, tangent_native_tensor,
    BridgeError, NativeTensorReadInput,
};
#[cfg(feature = "global-defaults")]
pub use tensor_element::TensorElement;

/// Extract a result whose error branch means validated internal state is inconsistent.
#[cfg(feature = "global-defaults")]
pub(crate) fn require_invariant<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: &str,
) -> T {
    let valid = result.is_ok();
    if let Err(error) = &result {
        assert!(valid, "{context}: {error}");
    }
    match result {
        Ok(value) => value,
        Err(_) => loop {
            std::hint::spin_loop();
        },
    }
}

#[cfg(all(test, feature = "global-defaults"))]
mod invariant_tests {
    use super::require_invariant;

    #[test]
    fn require_invariant_returns_success_and_reports_failure_context() {
        assert_eq!(require_invariant::<_, &str>(Ok(7), "valid state"), 7);

        let failure = std::panic::catch_unwind(|| {
            require_invariant::<(), _>(Err("broken state"), "tensor invariant")
        });
        let message = failure
            .unwrap_err()
            .downcast::<String>()
            .map(|message| *message)
            .unwrap_or_default();
        assert!(message.contains("tensor invariant: broken state"));
    }
}
