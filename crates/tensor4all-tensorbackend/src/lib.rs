#![warn(missing_docs)]
//! Tensor storage and linear algebra backend for tensor4all.
//!
//! This crate provides:
//! - [`Storage`]: Dynamic snapshot storage for logical tensor values
//! - [`StructuredStorage`]: `axis_classes`-aware materialized snapshots
//! - [`AnyScalar`]: Dynamic scalar type backed by rank-0 `tenferro::Tensor`
//! - tenferro-backed execution helpers for tensor algebra
//!
//! ## Feature Flags
//!
//! - `backend-tenferro` (default): Use tenferro backend for linalg/einsum

/// Dynamic scalar types supporting f32, f64, Complex32, and Complex64.
mod any_scalar;
/// Backend dispatch for dense linear algebra operations.
mod backend;
/// Process-global tenferro execution helpers.
mod context;
/// Dense column-major matrix type and backend-backed matrix utilities.
mod matrix;
/// Process-level memory pressure helpers.
mod memory;
/// Tensor snapshot storage types and low-level dense/diagonal kernels.
mod storage;
pub(crate) mod tenferro_bridge;
/// Supported public tensor element types and native constructor hooks.
mod tensor_element;

pub use any_scalar::AnyScalar;
pub use backend::{
    full_piv_lu_backend, full_piv_lu_matrix, qr_backend, solve_backend, solve_matrix,
    solve_matrix_owned, svd_backend, triangular_solve_backend, triangular_solve_matrix,
    triangular_solve_matrix_owned, BackendLinalgScalar, FullPivLuMatrixResult, FullPivLuResult,
    MatrixSolveScalar, MatrixTriangularSolveScalar, SvdResult,
};
pub use context::{default_eager_ctx, with_default_backend, EagerContextError};
pub use matrix::{
    batched_mat_mul_same_shape, batched_mat_mul_same_shape_owned, from_vec2d,
    hermitian_eigendecomposition, hermitian_exponential_first_column, lowest_hermitian_eigenpair,
    mat_mul, mat_mul_owned, submatrix, submatrix_argmax, swap_cols, swap_rows, transpose,
    try_from_vec2d, BlasMul, HermitianEigenError, HermitianEigenScalar,
    HermitianEigendecomposition, HermitianEigenpair, Matrix, MatrixScalar, MatrixShapeError,
    MatrixTensorConversionError,
};
pub use memory::{release_process_allocator_cached_memory, AllocatorPressureRelief};
pub use storage::{
    contract_storage, make_mut_storage, mindim, Storage, StorageError, StorageKind, StorageResult,
    StorageScalar, StructuredStorage, SumFromStorage,
};
pub use tenferro_bridge::{
    axpby_native_tensor, axpby_storage_native, conj_native_tensor, contract_native_tensor,
    contract_storage_native, dense_native_tensor_from_col_major, diag_native_tensor_from_col_major,
    einsum_native_tensor_reads, einsum_native_tensors, einsum_native_tensors_owned,
    native_tensor_primal_to_dense_c64_col_major, native_tensor_primal_to_dense_col_major,
    native_tensor_primal_to_dense_f64_col_major, native_tensor_primal_to_diag_c64,
    native_tensor_primal_to_diag_f64, native_tensor_primal_to_storage, outer_product_native_tensor,
    outer_product_storage_native, permute_native_tensor, permute_storage_native,
    print_and_reset_native_einsum_profile, qr_native_tensor, reset_native_einsum_profile,
    reshape_col_major_native_tensor, scale_native_tensor, scale_storage_native,
    storage_payload_native_read_input, storage_to_native_tensor, sum_native_tensor,
    svd_native_tensor, tangent_native_tensor, NativeTensorReadInput,
};
pub use tensor_element::TensorElement;
