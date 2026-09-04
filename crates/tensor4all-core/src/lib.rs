#![warn(missing_docs)]
//! Core tensor operations and types for tensor4all-rs.
//!
//! This crate provides the foundational types and operations for tensor networks:
//!
//! - **Index types**: [`DynIndex`], [`Index`], [`DynId`] for tensor indices
//! - **Tag sets**: [`TagSet`], [`TagSetLike`] for metadata tagging
//! - **Tensors**: [`IdxTensor`] for dynamic-rank dense tensors
//! - **Operations**: Contraction, SVD, QR decomposition, factorization
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{Index, DynIndex, IdxTensor};
//!
//! // Create indices with dynamic identity
//! let i = Index::new_dyn(2);
//! let j = Index::new_dyn(3);
//!
//! // Create a tensor
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

pub mod col_major_array;
pub use col_major_array::{ColMajorArray, ColMajorArrayMut, ColMajorArrayRef};

// Common (tags, utilities, scalar)
/// Dynamic scalar compatibility wrapper built on rank-0 `IdxTensor`.
pub mod any_scalar;
pub mod global_default;
/// Bit-packed integer keys for multi-index maps.
pub mod index_key;
pub mod index_like;
pub mod scalar;
/// Stack-allocated fixed-capacity string types for ITensors.jl compatibility.
pub mod smallstring;
/// Tag set types for tensor metadata.
pub mod tagset;
pub mod truncation;

// TCI substrate absorbed from tensor4all-tcicore (dissolved, #639): matrix
// cross interpolation (LU/LUCI/ACA), common scalar trait, floating-zone
// pivot search, caching, and index sets.
pub mod cached_function;
pub mod error;
pub mod floating_zone;
pub mod indexset;
mod matrix_luci;
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod traits;

pub use self::matrixluci::MatrixLuciScalar;
pub use cached_function::cache_key::CacheKey;
pub use cached_function::error::CacheKeyError;
pub use cached_function::index_int::IndexInt;
pub use cached_function::CachedFunction;
pub use error::{MatrixCIError, Result};
pub use floating_zone::floating_zone_walk;
pub use indexset::{IndexSet, LocalIndex, MultiIndex};
pub use matrix_luci::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix,
    matrix_luci_factors_from_matrix_owned, MatrixLUCI, MatrixLuciFactors,
};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_mut, RrLU, RrLUOptions};
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;

pub use scalar::Scalar as CommonScalar;

// Default concrete type implementations (index, tensor, linalg, etc.)
pub mod defaults;

// Backwards compatibility: re-export defaults submodules as top-level modules
// This allows `tensor4all_core::index::...` to work
pub use defaults::index;

pub use defaults::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, TagSet};
pub use index_like::{sort_indices_deterministic, ConjState, IndexLike};

/// Index operations (replacement, set operations, contraction preparation).
pub mod index_ops;
pub use index_ops::{
    check_unique_indices, common_ind_positions, common_inds, has_common_inds, has_inds, hasind,
    noncommon_inds, replace_indices, replace_indices_mut, union_inds, unique_inds,
    ReplaceIndsError,
};
pub use smallstring::{SmallChar, SmallString, SmallStringError};
pub use tagset::{Tag, TagSetError, TagSetLike};

// Tensor (storage, tensor types)
pub mod tensor_index;
pub mod tensor_like;

pub use tensor_index::TensorIndex;

// Krylov subspace methods (GMRES, etc.)
pub mod krylov;

// Block tensor for block matrix GMRES
pub mod block_tensor;

// Backwards compatibility: re-export defaults::idx_tensor as tensor
pub use defaults::idx_tensor as tensor;

pub use any_scalar::{AnyScalar, AnyScalarError};
pub use defaults::idx_tensor::{
    compute_permutation_from_indices, diag_idx_tensor, unfold_split, IdxTensor, IdxTensorError,
    StructuredSelectorError, TensorStorageError,
};
#[cfg(feature = "tenferro-cuda")]
pub use defaults::IdxTensorCudaError;
#[cfg(feature = "backend-tenferro")]
pub use tensor4all_tensorbackend::ExecutionContext;
pub use tensor4all_tensorbackend::TensorElement;
pub use tensor4all_tensorbackend::{
    print_and_reset_native_einsum_profile, reset_native_einsum_profile,
};
#[cfg(feature = "tenferro-cuda")]
pub use tensor4all_tensorbackend::{CudaExecutionContext, CudaExecutionContextError, CUDA_ORDINAL};
pub use tensor_like::{
    Canonical, DirectSumResult, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
    LinearizationOrder, TensorConstructionLike, TensorContractionLike, TensorFactorizationLike,
    TensorLike, TensorVectorSpace, TensorVectorSpaceError,
};

pub use defaults::contract::{
    contract, contract_owned, contract_owned_with_options, contract_pair,
    contract_pair_with_operand_options, contract_pair_with_options, contract_with_options,
    outer_product, print_and_reset_contract_profile, reset_contract_profile, tensordot,
    ContractionOptions, PairwiseContractionOptions, PreparedContraction,
};
pub use defaults::idx_tensor::{
    print_and_reset_pairwise_contract_profile, reset_pairwise_contract_profile,
};

// Re-export linear algebra modules from defaults for backwards compatibility
// This allows `tensor4all_core::svd::...`, `tensor4all_core::qr::...`, etc.
pub mod direct_sum {
    //! Re-export of direct sum operations.
    pub use crate::defaults::direct_sum::*;
}
pub mod factorize {
    //! Re-export of factorization operations.
    pub use crate::defaults::factorize::*;
}
pub mod qr {
    //! Re-export of QR decomposition operations.
    pub use crate::defaults::qr::*;
}
pub mod svd {
    //! Re-export of SVD decomposition operations.
    pub use crate::defaults::svd::{
        default_svd_truncation_policy, set_default_svd_truncation_policy, svd, svd_with,
        svd_with_in, SvdError, SvdOptions,
    };
}

// Re-export linear algebra items for top-level access
pub use defaults::direct_sum::direct_sum;
pub use defaults::factorize::{
    factorize, factorize_full_rank, factorize_full_rank_in, factorize_in,
};
pub use defaults::qr::{default_qr_rtol, qr, qr_with, set_default_qr_rtol, QrError, QrOptions};
pub use defaults::svd::{
    default_svd_truncation_policy, set_default_svd_truncation_policy, svd, svd_with, SvdError,
    SvdOptions,
};

// Global default and truncation utilities
pub use global_default::{GlobalDefault, InvalidRtolError};
pub use truncation::{
    validate_svd_truncation_options, DecompositionAlg, InvalidThresholdError, SingularValueMeasure,
    SvdTruncationOptionsError, SvdTruncationPolicy, ThresholdScale, TruncationRule,
};
