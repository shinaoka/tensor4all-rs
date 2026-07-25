//! Core tensor operations and types for tensor4all-rs.
//!
//! This crate provides the foundational types and operations for tensor networks:
//!
//! - **Index types**: [`DynIndex`], [`Index`], [`DynId`] for tensor indices
//! - **Tag sets**: [`TagSet`], [`TagSetLike`] for metadata tagging
//! - **Tensors**: [`TensorDynLen`] for dynamic-rank dense tensors
//! - **Operations**: Contraction, SVD, QR decomposition, factorization
//!
//! # Example
//!
//! ```
//! use tensor4all_core::{Index, DynIndex, TensorDynLen};
//!
//! // Create indices with dynamic identity
//! let i = Index::new_dyn(2);
//! let j = Index::new_dyn(3);
//!
//! // Create a tensor
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data).unwrap();
//! ```

#![warn(missing_docs)]

pub mod col_major_array;
pub use col_major_array::{ColMajorArray, ColMajorArrayMut, ColMajorArrayRef};

// Common (tags, utilities, scalar)
/// Dynamic scalar compatibility wrapper built on rank-0 `TensorDynLen`.
pub mod any_scalar;
pub mod global_default;
pub mod index_like;
pub mod scalar;
/// Stack-allocated fixed-capacity string types for ITensors.jl compatibility.
pub mod smallstring;
/// Tag set types for tensor metadata.
pub mod tagset;
pub mod truncation;

pub use scalar::CommonScalar;

// Default concrete type implementations (index, tensor, linalg, etc.)
pub mod defaults;

// Backwards compatibility: re-export defaults submodules as top-level modules
// This allows `tensor4all_core::index::...` to work
pub use defaults::index;

pub use defaults::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, TagSet};
pub use index_like::{ConjState, IndexLike};

/// Index operations (replacement, set operations, contraction preparation).
pub mod index_ops;
pub use index_ops::{
    check_unique_indices, common_ind_positions, common_inds, hascommoninds, hasind, hasinds,
    noncommon_inds, replaceinds, replaceinds_in_place, union_inds, unique_inds, ReplaceIndsError,
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

// Backwards compatibility: re-export defaults::tensordynlen as tensor
pub use defaults::tensordynlen as tensor;

pub use any_scalar::AnyScalar;
pub use defaults::tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len, unfold_split, StructuredSelectorError,
    TensorDynLen,
};
pub use tensor4all_tensorbackend::TensorElement;
pub use tensor4all_tensorbackend::{
    print_and_reset_native_einsum_profile, reset_native_einsum_profile,
};
pub use tensor_like::{
    Canonical, DirectSumResult, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
    LinearizationOrder, TensorConstructionLike, TensorContractionLike, TensorFactorizationLike,
    TensorLike, TensorVectorSpace,
};

pub use defaults::contract::{
    contract, contract_owned, contract_owned_with_options, contract_pair,
    contract_pair_with_operand_options, contract_pair_with_options, contract_with_options,
    outer_product, print_and_reset_contract_profile, reset_contract_profile, tensordot,
    ContractionOptions, PairwiseContractionOptions,
};
pub use defaults::tensordynlen::{
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
        default_svd_truncation_policy, set_default_svd_truncation_policy, svd, svd_with, SvdError,
        SvdOptions,
    };
}

// Re-export linear algebra items for top-level access
pub use defaults::direct_sum::direct_sum;
pub use defaults::factorize::{factorize, factorize_full_rank};
pub use defaults::qr::{default_qr_rtol, qr, qr_with, set_default_qr_rtol, QrError, QrOptions};
pub use defaults::svd::{
    default_svd_truncation_policy, set_default_svd_truncation_policy, svd, svd_with, SvdError,
    SvdOptions,
};

// Global default and truncation utilities
pub use global_default::{GlobalDefault, InvalidRtolError};
pub use truncation::{
    DecompositionAlg, InvalidThresholdError, SingularValueMeasure, SvdTruncationPolicy,
    ThresholdScale, TruncationRule,
};
