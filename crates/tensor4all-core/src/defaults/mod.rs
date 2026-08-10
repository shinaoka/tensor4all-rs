//! Default concrete type implementations.
//!
//! This module provides the default concrete types for tensor network operations:
//!
//! - [`DynId`]: Runtime identity (UUID-based unique identifier)
//! - [`TagSet`]: Tag set for metadata (Arc-wrapped for cheap cloning)
//! - [`Index`]: Generic index type (`Index<Id, Tags>`)
//! - [`DynIndex`]: Default index type (`Index<DynId, TagSet>`)
//! - [`TensorDynLen`]: Dense tensor with dynamic rank
//!
//! Linear algebra operations:
//! - [`svd::svd`]: Singular Value Decomposition
//! - [`qr::qr`]: QR decomposition
//! - [`factorize::factorize`]: Unified factorization interface
//! - [`direct_sum::direct_sum`]: Direct sum of tensors
//!
//! These types are suitable for most tensor network applications and provide
//! a good balance of flexibility and performance.

pub mod index;
/// Dynamic-length tensor implementation.
pub mod tensordynlen;

// Contraction
pub mod contract;
pub(crate) mod structured_contraction;

// Linear algebra modules
pub mod direct_sum;
pub mod factorize;
pub mod qr;
pub mod svd;

pub use contract::{
    contract, contract_owned, contract_owned_with_options, contract_pair,
    contract_pair_with_options, contract_with_options, outer_product,
    print_and_reset_contract_profile, reset_contract_profile, tensordot, ContractionOptions,
};
pub use index::{DefaultIndex, DefaultTagSet, DynId, DynIndex, Index, TagSet};
pub use tensordynlen::{
    compute_permutation_from_indices, diag_tensor_dyn_len,
    print_and_reset_pairwise_contract_profile, reset_pairwise_contract_profile, unfold_split,
    TensorDynLen, TensorDynLenError, TensorStorageError,
};

// Re-export linear algebra functions and types
pub use direct_sum::direct_sum;
pub use factorize::{
    factorize, Canonical, FactorizeAlg, FactorizeError, FactorizeOptions, FactorizeResult,
};
pub use qr::{default_qr_rtol, qr, qr_with, set_default_qr_rtol, QrError, QrOptions};
pub use svd::{svd, svd_with, SvdError, SvdOptions};
