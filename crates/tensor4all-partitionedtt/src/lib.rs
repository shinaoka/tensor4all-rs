//! Partitioned Tensor Train for tensor4all.
//!
//! **Deprecated:** new TreeTN-based work should use
//! [`tensor4all-partitionedtreetn`](https://docs.rs/tensor4all-partitionedtreetn).
//! Both crates coexist during migration. This crate remains buildable and
//! accepts correctness and security fixes only; no removal date has been set,
//! and removal requires a separate maintainer decision. This documentation-only
//! deprecation intentionally does not emit a compiler warning.
//!
//! This crate provides partitioned tensor train functionality for representing
//! functions over subdomains with non-overlapping projectors.
//!
//! # Main Types
//!
//! - [`Projector`]: Maps tensor indices (DynIndex) to fixed values, defining subdomains
//! - [`SubDomainTT`]: A tensor train restricted to a specific subdomain
//! - [`PartitionedTT`]: A collection of non-overlapping SubDomainTTs

mod adaptive_interpolation;
mod contract;
mod error;
mod partitioned_tt;
mod patching;
mod projector;
mod subdomain_tt;

#[cfg(test)]
mod test_utils;

#[cfg(feature = "adaptive-hataori-rayon")]
pub use adaptive_interpolation::adaptiveinterpolate_in;
#[cfg(feature = "adaptive-hataori-mpi")]
pub use adaptive_interpolation::adaptiveinterpolate_mpi;
pub use adaptive_interpolation::{
    adaptiveinterpolate, AcceptedPatchCache, AdaptiveInterpolateOptions,
    AdaptiveInterpolationResult,
};
pub use contract::{contract, proj_contract};
pub use error::{PartitionedTTError, Result};
pub use partitioned_tt::PartitionedTT;
pub use patching::{
    add_with_patching, contract_adaptive, truncate_adaptive, PatchSplitStrategy, PatchingOptions,
};
pub use projector::Projector;
pub use subdomain_tt::SubDomainTT;

// Re-export commonly used types from dependencies
pub use tensor4all_core::MultiIndex;
pub use tensor4all_core::{DynIndex, IdxTensor};
pub use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};
pub use tensor4all_tensorci::TCI2Options;
