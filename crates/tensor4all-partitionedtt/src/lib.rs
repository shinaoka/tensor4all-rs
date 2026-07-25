//! Partitioned Tensor Train for tensor4all
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

pub use adaptive_interpolation::{adaptiveinterpolate, AdaptiveInterpolateOptions};
pub use contract::{contract, proj_contract};
pub use error::{PartitionedTTError, Result};
pub use partitioned_tt::PartitionedTT;
pub use patching::{
    add_with_patching, contract_adaptive, truncate_adaptive, PatchSplitStrategy, PatchingOptions,
};
pub use projector::Projector;
pub use subdomain_tt::SubDomainTT;

// Re-export commonly used types from dependencies
pub use tensor4all_core::{DynIndex, TensorDynLen};
pub use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};
pub use tensor4all_tcicore::MultiIndex;
pub use tensor4all_tensorci::TCI2Options;
