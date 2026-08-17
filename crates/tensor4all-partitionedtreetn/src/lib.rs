#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

//! Partitioned Tree Tensor Network subdomains for tensor4all.
//!
//! This crate provides [`Projector`], [`SubDomainTreeTN`], and
//! [`PartitionedTreeTN`] patch algebra, including TreeTN-general adaptive
//! patching. Stored patch data is eagerly masked, and partition metadata is
//! validated transactionally.
//!
//! The representation follows the partitioned tensor-network approach used by
//! [PartitionedMPSs.jl](https://github.com/tensor4all/PartitionedMPSs.jl) and
//! the adaptive patching literature; this crate does not contain TCI-derived
//! adaptive interpolation code.

mod error;
mod partitioned_tree_tn;
mod patching;
mod projector;
mod subdomain_tree_tn;

pub use error::{PartitionedTreeTNError, Result};
pub use partitioned_tree_tn::PartitionedTreeTN;
pub use patching::{
    add_with_patching, contract_adaptive, truncate_adaptive, PatchSplitStrategy, PatchingOptions,
};
pub use projector::Projector;
pub use subdomain_tree_tn::SubDomainTreeTN;

pub use tensor4all_core::{DynIndex, IdxTensor};
pub use tensor4all_treetn::{SiteIndexNetwork, TreeTN};
