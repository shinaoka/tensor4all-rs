//! Alternating Cross Interpolation for tree tensor networks.
//!
//! This crate operates directly on [`tensor4all_treetn::TreeTN`] and keeps the
//! tree topology throughout interpolation. Unlike `tensor4all-aci`, it imposes
//! no path-order requirement: any validated tree topology is accepted, and no
//! `TensorTrain` conversion happens at any point.
//!
//! # Maturity
//!
//! **This is not yet a drop-in replacement for `tensor4all-aci`.** On a chain,
//! where the two are directly comparable, it reaches the same accuracy at the
//! same or lower rank for no more function evaluations, but remains several
//! times slower end to end (roughly 5x-17x on `tensor4all-aci`'s
//! `treeaci_parity` benchmark, chain, bond dimension 16 through 128). The
//! remaining gap is concentrated in candidate/pivot-search frame construction,
//! which does not yet route through a BLAS matrix-multiply primitive the way
//! `tensor4all-aci`'s cache does.
//!
//! Prefer `tensor4all-aci` for chain topologies. Use this crate when the
//! topology is genuinely a tree, where the alternative is not a slower run but
//! no run at all.

mod batch;
mod elementwise;
mod error;
mod frames;
mod global_guard;
mod hadamard;
mod initialize;
mod local_update;
mod options;
mod path_cover;
pub mod prelude;
mod problem;
mod result;
mod samples;
mod scalar;
mod schedule;
mod single_site;
mod state;
mod transaction;
mod traversal;

#[cfg(test)]
mod order_experiment;
#[cfg(test)]
mod skeleton;
#[cfg(test)]
mod validate;

pub use batch::TreeElementwiseBatch;
pub use elementwise::{tree_elementwise, tree_elementwise_batched};
pub use error::{Result, TreeAciError};
pub use hadamard::hadamard_many;
pub use options::TreeAciOptions;
pub use result::{TreeAciDiagnostics, TreeAciResult, TreeAciTermination};
pub use scalar::{TreeAciNode, TreeAciScalar};
pub use traversal::TreeAciTraversalStrategy;
