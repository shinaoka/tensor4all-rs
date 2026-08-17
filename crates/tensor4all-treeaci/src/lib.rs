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
//! same or lower rank for no more function evaluations -- but it currently
//! costs roughly two orders of magnitude more per evaluation, because the
//! per-evaluation caching that `tensor4all-aci` has (a persistent cache reused
//! across sweeps, and the contraction split placed at the varying site) has no
//! counterpart here yet. The gap is a constant factor, not a difference in
//! scaling.
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
