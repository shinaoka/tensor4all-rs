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
//! same or lower rank for no more function evaluations, and now converges in
//! a comparable number of sweeps (0.7x-1.3x `tensor4all-aci`'s sweep count on
//! `treeaci_parity`'s chain, bond dimension 16 through 128 -- down from
//! 1.7x-2.5x before `convergence_criterion` was changed to match
//! `tensor4all-aci`'s network-wide scalar max-rank stopping rule instead of
//! requiring every individual edge's rank to be simultaneously
//! non-increasing). It still remains slower end to end (a fresh run of
//! `treeaci_parity` measured roughly 1.5x-2.1x wall time on the chain across
//! bond dimension 16 through 256, with host-sensitive absolute timings), but
//! the gap is no longer chiefly a sweep-count effect -- it now reflects the
//! per-sweep cost described below.
//! `InputFrameStore::build_or_extend` used to rebuild every directed edge's
//! frame storage on every call: for an edge whose sample count had not
//! changed since the previous call, it eagerly copied every already-known
//! sample out of the previous store into a fresh `Matrix` and back out again
//! -- pure data movement with zero arithmetic, measured at 36.6% of total
//! wall time at chi=256. It now decides per edge: an edge whose sample count
//! is unchanged shares the previous store's `Rc<DirectedFrame<T>>`
//! allocation directly, with no copy at all, and an old sample that some
//! other (genuinely changed) edge's ancestor-priming recursion still needs
//! to read is pulled lazily, one row at a time, instead of being
//! pre-copied for every edge up front. Candidate/pivot-search frame
//! contraction, and sample materialization (`from_samples`/`extend`) for
//! single-incoming-edge nodes (which covers every node on a chain), route
//! through a BLAS matrix-multiply primitive instead of a
//! per-candidate/per-sample scalar loop when applicable.
//! `InputFrameStore::build_or_extend` retains edge-index order for accounting
//! and reuse decisions, but now materializes missing directed frames in
//! dependency order, so ancestor samples reach the batched path before a
//! dependent edge can prime them. Multi-incoming-edge nodes (genuine tree
//! branch points, not exercised by the chain benchmark) still use the
//! original per-candidate/per-sample scalar path throughout.
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
