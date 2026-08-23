//! Alternating Cross Interpolation for tree tensor networks.
//!
//! This crate operates directly on [`tensor4all_treetn::TreeTN`] and keeps the
//! tree topology throughout interpolation. Unlike `tensor4all-aci`, it imposes
//! no path-order requirement: any validated tree topology is accepted, and no
//! `TensorTrain` conversion happens at any point.
//!
//! # Maturity
//!
//! **This is not yet a drop-in replacement for `tensor4all-aci`.** Chain
//! parity tests cover accuracy, rank, and function-evaluation behavior, but
//! end-to-end timing remains host- and workload-sensitive and should be
//! measured for the intended problem.
//! `InputFrameStore::build_or_extend` used to rebuild every directed edge's
//! frame storage on every call: for an edge whose sample count had not
//! changed since the previous call, it eagerly copied every already-known
//! sample out of the previous store into a fresh matrix and back out again
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
//! dependent edge can prime them. Nodes with exactly two incoming edges use a
//! batched branch contraction; leaf nodes and nodes with three or more incoming
//! edges retain the scalar fallback.
//! Guard input evaluators persist across scans, reuse directed messages, and
//! share a bounded aggregate message-cache budget with each scan's output
//! evaluator.
//!
//! `tensor4all-aci` remains the narrower chain-specific API. Chain performance
//! between the two implementations is workload-dependent, so use the paired
//! benchmark when that distinction matters. This crate is the native choice
//! for genuinely branched trees.

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
