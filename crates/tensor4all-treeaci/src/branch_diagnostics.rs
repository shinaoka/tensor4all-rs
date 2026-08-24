//! Re-export of `tensor4all-treetn`'s per-node branch diagnostics registry.
//!
//! Named `branch_diagnostics`, not `diagnostics`, to stay distinct from
//! [`crate::TreeAciDiagnostics`] -- an unrelated, already-public sweep/
//! convergence report. This module is about branch-point performance
//! (issue #671); `TreeAciDiagnostics` is about ACI convergence history.

pub use tensor4all_treetn::diagnostics::{
    record_frame, record_guard, reset, snapshot, NodeDiagnostics,
};
