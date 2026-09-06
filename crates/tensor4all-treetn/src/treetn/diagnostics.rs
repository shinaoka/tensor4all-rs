//! Per-node cost attribution for TreeACI issues #671 and #732.
//!
//! Records, per tree node, Guard message-cache and TreeACI frame-cache
//! timing and hit/miss counts, plus the node's coordination number and
//! incident bond dimensions -- data needed to tell topology-necessary
//! `chi^z` cost apart from avoidable repeated work at a branch hub.
//!
//! One thread-local registry. Callers explicitly reset it at the start of a
//! diagnostics window and read it back via `snapshot()` at the end. It is not
//! cross-thread storage. Each row retains one operand/node/actual shape;
//! query times are inclusive and message times exclude recursive children.
//! Instrumentation overhead is included. Measure production wall time with the
//! feature disabled and report the observed diagnostic overhead separately.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::time::Duration;

mod cost;
pub use cost::{
    BatchDiagnostics, CacheDiagnostics, KernelDiagnostics, NodeShape, PhaseMeasurement,
};

/// Reads cumulative kernel counters for this thread only.
///
/// Use [`KernelDiagnostics::since`] to obtain a measurement delta.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::{kernel_snapshot, reset};
/// reset();
/// assert_eq!(kernel_snapshot().matmul_calls, 0);
/// ```
pub fn kernel_snapshot() -> KernelDiagnostics {
    KERNEL.with(|kernel| *kernel.borrow())
}

/// Adds kernel work to this thread's counters; unspecified fields are zero.
///
/// # Examples
/// ```
/// use tensor4all_treetn::diagnostics::{kernel_snapshot, record_kernel, reset, KernelDiagnostics};
/// reset();
/// record_kernel(KernelDiagnostics { matmul_calls: 2, ..Default::default() });
/// assert_eq!(kernel_snapshot().matmul_calls, 2);
/// ```
pub fn record_kernel(delta: KernelDiagnostics) {
    KERNEL.with(|kernel| {
        let mut kernel = kernel.borrow_mut();
        *kernel = kernel.plus(delta);
    });
}

pub(super) fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub(super) fn record_query(
    node: &str,
    shape: NodeShape,
    elapsed: Duration,
    points: usize,
    cache: CacheDiagnostics,
) {
    with_entry(node, shape, |entry| {
        entry.query_ns = entry.query_ns.saturating_add(nanos(elapsed));
        entry.query_batches.record(points as u64);
        entry.query_cache.record(cache);
    });
}

/// Per-node branch-point timing and cache statistics.
///
/// Unlike the aggregate contraction summary, these records distinguish actual
/// shapes and operands. [`NodeShape`] defines the tensor-size proxy.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::diagnostics::NodeDiagnostics;
///
/// let record = NodeDiagnostics::default();
/// assert_eq!(record.guard_cache_hits, 0);
/// assert!(record.bond_dims.is_empty());
/// ```
#[derive(Clone, Debug, Default)]
pub struct NodeDiagnostics {
    /// Operand namespace and node Debug label, e.g. `"input:0:3"` or `"output:3"`.
    pub node: String,
    /// Number of tree edges incident to this node.
    pub coordination_number: usize,
    /// Bond dimensions for each incident edge.
    ///
    /// Sorted multiset of actual incident dimensions; its length is the degree.
    pub bond_dims: Vec<usize>,
    /// Product of the physical dimensions at this node (one for no physical legs).
    pub physical_dim: usize,
    /// Checked product of incident bond dimensions; `None` indicates overflow.
    pub bond_product: Option<usize>,
    /// Checked `physical_dim * bond_product`, a tensor-size proxy, not a FLOP count.
    pub local_elements: Option<usize>,
    /// Inclusive complete query time attributed to its selected center.
    /// Do not add this to message or frame times.
    pub query_ns: u64,
    /// Full-point batches supplied by the caller, including duplicate points.
    pub query_batches: BatchDiagnostics,
    /// Whole-evaluator cache high-water observations; do not sum across nodes.
    pub query_cache: CacheDiagnostics,
    /// Unique component-assignment batches handled by Guard messages.
    pub guard_batches: BatchDiagnostics,
    /// Candidate sample batches handled by frame evaluation.
    pub frame_batches: BatchDiagnostics,
    /// Kernel work within this node's exclusive Guard message time.
    pub guard_kernel: KernelDiagnostics,
    /// Kernel work within this node's candidate-frame time.
    pub frame_kernel: KernelDiagnostics,
    /// Exclusive nanoseconds spent computing this node's Guard message: cache
    /// lookup plus any miss computation, excluding recursive child message work.
    ///
    /// This is NOT cache-lookup time alone -- a large value here can mean
    /// genuine `chi^z` contraction cost rather than cache overhead.
    pub guard_ns: u64,
    /// Total nanoseconds spent computing this node's TreeACI candidate
    /// frames: frame-cache lookup plus any miss computation, including the
    /// BLAS contraction work.
    ///
    /// This is NOT cache-lookup time alone -- a large value here can mean
    /// genuine `chi^z` contraction cost rather than cache overhead.
    pub frame_ns: u64,
    /// Number of Guard message-cache hits.
    pub guard_cache_hits: u64,
    /// Number of Guard message-cache misses.
    pub guard_cache_misses: u64,
    /// Number of TreeACI frame-cache hits.
    pub frame_cache_hits: u64,
    /// Number of TreeACI frame-cache misses.
    pub frame_cache_misses: u64,
}

thread_local! {
    // INVARIANT: opt-in observation storage, reset by the measurement owner;
    // one aggregate row per node and observed shape, never one per point.
    static REGISTRY: RefCell<BTreeMap<(String, NodeShape), NodeDiagnostics>> = const { RefCell::new(BTreeMap::new()) };
    static KERNEL: RefCell<KernelDiagnostics> = RefCell::new(KernelDiagnostics::default());
}

/// Clear the thread-local diagnostics registry.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use tensor4all_treetn::diagnostics::{record_guard, reset, snapshot, NodeShape, PhaseMeasurement};
///
/// record_guard("hub", NodeShape { physical_dim: 2, bond_dims: vec![2] },
///     PhaseMeasurement { elapsed: Duration::from_nanos(1), hits: 1, misses: 0, ..Default::default() });
/// assert_eq!(snapshot().len(), 1);
/// reset();
/// assert!(snapshot().is_empty());
/// ```
pub fn reset() {
    REGISTRY.with(|registry| registry.borrow_mut().clear());
    KERNEL.with(|kernel| *kernel.borrow_mut() = KernelDiagnostics::default());
}

/// Read back the current accumulated diagnostics as a snapshot.
///
/// Records are sorted by node label, physical dimension and sorted incident
/// dimensions. Shape changes produce distinct rows. Snapshot does not reset.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use tensor4all_treetn::diagnostics::{record_guard, reset, snapshot, NodeShape, PhaseMeasurement};
///
/// reset();
/// record_guard("hub", NodeShape { physical_dim: 2, bond_dims: vec![3, 4] },
///     PhaseMeasurement { elapsed: Duration::from_nanos(5), hits: 2, misses: 1, ..Default::default() });
/// let records = snapshot();
/// assert_eq!(records.len(), 1);
/// assert_eq!(records[0].node, "hub");
/// assert_eq!(records[0].guard_cache_hits, 2);
/// ```
pub fn snapshot() -> Vec<NodeDiagnostics> {
    REGISTRY.with(|registry| registry.borrow().values().cloned().collect())
}

fn with_entry(node: &str, mut shape: NodeShape, f: impl FnOnce(&mut NodeDiagnostics)) {
    shape.bond_dims.sort_unstable();
    REGISTRY.with(|registry| {
        let mut map = registry.borrow_mut();
        let entry = map
            .entry((node.to_owned(), shape.clone()))
            .or_insert_with(|| NodeDiagnostics {
                node: node.to_string(),
                coordination_number: shape.bond_dims.len(),
                physical_dim: shape.physical_dim,
                bond_product: shape.bond_product(),
                local_elements: shape.local_elements(),
                bond_dims: shape.bond_dims,
                ..Default::default()
            });
        f(entry);
    });
}

/// Record a Guard message computation and its cache hit/miss counts.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use tensor4all_treetn::diagnostics::{record_guard, reset, snapshot, NodeShape, PhaseMeasurement};
///
/// reset();
/// record_guard("hub", NodeShape { physical_dim: 2, bond_dims: vec![2, 2, 2] },
///     PhaseMeasurement { elapsed: Duration::from_nanos(10), hits: 4, misses: 1, ..Default::default() });
/// let record = &snapshot()[0];
/// assert_eq!(record.coordination_number, 3);
/// assert_eq!(record.guard_cache_hits, 4);
/// assert_eq!(record.guard_cache_misses, 1);
/// ```
///
///
/// `node` must include an operand namespace. `shape` preserves the actual
/// physical and bond dimensions, while `sample` excludes recursive child work.
pub fn record_guard(node: &str, shape: NodeShape, sample: PhaseMeasurement) {
    with_entry(node, shape, |entry| {
        entry.guard_ns = entry.guard_ns.saturating_add(nanos(sample.elapsed));
        entry.guard_cache_hits = entry.guard_cache_hits.saturating_add(sample.hits);
        entry.guard_cache_misses = entry.guard_cache_misses.saturating_add(sample.misses);
        entry
            .guard_batches
            .record(sample.hits.saturating_add(sample.misses));
        entry.guard_kernel = entry.guard_kernel.plus(sample.kernel);
    });
}

/// Record a TreeACI candidate-frame computation and its cache hit/miss
/// counts. The `node` key is namespaced by the operand index (`input`).
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use tensor4all_treetn::diagnostics::{record_frame, reset, snapshot, NodeShape, PhaseMeasurement};
///
/// reset();
/// record_frame("input:0:hub", NodeShape { physical_dim: 2, bond_dims: vec![2, 2, 2] },
///     PhaseMeasurement { elapsed: Duration::from_nanos(20), hits: 5, misses: 2, ..Default::default() });
/// let record = &snapshot()[0];
/// assert_eq!(record.node, "input:0:hub");
/// assert_eq!(record.frame_cache_hits, 5);
/// assert_eq!(record.frame_cache_misses, 2);
/// ```
pub fn record_frame(node: &str, shape: NodeShape, sample: PhaseMeasurement) {
    with_entry(node, shape, |entry| {
        entry.frame_ns = entry.frame_ns.saturating_add(nanos(sample.elapsed));
        entry.frame_cache_hits = entry.frame_cache_hits.saturating_add(sample.hits);
        entry.frame_cache_misses = entry.frame_cache_misses.saturating_add(sample.misses);
        entry
            .frame_batches
            .record(sample.hits.saturating_add(sample.misses));
        entry.frame_kernel = entry.frame_kernel.plus(sample.kernel);
    });
}

/// Reset the branch-vs-chain contraction-kernel counters (issue #671's
/// scratch investigation into where the branch kernel's per-call BLAS setup
/// cost goes; see `crate::treetn::cached_evaluator::contraction_diagnostics`).
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::diagnostics::{contraction_summary, reset_contraction};
///
/// reset_contraction();
/// assert!(contraction_summary().starts_with("branch:"));
/// ```
pub fn reset_contraction() {
    super::cached_evaluator::contraction_diagnostics::reset_all();
}

/// One human-readable line summarizing the branch-vs-chain contraction
/// counters accumulated since the last [`reset_contraction`].
/// The `blas_calls` field counts backend matrix-multiply dispatches; the
/// branch `blas_groups` field counts physical-value groups processed by the
/// branch kernel.
///
/// # Examples
///
/// ```
/// use tensor4all_treetn::diagnostics::{contraction_summary, reset_contraction};
///
/// reset_contraction();
/// let summary = contraction_summary();
/// assert!(summary.contains("branch: blas_calls=0"));
/// assert!(summary.contains("chain: blas_calls=0"));
/// ```
pub fn contraction_summary() -> String {
    super::cached_evaluator::contraction_diagnostics::summary()
}

#[cfg(test)]
mod tests;
