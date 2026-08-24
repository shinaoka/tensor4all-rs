//! Per-node branch-point diagnostics for TreeACI issue #671's investigation.
//!
//! Records, per tree node, Guard message-cache and TreeACI frame-cache
//! timing and hit/miss counts, plus the node's coordination number and
//! incident bond dimensions -- data needed to tell topology-necessary
//! `chi^z` cost apart from avoidable repeated work at a branch hub. See
//! `docs/superpowers/specs/2026-08-24-treeaci-branch-diagnostics-design.md`.
//!
//! One thread-local registry, reset at the start of a call and read back via
//! `snapshot()` at the end. Not a cross-call or cross-thread store.

use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Duration;

/// Per-node branch-point timing and cache statistics.
#[derive(Clone, Debug, Default)]
pub struct NodeDiagnostics {
    /// The registry key for this node. TreeACI's frame records use
    /// `"{input}:{node:?}"`, namespacing the node by its operand index;
    /// Guard records use the bare `"{node:?}"` (see [`record_guard`]).
    pub node: String,
    /// Number of tree edges incident to this node.
    pub coordination_number: usize,
    /// Bond dimensions for each incident edge.
    ///
    /// Always has exactly `coordination_number` entries; an edge whose bond
    /// dimension could not be looked up contributes a `0` sentinel rather
    /// than being skipped. The order of the entries is not guaranteed to be
    /// stable across calls or between the Guard and frame recording sites,
    /// so treat this as a multiset of incident bond dimensions.
    pub bond_dims: Vec<usize>,
    /// Total nanoseconds spent computing this node's Guard message: cache
    /// lookup plus any miss computation.
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
    static REGISTRY: RefCell<HashMap<String, NodeDiagnostics>> = RefCell::new(HashMap::new());
}

/// Clear the thread-local diagnostics registry.
pub fn reset() {
    REGISTRY.with(|registry| registry.borrow_mut().clear());
}

/// Read back the current accumulated diagnostics as a snapshot.
///
/// The returned `Vec`'s order is not guaranteed to be stable across calls:
/// it reflects `HashMap` iteration order. A caller that needs deterministic
/// output should sort the result (for example by `node`).
pub fn snapshot() -> Vec<NodeDiagnostics> {
    REGISTRY.with(|registry| registry.borrow().values().cloned().collect())
}

fn with_entry(
    node: &str,
    coordination_number: usize,
    bond_dims: &[usize],
    f: impl FnOnce(&mut NodeDiagnostics),
) {
    REGISTRY.with(|registry| {
        let mut map = registry.borrow_mut();
        let entry = map
            .entry(node.to_string())
            .or_insert_with(|| NodeDiagnostics {
                node: node.to_string(),
                ..Default::default()
            });
        entry.coordination_number = coordination_number;
        entry.bond_dims = bond_dims.to_vec();
        f(entry);
    });
}

/// Record a Guard message computation and its cache hit/miss counts.
///
/// # Known limitation
///
/// The `node` key carries no per-operand namespace: it is just the tree
/// node's `Debug` label. If a caller's diagnostics window spans more than
/// one distinct tree -- as `TreeAciProduct::combine` does, since it builds
/// one `TreeTNCachedEvaluator` per input tree -- colliding node labels
/// across those trees merge into a single registry entry. TreeACI's
/// frame-side key does not have this problem, because the operand index
/// `input` is available there and is included in the key. Fixing this on
/// the Guard side would require threading an operand identity through
/// `TreeTNCachedEvaluator`'s constructor.
pub fn record_guard(
    node: &str,
    coordination_number: usize,
    bond_dims: &[usize],
    elapsed: Duration,
    hits: u64,
    misses: u64,
) {
    with_entry(node, coordination_number, bond_dims, |entry| {
        entry.guard_ns += elapsed.as_nanos() as u64;
        entry.guard_cache_hits += hits;
        entry.guard_cache_misses += misses;
    });
}

/// Record a TreeACI candidate-frame computation and its cache hit/miss
/// counts. The `node` key is namespaced by the operand index (`input`).
pub fn record_frame(
    node: &str,
    coordination_number: usize,
    bond_dims: &[usize],
    elapsed: Duration,
    hits: u64,
    misses: u64,
) {
    with_entry(node, coordination_number, bond_dims, |entry| {
        entry.frame_ns += elapsed.as_nanos() as u64;
        entry.frame_cache_hits += hits;
        entry.frame_cache_misses += misses;
    });
}

/// Reset the branch-vs-chain contraction-kernel counters (issue #671's
/// scratch investigation into where the branch kernel's per-call BLAS setup
/// cost goes; see `crate::treetn::cached_evaluator::contraction_diagnostics`).
pub fn reset_contraction() {
    super::cached_evaluator::contraction_diagnostics::reset_all();
}

/// One human-readable line summarizing the branch-vs-chain contraction
/// counters accumulated since the last [`reset_contraction`].
pub fn contraction_summary() -> String {
    super::cached_evaluator::contraction_diagnostics::summary()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_then_snapshot_is_empty() {
        record_guard("a", 2, &[3, 3], Duration::from_nanos(10), 1, 0);
        reset();
        assert!(snapshot().is_empty());
    }

    #[test]
    fn record_guard_and_record_frame_accumulate_into_one_entry_per_node() {
        reset();
        record_guard("hub", 3, &[4, 4, 4], Duration::from_nanos(100), 2, 1);
        record_guard("hub", 3, &[4, 4, 4], Duration::from_nanos(50), 0, 1);
        record_frame("hub", 3, &[4, 4, 4], Duration::from_nanos(30), 5, 2);

        let snap = snapshot();
        assert_eq!(snap.len(), 1);
        let hub = &snap[0];
        assert_eq!(hub.node, "hub");
        assert_eq!(hub.coordination_number, 3);
        assert_eq!(hub.bond_dims, vec![4, 4, 4]);
        assert_eq!(hub.guard_ns, 150);
        assert_eq!(hub.guard_cache_hits, 2);
        assert_eq!(hub.guard_cache_misses, 2);
        assert_eq!(hub.frame_ns, 30);
        assert_eq!(hub.frame_cache_hits, 5);
        assert_eq!(hub.frame_cache_misses, 2);
    }

    #[test]
    fn distinct_nodes_get_distinct_entries() {
        reset();
        record_guard("a", 2, &[3, 3], Duration::from_nanos(10), 1, 0);
        record_guard("b", 3, &[5, 5, 5], Duration::from_nanos(20), 1, 0);
        let mut nodes: Vec<String> = snapshot().into_iter().map(|d| d.node).collect();
        nodes.sort();
        assert_eq!(nodes, vec!["a".to_string(), "b".to_string()]);
    }
}
