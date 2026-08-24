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

#[derive(Clone, Debug, Default)]
pub struct NodeDiagnostics {
    pub node: String,
    pub coordination_number: usize,
    pub bond_dims: Vec<usize>,
    pub guard_ns: u64,
    pub frame_ns: u64,
    pub guard_cache_hits: u64,
    pub guard_cache_misses: u64,
    pub frame_cache_hits: u64,
    pub frame_cache_misses: u64,
}

thread_local! {
    static REGISTRY: RefCell<HashMap<String, NodeDiagnostics>> = RefCell::new(HashMap::new());
}

pub fn reset() {
    REGISTRY.with(|registry| registry.borrow_mut().clear());
}

pub fn snapshot() -> Vec<NodeDiagnostics> {
    REGISTRY.with(|registry| registry.borrow().values().cloned().collect())
}

fn with_entry(node: &str, coordination_number: usize, bond_dims: &[usize], f: impl FnOnce(&mut NodeDiagnostics)) {
    REGISTRY.with(|registry| {
        let mut map = registry.borrow_mut();
        let entry = map.entry(node.to_string()).or_insert_with(|| NodeDiagnostics {
            node: node.to_string(),
            ..Default::default()
        });
        entry.coordination_number = coordination_number;
        entry.bond_dims = bond_dims.to_vec();
        f(entry);
    });
}

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
