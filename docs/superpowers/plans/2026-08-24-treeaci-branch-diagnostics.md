# TreeACI Branch-Point Diagnostics Implementation Plan

Historical implementation record. Issue #732 supersedes the diagnostic API,
node keys and timing semantics below; see [the current design](../../design/tree-aci.md)
and [the release experiment](../../../benchmarks/README.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `diagnostics` Cargo feature to `tensor4all-treetn` and `tensor4all-treeaci` that reports, per tree node, coordination number, incident bond dimensions, Guard vs. LUCI/frame timing, and cache hit/miss counts -- with zero generated code when the feature is off.

**Architecture:** A single thread-local registry (`NodeDiagnostics` records keyed by a node's `Debug` string) lives in `tensor4all-treetn::treetn::diagnostics`, gated by `#[cfg(feature = "diagnostics")]`. Two existing call sites record into it: `cached_evaluator.rs`'s `get_or_compute_node_message` (Guard's per-node message cache) and `frames.rs`'s three `candidate_cache` lookup sites (TreeACI's per-node frame cache). `tensor4all-treeaci` forwards the feature and re-exports the registry's public functions under `branch_diagnostics` (not `diagnostics`, to avoid reading as the same thing as the existing, unrelated `TreeAciDiagnostics` sweep/convergence struct already exported from `result.rs`).

**Tech Stack:** Rust, `std::cell::RefCell` + `thread_local!` (no new dependencies -- the workspace has no `serde` derive dependency today; `NodeDiagnostics` stays a plain struct and downstream crates serialize it themselves).

**Spec:** `docs/superpowers/specs/2026-08-24-treeaci-branch-diagnostics-design.md`

## Global Constraints

- Zero generated code when `diagnostics` is disabled (the default): every new item is behind `#[cfg(feature = "diagnostics")]`, and no existing public function signature changes.
- Do not change LUCI pivot selection, candidate enumeration, truncation tolerances, Guard's search policy, or any sampled value. This is observability only.
- Do not add a persistent/cross-call diagnostics store; `reset()` clears everything and each call's data is read via one `snapshot()`.
- Do not add `serde` or any other new dependency to `tensor4all-treetn`/`tensor4all-treeaci`.
- `cargo fmt --all` before every commit (repo-wide convention); `cargo clippy --workspace --all-targets -- -D warnings` must stay clean for changed crates.

---

## File Structure

- `crates/tensor4all-treetn/Cargo.toml` -- new `diagnostics` feature (no new deps).
- `crates/tensor4all-treetn/src/treetn/diagnostics.rs` -- new module: `NodeDiagnostics`, thread-local registry, `reset`/`snapshot`/`record_guard`/`record_frame`.
- `crates/tensor4all-treetn/src/treetn/mod.rs` -- `#[cfg(feature = "diagnostics")] pub mod diagnostics;`
- `crates/tensor4all-treetn/src/lib.rs` -- `#[cfg(feature = "diagnostics")] pub use treetn::diagnostics;`
- `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` -- feature-gated instrumentation inside `get_or_compute_node_message`.
- `crates/tensor4all-treeaci/Cargo.toml` -- new `diagnostics` feature, `= ["tensor4all-treetn/diagnostics"]`.
- `crates/tensor4all-treeaci/src/lib.rs` -- `#[cfg(feature = "diagnostics")] pub mod branch_diagnostics;` (thin re-export module).
- `crates/tensor4all-treeaci/src/branch_diagnostics.rs` -- new thin module: `pub use tensor4all_treetn::diagnostics::*;`.
- `crates/tensor4all-treeaci/src/frames.rs` -- new private `diagnostics_node_topology` helper; feature-gated instrumentation at the three `candidate_cache` lookup sites.

---

### Task 1: Diagnostics data model and registry (`tensor4all-treetn`)

**Files:**
- Create: `crates/tensor4all-treetn/src/treetn/diagnostics.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/mod.rs`
- Modify: `crates/tensor4all-treetn/src/lib.rs`
- Modify: `crates/tensor4all-treetn/Cargo.toml`
- Test: `crates/tensor4all-treetn/src/treetn/diagnostics.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces (used by Task 2 and by `tensor4all-treeaci`):
  ```rust
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
  pub fn reset();
  pub fn snapshot() -> Vec<NodeDiagnostics>;
  pub fn record_guard(node: &str, coordination_number: usize, bond_dims: &[usize], elapsed: std::time::Duration, hits: u64, misses: u64);
  pub fn record_frame(node: &str, coordination_number: usize, bond_dims: &[usize], elapsed: std::time::Duration, hits: u64, misses: u64);
  ```

- [ ] **Step 1: Add the `diagnostics` feature to `tensor4all-treetn/Cargo.toml`**

Open `crates/tensor4all-treetn/Cargo.toml` and add, inside the existing `[features]` table:

```toml
diagnostics = []
```

- [ ] **Step 2: Write the failing test**

Create `crates/tensor4all-treetn/src/treetn/diagnostics.rs`:

```rust
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
```

This module is not yet registered anywhere, so it does not compile as part of
the crate yet -- that is expected for this step.

- [ ] **Step 3: Wire the module in and verify the test fails to even build without the feature, then passes with it**

In `crates/tensor4all-treetn/src/treetn/mod.rs`, add near the other `mod`/`pub mod` declarations (alongside `pub mod contraction;`):

```rust
#[cfg(feature = "diagnostics")]
pub mod diagnostics;
```

In `crates/tensor4all-treetn/src/lib.rs`, add next to the existing `#[cfg(feature = "simplett-bridge")]` re-export block:

```rust
#[cfg(feature = "diagnostics")]
pub use treetn::diagnostics;
```

Run:

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --features diagnostics treetn::diagnostics -- --nocapture
```

Expected: the three tests in `diagnostics::tests` run and pass. If they do
not compile, fix the module (this is the first real compile of the file).

- [ ] **Step 4: Verify the default build is untouched**

```bash
cargo check --manifest-path crates/tensor4all-treetn/Cargo.toml
```

Expected: succeeds, and `diagnostics` does not appear as a symbol (the
feature is off by default, so the module is not compiled in).

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetn/Cargo.toml crates/tensor4all-treetn/src/treetn/mod.rs crates/tensor4all-treetn/src/treetn/diagnostics.rs crates/tensor4all-treetn/src/lib.rs
git commit -m "feat(treetn): add diagnostics feature with per-node registry"
```

---

### Task 2: Wire Guard's message cache into the registry (`tensor4all-treetn`)

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:2426-2760` (`get_or_compute_node_message`)
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` (new `#[cfg(all(test, feature = "diagnostics"))]` test in the existing test module)

**Interfaces:**
- Consumes: `tensor4all_treetn::diagnostics::record_guard` from Task 1 (same crate, so `super::diagnostics::record_guard` or `crate::treetn::diagnostics::record_guard`).
- Produces: nothing new for later tasks (Task 3 is independent, in a different crate).

Read `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs:2426-2760`
(`get_or_compute_node_message`) before starting: it has two return points --
an early return on a full cache hit (~line 2507-2540) and the normal return
after a partial/full miss is computed and merged (ends ~line 2757-2760). Both
need a `record_guard` call. `V: Debug` is already required in this file (see
existing `{:?}` uses in error messages), so `format!("{node:?}")` is valid
wherever `node: &V` is in scope.

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` block in
`crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`, next to
`degree_three_hub_keeps_raw_messages_when_center_is_a_leaf` (line ~5185).
That existing test already shows the exact fixture and API this new test
reuses verbatim: `star_tree()` (line ~5597) builds a 4-node tree where node
`0` is a degree-3 hub connected to leaves `1`, `2`, `3`; centering the
evaluator at leaf `1` and calling `build_environment_cache(&1, points)`
computes messages for every node except the center itself -- so this test's
snapshot will contain records for `0`, `2`, `3`, but not `1`.

```rust
#[cfg(feature = "diagnostics")]
#[test]
fn build_environment_cache_records_guard_diagnostics_per_node_with_correct_coordination_numbers() {
    use crate::treetn::diagnostics;

    let (tree, indices) = star_tree();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let shape = [4usize, 2usize];
    let values = [0usize, 0, 0, 0, 1, 0, 1, 1];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    diagnostics::reset();
    let _ = evaluator.build_environment_cache(&1, points).unwrap();

    let snapshot = diagnostics::snapshot();
    let hub_record = snapshot
        .iter()
        .find(|record| record.node == "0")
        .expect("hub node (0) recorded");
    assert_eq!(hub_record.coordination_number, 3);
    assert_eq!(hub_record.bond_dims.len(), 3);
    assert!(hub_record.guard_cache_hits + hub_record.guard_cache_misses > 0);

    for leaf in ["2", "3"] {
        let leaf_record = snapshot
            .iter()
            .find(|record| record.node == leaf)
            .unwrap_or_else(|| panic!("leaf node ({leaf}) recorded"));
        assert_eq!(leaf_record.coordination_number, 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --features diagnostics \
  build_environment_cache_records_guard_diagnostics_per_node_with_correct_coordination_numbers -- --nocapture
```

Expected: FAIL (`hub node (0) recorded` panics -- nothing has been recorded
yet).

- [ ] **Step 3: Instrument `get_or_compute_node_message`**

At the top of the function, right after the existing `#[cfg(test)] let
phase_start = std::time::Instant::now();` (line ~2441-2442), add:

```rust
#[cfg(feature = "diagnostics")]
let diag_start = std::time::Instant::now();
#[cfg(feature = "diagnostics")]
let (diag_coordination_number, diag_bond_dims) = {
    let neighbors: Vec<V> = self.tree.site_index_network().neighbors(node).collect();
    let bond_dims = neighbors
        .iter()
        .filter_map(|neighbor| {
            self.tree
                .edge_between(node, neighbor)
                .and_then(|edge| self.tree.bond_index(edge))
                .map(|index| index.dim())
        })
        .collect::<Vec<_>>();
    (neighbors.len(), bond_dims)
};
```

At the full-hit early return (the `if let Some(positions) =
cache.get_all_cached(&keys) { ... }` block, whose body currently ends with
`return Ok(StackedMessage { ... });` around line 2535-2539), add just before
that `return`:

```rust
#[cfg(feature = "diagnostics")]
diagnostics::record_guard(
    &format!("{node:?}"),
    diag_coordination_number,
    &diag_bond_dims,
    diag_start.elapsed(),
    keys.len() as u64,
    0,
);
```

and add `use super::diagnostics;` near the top of the file, gated:

```rust
#[cfg(feature = "diagnostics")]
use super::diagnostics;
```

For the miss/partial-hit path: right after the line that currently reads
`self.last_stats.message_cache_hits += hit_keys.len();` /
`self.last_stats.message_cache_misses += missing_indices.len();` (line
~2554-2555), capture the counts before `missing_indices` is consumed later
by `into_iter()`:

```rust
#[cfg(feature = "diagnostics")]
let (diag_hits, diag_misses) = (hit_keys.len() as u64, missing_indices.len() as u64);
```

Then, at the function's final `Ok(StackedMessage { assignment_index, tensor,
raw_values })` (the very end of the function, after the reconstruct block),
insert immediately before that final `Ok(...)`:

```rust
#[cfg(feature = "diagnostics")]
diagnostics::record_guard(
    &format!("{node:?}"),
    diag_coordination_number,
    &diag_bond_dims,
    diag_start.elapsed(),
    diag_hits,
    diag_misses,
);
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --features diagnostics \
  build_environment_cache_records_guard_diagnostics_per_node_with_correct_coordination_numbers -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Run the full crate test suite with and without the feature**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release --features diagnostics
```

Expected: both PASS. The first command is the existing default-feature
suite and must show no behavior change.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): record per-node Guard diagnostics in get_or_compute_node_message"
```

---

### Task 3: Wire TreeACI's frame cache into the registry (`tensor4all-treeaci`)

**Files:**
- Modify: `crates/tensor4all-treeaci/Cargo.toml`
- Create: `crates/tensor4all-treeaci/src/branch_diagnostics.rs`
- Modify: `crates/tensor4all-treeaci/src/lib.rs`
- Modify: `crates/tensor4all-treeaci/src/frames.rs` (three call sites: ~810-822, ~977-989, ~1113-1123)
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`

**Interfaces:**
- Consumes: `tensor4all_treetn::diagnostics::{record_frame, reset, snapshot}` (Task 1/2; cross-crate, `tensor4all-treeaci` already depends on `tensor4all-treetn`).
- Produces: `tensor4all_treeaci::branch_diagnostics::{reset, snapshot, NodeDiagnostics}` for downstream crates (`gw-rs`'s isolation harness).

- [ ] **Step 1: Add the `diagnostics` feature and the re-export module**

In `crates/tensor4all-treeaci/Cargo.toml`, add one line inside the existing
`[features]` table (it already has `default`, `tenferro-cpu-faer`, and
`tenferro-provider-inject` entries):

```toml
diagnostics = ["tensor4all-treetn/diagnostics"]
```

Create `crates/tensor4all-treeaci/src/branch_diagnostics.rs`:

```rust
//! Re-export of `tensor4all-treetn`'s per-node branch diagnostics registry.
//!
//! Named `branch_diagnostics`, not `diagnostics`, to stay distinct from
//! [`crate::TreeAciDiagnostics`] -- an unrelated, already-public sweep/
//! convergence report. This module is about branch-point performance
//! (issue #671); `TreeAciDiagnostics` is about ACI convergence history.

pub use tensor4all_treetn::diagnostics::{
    record_frame, record_guard, reset, snapshot, NodeDiagnostics,
};
```

In `crates/tensor4all-treeaci/src/lib.rs`, add next to the other `pub mod`/
`pub use` declarations:

```rust
#[cfg(feature = "diagnostics")]
pub mod branch_diagnostics;
```

- [ ] **Step 2: Run to verify it builds**

```bash
cargo check --manifest-path crates/tensor4all-treeaci/Cargo.toml --features diagnostics
cargo check --manifest-path crates/tensor4all-treeaci/Cargo.toml
```

Expected: both succeed (the second, default-feature, build does not see
`branch_diagnostics` at all).

- [ ] **Step 3: Commit the scaffolding**

```bash
git add crates/tensor4all-treeaci/Cargo.toml crates/tensor4all-treeaci/src/branch_diagnostics.rs crates/tensor4all-treeaci/src/lib.rs
git commit -m "feat(treeaci): forward diagnostics feature and re-export the registry"
```

- [ ] **Step 4: Write the failing test**

`candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges`
(line ~891 in `crates/tensor4all-treeaci/src/frames/tests/mod.rs`) is the
existing test to base this on: it uses `star_tree_for_fallback_dispatch()`
(line ~636, node `0` is the hub with edge `0 -> 1` having
`directed.incoming_to_from.len() == 2`, i.e. coordination number 3) and
calls both `frames.candidate_frames_for_edge(...)` (the batched two-incoming
path) and `frames.candidate_frame(...)` (the scalar path) on the same
candidates -- so a diagnostics test built the same way exercises both of
this task's instrumented sites at once. Add, right after that existing test:

```rust
#[cfg(feature = "diagnostics")]
#[test]
fn candidate_frames_for_edge_records_frame_diagnostics_with_hub_coordination_number_three() {
    use crate::branch_diagnostics;

    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);

    let seeds = vec![vec![0, 0, 0, 0], vec![0, 0, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let incoming_edge_a = directed.incoming_to_from[0];
    let incoming_edge_b = directed.incoming_to_from[1];
    let ids_a = &candidate_sets.ids[incoming_edge_a];
    let ids_b = &candidate_sets.ids[incoming_edge_b];

    let mut candidates = Vec::new();
    for local_coordinate in 0..2 {
        for &id_a in ids_a {
            for &id_b in ids_b {
                candidates.push(ComponentSample {
                    local_coordinate,
                    incoming: vec![(incoming_edge_a, id_a), (incoming_edge_b, id_b)],
                });
            }
        }
    }

    branch_diagnostics::reset();
    let _dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();

    let snapshot = branch_diagnostics::snapshot();
    let hub_record = snapshot
        .iter()
        .find(|record| record.node == "0")
        .expect("hub node (0) recorded in branch diagnostics");
    assert_eq!(hub_record.coordination_number, 3);
    assert_eq!(hub_record.bond_dims.len(), 3);
    assert!(hub_record.frame_cache_hits + hub_record.frame_cache_misses > 0);
}
```

- [ ] **Step 5: Run test to verify it fails**

```bash
cargo test --manifest-path crates/tensor4all-treeaci/Cargo.toml --features diagnostics \
  candidate_frames_for_edge_records_frame_diagnostics_with_hub_coordination_number_three -- --nocapture
```

Expected: FAIL (nothing recorded yet).

- [ ] **Step 6: Add the topology helper**

In `crates/tensor4all-treeaci/src/frames.rs`, near `fn outgoing_bond` (line
~1848), add:

```rust
#[cfg(feature = "diagnostics")]
fn diagnostics_node_topology<V: TreeAciNode>(
    tree: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    directed: &DirectedEdge<V>,
    directed_edge: DirectedEdgeId,
) -> (String, usize, Vec<usize>) {
    let coordination_number = directed.incoming_to_from.len() + 1;
    let mut bond_dims = Vec::with_capacity(coordination_number);
    if let Ok(index) = outgoing_bond(tree, problem, directed_edge) {
        bond_dims.push(index.dim());
    }
    for &incoming in &directed.incoming_to_from {
        if let Ok(index) = outgoing_bond(tree, problem, incoming) {
            bond_dims.push(index.dim());
        }
    }
    (format!("{:?}", directed.from), coordination_number, bond_dims)
}
```

- [ ] **Step 7: Instrument the single-incoming site (~line 730-830)**

This function (`candidate_frames_for_edge`, the single-incoming-edge batched
path) already has `directed`, `tree`, `outgoing_dim`, `incoming_dim`, and
`directed_edge` in scope by the time its candidate loop starts (line ~807).
Add timing and count accumulators right before that `for (candidate_index,
candidate) in candidates.iter().enumerate() {` loop:

```rust
#[cfg(feature = "diagnostics")]
let diag_start = std::time::Instant::now();
#[cfg(feature = "diagnostics")]
let (mut diag_hits, mut diag_misses) = (0u64, 0u64);
```

Inside the loop, next to the existing `#[cfg(test)]
candidate_debug_stats::record_hit();` (line ~815-817) add:

```rust
#[cfg(feature = "diagnostics")]
{
    diag_hits += 1;
}
```

and next to the existing `#[cfg(test)] candidate_debug_stats::record_miss();`
(line ~821) add:

```rust
#[cfg(feature = "diagnostics")]
{
    diag_misses += 1;
}
```

At the end of the function, immediately before its final `results
.into_iter() ... .collect()` (the function's tail expression), add:

```rust
#[cfg(feature = "diagnostics")]
{
    let (node, coordination_number, bond_dims) =
        diagnostics_node_topology(tree, problem, directed, directed_edge);
    diagnostics::record_frame(
        &node,
        coordination_number,
        &bond_dims,
        diag_start.elapsed(),
        diag_hits,
        diag_misses,
    );
}
```

and add `use tensor4all_treetn::diagnostics;` near the file's other `use`
statements, gated:

```rust
#[cfg(feature = "diagnostics")]
use tensor4all_treetn::diagnostics;
```

- [ ] **Step 8: Instrument the two-incoming site (~line 921-1104)**

Same pattern as Step 7, inside `candidate_frames_for_edge_two_incoming`: add
`diag_start`/`diag_hits`/`diag_misses` before its candidate loop (~line 976),
increment next to the existing hit/miss `candidate_debug_stats` calls
(~line 983-989), and add the same `diagnostics::record_frame(...)` call
before the function's final `results.into_iter()...collect()` (after line
1104).

- [ ] **Step 9: Instrument the general single-candidate site (~line 1106-1153)**

`candidate_frame` fetches `tree` only after the cache check today (line
~1124: `let tree = inputs.get(input)...`), and does not fetch `directed` at
all. First, move that `let tree = inputs.get(input)...` line (with its
`ok_or(...)` error arm) up to the top of the function, immediately after the
`let cache_key = self.candidate_cache_key(...)?;` line and before the `if
let Some(key) = cache_key { ... }` hit check -- it does not depend on the
cache lookup, so this reordering changes nothing about the function's
behavior, only makes `tree` available to the hit path below. Then add,
right after that relocated `tree` fetch:

```rust
#[cfg(feature = "diagnostics")]
let diag_start = std::time::Instant::now();
#[cfg(feature = "diagnostics")]
let directed = &problem.directed_edges[directed_edge];
```

On the hit path (`if let Some(cached) = self.candidate_cache.borrow()
.get(&key) { ... return Ok(cached.clone()); }`, now with `tree` and
`directed` both already in scope above it), add before that `return`:

```rust
#[cfg(feature = "diagnostics")]
{
    let (node, coordination_number, bond_dims) =
        diagnostics_node_topology(tree, problem, directed, directed_edge);
    diagnostics::record_frame(
        &node,
        coordination_number,
        &bond_dims,
        diag_start.elapsed(),
        1,
        0,
    );
}
```

At the function's end, right before its final `Ok(values)`, add:

```rust
#[cfg(feature = "diagnostics")]
{
    let (node, coordination_number, bond_dims) =
        diagnostics_node_topology(tree, problem, directed, directed_edge);
    diagnostics::record_frame(
        &node,
        coordination_number,
        &bond_dims,
        diag_start.elapsed(),
        0,
        1,
    );
}
```

- [ ] **Step 10: Run test to verify it passes**

```bash
cargo test --manifest-path crates/tensor4all-treeaci/Cargo.toml --features diagnostics \
  candidate_frames_for_edge_records_frame_diagnostics_with_hub_coordination_number_three -- --nocapture
```

Expected: PASS. This fixture's 8 candidates split across two
`local_coordinate` groups guarantee both a first-touch miss and at least one
within-call repeat; if `frame_cache_hits + frame_cache_misses` is `0`,
`diag_hits`/`diag_misses` are not being incremented at the same call sites
`candidate_debug_stats::record_hit`/`record_miss` already run at -- recheck
Step 7/8's placement against the surrounding `#[cfg(test)]` calls rather than
weakening the assertion.

- [ ] **Step 11: Run the full crate test suite with and without the feature**

```bash
cargo test --manifest-path crates/tensor4all-treeaci/Cargo.toml --release
cargo test --manifest-path crates/tensor4all-treeaci/Cargo.toml --release --features diagnostics
```

Expected: both PASS, no change to existing test outcomes.

- [ ] **Step 12: Commit**

```bash
git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/frames/tests/mod.rs
git commit -m "feat(treeaci): record per-node frame-cache diagnostics at the three candidate_cache sites"
```

---

### Task 4: Verification pass

**Files:** none (verification only).

- [ ] **Step 1: Default-feature build and clippy, both crates**

```bash
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml
cargo build --release --manifest-path crates/tensor4all-treeaci/Cargo.toml
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path crates/tensor4all-treeaci/Cargo.toml --all-targets -- -D warnings
```

Expected: all succeed with no warnings. This is the existing default build;
it must show zero difference from before this plan.

- [ ] **Step 2: `diagnostics`-feature build and clippy, both crates**

```bash
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml --features diagnostics
cargo build --release --manifest-path crates/tensor4all-treeaci/Cargo.toml --features diagnostics
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --features diagnostics --all-targets -- -D warnings
cargo clippy --manifest-path crates/tensor4all-treeaci/Cargo.toml --features diagnostics --all-targets -- -D warnings
```

Expected: all succeed with no warnings.

- [ ] **Step 3: `cargo fmt`**

```bash
cargo fmt --all -- --check
```

Expected: no diff. If there is one, run `cargo fmt --all` and re-commit.

- [ ] **Step 4: Do not rerun the branched-hotpaths synthetic benchmark**

Per the spec's Testing section: this change is instrumentation behind a
default-off feature, not a hot-path change, so the
`diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk`
benchmark from `docs/worklogs/2026-08-23-treeaci-branched-hotpaths.md` is
not part of this task's acceptance gate. Skip it.

- [ ] **Step 5: Final commit if `cargo fmt` made changes**

```bash
git add -A
git commit -m "chore: cargo fmt"
```

(Skip this step if Step 3 reported no diff.)

---

## After this plan

This plan produces the `diagnostics`/`branch_diagnostics` API that `gw-rs`'s
isolation harness (`docs/superpowers/specs/2026-08-24-aci-stage-isolation-design.md`
in the sibling `gw-rs` checkout) depends on. That harness's own implementation
plan is written separately, after this plan's PR is at least locally usable
via the sibling-checkout `[patch]` the gw-rs spec describes.
