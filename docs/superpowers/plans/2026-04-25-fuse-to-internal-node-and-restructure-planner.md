# fuse_to internal node and restructure_to planner fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `contract_node_group` to use the full tree graph instead of `SiteIndexNetwork` for connectivity and edge ordering with automatic Steiner tree expansion; relax `fuse_to` validation for internal nodes; fix `restructure_to` planner to use full graph connectivity.

**Architecture:** Both bugs stem from `SiteIndexNetwork` being an incomplete graph (omits internal nodes with bond-only indices). Fix `contract_node_group` to (a) query `self.graph` (full `NamedGraph`) for connectivity checks and parent-edge ordering, and (b) automatically expand the contracted node set to include internal connector nodes via Steiner tree — no expansion logic needed in `fuse_to`. Thread the full graph through the `restructure_to` planner.

**Tech Stack:** Rust, petgraph (StableGraph, Bfs, DfsPostOrder, astar), tensor4all-core

---

### Task 1: Add Steiner tree helper on full graph to `transform.rs`

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs`

- [ ] **Step 1: Add `steiner_tree_indices` helper function**

Insert at the top of `transform.rs`, after the existing imports:

```rust
use petgraph::stable_graph::{NodeIndex, StableGraph};
```

Add after the `use` block, as a module-level helper:

```rust
/// Compute the Steiner tree in the full graph spanning a set of terminal nodes.
///
/// For tree graphs, the Steiner tree is the union of unique shortest paths
/// from one terminal node to every other terminal node.
fn steiner_tree_indices(
    graph: &StableGraph<T, T::Index, petgraph::Undirected>,
    terminals: &HashSet<NodeIndex>,
) -> HashSet<NodeIndex> {
    if terminals.len() <= 1 {
        return terminals.clone();
    }
    let terms: Vec<NodeIndex> = terminals.iter().copied().collect();
    let root = terms[0];
    let mut result = HashSet::new();
    result.insert(root);
    for &term in &terms[1..] {
        if let Some((_, path)) = petgraph::algo::astar(
            graph,
            root,
            |n| n == term,
            |_| 1usize,
            |_| 0usize,
        ) {
            result.extend(path);
        }
    }
    result
}
```

(Verify: `use std::collections::{HashMap, HashSet}` already exists in transform.rs.)

- [ ] **Step 2: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles without errors.

---

### Task 2: Fix `contract_node_group` to use full graph + auto-expand via Steiner tree

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs:167-249`

- [ ] **Step 1: Add Steiner tree expansion + replace connectivity check**

Before the connectivity validation block (around line 199), insert Steiner tree expansion.
Replace the existing `node_indices` setup and connectivity check:

Old (lines 175-204):
```rust
        // Convert node names to NodeIndex
        let node_indices: HashSet<NodeIndex> = nodes
            .iter()
            .filter_map(|name| self.graph.node_index(name))
            .collect();

        if node_indices.len() != nodes.len() {
            return Err(anyhow::anyhow!(
                "Some nodes not found in graph: expected {} nodes, found {}",
                nodes.len(),
                node_indices.len()
            ));
        }

        // Single node case: just clone the tensor
        if nodes.len() == 1 {
            let node_name = nodes.iter().next().unwrap();
            let node_idx = self.graph.node_index(node_name).unwrap();
            return self
                .tensor(node_idx)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name));
        }

        // Validate connectivity
        if !self.site_index_network.is_connected_subset(&node_indices) {
            return Err(anyhow::anyhow!(
                "Nodes to contract do not form a connected subtree"
            ));
        }
```

New:
```rust
        // Convert node names to NodeIndex in the full graph
        let seed_indices: HashSet<NodeIndex> = nodes
            .iter()
            .filter_map(|name| self.graph.node_index(name))
            .collect();

        if seed_indices.len() != nodes.len() {
            return Err(anyhow::anyhow!(
                "Some nodes not found in graph: expected {} nodes, found {}",
                nodes.len(),
                seed_indices.len()
            ));
        }

        // Auto-expand to include internal connector nodes via Steiner tree.
        // This ensures internal nodes (no site indices) are naturally included.
        let g = self.graph.graph();
        let node_indices = if seed_indices.len() <= 1 {
            seed_indices
        } else {
            steiner_tree_indices(g, &seed_indices)
        };

        // Validate connectivity on the full graph (includes internal nodes)
        if !is_connected_subset_on_graph(g, &node_indices) {
            return Err(anyhow::anyhow!(
                "Nodes to contract do not form a connected subtree"
            ));
        }
```

- [ ] **Step 2: Add `is_connected_subset_on_graph` helper**

Add before `contract_node_group`:

```rust
/// Check if nodes form a connected induced subgraph in the given graph.
/// DFS restricted to edges where both endpoints are in `nodes`.
fn is_connected_subset_on_graph(
    graph: &StableGraph<T, T::Index, petgraph::Undirected>,
    nodes: &HashSet<NodeIndex>,
) -> bool {
    if nodes.is_empty() || nodes.len() == 1 {
        return true;
    }
    let start = *nodes.iter().next().unwrap();
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    seen.insert(start);
    while let Some(v) = stack.pop() {
        for nb in graph.neighbors(v) {
            if nodes.contains(&nb) && seen.insert(nb) {
                stack.push(nb);
            }
        }
    }
    seen.len() == nodes.len()
}
```

- [ ] **Step 3: Replace edge ordering with full-graph parent edges**

Replace lines 210-220 (the edges and internal_edges computation):

Old:
```rust
        // Get edges within the group, ordered from leaves to root
        let edges = self
            .site_index_network
            .edges_to_canonicalize(None, root_idx);

        // Filter to only edges within our group
        let internal_edges: Vec<(NodeIndex, NodeIndex)> = edges
            .iter()
            .filter(|(from, to)| node_indices.contains(from) && node_indices.contains(to))
            .cloned()
            .collect();
```

New:
```rust
        // Get edges within the group, ordered from leaves to root.
        // Use the full graph (includes internal nodes).
        let g = self.graph.graph();
        let post_order: Vec<NodeIndex> =
            petgraph::visit::DfsPostOrder::new(g, root_idx)
                .iter(g)
                .collect();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut bfs = petgraph::visit::Bfs::new(g, root_idx);
        while let Some(node) = bfs.next(g) {
            for neighbor in g.neighbors(node) {
                if neighbor != root_idx {
                    parent.entry(neighbor).or_insert(node);
                }
            }
        }
        let mut internal_edges: Vec<(NodeIndex, NodeIndex)> = Vec::new();
        for &node in &post_order {
            if node != root_idx && node_indices.contains(&node) {
                if let Some(&p) = parent.get(&node) {
                    if node_indices.contains(&p) {
                        internal_edges.push((node, p));
                    }
                }
            }
        }
```

- [ ] **Step 4: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles without errors.

- [ ] **Step 5: Run existing tests to check for regressions**

```bash
cargo test -p tensor4all-treetn
```

Expected: all 298 tests pass.

---

### Task 3: Fix `fuse_to` to skip internal nodes in validation

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform.rs:61-160`

- [ ] **Step 1: Skip internal nodes in validation**

Replace lines 122-131 (the "all current nodes are accounted for" check):

Old:
```rust
        // Check all current nodes are accounted for
        for current_name in self.node_names() {
            if !current_to_target.contains_key(&current_name) {
                return Err(anyhow::anyhow!(
                    "Current node {:?} has no corresponding target node",
                    current_name
                ))
                .context("fuse_to: missing target for current node");
            }
        }
```

New:
```rust
        // Check all current nodes are accounted for.
        // Nodes without site indices (internal bonding nodes) are skipped —
        // `contract_node_group` auto-expands via Steiner tree to include them.
        for current_name in self.node_names() {
            if !current_to_target.contains_key(&current_name) {
                // Only flag as missing if the node has site indices
                if self.site_space(&current_name).is_some() {
                    return Err(anyhow::anyhow!(
                        "Current node {:?} has site indices but no corresponding target node",
                        current_name
                    ))
                    .context("fuse_to: missing target for current node");
                }
            }
        }
```

- [ ] **Step 2: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles without errors.

- [ ] **Step 3: Run existing tests**

```bash
cargo test -p tensor4all-treetn
```

Expected: all 298 tests pass.

---

### Task 4: Add `fuse_to` test with pure internal node (no site index on center)

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/transform/tests/mod.rs`

- [ ] **Step 1: Add test for Y-shape with site-less center node**

Add after the existing Y-shape tests (after `test_fuse_to_y_shape_identity`):

```rust
#[test]
fn test_fuse_to_y_shape_internal_center_node() {
    // Y-shape where center node B has NO site index:
    //     A(site_a)
    //       |
    //     B(no site)   <-- internal node, only bonds
    //      / \
    // C(site_c) D(site_d)
    let mut tn = TreeTN::<IdxTensor, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    let tensor_a =
        IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // B has NO site index — only bond indices
    let tensor_b = IdxTensor::from_dense(
        vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
        vec![1.0; 27],
    )
    .unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    let tensor_c =
        IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    let tensor_d =
        IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("D".to_string(), tensor_d).unwrap();

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();
    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

    // Fuse B+C+D into one node (B is internal, has no site index)
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a.clone()]))
        .unwrap();
    target
        .add_node(
            "BCD".to_string(),
            HashSet::from([site_c.clone(), site_d.clone()]),
        )
        .unwrap();
    target
        .add_edge(&"A".to_string(), &"BCD".to_string())
        .unwrap();

    // Should succeed: B is pulled in implicitly as the connector node
    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "fused Y-shape with internal center should match original contraction"
    );
}
```

- [ ] **Step 2: Run the new test**

```bash
cargo test -p tensor4all-treetn -- test_fuse_to_y_shape_internal_center_node
```

Expected: test passes.

- [ ] **Step 3: Run all tests**

```bash
cargo test -p tensor4all-treetn
```

Expected: all tests pass.

---

### Task 5: Fix `restructure_to` planner — thread full graph for connectivity

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/restructure/mod.rs`

- [ ] **Step 1: Change `build_plan` and `target_nodes_span_connected_currents` to accept full graph info**

Change `build_plan` signature (line 842) to accept a closure or the full graph's node-name set:

```rust
fn build_plan<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    current_full_graph: &NamedGraph<CurrentV, T, T::Index>,
) -> Result<RestructurePlan<CurrentV, TargetV, T::Index>>
```

Update the call site in `restructure_to` (look for `build_plan::<T, V, TargetV>(...)` near line 1016):

Old:
```rust
    let plan = build_plan::<T, V, TargetV>(
        current.site_index_network(),
        target,
    )?;
```

New:
```rust
    let plan = build_plan::<T, V, TargetV>(
        current.site_index_network(),
        target,
        &current.graph, // NamedGraph field is pub(crate), accessible within crate
    )?;
```

- [ ] **Step 2: Update `target_nodes_span_connected_currents` to use full graph**

Change signature and connectivity check:

```rust
fn target_nodes_span_connected_currents<T, CurrentV, TargetV>(
    current: &SiteIndexNetwork<CurrentV, T::Index>,
    target: &SiteIndexNetwork<TargetV, T::Index>,
    site_to_current: &HashMap<<T::Index as IndexLike>::Id, CurrentV>,
    full_graph: &NamedGraph<CurrentV, T, T::Index>,
) -> Result<bool>
```

Replace the connectivity check inside (line 265-267). Old:
```rust
        if !current.is_connected_subset(&current_nodes) {
            return Ok(false);
        }
```

New:
```rust
        // Check connectivity on the full graph (includes internal nodes
        // that may connect the site-bearing members)
        let full_node_indices: HashSet<NodeIndex> = current_nodes
            .iter()
            .filter_map(|n| full_graph.node_index(n))
            .collect();
        if full_node_indices.is_empty() {
            return Ok(false);
        }
        let g = full_graph.graph();
        let start = *full_node_indices.iter().next().unwrap();
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        seen.insert(start);
        while let Some(v) = stack.pop() {
            for nb in g.neighbors(v) {
                if full_node_indices.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        if seen.len() != full_node_indices.len() {
            return Ok(false);
        }
```

Update all call sites of `target_nodes_span_connected_currents` in `build_plan` (line 862) to pass the new parameter:

```rust
    if current_nodes_map_uniquely_to_targets::<T, CurrentV, TargetV>(current, &site_to_target)? {
        if target_nodes_span_connected_currents::<T, CurrentV, TargetV>(
            current,
            target,
            &site_to_current,
            current_full_graph,
        )? {
            return Ok(RestructurePlan {
                kind: RestructurePlanKind::FuseOnly,
            });
        }
```

- [ ] **Step 3: Add missing imports in `restructure/mod.rs`**

Ensure these imports exist in the restructure module:
```rust
use crate::named_graph::NamedGraph;
use std::collections::{HashMap, HashSet};
```

(Check: `HashMap`, `HashSet` are already imported. `NamedGraph` may need to be added at the top of restructure/mod.rs.)

- [ ] **Step 4: Verify compilation**

```bash
cargo build -p tensor4all-treetn
```

Expected: compiles without errors.

- [ ] **Step 5: Run existing tests**

```bash
cargo test -p tensor4all-treetn
```

Expected: all 298 tests pass.

---

### Task 6: Add `restructure_to` test with internal center node

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/restructure/mod.rs` (inline tests)

- [ ] **Step 1: Add test for Y-shape with site-less center via restructure_to**

Add after the existing Y-shape tests:

```rust
    #[test]
    fn test_restructure_to_y_shape_internal_center() -> anyhow::Result<()> {
        // Y-shape with internal center (B has no site index):
        //     A(site_a)
        //       |
        //     B(no site)
        //      / \
        // C(site_c) D(site_d)
        let mut tn = TreeTN::<IdxTensor, String>::new();
        let site_a = DynIndex::new_dyn(2);
        let site_c = DynIndex::new_dyn(2);
        let site_d = DynIndex::new_dyn(2);
        let bond_ab = DynIndex::new_dyn(3);
        let bond_bc = DynIndex::new_dyn(3);
        let bond_bd = DynIndex::new_dyn(3);

        let tensor_a = IdxTensor::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6])?;
        tn.add_tensor("A".to_string(), tensor_a)?;
        // B has no site index
        let tensor_b = IdxTensor::from_dense(
            vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
            vec![1.0; 27],
        )?;
        tn.add_tensor("B".to_string(), tensor_b)?;
        let tensor_c = IdxTensor::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6])?;
        tn.add_tensor("C".to_string(), tensor_c)?;
        let tensor_d = IdxTensor::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6])?;
        tn.add_tensor("D".to_string(), tensor_d)?;

        let n_a = tn.node_index(&"A".to_string()).unwrap();
        let n_b = tn.node_index(&"B".to_string()).unwrap();
        let n_c = tn.node_index(&"C".to_string()).unwrap();
        let n_d = tn.node_index(&"D".to_string()).unwrap();
        tn.connect(n_a, &bond_ab, n_b, &bond_ab)?;
        tn.connect(n_b, &bond_bc, n_c, &bond_bc)?;
        tn.connect(n_b, &bond_bd, n_d, &bond_bd)?;

        // Restructure: fuse C+D into one (B is internal connector)
        let mut target: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
        target.add_node("A".to_string(), HashSet::from([site_a])).map_err(anyhow::Error::msg)?;
        target.add_node("CD".to_string(), HashSet::from([site_c, site_d])).map_err(anyhow::Error::msg)?;
        target.add_edge(&"A".to_string(), &"CD".to_string()).map_err(anyhow::Error::msg)?;

        let result = tn.restructure_to(&target, &RestructureOptions::default())?;
        assert_eq!(result.node_count(), 2);
        let dense_expected = tn.contract_to_tensor()?;
        let dense_actual = result.contract_to_tensor()?;
        assert!(dense_actual.distance(&dense_expected).unwrap() < 1e-12);
        Ok(())
    }
```

- [ ] **Step 2: Run the new test**

```bash
cargo test -p tensor4all-treetn -- test_restructure_to_y_shape_internal_center
```

Expected: test passes.

- [ ] **Step 3: Run all tests**

```bash
cargo test -p tensor4all-treetn
```

Expected: all tests pass.

---

### Task 7: Final verification

- [ ] **Step 1: Run full test suite**

```bash
cargo test --workspace --release
```

Expected: all tests pass.

- [ ] **Step 2: Format**

```bash
cargo fmt --all --check
```

Expected: no formatting issues.
