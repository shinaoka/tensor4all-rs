# Steiner Tree Partial Apply Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `apply_linear_operator` to handle non-contiguous operator node sets by computing the Steiner tree and inserting identity tensors at intermediate nodes, generalizing Julia's `matchsite_indices` to tree tensor networks.

**Architecture:** Add a `steiner_tree` method to `SiteIndexNetwork`, then modify `extend_operator_to_full_space` to first expand the operator's effective node set to its Steiner tree (filling intermediate nodes with identity), before composing with identity on the remaining gap nodes. This removes the connected-subtree requirement from partial apply.

**Tech Stack:** Rust, petgraph, tensor4all-treetn

---

## Current State

`apply_linear_operator` (apply.rs:139) handles partial apply by calling
`extend_operator_to_full_space` (apply.rs:208), which calls
`compose_exclusive_linear_operators` (compose.rs:168). The latter requires
each operator to be a connected subtree (`is_connected_subset` check at
compose.rs:74). This blocks non-contiguous node sets like interleaved
quantics encoding.

## Target State

```
State:     x₁ — y₁ — x₂ — y₂ — x₃ — y₃   (interleaved 2D, chain)
Operator:  x₁        x₂        x₃           (non-contiguous)
Steiner:   x₁ — y₁ — x₂ — y₂ — x₃          (minimal connected subtree)
Gap (Steiner): y₁, y₂                        (identity inserted)
Gap (outer): y₃                               (identity inserted)
```

Also works for trees:
```
Tree:    A — B — C — D
              |
              E — F
Operator: {A, F}
Steiner:   A — B — E — F
Gap (Steiner): B, E (identity)
Gap (outer): C, D (identity)
```

---

### Task 1: Add `steiner_tree` to `NodeNameNetwork`

**Files:**
- Modify: `crates/tensor4all-treetn/src/node_name_network.rs`

- [ ] **Step 1: Write a test for steiner_tree**

Add to the existing test module in `node_name_network.rs` or create a test file:

```rust
#[test]
fn test_steiner_tree_chain() {
    // Chain: 0 — 1 — 2 — 3 — 4
    let mut net = NodeNameNetwork::new();
    let n: Vec<_> = (0..5).map(|i| net.add_node(i)).collect();
    for i in 0..4 {
        net.add_edge(n[i], n[i + 1], ());
    }

    // Steiner tree of {0, 2, 4} = all nodes on paths = {0, 1, 2, 3, 4}
    let steiner = net.steiner_tree_nodes(&[n[0], n[2], n[4]].into_iter().collect());
    assert_eq!(steiner.len(), 5);

    // Steiner tree of {0, 4} = {0, 1, 2, 3, 4}
    let steiner2 = net.steiner_tree_nodes(&[n[0], n[4]].into_iter().collect());
    assert_eq!(steiner2.len(), 5);

    // Steiner tree of {1, 3} = {1, 2, 3}
    let steiner3 = net.steiner_tree_nodes(&[n[1], n[3]].into_iter().collect());
    assert_eq!(steiner3.len(), 3);
    assert!(steiner3.contains(&n[1]));
    assert!(steiner3.contains(&n[2]));
    assert!(steiner3.contains(&n[3]));
}

#[test]
fn test_steiner_tree_branching() {
    // Tree:  0 — 1 — 2
    //             |
    //             3 — 4
    let mut net = NodeNameNetwork::new();
    let n: Vec<_> = (0..5).map(|i| net.add_node(i)).collect();
    net.add_edge(n[0], n[1], ());
    net.add_edge(n[1], n[2], ());
    net.add_edge(n[1], n[3], ());
    net.add_edge(n[3], n[4], ());

    // Steiner tree of {0, 4} = {0, 1, 3, 4} (goes through branch)
    let steiner = net.steiner_tree_nodes(&[n[0], n[4]].into_iter().collect());
    assert_eq!(steiner.len(), 4);
    assert!(!steiner.contains(&n[2])); // node 2 not on path

    // Steiner tree of {0, 2, 4} = {0, 1, 2, 3, 4} (all nodes needed)
    let steiner2 = net.steiner_tree_nodes(&[n[0], n[2], n[4]].into_iter().collect());
    assert_eq!(steiner2.len(), 5);

    // Single node
    let steiner3 = net.steiner_tree_nodes(&[n[2]].into_iter().collect());
    assert_eq!(steiner3.len(), 1);
}
```

- [ ] **Step 2: Implement steiner_tree_nodes**

Algorithm for trees (simpler than general Steiner tree since trees have unique paths):
1. Pick any terminal node as root
2. BFS/DFS from root, recording paths to all terminal nodes
3. Steiner tree = union of all nodes on paths between terminals

Efficient approach using `path_between`:
1. For each pair of terminal nodes, get path
2. Union all path nodes

Or more efficiently:
1. Pick one terminal as root
2. For each other terminal, get `path_between(root, terminal)`
3. Union all path nodes

```rust
/// Compute the Steiner tree nodes for a set of terminal nodes in this tree network.
///
/// Returns the minimal set of nodes that forms a connected subtree containing
/// all terminal nodes. In a tree, this is the union of all paths between
/// terminal nodes.
pub fn steiner_tree_nodes(&self, terminals: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
    if terminals.len() <= 1 {
        return terminals.clone();
    }

    let mut result = HashSet::new();
    let terminals_vec: Vec<NodeIndex> = terminals.iter().copied().collect();
    let root = terminals_vec[0];

    for &terminal in &terminals_vec[1..] {
        if let Some(path) = self.path_between(root, terminal) {
            result.extend(path);
        }
    }

    // The union of paths from one terminal to all others covers the Steiner tree
    // in a tree graph, but we need paths between ALL pairs to get intermediate branches.
    // For a tree, union of paths from root to all terminals IS the Steiner tree
    // IF we then prune leaves that aren't terminals.
    // Simpler: collect all path nodes, then prune non-terminal leaves iteratively.
    prune_non_terminal_leaves(&self.graph, &mut result, terminals);

    result
}
```

Actually, for a tree the simplest correct approach is:
1. Union paths from an arbitrary terminal to all other terminals
2. Iteratively prune leaf nodes that are not terminals

Implement the pruning step:

```rust
fn prune_non_terminal_leaves(
    graph: &StableGraph<V, E>,
    nodes: &mut HashSet<NodeIndex>,
    terminals: &HashSet<NodeIndex>,
) {
    loop {
        let leaves_to_remove: Vec<NodeIndex> = nodes
            .iter()
            .copied()
            .filter(|&n| {
                !terminals.contains(&n)
                    && graph
                        .neighbors(n)
                        .filter(|nb| nodes.contains(nb))
                        .count()
                        <= 1
            })
            .collect();
        if leaves_to_remove.is_empty() {
            break;
        }
        for n in leaves_to_remove {
            nodes.remove(&n);
        }
    }
}
```

- [ ] **Step 3: Run tests**

```bash
cargo nextest run --release -p tensor4all-treetn --test '*' -E 'test(steiner)'
```

- [ ] **Step 4: Expose via SiteIndexNetwork**

Add a forwarding method in `crates/tensor4all-treetn/src/site_index_network.rs`:

```rust
pub fn steiner_tree_nodes(&self, terminals: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
    self.topology.steiner_tree_nodes(terminals)
}
```

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-treetn/src/node_name_network.rs \
       crates/tensor4all-treetn/src/site_index_network.rs
git commit -m "feat(treetn): add steiner_tree_nodes for minimal connected subtree computation"
```

---

### Task 2: Modify `extend_operator_to_full_space` to use Steiner tree

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply.rs`

- [ ] **Step 1: Write test for non-contiguous partial apply**

Add to `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`:

```rust
#[test]
fn test_apply_linear_operator_non_contiguous() {
    use crate::operator::apply_linear_operator;
    use crate::operator::ApplyOptions;

    // Create a 4-site chain state: site0 — site1 — site2 — site3
    // Operator covers site0 and site2 (non-contiguous, gap at site1)
    // This should work by inserting identity at site1 (Steiner tree)
    // and identity at site3 (outer gap)

    let mut state = TreeTN::<TensorDynLen, String>::new();
    let s0 = make_index(2);
    let s1 = make_index(2);
    let s2 = make_index(2);
    let s3 = make_index(2);
    let b01 = make_index(3);
    let b12 = make_index(3);
    let b23 = make_index(3);

    let t0 = TensorDynLen::random_f64(&mut rng(), vec![s0.clone(), b01.clone()]);
    let t1 = TensorDynLen::random_f64(&mut rng(), vec![b01.clone(), s1.clone(), b12.clone()]);
    let t2 = TensorDynLen::random_f64(&mut rng(), vec![b12.clone(), s2.clone(), b23.clone()]);
    let t3 = TensorDynLen::random_f64(&mut rng(), vec![b23.clone(), s3.clone()]);

    state.add_tensor("site0".into(), t0).unwrap();
    state.add_tensor("site1".into(), t1).unwrap();
    state.add_tensor("site2".into(), t2).unwrap();
    state.add_tensor("site3".into(), t3).unwrap();
    // connect chain...

    // Build identity operator on {site0, site2} (non-contiguous)
    // ... (build 2-site identity MPO with site0 and site2)

    let result = apply_linear_operator(&operator, &state, ApplyOptions::default()).unwrap();
    assert_eq!(result.node_count(), 4);

    // Verify result equals original state (identity operator)
    let dense_orig = state.to_dense().unwrap();
    let dense_result = result.to_dense().unwrap();
    let diff = dense_orig.subtract(&dense_result).unwrap();
    assert!(diff.maxabs() < 1e-10);
}
```

The worker must construct the full test with proper tensor construction and operator building. Use the existing `test_apply_linear_operator_partial` as a template but with a 4-site chain and non-contiguous operator on {site0, site2}.

- [ ] **Step 2: Run test, verify it fails**

```bash
cargo nextest run --release -p tensor4all-treetn -E 'test(non_contiguous)'
```

Expected: FAIL (currently `compose_exclusive_linear_operators` rejects non-connected operator)

- [ ] **Step 3: Modify extend_operator_to_full_space**

The key change: before calling `compose_exclusive_linear_operators`, expand the
operator to cover its Steiner tree by inserting identity tensors at intermediate
nodes. Then the operator IS a connected subtree.

In `apply.rs`, modify `extend_operator_to_full_space`:

```rust
fn extend_operator_to_full_space<T, V>(
    operator: &LinearOperator<T, V>,
    state: &TreeTN<T, V>,
) -> Result<LinearOperator<T, V>>
where ...
{
    let state_network = state.site_index_network();
    let op_nodes: HashSet<V> = operator.node_names();
    let state_nodes: HashSet<V> = state.node_names().into_iter().collect();

    // Step 1: Compute Steiner tree of operator nodes in state graph
    let op_node_indices: HashSet<NodeIndex> = op_nodes
        .iter()
        .filter_map(|name| state_network.node_index(name))
        .collect();
    let steiner_indices = state_network.steiner_tree_nodes(&op_node_indices);
    let steiner_names: HashSet<V> = steiner_indices
        .iter()
        .filter_map(|idx| state_network.node_name(*idx).cloned())
        .collect();

    // Step 2: Intermediate nodes = Steiner tree - operator nodes
    // These need identity tensors added to the operator's MPO
    let intermediate_nodes: Vec<V> = steiner_names.difference(&op_nodes).cloned().collect();

    // Step 3: Build expanded operator that includes identity at intermediate nodes
    let expanded_operator = if intermediate_nodes.is_empty() {
        operator.clone()
    } else {
        expand_operator_with_steiner_identities(operator, state, &intermediate_nodes)?
    };

    // Step 4: Now expanded_operator IS a connected subtree.
    // Remaining gaps = state_nodes - steiner_names
    let remaining_gap_nodes: Vec<V> = state_nodes.difference(&steiner_names).cloned().collect();

    if remaining_gap_nodes.is_empty() {
        return Ok(expanded_operator);
    }

    // Step 5: Compose expanded operator with identity at remaining gaps
    // (This uses the existing compose_exclusive_linear_operators path)
    // ... (same logic as current extend_operator_to_full_space but using expanded_operator)
}
```

The worker must implement `expand_operator_with_steiner_identities` which:
1. For each intermediate node, creates identity site index pairs (same as current gap logic)
2. Adds identity tensors to the operator's MPO at the intermediate positions
3. Connects them properly in the tree structure
4. Returns a new LinearOperator that covers the Steiner tree

This is the most complex part. The worker should study how `compose_exclusive_linear_operators` builds identity tensors (compose.rs:230+) and adapt that logic.

- [ ] **Step 4: Run test, verify it passes**

```bash
cargo nextest run --release -p tensor4all-treetn -E 'test(non_contiguous)'
```

- [ ] **Step 5: Run full test suite**

```bash
cargo nextest run --release -p tensor4all-treetn
```

All existing tests must still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply.rs \
       crates/tensor4all-treetn/src/operator/apply/tests/mod.rs
git commit -m "feat(treetn): support non-contiguous partial apply via Steiner tree"
```

---

### Task 3: Add test for tree topology (not just chain)

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs`

- [ ] **Step 1: Write test for branching tree**

```rust
#[test]
fn test_apply_linear_operator_non_contiguous_tree() {
    // Tree:  A — B — C
    //             |
    //             D — E
    // Operator on {A, E} — Steiner tree is {A, B, D, E}, gap B and D get identity
    // Outer gap: C

    // Build state, build identity operator on {A, E}, apply, verify result == state
}
```

- [ ] **Step 2: Run test**

```bash
cargo nextest run --release -p tensor4all-treetn -E 'test(non_contiguous_tree)'
```

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-treetn/src/operator/apply/tests/mod.rs
git commit -m "test(treetn): add non-contiguous partial apply test for tree topology"
```

---

### Task 4: Update qft.md with working 2D QFT example

**Files:**
- Modify: `docs/book/src/guides/qft.md`

After the Steiner tree feature is implemented, update the 2D QFT section
with a complete working example that uses interleaved quantics encoding
and applies 1D Fourier to each variable via partial apply.

- [ ] **Step 1: Write the 2D QFT example**

The example should:
1. Construct a 2D function f(x,y) on interleaved quantics sites [x₁,y₁,...,xᵣ,yᵣ]
2. Build 1D Fourier operator for x-variable sites
3. Apply via partial apply (now supported for non-contiguous nodes)
4. Build 1D Fourier operator for y-variable sites
5. Apply second transform
6. Verify result against 2D DFT reference values

Note: The LinearOperator node names must match the state's node names for the target variable.
This requires careful construction of the operator with the correct node name mapping.

- [ ] **Step 2: Verify mdbook builds**

```bash
mdbook build docs/book
```

- [ ] **Step 3: Commit**

```bash
git add docs/book/src/guides/qft.md
git commit -m "docs(book): update QFT guide with working 2D interleaved example"
```

---

### Task 5: Final Verification

- [ ] **Step 1: Run all checks**

```bash
cargo fmt --all
cargo clippy --workspace
cargo nextest run --release --workspace
cargo doc --workspace --no-deps
mdbook build docs/book
```

- [ ] **Step 2: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: final formatting and verification"
```
