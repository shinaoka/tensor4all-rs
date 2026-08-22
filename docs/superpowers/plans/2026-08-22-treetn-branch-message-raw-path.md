# TreeTNCachedEvaluator Branch-Node Raw Message Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 25.84x wall-time gap (measured, see below) that a single degree-3 branch node introduces into `TreeTNCachedEvaluator::evaluate_batched`'s realistic floating-zone-walk call pattern, by giving the exactly-two-children case a raw-array fast path analogous to the existing exactly-one-child (`try_compute_chain_message_raw`) path, instead of falling through to the generic `compute_stacked_message` (`IdxTensor` + `contract_with_options`) path.

**Architecture:** Add `BranchContractionSpec` + `grouped_branch_message_contraction` (mirroring `ChainContractionSpec` + `grouped_chain_message_contraction`), computed in two BLAS/vectorized steps per physical-value group instead of one, because this evaluator's points are independent per-point assignments (not a cartesian product like the `tensor4all-treeaci` fix): Step A folds in child 1 via one `mat_mul_owned` call per physical-value group (T sliced at that physical value is shared across the group, so all points' child-1 columns batch into one matmul, exactly like the chain kernel); Step B folds in child 2 via a vectorized accumulate loop over `child_dim_2` (this step cannot be a single shared-matrix matmul, since child 2's contribution differs per point), which is still allocation-free, function-call-free array arithmetic instead of `contract_with_options`'s generic per-call `IdxTensor` construction and n-ary contraction planning. Add `try_compute_branch_message_raw` / `_complex_raw` (mirroring the chain raw functions) and wire them into `get_or_compute_node_message` as a new attempt between the existing chain attempt and the `compute_stacked_message` fallback, for both the real and complex branches. Nodes with 0, 1, or 3+ children are untouched.

**Tech Stack:** Rust, `tensor4all_tensorbackend::{Matrix, mat_mul_owned, BlasMul}` (already used by the existing chain kernel).

**Spec:** This plan's own "Architecture" section, empirically validated in-session: `diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk` (temporary, added to `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`'s test module) measured a 16-site chain (bond=128) at 83.7ms vs a same-size comb tree (one degree-3 hub, three 5-site arms, bond=128) at 2.16s for the same `NSEARCH=5`, `MAX_SWEEPS=100` floating-zone walk -- 25.84x, at a scale close to the real gw-rs downstream workload (observed bond ~90). This follows on from `tensor4all-rs#671`/`docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`, which fixed an analogous but structurally distinct bug in a different crate (`tensor4all-treeaci/frames.rs`) that turned out not to be the dominant cost once `aci_global_guard=true` routes most evaluation through this evaluator instead.

## Global Constraints

- Crate-scoped commands only (`--manifest-path crates/tensor4all-treetn/Cargo.toml` or `-p tensor4all-treetn`); do not build the full workspace.
- `cargo test --release` for this crate (debug-mode is slow for these fixtures).
- No `unwrap()`/`expect()` in library code (test code may use them, matching existing style in this file's `tests` module).
- `cargo fmt --all` before considering any task done.
- Every new raw path must be covered by a test asserting exact equality against the untouched generic `compute_stacked_message` path's result, mirroring this file's own `raw_chain_message_matches_generic_contraction` / `raw_complex_chain_message_matches_generic_contraction` convention.
- Do not touch `try_compute_chain_message_raw`, `grouped_chain_message_contraction`, `compute_stacked_message`, or the 0-/1-/3+-child dispatch behavior -- this plan only adds a new attempt for exactly 2 children, inserted before the generic fallback.
- Do not touch `CHAIN_BLAS_WORK_THRESHOLD` / `CHAIN_BLAS_MIN_GROUP_POINTS` or the chain path's own scalar-fallback thresholds; add separate constants for the branch path so neither path's tuning affects the other.

---

## File Structure

- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
  - New `BranchContractionSpec` struct + `BRANCH_BLAS_WORK_THRESHOLD` / `BRANCH_BLAS_MIN_GROUP_POINTS` constants, placed near `ChainContractionSpec`.
  - New `scalar_branch_message_contraction` (naive reference, mirrors `scalar_chain_message_contraction`) and `grouped_branch_message_contraction` (BLAS/vectorized path), placed near `grouped_chain_message_contraction`.
  - New methods `try_compute_branch_message_raw` / `try_compute_branch_message_complex_raw` on `TreeTNCachedEvaluator`, placed after `try_compute_chain_message_complex_raw`.
  - Dispatch edits inside `get_or_compute_node_message` (both the complex and real branches), inserting a `try_compute_branch_message_*_raw` attempt between the chain attempt and `compute_stacked_message`.
  - Test module: new fixtures/tests (Task 3), and the existing temporary diagnostic test updated in place once the fix lands (Task 4) rather than reverted, mirroring how `frames.rs`'s equivalent diagnostic became a kept measurement.

## Global Constraints Recap for Every Task

Every task below ends with, at minimum: `cargo fmt --all -- --check` clean and `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator:: --no-fail-fast` green, before moving to the next task.

---

### Task 1: `BranchContractionSpec` + `grouped_branch_message_contraction` (real-valued)

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` (insert near `ChainContractionSpec`/`grouped_chain_message_contraction`)
- Test: same file's `tests` module

**Interfaces:**
- Produces:
  ```rust
  #[derive(Clone, Copy, Debug)]
  struct BranchContractionSpec {
      strides: [usize; 4],
      physical_axis: usize,
      parent_axis: usize,
      child_axis_1: usize,
      child_axis_2: usize,
      parent_dim: usize,
      child_dim_1: usize,
      child_dim_2: usize,
  }

  fn grouped_branch_message_contraction<T>(
      spec: BranchContractionSpec,
      raw: &[T],
      physical_values: &[usize],
      child1_columns: &[T],
      child2_columns: &[T],
  ) -> Result<Vec<T>>
  where
      T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>;
  ```
  `child1_columns` is `point_count * child_dim_1` (point-major, i.e. point `p`'s column is `child1_columns[p*child_dim_1..(p+1)*child_dim_1]`, matching `try_compute_chain_message_raw`'s existing `child_columns` convention). Same for `child2_columns` with `child_dim_2`. Returns `point_count * parent_dim` (point-major, matching `grouped_chain_message_contraction`'s output convention exactly, since callers treat both outputs identically).
- Consumes: `Matrix`, `mat_mul_owned` (`tensor4all_tensorbackend`, already imported in this file).

- [ ] **Step 1: Write the failing test**

Add to the `tests` module, near `grouped_chain_contraction_matches_scalar_reference_for_real_values`:

```rust
#[test]
fn grouped_branch_contraction_matches_direct_reference_for_real_values() {
    // 2x2x2x2 tensor (parent=2, child1=2, child2=2, physical=2), axis order
    // [physical, parent, child1, child2] so strides are trivial to hand-check:
    // strides = [1, 2, 4, 8].
    let raw: Vec<f64> = (0..16).map(|v| v as f64 + 1.0).collect();
    let spec = BranchContractionSpec {
        strides: [1, 2, 4, 8],
        physical_axis: 0,
        parent_axis: 1,
        child_axis_1: 2,
        child_axis_2: 3,
        parent_dim: 2,
        child_dim_1: 2,
        child_dim_2: 2,
    };
    // 3 points: point 0 at physical=0, points 1-2 at physical=1 (a mixed
    // group split so the BLAS path -- once it engages above its point-count
    // threshold -- and the scalar path are both exercised across this
    // suite's other size variants; this test itself stays small enough to
    // hand-verify and therefore always takes the scalar branch).
    let physical_values = vec![0usize, 1, 1];
    let child1_columns: Vec<f64> = vec![1.0, 0.5, /*pt0*/ 2.0, 1.0, /*pt1*/ 0.25, 3.0 /*pt2*/];
    let child2_columns: Vec<f64> = vec![1.0, 1.0, /*pt0*/ 0.5, 2.0, /*pt1*/ 1.5, 0.5 /*pt2*/];

    let actual = grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child1_columns,
        &child2_columns,
    )
    .unwrap();

    // Direct reference: message[parent,p] = sum_{c1,c2} raw[v,parent,c1,c2] * child1[c1,p] * child2[c2,p]
    let mut expected = vec![0.0f64; 3 * 2];
    for (p, &v) in physical_values.iter().enumerate() {
        for parent in 0..2 {
            let mut sum = 0.0;
            for c1 in 0..2 {
                for c2 in 0..2 {
                    let flat = v + 2 * parent + 4 * c1 + 8 * c2;
                    sum += raw[flat] * child1_columns[p * 2 + c1] * child2_columns[p * 2 + c2];
                }
            }
            expected[p * 2 + parent] = sum;
        }
    }
    assert_eq!(actual, expected);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::grouped_branch_contraction_matches_direct_reference_for_real_values`
Expected: FAIL to compile -- `BranchContractionSpec`/`grouped_branch_message_contraction` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Insert near `ChainContractionSpec` (after its definition):

```rust
/// Minimum scalar multiply count / group size before the backend setup cost
/// is amortized by the grouped branch kernel, mirroring `CHAIN_BLAS_WORK_THRESHOLD`
/// / `CHAIN_BLAS_MIN_GROUP_POINTS` but tuned independently: a branch step
/// does two contractions per group instead of one, so its fixed per-call
/// setup cost differs from the chain kernel's.
const BRANCH_BLAS_WORK_THRESHOLD: usize = 4096;
const BRANCH_BLAS_MIN_GROUP_POINTS: usize = 4;

#[derive(Clone, Copy, Debug)]
struct BranchContractionSpec {
    strides: [usize; 4],
    physical_axis: usize,
    parent_axis: usize,
    child_axis_1: usize,
    child_axis_2: usize,
    parent_dim: usize,
    child_dim_1: usize,
    child_dim_2: usize,
}
```

Insert near `grouped_chain_message_contraction` (after it):

```rust
fn scalar_branch_message_contraction<T>(
    spec: BranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child1_columns: &[T],
    child2_columns: &[T],
) -> Result<Vec<T>>
where
    T: Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let BranchContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis_1,
        child_axis_2,
        parent_dim,
        child_dim_1,
        child_dim_2,
    } = spec;
    let point_count = physical_values.len();
    let mut output = vec![T::default(); point_count * parent_dim];
    for (point, &physical_value) in physical_values.iter().enumerate() {
        for parent in 0..parent_dim {
            let mut sum = T::default();
            for c1 in 0..child_dim_1 {
                let child1_value = child1_columns[point * child_dim_1 + c1];
                for c2 in 0..child_dim_2 {
                    let child2_value = child2_columns[point * child_dim_2 + c2];
                    let mut axis_values = [0usize; 4];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent;
                    axis_values[child_axis_1] = c1;
                    axis_values[child_axis_2] = c2;
                    let flat = axis_values[0] * strides[0]
                        + axis_values[1] * strides[1]
                        + axis_values[2] * strides[2]
                        + axis_values[3] * strides[3];
                    sum += *raw.get(flat).ok_or_else(|| {
                        anyhow::anyhow!("branch tensor offset {flat} is out of bounds")
                    })? * child1_value
                        * child2_value;
                }
            }
            output[point * parent_dim + parent] = sum;
        }
    }
    Ok(output)
}

/// Contracts a branch node's raw tensor data against two children's
/// already-computed raw message columns, generalizing
/// [`grouped_chain_message_contraction`] from one child to two.
///
/// Unlike the cartesian-product batching in `tensor4all-treeaci/frames.rs`'s
/// analogous fix, this evaluator's `points` are independent per-point
/// assignments (point `p` names one specific `(child1_assignment,
/// child2_assignment)` pair, not every combination), so only the FIRST
/// child's contraction reduces to a single shared-matrix `mat_mul_owned`
/// call per physical-value group (the node's raw tensor slice at that
/// physical value is the same for every point in the group, so all of the
/// group's child-1 columns can be batched against it in one matmul). The
/// second child's contraction cannot share a single matrix across points
/// this way -- the intermediate from step one already differs per point --
/// so it is folded in via a vectorized accumulate loop over `child_dim_2`
/// instead: still allocation-free, per-element-function-call-free array
/// arithmetic, just not a single BLAS call.
fn grouped_branch_message_contraction<T>(
    spec: BranchContractionSpec,
    raw: &[T],
    physical_values: &[usize],
    child1_columns: &[T],
    child2_columns: &[T],
) -> Result<Vec<T>>
where
    T: BlasMul + Copy + Default + std::ops::AddAssign + std::ops::Mul<Output = T>,
{
    let BranchContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis_1,
        child_axis_2,
        parent_dim,
        child_dim_1,
        child_dim_2,
    } = spec;
    let point_count = physical_values.len();
    anyhow::ensure!(
        child1_columns.len() == point_count * child_dim_1,
        "branch child-1 message length {} does not match {} points x {} child values",
        child1_columns.len(),
        point_count,
        child_dim_1
    );
    anyhow::ensure!(
        child2_columns.len() == point_count * child_dim_2,
        "branch child-2 message length {} does not match {} points x {} child values",
        child2_columns.len(),
        point_count,
        child_dim_2
    );

    if point_count < 2 * BRANCH_BLAS_MIN_GROUP_POINTS {
        return scalar_branch_message_contraction(
            spec,
            raw,
            physical_values,
            child1_columns,
            child2_columns,
        );
    }

    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for (point, &physical_value) in physical_values.iter().enumerate() {
        groups.entry(physical_value).or_default().push(point);
    }
    let scalar_work = parent_dim * child_dim_1 * child_dim_2 * point_count;
    if scalar_work < BRANCH_BLAS_WORK_THRESHOLD
        || groups.len() > 8
        || groups
            .values()
            .any(|points| points.len() < BRANCH_BLAS_MIN_GROUP_POINTS)
    {
        return scalar_branch_message_contraction(
            spec,
            raw,
            physical_values,
            child1_columns,
            child2_columns,
        );
    }

    let mut output = vec![T::default(); point_count * parent_dim];
    for (physical_value, points) in groups {
        // Step A: fold in child 1 via one matmul for the whole group. `left`
        // is (parent_dim*child_dim_2) x child_dim_1, laid out so child_dim_1
        // is the trailing (column) axis -- child2 rides along in "rows",
        // ordered slower than parent so a fixed c2 selects a contiguous
        // parent_dim-length row block within each column below.
        let left_len = parent_dim * child_dim_2 * child_dim_1;
        let mut left = vec![T::default(); left_len];
        for c1 in 0..child_dim_1 {
            for c2 in 0..child_dim_2 {
                for parent in 0..parent_dim {
                    let mut axis_values = [0usize; 4];
                    axis_values[physical_axis] = physical_value;
                    axis_values[parent_axis] = parent;
                    axis_values[child_axis_1] = c1;
                    axis_values[child_axis_2] = c2;
                    let flat = axis_values[0] * strides[0]
                        + axis_values[1] * strides[1]
                        + axis_values[2] * strides[2]
                        + axis_values[3] * strides[3];
                    let left_row = parent + parent_dim * c2;
                    let left_offset = left_row + parent_dim * child_dim_2 * c1;
                    left[left_offset] = *raw.get(flat).ok_or_else(|| {
                        anyhow::anyhow!("branch tensor offset {flat} is out of bounds")
                    })?;
                }
            }
        }
        let mut right = Vec::with_capacity(child_dim_1 * points.len());
        for &point in &points {
            let start = point * child_dim_1;
            right.extend_from_slice(&child1_columns[start..start + child_dim_1]);
        }
        let intermediate = mat_mul_owned(
            Matrix::from_col_major_vec(parent_dim * child_dim_2, child_dim_1, left),
            Matrix::from_col_major_vec(child_dim_1, points.len(), right),
        )
        .map_err(anyhow::Error::from)?;
        let intermediate = intermediate.as_col_major_slice();

        // Step B: fold in child 2 via a vectorized accumulate over child_dim_2
        // -- the intermediate's "rows" already interleave parent (fast) and
        // c2 (slow) per group column, so for a fixed c2 the parent_dim-length
        // slice at rows [c2*parent_dim, (c2+1)*parent_dim) within each
        // group-column is contiguous.
        for (column, &point) in points.iter().enumerate() {
            let column_base = column * parent_dim * child_dim_2;
            let destination = point * parent_dim;
            for c2 in 0..child_dim_2 {
                let child2_value = child2_columns[point * child_dim_2 + c2];
                let row_base = column_base + c2 * parent_dim;
                for parent in 0..parent_dim {
                    output[destination + parent] += intermediate[row_base + parent] * child2_value;
                }
            }
        }
    }
    Ok(output)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::grouped_branch_contraction_matches_direct_reference_for_real_values -- --nocapture`
Expected: PASS (this small test always takes the scalar fallback since `point_count=3 < 2*BRANCH_BLAS_MIN_GROUP_POINTS=8`, so it validates `scalar_branch_message_contraction`'s indexing first).

- [ ] **Step 5: Add a large-group test that forces the BLAS path, and a complex-value variant**

Mirror `grouped_chain_contraction_large_real_groups_match_scalar_reference` / `grouped_chain_contraction_large_complex_groups_match_scalar_reference`: build a `BranchContractionSpec` with `parent_dim`/`child_dim_1`/`child_dim_2` large enough that `scalar_work >= BRANCH_BLAS_WORK_THRESHOLD` and at least one physical-value group with `>= BRANCH_BLAS_MIN_GROUP_POINTS` points (e.g. `parent_dim=8, child_dim_1=8, child_dim_2=8`, physical dim 2, `point_count=40` split across the 2 physical values), random `f64` raw/child columns via `rand::rngs::StdRng` (seeded, matching this file's existing test RNG usage), and assert `grouped_branch_message_contraction(...)` equals `scalar_branch_message_contraction(...)` (the untouched reference) element-wise via `assert_eq!` for `f64`, or `Complex64` with the same structure for the complex variant (`T: BlasMul` must hold for `Complex64` too -- confirm by checking `grouped_chain_contraction_matches_scalar_reference_for_complex_values`'s existing pattern and mirror its `Complex64` construction).

- [ ] **Step 6: Run full test module, commit**

```bash
cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator:: --no-fail-fast
cargo fmt --all
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): add batched two-child branch message contraction primitive"
```

---

### Task 2: `try_compute_branch_message_raw` / `_complex_raw` + dispatch wiring

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` (new methods after `try_compute_chain_message_complex_raw`; dispatch edits inside `get_or_compute_node_message`, both the `tensor_is_complex` branch around line 2001 and the real branch around line 2034 -- exact line numbers will have shifted after Task 1's insertion, locate by function name)

**Interfaces:**
- Produces:
  ```rust
  fn try_compute_branch_message_raw(
      &self,
      node: &V,
      values: ColMajorArrayRef<'_, usize>,
      points: &[usize],
      plan: &RootedMessagePlan<V>,
      assignment_batches: &HashMap<V, AssignmentBatch>,
      messages: &HashMap<V, StackedMessage>,
  ) -> Result<Option<Vec<f64>>>

  fn try_compute_branch_message_complex_raw(
      &self,
      node: &V,
      values: ColMajorArrayRef<'_, usize>,
      points: &[usize],
      plan: &RootedMessagePlan<V>,
      assignment_batches: &HashMap<V, AssignmentBatch>,
      messages: &HashMap<V, StackedMessage>,
  ) -> Result<Option<Vec<Complex64>>>
  ```
  Same `Ok(None)` escape-hatch contract as `try_compute_chain_message_raw`: return `None` when `node` is not eligible, so the caller falls back to `compute_stacked_message`.
- Consumes: `grouped_branch_message_contraction` (Task 1), the same tree/plan/message-cache accessors `try_compute_chain_message_raw` already uses (`self.layout.entries_by_node`, `plan.children`, `tensor_for_node`, `self.tree.edge_between`/`bond_index`, `messages.get(child)`, `assignment_batches.get(child)`).

- [ ] **Step 1: Write the failing test**

Add to the `tests` module, near `raw_chain_message_matches_generic_contraction`. First read that existing test in full (`crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`, search `fn raw_chain_message_matches_generic_contraction`) to copy its exact setup pattern (how it builds a small tree, a `RootedMessagePlan`, `assignment_batches`, and calls both the raw and generic paths on the SAME inputs to compare). Build an analogous test using a **star-shaped** tree (hub + 3 leaves, mirroring this file's existing `star_tree()` fixture) with `center` set to one leaf (so the hub has 2 children, the other 2 leaves), and assert:

```rust
#[test]
fn raw_branch_message_matches_generic_contraction() {
    // Build using star_tree(), fix center to leaf 1 (so node 0 / the hub
    // has children [2, 3] when rooted toward center 1), drive both
    // try_compute_branch_message_raw and compute_stacked_message for the
    // hub node on the same points, assert equal results.
    // ... (mirror raw_chain_message_matches_generic_contraction's exact
    // plan/assignment_batches/messages setup, substituting star_tree()
    // and center=1 for that test's chain fixture and center choice)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::raw_branch_message_matches_generic_contraction`
Expected: FAIL to compile -- `try_compute_branch_message_raw` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Insert after `try_compute_chain_message_complex_raw`'s closing `}`:

```rust
/// Computes a branch node's (exactly two children) message directly from
/// raw data, generalizing [`Self::try_compute_chain_message_raw`] from one
/// child to two via [`grouped_branch_message_contraction`].
///
/// Returns `Ok(None)` when `node` is not eligible (not exactly one physical
/// index and exactly two children, a complex-valued tensor, or an
/// unexpected axis count), so the caller falls back to
/// [`Self::compute_stacked_message`].
fn try_compute_branch_message_raw(
    &self,
    node: &V,
    values: ColMajorArrayRef<'_, usize>,
    points: &[usize],
    plan: &RootedMessagePlan<V>,
    assignment_batches: &HashMap<V, AssignmentBatch>,
    messages: &HashMap<V, StackedMessage>,
) -> Result<Option<Vec<f64>>> {
    let entries = self
        .layout
        .entries_by_node
        .get(node)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let [entry] = entries else {
        return Ok(None);
    };
    let children = plan.children.get(node).map(Vec::as_slice).unwrap_or(&[]);
    let [child_1, child_2] = children else {
        return Ok(None);
    };

    let tensor = tensor_for_node(self.tree, node)?;
    if tensor.is_complex() {
        return Ok(None);
    }
    let tensor_indices = tensor.indices();
    if tensor_indices.len() != 4 {
        return Ok(None);
    }
    let Some(physical_axis) = tensor_indices.iter().position(|idx| idx == &entry.index) else {
        return Ok(None);
    };
    let Some(child_1_edge) = self.tree.edge_between(node, child_1) else {
        return Ok(None);
    };
    let Some(child_1_bond_index) = self.tree.bond_index(child_1_edge) else {
        return Ok(None);
    };
    let Some(child_axis_1) = tensor_indices
        .iter()
        .position(|idx| idx == child_1_bond_index)
    else {
        return Ok(None);
    };
    let Some(child_2_edge) = self.tree.edge_between(node, child_2) else {
        return Ok(None);
    };
    let Some(child_2_bond_index) = self.tree.bond_index(child_2_edge) else {
        return Ok(None);
    };
    let Some(child_axis_2) = tensor_indices
        .iter()
        .position(|idx| idx == child_2_bond_index)
    else {
        return Ok(None);
    };
    let Some(parent_axis) = (0..4).find(|&axis| {
        axis != physical_axis && axis != child_axis_1 && axis != child_axis_2
    }) else {
        return Ok(None);
    };

    let child_1_message = messages.get(child_1).ok_or_else(|| {
        anyhow::anyhow!(
            "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing message for child {:?}",
            child_1
        )
    })?;
    let child_2_message = messages.get(child_2).ok_or_else(|| {
        anyhow::anyhow!(
            "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing message for child {:?}",
            child_2
        )
    })?;
    let Some(child_1_values) = raw_real_message_values(child_1_message)? else {
        return Ok(None);
    };
    let Some(child_2_values) = raw_real_message_values(child_2_message)? else {
        return Ok(None);
    };

    let child_1_assignment_batch = assignment_batches.get(child_1).ok_or_else(|| {
        anyhow::anyhow!(
            "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing assignment batch for child {:?}",
            child_1
        )
    })?;
    let child_2_assignment_batch = assignment_batches.get(child_2).ok_or_else(|| {
        anyhow::anyhow!(
            "TreeTNCachedEvaluator::try_compute_branch_message_raw: missing assignment batch for child {:?}",
            child_2
        )
    })?;

    let dims = tensor.dims();
    let parent_dim = dims[parent_axis];
    let child_dim_1 = dims[child_axis_1];
    let child_dim_2 = dims[child_axis_2];
    let strides = [
        1usize,
        dims[0],
        dims[0].checked_mul(dims[1]).ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
        dims[0]
            .checked_mul(dims[1])
            .and_then(|value| value.checked_mul(dims[2]))
            .ok_or_else(|| anyhow::anyhow!("branch tensor strides overflow usize"))?,
    ];
    let spec = BranchContractionSpec {
        strides,
        physical_axis,
        parent_axis,
        child_axis_1,
        child_axis_2,
        parent_dim,
        child_dim_1,
        child_dim_2,
    };
    let raw = tensor.to_vec::<f64>()?;

    let mut physical_values = Vec::with_capacity(points.len());
    let mut child_1_columns = Vec::with_capacity(child_dim_1 * points.len());
    let mut child_2_columns = Vec::with_capacity(child_dim_2 * points.len());
    for &point in points {
        let physical_value = value_at(
            values,
            entry.input_position,
            point,
            "TreeTNCachedEvaluator::try_compute_branch_message_raw",
        )?;
        physical_values.push(physical_value);

        let child_1_assignment = child_1_assignment_batch
            .point_to_assignment
            .get(point)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing child-1 assignment for point {point}"))?;
        let start_1 = child_1_assignment * child_dim_1;
        child_1_columns.extend_from_slice(
            child_1_values
                .get(start_1..start_1 + child_dim_1)
                .ok_or_else(|| anyhow::anyhow!("child-1 assignment is out of bounds"))?,
        );

        let child_2_assignment = child_2_assignment_batch
            .point_to_assignment
            .get(point)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("missing child-2 assignment for point {point}"))?;
        let start_2 = child_2_assignment * child_dim_2;
        child_2_columns.extend_from_slice(
            child_2_values
                .get(start_2..start_2 + child_dim_2)
                .ok_or_else(|| anyhow::anyhow!("child-2 assignment is out of bounds"))?,
        );
    }
    let result = Self::grouped_branch_message_contraction(
        spec,
        &raw,
        &physical_values,
        &child_1_columns,
        &child_2_columns,
    )?;
    Ok(Some(result))
}
```

Then a `raw_real_message_values` helper (shared with the complex counterpart's structure) extracting a real `Vec<f64>` from a `StackedMessage`, matching exactly the inline logic `try_compute_chain_message_raw` already has for `child_values` (lines around 1608-1625 as read during investigation: prefer `raw_values` if present, else convert a real, host-resident `tensor`, returning `Ok(None)` on complex/`contract_with_options`-produced messages) -- factor that inline block out of `try_compute_chain_message_raw` into this helper **only if** doing so doesn't change `try_compute_chain_message_raw`'s behavior (verify with its existing tests still green after the extraction); otherwise duplicate the block rather than risk an unintended behavior change to the untouched chain path.

Write `try_compute_branch_message_complex_raw` as the direct `Complex64` counterpart, mirroring `try_compute_chain_message_complex_raw`'s existing relationship to `try_compute_chain_message_raw` (same structure, `Complex64` values, no `tensor.is_complex()` early return).

Wire both into `get_or_compute_node_message`: in the `tensor_is_complex` branch, insert `try_compute_branch_message_complex_raw` between the existing `try_compute_chain_message_complex_raw` attempt and the `compute_stacked_message` fallback (same `if leaf.is_some() { leaf } else { chain_result }` chaining pattern, extended to `if chain_result.is_some() { chain_result } else { branch_result }`); mirror in the real (non-complex) branch with `try_compute_branch_message_raw`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::raw_branch_message_matches_generic_contraction -- --nocapture`
Expected: PASS. If it fails on a value mismatch, re-check `strides`/axis assignment against `try_compute_chain_message_raw`'s exact convention (`dims[0]` fastest) and `grouped_branch_message_contraction`'s row/column layout (Task 1 Step 4 already validated that function in isolation, so a mismatch here points at this method's tensor-slicing/assignment-lookup glue, not the contraction math itself).

- [ ] **Step 5: Add a complex-value variant, run full module, commit**

Mirror `raw_complex_chain_message_matches_generic_contraction` for `try_compute_branch_message_complex_raw`.

```bash
cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator:: --no-fail-fast
cargo fmt --all
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "feat(treetn): wire the two-child branch raw message path into get_or_compute_node_message"
```

---

### Task 3: End-to-end correctness on a real branching tree

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` (`tests` module)

**Interfaces:**
- Consumes: `star_tree()` (existing fixture), `TreeTNCachedEvaluator::evaluate_batched`, `tree.evaluate` (ground truth), `assert_scalars_close` (existing helper).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn cached_evaluator_matches_tree_evaluate_on_star_tree_with_fixed_leaf_center() {
    let (tree, indices) = star_tree();
    let values = vec![
        0, 0, 0, 0, //
        1, 1, 0, 1, //
        0, 1, 1, 0, //
        1, 0, 1, 1,
    ];
    let shape = [4, 4];
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();

    let expected = tree.evaluate(&indices, points).unwrap();
    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let actual = evaluator.evaluate_batched(points).unwrap();

    assert_scalars_close(&actual, &expected);
}
```

This is deliberately close to the existing `cached_evaluator_matches_tree_evaluate_on_star_tree` test (check whether that test already exists and already fixes `center: Some(1)` or lets greedy search pick it -- read it first; if it already covers this exact scenario, skip this task and note in the commit message that Task 2's dispatch change is already covered by existing end-to-end tests instead of adding a redundant one).

- [ ] **Step 2: Run to verify it currently passes (it should -- Task 2 didn't change correctness, only which internal path computes the same result) and would have failed before Task 2 if the new path had a bug**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::cached_evaluator_matches_tree_evaluate_on_star_tree_with_fixed_leaf_center -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs
git commit -m "test(treetn): add end-to-end star-tree coverage pinning the branch raw path's center choice"
```

---

### Task 4: Full crate verification, update the diagnostic to show before/after, worklog

**Files:**
- Test: full `tensor4all-treetn` suite
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` (update the temporary diagnostic in place; keep it, matching how `frames.rs`'s equivalent diagnostic became a kept measurement in the prior fix)
- Create: `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md`

- [ ] **Step 1: Full crate verification**

```bash
cargo fmt --all -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --no-fail-fast
cargo doc --manifest-path crates/tensor4all-treetn/Cargo.toml --no-deps
```

- [ ] **Step 2: Re-run the diagnostic to measure the fix**

Run: `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::cached_evaluator::tests::diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk -- --ignored --nocapture`

Record the new `ratio (comb/chain)` -- compare against the pre-fix 25.84x. Update the test's doc comment from "TEMPORARY... revert after root-cause confirmation" to a permanent doc comment describing it as a kept regression-style measurement (mirroring `message_cache_wall_time_on_realistic_floating_zone_walk`'s own framing: "measurement tooling rather than a wall-clock regression assertion").

- [ ] **Step 3: Write the worklog**

Create `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md`: root cause (this plan's own root cause section), the 25.84x pre-fix measurement, the fix (Tasks 1-3), the post-fix measurement from Step 2, and an explicit note that this is a *different* crate/bug from `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`'s fix -- the two are structurally analogous but independent, and closing this one is what should actually move `pi_rtau` wall time on the downstream gw-rs `aci_global_guard=true` pipeline, per the user's own re-run finding that the first fix alone did not.

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md
git commit -m "test(treetn): confirm branch message fix closes the chain-vs-comb gap, add worklog"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the primitive; Task 2 covers dispatch wiring for both real/complex; Task 3 covers end-to-end correctness through the public `evaluate_batched` API; Task 4 covers full verification plus the before/after measurement the investigation promised.
- **Placeholder scan:** Task 3's exact test body is conditional on checking existing coverage first -- this is an explicit, actionable instruction ("read it first; if it already covers this, skip and note why"), not a vague placeholder, and is the one deliberate exception to "no placeholders" since it depends on a file-content check the plan author (me) did not exhaustively perform for every existing test in a 4787-line file before writing this plan.
- **Type consistency:** `BranchContractionSpec`/`grouped_branch_message_contraction`'s signature (Task 1) is used identically by `try_compute_branch_message_raw`/`_complex_raw` (Task 2). Output/input column layouts (point-major) are stated once and referenced consistently.
- **Not in scope:** nodes with 0, 1, or 3+ children; `compute_stacked_message`/`contract_with_options` themselves; message-cache internals (`PackedMessageCache`, `get_or_compute_batch`); the `tensor4all-treeaci/frames.rs` fix from the prior plan (already shipped, separate bug).
