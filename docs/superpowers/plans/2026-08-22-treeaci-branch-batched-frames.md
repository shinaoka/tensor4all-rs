# TreeACI Branch-Point Batched Frame Contraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the fixed per-oracle-call overhead at branching (2-incoming-edge) tree nodes reported in issue #671 by giving `InputFrameStore::candidate_frames_for_edge` and `FrameBuilder::compute_batch` a BLAS-batched contraction path for exactly-two-incoming-edge directed edges, instead of falling back to the scalar `accumulate_incoming` walk per candidate/sample.

**Architecture:** Add one new low-level primitive, `two_incoming_core_matrix_batched`, that contracts a node's core against two incoming-edge candidate-frame matrices (`v1`: `D1 x N1`, `v2`: `D2 x N2`) via `D2 + 1` BLAS `mat_mul` calls, producing every `(n1, n2)` combination in one shot. It is built by looping the *existing* `single_incoming_core_matrix` + `contract_prepared_core_batched` pair over the second incoming edge's dimension, then doing one more `contract_prepared_core_batched` call to fold in the second edge — no new BLAS-call machinery, no hand-rolled multi-axis transpose. Wire this primitive into both existing dispatch points (`candidate_frames_for_edge`'s pivot-search-candidate path, `FrameBuilder::compute_batch`'s sample-materialization path) as a new middle case between the existing 1-incoming (batched) and 0-or-3+-incoming (scalar) cases. Nodes with 3+ incoming edges (degree ≥ 5) keep using the scalar fallback — out of scope for this plan, since issue #671's reported topology (3-arm comb, i.e. exactly 2 incoming edges at every hub) does not need it, and generalizing further needs an inter-step transpose the 2-incoming case avoids.

**Tech Stack:** Rust, `tensor4all_tensorbackend::Matrix` + `mat_mul` (already used by the existing single-incoming batched path — no new dependency).

**Spec:** This plan's own "Architecture" section above (root-caused and empirically validated in-session against gw-rs issue #671; see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md` for the prior, deliberately-deferred documentation of this exact gap).

## Global Constraints

- Crate-scoped commands only (`--manifest-path crates/tensor4all-treeaci/Cargo.toml` or `-p tensor4all-treeaci`); do not build the full workspace.
- `cargo test`/`cargo nextest run` for this crate must use `--release` (debug-mode tree-ACI tests are slow).
- No `unwrap()`/`expect()` in library code (test code may use them, matching existing test style in `frames/tests/mod.rs`).
- `cargo fmt --all` before considering any task done.
- New public-crate-internal (`pub(crate)`) functions still need doc comments per this repo's rustdoc standard; free functions in `frames.rs` follow the existing terse-comment style already used for `single_incoming_core_matrix`/`contract_prepared_core_batched`.
- Every new dispatch branch must be covered by a test that asserts bit-for-bit (or, if that turns out not to hold at f64 precision once measured, `approx::assert_abs_diff_eq!` with a tight tolerance — decide from what the TDD step actually shows, do not assume) equality against the untouched scalar ground truth (`contract_prepared_core`/`candidate_frame`), mirroring the existing `..._falls_back_on_a_branch_edge_with_two_incoming_edges` test convention.
- Do not touch `accumulate_incoming`, `contract_prepared_core`, or the 0-/3+-incoming scalar fallback behavior — this plan only adds a new dispatch branch for exactly 2 incoming edges.

---

## File Structure

- Modify: `crates/tensor4all-treeaci/src/frames.rs`
  - New free function `two_incoming_core_matrix_batched` (the BLAS primitive), placed after `contract_prepared_core_batched`.
  - New method `InputFrameStore::candidate_frames_for_edge_two_incoming`, placed after `candidate_frames_for_edge`.
  - New method `FrameBuilder::compute_batch_two_incoming`, placed after `compute_batch`.
  - Dispatch edits inside `candidate_frames_for_edge` and `compute_batch`.
  - Doc-comment updates on both dispatching functions.
- Modify: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`
  - Rename/adapt `candidate_frames_for_edge_falls_back_on_a_branch_edge_with_two_incoming_edges` → `candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges` (it now exercises the new batched path, not a fallback).
  - Rename/adapt `compute_batch_falls_back_correctly_on_a_branch_edge` → `compute_batch_batches_a_branch_edge_with_two_incoming_edges`.
  - New fixture `four_arm_star_tree_for_three_incoming_fallback` (degree-4 hub, so one directed edge has exactly 3 incoming edges) plus two new tests asserting the scalar fallback is still used for 3 incoming edges, for both call sites.
  - New isolated unit test for `two_incoming_core_matrix_batched` itself.
- Create: `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md` — root cause, fix, before/after measurement, following this repo's existing treeaci worklog convention.

## Global Constraints Recap for Every Task

Every task below ends with, at minimum: `cargo fmt --all -- --check` clean and `cargo test --release -p tensor4all-treeaci --no-fail-fast` green, before moving to the next task.

---

### Task 1: `two_incoming_core_matrix_batched` primitive

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs` (insert after `contract_prepared_core_batched`, i.e. after the closing `}` currently at line 1165, before `fn outgoing_bond<'a, V: TreeAciNode>` at line 1167)
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs` (append near the end, after the existing `two_node_frames_are_exact_for_real_and_complex_inputs`-style tests, or directly after the `star_tree_for_fallback_dispatch` fixture — exact placement is not load-bearing, put it near `star_tree_for_fallback_dispatch` since it reuses that fixture)

**Interfaces:**
- Produces: `fn two_incoming_core_matrix_batched<T: TreeAciScalar>(core: &PreparedCore<T>, outgoing_axis: usize, incoming_axis_1: usize, incoming_axis_2: usize, physical_base_offset: usize, outgoing_dim: usize, incoming_dim_1: usize, incoming_dim_2: usize, v1: &Matrix<T>, v2: &Matrix<T>) -> Result<Matrix<T>>` — `v1` is `incoming_dim_1 x n1`, `v2` is `incoming_dim_2 x n2`; returns a `(outgoing_dim * n1) x n2` matrix where `result[[outgoing_dim * n1_index + out, n2_index]]` equals what `contract_prepared_core` would produce at `out` for the `(n1_index, n2_index)` pair.
- Consumes: existing `single_incoming_core_matrix`, `contract_prepared_core_batched`, `PreparedCore` (all already in `frames.rs`).

- [ ] **Step 1: Write the failing test**

Add to `crates/tensor4all-treeaci/src/frames/tests/mod.rs`, near `star_tree_for_fallback_dispatch`:

```rust
#[test]
fn two_incoming_core_matrix_batched_matches_scalar_contraction_on_every_pair() {
    let inputs = vec![star_tree_for_fallback_dispatch()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let cores = super::prepare_cores::<f64, usize>(&inputs[0], &problem).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("star tree must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);
    let incoming_edge_1 = directed.incoming_to_from[0];
    let incoming_edge_2 = directed.incoming_to_from[1];

    let node = *problem.node_positions.get(&directed.from).unwrap();
    let core = &cores[node];
    let outgoing = super::outgoing_bond(&inputs[0], &problem, edge).unwrap();
    let outgoing_axis = super::axis_of(&core.indices, outgoing).unwrap();
    let incoming_bond_1 = super::outgoing_bond(&inputs[0], &problem, incoming_edge_1).unwrap();
    let incoming_bond_2 = super::outgoing_bond(&inputs[0], &problem, incoming_edge_2).unwrap();
    let incoming_axis_1 = super::axis_of(&core.indices, incoming_bond_1).unwrap();
    let incoming_axis_2 = super::axis_of(&core.indices, incoming_bond_2).unwrap();
    let outgoing_dim = core.dims[outgoing_axis];
    let incoming_dim_1 = core.dims[incoming_axis_1];
    let incoming_dim_2 = core.dims[incoming_axis_2];

    // Two arbitrary, non-trivial frame vectors per incoming edge (n1=2, n2=2)
    // so the test exercises real cross-combination behavior, not a
    // degenerate 1-candidate case.
    let v1_cols = [vec![1.0, 0.5], vec![0.25, 2.0]];
    let v2_cols = [vec![3.0, -1.0], vec![0.1, 4.0]];
    let mut v1_data = Vec::new();
    for col in &v1_cols {
        v1_data.extend(col.iter().copied());
    }
    let v1 = tensor4all_tensorbackend::Matrix::from_col_major_vec(incoming_dim_1, 2, v1_data);
    let mut v2_data = Vec::new();
    for col in &v2_cols {
        v2_data.extend(col.iter().copied());
    }
    let v2 = tensor4all_tensorbackend::Matrix::from_col_major_vec(incoming_dim_2, 2, v2_data);

    let batched = super::two_incoming_core_matrix_batched(
        core,
        outgoing_axis,
        incoming_axis_1,
        incoming_axis_2,
        0,
        outgoing_dim,
        incoming_dim_1,
        incoming_dim_2,
        &v1,
        &v2,
    )
    .unwrap();

    for (n1, v1_vec) in v1_cols.iter().enumerate() {
        for (n2, v2_vec) in v2_cols.iter().enumerate() {
            let incoming_frames = vec![
                (incoming_edge_1, v1_vec.clone()),
                (incoming_edge_2, v2_vec.clone()),
            ];
            let expected = super::contract_prepared_core(
                &inputs[0],
                &problem,
                &cores,
                edge,
                0,
                &incoming_frames,
            )
            .unwrap();
            let actual: Vec<f64> = (0..outgoing_dim)
                .map(|out| batched[[out + outgoing_dim * n1, n2]])
                .collect();
            assert_eq!(
                actual, expected,
                "mismatch at (n1={n1}, n2={n2})"
            );
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci --lib frames::tests::two_incoming_core_matrix_batched_matches_scalar_contraction_on_every_pair`
Expected: FAIL to compile — `two_incoming_core_matrix_batched` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Insert into `crates/tensor4all-treeaci/src/frames.rs`, immediately after `contract_prepared_core_batched`'s closing `}` (currently line 1165) and before `fn outgoing_bond<'a, V: TreeAciNode>`:

```rust
/// Contracts a core slice's two incoming axes against batches of candidate
/// frame vectors for both incoming edges, computing every combination in
/// the cartesian product of `v1`'s and `v2`'s columns via `incoming_dim_2 + 1`
/// BLAS `mat_mul` calls (`incoming_dim_2` calls fold in `v1` one slice of the
/// second axis at a time, then one final call folds in `v2`) instead of one
/// scalar [`accumulate_incoming`] walk per `(n1, n2)` combination.
///
/// `v1` is `incoming_dim_1 x n1`, `v2` is `incoming_dim_2 x n2`. Returns an
/// `(outgoing_dim * n1) x n2` matrix: column `n2`, rows
/// `[outgoing_dim * n1_index, outgoing_dim * (n1_index + 1))`, holds the
/// `outgoing_dim`-length frame vector [`contract_prepared_core`] would
/// produce for the `(n1_index, n2)` candidate alone.
fn two_incoming_core_matrix_batched<T: TreeAciScalar>(
    core: &PreparedCore<T>,
    outgoing_axis: usize,
    incoming_axis_1: usize,
    incoming_axis_2: usize,
    physical_base_offset: usize,
    outgoing_dim: usize,
    incoming_dim_1: usize,
    incoming_dim_2: usize,
    v1: &Matrix<T>,
    v2: &Matrix<T>,
) -> Result<Matrix<T>> {
    let n1 = v1.ncols();
    let stride_2 = core.strides[incoming_axis_2];
    let mut stage1_data = Vec::with_capacity(outgoing_dim * n1 * incoming_dim_2);
    for i2 in 0..incoming_dim_2 {
        let core_matrix = single_incoming_core_matrix(
            core,
            outgoing_axis,
            incoming_axis_1,
            physical_base_offset + i2 * stride_2,
            outgoing_dim,
            incoming_dim_1,
        );
        let stage1 = contract_prepared_core_batched(&core_matrix, v1)?;
        stage1_data.extend(stage1.into_col_major_vec());
    }
    let stage1_matrix = Matrix::from_col_major_vec(outgoing_dim * n1, incoming_dim_2, stage1_data);
    contract_prepared_core_batched(&stage1_matrix, v2)
}
```

Also make `contract_prepared_core`, `outgoing_bond`, `axis_of`, and `prepare_cores` reachable from the test module at their current `pub(crate)`/private visibility — they already are, since `frames/tests/mod.rs` uses `super::` to reach other private items in this file (see existing use of `super::InputFrameStore`, `super::prepare_cores` pattern already present in `build_frame_builder`). No visibility changes should be needed; if the compiler disagrees, add `pub(crate)` (or `pub(super)`) only to the specific item that fails to resolve, not broadly.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --release -p tensor4all-treeaci --lib frames::tests::two_incoming_core_matrix_batched_matches_scalar_contraction_on_every_pair -- --nocapture`
Expected: PASS. If it fails with a numeric mismatch (not a compile error), the bug is in the offset/reshape bookkeeping in `two_incoming_core_matrix_batched` — check `stride_2` is `core.strides[incoming_axis_2]` (not axis_1), and that `stage1_data` is appended in `i2` order (so `stage1_matrix`'s columns are indexed by `i2`, matching `contract_prepared_core_batched`'s expectation that its second argument's rows are the axis being contracted).

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/frames/tests/mod.rs
git commit -m "feat(treeaci): add batched two-incoming-edge core contraction primitive"
```

---

### Task 2: Wire into `candidate_frames_for_edge` (pivot-search candidate path)

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs` (`candidate_frames_for_edge`, currently lines 481-642; insert new method `candidate_frames_for_edge_two_incoming` after it, before `candidate_frame` at line 644)
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs` (adapt the existing branch test at lines 653-711; add a new 3-incoming fixture/test)

**Interfaces:**
- Consumes: `two_incoming_core_matrix_batched` (Task 1), `InputFrameStore::frame_values` (existing), `InputFrameStore::candidate_cache`/`candidate_cache_bytes` (existing fields).
- Produces: `InputFrameStore::candidate_frames_for_edge_two_incoming(&self, inputs, problem, input, directed_edge, candidates: &[ComponentSample]) -> Result<Vec<Vec<T>>>` — same contract as `candidate_frames_for_edge` (one frame vector per input candidate, same order), used only when the edge has exactly two incoming edges.

- [ ] **Step 1: Adapt the existing branch test to assert against the new path (still red — the code hasn't changed yet)**

In `crates/tensor4all-treeaci/src/frames/tests/mod.rs`, rename
`candidate_frames_for_edge_falls_back_on_a_branch_edge_with_two_incoming_edges`
to `candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges` and update its doc comment (the one above `star_tree_for_fallback_dispatch` references it by name, at line 583 — update that cross-reference too). The test body's assertions (`dispatched == scalar`) stay exactly as they are — they already compare `candidate_frames_for_edge`'s output against `candidate_frame`'s scalar output, which is exactly what proves the new batched path is correct once wired in. This step is a rename/doc-only change; it does not need to "fail" first since the behavior under test doesn't change yet — skip to Step 2.

- [ ] **Step 2: Write the new 3-incoming-edges-still-scalar fixture and test**

Add to `crates/tensor4all-treeaci/src/frames/tests/mod.rs`, after the (renamed) two-incoming test:

```rust
/// 4-arm star: hub node 0 with three leaves plus a fourth arm, so directed
/// edge `0 -> 1` has exactly three incoming edges (`2 -> 0`, `3 -> 0`,
/// `4 -> 0`). Used to pin that 3+-incoming-edge dispatch still uses the
/// scalar fallback after Task 2/3 add a batched path for exactly two.
fn four_arm_star_tree_for_three_incoming_fallback() -> TreeTN<IdxTensor, usize> {
    let s0 = DynIndex::new_dyn(1);
    let s1 = DynIndex::new_dyn(1);
    let s2 = DynIndex::new_dyn(2);
    let s3 = DynIndex::new_dyn(2);
    let s4 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond02 = DynIndex::new_dyn(2);
    let bond03 = DynIndex::new_dyn(2);
    let bond04 = DynIndex::new_dyn(2);

    let node0 = IdxTensor::from_dense(
        vec![s0, bond01.clone(), bond02.clone(), bond03.clone(), bond04.clone()],
        (1..=16).map(|value| value as f64).collect(),
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(vec![bond01, s1], vec![1.0, 2.0]).unwrap();
    let node2 = IdxTensor::from_dense(vec![bond02, s2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let node3 = IdxTensor::from_dense(vec![bond03, s3], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let node4 = IdxTensor::from_dense(vec![bond04, s4], vec![9.0, 10.0, 11.0, 12.0]).unwrap();

    TreeTN::from_tensors(vec![node0, node1, node2, node3, node4], vec![0, 1, 2, 3, 4]).unwrap()
}

#[test]
fn candidate_frames_for_edge_still_falls_back_on_three_incoming_edges() {
    let inputs = vec![four_arm_star_tree_for_three_incoming_fallback()];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("4-arm star must have a directed edge 0 -> 1");
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 3);

    let seeds = vec![vec![0, 0, 0, 0, 0], vec![0, 0, 1, 1, 1]];
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let mut candidates = Vec::new();
    let incoming: Vec<_> = directed.incoming_to_from.clone();
    for &id_a in &candidate_sets.ids[incoming[0]] {
        for &id_b in &candidate_sets.ids[incoming[1]] {
            for &id_c in &candidate_sets.ids[incoming[2]] {
                candidates.push(ComponentSample {
                    local_coordinate: 0,
                    incoming: vec![
                        (incoming[0], id_a),
                        (incoming[1], id_b),
                        (incoming[2], id_c),
                    ],
                });
            }
        }
    }

    let dispatched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let scalar = candidates
        .iter()
        .map(|candidate| {
            frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(dispatched, scalar);
}
```

- [ ] **Step 3: Run both tests to confirm current (pre-wiring) behavior**

Run: `cargo test --release -p tensor4all-treeaci --lib frames::tests::candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges frames::tests::candidate_frames_for_edge_still_falls_back_on_three_incoming_edges -- --nocapture`
Expected: both PASS already (the dispatch hasn't changed yet, so both still exercise the old scalar fallback and trivially agree with themselves). This confirms the test fixtures compile and are meaningful before the dispatch change makes the two-incoming one actually exercise new code.

- [ ] **Step 4: Add the new orchestration method and wire the dispatch**

In `crates/tensor4all-treeaci/src/frames.rs`, replace the fallback check at the top of `candidate_frames_for_edge` (currently):

```rust
        if directed.incoming_to_from.len() != 1 {
            return candidates
                .iter()
                .map(|candidate| {
                    self.candidate_frame(inputs, problem, input, directed_edge, candidate)
                })
                .collect();
        }
```

with:

```rust
        if directed.incoming_to_from.len() == 2 {
            return self.candidate_frames_for_edge_two_incoming(
                inputs,
                problem,
                input,
                directed_edge,
                candidates,
            );
        }
        if directed.incoming_to_from.len() != 1 {
            return candidates
                .iter()
                .map(|candidate| {
                    self.candidate_frame(inputs, problem, input, directed_edge, candidate)
                })
                .collect();
        }
```

Then insert this new method into the `impl<T: TreeAciScalar> InputFrameStore<T>` block, directly after `candidate_frames_for_edge`'s closing `}` (before `pub(crate) fn candidate_frame` at line 644):

```rust
    /// Batched counterpart to [`Self::candidate_frames_for_edge`]'s
    /// single-incoming-edge path, for directed edges whose source node has
    /// exactly two incoming edges (every hub of a 3-valent tree branch
    /// point). Groups candidates by `local_coordinate` exactly as the
    /// single-incoming path does, then for each group gathers the distinct
    /// sample ids referenced on each incoming edge, builds one frame-vector
    /// matrix per incoming edge, and contracts both via
    /// [`two_incoming_core_matrix_batched`] in one shot -- computing the
    /// full cartesian product of the group's distinct incoming ids (a
    /// superset of the group's actual candidates whenever the group is not
    /// already the full product) and reading back only the entries the
    /// group's candidates actually need.
    fn candidate_frames_for_edge_two_incoming<V: TreeAciNode>(
        &self,
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        candidates: &[ComponentSample],
    ) -> Result<Vec<Vec<T>>> {
        let directed =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate frame references an unknown directed edge",
                })?;
        let node =
            *problem
                .node_positions
                .get(&directed.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate source has no prepared node position",
                })?;
        let tree = inputs.get(input).ok_or(TreeAciError::InternalInvariant {
            message: "candidate frame references an unknown input",
        })?;
        let cores = self
            .cores
            .get(input)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate frame has no prepared input cores",
            })?;
        let core = cores.get(node).ok_or(TreeAciError::InternalInvariant {
            message: "candidate frame source node has no prepared core",
        })?;
        let outgoing = outgoing_bond(tree, problem, directed_edge)?;
        let outgoing_axis = axis_of(&core.indices, outgoing)?;
        let physical = &problem.physical[node];
        let physical_axes = physical
            .indices
            .iter()
            .map(|index| axis_of(&core.indices, index))
            .collect::<Result<Vec<_>>>()?;
        let incoming_edge_1 = directed.incoming_to_from[0];
        let incoming_edge_2 = directed.incoming_to_from[1];
        let incoming_bond_1 = outgoing_bond(tree, problem, incoming_edge_1)?;
        let incoming_bond_2 = outgoing_bond(tree, problem, incoming_edge_2)?;
        let incoming_axis_1 = axis_of(&core.indices, incoming_bond_1)?;
        let incoming_axis_2 = axis_of(&core.indices, incoming_bond_2)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim_1 = core.dims[incoming_axis_1];
        let incoming_dim_2 = core.dims[incoming_axis_2];

        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut results: Vec<Option<Vec<T>>> = vec![None; candidates.len()];
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let key: CandidateCacheKey = (
                input,
                directed_edge,
                candidate.local_coordinate,
                candidate.incoming.clone(),
            );
            if let Some(cached) = self.candidate_cache.borrow().get(&key) {
                #[cfg(test)]
                candidate_debug_stats::record_hit();
                results[candidate_index] = Some(cached.clone());
                continue;
            }
            #[cfg(test)]
            candidate_debug_stats::record_miss();
            if candidate.incoming.len() != 2
                || candidate.incoming[0].0 != incoming_edge_1
                || candidate.incoming[1].0 != incoming_edge_2
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "two-incoming-edge candidate does not match the edge's incoming order",
                });
            }
            groups
                .entry(candidate.local_coordinate)
                .or_default()
                .push(candidate_index);
        }

        for (local_coordinate, indices) in groups {
            let mut base_offset = 0usize;
            for (physical_axis, &axis) in physical_axes.iter().enumerate() {
                let wanted = (local_coordinate / physical.strides[physical_axis])
                    % physical.dims[physical_axis];
                base_offset += wanted * core.strides[axis];
            }

            let mut ids_1: Vec<SampleId> = Vec::new();
            let mut position_1: HashMap<SampleId, usize> = HashMap::new();
            let mut ids_2: Vec<SampleId> = Vec::new();
            let mut position_2: HashMap<SampleId, usize> = HashMap::new();
            for &candidate_index in &indices {
                let (_, sample_1) = candidates[candidate_index].incoming[0];
                let (_, sample_2) = candidates[candidate_index].incoming[1];
                position_1.entry(sample_1).or_insert_with(|| {
                    ids_1.push(sample_1);
                    ids_1.len() - 1
                });
                position_2.entry(sample_2).or_insert_with(|| {
                    ids_2.push(sample_2);
                    ids_2.len() - 1
                });
            }

            let mut v1_data = Vec::with_capacity(incoming_dim_1 * ids_1.len());
            for &sample in &ids_1 {
                let values = self.frame_values(input, incoming_edge_1, sample)?;
                if values.len() != incoming_dim_1 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v1_data.extend(values);
            }
            let v1 = Matrix::from_col_major_vec(incoming_dim_1, ids_1.len(), v1_data);

            let mut v2_data = Vec::with_capacity(incoming_dim_2 * ids_2.len());
            for &sample in &ids_2 {
                let values = self.frame_values(input, incoming_edge_2, sample)?;
                if values.len() != incoming_dim_2 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v2_data.extend(values);
            }
            let v2 = Matrix::from_col_major_vec(incoming_dim_2, ids_2.len(), v2_data);

            let batched = two_incoming_core_matrix_batched(
                core,
                outgoing_axis,
                incoming_axis_1,
                incoming_axis_2,
                base_offset,
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                &v1,
                &v2,
            )?;

            for &candidate_index in &indices {
                let (_, sample_1) = candidates[candidate_index].incoming[0];
                let (_, sample_2) = candidates[candidate_index].incoming[1];
                let n1 = position_1[&sample_1];
                let n2 = position_2[&sample_2];
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|out| batched[[out + outgoing_dim * n1, n2]])
                    .collect();
                let entry_bytes = values.len().saturating_mul(size_of::<T>());
                let candidate_bytes = self.candidate_cache_bytes.get();
                let projected = self
                    .retained_bytes
                    .saturating_add(candidate_bytes)
                    .saturating_add(entry_bytes);
                if projected <= problem.max_frame_bytes {
                    let candidate = &candidates[candidate_index];
                    let key: CandidateCacheKey = (
                        input,
                        directed_edge,
                        candidate.local_coordinate,
                        candidate.incoming.clone(),
                    );
                    self.candidate_cache_bytes
                        .set(candidate_bytes.saturating_add(entry_bytes));
                    self.candidate_cache
                        .borrow_mut()
                        .insert(key, values.clone());
                }
                results[candidate_index] = Some(values);
            }
        }

        results
            .into_iter()
            .map(|value| {
                value.ok_or(TreeAciError::InternalInvariant {
                    message: "two-incoming candidate frame batching left a candidate unfilled",
                })
            })
            .collect()
    }
```

- [ ] **Step 5: Run tests to verify the new path is correct**

Run: `cargo test --release -p tensor4all-treeaci --lib frames:: -- --nocapture`
Expected: PASS, including `candidate_frames_for_edge_batches_a_branch_edge_with_two_incoming_edges` (now genuinely exercising the new batched code, since `directed.incoming_to_from.len() == 2` on that fixture's edge `0 -> 1`) and `candidate_frames_for_edge_still_falls_back_on_three_incoming_edges` (unaffected, still scalar). If the two-incoming test fails on a value mismatch, re-check `candidate_frames_for_edge_two_incoming`'s `n1`/`n2` indexing against `two_incoming_core_matrix_batched`'s documented output layout (`out + outgoing_dim * n1`, column `n2`).

- [ ] **Step 6: Update the function's doc comment**

`candidate_frames_for_edge`'s doc comment (currently starting "Computes every candidate's frame vector for one input and directed edge, using the batched BLAS path ... when the edge's source node has exactly one incoming edge, and falling back to the scalar ... path ... otherwise.") needs updating to describe the new three-way dispatch. Replace the first paragraph with:

```rust
    /// Computes every candidate's frame vector for one input and directed
    /// edge. Dispatches to a batched BLAS path when the edge's source node
    /// has exactly one incoming edge (one `mat_mul` call per distinct
    /// `local_coordinate`) or exactly two incoming edges (see
    /// [`Self::candidate_frames_for_edge_two_incoming`] and
    /// [`two_incoming_core_matrix_batched`]), and falls back to the scalar
    /// [`Self::candidate_frame`] path (called once per candidate, still
    /// consulting/populating `candidate_cache`) for a leaf edge (zero
    /// incoming edges) or a node with three or more incoming edges (out of
    /// scope for the batched paths -- see issue #671 and
    /// `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`).
```

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/frames/tests/mod.rs
git commit -m "feat(treeaci): batch candidate_frames_for_edge's two-incoming-edge branch dispatch"
```

---

### Task 3: Wire into `FrameBuilder::compute_batch` (sample-materialization path)

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs` (`compute_batch`, currently lines 861-1006; insert new method `compute_batch_two_incoming` after it, before `outgoing_bond` method at line 1008)
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs` (adapt existing branch test around line 1125; add a new 3-incoming test reusing Task 2's fixture)

**Interfaces:**
- Consumes: `two_incoming_core_matrix_batched` (Task 1), `FrameBuilder::compute` (existing, for priming), `FrameBuilder::outgoing_bond` (existing), `FrameBuilder::memo` (existing field).
- Produces: `FrameBuilder::compute_batch_two_incoming(&mut self, edge: DirectedEdgeId, samples: std::ops::Range<SampleId>) -> Result<()>` — same contract as `compute_batch`: every result lands in `self.memo[edge]`.

- [ ] **Step 1: Rename the existing branch test**

In `crates/tensor4all-treeaci/src/frames/tests/mod.rs`, rename `compute_batch_falls_back_correctly_on_a_branch_edge` (around line 1125) to `compute_batch_batches_a_branch_edge_with_two_incoming_edges`. Update its doc comment and the cross-reference in `compute_batch`'s own doc comment (the one describing its dispatch) accordingly. Leave the test body's assertions unchanged for now — same rationale as Task 2 Step 1.

- [ ] **Step 2: Add a 3-incoming-edges-still-scalar test for `compute_batch`**

Add to `crates/tensor4all-treeaci/src/frames/tests/mod.rs`, near the renamed test:

```rust
#[test]
fn compute_batch_still_falls_back_on_three_incoming_edges() {
    let input = four_arm_star_tree_for_three_incoming_fallback();
    let problem =
        prepare_problem(std::slice::from_ref(&input), &TreeAciOptions::default()).unwrap();

    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .expect("4-arm star must have a directed edge 0 -> 1");
    assert_eq!(problem.directed_edges[edge].incoming_to_from.len(), 3);

    let seeds = vec![vec![0, 0, 0, 0, 0], vec![0, 0, 1, 1, 1]];
    let (arena, _candidates) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();

    let mut scalar_builder = build_frame_builder(&input, &problem, &arena);
    let sample_count = arena.directed_record_count(edge).unwrap();
    for sample in 0..sample_count {
        scalar_builder.compute(edge, sample).unwrap();
    }
    let scalar_values: Vec<Vec<f64>> = (0..sample_count)
        .map(|sample| scalar_builder.memo[edge][sample].clone().unwrap())
        .collect();

    let mut batched_builder = build_frame_builder(&input, &problem, &arena);
    batched_builder.compute_batch(edge, 0..sample_count).unwrap();
    let batched_values: Vec<Vec<f64>> = (0..sample_count)
        .map(|sample| batched_builder.memo[edge][sample].clone().unwrap())
        .collect();

    assert_eq!(batched_values, scalar_values);
}
```

- [ ] **Step 3: Run both tests to confirm current (pre-wiring) behavior**

Run: `cargo test --release -p tensor4all-treeaci --lib frames::tests::compute_batch_batches_a_branch_edge_with_two_incoming_edges frames::tests::compute_batch_still_falls_back_on_three_incoming_edges -- --nocapture`
Expected: both PASS (dispatch not yet changed).

- [ ] **Step 4: Add the new orchestration method and wire the dispatch**

In `crates/tensor4all-treeaci/src/frames.rs`, inside `FrameBuilder::compute_batch`, replace:

```rust
        let directed = &self.problem.directed_edges[edge];
        if directed.incoming_to_from.len() != 1 {
            for sample in samples {
                self.compute(edge, sample)?;
            }
            return Ok(());
        }
        let incoming_edge = directed.incoming_to_from[0];
```

with:

```rust
        let directed = &self.problem.directed_edges[edge];
        if directed.incoming_to_from.len() == 2 {
            return self.compute_batch_two_incoming(edge, samples);
        }
        if directed.incoming_to_from.len() != 1 {
            for sample in samples {
                self.compute(edge, sample)?;
            }
            return Ok(());
        }
        let incoming_edge = directed.incoming_to_from[0];
```

Then insert this new method into `impl<T: TreeAciScalar, V: TreeAciNode> FrameBuilder<'_, T, V>`, directly after `compute_batch`'s closing `}` (before `fn outgoing_bond(&self, ...)` at line 1008):

```rust
    /// Batched counterpart to [`Self::compute_batch`]'s single-incoming-edge
    /// path, for directed edges whose source node has exactly two incoming
    /// edges. Primes both incoming edges' needed samples via [`Self::compute`]
    /// (as `compute_batch` already does for its one incoming edge), then
    /// groups the pending samples by `local_coordinate` and contracts each
    /// group via [`two_incoming_core_matrix_batched`], mirroring
    /// [`InputFrameStore::candidate_frames_for_edge_two_incoming`]'s
    /// structure but reading incoming frame vectors from `self.memo` instead
    /// of a committed `InputFrameStore`.
    fn compute_batch_two_incoming(
        &mut self,
        edge: DirectedEdgeId,
        samples: std::ops::Range<SampleId>,
    ) -> Result<()> {
        let directed = &self.problem.directed_edges[edge];
        let incoming_edge_1 = directed.incoming_to_from[0];
        let incoming_edge_2 = directed.incoming_to_from[1];

        let mut pending: Vec<(SampleId, ComponentSample)> = Vec::new();
        for sample in samples {
            if self.memo[edge][sample].is_some() {
                continue;
            }
            let record = self.arena.record(edge, sample)?.clone();
            if record.incoming.len() != 2
                || record.incoming[0].0 != incoming_edge_1
                || record.incoming[1].0 != incoming_edge_2
            {
                return Err(TreeAciError::InternalInvariant {
                    message:
                        "two-incoming-edge sample does not have exactly two incoming samples on the expected edges",
                });
            }
            pending.push((sample, record));
        }
        if pending.is_empty() {
            return Ok(());
        }

        for (_, record) in &pending {
            self.compute(incoming_edge_1, record.incoming[0].1)?;
            self.compute(incoming_edge_2, record.incoming[1].1)?;
        }

        let node = *self.problem.node_positions.get(&directed.from).ok_or(
            TreeAciError::InternalInvariant {
                message: "frame source has no prepared node position",
            },
        )?;
        let core = &self.cores[node];
        let outgoing = self.outgoing_bond(edge)?;
        let outgoing_axis = axis_of(&core.indices, outgoing)?;
        let physical = &self.problem.physical[node];
        let physical_axes = physical
            .indices
            .iter()
            .map(|index| axis_of(&core.indices, index))
            .collect::<Result<Vec<_>>>()?;
        let incoming_bond_1 = self.outgoing_bond(incoming_edge_1)?;
        let incoming_bond_2 = self.outgoing_bond(incoming_edge_2)?;
        let incoming_axis_1 = axis_of(&core.indices, incoming_bond_1)?;
        let incoming_axis_2 = axis_of(&core.indices, incoming_bond_2)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim_1 = core.dims[incoming_axis_1];
        let incoming_dim_2 = core.dims[incoming_axis_2];

        let mut groups: std::collections::BTreeMap<usize, Vec<(SampleId, SampleId, SampleId)>> =
            std::collections::BTreeMap::new();
        for (sample, record) in &pending {
            let (_, sample_1) = record.incoming[0];
            let (_, sample_2) = record.incoming[1];
            groups
                .entry(record.local_coordinate)
                .or_default()
                .push((*sample, sample_1, sample_2));
        }

        for (local_coordinate, group_samples) in groups {
            let mut base_offset = 0usize;
            for (physical_axis, &axis) in physical_axes.iter().enumerate() {
                let wanted = (local_coordinate / physical.strides[physical_axis])
                    % physical.dims[physical_axis];
                base_offset += wanted * core.strides[axis];
            }

            let mut ids_1: Vec<SampleId> = Vec::new();
            let mut position_1: HashMap<SampleId, usize> = HashMap::new();
            let mut ids_2: Vec<SampleId> = Vec::new();
            let mut position_2: HashMap<SampleId, usize> = HashMap::new();
            for &(_, sample_1, sample_2) in &group_samples {
                position_1.entry(sample_1).or_insert_with(|| {
                    ids_1.push(sample_1);
                    ids_1.len() - 1
                });
                position_2.entry(sample_2).or_insert_with(|| {
                    ids_2.push(sample_2);
                    ids_2.len() - 1
                });
            }

            let mut v1_data = Vec::with_capacity(incoming_dim_1 * ids_1.len());
            for &sample_1 in &ids_1 {
                let values = self.memo[incoming_edge_1][sample_1].clone().ok_or(
                    TreeAciError::InternalInvariant {
                        message: "incoming sample frame was not memoized before batched contraction",
                    },
                )?;
                if values.len() != incoming_dim_1 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v1_data.extend(values);
            }
            let v1 = Matrix::from_col_major_vec(incoming_dim_1, ids_1.len(), v1_data);

            let mut v2_data = Vec::with_capacity(incoming_dim_2 * ids_2.len());
            for &sample_2 in &ids_2 {
                let values = self.memo[incoming_edge_2][sample_2].clone().ok_or(
                    TreeAciError::InternalInvariant {
                        message: "incoming sample frame was not memoized before batched contraction",
                    },
                )?;
                if values.len() != incoming_dim_2 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v2_data.extend(values);
            }
            let v2 = Matrix::from_col_major_vec(incoming_dim_2, ids_2.len(), v2_data);

            let batched = two_incoming_core_matrix_batched(
                core,
                outgoing_axis,
                incoming_axis_1,
                incoming_axis_2,
                base_offset,
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                &v1,
                &v2,
            )?;

            for (sample, sample_1, sample_2) in group_samples {
                let n1 = position_1[&sample_1];
                let n2 = position_2[&sample_2];
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|out| batched[[out + outgoing_dim * n1, n2]])
                    .collect();
                #[cfg(test)]
                debug_stats::record_batched_compute_call();
                let slot = self
                    .memo
                    .get_mut(edge)
                    .and_then(|s| s.get_mut(sample))
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "computed frame has no memoization slot",
                    })?;
                *slot = Some(values);
            }
        }
        Ok(())
    }
```

- [ ] **Step 5: Run tests to verify the new path is correct**

Run: `cargo test --release -p tensor4all-treeaci --lib frames:: -- --nocapture`
Expected: PASS, including `compute_batch_batches_a_branch_edge_with_two_incoming_edges` (now genuinely exercising the new code) and `compute_batch_still_falls_back_on_three_incoming_edges`. Also re-run the full crate suite once here, since `compute_batch` is used by `InputFrameStore::from_samples`/`extend`, which many other existing tests (`extend_matches_a_full_rebuild_on_the_grown_arena`, `extend_reuses_unchanged_edges_via_rc_instead_of_rebuilding_them`, etc.) depend on transitively for any fixture with a 2-incoming-edge node (e.g. `y_tree`, `star_tree_for_fallback_dispatch`):

Run: `cargo test --release -p tensor4all-treeaci --no-fail-fast`
Expected: all green, no regressions in the extend/rebuild parity tests.

- [ ] **Step 6: Update `compute_batch`'s doc comment**

`compute_batch`'s doc comment currently says: "using the batched BLAS path ... when `edge`'s source node has exactly one incoming edge ... and falling back to [`Self::compute`] per sample otherwise (0 or >=2 incoming edges)." Update the parenthetical to: "and falling back to [`Self::compute`] per sample otherwise (0 incoming edges, or 3+)." and add a sentence referencing the new two-incoming case and `compute_batch_two_incoming`, matching Task 2 Step 6's edit to `candidate_frames_for_edge`'s doc comment in tone.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/frames/tests/mod.rs
git commit -m "feat(treeaci): batch compute_batch's two-incoming-edge branch dispatch"
```

---

### Task 4: Full crate verification, worklog, and issue-facing measurement

**Files:**
- Test: full `tensor4all-treeaci` suite
- Create: `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces: a worklog entry following this repo's existing treeaci worklog convention (see `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`, `docs/worklogs/2026-08-21-treeaci-message-cache-budget.md` for style/structure).

- [ ] **Step 1: Full crate verification**

```bash
cargo fmt --all -- --check
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
cargo test --release -p tensor4all-treeaci --no-fail-fast
cargo doc -p tensor4all-treeaci --no-deps
```

Expected: all clean/green. If clippy flags the new `HashMap<SampleId, usize>` closures (`or_insert_with` capturing `ids_1`/`ids_2` by mutable reference) with a lint about entry-API style, resolve by following clippy's suggestion rather than allowing the lint, unless the suggestion would change behavior (it should not here).

- [ ] **Step 2: Measure the actual improvement**

Reuse the star fixture at larger scale (mirroring the investigation's diagnostic, this time comparing the *now-fixed* `candidate_frames_for_edge` against a manual per-candidate loop over the still-unchanged `candidate_frame` scalar method, both reachable through the existing public-to-the-module API — no temporary test code needed, since both entry points are permanent). Write a small `#[ignore]`d benchmark-style test (kept this time, not reverted, since it directly answers issue #671's "Suggested next step" for a reproducible branching-tree measurement) in `crates/tensor4all-treeaci/src/frames/tests/mod.rs`:

```rust
/// Reproduces #671's core question directly inside the crate: at a
/// 2-incoming-edge branch point, how much faster is the batched path
/// (`candidate_frames_for_edge`, since Task 2) than looping the scalar
/// `candidate_frame` path per candidate (the code path every branch point
/// used before this fix, and what 3+-incoming nodes still use today)? Not
/// run by default (`#[ignore]`): it is a timing report, not a correctness
/// gate. Run explicitly with `--ignored --nocapture`.
#[test]
#[ignore]
fn branch_point_batched_speedup_vs_scalar_at_realistic_scale() {
    use std::time::Instant;

    let s0 = DynIndex::new_dyn(1);
    let s1 = DynIndex::new_dyn(1);
    let m = 40usize;
    let d = 32usize;
    let s2 = DynIndex::new_dyn(m);
    let s3 = DynIndex::new_dyn(m);
    let bond01 = DynIndex::new_dyn(4);
    let bond02 = DynIndex::new_dyn(d);
    let bond03 = DynIndex::new_dyn(d);
    let node0 = IdxTensor::from_dense(
        vec![s0, bond01.clone(), bond02.clone(), bond03.clone()],
        (0..4 * d * d).map(|v| v as f64).collect(),
    )
    .unwrap();
    let node1 = IdxTensor::from_dense(vec![bond01, s1], (0..4).map(|v| v as f64).collect()).unwrap();
    let node2 =
        IdxTensor::from_dense(vec![bond02, s2], (0..d * m).map(|v| v as f64).collect()).unwrap();
    let node3 =
        IdxTensor::from_dense(vec![bond03, s3], (0..d * m).map(|v| v as f64).collect()).unwrap();
    let star = TreeTN::from_tensors(vec![node0, node1, node2, node3], vec![0, 1, 2, 3]).unwrap();

    let inputs = vec![star];
    let options = TreeAciOptions::default();
    let problem = prepare_problem(&inputs, &options).unwrap();
    let edge = problem
        .directed_edges
        .iter()
        .position(|arc| arc.from == 0 && arc.to == 1)
        .unwrap();
    let directed = &problem.directed_edges[edge];
    assert_eq!(directed.incoming_to_from.len(), 2);

    let seeds: Vec<Vec<usize>> = (0..m).map(|i| vec![0, 0, i, i]).collect();
    let (arena, candidate_sets) = SampleArena::from_global_seeds(&problem, &seeds).unwrap();
    let frames = InputFrameStore::<f64>::from_samples(&inputs, &problem, &arena).unwrap();

    let incoming_a = directed.incoming_to_from[0];
    let incoming_b = directed.incoming_to_from[1];
    let ids_a = &candidate_sets.ids[incoming_a];
    let ids_b = &candidate_sets.ids[incoming_b];
    let mut candidates = Vec::new();
    for &id_a in ids_a {
        for &id_b in ids_b {
            candidates.push(ComponentSample {
                local_coordinate: 0,
                incoming: vec![(incoming_a, id_a), (incoming_b, id_b)],
            });
        }
    }

    let batched_start = Instant::now();
    let batched = frames
        .candidate_frames_for_edge(&inputs, &problem, 0, edge, &candidates)
        .unwrap();
    let batched_elapsed = batched_start.elapsed();

    let scalar_start = Instant::now();
    let scalar: Vec<Vec<f64>> = candidates
        .iter()
        .map(|candidate| {
            frames
                .candidate_frame(&inputs, &problem, 0, edge, candidate)
                .unwrap()
        })
        .collect();
    let scalar_elapsed = scalar_start.elapsed();

    assert_eq!(batched, scalar);
    eprintln!(
        "batched (Task 2 fix): {batched_elapsed:?}\nscalar (old fallback, still used for 3+ incoming): {scalar_elapsed:?}\nspeedup: {:.2}x",
        scalar_elapsed.as_secs_f64() / batched_elapsed.as_secs_f64()
    );
}
```

Run: `cargo test --release -p tensor4all-treeaci --lib frames::tests::branch_point_batched_speedup_vs_scalar_at_realistic_scale -- --ignored --nocapture`

Record the printed speedup number for the worklog (Step 3). Note this test intentionally shares `assert_eq!(batched, scalar)` as a correctness belt-and-braces check even though Tasks 1-3's tests already cover this; keep it since it's the number that answers #671.

- [ ] **Step 3: Write the worklog**

Create `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md` with: the root cause (dispatch condition in `frames.rs`, cross-referencing `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s prior note that this was left unresolved), the matched-FLOP diagnostic numbers from the investigation (1.76x/3.61x/4.17x at D=64/256/1024), the fix (this plan's Tasks 1-3), and the Step 2 measurement's actual speedup number -- fill in the real number from the run, do not estimate it. Follow the existing worklog files' structure (problem statement, root cause, fix, verification, results table).

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/frames/tests/mod.rs docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md
git commit -m "test(treeaci): add branch-point batched-vs-scalar speedup measurement and worklog for #671"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1 covers the primitive; Tasks 2-3 cover both call sites the investigation identified (`candidate_frames_for_edge` and `compute_batch`); Task 4 covers full verification plus the issue-facing measurement/worklog the investigation promised. The 0-/3+-incoming scalar fallback is explicitly left untouched and pinned by new tests (Task 2 Step 2, Task 3 Step 2) rather than silently left unverified.
- **Placeholder scan:** every step has literal, complete code; no "TBD"/"similar to Task N" shortcuts.
- **Type consistency:** `two_incoming_core_matrix_batched`'s signature (Task 1) is used identically in Task 2's `candidate_frames_for_edge_two_incoming` and Task 3's `compute_batch_two_incoming`. `SampleId = usize` (from `samples.rs`) is used consistently for `position_1`/`position_2`/`ids_1`/`ids_2` across both call sites.
- **Not in scope:** 3+-incoming-edge nodes, changing `accumulate_incoming`/`contract_prepared_core` themselves, and any change to `local_update.rs`/`schedule.rs` (the investigation found the bottleneck entirely inside `frames.rs`'s dispatch, not in how callers use it).
