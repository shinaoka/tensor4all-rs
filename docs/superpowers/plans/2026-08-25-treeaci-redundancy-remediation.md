# TreeACI Redundancy Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the confirmed redundant work remaining in `tensor4all-treeaci` while preserving candidate ordering, floating-point contraction order, transaction atomicity, and low-temperature/high-`R` numerical behavior.

**Architecture:** Work from the owning TreeACI abstraction, one candidate at a time. Mechanical cleanups are isolated from convergence changes; every frame or Guard optimization keeps a real/complex scalar oracle and the existing saved-artifact comparison as a gate. The global-Guard start-point reuse is deferred until the false-convergence regression in #687 has a deterministic test, because changing its evaluated-point state machine can affect low-temperature large-`R` behavior.

**Tech Stack:** Rust, `tensor4all-treeaci`, TreeTN/TreeACI internal tests, release-mode Cargo tests, Git history, and the retained `gw-rs` checkpoints.

**Spec:** GitHub issue #686, with correctness scope cross-checked against issue #687 and `docs/worklogs/2026-08-23-treeaci-audit-remediation.md`.

## Global Constraints

- Do not change numerical reduction order, candidate ordering, convergence thresholds, or transaction rollback semantics without a dedicated regression oracle.
- Preserve real and `Complex64` behavior for every shared numerical path.
- Keep the 3+ incoming scalar fallback as the correctness reference until an equivalent implementation is proven.
- Use release mode for numerical/performance-sensitive verification, and do not relax tolerances.
- Record intentional retained costs with an `// INVARIANT:` marker or a worklog entry.
- Do not push or create a PR without explicit user approval.

---

### Task 1: Remove dead schedule rank bookkeeping

**Files:**
- Modify: `crates/tensor4all-treeaci/src/schedule.rs`
- Test: `crates/tensor4all-treeaci/src/schedule/tests/mod.rs` (only if compilation exposes a stale test reference)

**Interfaces:**
- Consumes: `run_directional_pass` and its private `PassReport`.
- Produces: the same pass behavior and report data, without the unused `rank_changed` field or `ranks_before` clone.

- [x] **Step 1: Confirm the dead-data boundary.** Verify that `rank_changed` and `ranks_before` have no production or test consumers other than the report construction.
- [x] **Step 2: Remove only the field, clone, comparison, and initializer.** Leave `updated_edges`, deferred canonicalization, edge update order, and all convergence vectors unchanged.
- [x] **Step 3: Run the focused schedule tests.**

  ```bash
  cargo test --release -p tensor4all-treeaci schedule --no-fail-fast
  ```

- [x] **Step 4: Run the TreeACI crate tests.**

  ```bash
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  ```

### Task 2: Hoist invariant physical-offset arithmetic

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`

**Interfaces:**
- Consumes: `single_incoming_all_physical_core_matrix` and the existing scalar-vs-batched frame tests.
- Produces: identical column-major matrix values while computing each local physical offset once per local coordinate rather than once per incoming value.

- [x] **Step 1: Add or strengthen a real/complex matrix oracle** using the existing scalar contraction path and a fixture with multiple physical axes and nontrivial axis order.
- [x] **Step 2: Run that oracle before changing production code and record its baseline result.**
- [x] **Step 3: Move the `physical_offset` calculation outside the `incoming_value` loop without changing loop nesting, source offsets, or write order.**
- [x] **Step 4: Run the oracle and the release frame tests.**

  ```bash
  cargo test --release -p tensor4all-treeaci frames --no-fail-fast
  ```

### Task 3: Reuse the already computed Guard evaluation hint

**Files:**
- Modify: `crates/tensor4all-treeaci/src/global_guard.rs`
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`

**Interfaces:**
- Consumes: `InputEvaluators::evaluate_expanded` and `GuardOutputEvaluator::evaluate_expanded`.
- Produces: the same input/output values and hint choice, with one `EvaluationHint` calculation per immutable point batch.

- [x] **Step 1: Add a Guard test that compares the output values and evaluated-point accounting for a single-site varying batch and a multi-site fallback batch.**
- [x] **Step 2: Run the test before the refactor.**
- [x] **Step 3: Pass the existing hint from the input evaluator into the output evaluator instead of recomputing it.**
- [x] **Step 4: Run the focused Guard tests and inspect the retained T=0.01 artifacts; a post-change pipeline replay remains the #687 follow-up gate.**

### Task 4: Short-circuit empty global-pivot injection after validation

**Files:**
- Modify: `crates/tensor4all-treeaci/src/global_guard.rs`
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`

**Interfaces:**
- Consumes: `inject_global_pivots` and its growth-capacity validation.
- Produces: `Ok(0)` for an empty point list after capacity-length validation, without checkpointing or cloning state.

- [x] **Step 1: Add tests for an empty point list with valid capacity and an empty point list with invalid capacity.** The first must leave generation, candidates, arena record count, ranks, and output unchanged; the second must still return the existing invariant error.
- [x] **Step 2: Run the tests before the production short-circuit.**
- [x] **Step 3: Add the early return immediately after the capacity-length check.**
- [x] **Step 4: Run all Guard and TreeACI release tests.**

### Task 5: Eliminate owned sample clones at transaction commit

**Files:**
- Modify: `crates/tensor4all-treeaci/src/transaction.rs`
- Test: `crates/tensor4all-treeaci/src/transaction/tests/mod.rs`

**Interfaces:**
- Consumes: owned `row_samples` and `col_samples` staged by `commit_edge_proposal`.
- Produces: identical candidate/pivot/sample state, consuming the vectors with `into_iter()` rather than cloning each `ComponentSample`.

- [x] **Step 1: Run the existing success, no-op, and staged-error transaction tests as the baseline.**
- [x] **Step 2: Replace only the owned-vector iteration and preserve all validation and commit ordering.**
- [x] **Step 3: Re-run transaction tests plus the TreeACI release suite.**

### Task 6: Avoid copies when appending memoized incoming frames

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`

**Interfaces:**
- Consumes: memoized incoming frame slices in the two-incoming batched path.
- Produces: the same matrices and candidate values using borrowed slices/`extend_from_slice`.

- [x] **Step 1: Run the existing real/complex batched-vs-scalar frame tests and the branch performance fixture.**
- [x] **Step 2: Confirm the append path already uses `extend_from_slice`; git history shows this cleanup is already present, so no production change is needed.**
- [x] **Step 3: Re-run the scalar-vs-batched tests and inspect the saved T=0.01 stage artifacts.**

### Task 7: Deduplicate FrameBuilder priming requests

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`

**Interfaces:**
- Consumes: the priming path that requests identical incoming `SampleId`s.
- Produces: one owned frame-vector copy per distinct incoming sample while retaining the existing memo contract for ordinary `compute` callers.

- [x] **Step 1: Add a debug-stat regression that feeds duplicate incoming IDs during priming and asserts one copy/materialization per distinct ID, while checking real and complex values against scalar computation.**
- [x] **Step 2:** Run it before the implementation change and verify it fails for repeated priming copies.
- [x] **Step 3: Add a priming-only deduplication seam; do not change the general owned-returning `compute` contract.**
- [x] **Step 4: Run all frame tests and the TreeACI release suite.**

### Task 8: Reuse algebraic edge bounds during initialization

**Files:**
- Modify: `crates/tensor4all-treeaci/src/state.rs`, `crates/tensor4all-treeaci/src/initialize.rs`
- Test: `crates/tensor4all-treeaci/src/state/tests/mod.rs`, `crates/tensor4all-treeaci/src/initialize/tests/mod.rs`

**Interfaces:**
- Consumes: one `algebraic_edge_bounds` result from state initialization.
- Produces: identical initial ranks and bounds without a second recursive/dynamic-program calculation.

- [x] **Step 1: Trace state initialization and prove the second calculation is pure and identical for the same prepared problem.**
- [x] **Step 2: Add a real/complex initial-rank regression that asserts the existing bounds and ranks.**
- [x] **Step 3: Thread the already computed vector into `initial_edge_ranks` and remove the duplicate call.**
- [x] **Step 4: Run initialize/state release tests and the full TreeACI suite.**

### Task 9: Keep optional candidate-cache key work one-pass

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`

**Interfaces:**
- Consumes: compact candidate cache keys produced by batched contractions.
- Produces: the same optional cache hits and values without reconstructing a key after validation.

- [x] **Step 1: Run cache-hit/miss and scalar-vs-batched tests as a baseline.**
- [x] **Step 2: Carry the validated key into insertion instead of rebuilding it, retaining the skip path for degree four or higher.**
- [x] **Step 3: Re-run frame tests and validate cache diagnostics where enabled.**

### Task 10: Review structural candidates only after safe cleanups

**Files:**
- Inspect and, only with a measured need, modify: `initialize.rs`, `samples.rs`, `schedule.rs`, `frames.rs`
- Tests: owning module test files and the retained `gw-rs` checkpoint harness

**Interfaces:**
- Consumes: the post-Task-9 implementation and issue #686's structural candidates.
- Produces: no implementation change unless a paired release measurement proves meaningful end-to-end cost and the numerical oracle remains unchanged.

- [ ] **Step 1:** Measure component traversal, projection scratch, phase cloning, `updated_edges` storage, and metadata scans on chain, two-incoming branch, and 3+ incoming branch cases.
- [ ] **Step 2: Add `// INVARIANT:` markers for costs proven necessary, including any retained rollback or deterministic-order storage.
- [ ] **Step 3: Implement only one measured candidate at a time with a failing regression or source-contract test first.
- [ ] **Step 4:** Re-run all affected release tests and the paired performance protocol.

### Task 11: Revisit Guard start-point reuse only after the correctness regression exists

**Files:**
- Potentially modify: `crates/tensor4all-core/src/floating_zone.rs`, `crates/tensor4all-treeaci/src/global_guard.rs`
- Tests: `crates/tensor4all-core/src/floating_zone.rs` tests, `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`, and the `gw-rs` T=0.01/R=10 retained-artifact comparison

**Interfaces:**
- Consumes: a tested API that lets the floating-zone walk reuse the initial error without changing its trajectory.
- Produces: fewer duplicate start evaluations while preserving `evaluated_points`, threshold scaling, pivot selection, and false-convergence diagnostics.

- [ ] **Step 1: Land or identify a deterministic #687 regression at low temperature and large `R` that distinguishes false convergence from rank growth.**
- [ ] **Step 2: Add a core floating-zone test proving the reused initial error follows the byte-for-byte same pivot trajectory as the old path.**
- [ ] **Step 3: Implement the smallest API seam and update Guard accounting explicitly.**
- [ ] **Step 4: Replay the retained CTTN/nblock artifacts and run real/complex Guard tests before considering this cleanup safe.**
