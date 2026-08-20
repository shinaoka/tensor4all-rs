# TreeACI Spine-Only Return Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TreeACI's redundant full reverse walk with a spine-only return so every tree edge is updated once in each direction per forward/return round, while preserving train-chain numerical correctness.

**Architecture:** Keep `continuous_walk` as the existing open forward walk. Store the selected spine explicitly and construct the reverse schedule from that spine only; validate coverage over the concatenated round rather than requiring the reverse pass to cover every edge or exactly invert the forward excursion list. Preserve the existing scalar maximum-rank convergence policy used by train ACI; per-edge bond dimensions are diagnostics and are not equality requirements.

**Tech Stack:** Rust, `tensor4all-treeaci` unit tests, `TreeTN` dense reference checks, LaTeX/latexmk for the external theory notes.

**Spec:** `docs/superpowers/specs/2026-08-20-treeaci-euler-round-schedule-design.md`

## Global Constraints

- Do not convert `TreeTN` inputs to `SimpleTensorTrain`.
- Keep the public `TreeAciTraversalStrategy` surface unchanged.
- Preserve the original train ACI convergence semantics: scalar network-wide maximum rank, error tolerance, and global-pivot dwell; never require element-wise per-edge rank monotonicity.
- Treat exact per-edge bond-dimension equality with the old schedule or train ACI as non-required; validate represented values, residuals, and termination instead.
- Do not add a tree benchmark in this change; leave benchmark execution to the user’s separate workflow.
- Use TDD for Rust behavior: write each new structural/correctness test, run it red, implement the smallest change, then run it green.
- Preserve unrelated untracked plan files in the worktree and do not stage them.

---

### Task 1: Update the theory notes to distinguish open walks from Euler rounds

**Files:**
- Modify: `/Users/lingruicheng/treeaci/tree-aci-theory.tex:472-505` (train reduction)
- Modify: `/Users/lingruicheng/treeaci/tree-aci-theory.tex:609-665` (execution schedule)
- Modify: `/Users/lingruicheng/treeaci/tree-aci-theory.tex:790-805` (algorithm summary)
- Modify: `/Users/lingruicheng/treeaci/tree-aci-theory.tex:1025-1045` (verification obligations)

**Interfaces:**
- Consumes: the approved spine-only return design and the existing definitions of
  `continuous_walk`, selected spine, forward pass, and reverse pass.
- Produces: a self-contained theory note that states the schedule actually
  implemented by PR #646 and does not call a repeated-vertex walk a strict path.

- [ ] **Step 1: Rewrite the train reduction paragraph.**

  State that for a path the selected spine is the whole tree, so the forward
  walk has `|E|` updates and the spine-only return has `|E|` updates. Call their
  concatenation one forward/backward round. State explicitly that tree branch
  excursions are already bidirectional in the forward walk and are not
  repeated in the return.

- [ ] **Step 2: Add the Euler-round statement to the execution-schedule section.**

  Retain the existing unique-centre Euler reference, then add the implemented
  schedule as a separate paragraph:

  ```tex
  Let P be the selected spine and let W_f be the open forward walk.  Every
  edge outside P occurs in W_f once in each orientation, while every edge in
  P occurs once in the forward orientation.  The return walk W_r is P in
  reverse order.  Thus W_f followed by W_r is a directed Euler tour of the
  bidirected tree and has exactly 2|E| steps.
  ```

  Explain that the old full inverse would cost `4|E|-2|P|` and is not needed
  for once-per-direction coverage.

- [ ] **Step 3: Update the algorithm summary and verification obligations.**

  Replace “build a deterministic shortest continuous edge-covering walk” with
  “build the forward open walk and its spine-only return.” Replace any claim
  that the reverse is the exact inverse. Change the chain verification item to
  compare numerical values/residuals and update counts, not exact bond ranks.

- [ ] **Step 4: Build the TeX notes and verify the generated artifact.**

  Run from `/Users/lingruicheng/treeaci`:

  ```bash
  make
  ```

  Expected: `latexmk` completes successfully and writes
  `/Users/lingruicheng/treeaci/build/tree-aci-theory.pdf`.

- [ ] **Step 5: Commit the theory update separately.**

  The external notes directory is not a Git worktree, so report the changed
  TeX path and generated PDF to the user. In the PR worktree, commit only the
  spec clarification if it has not already been committed; do not copy binary
  PDF build artifacts into the Rust repository.

---

### Task 2: Add failing path-plan tests for spine-only return coverage

**Files:**
- Modify: `crates/tensor4all-treeaci/src/path_cover/tests/mod.rs`

**Interfaces:**
- Consumes: `SweepPlan::forward`, `SweepPlan::reverse`, and the existing
  labelled-tree fixtures.
- Produces: tests that define the new structural contract before production
  code changes.

- [ ] **Step 1: Add helpers that flatten both passes and count directed steps.**

  Add a helper that returns `(edge, from, to)` triples for an arbitrary pass,
  and a helper that concatenates forward then reverse. Count keys by the full
  triple, not only by undirected edge number.

- [ ] **Step 2: Replace the exact-inverse assertion with the new contract.**

  For each fixture, assert that:

  ```rust
  let round = forward_steps(plan)
      .into_iter()
      .chain(reverse_steps(plan))
      .collect::<Vec<_>>();
  assert_eq!(round.first().map(|step| step.1), round.last().map(|step| step.2));
  assert_eq!(round.len(), 2 * edges.len());
  assert_eq!(directed_counts(&round).len(), 2 * edges.len());
  assert!(directed_counts(&round).values().all(|count| *count == 1));
  ```

  The exact implementation may use a sorted vector rather than a map, but it
  must detect duplicate `(edge, from, to)` occurrences.

- [ ] **Step 3: Add explicit chain and star expectations.**

  For a four-edge chain, assert forward and reverse each have four updates.
  For a four-edge star, assert the forward has `2E-D` updates and the reverse
  has `D` updates, while the concatenated round has `2E` directed updates.

- [ ] **Step 4: Run the focused test and confirm RED.**

  Run:

  ```bash
  cargo test -p tensor4all-treeaci path_cover --lib
  ```

  Expected: failure because the current reverse is the full inverse and the
  current validator rejects a partial return.

- [ ] **Step 5: Commit the failing-test checkpoint only if the implementation
  workflow requires a checkpoint.**

  Do not stage unrelated `docs/superpowers/plans/2026-08-*.md` files.

---

### Task 3: Implement spine-only return and round validation

**Files:**
- Modify: `crates/tensor4all-treeaci/src/path_cover.rs:20-125,194-324`
- Test: `crates/tensor4all-treeaci/src/path_cover/tests/mod.rs`

**Interfaces:**
- Consumes: the failing structural tests from Task 2.
- Produces: `SweepPlan` values whose forward walk is unchanged and whose
  reverse contains only the selected spine in reverse orientation.

- [ ] **Step 1: Add a failing test for explicit spine endpoint continuity.**

  In the star/comb fixture, assert the first reverse step starts at the last
  forward vertex and the last reverse step ends at `plan.start`. This catches
  a return path that has the right edge set but the wrong order.

- [ ] **Step 2: Preserve the selected spine during planning.**

  In `from_minimum_retracing_walk`, retain the ordered `path_nodes(start, end)`
  result or its ordered edge equivalent. Keep the call to `continuous_walk`
  unchanged. Construct `reverse` from the spine edges in reverse order, flipping
  each orientation. Do not call the existing full `reverse_pass` helper.

- [ ] **Step 3: Replace per-pass coverage validation with round validation.**

  Keep edge-reference, orientation, continuity, and phase-disjointness checks
  for every pass. Remove the requirement that the return pass independently
  visits every undirected edge and remove the exact-inverse assertion. Add a
  validator for the concatenated forward+return sequence that requires:

  - continuity at the pass boundary and closure at the starting node;
  - exactly two occurrences per undirected edge;
  - exactly one occurrence of each orientation `(edge, from, to)`.

- [ ] **Step 4: Run the focused tests and confirm GREEN.**

  Run:

  ```bash
  cargo test -p tensor4all-treeaci path_cover --lib
  ```

  Expected: all path-cover tests pass, including the exhaustive labelled-tree
  cases.

- [ ] **Step 5: Run schedule tests and repair only assumptions invalidated by
  the new contract.**

  Run:

  ```bash
  cargo test -p tensor4all-treeaci schedule --lib
  ```

  Update tests that currently assert the reverse is the exact inverse or that
  every reverse pass touches every edge. Keep the scalar convergence tests
  unchanged; they already encode the original train ACI max-rank policy.

- [ ] **Step 6: Commit the scheduler change.**

  ```bash
  git add crates/tensor4all-treeaci/src/path_cover.rs \
    crates/tensor4all-treeaci/src/path_cover/tests/mod.rs \
    crates/tensor4all-treeaci/src/schedule/tests/mod.rs
  git commit -m "perf(treeaci): avoid repeating branch edges on return"
  ```

---

### Task 4: Prove chain numerical correctness without requiring rank equality

**Files:**
- Modify: `crates/tensor4all-treeaci/src/elementwise/tests/mod.rs`
- Inspect/modify only if required: `crates/tensor4all-treeaci/src/schedule.rs`

**Interfaces:**
- Consumes: the new path plan and scalar train-compatible convergence policy.
- Produces: a deterministic chain correctness gate suitable for the user’s
  later independent benchmark run.

- [ ] **Step 1: Add a dedicated chain value test before changing any production
  code in this task.**

  Use the existing `product_tree` fixture with at least five sites and two
  inputs. Run `tree_elementwise` with the global guard disabled and compare one
  materialized dense result against the pointwise sum of the two input dense
  tensors. Assert the maximum absolute value error is below the existing test
  tolerance. Assert only that `max_ranks.len() == max_errors.len()` and that
  termination is a valid convergence/rank-limited/max-sweeps outcome; do not
  assert exact edge-rank values.

- [ ] **Step 2: Run the dedicated test and confirm it passes on the unchanged
  implementation.**

  This establishes that the test exercises a real existing chain result before
  the scheduler change. If it passes immediately, keep it as a regression gate
  and use the structural tests from Task 2 as the red test for the code change.

- [ ] **Step 3: Verify the convergence implementation against train ACI.**

  Confirm `schedule::convergence_criterion` remains equivalent to
  `tensor4all_aci::convergence_criterion_like_julia`: scalar maximum rank,
  trailing dwell, error threshold, and no global pivots. If a code change is
  needed, first add a focused failing test for that discrepancy, then make the
  smallest correction. Do not introduce per-edge rank-vector monotonicity.

- [ ] **Step 4: Run the chain correctness tests in release mode.**

  ```bash
  cargo test -p tensor4all-treeaci --release --lib elementwise
  cargo test -p tensor4all-treeaci --release --test public_api
  cargo test -p tensor4all-treeaci --release --test rank_scaling
  ```

  Expected: all tests pass and no assertion compares TreeACI edge ranks with a
  train or old-schedule edge-rank vector.

- [ ] **Step 5: Commit the chain correctness gate.**

  ```bash
  git add crates/tensor4all-treeaci/src/elementwise/tests/mod.rs
  git commit -m "test(treeaci): lock down chain values without rank equality"
  ```

---

### Task 5: Repository verification and handoff

**Files:**
- Inspect: all files changed by Tasks 1–4

**Interfaces:**
- Consumes: the committed theory update, spine-only scheduler, and chain
  correctness tests.
- Produces: a clean, locally verified PR #646 branch ready for the user’s
  independent correctness and performance benchmark.

- [ ] **Step 1: Format and run the changed-crate checks.**

  ```bash
  cargo fmt --all
  cargo fmt --all -- --check
  cargo test -p tensor4all-treeaci --release --lib
  cargo test -p tensor4all-treeaci --release --test public_api
  cargo test -p tensor4all-treeaci --release --test rank_scaling
  ```

- [ ] **Step 2: Run scoped clippy.**

  ```bash
  cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
  ```

- [ ] **Step 3: Review the final diff and working tree.**

  Verify the diff contains only the intended theory-note change (outside the
  Rust repository), the spec clarification/plan, TreeACI scheduler/tests, and
  no benchmark binaries or generated TeX files. Confirm the pre-existing
  untracked plan documents remain untouched.

- [ ] **Step 4: Report the handoff without pushing.**

  Report commit hashes, tests run, the chain correctness result, and the fact
  that no tree benchmark was added or run. Do not push or create/update the PR
  unless the user separately authorizes it.
