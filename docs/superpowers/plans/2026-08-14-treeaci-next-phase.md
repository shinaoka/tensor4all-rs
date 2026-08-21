# TreeACI Next Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the interpolation invariant for tree ACI, settle experimentally whether edge order affects correctness, remove the injection path that can build a singular pivot block, and give the global guard's input evaluators run-lifetime ownership.

**Architecture:** A test-only skeleton reconstructor rebuilds `Y` directly from the active pivot pairs and serves as the independent measuring instrument for all four workstreams. The pivot state is split into pivot pairs (which set bond rank and `P_e`) and candidate sample sets (which feed neighbouring edges' candidate spaces), matching what train ACI already does implicitly with frames. Global pivot injection touches only candidate sets; the existing output-bond padding remains necessary when a later committed rank changes the dense payload shape, but injection itself never changes rank. Finally, input ownership moves out of `TreeAciState` so cached evaluators can live for the whole run.

**Tech Stack:** Rust, `tensor4all-treeaci`, `tensor4all-treetn`, `tensor4all-tcicore`, `tensor4all-tensorbackend`, `tensor4all-core`.

## Global Constraints

- Worktree: `/Users/lingruicheng/tensor4all-rust/tensor4all-rs-treeaci`, branch `codex/treeaci`. All work happens here, not in the primary worktree.
- Spec: `docs/superpowers/specs/2026-08-14-treeaci-next-phase-design.md`. Read it before starting.
- Reference baseline is the **current Rust `tensor4all-aci`**, never `AlternatingCrossInterpolation.jl`. The Rust port has fixed bugs still present in the Julia original.
- TreeACI must not gain a normal Cargo dependency on `tensor4all-simplett`, `tensor4all-aci`, or `tensor4all-treetci`. Reading those crates for design guidance is expected; depending on them is not.
- No new public API on `tensor4all-treetn`. If a task appears to need one, stop and report rather than reaching into TreeTN internals.
- Downstream project names must not appear anywhere in the repository.
- Tests run in release mode: `cargo test --release -p tensor4all-treeaci`. `cargo-nextest` is not installed in this environment.
- Every new test goes in the normal suite. No `#[ignore]`, no opt-in feature gating. Fixtures stay small: perfect binary tree of height 3, `d = 2`, single-digit rank caps.
- Cost assertions use `TreeAciDiagnostics::evaluated_points`, never wall-clock time.
- Run `cargo fmt --all` before every commit.
- Doc examples must be runnable with assertions. `ignore` and `no_run` doctest fences are prohibited.
- Avoid `unwrap()`/`expect()` in library code. Test code may use them.
- Never push or open a PR without explicit user approval.

## File Structure

**Created:**
- `crates/tensor4all-treeaci/src/skeleton.rs` — test-only. Rebuilds `Y` from active pivot sets as `T_v` tensors and `P_e^{-1}` gauges. Depends only on `problem`, `samples`, and an oracle closure.
- `crates/tensor4all-treeaci/src/validate.rs` — test-only. Nesting, interpolation, and gauge-equivalence checks built on `skeleton.rs`.
- `crates/tensor4all-treeaci/src/order_experiment.rs` — test-only. Pivot-only update path and the three-arm edge-order experiment driver.
- `crates/tensor4all-treeaci/tests/rank_scaling.rs` — integration test guarding final ranks at fixed tolerance, mirroring train ACI's file of the same name.

**Modified:**
- `crates/tensor4all-treeaci/src/samples.rs` — split `ActivePivotSets` into `PivotPairs` and `CandidateSets`; injection writes only candidate sets.
- `crates/tensor4all-treeaci/src/local_update.rs` — `enumerate_candidates` reads `CandidateSets`.
- `crates/tensor4all-treeaci/src/transaction.rs` — commit writes both structures.
- `crates/tensor4all-treeaci/src/global_guard.rs` — delete `pad_output_bonds`; move input evaluators out.
- `crates/tensor4all-treeaci/src/state.rs` — hold both structures; borrow inputs instead of owning them.
- `crates/tensor4all-treeaci/src/elementwise.rs` — own the inputs and the run-lifetime evaluators.
- `crates/tensor4all-treeaci/src/result.rs` — candidate-size diagnostic.
- `crates/tensor4all-treeaci/src/lib.rs` — register new modules.
- `~/treeaci/tree-aci-theory.tex` — nesting definition, interpolation theorem, rewritten section 6.1.

**Task order rationale:** B3 first, because it changes the state types the validator reads. B1 second, so the validator exists before it is used as an instrument. B2 third, since it consumes both. B4 last, because its lifetime refactor touches `TreeAciState` broadly and would conflict with everything else.

---

## Phase A — B3: split candidate sets from pivot pairs

### Task 1: Introduce `PivotPairs` and `CandidateSets`

**Files:**
- Modify: `crates/tensor4all-treeaci/src/samples.rs`
- Test: `crates/tensor4all-treeaci/src/samples/tests/mod.rs`

**Interfaces:**
- Consumes: `SampleId`, `ComponentSample`, `SampleArena`, `PreparedTreeProblem<V>`, `DirectedEdgeId` (all existing in `samples.rs` / `problem.rs`).
- Produces:
  - `pub(crate) struct CandidateSets { pub(crate) generation: u64, pub(crate) ids: Vec<Vec<SampleId>> }` indexed by `DirectedEdgeId`
  - `CandidateSets::new(directed_edge_count: usize) -> Self`
  - `CandidateSets::push_unique(&mut self, edge: DirectedEdgeId, id: SampleId) -> bool` (returns `true` when the id was absent and got appended)
  - `pub(crate) struct PivotPairs { pub(crate) per_edge: Vec<Vec<(SampleId, SampleId)>> }` indexed by undirected edge number (`forward / 2`)
  - `PivotPairs::new(edge_count: usize) -> Self`
  - `PivotPairs::rank(&self, edge_number: usize) -> usize`
  - `PivotPairs::set(&mut self, edge_number: usize, pairs: Vec<(SampleId, SampleId)>)`
  - `PivotPairs::forward_ids(&self, edge_number: usize) -> Vec<SampleId>` and `PivotPairs::reverse_ids(&self, edge_number: usize) -> Vec<SampleId>`

Note the naming decision used consistently across all later tasks: `CandidateSets` is indexed by **directed** edge, `PivotPairs` by **undirected** edge number.

- [x] **Step 1: Write the failing test**

Append to `crates/tensor4all-treeaci/src/samples/tests/mod.rs`:

```rust
#[test]
fn candidate_sets_append_only_unique_ids() {
    let mut candidates = CandidateSets::new(4);
    assert!(candidates.push_unique(0, 7));
    assert!(!candidates.push_unique(0, 7));
    assert!(candidates.push_unique(0, 9));
    assert_eq!(candidates.ids[0], vec![7, 9]);
    assert!(candidates.ids[1].is_empty());
}

#[test]
fn pivot_pairs_report_rank_and_projections() {
    let mut pairs = PivotPairs::new(2);
    assert_eq!(pairs.rank(0), 0);
    pairs.set(0, vec![(1, 10), (2, 20), (3, 30)]);
    assert_eq!(pairs.rank(0), 3);
    assert_eq!(pairs.forward_ids(0), vec![1, 2, 3]);
    assert_eq!(pairs.reverse_ids(0), vec![10, 20, 30]);
    assert_eq!(pairs.rank(1), 0);
}
```

Add `CandidateSets` and `PivotPairs` to the `use crate::samples::{...}` line at the top of that test module.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci candidate_sets_append_only_unique_ids`
Expected: FAIL — `cannot find type CandidateSets in this scope`.

- [x] **Step 3: Write minimal implementation**

In `crates/tensor4all-treeaci/src/samples.rs`, add next to the existing `ActivePivotSets` definition (leave `ActivePivotSets` in place for now; Task 2 removes it):

```rust
/// Candidate component samples per directed cut.
///
/// These feed the candidate row and column spaces of *neighbouring* edges. They
/// are replaced when their own edge is updated and appended to by global pivot
/// injection. They are deliberately not the same thing as the pivot pairs that
/// set a bond's rank: a duplicate here is a harmless redundant candidate, while
/// a duplicate in a pivot list makes `P_e` singular.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CandidateSets {
    pub(crate) generation: u64,
    pub(crate) ids: Vec<Vec<SampleId>>,
}

impl CandidateSets {
    pub(crate) fn new(directed_edge_count: usize) -> Self {
        Self {
            generation: 0,
            ids: vec![Vec::new(); directed_edge_count],
        }
    }

    /// Appends `id` to `edge` unless it is already present.
    ///
    /// Returns `true` when the id was appended.
    pub(crate) fn push_unique(&mut self, edge: DirectedEdgeId, id: SampleId) -> bool {
        let ids = &mut self.ids[edge];
        if ids.contains(&id) {
            return false;
        }
        ids.push(id);
        true
    }
}

/// Selected cross pivots per undirected edge.
///
/// Entry `k` of edge `e` is the pair of component samples whose intersection is
/// the `k`-th pivot of `P_e`. The forward and reverse projections therefore have
/// equal length by construction, and neither may contain a repeat.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PivotPairs {
    pub(crate) per_edge: Vec<Vec<(SampleId, SampleId)>>,
}

impl PivotPairs {
    pub(crate) fn new(edge_count: usize) -> Self {
        Self {
            per_edge: vec![Vec::new(); edge_count],
        }
    }

    pub(crate) fn rank(&self, edge_number: usize) -> usize {
        self.per_edge[edge_number].len()
    }

    pub(crate) fn set(&mut self, edge_number: usize, pairs: Vec<(SampleId, SampleId)>) {
        self.per_edge[edge_number] = pairs;
    }

    pub(crate) fn forward_ids(&self, edge_number: usize) -> Vec<SampleId> {
        self.per_edge[edge_number]
            .iter()
            .map(|(forward, _)| *forward)
            .collect()
    }

    pub(crate) fn reverse_ids(&self, edge_number: usize) -> Vec<SampleId> {
        self.per_edge[edge_number]
            .iter()
            .map(|(_, reverse)| *reverse)
            .collect()
    }
}
```

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test --release -p tensor4all-treeaci samples::`
Expected: PASS, including the two new tests.

- [x] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/samples.rs crates/tensor4all-treeaci/src/samples/tests/mod.rs
git commit -m "feat(treeaci): add CandidateSets and PivotPairs alongside ActivePivotSets"
```

---

### Task 2: Route the sweep through the split state

**Files:**
- Modify: `crates/tensor4all-treeaci/src/samples.rs` (replace `ActivePivotSets` uses)
- Modify: `crates/tensor4all-treeaci/src/state.rs:22-34` (state fields), `crates/tensor4all-treeaci/src/state.rs:73-89` (initialization)
- Modify: `crates/tensor4all-treeaci/src/local_update.rs:39` and `:235-287` (`enumerate_candidates`)
- Modify: `crates/tensor4all-treeaci/src/transaction.rs:88-127` (commit)
- Modify: `crates/tensor4all-treeaci/src/frames.rs` (any `ActivePivotSets` parameter)
- Test: `crates/tensor4all-treeaci/src/transaction/tests/mod.rs`

**Interfaces:**
- Consumes: `CandidateSets`, `PivotPairs` from Task 1; `LocalUpdateResult<T>` from `local_update.rs` (fields `row_samples`, `col_samples`, `left`, `right`, `pivot_errors`, `sampled_scale`, `row_count`, `col_count`, `local_values`).
- Produces:
  - `TreeAciState` fields `pub(crate) candidates: CandidateSets` and `pub(crate) pivots: PivotPairs`, replacing `pub(crate) active: ActivePivotSets`.
  - `SampleArena::from_global_seeds` now returns `Result<(SampleArena, CandidateSets, PivotPairs)>`.
  - `enumerate_candidates(problem, candidates: &CandidateSets, edge, resource, limit)`.

The commit rule established here and relied on by every later task:

> On commit at edge `e = (u, v)`: `pivots.set(e/2, zip(left_ids, right_ids))`; `candidates.ids[forward] = left_ids`; `candidates.ids[reverse] = right_ids`. Replacement, not union — that is what keeps growth bounded without an eviction policy, and it mirrors train ACI rebuilding its frames from the selected pivots.

- [ ] **Step 1: Write the failing test**

Append to `crates/tensor4all-treeaci/src/transaction/tests/mod.rs`:

```rust
#[test]
fn commit_sets_pivot_pairs_and_replaces_candidate_sets() {
    let (mut state, options, mut operator) = two_node_product_fixture();
    let report = update_edge_transaction(&mut state, 0, &options, true, &mut operator)
        .expect("edge update must commit");

    let edge_number = 0;
    assert_eq!(state.pivots.rank(edge_number), report.new_rank);
    assert_eq!(
        state.pivots.forward_ids(edge_number),
        state.candidates.ids[0]
    );
    assert_eq!(
        state.pivots.reverse_ids(edge_number),
        state.candidates.ids[1]
    );

    let mut forward = state.pivots.forward_ids(edge_number);
    let before = forward.len();
    forward.sort_unstable();
    forward.dedup();
    assert_eq!(forward.len(), before, "pivot rows must not repeat");
}
```

`two_node_product_fixture` already exists in that test module; if its name differs, reuse whatever fixture the neighbouring tests in the file use and keep the assertions identical.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci commit_sets_pivot_pairs_and_replaces_candidate_sets`
Expected: FAIL — `no field pivots on type TreeAciState`.

- [ ] **Step 3: Write minimal implementation**

Delete `ActivePivotSets` and its `impl` block from `samples.rs`.

Change `SampleArena::from_global_seeds` to build both structures. Replace its body's active-set handling:

```rust
    pub(crate) fn from_global_seeds<V: TreeAciNode>(
        problem: &PreparedTreeProblem<V>,
        seeds: &[Vec<usize>],
    ) -> Result<(Self, CandidateSets, PivotPairs)> {
        let deterministic_seed;
        let seeds = if seeds.is_empty() {
            deterministic_seed = vec![vec![0; problem.node_order.len()]];
            deterministic_seed.as_slice()
        } else {
            seeds
        };
        let mut arena = Self {
            directed: vec![DirectedSampleArena::default(); problem.directed_edges.len()],
            retained_bytes: 0,
            max_retained_bytes: problem.max_sample_arena_bytes,
        };
        let mut candidates = CandidateSets::new(problem.directed_edges.len());
        for point in seeds {
            arena.validate_point(problem, point)?;
            for directed_edge in 0..problem.directed_edges.len() {
                let id = arena.project_component(problem, directed_edge, point)?;
                candidates.push_unique(directed_edge, id);
            }
        }
        let edge_count = problem.directed_edges.len() / 2;
        let mut pivots = PivotPairs::new(edge_count);
        for edge_number in 0..edge_count {
            let forward = &candidates.ids[2 * edge_number];
            let reverse = &candidates.ids[2 * edge_number + 1];
            let rank = forward.len().min(reverse.len());
            pivots.set(
                edge_number,
                (0..rank).map(|k| (forward[k], reverse[k])).collect(),
            );
        }
        Ok((arena, candidates, pivots))
    }
```

Change `inject_global_point_impl`'s signature to take `candidates: &mut CandidateSets` instead of `active: &mut ActivePivotSets`, replace every `active.ids[...]` read with `candidates.ids[...]`, and replace the push loop body with `candidates.push_unique(edge, id);`. Leave the `pair_opposite_cuts` parameter and its logic alone in this task — Task 3 removes it.

In `local_update.rs`, change the `active: &ActivePivotSets` parameter of `materialize_and_factor_edge` to `candidates: &CandidateSets`, and in `enumerate_candidates` change the parameter type the same way; every `active.ids` becomes `candidates.ids`.

In `state.rs`, replace the `active` field:

```rust
    pub(crate) candidates: CandidateSets,
    pub(crate) pivots: PivotPairs,
```

and in `initialize`, replace the bootstrap line and the struct literal fields:

```rust
        let (sample_arena, candidates, pivots) = bootstrap_samples(&problem, &edge_ranks)?;
        let input_frames = InputFrameStore::from_samples(&inputs, &problem, &sample_arena)?;
        let generation = candidates.generation;
```

Update `initialize::bootstrap_samples` to return the new triple, forwarding whatever `from_global_seeds` gives it.

In `transaction.rs`, replace the active-set section of `commit_edge_proposal`:

```rust
    let mut proposed_candidates = state.candidates.clone();
    proposed_candidates.ids[forward] = left_ids.clone();
    proposed_candidates.ids[reverse] = right_ids.clone();
    proposed_candidates.generation = next_generation;

    let mut proposed_pivots = state.pivots.clone();
    proposed_pivots.set(
        edge_number,
        left_ids.into_iter().zip(right_ids).collect(),
    );
```

and in the commit block replace `state.active = proposed_active;` with:

```rust
    state.candidates = proposed_candidates;
    state.pivots = proposed_pivots;
```

Move the existing `let edge_number = forward / 2;` binding above the pivot assignment so it is in scope. Replace `state.edge_ranks[edge_number] = new_rank;` with `state.edge_ranks[edge_number] = state.pivots.rank(edge_number);` and `state.generation = next_generation;` stays as is.

Fix the remaining compile errors mechanically: every other `state.active` becomes `state.candidates`, and `ActivePivotSets::new(n)` becomes `CandidateSets::new(n)`.

**Consistency caution.** `state.rs` initialization asserts that each output bond dimension equals the corresponding entry of `initial_edge_ranks`. The `PivotPairs` built in `from_global_seeds` must agree with that vector, or initialization fails with `initialized output rank differs from active rank target`. If the assertion fires, the seed-derived pivot count and `initial_edge_ranks` disagree; make `initial_edge_ranks` derive from `pivots.rank(edge_number)` rather than computing its own count, so there is a single source of truth.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --release -p tensor4all-treeaci`
Expected: PASS — the whole crate suite, including the new test. Existing tests that referenced `state.active` need their field name updated; that is expected mechanical churn, not a behaviour change.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
git add crates/tensor4all-treeaci/src
git commit -m "refactor(treeaci): route the sweep through split pivot pairs and candidate sets"
```

---

### Task 3: Lock the injection invariant (revised during execution)

**Status: complete.** The original task called for deleting `pad_output_bonds`
and letting rank growth emerge. That premise was wrong — see the "Correction,
found during implementation" section of the spec. An output bond's dimension
equals the candidate-set size on that cut, so enlarging a candidate set requires
padding the bond, and both directed sets must grow together.

The real defect is already fixed by Task 2: injection writes `CandidateSets` and
never `PivotPairs`, so a repeated id can no longer reach `P_e`. What remained was
to lock that in.

- [x] **Step 1: Add the regression test**

`injection_leaves_pivot_pairs_untouched_and_keeps_bonds_in_step` in
`crates/tensor4all-treeaci/src/global_guard/tests/mod.rs` injects the same point
twice and asserts that the pivot pairs are unchanged, that no pivot row repeats,
and that each bond dimension still equals both directed candidate-set sizes.

- [x] **Step 2: Record the finding in the spec**

- [x] **Step 3: Commit**

### Task 4: Candidate-size diagnostic and the injection ordering regression

**Files:**
- Modify: `crates/tensor4all-treeaci/src/result.rs:42-52` (`TreeAciDiagnostics`)
- Modify: `crates/tensor4all-treeaci/src/elementwise.rs` (diagnostics assembly)
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`

**Interfaces:**
- Consumes: `CandidateSets` from Task 1, the injection behaviour from Task 3.
- Produces: `TreeAciDiagnostics::candidate_set_sizes: Vec<(V, V, usize)>`, one entry per directed edge as `(from, to, len)`.

- [ ] **Step 1: Write the failing test**

Append to `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`:

```rust
#[test]
fn injected_pivot_survives_a_neighbouring_edge_update_first() {
    // The old zero-padding scheme broke here: a neighbouring edge recomputed its
    // core against the enlarged set while P_e was still block-diagonal padded.
    let (mut state, options, mut operator) = star_three_leaf_fixture();
    let mask = vec![true; state.edge_ranks.len()];
    inject_global_pivots(&mut state, &[remote_peak_point()], &mask).expect("injection");

    // Update a neighbour of the injected cut before the injected cut itself.
    update_edge_transaction(&mut state, 2, &options, true, &mut operator)
        .expect("neighbour update must commit");
    update_edge_transaction(&mut state, 0, &options, true, &mut operator)
        .expect("injected edge update must commit");

    let error = crate::validate::check_interpolation_for_state(&state, &mut operator)
        .expect("interpolation check must run");
    assert!(error < 1e-10, "interpolation error after reordering: {error}");
}
```

This test depends on `validate::check_interpolation_for_state`, which Task 6 creates. Write the test now but leave it commented out with a `// Enabled by Task 6.` marker, and uncomment it in Task 6 Step 4. Everything else in this task is independent.

- [ ] **Step 2: Write the diagnostic test**

```rust
#[test]
fn diagnostics_report_candidate_set_sizes() {
    let result = run_star_three_leaf_to_completion();
    let sizes = &result.diagnostics.candidate_set_sizes;
    assert_eq!(sizes.len(), 6, "one entry per directed edge on a three-leaf star");
    assert!(
        sizes.iter().all(|(_, _, len)| *len > 0),
        "every directed cut must retain at least one candidate"
    );
}
```

- [ ] **Step 3: Write the growth-gate tests**

The spec requires each of the three growth gates inherited from train ACI to demonstrably bind in a fixture, so that removing one would fail a test rather than silently inflating ranks.

```rust
#[test]
fn max_nglobal_pivots_caps_injections_per_guard_run() {
    let (mut state, mut options, _operator) = star_three_leaf_fixture();
    options.max_nglobal_pivots = 2;
    let mask = vec![true; state.edge_ranks.len()];
    let points = five_distinct_far_points();
    let injected = inject_global_pivots(&mut state, &points[..options.max_nglobal_pivots], &mask)
        .expect("injection");
    assert_eq!(injected, 2, "the guard must not offer more than max_nglobal_pivots");
}

#[test]
fn algebraically_saturated_cuts_do_not_take_candidates() {
    let (mut state, _options, _operator) = two_node_full_rank_state_fixture();
    // Both directed cuts already enumerate their whole index space.
    let saturated = state.candidates.ids[0].len();
    let mask = vec![true; state.edge_ranks.len()];
    inject_global_pivots(&mut state, &[vec![0, 0]], &mask).expect("injection");
    assert_eq!(
        state.candidates.ids[0].len(),
        saturated,
        "a saturated cut must not grow"
    );
}

#[test]
fn an_already_represented_point_adds_nothing() {
    let (mut state, _options, _operator) = star_three_leaf_fixture();
    let mask = vec![true; state.edge_ranks.len()];
    let point = seed_point_of(&state);
    let injected = inject_global_pivots(&mut state, &[point], &mask).expect("injection");
    assert_eq!(injected, 0, "a point already in every candidate set adds nothing");
}
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cargo test --release -p tensor4all-treeaci diagnostics_report_candidate_set_sizes`
Expected: FAIL — `no field candidate_set_sizes on type TreeAciDiagnostics`.

Run: `cargo test --release -p tensor4all-treeaci max_nglobal_pivots_caps_injections_per_guard_run`
Expected: FAIL or PASS depending on the fixture helpers; if the gate already works, the test passes immediately and simply locks the behaviour in. That is an acceptable outcome for a characterization test — do not remove it.

- [ ] **Step 5: Write minimal implementation**

In `result.rs`, add to `TreeAciDiagnostics`:

```rust
    /// Candidate component samples retained per directed cut, as
    /// `(from, to, len)`.
    ///
    /// Candidate sets are replaced when their own edge is updated and appended
    /// to by global pivot injection, so these numbers stay bounded without an
    /// eviction policy. They are reported so that a future phase can design one
    /// against measurements should that stop being true.
    pub candidate_set_sizes: Vec<(V, V, usize)>,
```

In `elementwise.rs`, where `TreeAciDiagnostics` is constructed, add:

```rust
        candidate_set_sizes: state
            .problem
            .directed_edges
            .iter()
            .map(|edge| {
                (
                    edge.from.clone(),
                    edge.to.clone(),
                    state.candidates.ids[edge.id].len(),
                )
            })
            .collect(),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --release -p tensor4all-treeaci`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src
git commit -m "feat(treeaci): report candidate set sizes in diagnostics"
```

---

## Phase B — B1: interpolation invariant

### Task 5: Skeleton reconstructor

**Status: complete.** The test-only reconstructor and two-node/Y-tree checks
landed in `eef8c33`; the implementation uses the inverse orientation
`P_e^{-1}[reverse, forward]` and the plan's separable rank-one Y-tree oracle was
replaced by a nonsingular multilinear fixture.

**Files:**
- Create: `crates/tensor4all-treeaci/src/skeleton.rs`
- Modify: `crates/tensor4all-treeaci/src/lib.rs` (register the module)
- Test: `crates/tensor4all-treeaci/src/skeleton/tests/mod.rs`

**Interfaces:**
- Consumes: `PreparedTreeProblem<V>`, `SampleArena`, `CandidateSets`, `PivotPairs`, `SampleArena::materialize_global_point`.
- Produces:
  - `pub(crate) struct SkeletonTensors<T> { pub(crate) node: Vec<Vec<T>>, pub(crate) node_shape: Vec<Vec<usize>>, pub(crate) gauge: Vec<Matrix<T>> }` — `node[v]` is column-major over `Sigma_v x prod_{a in N(v)} I_{a->v}`, `gauge[e]` is `P_e^{-1}`.
  - `pub(crate) fn skeleton_tensors<T, V, O>(problem, arena, pivots, oracle: &mut O) -> Result<SkeletonTensors<T>>` where `O: FnMut(&[usize]) -> Result<T>`.
  - `pub(crate) fn skeleton_evaluate<T>(tensors: &SkeletonTensors<T>, problem, sigma: &[usize]) -> Result<T>`.

Register the module in `lib.rs` as `#[cfg(test)] mod skeleton;`.

- [x] **Step 1: Write the failing test**

Create `crates/tensor4all-treeaci/src/skeleton/tests/mod.rs`:

```rust
use super::*;

/// On a two-node tree with rank equal to the full algebraic bound, the skeleton
/// must reproduce the oracle exactly at every configuration, because the cross
/// is then the whole matrix.
#[test]
fn full_rank_skeleton_reproduces_the_oracle_exactly() {
    let (problem, arena, pivots) = two_node_full_rank_fixture();
    let mut oracle = |sigma: &[usize]| -> crate::Result<f64> {
        let a = sigma[0] as f64;
        let b = sigma[1] as f64;
        Ok(1.0 + 2.0 * a + 3.0 * b + 5.0 * a * b)
    };
    let tensors = skeleton_tensors(&problem, &arena, &pivots, &mut oracle)
        .expect("skeleton must build");

    let mut worst = 0.0_f64;
    for a in 0..2 {
        for b in 0..2 {
            let sigma = vec![a, b];
            let expected = oracle(&sigma).expect("oracle");
            let actual = skeleton_evaluate(&tensors, &problem, &sigma).expect("evaluate");
            worst = worst.max((expected - actual).abs());
        }
    }
    assert!(worst < 1e-12, "skeleton deviated by {worst}");
}
```

Write `two_node_full_rank_fixture` in the same test module, building a two-node path with `d = 2` per node, seeding all four configurations so both directed candidate sets hold both local coordinates, and setting `PivotPairs` for edge 0 to the two cross pairs.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci full_rank_skeleton_reproduces_the_oracle_exactly`
Expected: FAIL — `cannot find function skeleton_tensors`.

- [x] **Step 3: Write minimal implementation**

Create `crates/tensor4all-treeaci/src/skeleton.rs`. The construction, stated precisely so the implementation is unambiguous:

- For node `v`, enumerate `Sigma_v x prod_{a in N(v)} I_{a->v}` in the same mixed-radix order `local_update::enumerate_candidates` uses — local coordinate fastest, then incoming edges in `directed.incoming_to_from` order — but over *all* incident edges rather than all-but-one. Take `I_{a->v} = pivots.forward_ids(edge_number_of(a,v))` oriented so the ids belong to the `a -> v` direction. For each entry, materialize the global configuration with `SampleArena::materialize_global_point` from the node's local coordinate and the incoming sample ids, then call `oracle`. Store column-major.
- For edge `e` with pivot pairs `[(i_k, j_k)]`, build `P_e[k][l] = oracle(glue(i_k, j_l))` and invert it. Use `tensor4all_tensorbackend::Matrix` and `solve_matrix_owned(P_e, I)`. The returned numerical inverse has row labels from the reverse component and column labels from the forward component, so the evaluator must contract it as `P_e^{-1}[j, i]`. The test oracle must be nonsingular on every selected cross; a separable product oracle is rank one and is not a valid full-rank fixture.
- `skeleton_evaluate` contracts the node tensors and gauges over the tree. On the small fixtures this plan uses, a straightforward recursive contraction from the leaves toward `problem.root` is sufficient and clearer than a generic einsum.

Add `#[cfg(test)] mod skeleton;` to `lib.rs` and `#[cfg(test)] mod tests;` at the bottom of `skeleton.rs`.

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test --release -p tensor4all-treeaci skeleton::`
Expected: PASS.

- [x] **Step 5: Add the dense cross-check test**

```rust
#[test]
fn skeleton_matches_a_dense_contraction_on_a_y_tree() {
    let (problem, arena, pivots) = y_tree_full_rank_fixture();
    let mut oracle = |sigma: &[usize]| -> crate::Result<f64> {
        Ok(1.0 / (1.0 + sigma[0] as f64 + 2.0 * sigma[1] as f64 + 3.0 * sigma[2] as f64))
    };
    let tensors = skeleton_tensors(&problem, &arena, &pivots, &mut oracle).expect("skeleton");
    let mut worst = 0.0_f64;
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                let sigma = vec![a, b, c];
                let expected = oracle(&sigma).expect("oracle");
                let actual = skeleton_evaluate(&tensors, &problem, &sigma).expect("evaluate");
                worst = worst.max((expected - actual).abs());
            }
        }
    }
    assert!(worst < 1e-12, "Y-tree skeleton deviated by {worst}");
}
```

Run: `cargo test --release -p tensor4all-treeaci skeleton::`
Expected: PASS.

- [x] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src/skeleton.rs crates/tensor4all-treeaci/src/skeleton crates/tensor4all-treeaci/src/lib.rs
git commit -m "test(treeaci): add a skeleton reconstructor as an independent measuring instrument"
```

---

### Task 6: Nesting, interpolation, and gauge-equivalence checks

**Status: complete.** The nesting, pivot interpolation, gauge-equivalence, and
stale-record validators landed in `de30e76`; the global-guard neighbouring-edge
regression is covered by the expanded injection-order test.

**Files:**
- Create: `crates/tensor4all-treeaci/src/validate.rs`
- Modify: `crates/tensor4all-treeaci/src/lib.rs`
- Modify: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs` (uncomment the Task 4 test)
- Test: `crates/tensor4all-treeaci/src/validate/tests/mod.rs`

**Interfaces:**
- Consumes: `SkeletonTensors<T>`, `skeleton_tensors`, `skeleton_evaluate` from Task 5; `TreeAciState` fields `candidates`, `pivots`, `sample_arena`, `output`, `problem`.
- Produces:
  - `pub(crate) struct NestingReport { pub(crate) nested: Vec<bool> }` indexed by directed edge, plus `NestingReport::fraction(&self) -> f64`.
  - `pub(crate) fn check_nesting<V>(problem, arena, candidates, pivots) -> Result<NestingReport>`.
  - `pub(crate) fn check_interpolation_for_state<T, V, F>(state, operator: &mut F) -> Result<f64>` — max absolute deviation between the skeleton and the oracle over the pivot configurations.
  - `pub(crate) fn check_gauge_equivalence<T, V, F>(state, operator: &mut F) -> Result<f64>` — max absolute deviation between `state.output` evaluated through TreeTN and the skeleton, over a deterministic sample of configurations.

- [x] **Step 1: Write the failing test**

Create `crates/tensor4all-treeaci/src/validate/tests/mod.rs`:

```rust
use super::*;

#[test]
fn seed_initialization_is_fully_nested() {
    let (state, _options, _operator) = binary_tree_height_three_fixture();
    let report = check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )
    .expect("nesting check must run");
    assert!(
        (report.fraction() - 1.0).abs() < f64::EPSILON,
        "seed projections must be nested everywhere, got {}",
        report.fraction()
    );
}

#[test]
fn continuous_walk_interpolates_at_pivots_to_machine_precision() {
    let (mut state, options, mut operator) = binary_tree_height_three_fixture();
    run_one_directional_pass(&mut state, &options, &mut operator).expect("pass must run");
    let error = check_interpolation_for_state(&state, &mut operator).expect("check must run");
    assert!(error < 1e-10, "interpolation error {error} is not machine precision");
}

#[test]
fn stored_output_is_a_gauge_of_the_skeleton_after_one_pass() {
    let (mut state, options, mut operator) = binary_tree_height_three_fixture();
    // build_random_output leaves the stored cores partly random until every edge
    // has been visited once, so this only becomes meaningful after a full pass.
    run_one_directional_pass(&mut state, &options, &mut operator).expect("pass must run");
    let deviation = check_gauge_equivalence(&state, &mut operator).expect("check must run");
    assert!(deviation < 1e-8, "stored output deviated from the skeleton by {deviation}");
}

#[test]
fn nesting_report_detects_a_stale_incoming_sample() {
    let (mut state, options) = binary_tree_height_three_fixture();
    run_one_directional_pass(&mut state, &options).expect("pass must run");
    let (directed, incoming) = find_a_recursive_pivot_record(&state);
    state.candidates.ids[incoming].clear();
    let report = check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )
    .expect("nesting check must run");
    assert!(!report.nested[directed]);
}
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cargo test --release -p tensor4all-treeaci validate::`
Expected: FAIL — `cannot find function check_nesting`.

- [x] **Step 3: Write minimal implementation**

Create `crates/tensor4all-treeaci/src/validate.rs`.

`check_nesting`: for each directed edge `u -> v` with edge number `e`, take each pivot row id `i` in `pivots.forward_ids(e)` oriented toward `u -> v`, read its arena record, and check that every `(incoming_edge, incoming_id)` pair in that record satisfies `candidates.ids[incoming_edge].contains(&incoming_id)`. Set `nested[directed] = true` when all pivots pass. `fraction` is the count of `true` divided by the length. The validator test deliberately removes one current incoming candidate after a pass and asserts that the corresponding stale pivot is detected; it does not claim that every serial forward fixture must exhibit partial nesting, since the current schedule can preserve all nesting on a full-rank or stable-pivot fixture.

`check_interpolation_for_state`: build the oracle from `operator` by evaluating all inputs at a configuration through `SampleArena::materialize_global_point`-style expansion and applying the batched callback with a batch of one; build `skeleton_tensors`; enumerate the pivot configurations of every edge (glue `i_k` and `j_k`); return the maximum `|oracle(sigma) - skeleton_evaluate(sigma)|`.

`check_gauge_equivalence`: evaluate `state.output` through the public `TreeTN::evaluate_point` API at a deterministic pseudo-random set of 64 configurations from a fixed local LCG seed, evaluate the skeleton at the same configurations, and return the maximum absolute difference. This is test-only and intentionally avoids depending on the guard's cache ownership.

Add `#[cfg(test)] mod validate;` to `lib.rs`.

Write `binary_tree_height_three_fixture` and `run_one_directional_pass` in the test module. The latter should call the crate's existing directional-pass entry point (`schedule::run_directional_pass`) with the forward direction.

- [x] **Step 4: Run tests to verify they pass, then enable the Task 4 test**

Run: `cargo test --release -p tensor4all-treeaci validate::`
Expected: PASS, all four.

Uncomment `injected_pivot_survives_a_neighbouring_edge_update_first` in `global_guard/tests/mod.rs`.

Run: `cargo test --release -p tensor4all-treeaci injected_pivot_survives_a_neighbouring_edge_update_first`
Expected: PASS. If it fails, the injection restructure from Task 3 is incomplete — do not weaken the tolerance; fix the injection path.

- [x] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
git add crates/tensor4all-treeaci/src
git commit -m "test(treeaci): add nesting, interpolation, and gauge-equivalence validators"
```

---

### Task 7: Theory document — nesting definition and interpolation theorem

**Status: complete.** The theory note now distinguishes immutable arena
validity, current candidate sets, and active pivot projections; states the
selected-pivot interpolation theorem with the inverse axis orientation; and
documents candidate-only global injection plus representation-level output
padding. `latexmk` produced the updated PDF without unresolved references.

**Files:**
- Modify: `~/treeaci/tree-aci-theory.tex`

This file is outside the repository. It has no test cycle; the gate is that `latexmk` builds it and the statements match the code that Tasks 1-6 produced.

- [x] **Step 1: Add the nesting definition**

In section 4, immediately after Assumption 4.2 (`Versioned nested-sample invariant`), insert:

The document distinguishes candidate sets $C_{u\to v}$ from the active pivot
projections $I_{u\to v}$.  Define a directed pivot projection to be nested when
\begin{equation}
 I_{u\to v}\subseteq
 \Sigma_u\times\prod_{a\in N(u)\setminus\{v\}} C_{a\to u},
 \label{eq:nested}
\end{equation}
against the \emph{current} incoming candidate sets.  A state is fully nested
when this holds for every directed edge; an outward pass may only guarantee the
outward half until the reverse pass refreshes the opposite orientation.

- [x] **Step 2: Add the interpolation theorem**

After the reduction-to-train-ACI subsection, insert:

State the resulting skeleton explicitly in terms of the node target tables on
the selected incoming pivot projections and the edge gauges $P_e^{-1}$, then
prove exactness at every diagonal pivot pair by recursive leaf elimination.
The proof must record the inverse orientation (rows from $I_{v\to u}$, columns
from $I_{u\to v}$), since this is an implementation-sensitive detail.

- [x] **Step 3: Rewrite the discussion after Assumption 4.2**

Replace the paragraph that currently permits records to reference stale sets with
text distinguishing immutable arena validity from current nestedness: old
records remain valid, while current pivot projections can become stale when an
incoming candidate set is replaced.  A directional pass may preserve only its
outward half; the reverse pass refreshes the opposite orientation.  This is
isomorphic to train two-site TCI, where a forward sweep maintains one side's
nesting while the other side is replaced, and is not a defect specific to trees.

- [x] **Step 4: Rewrite section 6.1**

Replace the atomic-growth and zero-padding rules with the candidate/pivot-pair
split as implemented in Tasks 1-3: injection adds projections to candidate sets
only; pivot pairs and bond dimensions are untouched; rank growth happens at the
edge's next local update if the rank-revealing step selects the injected
candidate.  State explicitly that the dense output payload still needs padding
when that later committed rank changes shape, and that a repeated entry on
either side of a pivot pair list makes `P_e` singular.

- [x] **Step 5: Tighten Proposition 6.2's wording**

Change its statement so it says what its proof shows — that a generic TreeTN canonicalization destroys the bond coordinate labels — rather than implying discontinuity is itself the obstruction. Leave the conclusion standing; Task 10 revisits it with the experiment's verdict.

- [x] **Step 6: Build and check**

```bash
cd ~/treeaci && latexmk -pdf tree-aci-theory.tex
```

Expected: builds without errors; no unresolved `\cref` warnings for the new labels.

- [x] **Step 7: Commit**

`~/treeaci` is a separate directory. If it is a git repository, commit there:

```bash
cd ~/treeaci && git add tree-aci-theory.tex && git commit -m "theory: add nestedness and the interpolation theorem, rewrite global injection"
```

If it is not a git repository, record that the PDF/source build succeeded but
leave the external document uncommitted; do not initialize a repository.

---

## Phase C — B2: the edge-order experiment

### Task 8: Pivot-only update path

**Files:**
- Create: `crates/tensor4all-treeaci/src/order_experiment.rs`
- Modify: `crates/tensor4all-treeaci/src/lib.rs`
- Test: `crates/tensor4all-treeaci/src/order_experiment/tests/mod.rs`

**Interfaces:**
- Consumes: `materialize_and_factor_edge` from `local_update.rs`; `CandidateSets`, `PivotPairs`; `SkeletonTensors` and `skeleton_evaluate` from Task 5.
- Produces:
  - `pub(crate) fn pivot_only_update<T, V, F>(state: &mut TreeAciState<T, V>, forward: DirectedEdgeId, options: &TreeAciOptions<V>, operator: &mut F) -> Result<usize>` — runs the local factorization, writes `pivots` and `candidates` and `edge_ranks`, and returns the evaluated point count. It does **not** call `replace_edge_cores` and does **not** canonicalize.
  - `pub(crate) enum EdgeOrder { ContinuousWalk, RandomPermutation, EdgeIndex }`
  - `pub(crate) struct ArmOutcome { pub(crate) edge_updates: usize, pub(crate) evaluated_points: usize, pub(crate) held_out_error: f64, pub(crate) pivot_error: f64, pub(crate) ranks: Vec<usize>, pub(crate) nested_fraction: f64 }`
  - `pub(crate) fn run_arm<T, V, F>(state, options, order: EdgeOrder, sweeps: usize, operator: &mut F) -> Result<ArmOutcome>`

- [ ] **Step 1: Write the failing test**

Create `crates/tensor4all-treeaci/src/order_experiment/tests/mod.rs`:

```rust
use super::*;

#[test]
fn pivot_only_update_changes_pivots_but_not_the_stored_output() {
    let (mut state, options, mut operator) = binary_tree_height_three_fixture();
    let output_before = state.output.clone();
    let pivots_before = state.pivots.clone();

    pivot_only_update(&mut state, 0, &options, &mut operator).expect("update must run");

    assert_ne!(state.pivots, pivots_before, "pivots must change");
    assert_eq!(
        state.output.link_dims_snapshot(),
        output_before.link_dims_snapshot(),
        "the stored output must be untouched by a pivot-only update"
    );
}
```

If `TreeTN` has no `link_dims_snapshot`, compare each edge's `bond_index(edge).dim()` across both networks in a loop instead — the assertion is that no bond dimension changed.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci pivot_only_update_changes_pivots_but_not_the_stored_output`
Expected: FAIL — `cannot find function pivot_only_update`.

- [ ] **Step 3: Write minimal implementation**

Create `order_experiment.rs` with `pivot_only_update` mirroring `transaction::commit_edge_proposal`, minus the output half:

```rust
pub(crate) fn pivot_only_update<T, V, F>(
    state: &mut TreeAciState<T, V>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    operator: &mut F,
) -> Result<usize>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let proposal = materialize_and_factor_edge(
        &state.inputs,
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.input_frames,
        forward,
        options,
        true,
        operator,
    )?;
    let reverse = state.problem.directed_edges[forward].reverse;
    let edge_number = forward / 2;
    let evaluated_points = proposal.row_count * proposal.col_count;

    let mut arena = state.sample_arena.clone();
    let left_ids = proposal
        .row_samples
        .iter()
        .cloned()
        .map(|sample| arena.intern_component(&state.problem, forward, sample))
        .collect::<Result<Vec<_>>>()?;
    let right_ids = proposal
        .col_samples
        .iter()
        .cloned()
        .map(|sample| arena.intern_component(&state.problem, reverse, sample))
        .collect::<Result<Vec<_>>>()?;

    state.candidates.ids[forward] = left_ids.clone();
    state.candidates.ids[reverse] = right_ids.clone();
    state.candidates.generation += 1;
    state.pivots.set(edge_number, left_ids.into_iter().zip(right_ids).collect());
    state.edge_ranks[edge_number] = state.pivots.rank(edge_number);
    state.input_frames = InputFrameStore::from_samples(&state.inputs, &state.problem, &arena)?;
    state.sample_arena = arena;
    state.generation = state.candidates.generation;
    Ok(evaluated_points)
}
```

Add `#[cfg(test)] mod order_experiment;` to `lib.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --release -p tensor4all-treeaci order_experiment::`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treeaci/src
git commit -m "test(treeaci): add a pivot-only update path for the edge-order experiment"
```

---

### Task 9: The three arms across five fixtures

**Files:**
- Modify: `crates/tensor4all-treeaci/src/order_experiment.rs`
- Create: `crates/tensor4all-treeaci/tests/rank_scaling.rs`
- Test: `crates/tensor4all-treeaci/src/order_experiment/tests/mod.rs`

**Interfaces:**
- Consumes: `pivot_only_update`, `EdgeOrder`, `ArmOutcome` from Task 8; `check_nesting` from Task 6.
- Produces: the recorded verdict, written into the spec's decision table.

- [x] **Step 1: Implement `run_arm`**

```rust
pub(crate) fn run_arm<T, V, F>(
    state: &mut TreeAciState<T, V>,
    options: &TreeAciOptions<V>,
    order: EdgeOrder,
    sweeps: usize,
    operator: &mut F,
) -> Result<ArmOutcome>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let edge_count = state.edge_ranks.len();
    let mut rng = deterministic_rng(options.rng_seed);
    let mut edge_updates = 0usize;
    let mut evaluated_points = 0usize;
    for _ in 0..sweeps {
        let plan = match order {
            EdgeOrder::ContinuousWalk => continuous_walk_edges(&state.problem),
            EdgeOrder::EdgeIndex => (0..edge_count).map(|e| 2 * e).collect(),
            EdgeOrder::RandomPermutation => {
                let mut edges: Vec<_> = (0..edge_count).map(|e| 2 * e).collect();
                shuffle(&mut edges, &mut rng);
                edges
            }
        };
        for forward in plan {
            evaluated_points += pivot_only_update(state, forward, options, operator)?;
            edge_updates += 1;
        }
    }
    let nesting = crate::validate::check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )?;
    Ok(ArmOutcome {
        edge_updates,
        evaluated_points,
        held_out_error: held_out_error(state, operator)?,
        pivot_error: crate::validate::check_interpolation_for_state(state, operator)?,
        ranks: state.edge_ranks.clone(),
        nested_fraction: nesting.fraction(),
    })
}
```

`continuous_walk_edges` reads `problem.schedule` and returns its forward directed edge ids. `deterministic_rng` and `shuffle` are a small local xorshift and Fisher-Yates, written in this test-only module rather than adding a dependency. `held_out_error` evaluates the skeleton and the oracle at 128 deterministic configurations and returns the maximum absolute difference.

- [x] **Step 2: Write the arm-comparison test**

```rust
fn assert_arms_agree(fixture: fn() -> (TreeAciState<f64, u32>, TreeAciOptions<u32>, TestOperator)) {
    let sweeps = 6;
    let mut outcomes = Vec::new();
    for order in [
        EdgeOrder::ContinuousWalk,
        EdgeOrder::RandomPermutation,
        EdgeOrder::EdgeIndex,
    ] {
        let (mut state, options, mut operator) = fixture();
        outcomes.push(
            run_arm(&mut state, &options, order, sweeps, &mut operator).expect("arm must run"),
        );
    }
    let baseline = &outcomes[0];
    for (index, arm) in outcomes.iter().enumerate().skip(1) {
        assert!(
            arm.pivot_error < 1e-10,
            "arm {index} lost interpolation at pivots: {}",
            arm.pivot_error
        );
        assert!(
            arm.held_out_error <= baseline.held_out_error.max(1e-8) * 10.0,
            "arm {index} held-out error {} vs baseline {}",
            arm.held_out_error,
            baseline.held_out_error
        );
        for (edge, (rank, base)) in arm.ranks.iter().zip(&baseline.ranks).enumerate() {
            assert!(
                *rank <= base + 1,
                "arm {index} inflated rank on edge {edge}: {rank} vs {base}"
            );
        }
    }
}

#[test]
fn edge_order_does_not_change_the_result_on_a_path() {
    assert_arms_agree(path_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_y_tree() {
    assert_arms_agree(y_tree_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_comb() {
    assert_arms_agree(comb_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_binary_tree() {
    assert_arms_agree(binary_tree_height_three_fixture);
}

#[test]
fn edge_order_does_not_change_the_result_on_a_degree_four_star() {
    assert_arms_agree(degree_four_star_fixture);
}
```

The `pivot_error < 1e-10` assertion is the one that distinguishes the third decision-rule outcome. A coordinate permutation blows it up while the held-out error can still look small — that is exactly the failure mode of the original perfect-binary-tree regression.

- [x] **Step 3: Run the arm tests**

Run: `cargo test --release -p tensor4all-treeaci edge_order_does_not_change`
Expected: PASS on all five, if the analysis in the spec holds.

**If a fixture fails:** do not weaken any assertion. Record which fixture, which arm, and which metric failed, and stop — that is decision-rule outcome 3, and it means continuity is a genuine necessary condition. Report it to the user before continuing.

- [x] **Step 4: Write the rank-scaling integration test**

Create `crates/tensor4all-treeaci/tests/rank_scaling.rs`, mirroring train ACI's file of the same name. Drive the public `tree_elementwise` entry point on a binary tree of height 3 at a fixed tolerance, and assert an upper bound on both the final maximum rank and `result.diagnostics.evaluated_points`. Pick the bounds by running once and taking the observed values with modest headroom, then state the observed numbers in a comment so a future regression is legible.

- [x] **Step 5: Run the whole suite**

Run: `cargo test --release -p tensor4all-treeaci`
Expected: PASS. The suite must remain fast — a slow run is evidence of an algorithmic defect, not a reason to gate tests.

- [x] **Step 6: Record the verdict**

Update the decision table in `docs/superpowers/specs/2026-08-14-treeaci-next-phase-design.md` with the observed outcome, including the per-fixture rank vectors and `evaluated_points` for each arm.

- [x] **Step 7: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
git add crates/tensor4all-treeaci docs/superpowers/specs
git commit -m "test(treeaci): run the three-arm edge-order experiment and record its verdict"
```

---

### Task 10: Reflect the verdict in the theory document

**Files:**
- Modify: `~/treeaci/tree-aci-theory.tex`

- [x] **Step 1: Update Proposition 6.2 and section 6**

If the experiment confirmed order-freedom, state that the minimum-retracing walk is a scheduling heuristic rather than a correctness requirement, that the obstruction is specifically generic canonicalization between updates, and that the in-place moving-centre representation is what makes continuity necessary for this implementation. Keep the walk-length result as an efficiency statement.

If the experiment showed slower convergence or rank inflation, state that instead, with the measured rank vectors.

If the experiment failed outright, promote continuity to a necessary condition and record the counterexample.

- [x] **Step 2: Build and verify**

```bash
cd ~/treeaci && latexmk -pdf -interaction=nonstopmode -halt-on-error tree-aci-theory.tex
```

The source and PDF were rebuilt and visually checked. `~/treeaci` is not a
Git repository, so the external theory document remains uncommitted.

---

## Phase D — B4: guard evaluator lifetime

### Task 11: Move input ownership out of `TreeAciState`

**Files:**
- Modify: `crates/tensor4all-treeaci/src/state.rs:22-34`
- Modify: `crates/tensor4all-treeaci/src/elementwise.rs`
- Modify: `crates/tensor4all-treeaci/src/transaction.rs`, `schedule.rs`, `global_guard.rs`, `local_update.rs`, `frames.rs` (lifetime propagation)
- Test: `crates/tensor4all-treeaci/src/state/tests/mod.rs`

**Interfaces:**
- Consumes: nothing new.
- Produces: `TreeAciState<'a, T, V>` with `pub(crate) inputs: &'a [TreeTN<IdxTensor, V>]`, and `TreeAciState::initialize(inputs: &'a [TreeTN<IdxTensor, V>], options) -> Result<Self>`.

This is the enabling refactor. `TreeTNCachedEvaluator<'a, V>` borrows its network, unlike train ACI's `TTCache<T>` which owns its data, so the evaluators cannot live in a struct that also owns the inputs. Moving ownership up to `tree_elementwise_batched` is the way to give them run lifetime without a new TreeTN API.

- [x] **Step 1: Write the failing test**

Append to `crates/tensor4all-treeaci/src/state/tests/mod.rs`:

```rust
#[test]
fn state_borrows_inputs_rather_than_owning_them() {
    let inputs = two_node_inputs();
    let options = TreeAciOptions::default();
    let state = TreeAciState::<f64, u32>::initialize(&inputs, &options).expect("initialize");
    assert_eq!(state.inputs.len(), inputs.len());
    assert!(std::ptr::eq(&state.inputs[0], &inputs[0]));
}
```

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci state_borrows_inputs_rather_than_owning_them`
Expected: FAIL — `initialize` takes `Vec<TreeTN<...>>` by value.

- [x] **Step 3: Write minimal implementation**

Change the struct and its constructor:

```rust
pub(crate) struct TreeAciState<'a, T: TreeAciScalar, V: TreeAciNode> {
    pub(crate) problem: PreparedTreeProblem<V>,
    pub(crate) inputs: &'a [TreeTN<IdxTensor, V>],
    pub(crate) output: TreeTN<IdxTensor, V>,
    // ... remaining fields unchanged
}

impl<'a, T: TreeAciScalar, V: TreeAciNode> TreeAciState<'a, T, V> {
    pub(crate) fn initialize(
        inputs: &'a [TreeTN<IdxTensor, V>],
        options: &TreeAciOptions<V>,
    ) -> Result<Self> {
        let problem = prepare_problem(inputs, options)?;
        // ... body unchanged except that `inputs` is already a slice
    }
}
```

Propagate `'a` through every `TreeAciState<T, V>` mention: `TreeAciState<'_, T, V>` in function parameters that do not name the lifetime. In `elementwise.rs`, hold the inputs in a local binding that outlives the state:

```rust
    let inputs = inputs.to_vec();
    let mut state = TreeAciState::<T, V>::initialize(&inputs, options)?;
```

The `#[derive(Clone)]` on `TreeAciState` still works; a shared slice is `Copy`.

- [x] **Step 4: Run the suite**

Run: `cargo test --release -p tensor4all-treeaci`
Expected: PASS. This task must not change any behaviour — every previously passing test still passes with identical assertions.

- [x] **Step 5: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
git add crates/tensor4all-treeaci/src
git commit -m "refactor(treeaci): borrow the inputs in TreeAciState instead of owning them"
```

---

### Task 12: Run-lifetime input evaluators and split batch evaluation

**Files:**
- Modify: `crates/tensor4all-treeaci/src/global_guard.rs:323-440` (`GuardEvaluators`)
- Modify: `crates/tensor4all-treeaci/src/elementwise.rs`, `schedule.rs` (thread the evaluators through)
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`

**Interfaces:**
- Consumes: `TreeAciState<'a, T, V>` from Task 11.
- Produces:
  - `pub(crate) struct InputEvaluators<'a, V: TreeAciNode> { inputs: Vec<TreeTNCachedEvaluator<'a, V>>, indices_per_node: Vec<usize>, strides: Vec<Vec<usize>>, dims: Vec<Vec<usize>> }`
  - `InputEvaluators::new(inputs: &'a [TreeTN<IdxTensor, V>], problem: &PreparedTreeProblem<V>) -> Result<Self>` — constructed once per run.
  - `InputEvaluators::evaluate<T>(&mut self, points: &[Vec<usize>], split: Option<usize>) -> Result<Vec<T>>`
  - `find_global_pivots` and `inject_global_pivots` gain an `&mut InputEvaluators<'a, V>` parameter.

The output evaluator stays per guard invocation, matching #621's reasoning: the solution is fixed during a search but changes between them.

- [x] **Step 1: Write the failing test**

```rust
#[test]
fn guard_reuses_input_evaluators_across_invocations() {
    let inputs = binary_tree_height_three_inputs();
    let options = TreeAciOptions::default();
    let mut state = TreeAciState::<f64, u32>::initialize(&inputs, &options).expect("initialize");
    let mut evaluators =
        InputEvaluators::new(&inputs, &state.problem).expect("evaluators must build");

    let first = guard_evaluated_points(&mut state, &mut evaluators, &options);
    let second = guard_evaluated_points(&mut state, &mut evaluators, &options);
    assert!(
        second < first,
        "the second guard run must benefit from the warm cache: {second} vs {first}"
    );
}
```

`guard_evaluated_points` runs one guard invocation and returns the number of input evaluations it performed. Add a counter to `InputEvaluators` for this purpose, exposed as `InputEvaluators::evaluations(&self) -> usize`, and have the helper return the delta across the call.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test --release -p tensor4all-treeaci guard_reuses_input_evaluators_across_invocations`
Expected: FAIL — `cannot find type InputEvaluators`.

- [x] **Step 3: Write minimal implementation**

Split `GuardEvaluators` into `InputEvaluators` (run lifetime, owns the per-input `TreeTNCachedEvaluator`s and the coordinate-expansion tables) and a per-invocation output evaluator constructed inside `find_global_pivots`. Move `expand_points` onto `InputEvaluators`, since both halves need it and the tables come from `problem`.

Add the split argument to the batch path, forwarding it to `TreeTNCachedEvaluator::evaluate_batched` so a floating-zone walk that varies one node contracts the shared side once. If `evaluate_batched` has no split parameter, pass the points through unchanged and record in the commit body that only the lifetime half of #621 landed — do not add a new TreeTN public API to get the other half.

Thread `&mut InputEvaluators` from `tree_elementwise_batched` through `schedule::run_sweeps` into the guard.

- [x] **Step 4: Run the suite**

Run: `cargo test --release -p tensor4all-treeaci`
Expected: PASS.

- [ ] **Step 5: Add the cost regression test**

```rust
#[test]
fn guard_cost_stays_within_budget_on_a_binary_tree() {
    // Observed after the cache-lifetime change; keep modest headroom over the
    // measured guard-on cost and retain the guard-off baseline in the test
    // documentation.
    let result = run_binary_tree_height_three_with_guard();
    assert!(
        result.diagnostics.evaluated_points < GUARD_EVALUATION_BUDGET,
        "guard evaluated {} points, budget {GUARD_EVALUATION_BUDGET}",
        result.diagnostics.evaluated_points
    );
}
```

Set `GUARD_EVALUATION_BUDGET` from the observed post-change value with modest headroom, and record the guard-off baseline and guard-on number in a comment. A direct pre-change cache-lifetime measurement was not available, so do not infer one.

- [x] **Step 6: Commit**

```bash
cargo fmt --all
cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings
git add crates/tensor4all-treeaci/src
git commit -m "perf(treeaci): give guard input evaluators run lifetime and split walk batches"
```

---

## Task 13: Final verification and progress note

**Files:**
- Modify: `~/treeaci/2026-08-14-treeaci-progress.md`

- [x] **Step 1: Run the full gate**

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --release -p tensor4all-treeaci
cargo doc -p tensor4all-treeaci --no-deps
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
git diff --check
```

The TreeACI crate-level gates pass. Workspace clippy remains blocked by 45
pre-existing `nonminimal_bool` warnings in `tensor4all-core`; report that
environmental baseline rather than changing unrelated code in this phase.

- [x] **Step 2: Confirm the dependency restriction still holds**

```bash
grep -n "simplett\|tensor4all-aci\|treetci" crates/tensor4all-treeaci/Cargo.toml
```

Expected: no normal dependency on any of them.

- [x] **Step 3: Update the progress note**

Rewrite the "Remaining work" section of `~/treeaci/2026-08-14-treeaci-progress.md`: the interpolation invariant is now stated and machine-checked; the edge-order question has a recorded verdict; injection no longer pads bonds; the guard's input evaluators have run lifetime. What remains: the TreeTN bounded cross-call cache seam, path parity benchmarks, the remaining correctness fixtures, parallel execution, and deterministic skeleton initialization to replace `build_random_output`.

- [x] **Step 4: Report to the user**

Summarize the verdict of the B2 experiment and the measured guard cost change. Do not push or open a pull request without explicit approval.
