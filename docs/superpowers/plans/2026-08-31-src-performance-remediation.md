# SRC Performance Remediation Implementation Plan

> **Execution override:** The user explicitly requires this plan to be executed
> inline without any further superpowers skills or sub-skill dispatch.

**Goal:** Remove the degree-exponential TreeTN SRC intermediate first, then
reduce the remaining chain adaptive SRC overhead without changing numerical or
probe-stream semantics.

**Architecture:** Tree messages will contract raw local operands, factorized
probe batches, and incoming messages in one planner-visible expression, so no
full local A-by-B product is materialized at a branch. Chain work will be a
separate follow-up that introduces one owner-level flattened-batch contraction
seam before changing SRC call sites.

**Tech Stack:** Rust, `tensor4all-core`, `tensor4all-treetn`, tenferro-backed
`TensorLike`, release-mode tests and deterministic benchmark binaries.

**Spec:** `docs/plans/2026-08-26-treetn-src-contraction-plan.md`, supplemented
by the measured findings in
`docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md`.

## Global Constraints

- Preserve fixed-seed probe prefixes and all existing numerical tolerances.
- Do not materialize a local tensor with both operands' bonds for every branch.
- Keep factorized MPO-MPO probes; do not fuse physical legs into a `d^2` probe.
- Use existing tensor/backend contraction APIs; no local dense linear algebra.
- Run all tests and benchmarks in release mode with provider thread counts set
  to one for timing.
- Do not change test tolerances.
- Do not push, create a PR, or commit without explicit user approval.

---

### Task 1: Pin the bounded-degree and degree-scaling performance gates

**Files:**
- Modify: `crates/tensor4all-treetn/examples/benchmark_src.rs`
- Test: `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`

**Interfaces:**
- Produces: deterministic `binary-tree` and `star` fixtures plus an algorithm
  selector that can run `zipup`, `src-fixed`, or `src-adaptive` independently.

- [ ] Add a `make_binary_tree_pair(levels, physical_dim, bond_dim, seed)`
  fixture whose maximum degree is three and whose input payload is bounded by
  `O(physical_dim^2 * bond_dim^3)` per node.
- [ ] Add `T4A_BENCH_ALGORITHM=zipup|src-fixed|src-adaptive|all`; default to
  `all` so existing commands keep their behavior.
- [ ] Add a release integration test constructing a degree-five, bond-four
  star and assert SRC agrees with one dense whole-result oracle at the existing
  contraction tolerance and returns maximum bond four.
- [ ] Run the new targeted test and the existing SRC contraction tests:

```bash
cargo test --release -p tensor4all-treetn src_ -- --nocapture
```

- [ ] Record the current `2daf5e3` gates before implementation:

```bash
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_BENCH_SKIP_EXACT=1 T4A_BENCH_ALGORITHM=src-fixed \
cargo run --release -p tensor4all-treetn --example benchmark_src -- 5 4 5 tree 3 false
```

Expected baseline: degree four is about 10 ms at bond four; degree five is
about 176 ms. No timing assertion is placed in tests.

### Task 2: Replace pre-paired tree sites with planner-visible site factors

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs:136-254`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:314-445`
- Test: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:725`

**Interfaces:**
- Produces:

```rust
pub(super) fn probe_batch_tensors<T>(
    outputs: &[T::Index],
    probes: &ProbeBank<T::Index>,
    first_column: usize,
    width: usize,
    batch: &T::Index,
) -> Result<Vec<T>>
```

```rust
fn directed_messages_batched<T, V>(
    tn: &TreeTN<T, V>,
    edges: &[(V, V)],
    local: &HashMap<V, (&T, &T)>,
    probe_batches: &HashMap<V, Vec<T>>,
    batch: &T::Index,
) -> Result<HashMap<(V, V), T>>
```

- [ ] Write a unit test for a degree-four center that compares every directed
  message from the new factorized routine with the current pre-paired reference
  on bond dimension two.
- [ ] Run the test before implementation and verify it fails because the new
  interface does not exist.
- [ ] Extract probe construction from `probed_site_pair_batch_range` into
  `probe_batch_tensors`; keep the existing helper as a chain caller that builds
  probes and then performs its current local-pair contraction.
- [ ] Change `EnvironmentCache::batch` and `grow_segment` to cache only the
  small probe tensors per node while building messages; do not build a
  `HashMap<V, T>` of pre-paired local products.
- [ ] For each directed message, pass raw A, raw B, all local probe tensors,
  and all incoming messages except the destination to one
  `contract_retaining` call. The retained batch index must remain last.
- [ ] Run the directed-message unit test, all SRC tests, and Clippy for the
  changed crate.

### Task 3: Verify that branch complexity no longer depends on `chi^(2d)`

**Files:**
- Modify: `docs/worklogs/2026-08-31-src-performance-remediation.md`

**Interfaces:**
- Consumes: Task 1 benchmark selector and Task 2 message-first implementation.
- Produces: before/after timing, RSS, contraction signatures, and a performance
  classification.

- [ ] Run degree three, four, and five stars at bond four, five repetitions,
  separately for zip-up and both SRC modes.
- [ ] Run a bounded-degree binary tree across bond dimensions 4, 6, and 8.
- [ ] Profile one degree-four/bond-eight SRC call with
  `T4A_PROFILE_CONTRACT=1` and confirm no signature contains a raw local input
  with eight uncontracted bond axes.
- [ ] Classify the tree change `PASS` only if numerical tests pass, peak RSS no
  longer follows `chi^(2d)`, and no measured SRC case regresses by more than
  10%; otherwise classify `FAIL` or `INCONCLUSIVE` and retain all results.
- [ ] Document exact commit, CPU, Rust version, environment variables, complete
  cases, and limitations in the worklog.

### Task 4: Remove systematic ragged adaptive-cache fallback

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:506-740`
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:408-557`
- Test: corresponding `src_chain.rs` and `src_tree.rs` test modules

**Interfaces:**
- Produces: fixed-grid segment requests where a narrow request returns a view
  or slice of one cached segment rather than installing a ragged boundary.

- [ ] Add counters in test-only cache fixtures for aligned hits, sliced hits,
  and split/restack fallbacks; reproduce the `2,4,8,16,32` width sequence with
  rank increment three.
- [ ] Assert the current implementation takes the split/restack fallback.
- [ ] Grow full `batch_size` segments up to the global cache maximum while
  respecting each caller's returned width; select a contiguous batch range
  from one segment when the request is narrower.
- [ ] Assert the common chain sequence uses no split/restack fallback and that
  every returned tensor equals the direct reference.
- [ ] Rerun chain adaptive increments 1, 3, 8, and 16; keep the change only if
  increment three improves and all other cases remain within 10%.

### Task 5: Add an owner-level flattened probe-batch contraction seam

**Files:**
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/idx_tensor.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs` only if the
  existing matrix/dot-general API cannot express the flattened operation
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs:304-435`
- Test: core tensor default tests and TreeTN SRC numerical tests

**Interfaces:**
- Produces a generic batch contraction operation that treats the retained probe
  column as an outer batch in the API but lowers it by folding compatible axes
  into one GEMM dimension on CPU.

- [ ] Write an f64/complex generic test comparing the flattened operation with
  `contract_retaining_indices` for widths 1, 3, and 8 and both operand layouts.
- [ ] Add a default `TensorLike` implementation in terms of existing public
  operations, then override `IdxTensor` with a checked column-major lowering.
- [ ] Replace only the chain probe contraction call sites proven equivalent by
  the test; keep tree multi-factor message contraction planner-visible.
- [ ] Measure contraction call count and wall time at chain bond dimensions
  4, 8, 16, and 32 against both zip-up and the recorded Python reference.
- [ ] Keep the override only if all correctness tests pass and the complete
  paired suite improves adaptive SRC without a regression above 10%.

### Task 6: Final verification and documentation

**Files:**
- Modify: `docs/worklogs/2026-08-31-src-performance-remediation.md`
- Check: `README.md`, TreeTN rustdoc, and tutorial references for surface drift

- [ ] Run `cargo fmt --all` and `cargo fmt --all -- --check`.
- [ ] Run `cargo clippy --workspace --all-targets -- -D warnings
  -D clippy::missing_errors_doc -D clippy::missing_panics_doc`.
- [ ] Run `cargo test --release -p tensor4all-treetn` followed by workspace CI
  tests required by `AGENTS.md` if the lower-layer Task 5 seam was changed.
- [ ] Run workspace doctests and rustdoc build.
- [ ] Run the repository-rules dry-run review and its self-tests.
- [ ] Record remaining risks: provider-specific batching, CUDA not measured,
  allocator retention, heterogeneous cuts, and fixed-rank over-requesting.
- [ ] Confirm `git diff --check` and that no benchmark output or generated API
  inventory is staged.
