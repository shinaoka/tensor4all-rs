# TreeACI Branch-Point Diagnostics

## Status

Design draft for review. Companion to `gw-rs`'s
`docs/superpowers/specs/2026-08-24-aci-stage-isolation-design.md`, which
consumes the interface this document defines.

## Problem

Issue #671 tracks a wall-time gap between chain-shaped (NBlock) and
branch-shaped (Comb/CTTN) TreeACI runs. `docs/worklogs/2026-08-23-treeaci-branched-hotpaths.md`
already reduced the fixed per-node overhead (76.3% median reduction on the
synthetic comb benchmark) and ruled out two structural bug candidates
(redundant sweeps, cache-lifetime errors), but the residual ~13.7x
synthetic-benchmark gap and the downstream 5-13x stage-level gaps are not yet
attributed to a cause.

`shinaoka` (issue #671 comment) proposed the missing distinction: for a node
with coordination number `z` and bond dimension `chi`, the local tensor has
`O(chi^z)` degrees of freedom, so branching (`z=3`) is expected to cost more
than a chain interior (`z=2`) by a factor on the order of `chi` even with a
perfect implementation. The open question is whether the observed gap is
explained by this topology-necessary `chi^z` scaling, or whether some of it is
avoidable repeated work (e.g. environment reconstruction or cache misses
concentrated at the branch hub). Answering that requires per-node data that
TreeACI does not currently expose: per-node timing broken down by algorithm
phase, each node's actual bond dimensions, and cache hit/miss counts at each
node. The existing `#[cfg(test)]` phase-timing counters in
`tensor4all-treetn/src/treetn/cached_evaluator.rs` were built for an unrelated
investigation (why the message cache had no net speedup) and are global
sums, not per-node, and not available in release builds.

## Goals

1. For a given `TreeACI`/`Guard`-driven `tree_elementwise` call, report, per
   node: coordination number, the bond dimension of each incident edge, time
   spent in Guard's environment/message-cache construction, time spent in
   TreeACI's LUCI/frame construction, and cache hit/miss counts for both the
   Guard message cache and the TreeACI candidate-frame cache.
2. Make this available in a release build behind an explicit opt-in, with
   zero generated code (not just zero measured cost) when the opt-in is
   disabled.
3. Make the collected data consumable by a downstream crate (`gw-rs`) without
   that crate depending on TreeACI's internal node-key generic type.

## Non-goals

- Do not change LUCI pivot selection, candidate enumeration, truncation
  tolerances, Guard's search policy, or any sampled value. This is
  observability only.
- Do not add a persistent/cross-call diagnostics store. One call's
  diagnostics are collected, snapshotted, and discarded on the next reset.
- Do not instrument every cache in the two crates -- only the two identified
  in the Problem section (Guard's `EnvironmentCache<V>`, TreeACI's
  `candidate_cache`). Other internal caches (e.g. `InputFrameStore::frames`,
  which is a memoized-computation store rather than a hit/miss cache in the
  same sense) are out of scope.
- Do not change the public signature of `tree_elementwise`, `TreeAciOptions`,
  or `TreeTNCachedEvaluator`'s existing methods. Diagnostics are read out of
  band via a snapshot function, not threaded through return types.

## Design

### Feature flag

Add a `diagnostics` Cargo feature to `tensor4all-treetn` and to
`tensor4all-treeaci`. `tensor4all-treeaci/Cargo.toml` declares
`diagnostics = ["tensor4all-treetn/diagnostics"]` so enabling it on the
downstream-facing crate enables it transitively. With the feature disabled
(the default), every diagnostics-related item in this design is behind
`#[cfg(feature = "diagnostics")]` and compiles to nothing; existing release
builds and existing benchmarks (including the ones the branched-hotpaths
worklog measured) are unaffected.

This mirrors the existing `#[cfg(test)]`-gated `phase_timing` module in
`cached_evaluator.rs`, generalized from a test-only global-sum prototype to a
per-node, feature-gated, cross-crate-consumable one. The existing
`phase_timing` module stays as is; it is a different, narrower investigation
and this design does not fold it in.

### Data model

Both crates report through a shared, serializable record. It lives in
`tensor4all-treetn` (the lower crate) so `tensor4all-treeaci` can extend the
same entries rather than keeping a parallel table.

```rust
// tensor4all-treetn/src/treetn/diagnostics.rs, cfg(feature = "diagnostics")

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct NodeDiagnostics {
    /// `format!("{node:?}")` of the tree node. A string, not the generic
    /// `V`, so downstream crates (gw-rs) do not need `V: Serialize` or to
    /// depend on TreeACI's/TreeTN's node-key type at all.
    pub node: String,
    /// Number of tree edges incident to this node (Guard/TreeACI's `z`).
    pub coordination_number: usize,
    /// Bond dimension of each incident edge, in the same order as
    /// discovered (not guaranteed stable across calls).
    pub bond_dims: Vec<usize>,
    /// Nanoseconds spent building this node's Guard environment/message
    /// during `build_environment_cache` and its callers.
    pub guard_ns: u64,
    /// Nanoseconds spent in this node's TreeACI LUCI/frame construction
    /// (`candidate_frame`, `directed_frame`, and the batched equivalents in
    /// `frames.rs`).
    pub frame_ns: u64,
    pub guard_cache_hits: u64,
    pub guard_cache_misses: u64,
    pub frame_cache_hits: u64,
    pub frame_cache_misses: u64,
}
```

### Collection mechanism

A feature-gated global registry, following the existing atomic-counter
pattern in `cached_evaluator.rs` but keyed per node instead of summed:

```rust
// tensor4all-treetn/src/treetn/diagnostics.rs, cfg(feature = "diagnostics")

thread_local! {
    static REGISTRY: RefCell<HashMap<String, NodeDiagnostics>> = ...;
}

pub fn reset();
pub fn record_guard(node: &str, coordination_number: usize, bond_dims: &[usize], elapsed: Duration, hit: bool);
pub fn record_frame(node: &str, elapsed: Duration, hit: bool);
pub fn snapshot() -> Vec<NodeDiagnostics>;
```

`thread_local`, not a `Mutex`-guarded global: both Guard and TreeACI already
run single-threaded per `tree_elementwise` call in the code paths this
investigation cares about (the R=10 checkpoints were produced with
`RAYON_NUM_THREADS=1` for the isolated G0 worker and the downstream product
stages do not fan the Guard/TreeACI sweep itself across threads). A
thread-local avoids lock contention entirely and matches the "one call, one
snapshot" usage model. If a future caller does run diagnostics-enabled sweeps
across threads, `snapshot()` only sees its own thread's records -- documented
as a known limitation, not silently wrong data.

Call sites:

- `cached_evaluator.rs`'s `build_environment_cache` (~line 1043): wrap each
  per-node message construction with a timer; call `record_guard` with the
  node's `Debug` string, its coordination number (`self.neighbors[node].len()`,
  from the existing `ComponentCostIndex`/ evaluator's `neighbors: HashMap<V,
  Vec<V>>`), and the bond dimensions read off the constructed `StackedMessage`.
  `hit: false` here, since this call site is a cache *miss* by construction
  (it is the rebuild path); a `hit: true` record is added wherever an
  existing `EnvironmentCache` entry is reused instead of rebuilt.
- `frames.rs`'s three `candidate_cache.borrow().get(&key)` sites (lines 814,
  982, 1116): record a frame-cache hit when `Some`, and wrap the
  corresponding compute-and-insert fallback (around line 530) with a timer
  and a miss record. The node identity for a `CandidateCacheKey` is derived
  from `problem.directed_edges[key.directed_edge].to` (or `.from`, whichever
  is the node the frame is being built *at* -- confirm against
  `PreparedTreeProblem`'s existing edge-orientation convention during
  implementation). Coordination number is the count of directed edges in
  `problem.directed_edges` sharing that node.

All of the above only executes inside `#[cfg(feature = "diagnostics")]`
blocks; the non-diagnostic build keeps exactly the code it has today.

### Public accessors

```rust
// re-exported from tensor4all_treetn::diagnostics (feature = "diagnostics")
pub fn reset();
pub fn snapshot() -> Vec<NodeDiagnostics>;
```

`tensor4all_treeaci` re-exports the same two functions (or gw-rs depends on
`tensor4all_treetn` directly for them -- an implementation-time choice with no
design impact, since `tensor4all-treeaci` already depends on
`tensor4all-treetn`).

## Error handling

Diagnostics collection must not turn a successful `tree_elementwise` call
into a failing one. If a record cannot be attributed (e.g. a node `Debug`
string collides after formatting, which is not expected but is not a
correctness invariant this design proves), the later write silently
overwrites the earlier one rather than panicking or returning `Result`.
`reset`/`snapshot` are infallible.

## Testing

1. A `diagnostics`-feature unit test in `tensor4all-treetn` that runs a small
   labeled tree (chain and a 3-arm star) through the existing Guard evaluator
   path, calls `snapshot()`, and asserts: one record per node, the star's
   center has `coordination_number == 3` and the chain's interior nodes have
   `coordination_number == 2`, and `guard_cache_hits + guard_cache_misses`
   matches the number of `EnvironmentCache` lookups the test setup is known
   to perform.
2. A `diagnostics`-feature unit test in `tensor4all-treeaci` that runs
   `tree_elementwise` over a small branched tree and asserts every node
   reachable in `PreparedTreeProblem` appears in `snapshot()`, with
   `frame_cache_hits + frame_cache_misses > 0` for at least one node after a
   multi-sweep run (candidate reuse across sweeps is the documented 45-65%
   hit-rate behavior from the message-cache worklog).
3. Zero-cost-when-disabled is enforced by construction (every diagnostics
   item lives behind `#[cfg(feature = "diagnostics")]`), not by a runtime
   test. The default-feature build
   (`cargo build --release --manifest-path crates/tensor4all-treeaci/Cargo.toml`,
   no `diagnostics` feature) must succeed and is covered by existing CI; this
   design adds no new default-feature test for the module's absence.
4. Release build check: `cargo build --release --manifest-path
   crates/tensor4all-treeaci/Cargo.toml --features diagnostics` succeeds and
   `cargo clippy --manifest-path crates/tensor4all-treeaci/Cargo.toml
   --features diagnostics --all-targets -- -D warnings` is clean.
5. Do not rerun the branched-hotpaths synthetic benchmark as part of this
   change's acceptance gate -- this is instrumentation, not a hot-path
   change, and the feature is off by default in that benchmark's build.

## Files expected to change

- `crates/tensor4all-treetn/Cargo.toml`: new `diagnostics` feature (likely
  pulling in `serde`'s `derive` feature, already a dependency elsewhere in
  the workspace).
- `crates/tensor4all-treetn/src/treetn/diagnostics.rs`: new module (data
  model, registry, `reset`/`snapshot`).
- `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`: feature-gated
  call sites in `build_environment_cache` and wherever an `EnvironmentCache`
  hit is served.
- `crates/tensor4all-treeaci/Cargo.toml`: new `diagnostics` feature
  forwarding to `tensor4all-treetn/diagnostics`.
- `crates/tensor4all-treeaci/src/frames.rs`: feature-gated call sites at the
  three `candidate_cache` lookup sites and the compute-and-insert fallback.
- `crates/tensor4all-treeaci/src/lib.rs`: re-export `diagnostics` module
  when the feature is enabled.
- Tests adjacent to the changed modules.
- `docs/superpowers/plans/2026-08-24-treeaci-branch-diagnostics.md`: the
  implementation plan after this spec is approved.

## Alternatives considered

### Pass a `&mut Diagnostics` collector through call signatures

Rejected: it would touch `tree_elementwise`, `TreeAciOptions`, `Guard`'s
constructor, and every internal call in the affected paths, even when the
feature is disabled (an `Option<&mut Diagnostics>` parameter still exists in
the signature). The global/thread-local registry keeps the change entirely
inside `#[cfg(feature = "diagnostics")]` blocks and leaves every public
signature untouched.

### Expose `V` directly instead of a `String` node label

Rejected for this design: it would require `V: Serialize` (not currently
bounded on `TreeAciNode`/the evaluator's node type) and would leak the
node-key type into `gw-rs`'s dependency surface for no benefit the isolation
harness needs -- `gw-rs` only needs to identify "the hub" (by
`coordination_number`) and print/serialize labels, not reconstruct `V`.

### Fold into the existing `phase_timing` module

Rejected: that module is explicitly scoped to a different, already-resolved
question (message-cache net speedup) and is a global sum, not per-node. Reusing
it would conflate two investigations and require changing its meaning
retroactively.
