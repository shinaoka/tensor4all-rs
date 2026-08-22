# Closing the branch-point batching gap for issue #671

Work log. Fixes gw-rs issue #671 (upstream `tensor4all-rs` side): `tree_elementwise`
was ~5-6.5x slower per evaluated point on a branching (comb) tree than on a
chain, at comparable sweep counts. Follows on from
`docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`, which added the
batched BLAS path for single-incoming-edge (chain) directed edges and
explicitly left multi-incoming-edge (branch-point) nodes on the scalar path,
"work there needs its own benchmark with actual branch points."

## Root cause

`InputFrameStore::candidate_frames_for_edge` and `FrameBuilder::compute_batch`
(`crates/tensor4all-treeaci/src/frames.rs`) dispatch to a BLAS-backed batched
contraction (`contract_prepared_core_batched`, one `mat_mul` call per group of
candidates sharing a local coordinate) only when a directed edge's source node
has exactly one incoming edge -- every interior chain node. Any node with 2+
incoming edges -- every branch/hub point in a comb/tree topology -- fell
through to `candidate_frame`/`contract_prepared_core` -> `accumulate_incoming`,
a scalar recursive nested loop evaluated independently per candidate, with no
BLAS and no cross-candidate batching.

## Confirming it before fixing it

Before touching any dispatch code, a temporary `#[ignore]`d diagnostic
(`diagnostic_branch_scalar_vs_chain_batched_at_matched_flops`, added and later
reverted) timed the scalar branch path against the existing single-incoming
batched path at **matched total element counts** (chain: one incoming edge of
dimension `D`; branch: two incoming edges of dimension `sqrt(D)` each, so both
touch the same number of tensor elements per candidate), at increasing scale:

| D (chain) / D x D (branch) | ratio (scalar/batched) |
|---:|---:|
| 64 / 8x8 | 1.76x |
| 256 / 16x16 | 3.61x |
| 1024 / 32x32 | 4.17x |

Same FLOP count, but the ratio grows with dimension -- the signature of BLAS's
cache/SIMD/threading advantage widening at larger matrix sizes versus a naive
scalar loop, not of genuinely more required arithmetic. This confirmed the gap
was closeable by extending the existing batched-path pattern to the
two-incoming-edge case, not an inherent cost of tree branching.

## Fix

Added `two_incoming_core_matrix_batched` (`frames.rs`): contracts a node's
core against two incoming-edge candidate-frame matrices (`v1`: `D1 x N1`,
`v2`: `D2 x N2`) via `D2 + 1` BLAS `mat_mul` calls -- `D2` calls reuse the
existing `single_incoming_core_matrix` + `contract_prepared_core_batched` pair
to fold in `v1` one slice of the second incoming axis at a time, then one
final `contract_prepared_core_batched` call folds in `v2` -- producing every
`(n1, n2)` combination in one shot instead of one scalar
`accumulate_incoming` walk per combination.

Wired this into both existing dispatch points as a new middle case:

- `InputFrameStore::candidate_frames_for_edge` (pivot-search candidate path)
  gained `candidate_frames_for_edge_two_incoming`, mirroring the existing
  single-incoming orchestration (group by `local_coordinate`, consult/populate
  `candidate_cache`) but gathering distinct sample ids on *both* incoming
  edges before the batched contraction.
- `FrameBuilder::compute_batch` (sample-materialization path, used by
  `InputFrameStore::from_samples`/`extend`) gained `compute_batch_two_incoming`,
  the same structure reading incoming frame vectors from `self.memo` instead
  of a committed `InputFrameStore`.

Nodes with 0 or 3+ incoming edges are untouched and still use the scalar path
-- deliberately out of scope: issue #671's reported topology (3-arm comb) only
ever produces exactly 2 incoming edges at a hub, and generalizing further would
need an inter-step buffer transpose the 2-incoming case avoids by reusing the
existing single-incoming primitives directly.

## Verification

- TDD throughout: every new function/branch had a failing test confirmed red
  before implementation (compile error for the new primitive; both
  renamed/adapted branch tests already asserted `dispatched == scalar`, so
  they were confirmed still green pre-wiring, then re-confirmed green
  post-wiring now exercising the new code path for real).
- New tests: `two_incoming_core_matrix_batched_matches_scalar_contraction_on_every_pair`
  (isolated primitive check), `candidate_frames_for_edge_still_falls_back_on_three_incoming_edges`
  and `compute_batch_still_falls_back_on_three_incoming_edges` (new 4-arm-star
  fixture, pinning that 3+-incoming dispatch is unchanged).
- `cargo fmt --all -- --check` clean.
- `cargo clippy -p tensor4all-treeaci --all-targets -- -D warnings` clean
  (needed one `#[allow(clippy::too_many_arguments)]`, matching the crate's
  existing convention for wide-signature internal contraction primitives).
- `cargo test --release -p tensor4all-treeaci --no-fail-fast`: 114 lib tests
  (0 failed, 1 ignored -- the speedup diagnostic below), 7 `public_api`
  integration tests, 1 `rank_scaling` test, 18 doctests, all green.
- `cargo doc -p tensor4all-treeaci --no-deps` clean.

## Measured speedup

`branch_point_batched_speedup_vs_scalar_at_realistic_scale` (kept as an
`#[ignore]`d test; run with `--ignored --nocapture`) reproduces #671's
question directly: a 3-arm star with a hub node whose branch edge has two
incoming edges of dimension 32 each, 40 distinct candidate samples per
incoming edge (1600 candidates total), comparing the new batched
`candidate_frames_for_edge` against the still-present scalar `candidate_frame`
loop -- each measured against its own freshly-built `InputFrameStore` so
neither run's timing is contaminated by the other populating the shared
`candidate_cache`:

```
batched (two-incoming fix): 2.194234ms
scalar (old fallback, still used for 3+ incoming): 30.975879ms
speedup: 14.12x
```

14.12x at this problem size, comfortably exceeding the 5-6.5x gap #671
reported for the real gw-rs pipeline (which also includes non-branch-edge
work diluting the aggregate ratio, so the two numbers are not directly
comparable, but this confirms the fix removes the branch-point-local
bottleneck rather than only partially narrowing it).

## Follow-up

Nodes with 3+ incoming edges (degree >= 5) remain on the scalar path. Not
needed for #671's reported topology; if a future workload hits this (e.g. a
Bethe-lattice-like tree), the same technique generalizes via an inter-step
buffer transpose between each pair of incoming-edge contractions, at the cost
of noticeably more implementation and testing complexity than the 2-incoming
case needed.
