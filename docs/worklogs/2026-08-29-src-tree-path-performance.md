# SRC Tree-Path (`ensure_width`/`batch`) Performance Measurement

## Summary

The audit findings F-4 ("`ensure_width`/`batch` repeatedly materialize local
site pairs") and F-5 ("the width-keyed environment cache in
`EnvironmentCache` recomputes on every new width it sees") were established
by index-counting arguments in WS-tree-probe, not by profiling. This worklog
measures the actual cost of `EnvironmentCache::ensure_width` and
`EnvironmentCache::batch` in
`crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:314-398` on a
representative interior-center tree contraction, per the task-7 spec's
"measure first" gate.

**Conclusion: no code change is made in this task.** The width-keyed cache
does trigger a small, bounded number of full re-materializations as the
per-edge required width grows toward the tree's root — the mechanism F-4/F-5
describe is real — but at measured problem sizes that redundancy is a minor
fraction of a single contraction's cost (at most ~24% of the
column-materialization work, itself only one component of per-edge cost),
not the dominant contributor to the tree path's slowdown relative to the
chain-specialized fast path. The dominant contributor is the tree path's
inherently different architecture (global directed-message-passing per width,
versus the chain path's single-direction incremental sweep), which is general
wrapper/planner overhead outside F-4/F-5's specific claim and outside this
task's scope per the spec.

## Context and sources

- `docs/plans/2026-08-29-src-audit-remediation.md` (this remediation plan;
  see task 7's brief for the exact gate and step sequence).
- `docs/worklogs/2026-08-29-backend-incremental-qr-refactor.md`: an earlier
  session in this same remediation effort measured the adaptive-SRC QR path
  and found it not the dominant cost either, explicitly flagging "the
  audit's F-4/F-5 tree-probe candidates remain higher-priority profiling
  targets." This worklog closes that flag.
- `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs:314-428`
  (`EnvironmentCache::ensure_width`, `::batch`, `::column`) — the functions
  under measurement.
- `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` and
  `src_probe.rs::contract_prefix_with_probed_site_pair_batch_range` — the
  chain path's incremental-batch pattern that F-4's fix direction is
  supposed to imitate, read for comparison (not adapted, since no fix was
  made).
- `crates/tensor4all-treetn/examples/benchmark_src.rs` — pre-existing,
  uncommitted-until-this-PR benchmark harness (added by this PR's commit
  `2395ec5`, "bench(treetn): add deterministic SRC comparison"); used
  unmodified for timing.

## Measurement methodology

- Host: AMD Ryzen 9 6900HX, WSL2, `rustc 1.98.0 (88d9e12ae 2026-08-18)`.
- Baseline commit: `a78d2a0f09a583db87a1c0a1a4e2b66f52f51925` (worktree HEAD
  for this task).
- Release profile: `cargo build --release --manifest-path
  crates/tensor4all-treetn/Cargo.toml --example benchmark_src`.
- `RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1` for reproducible single-threaded CPU measurements.
- `mpo-mpo` mode (10 sites, physical dim 2, seed 11 for the pair
  construction, seed 1234 for SRC), rank increment 3, final SVD disabled,
  10 repetitions per case, as instructed by the task brief.
- Two centers: the default endpoint center (`S0`, the lexicographically
  smallest node name, which for this chain-shaped `TreeTN` topology routes
  through `contraction.rs`'s `chain_order` check into the chain-specialized
  fast path in `src_chain.rs`), and an explicit interior center
  (`T4A_BENCH_CENTER=S5`), which is not a degree-1 node in the chain graph,
  so `chain_order(center)` either returns `None` or a chain whose `.last()`
  is not the requested center, and `contraction.rs::contract` dispatches to
  the general tree path in `src_tree.rs` — the exact code under
  measurement — instead.
- Two bond dimensions: 4 and 8 (`max_rank` 16 and 64 respectively).
- Both `src-fixed` (uses `EnvironmentCache::batch` exclusively — fixed rank
  never calls `ensure_width`/`column`) and `src-adaptive` (uses
  `EnvironmentCache::ensure_width`/`column`, single probe column at a time)
  SRC option presets.
- A second, temporary (not committed) instrumentation pass added
  `eprintln!` tracing gated on `T4A_DEBUG_TREE_CACHE=1` inside
  `EnvironmentCache::batch` (cache hit vs. miss per width) and
  `EnvironmentCache::ensure_width` (column-range extended per call), plus a
  trace of each edge's computed `site_max_width` in the `contract` function.
  This was used only to inspect the cache's actual hit/miss behavior on this
  benchmark's topology and was reverted before finishing this task; it does
  not appear in the diff.

## Results

### Wall-clock timings (10 reps, `mpo-mpo`, per-run seconds)

| Bond dim | Center      | Path                        | src-fixed s/run | src-adaptive s/run |
| -------: | ----------- | ---------------------------- | ---------------: | -------------------: |
|        4 | `S0` (endpoint) | chain fast path (`src_chain.rs`) | 0.011224 | 0.056837 |
|        4 | `S5` (interior) | tree path (`src_tree.rs`)        | 0.012174 | 0.090985 |
|        8 | `S0` (endpoint) | chain fast path (`src_chain.rs`) | 0.027242 | 0.199015 |
|        8 | `S5` (interior) | tree path (`src_tree.rs`)        | 0.072531 | 0.383381 |

Interior/endpoint ratio: bond 4 fixed 1.08x, bond 4 adaptive 1.60x, bond 8
fixed 2.66x, bond 8 adaptive 1.93x. Relative error against
`contract_naive` stayed at `1e-19`--`1e-22` in every case (correctness
unaffected either way; not gated on here).

A `reps=5` run at both bond dimensions and both centers reproduced these
numbers within noise before the `reps=10` runs above were taken as the
recorded set.

### Cache behavior trace (`T4A_DEBUG_TREE_CACHE=1`, interior center `S5`, single rep)

`EnvironmentCache::batch` (fixed rank, bond dim 8, `max_rank=64`): the 9
rooted edges produced only 3 distinct `site_max_width` values (4, 16, 64,
saturating at the cut dimension `parent_bond_a.dim() * parent_bond_b.dim() =
64` once the accumulated local Hilbert space exceeds it) — i.e. 3 cache
misses (full re-materialization across all 10 nodes plus one
`directed_messages_batched` tree-wide pass) and 6 cache hits. At bond dim 4
(`max_rank=16`) the pattern was even more favorable: only 2 distinct widths
(4, 16) across the same 9 edges, 2 misses and 7 hits.

`EnvironmentCache::ensure_width` (adaptive, bond dim 8): the per-column
cache (`self.environments`, a growing `Vec`, unlike `batch`'s width-keyed
`HashMap`) extended monotonically and was never re-materialized for an
already-covered column — once width reached 64 partway through the edge
list (at edge `S2->S3`), the remaining 6 edges required no further
extension at all. All of the adaptive path's cost is single-column,
unbatched `directed_messages` calls (64 of them, one per required column, to
reach width 64), which is architecturally distinct from `batch`'s
width-keyed re-materialization and is the "general adaptive-vs-fixed
overhead" the spec explicitly says to exclude from this task's justification
threshold.

## Analysis

The number of distinct widths `batch()` sees is bounded by how many times
the per-edge row dimension needs to grow before saturating at the cut
dimension (`parent_bond_a.dim() * parent_bond_b.dim()`), which for this
benchmark's uniform bond dimension is `O(log(max_rank))` (2-3 values), not
`O(edges)`. This is inherent to the algorithm's structure — not something
`ensure_width`/`batch`'s implementation choices could avoid — and it means
the *number* of full re-materializations F-4 describes is already small on
typical tree/chain topologies, growing only with rank, not tree size.

The actual avoidable waste is narrower than "repeated materialization" in
general: `batch()` recomputes the *entire* `0..width` range from scratch on
each cache miss rather than extending the previous smaller-width result
in place (`contract_prefix_with_probed_site_pair_batch_range` in
`src_probe.rs`, used by the chain path, does support exactly this kind of
incremental extension via its `first_column` parameter — this is the "fix
direction" the spec points to). For the observed bond-8 sequence (widths 4,
16, 64), the total column-materialization work done is `4 + 16 + 64 = 84`
column-widths, versus `4 + (16-4) + (64-16) = 64` column-widths for a
perfectly incremental version — about 24% avoidable overhead in the
materialization step specifically, itself one of several per-edge costs
(alongside QR factorization and the local tensor contractions independent of
`batch`/`ensure_width`).

That ~24% upper bound on the avoidable fraction of one component of the
per-edge cost does not explain the observed 2.66x (bond 8, fixed) or 1.93x
(bond 8, adaptive) wall-clock gap between the interior tree path and the
endpoint chain fast path. The dominant contributor to that gap is
architectural: the tree path performs a `directed_messages`/
`directed_messages_batched` tree-wide message-passing computation on every
materialization (even the "efficient" ones), while the chain path's
`src_chain.rs` recurrence is a single incremental one-directional sweep with
no analogous whole-tree message-passing step. This is general wrapper/
planner overhead distinguishing the two code paths' architectures, not
specifically F-4/F-5's redundant-materialization claim, and it is the kind
of cost the spec explicitly excludes from this task's justification
threshold ("not just general adaptive-vs-fixed overhead, which the audit
already explained separately" — the same reasoning extends to fixed-vs-fixed
chain/tree architectural overhead, which is likewise not what F-4/F-5 name).

## Decision

F-4 and F-5 are confirmed as real, small, bounded inefficiencies:
`EnvironmentCache::batch` does discard and recompute smaller-width prefix
work on every new cache miss, instead of extending incrementally the way the
chain path's own `contract_prefix_with_probed_site_pair_batch_range`
already does. But at the problem sizes measured here (bond dim 4-8, 10
sites), this redundancy is a modest fraction (an estimated ~24% ceiling) of
one component of one contraction path's cost, not the dominant driver of the
interior-center tree path being 1.6x-2.7x slower than the endpoint-center
chain fast path — which is explained by the tree path's fundamentally
different, more general architecture (global message passing vs. a
one-directional incremental sweep).

Per the task-7 spec's explicit instruction not to force a code change to
"have done something," this task makes **no code change** to
`ensure_width`/`batch`. F-4/F-5 remain valid code-quality observations for a
future task if profiling at larger problem sizes (larger trees, more
imbalanced per-edge bond dimensions producing more distinct widths, or
genuinely branching topologies rather than chain-shaped trees, where the
number of distinct widths could scale differently) shows the redundant
prefix recomputation becoming a larger share of total cost.

## Caveat / possible follow-up

This benchmark only exercises a chain-shaped `TreeTN` with an interior
center — the general tree path's simplest non-degenerate case. A genuinely
branching topology (a star or a wider tree, where independent branches can
require different widths simultaneously, driving more distinct cache keys)
was not measured: an attempt to run `benchmark_src`'s existing `tree` mode
(star topology) hit an unrelated, much larger pre-existing performance
issue in the zip-up contraction path used as this benchmark's first case
(dozens of seconds for an 8-node star at bond dim 8), which made a full
`tree`-mode run impractical within this task's scope and is out of scope for
F-4/F-5 (it is in `contract_zipup_impl`, a different code path). If a future
task revisits F-4/F-5, a genuinely branching-topology benchmark (bypassing
or fixing the zip-up cost separately) would be a better test of whether
distinct-width diversity — and therefore `batch()`'s redundant
re-materialization — grows large enough to dominate.
