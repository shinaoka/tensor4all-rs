# Batch-Native Adaptive SRC Probe/QR: Before/After Results

## Summary

The 7-task plan in
`docs/superpowers/plans/2026-08-30-src-adaptive-batch-probe-columns.md`
replaced the adaptive SRC growth loop's per-column probe/QR round trip
(`select_indices` one column at a time, then `stack_along_new_index` +
`permute_indices` to reassemble a batch) with a batch-native interface
(`factorize_probe_batches`, `IncrementalQr`'s existing whole-`Matrix`
`append`) on both the chain (`src_chain.rs`/`PrefixCache`) and general-tree
(`src_tree.rs`/`EnvironmentCache`) adaptive paths. This task deletes the
now-dead `factorize_probe_columns` and measures the actual effect.

**The result is a genuine, substantial improvement, with one honest
caveat.**

- **Chain path** (`mpo-mps` mode, matching the design spec's original
  measurement exactly): `src-adaptive` per-run wall-clock dropped
  **~19-23%** at every bond dimension tested (4, 8, 16, 32) relative to the
  pre-plan baseline. This narrows, but does not close, the Rust/Python gap
  the design spec identified (`src-adaptive` was 4-12x slower than Python
  and not shrinking with problem size; after this plan it is roughly
  3.5-9.5x, still not shrinking with size).
- **General-tree path, the scenario that actually needed many growth steps**
  (interior-center contraction on a 10-site chain-shaped `TreeTN`, forcing
  `contract`'s general-tree dispatch instead of the chain fast path, per the
  method in `docs/worklogs/2026-08-29-src-tree-path-performance.md`):
  `src-adaptive` per-run wall-clock dropped **35-66%** (1.5x-3.0x speedup)
  at bond dimensions 4, 8, and 16. This is the largest win in the whole
  plan, and it is exactly the mechanism the design spec predicted: the old
  tree path issued one whole-tree `directed_messages` pass *per single probe
  column* (64 of them to reach width 64 at bond dim 8, per the 2026-08-29
  trace); the new path issues far fewer, wider batched passes.
- **General-tree path, star topology with a small required rank** (few
  growth steps needed): measured as a supplementary check using
  `benchmark_src`'s `tree` mode at a reduced, tractable size (see
  "Methodology" for why `n_sites=10` as specified in the task brief could
  not be run). Here the batch-native path was **17-38% *slower*** than the
  pre-plan baseline, not faster. This is reported honestly below, with an
  explanation in "Analysis."

No preset performance target existed for this task; both outcomes above are
reported as measured.

## Context and sources

- `docs/superpowers/specs/2026-08-30-src-adaptive-batch-probe-columns-design.md`
  -- the design spec, in particular its "Problem, with evidence" section
  (the pre-plan baseline numbers and the Rust/Python ratio table) and its
  "Non-goals"/"Testing" sections (no `tenferro` changes; measure, don't
  target).
- `docs/worklogs/2026-08-29-src-tree-path-performance.md` -- the established
  worklog format this document follows, and the source of the
  interior-center/`T4A_BENCH_CENTER=S5` measurement methodology used below
  for the general-tree path (chosen there, and reused here, specifically
  because `benchmark_src`'s `tree` mode star topology has a large
  pre-existing, unrelated performance problem in its `zipup` case -- see
  that worklog's "Caveat / possible follow-up" and "Measurement
  methodology" below).
- `crates/tensor4all-treetn/examples/benchmark_src.rs` -- the benchmark
  harness, used unmodified.
- Commits actually measured:
  - Pre-plan baseline: `5fdcc08` ("fix(treetn): stop ordering SRC probe/QR
    indices by random `Index::id()`"), the commit immediately before this
    plan's design-spec commit (`02225c3`). Confirmed to reproduce the design
    spec's recorded pre-plan numbers within noise (see "Results").
  - After: `95ce662` ("chore(treetn): remove the now-dead per-column
    `factorize_probe_columns`"), this task's own commit, i.e. the full
    7-task plan applied.

## Measurement methodology

- Host: AMD Ryzen 9 6900HX, WSL2, `rustc 1.98.0 (88d9e12ae 2026-08-18)` --
  same host as the design spec's and the 2026-08-29 worklog's measurements.
- `RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1` for reproducible single-threaded CPU measurements, per
  this plan's Global Constraints.
- Release build: `cargo build --release --manifest-path
  crates/tensor4all-treetn/Cargo.toml --example benchmark_src`, built fresh
  at each of the two commits compared (`git checkout <sha> --detach`,
  rebuild, measure, `git checkout feature/treetn-src` to return).
- `T4A_BENCH_SKIP_EXACT=1` throughout (skips the naive dense reference
  contraction, irrelevant to timing and, at the larger sizes below,
  intractable on its own).
- **Chain path**: `mpo-mps` mode, `n_sites=10`, `rank_increment=3`,
  `final_svd=false`, bond dimensions 4/8/16/32, `reps=5` -- the exact
  command from this task's brief. Run twice on the "after" commit to check
  reproducibility (both runs agreed within ~3%); the first run is the
  recorded set below.
- **General-tree path, primary measurement**: `mpo-mpo` mode,
  `n_sites=10`, `rank_increment=3`, `final_svd=false`,
  `T4A_BENCH_CENTER=S5` (an interior, non-degree-1 node, which routes
  `contract` into the general-tree path in `src_tree.rs` instead of the
  chain fast path -- exactly the 2026-08-29 worklog's method), bond
  dimensions 4/8 at `reps=10` and 16 at `reps=5`. The pre-plan baseline
  reproduced the 2026-08-29 worklog's previously recorded numbers within
  ~1% (91.9ms vs. 91.0ms at bond 4; 393.7ms vs. 383.4ms at bond 8),
  confirming this measurement is consistent with that earlier session.
  Bond dimension 32 was attempted (`reps=1`) but abandoned: `src-fixed`
  (unmodified by this plan; a fixed-width probe unrelated to the adaptive
  growth loop) alone did not complete within 9 minutes of wall-clock time
  at `max_rank=1024` on this topology/center, at both the baseline and
  after commits equally -- a pre-existing cost unrelated to this plan, not
  a regression it introduces, but impractical to wait out for this task.
- **General-tree path, star topology (task-brief's literal command)**: the
  brief's exact instruction, `benchmark_src 10 8 5 tree 3` (a 9-leaf star
  at bond dim 8), was attempted first and killed by the OOM killer (RSS
  climbing past 19GB within ~30 seconds) at *both* the baseline and after
  commits -- confirmed to be pre-existing, not something this plan makes
  worse (checked by running the unmodified baseline commit through the
  same command and observing the identical trajectory). This matches the
  2026-08-29 worklog's own note that `tree` mode's star topology "hit an
  unrelated, much larger pre-existing performance issue in the zip-up
  contraction path," making a full `tree`-mode run "impractical." Two much
  smaller, tractable star sizes were substituted instead, as a
  supplementary check specifically of the star topology (not a substitute
  for the primary general-tree measurement above): `n_sites=5` (4 leaves)
  at bond dim 8, `reps=10`, and `n_sites=6` (5 leaves) at bond dim 4,
  `reps=20`. Both reached only a small final bond dimension (4) regardless
  of nominal input bond dimension, i.e. the adaptive loop needed very few
  growth steps to converge in these small, low-effective-rank
  constructions.

## Results

### Chain path (`mpo-mps` mode, task brief's exact command)

| bond_dim | before per_run | after per_run (run 1) | after per_run (run 2) | change |
| -------: | --------------: | ----------------------: | ----------------------: | ------: |
|        4 |        36.905ms |                 28.135ms |                 28.889ms | -23.8% (run 1) |
|        8 |        53.085ms |                 43.220ms |                 43.967ms | -18.6% |
|       16 |        60.251ms |                 49.268ms |                 49.320ms | -18.2% |
|       32 |        74.265ms |                 60.037ms |                 59.417ms | -19.2% |

The "before" column reproduces the design spec's recorded pre-plan numbers
(37.735ms / 54.610ms / 60.314ms / 74.955ms) within ~2%, confirming
`5fdcc08` is the right baseline commit.

### General-tree path, primary measurement (`mpo-mpo` mode, interior center `S5`)

| bond_dim | before per_run | after per_run | change |
| -------: | --------------: | --------------: | ------: |
|        4 |         91.935ms |         33.641ms | -63.4% (2.73x) |
|        8 |        393.724ms |        133.026ms | -66.2% (2.96x) |
|       16 |       2469.209ms |       1602.920ms | -35.1% (1.54x) |
|       32 |     not measured (`src-fixed` alone exceeded 9 minutes, both commits) | not measured | -- |

### General-tree path, star-topology supplement (small required rank)

| star size | bond_dim | before per_run | after per_run | change |
| --------: | -------: | --------------: | --------------: | ------: |
| 4 leaves (`n_sites=5`) |        8 |  1949.025ms |  2293.096ms | **+17.6% (slower)** |
| 5 leaves (`n_sites=6`) |        4 |    84.659ms |   116.504ms | **+37.6% (slower)** |
| 9 leaves (`n_sites=10`, task brief's literal command) | 8 | OOM-killed (pre-existing, both commits) | OOM-killed (pre-existing, both commits) | not comparable |

Every measured "after" number above was reproduced at least once (repeat
full runs for the chain path; `reps=10`/`20` internal repetition, run twice
end-to-end for the two star configurations) and was stable to within a few
percent across repeats -- these are not one-off noise.

### Python reference (from the design spec, back-derived from its ratio table)

The design spec's "Problem" table records Rust/Python ratios, not raw
Python milliseconds, for the pre-plan baseline. Back-computing from
`rust_ms / ratio` at each bond dimension gives:

| bond_dim | pre-plan rust src-adaptive | python reference | pre-plan ratio | post-plan rust src-adaptive (chain) | post-plan ratio |
| -------: | ---------------------------: | -----------------: | ---------------: | -------------------------------------: | -----------------: |
|        4 |                     37.735ms |            ~3.20ms |            11.8x |                                28.14ms |             ~8.8x |
|        8 |                     54.610ms |            ~9.75ms |             5.6x |                                43.22ms |             ~4.4x |
|       16 |                     60.314ms |           ~13.71ms |             4.4x |                                49.27ms |             ~3.6x |
|       32 |                     74.955ms |           ~14.42ms |             5.2x |                                60.04ms |             ~4.9x |

## Analysis

**Chain path**: a consistent ~19-24% wall-clock reduction across all four
bond dimensions, matching the plan's mechanism directly -- each growth step
now issues one `IncrementalQr::append` on a whole batch instead of
`rank_increment` individual `select_indices` calls plus a
stack/permute/`to_vec` reassembly, and the chain path's own last-site step
now uses the batched fast path instead of contracting one probe column at a
time. This narrows the Rust/Python gap from 4.4x-11.8x to roughly 3.6x-8.8x
but does **not** close it, and the gap still does not shrink with problem
size the way `zipup`/`src-fixed`'s do. This is consistent with the design
spec's own diagnosis: over 90% of each `contract()` call's cost sits inside
`tenferro`'s compile-program/scoped-execute engine, which this plan
explicitly does not touch (see "Non-goals"). Reducing the adaptive loop's
*call count* by roughly `rank_increment`-fold (this benchmark uses
`rank_increment=3`) produced a real but sub-proportional wall-clock
reduction -- consistent with `tenferro`'s per-call fixed cost still being
paid on every remaining batched call, and with other non-`select_indices`
per-step costs (QR append, the local site contraction itself) being
unaffected by this plan.

**General-tree path, primary case**: the improvement here is dramatically
larger (35-66%, up to ~3x) than the chain path's, and for good reason: the
2026-08-29 worklog measured the *old* tree path issuing one whole-tree
`directed_messages` pass per single new probe column -- 64 of them to reach
width 64 at bond dim 8 on this exact topology -- with no batching at all
(worse than the chain path's old per-column-but-locally-scoped cost). This
plan's `EnvironmentCache::request` collapses that into a handful of
segment-sized batched passes. The size of the win (66% at bond 8) directly
confirms this was the single largest inefficiency this whole plan
addressed, exactly as the design spec predicted when it called the tree
path's per-column `directed_messages` calls "worse yet" than the chain
path's per-column contractions. The win *shrinks* at bond 16 (35% versus
63-66% at bond 4/8) rather than growing, which is a genuine, if secondary,
finding: at bond 16 the final width (256) is large enough that the fixed
per-segment grid in `EnvironmentCache` (from the Task 5 fix, "give
`EnvironmentCache` a fixed segment grid and a misaligned-request fallback")
produces more distinct segment boundaries to cross, so the number of whole-
tree passes saved per unit of final rank is smaller than at bond 4/8; other
per-step costs (QR append growing with accumulated width, the local
tensor contractions themselves) also grow with bond dimension and are
untouched by this plan, so they claim a larger share of the after-total as
bond dimension grows -- the same "fixed cost dilution" mechanism visible in
the chain path's numbers, just operating on a much smaller starting
overhead here.

**General-tree path, star-topology supplement -- the honest negative
result**: at both tested star sizes, the batch-native path is *slower* than
the old per-column path, by 18-38%. This is not noise (stable across
repeated full runs and across two different star sizes/bond dimensions).
The mechanism is very plausibly the mirror image of the primary case's win:
these star constructions converge to a small final rank (bond 4) in very
few growth steps (`min_rank=2`, `rank_increment=3`, so typically one or two
widening steps total), meaning the *old* per-column path issued few
`directed_messages` calls to begin with -- there was little redundant
per-column cost left to amortize away. Against that small baseline, the new
segment-based `EnvironmentCache::request` path pays extra fixed overhead
that the old path did not: computing a whole batch-indexed tensor and its
segment bookkeeping (segment-grid lookup, the misaligned-request fallback
added in the Task 5 fix commit `fabf567`) up front, even when the
adaptive stopping criterion would have been satisfied by (and the old path
would have requested) far fewer individual columns. In other words: this
plan's batching wins big when many small calls are being collapsed into few
large ones, but *loses* when only one or two calls were ever going to
happen -- the new call is simply doing more work than the few small calls
it replaced, with no repeated-computation waste left to eliminate. This is
exactly the shape of risk the design spec's own "Non-goals"/"Testing"
section flagged as possible without a hard target to hit, and it shows up
concretely here in the specific case of a wide, shallow star with abundant
distinct branches but low required rank per branch -- the opposite regime
from the primary (narrow, deep, high-required-rank) measurement above.

## Decision

This plan's change is kept as-is; no further code change is made in this
task (Task 7 is deletion + measurement + reporting only, per its brief and
this plan's Global Constraints).

The chain-path and primary general-tree-path results are a clear,
substantial, reproducible win -- the general-tree path in particular saw
its single largest inefficiency (unbatched per-column whole-tree message
passing) sharply reduced, which is the specific problem this plan set out
to fix per the design spec's "Problem" section. The chain-path win, while
smaller, is consistent and matches the mechanism intended.

The star-topology regression is reported honestly rather than omitted or
downplayed: it is real, reproducible, and traceable to a specific
mechanism (fixed per-request overhead in the new segment-based cache not
being amortized when very few growth steps are needed). It does not
invalidate the plan's overall benefit -- both measured star configurations
are small, low-effective-rank corner cases, not the primary target
scenario the design spec's evidence was built on (which used chain/
interior-center topologies requiring many growth steps to reach a large
final rank) -- but it is a legitimate, scoped follow-up candidate: if a
future task revisits the general-tree path, profiling a wide star that
requires many growth steps (large required rank per branch, not just many
branches) would show whether the segment-based cache's fixed overhead is a
one-time cost paid once per topology or a recurring one paid per branch,
and whether it is worth special-casing away for the specific
few-growth-steps case. `benchmark_src`'s `tree` mode star topology remains
impractical to run at the task brief's literal `n_sites=10` scale due to
the pre-existing, unrelated `zipup` cost blowup already flagged in the
2026-08-29 worklog -- this remains out of scope for both that worklog and
this plan, since it lives in `contract_zipup_impl`, a code path neither
touches.
