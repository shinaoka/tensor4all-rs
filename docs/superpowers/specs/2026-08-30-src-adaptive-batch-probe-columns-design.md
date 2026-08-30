# Batch-Native Adaptive SRC Probe/QR Interface Design

## Status

Proposed 2026-08-30, following the SRC per-run non-determinism investigation
recorded in `docs/worklogs/2026-08-30-src-probe-order-nondeterminism.md`.

## Goal

Close (a meaningful, measured fraction of) the wall-clock gap between this
crate's adaptive SRC contraction and the reference Python implementation
(`chriscamano/RandomMPOMPS`), by eliminating a per-growth-step
split-then-restack round trip that currently sits on both the chain
(`src_chain.rs`) and general-tree (`src_tree.rs`) adaptive paths, without
changing the algorithm itself (the sketch -> incremental-QR -> projection
core is Camaño-Epperly-Tropp's Algorithm 1 / Appendix C, already implemented
correctly per the 2026-08-26 reimplementation and its provenance audit; this
spec only changes how columns are batched and handed to that unchanged core).

No hard numeric target is set (see "Measurement" below) -- the repository's
existing practice (`docs/worklogs/2026-08-29-src-tree-path-performance.md`)
is to measure first and report the actual effect rather than force a change
to hit a preset number.

## Problem, with evidence

Measured this session (release build, single-threaded: `RAYON_NUM_THREADS`/
`OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS=1`, `benchmark_src`
vs. a matching standalone Python script against the same reference
implementation, `n_sites=10`, bond dim 4/8/16/32):

| bond_dim | zipup rust/python | src-fixed rust/python | src-adaptive rust/python |
| -------: | -----------------: | ----------------------: | -------------------------: |
|        4 |               28.6x |                   10.7x |                      11.8x |
|        8 |                4.5x |         0.06x (Python's `FixedDimension` here forces an unneededly high rank; not a fair comparison) |                       5.6x |
|       16 |                0.9x |                    0.2x |                       4.4x |
|       32 |                0.4x |                   ~1.0x |                       5.2x |

`zipup` and `src-fixed`'s Rust/Python ratio shrinks (and inverts) as problem
size grows -- ordinary fixed-per-call-overhead behavior, benign, not
addressed here. **`src-adaptive` alone stays in the 4-12x range across every
bond dimension tested, not shrinking with size** -- evidence of a real,
size-independent inefficiency in how the adaptive algorithm is realized, not
just amortizable fixed cost.

A minimal isolated micro-benchmark (`contract()` on tensors with every
dimension set to 1 -- zero real FLOP content) still cost ~68us/call, matching
the ~70-230us/call range seen in `benchmark_src`'s own
`T4A_PROFILE_CONTRACT=1` output. Manual phase timing inside
`contract_with_options_impl`/`contract_impl` showed the generic
connectivity-check and contraction-plan-building steps cost under 10us
combined; **over 90% of each call's cost is inside
`tensor4all-tensorbackend::einsum_native_tensor_reads`**, which dispatches
into `tenferro`'s compile-program/scoped-execute engine. This crate has no
visibility into, or authority to change, that engine's per-call cost -- see
"Non-goals".

Given that fixed per-call tax is real and roughly constant regardless of
tensor size, the fix available to this crate is reducing *how many* such
calls the adaptive algorithm issues per unit of useful work. Tracing the
adaptive growth loop found exactly that: each `rank_increment`-sized growth
step currently pays roughly `rank_increment` `select_indices` calls (splitting
an already-computed batch tensor into individual columns) plus a
`stack_along_new_index` + `permute_indices` + `.to_vec()` sequence
(re-assembling those same individual columns back into one tensor to feed
`IncrementalQr::append`, which already accepts a whole `Matrix` of new
columns at once). The chain path's last-site step is worse still: it never
uses the already-existing `PrefixCache::fresh_segment` batched fast path at
all, and contracts each column against the site's local tensors one at a
time. The general-tree path's adaptive `EnvironmentCache::ensure_width`/
`column` is worse yet: it recomputes one whole-tree `directed_messages` pass
per single new probe column, never batching at all (this was already flagged,
but judged low-priority, by `docs/worklogs/2026-08-29-src-tree-path-performance.md`
-- see "Relationship to prior work").

## Non-negotiable invariants

- **Numerics**: every existing dense-oracle regression test
  (`src_fixed_matches_exact_contraction_when_probe_cap_is_full`,
  `src_adaptive_matches_exact_contraction_on_a_small_chain`, the isometry
  check, etc.) must keep passing at its existing tolerance. This is a
  call-batching change, not an algorithm change.
- **RNG/probe-stream semantics**: `ProbeBank`'s append-only prefix property
  (an adaptive run observes exactly the same prefix as a fixed-width run with
  the same seed) is unchanged -- this spec does not touch `ProbeBank` or
  `generate_id()`/index identity at all.
- **No `tenferro` changes**: everything in this spec stays inside
  `tensor4all-core`/`tensor4all-treetn`/`tensor4all-tensorbackend`. See
  "Non-goals".
- **No silent behavior change on cache-reuse paths**: the current fallback
  (fetch already-cached individual columns, `stack_along_new_index` them)
  that lets a later-processed site/edge reuse an earlier one's wider,
  already-computed prefix/environment cache must keep working -- see
  "Design, Section 3" for why a naive per-request batch interface would have
  silently regressed exactly this reuse path.

## Design

### 1. Layering

Three layers, matching where responsibility already lives:

1. **`tensor4all-core/src/defaults/idx_tensor.rs`**: turns "one already
   batch-indexed tensor" into a `Matrix<S>` and feeds `IncrementalQr`.
   Doesn't know about chain vs. tree.
2. **`src_probe.rs`** (shared by both paths): the adaptive growth/stopping
   loop (how wide to grow, when the error estimate says stop). Doesn't know
   how a batch is produced.
3. **`src_chain.rs` / `src_tree.rs`**: each produces its own next batch,
   using its own topology-specific cache.

### 2. New interfaces

`tensor4all-core/src/defaults/idx_tensor.rs`:

```rust
// Replaces probe_columns_matrix(columns: &[&IdxTensor], left_inds) -> Matrix<S>
fn probe_batch_matrix<S: TensorElement>(
    batch_tensor: &IdxTensor,
    batch_index: &DynIndex,
    left_inds: &[DynIndex],
) -> Result<Matrix<S>, FactorizeError>;
// permute_indices(left_inds + [batch_index]) once, then to_vec() once.
// No stack_along_new_index: the input is already batch-shaped.

// Replaces factorize_probe_columns_incremental(previous, all_columns, appended_columns, left_inds)
fn factorize_probe_batch_incremental(
    previous: Option<&FactorizeResult<IdxTensor>>,
    batch_tensor: &IdxTensor,
    batch_index: &DynIndex,
    left_inds: &[DynIndex],
) -> Result<FactorizeResult<IdxTensor>, FactorizeError>;
// previous == None: batch_tensor is the from-scratch first batch (IncrementalQr::new).
// previous == Some: batch_tensor is this step's newly appended batch (IncrementalQr::append).
// No separate all_columns/appended_columns views needed.
```

`src_probe.rs` (shared growth loop):

```rust
// Replaces factorize_probe_columns(left_indices, initial_width, maximum_width, src_options, label, make_column)
pub(super) fn factorize_probe_batches<T, F>(
    left_indices: &[T::Index],
    initial_width: usize,
    maximum_width: usize,
    src_options: &SrcOptions,
    label: &str,
    mut make_batch: F,
) -> Result<(T, T::Index)>
where
    F: FnMut(usize /* start */, usize /* width */) -> Result<(T, T::Index)>;
```

Growth/stopping logic is unchanged from `factorize_probe_columns`; only the
per-step body changes from "call `make_column` `width - previous_width`
times, accumulate into a `Vec<T>`" to "call `make_batch(previous_width, width
- previous_width)` once."

### 3. Segment-based caches (the reuse-preserving part)

`PrefixCache` (chain) and `EnvironmentCache` (tree) currently cache
*individual columns*. Naively making `make_batch` always return a freshly
constructed batch tensor would force a `stack_along_new_index` even when a
later-processed site/edge is simply reading a range some *earlier* one
already computed and cached -- today that read is a zero-cost `Vec::get`.
Since both caches already serve their *entire* chain/tree from one shared
growth call (a single `grow_one_segment`/`directed_messages_batched` call
updates every site's/edge's cache entry at once), this reuse is common, not
an edge case, and must not regress.

Fix: cache **segments** (whatever chunk one growth call produced, already
batch-shaped), not individual columns.

```rust
// PrefixCache, per site:
segments: Vec<Vec<(T, T::Index, usize)>>, // (batch tensor, batch index, width) per site
total_width: usize,

// EnvironmentCache:
segments: Vec<(usize /* start */, usize /* width */, HashMap<(V, V), T>)>,
total_width: usize,
```

`request(site_or_edge, start, width)` replaces `column`/`fresh_segment`:

- `start + width <= total_width`: look up the covering segment(s).
  **Common case** (`start`/`width` align to one stored segment's boundary):
  return that segment directly, no extra work. **Misaligned case** (a
  request spans a partial segment boundary): fall back to fetching the
  covering columns via `select_indices` and `stack_along_new_index`-ing them
  -- the same mechanism as today, now rare rather than routine (see below).
- `start + width > total_width`: grow (`grow_one_segment` /
  `probed_site_pair_batch_range` + `directed_messages_batched`) to cover the
  gap, store the new segment(s), then resolve as above.

**Why misalignment is rare, not routine, after this change**: every caller
grows in `sketch_options.rank_increment`-sized steps starting from 0 (the
same shared value for every site/edge within one `contract()` call), so
segment boundaries naturally line up across callers *except* right after a
call whose own `maximum_width` capped a growth step short of a full
`rank_increment` (a "ragged" final segment for that call). A later
caller needing to grow past that specific ragged boundary is the only case
that still pays a restack -- bounded by the number of distinct
`maximum_width` values seen in one `contract()` call (a handful), not by the
number of columns.

Tree's `EnvironmentCache` doesn't have chain's "different sites, same shared
cache" reuse subtlety in the first place -- one `directed_messages_batched`
call inherently produces every edge's segment together -- so this class of
misalignment cannot arise there; its segment cache is simpler.

> **Correction (recorded after implementation, 2026-08-30):** This claim
> turned out to be false. Task 5's review found that `EnvironmentCache`,
> when shared across multiple tree edges with different required widths,
> DOES need a fixed growth grid and a genuine misaligned-request fallback,
> mirroring the chain path -- the same class of misalignment this section
> claims cannot arise on the tree side. The finished implementation has
> both a fixed grid and the fallback, plus a dedicated regression test,
> `request_shared_across_edges_with_misaligned_widths_matches_a_direct_reference`
> in `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`, proving
> misalignment does arise there too. This paragraph is left as originally
> written for the historical record; see the Task 5 fix round for what was
> actually built instead.

### 4. Call-site changes

- **`src_chain.rs` interior sites**: the closure drops its final
  per-position `select_indices` split-into-`pending_columns` loop entirely;
  it returns the batch-contracted tensor straight from the existing
  `contract_retaining` three-call sequence. `pending_columns`/`next_column`
  bookkeeping is deleted (no longer needed: batches are requested directly by
  range, not pulled one at a time).
- **`src_chain.rs` last site**: switches from `prefixes.column(...)` (never
  batched, never used `fresh_segment`) to `prefixes.request(...)`, and
  contracts the whole batch against `local[last].0`/`.1` once per growth step
  instead of twice per column.
- **`src_tree.rs` `EnvironmentCache`**: `ensure_width`/`column`/`batch`
  collapse into the single `request` method described above, backed by
  `probed_site_pair_batch_range` + `directed_messages_batched` (already used
  by today's fixed-rank `batch()`) instead of the adaptive path's current
  one-column-at-a-time `probed_site_pair` + `directed_messages`.

  > **Correction (recorded after implementation, 2026-08-30):** This is not
  > what was actually done, correctly. The plan's own Global Constraints
  > explicitly kept `batch`/`batched_environments` (the fixed-rank tree
  > path) untouched and out of scope, and Task 6 only replaced
  > `ensure_width`/`column` (the adaptive path) with `request`, leaving
  > `batch` as its own separate method. This paragraph's wording is stale
  > relative to what the plan and implementation correctly decided to do
  > instead; it is left as originally written for the historical record.

### 5. Error handling

No new error categories. The rare misaligned-boundary fallback path gets an
error message naming the site/edge and the requested/cached ranges (so a
future profiling session that finds this path unexpectedly hot has enough
context without re-deriving it). Everything else keeps today's
`anyhow::anyhow!("contract_src: ...")` context conventions.

### 6. Migration and cleanup

`factorize_probe_columns` / `factorize_probe_columns_incremental` /
`probe_columns_matrix` currently have exactly two call sites (chain's
interior+last-site adaptive path, tree's adaptive path) and no other
consumers. Once both migrate to the batch-native versions, delete the old
ones outright -- no compatibility shim.

## Testing

1. **Existing dense-oracle and isometry tests** (`src_fixed_matches_exact_contraction_when_probe_cap_is_full`,
   `src_adaptive_matches_exact_contraction_on_a_small_chain`,
   `src_result_tensor_is_numerically_isometric`, etc.) must keep passing
   unmodified at their existing tolerances.
2. **New: cross-site/edge segment-reuse test.** A fixture where two sites (or
   two edges, for the tree variant) have different `maximum_width` values
   such that one needs strictly more columns than the other, with the
   smaller one's cap landing on a non-`rank_increment`-aligned width (forcing
   the rare fallback path in the same test that exercises reuse). Assert (a)
   the final contraction still matches `contract_naive` within tolerance,
   and (b) the shared cache is not recomputed from scratch for the
   already-covered range (a call counter around `grow_one_segment` /
   `probed_site_pair_batch_range`, asserting the count matches the expected
   number of *new* segments, not columns).
3. **Performance measurement, not a gate.** Re-run `benchmark_src` at the
   same bond dimensions used in this spec's "Problem" section
   (single-threaded, before/after), and record the result the same way
   `docs/worklogs/2026-08-29-src-tree-path-performance.md` did. Include the
   same Python-comparison numbers so the before/after is legible against the
   original motivation, not just against the old Rust baseline.

## Non-goals

- Changing anything inside `tenferro` (`tenferro-rs`, a sibling repository
  this crate has no PR access to, and which is expected to already be more
  mature than this crate's own code -- a change there is only in scope if a
  genuine bug is found with very high confidence, which this investigation
  did not find: the per-call cost looks like inherent engine dispatch
  overhead, not an obvious defect).
- Redesigning the general-tree path's message-passing algorithm itself
  (e.g. the scan/blocking/checkpoint ideas discussed on issue #563). This
  spec only batches the *existing* per-column message-passing calls into
  per-segment ones; it does not change what gets computed.
- Touching `ProbeBank`, `generate_id()`, or anything from the
  non-determinism investigation earlier this session -- unrelated concern,
  already fixed and committed separately.
- A hard performance target. See "Testing" item 3.

## Relationship to prior work

`docs/worklogs/2026-08-29-src-tree-path-performance.md` measured the same
tree-path adaptive cache and concluded (at the problem sizes it tested) that
redundant re-materialization was "a modest fraction... not the dominant
driver," with the dominant driver attributed to the tree path's architecture
(whole-tree message passing per width) rather than to F-4/F-5's specific
redundant-recomputation claim. That worklog's measurement was in wall-clock
terms, without visibility into per-call fixed cost -- this session's
`T4A_PROFILE_CONTRACT=1` and isolated micro-benchmark evidence (a ~60-90us
floor per call regardless of tensor size) puts a different number on exactly
the same "whole-tree message passing per width" cost that worklog already
identified as dominant: if every one of those per-column whole-tree passes
pays a ~60-90us fixed tax on top of its real work, batching them into
per-segment passes (this spec's change) directly attacks that previously
un-quantified fixed cost, not a separate mechanism.
