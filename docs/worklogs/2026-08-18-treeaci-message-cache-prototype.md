# Does a persistent message cache close the TreeACI/TT gap, and what's next?

Work log. Continues from `2026-08-17-treeaci-per-evaluation-cost.md` (retracted
by `2026-08-17-treeaci-train-parity.md`, which was itself never posted back to
the #646 review thread before this work started). Covers: re-confirming
non-convergence at realistic bond dimension, measuring the real cache-hit
potential on the actual call pattern, building and wiring a prototype
persistent cache, and measuring that the prototype does not close the gap --
plus why, and what does.

## Where this picks up

`2026-08-17-treeaci-train-parity.md`, committed as `f10bf9c0` at 14:06:55 UTC
on 2026-08-17 -- five minutes after Hiroshi's 13:56:47 UTC review comment
asking for exactly this measurement -- found that TreeACI pays `O(chi^2)` per
evaluated point where `tensor4all-aci` does not, and explicitly retracted the
prior worklog's "defer the cache" recommendation. That result was never
communicated back to the #646 thread. This session found the gap first
(independently, before discovering `f10bf9c0` already existed), then found the
existing worklog, then proceeded past it: is the fix a cache, and if so, does
prototyping one actually help.

## Step 1 (repeat): TreeTNCachedEvaluator vs TTCache at realistic bond, warm call

Extended `crates/tensor4all-treetn/benches/cached_evaluator.rs` with a new
`bench_treetn_vs_ttcache_large_bond` group: both evaluator and `TTCache` built
outside the timed closure, same chain, same floating-zone two-point batch,
`TreeTNCachedEvaluator`'s centre and `TTCache`'s split both pinned to the
varying site, bond in {32, 64, 128, 256}, `n_sites` in {8, 32}.

First run used the same `TTCache` object across all 100 criterion samples
without clearing it. Result looked decisive -- TTCache flat at 1.4-2.4 us
regardless of bond -- and wrong: at bond 256, `n_sites=32`, 2.4 us implies
roughly 240 GFLOP/s for what should be `O(n_sites * bond^2)` work. The
instrument was measuring a warm cache hit on the exact same repeated indices,
not a cold contraction: `TTCache::evaluate_many` memoizes by index, and every
sample after the first was a lookup.

Fixed with `TTCache::clear_cache()` in criterion's `iter_batched` setup so
every sample is a cold contraction, matching `TreeTNCachedEvaluator` (which
never gets a cross-call cache to hit, since it rebuilds `environment_cache`
locally every call). Corrected result, `cargo bench -p tensor4all-treetn`:

| bond | n=8 TreeTN | n=8 TTCache(cold) | ratio | n=32 TreeTN | n=32 TTCache(cold) | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2.048 ms | 1.51 us | 1355x | 7.675 ms | 1.96 us | 3912x |
| 64 | 2.417 ms | 1.49 us | 1620x | 10.758 ms | 2.20 us | 4901x |
| 128 | 5.185 ms | 1.70 us | 3060x | 21.849 ms | 3.19 us | 6850x |
| 256 | 5.714 ms | 2.80 us | 2038x | 22.325 ms | 13.01 us | 1716x |

No convergence at realistic bond; the gap is larger than the chi=2 measurement
suggested (80-110x) and does not shrink monotonically with bond. This
corroborates `f10bf9c0`'s independent finding via a different method (single
warm call vs. whole-run `elementwise`), reusing neither its benchmark nor its
tree construction. The bench file addition (`bench_treetn_vs_ttcache_large_bond`)
was reverted after recording these numbers here, since `f10bf9c0` already
committed the more decisive, pre-registered version of this comparison and a
second uncommitted copy would only duplicate it.

## Step 2: would-be cache-hit rate on the real call pattern

Hiroshi's ordering (13:56:47 UTC comment) asks for this before building
anything: measure hit rate and bond distribution on the real workload, not a
synthetic one. Added temporary instrumentation to `TreeTNCachedEvaluator`
(`message_log: Vec<(V, Vec<usize>)>`, `#[cfg(test)]`-gated, since removed) that
recorded every `(node, full_point)` pair `build_environment_cache` computed a
message for, then drove `tensor4all_core::floating_zone_walk` -- the same
primitive `find_global_pivots` uses -- with `find_global_pivots`'s own
defaults (`nsearch_global_pivots = 5`, `nsweeps_global_search = 100`) against a
persistent evaluator on a 16-site chain, bond 128.

First key was `index_vals_for_point`, which returns only a node's own local
physical entry, not its subtree. Result: 99.18% hit rate -- implausible,
because it collapses distinct messages that merely share one node's local
value. Corrected to the full evaluated point per message (always correct,
since two identical full points must produce identical subtree assignments
everywhere; coarser than a true subtree-restricted key, so this
undercounts achievable reuse rather than overcounting it). Corrected result:

**3675 total message computations, 941 unique, hit rate 74.4%** (a lower
bound). Reverted after recording this number; not committed.

## Step 3: prototype the cache

Built `PackedMessageCache<K, T>` in `cached_evaluator.rs`, TDD throughout
(RED verified for each test before implementation): batch get-or-compute with
per-key dedup, cross-call persistence, a byte budget that stops retaining
without evicting once exhausted (`CacheSlot::Cached(usize)` /
`CacheSlot::Uncached(Vec<T>)`, matching Hiroshi's "continue evaluating new
messages without caching them" rather than evicting to make room), and
hit/miss counters. One method (`get_all_cached`) was written just ahead of its
test rather than strictly after, a minor deviation from the RED-first
discipline, noted here rather than silently passed over.

Wired into `TreeTNCachedEvaluator` via a new `get_or_compute_node_message`,
replacing the direct `compute_stacked_message` call in
`build_environment_cache`'s postorder loop. `RootedMessagePlan` gained a
`parent: ParentMap<V>` field (previously computed and discarded) so a node's
parent-bond `DynIndex` is resolvable via the existing
`TreeTN::edge_between`/`bond_index` API (found by a research pass rather than
guessed: getting this index identity wrong would silently connect the wrong
tensor axes, which is why it was not attempted without first confirming the
API existed).

First version cached at node-batch granularity: a node's whole
`assignment_batch` had to be entirely cache-hit to skip computation, otherwise
the whole batch was recomputed through the unchanged path (loses to a partial
hit inside a batch of 2). Verified correct against `tree.evaluate` (the
uncached oracle) via `assert_scalars_close` at every step; 4 cache tests plus
one evaluator-level integration test, all passing throughout.

## Step 4: does it actually help? (measured, not assumed)

Added `message_cache_wall_time_on_realistic_floating_zone_walk`: the same
`find_global_pivots`-pattern walk as step 2, run twice on identically
constructed trees (`build_tree(7)`, same seed) -- once through the normal
(now cached) `evaluate_batched`, once through a hand-duplicated uncached path
(`evaluate_batched_uncached`, calling `compute_stacked_message` directly) --
on one evaluator per arm so construction cost cancels, `Instant`-timed in
release build.

**First result: cached 3.696s, uncached 2.028s -- the cache made the walk
1.8x slower**, despite the 74.4% hit rate confirmed to reproduce exactly
(2734 hits / 941 misses / 3675 total, matching step 2's count exactly).

Iterated, remeasuring the same walk after each fix:

| change | cached vs uncached |
|---|---:|
| node-batch granularity, full-point `Vec<usize>` key | 0.55x (82% slower) |
| + point-level partial caching (only missing points recomputed, merged with cached columns in original order; required generalizing `compute_stacked_message` to take an explicit `points: &[usize]` subset instead of always reading `assignment_batch.first_points`) | 0.79x |
| + memoize `edge_between`/`bond_index` per node (constant under a fixed rooting; was being looked up on every call) | 0.93x (three repeated runs: 0.83-0.89x -- within host noise of 1.0x, not confidently below or at parity) |
| + swap the key from full-point `Vec<usize>` to #645's `IndexKey` (`FlatIndexer::encode`, from `origin/issue-628-index-key`, merged into this branch per this branch's own decision to use it directly rather than wait for #645 to land) | 0.83-0.89x across three runs -- **no improvement, possibly slightly worse** |

Every step preserved correctness (`assert_scalars_close` against the uncached
oracle; full `cargo test -p tensor4all-treetn --lib` green throughout, 414
tests, `cargo clippy -- -D warnings` clean).

### Why the key swap didn't help: the real cost is tensor construction, not key size

The IndexKey result contradicted the working assumption that key
construction/hashing was the dominant remaining overhead. Rereading Hiroshi's
own per-primitive breakdown (`2026-08-17-treeaci-per-evaluation-cost.md`):
backend session (13%) and `EagerTensor` construction (15%) are 28% of one
message's cost and are independent of whether the contraction itself runs.
`get_or_compute_node_message` calls `IdxTensor::from_dense_any` to materialize
a `StackedMessage.tensor` on **every** call, hit or miss, because the parent's
own `compute_stacked_message` takes that field as an `IdxTensor` operand. The
cache removes the contraction and slice/gather cost (66% + 30% of the original
breakdown) but never removes the construction/session cost, so at high hit
rates the remaining fixed cost per message is a larger fraction of a smaller
total, and the extra bookkeeping (key construction, cache lookups, the merge
loop) does not have enough removed work left to pay for itself.

This matches Hiroshi's own line from the 14:01:26 UTC comment, read
initially as a caution and turned out to be load-bearing: "Do not allocate one
output tensor for every cache entry."

### What actually would fix it: a proven, already-shipped precedent

`crates/tensor4all-simplett/src/einsum_helper.rs::row_vector_times_matrix`
(the primitive under `TTCache::evaluate_left`/`evaluate_right`, i.e. the
reason TTCache is fast) does not call `tenferro-einsum` at all. Its own
comment: "`einsum_tensors` re-traces and re-compiles the graph on every call
... which costs ~70 us even for a 2x2 product... so the mat-vec is computed
inline." It is a hand-written loop over `&[T]` slices.

Replicating this for `tensor4all-treeaci`'s message computation means
replacing `StackedMessage.tensor: IdxTensor` with a packed-buffer
representation carried through the whole leaf-to-centre pass, and writing an
N-ary generalization of `row_vector_times_matrix` (the train side never needs
more than one child, so no N-ary version exists yet) matching the contraction
shapes Hiroshi specified: `abq,aq->bq` for one child, `abcq,bq,cq->aq` for two,
generalizing by node degree. This is what Hiroshi called "step 4" and
separated from "step 3, prototype the cache" -- confirmed here to be a real
separation, not an ordering preference: the cache alone cannot close the gap
because the fixed cost it doesn't touch is where the remaining time is.

## What this changes about the #646 plan

- The persistent cache (step 3) is built, wired, and correctness-verified, but
  does not deliver a net win on its own -- report this plainly rather than as
  a completed optimization.
- The next lever is the packed-buffer contraction path (step 4): no
  `IdxTensor` construction in the per-message hot path, an N-ary raw-loop
  contraction kernel following `row_vector_times_matrix`'s proven shape, and
  `PackedMessageCache`'s already-packed `Vec<AnyScalar>` columns feed it
  directly with no conversion once it exists.
- `message_cache_wall_time_on_realistic_floating_zone_walk` and
  `evaluate_batched_uncached` are kept (not reverted) specifically to
  re-measure this once the packed-buffer path lands.
- `2026-08-17-treeaci-train-parity.md`'s finding and this session's findings
  should both be posted to #646 -- neither has been communicated to Hiroshi
  yet as of this commit.

## What this does not establish

A chain topology throughout (matching every existing bench/test in this file).
Branching correctness of the point-level partial-cache path is exercised by
the existing branched unit tests in `cached_evaluator.rs`'s `mod tests`
(pre-existing fixtures, unchanged), but the wall-clock measurement above is
chain-only; a branched-tree wall-clock measurement has not been made. The
`IndexKey` swap uses one `FlatIndexer` built from all evaluated indices'
dimensions at construction time; it has not been checked against non-uniform
local dimensions beyond what the existing `index_key` test suite (merged from
#645) already covers.

## Update: root-caused and fixed for chain topology (same session, systematic debugging)

The "what fixes it" section above was itself a hypothesis, arrived at by
extrapolating Hiroshi's bond=2 primitive-cost table
(`2026-08-17-treeaci-per-evaluation-cost.md`) rather than by measuring this
code path directly. Continuing under `superpowers:systematic-debugging`
rather than starting the step-4 rewrite on that assumption:

**Phase 1, evidence.** Added `#[cfg(test)]`-gated `AtomicU64` phase timers
inside `get_or_compute_node_message`
(`message_cache_phase_breakdown_on_realistic_floating_zone_walk`, kept) and
reran the same 16-site/bond-128 walk. Result:

```
key_and_lookup=4.7ms (0.2%) contract=619.9ms (25.1%) tensor_values=1760.1ms (71.3%)
insert=9.2ms (0.4%) reconstruct=74.6ms (3.0%) total=2468.6ms
```

`IdxTensor::from_dense_any` (the reconstruct step, i.e. the mechanism the
original hypothesis blamed) is 3.0%. The actual dominant cost, at 71.3% and
**three times the contraction that produced the data**, is
`tensor_values_any` -- converting a cache miss's freshly contracted message
back into a plain `Vec<AnyScalar>` so it can be stored.

**Root cause, traced into `tenferro-ad`'s source rather than inferred:**
`IdxTensor::to_vec` calls `EagerTensor::duplicate_value`, which has a cheap
path (`duplicate_host_tensor`, a plain slice copy) and an expensive fallback
(`with_execution_session(|s| s.to_contiguous_read(...))`) taken whenever the
value is non-contiguous or not yet host-resident. A `contract_with_options`
result hits the fallback every time; the uncached code never paid this cost
because it keeps every intermediate message as an `IdxTensor` and only
materializes once, at the very end, for the final scalar output. The cache
introduced a new per-miss materialization that the original code structurally
avoided.

**Fix (chain topology; branching and complex scalars fall back to the
existing generic path).** `try_compute_leaf_message_raw` and
`try_compute_chain_message_raw`: hand-written loops over each node's own
`Vec<f64>` (read once, host-resident by construction) and, for a one-child
node, the child's already-`Vec<f64>` message, generalizing
`row_vector_times_matrix` from `crates/tensor4all-simplett/src/einsum_helper.rs`
to the `abq,aq->bq` chain-message shape. Never calls `contract_with_options`
or produces a non-contiguous intermediate, so there is nothing for
`to_vec`/`duplicate_value` to fall back on. TDD throughout: each kernel has a
test asserting bit-for-bit agreement with `compute_stacked_message`'s existing
(oracle) output before being wired in
(`raw_leaf_message_matches_generic_contraction`,
`raw_chain_message_matches_generic_contraction`).

**Result**, same walk, three repeated runs: **0.96x, 1.40x, 1.09x** -- cached
is now faster than uncached, versus 0.83-0.89x before this fix. Phase
breakdown after the fix: `total` drops from 2468.6ms to 2014.4ms; `contract`
(now covering both the raw and any remaining generic-path computation)
absorbs what were separately the contract and tensor_values phases, since the
raw path produces no separately-timed materialization step at all.

417 tests pass (up from 414: three new tests for this fix, all TDD'd
RED-then-GREEN), `cargo clippy -- -D warnings` clean, `tensor4all-treeaci`
still compiles against the unchanged public surface.

**What is not yet covered by the fix:** nodes with two or more children
(branching) and complex-valued (`Complex64`) trees still go through the
original `compute_stacked_message`/`contract_with_options` path and pay the
same materialization cost measured above. `try_compute_chain_message_raw`'s
eligibility check (`entries.len() == 1`, `children.len() == 1`,
`!tensor.is_complex()`) returns `Ok(None)` for those cases rather than
guessing, so correctness cannot regress for them -- but they get none of this
session's speedup. Generalizing to N children, matching Hiroshi's
`abcq,bq,cq->aq` shape and beyond, is the natural next increment and was not
attempted here.

## Update 2: this fix didn't move the number that matters, so it kept going

Rerunning `crates/tensor4all-aci/benches/treeaci_parity.rs` (the same
end-to-end tree-vs-train comparison `2026-08-17-treeaci-train-parity.md`
used) after the fix above showed the tree/train ratio essentially unchanged:
677x -> 622x at chi=128, within run-to-run noise. The reason: that benchmark
sets `enable_global_guard: false`, so `find_global_pivots` -- the only place
the message cache this session built actually runs -- never executes. The
fix was real and verified, but for a code path this benchmark does not
exercise.

**Phase 1, again, on the actual dominant path.** Added temporary
`AtomicU64` instrumentation (reverted after use, not committed) around
`contract_prepared_core` in `crates/tensor4all-treeaci/src/frames.rs` and ran
the same parity benchmark. At chi=128: **this one function was 20.7s of the
21.4s total tree run -- 96.7% -- visiting 4.99 billion elements across
582,282 calls**, at a remarkably constant 4.05-4.56 ns/element across all
four chi values (consistent with a fixed per-element cost, not one that
improves with scale).

**Root cause.** `contract_prepared_core` scanned every element of a node's
*entire* dense core (`left_dim x local_dim x right_dim`) for every candidate
frame, decoding each element's per-axis coordinate via integer division and
modulo (`axis_coordinate`), and discarding (`continue`) whatever did not
match the wanted physical value -- rather than computing the one flat offset
that already selects the right physical slice and iterating only the axes
that are actually contracted.

**Fix.** Compute the physical-fixed base offset once (a sum of `wanted axis
coordinate x stride`, one add per physical axis, no division), then a
recursive helper (`accumulate_incoming`) walks the cartesian product of the
incoming axes directly via stride arithmetic, touching exactly
`outgoing.dim() * product(incoming dims)` elements -- never the padded
`local_dim`-inclusive volume, never a per-element divmod. Existing tests in
`crates/tensor4all-treeaci/src/frames/tests/mod.rs` (two-node chain, branching
Y-tree, multiple-physical-axes-per-node, real and complex scalars, resource
limits) already covered this function's correctness and served as the
regression suite; all 6 passed unchanged before and after.

**A second bug found by running the full downstream suite, not just the
crate I was editing.** After the frames.rs rewrite, running
`tensor4all-treeaci`'s own test suite (not just `tensor4all-treetn`'s, which
this session had been checking after every change) surfaced two failures
that predated the frames.rs work: `global_guard_recovers_a_separated_feature_end_to_end`
and `...at_the_default_search_count`, both failing with "center contraction
left non-scalar indices" -- and confirmed (`git stash` + retest) to already
fail at commit `09656ad8`, i.e. caused by the message-cache work from Update 1,
not by this update. Root cause: `global_guard.rs` calls
`evaluate_batched_with_hint` with a *different* centre on every call
(`EvaluationHint::around`, pinning the contraction centre to whichever site
the current batch varies). A node's "message toward its parent" names a
different neighbour under a different rooting, so `message_caches` and
`parent_bond_indices` -- keyed only by node, implicitly assuming one fixed
rooting -- served stale-and-wrong data across a centre change. This is a
correctness bug, not merely a missed optimization: it would have shipped
silently wrong numbers in any run where the guard's per-call centre hint
actually changed between calls, most of the time.

Fixed by tracking which centre the caches were built for
(`rooted_for_center: Option<V>`) and clearing both caches whenever
`build_environment_cache` is called with a different one. This is exactly
correctness-first: it costs cache reuse across a centre change, but a fast
wrong answer is worse than a slow right one. `message_cache_wall_time_on_realistic_floating_zone_walk`
does not exercise this path (it never passes a hint, so centre never
changes), which is why it did not catch the bug -- a gap in that test's
realism, noted for anyone extending it.

**Result**, `treeaci_parity.rs` chi=16/32/64/128, before vs after the
frames.rs fix (both with the global-guard correctness fix also applied,
though inert here since the guard is disabled in this benchmark):

| chi | train | tree before | tree after | ratio before | ratio after |
|---:|---:|---:|---:|---:|---:|
| 16 | 22.9 ms | 455.6 ms | 302.2 ms | 26x | 13.2x |
| 32 | 16.9 ms | 1934.3 ms | 1207.1 ms | 139x | 71.4x |
| 64 | 28.6 ms | 7983.5 ms | 4723.3 ms | 303x | 165.2x |
| 128 | 36.0 ms | 21594 ms | 14567 ms | 677x | 404.6x |

Roughly a 33-40% reduction in tree wall time at every chi, halving the ratio
to train ACI at each point. Not parity -- still 13x-405x slower -- but a real,
verified, end-to-end improvement from fixing the function that actually
dominates, unlike Update 1's fix. `tensor4all-treetn` (417 tests) and
`tensor4all-treeaci` (83 tests, including both previously-failing
global-guard tests) both green; `cargo clippy -- -D warnings` clean on both.

## What remains

The gap did not close, only halve. `contract_prepared_core` was the single
largest item found, not necessarily the only one -- `local_update.rs`'s LU
factorization (`matrix_luci_factors_from_matrix_owned`) and the surrounding
candidate-enumeration/frame-assembly machinery have not been individually
profiled the way `contract_prepared_core` was. The natural next step is the
same discipline again: instrument the post-fix run, find what now dominates,
and decide from measurement rather than assumption whether it is worth
fixing.

## Update 3: the actual dominant cost was `InputFrameStore` discarding itself every commit (#646 continuation, systematic debugging)

Picked up from "What remains" above, continuing under
`superpowers:systematic-debugging`. Re-baselined `treeaci_parity` at
`fa2c47c9` first (measured, not assumed): 13.2x/71.4x/165.2x/404.6x at
chi=16/32/64/128, matching this file's own recorded numbers, confirming no
drift since Update 2.

**Phase 1, evidence.** Added temporary phase timers (not committed) around
`InputFrameStore::from_samples` and a call counter on
`FrameBuilder::compute`'s cache-miss branch, then reran `treeaci_parity` with
the criterion timing loop skipped (single un-timed run per chi, matching this
file's established methodology). Result: `from_samples` alone is 78.4% of a
chi=16 run and climbs to 96.3% at chi=128 -- both the dominant cost *and* the
reason the ratio grows with chi, not just a large constant.

**Root cause, traced into the call site rather than inferred.**
`transaction.rs::commit_edge_proposal` calls
`InputFrameStore::from_samples(state.inputs, &state.problem, &proposed_arena)`
after *every single directed-edge commit*, discarding the current store and
recomputing every sample on every directed edge from scratch --
`FrameBuilder`'s own memoization only covers repeated lookups *within* one
such call, never across calls. `samples.rs` confirms `SampleArena` is
append-only and `SampleId`s are immutable per directed edge (`intern_key`
only ever pushes, `PartialEq`-deduplicated) -- a sample already computed
names exactly the same component forever. So this was `O(edges)` real work
(recompute everything) repeated on `O(edges)` edge commits per sweep, i.e.
`O(edges^2)` where an incremental update needs only `O(edges)` total: the
same architectural class of bug as Update 1's fix, but for the *always-on*
frame store rather than the guard-only message cache, and much larger in
practice since this path runs on every commit, not only during global-pivot
search.

**Fix.** `InputFrameStore::extend(&self, inputs, problem, arena) -> Result<Self>`
(`frames.rs`): reuses every already-computed `DirectedFrame` row from `self`
(seeding `FrameBuilder`'s per-edge memo with them, converted back to
`Vec<T>` via a new `DirectedFrame::row` helper) and only calls
`contract_prepared_core` for samples in `existing_sample_count..new_count`
per directed edge. `from_samples` and `extend` now share one
`build_or_extend` implementation, parameterized by `existing: Option<&Self>`,
so there is one code path to keep correct rather than two. As a side effect,
`extend` also reuses `self.cores` instead of calling `prepare_cores` (which
re-extracts every input tensor's dense values via `to_vec::<T>()` on every
commit) -- free, since the fixed operands never change across the run.
`transaction.rs` now calls `state.input_frames.extend(...)` instead of
`InputFrameStore::from_samples(...)`.

TDD: `frames/tests/mod.rs` gained
`extend_matches_a_full_rebuild_on_the_grown_arena` (grows a `y_tree` arena by
one global point, asserts `extend`'s frame values equal a from-scratch
rebuild's for every sample, old and new -- correctness) and
`extend_recomputes_only_the_newly_interned_samples` (same setup, asserts
`extend`'s `contract_prepared_core` call count is strictly less than a
from-scratch rebuild's on the same grown arena -- proves the reuse claim,
not just correctness). The call counter is a `#[cfg(test)]`-gated
`debug_stats` module in `frames.rs`, kept as permanent regression
infrastructure rather than reverted, since it is what would catch a
regression back to full-rebuild-per-commit.

**Result**, `treeaci_parity.rs` chi=16/32/64/128, clean run (no
instrumentation) after the fix:

| chi | train | tree | ratio before | ratio after |
|---:|---:|---:|---:|---:|
| 16 | 21.7 ms | 81.8 ms | 13.2x | 3.8x |
| 32 | 14.1 ms | 180.6 ms | 71.4x | 12.8x |
| 64 | 24.7 ms | 391.9 ms | 165.2x | 15.9x |
| 128 | 30.6 ms | 911.9 ms | 404.6x | 29.8x |

The ratio's growth across the chi range shrank from roughly 30x (13.2x to
404.6x) to roughly 8x (3.8x to 29.8x): most, though not all, of the
super-linear-in-chi behaviour reported in the #646 review was this bug, not
an inherent property of the tree algorithm.

**What was checked and deliberately not fixed.** A second phase breakdown
after this fix (chi=128, single run) showed `from_samples` down to 37% of
total time; the next-largest item is `local_update.rs`'s `candidate_frame`
calls (28% and growing faster than the rest, roughly chi^1.7). Unlike the bug
above, these candidates are generally *not* already-interned samples --
`transaction.rs`'s own comment records that candidate sets are replaced
rather than accumulated specifically to keep candidate growth bounded
without an eviction policy, so caching every enumerated candidate would
undo a deliberate memory/recompute tradeoff, not fix a bug. The more likely
remaining lever is the same one Update 1 identified for the guard-only
message cache and did not reach: `contract_prepared_core` and
`local_update.rs`'s row/col dot-product loop are hand-written scalar loops,
while `tensor4all-aci`'s analogous frame and local-matrix construction
(`state.rs`, `local.rs`) routes through `mat_mul`/`matmul_checked_owned`
(BLAS). Not attempted here -- flagged for a follow-up investigation rather
than assumed.

Full verification: `cargo fmt --all -- --check` clean;
`cargo clippy --workspace --all-targets --exclude tensor4all-hdf5 --exclude
book-tests -- -D warnings` clean (the excluded crates fail to build in this
environment for an unrelated, pre-existing reason: no local HDF5 via
Homebrew); `cargo test -p tensor4all-treeaci -p tensor4all-treetn --release`
green end to end (unit, integration, and doctests, exit code 0).

## Update 4: the BLAS hypothesis above was wrong; `candidate_frame` was redundant, not merely scalar (#646 continuation)

Continuing under `superpowers:systematic-debugging`. Update 3's "what was
checked and deliberately not fixed" section guessed the remaining
`candidate_frame` cost was a missing-BLAS problem. That guess was never
measured on its own terms and turned out to be wrong.

**Phase 1, evidence.** Instrumented `local_update.rs`'s own phases
(candidates/frames/matrix_build/operator/factor) plus a hit/miss counter on
`InputFrameStore::candidate_frame`. Two findings, neither assumed:

- `local_update.rs`'s own row/col dot-product loop (`matrix_build`, the
  candidate BLAS hypothesis's actual target) was 0.8% of total time at
  chi=128 -- not a meaningful share of the workload. Not touched.
- Counting distinct `(input, directed_edge, local_coordinate, incoming)`
  keys against total `candidate_frame` calls: 45-65% of calls across chi =
  16..128 are exact duplicates of an already-computed candidate. Since a
  candidate's frame is a pure function of that key (plus the fixed input
  cores), a repeat is genuine wasted work, not new work as Update 3 assumed.

**Root cause.** `candidate_frame` (unlike the persistent `frames` cache
fixed in Update 3) had no cache at all: every pivot-search candidate, most
of which are proposed and never selected or interned into `SampleArena`, re-ran
`contract_prepared_core` from scratch on every call, including calls with
identical keys from earlier sweeps or neighbouring edges once ranks
stabilize.

**Fix.** A bounded cache on `InputFrameStore`, keyed by candidate identity,
sharing `retained_bytes`'s budget against `max_frame_bytes` (degrades by
skipping the cache insert once the shared budget is exhausted, never by
evicting or erroring -- `options.rs`'s doc for `max_frame_bytes` updated to
say so). TDD: `candidate_frame_hits_the_cache_on_a_repeated_lookup` (a
repeated identical lookup must hit and return the same value) and
`candidate_frame_stays_correct_when_the_shared_budget_has_no_headroom_for_caching`
(correctness survives even when caching is skipped).

**A self-inflicted regression, caught by measurement before it shipped.**
The first version carried the cache across `extend` calls (needed, since
duplicates recur across the whole run, not one local update) by deep-cloning
the `HashMap` each time. `extend` runs once per directed-edge commit --
exactly the call site Update 3 fixed for the same reason -- so this
reintroduced an `O(edges)`-deep-clone repeated `O(edges)` times, just
relocated from `frames` to the new candidate cache. A diagnostic single run
measured chi=128 total time *nearly doubling* (864ms -> ~1.7s) instead of
improving. Fixed by making the cache `Rc<RefCell<HashMap<...>>>`-shared:
`extend` now clones two `Rc` pointers (refcount bumps) instead of the map's
contents.

**Measurement was itself unreliable for a while, and that was diagnosed
rather than papered over.** After the `Rc` fix, repeated `cargo bench`
criterion runs disagreed with each other by 20-30% at the same chi, and
`tensor4all-aci`'s own arm -- untouched by this change -- showed confidence
intervals spanning nearly 2x within a single run (e.g. chi=64 tree:
`[531.72ms 731.54ms 948.45ms]`). Comparing CI width across every log
recorded this session showed a clear trend: baseline runs early in the
session had ~1-17% CI width; by this point the same unchanged `train` arm
was regularly 20-80%. This was host contention (`top -o cpu` caught
`ANECompilerService`/`ecosystemd`/`ecosystemanalyticsd` at 60-90% CPU after
a machine restart, and multiple `git worktree`s exist on this host that can
run concurrent builds), not a property of the code. Declining to conclude
from noisy runs and instead re-measuring once `top` showed CPU usage settled
produced consistent, tight results (CI width 0.5-3.3%), confirming the
fix's effect was real and small, not an artifact of the earlier noise.

**Result**, `treeaci_parity.rs` chi=16/32/64/128, two clean back-to-back
runs (with vs. without the candidate cache, same quiet host state,
`git stash` used to isolate the change so both arms saw identical
conditions):

| chi | ratio without cache | ratio with cache | improvement |
|---:|---:|---:|---:|
| 16 | 3.71x | 3.56x | 4.0% |
| 32 | 10.86x | 10.13x | 6.7% |
| 64 | 13.75x | 12.98x | 5.6% |
| 128 | 27.13x | 24.65x | 9.1% |

A real, reproducible, modest win that grows with chi (matching the
duplication mechanism: candidates recur more as sweeps and neighbourhoods
stabilize at higher rank), not the second `Update-3`-sized win a first,
noise-contaminated measurement briefly suggested. Consistent with a
back-of-envelope ceiling computed from the hit/miss data: eliminating every
duplicate `candidate_frame` call for free (zero cache overhead) would save
at most ~13% of chi=128's total time, since that phase is only ~28% of the
run; the observed 9.1% is a reasonable fraction of that ceiling once real
cache bookkeeping cost is accounted for.

Full verification: `cargo fmt --all -- --check` clean; `cargo clippy
-p tensor4all-treeaci --all-targets -- -D warnings` and `cargo clippy
--workspace --all-targets --exclude tensor4all-hdf5 --exclude book-tests
-- -D warnings` both clean; `cargo test -p tensor4all-treeaci --release`
(87 lib tests, up from 85) and `cargo test -p tensor4all-treetn --release`
(121 tests) both green, exit code 0 on every run.

## What remains after Update 4

`candidate_frame` is now cached but `frames` (its containing phase) is still
the largest single item in `local_update.rs` at high chi. The BLAS
hypothesis is now known-wrong for the row/col dot-product loop specifically
(measured at 0.8% of total); whether it would help `contract_prepared_core`
itself (the shared primitive both `frames.rs`'s persistent cache and the new
candidate cache route through) has not been measured and should not be
assumed either way without doing so first.

## Update 5: `TreeAciState::initialize` -- the same clone-everything shape a third time, and the largest single win yet (#646 continuation)

Continuing under `superpowers:systematic-debugging`, prompted by "the ratio
still grows a lot with chi, describe the current problem" rather than a new
hypothesis. Phase 1 evidence: added phase timers to `transaction.rs`
(materialize/arena_clone/intern/candidates_clone/pivots_clone/
output_clone_replace/verify/extend) covering the whole of
`update_edge_transaction`. At chi=128 these summed to only ~38% of total
wall time; the remaining ~62% ("other") was *larger and growing faster*
than the already-large `materialize` phase, and was not inside the sweep
loop at all -- it had to be somewhere between `tree_elementwise_batched`'s
entry and the first `update_edge_transaction` call.

Timed that boundary directly (`elementwise.rs`): `TreeAciState::initialize`,
called exactly once per run before any sweep, accounted for 58% of total
time at chi=128 on its own. Timed inside `initialize` (`state.rs`):
`bootstrap_samples` (213.0ms) and the initial, necessarily-non-incremental
`InputFrameStore::from_samples` (217.9ms, scaling ~chi^2) were the two
dominant pieces, ~92% of `initialize`'s cost between them.

**Root cause, in `bootstrap_samples` (`initialize.rs`).** For every edge,
the function enumerates candidate points one at a time until that edge's
target rank (~chi) is reached, calling `arena.inject_global_point(...)` for
each. `inject_global_point` (`samples.rs`) is built for a different job: it
clones the *entire* `SampleArena` first ("Clone-and-commit is intentionally
simple in phase one", a correctness-motivated design for callers that need
to safely attempt an injection that might fail), then projects the point
onto *every* directed edge, not just the one edge the caller asked about,
checking membership on each with an O(len) `Vec::contains` scan. This is
the third occurrence of the same shape as Update 3's and Update 5's
predecessors: an O(n)-or-worse operation, repeated O(n) times, where the
"everything" being redone every time is unrelated to what the immediate
caller needs. `bootstrap_samples` never needed the clone-based rollback
safety (any failure here aborts the whole `initialize` call regardless) or
the all-edges projection (it processes edges one at a time in its own outer
loop, and each edge's own iterations already reach its own target).

**Fix.** `SampleArena::project_point_onto_edge`: projects onto exactly the
requested edge (and, through `project_component`'s existing recursion, that
edge's ancestor chain only) by direct mutation, no clone, no work on any
other edge. `bootstrap_samples` now calls this plus the existing
`CandidateSets::push_unique` instead of `inject_global_point`.
`inject_global_point` itself is untouched and still correct for callers
that do need its atomicity (e.g. global-pivot injection during a sweep).

TDD: `project_point_onto_edge_touches_only_the_requested_edges_ancestor_chain`
(the whole point of the fix -- other edges' candidate sets must stay empty)
and `project_point_onto_edge_materializes_to_the_same_point_as_inject_global_point`
(the cheaper path must not change the result, only the cost), both in
`samples/tests/mod.rs`. Also added `initialize/tests/mod.rs` -- there was no
direct test coverage of `bootstrap_samples` at all before this -- asserting
every edge reaches exactly its requested rank with valid, materializable
pivots.

**A pre-existing, unrelated flaky test found and fixed along the way.**
Running the full suite surfaced `frames::tests::
candidate_frame_hits_the_cache_on_a_repeated_lookup` failing intermittently
under the default parallel test runner (passed 100% of the time under
`--test-threads=1`). Root cause: Update 4's cache-hit/miss counters
(`frames.rs`'s `debug_stats`/`candidate_debug_stats`) were `static
AtomicU64`s -- process-global, so shared and raced on by every test in the
binary that happens to run concurrently and touch the same code path, not
isolated to the one test reading them. Rust's default test harness runs
each `#[test]` fn on its own thread, so the fix is `thread_local!` instead
of `static`: each test's counters are then genuinely private to it. Verified
with 8 consecutive full default-parallel runs, all green (previously
observed to fail within a handful of runs once the suite grew).

**Result**, `treeaci_parity.rs` chi=16/32/64/128, clean back-to-back runs
(`git stash` isolating the fix, both arms on the same host state):

| chi | ratio before | ratio after | improvement |
|---:|---:|---:|---:|
| 16 | 4.51x | 2.49x | 44.8% |
| 32 | 10.17x | 3.80x | 62.6% |
| 64 | 12.40x | 6.21x | 49.9% |
| 128 | 25.36x | 12.77x | 49.6% |

The ratio roughly *halved at every chi* -- the largest single win of the
three fixes in this file, larger than Updates 3 and 4 combined. Sweep counts
shifted by up to +1 at some chi values (bootstrapping candidates in a
different order changes which points a sweep discovers when), but every run
still converges with error under tolerance and the same or better rank than
before, so this is a benign path change, not a correctness regression.

Full verification: `cargo fmt --all -- --check` clean; `cargo test
-p tensor4all-treeaci --release` green (90 lib tests, up from 87 -- 3 new,
plus the flaky one now reliable), including 8 consecutive default-parallel
runs with zero failures; integration and doctests green.

## What remains after Update 5

Not yet re-profiled after this fix. `from_samples`'s initial, non-incremental
build (~chi^2 at chi=128, per Update 5's own measurement) is a plausible next
place to look, but that is a hypothesis carried over from before this fix
landed, not a conclusion re-checked against the current code -- the same
discipline this file has followed throughout says to measure the post-fix
run before deciding whether it is still the largest remaining piece, and
whether it is inherent (there is no "previous" store to extend from at the
very start of a run) or has its own redundant-work shape to find.

## Update 6: candidate-frame contraction through BLAS `mat_mul` for single-incoming-edge nodes

Separate plan (`2026-08-18-treeaci-blas-candidate-contraction`), executed as
four tasks. Root cause targeted here is different in kind from Updates 3-5:
those were all "redo unrelated work O(n) times" shapes fixed by scoping the
work correctly; this one is "do the necessary work through a scalar loop
instead of a BLAS primitive." `frames.rs`'s `accumulate_incoming` /
`contract_prepared_core` path contracted each pivot-search candidate's
incoming frame against a node's prepared core one scalar multiply-add at a
time, once per candidate, even though on a chain every node has exactly one
incoming edge for its outgoing directed edges -- exactly the shape a single
BLAS matrix-multiply can batch across the whole candidate set for that edge
in one call.

**Fix, in four steps.** (1) `TreeAciScalar` widened to require
`tensor4all_tensorbackend::MatrixScalar` as a supertrait -- a breaking but
harmless change, since `f32`/`f64`/`Complex32`/`Complex64` already
independently implement both, so the crate built with zero fallout; proved
non-vacuous with a test that only typechecks because of the new supertrait
bound. (2) Two new private helpers in `frames.rs`:
`single_incoming_core_matrix` gathers a `PreparedCore`'s fixed-physical-value
slice into a plain `outgoing_dim x incoming_dim` column-major `Matrix<T>`;
`contract_prepared_core_batched` runs one `mat_mul` against an
`incoming_dim x n_candidates` matrix of candidate incoming-frame vectors,
returning `outgoing_dim x n_candidates` in one call. (3) A new
`pub(crate)` dispatcher, `InputFrameStore::candidate_frames_for_edge`: falls
back to the existing scalar per-candidate loop whenever an edge has 0 or
\>=2 incoming edges, and otherwise groups candidates by
`local_coordinate` (candidates sharing a physical value are strided, not
contiguous, per `enumerate_candidates`'s mixed-radix encoding in
`local_update.rs`) and issues one batched `mat_mul` per group.
`local_update.rs`'s `candidate_frames` was reduced to a one-line dispatch
over this new method; its own signature and return type are unchanged.
(4) Before wiring the dispatcher in, measured `candidate_cache`'s hit rate
on a representative multi-sweep run using the existing
`candidate_debug_stats` counters to decide, with evidence rather than a
guess, whether the batched path still needed to consult/populate that
cache.

TDD: new tests in `frames/tests/mod.rs` for both Task 3 helpers (matrix
gather correctness against hand-computed values, plus a genuine cross-check
against the scalar `accumulate_incoming` accumulator itself) and for the
Task 4 dispatcher's per-group batching against the previous scalar path
(`candidate_frames_for_edge_falls_back_on_a_leaf_edge_with_zero_incoming_edges`
and `candidate_frames_for_edge_falls_back_on_a_branch_edge_with_two_incoming_edges`,
also in `frames/tests/mod.rs`, exercising the dispatcher's 0- and
\>=2-incoming-edge fallback directly), plus the Task 2 supertrait-implication
test in `scalar.rs`. The dispatcher's single-incoming-edge batched path
itself is cross-checked against the scalar path by
`candidate_frames_batched_path_matches_scalar_path_on_a_chain` in
`local_update/tests/mod.rs`, not in `frames/tests/mod.rs` -- an earlier draft
of this update misattributed that test's location. A first draft of this
paragraph also claimed the fallback branch was covered only transitively
through full-run integration tests, including for the >=2-incoming-edges
(genuine branch-point) case; that was false for >=2 specifically -- every
fixture used elsewhere in the crate at the time was a chain topology, so no
test, direct or transitive, exercised a node with two or more incoming edges
on any of its outgoing directed edges. The two fallback tests above close
that gap with a 4-node star fixture built for the purpose.

**Result**, `treeaci_parity.rs` chi=16/32/64/128, same benchmark invocation
as the baseline capture. The first pass at this measurement ran on a host
with heavy, unrelated background CPU contention (post-restart Spotlight/
ANE/ecosystem-analytics activity, the same class of noise Update 4 already
documented) and produced numbers that were not independently re-verified
before being written here; that draft has been superseded by a clean re-run
after the contention settled, with the controlling session independently
re-executing every verification command itself rather than trusting the
implementer's report of having run them:

| chi | ratio before | ratio after | improvement |
|---:|---:|---:|---:|
| 16 | 2.53x | 2.70x | -6.7% (regressed, within run-to-run noise) |
| 32 | 4.03x | 3.91x | 3.0% |
| 64 | 6.17x | 5.33x | 13.6% |
| 128 | 13.78x | 9.43x | 31.6% |

The improvement grows with chi, as expected: batching helps most when a
node's candidate set is large enough that one BLAS call amortizes better
than the fixed cost of many scalar-loop calls, and per-edge candidate counts
grow with chi. At chi=16, candidate sets are small enough that this
overhead is close to a wash, and the small measured regression there is not
distinguishable from ordinary criterion run-to-run noise (this file's other
updates have shown similar single-digit-percent noise on unrelated re-runs).
The gap is smaller than it was but not closed: the crate's own maturity doc
comment (`lib.rs`) has been updated from "2.5x-13.8x" to "2.7x-9.4x", and
multi-incoming-edge nodes -- genuine tree branch points, not reachable by
this chain benchmark -- remain on the original scalar path, unchanged by
this update and called out explicitly as such in that doc comment.

Full verification, independently re-run by the controlling session (not
merely reported by an implementer) on the quieted host: `cargo fmt --all
-- --check` clean; scoped `cargo clippy -p tensor4all-treeaci
-p tensor4all-aci --all-targets -- -D warnings` clean; `cargo test --release
-p tensor4all-treeaci --no-fail-fast` green (94 lib tests, 7 integration,
1 rank-scaling, 18 doctests, all passing); `cargo doc -p tensor4all-treeaci
-p tensor4all-aci --no-deps` clean (workspace-wide `cargo doc` still fails on
the unrelated, pre-existing `tensor4all-hdf5`/`hdf5-metno-sys` build script,
which probes only a fixed list of Homebrew HDF5 formula names --
`hdf5@2.1`/`hdf5@2.0`/`hdf5@1.14`/down to `hdf5@1.8`/`hdf5-mpi` -- that does
not include the plain `hdf5` formula (2.2.0) actually installed on this
host; unrelated to this change, not investigated further here).

## What remains after Update 6

Multi-incoming-edge (genuine tree branch point) nodes still use the scalar
`contract_prepared_core` path; batching a combinatorial candidate space
across multiple incoming edges is a materially different and materially
riskier problem than the single-incoming-edge case handled here, and the
chain benchmark used throughout this file cannot validate it -- any future
work there needs its own benchmark with actual branch points, not an
opportunistic extension of this update. `from_samples`'s initial,
non-incremental build noted as the likely next hotspot after Update 5
remains unprofiled against the current code.

## Update 7: sample materialization (`from_samples`/`extend`) through BLAS `mat_mul`, same shape as Update 6

Separate plan (`2026-08-18-treeaci-blas-sample-materialization`), executed as
three tasks. This closes the item Update 6 left explicitly unprofiled
("`from_samples`'s initial, non-incremental build ... remains unprofiled
against the current code").

**Root cause, traced into the call site rather than inferred.** A phase-timer
profiling pass (same technique as Updates 3-4) measured
`InputFrameStore::from_samples`/`extend` at roughly 70% of wall time at
chi=128. `InputFrameStore::build_or_extend` (`frames.rs`) is the shared
implementation behind both entry points, and its inner loop --
`for sample in known..builder.memo[edge].len() { builder.compute(edge,
sample)?; }` -- called `FrameBuilder::compute` once per sample.
`compute`'s single-incoming-edge case bottoms out in the same
`contract_prepared_core`/`accumulate_incoming` scalar per-candidate loop that
Update 6 had already replaced with a BLAS `mat_mul` at a *different* call
site (`candidate_frames_for_edge`, used by pivot-search candidate
contraction). Sample materialization shares the same underlying scalar
primitive but was never routed through Update 6's batching, because it goes
through `compute`, not `candidate_frame` -- a separate call site doing
functionally the same per-candidate/per-sample contraction, requiring its own
batched entry point rather than reuse of Update 6's.

**Fix, in two tasks.** (1) `FrameBuilder::compute_batch(&mut self, edge:
DirectedEdgeId, samples: Range<SampleId>) -> Result<()>` (`frames.rs`): the
single-incoming-edge case groups the requested sample range by
`local_coordinate` -- same rationale as `candidate_frames_for_edge`, samples
sharing a local coordinate are not guaranteed contiguous in the range -- then
gathers each group's incoming frames into one `incoming_dim x
group_size` matrix and issues one `mat_mul` against that local coordinate's
slice of the node's prepared core, one batched `mat_mul` call per group,
writing every group's memoized results from that group's batched result; 0-
or >=2-incoming-edge cuts fall back to the unchanged per-sample scalar
`compute`, mirroring Update 6's `candidate_frames_for_edge` dispatcher
exactly -- the same local-coordinate-grouped, one-`mat_mul`-per-group
strategy, applied to sample materialization instead of candidate
contraction. (2)
`InputFrameStore::build_or_extend`'s inner loop was reduced from the
per-sample `compute` loop above to one `builder.compute_batch(edge,
known..builder.memo[edge].len())?` call -- a three-line diff in `frames.rs`,
no other file touched. `compute_batch` itself was added in a prior commit
with zero production call sites (verified by grep) and wired in only in the
second commit, so the batched path had standalone test coverage before it
carried any real traffic.

TDD: three new tests in `frames/tests/mod.rs`
(`compute_batch_matches_scalar_compute_on_a_chain_edge`,
`compute_batch_falls_back_correctly_on_a_leaf_edge`,
`compute_batch_falls_back_correctly_on_a_branch_edge`), each building two
independent `FrameBuilder`s over the same input/problem/arena and comparing
`compute_batch`'s memoized results against the pre-existing scalar `compute`
loop's, exact (`assert_eq!`, both paths perform the same floating-point
operations in the same order). None of the crate's existing fixtures had a
directed edge with exactly one incoming edge, so a new 3-node chain fixture
(`chain_tree_for_batched_compute`) was added for the batched-branch test.
A task-reviewer fix round then added
`extend_matches_a_full_rebuild_on_a_chain_with_a_batched_edge`, closing a
coverage gap the reviewer identified: the two pre-existing "designated"
extend-correctness tests both use the `y_tree` fixture, whose directed edges
are always 0- or 2-incoming (center degree 3, leaves degree 1), so neither
ever drove `compute_batch`'s batched branch with a nonzero `known` offset --
the exact `extend`-plus-batching interaction this task exists to scrutinize.
The new test grows a chain arena, asserts `known > 0` and `grown_count >
known` (so the setup is non-vacuous), then compares a genuine `extend` call
against a from-scratch rebuild sample-by-sample.

**Result**, `treeaci_parity.rs` chi=16/32/64/128, same benchmark invocation
as the Update 6 baseline capture. The host was under sustained,
unrelated background CPU contention throughout this measurement --
`ecosystemd`/`EcosystemAnalytics`/`trustd` daemons active shortly after a
restart, load average oscillating 4.4-7.7 against 8 cores and not settling
after several minutes of waiting, the same class of noise Update 4 and
Update 6 both already documented. Rather than discard the run outright, two
independent full benchmark invocations were run back to back and their
per-chi tree/train ratios compared: they agreed to within 2.1% of each
other at every chi (chi=16, the loosest agreement; tighter at chi=64/128,
within 0.8% or better -- e.g. chi=128: 6.01x and 5.97x), even though each
run's own criterion change-detection against its stored baseline showed
inconsistent, noise-driven regressed/improved swings on individual arms.
Reported numbers
below are the two runs' average:

| chi | ratio before (Update 6) | ratio after | improvement |
|---:|---:|---:|---:|
| 16 | 2.70x | 2.85x | -5.4% (regressed, within run-to-run noise) |
| 32 | 3.91x | 3.74x | 4.4% |
| 64 | 5.33x | 4.24x | 20.4% |
| 128 | 9.43x | 5.99x | 36.5% |

The improvement grows with chi, the same pattern Update 6 found and larger in
magnitude at every chi from 32 up (Update 6: 3.0%/13.6%/31.6% at
chi=32/64/128; this update: 4.4%/20.4%/36.5%) -- consistent with the
profiling finding that sample materialization was a larger fraction of wall
time than candidate-frame contraction was. At chi=16 the regression is not
distinguishable from ordinary run-to-run noise, the same conclusion Update 6
reached at chi=16 for the same reason (candidate/sample counts too small for
one BLAS call to amortize better than the scalar loop's fixed overhead).

The ratio does not approach 1x even though the dominant cost is now
BLAS-backed on both arms of the comparison, and the benchmark's own printed
diagnostics point at part of why: at every chi, the tree solver takes
noticeably more sweeps to converge than chain `tensor4all-aci` on this
benchmark (chi=128: 2 sweeps for train, 5 for tree; chi=64: 3 vs 6; chi=32: 2
vs 4; chi=16: 3 vs 5) -- roughly 2x-2.5x, unrelated to and unmoved by this
change, since neither task touched scheduling or convergence logic. A sweep
count multiplier of that size alone puts a floor under the ratio well above
1x regardless of how fast per-sweep contraction becomes. This is offered as
a directly observed, not inferred, partial explanation rather than a full
accounting of the remaining gap; the ~20% "other" phase (`initialize` prep,
`commit_edge_proposal` bookkeeping) flagged as unresolved after Update 5
remains unprofiled against the current code, and multi-incoming-edge nodes
remain on the original scalar path for both candidate contraction and sample
materialization, as before. The crate's own maturity doc comment (`lib.rs`)
has been updated from "2.7x-9.4x" to "2.8x-6.0x".

Full verification: `cargo fmt --all -- --check` clean; scoped `cargo clippy
-p tensor4all-treeaci -p tensor4all-aci --all-targets -- -D warnings` clean;
`cargo test --release -p tensor4all-treeaci --no-fail-fast` green (100 lib
tests, 7 integration, 1 rank-scaling, 18 doctests, 126 total, all passing);
`cargo doc -p tensor4all-treeaci -p tensor4all-aci --no-deps` clean
(workspace-wide `cargo doc`/`cargo clippy` still fail on the same unrelated,
pre-existing `tensor4all-hdf5` Homebrew-formula-name mismatch documented in
Update 6, not investigated further here).

## What remains after Update 7

Multi-incoming-edge (genuine tree branch point) nodes still use the scalar
path for both candidate contraction and sample materialization -- unchanged
by this update, same scope boundary as Update 6. The ~20% "other" phase
(`initialize` prep, `commit_edge_proposal` bookkeeping) identified after
Update 5 remains unprofiled. The sweep-count gap between tree and chain
convergence on this benchmark (roughly 2x-2.5x, noted above) is a newly
observed candidate explanation for part of the remaining ratio, but is not
itself investigated here -- if it turns out to dominate what remains, that is
a new root-cause pass on scheduling/convergence behavior, not an
opportunistic extension of this update's sample-materialization scope.

## Update 8: final-review fix round for Update 7 -- `compute_batch` was redundantly re-materializing samples an earlier edge had already primed

Final-review fix round on the same plan Update 7 shipped
(`2026-08-18-treeaci-blas-sample-materialization`). Root cause: Update 7's
`InputFrameStore::build_or_extend` calls `FrameBuilder::compute_batch` once
per directed edge, in the edge's index order (`0..problem.directed_edges.len()`),
which is not topological order. `compute_batch`'s single-incoming-edge branch
primes its group's incoming frames via the ordinary scalar `self.compute`
recursion before batching -- and `compute` walks the ancestor chain, so an
earlier edge's priming recursion can reach and scalar-materialize samples that
belong to a *later* edge before `build_or_extend`'s own loop gets there. When
the loop later calls `compute_batch` for that later edge directly, its old
code re-fetched and re-grouped every sample in the requested range
unconditionally -- including the ones the earlier edge's recursion had just
computed -- and pushed them through a second, wasted `mat_mul`. The result
was still correct (both paths compute the same value), but did real duplicate
work that Update 7's batching was specifically meant to eliminate.

**Fix** (`crates/tensor4all-treeaci/src/frames.rs`, `compute_batch`'s
single-incoming-edge branch): before grouping a sample for batching, check
`self.memo[edge][sample].is_some()` and skip samples already memoized,
collecting the rest into a `pending: Vec<(SampleId, ComponentSample)>`. Each
pending sample's `ComponentSample` record is now fetched from the arena
exactly once and reused across the priming, local-coordinate-grouping, and
frame-gathering steps, instead of being re-fetched from `self.arena` in each
of those three separate loops as before. This mirrors the existing
`candidate_cache` check `candidate_frames_for_edge` already had at the
equivalent point in its own loop (Update 6) -- `compute_batch` was the one
call site that hadn't gotten the same guard.

**Test** (`crates/tensor4all-treeaci/src/frames/tests/mod.rs`):
`from_samples_issues_exactly_one_compute_call_per_memo_slot_on_a_five_node_chain`,
against a new 5-node chain fixture (`chain5_tree_for_dedup_regression`). Five
nodes is the minimum needed to reproduce the bug: it requires a directed edge
whose own incoming edge is itself single-incoming (not a leaf), which first
appears at 5 nodes -- the pre-existing 3-node
`chain_tree_for_batched_compute` fixture is too short. The test asserts
`debug_stats::compute_calls()` (which increments once per cache-miss
materialization, whether scalar `compute` or a batched column write) equals
the total number of memo slots filled across every directed edge -- an exact
equality, not a loose bound, so any redundant recomputation fails it. A
second assertion (`total_memo_slots > seeds.len() * directed_edges.len() /
2`) guards against the comparison being vacuously satisfied by a fixture
where every edge only ever has one sample. Before the fix this test failed
(`calls` exceeded `total_memo_slots`); after the fix it passes.

**Verification.** `cargo fmt --all -- --check` clean. Scoped `cargo clippy -p
tensor4all-treeaci -p tensor4all-aci --all-targets -- -D warnings` clean.
`cargo test -p tensor4all-treeaci --release --no-fail-fast`: 101 lib tests (up
from Update 7's 100 -- the one new regression test), 7 integration, 1
rank-scaling, 18 doctests, all passing.

**Benchmark.** Same `treeaci_parity.rs` chi=16/32/64/128 invocation as Update
7's baseline capture. The host was again under sustained, unrelated
background CPU contention throughout (`ecosystemd`/`EcosystemAnalytics`/
`trustd` daemons, load average 5.9-8.0, idle consistently 33-45% and not
settling), the same recurring class of noise Updates 4, 6, and 7 all
document. This round's run-to-run agreement was worse than Update 7's (which
reported 0.8-2.1% agreement between two runs): four independent full
benchmark invocations were run back to back, and their per-chi tree/train
ratios varied by as much as ~20% at chi=16 and ~10% at chi=64 run to run,
comparable in size to the improvement itself. Rather than pick a favorable
pair, all four runs are reported, with the mean used as the headline number:

| chi | Update 7 baseline | run 1 | run 2 | run 3 | run 4 | mean (after) | change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2.85x | 2.40x | 2.87x | 2.67x | 2.99x | 2.73x | -4.2% |
| 32 | 3.74x | 3.52x | 3.72x | 3.59x | 3.74x | 3.64x | -2.7% |
| 64 | 4.24x | 4.04x | 4.35x | 4.00x | 4.47x | 4.22x | -0.5% |
| 128 | 5.99x | 5.71x | 5.94x | 5.61x | 5.86x | 5.78x | -3.5% |

The mean sits at or below the Update 7 baseline at every chi, consistent with
the fix's mechanism (it only ever removes work, never adds any), but the
per-run spread is large enough that no single run, and arguably not even the
4-run mean, cleanly separates from host noise the way Update 6's and Update
7's own 2-run comparisons did. Two things are worth naming plainly: (1) the
direction is right and consistent with the fix -- of the 16 (chi, run) cells,
12 sit at or below the corresponding baseline value; (2) the *size* of the
improvement this round is smaller than Update 7's, which is expected --
Update 7 eliminated a full second scalar-vs-BLAS pass for every
single-incoming-edge sample; this fix only eliminates the subset of samples
that happened to get double-materialized because of index-vs-topological
edge ordering, which on this benchmark's particular chain topology and
traversal schedule is a minority of the total sample count. The crate's
maturity doc comment (`lib.rs`) has been updated from Update 7's "2.8x-6.0x"
(and an interrupted pass's unmeasured placeholder of "2.8x-5.9x") to
"2.7x-5.8x" (the 4-run mean, rounded).

As with Update 7, `InputFrameStore::build_or_extend`'s edge-processing loop
was deliberately left in index order rather than reordered to topological
order -- that remains a separate, riskier future change, out of scope for
this fix. The memo-check skip is the complete, intended fix for this round.

## Update 9: the sweep-count gap flagged after Update 7 was the convergence criterion, not a real algorithmic difference

Update 7's "What remains after Update 7" section named the tree/chain
sweep-count gap (roughly 2x-2.5x on `treeaci_parity`) as "a newly observed
candidate explanation for part of the remaining ratio" and left it
uninvestigated. This update is that follow-up pass: a per-sweep rank/error
trajectory comparison at chi=32 (informal instrumentation, not committed)
showed both `tensor4all-aci` (chain) and `tensor4all-treeaci` (tree) reaching
their near-final rank and error by sweep 0-1 -- the two algorithms were doing
essentially the same convergence work per sweep, at the same pace. The gap
was entirely in how many *additional* sweeps each side's stopping rule
demanded before declaring convergence, not in how many sweeps were needed to
reach the answer.

**Root cause** (`crates/tensor4all-treeaci/src/schedule.rs`,
`convergence_criterion`): the tree crate's dwell-window check required every
individual directed edge's rank to be simultaneously non-increasing across
the window (`ranks: &[Vec<usize>]`, per-edge comparison). `tensor4all-aci`'s
`convergence_criterion_like_julia` (`crates/tensor4all-aci/src/elementwise.rs`)
only requires the network-wide **scalar max rank** to be non-increasing;
individual bonds are free to fluctuate underneath that max. On a tree, with
more edges than a chain has bonds, the probability that at least one edge
ticks up by one while the network max is already flat and stable is higher
than on a chain -- and every such tick reset tree's dwell window, forcing it
to re-accumulate `min_sweeps` more clean passes before it could converge,
even though the quantity chain's criterion actually cares about (the network
max) had already stopped moving. This is exactly the mechanism informally
observed in the chi=32 trajectory: individual edges fluctuated by one for a
sweep or two while the max rank held flat, and each fluctuation cost tree
roughly one more sweep it did not algorithmically need.

**Fix** (Task 2 of this plan, commit `8725c5b0`,
`crates/tensor4all-treeaci/src/schedule.rs`): changed `convergence_criterion`
to read the already-computed `max_ranks: Vec<usize>` (pushed from
`report.max_rank` every pass -- this was already being computed and returned
in `SweepHistory`, just not used for the stability check) instead of the
per-edge `rank_vectors: Vec<Vec<usize>>`, which was then dead and removed.
The window check is now structurally identical to `tensor4all-aci`'s
reference criterion: non-increasing scalar max across the dwell window, plus
the unchanged `error <= tolerance` gate and the unchanged "zero global pivots
found in the window" clause. Neither of those two other gates was touched.

**TDD.** The old test
`convergence_requires_full_rank_vector_stability_and_minimum_dwell` was
replaced with `convergence_requires_stable_max_rank_and_minimum_dwell`
(`crates/tensor4all-treeaci/src/schedule/tests/mod.rs`), which first asserts
the new signature fails to compile against the old `&[Vec<usize>]` parameter
(confirmed: 5 call-site type errors), then after the fix asserts: (1) below
minimum dwell never converges; (2) two sweeps with a stable scalar max rank,
error at tolerance, and no global pivots in the window converges even though
this is exactly the per-edge-fluctuation-underneath-a-stable-max scenario the
old code used to reject; (3) a max rank that actually increases within the
window still blocks convergence (unchanged); (4) error above tolerance still
blocks (unchanged); (5) a global pivot found in the window still blocks
(unchanged). No other test in the crate needed an updated expected rank,
sweep count, or error value -- the full 127-test suite (101 unit + 7
`public_api` + 1 `rank_scaling` + 18 doctests) passed unchanged both before
and after, because no pre-existing test scenario happened to exercise an edge
fluctuating underneath an already-stable network max.

**Result.** Re-ran `cargo bench -p tensor4all-aci --bench treeaci_parity`
twice back to back (host load average 5.5, CPU 28% idle -- moderately loaded
but the two runs agreed closely, see table). Sweep counts are deterministic
on this fixed-seed benchmark and were identical between the two runs:

| chi | chain sweeps | tree sweeps (before) | tree sweeps (after) | sweep ratio (before) | sweep ratio (after) |
|---:|---:|---:|---:|---:|---:|
| 16 | 3 | 5 | 4 | 1.67x | 1.33x |
| 32 | 2 | 4 | 2 | 2.00x | 1.00x |
| 64 | 3 | 6 | 2 | 2.00x | 0.67x |
| 128 | 2 | 5 | 2 | 2.50x | 1.00x |

Sweep counts converged from a 1.67x-2.50x tree/chain gap down to 0.67x-1.33x
-- at chi=32 and chi=128 tree now matches chain's sweep count exactly, and at
chi=64 tree actually converges in *fewer* sweeps than chain. Wall-time
followed the same direction, using the mean of the two back-to-back runs'
center estimates:

| chi | chain time (ms) | tree time (ms) | wall ratio (before, Task 1) | wall ratio (after) |
|---:|---:|---:|---:|---:|
| 16 | 22.36 | 52.42 | 2.84x | 2.35x |
| 32 | 16.65 | 37.74 | 3.62x | 2.27x |
| 64 | 28.13 | 59.48 | 4.37x | 2.11x |
| 128 | 33.94 | 122.19 | 5.82x | 3.60x |

The wall-time ratio dropped at every chi (from 2.84x-5.82x to 2.11x-3.60x),
and no longer grows monotonically with chi the way the pre-fix numbers did --
chi=64's ratio (2.11x) is now the lowest of the four, not the second highest.
The remaining 2.1x-3.6x gap is consistent with Update 6/7's still-unfinished
per-sweep BLAS-batching scope (multi-incoming-edge nodes and the
scalar-primed subset of single-incoming-edge nodes still use the
per-candidate/per-sample scalar path) rather than with sweep count, which
this update brought to near parity. The crate's maturity doc comment
(`crates/tensor4all-treeaci/src/lib.rs`) has been updated to report both the
new sweep-count ratio (0.7x-1.3x, down from 1.7x-2.5x) and the new wall-time
ratio (2.1x-3.6x, down from 2.8x-5.8x), and the "still growing with bond
dimension" characterization of the wall-time gap has been dropped since it is
no longer accurate (the new ratio dips at chi=64 before rising again at
chi=128).

**Verification.** `cargo fmt --all -- --check` clean. Scoped `cargo clippy -p
tensor4all-treeaci -p tensor4all-aci --all-targets -- -D warnings` clean.
`cargo test --release -p tensor4all-treeaci --no-fail-fast`: 101 lib tests,
7 integration, 1 rank-scaling, 18 doctests, all passing. `cargo doc -p
tensor4all-treeaci -p tensor4all-aci --no-deps` builds clean.

This closes the sweep-count-gap question Update 7 raised and left open: it
was the stopping rule, not a real per-sweep algorithmic difference between
tree and chain traversal.

## Update 10: `InputFrameStore::build_or_extend` was rebuilding every directed edge's frame storage on every call -- unchanged edges now shared via `Rc`

Update 9 closed the sweep-count gap and pointed the remaining 2.1x-3.6x
wall-time ratio at per-sweep cost. This update is a phase-breakdown of that
per-sweep cost at the algebraic rank ceiling (chi=256 on `treeaci_parity`'s
16-site chain, where frame storage is largest: `3,602,144` bytes), followed
by the fix it identified and this task's fresh measurement of the result.

**The mechanism: `seed` and `reconstruct`, pure data movement.** Before this
plan, `InputFrameStore::build_or_extend`'s per-input, per-directed-edge loop
did three things unconditionally, on *every* call, for *every* edge,
regardless of whether that edge's sample set had changed since the previous
call:

1. allocate `memo[edge]` and **eagerly seed** it by copying every
   already-known sample's row out of the previous store (`previous.row(sample)`,
   an `O(count * bond_dim)` copy plus one `Vec` allocation per sample);
2. call `compute_batch(edge, known..count)` -- a no-op range when the edge's
   sample count had not grown, but still a call;
3. **reconstruct** a brand-new `Matrix` for the edge by copying every row back
   out of `memo`, a second `O(count * bond_dim)` copy.

For an edge whose sample count had not changed, steps 1 and 3 together
produced an exact bitwise duplicate of a frame the previous store already
held, at the cost of two full copies of it -- no arithmetic, purely data
movement. Informal instrumentation at chi=256 (this session, not committed as
its own artifact) attributed **36.6% of total ACI wall time** to this
seed/reconstruct pair. Two adjacent phases were also measured and are
explicitly out of this plan's scope (see "What this plan deliberately does
not do" in `docs/superpowers/plans/2026-08-19-treeaci-shared-unchanged-frames.md`):
first-materialization via scalar priming (~20%, edge-index-vs-topological-order
gap) and `TreeAciState::initialize`'s non-`build_or_extend` bootstrap/
canonicalization cost (~20%). This update's fix addresses only the 36.6% share.

**Baseline (Task 1).** Re-ran `treeaci_parity` with `CHI_VALUES` extended to
`[16, 32, 64, 128, 200, 256]` before any fix landed: wall-time ratio (tree /
train) at chi=256 was **5.003x** (mean of two clean back-to-back runs; a
third run's chi=256 tree time was a host-noise outlier at 302 ms against 259
and 260 ms for the other two, discarded). This became the fix's target.

**The fix (Tasks 2-5 of this plan, commits `d23ceeec`..`cf089f03`).** Four
increments, each independently tested and committed:

- **Task 2** (`d23ceeec`): `InputFrameStore.cores` changed from
  `Vec<Vec<PreparedCore<T>>>` to `Vec<Rc<Vec<PreparedCore<T>>>>`. An unchanged
  input's prepared cores are now shared by reference across `extend` calls
  instead of cloned by value. New test
  `extend_reuses_the_same_cores_allocation_instead_of_cloning_it` asserts
  `Rc::ptr_eq`, not just value equality, since a regression back to `.clone()`
  would still pass every pre-existing value-equality test.
- **Task 3** (`500de8a7`): `InputFrameStore.frames` changed from
  `Vec<Vec<DirectedFrame<T>>>` to `Vec<Vec<Rc<DirectedFrame<T>>>>`. Type-only
  change -- `build_or_extend` still rebuilt every edge on every call at this
  point; this just made the storage shareable in preparation for Task 5.
- **Task 4** (`36bbaa36`): added `FrameBuilder::existing_frames: Option<&'a
  [Rc<DirectedFrame<T>>]>` and a lazy-pull branch in `FrameBuilder::compute`,
  inserted between the memo-hit check and the genuine `contract_prepared_core`
  computation -- a sample already known to the previous store is pulled via
  `DirectedFrame::row` (one `O(bond_dim)` copy) and memoized, instead of
  recomputed, and does not increment the `debug_stats` compute-call counter.
  This added the capability without activating it: `build_or_extend`'s
  construction site still passed `existing_frames: None` and still eagerly
  seeded `memo` exactly as before.
- **Task 5** (`cf089f03`): wired it up and removed the eager seed loop.
  `build_or_extend` now decides per edge: if the previous store had a frame
  for that edge *and* its sample count is unchanged, the edge shares the
  previous store's `Rc<DirectedFrame<T>>` directly -- no `compute_batch` call,
  no memo fill, no fresh `Matrix`, just an `Rc::clone`. Only edges that grew or
  are new get rebuilt, using the Task 4 lazy-pull path for any old sample an
  ancestor-priming recursion still needs to read.

  Two subtleties surfaced during Task 5's implementation and review, both
  documented in `task-5-report.md`:
  - **Edge ordering.** A naive two-pass structure ("push reused edges, then
    push rebuilt edges") would silently reorder `input_frames` relative to
    `edge_count`, mis-associating frames with edges in every mixed case (the
    normal case for `extend`) -- a value-corruption bug, not a crash. The fix
    pre-sizes `input_frames: Vec<Option<Rc<DirectedFrame<T>>>>` to
    `edge_count` and has both passes write by index (`input_frames[edge] =
    Some(...)`), never push, so no ordering assumption exists to violate.
  - **The memo spine must stay allocated for every edge, including reused
    ones.** A grown edge's `compute_batch` priming recursion walks its
    ancestor chain and calls `self.compute(incoming_edge, incoming_sample)`
    regardless of whether `incoming_edge` is itself being reused; the
    lazy-pull branch (Task 4) writes its pulled row back into
    `self.memo[incoming_edge][incoming_sample]`, and `compute_batch`'s
    batched branch reads that slot by direct indexing. An empty spine on a
    reused edge would therefore return a graceful `Result::Err`
    (`TreeAciError::InternalInvariant`) from `compute`'s write, or panic via
    an out-of-bounds index from `compute_batch`'s direct-indexed reads, the
    first time a grown edge's ancestor chain passed through it. So `memo` is
    still allocated at full size for every edge from the pre-computed
    `sample_counts` -- reused edges' spines simply stay `None`-filled unless
    something reads through them.
    The cost is one `Option<Vec<T>>` (24 bytes) per sample of spine, against
    the `bond_dim * size_of::<T>()` row it would have held (2 KiB at
    chi=256, `f64`) -- roughly 1% of the movement the fix removes, not
    worth chasing further.

**This task's measurement (Task 6).** Re-extended `CHI_VALUES` to `[16, 32,
64, 128, 200, 256]` (same edit as Task 1, reverted afterward), rebuilt in
`--release`, and ran `cargo bench -p tensor4all-aci --bench treeaci_parity`
three times back to back. Host load was moderate throughout (load average
4.2-8.9, 27-31% user CPU at the start, rising toward the end of the third
run), consistent with Task 1's conditions. Cross-run agreement was good at
every chi except two individual points flagged by Criterion itself as noisy
(run 1's chi=64 tree time and chi=128 train time; run 3's chi=256 train
time -- each shows a wide confidence interval and a "regressed"/outlier flag
against that same benchmark's own history, both symptomatic of a background
CPU spike rather than a real change). Using the two agreeing runs per chi
point:

| chi | train (ms) | tree (ms) | ratio (after) | ratio (before, Task 1) |
|---:|---:|---:|---:|---:|
| 16  | 19.66  | 38.44  | 1.96x | 2.33x |
| 32  | 14.09  | 29.53  | 2.10x | 2.31x |
| 64  | 25.19  | 44.32  | 1.76x | 2.11x |
| 128 | 32.74  | 93.04  | 2.84x | 3.61x |
| 200 | 42.61  | 120.48 | 2.83x | 3.74x |
| 256 | 52.09  | 153.94 | **2.96x** | **5.003x** |

(chi=16/32/64/128 "before" column from Task 1's cross-run average table;
chi=200/256 "before" from the same table. chi=256 "after" uses runs 1-2's
mean, since run 3's train time was the flagged outlier for that point,
mirroring exactly how Task 1 itself excluded its own run-3 chi=256 outlier.)

The wall-time ratio at chi=256 dropped from **5.003x to 2.96x**, a **~41%
reduction** -- substantial, and in the direction and rough magnitude the
36.6%-of-wall-time mechanism predicted (removing a component that was pure
data movement should show up close to linearly in wall time, since it adds
no arithmetic and unblocks no other phase). The ratio also improved at every
other chi point, including the smallest (16 and 64), where frame storage is
far from its cap -- consistent with the fix mattering on every `extend` call,
not only at the largest bond dimension where the previous eager-copy volume
happened to be biggest.

The crate's maturity doc comment (`crates/tensor4all-treeaci/src/lib.rs`) has
been updated to report the new wall-time range (roughly 1.8x-3.0x across chi
16 through 256, chi=256 specifically ~2.9x-3.0x, down from 2.1x-3.6x on the
benchmark's previous default chi 16-128 range and from a measured 5.0x at
chi=256 alone) and to describe the fix: unchanged directed edges' frame
storage is now shared via `Rc` instead of rebuilt, with old samples any
other edge still needs pulled lazily instead of pre-copied.

**Verification.** `cargo fmt --all -- --check` clean. Scoped `cargo clippy -p
tensor4all-treeaci -p tensor4all-aci --all-targets -- -D warnings` clean.
`cargo test --release -p tensor4all-treeaci --no-fail-fast`: 104 lib tests, 7
integration, 1 rank-scaling, 18 doctests, all passing (unchanged from Task
5's count -- this task touched no `.rs` file other than `lib.rs`'s doc
comment). `cargo doc -p tensor4all-treeaci -p tensor4all-aci --no-deps`
builds clean.

This closes the seed/reconstruct question this session's phase breakdown
raised: the 36.6%-of-wall-time mechanism was real, specific, and fixable
without touching `InputFrameStore`'s public API, and removing it closed
roughly two-fifths of the tree/chain wall-time gap at the algebraic rank
ceiling. What remains -- first-materialization via scalar priming (~20%) and
`TreeAciState::initialize`'s bootstrap/canonicalization cost (~20%) -- is
explicitly out of this plan's scope and is each already identified as a
separate follow-up.

## Update 11: topological frame materialization removes first-materialization scalar priming

Update 10's phase breakdown identified a remaining, measured ~20% cost: on a
chain, `InputFrameStore::build_or_extend` visited directed edges by numeric id.
`compute_batch` on an edge with one incoming edge recursively called scalar
`FrameBuilder::compute` for that incoming frame. Because numeric edge order is
not dependency order, that recursion could materialize a later single-incoming
edge through the scalar path before that edge's own batched call ran. This was
not an inherent tree-branching cost; it was a missed batching opportunity in a
known sibling call path.

**RED evidence.** A five-node chain fixture with four seeds recorded 6 scalar
frame contractions, while only 3 samples belonged to non-batchable edges (the
two leaf directions in this fixture). The new regression
`from_samples_uses_scalar_only_for_non_batch_edges_on_a_chain` failed with
`left: 6, right: 3` before the production change. A separate star-topology
invariant test checks that every `incoming_to_from` dependency precedes its
dependent in the computed order.

**The fix.** `build_or_extend` now keeps the original numeric edge order for
resource accounting and reuse decisions, then materializes missing edges in a
Kahn topological order of the directed-frame dependency graph. Consequently,
single-incoming ancestors are fully materialized through their grouped BLAS
path before dependent edges read them. Multi-incoming edges retain the scalar
fallback. The test-only counters now distinguish scalar contractions from
batched columns, while the existing aggregate counter remains unchanged.

**Correctness verification.** The targeted regression passed after the fix,
and the full `tensor4all-treeaci` suite passed: 106 lib tests, 7 integration
tests, 1 rank-scaling test, and 18 doctests. The existing chain, star, branch,
extend, cache, and exact-value tests all remained green.

**Fresh efficiency measurement.** Using the committed `treeaci_parity` setup
(16 sites, local dimension 2, two deterministic TT inputs, chain topology,
unseeded, global guard disabled), one fresh Criterion run gave these midpoint
times; the temporary 200/256 entries were reverted after measurement:

| chi | train (ms) | tree (ms) | tree/train |
|---:|---:|---:|---:|
| 16 | 22.345 | 46.169 | 2.07x |
| 32 | 15.708 | 33.103 | 2.11x |
| 64 | 28.656 | 43.545 | 1.52x |
| 128 | 35.139 | 68.993 | 1.96x |
| 200 | 45.156 | 83.601 | 1.85x |
| 256 | 53.402 | 108.54 | 2.03x |

The run did not show the previous monotonic ratio growth through chi=256; the
remaining roughly 2x gap is now a separate per-sweep overhead question, not
evidence that this scalar-priming bug is still present. Absolute values remain
host-sensitive, and the benchmark still compares chain versus chain rather
than a genuinely branched tree.

## Update 12: chain raw-message BLAS dispatch is guarded by batch shape

The remaining downstream-shaped guard cost led to a focused optimization of
`tensor4all-treetn`'s existing raw interior-chain message path. The general
tree evaluator and all non-chain eligibility fallbacks remain unchanged. The
chain path now groups missing message columns by physical value and uses the
tensorbackend matrix multiply only when the grouped contraction is large
enough; real and complex paths share the same checked contraction logic.

The first implementation used only a total scalar-work threshold. A direct
release measurement exposed a second performance bug in that dispatch rule:
the global guard commonly supplies one or two points per floating-zone callback,
so the rule created `d x d` matrix-vector backend calls whose setup cost was
higher than the scalar loop. At bond dimension 128, the measured scalar versus
grouped times were `0.47 ms` versus `0.95 ms` for one point, and `0.87 ms`
versus `1.77 ms` for two points. The crossover became consistently favorable
when each physical-value group had at least four columns (for example, `4.59`
versus `2.47 ms` at bond 128 with eight points).

The final dispatch therefore keeps the scalar path for batches below eight
points or whenever any physical group has fewer than four points, in addition
to the existing work and group-count checks. This is an optimization guard,
not a deprecation: large chain batches still use BLAS, while branch and
ineligible tensor layouts still use the original generic contraction.

Correctness coverage includes real and complex large-group contractions, the
existing raw chain oracle comparisons, and the star-tree fallback comparison.
The focused BLAS-path tests and the full release suites pass: 439
`tensor4all-treetn` unit tests and 111 `tensor4all-treeaci` unit tests. The
fixed-center output guard regression also passes, confirming that the cache
optimization does not reintroduce per-scan center changes.

An upstream-shaped diagnostic using 15-site and 21-site chains (the R=5/R=7
site counts), one normalized input, one sweep, and the guard's default
five-start/100-step search showed guard-on versus guard-off times of
`0.937 s` versus `0.011 s` and `2.177 s` versus `0.015 s`, respectively; the
held-out point errors were `3.39e-12` and `2.38e-12`. These are diagnostic
measurements, not the downstream SGW benchmark, and the low-rank fixture does
not establish the final R=7 production ratio. The direct cached-evaluator
workload at bond 128 remains correct and showed cached times around
`0.98–1.12 s` versus `1.92–1.96 s` uncached across repeated runs; the
small-batch dispatch change is intentionally reported as no-regression rather
than as a precise end-to-end speedup because host noise overlaps its modest
effect.

## Update 13: lightweight cache scalars remove the dominant chain guard cost

The previous update's raw chain kernel was correct, but it still stored every
new message element in `PackedMessageCache<IndexKey, AnyScalar>`. Although the
raw real/complex kernels returned primitive values, the cache insertion path
then converted each element to `AnyScalar`; `AnyScalar::from_value` constructs
a rank-zero `IdxTensor` for each scalar. This was a second independent
performance bug: the numerical representation was unnecessarily expensive even
after the contraction itself had been optimized.

The cache now stores a private `CachedScalar` enum containing either `f64` or
`Complex64`. Raw real and complex paths populate it directly, and conversion
back to `AnyScalar` is limited to the mixed-type generic fallback. Message
reconstruction still returns the same `IdxTensor` and public `AnyScalar`
values, so this is an internal storage optimization rather than a change to
the evaluator's type or numerical semantics. The existing generic tree path,
branch fallback, fixed-center policy, and cache budget behavior are unchanged.

On the same 16-site, local-dimension-2, bond-128 floating-zone walk
(`NSEARCH=5`, `MAX_SWEEPS=100`), the call pattern was identical in all runs:
165 calls, 3150 cache hits, 525 misses, and hit rate 0.857. Three sequential
release runs measured:

| run | cached | uncached | speedup |
|---:|---:|---:|---:|
| 1 | 151.45 ms | 2.029 s | 13.40x |
| 2 | 152.88 ms | 2.048 s | 13.40x |
| 3 | 154.38 ms | 1.982 s | 12.84x |

The cached path is therefore consistently about 0.15 s and the uncached path
about 2.0 s on this host, with a measured speedup range of 12.84–13.40x.
This is the direct cached-evaluator workload, not the downstream `gw-rs`
benchmark; no downstream source was changed. The measurement is not used as a
wall-clock pass/fail assertion.

**Verification after the fix.** Rust `1.97.1` (Homebrew) was used. Focused
real/complex raw-chain, star-tree, and fixed-center tests passed. The complete
release unit suites passed: 440 `tensor4all-treetn` tests and 111
`tensor4all-treeaci` tests. Formatting was clean, and the temporary raw-kernel
timers/counters used during diagnosis were removed before this measurement.

## Update 14: avoid materializing cached chain messages for every guard batch

The downstream-shaped complex chain probe still showed a second independent
cost after the raw contraction and `CachedScalar` fixes. For every small
floating-zone batch, `TreeTNCachedEvaluator` rebuilt an `IdxTensor` for every
directed message from already-packed cache columns, even though the next
chain contraction could consume those columns directly. A phase probe with 30
sites, local dimension 2, bond dimension 34, and 300 two-point calls measured
`reconstruct=151.3 ms` out of the message-building work. The leaf-centre
contraction also still used a backend tensor contraction for a mathematically
simple bond-vector dot product (`66.1 ms` before the raw centre path).

The evaluator now memoizes the rooted message plan and cache layouts for the
fixed centre, keeps eligible chain messages as private `CachedScalar` columns,
and consumes those columns directly in the parent raw contraction and leaf
centre contraction. The generic `IdxTensor` materialization remains in place
for branches, interior centres, mixed scalar kinds, and non-standard layouts.
The raw mode is therefore an optimization path with explicit eligibility
checks, not a change to the general tree evaluator.

The same probe remained numerically identical (`2` sweeps, rank `3`, final
error `7.056e-5`): TreeACI guard-on time moved from `1.557 s` after the earlier
raw-centre fix to `0.443 s`; guard-off was `38.6 ms`. In the isolated phase
probe, message reconstruction fell from `151.3 ms` to `1.4 ms`, environment
build from `192.0 ms` to `28.8 ms`, and centre contraction remained about
`12.3 ms`. The direct input/output cached replays were `48.5 ms` and `38.0 ms`
for 300 calls. These are upstream diagnostic measurements, not the downstream
SGW benchmark; the downstream checkout was not modified.

Correctness after the raw-column path passed the cached-evaluator suite
(39 focused tests), the complete TreeACI release suite (111 unit tests, 7
integration tests, 1 rank-scaling test, and 18 doctests), formatting, and
scoped clippy. The final full-crate release verification remains the gate
before commit.
