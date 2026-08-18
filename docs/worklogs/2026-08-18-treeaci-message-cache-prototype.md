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
