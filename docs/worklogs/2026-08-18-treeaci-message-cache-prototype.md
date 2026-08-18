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
