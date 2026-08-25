# Parallel adaptive TCI patch execution

## Status and scope

This document specifies parallel execution for
`tensor4all-partitionedtt::adaptiveinterpolate`. It does not change the TCI2
algorithm or the mathematical patch partition. Hataori schedules independent
patches; TCI2 continues to process one patch at a time.

`tensor4all-partitionedtt` is otherwise behavior-frozen during the
`tensor4all-partitionedtreetn` migration. Adaptive interpolation is explicitly
excluded from that successor's migration scope, so this user-approved change
stays with the existing adaptive-interpolation owner rather than introducing a
second implementation in the TreeTN crate.

The required behavior is:

- Rayon is the default rank-local scheduler when Hataori support is enabled;
- Hataori and MPI remain optional dependencies/features;
- Rayon work inside a patch may be nested under Rayon patch-level work;
- a rejected parent's sample cache is transferred to its children;
- each child receives only samples belonging to that child; and
- accepted-patch caches are returned with the interpolation result.

## Cargo features

Use one optional dependency and two additive crate features:

```toml
[features]
default = ["tenferro-cpu-faer", "adaptive-hataori-rayon"]
adaptive-hataori-rayon = ["dep:hataori", "hataori/rayon"]
adaptive-hataori-mpi = [
    "adaptive-hataori-rayon",
    "hataori/mpi",
    "dep:mpi",
    "dep:serde",
    "num-complex/serde",
]

[dependencies]
hataori = { git = "https://github.com/shinaoka/hataori-rs.git", rev = "59d0ccd6b403e285b7e716a8ae6d12d851da13b1", default-features = false, optional = true }
mpi = { version = "=0.8.1", default-features = false, optional = true }
serde = { version = "1", features = ["derive"], optional = true }
```

Thus `--no-default-features` does not build Hataori, while an ordinary build has
rank-local Rayon execution available. MPI is never enabled by default.
`adaptive-hataori-mpi` must use the same upstream `mpi` package/version as
Hataori when naming the communicator trait.

An `rsmpi-rt` forwarding feature is deliberately deferred. It can be added as a
mutually exclusive sibling of `adaptive-hataori-mpi` when a runtime-loaded MPI
consumer is ready to test it.

## Public entry points

Keep the existing sequential entry-point name for minimal and
no-default-feature builds, but change its return type to
`AdaptiveInterpolationResult<T>` so the required caches are returned uniformly.
It uses the same wave processor and per-path seed derivation as parallel modes,
with a sequential scheduler. Add feature-gated entry points rather than putting
feature-dependent bounds on the existing function:

```rust
#[cfg(feature = "adaptive-hataori-rayon")]
pub fn adaptiveinterpolate_in<T, F, B>(
    domain: &hataori::Domain,
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<AdaptiveInterpolationResult<T>>
where
    T: /* existing scalar bounds */ + Send + Sync,
    F: Fn(&MultiIndex) -> T + Send + Sync,
    B: Fn(&[MultiIndex]) -> Vec<T> + Send + Sync;
```

The Hataori domain is explicit. Tensor4all must not create a hidden Rayon pool,
guess CPU affinity, or establish a second threading policy. Rank-local patch
execution always uses `hataori::LocalMode::Outer`. The domain must be backed by
a Rayon pool. Call `adaptiveinterpolate_in` from outside a Rayon worker or from
within that same domain's pool; Hataori rejects a sequential domain with
`MissingPool` and a foreign pool worker with `ForeignPool`.

The MPI entry point is collective and returns the result only on `root`, matching
`hataori::pmap`:

```rust
#[cfg(feature = "adaptive-hataori-mpi")]
pub fn adaptiveinterpolate_mpi<C, T, F, B>(
    world: &C,
    domain: &hataori::Domain,
    root: i32,
    f: F,
    batched_f: Option<B>,
    site_indices: Vec<DynIndex>,
    initial_pivots: Vec<MultiIndex>,
    options: AdaptiveInterpolateOptions,
) -> Result<Option<AdaptiveInterpolationResult<T>>>
where
    C: mpi::traits::Communicator,
    T: /* local bounds */ + serde::Serialize + serde::de::DeserializeOwned,
    F: Fn(&MultiIndex) -> T + Send + Sync,
    B: Fn(&[MultiIndex]) -> Vec<T> + Send + Sync;
```

Every rank supplies equivalent callbacks and metadata. Only the root supplies
pending work. Invalid common input is rejected collectively before the first
patch wave. Every rank passes its own pool-backed, id-zero `Domain`; the call
originates on the MPI main thread outside any Rayon worker, and MPI must provide
`MPI_THREAD_FUNNELED` or stronger, matching Hataori preflight.

All three entry points return the same result shape:

```text
AdaptiveInterpolationResult<T> {
    partitioned_tt: PartitionedTT,
    patch_caches: Vec<AcceptedPatchCache<T>>,
}
```

The result owns the final per-patch caches. It provides `partitioned_tt()`,
`patch_caches()`, and `into_parts()` accessors. Each returned cache is paired
with exactly one accepted projector and provides local lookup, `len()`,
`is_empty()`, `clear()`, and retained-entry statistics. This replaces the old
bare-`PartitionedTT` adaptive return rather than adding a second parallel-only
result convention. The same branch updates all affected rustdoc, tests,
`crates/tensor4all-partitionedtt/README.md`, generated API references, and
`skills/use-tensor4all-rs/references/{recipes,crates}.md`; no stale bare-result
examples remain.

## Stable patch identity

Represent a pending patch by its path in the subdivision tree:

```text
PendingPatch<T> {
    path: Vec<usize>,
    recycled_pivots: Vec<MultiIndex>,
    cache: PatchCache<T>,
}
```

`path[d]` is the selected value of `patch_order[d]`. Because adaptive splitting
always selects the first unprojected index in `patch_order`, the path reconstructs
the full `Projector` without serializing `DynIndex` or matching indices by ID.
Full index identity remains local in `site_indices` and `patch_order`.

The path also supplies a schedule-independent RNG seed. All entry points,
including the no-default-feature sequential entry point, derive each patch seed
from the configured root seed and all path coordinates with one fixed documented
integer mixer. The implementation uses SplitMix64: initialize with
`splitmix64(root_seed ^ 0x6a09e667f3bcc909)`, then for every `(depth, value)`
apply `splitmix64(state ^ rotate_left(depth as u64, 32) ^ value as u64)`. TCI's
inner seed is one further `splitmix64` application. Do not use `DefaultHasher`,
whose stability is not a contract. This intentionally replaces the legacy
single RNG stream so sequential, Rayon, and MPI modes generate identical
candidates independent of completion order.

## Wavefront scheduling

Replace the sequential FIFO loop with breadth-first waves:

1. The root starts with the empty-path patch and an empty cache.
2. Process every patch in the current wave with Hataori.
3. Preserve input order in the returned outcomes.
4. Append accepted patches and their returned caches to the result; append
   rejected patches' children to the next wave, in path order.
5. Stop when the next wave is empty.

The rank-local implementation uses:

```text
hataori::map_in(domain, LocalMode::Outer, wave, process_patch)
```

The MPI implementation uses one collective `hataori::pmap` per wave with
`batch_size = 1` and `LocalMode::Outer`. Unit batches let remote ranks receive
small child waves instead of allowing the root's local batch to consume every
child; nested Rayon work inside each patch still uses the supplied domain pool.
After `pmap` returns, root
returns, root performs every fallible root-only step for that wave, including
wire-core reconstruction, cache/result alignment, next-wave construction, and,
for the last wave, `PartitionedTT::from_subdomains`. Root then broadcasts one
serializable control result:

```text
WaveControl = Continue | Stop | Fail(WireAdaptiveError)
```

Every rank consumes this control before either entering another `pmap` or
returning. `Fail` preserves the diagnostic message and maps it to the public
`DistributedAdaptiveInterpolation` error category on every rank; patch paths
are included when the originating diagnostic provides one. `Stop` is
broadcast only after final root validation succeeds. Thus no root-side failure
window can leave non-root ranks in the next collective or return success on
only part of the communicator. Any failure returns no partial partition or
cache.

Breadth-first waves preserve the current FIFO subdivision order. Final accepted
patches are sorted by `path` before constructing `PartitionedTT`, making output
order independent of scheduling.

## Nested Rayon contract

`LocalMode::Outer` executes different patches in parallel. A patch callback may
itself use Rayon (`join`, `scope`, or parallel iterators). Because Hataori enters
the supplied domain pool before invoking the callback, nested Rayon work uses
the same pool and work-stealing scheduler; Tensor4all must not create another
pool or divide the thread count manually.

This is intentional nested parallelism. It avoids oversubscription as long as
inner work remains on the current Rayon pool. Callbacks that create their own
thread pools are outside the contract.

## Patch-owned sample cache

Add one private cache owned by each pending patch:

```text
PatchCache<T> {
    active_dims: Vec<usize>,
    entries: HashMap<MultiIndex, T>,
}
```

Keys use coordinates of the patch's active sites in their existing order. A
pending cache moves into either its children or the accepted result; it is never
silently dropped at a scheduling boundary. Its size is naturally bounded by the
finite patch volume and the number of distinct points requested by TCI2. No
global or thread-local cache is introduced.

All evaluations for a patch go through one cache wrapper, including:

- seeded candidate checks;
- scalar TCI2 evaluations;
- batch TCI2 evaluations; and
- final site-tensor and global-pivot evaluations.

For a batch, look up and deduplicate keys first, call the user batch callback
only for misses, validate the returned length, then insert and restore the
original order. Do not hold a cache lock while executing user code. A
patch-local lock is sufficient for the `Fn` callback required by TCI2; caches
are never shared between concurrently scheduled patches.

## One-pass cache projection on split

When a parent fixes active coordinate `split_pos` with dimension `m`, consume
the parent cache and partition it into all `m` child caches in one pass:

```text
split_cache(parent, split_pos, m):
    children = m empty caches with the split dimension removed
    for (key, value) in parent.entries:
        child = key[split_pos]
        child_key = key with coordinate split_pos removed
        children[child].insert(child_key, value)
    return children
```

Mathematically, child `v` receives exactly

```text
C_v = { (x without split_pos, y) | (x, y) in C_parent
                                      and x[split_pos] = v }.
```

Therefore every parent entry follows exactly one edge of the split tree. No
entry is copied to a sibling, and no child receives a sample outside its patch.
The operation is `O(number of parent entries)` plus child allocation, rather
than scanning the parent once per child.

The same child-construction loop must zip together:

- child path;
- reconstructed projector value;
- projected cache bucket; and
- compatible recycled pivots.

This keeps projector, cache, and pivot projection on one canonical split path.
Malformed cache keys or out-of-range split coordinates are internal invariant
errors, not silently dropped entries.

## Returned per-patch caches

An accepted worker outcome contains its path, TT cores, and the final
`PatchCache<T>`. The root returns these caches as a collection aligned one-to-one
with the accepted `SubDomainTT`s. There is no shared parent cache and no cache
stored on the subdivision tree.

```text
AcceptedPatchCache<T> {
    projector: Projector,
    active_positions: Vec<usize>,
    cache: PatchCache<T>,
}
```

A split physically consumes the parent's map and creates independent child
maps. Each entry is moved into exactly one child map; siblings share neither the
map nor its entries. The subdivision path is only stable patch identity and MPI
metadata, not cache storage or cache lookup machinery.

`into_parts()` returns the `PartitionedTT` and the aligned cache collection
without copying entries. A global full-index lookup index is deferred; add one
only if a measured caller needs it.

## MPI wire representation

Hataori serializes work and outcomes, but closures remain rank-local. Keep wire
types private:

- pending path, recycled pivots, and `PatchCache<T>` are serialized directly;
- a rejected outcome returns child pending patches;
- an accepted outcome returns the path, its final `PatchCache<T>`, and TT cores
  as column-major core buffers plus their three dimensions.

The accepted wire form serializes only TT parameters, not the full dense tensor.
The root reconstructs each core through existing `tensor3_from_data` and embeds
it with the projector reconstructed from the path. Shape products and wire
lengths are checked before allocation. No public tensor serialization format is
created by this feature.

## Error and cancellation semantics

- Callback values must describe the same deterministic function on every rank.
- Batch length mismatches remain typed adaptive/TCI errors.
- Hataori scheduling, wire, or collective failures are wrapped with their patch
  path and preserved as the source of `PartitionedTTError`.
- After any wave failure, no later wave starts and no partial result is returned.
- MPI ranks converge on the same failure through Hataori before leaving the
  collective entry point.

## Verification matrix

Required focused tests:

1. the public sequential entry point and `adaptiveinterpolate_in` agree
   numerically, have identical projector paths, and use identical per-path
   candidates for a problem that splits more than once;
2. nested Rayon work inside `process_patch` completes on the supplied domain
   without creating another pool;
3. one-pass cache projection sends every entry to exactly one child and removes
   the fixed coordinate from its key;
4. child evaluation hits inherited in-patch samples and re-evaluates samples
   belonging to siblings;
5. cache projection works when the split site is first, middle, or last among
   active sites;
6. fixed seed results are independent of Rayon scheduling order;
7. invalid common MPI input is rejected collectively before the first wave;
8. batch-cache lookup deduplicates misses, calls the user callback only for
   misses, validates returned length, and restores input order; the mismatch
   path returns no partial result;
9. `--no-default-features` builds without Hataori;
10. the default feature set builds Rayon support without MPI;
11. returned caches are aligned one-to-one with accepted patches, have distinct
    map allocations, contain no sibling-only samples, and correctly implement
    `clear()` and retained-entry statistics;
12. malformed wire core shape products and payload lengths are rejected before
    root allocation/reconstruction and the failure converges on every rank;
13. the MPI feature build and a multi-rank smoke test agree with rank-local
    execution, including a split whose accepted TT and cache are returned from
    a worker;
14. fixed-seed results are independent of MPI completion order;
15. callback, malformed wire, root post-processing, and batch-length failures
    return no partial partition or cache on any rank.

Performance evidence should report patch counts, cache hits, newly evaluated
samples, and wall time for sequential, Rayon, and MPI runs. It must also confirm
that cache projection is linear in retained entries and that MPI transfers TT
core parameters rather than dense-domain values.

## Deferred work

Do not add a general executor trait, global cache, work-stealing patch queue, or
cache eviction policy in the first implementation. Hataori already owns
scheduling, and breadth-first waves are sufficient until measurements show a
load-balance problem that bounded prefetch cannot address.
