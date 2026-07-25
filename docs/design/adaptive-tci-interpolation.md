# Adaptive TCI interpolation

## Scope

`tensor4all-partitionedtt::adaptiveinterpolate` approximates a discrete function
with a collection of mutually disjoint tensor-train patches. It is intended for
functions that are low-rank only after restricting selected site indices.

The queue and split flow follow the `adaptiveinterpolate`, `createpatch`, and
`_globalpivots` design in TCIAlgorithms.jl. The Rust implementation remains
sequential so callback thread-safety is not part of the API contract.

## Patch model

A patch consists of a `Projector` and optional full-domain recycled pivots. The
projector fixes zero-based coordinates at selected site indices. All other sites
are active and are passed to TCI2 in their original order.

The algorithm processes a FIFO queue:

1. Evaluate patches with zero or one active site exactly.
2. For larger patches, filter user and recycled pivots for compatibility,
   deduplicate them, and replenish the candidate set with a seeded generator.
3. Run TCI2 on the active sites.
4. Accept only a patch for which TCI2 reports `TCI2Termination::Converged` and
   whose accepted error satisfies the requested tolerance.
5. Otherwise fix the next active index in `patch_order` and enqueue one child
   for each coordinate of that index.

Reaching `max_bond_dim` or exhausting `max_iter` is a stopping condition for
TCI2, not proof that an adaptive patch converged. Such a patch is split.

The patch order must be an exact permutation of the site indices. Index
identity includes the ID, dimension, tags, and prime level; matching by ID alone
is not sufficient.

## Pivot policy

Initial pivots and recycled pivots use full-domain coordinates. Each patch keeps
only compatible pivots and converts them to active-site coordinates.

Pivot recycling is opt-in. When enabled, diagonal TCI pivots from a rejected
parent seed its children. A child that receives no compatible recycled pivot is
not classified as zero: seeded candidates replenish it to
`n_initial_pivots`, bounded by the number of points in the patch.

The product of active dimensions and the random-attempt count use checked
arithmetic. An unrepresentable count is rejected before enumeration or
allocation.

## Sampled-zero policy

A patch may be represented as zero when every supplied and seeded candidate is
sampled as zero. This is a finite-sampling policy, not a mathematical proof.
Sparse functions should supply pivots in known nonzero regions. The numerical
zero predicate must not silently discard nonzero functions merely because their
scale is small.

## Embedding active tensor trains

An accepted active-site TT is embedded back into the full site order. Fixed
boundary sites have unit carried bonds. A fixed middle site carries equal left
and right bonds with a compact copy-selector tensor:

```text
scale * delta(left, right) * delta(site, selected_value)
```

The owning `tensor4all-core` constructor uses structured axis classes
`[0, 1, 0]`. Its payload contains `bond_dim * site_dim` elements rather than a
dense `bond_dim * site_dim * bond_dim` buffer. Bond products, payload sizes, and
stride conversions are checked before allocation.

Tests of this path must verify both compact representation and dense numerical
behavior.

## Callback contract

The scalar callback receives one full zero-based multi-index. The optional batch
callback receives full multi-indices and must return one value per input in the
same order. Both callbacks describe the same deterministic function; batching
may change evaluation strategy but not values.

## Output and complexity

Accepted projectors are mutually disjoint and cover the original domain because
every rejected patch is replaced by all children of one complete coordinate
split. `PartitionedTT::from_subdomains` validates disjointness at construction.

Patch execution is sequential and deterministic for a fixed seed and
deterministic callbacks. The number of patches can grow with the subdivision
tree; callers should choose rank limits and patch order with that tradeoff in
mind. A general improvement to the quadratic final disjointness validation is
outside this design change and must preserve validation rather than bypass it.
