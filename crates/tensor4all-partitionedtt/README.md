# tensor4all-partitionedtt

> **Deprecated:** use [`tensor4all-partitionedtreetn`](../tensor4all-partitionedtreetn/)
> for new TreeTN-based partitioned tensor-network work.
>
> Both crates coexist during migration. This crate remains buildable and accepts
> correctness and security fixes only; no removal date has been set, and removal
> requires a separate maintainer decision.

Partitioned Tensor Train for representing functions over non-overlapping subdomains
with projectors.

## Key Types

- `PartitionedTT` — collection of non-overlapping subdomain tensor trains
- `SubDomainTT` — tensor train restricted to a specific subdomain
- `Projector` — maps tensor indices to fixed values defining subdomains
- `adaptiveinterpolate` — runs TCI2 per patch and subdivides patches that miss the requested tolerance
- `AdaptiveInterpolateOptions` — controls TCI2, patch order, initial pivots, and opt-in pivot recycling
- `AdaptiveInterpolationResult` — partitioned TT plus one returned sample cache per accepted patch

## Adaptive interpolation

```rust
use tensor4all_partitionedtt::{
    adaptiveinterpolate, AdaptiveInterpolateOptions, DynIndex, MultiIndex,
};

let sites = vec![DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
let f = |index: &MultiIndex| ((index[0] + 1) * (index[1] + 1)) as f64;
let result = adaptiveinterpolate::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
    f,
    None,
    sites,
    vec![vec![1, 1]],
    AdaptiveInterpolateOptions::default(),
)
.unwrap();

assert_eq!(result.partitioned_tt().len(), 1);
assert!(result.partitioned_tt().projectors().next().unwrap().is_empty());
assert_eq!(result.patch_caches().len(), 1);
assert!(!result.patch_caches()[0].is_empty());
```

Set `recycle_pivots` to reuse compatible parent TCI pivots in child patches.
Children with no compatible recycled pivots are replenished with seeded random
candidates rather than being treated as zero. A patch is classified as sampled
zero only when all of its initial candidates evaluate below `1e-30`; provide
known nonzero pivots for very sparse functions. When a patch splits, its cache
is consumed and partitioned among all children in one pass; each sample moves to
exactly one child. The accepted per-patch caches are returned with the result.

The default feature set enables optional Hataori rank-local scheduling. Use
`adaptiveinterpolate_in` with an explicit Hataori domain for
`LocalMode::Outer` Rayon patch execution; Rayon used inside a callback is nested
in the same pool. MPI remains default-off behind `adaptive-hataori-mpi` and uses
`adaptiveinterpolate_mpi` collectively. Building that feature requires a system
MPI implementation and development headers (`libopenmpi-dev`/`openmpi-bin` on
Ubuntu).

## Documentation

- [User Guide: Tensor Train](https://tensor4all.org/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_partitionedtt/)
