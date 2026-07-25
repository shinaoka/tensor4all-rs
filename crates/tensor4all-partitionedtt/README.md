# tensor4all-partitionedtt

Partitioned Tensor Train for representing functions over non-overlapping subdomains
with projectors.

## Key Types

- `PartitionedTT` — collection of non-overlapping subdomain tensor trains
- `SubDomainTT` — tensor train restricted to a specific subdomain
- `Projector` — maps tensor indices to fixed values defining subdomains
- `adaptiveinterpolate` — runs TCI2 per patch and subdivides patches that miss the requested tolerance
- `AdaptiveInterpolateOptions` — controls TCI2, patch order, initial pivots, and opt-in pivot recycling

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

assert_eq!(result.len(), 1);
assert!(result.projectors().next().unwrap().is_empty());
```

Set `recycle_pivots` to reuse compatible parent TCI pivots in child patches.
Children with no compatible recycled pivots are replenished with seeded random
candidates rather than being treated as zero. A patch is classified as sampled
zero only when all of its initial candidates evaluate below `1e-30`; provide
known nonzero pivots for very sparse functions.

## Documentation

- [User Guide: Tensor Train](https://tensor4all.org/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_partitionedtt/)
