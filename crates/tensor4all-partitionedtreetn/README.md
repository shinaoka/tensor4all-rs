# tensor4all-partitionedtreetn

TreeTN-native partitioned tensor networks with eagerly masked subdomain
patches. The crate supports arbitrary named tree topologies, multiple site
indices per node, homogeneous `f64`/`Complex64` partitions, strict algebra, and
volume-proportional adaptive patching.

Adaptive patching is bond-cap-driven and independent of adaptive interpolation:
this crate does **not** provide TCI, sampled-zero inference, or a dependency on
`tensor4all-treetci`.

## Truncation convention

The scalar adaptive truncation parameter is `PatchingOptions::cutoff`, a local
discarded-weight cutoff following the ITensorMPS convention. One absolute local
threshold `cutoff * ||F||^2 * volume_p / total_volume` is derived per operation
and applied whole at every local SVD. It is **best effort** for the final
whole-network error; `max_bond_dim` is a hard cap and takes precedence. No API
in this crate claims a global relative-error bound.

## Quick start

```rust
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_partitionedtreetn::{
    add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
};
use tensor4all_treetn::TreeTN;

let site0 = DynIndex::new_dyn(2);
let bond = DynIndex::new_dyn(2);
let site1 = DynIndex::new_dyn(2);
let left = IdxTensor::from_dense(
    vec![site0.clone(), bond.clone()],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let right = IdxTensor::from_dense(
    vec![bond, site1],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let tree = TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?;
let patch = SubDomainTreeTN::from_treetn(tree)?;
let result = add_with_patching(
    vec![patch],
    &0,
    &PatchingOptions {
        cutoff: 0.0,
        max_bond_dim: Some(1),
        patch_order: vec![site0],
        split_strategy: PatchSplitStrategy::Sequential,
    },
)?;

assert_eq!(result.len(), 2);
assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
# Ok::<(), Box<dyn std::error::Error>>(())
```

All flat tensor buffers use column-major order: the first listed index varies
fastest. Coordinates are zero-based. An operation that truncates or contracts
requires an explicit existing TreeTN node name as its center.

## Documentation

- [Tensor4all-rs user guide](https://tensor4all.org/tensor4all-rs/)
- [Partitioned TreeTN guide](https://tensor4all.org/tensor4all-rs/guides/partitioned-treetn.html)
- [API reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_partitionedtreetn/)
- [Migration design](../../docs/design/partitioned-treetn.md)
- [Provenance and citation policy](../../docs/PROVENANCE_AND_CITATION_POLICY.md)

The deprecated `tensor4all-partitionedtt` crate remains buildable during the
migration window. It is limited to correctness and security fixes; no removal
date has been set.
