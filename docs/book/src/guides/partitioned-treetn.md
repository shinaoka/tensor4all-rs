# Partitioned TreeTNs

`tensor4all-partitionedtreetn` stores TreeTN subdomains as eagerly masked
patches. It is the TreeTN-native successor to the deprecated
`tensor4all-partitionedtt` crate and supports named chains, branched trees, and
multiple site indices on one node.

This crate provides partition algebra and TreeTN-general adaptive patching. It
does not provide adaptive interpolation or TCI.

## Construct an eager patch

Projectors use zero-based coordinates and full index identity. Construction
retains every site axis but masks values outside the selected coordinates:

```rust
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{Projector, SubDomainTreeTN};
# use tensor4all_treetn::TreeTN;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let site = DynIndex::new_dyn(2);
let tensor = IdxTensor::from_dense(
    vec![site.clone()],
    vec![3.0_f64, 1.0e12],
)?;
let tree = TreeTN::from_tensors(vec![tensor], vec!["root".to_string()])?;
let patch = SubDomainTreeTN::new(
    tree,
    Projector::from_pairs([(site.clone(), 0)])?,
)?;

let node = patch.data().node_index(&"root".to_string()).ok_or("missing root")?;
assert_eq!(patch.data().tensor(node).ok_or("missing tensor")?.to_vec::<f64>()?,
           vec![3.0, 0.0]);
assert!((patch.norm_squared()? - 9.0).abs() < 1.0e-12);
# Ok(())
# }
```

Norms, inner products, contraction, truncation, and summation use this stored
masked value directly. No projector is re-applied and no full network is
densified.

## Adaptive patching

Every truncating or contracting operation takes an explicit existing node name
as its center. `add_with_patching` first assigns absolute squared-tail budgets
proportional to logical patch volume, then splits patches that remain above the
bond cap:

```rust
# use tensor4all_core::{DynIndex, IdxTensor};
# use tensor4all_partitionedtreetn::{
#     add_with_patching, PatchSplitStrategy, PatchingOptions, SubDomainTreeTN,
# };
# use tensor4all_treetn::TreeTN;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
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
let patch = SubDomainTreeTN::from_treetn(
    TreeTN::from_tensors(vec![left, right], vec![0usize, 1])?,
)?;
let result = add_with_patching(
    vec![patch],
    &0,
    &PatchingOptions {
        rtol: 0.0,
        max_bond_dim: Some(1),
        patch_order: vec![site0],
        split_strategy: PatchSplitStrategy::Sequential,
    },
)?;

assert_eq!(result.len(), 2);
assert!(result.values().all(|patch| patch.max_bond_dim() <= 1));
# Ok(())
# }
```

`PatchSplitStrategy::Sequential` follows `patch_order`. The default
`ExactParameterGain` forms and budget-truncates every candidate's children,
then compares checked sums of logical local tensor element counts. Structured
storage payload length and AD state are not used as the metric.

## Dtype and topology

A partition is homogeneous: all patches must use the same `IdxTensor` scalar
dtype and the same named topology and site-index assignment. Both `f64` and
`Complex64` are supported. Topology is not restricted to a chain; a TreeTN
with a central named node and three named leaves is a valid partition input.
See the [Tree Tensor Networks guide](tree-tn.md) for constructing branched
networks and selecting contraction/truncation options.

## Migration

Use this crate for new named TreeTN partition work. The old
`tensor4all-partitionedtt` crate remains buildable during migration and receives
correctness and security fixes only; no removal date has been set.
