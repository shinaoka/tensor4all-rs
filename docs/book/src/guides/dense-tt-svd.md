# Dense Tensor to Tensor Train with TT-SVD

Use `TensorTrain::from_dense` when all values of an indexed tensor already fit
in memory and you want a sequential SVD decomposition. For sampled functions
whose full tensor does not fit in memory, use TCI instead.

## Complete example

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::{DynIndex, IdxTensor, SvdTruncationPolicy};
# use tensor4all_itensorlike::{CanonicalForm, SvdOptions, TensorTrain};
let sites = [
    DynIndex::new_dyn(2),
    DynIndex::new_dyn(2),
    DynIndex::new_dyn(2),
];

// IdxTensor dense buffers are column-major: sites[0] varies fastest.
// These values are [1, 2] ⊗ [1, 3] ⊗ [1, 4].
let dense = IdxTensor::from_dense(
    sites.to_vec(),
    vec![1.0, 2.0, 3.0, 6.0, 4.0, 8.0, 12.0, 24.0],
)?;
let options = SvdOptions::new()
    .with_policy(SvdTruncationPolicy::new(1.0e-12))
    .with_max_bond_dim(16);
let train = TensorTrain::from_dense(&dense, &sites, &options)?;

let reconstructed = train.to_dense()?;
assert!(dense.distance(&reconstructed)? < 1.0e-12);
assert_eq!(train.bond_dims(), vec![1, 1]);
assert_eq!(train.ortho_center(), Some(2));
assert_eq!(train.canonical_form(), Some(CanonicalForm::Unitary));
# Ok(())
# }
```

The `sites` slice defines the tensor-train order. It must contain every full
`DynIndex` of the input exactly once, but it need not match the input tensor's
axis order. Prime levels and tags are part of index identity.

## Sweep and canonical form

TT-SVD splits from left to right. At each bond, `U` becomes the completed
left-canonical core and `S Vᴴ` is carried into the next split. The final core
therefore holds the remaining norm, and the returned train records the final
site as its orthogonality center.

## Truncation

`SvdOptions` has two independent controls:

- `with_policy(...)` chooses which singular-value tail is discarded at each
  bond;
- `with_max_bond_dim(n)` caps every retained bond at `n`.

For a discarded-Frobenius-weight rule, use squared singular values and a tail
sum:

```rust
# use tensor4all_core::SvdTruncationPolicy;
let local_policy = SvdTruncationPolicy::new(1.0e-10)
    .with_squared_values()
    .with_discarded_tail_sum();
assert_eq!(local_policy.threshold, 1.0e-10);
```

The policy is applied separately at each of the `number_of_sites - 1` splits.
It is therefore not, by itself, a statement of the final global relative error.
When a global tolerance matters, allocate a tighter per-bond error budget and
verify the result with `dense.distance(&train.to_dense()?)`, as in the example.
A hard `max_bond_dim` can force a larger error regardless of the policy.

## Memory cost

This is an explicitly dense algorithm. The input already costs the product of
all site dimensions, and SVD workspaces plus the carried tensor can require
several times that storage. Do not use dense TT-SVD as a hidden compression
path for data that cannot already be materialized. Use TCI for oracle-defined
or otherwise non-materialized data.
