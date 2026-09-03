# Tensor Basics

This guide covers the `tensor4all-core` crate, which provides the foundation for
all tensor operations: indices, tensors, contraction, and factorization.

If you have not set up a dependency yet, add `tensor4all-core` to your
`Cargo.toml`. Use a git dependency from an external project:

```toml
[dependencies]
tensor4all-core = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-core" }
```

Or use a path dependency when working from a local checkout:

```toml
[dependencies]
tensor4all-core = { path = "../tensor4all-rs/crates/tensor4all-core" }
```

## Index

Every tensor axis is identified by an `Index`. Indices carry a unique identity
(so two indices with the same dimension are still distinct), an optional tag, and
an optional prime level.

```rust
use tensor4all_core::index::{Index, DynId};
use tensor4all_core::IndexLike; // needed for .dim() and .plev()

// Simplest form: just give a dimension.
let i = Index::new_dyn(3);   // dimension 3, auto-generated ID, no tags
let j = Index::new_dyn(4);

// A tag names the index (useful for debugging and tag-based operations).
let site = Index::new_dyn_with_tag(2, "Site").unwrap();
assert_eq!(site.dim(), 2);

// Two indices created independently are always distinct, even with the same dim.
let a = Index::new_dyn(3);
let b = Index::new_dyn(3);
assert_ne!(a, b);

// Prime levels distinguish related indices (e.g. ket vs bra in quantum physics).
let bra = site.prime();       // plev 0 -> 1
let ket = site.noprime();     // always plev 0
assert_ne!(bra, ket);

// Inspect properties.
assert_eq!(site.dim(), 2);
assert_eq!(site.plev(), 0);
assert_eq!(bra.plev(), 1);
```

## Tensor (IdxTensor)

`IdxTensor` is a dynamic-rank tensor parameterized by a list of `Index`
values and backed by compact storage that may be dense, diagonal, or explicitly
structured. Each index uniquely identifies an axis; there is no fixed axis
ordering in the abstract sense — operations match axes by index identity.

### Creating tensors

```rust
use tensor4all_core::{IdxTensor, Index};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);

// From explicit column-major data (2×3 tensor, 6 elements).
let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
let t = IdxTensor::from_dense(vec![i.clone(), j.clone()], data).unwrap();
assert_eq!(t.dims(), vec![2, 3]);

// All-zeros tensor.
let zeros = IdxTensor::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();

// Random tensor (standard normal).
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
let mut rng = ChaCha8Rng::seed_from_u64(42);
let rand_t: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]).unwrap();
assert_eq!(rand_t.dims(), vec![2, 3]);
```

### Extracting data

```rust
use tensor4all_core::{IdxTensor, Index};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let data = vec![10.0_f64, 20.0];
let t = IdxTensor::from_dense(vec![i], data).unwrap();

// Extract all elements in column-major order.
let out: Vec<f64> = t.to_vec().unwrap();
assert_eq!(out, vec![10.0, 20.0]);

// Sum all elements.
let s = t.sum().unwrap();
assert_eq!(s.real(), 30.0);
```

## Contraction

Contraction sums over all shared (common) indices between two or more tensors.
Think of it as a generalization of matrix multiplication.

### Pairwise contraction

```rust
use tensor4all_core::{IdxTensor, Index, contract};
use tensor4all_core::index::DynId;

// A[i,j] and B[j,k] — contracting over j gives C[i,k].
let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);

let a = IdxTensor::zeros::<f64>(vec![i.clone(), j.clone()]).unwrap();
let b = IdxTensor::zeros::<f64>(vec![j.clone(), k.clone()]).unwrap();

let c = contract(&[&a, &b]).unwrap();
assert_eq!(c.dims(), vec![2, 4]);  // j is summed away
```

### Multi-tensor contraction

`contract` contracts a connected list of tensors. Disconnected inputs are an
error; use `outer_product` explicitly when a tensor product of disconnected
pieces is intended.

```rust
use tensor4all_core::{
    IdxTensor, Index, contract, outer_product,
};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);
let l = Index::new_dyn(5);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(0)
};
let a: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]).unwrap();
let b: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![j.clone(), k.clone()]).unwrap();
let c: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![k.clone(), l.clone()]).unwrap();

// Contract A(i,j) * B(j,k) * C(k,l) -> result(i,l)
let result = contract(&[&a, &b, &c]).unwrap();
assert_eq!(result.dims().iter().product::<usize>(), 2 * 5); // i * l

// Disconnected products are explicit.
let product = outer_product(&a, &c).unwrap();
assert_eq!(product.dims().iter().product::<usize>(), 2 * 3 * 4 * 5);
```

### Repeated contraction with a prepared plan

When repeated operands keep exactly the same ordered indices, dimensions, and
axis classes, prepare the index matching and label assignment once. Values,
dtypes, and gradient-tracking state may change; fresh index identities require a
new plan.

```rust
use tensor4all_core::{ContractionOptions, IdxTensor, Index, PreparedContraction};

let i = Index::new_dyn(2);
let j = Index::new_dyn(2);
let k = Index::new_dyn(2);
let a = IdxTensor::from_dense(
    vec![i.clone(), k.clone()],
    vec![1.0, 2.0, 3.0, 4.0],
).unwrap();
let b = IdxTensor::from_dense(
    vec![k, j.clone()],
    vec![5.0, 6.0, 7.0, 8.0],
).unwrap();
let c = IdxTensor::from_dense(vec![j], vec![1.0, 2.0]).unwrap();

let plan = PreparedContraction::new(&[&a, &b, &c], ContractionOptions::new()).unwrap();
let result = plan.execute(&[&a, &b, &c]).unwrap();
assert_eq!(result.indices(), &[i]);
assert_eq!(result.to_vec::<f64>().unwrap(), vec![85.0, 126.0]);
```

## Factorization

The unified `factorize()` function dispatches to SVD, QR, LU, or CI based on
`FactorizeOptions`. The result splits the input tensor into a `left` and `right`
factor connected by a new bond index.

### SVD with truncation

```rust
use tensor4all_core::{IdxTensor, Index, factorize, FactorizeOptions, SvdTruncationPolicy};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(4);
let j = Index::new_dyn(6);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(1)
};
let t: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]).unwrap();

// SVD: split along i | j, discarding singular values below the chosen policy threshold.
let opts = FactorizeOptions::svd().with_svd_policy(SvdTruncationPolicy::new(1e-10));
let result = factorize(&t, &[i.clone()], &opts).unwrap();

// result.left  has indices [i, bond]
// result.right has indices [bond, j]
// result.left * result.right ≈ t  (within tolerance)
let bond_dim = result.rank;
println!("bond dimension after SVD: {bond_dim}");

// Limit bond dimension explicitly.
let opts_capped = FactorizeOptions::svd().with_max_bond_dim(2);
let result_capped = factorize(&t, &[i], &opts_capped).unwrap();
assert!(result_capped.rank <= 2);
```

### QR decomposition

```rust
use tensor4all_core::{IdxTensor, Index, factorize, FactorizeOptions};
use tensor4all_core::index::DynId;

let i = Index::new_dyn(4);
let j = Index::new_dyn(6);

let mut rng = {
    use rand::SeedableRng;
    rand_chacha::ChaCha8Rng::seed_from_u64(2)
};
let t: IdxTensor =
    IdxTensor::random::<f64, _>(&mut rng, vec![i.clone(), j.clone()]).unwrap();

// QR: left factor is orthogonal (Q), right factor is upper-triangular (R).
let opts = FactorizeOptions::qr();
let result = factorize(&t, &[i], &opts).unwrap();
// result.left  = Q  (orthogonal columns)
// result.right = R  (upper triangular)
```

Both `FactorizeOptions::svd()` and `FactorizeOptions::qr()` return builder
structs. Use `.with_svd_policy(policy)` for SVD and `.with_qr_rtol(tol)` for
QR-specific rank control. `with_max_bond_dim(n)` remains available as an
algorithm-independent hard cap. For SVD, `result.singular_values` holds the
retained singular values.
