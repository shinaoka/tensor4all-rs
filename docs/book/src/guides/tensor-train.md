# Tensor Train

A **Tensor Train** (TT), also known as a Matrix Product State (MPS), represents a
high-dimensional tensor as a chain of low-rank cores.  tensor4all-rs provides two
complementary implementations:

| Crate | Best for |
|---|---|
| `tensor4all-simplett` | Lightweight numerical work with raw arrays |
| `tensor4all-itensorlike` | ITensors.jl-like Index semantics, orthogonality tracking, canonical forms |

**When to choose which:**
Use `tensor4all-simplett` when you want fast numerics with minimal boilerplate
(no named indices needed). Use `tensor4all-itensorlike` when you need named
indices, automatic orthogonality tracking, or ITensors.jl compatibility.
If you already have a full `IdxTensor`, follow [Dense Tensor to Tensor Train
with TT-SVD](dense-tt-svd.md).

---

## SimpleTT

The `tensor4all-simplett` crate offers a minimal, efficient TT implementation.
It works with plain `f64` and `Complex64` scalars and does not require you to
manage named indices.

### Creating a tensor train

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_simplett::prelude::*;

// Constant TT: all entries equal to 1.0, physical dimensions [2, 3, 4]
let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
assert_eq!(tt.len(), 3);
assert_eq!(tt.site_dims(), vec![2, 3, 4]);
assert_eq!(tt.link_dims(), vec![1, 1]); // bond dim = 1 for a constant

// Zero TT: all entries are zero
let zero_tt = SimpleTensorTrain::<f64>::zeros(&[2, 3, 4]);
assert!((zero_tt.sum()).abs() < 1e-14);
# Ok(())
# }
```

### Evaluating and summing

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_simplett::prelude::*;
# let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
// Evaluate the tensor at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;
assert!((value - 1.0).abs() < 1e-12);

// Sum over all multi-indices (equivalent to contracting with all-ones vectors)
let total = tt.sum();
// For the constant TT: sum = 1.0 * 2 * 3 * 4 = 24.0
assert!((total - 24.0).abs() < 1e-10);
# Ok(())
# }
```

### Compressing

`CompressionOptions` controls the accuracy--cost trade-off:

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_simplett::prelude::*;
// Build a TT with artificially inflated bond dimension by adding two constants
let a = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
let b = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 2.0);
let big = a.add(&b)?; // bond dim = 2, but rank-1 would suffice
assert_eq!(big.rank(), 2);

let options = CompressionOptions {
    tolerance: 1e-10,
    max_bond_dim: Some(20),
    ..Default::default()
};
let compressed = big.compressed(&options)?;

// Compression found the optimal rank
assert_eq!(compressed.rank(), 1);
// Values are preserved: 1.0 + 2.0 = 3.0
assert!((compressed.evaluate(&[0, 1, 2])? - 3.0).abs() < 1e-10);
# Ok(())
# }
```

The compression reduces bond dimensions while keeping the approximation error
below `tolerance` (relative truncation threshold), up to `max_bond_dim`.

**Tolerance guidance:**
- `1e-12` (default): near machine precision, almost lossless.
- `1e-8` to `1e-6`: good for most scientific applications.
- Tighter tolerances produce larger bond dimensions and slower evaluation.

### End-to-end workflow

This example shows the complete lifecycle: create, add, compress, evaluate, and
verify.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_simplett::prelude::*;

// Step 1: Create two constant TTs
let a = SimpleTensorTrain::<f64>::constant(&[4, 4, 4], 1.0);
let b = SimpleTensorTrain::<f64>::constant(&[4, 4, 4], 2.0);

// Step 2: Add them (bond dim doubles)
let sum = a.add(&b)?;
assert_eq!(sum.rank(), 2);

// Step 3: Compress
let compressed = sum.compressed(&CompressionOptions::default())?;
assert_eq!(compressed.rank(), 1);

// Step 4: Evaluate and verify
for i in 0..4 {
    for j in 0..4 {
        let val = compressed.evaluate(&[i, j, 0])?;
        assert!((val - 3.0).abs() < 1e-10);
    }
}

// Step 5: Check norm
// norm^2 = 3^2 * 4^3 = 576, norm = 24
assert!((compressed.norm() - 24.0).abs() < 1e-10);
# Ok(())
# }
```

---

## ITensorLike TensorTrain

The `tensor4all-itensorlike` crate provides a higher-level API modelled after
[ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).  Each tensor carries
named `DynIndex` objects so that contractions are unambiguous regardless of axis
ordering.

> **Key conventions**
> - Sites are **0-indexed** (Julia uses 1-indexed).
> - `IdxTensor::from_dense` expects data in **column-major** (Fortran) order.
> - `inner()` computes `<self|other>` with complex conjugation on `self`.

### Creating and orthogonalizing

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_itensorlike::prelude::*;

// Site and bond indices
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

// Build site tensors (column-major data)
let t0 = IdxTensor::from_dense(
    vec![s0.clone(), b01.clone()],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let t1 = IdxTensor::from_dense(
    vec![b01.clone(), s1.clone(), b12.clone()],
    vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
)?;
let t2 = IdxTensor::from_dense(
    vec![b12.clone(), s2.clone()],
    vec![1.0, 0.0, 0.0, 1.0],
)?;

// Assemble and orthogonalize with center at site 1
let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
let norm_before = tt.norm()?;
tt.orthogonalize(1)?;

assert!(tt.is_ortho());
assert_eq!(tt.ortho_center(), Some(1));

// Orthogonalization preserves the tensor train value
assert!((tt.norm()? - norm_before).abs() < 1e-10);
# Ok(())
# }
```

### Truncating

After orthogonalization you can truncate bond dimensions by SVD:

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_itensorlike::prelude::*;
# let s0 = DynIndex::new_dyn(2);
# let s1 = DynIndex::new_dyn(2);
# let s2 = DynIndex::new_dyn(2);
# let b01 = DynIndex::new_bond(4)?;
# let b12 = DynIndex::new_bond(4)?;
# let t0 = IdxTensor::from_dense(
#     vec![s0, b01.clone()],
#     (0..8).map(|i| i as f64).collect(),
# )?;
# let t1 = IdxTensor::from_dense(
#     vec![b01, s1, b12.clone()],
#     (0..32).map(|i| i as f64).collect(),
# )?;
# let t2 = IdxTensor::from_dense(
#     vec![b12, s2],
#     (0..8).map(|i| i as f64).collect(),
# )?;
# let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
# tt.orthogonalize(1)?;
tt.truncate(
    &TruncateOptions::svd()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
        .with_max_bond_dim(2),
)?;
assert!(tt.max_bond_dim() <= 2);
# Ok(())
# }
```

`SvdTruncationPolicy::new(threshold)` uses the default relative per-value rule.
To emulate ITensor-style discarded-weight cutoffs, use
`.with_squared_values().with_discarded_tail_sum()`.

### Norm and inner product

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_itensorlike::prelude::*;
# let s0 = DynIndex::new_dyn(2);
# let s1 = DynIndex::new_dyn(2);
# let b = DynIndex::new_bond(2)?;
# let t0 = IdxTensor::from_dense(
#     vec![s0, b.clone()],
#     vec![1.0_f64, 0.0, 0.0, 1.0],
# )?;
# let t1 = IdxTensor::from_dense(
#     vec![b, s1],
#     vec![1.0, 0.0, 0.0, 1.0],
# )?;
# let tt = TensorTrain::new(vec![t0, t1])?;
let norm = tt.norm()?;
assert!(norm.is_finite());

// <tt|tt> = ||tt||^2
let inner = tt.inner(&tt)?;
assert!((inner.real() - norm * norm).abs() < 1e-10);
# Ok(())
# }
```

### Complex scalars

The same API works with `Complex64`:

```rust
# fn main() -> anyhow::Result<()> {
use num_complex::Complex64;
use tensor4all_itensorlike::prelude::*;

let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let b  = DynIndex::new_bond(2)?;

let t0 = IdxTensor::from_dense(
    vec![s0, b.clone()],
    vec![
        Complex64::new(1.0,  0.0), Complex64::new(0.0,  1.0),
        Complex64::new(0.0, -1.0), Complex64::new(1.0,  0.0),
    ],
)?;
let t1 = IdxTensor::from_dense(
    vec![b, s1],
    vec![
        Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0),
    ],
)?;

let tt = TensorTrain::new(vec![t0, t1])?;

// Norm is sqrt(<tt|tt>) = sqrt(conj(tt) * tt summed over all indices)
let norm = tt.norm()?;
assert!(norm > 0.0);

// For complex tensors, inner product uses complex conjugation on self
let inner = tt.inner(&tt)?;
assert!((inner.real() - norm * norm).abs() < 1e-10);
# Ok(())
# }
```

---

## Differences from ITensorMPS.jl

| Feature | tensor4all-itensorlike | ITensorMPS.jl |
|---|---|---|
| Site indexing | 0-indexed | 1-indexed |
| Canonical forms | Unitary (QR), LU, CI | Unitary (SVD) |
| Conjugation | `conj` | `dag` (with QN direction flip) |
| Sweep counting | `nhalfsweeps` | `nsweeps` |
