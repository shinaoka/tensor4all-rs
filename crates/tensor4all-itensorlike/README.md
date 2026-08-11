# tensor4all-itensorlike

ITensors.jl-inspired TensorTrain API with orthogonality tracking and multiple canonical forms.

## Key Types

- `TensorTrain` — tensor train with orthogonality center tracking
- `orthogonalize()` — move orthogonality center to a given site
- `truncate()` — SVD-based bond dimension truncation
- `inner()` — inner product `<self|other>` with complex conjugation on `self`
- `norm()` — efficient norm via the orthogonality center

## Conventions

- Dense data passed to `TensorDynLen::from_dense` is **column-major**: the
  first listed index varies fastest.
- Complex tensors use `num_complex::Complex64`; `inner()` conjugates the left
  operand, so `tt.inner(&tt)` equals the square of `tt.norm()?` for real and
  complex tensor trains.
- Import `tensor4all_core::IndexLike` when calling trait methods such as
  `.dim()` on dynamic indices.
- Use `DynIndex::new_dyn(dim)` for ordinary site indices and
  `DynIndex::new_bond(dim)?` for bond indices; bond creation is fallible because
  bond metadata is validated.
- `ContractOptions` and `LinsolveOptions` provide
  `with_nsweeps(n)` as a convenience wrapper for `with_nhalfsweeps(2 * n)`.
- `truncate()` uses SVD truncation and leaves the tensor train in unitary
  canonical form; use `orthogonalize_with(site, CanonicalForm::...)` when you
  need LU or CI canonicalization instead.

## Example

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
use tensor4all_itensorlike::{TensorTrain, TruncateOptions};

// Build a 3-site tensor train
let s0 = DynIndex::new_dyn(2);
let s1 = DynIndex::new_dyn(2);
let s2 = DynIndex::new_dyn(2);
let b01 = DynIndex::new_bond(2)?;
let b12 = DynIndex::new_bond(2)?;

let t0 = TensorDynLen::from_dense(vec![s0, b01.clone()], vec![1.0, 0.0, 0.0, 1.0])?;
let t1 = TensorDynLen::from_dense(vec![b01, s1.clone(), b12.clone()], vec![1.0; 8])?;
let t2 = TensorDynLen::from_dense(vec![b12, s2], vec![1.0, 0.0, 0.0, 1.0])?;

let mut tt = TensorTrain::new(vec![t0, t1, t2])?;
tt.orthogonalize(1)?;
tt.truncate(&TruncateOptions::svd()
    .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-10))
    .with_max_bond_dim(2))?;

assert!(tt.isortho());
assert_eq!(s1.dim(), 2);
let norm = tt.norm()?;
assert!(norm.is_finite());

// Inner product: <tt|tt> = norm^2
let inner = tt.inner(&tt)?;
assert!((inner.real() - norm * norm).abs() < 1e-10);
# Ok(())
# }
```

## Complex Values

```rust
# fn main() -> anyhow::Result<()> {
use num_complex::Complex64;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_itensorlike::TensorTrain;

let site = DynIndex::new_dyn(2);
let tensor = TensorDynLen::from_dense(
    vec![site],
    vec![Complex64::new(1.0, 2.0), Complex64::new(0.0, -1.0)],
)?;
let tt = TensorTrain::new(vec![tensor])?;

let inner = tt.inner(&tt)?;
assert!((inner.real() - 6.0).abs() < 1e-12);
assert!(inner.imag().abs() < 1e-12);
# Ok(())
# }
```

## Documentation

- [User Guide: Tensor Train](https://tensor4all.org/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_itensorlike/)
