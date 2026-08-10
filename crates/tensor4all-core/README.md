# tensor4all-core

Core tensor library: Index system, dynamic-rank Tensor, contraction, SVD/QR/LU factorization.

## Key Types

- `Index` — flexible index with tags and prime levels
- `TensorDynLen` — dynamic-rank tensor with flexible index types
- `Storage` — compact dense/diagonal/structured snapshots for `f64` and
  `Complex64`; `f32` and `Complex32` tensors retain eager authoritative payloads
- `contract()` / `contract_with_options()` — connected tensor-network contraction
- `svd()` / `qr()` — factorizations with truncation support

## Example

```rust
use tensor4all_core::{
    factorize, DynIndex, FactorizeOptions, TensorContractionLike, TensorDynLen,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create indices
let i = DynIndex::new_dyn(3);
let j = DynIndex::new_dyn(4);

// Create a dense tensor
let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], data)?;
assert_eq!(tensor.dims(), vec![3, 4]);

// SVD factorization
let result = factorize(&tensor, &[i], &FactorizeOptions::svd())?;
let recovered = result.left.contract_pair(&result.right)?;
assert!(tensor.distance(&recovered)? < 1e-12);
assert!(result.singular_values.is_some());

Ok(())
}
```

## Documentation

- [User Guide: Tensor Basics](https://tensor4all.org/tensor4all-rs/guides/tensor-basics.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_core/)
