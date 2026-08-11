# tensor4all-simplett

Simple, efficient Tensor Train (MPS) implementation for numerical computation.

## Key Types

- `SimpleTensorTrain` — basic MPS with `f64` and `Complex64` support
- `SiteTensorTrain` — center-canonical MPS with specified orthogonality center
- `VidalTensorTrain` — Vidal canonical form with explicit singular values
- `CompressionOptions` — controls tolerance and maximum bond dimension for compression

## Example

```rust
use tensor4all_simplett::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
// Create a constant tensor train (all entries = 1.0) over a 2x3x4 grid
let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 4], 1.0);

// Evaluate at a specific multi-index
let value = tt.evaluate(&[0, 1, 2])?;
assert!((value - 1.0).abs() < 1e-15);

// Sum over all indices: 2 * 3 * 4 = 24
let total = tt.sum();
assert!((total - 24.0).abs() < 1e-10);

// Compress with tolerance
let options = CompressionOptions {
    tolerance: 1e-10,
    max_bond_dim: Some(20),
    ..Default::default()
};
let compressed = tt.compressed(&options)?;
assert!(compressed.rank() <= tt.rank());

Ok(())
}
```

## Documentation

- [User Guide: Tensor Train](https://tensor4all.org/tensor4all-rs/guides/tensor-train.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_simplett/)
