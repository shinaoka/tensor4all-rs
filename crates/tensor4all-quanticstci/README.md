# tensor4all-quanticstci

High-level Quantics Tensor Train interpolation interface. Port of QuanticsTCI.jl.

## Key Types

- `QuanticsTensorCI2` — result of a quantics TCI run; supports `evaluate()`, `sum()`, `integral()`
- `DiscretizedGrid` — maps grid indices to physical coordinates for continuous domains
- `quanticscrossinterpolate()` — main entry point for continuous-domain interpolation
- `quanticscrossinterpolate_discrete()` — entry point for integer grids

## Example

```rust
use tensor4all_quanticstci::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
// Interpolate f(i, j) = i + j on a 16x16 discrete grid (1-indexed)
let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;

let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete(
    &[16, 16],   // grid sizes (must be equal powers of 2)
    f,
    None,        // auto-select initial pivot
    QtciOptions::default().with_tolerance(1e-10),
)?;

// Evaluate at a point (1-indexed)
let value = qtci.evaluate(&[5, 10])?;
assert!((value - 15.0).abs() < 1e-10);
assert!(errors.last().copied().unwrap() < 1e-10);

Ok(())
}
```

## Documentation

- [User Guide: TCI](https://tensor4all.org/tensor4all-rs/guides/tci.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_quanticstci/)
