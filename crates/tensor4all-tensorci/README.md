# tensor4all-tensorci

Tensor Cross Interpolation algorithms. TCI2 is the primary maintained implementation; TCI1 is available for legacy one-site workflows and parity with TensorCrossInterpolation.jl.

## Key Types

- `crossinterpolate2()` — main entry point for two-site TCI
- `TCI2Options` — tolerance, pivot strategy, and convergence settings
- `TCI2OptimizationResult` and `TCI2Termination` — sweep histories and explicit stop reason
- `TensorCI2` — two-site TCI state; convert it to `TensorTrain` for repeated evaluation
- `TensorCI2::from_tensor_train()` and `TensorCI2::from_index_sets()` — non-dense conversion constructors
- `crossinterpolate1()` — legacy one-site TCI entry point
- `TCI1Options` and `TCI1SweepStrategy` — tolerance, sweep direction, and pivot settings for TCI1
- `TensorCI1` — one-site TCI state; convert it to `TensorTrain` for repeated evaluation

## Example

```rust
use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};
use tensor4all_simplett::AbstractTensorTrain;

// Function to interpolate: f(i, j) = (i + j + 1) as f64
let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;

// Run TCI on a 4x4 grid
let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    vec![4, 4],         // local dimensions
    vec![vec![3, 3]],   // initial pivot (0-indexed); pick where f is large
    TCI2Options {
        tolerance: 1e-10,
        ..Default::default()
    },
).unwrap();

// Verify full convergence, including stable ranks and no recent global pivots
assert_eq!(result.termination, TCI2Termination::Converged);

// Verify interpolation accuracy
let tt = result.tci.to_tensor_train().unwrap();
let val = tt.evaluate(&[2, 3]).unwrap();  // f(2, 3) = 2 + 3 + 1 = 6.0
assert!((val - 6.0).abs() < 1e-10);
```

## Documentation

- [User Guide: TCI](https://tensor4all.org/tensor4all-rs/guides/tci.html)
- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_tensorci/)
