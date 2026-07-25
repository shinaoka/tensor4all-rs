# Compressing Existing Data

This guide corresponds to the Julia [Compressing existing data](https://tensor4all.org/T4APlutoExamples/compress.html) notebook.

## Problem

The goal is to compress a large 3D array without materializing the full tensor in memory. Instead of allocating `128 x 128 x 128` values up front, define a function that computes the element on demand and let TCI discover the low-rank structure.

**When compression works well:** Functions with low-rank structure (sums of
separable terms, smooth functions, oscillatory functions). The example below
-- `cos(x) + cos(y) + cos(z)` -- is a sum of three single-variable functions
and has exact TT rank 2.

**Tolerance selection:** Start with `1e-12` (near machine precision). If the
resulting bond dimensions are too large for your application, relax to `1e-8`
or `1e-6`.

## TCI Compression

```rust
# fn main() -> anyhow::Result<()> {
use std::f64::consts::PI;
use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};
use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};

let local_dims = vec![128, 128, 128];
let f = |idx: &Vec<usize>| {
    let x = 2.0 * PI * idx[0] as f64 / 128.0;
    let y = 2.0 * PI * idx[1] as f64 / 128.0;
    let z = 2.0 * PI * idx[2] as f64 / 128.0;
    x.cos() + y.cos() + z.cos()
};

let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims.clone(),
    vec![vec![0, 0, 0]],
    TCI2Options {
        tolerance: 1e-12,
        max_bond_dim: 64,
        ..Default::default()
    },
)?;

assert_eq!(result.termination, TCI2Termination::Converged);

let tt = result.tci.to_tensor_train()?;
for point in [
    vec![0, 0, 0],
    vec![1, 2, 3],
    vec![17, 33, 65],
    vec![127, 127, 127],
] {
    let x = 2.0 * PI * point[0] as f64 / 128.0;
    let y = 2.0 * PI * point[1] as f64 / 128.0;
    let z = 2.0 * PI * point[2] as f64 / 128.0;
    let exact = x.cos() + y.cos() + z.cos();
    let got = tt.evaluate(&point)?;
    assert!((got - exact).abs() < 1e-8);
}
# Ok(())
# }
```

## Compression Quality

The quality of the compression is visible in the bond dimensions and the parameter count of the tensor train.

```rust
# fn main() -> anyhow::Result<()> {
# use std::f64::consts::PI;
# use tensor4all_simplett::{AbstractTensorTrain, Tensor3Ops};
# use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
# let local_dims = vec![128, 128, 128];
# let f = |idx: &Vec<usize>| {
#     let x = 2.0 * PI * idx[0] as f64 / 128.0;
#     let y = 2.0 * PI * idx[1] as f64 / 128.0;
#     let z = 2.0 * PI * idx[2] as f64 / 128.0;
#     x.cos() + y.cos() + z.cos()
# };
# let tensor4all_tensorci::TCI2OptimizationResult { tci, .. } = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
#     f, None, local_dims.clone(), vec![vec![0, 0, 0]],
#     TCI2Options { tolerance: 1e-12, max_bond_dim: 64, ..Default::default() },
# )?;
# let tt = tci.to_tensor_train()?;
let bond_dims = tt.link_dims();
assert!(!bond_dims.is_empty());

let full_size = local_dims.iter().product::<usize>();
let compressed_params: usize = (0..tt.len())
    .map(|i| {
        let tensor = tt.site_tensor(i);
        tensor.left_dim() * tensor.site_dim() * tensor.right_dim()
    })
    .sum();

let compression_ratio = full_size as f64 / compressed_params as f64;

assert!(compression_ratio > 10.0);
assert!(bond_dims.iter().copied().max().unwrap_or(0) <= 64);
# Ok(())
# }
```
