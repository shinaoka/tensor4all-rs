# Tensor Cross Interpolation

Tensor Cross Interpolation (TCI) approximates a high-dimensional function as a tensor train by sampling only a small fraction of its entries. This crate provides two levels of API:

- `tensor4all-tensorci` for low-level TCI directly on integer indices.
- `tensor4all-quanticstci` for quantics interpolation on discrete or continuous grids.

## Low-Level TCI (`tensor4all-tensorci`)

Use `crossinterpolate2` when you already know the local dimensions and want direct control over the algorithm.

### Defining the function

The function must accept a `&Vec<usize>` of 0-indexed multi-indices and return a scalar value. Here is a simple 2D example where `f(i, j) = i + j + 1`:

```rust
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};

let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
let local_dims = vec![4, 4];
let initial_pivots = vec![vec![3, 3]]; // pick where |f| is large

let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims,
    initial_pivots,
    TCI2Options {
        tolerance: 1e-10,
        seed: Some(42),
        ..Default::default()
    },
).unwrap();

assert_eq!(result.termination, TCI2Termination::Converged);

let tt = result.tci.to_tensor_train().unwrap();
let value = tt.evaluate(&[2, 3]).unwrap();
assert!((value - 6.0).abs() < 1e-10);
assert!(tt.rank() >= 1);
```

### Choosing `TCI2Options`

The most important parameters:

| Parameter | Default | Guidance |
|---|---|---|
| `tolerance` | `1e-8` | Relative convergence threshold. Use `1e-6` for quick exploration, `1e-12` for high accuracy. |
| `max_bond_dim` | `usize::MAX` | Set to `50`--`500` for expensive functions to prevent runaway computation. |
| `max_iter` | `20` | Increase to `50`--`100` for difficult functions that need more sweeps. |
| `seed` | `None` | Set to `Some(42)` for reproducible results. |
| `normalize_error` | `true` | When `true`, `tolerance` is relative to max |f|. Set `false` for absolute tolerance. |

### Interpreting the results

`crossinterpolate2` returns a `TCI2OptimizationResult`:

| Field | Type | Description |
|-------|------|-------------|
| `tci` | `TensorCI2<T>` | Completed TCI object; call `.to_tensor_train()` to get a `TensorTrain`. |
| `ranks` | `Vec<usize>` | Bond dimensions after each sweep. |
| `errors` | `Vec<f64>` | Error estimate after each sweep. |
| `termination` | `TCI2Termination` | Whether the full criterion converged, the rank cap was reached, or iterations were exhausted. |

Do not infer convergence from `errors.last()` alone. Full convergence also
requires no recent global pivots and a stable rank history.

### Convergence diagnostics

The `errors` vector tracks the normalized bond error after each half-sweep. The algorithm converges when:

1. The last `ncheck_history` (default: 3) entries are all below `tolerance`.
2. No global pivots were added in those iterations.
3. The rank has stabilized.

If the errors plateau above your tolerance, try:
- Increasing `max_bond_dim` (the function may need higher rank).
- Increasing `max_iter` (more sweeps may be needed).
- Choosing better initial pivots (where `|f|` is large).

Convert to a tensor train for further manipulation:

```rust
# use tensor4all_simplett::AbstractTensorTrain;
# use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
# let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
# let tensor4all_tensorci::TCI2OptimizationResult { tci, ranks: _ranks, errors: _errors, .. } = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
#     f, None, vec![4, 4], vec![vec![3, 3]],
#     TCI2Options { seed: Some(42), ..Default::default() },
# ).unwrap();
let tt = tci.to_tensor_train().unwrap();
assert!(tt.rank() >= 1);

// Evaluate the tensor train at specific indices
let val = tt.evaluate(&[1, 2]).unwrap();
assert!((val - 4.0).abs() < 1e-10); // f(1,2) = 1+2+1 = 4
```

### Continuous vs discrete

`crossinterpolate2` works on discrete integer indices. For functions on continuous domains, use the quantics representation provided by `tensor4all-quanticstci` (see below), which maps floating-point coordinates to binary tensor-train indices.

## High-Level Quantics TCI (`tensor4all-quanticstci`)

The quantics representation encodes each grid index in binary and arranges the bits across tensor-train sites. This often yields much lower bond dimensions than a naive encoding.

### Important conventions

- Indexing differs between the two APIs:
  - `crossinterpolate2` (low-level): indices and pivots are 0-indexed (`0..local_dim`)
  - `quanticscrossinterpolate_discrete` (high-level): grid indices are 1-indexed (`1..=grid_size`), matching the Julia `QuanticsTCI.jl` convention
- Equal dimensions: `quanticscrossinterpolate_discrete` requires all dimensions to have the same number of points.
- Power-of-2 grid sizes: all grid dimensions must be powers of 2 (4, 8, 16, 32, ...).

### Choosing between discrete and continuous APIs

| Scenario | Function to use |
|---|---|
| Function on integer grid (lattice, combinatorial) | `quanticscrossinterpolate_discrete` |
| Function on continuous interval `[a, b)` | `quanticscrossinterpolate` with `DiscretizedGrid` |
| Grid points given as explicit coordinate arrays | `quanticscrossinterpolate_from_arrays` |
| Vector/tensor-valued function | `quanticscrossinterpolate_batched` |

**Tip on `nrandominitpivot`**: The `nrandominitpivot` option (default: 5) controls how many random initial pivot points are used to seed the TCI algorithm. For functions with multiple separated features or high-dimensional problems, increase this to 10--20 to improve robustness.

### Discrete grid interpolation

Use `quanticscrossinterpolate_discrete` when your function is naturally defined on an integer grid. Indices are passed as `&[i64]` and are 1-indexed.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
let sizes = vec![16, 16];

let (qtci, ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes,
    f,
    None,
    QtciOptions::default().with_tolerance(1e-10),
)?;

assert!(*errors.last().unwrap() < 1e-10);
assert!(!ranks.is_empty());

let value = qtci.evaluate(&[5, 10])?;
assert!((value - 15.0).abs() < 1e-8);

// Sum of (i + j) for i, j in 1..=16 = 2 * 16 * (16 * 17 / 2) = 4352
let total = qtci.sum()?;
assert!((total - 4352.0).abs() < 1e-6);
# Ok(())
# }
```

### Continuous grid interpolation with `DiscretizedGrid`

For functions on continuous domains, build a `DiscretizedGrid` that maps grid indices to physical coordinates. The number of quantics bits per dimension is set via the builder.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
};

// 2^4 = 16 grid points on [0, 1)
let grid = DiscretizedGrid::builder(&[4])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[1.0])
    .build()
    .unwrap();

let f = |x: &[f64]| x[0] * x[0];

let (qtci, _ranks, errors) = quanticscrossinterpolate::<f64, _>(
    &grid,
    f,
    None,
    QtciOptions::default(),
)?;

assert!(*errors.last().unwrap() < 1e-8);

// Verify the interpolation at grid point 1 (x = 0.0)
let value = qtci.evaluate(&[1])?;
assert!((value - 0.0).abs() < 1e-10);
# Ok(())
# }
```

### Integration

`integral()` computes a left Riemann sum:

```text
integral ≈ Σ f(xᵢ) × Δx
```

This has O(h) convergence where h is the grid spacing. The result depends on whether `include_endpoint` is set on the `DiscretizedGrid`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
# };
# let grid = DiscretizedGrid::builder(&[4])
#     .with_lower_bound(&[0.0])
#     .with_upper_bound(&[1.0])
#     .build()
#     .unwrap();
# let f = |x: &[f64]| x[0] * x[0];
# let (qtci, _, _) = quanticscrossinterpolate::<f64, _>(
#     &grid, f, None, QtciOptions::default(),
# )?;
let integral = qtci.integral()?;
// Left Riemann sum of x^2 over [0, 1) with 16 points
assert!((integral - 1.0 / 3.0).abs() < 5e-2);

let sum = qtci.sum()?;
assert!((sum * grid.grid_step()[0] - integral).abs() < 1e-12);
# Ok(())
# }
```

For discrete grids created without a continuous domain, `integral()` returns the plain sum identical to `sum()`.

## Practical Example: Multi-scale 1D Function

This section corresponds to the Julia [Quantics TCI of univariate function](https://tensor4all.org/T4APlutoExamples/quantics1d.html) notebook.

The function below mixes several length scales:

`f(x) = cos(x/B) * cos(x/(4*sqrt(5)*B)) * exp(-x^2) + 2*exp(-x)`

with `B = 2^-30` on `[0, ln(20))` using `R = 40` quantics bits.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstci::{quanticscrossinterpolate, DiscretizedGrid, QtciOptions};

// Use R = 10 bits (2^10 = 1024 points) for a fast doctest.
// In practice, set R = 40 for the full multi-scale resolution.
let r = 10;
let x_max = 20.0_f64.ln();
let grid = DiscretizedGrid::builder(&[r])
    .with_lower_bound(&[0.0])
    .with_upper_bound(&[x_max])
    .include_endpoint(false)
    .build()
    .unwrap();

// A smooth multi-scale function: f(x) = cos(10x) * exp(-x^2) + 2*exp(-x)
let f = move |coords: &[f64]| {
    let x = coords[0];
    (10.0 * x).cos() * (-x * x).exp() + 2.0 * (-x).exp()
};

let tol = 1e-8;
let (qtci, _ranks, errors) = quanticscrossinterpolate::<f64, _>(
    &grid,
    f,
    None,
    QtciOptions::default()
        .with_tolerance(tol)
        .with_maxbonddim(64)
        .with_nrandominitpivot(8),
)?;

assert!(*errors.last().unwrap() < tol);

// Verify at several grid points across the domain
for &grid_idx in &[1_i64, 100, 512, 1024] {
    let got = qtci.evaluate(&[grid_idx])?;
    assert!(got.is_finite(), "value at grid index {} should be finite", grid_idx);
}

// Check the integral is reasonable (positive, since f > 0 near x = 0)
let integral = qtci.integral()?;
assert!(integral > 1.0);
# Ok(())
# }
```

## Multivariate (2D) Example

This section corresponds to the Julia [Quantics TCI of multivariate function](https://tensor4all.org/T4APlutoExamples/quantics2d.html) notebook.

```rust
# fn main() -> anyhow::Result<()> {
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};

// Use 64x64 for a faster doctest (256x256 in practice)
let sizes = vec![64, 64];
let f = |idx: &[i64]| {
    let x = idx[0] as f64;
    let y = idx[1] as f64;
    (x / 24.0).cos() + (y / 17.0).cos() + 0.1 * ((x + y) / 13.0).sin()
};

let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes,
    f,
    None,
    QtciOptions::default()
        .with_tolerance(1e-10)
        .with_maxbonddim(64)
        .with_nrandominitpivot(8),
)?;

assert!(*errors.last().unwrap() < 1e-8);

for &(i, j) in &[(1_i64, 1_i64), (17, 33), (32, 50), (64, 64)] {
    let exact = (i as f64 / 24.0).cos()
        + (j as f64 / 17.0).cos()
        + 0.1 * (((i + j) as f64) / 13.0).sin();
    let got = qtci.evaluate(&[i, j])?;
    assert!((got - exact).abs() < 1e-6);
}
# Ok(())
# }
```
