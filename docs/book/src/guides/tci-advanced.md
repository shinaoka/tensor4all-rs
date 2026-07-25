# TCI Advanced Topics

This guide corresponds to the Julia [Quantics TCI (advanced topics)](https://tensor4all.org/T4APlutoExamples/quantics1d_advanced.html) notebook.

## Direct `crossinterpolate2` Usage

This path bypasses the high-level quantics wrapper and works directly with quantics bits. Use `vec![2; R]` as `local_dims`, because each site is binary.

```rust
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};

let r = 8;
let local_dims = vec![2; r];
let x_max = 1.0_f64;
let step = x_max / (1usize << r) as f64;

let f = move |idx: &Vec<usize>| {
    let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
    let x = q as f64 * step;
    (-3.0 * x).exp()
};

let initial_pivots = vec![vec![0; r]];
let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    f,
    None,
    local_dims.clone(),
    initial_pivots,
    TCI2Options {
        tolerance: 1e-12,
        seed: Some(42),
        ..Default::default()
    },
).unwrap();

assert_eq!(result.termination, TCI2Termination::Converged);

let tt = result.tci.to_tensor_train().unwrap();
for bits in [
    vec![0, 0, 0, 0, 0, 0, 0, 0],
    vec![1, 0, 0, 0, 0, 0, 0, 0],
    vec![1, 1, 0, 0, 0, 0, 0, 0],
] {
    let q = bits.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
    let exact = (-3.0 * (q as f64 * step)).exp();
    let got = tt.evaluate(&bits).unwrap();
    assert!((got - exact).abs() < 1e-8);
}
```

## Initial Pivot Selection

Choose an initial pivot where `|f|` is large. That usually helps the first local solve. You can use `opt_first_pivot` for automatic local search, or pick candidates manually:

```rust
# let r = 8;
# let x_max = 1.0_f64;
# let step = x_max / (1usize << r) as f64;
# let f = move |idx: &Vec<usize>| {
#     let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
#     let x = q as f64 * step;
#     (-3.0 * x).exp()
# };
let candidates = vec![
    vec![0; r],
    vec![1; r],
    vec![0, 1, 0, 1, 0, 1, 0, 1],
    vec![1, 1, 1, 1, 1, 1, 1, 1],
];

let first_pivot = candidates
    .into_iter()
    .max_by(|a, b| {
        let va = f(a).abs();
        let vb = f(b).abs();
        va.partial_cmp(&vb).unwrap()
    })
    .unwrap();

// f(0,...,0) = exp(0) = 1.0 is the maximum of exp(-3x) on [0,1)
assert_eq!(first_pivot, vec![0; r]);
```

Alternatively, use `opt_first_pivot` to refine any starting guess:

```rust
use tensor4all_tensorci::opt_first_pivot;

let f = |idx: &Vec<usize>| (idx[0] as f64 + idx[1] as f64 + 1.0).powi(2);
let local_dims = vec![4, 4];
let start = vec![0, 0];

let pivot = opt_first_pivot::<f64, _>(&f, &local_dims, &start, 1000);
assert_eq!(pivot, vec![3, 3]); // f(3,3) = 49.0 is the maximum
```

## `CachedFunction`

Wrap expensive evaluations with `CachedFunction` to avoid redundant calls. The cache is shared across all TCI sweeps.

```rust
use tensor4all_tcicore::CachedFunction;
use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tensorci::{crossinterpolate2, TCI2Options};

let r = 8;
let local_dims = vec![2; r];
let x_max = 1.0_f64;
let step = x_max / (1usize << r) as f64;

let cf = CachedFunction::new(
    |idx: &[usize]| {
        let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
        let x = q as f64 * step;
        (-3.0 * x).exp()
    },
    &local_dims,
).unwrap();

let cached_f = |idx: &Vec<usize>| cf.eval(idx);
let tensor4all_tensorci::TCI2OptimizationResult { tci: tci_cached, ranks: _ranks, errors: _errors, .. } = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
    cached_f,
    None,
    local_dims.clone(),
    vec![vec![0; r]],
    TCI2Options {
        tolerance: 1e-12,
        seed: Some(42),
        ..Default::default()
    },
).unwrap();

assert!(cf.cache_size() > 0);
assert!(tci_cached.rank() >= 1);

// The cache avoids recomputing values seen in previous sweeps
assert!(cf.num_cache_hits() > 0);
```

### Performance guidance

- `CachedFunction` is most useful when the function evaluation is expensive
  (e.g., solving a differential equation for each index).
- For cheap functions (arithmetic, elementary functions), the caching overhead
  may not be worth it.
- The cache grows with the number of unique indices evaluated. For very
  high-dimensional problems, memory usage may become significant.

## Manual Integral

For a uniform half-open grid on `[x_min, x_max)`, the quantics tensor train sum becomes a Riemann integral after multiplying by the cell width:

`integral = tt.sum() * (x_max - x_min) / 2^R`

```rust
# use tensor4all_simplett::AbstractTensorTrain;
# use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
# let r = 8;
# let local_dims = vec![2; r];
# let x_max = 1.0_f64;
# let step = x_max / (1usize << r) as f64;
# let f = move |idx: &Vec<usize>| {
#     let q = idx.iter().fold(0usize, |acc, &bit| (acc << 1) | bit);
#     let x = q as f64 * step;
#     (-3.0 * x).exp()
# };
# let tensor4all_tensorci::TCI2OptimizationResult { tci, ranks: _ranks, errors: _errors, .. } = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
#     f, None, local_dims, vec![vec![0; r]],
#     TCI2Options { tolerance: 1e-12, seed: Some(42), ..Default::default() },
# ).unwrap();
let tt = tci.to_tensor_train().unwrap();
let integral = tt.sum() * (x_max - 0.0) / (1usize << r) as f64;

// For f(x) = exp(-3x) on [0, 1): integral = (1 - e^{-3}) / 3
let exact_integral = (1.0 - (-3.0_f64).exp()) / 3.0;
// R=8 gives 256 grid points; Riemann sum error is O(h) ~ 1/256
assert!((integral - exact_integral).abs() < 1e-2);
```
