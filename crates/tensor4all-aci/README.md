# tensor4all-aci

Alternating Cross Interpolation (ACI) elementwise operations for tensor trains.

This crate ports the public Rust API for
[AlternatingCrossInterpolation.jl](https://github.com/tensor4all/AlternatingCrossInterpolation.jl),
originally authored by Marc Ritter and contributors.

## Key Types

- `elementwise()` - approximate an elementwise operation on tensor trains
- `elementwise_batched()` - batch-evaluation entry point for amortized function calls
- `ElementwiseBatch` - column-major batch of tensor-train input values
- `AciOptions` - tolerance, sweep, rank, and pivot-search controls
- `AciResult` - output tensor train and convergence diagnostics

## Examples

### Julia-style elementwise multiplication

`elementwise()` is the Rust counterpart of Julia
`AlternatingCrossInterpolation.elementwise`. The operator receives one slice of
input values per interpolation point.

```rust
use tensor4all_aci::prelude::*;

let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);

let result = elementwise(
    |xs: &[f64]| xs[0] * xs[1],
    &[a, b],
    &AciOptions::default(),
)
.unwrap();

assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 8.0).abs() < 1e-10);
```

### Batched operator callback

`elementwise_batched()` is a Rust extension for amortizing operator calls. The
batch is a borrowed column-major view with `n_inputs` rows and `n_points`
columns. Use `batch.get(input, point)` for checked access.

```rust
use tensor4all_aci::prelude::*;

let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);

let result = elementwise_batched(
    |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        for point in 0..batch.n_points() {
            let a = batch.get(0, point)?;
            let b = batch.get(1, point)?;
            output[point] = a * b + 1.0;
        }
        Ok(())
    },
    &[a, b],
    &AciOptions::default(),
)
.unwrap();

assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 9.0).abs() < 1e-10);
```

For hot callbacks, the flat batch slice can be read directly. Values are stored
as `data[input + n_inputs * point]`.

```rust
use tensor4all_aci::prelude::*;

let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);

let result = elementwise_batched(
    |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
        let data = batch.as_col_major_slice();
        let n_inputs = batch.n_inputs();

        for point in 0..batch.n_points() {
            let base = n_inputs * point;
            output[point] = data[base] * data[base + 1];
        }
        Ok(())
    },
    &[a, b],
    &AciOptions::default(),
)
.unwrap();

assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
assert!((result.tensor_train.evaluate(&[0, 1]).unwrap() - 8.0).abs() < 1e-10);
```

`ElementwiseBatch` is not `tensor4all_simplett::TTCache`. ACI maintains
left/right frames internally and uses `ElementwiseBatch` only to pass local
matrix entries to the user operator efficiently.

## Citation

If you use ACI in research, please cite the original ACI paper:

> M. K. Ritter, "Fast elementwise operations on tensor trains with alternating
> cross interpolation", arXiv:2604.00037 (2026).

## Documentation

- [API Reference](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_aci/)
