# tensor4all-interpolativeqtt

Interpolative Quantics Tensor Train construction for tensor4all-rs.

This crate ports the tested public algorithms from
`InterpolativeQTT.jl` to Rust and returns `tensor4all-simplett`
`SimpleTensorTrain<f64>` values.

```rust
use tensor4all_interpolativeqtt::{
    interpolate_single_scale, AbstractTensorTrain, InterpolativeQttOptions,
};

let tt = interpolate_single_scale(
    |x| (-x * x).exp(),
    -2.0,
    2.0,
    5,
    12,
    &InterpolativeQttOptions::default(),
).unwrap();

let value = tt.evaluate(&[0, 0, 0, 0, 0]).unwrap();
let expected = (-4.0_f64).exp();
assert!((value - expected).abs() < 1e-10);
```
