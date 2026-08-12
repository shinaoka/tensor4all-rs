# Affine Transformation

An affine transformation evaluates a function at shifted or mixed coordinates.
The tutorial uses the pullback point of view: to get the new value at an output
point, look up the old function at the transformed input point.

Runnable source: [`docs/tutorial-code/src/bin/qtt_affine.rs`](../../../../tutorial-code/src/bin/qtt_affine.rs)

## Key API Pieces

`AffineParams` stores the matrix and offset. Boundary conditions say what
happens when transformed coordinates leave the grid. The operator is
applied via `tensor_train_to_treetn`, `align_to_state`, and
`apply_linear_operator`. The source function is
`f(u, v) = sin(2πu/N) + 0.5 cos(2πv/N) + 0.25 sin(2π(u + 2v)/N)`.

```rust
# fn main() -> anyhow::Result<()> {
# use tensor4all_core::TensorIndex;
# use tensor4all_quanticstci::{
#     quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
# };
# use tensor4all_quanticstransform::{
#     affine_operator, AffineParams, BoundaryCondition,
# };
# use tensor4all_treetn::{apply_linear_operator, tensor_train_to_treetn, ApplyOptions};
use std::f64::consts::PI;

let bits = 7;
let n = 1usize << bits;
let source_grid = [n, n];
let source_function = move |grid_idx: &[usize]| -> f64 {
    let u = grid_idx[0] as f64;
    let v = grid_idx[1] as f64;
    let n = n as f64;
    (2.0 * PI * u / n).sin()
        + 0.5 * (2.0 * PI * v / n).cos()
        + 0.25 * (2.0 * PI * (u + 2.0 * v) / n).sin()
};
let source_options = QtciOptions::default()
    .with_unfoldingscheme(UnfoldingScheme::Fused)
    .with_verbosity(0);

let (source, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    &source_grid,
    source_function,
    None,
    source_options,
)?;

let params = AffineParams::from_integers(vec![1, 1, 0, 1], vec![0, 0], 2, 2)?;
let mut operator = affine_operator(bits, &params, &[BoundaryCondition::Periodic; 2])?
    .transpose();

let source_tt = source.tensor_train();
let (state, _indices) = tensor_train_to_treetn(&source_tt)?;

operator.align_to_state(&state)?;
let result = apply_linear_operator(&operator, &state, ApplyOptions::naive())?;

let external = TensorIndex::external_indices(&result);
assert_eq!(external.len(), bits);
assert!(result.node_count() >= state.node_count());
# Ok(())
# }
```

The full tutorial repeats the same workflow for three boundary conditions. The
anti-periodic case uses `[AntiPeriodic, Periodic]`, so only the wrapped
`u = x + y` coordinate changes sign.

## What It Computes

The example builds a two-dimensional QTT, creates affine operators, applies
them to the QTT, and compares the transformed values with direct references.
The anti-periodic pullback matches the periodic wrap except that the region
`x + y >= N` is multiplied by `-1`.

![Affine transformed values](qtt_affine_values.png)

![Affine transformation error](qtt_affine_error.png)

The operator has its own bond dimensions, and the transformed QTT has another
set. Both are useful when judging the cost of the operation.

![Bond dimensions after the affine transformation](qtt_affine_bond_dims.png)

![Bond dimensions of the affine operator](qtt_affine_operator_bond_dims.png)
