# Quantum Fourier Transform

This guide demonstrates how to apply the quantics Fourier transform (QFT)
operator to tensor trains. It corresponds to the Julia
[Quantum Fourier Transform](https://tensor4all.org/T4APlutoExamples/qft.html) notebook.

## Background

The QFT operator converts a function from position space to frequency space
(and vice versa) using a matrix product operator (MPO) construction due to
Chen and Lindsey (arXiv:2404.03182). In quantics representation, the DFT
of a function on 2^R grid points is expressed as a compact tensor train with
small bond dimension.

Key properties:
- The output is in **bit-reversed** frequency order.
- Forward transform uses sign = -1 in the exponent; inverse uses +1.
- When `normalize = true` (default), the transform is an isometry.

## Simple QFT Example

Before working with TCI-constructed states, here is a minimal example that
creates a QFT operator and verifies its structure.

```rust
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions, FTCore};

let r = 4;

// Create forward and inverse QFT operators
let ft = FTCore::new(r, FourierOptions::default()).unwrap();
let fwd = ft.forward().unwrap();
let bwd = ft.backward().unwrap();

// Each operator has r sites
assert_eq!(fwd.mpo().node_count(), r);
assert_eq!(bwd.mpo().node_count(), r);

// Both operators have input and output mappings for all sites
for i in 0..r {
    assert!(fwd.get_input_mapping(&i).is_some());
    assert!(fwd.get_output_mapping(&i).is_some());
    assert!(bwd.get_input_mapping(&i).is_some());
    assert!(bwd.get_output_mapping(&i).is_some());
}
```

## 1D QFT Application

This example applies the QFT to a product state |0> (the uniform function)
and verifies that the result has uniform magnitude 1/sqrt(N) at all
frequency components -- a well-known property of the DFT.

```rust
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_core::TensorIndex;
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, SimpleTensorTrain};
use tensor4all_treetn::{apply_linear_operator, ApplyOptions, tensor_train_to_treetn};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};

let r = 3;
let n = 1usize << r; // 8

// Create the |0> product state as a SimpleTensorTrain
// |0> = |0> x |0> x ... x |0>  (all bits zero)
let mut tensors = Vec::new();
for _ in 0..r {
    let mut t = tensor3_zeros(1, 2, 1);
    t.set3(0, 0, 0, Complex64::one()); // bit = 0
    tensors.push(t);
}
let mps = SimpleTensorTrain::new(tensors).unwrap();

// Convert MPS to TreeTN
let (treetn, site_indices) = tensor_train_to_treetn(&mps).unwrap();

// Create the forward QFT operator
let qft_op = quantics_fourier_operator(r, FourierOptions::forward()).unwrap();

// Replace TreeTN site indices with operator input indices
let mut state = treetn;
for i in 0..r {
    let op_input = qft_op
        .get_input_mapping(&i)
        .unwrap()
        .true_index
        .clone();
    state = state.replaceind(&site_indices[i], &op_input).unwrap();
}

// Apply the QFT with local exact naive apply. This can grow bond dimensions as
// state/operator products, so use it for small exact/debug cases.
let result = apply_linear_operator(&qft_op, &state, ApplyOptions::naive()).unwrap();

// The result should exist and have the same number of nodes
assert_eq!(result.node_count(), r);

// This example is intentionally small, so dense verification is acceptable.
// For production-size networks, prefer scalable residual norms or sampled
// `evaluate()` checks rather than `contract_to_tensor()`.
let dense = result.contract_to_tensor().unwrap();
let data = dense.to_vec::<Complex64>().unwrap();

// For |0> input, all Fourier coefficients should have magnitude 1/sqrt(N)
let expected_mag = 1.0 / (n as f64).sqrt();
for val in &data {
    let mag = val.norm();
    assert!((mag - expected_mag).abs() < 1e-6,
        "Expected magnitude {}, got {}", expected_mag, mag);
}
```

### Interpreting QFT Output

The QFT output represents the discrete Fourier transform of the input function.
For a function f(x) on N = 2^R points, the k-th Fourier coefficient is:

```text
F(k) = (1/sqrt(N)) * sum_{x=0}^{N-1} f(x) * exp(-2*pi*i*k*x/N)
```

The output is in **bit-reversed** frequency order: the output at site
configuration (b_0, b_1, ..., b_{R-1}) corresponds to frequency index
k = b_{R-1} * 2^{R-1} + ... + b_1 * 2 + b_0 (i.e., the bit-reversal of
b_0 * 2^{R-1} + ... + b_{R-1}).

## 2D QFT via Partial Apply

A dedicated multivar Fourier API is not yet available. This example
shows how to use partial apply to perform a 2D transform by applying a 1D
Fourier operator to non-contiguous sites.

For a 2D function in interleaved quantics encoding with R bits per variable,
the TreeTN has 2R sites: `[x_1, y_1, x_2, y_2, ..., x_R, y_R]` (node names
`0, 1, 2, 3, ..., 2R-1`).

### Approach

To apply a 1D Fourier transform to the x-variable:

1. Build the 1D Fourier operator (R sites, node names 0..R-1)
2. Rename operator nodes to match x-variable sites: 0 -> 0, 1 -> 2, 2 -> 4, ...
3. Set input/output mappings from the state
4. Call `apply_linear_operator` -- Steiner tree partial apply handles the gaps

The Steiner tree mechanism automatically inserts identity tensors at the
y-variable sites (1, 3, 5) so the operator acts only on the x-variable.
The same approach works for applying the Fourier transform to the y-variable
(rename operator nodes to 1, 3, 5 instead).

```rust
use std::f64::consts::PI;
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetn::operator::{apply_linear_operator, ApplyOptions};
use tensor4all_treetn::Operator;

let r = 3;
let n = 1usize << r; // 8

// f(x, y) = cos(2*pi*(x-1)/N), 1-indexed -- depends only on x
let f = move |idx: &[i64]| -> f64 {
    let x = (idx[0] - 1) as f64;
    (2.0 * PI * x / n as f64).cos()
};

// Build QTT with interleaved encoding
let sizes = vec![n, n];
let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
    &sizes, f, None,
    QtciOptions::default()
        .with_tolerance(1e-12)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved),
).unwrap();
assert!(*errors.last().unwrap() < 1e-10);

// Convert to TreeTN (6 sites: 0,1,2,3,4,5)
let tci_state = qtci.tci();
let batch_eval = move |batch: tensor4all_treetci::GlobalIndexBatch<'_>|
    -> anyhow::Result<Vec<f64>>
{
    let mut values = Vec::with_capacity(batch.n_points());
    for p in 0..batch.n_points() {
        let mut x_val = 0usize;
        for bit in 0..r {
            x_val |= batch.get(2 * bit, p).unwrap() << bit;
        }
        values.push((2.0 * PI * x_val as f64 / n as f64).cos());
    }
    Ok(values)
};
let state = to_treetn(tci_state, batch_eval, Some(0)).unwrap();

// Build 1D Fourier operator (nodes 0,1,2)
let fourier_op = quantics_fourier_operator(
    r, FourierOptions { normalize: true, ..Default::default() },
).unwrap();

// Rename nodes to x-variable sites: 0->0, 1->2, 2->4
let x_site_mapping: Vec<_> = (0..r).map(|i| (i, 2 * i)).collect();
let mut fourier_op = fourier_op.rename_nodes(&x_site_mapping).unwrap();

// Match operator's true indices to state's site indices
fourier_op.set_input_space_from_state(&state).unwrap();
fourier_op.set_output_space_from_state(&state).unwrap();

// Apply -- Steiner tree inserts identity at y-sites {1, 3, 5}
let result = apply_linear_operator(
    &fourier_op, &state, ApplyOptions::default(),
).unwrap();
assert_eq!(result.node_count(), 2 * r);
```
