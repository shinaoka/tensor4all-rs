# Quantics Transform

The `tensor4all-quanticstransform` crate provides `LinearOperator` constructors for
applying transformations to functions represented as quantics tensor trains.
It is a Rust port of the transformation functionality from
[Quantics.jl](https://github.com/tensor4all/Quantics.jl).

If you have not set up dependencies yet, add the transform and TreeTN crates to
your `Cargo.toml`. Use git dependencies from an external project:

```toml
[dependencies]
tensor4all-quanticstransform = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-quanticstransform" }
tensor4all-treetn = { git = "https://github.com/tensor4all/tensor4all-rs", package = "tensor4all-treetn" }
```

Or use path dependencies when working from a local checkout:

```toml
[dependencies]
tensor4all-quanticstransform = { path = "../tensor4all-rs/crates/tensor4all-quanticstransform" }
tensor4all-treetn = { path = "../tensor4all-rs/crates/tensor4all-treetn" }
```

---

## Operator Overview

Every constructor returns a `LinearOperator` from `tensor4all-treetn`, so all
operators are applied in the same way regardless of their mathematical meaning.

| Operator | Mathematical effect | Constructor |
|----------|---------------------|-------------|
| Flip | f(x) = g(2^R - x) | `flip_operator` |
| Shift | f(x) = g(x + offset) mod 2^R | `shift_operator` |
| Phase Rotation | f(x) = exp(i*theta*x) * g(x) | `phase_rotation_operator` |
| Cumulative Sum | y_i = sum of x_j for j < i | `cumsum_operator` |
| Fourier Transform | Quantics Fourier Transform (QFT) | `quantics_fourier_operator` |
| Affine Transform | y = A*x + b (rational coefficients) | `affine_operator` |

The parameter `r` that appears in every constructor is the number of quantics
bits (sites) per variable.  A single variable is discretized on `2^r` grid
points.

### Error Conditions

Constructors return `Err` for invalid inputs:

- `r == 0` -- no sites to operate on
- `r == 1` for `cumsum_operator`, `triangle_operator`, `quantics_fourier_operator` -- requires at least 2 sites
- `r >= 64` for `shift_operator` -- would overflow a 64-bit integer
- NaN or Inf `theta` for `phase_rotation_operator` -- invalid rotation angle

---

## Creating Operators

```rust
use tensor4all_quanticstransform::{
    flip_operator, shift_operator, phase_rotation_operator,
    cumsum_operator, quantics_fourier_operator,
    BoundaryCondition, FourierOptions,
};

// 8-bit quantics representation (2^8 = 256 grid points)
let r = 8;

// Flip: f(x) = g(2^R - x)
let flip_op = flip_operator(r, BoundaryCondition::Periodic).unwrap();
assert_eq!(flip_op.mpo().node_count(), r);

// Shift by 10: f(x) = g(x + 10) mod 2^R
let shift_op = shift_operator(r, 10, BoundaryCondition::Periodic).unwrap();
assert_eq!(shift_op.mpo().node_count(), r);

// Phase rotation: f(x) = exp(i*pi/4*x) * g(x)
let phase_op = phase_rotation_operator(r, std::f64::consts::PI / 4.0).unwrap();
assert_eq!(phase_op.mpo().node_count(), r);

// Cumulative sum
let cumsum_op = cumsum_operator(r).unwrap();
assert_eq!(cumsum_op.mpo().node_count(), r);

// Fourier transform (forward)
let ft_op = quantics_fourier_operator(r, FourierOptions::forward()).unwrap();
assert_eq!(ft_op.mpo().node_count(), r);
```

### Affine Transform

For transformations of the form **y = A*x + b** with rational coefficients:

```rust
use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
use num_rational::Rational64;

let r = 4;

// A = [[1, 1], [1, -1]], b = [0, 0]  (2 outputs, 2 inputs)
let a = vec![
    Rational64::from_integer(1), Rational64::from_integer(1),
    Rational64::from_integer(1), Rational64::from_integer(-1),
];
let b = vec![Rational64::from_integer(0), Rational64::from_integer(0)];
let params = AffineParams::new(a, b, 2, 2).unwrap();
let bc = vec![BoundaryCondition::Periodic; 2];
let affine_op = affine_operator(r, &params, &bc).unwrap();
assert_eq!(affine_op.mpo().node_count(), r);
```

---

## Applying Operators to a Tensor Train

### Index Mapping Flow

Operators define abstract *input* and *output* indices labeled `0, 1, ..., r-1`.
To apply an operator, bind those abstract indices to the concrete site indices
of the target state, then call `apply_linear_operator`. When the target
TreeTN's quantics sites are tagged as `x=1`, `x=2`, ..., use
`apply_linear_operator_to_numbered_tags`; otherwise use explicit index binding
with `apply_linear_operator_to_indices`.

```text
Original TreeTN          Operator            Result TreeTN
+----------------+    +----------------+    +----------------+
|  site_idx_0    |--->|  input_0       |    |  output_0      |
|  site_idx_1    |--->|  input_1       |--->|  output_1      |
|  site_idx_2    |--->|  input_2       |    |  output_2      |
|  other_idx     |    |                |    |  other_idx     |  (passes through)
+----------------+    +----------------+    +----------------+
```

`apply_linear_operator`:
1. Contracts the operator's tensors with the TreeTN.
2. Replaces input indices with output indices.
3. Leaves all unrelated indices unchanged.

### Apply Method Selection

`ApplyOptions` controls how the operator-state contraction is performed.
Four methods are available, each with different tradeoffs:

| Method | Algorithm | When to use |
|--------|-----------|-------------|
| `ApplyOptions::naive()` | Local exact operator-state apply with product links | Small exact/debug cases; bond dimensions may grow as products |
| `ApplyOptions::zipup()` | Single-sweep contraction with SVD truncation | Default choice; fast, good accuracy |
| `ApplyOptions::fit()` | Iterative variational optimization | Best deterministic compression; use when bond dim must be small |
| `ApplyOptions::src()` | Successive randomized compression with directed tree environments | Fast randomized compression; use a fixed seed for reproducible runs |

```rust
use tensor4all_core::SvdTruncationPolicy;
use tensor4all_treetn::{ApplyOptions, SrcOptions};

// Naive: local exact apply with no truncation or full dense materialization.
// Use for small exact/debug cases; output bond dimensions can grow as
// state/operator bond products.
let opts = ApplyOptions::naive();
assert_eq!(opts.max_bond_dim, None);

// ZipUp (default): single-pass, controllable truncation.
let opts = ApplyOptions::zipup()
    .with_max_bond_dim(64)
    .with_svd_policy(SvdTruncationPolicy::new(1e-10));
assert_eq!(opts.max_bond_dim, Some(64));

// Fit: iterative sweeps for best compression.
let opts = ApplyOptions::fit()
    .with_max_bond_dim(32)
    .with_nfullsweeps(3);
assert_eq!(opts.nfullsweeps, 3);

// SRC: randomized range finding with a deterministic probe stream.
let opts = ApplyOptions::src()
    .with_max_bond_dim(32)
    .with_src_options(SrcOptions::fixed().with_seed(7));
assert_eq!(opts.max_bond_dim, Some(32));
assert_eq!(opts.src_options.seed, 7);
```

SRC is implemented on the TreeTN contraction path and therefore also handles
non-chain operator/state topologies. Use `ApplyOptions::with_src_options` to
select adaptive tolerance control, rank increments, or the optional final SVD.

### Steiner Tree Partial Apply

When applying an operator to **a subset of sites** in a tensor network
(e.g., Fourier-transforming only the x-variable of a 2D function),
`apply_linear_operator` automatically handles the non-contiguous case.

If the operator's nodes are a subset of the state's nodes, the algorithm
constructs a **Steiner tree** -- the minimal subtree connecting all operator
nodes -- and inserts identity tensors at intermediate nodes that are not
covered by the operator. This means:

- You do not need to manually insert identity tensors.
- The operator can act on non-contiguous nodes (e.g., every other site in
  an interleaved encoding).
- Indices on nodes outside the Steiner tree pass through unchanged.

This feature is essential for multi-variable quantics, where variables are
interleaved: applying a 1D operator to variable x means acting on
sites `{0, 2, 4, ...}` while leaving `{1, 3, 5, ...}` (the y-variable)
untouched.

---

## Bit Ordering and Encoding

### Big-Endian Convention

All operators use **big-endian** bit ordering, matching Julia's Quantics.jl:

- Site 0 = Most Significant Bit (MSB)
- Site R-1 = Least Significant Bit (LSB)
- Integer value: x = sum over n of x_n * 2^(R-1-n)

For example, with R = 3 sites the value 5 = 101 in binary is stored as:

| Site | Bit | Contribution |
|------|-----|-------------|
| 0 | 1 | 2^2 = 4 |
| 1 | 0 | 0 |
| 2 | 1 | 2^0 = 1 |

### Multi-Variable Encoding

The `_multivar` variants (`flip_operator_multivar`, `shift_operator_multivar`,
`phase_rotation_operator_multivar`) use **interleaved bit encoding** for
multiple variables.  Each site simultaneously encodes one bit from each
variable:

```text
site index s_n encodes: bit_var0 + 2 * bit_var1 + 4 * bit_var2 + ...
```

Each site's local dimension is `2^num_vars`.  This interleaved layout is the
standard quantics multi-variable representation and is equivalent to
interleaving the bit planes of all variables.

---

## Boundary Conditions

Two boundary conditions are supported for operators that wrap indices (flip,
shift):

- **Periodic** -- results wrap modulo 2^R.
- **Open** -- indices that fall outside [0, 2^R) produce a zero vector.

```rust
use tensor4all_quanticstransform::{shift_operator, BoundaryCondition};

// Periodic: shift(7, 2) in 3-bit (mod 8) wraps to 1
let shift_periodic = shift_operator(3, 2, BoundaryCondition::Periodic).unwrap();
assert_eq!(shift_periodic.mpo().node_count(), 3);

// Open: shift(7, 2) in 3-bit goes to 9 >= 8, producing zero
let shift_open = shift_operator(3, 2, BoundaryCondition::Open).unwrap();
assert_eq!(shift_open.mpo().node_count(), 3);
```

---

## Fourier Transform Convention

`quantics_fourier_operator` produces output in **bit-reversed order**.  This is
inherent to the QFT algorithm: the output site ordering corresponds to the
bit-reversal of the frequency index.  If you need natural frequency ordering,
apply a bit-reversal permutation after the transform.

```rust
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};

let r = 4;

// Forward QFT (bit-reversed output)
let fwd = quantics_fourier_operator(r, FourierOptions::forward()).unwrap();
assert_eq!(fwd.mpo().node_count(), r);

// Inverse QFT
let inv = quantics_fourier_operator(r, FourierOptions::inverse()).unwrap();
assert_eq!(inv.mpo().node_count(), r);
```

---

## Reference

- [Quantics.jl](https://github.com/tensor4all/Quantics.jl) -- Julia implementation that this crate ports.
- J. Chen and M. Lindsey, "Direct Interpolative Construction of the Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182 (2024) -- QFT algorithm.
