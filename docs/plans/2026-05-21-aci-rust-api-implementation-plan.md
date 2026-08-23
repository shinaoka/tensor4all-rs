# ACI Rust API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Rust public API for alternating cross interpolation elementwise operations on `tensor4all-simplett::TensorTrain`.

**Architecture:** Implement a new `tensor4all-aci` crate that depends on `tensor4all-simplett`, `tensor4all-tcicore`, and `tensor4all-tensorbackend`. The public API exposes scalar and batched elementwise entry points, while the internal algorithm uses persistent ACI frames, one-bond local entry caches, and lazy MatrixLUCI block requests.

**Tech Stack:** Rust 2021, `thiserror`, `rand`, `rand_chacha`, `rand_distr`, `criterion`, `tensor4all-simplett`, `tensor4all-tcicore::matrix_luci_factors_from_blocks`, `tensor4all-tensorbackend::Matrix`, Julia 1.10+, `BenchmarkTools.jl`, `AlternatingCrossInterpolation.jl`.

---

## References

- Design: `docs/plans/2026-05-21-aci-rust-api-design.md`
- Upstream Julia package: `/tmp/AlternatingCrossInterpolation.jl` if still present, or clone `https://github.com/tensor4all/AlternatingCrossInterpolation.jl`
- Existing APIs:
  - `docs/api/tensor4all_simplett.md`
  - `docs/api/tensor4all_tcicore.md`
  - `docs/api/tensor4all_tensorbackend.md`
- Source files likely needed for implementation:
  - `crates/tensor4all-simplett/src/types.rs`
  - `crates/tensor4all-simplett/src/tensortrain.rs`
  - `crates/tensor4all-simplett/src/traits.rs`
  - `crates/tensor4all-tcicore/src/matrix_luci.rs`
  - `crates/tensor4all-tensorbackend/src/matrix.rs` or the current `Matrix` module path
- Benchmark references:
  - `benchmarks/README.md`
  - `benchmarks/julia/Project.toml`
  - Existing Rust/Julia paired benchmark scripts under `benchmarks/rust/` and `benchmarks/julia/`

## Task 1: Scaffold `tensor4all-aci`

**Files:**
- Modify: `Cargo.toml`
- Create: `crates/tensor4all-aci/Cargo.toml`
- Create: `crates/tensor4all-aci/src/lib.rs`
- Create: `crates/tensor4all-aci/NOTICE`

**Step 1: Write the minimal crate files**

Add `crates/tensor4all-aci` to workspace members in root `Cargo.toml`.

Create `crates/tensor4all-aci/Cargo.toml`:

```toml
[package]
name = "tensor4all-aci"
description = "Alternating Cross Interpolation elementwise operations for tensor4all tensor trains"
version.workspace = true
edition.workspace = true
authors = [
    "Marc Ritter <mritter@flatironinstitute.org>",
    "Hiroshi Shinaoka <h.shinaoka@gmail.com>",
    "tensor4all contributors",
]
license.workspace = true
repository.workspace = true

[features]
default = ["tenferro-cpu-faer"]
tenferro-cpu-faer = [
    "tensor4all-simplett/tenferro-cpu-faer",
    "tensor4all-tcicore/tenferro-cpu-faer",
    "tensor4all-tensorbackend/tenferro-cpu-faer",
]
tenferro-provider-inject = [
    "tensor4all-simplett/tenferro-provider-inject",
    "tensor4all-tcicore/tenferro-provider-inject",
    "tensor4all-tensorbackend/tenferro-provider-inject",
]

[dependencies]
num-complex.workspace = true
rand.workspace = true
rand_chacha.workspace = true
rand_distr.workspace = true
thiserror.workspace = true
tensor4all-simplett = { path = "../tensor4all-simplett", default-features = false }
tensor4all-tcicore = { path = "../tensor4all-tcicore", default-features = false }
tensor4all-tensorbackend = { path = "../tensor4all-tensorbackend", default-features = false }

[dev-dependencies]
approx.workspace = true
```

Create `src/lib.rs`:

```rust
#![warn(missing_docs)]
//! Alternating Cross Interpolation for elementwise tensor-train operations.
//!
//! This crate ports the Rust public API for
//! `AlternatingCrossInterpolation.jl`, originally authored by
//! Marc Ritter <mritter@flatironinstitute.org> and contributors.

/// Placeholder item until the public API is implemented.
pub fn crate_ready() -> bool {
    true
}
```

Create `NOTICE`:

```text
This crate ports AlternatingCrossInterpolation.jl, originally authored by
Marc Ritter <mritter@flatironinstitute.org> and contributors.

Upstream copyright:

Copyright (c) 2026 Marc Ritter <mritter@flatironinstitute.org> and contributors
```

**Step 2: Run the crate check**

Run:

```bash
cargo check -p tensor4all-aci
```

Expected: PASS.

**Step 3: Commit**

```bash
git add Cargo.toml crates/tensor4all-aci
git commit -m "Add tensor4all-aci crate scaffold"
```

## Task 2: Add Public Error, Options, Result, and Batch Types

**Files:**
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Create: `crates/tensor4all-aci/src/error.rs`
- Create: `crates/tensor4all-aci/src/options.rs`
- Create: `crates/tensor4all-aci/src/result.rs`
- Create: `crates/tensor4all-aci/src/batch.rs`
- Create: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write tests for public type behavior**

In `src/tests.rs`, add tests for default options and column-major batch access:

```rust
use crate::{AciOptions, ElementwiseBatch};

#[test]
fn default_options_are_conservative() {
    let options = AciOptions::<f64>::default();
    assert_eq!(options.max_iters, 20);
    assert_eq!(options.min_iters, 2);
    assert_eq!(options.max_bond_dim, usize::MAX);
    assert!((options.tolerance - 1e-12).abs() < 1e-15);
    assert!(!options.scale_tolerance);
    assert!(options.initial_guess.is_none());
}

#[test]
fn elementwise_batch_uses_column_major_input_point_layout() {
    let values = vec![10.0, 20.0, 11.0, 21.0, 12.0, 22.0];
    let batch = ElementwiseBatch::new(&values, 2, 3).unwrap();
    assert_eq!(batch.n_inputs(), 2);
    assert_eq!(batch.n_points(), 3);
    assert_eq!(batch.get(0, 0).unwrap(), 10.0);
    assert_eq!(batch.get(1, 0).unwrap(), 20.0);
    assert_eq!(batch.get(0, 2).unwrap(), 12.0);
    assert_eq!(batch.get(1, 2).unwrap(), 22.0);
    assert_eq!(batch.as_col_major_slice(), values.as_slice());
}

#[test]
fn elementwise_batch_rejects_bad_length() {
    let err = ElementwiseBatch::<f64>::new(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
    assert!(err.to_string().contains("length"));
}
```

**Step 2: Verify tests fail**

Run:

```bash
cargo test --release -p tensor4all-aci default_options_are_conservative
cargo test --release -p tensor4all-aci elementwise_batch
```

Expected: FAIL because the types do not exist.

**Step 3: Implement public types**

`src/error.rs`:

```rust
use thiserror::Error;

/// Result type for ACI operations.
pub type Result<T> = std::result::Result<T, AciError>;

/// Errors returned by the ACI Rust public API.
#[derive(Debug, Error)]
pub enum AciError {
    /// No input tensor trains were provided.
    #[error("at least one input tensor train is required")]
    EmptyInputs,

    /// Input tensor trains have different numbers of sites.
    #[error("tensor train length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        /// Expected number of sites.
        expected: usize,
        /// Actual number of sites.
        got: usize,
    },

    /// Input tensor trains have different physical dimensions at one site.
    #[error("site dimension mismatch at site {site}: expected {expected}, got {got}")]
    SiteDimMismatch {
        /// Site where the mismatch was found.
        site: usize,
        /// Expected site dimension.
        expected: usize,
        /// Actual site dimension.
        got: usize,
    },

    /// The caller supplied invalid options.
    #[error("invalid ACI options: {message}")]
    InvalidOptions {
        /// Explanation.
        message: String,
    },

    /// The initial guess is incompatible with the inputs.
    #[error("invalid initial guess: {message}")]
    InvalidInitialGuess {
        /// Explanation.
        message: String,
    },

    /// Batched operator returned or wrote an invalid number of values.
    #[error("operator output length mismatch: expected {expected}, got {got}")]
    OperatorOutputLength {
        /// Expected output length.
        expected: usize,
        /// Actual output length.
        got: usize,
    },

    /// User callback failed.
    #[error("operator failed: {message}")]
    Operator {
        /// Error context.
        message: String,
    },

    /// Matrix cross interpolation failed.
    #[error("matrix CI error: {0}")]
    MatrixCI(#[from] tensor4all_tcicore::MatrixCIError),

    /// Tensor train operation failed.
    #[error("tensor train error: {0}")]
    TensorTrain(#[from] tensor4all_simplett::SimpleTensorTrainError),
}
```

`src/options.rs`:

```rust
use tensor4all_simplett::{TTScalar, TensorTrain};

/// Configuration for ACI elementwise operations.
#[derive(Debug, Clone)]
pub struct AciOptions<T: TTScalar> {
    /// Maximum number of full sweeps.
    pub max_iters: usize,
    /// Minimum number of sweeps before convergence checks can stop the run.
    pub min_iters: usize,
    /// Maximum allowed output bond dimension.
    pub max_bond_dim: usize,
    /// Pivot-error tolerance.
    pub tolerance: f64,
    /// Whether to scale tolerance by an observed problem scale.
    pub scale_tolerance: bool,
    /// Optional initial guess for the output tensor train.
    pub initial_guess: Option<TensorTrain<T>>,
    /// Seed used when generating a default initial guess.
    pub rng_seed: u64,
}

impl<T: TTScalar> Default for AciOptions<T> {
    fn default() -> Self {
        Self {
            max_iters: 20,
            min_iters: 2,
            max_bond_dim: usize::MAX,
            tolerance: 1e-12,
            scale_tolerance: false,
            initial_guess: None,
            rng_seed: 0,
        }
    }
}
```

`src/result.rs`:

```rust
use tensor4all_simplett::{TTScalar, TensorTrain};

/// Output of an ACI elementwise run.
#[derive(Debug, Clone)]
pub struct AciResult<T: TTScalar> {
    /// Resulting tensor train.
    pub tensor_train: TensorTrain<T>,
    /// Maximum bond dimension after each sweep.
    pub ranks: Vec<usize>,
    /// Maximum pivot error after each sweep.
    pub errors: Vec<f64>,
}
```

`src/batch.rs`:

```rust
use crate::{AciError, Result};

/// Column-major 2D view passed to batched elementwise callbacks.
#[derive(Debug, Clone, Copy)]
pub struct ElementwiseBatch<'a, T> {
    values: &'a [T],
    n_inputs: usize,
    n_points: usize,
}

impl<'a, T> ElementwiseBatch<'a, T> {
    /// Create a view over `values` with shape `(n_inputs, n_points)`.
    pub fn new(values: &'a [T], n_inputs: usize, n_points: usize) -> Result<Self> {
        let expected = n_inputs.checked_mul(n_points).ok_or_else(|| AciError::InvalidOptions {
            message: "batch shape overflows usize".to_string(),
        })?;
        if values.len() != expected {
            return Err(AciError::OperatorOutputLength {
                expected,
                got: values.len(),
            });
        }
        Ok(Self { values, n_inputs, n_points })
    }

    /// Number of input tensor trains.
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Number of points in the batch.
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Return the value for one input and point.
    pub fn get(&self, input: usize, point: usize) -> Result<T>
    where
        T: Copy,
    {
        if input >= self.n_inputs || point >= self.n_points {
            return Err(AciError::InvalidOptions {
                message: format!("batch index out of bounds: input={input}, point={point}"),
            });
        }
        Ok(self.values[input + self.n_inputs * point])
    }

    /// Borrow the raw column-major buffer.
    pub fn as_col_major_slice(&self) -> &'a [T] {
        self.values
    }
}
```

Update `src/lib.rs` to export the modules and remove `crate_ready`.

**Step 4: Run tests**

Run:

```bash
cargo test --release -p tensor4all-aci
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI public API types"
```

## Task 3: Add Input Validation and Initial Guess Generation

**Files:**
- Create: `crates/tensor4all-aci/src/validation.rs`
- Create: `crates/tensor4all-aci/src/random_tt.rs`
- Create: `crates/tensor4all-aci/src/scalar.rs`
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Modify: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write failing validation tests**

Add tests:

```rust
use crate::{initial_guess, AciOptions};
use crate::validation::validate_inputs;
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

#[test]
fn validate_inputs_rejects_empty_inputs() {
    let err = validate_inputs::<f64>(&[]).unwrap_err();
    assert!(err.to_string().contains("at least one"));
}

#[test]
fn validate_inputs_rejects_length_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("length mismatch"));
}

#[test]
fn validate_inputs_rejects_site_dim_mismatch() {
    let a = TensorTrain::<f64>::constant(&[2, 3], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 4], 1.0);
    let err = validate_inputs(&[a, b]).unwrap_err();
    assert!(err.to_string().contains("site dimension mismatch"));
}

#[test]
fn default_initial_guess_matches_input_site_dims() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 4], 2.0);
    let guess = initial_guess(&[a, b], &AciOptions::default()).unwrap();
    assert_eq!(guess.site_dims(), vec![2, 3, 4]);
}
```

**Step 2: Run tests and verify failure**

Run:

```bash
cargo test --release -p tensor4all-aci validate_inputs default_initial_guess
```

Expected: FAIL because validation helpers are missing.

**Step 3: Implement validation**

`validation.rs`:

```rust
use crate::{AciError, Result};
use tensor4all_simplett::{AbstractTensorTrain, TTScalar, TensorTrain};

pub(crate) fn validate_options<T: TTScalar>(options: &crate::AciOptions<T>) -> Result<()> {
    if options.max_iters == 0 {
        return Err(AciError::InvalidOptions { message: "max_iters must be positive".to_string() });
    }
    if options.min_iters > options.max_iters {
        return Err(AciError::InvalidOptions {
            message: "min_iters must be <= max_iters".to_string(),
        });
    }
    if !(options.tolerance >= 0.0 && options.tolerance.is_finite()) {
        return Err(AciError::InvalidOptions {
            message: "tolerance must be finite and non-negative".to_string(),
        });
    }
    Ok(())
}

pub(crate) fn validate_inputs<T: TTScalar>(inputs: &[TensorTrain<T>]) -> Result<Vec<usize>> {
    let first = inputs.first().ok_or(AciError::EmptyInputs)?;
    let n = first.len();
    let site_dims = first.site_dims();
    for input in inputs.iter().skip(1) {
        if input.len() != n {
            return Err(AciError::LengthMismatch { expected: n, got: input.len() });
        }
        for site in 0..n {
            let got = input.site_dim(site);
            let expected = site_dims[site];
            if got != expected {
                return Err(AciError::SiteDimMismatch { site, expected, got });
            }
        }
    }
    Ok(site_dims)
}
```

`random_tt.rs` should generate a deterministic random tensor train whose link
dimensions are clamped by the product of left and right physical dimensions and
by the minimum input link dimension where available.

Define small internal scalar traits in `scalar.rs`:

```rust
pub(crate) trait AciRandomScalar: tensor4all_simplett::TTScalar {
    fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self;
}

pub(crate) trait AciScalar:
    tensor4all_simplett::TTScalar
    + tensor4all_tcicore::MatrixLuciScalar
    + AciRandomScalar
    + Copy
{
}

impl<T> AciScalar for T
where
    T: tensor4all_simplett::TTScalar
        + tensor4all_tcicore::MatrixLuciScalar
        + AciRandomScalar
        + Copy
{
}
```

Implement `AciRandomScalar` for `f64` and `num_complex::Complex64`.

**Step 4: Run tests**

Run:

```bash
cargo test --release -p tensor4all-aci validate_inputs default_initial_guess
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI input validation and initial guess"
```

## Task 4: Implement Frame State Initialization

**Files:**
- Create: `crates/tensor4all-aci/src/state.rs`
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Modify: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write failing frame tests**

Add tests for two three-site inputs:

```rust
use crate::{AciOptions, ElementwiseProblem};
use tensor4all_simplett::TensorTrain;

#[test]
fn elementwise_problem_initializes_boundary_frames() {
    let a = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let b = TensorTrain::<f64>::constant(&[2, 2, 2], 2.0);
    let problem = ElementwiseProblem::new(vec![a, b], AciOptions::default()).unwrap();
    assert_eq!(problem.len(), 3);
    assert_eq!(problem.n_inputs(), 2);
    assert_eq!(problem.left_frame_shape(0, 0), Some((1, 1)));
    assert_eq!(problem.right_frame_shape(0, 3), Some((1, 1)));
}
```

Keep `ElementwiseProblem` crate-private if possible. Tests may live in the same
crate so they can access `pub(crate)` helpers.

**Step 2: Run and verify failure**

Run:

```bash
cargo test --release -p tensor4all-aci elementwise_problem_initializes_boundary_frames
```

Expected: FAIL because `ElementwiseProblem` is missing.

**Step 3: Implement frame state**

Use:

```rust
struct ElementwiseProblem<T: AciScalar> {
    inputs: Vec<TensorTrain<T>>,
    solution: TensorTrain<T>,
    left_frames: Vec<Vec<Option<Matrix<T>>>>,
    right_frames: Vec<Vec<Option<Matrix<T>>>>,
    pivot_errors: Vec<f64>,
}
```

Indexing convention:

- `left_frames[input][0] = 1 x 1`
- `left_frames[input][site]` is available after site `site - 1`
- `right_frames[input][n] = 1 x 1`
- `right_frames[input][site]` is available before site `site`

Implement:

```rust
fn update_left_frame(&mut self, input: usize, site: usize, row_indices: &[usize]) -> Result<()>;
fn update_right_frame(&mut self, input: usize, site: usize, col_indices: &[usize]) -> Result<()>;
fn update_left_frames(&mut self, site: usize, row_indices: &[usize]) -> Result<()>;
fn update_right_frames(&mut self, site: usize, col_indices: &[usize]) -> Result<()>;
```

Manual loop formula for left update:

```text
new_full[(lf_row, s), right_bond] =
    sum left_frame[lf_row, input_left] * site_tensor[input_left, s, right_bond]
```

Then select `row_indices`.

Manual loop formula for right update:

```text
new_full[left_bond, (s, rf_col)] =
    sum site_tensor[left_bond, s, input_right] * right_frame[input_right, rf_col]
```

Then select `col_indices`.

**Step 4: Initialize solution CI-canonical form**

Port the Julia right-to-left initialization:

```text
for site in (1..n).rev():
    factorize solution[site].as_right_matrix() with left_orthogonal = false
    replace solution[site] with right factor reshaped as (rank, site_dim, right_dim)
    multiply previous core's right matrix by left factor
    update right frames at `site` using selected column indices
```

Validate selected column indices before indexing input frame products. Return
`AciError::InvalidInitialGuess` instead of panicking.

**Step 5: Run tests**

Run:

```bash
cargo test --release -p tensor4all-aci elementwise_problem
```

Expected: PASS.

**Step 6: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI frame state initialization"
```

## Task 5: Implement Lazy Local Block Evaluation

**Files:**
- Create: `crates/tensor4all-aci/src/local.rs`
- Modify: `crates/tensor4all-aci/src/state.rs`
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Modify: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write failing tests for local value and batch layout**

Create a test that compares the local block evaluator against an explicit
small dense `pitensor` calculation for one input and one bond.

The point ordering must match MatrixLUCI block layout:

```text
point = row_position + rows.len() * col_position
out[point] = A[rows[row_position], cols[col_position]]
```

**Step 2: Implement local value evaluation**

For one input tensor train and one candidate matrix entry:

```rust
fn local_input_value(
    &self,
    input: usize,
    bond: usize,
    row: usize,
    col: usize,
) -> Result<T>
```

Decode row and column:

```text
r_left = left_frame.nrows()
d_left = input.site_dim(bond)
d_right = input.site_dim(bond + 1)
r_right = right_frame.ncols()

left_pivot = row % r_left
site_left = row / r_left
site_right = col % d_right
right_pivot = col / d_right
```

Then compute:

```text
sum_{a,m,b}
    left_frame[left_pivot, a]
  * core_left[a, site_left, m]
  * core_right[m, site_right, b]
  * right_frame[b, right_pivot]
```

Use `Tensor3Ops` and `Matrix` indexing. Return an error if any decoded index is
out of range.

**Step 3: Implement `fill_local_block`**

Build an input values scratch buffer with shape `(n_inputs, n_points)` in
column-major order:

```text
values[input + n_inputs * point]
```

Call `ElementwiseBatch::new(&values, n_inputs, n_points)` and then the user
batched operator. Write output values back to the MatrixLUCI block output in
the same point order.

**Step 4: Add one-update local entry cache**

Inside the local block evaluator, keep:

```rust
HashMap<usize, T>
```

with key:

```rust
row + nrows * col
```

Use checked multiplication/addition. Cache only output elementwise values, not
per-input local values. Drop this cache at the end of each bond update.

**Step 5: Run tests**

Run:

```bash
cargo test --release -p tensor4all-aci local
```

Expected: PASS.

**Step 6: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI lazy local block evaluation"
```

## Task 6: Implement One Bond Update

**Files:**
- Modify: `crates/tensor4all-aci/src/local.rs`
- Modify: `crates/tensor4all-aci/src/state.rs`
- Modify: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write failing one-bond update test**

Use two two-site constant tensor trains. Run one local update at bond `0` with
multiplication and assert the resulting two cores evaluate to the dense product
on all four points.

**Step 2: Implement local update**

Add:

```rust
fn local_update<F>(
    &mut self,
    bond: usize,
    left_orthogonal: bool,
    options: &AciOptions<T>,
    op: &mut F,
) -> Result<()>
where
    F: FnMut(ElementwiseBatch<'_, T>, &mut [T]) -> Result<()>;
```

Inside:

1. Determine candidate matrix dimensions:
   - `nrows = left_solution_rank * site_dim[bond]`
   - `ncols = site_dim[bond + 1] * right_solution_rank`
2. Call `matrix_luci_factors_from_blocks`.
3. Use `RrLUOptions`:
   - `max_rank = options.max_bond_dim`
   - `rel_tol = if options.scale_tolerance { options.tolerance } else { 0.0 }`
   - `abs_tol = if options.scale_tolerance { 0.0 } else { options.tolerance }`
   - `left_orthogonal = left_orthogonal`
4. Reshape factors:
   - left factor `(nrows, rank)` to `Tensor3(left_rank, site_dim_left, rank)`
   - right factor `(rank, ncols)` to `Tensor3(rank, site_dim_right, right_rank)`
5. Replace the two solution site tensors.
6. Update left or right frames with selected LUCI row or column indices.
7. Store `last_pivot_error` for that bond.

**Step 3: Run one-bond tests**

Run:

```bash
cargo test --release -p tensor4all-aci one_bond
```

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI local bond update"
```

## Task 7: Implement Public Sweep APIs

**Files:**
- Create: `crates/tensor4all-aci/src/elementwise.rs`
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Modify: `crates/tensor4all-aci/src/tests.rs`

**Step 1: Write failing public API tests**

Add tests:

```rust
use crate::{elementwise, elementwise_batched, AciOptions};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

#[test]
fn elementwise_multiplies_constant_tensor_trains() {
    let a = TensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let b = TensorTrain::<f64>::constant(&[2, 3, 2], 4.0);
    let result = elementwise(|xs: &[f64]| xs[0] * xs[1], &[a, b], &AciOptions::default()).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                assert!((result.tensor_train.evaluate(&[i, j, k]).unwrap() - 8.0).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn scalar_and_batched_paths_match() {
    let a = TensorTrain::<f64>::constant(&[2, 2], 3.0);
    let b = TensorTrain::<f64>::constant(&[2, 2], 5.0);
    let options = AciOptions::default();
    let scalar = elementwise(|xs: &[f64]| xs[0] + xs[1], &[a.clone(), b.clone()], &options).unwrap();
    let batched = elementwise_batched(
        |batch, out| {
            for p in 0..batch.n_points() {
                out[p] = batch.get(0, p)? + batch.get(1, p)?;
            }
            Ok(())
        },
        &[a, b],
        &options,
    ).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            let idx = [i, j];
            assert!((scalar.tensor_train.evaluate(&idx).unwrap()
                - batched.tensor_train.evaluate(&idx).unwrap()).abs() < 1e-12);
        }
    }
}
```

**Step 2: Implement sweep**

`elementwise_batched` flow:

1. Validate options and inputs.
2. Build or validate initial guess.
3. Construct `ElementwiseProblem`.
4. For `iteration in 0..max_iters`:
   - forward if `iteration` is even, backward otherwise
   - visit bonds in order or reverse order
   - call `local_update`
   - push `solution.rank()` and max pivot error
   - stop after `min_iters` if error <= tolerance and rank history is stable
5. Return `AciResult`.

Use the Julia convergence rule as the first implementation:

```text
iteration >= min_iters
max_error <= tolerance
last min_iters ranks do not exceed the rank at the start of that window
```

**Step 3: Implement scalar wrapper**

Wrap scalar callback into batched callback with a small scratch vector:

```rust
pub fn elementwise<T, F>(mut op: F, inputs: &[TensorTrain<T>], options: &AciOptions<T>) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: FnMut(&[T]) -> T,
{
    let mut scratch = Vec::new();
    elementwise_batched(
        |batch, out| {
            scratch.resize(batch.n_inputs(), T::zero());
            for p in 0..batch.n_points() {
                for k in 0..batch.n_inputs() {
                    scratch[k] = batch.get(k, p)?;
                }
                out[p] = op(&scratch);
            }
            Ok(())
        },
        inputs,
        options,
    )
}
```

**Step 4: Run public API tests**

Run:

```bash
cargo test --release -p tensor4all-aci elementwise
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI elementwise sweep API"
```

## Task 8: Add Dense Oracle and Cache Behavior Tests

**Files:**
- Create: `crates/tensor4all-aci/tests/elementwise.rs`
- Modify: `crates/tensor4all-aci/src/local.rs`

**Step 1: Add dense oracle helper**

In the integration test, use `TensorTrain::to_dense()` for small tensors only.
Materialize each input once and compare whole dense buffers.

Test cases:

- rank-1 constant multiplication
- addition of two non-constant small hand-built tensor trains
- nonlinear scalar operation such as `xs[0].sin() + xs[1] * xs[1]`

**Step 2: Add cache behavior test**

Use a counting batched callback:

```rust
let calls = std::cell::Cell::new(0usize);
let op = |batch: ElementwiseBatch<'_, f64>, out: &mut [f64]| {
    calls.set(calls.get() + batch.n_points());
    for p in 0..batch.n_points() {
        out[p] = batch.get(0, p)? * batch.get(1, p)?;
    }
    Ok(())
};
```

Assert that repeated local block requests inside one local update do not call
the callback for every duplicate entry. Do not assert a fragile exact count
unless the test controls the local block requests directly.

**Step 3: Run tests**

Run:

```bash
cargo test --release -p tensor4all-aci --test elementwise
```

Expected: PASS.

**Step 4: Commit**

```bash
git add crates/tensor4all-aci
git commit -m "Add ACI dense oracle tests"
```

## Task 9: Add Docs and API Dump

**Files:**
- Modify: `crates/tensor4all-aci/src/lib.rs`
- Modify: `README.md`
- Modify: `docs/api/*.md` through api-dump generation

**Step 1: Add crate-level runnable example**

In `src/lib.rs`, add a doc example:

```rust
//! ```
//! use tensor4all_aci::{elementwise, AciOptions};
//! use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
//!
//! let a = TensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let b = TensorTrain::<f64>::constant(&[2, 3], 4.0);
//! let result = elementwise(
//!     |xs: &[f64]| xs[0] * xs[1],
//!     &[a, b],
//!     &AciOptions::default(),
//! ).unwrap();
//!
//! assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 8.0).abs() < 1e-10);
//! assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
//! ```
```

**Step 2: Update README crate table**

Add:

```markdown
| [tensor4all-aci](crates/tensor4all-aci/) | Alternating Cross Interpolation for elementwise tensor train operations |
```

Do not claim C API or language binding support.

**Step 3: Generate API docs**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: PASS and a new or updated `docs/api/tensor4all_aci.md`.

**Step 4: Run doc tests**

Run:

```bash
cargo test --doc --release -p tensor4all-aci
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-aci README.md docs/api
git commit -m "Document ACI Rust API"
```

## Task 10: Add Julia Parity and Chi-Scaling Benchmarks

**Files:**
- Modify: `crates/tensor4all-aci/Cargo.toml`
- Create: `crates/tensor4all-aci/benches/elementwise_scaling.rs`
- Modify: `benchmarks/julia/Project.toml`
- Create: `benchmarks/julia/benchmark_aci_elementwise.jl`
- Modify: `benchmarks/README.md`
- Create: `benchmarks/results/<date>-aci-elementwise.md`

**Step 1: Add Criterion benchmark target**

In the existing `crates/tensor4all-aci/Cargo.toml` `[dev-dependencies]`
section, add `criterion.workspace = true`, preserving existing entries:

```toml
[dev-dependencies]
approx.workspace = true
criterion.workspace = true

[[bench]]
name = "elementwise_scaling"
harness = false
```

Do not create a second `[dev-dependencies]` section.

**Step 2: Implement Rust chi-scaling benchmark**

Create `crates/tensor4all-aci/benches/elementwise_scaling.rs`.

Benchmark requirements:

- Benchmark group name: `aci_elementwise_chi_scaling`.
- Fixed parameters for the primary comparison:
  - `n_sites = 12`
  - `local_dim = 2`
  - `n_inputs = 2`
  - `tolerance = 1e-10`
  - `max_iters = 20`
  - `chi_values = [2, 4, 8, 16]`
- Add `chi = 32` as an optional long case behind a command-line filter or a
  separate benchmark group so the smoke run remains practical.
- Build deterministic input tensor trains from the same closed-form core formula
  used by the Julia script. Do not rely on Rust and Julia RNG streams matching.
- Time the batched public API path as the primary benchmark. The scalar wrapper
  can be included as a secondary callback-overhead benchmark, but the completion
  criterion is the batched path.
- Record enough observable metadata to compare with Julia:
  - `chi`
  - final maximum output bond dimension
  - number of sweeps recorded in `AciResult::ranks`
  - final maximum pivot error from `AciResult::errors`
  - sampled max absolute error against direct dense point evaluation
- Use `criterion::{criterion_group, criterion_main, BenchmarkId, Criterion}` and
  `black_box`.

The sampled correctness check should evaluate at least 64 deterministic points
for each `chi` and assert the sampled max absolute error is less than `1e-8`.
This check must run outside the timed closure so benchmark timing measures ACI,
not validation.

**Step 3: Implement Julia comparison benchmark**

Add `BenchmarkTools` to `benchmarks/julia/Project.toml`:

```toml
BenchmarkTools = "6e4b80f9-dda5-5e3a-8b7e-868b0140f7fe"
```

Create `benchmarks/julia/benchmark_aci_elementwise.jl` that:

- Loads upstream `AlternatingCrossInterpolation.jl` as `ACI`.
  - If `ENV["ACI_JL_PATH"]` is set, use `Pkg.develop(path=ENV["ACI_JL_PATH"])`.
  - Otherwise use `Pkg.add(url="https://github.com/tensor4all/AlternatingCrossInterpolation.jl.git")`.
- Uses the same benchmark parameters and the same deterministic core formula as
  the Rust benchmark.
- Uses `ACI.TruncationParameters(typemax(Int), 1e-10, false)` unless the Rust
  benchmark uses a smaller `max_bond_dim`; keep the two sides matched.
- Runs `ACI.elementwise(*, inputs; truncationparameters=...)`.
- Prints CSV rows with this schema:

```text
impl,n_sites,local_dim,chi,tolerance,median_ms,min_ms,mean_ms,max_ms,output_max_chi,n_sweeps,final_error,sampled_max_abs_error
```

Use `BenchmarkTools.@benchmark` or equivalent repeated timing. Do not include
package installation time or input construction in the measured closure.

**Step 4: Update benchmark README**

In `benchmarks/README.md`, add commands:

```bash
RAYON_NUM_THREADS=1 cargo bench -p tensor4all-aci --bench elementwise_scaling -- --sample-size 10
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_aci_elementwise.jl --chis 2,4,8,16
```

Document that `chi`/`χ` is the TT bond dimension scaling axis and that `chi=32`
is a longer optional run.

**Step 5: Run Rust benchmark smoke check**

Run:

```bash
cargo bench --no-run -p tensor4all-aci --bench elementwise_scaling
RAYON_NUM_THREADS=1 cargo bench -p tensor4all-aci --bench elementwise_scaling -- --sample-size 10 --measurement-time 1 --warm-up-time 1
```

Expected: benchmark compiles and prints timings for `chi = 2, 4, 8, 16`.

**Step 6: Run Julia comparison**

Run:

```bash
ACI_JL_PATH=/tmp/AlternatingCrossInterpolation.jl BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_aci_elementwise.jl --chis 2,4,8,16
```

Expected: Julia benchmark prints the same CSV schema and reports sampled errors
below `1e-8`. If `/tmp/AlternatingCrossInterpolation.jl` is absent, clone the
upstream repository or let the script install by URL.

**Step 7: Record comparison results**

Create `benchmarks/results/<date>-aci-elementwise.md` with:

- Exact Rust and Julia commands.
- Host CPU/thread settings if available.
- A table comparing Rust and Julia for each `chi`.
- A short interpretation of scaling versus `chi`, including whether runtime
  growth appears consistent with the expected local two-site block cost.
- Numerical parity notes: sampled max absolute errors for both implementations
  and any rank/error trajectory differences.

Do not claim Rust is faster or slower in docs unless the benchmark was actually
run in the current branch and the raw commands/results are recorded.

**Step 8: Commit**

```bash
git add crates/tensor4all-aci/Cargo.toml crates/tensor4all-aci/benches benchmarks/julia/Project.toml benchmarks/julia/benchmark_aci_elementwise.jl benchmarks/README.md benchmarks/results
git commit -m "Benchmark ACI elementwise chi scaling"
```

## Task 11: Final Verification

**Files:**
- No planned edits unless verification finds a bug.

**Step 1: Format**

Run:

```bash
cargo fmt --all
cargo fmt --all -- --check
```

Expected: PASS.

**Step 2: Lint**

Run:

```bash
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: PASS. Fix warnings by improving code, not by suppressing warnings
unless the suppression has a narrow justification.

**Step 3: Targeted tests**

Run:

```bash
cargo test --release -p tensor4all-aci
cargo test --doc --release -p tensor4all-aci
```

Expected: PASS.

**Step 4: Full tests**

Run:

```bash
cargo nextest run --release --workspace
```

Expected: PASS.

**Step 5: Benchmark compile check**

Run:

```bash
cargo bench --no-run -p tensor4all-aci --bench elementwise_scaling
```

Expected: PASS.

**Step 6: Confirm Julia comparison was recorded**

Check:

```bash
ls benchmarks/results/*aci-elementwise*.md
```

Expected: at least one result file from Task 10 containing the Rust/Julia
comparison and chi-scaling table.

**Step 7: Commit any verification fixes**

If formatting or test fixes were required:

```bash
git add <changed-files>
git commit -m "Fix ACI verification issues"
```

If no fixes were required, do not create an empty commit.
