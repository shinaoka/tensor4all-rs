> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# TensorCI1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port public legacy `TensorCI1` and `crossinterpolate1` from `TensorCrossInterpolation.jl`.

**Architecture:** Add a focused `tensorci1` module in `tensor4all-tensorci`. Keep `MatrixCI` as a private helper, use solve-based pivot-block operations, and convert the final state to `tensor4all-simplett::TensorTrain` for repeated evaluation.

**Tech Stack:** Rust, `tensor4all-tensorci`, `tensor4all-tcicore`, `tensor4all-tensorbackend::Matrix`, `tensor4all-simplett`.

---

## File Structure

- Create `crates/tensor4all-tensorci/src/tensorci1.rs`: public TCI1 state, options, entry point, and helpers.
- Create `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`: private `MatrixCI` helper.
- Create `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`: TCI1 tests.
- Create `crates/tensor4all-tensorci/examples/tci1_speed.rs`: deterministic Rust TCI1 speed benchmark used by the pre-PR gate.
- Create `scripts/compare-tci1-speed.sh`: runs the Rust benchmark and the sibling `../TensorCrossInterpolation.jl` benchmark on the same cases.
- Modify `crates/tensor4all-tensorci/src/lib.rs`: expose `tensorci1`, `TensorCI1`, `TCI1Options`, `crossinterpolate1`.
- Modify `docs/api/tensor4all_tensorci.md`: regenerate via `api-dump`.

### Task 1: Add Public API Shell And Failing Tests

**Files:**
- Create: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Create: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`
- Modify: `crates/tensor4all-tensorci/src/lib.rs`

- [ ] **Step 1: Add the public shell**

```rust
//! TensorCI1 - legacy one-site Tensor Cross Interpolation algorithm.

use crate::error::{Result, TCIError};
use tensor4all_simplett::{TTScalar, TensorTrain};
use tensor4all_tcicore::{MatrixLuciScalar, MultiIndex, Scalar};

mod matrix_ci;

/// Configuration for [`crossinterpolate1`].
#[derive(Debug, Clone)]
pub struct TCI1Options {
    /// Relative convergence tolerance.
    pub tolerance: f64,
    /// Maximum number of sweeps.
    pub max_iter: usize,
    /// Tolerance for accepting one new local pivot.
    pub pivot_tolerance: f64,
    /// Whether to normalize pivot errors by the maximum sampled value.
    pub normalize_error: bool,
    /// Additional global pivots inserted before sweeps.
    pub additional_pivots: Vec<MultiIndex>,
}

impl Default for TCI1Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 200,
            pivot_tolerance: 1e-12,
            normalize_error: true,
            additional_pivots: Vec::new(),
        }
    }
}

/// Legacy one-site tensor cross interpolation state.
#[derive(Debug, Clone)]
pub struct TensorCI1<T: Scalar + TTScalar> {
    local_dims: Vec<usize>,
    pivot_errors: Vec<f64>,
    max_sample_value: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T> TensorCI1<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
    /// Construct an empty TCI1 state.
    pub fn new(local_dims: Vec<usize>) -> Result<Self> {
        if local_dims.len() < 2 {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims should have at least 2 elements".to_string(),
            });
        }
        Ok(Self {
            pivot_errors: vec![f64::INFINITY; local_dims.len() - 1],
            local_dims,
            max_sample_value: 0.0,
            _marker: std::marker::PhantomData,
        })
    }

    /// Number of sites.
    pub fn len(&self) -> usize {
        self.local_dims.len()
    }

    /// Whether the state has no sites.
    pub fn is_empty(&self) -> bool {
        self.local_dims.is_empty()
    }

    /// Local dimensions.
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Maximum sampled absolute value.
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Pivot errors from the latest sweep.
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }
}

/// Approximate a function with the legacy TCI1 algorithm.
pub fn crossinterpolate1<T, F>(
    _f: F,
    local_dims: Vec<usize>,
    _first_pivot: MultiIndex,
    _options: TCI1Options,
) -> Result<(TensorCI1<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
{
    let tci = TensorCI1::new(local_dims)?;
    Ok((tci, Vec::new(), Vec::new()))
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Re-export from `lib.rs`**

```rust
pub mod tensorci1;
pub use tensorci1::{crossinterpolate1, TCI1Options, TensorCI1};
```

- [ ] **Step 3: Add failing behavior tests**

```rust
use super::*;
use tensor4all_simplett::AbstractTensorTrain;

#[test]
fn test_crossinterpolate1_rank2_function() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![4, 4],
        vec![3, 3],
        TCI1Options {
            tolerance: 1e-12,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    let tt = tci.to_tensor_train().unwrap();
    assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
}
```

- [ ] **Step 4: Run the failing test**

```bash
cargo test --release -p tensor4all-tensorci test_crossinterpolate1_rank2_function
```

Expected: compile failure because `to_tensor_train` does not exist, then assertion failures until the real algorithm is implemented.

### Task 2: Implement Private MatrixCI Helper

**Files:**
- Create: `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`

- [ ] **Step 1: Add helper structure and solve-based evaluation**

```rust
use crate::error::{Result, TCIError};
use tensor4all_tcicore::Scalar;
use tensor4all_tensorbackend::{solve_matrix, transpose, Matrix};

#[derive(Debug, Clone)]
pub(super) struct MatrixCI<T: Scalar> {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    left: Matrix<T>,
    right: Matrix<T>,
}

impl<T: Scalar> MatrixCI<T> {
    pub(super) fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        left: Matrix<T>,
        right: Matrix<T>,
    ) -> Result<Self> {
        if left.ncols() != right.nrows() {
            return Err(TCIError::DimensionMismatch {
                message: "MatrixCI left/right rank mismatch".to_string(),
            });
        }
        Ok(Self {
            row_indices,
            col_indices,
            left,
            right,
        })
    }

    pub(super) fn rank(&self) -> usize {
        self.left.ncols()
    }

    pub(super) fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    pub(super) fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    pub(super) fn evaluate(&self, row: usize, col: usize) -> T {
        let mut value = T::zero();
        for k in 0..self.rank() {
            value = value + self.left[[row, k]] * self.right[[k, col]];
        }
        value
    }
}
```

- [ ] **Step 2: Add helper tests**

Test that `evaluate` reconstructs a rank-1 outer product at selected entries and that mismatched factor shapes return `TCIError::DimensionMismatch`.

- [ ] **Step 3: Run helper tests**

```bash
cargo test --release -p tensor4all-tensorci matrix_ci
```

Expected: PASS.

### Task 3: Port TCI1 State And Pivot Updates

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs`

- [ ] **Step 1: Replace placeholder fields with TCI1 state**

Store:

```rust
i_set: Vec<Vec<MultiIndex>>,
j_set: Vec<Vec<MultiIndex>>,
local_dims: Vec<usize>,
site_tensors: Vec<Tensor3<T>>,
pivots: Vec<Matrix<T>>,
pivot_errors: Vec<f64>,
max_sample_value: f64,
```

- [ ] **Step 2: Add initialization from first pivot**

Implement `TensorCI1::from_first_pivot(f, local_dims, first_pivot)`:

1. Validate `first_pivot.len() == local_dims.len()`.
2. Validate each coordinate is in range `0..local_dims[p]`.
3. Evaluate `f(&first_pivot)` and reject zero sample value.
4. Initialize `i_set[p] = first_pivot[..p]` and `j_set[p] = first_pivot[p+1..]`.
5. Fill initial site tensors and pivot matrices with function samples.

- [ ] **Step 3: Add solve helpers**

Add private helpers equivalent to Julia `AtimesBinv` and `AinvtimesB`:

```rust
fn right_solve<T: Scalar>(a: &Matrix<T>, p: &Matrix<T>) -> Result<Matrix<T>> {
    let solved_t = solve_matrix(&transpose(p), &transpose(a))?;
    Ok(transpose(&solved_t))
}

fn left_solve<T: Scalar>(p: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>> {
    solve_matrix(p, b).map_err(Into::into)
}
```

- [ ] **Step 4: Add `add_pivot_at_bond` and sweep logic**

Implement forward/backward sweeps that add one pivot per bond by evaluating local error candidates, updating I/J sets, site tensors, pivot matrix, pivot errors, and `max_sample_value`.

- [ ] **Step 5: Run rank-2 test**

```bash
cargo test --release -p tensor4all-tensorci test_crossinterpolate1_rank2_function
```

Expected: PASS.

### Task 4: Add TensorTrain Conversion And Evaluation

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`

- [ ] **Step 1: Add `to_tensor_train`**

```rust
pub fn to_tensor_train(&self) -> Result<TensorTrain<T>> {
    TensorTrain::new(self.site_tensors.clone()).map_err(Into::into)
}
```

Call `right_solve` on each non-final site tensor matrix and its corresponding
pivot matrix before reshaping it back to `Tensor3<T>`. Leave the final site
tensor unchanged, matching Julia's `sitetensors(tci::TensorCI1)` behavior.

- [ ] **Step 2: Add single-point `evaluate`**

```rust
pub fn evaluate(&self, index: &[usize]) -> Result<T> {
    self.to_tensor_train()?.evaluate(index).map_err(Into::into)
}
```

- [ ] **Step 3: Add tests for `evaluate` and `to_tensor_train`**

Check every point of a `4 x 4` rank-2 function by materializing the tensor train once and evaluating all grid points through `tt.evaluate`.

- [ ] **Step 4: Run TCI1 tests**

```bash
cargo test --release -p tensor4all-tensorci tensorci1
```

Expected: PASS.

### Task 5: Port Julia Coverage And Required Rust Tests

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs`

- [ ] **Step 1: Add trivial constant MPS tests**

Port the important public behavior from
`../TensorCrossInterpolation.jl/test/test_tensorci1.jl`'s `"trivial MPS"`
test. Use zero-based Rust indices.

```rust
#[test]
fn test_tensorci1_trivial_constant_function_stays_rank_one() {
    let n = 5;
    let f = |_idx: &MultiIndex| 1.0_f64;
    let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
        f,
        vec![2; n],
        vec![0; n],
        TCI1Options {
            tolerance: 1e-12,
            max_iter: 8,
            pivot_tolerance: 1e-8,
            ..TCI1Options::default()
        },
    )
    .unwrap();

    assert_eq!(tci.len(), n);
    assert_eq!(tci.link_dims(), vec![1; n - 1]);
    assert_eq!(tci.rank(), 1);
    assert!(errors.iter().all(|&error| error <= 1e-12));
    assert!(ranks.iter().all(|&rank| rank == 1));

    let tt = tci.to_tensor_train().unwrap();
    for index in all_indices(&[2, 2, 2, 2, 2]) {
        assert_abs_diff_eq!(tt.evaluate(&index).unwrap(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(tci.evaluate(&index).unwrap(), 1.0, epsilon = 1e-12);
    }
}
```

Add `all_indices(local_dims: &[usize]) -> Vec<MultiIndex>` as a test helper in
the same module. It should enumerate indices in lexicographic order without
depending on internal TCI storage.

- [ ] **Step 2: Add real and complex Lorentz tests**

Port the Julia `"Lorentz MPS"` scenarios for both `f64` and `Complex64`.
Use local dimensions `vec![10; 5]`, first pivot `vec![0; 5]`, Julia's
`globalpivot = [2, 9, 10, 5, 7]` converted to Rust as
`vec![1, 8, 9, 4, 6]`, and additional pivots converted by subtracting one:
`[9, 7, 9, 3, 3]`, `[4, 3, 7, 8, 2]`, `[6, 6, 9, 4, 8]`,
`[6, 6, 9, 4, 8]`.

Use `num_complex::Complex64` and this generic shape:

```rust
fn lorentz_f64(idx: &MultiIndex) -> f64 {
    let norm: f64 = idx.iter().map(|&x| (x + 1) as f64).map(|x| x * x).sum();
    1.0 / (norm + 1.0)
}

fn lorentz_c64(idx: &MultiIndex) -> Complex64 {
    let norm: f64 = idx.iter().map(|&x| (x + 1) as f64).map(|x| x * x).sum();
    Complex64::new(0.0, 1.0) / Complex64::new(norm + 1.0, 0.0)
}
```

For both scalar types:

1. Construct with `crossinterpolate1` and `max_iter: 200`, `tolerance: 1e-12`.
2. Assert every pivot error is `<= 1e-12`.
3. Assert `rank() <= 200` and every link dimension is `<= 200`.
4. Convert to `TensorTrain` and evaluate all points in the `0..3` grid for
   every site, matching Julia's `Iterators.product([1:3 for p in 1:n]...)`
   check after zero-based conversion.
5. Assert both `tci.evaluate` and `tt.evaluate` match the reference function
   within `1e-8` for `f64` and `1e-8` norm for `Complex64`.

- [ ] **Step 3: Add additional and duplicate global pivot tests**

Add coverage for Julia's `additionalpivots` behavior:

```rust
#[test]
fn test_crossinterpolate1_additional_pivots_are_inserted_and_deduplicated() {
    let additional_pivots = vec![
        vec![9, 7, 9, 3, 3],
        vec![4, 3, 7, 8, 2],
        vec![6, 6, 9, 4, 8],
        vec![6, 6, 9, 4, 8],
    ];

    let (tci, _ranks, _errors) = crossinterpolate1::<f64, _>(
        lorentz_f64,
        vec![10; 5],
        vec![0; 5],
        TCI1Options {
            tolerance: 1e-12,
            max_iter: 200,
            additional_pivots: additional_pivots.clone(),
            ..TCI1Options::default()
        },
    )
    .unwrap();

    for pivot in &additional_pivots {
        assert_abs_diff_eq!(
            tci.evaluate(pivot).unwrap(),
            lorentz_f64(pivot),
            epsilon = 1e-8
        );
    }

    assert!(tci.link_dims().iter().all(|&dim| dim <= 200));
    assert!(tci.rank() <= 200);
}
```

Add a module-private test for the direct global-pivot insertion helper. Insert
the same global pivot twice and assert the second insertion does not increase
`rank()` or `link_dims()`.

- [ ] **Step 4: Add manual sweep parity tests**

Keep local pivot insertion crate-private unless it is needed by Rust users.
Within `tensorci1` module tests, port the Julia manual-sweep checks:

1. Build a `TensorCI1` from the first pivot for real and complex Lorentz
   functions.
2. Assert initial `link_dims() == vec![1; n - 1]` and `rank() == 1`.
3. Call the internal `add_pivot_at_bond` for every bond once and assert
   `link_dims() == vec![2; n - 1]` and `rank() == 2`.
4. Insert global pivot `vec![1, 8, 9, 4, 6]` and assert
   `link_dims() == vec![3; n - 1]`, `rank() == 3`, and the pivot evaluates
   accurately.
5. Insert the same global pivot again and assert rank/link dimensions remain
   unchanged.
6. Run additional local sweeps for target ranks 4 through 8 and assert the
   link dimensions and rank match the target after each sweep.

- [ ] **Step 5: Add invalid first-pivot tests**

Assert wrong length, out-of-range coordinate, and zero-valued initial sample return typed errors.

- [ ] **Step 6: Run crate tests**

```bash
cargo test --release -p tensor4all-tensorci tensorci1
```

Expected: PASS.

### Task 6: Add Pre-PR Julia Speed Benchmark Gate

**Files:**
- Create: `crates/tensor4all-tensorci/examples/tci1_speed.rs`
- Create: `scripts/compare-tci1-speed.sh`

- [ ] **Step 1: Add Rust benchmark example**

Create an example binary that runs the same deterministic TCI1 cases as the
Julia gate and prints parseable key/value lines:

```rust
use std::time::Instant;

use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
use tensor4all_tcicore::MultiIndex;

fn lorentz(idx: &MultiIndex) -> f64 {
    let norm: f64 = idx.iter().map(|&x| (x + 1) as f64).map(|x| x * x).sum();
    1.0 / (norm + 1.0)
}

fn main() {
    let cases = [
        ("lorentz_10x5_tol1e-8", 1e-8, 40_usize),
        ("lorentz_10x5_tol1e-12", 1e-12, 200_usize),
    ];

    for (name, tolerance, max_iter) in cases {
        let start = Instant::now();
        let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
            lorentz,
            vec![10; 5],
            vec![0; 5],
            TCI1Options {
                tolerance,
                max_iter,
                ..TCI1Options::default()
            },
        )
        .expect("TCI1 benchmark should converge");
        let elapsed = start.elapsed().as_secs_f64();
        let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
        let final_rank = ranks.last().copied().unwrap_or_else(|| tci.rank());
        println!(
            "case={name} impl=rust seconds={elapsed:.9} rank={final_rank} error={final_error:.6e}"
        );
    }
}
```

- [ ] **Step 2: Add comparison script**

Create `scripts/compare-tci1-speed.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

JULIA_REPO="${1:-../TensorCrossInterpolation.jl}"
THRESHOLD="${TCI1_SPEED_RATIO_THRESHOLD:-2.0}"

if [[ ! -d "$JULIA_REPO" ]]; then
  echo "missing Julia repo: $JULIA_REPO" >&2
  exit 2
fi

rust_output="$(cargo run --release -p tensor4all-tensorci --example tci1_speed)"
julia_output="$(julia --project="$JULIA_REPO" -e '
using TensorCrossInterpolation

function lorentz(v)
    return 1.0 / (sum(v .^ 2) + 1.0)
end

cases = [
    ("lorentz_10x5_tol1e-8", 1e-8, 40),
    ("lorentz_10x5_tol1e-12", 1e-12, 200),
]

for (name, tolerance, maxiter) in cases
    elapsed = @elapsed begin
        tci, ranks, errors = TensorCrossInterpolation.crossinterpolate1(
            Float64,
            lorentz,
            fill(10, 5),
            ones(Int, 5);
            tolerance=tolerance,
            maxiter=maxiter,
        )
    end
    final_rank = isempty(ranks) ? TensorCrossInterpolation.rank(tci) : ranks[end]
    final_error = isempty(errors) ? Inf : errors[end]
    println("case=$(name) impl=julia seconds=$(elapsed) rank=$(final_rank) error=$(final_error)")
end
')"

combined_output="$(printf "%s\n%s\n" "$rust_output" "$julia_output")"
printf "%s\n" "$combined_output"

COMBINED_OUTPUT="$combined_output" python3 - "$THRESHOLD" <<'PY'
import os
import re
import statistics
import sys

threshold = float(sys.argv[1])
records = {}
pattern = re.compile(r"case=(\S+) impl=(rust|julia) seconds=([0-9.eE+-]+)")
for line in os.environ["COMBINED_OUTPUT"].splitlines():
    match = pattern.search(line)
    if match:
        case, impl, seconds = match.groups()
        records.setdefault(case, {})[impl] = float(seconds)

ratios = []
for case, values in sorted(records.items()):
    if "rust" not in values or "julia" not in values:
        raise SystemExit(f"missing rust/julia timing for {case}")
    ratio = values["rust"] / values["julia"]
    ratios.append(ratio)
    print(f"case={case} rust_seconds={values['rust']:.9f} julia_seconds={values['julia']:.9f} rust_julia_ratio={ratio:.3f}")

median_ratio = statistics.median(ratios)
print(f"median_rust_julia_ratio={median_ratio:.3f} threshold={threshold:.3f}")
if median_ratio > threshold:
    raise SystemExit(1)
PY
```

Make the script executable with `chmod +x scripts/compare-tci1-speed.sh`.

- [ ] **Step 3: Run speed gate against Julia**

```bash
./scripts/compare-tci1-speed.sh ../TensorCrossInterpolation.jl
```

Expected: exits 0 and prints one Rust/Julia ratio per case plus a median
ratio. If the median Rust/Julia ratio is greater than `2.0`, do not mark the PR
ready without explicit user approval; include the measured output in the PR
notes either way.

### Task 7: Docs, API Dump, And Commit

**Files:**
- Modify: `crates/tensor4all-tensorci/src/lib.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci1.rs`
- Modify: `docs/api/tensor4all_tensorci.md`
- Include: `crates/tensor4all-tensorci/examples/tci1_speed.rs`
- Include: `scripts/compare-tci1-speed.sh`

- [ ] **Step 1: Add rustdoc examples**

Add runnable examples for `TCI1Options`, `TensorCI1::new`, and `crossinterpolate1`. Each example must include assertions.

- [ ] **Step 2: Verify**

```bash
cargo fmt --all
cargo test --release -p tensor4all-tensorci tensorci1
cargo test --doc --release -p tensor4all-tensorci
cargo run -p api-dump --release -- . -o docs/api
./scripts/compare-tci1-speed.sh ../TensorCrossInterpolation.jl
```

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-tensorci/src/lib.rs \
  crates/tensor4all-tensorci/src/tensorci1.rs \
  crates/tensor4all-tensorci/src/tensorci1/matrix_ci.rs \
  crates/tensor4all-tensorci/src/tensorci1/tests/mod.rs \
  crates/tensor4all-tensorci/examples/tci1_speed.rs \
  scripts/compare-tci1-speed.sh \
  docs/api/tensor4all_tensorci.md
git commit -m "feat(tensorci): add public TensorCI1"
```
