> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# TensorCI Conversion Constructors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add non-dense TensorCI conversion constructors, starting with consuming `TensorTrain<T> -> TensorCI2<T>`.

**Architecture:** Implement conversion logic in a new focused conversion module for `tensor4all-tensorci`. Reuse one-site LU index extraction instead of full dense tensor materialization.

**Tech Stack:** Rust, `tensor4all-tensorci`, `tensor4all-simplett::TensorTrain`, `tensor4all-tcicore::MatrixLUCI`.

---

## File Structure

- Create `crates/tensor4all-tensorci/src/conversion.rs`: conversion options and helpers.
- Modify `crates/tensor4all-tensorci/src/lib.rs`: expose conversion module and options.
- Modify `crates/tensor4all-tensorci/src/tensorci2.rs`: add constructor methods or call conversion helpers.
- Create `crates/tensor4all-tensorci/src/conversion/tests/mod.rs`: conversion tests.
- Modify `docs/api/tensor4all_tensorci.md`: regenerate via `api-dump`.

### Task 1: Add Options And Failing Tests

**Files:**
- Create: `crates/tensor4all-tensorci/src/conversion.rs`
- Create: `crates/tensor4all-tensorci/src/conversion/tests/mod.rs`
- Modify: `crates/tensor4all-tensorci/src/lib.rs`

- [ ] **Step 1: Add the module shell**

```rust
//! Conversion constructors for TensorCI state objects.

/// Options for constructing [`TensorCI2`](crate::TensorCI2) from a tensor train.
#[derive(Debug, Clone)]
pub struct TensorCI2FromTensorTrainOptions {
    /// Relative tolerance used during one-site index extraction.
    pub tolerance: f64,
    /// Maximum bond dimension retained during index extraction.
    pub max_bond_dim: usize,
    /// Maximum number of alternating one-site index extraction sweeps.
    pub max_iter: usize,
}

impl Default for TensorCI2FromTensorTrainOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            max_iter: 3,
        }
    }
}

#[cfg(test)]
mod tests;
```

- [ ] **Step 2: Register the module**

In `lib.rs`:

```rust
pub mod conversion;
pub use conversion::TensorCI2FromTensorTrainOptions;
```

- [ ] **Step 3: Add a failing conversion test**

```rust
use crate::{TensorCI2, TensorCI2FromTensorTrainOptions};
use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};

#[test]
fn test_tensorci2_from_tensor_train_preserves_values() {
    let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 2.5);
    let tci = TensorCI2::from_tensor_train(tt, TensorCI2FromTensorTrainOptions::default()).unwrap();
    let roundtrip = tci.to_tensor_train().unwrap();
    assert!((roundtrip.evaluate(&[1, 2, 1]).unwrap() - 2.5).abs() < 1e-12);
}
```

- [ ] **Step 4: Run the failing test**

```bash
cargo test --release -p tensor4all-tensorci test_tensorci2_from_tensor_train_preserves_values
```

Expected: compile failure because `TensorCI2::from_tensor_train` does not exist.

### Task 2: Implement Consuming `from_tensor_train`

**Files:**
- Modify: `crates/tensor4all-tensorci/src/conversion.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`

- [ ] **Step 1: Add the constructor method**

```rust
impl<T> TensorCI2<T>
where
    T: tensor4all_tcicore::Scalar
        + tensor4all_simplett::TTScalar
        + Default
        + tensor4all_tcicore::MatrixLuciScalar,
{
    /// Construct a `TensorCI2` from an existing tensor train, consuming the train.
    pub fn from_tensor_train(
        tt: TensorTrain<T>,
        options: crate::conversion::TensorCI2FromTensorTrainOptions,
    ) -> Result<Self> {
        crate::conversion::tensorci2_from_tensor_train(tt, options)
    }
}
```

- [ ] **Step 2: Implement `tensorci2_from_tensor_train`**

Use the Julia algorithm in `conversion.jl::TensorCI2(tt)` as the reference:

1. Extract local dimensions from the consumed tensor train.
2. Run forward one-site LU extraction to produce `Iset`.
3. Run backward one-site LU extraction to produce `Jset` and pivot errors.
4. Alternate up to `options.max_iter` until index sets stabilize.
5. Build `TensorCI2` with the extracted index sets and the original site tensors.

- [ ] **Step 3: Add a helper for one-site index extraction**

Keep this helper private to `conversion.rs`:

```rust
fn sweep1site_get_indices<T>(
    tt: &mut TensorTrain<T>,
    forward: bool,
    spectator_indices: Option<&mut [Vec<MultiIndex>]>,
    options: &TensorCI2FromTensorTrainOptions,
) -> Result<(Vec<Vec<MultiIndex>>, Vec<f64>)>
where
    T: tensor4all_tcicore::Scalar
        + tensor4all_simplett::TTScalar
        + Default
        + tensor4all_tcicore::MatrixLuciScalar,
```

- [ ] **Step 4: Run the focused test**

```bash
cargo test --release -p tensor4all-tensorci test_tensorci2_from_tensor_train_preserves_values
```

Expected: PASS.

### Task 3: Add Validation Coverage

**Files:**
- Modify: `crates/tensor4all-tensorci/src/conversion/tests/mod.rs`

- [ ] **Step 1: Add max bond dimension test**

```rust
#[test]
fn test_tensorci2_from_tensor_train_respects_max_bond_dim() {
    let tt = TensorTrain::<f64>::constant(&[2, 2, 2], 1.0);
    let options = TensorCI2FromTensorTrainOptions {
        max_bond_dim: 1,
        ..TensorCI2FromTensorTrainOptions::default()
    };
    let tci = TensorCI2::from_tensor_train(tt, options).unwrap();
    assert!(tci.link_dims().iter().all(|&dim| dim <= 1));
}
```

- [ ] **Step 2: Add complex scalar test**

```rust
#[test]
fn test_tensorci2_from_tensor_train_complex_constant() {
    let value = num_complex::Complex64::new(1.25, -0.5);
    let tt = TensorTrain::<num_complex::Complex64>::constant(&[2, 2], value);
    let tci = TensorCI2::from_tensor_train(tt, TensorCI2FromTensorTrainOptions::default()).unwrap();
    let roundtrip = tci.to_tensor_train().unwrap();
    let actual = roundtrip.evaluate(&[1, 1]).unwrap();
    assert!((actual - value).norm() < 1e-12);
}
```

- [ ] **Step 3: Run conversion tests**

```bash
cargo test --release -p tensor4all-tensorci conversion
```

Expected: PASS.

### Task 4: Add Strict Index-Set Constructor

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`
- Modify: `crates/tensor4all-tensorci/src/conversion/tests/mod.rs`

- [ ] **Step 1: Add failing validation test**

```rust
#[test]
fn test_tensorci2_from_index_sets_rejects_wrong_lengths() {
    let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
    let err = TensorCI2::from_index_sets(vec![2, 2], vec![vec![vec![]]], vec![], &f).unwrap_err();
    assert!(err.to_string().contains("I/J set length"));
}
```

- [ ] **Step 2: Implement constructor**

```rust
pub fn from_index_sets<F>(
    local_dims: Vec<usize>,
    i_set: Vec<Vec<MultiIndex>>,
    j_set: Vec<Vec<MultiIndex>>,
    f: &F,
) -> Result<Self>
where
    F: Fn(&MultiIndex) -> T,
```

Validate lengths, index ranges, and nonzero maximum sample over reconstructed global pivots before returning the new `TensorCI2`.

- [ ] **Step 3: Run validation test**

```bash
cargo test --release -p tensor4all-tensorci test_tensorci2_from_index_sets_rejects_wrong_lengths
```

Expected: PASS.

### Task 5: Docs, API Dump, And Commit

**Files:**
- Modify: `docs/api/tensor4all_tensorci.md`

- [ ] **Step 1: Add rustdoc examples**

Add runnable examples for `TensorCI2FromTensorTrainOptions` and `TensorCI2::from_tensor_train`, each with assertions.

- [ ] **Step 2: Verify**

```bash
cargo fmt --all
cargo test --release -p tensor4all-tensorci conversion
cargo test --doc --release -p tensor4all-tensorci
cargo run -p api-dump --release -- . -o docs/api
```

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-tensorci/src/lib.rs \
  crates/tensor4all-tensorci/src/tensorci2.rs \
  crates/tensor4all-tensorci/src/conversion.rs \
  crates/tensor4all-tensorci/src/conversion/tests/mod.rs \
  docs/api/tensor4all_tensorci.md
git commit -m "feat(tensorci): add TensorCI2 conversion constructors"
```
