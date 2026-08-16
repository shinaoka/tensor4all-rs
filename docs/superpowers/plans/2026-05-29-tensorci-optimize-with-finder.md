> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# TensorCI Optimize With Finder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public TCI2 optimizer entry point that accepts a custom `GlobalPivotFinder`.

**Architecture:** Keep `TCI2Options` as a plain configuration object. Extract the existing `crossinterpolate2` optimization loop into a generic `optimize_with_finder` function, then make `crossinterpolate2` a thin wrapper using `DefaultGlobalPivotFinder`.

**Tech Stack:** Rust, `tensor4all-tensorci`, `tensor4all-simplett`, `tensor4all-tcicore`, release-mode cargo tests.

---

## File Structure

- Modify `crates/tensor4all-tensorci/src/tensorci2.rs`: add `optimize_with_finder`, preserve `crossinterpolate2`.
- Modify `crates/tensor4all-tensorci/src/lib.rs`: re-export `optimize_with_finder`.
- Modify `crates/tensor4all-tensorci/src/tensorci2/tests/mod.rs`: add custom finder regression.
- Modify `docs/api/tensor4all_tensorci.md`: regenerate via `api-dump`, do not edit manually.

### Task 1: Add Failing Custom Finder Test

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci2/tests/mod.rs`

- [ ] **Step 1: Add this failing test**

```rust
#[test]
fn test_optimize_with_finder_invokes_custom_finder() {
    use crate::globalpivot::{GlobalPivotFinder, GlobalPivotSearchInput};
    use std::cell::Cell;
    use std::rc::Rc;

    struct CountingFinder {
        calls: Rc<Cell<usize>>,
    }

    impl GlobalPivotFinder for CountingFinder {
        fn find_global_pivots<T, F>(
            &self,
            input: &GlobalPivotSearchInput<T>,
            _f: &F,
            _abs_tol: f64,
            _rng: &mut impl rand::Rng,
        ) -> Vec<MultiIndex>
        where
            T: tensor4all_tcicore::Scalar + tensor4all_simplett::TTScalar,
            F: Fn(&MultiIndex) -> T,
        {
            self.calls.set(self.calls.get() + 1);
            vec![vec![input.local_dims[0] - 1, input.local_dims[1] - 1]]
        }
    }

    let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 2)) as f64;
    let mut tci = TensorCI2::<f64>::new(vec![4, 4]).unwrap();
    tci.add_global_pivots(&[vec![0, 0]]).unwrap();

    let calls = Rc::new(Cell::new(0));
    let finder = CountingFinder {
        calls: Rc::clone(&calls),
    };
    let (tci, ranks, errors) = optimize_with_finder::<
        f64,
        _,
        fn(&[MultiIndex]) -> Vec<f64>,
        _,
    >(tci, f, None, TCI2Options::default(), finder)
    .unwrap();

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(tci.rank() >= 1);
    assert!(calls.get() > 0);
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
cargo test --release -p tensor4all-tensorci test_optimize_with_finder_invokes_custom_finder
```

Expected: compile failure because `optimize_with_finder` is not defined.

### Task 2: Extract Optimizer

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`

- [ ] **Step 1: Add the public function signature before `crossinterpolate2`**

```rust
pub fn optimize_with_finder<T, F, B, G>(
    mut tci: TensorCI2<T>,
    f: F,
    batched_f: Option<B>,
    options: TCI2Options,
    finder: G,
) -> Result<(TensorCI2<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
    G: GlobalPivotFinder,
```

- [ ] **Step 2: Move the body of the existing `crossinterpolate2` main loop into `optimize_with_finder`**

Keep the logic from after `tci.max_sample_value` initialization through the final `make_canonical`/return path. Use the passed `finder` instead of constructing `DefaultGlobalPivotFinder`.

- [ ] **Step 3: Keep pivot initialization in `crossinterpolate2`**

Make `crossinterpolate2` create `TensorCI2`, add initial pivots, initialize `max_sample_value`, then call:

```rust
let finder = DefaultGlobalPivotFinder::new(
    options.nsearch,
    options.max_nglobal_pivot,
    options.tol_margin_global_search,
);
optimize_with_finder(tci, f, batched_f, options, finder)
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
cargo test --release -p tensor4all-tensorci test_optimize_with_finder_invokes_custom_finder
```

Expected: PASS.

### Task 3: Preserve Existing Behavior

**Files:**
- Modify: `crates/tensor4all-tensorci/src/tensorci2/tests/mod.rs`

- [ ] **Step 1: Run existing TCI2 tests**

Run:

```bash
cargo test --release -p tensor4all-tensorci tensorci2
```

Expected: all TCI2 tests pass.

- [ ] **Step 2: Check moved-loop invariants**

Review the extracted function and confirm these concrete invariants still match
the previous `crossinterpolate2` behavior:

- `ranks.push(tci.rank())` happens once per main iteration after global pivots
  are added.
- `errors.push(error)` records the normalized `tci.max_bond_error()` value.
- `nglobal_pivots_history.push(global_pivots.len())` happens before the
  convergence check.
- `StdRng::seed_from_u64(options.seed)` is used for deterministic runs.
- `StdRng::from_os_rng()` is used when `options.seed` is `None`.
- `tci.make_canonical(...)` remains the final cleanup before returning.

### Task 4: Re-export And Document

**Files:**
- Modify: `crates/tensor4all-tensorci/src/lib.rs`
- Modify: `crates/tensor4all-tensorci/src/tensorci2.rs`

- [ ] **Step 1: Re-export the function**

```rust
pub use tensorci2::{
    crossinterpolate2, optimize_with_finder, PivotSearchStrategy, Sweep2Strategy, TCI2Options,
    TensorCI2,
};
```

- [ ] **Step 2: Add rustdoc example to `optimize_with_finder`**

Use a small 2D function and a custom finder that returns no pivots. Assert that ranks and errors are non-empty and that the output tensor train evaluates a known point.

- [ ] **Step 3: Verify docs compile for this crate**

Run:

```bash
cargo test --doc --release -p tensor4all-tensorci
```

Expected: PASS.

### Task 5: Final Verification And Commit

**Files:**
- Modify: `docs/api/tensor4all_tensorci.md`

- [ ] **Step 1: Format**

```bash
cargo fmt --all
```

- [ ] **Step 2: Run crate tests**

```bash
cargo test --release -p tensor4all-tensorci
```

- [ ] **Step 3: Regenerate API docs**

```bash
cargo run -p api-dump --release -- . -o docs/api
```

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-tensorci/src/lib.rs \
  crates/tensor4all-tensorci/src/tensorci2.rs \
  crates/tensor4all-tensorci/src/tensorci2/tests/mod.rs \
  docs/api/tensor4all_tensorci.md
git commit -m "feat(tensorci): expose optimize_with_finder"
```
