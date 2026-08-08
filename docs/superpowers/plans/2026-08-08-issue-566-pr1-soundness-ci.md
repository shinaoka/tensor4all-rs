# Issue #566 PR 1 Soundness and CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve issue #566 Phase 0 and establish the Phase 1 release-test, panic, documentation, boundary, doctest, housekeeping, and release-coverage enforcement needed for later backlog burn-down.

**Architecture:** Validate all user- and file-derived dimensions at the owning public boundary before allocation, unsafe access, or fallback behavior. Put deterministic, stdlib-only audit scripts in the existing `scripts` CI job, keep known Phase 2/3 backlogs exact and shrinking, and restore release tests as a separate required job. Complete implementation and local review before the first push.

**Tech Stack:** Rust, thiserror/anyhow at existing crate boundaries, C ABI status helpers, HDF5, GitHub Actions, Python 3 standard library audit scripts, cargo-nextest, cargo-llvm-cov.

## Global Constraints

- Work only in `/home/shinaoka/tensor4all/tensor4all-rs/.worktrees/issue-566-remediation` on branch `audit/issue-566-remediation`.
- Read `AGENTS.md`, `README.md`, `REPOSITORY_RULES.md`, relevant `docs/api/*.md`, and `docs/CAPI_DESIGN.md` before touching their owning areas.
- Preserve unrelated user changes; do not modify the original dirty checkout.
- Luna Max is the sole writer. Review agents are read-only.
- Follow strict TDD: establish a failing test or deterministic audit before implementation, then make the minimum root-cause change.
- Use release-mode tests. Run `cargo fmt --all` before every commit.
- Do not lower test tolerances or coverage thresholds.
- Do not add compatibility shims, TODO placeholders, duplicated validation, or new dependencies.
- Every new unsafe block requires a nearby `// SAFETY:` invariant. Prefer existing safe helpers.
- Flat dense buffers remain column-major.
- Check `df -h /home/shinaoka` and `du -sh target` before and after workspace-wide builds, coverage runs, and benchmarks. Remove only regenerable build/coverage artifacts at PR boundaries or when capacity is constrained.
- Update `docs/worklogs/2026-08-08-issue-566-pr1-soundness-ci.md` with decisions, commands, and evidence as tasks land.

---

### Task 1: Enforce Matrix shape and indexing invariants

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/matrix.rs:320-503,772-825`
- Modify: `crates/tensor4all-tensorbackend/src/matrix/tests/mod.rs`

**Interfaces:**
- Consumes: existing infallible `Matrix::from_col_major_vec`, `Matrix::from_elem`, `Matrix::zeros`, and `Index<[usize; 2]>` APIs.
- Produces: private `checked_matrix_len(nrows: usize, ncols: usize) -> usize`; axis-checked private `Matrix::offset`; stable panic diagnostics for overflow and out-of-range axes.

- [ ] **Step 1: Add failing constructor-overflow and axis-aliasing tests**

Add tests with these exact behaviors to `matrix/tests/mod.rs`:

```rust
#[test]
fn matrix_constructors_reject_shape_product_overflow() {
    assert!(std::panic::catch_unwind(|| {
        Matrix::from_col_major_vec(usize::MAX, 2, Vec::<u8>::new())
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| {
        Matrix::from_elem(usize::MAX, 2, 0_u8)
    })
    .is_err());
    assert!(std::panic::catch_unwind(|| Matrix::<u8>::zeros(usize::MAX, 2)).is_err());

    let zero = Matrix::<u8>::zeros(0, usize::MAX);
    assert_eq!(zero.nrows(), 0);
    assert_eq!(zero.ncols(), usize::MAX);
    assert!(zero.as_col_major_slice().is_empty());
}

#[test]
fn matrix_indexing_rejects_each_axis_before_linearization() {
    let mut matrix = Matrix::from_col_major_vec(2, 2, vec![1, 3, 2, 4]);
    assert!(std::panic::catch_unwind(|| matrix[[2, 0]]).is_err());
    assert!(std::panic::catch_unwind(|| matrix[[0, 2]]).is_err());
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        matrix[[2, 0]] = 99;
    }))
    .is_err());
    assert_eq!(matrix.as_col_major_slice(), &[1, 3, 2, 4]);
    matrix[[1, 1]] = 8;
    assert_eq!(matrix[[1, 1]], 8);
}
```

Also add one normal `Complex64` construction/index test using the same column-major order.

- [ ] **Step 2: Run RED tests**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend matrix_constructors_reject_shape_product_overflow
cargo test --release -p tensor4all-tensorbackend matrix_indexing_rejects_each_axis_before_linearization
```

Expected: overflow test fails because multiplication wraps or allocation behavior is wrong; row-alias test fails because `[2, 0]` aliases an in-range element.

- [ ] **Step 3: Add one checked shape helper and axis checks**

Implement the shared helper and route all three constructors through it:

```rust
fn checked_matrix_len(nrows: usize, ncols: usize) -> usize {
    nrows.checked_mul(ncols).unwrap_or_else(|| {
        panic!("matrix shape product overflow: {nrows} rows * {ncols} columns")
    })
}
```

In `from_col_major_vec`, compare `data.len()` with `checked_matrix_len`. In `from_elem` and `zeros`, allocate exactly that checked length. In `offset`, assert `row < self.nrows` and `col < self.ncols` before computing `row + self.nrows * col`; diagnostics must name the bad axis, index, and bound.

- [ ] **Step 4: Run GREEN and sibling regressions**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend matrix
cargo test --doc --release -p tensor4all-tensorbackend
```

Expected: PASS. Inspect `submatrix`, `submatrix_argmax`, and every direct `offset` caller to confirm optimized unchecked paths validate ranges independently.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-tensorbackend/src/matrix.rs crates/tensor4all-tensorbackend/src/matrix/tests/mod.rs
git commit -m "fix(tensorbackend): validate matrix shapes and indices"
```

### Task 2: Validate RRLU invariants before unsafe kernels

**Files:**
- Modify: `crates/tensor4all-tcicore/src/matrixlu.rs:430-540,599-603,713-825`
- Modify: `crates/tensor4all-tcicore/src/matrixlu/tests/mod.rs`

**Interfaces:**
- Consumes: Matrix representation invariant from Task 1 and `MatrixCIError::InvalidArgument`.
- Produces: private `validate_col_major_matrix_len(nrows, ncols, actual_len) -> Result<()>`; public `rrlu` and `rrlu_inplace` validate once before any unchecked helper.

- [ ] **Step 1: Add failing validation-helper tests**

Add module-local tests:

```rust
#[test]
fn rrlu_shape_validation_rejects_overflow_and_length_mismatch() {
    assert!(validate_col_major_matrix_len(usize::MAX, 2, 0)
        .unwrap_err()
        .to_string()
        .contains("overflows"));
    assert!(validate_col_major_matrix_len(2, 2, 3)
        .unwrap_err()
        .to_string()
        .contains("expected 4"));
    validate_col_major_matrix_len(0, usize::MAX, 0).unwrap();
}
```

Add a `Complex64` public `rrlu` happy test that reconstructs the original matrix or checks a known factorization identity. Do not create malformed `Matrix` values with unsafe or test-only constructors.

- [ ] **Step 2: Run RED**

```bash
cargo test --release -p tensor4all-tcicore rrlu_shape_validation_rejects_overflow_and_length_mismatch
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement entry validation**

Use checked multiplication and exact length comparison:

```rust
fn validate_col_major_matrix_len(
    nrows: usize,
    ncols: usize,
    actual_len: usize,
) -> crate::Result<()> {
    let expected = nrows.checked_mul(ncols).ok_or_else(|| {
        MatrixCIError::InvalidArgument {
            message: format!("matrix shape {nrows} x {ncols} overflows usize"),
        }
    })?;
    if actual_len != expected {
        return Err(MatrixCIError::InvalidArgument {
            message: format!("column-major matrix length mismatch: expected {expected}, got {actual_len}"),
        });
    }
    Ok(())
}
```

Call it at the beginning of `rrlu_inplace`, before max-rank shortcuts and before taking unchecked paths. Keep nearby debug assertions inside leaf kernels as invariant documentation and add/retain `// SAFETY:` comments for unsafe blocks.

- [ ] **Step 4: Run GREEN and downstream factorization regressions**

```bash
cargo test --release -p tensor4all-tcicore matrixlu
cargo test --release -p tensor4all-simplett canonical
cargo test --release -p tensor4all-simplett compression
```

Expected: PASS.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-tcicore/src/matrixlu.rs crates/tensor4all-tcicore/src/matrixlu/tests/mod.rs
git commit -m "fix(tcicore): validate RRLU matrix invariants"
```

### Task 3: Propagate quanticstci coordinate errors

**Files:**
- Modify: `crates/tensor4all-quanticstci/src/quantics_tci.rs:458-540,621-770`
- Modify: `crates/tensor4all-quanticstci/src/quantics_tci/tests/mod.rs`

**Interfaces:**
- Consumes: existing `anyhow::Result` public contract and fallible TreeTCI batch callback.
- Produces: `fn evaluate_grid_point<V>(quantics: &[i64], to_coord: impl FnOnce(&[i64]) -> anyhow::Result<Vec<f64>>, evaluate: impl FnOnce(&[f64]) -> V) -> anyhow::Result<V>`; no `V::default()` fallback and no silent invalid-initial-pivot dropping.

- [ ] **Step 1: Add a deterministic failing helper test**

Extracting a private helper for injection is allowed because valid grid metadata makes the corruption branch difficult to reach end-to-end. Add a test-only conversion closure returning an error and assert:

```rust
#[test]
fn grid_evaluation_propagates_coordinate_conversion_failure() {
    let called = std::cell::Cell::new(false);
    let result = evaluate_grid_point(
        &[3_i64],
        |_point| anyhow::bail!("synthetic coordinate failure"),
        |_coord| {
            called.set(true);
            1.0_f64
        },
    );
    let error = result.unwrap_err().to_string();
    assert!(error.contains("synthetic coordinate failure"));
    assert!(error.contains("[3]"));
    assert!(!called.get());
}

#[test]
fn grid_evaluation_passes_converted_coordinates_to_callback() {
    let value = evaluate_grid_point(
        &[1_i64],
        |_point| Ok(vec![0.25]),
        |coord| coord[0] * 4.0,
    )
    .unwrap();
    assert_eq!(value, 1.0);
}
```

Keep the helper private; production and tests use the same function.

- [ ] **Step 2: Run RED**

```bash
cargo test --release -p tensor4all-quanticstci grid_evaluation_propagates_coordinate_conversion_failure
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Make evaluation and initial-pivot conversion fallible**

Change the local point evaluator to return `Result<V>`. Map coordinate errors with the offending quantics index, return cache hits as `Ok(value)`, and collect batch values with `?`. Replace initial-pivot `filter_map(...ok())` with fallible collection so invalid pivots return context instead of disappearing:

```rust
let initial_points = initial_pivots
    .iter()
    .map(|pivot| {
        grid.origcoord_to_quantics(pivot).map_err(|error| {
            anyhow::anyhow!("initial pivot {pivot:?} conversion failed: {error}")
        })
    })
    .collect::<Result<Vec<_>>>()?;
```

Do not introduce the PR 2 typed error enum early.

- [ ] **Step 4: Run GREEN and public interpolation tests**

```bash
cargo test --release -p tensor4all-quanticstci quantics_tci
cargo test --release -p tensor4all-quanticstci --test qft_2d_test
```

Expected: PASS with no `V::default()` corruption path in source.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstci/src/quantics_tci.rs crates/tensor4all-quanticstci/src/quantics_tci/tests/mod.rs
git commit -m "fix(quanticstci): propagate coordinate conversion errors"
```

### Task 4: Check all tensor C API lengths before pointer use

**Files:**
- Read first: `docs/CAPI_DESIGN.md`
- Modify: `crates/tensor4all-capi/src/tensor.rs:75-140,789-845`
- Modify: `crates/tensor4all-capi/src/tensor/tests/mod.rs`

**Interfaces:**
- Consumes: `capi_error`, `T4A_INVALID_ARGUMENT`, `read_c64_slice`, and `run_status`.
- Produces: `checked_dims_product` and checked raw-slice byte-length validation shared by f64/c64 constructors.

- [ ] **Step 1: Add failing FFI status tests**

Add tests following existing out-pointer and `t4a_last_error_message` patterns:

```rust
#[test]
fn dense_constructor_rejects_dimension_product_overflow() {
    let huge = new_index(usize::MAX);
    let two = new_index(2);
    let indices = [huge as *const t4a_index, two as *const t4a_index];

    let mut real = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_f64(
            indices.len(),
            indices.as_ptr(),
            std::ptr::null(),
            0,
            &mut real,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(real.is_null());
    assert!(last_error().contains("dimension product"));

    let mut complex = std::ptr::null_mut();
    assert_eq!(
        t4a_tensor_new_dense_c64(
            indices.len(),
            indices.as_ptr(),
            std::ptr::null(),
            0,
            &mut complex,
        ),
        T4A_INVALID_ARGUMENT
    );
    assert!(complex.is_null());
    assert!(last_error().contains("dimension product"));

    t4a_index_release(two);
    t4a_index_release(huge);
}

#[test]
fn c64_reader_rejects_interleaved_length_overflow_before_dereference() {
    let error = read_c64_slice("data_interleaved", std::ptr::NonNull::<f64>::dangling().as_ptr(), usize::MAX)
        .unwrap_err();
    assert_eq!(error.0, T4A_INVALID_ARGUMENT);
    assert!(error.1.contains("overflows"));
}
```

Also test generic raw-slice byte limit (`len > isize::MAX / size_of::<T>()`) and retain null-pointer precedence for a valid positive length.

- [ ] **Step 2: Run RED**

```bash
cargo test --release -p tensor4all-capi dense_constructor_rejects_dimension_product_overflow
cargo test --release -p tensor4all-capi c64_reader_rejects_interleaved_length_overflow_before_dereference
```

Expected: FAIL because dimension products/raw-slice sizes are unchecked.

- [ ] **Step 3: Centralize checked arithmetic**

Implement:

```rust
fn checked_dims_product(name: &str, dims: &[usize]) -> CapiResult<usize> {
    dims.iter().try_fold(1usize, |product, &dim| {
        product.checked_mul(dim).ok_or_else(|| {
            capi_error(
                T4A_INVALID_ARGUMENT,
                format!("{name} dimension product overflows usize for dims {dims:?}"),
            )
        })
    })
}
```

Before every `from_raw_parts`, reject element counts whose byte length exceeds `isize::MAX`. Use `read_c64_slice` instead of recreating interleaved conversion in the public c64 constructor. Preserve detailed errors through `run_status`; do not add `catch_unwind` that discards messages.

- [ ] **Step 4: Run GREEN and C API regressions**

```bash
cargo test --release -p tensor4all-capi tensor
cargo test --release -p tensor4all-capi
```

If exported signatures changed unexpectedly, stop; this task should require no header change.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-capi/src/tensor.rs crates/tensor4all-capi/src/tensor/tests/mod.rs
git commit -m "fix(capi): check tensor dimensions before pointer use"
```

### Task 5: Validate HDF5 integers before conversion and allocation

**Files:**
- Modify: `crates/tensor4all-hdf5/src/index.rs:119-208`
- Modify: `crates/tensor4all-hdf5/src/index/tests/mod.rs`
- Modify: `crates/tensor4all-hdf5/src/mps.rs:100-147`
- Modify: `crates/tensor4all-hdf5/tests/test_hdf5.rs:230-359`

**Interfaces:**
- Consumes: current ITensors-compatible dataset schema.
- Produces: private checked file-integer conversion helpers with dataset/value context; allocation only after consistency with actual group members is established.

- [ ] **Step 1: Add corrupt-file regression tests**

Using temporary HDF5 files and current writer schema, mutate/create datasets and assert each read returns `Err`, never panics:

```rust
#[test]
fn negative_index_dimension_is_rejected() {
    let path = temp_path("negative_index_dim");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("tensor/inds/index_1")
        .unwrap()
        .dataset("dim")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);
    let error = load_itensor(&path, "tensor").unwrap_err().to_string();
    assert!(error.contains("dim"));
    assert!(error.contains("-1"));
    std::fs::remove_file(path).ok();
}

#[test]
fn negative_index_set_length_is_rejected() {
    let path = temp_path("negative_index_set_length");
    save_itensor(&path, "tensor", &make_test_tensor_f64()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("tensor/inds")
        .unwrap()
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);
    assert!(load_itensor(&path, "tensor")
        .unwrap_err()
        .to_string()
        .contains("length"));
    std::fs::remove_file(path).ok();
}

#[test]
fn negative_mps_length_is_rejected() {
    let path = temp_path("negative_mps_length");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    file.group("mps")
        .unwrap()
        .dataset("length")
        .unwrap()
        .as_writer()
        .write_scalar(&-1_i64)
        .unwrap();
    drop(file);
    assert!(load_mps(&path, "mps")
        .unwrap_err()
        .to_string()
        .contains("length"));
    std::fs::remove_file(path).ok();
}

#[test]
fn oversized_mps_limits_are_rejected() {
    let path = temp_path("oversized_mps_limits");
    save_mps(&path, "mps", &make_test_mps()).unwrap();
    let file = File::open_rw(&path).unwrap();
    let group = file.group("mps").unwrap();
    group
        .dataset("llim")
        .unwrap()
        .as_writer()
        .write_scalar(&i64::MAX)
        .unwrap();
    group
        .dataset("rlim")
        .unwrap()
        .as_writer()
        .write_scalar(&i64::MIN)
        .unwrap();
    drop(file);
    let error = load_mps(&path, "mps").unwrap_err().to_string();
    assert!(error.contains("llim") || error.contains("rlim"));
    std::fs::remove_file(path).ok();
}
```

Add a mismatch test where positive `length` exceeds available `Index_N`/`MPS[N]` child groups; it must error before `Vec::with_capacity(length)` can reserve attacker-controlled memory. Retain valid zero-length behavior only if the existing schema supports it.

- [ ] **Step 2: Run RED with cargo test, not nextest**

```bash
cargo test --release -p tensor4all-hdf5 negative_index_dimension_is_rejected
cargo test --release -p tensor4all-hdf5 negative_mps_length_is_rejected
```

Expected: current direct casts fail the assertions or attempt invalid capacity.

- [ ] **Step 3: Convert and cross-check before allocation**

Use `usize::try_from` and `i32::try_from` immediately after reading signed datasets:

```rust
fn read_nonnegative_usize(name: &'static str, value: i64) -> anyhow::Result<usize> {
    usize::try_from(value)
        .with_context(|| format!("HDF5 dataset {name} must be non-negative, got {value}"))
}
```

Before allocating from a file length, compare the length to the actual expected child-group names/member count. Validate `llim` and `rlim` after checked conversion and before constructing the MPS. Audit all other reader-side `as usize`/`as i32` conversions in `tensor4all-hdf5`; fix siblings with the same untrusted-file root cause in this commit.

- [ ] **Step 4: Run GREEN and round trips**

```bash
cargo test --release -p tensor4all-hdf5
```

Expected: PASS for corrupt files and valid ITensors-compatible round trips.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-hdf5/src/index.rs crates/tensor4all-hdf5/src/index/tests/mod.rs crates/tensor4all-hdf5/src/mps.rs crates/tensor4all-hdf5/tests/test_hdf5.rs
git commit -m "fix(hdf5): validate file-derived dimensions"
```

### Task 6: Centralize multivariable shift-width validation

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/common.rs:452-495`
- Modify: `crates/tensor4all-quanticstransform/src/shift.rs:81-175`
- Modify: `crates/tensor4all-quanticstransform/src/shift/tests/mod.rs`
- Inspect and modify if affected: `crates/tensor4all-quanticstransform/src/affine.rs:500-525`

**Interfaces:**
- Consumes: all multivariable operator constructors and `embed_single_var_mpo`.
- Produces: private `checked_multivar_dims(nvariables) -> Result<(usize, usize)>`, returning local dimension and squared site dimension without overflow.

- [ ] **Step 1: Add failing boundary tests**

```rust
#[test]
fn shift_multivar_rejects_usize_shift_width() {
    let error = shift_operator_multivar(
        1,
        0,
        BoundaryCondition::Periodic,
        usize::BITS as usize,
        0,
    )
    .unwrap_err();
    assert!(error.to_string().contains("nvariables"));
}

#[test]
fn multivar_dims_reject_squared_site_dimension_overflow() {
    let nvariables = usize::BITS as usize / 2;
    assert!(checked_multivar_dims(nvariables)
        .unwrap_err()
        .to_string()
        .contains("site dimension"));
}
```

Keep existing tests for `nvariables < 2`, invalid target, and two-variable behavior.

- [ ] **Step 2: Run RED**

```bash
cargo test --release -p tensor4all-quanticstransform shift_multivar_rejects_usize_shift_width
```

Expected: debug panic/wrapped release behavior rather than `Err`.

- [ ] **Step 3: Implement one shared checked helper**

Use `checked_shl` and `checked_mul`:

```rust
fn checked_multivar_dims(nvariables: usize) -> Result<(usize, usize)> {
    if nvariables < 2 {
        anyhow::bail!("nvariables must be at least 2, got {nvariables}");
    }
    let shift = u32::try_from(nvariables).map_err(|_| {
        anyhow::anyhow!("nvariables {nvariables} exceeds usize shift width")
    })?;
    let local_dim = 1usize.checked_shl(shift).ok_or_else(|| {
        anyhow::anyhow!("nvariables {nvariables} exceeds usize shift width")
    })?;
    let site_dim = local_dim.checked_mul(local_dim).ok_or_else(|| {
        anyhow::anyhow!("multi-variable site dimension overflows usize")
    })?;
    Ok((local_dim, site_dim))
}
```

Reuse it from `embed_single_var_mpo`, `shift_operator_multivar`, and sibling multivariable operators. Audit affine shifts/products at the cited lines and route identical width arithmetic through checked helpers rather than leaving a related bug.

- [ ] **Step 4: Run GREEN and binding regressions**

```bash
cargo test --release -p tensor4all-quanticstransform shift
cargo test --release -p tensor4all-quanticstransform --test integration_test
cargo test --release -p tensor4all-capi quanticstransform
```

Expected: PASS and no unchecked `1usize << nvariables`/`1 << nvariables` remains.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/common.rs crates/tensor4all-quanticstransform/src/shift.rs crates/tensor4all-quanticstransform/src/shift/tests/mod.rs crates/tensor4all-quanticstransform/src/affine.rs
git commit -m "fix(quanticstransform): validate multivariable dimensions"
```

### Task 7: Restore always-on release tests

**Files:**
- Modify: `.github/workflows/CI_rs.yml:59-88,121-128`

**Interfaces:**
- Produces: required `test` job for pushes and pull requests to `main` and `develop`; rollup depends on it.

- [ ] **Step 1: Add a deterministic workflow contract check**

Create a small fixture assertion in the existing maintenance-script test style or use a Python one-liner in local validation that parses text and verifies all of these strings are active, not commented:

```text
test:
cargo nextest run --release --workspace --exclude tensor4all-hdf5
cargo test --release -p tensor4all-hdf5
- test
```

Run it before changing the workflow; expected result is failure because the job is commented.

- [ ] **Step 2: Restore the existing job and rollup dependency**

Uncomment the existing test job without changing its release commands. Keep HDF5 outside nextest and execute it with cargo test. Add `test` to `rollup-rs.needs`.

- [ ] **Step 3: Validate syntax and commands locally**

```bash
actionlint .github/workflows/CI_rs.yml
cargo nextest run --release --workspace --exclude tensor4all-hdf5
cargo test --release -p tensor4all-hdf5
```

If `actionlint` is unavailable, run the repository's documented workflow linter or install/use the existing tool without adding a project dependency.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/CI_rs.yml
git commit -m "ci: restore release workspace tests"
```

### Task 8: Make the library-panic audit blocking and source-aware

**Files:**
- Modify: `scripts/audit-library-panics.py`
- Create: `scripts/test-audit-library-panics.py`
- Create if an audited baseline is necessary: `scripts/library-panics-baseline.json`
- Modify the six current production hit owners as classified by the audit
- Modify: `.github/workflows/CI_rs.yml`

**Interfaces:**
- Produces: deterministic scanner that rejects production `panic!`, `unreachable!`, raw `unwrap`/`expect`, and public-path `assert!`/`debug_assert!`; exact baseline entries are shrinking and stale entries fail.

- [ ] **Step 1: Write fixture-based scanner tests**

Use `tempfile.TemporaryDirectory` and subprocess invocation. Cover production hit failure, test-module/file exclusion, comments/rustdoc exclusion, public function assertions, private helper classification, matching baseline, new finding, and stale baseline. Assertions must check exit codes and normalized `path:line:kind` output.

- [ ] **Step 2: Run RED**

```bash
python3 scripts/test-audit-library-panics.py
```

Expected: FAIL because source-aware assertion handling and baseline semantics do not exist.

- [ ] **Step 3: Implement the scanner and fix all six raw panic hits**

Classify and remove/fix:

```text
crates/tensor4all-core/src/defaults/tensordynlen.rs:2354
crates/tensor4all-simplett/src/mpo/test_support.rs:40
crates/tensor4all-tensorbackend/src/context.rs:115
crates/tensor4all-tensorbackend/src/matrix.rs:901
crates/tensor4all-treetn/src/linsolve/square/projected_state.rs:94
crates/tensor4all-treetn/src/tdvp/plan.rs:176
```

Exclude `test_support.rs` structurally only if Cargo compilation proves it is test-only; otherwise replace the `expect`. For public-path assertions, use a conservative lexical public-function/method scanner with fixture coverage and an exact reviewed baseline for pre-existing assertions. Never baseline the six raw panic-style hits. Make stale baseline entries fail so backlog removal is enforced.

- [ ] **Step 4: Wire self-test and blocking audit into `jobs.scripts`**

Add:

```yaml
- name: Test library panic audit
  run: python3 scripts/test-audit-library-panics.py
- name: Audit library panic paths
  run: python3 scripts/audit-library-panics.py
```

- [ ] **Step 5: Run GREEN and affected crate tests**

```bash
python3 scripts/test-audit-library-panics.py
python3 scripts/audit-library-panics.py
cargo test --release -p tensor4all-core
cargo test --release -p tensor4all-tensorbackend
cargo test --release -p tensor4all-treetn
```

Expected: all pass; scanner output reports zero unbaselined findings and zero stale baselines.

- [ ] **Step 6: Format and commit**

```bash
cargo fmt --all
git add scripts/audit-library-panics.py scripts/test-audit-library-panics.py scripts/library-panics-baseline.json .github/workflows/CI_rs.yml crates/
git commit -m "ci: enforce library panic safety"
```

Stage only files actually changed for this task; do not use a broad `crates/` add if unrelated modifications appear.

### Task 9: Add incremental public-error-doc and crate-boundary gates

**Files:**
- Create: `scripts/check-public-error-docs.py`
- Create: `scripts/test-check-public-error-docs.py`
- Create: `scripts/check-crate-boundaries.py`
- Create: `scripts/test-check-crate-boundaries.py`
- Modify: `crates/tensor4all-tcicore/Cargo.toml` and test ownership needed to remove the tcicore↔tensorci dev cycle
- Modify: `.github/workflows/CI_rs.yml`

**Interfaces:**
- Consumes: upstream tenferro script shapes under `/home/shinaoka/tensor4all/tenferro-rs/scripts/`.
- Produces: `check-public-error-docs.py --changed-from REV`; exact shrinking tenferro dependency exceptions; repository-wide dev-cycle rejection.

- [ ] **Step 1: Port and extend public-error-doc fixture tests**

Copy the upstream behavior, then add repository-specific synthetic-git tests: an undocumented pre-existing API is ignored in changed mode; an added undocumented `Result` API fails; a concrete `# Errors` section passes; deleted files do not trigger whole-repo fallback; missing base commit fails loudly.

- [ ] **Step 2: Write crate-boundary fixture tests**

Temporary manifests must cover: tensorbackend normal tenferro dependency passes; a new feature-crate normal dependency fails; renamed `package = "tenferro-*"` fails; dev-only tenferro dependency is allowed; acyclic dev graph passes; the tcicore→tensorci(dev)→tcicore(normal) cycle fails with the full path; stale exception fails.

- [ ] **Step 3: Run RED**

```bash
python3 scripts/test-check-public-error-docs.py
python3 scripts/test-check-crate-boundaries.py
```

Expected: missing scripts or repository-specific cases fail.

- [ ] **Step 4: Implement exact incremental policies**

Use Python `tomllib`. Allow normal tenferro dependencies only in tensorbackend, plus exact temporary exception tuples for current core, tcicore, simplett, and treetci dependencies. Reject new tuples and stale exceptions. Permit direct tenferro dev-dependencies as test fixtures. Remove/rehome the current tcicore test/dev dependency on tensorci before enabling repository-wide cycle rejection.

Do not add broad clippy `missing_errors_doc`/`missing_panics_doc` denies yet: PR 2 removes the public error/doc backlog and then activates those repository-wide denies. PR 1 must block newly added undocumented APIs through changed-from mode.

- [ ] **Step 5: Wire CI with full base history**

Set `fetch-depth: 0` for the scripts checkout. Resolve the base SHA from `github.event.pull_request.base.sha` on PRs and `github.event.before` on pushes. Add self-tests, changed-from docs check, and repository-wide boundary check. A missing/unavailable base must fail, not silently pass.

- [ ] **Step 6: Run GREEN**

```bash
python3 scripts/test-check-public-error-docs.py
python3 scripts/test-check-crate-boundaries.py
python3 scripts/check-crate-boundaries.py
cargo metadata --format-version 1 --no-deps >/dev/null
```

Also run the docs checker against the local merge base with `origin/main` and confirm only changed APIs are evaluated.

- [ ] **Step 7: Commit**

```bash
git add scripts/check-public-error-docs.py scripts/test-check-public-error-docs.py scripts/check-crate-boundaries.py scripts/test-check-crate-boundaries.py .github/workflows/CI_rs.yml crates/tensor4all-tcicore/Cargo.toml
git commit -m "ci: enforce incremental API docs and crate boundaries"
```

Include any moved test file required to remove the dev cycle.

### Task 10: Replace prohibited doctests and remove kryst

**Files:**
- Modify: `Cargo.toml`, `Cargo.lock`
- Modify: `crates/tensor4all-treetn/Cargo.toml`
- Modify: `crates/tensor4all-treetn/src/linsolve/square/mod.rs:1-23,104-139`
- Modify: `crates/tensor4all-treetn/src/treetn/partial_contraction.rs:808-866`

**Interfaces:**
- Produces: no live `kryst` dependency/claim and no `ignore`/`no_run` fences; examples assert numerical behavior.

- [ ] **Step 1: Prove kryst is unused and record RED documentation state**

```bash
cargo tree -i kryst --workspace
git grep -n kryst -- ':!docs/superpowers/specs/**' ':!docs/superpowers/plans/**'
git grep -nE '```(ignore|no_run)' -- '*.rs' 'docs/**/*.md'
```

Expected: treetn is the sole kryst consumer declaration and exactly two prohibited fences are reported.

- [ ] **Step 2: Replace examples with runnable numerical assertions**

For `partial_contract`, replace the weak node-count assertion with a one-time dense scalar extraction:

```rust
let result = partial_contract(&a, &b, &spec, &0usize, ContractionOptions::default()).unwrap();
let scalar = result.contract_to_tensor().unwrap().only().unwrap().real();
assert!((scalar - 11.0).abs() < 1.0e-12);
```

Replace the `square_linsolve` example's invalid one-index operator with a one-site identity operator and explicit mappings:

```rust
use std::collections::HashMap;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::{square_linsolve, IndexMapping, LinsolveOptions, TreeTN};

# fn main() -> anyhow::Result<()> {
let site = DynIndex::new_dyn(2);
let s_in = DynIndex::new_dyn(2);
let s_out = DynIndex::new_dyn(2);
let operator_tensor = TensorDynLen::from_dense(
    vec![s_out.clone(), s_in.clone()],
    vec![1.0_f64, 0.0, 0.0, 1.0],
)?;
let rhs_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![1.0_f64, 2.0])?;
let init_tensor = TensorDynLen::from_dense(vec![site.clone()], vec![0.0_f64, 0.0])?;
let operator = TreeTN::<TensorDynLen, usize>::from_tensors(vec![operator_tensor], vec![0])?;
let rhs = TreeTN::<TensorDynLen, usize>::from_tensors(vec![rhs_tensor], vec![0])?;
let init = TreeTN::<TensorDynLen, usize>::from_tensors(vec![init_tensor], vec![0])?;
let mut input_mapping = HashMap::new();
input_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_in });
let mut output_mapping = HashMap::new();
output_mapping.insert(0usize, IndexMapping { true_index: site, internal_index: s_out });
let result = square_linsolve(
    &operator,
    &rhs,
    init,
    &0usize,
    LinsolveOptions::default(),
    Some(input_mapping),
    Some(output_mapping),
)?;
assert!(result.residual.unwrap() < 1.0e-10);
let node = result.solution.node_index(&0usize).unwrap();
assert_eq!(result.solution.tensor(node).unwrap().to_vec::<f64>()?, vec![1.0, 2.0]);
# Ok(())
# }
```

- [ ] **Step 3: Remove kryst and correct documentation**

Delete both workspace/crate dependency declarations. Update linsolve docs to name `tensor4all_core::krylov::gmres`. Regenerate `Cargo.lock` through Cargo, not manual edits.

- [ ] **Step 4: Run GREEN**

```bash
cargo test --doc --release -p tensor4all-treetn
cargo test --doc --release --workspace
cargo test --release -p tensor4all-treetn linsolve
test -z "$(git grep -nE '```(ignore|no_run)' -- '*.rs' 'docs/**/*.md' || true)"
! cargo tree -i kryst --workspace
```

Expected: all pass/no matches.

- [ ] **Step 5: Format and commit**

```bash
cargo fmt --all
git add Cargo.toml Cargo.lock crates/tensor4all-treetn/Cargo.toml crates/tensor4all-treetn/src/linsolve/square/mod.rs crates/tensor4all-treetn/src/treetn/partial_contraction.rs
git commit -m "docs(treetn): make solver examples runnable"
```

### Task 11: Remove audited debris and preserve live decisions

**Files:**
- Delete: `debug.md`
- Delete: `plan/`
- Delete: `internal/tenferro-internal-ad-linalg/tests/eager_dyn_extra.rs`
- Delete: `coverage-local.json`
- Create/modify only if live rationale is found: focused files under `docs/design/` or `docs/worklogs/`

**Interfaces:**
- Produces: no committed machine-local/generated debris; any still-normative decision is moved to its owning durable document.

- [ ] **Step 1: Audit every deletion candidate**

For each file, inspect `git log -- <path>`, headings, unique API names, linked issue/PR state, and `git grep` references. Confirm the orphan Rust file is outside all workspace members and no `include!`/path dependency names it. Confirm no script/workflow consumes `coverage-local.json`.

- [ ] **Step 2: Move only still-live decisions**

If a plan contains a constraint still implemented but undocumented elsewhere, add that exact invariant to the appropriate `docs/design/*.md` and link its source/tests. Do not preserve historical task lists or debugging transcripts.

- [ ] **Step 3: Delete debris and run absence checks**

```bash
rm -rf debug.md plan coverage-local.json internal/tenferro-internal-ad-linalg/tests/eager_dyn_extra.rs
test ! -e debug.md
test ! -e plan
test ! -e coverage-local.json
test ! -e internal/tenferro-internal-ad-linalg/tests/eager_dyn_extra.rs
cargo metadata --format-version 1 --no-deps >/dev/null
```

- [ ] **Step 4: Run focused tests for any migrated design owner and commit**

```bash
git add -A debug.md plan coverage-local.json internal/tenferro-internal-ad-linalg/tests/eager_dyn_extra.rs docs/design docs/worklogs
git commit -m "chore: remove committed audit debris"
```

Do not run `git add -A` from outside the isolated worktree.

### Task 12: Switch coverage to release and document threshold rationale

**Files:**
- Modify: `.github/workflows/CI_rs.yml:97-119`
- Modify: `coverage-thresholds.json`
- Modify: `scripts/check-coverage.py`
- Create: `scripts/test-check-coverage.py`

**Interfaces:**
- Produces: release-mode llvm-cov; machine-readable rationale clusters that do not alter numeric enforcement; tested coverage parser.

- [ ] **Step 1: Add coverage-checker fixture tests**

Tests must prove default threshold pass/fail, exact per-file override, missing-file behavior, and that top-level `_comment_*` rationale keys are ignored for enforcement.

- [ ] **Step 2: Run RED**

```bash
python3 scripts/test-check-coverage.py
```

Expected: FAIL for the new rationale/fixture contract until implementation is complete.

- [ ] **Step 3: Add rationale clusters without changing numbers**

Add `_comment_tooling`, `_comment_hdf5`, `_comment_capi`, `_comment_expensive_algorithms`, and other narrowly justified strings. Each string states why release llvm-cov misses the path, what deterministic test/artifact covers it, and when the exception can be removed. Keep default 75 and every per-file threshold unchanged unless tests raise coverage; never lower a number.

- [ ] **Step 4: Change CI to release coverage**

Use:

```yaml
- name: Generate release coverage JSON
  run: cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
```

Remove or replace debug-profile-only environment settings. Wire `scripts/test-check-coverage.py` into `jobs.scripts`.

- [ ] **Step 5: Run release coverage and fix deficits with tests**

Check disk first, then run:

```bash
df -h /home/shinaoka
du -sh target || true
cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
python3 scripts/test-check-coverage.py
```

If release coverage falls below an existing threshold, add meaningful path tests. Do not reduce thresholds. Record before/after percentages in the work log.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/CI_rs.yml coverage-thresholds.json scripts/check-coverage.py scripts/test-check-coverage.py
git commit -m "ci: enforce release-mode coverage"
```

### Task 13: Integrate, review, verify, and prepare the single push

**Files:**
- Create/update: `docs/worklogs/2026-08-08-issue-566-pr1-soundness-ci.md`
- Modify as needed: `docs/superpowers/specs/2026-08-08-issue-566-remediation-design.md`
- No unrelated production edits.

**Interfaces:**
- Produces: locally verified PR 1 branch with independent review evidence and a Phase 0/1 issue ledger.

- [ ] **Step 1: Update the work log and evidence ledger**

Record files read, related-bug audits, decisions, rejected alternatives, every validation command/result, coverage percentages, disk cleanup, and residual risks. Map all eight Phase 0 items and seven incomplete Phase 1 items to commits/tests. Mark raw Clippy denies and full boundary enforcement as scheduled activation in PR 2, not as completed in #566 yet.

- [ ] **Step 2: Run specification-compliance review**

Dispatch a fresh read-only reviewer against issue #566, the design, this plan, and `origin/main...HEAD`. Require exact missing/extra scope findings with file/line evidence. Luna Max fixes every confirmed finding, then rerun affected tests.

- [ ] **Step 3: Run independent correctness/code-quality review**

Use a different model family from Luna Max when available. Focus on safe-public-API reachability, checked arithmetic ordering, FFI slice validity, corrupted HDF5 inputs, workflow gates, false positives/negatives in audit scripts, and test quality. Luna Max fixes confirmed findings.

- [ ] **Step 4: Run complete local PR validation**

```bash
df -h /home/shinaoka
du -sh target || true
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace --exclude tensor4all-hdf5
cargo test --release -p tensor4all-hdf5
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
python3 scripts/test-audit-library-panics.py
python3 scripts/audit-library-panics.py
python3 scripts/test-check-public-error-docs.py
python3 scripts/test-check-crate-boundaries.py
python3 scripts/check-crate-boundaries.py
python3 scripts/test-check-coverage.py
python3 scripts/test-repository-rules-review.py
./scripts/repository-rules-review.py --base origin/main --worktree --dry-run
cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: every command exits 0. Inspect docs/API output and the final diff, not only exit codes.

- [ ] **Step 5: Synchronize with current main and revalidate if needed**

```bash
git fetch origin
git merge-base --is-ancestor origin/main HEAD
```

If false, integrate current `origin/main`, resolve conflicts without discarding work, and rerun the complete validation suite. Green results from before synchronization are insufficient.

- [ ] **Step 6: Commit final docs and confirm clean tree**

```bash
cargo fmt --all
git add docs/worklogs/2026-08-08-issue-566-pr1-soundness-ci.md docs/superpowers/specs/2026-08-08-issue-566-remediation-design.md
git commit -m "docs: record issue 566 phase 0 evidence"
git status --short
```

Expected: no source changes remain unstaged/uncommitted; only ignored runtime artifacts may exist.

- [ ] **Step 7: Parent push and PR actions**

The parent—not Luna Max—pushes once, creates the PR referencing #566, enables squash auto-merge, and monitors CI. Do not close #566 after PR 1. After merge, update the evidence ledger, clean regenerable build artifacts if warranted, fetch `origin/main`, and create the next isolated worktree for the shared-rules/PR 2 plan.
