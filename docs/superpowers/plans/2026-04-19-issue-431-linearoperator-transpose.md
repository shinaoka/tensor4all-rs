# Issue #431 — LinearOperator::transpose & affine pullback unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `LinearOperator::transpose()` in Rust (O(1) swap of input/output mappings) and eliminate the duplicated pullback path from both Rust (`affine_pullback_operator`) and the C API (`t4a_qtransform_affine_pullback_materialize`). The Julia side (Tensor4all.jl) will derive pullback by calling its own pure-Julia `transpose` on the forward operator in a follow-up PR.

**Architecture:** The pullback of an affine operator is mathematically the transpose of the forward operator. Since both already rely on the same underlying MPO (`affine_transform_mpo`) and differ only in how site indices are permuted to match `LinearOperator`'s `(input, output)` convention, swapping `input_mapping` and `output_mapping` on the forward operator is identical to building the pullback directly. We expose `transpose` as the single seam and delete the redundant code paths.

**Tech Stack:** Rust (2021 edition, workspace), cargo nextest, cbindgen (generates `tensor4all_capi.h`), mdarray, anyhow.

**Spec:** `docs/superpowers/specs/2026-04-19-issue-431-linearoperator-transpose-design.md`.

---

### Task 1: Add `LinearOperator::transpose()` with unit tests

**Files:**
- Modify: `crates/tensor4all-treetn/src/operator/linear_operator.rs`
- Modify: `crates/tensor4all-treetn/src/operator/linear_operator/tests/mod.rs`

- [ ] **Step 1: Write the failing unit tests**

Append to `crates/tensor4all-treetn/src/operator/linear_operator/tests/mod.rs` after the existing tests:

```rust
#[test]
fn test_linear_operator_transpose_swaps_mappings() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Snapshot the expected swapped mappings before consuming `op`.
    let original_input = op.input_mapping.clone();
    let original_output = op.output_mapping.clone();

    let transposed = op.transpose();

    // After transpose, the new input_mapping is the original output_mapping,
    // and vice versa.
    assert_eq!(transposed.input_mapping.len(), original_output.len());
    assert_eq!(transposed.output_mapping.len(), original_input.len());
    let tin = transposed
        .input_mapping
        .get("A")
        .expect("transposed input mapping for A");
    let tout = transposed
        .output_mapping
        .get("A")
        .expect("transposed output mapping for A");
    let expected_in = original_output.get("A").unwrap();
    let expected_out = original_input.get("A").unwrap();
    assert!(tin.true_index.same_id(&expected_in.true_index));
    assert!(tin.internal_index.same_id(&expected_in.internal_index));
    assert!(tout.true_index.same_id(&expected_out.true_index));
    assert!(tout.internal_index.same_id(&expected_out.internal_index));
}

#[test]
fn test_linear_operator_transpose_is_involutive() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let original_input = op.input_mapping.clone();
    let original_output = op.output_mapping.clone();

    let round_trip = op.transpose().transpose();

    // Double transpose == identity on mappings.
    assert_eq!(round_trip.input_mapping.len(), original_input.len());
    assert_eq!(round_trip.output_mapping.len(), original_output.len());
    let rin = round_trip.input_mapping.get("A").unwrap();
    let rout = round_trip.output_mapping.get("A").unwrap();
    let ein = original_input.get("A").unwrap();
    let eout = original_output.get("A").unwrap();
    assert!(rin.true_index.same_id(&ein.true_index));
    assert!(rin.internal_index.same_id(&ein.internal_index));
    assert!(rout.true_index.same_id(&eout.true_index));
    assert!(rout.internal_index.same_id(&eout.internal_index));
}

#[test]
fn test_linear_operator_transpose_preserves_mpo() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();
    let original_node_count = op.mpo().node_count();

    let transposed = op.transpose();

    assert_eq!(transposed.mpo().node_count(), original_node_count);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn test_linear_operator_transpose
```

Expected: FAIL with "no method named `transpose` found for struct `LinearOperator`".

- [ ] **Step 3: Add the `transpose()` method and rustdoc**

In `crates/tensor4all-treetn/src/operator/linear_operator.rs`, inside the existing
`impl<T, V> LinearOperator<T, V>` block with the same generic bounds as `new`
(around line 78, the block that contains `new`, `from_mpo_and_state`, `apply_local`,
`mpo`, etc.), add after the `align_to_state` method (before the closing `}` of
that impl, around line 440):

```rust
    /// Returns the transposed operator by swapping input and output mappings.
    ///
    /// The pullback of a forward operator is its transpose: if the forward
    /// operator realizes the matrix `M_{y,x}`, the transposed operator
    /// realizes `M_{x,y}`. This method swaps `input_mapping` and
    /// `output_mapping` without copying the underlying MPO tensors — it is
    /// an O(1) operation.
    ///
    /// `.transpose().transpose()` yields an operator equivalent to the
    /// original (mappings restored, MPO unchanged).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IndexLike, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let site_in = DynIndex::new_dyn(2);
    /// let site_out = DynIndex::new_dyn(3);
    /// let s_in_tmp = DynIndex::new_dyn(2);
    /// let s_out_tmp = DynIndex::new_dyn(3);
    ///
    /// let mpo_tensor = IdxTensor::from_dense(
    ///     vec![s_in_tmp.clone(), s_out_tmp.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<_, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: site_in.clone(), internal_index: s_in_tmp.clone() },
    /// );
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: site_out.clone(), internal_index: s_out_tmp.clone() },
    /// );
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// let t = op.transpose();
    ///
    /// // Input/output mappings are swapped.
    /// assert!(t.input_mapping[&0].true_index.same_id(&site_out));
    /// assert!(t.output_mapping[&0].true_index.same_id(&site_in));
    /// ```
    pub fn transpose(self) -> Self {
        Self {
            mpo: self.mpo,
            input_mapping: self.output_mapping,
            output_mapping: self.input_mapping,
        }
    }
```

Note: the bounds on the containing `impl` block already cover the types that
`transpose` needs. If rustc reports missing bounds, add them to a fresh `impl`
block instead — the method needs no more than `T: TensorLike`, `V: Clone + Hash
+ Eq + Send + Sync + Debug`.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
cargo nextest run --release -p tensor4all-treetn test_linear_operator_transpose
cargo test --doc --release -p tensor4all-treetn
```

Expected: All three unit tests PASS and the doc example compiles and runs.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-treetn/src/operator/linear_operator.rs \
        crates/tensor4all-treetn/src/operator/linear_operator/tests/mod.rs
git commit -m "feat(treetn): add LinearOperator::transpose (O(1) mapping swap)"
```

---

### Task 2: Add a new transpose equivalence test against the dense forward matrix

This adds an **independent** test that `affine_operator(...).transpose()` produces
the transpose of the forward dense matrix. Independent of the existing
pullback test so we can remove `affine_pullback_operator` without deleting
coverage.

**Files:**
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs`

- [ ] **Step 1: Add the new test**

Append to `crates/tensor4all-quanticstransform/tests/integration_test.rs`
immediately before the existing `test_affine_pullback_mpo_matches_transposed_matrix`
(around line 2103):

```rust
#[test]
fn test_affine_operator_transpose_matches_forward_matrix_transposed() {
    // For each test case, forward matrix M satisfies M[y, x] = 1 iff y = A x + b.
    // Thus M^T[x, y] = M[y, x]. We verify that
    //   dense(affine_operator(...).transpose())[a, b] == M[b, a]
    // for the same (a_flat, b_vec, r, bc), covering both 1×1 and a 2×2 swap.
    let test_cases: Vec<(Vec<i64>, Vec<i64>, usize, usize, Vec<BoundaryCondition>)> = vec![
        (vec![1], vec![0], 1, 1, vec![BoundaryCondition::Periodic]),
        (vec![1], vec![3], 1, 1, vec![BoundaryCondition::Periodic]),
        (vec![-1], vec![0], 1, 1, vec![BoundaryCondition::Periodic]),
        (vec![1], vec![1], 1, 1, vec![BoundaryCondition::Open]),
    ];

    for (a_flat, b_vec, m, n, bc) in &test_cases {
        for r in [2, 3] {
            let params =
                AffineParams::from_integers(a_flat.clone(), b_vec.clone(), *m, *n).unwrap();

            let ref_matrix = affine_transform_matrix(r, &params, bc).unwrap();

            let op = affine_operator(r, &params, bc).unwrap().transpose();

            // All test cases use m = n = 1, so per-site bit counts are (r, r),
            // matching the existing pullback-vs-forward test in this file.
            let mpo_dense = apply_operator_to_dense_matrix(&op, r, r);

            let in_dim = 1usize << (r * *m);
            let out_dim = 1usize << (r * *n);
            for y in 0..out_dim {
                for x in 0..in_dim {
                    let expected = *ref_matrix.get(x, y).unwrap_or(&0.0);
                    let actual = mpo_dense[y][x];
                    assert!(
                        (actual.re - expected).abs() < 1e-10,
                        "transpose real mismatch a={:?} b={:?} r={} bc={:?} at ({},{}): \
                         actual={} expected={}",
                        a_flat,
                        b_vec,
                        r,
                        bc,
                        y,
                        x,
                        actual.re,
                        expected,
                    );
                    assert!(
                        actual.im.abs() < 1e-10,
                        "transpose imag nonzero a={:?} b={:?} r={} bc={:?} at ({},{}): \
                         actual_im={}",
                        a_flat,
                        b_vec,
                        r,
                        bc,
                        y,
                        x,
                        actual.im,
                    );
                }
            }
        }
    }
}
```

- [ ] **Step 2: Run the new test**

```bash
cargo nextest run --release -p tensor4all-quanticstransform --test integration_test test_affine_operator_transpose_matches_forward_matrix_transposed
```

Expected: PASS (this validates the refactor's mathematical correctness).

- [ ] **Step 3: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/tests/integration_test.rs
git commit -m "test(quanticstransform): add transpose equivalence test for affine operator"
```

---

### Task 3: Migrate `test_affine_pullback_mpo_matches_transposed_matrix` to use `.transpose()`

**Files:**
- Modify: `crates/tensor4all-quanticstransform/tests/integration_test.rs:1855-1857, 2103-2184`

- [ ] **Step 1: Rewrite the test body**

Replace the body of `test_affine_pullback_mpo_matches_transposed_matrix` (lines
2103-2184). The test name may stay the same for git-blame continuity; or rename
to `test_affine_pullback_via_transpose_matches_forward_matrix_transposed` —
either is acceptable. Below uses the original name.

Replace the current `let op = affine_pullback_operator(r, &params, bc)...` block
with:

```rust
            let op = affine_operator(r, &params, bc).unwrap_or_else(|e| {
                panic!(
                    "Failed forward operator a={:?} b={:?} r={} bc={:?}: {}",
                    a_flat, b_vec, r, bc, e
                )
            });
            let op = op.transpose();
```

(Leave the surrounding loops, parameter construction, `ref_matrix`
construction, and assertion loop untouched.)

- [ ] **Step 2: Drop the now-unused import**

At line 1855, change:

```rust
use tensor4all_quanticstransform::{
    affine_operator, affine_pullback_operator, affine_transform_matrix, AffineParams,
};
```

to:

```rust
use tensor4all_quanticstransform::{
    affine_operator, affine_transform_matrix, AffineParams,
};
```

- [ ] **Step 3: Run the migrated test**

```bash
cargo nextest run --release -p tensor4all-quanticstransform --test integration_test test_affine_pullback_mpo_matches_transposed_matrix
cargo nextest run --release -p tensor4all-quanticstransform --test integration_test test_affine_operator_transpose_matches_forward_matrix_transposed
```

Expected: both PASS.

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/tests/integration_test.rs
git commit -m "test(quanticstransform): migrate pullback test to use affine_operator.transpose()"
```

---

### Task 4: Migrate unit tests in `src/affine/tests/mod.rs` off `affine_pullback_operator`

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine/tests/mod.rs:80-123`

Two unit tests call `affine_pullback_operator`:

- `test_affine_pullback_operator_creation` (line 81)
- `test_affine_pullback_bc_mismatch` (line 115)

- [ ] **Step 1: Rewrite `test_affine_pullback_operator_creation`**

Replace the body (lines 80-90):

```rust
#[test]
fn test_affine_pullback_operator_creation_via_transpose() {
    // Verify we can build the pullback of a 1->2 affine map via
    // affine_operator(...).transpose(): the pullback direction swaps
    // input/output roles without panicking or erroring.
    let a = vec![1i64, 0];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Open];

    let op = affine_operator(4, &params, &bc);
    assert!(op.is_ok());
    let _transposed = op.unwrap().transpose();
}
```

- [ ] **Step 2: Rewrite `test_affine_pullback_bc_mismatch`**

Replace the body (lines 114-123). `affine_operator` already rejects BC/m
mismatches the same way `affine_pullback_operator` did, so the test becomes a
forward-only error check (which is now the only API surface):

```rust
#[test]
fn test_affine_pullback_bc_mismatch_via_transpose() {
    // Boundary-condition length must equal params.m; affine_operator
    // rejects mismatches regardless of whether the caller later transposes.
    let a = vec![1i64, 0];
    let b = vec![0i64];
    let params = AffineParams::from_integers(a, b, 1, 2).unwrap();
    let bc = vec![BoundaryCondition::Periodic; 2];

    let result = affine_operator(4, &params, &bc);
    assert!(result.is_err());
}
```

- [ ] **Step 3: Run the migrated tests**

```bash
cargo nextest run --release -p tensor4all-quanticstransform \
    test_affine_pullback_operator_creation_via_transpose \
    test_affine_pullback_bc_mismatch_via_transpose
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/affine/tests/mod.rs
git commit -m "test(quanticstransform): migrate affine pullback unit tests to transpose form"
```

---

### Task 5: Delete C API pullback tests

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`

- [ ] **Step 1: Delete `test_affine_pullback_fused_materialization_matches_rust_reference`**

Remove the entire function (lines 337-378 of `crates/tensor4all-capi/src/quanticstransform/tests/mod.rs`).

- [ ] **Step 2: Delete `test_affine_pullback_fused_swap_matches_rust_reference`**

Remove the entire function (lines 380-434 of the same file).

- [ ] **Step 3: Remove the pullback assertion from `test_layout_and_affine_validation_errors_are_reported`**

Delete lines 486-500 (the block starting with `assert_eq!(t4a_qtransform_affine_pullback_materialize(...)` through the `assert!(last_error().contains("fused layouts only"));` that follows it). Keep the `t4a_qtransform_affine_materialize` interleaved-layout rejection check above it and the `zero_den` check below.

- [ ] **Step 4: Drop the unused `affine_pullback_operator` import from the tests module**

At the top of the file (line 7), change:

```rust
use tensor4all_quanticstransform::{
    affine_operator, affine_pullback_operator, cumsum_operator, flip_operator,
    phase_rotation_operator, quantics_fourier_operator, shift_operator, AffineParams,
    BoundaryCondition, FourierOptions,
};
```

to:

```rust
use tensor4all_quanticstransform::{
    affine_operator, cumsum_operator, flip_operator, phase_rotation_operator,
    quantics_fourier_operator, shift_operator, AffineParams, BoundaryCondition,
    FourierOptions,
};
```

- [ ] **Step 5: Run C API tests to confirm nothing else broke**

```bash
cargo nextest run --release -p tensor4all-capi
```

Expected: PASS (tests we deleted are gone; everything else still works since
`t4a_qtransform_affine_pullback_materialize` still exists at this point).

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-capi/src/quanticstransform/tests/mod.rs
git commit -m "test(capi): remove tests for t4a_qtransform_affine_pullback_materialize"
```

---

### Task 6: Remove C API `t4a_qtransform_affine_pullback_materialize` and inline the helper

Do C API cleanup first so that by the time we touch Rust in Task 7,
`affine_pullback_operator` has no remaining callers inside the workspace and
can be deleted cleanly.

**Files:**
- Modify: `crates/tensor4all-capi/src/quanticstransform.rs`

- [ ] **Step 1: Delete `t4a_qtransform_affine_pullback_materialize`**

Remove the entire `t4a_qtransform_affine_pullback_materialize` function from
`crates/tensor4all-capi/src/quanticstransform.rs:796-849` (the rustdoc `/// Materialize the pullback operator ...` through the function's closing `}`).

- [ ] **Step 2: Inline `materialize_affine_family` into `t4a_qtransform_affine_materialize`**

Change the body of `t4a_qtransform_affine_materialize` (lines 763-794) so the
logic formerly in `materialize_affine_family` is inlined. Full replacement —
note the `AffineMaterializeArgs` struct and `materialize_affine_family`
function are deleted along with this edit:

1. Delete the `AffineMaterializeArgs` struct (lines 502-511).
2. Delete the `materialize_affine_family` function (lines 513-548).
3. Rewrite `t4a_qtransform_affine_materialize` (lines 763-794) as:

```rust
/// Materialize the forward affine operator `y = A * x + b` as a chain-shaped
/// TreeTN using the Fused QTT layout.
///
/// `a_num[i + k * m]` and `a_den[i + k * m]` hold the numerator and
/// denominator of `A[i, k]` (column-major, length `m * n`, where `i`
/// is the row index 0..m and `k` is the column index 0..n). `b_num[i]`
/// and `b_den[i]` describe the `i`-th component of `b` (length `m`).
/// `bc[i]` is the boundary condition applied to output coordinate `i`.
/// The resulting TreeTN has `layout->nsites()` nodes, each with fused
/// input and output site indices of dimensions `2^n` and `2^m`
/// respectively.
///
/// To obtain the pullback operator `f(y) = g(A * y + b)`, materialize the
/// forward operator with this function and transpose at the binding layer
/// (the pullback is exactly the transpose of the forward operator).
///
/// # Errors
///
/// Returns `T4A_INVALID_ARGUMENT` if `m == 0`, `n == 0`, `layout->kind()`
/// is not `Fused`, `b_den[i] == 0`, or `a_den[i + k * m] == 0`.
#[unsafe(no_mangle)]
pub extern "C" fn t4a_qtransform_affine_materialize(
    layout: *const t4a_qtt_layout,
    a_num: *const i64,
    a_den: *const i64,
    b_num: *const i64,
    b_den: *const i64,
    m: usize,
    n: usize,
    bc: *const t4a_boundary_condition,
    out: *mut *mut t4a_treetn,
) -> StatusCode {
    let layout_ref = match require_layout(layout) {
        Ok(layout) => layout,
        Err((code, msg)) => {
            set_last_error(&msg);
            return code;
        }
    };

    run_catching(out, || {
        if m == 0 || n == 0 {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "affine materialization requires m > 0 and n > 0",
            ));
        }
        if layout_ref.kind() != t4a_qtt_layout_kind::Fused {
            return Err(capi_error(
                T4A_INVALID_ARGUMENT,
                "affine materialization currently supports fused layouts only",
            ));
        }

        let a = parse_rationals(a_num, a_den, m * n, "a")?;
        let b = parse_rationals(b_num, b_den, m, "b")?;
        let bc = parse_boundary_conditions(bc, m)?;
        let params = AffineParams::new(a, b, m, n)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        let source = affine_operator(layout_ref.nsites(), &params, &bc)
            .map_err(|err| capi_error(T4A_INVALID_ARGUMENT, err))?;
        Ok(t4a_treetn::new(source.mpo))
    })
}
```

- [ ] **Step 3: Clean up unused imports in the C API crate**

Scan `crates/tensor4all-capi/src/quanticstransform.rs` (top of file, the
`use tensor4all_quanticstransform::{ ... };` block) and remove any symbols
that are no longer referenced after Steps 1–2. Concretely, remove
`affine_pullback_operator` from the import list. Also remove the
`LinearOperator` import if it existed solely for the generic
`build_operator: F` signature of the deleted helper. Let the compiler
surface any other newly-unused symbols via `cargo build --workspace`.

- [ ] **Step 4: Run tests**

```bash
cargo build --workspace
cargo nextest run --release --workspace
cargo test --doc --release --workspace
```

Expected: PASS. The forward C API test
(`test_affine_fused_materialization_matches_rust_reference` and similar) still
passes since we only inlined the helper.

- [ ] **Step 5: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-capi/src/quanticstransform.rs
git commit -m "refactor(capi): remove t4a_qtransform_affine_pullback_materialize"
```

---

### Task 7: Remove `affine_pullback_operator` and its helpers from Rust

With the C API no longer calling `affine_pullback_operator`, the Rust symbol
has zero remaining callers in the workspace and can be removed.

**Files:**
- Modify: `crates/tensor4all-quanticstransform/src/affine.rs`
- Modify: `crates/tensor4all-quanticstransform/src/lib.rs`

- [ ] **Step 1: Delete `affine_pullback_operator`**

Remove the function block in `crates/tensor4all-quanticstransform/src/affine.rs`
at lines 407-473 (from the rustdoc `/// Create the operator that realizes the
pullback ...` through the closing `}` of the function).

- [ ] **Step 2: Delete `remap_affine_site_indices_pullback`**

Remove the function block in the same file at lines 265-311 (including its
rustdoc).

- [ ] **Step 3: Update `affine_operator` rustdoc to note transpose pathway**

In the doc comment for `affine_operator` (lines 313-369 of
`crates/tensor4all-quanticstransform/src/affine.rs`), delete the paragraphs
about the pullback direction (the "For the **pullback** direction ..."
paragraph and the "Forward and pullback share the same underlying MPO
construction ..." paragraph immediately after it), and replace them with a
short note that pullback is derived via `.transpose()`. The resulting summary
and description should read:

```
/// Create the operator that realizes the coordinate map `y = A * x + b`.
///
/// This is the **forward** affine operator. It maps a quantics tensor train
/// representing an `N`-variable state `x` to the quantics tensor train of
/// the `M`-variable state `y = A * x + b`.
///
/// To build the **pullback** (`f(y) = g(A * y + b)`), call `.transpose()`
/// on the returned operator; the pullback is exactly the transpose of the
/// forward operator.
```

(Keep the `# Arguments`, `# Errors`, and `# Examples` sections unchanged.)

- [ ] **Step 4: Drop the re-export from `lib.rs`**

Change `crates/tensor4all-quanticstransform/src/lib.rs:40-43` from:

```rust
pub use affine::{
    affine_operator, affine_pullback_operator, affine_transform_matrix,
    affine_transform_tensors_unfused, AffineParams, UnfusedTensorInfo,
};
```

to:

```rust
pub use affine::{
    affine_operator, affine_transform_matrix, affine_transform_tensors_unfused,
    AffineParams, UnfusedTensorInfo,
};
```

- [ ] **Step 5: Run tests**

```bash
cargo build --workspace
cargo nextest run --release --workspace
cargo test --doc --release --workspace
```

Expected: PASS. If the compiler reports remaining references to
`affine_pullback_operator` or `remap_affine_site_indices_pullback` anywhere
in the workspace, grep for them and remove those too — they should have all
been migrated by Tasks 2–6.

- [ ] **Step 6: Commit**

```bash
cargo fmt --all
git add crates/tensor4all-quanticstransform/src/affine.rs \
        crates/tensor4all-quanticstransform/src/lib.rs
git commit -m "refactor(quanticstransform): remove affine_pullback_operator (use .transpose())"
```

---

### Task 8: Regenerate C header with cbindgen

**Files:**
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h` (auto-generated)

- [ ] **Step 1: Run cbindgen**

```bash
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

(If `cbindgen` is not on `PATH`, install with `cargo install cbindgen`.)

- [ ] **Step 2: Verify `affine_pullback` is gone from the header**

```bash
grep -n "affine_pullback" crates/tensor4all-capi/include/tensor4all_capi.h
grep -n "t4a_qtransform_affine_materialize" crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: the first command returns no matches; the second returns at least
one line (the forward entry point is still exported).

- [ ] **Step 3: Commit**

```bash
git add crates/tensor4all-capi/include/tensor4all_capi.h
git commit -m "chore(capi): regenerate C header (drop affine_pullback_materialize)"
```

---

### Task 9: Update design docs

**Files:**
- Modify: `docs/CAPI_DESIGN.md:156-169`
- Modify: `docs/design/quanticstransform_julia_comparison.md:122`

- [ ] **Step 1: Update `docs/CAPI_DESIGN.md` materializers list**

Replace lines 156-169 (the "Current materializers:" list and the limitation
paragraph) with:

```
Current materializers:

- `t4a_qtransform_shift_materialize`
- `t4a_qtransform_flip_materialize`
- `t4a_qtransform_phase_rotation_materialize`
- `t4a_qtransform_cumsum_materialize`
- `t4a_qtransform_fourier_materialize`
- `t4a_qtransform_affine_materialize`

Current intentional limitation: affine materialization requires `Fused`
layout. The pullback operator `f(y) = g(A * y + b)` is not exposed as a
separate C API entry point; bindings derive it by calling
`t4a_qtransform_affine_materialize` and transposing the resulting
`LinearOperator` at the binding layer (the pullback is exactly the transpose
of the forward operator).

These constraints should stay explicit in both the function documentation and
the error message returned to bindings.
```

- [ ] **Step 2: Update `docs/design/quanticstransform_julia_comparison.md`**

Line 122 contains a forward-looking note that references
`affine_pullback_operator`, which we have removed. Replace:

```
> **注 (2026-04-18):** 以下は 2026-02-13 時点の記録。issue #428 で Rust 側の `binaryop_operator` は削除済み。現状は Julia の binaryop.jl を `affine_pullback_operator` ベースに再実装する follow-up PR で対応予定。
```

with:

```
> **注 (2026-04-19):** 以下は 2026-02-13 時点の記録。issue #428 で Rust 側の `binaryop_operator` は削除済み。issue #431 で `affine_pullback_operator` も削除し、pullback は `affine_operator(...).transpose()` で取得する方針に変更。Julia の binaryop.jl は `affine_operator + transpose + SVD` ベースで再実装する follow-up PR で対応予定。
```

- [ ] **Step 3: Verify mdbook guide snippets still build**

```bash
./scripts/test-mdbook.sh
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add docs/CAPI_DESIGN.md docs/design/quanticstransform_julia_comparison.md
git commit -m "docs: drop affine_pullback_materialize references"
```

---

### Task 10: Pre-PR checks (fmt, clippy, test, doc, mdbook, coverage)

**Files:**
- None (may touch any file if a check fails)

- [ ] **Step 1: Run formatter and linter**

```bash
cargo fmt --all
cargo clippy --workspace
```

Expected: no diffs from fmt, no new clippy warnings introduced by this plan.
If clippy flags issues in code touched by earlier tasks, fix them and re-run.

- [ ] **Step 2: Run the full test suite**

```bash
cargo nextest run --release --workspace
cargo test --doc --release --workspace
```

Expected: all tests pass.

- [ ] **Step 3: Build rustdoc**

```bash
cargo doc --workspace --no-deps
```

Expected: no errors or warnings.

- [ ] **Step 4: mdbook guide examples**

```bash
./scripts/test-mdbook.sh
```

Expected: PASS.

- [ ] **Step 5: Coverage check**

```bash
cargo llvm-cov --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
```

Expected: PASS at current thresholds. If a threshold drops, add tests —
**do not** lower thresholds (per `AGENTS.md`: threshold changes need explicit
user approval).

- [ ] **Step 6: Commit any fmt/clippy fixups**

If Step 1 introduced any formatting or clippy-fix changes:

```bash
git add -A
git commit -m "chore: fmt and clippy fixups"
```

Otherwise skip this step.

---

## Follow-up after merge (not part of this plan)

- Update `Tensor4all.jl`'s `deps/build.jl` pin to the merged tensor4all-rs
  commit, add a pure-Julia `transpose` on the Julia operator wrapper, and
  migrate the `_materialize_affine_pullback` call sites to
  `_materialize_affine + transpose`. Tracked in
  [Tensor4all.jl#58](https://github.com/tensor4all/Tensor4all.jl/issues/58).
- Do **not** open a PR for tensor4all-rs without user approval; AGENTS.md
  forbids it.
