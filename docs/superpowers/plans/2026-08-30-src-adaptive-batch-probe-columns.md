# Batch-Native Adaptive SRC Probe/QR Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-growth-step split-then-restack round trip in adaptive SRC's chain (`src_chain.rs`) and general-tree (`src_tree.rs`) contraction paths by giving `IncrementalQr`'s already-batch-native `append` a batch-native path all the way up through `factorize_probe_columns`'s growth loop and `PrefixCache`/`EnvironmentCache`'s caches, without changing the sketch/QR/projection algorithm itself.

**Architecture:** Three layers. (1) `tensor4all-core`: new `probe_batch_matrix` + `factorize_probe_batch_incremental` (a new trait method on `TensorFactorizationLike`, additive — the existing column-slice-based `factorize_probe_columns_incremental` is public API and stays untouched) turn one already-batch-indexed tensor into a `Matrix` and feed `IncrementalQr`. (2) `src_probe.rs`: new `factorize_probe_batches` (the shared adaptive growth/stop loop, replacing `factorize_probe_columns`, which becomes dead once both callers migrate). (3) `src_chain.rs`/`src_tree.rs`: `PrefixCache`/`EnvironmentCache` rewritten to cache whole growth *segments* (never split into individual columns), each producing its next batch on demand.

**Tech Stack:** Rust, existing `tensor4all-core`/`tensor4all-treetn`/`tensor4all-tensorbackend` crates in this worktree (`feature/treetn-src` branch). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-30-src-adaptive-batch-probe-columns-design.md`

## Global Constraints

- Every existing dense-oracle and isometry regression test in
  `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs` must keep
  passing, unmodified, at its existing tolerance — this is a call-batching
  change, not an algorithm change.
- `ProbeBank`'s append-only prefix property and `generate_id()`/index
  identity are out of scope — do not touch `src_probe.rs`'s `ProbeBank`
  struct or `tensor4all-core`'s index-identity machinery.
- No changes to any `tenferro`/`tenferro-rs` crate or dependency version.
- The existing column-slice-based `factorize_probe_columns_incremental`
  trait method (public API, has a doctest) is never modified or removed.
- `EnvironmentCache::batch` (the fixed-rank tree path) is out of scope —
  only its adaptive `ensure_width`/`column` methods are touched.
- `cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check`
  and `cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml
  --all-targets -- -D warnings` must stay clean after every task (also run
  the same two commands with `--manifest-path crates/tensor4all-core/Cargo.toml`
  for Task 1).
- Per this repo's conventions: commit messages use the `type(scope): summary`
  form seen in `git log`; every commit here should be reviewable standalone.

---

## File Structure

- `crates/tensor4all-core/src/tensor_like.rs`: add the new
  `factorize_probe_batch_incremental` trait method (with a from-scratch-only
  default) to `TensorFactorizationLike`.
- `crates/tensor4all-core/src/defaults/idx_tensor.rs`: add `probe_batch_matrix`,
  a shared `incremental_probe_factorize_from_matrices` helper (extracted from
  the existing `incremental_probe_factorize_typed` to share Q/R-state logic
  with zero duplication), `incremental_probe_factorize_batch_typed`, and the
  `IdxTensor` override of `factorize_probe_batch_incremental`.
- `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs`: add
  `factorize_probe_batches`; delete `factorize_probe_columns` in Task 7 once
  both callers have migrated.
- `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs`: rewrite
  `PrefixCache` to segment-based storage with a `request` method; wire the
  interior-sites loop and last-site step to `factorize_probe_batches` +
  `PrefixCache::request`, removing the `pending_columns`/`next_column`
  bookkeeping.
- `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`: rewrite
  `EnvironmentCache`'s adaptive path (`ensure_width`/`column`) to
  segment-based storage with a `request` method; wire the tree's adaptive
  branch in `contract()` to it.
- `crates/tensor4all-treetn/examples/benchmark_src.rs`: unmodified — used
  as-is for Task 7's before/after measurement.
- `docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md`
  (new, Task 7): records the before/after measurement.

---

### Task 1: Batch-native QR-feeding primitives in `tensor4all-core`

**Files:**
- Modify: `crates/tensor4all-core/src/tensor_like.rs`
- Modify: `crates/tensor4all-core/src/defaults/idx_tensor.rs`
- Test: inline `#[cfg(test)]` modules in both files (see steps)

**Interfaces:**
- Consumes: nothing from later tasks.
- Produces (for Task 2 and beyond):
  ```rust
  // tensor_like.rs, on TensorFactorizationLike
  fn factorize_probe_batch_incremental(
      previous: Option<&FactorizeResult<Self>>,
      batch_tensor: &Self,
      batch_index: &<Self as TensorIndex>::Index,
      left_inds: &[<Self as TensorIndex>::Index],
  ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>
  where
      Self: TensorVectorSpace + TensorConstructionLike;
  ```
  Called generically as `T::factorize_probe_batch_incremental(...)` from
  `src_probe.rs` (Task 2).

- [ ] **Step 1: Read the exact current body of `incremental_probe_factorize_typed`**

Before writing anything, re-read
`crates/tensor4all-core/src/defaults/idx_tensor.rs`'s
`incremental_probe_factorize_typed` function in full (search for `fn
incremental_probe_factorize_typed`) and confirm its current line range and
exact body match what this task assumes below. It has this shape (confirmed
while writing this plan):

```rust
fn incremental_probe_factorize_typed<S>(
    previous: Option<&FactorizeResult<IdxTensor>>,
    all_columns: &[&IdxTensor],
    appended_columns: &[&IdxTensor],
    left_inds: &[DynIndex],
) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    S: IncrementalQrStateScalar,
{
    let (mut state, previous_left, previous_cap, previous_rank) = if let Some(previous) = previous {
        if appended_columns.is_empty() {
            return Ok(previous.clone());
        }
        // ... resume-or-rebuild IncrementalQr<S> `state` from `previous` ...
    } else {
        let initial_matrix = probe_columns_matrix::<S>(all_columns, left_inds)?;
        (
            IncrementalQr::new(initial_matrix)
                .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?,
            None,
            None,
            0,
        )
    };
    if !appended_columns.is_empty() && previous.is_some() {
        let appended_matrix = probe_columns_matrix::<S>(appended_columns, left_inds)?;
        state
            .append(&appended_matrix)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    }
    // ... rank checks, build `left`/`cap`/`right` from `state`, return FactorizeResult ...
}
```

The only two places `all_columns`/`appended_columns` are used are the two
`probe_columns_matrix::<S>(...)` calls and the two `appended_columns.is_empty()`
checks (one gating the early return, one gating the `state.append` call —
same underlying condition, checked twice). Everything else operates on
`state`/`previous` and is agnostic to how the `Matrix<S>` was built.

- [ ] **Step 2: Extract the shared Q/R-state logic into a matrix-agnostic helper**

Rename `incremental_probe_factorize_typed` to
`incremental_probe_factorize_from_matrices` and change its signature to take
two closures instead of column slices, replacing the two
`probe_columns_matrix::<S>(X, left_inds)` call sites with `initial_matrix()?`/
`appended_matrix()?`, and the two `appended_columns.is_empty()` checks with
the new `appended_is_empty` parameter:

```rust
fn incremental_probe_factorize_from_matrices<S>(
    previous: Option<&FactorizeResult<IdxTensor>>,
    appended_is_empty: bool,
    initial_matrix: impl FnOnce() -> std::result::Result<Matrix<S>, FactorizeError>,
    appended_matrix: impl FnOnce() -> std::result::Result<Matrix<S>, FactorizeError>,
) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    S: IncrementalQrStateScalar,
{
    let (mut state, previous_left, previous_cap, previous_rank) = if let Some(previous) = previous {
        if appended_is_empty {
            return Ok(previous.clone());
        }
        // ... unchanged body that resumes/rebuilds `state` from `previous` ...
    } else {
        let initial_matrix = initial_matrix()?;
        (
            IncrementalQr::new(initial_matrix)
                .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?,
            None,
            None,
            0,
        )
    };
    if !appended_is_empty && previous.is_some() {
        let appended_matrix = appended_matrix()?;
        state
            .append(&appended_matrix)
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    }
    // ... unchanged rank checks / left/cap/right construction / return ...
}
```

Note `left_inds` is no longer a parameter of this helper — it was only ever
used inside the (now-removed) `probe_columns_matrix::<S>(_, left_inds)`
calls, which the closures now capture themselves.

- [ ] **Step 3: Re-add `incremental_probe_factorize_typed` as a thin wrapper**

Directly below the renamed function, restore the old name/signature as a
one-line wrapper so every existing caller (`factorize_probe_columns_incremental`,
the public trait method's `IdxTensor` override) keeps compiling unchanged:

```rust
fn incremental_probe_factorize_typed<S>(
    previous: Option<&FactorizeResult<IdxTensor>>,
    all_columns: &[&IdxTensor],
    appended_columns: &[&IdxTensor],
    left_inds: &[DynIndex],
) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    S: IncrementalQrStateScalar,
{
    incremental_probe_factorize_from_matrices::<S>(
        previous,
        appended_columns.is_empty(),
        || probe_columns_matrix::<S>(all_columns, left_inds),
        || probe_columns_matrix::<S>(appended_columns, left_inds),
    )
}
```

- [ ] **Step 4: Run the existing test suite to confirm this pure refactor changed nothing**

```bash
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --lib
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --doc
```

Expected: PASS, identical to before this task (this step is a pure
extract-a-helper refactor; no behavior changed yet).

- [ ] **Step 5: Add `probe_batch_matrix`**

Directly below `probe_columns_matrix` (search for `fn probe_columns_matrix`),
add:

```rust
/// Like [`probe_columns_matrix`], but for a tensor that already carries a
/// batch axis (`batch_index`) instead of being split into separate column
/// tensors — skips the `stack_along_new_index` call `probe_columns_matrix`
/// needs to re-assemble one.
fn probe_batch_matrix<S>(
    batch_tensor: &IdxTensor,
    batch_index: &DynIndex,
    left_inds: &[DynIndex],
) -> std::result::Result<Matrix<S>, FactorizeError>
where
    S: TensorElement,
{
    let mut ordered_indices = left_inds.to_vec();
    ordered_indices.push(batch_index.clone());
    let ordered = batch_tensor
        .permute_indices(&ordered_indices)
        .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?;
    let nrows = left_inds
        .iter()
        .try_fold(1usize, |size, index| size.checked_mul(index.dim()))
        .ok_or_else(|| {
            FactorizeError::ComputationError(anyhow::anyhow!(
                "incremental SRC sketch row dimension overflows usize"
            ))
        })?;
    Ok(Matrix::from_col_major_vec(
        nrows,
        batch_index.dim(),
        ordered
            .to_vec::<S>()
            .map_err(|error| FactorizeError::ComputationError(anyhow::Error::new(error)))?,
    ))
}
```

- [ ] **Step 6: Add `incremental_probe_factorize_batch_typed` and its dtype-dispatching wrapper**

Directly below `incremental_probe_factorize_typed`:

```rust
fn incremental_probe_factorize_batch_typed<S>(
    previous: Option<&FactorizeResult<IdxTensor>>,
    batch_tensor: &IdxTensor,
    batch_index: &DynIndex,
    left_inds: &[DynIndex],
) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError>
where
    S: IncrementalQrStateScalar,
{
    incremental_probe_factorize_from_matrices::<S>(
        previous,
        batch_index.dim() == 0,
        || probe_batch_matrix::<S>(batch_tensor, batch_index, left_inds),
        || probe_batch_matrix::<S>(batch_tensor, batch_index, left_inds),
    )
}

fn factorize_probe_batch_incremental_impl(
    previous: Option<&FactorizeResult<IdxTensor>>,
    batch_tensor: &IdxTensor,
    batch_index: &DynIndex,
    left_inds: &[DynIndex],
) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError> {
    if batch_tensor.is_f64() {
        incremental_probe_factorize_batch_typed::<f64>(previous, batch_tensor, batch_index, left_inds)
    } else if batch_tensor.is_c64() {
        incremental_probe_factorize_batch_typed::<Complex64>(
            previous,
            batch_tensor,
            batch_index,
            left_inds,
        )
    } else {
        Err(FactorizeError::UnsupportedStorage(
            "incremental SRC factorization currently supports f64 and Complex64 tensors",
        ))
    }
}
```

This mirrors `factorize_probe_columns_incremental`'s existing dtype-dispatch
shape exactly (search for `fn factorize_probe_columns_incremental` in this
same file to compare side by side).

- [ ] **Step 7: Add the trait method to `TensorFactorizationLike`**

In `crates/tensor4all-core/src/tensor_like.rs`, directly below the existing
`factorize_probe_columns_incremental` trait method (search for its closing
`}` at the end of its default-impl body), add:

```rust
    /// Batch-native variant of [`Self::factorize_probe_columns_incremental`]:
    /// `batch_tensor` carries a batch axis (`batch_index`) instead of being
    /// split into separate column tensors, letting an implementation avoid
    /// splitting an already-computed batch into columns and re-stacking them
    /// only to feed this call.
    ///
    /// The default implementation only supports the from-scratch case
    /// (`previous.is_none()`); it returns
    /// [`FactorizeError::UnsupportedStorage`] when asked to extend a
    /// previous factorization, exactly like this trait's
    /// [`Self::src_error_estimate`] default does for a capability a generic
    /// implementation cannot provide. `IdxTensor` overrides this with a true
    /// incremental implementation.
    ///
    /// # Errors
    /// Returns [`FactorizeError`] when the batch cannot be factorized, or
    /// (default implementation only) when asked to extend a previous
    /// factorization.
    fn factorize_probe_batch_incremental(
        previous: Option<&FactorizeResult<Self>>,
        batch_tensor: &Self,
        batch_index: &<Self as TensorIndex>::Index,
        left_inds: &[<Self as TensorIndex>::Index],
    ) -> std::result::Result<FactorizeResult<Self>, FactorizeError>
    where
        Self: TensorVectorSpace + TensorConstructionLike,
    {
        if previous.is_some() {
            return Err(FactorizeError::UnsupportedStorage(
                "incremental probe-batch growth is not supported for this tensor type",
            ));
        }
        let _ = batch_index;
        batch_tensor.factorize_full_rank(left_inds, FactorizeAlg::QR, Canonical::Left)
    }
```

- [ ] **Step 8: Override the trait method for `IdxTensor`**

In `crates/tensor4all-core/src/defaults/idx_tensor.rs`, find the `impl
TensorFactorizationLike for IdxTensor` block (search for `fn
factorize_probe_columns_incremental` inside an `impl` block, around line
5648 per this plan's initial investigation) and add, alongside the existing
`factorize_probe_columns_incremental` method in that same `impl` block:

```rust
    fn factorize_probe_batch_incremental(
        previous: Option<&FactorizeResult<IdxTensor>>,
        batch_tensor: &IdxTensor,
        batch_index: &DynIndex,
        left_inds: &[DynIndex],
    ) -> std::result::Result<FactorizeResult<IdxTensor>, FactorizeError> {
        factorize_probe_batch_incremental_impl(previous, batch_tensor, batch_index, left_inds)
    }
```

- [ ] **Step 9: Write the failing test**

Add to the `#[cfg(test)]` module at the bottom of
`crates/tensor4all-core/src/defaults/idx_tensor.rs` (find an existing `mod
tests` block; if the file's tests live in a separate `tests/` submodule
instead, check `crates/tensor4all-core/src/defaults/idx_tensor.rs`'s own
`mod tests` — this file is large enough that it may have one already; add
alongside it):

```rust
#[test]
fn factorize_probe_batch_incremental_matches_the_column_based_path() {
    let row = DynIndex::new_dyn(3);
    let batch = DynIndex::new_dyn(4);
    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.37 - 1.5).collect();
    let batch_tensor = IdxTensor::from_dense(vec![row.clone(), batch.clone()], data.clone())
        .unwrap();

    let columns: Vec<IdxTensor> = (0..4)
        .map(|position| {
            batch_tensor
                .select_indices(&[batch.clone()], &[position])
                .unwrap()
        })
        .collect();
    let column_refs: Vec<&IdxTensor> = columns.iter().collect();

    let from_batch =
        IdxTensor::factorize_probe_batch_incremental(None, &batch_tensor, &batch, &[row.clone()])
            .unwrap();
    let from_columns = IdxTensor::factorize_probe_columns_incremental(
        None,
        &column_refs,
        &column_refs,
        &[row],
    )
    .unwrap();

    assert_eq!(from_batch.rank, from_columns.rank);
    let batch_dense = from_batch.left.to_dense().unwrap();
    let columns_dense = from_columns.left.to_dense().unwrap();
    assert!(
        (batch_dense.sub(&columns_dense).unwrap().maxabs().unwrap()) < 1e-12,
        "batch-native and column-based factorizations disagree"
    );
}

#[test]
fn factorize_probe_batch_incremental_default_rejects_incremental_growth() {
    // Exercise the *default* trait implementation's `previous.is_some()`
    // error path directly (not IdxTensor's override): a type with no
    // FactorizeResult of its own to pass as `previous` can't exercise the
    // `Some` branch through IdxTensor, so this test only needs to confirm
    // the *from-scratch* default path behaves correctly for a type that
    // does not override the method. IdxTensor always overrides it, so this
    // is documentation-by-test for the trait default rather than a
    // reachable-in-practice code path today; see the trait doc comment.
    let row = DynIndex::new_dyn(2);
    let batch = DynIndex::new_dyn(2);
    let batch_tensor =
        IdxTensor::from_dense(vec![row.clone(), batch.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    let result =
        IdxTensor::factorize_probe_batch_incremental(None, &batch_tensor, &batch, &[row]).unwrap();
    assert_eq!(result.rank, 2);
}
```

- [ ] **Step 10: Run the tests to verify they fail to compile (methods don't exist yet if Steps 1-8 were skipped) or pass (if done in order)**

```bash
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --lib factorize_probe_batch_incremental
```

Expected: PASS (Steps 1-8 already added the methods before this test was
written, per this plan's step order — if you are following strict
test-first TDD instead, write Step 9 before Steps 5-8 and confirm a
compile failure first, then implement).

- [ ] **Step 11: Full crate check, fmt, clippy**

```bash
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --lib
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --doc
cargo fmt --manifest-path crates/tensor4all-core/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-core/Cargo.toml --all-targets -- -D warnings
```

Expected: all clean.

- [ ] **Step 12: Commit**

```bash
git add crates/tensor4all-core/src/tensor_like.rs crates/tensor4all-core/src/defaults/idx_tensor.rs
git commit -m "feat(core): add batch-native factorize_probe_batch_incremental"
```

---

### Task 2: Batch-native growth loop in `src_probe.rs`

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs`

**Interfaces:**
- Consumes: `T::factorize_probe_batch_incremental` (Task 1).
- Produces (for Tasks 4 and 6):
  ```rust
  pub(super) fn factorize_probe_batches<T, F>(
      left_indices: &[T::Index],
      initial_width: usize,
      maximum_width: usize,
      src_options: &SrcOptions,
      label: &str,
      mut make_batch: F,
  ) -> Result<(T, T::Index)>
  where
      T: TensorLike,
      T::Index: IndexLike + Clone + Hash + Eq,
      F: FnMut(usize, usize) -> Result<(T, T::Index)>;
  ```

- [ ] **Step 1: Write the failing test**

Add to `src_probe.rs`'s existing `#[cfg(test)] mod tests` block (the same
one containing `probe_bank_extension_preserves_the_existing_prefix`):

```rust
#[test]
fn factorize_probe_batches_grows_by_rank_increment_and_stops_on_error_estimate() {
    use super::factorize_probe_batches;

    let row = DynIndex::new_dyn(4);
    // A 4x4 identity-ish sketch (well-conditioned, rank 4) split into two
    // width-2 batches, so a rank_increment of 2 should need exactly two
    // `make_batch` calls to reach the full rank-4 estimate below tolerance.
    let batch0 = DynIndex::new_dyn(2);
    let batch1 = DynIndex::new_dyn(2);
    let full = IdxTensor::from_dense(
        vec![row.clone(), DynIndex::new_dyn(4)],
        vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    )
    .unwrap();
    let column_index = full.indices()[1].clone();
    let col = |position: usize| full.select_indices(&[column_index.clone()], &[position]).unwrap();
    let first_two = IdxTensor::stack_along_new_index(&[&col(0), &col(1)], batch0.clone(), -1).unwrap();
    let last_two = IdxTensor::stack_along_new_index(&[&col(2), &col(3)], batch1.clone(), -1).unwrap();

    let mut calls = Vec::new();
    let src_options = SrcOptions::adaptive(1.0e-10, 4).with_min_rank(1).with_rank_increment(2);
    let (result, result_batch) = factorize_probe_batches::<IdxTensor, _>(
        &[row],
        2,
        4,
        &src_options,
        "test",
        |start, width| {
            calls.push((start, width));
            if start == 0 {
                Ok((first_two.clone(), batch0.clone()))
            } else {
                Ok((last_two.clone(), batch1.clone()))
            }
        },
    )
    .unwrap();

    assert_eq!(calls, vec![(0, 2), (2, 2)]);
    assert_eq!(result_batch.dim(), 4);
    let dense = result.to_dense().unwrap();
    assert_eq!(dense.data().len(), 16);
}
```

(If `SrcOptions::adaptive`/`with_min_rank`/`with_rank_increment` names differ
slightly from this plan's assumption, check `SrcOptions`'s definition in
`src_tree.rs`/`src_chain.rs`'s `use super::{SrcOptions, ...}` origin module
— search for `struct SrcOptions` — and adjust the test to its real builder
method names; the growth/stop *behavior* under test does not depend on the
exact builder spelling.)

- [ ] **Step 2: Run test to verify it fails**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib factorize_probe_batches_grows
```

Expected: FAIL to compile (`factorize_probe_batches` does not exist yet).

- [ ] **Step 3: Implement `factorize_probe_batches`**

Directly below the existing `factorize_probe_columns` function (search for
`pub(super) fn factorize_probe_columns`), add:

```rust
/// Batch-native counterpart of [`factorize_probe_columns`]: `make_batch`
/// receives `(start, width)` and returns one batch-indexed tensor covering
/// exactly `[start, start + width)`, instead of being asked for individual
/// columns one at a time. See
/// `docs/superpowers/specs/2026-08-30-src-adaptive-batch-probe-columns-design.md`.
pub(super) fn factorize_probe_batches<T, F>(
    left_indices: &[T::Index],
    initial_width: usize,
    maximum_width: usize,
    src_options: &SrcOptions,
    label: &str,
    mut make_batch: F,
) -> Result<(T, T::Index)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize, usize) -> Result<(T, T::Index)>,
{
    if maximum_width == 0 {
        anyhow::bail!("contract_src: {label} has no usable probe columns");
    }
    let mut width = initial_width.min(maximum_width).max(1);
    let mut previous_width = 0;
    let mut previous = None;
    loop {
        let (batch_tensor, batch_index) = make_batch(previous_width, width - previous_width)?;
        let factorized =
            T::factorize_probe_batch_incremental(previous.as_ref(), &batch_tensor, &batch_index, left_indices)
                .map_err(|error| anyhow::anyhow!("contract_src: {label} QR failed: {error}"))?;
        let saturated = factorized.rank < width || width == maximum_width;
        let stop = if src_options.rtol.is_none() || saturated {
            true
        } else {
            match factorized.right.src_error_estimate() {
                Ok(estimate) => {
                    estimate.error
                        <= src_options.atol + src_options.rtol.unwrap_or(0.0) * estimate.norm
                }
                Err(error) => {
                    return Err(anyhow::anyhow!(
                        "contract_src: {label} adaptive estimator unavailable: {error}"
                    ));
                }
            }
        };
        if !stop {
            previous_width = width;
            previous = Some(factorized);
            width = width
                .saturating_add(src_options.rank_increment)
                .min(maximum_width);
            continue;
        }
        return Ok((factorized.left, factorized.bond_index));
    }
}
```

Note this loop shape is a direct one-for-one translation of
`factorize_probe_columns`'s existing growth/stop logic (same field names:
`saturated`, `stop`, `src_options.rtol`/`atol`/`rank_increment`) — cross-check
against `factorize_probe_columns`'s current body (a few lines above where
this new function is being inserted) if anything here looks off, since that
is the ground truth this function must preserve exactly, just restructured
around whole-batch calls instead of individual-column accumulation.

- [ ] **Step 4: Run test to verify it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib factorize_probe_batches_grows
```

Expected: PASS.

- [ ] **Step 5: Run the full crate test suite (nothing else changed yet, so this must be a no-op check)**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

Expected: all clean; `factorize_probe_columns` is unused by anything new
yet (still used by the existing chain/tree call sites), so no dead-code
warnings should appear for it.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs
git commit -m "feat(treetn): add factorize_probe_batches, the batch-native SRC growth loop"
```

---

### Task 3: `PrefixCache` segment-based storage (chain), added alongside the existing cache

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs`

**Interfaces:**
- Consumes: nothing new from Tasks 1-2 yet (this task only changes
  `PrefixCache`'s internals and adds `request`; nothing calls `request` yet
  — that's Task 4).
- Produces (for Task 4):
  ```rust
  impl<'a, T> PrefixCache<'a, T> {
      fn request(&mut self, site: usize, start: usize, width: usize) -> Result<(T, T::Index)>;
  }
  ```

- [ ] **Step 1: Read `PrefixCache`'s current full definition**

Re-read `struct PrefixCache` through the end of its `impl` block in
`src_chain.rs` (search `struct PrefixCache`, ends before `struct
FactorizeSiteRequest`) in full before changing anything — this task rewrites
its internals, and the exact current field names/`grow_one_segment`/`column`/
`fresh_segment` bodies (already quoted in the design spec's Section 3) must
match what's actually in the file before you delete them.

- [ ] **Step 2: Write the failing test for basic segment growth**

Add a new `#[cfg(test)] mod tests` block at the end of `src_chain.rs` (it
does not have one yet — confirm with `grep -n "mod tests" src_chain.rs`
before adding; if one already exists, add into it instead):

```rust
#[cfg(test)]
mod tests {
    use super::PrefixCache;
    use crate::treetn::contraction::src_probe::ProbeBank;
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

    fn two_site_local(dim: usize) -> (IdxTensor, IdxTensor, IdxTensor, IdxTensor) {
        let s0_out = DynIndex::new_dyn(dim);
        let s0_in = DynIndex::new_dyn(dim);
        let s1_out = DynIndex::new_dyn(dim);
        let elements = dim * dim;
        let a0 = IdxTensor::from_dense(vec![s0_out.clone(), s0_in.clone()], vec![0.1; elements]).unwrap();
        let b0 = IdxTensor::from_dense(vec![s0_in.clone()], vec![0.2; dim]).unwrap();
        let a1 = IdxTensor::from_dense(vec![s1_out.clone()], vec![0.3; dim]).unwrap();
        let b1 = IdxTensor::from_dense(vec![s1_out], vec![0.4; dim]).unwrap();
        let _ = s0_out;
        (a0, b0, a1, b1)
    }

    #[test]
    fn request_grows_a_fresh_segment_and_reuses_a_previously_cached_one() {
        let (a0, b0, a1, b1) = two_site_local(3);
        let local = vec![(&a0, &b0), (&a1, &b1)];
        let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
        let mut probes = ProbeBank::new(
            outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
            1,
            42,
        )
        .unwrap();
        let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3);

        let (first, first_batch) = cache.request(0, 0, 3).unwrap();
        assert_eq!(first_batch.dim(), 3);
        assert_eq!(first.dims().len(), 2); // [site-0 output index, batch]

        // Requesting the same already-cached range again must not recompute
        // it -- assert the returned tensor is bit-identical (a fresh
        // recomputation of the same probes would also be numerically
        // identical here since ProbeBank is deterministic, so this alone
        // isn't proof of reuse; Task 3 Step 4 below adds a call-counting
        // variant that actually proves it).
        let (again, again_batch) = cache.request(0, 0, 3).unwrap();
        assert_eq!(again_batch.dim(), first_batch.dim());
        assert_eq!(again.to_dense().unwrap(), first.to_dense().unwrap());
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib request_grows_a_fresh_segment
```

Expected: FAIL to compile (`PrefixCache::request` does not exist yet).

- [ ] **Step 4: Add the segment-based fields and `request` method, keeping the old fields/methods for now**

`PrefixCache`'s struct gains two new fields (do not remove the old
`prefixes: Vec<Vec<T>>` field yet — Task 4 removes it once nothing reads it):

```rust
struct PrefixCache<'a, T>
where
    T: TensorLike,
{
    local: &'a [(&'a T, &'a T)],
    outputs: &'a [Vec<T::Index>],
    probes: &'a mut ProbeBank<T::Index>,
    prefixes: Vec<Vec<T>>, // existing field -- untouched, still used by `column`/`fresh_segment`/`grow_one_segment` below
    batch_size: usize,     // existing field -- untouched
    // New: per-site list of (batch tensor, batch index, width) segments,
    // storing whatever chunk each `grow_segment` call actually produced --
    // segments need not all be the same width (see the design spec's
    // Section 3 on ragged final segments).
    segments: Vec<Vec<(T, T::Index, usize)>>,
    segment_total_width: usize,
}
```

In `PrefixCache::new`, initialize the two new fields alongside the existing
ones:

```rust
    fn new(
        local: &'a [(&'a T, &'a T)],
        outputs: &'a [Vec<T::Index>],
        probes: &'a mut ProbeBank<T::Index>,
        batch_size: usize,
    ) -> Self {
        Self {
            local,
            outputs,
            probes,
            prefixes: (0..local.len() - 1).map(|_| Vec::new()).collect(),
            batch_size: batch_size.max(1),
            segments: (0..local.len() - 1).map(|_| Vec::new()).collect(),
            segment_total_width: 0,
        }
    }
```

Add `grow_segment` (a segment-native sibling of the existing
`grow_one_segment`, computing exactly one new segment covering `[start,
start + width)` without ever splitting it into individual columns) and
`request`:

```rust
    /// Compute one new segment covering `[start, start + width)` for every
    /// site, without splitting it into individual per-column tensors
    /// (unlike `grow_one_segment`, which this method does not call).
    fn grow_segment(&mut self, start: usize, width: usize) -> Result<()> {
        let batch = T::Index::new_link(width)?;
        let mut prefix = probed_site_pair_batch_range(
            self.local[0].0,
            self.local[0].1,
            &self.outputs[0],
            self.probes,
            start,
            width,
            &batch,
        )?;
        self.segments[0].push((prefix.clone(), batch.clone(), width));
        for site in 1..self.local.len() - 1 {
            prefix = contract_prefix_with_probed_site_pair_batch_range(
                &prefix,
                self.local[site].0,
                self.local[site].1,
                &self.outputs[site],
                self.probes,
                start,
                width,
                &batch,
            )?;
            self.segments[site].push((prefix.clone(), batch.clone(), width));
        }
        self.segment_total_width += width;
        Ok(())
    }

    /// Return `[start, start + width)` as one batch-indexed tensor for
    /// `site`, growing new segments first if the requested range extends
    /// past what is cached.
    ///
    /// The common case (the request aligns exactly with one already-grown
    /// or newly-grown segment's boundaries) returns that segment directly,
    /// with no `select_indices`/`stack_along_new_index` at all. A request
    /// that only partially overlaps a segment boundary (possible when an
    /// earlier caller's own `maximum_width` capped a segment at a width
    /// narrower than `batch_size` -- see the design spec's Section 3) falls
    /// back to splitting and re-stacking the covering segments; this is
    /// expected to be rare, not routine.
    fn request(&mut self, site: usize, start: usize, width: usize) -> Result<(T, T::Index)> {
        self.probes.extend_to(start + width)?;
        while self.segment_total_width < start + width {
            let next_start = self.segment_total_width;
            let next_width = self.batch_size.min(start + width - next_start);
            self.grow_segment(next_start, next_width)?;
        }

        let site_segments = &self.segments[site];
        let mut cursor = 0usize;
        for (tensor, batch_index, segment_width) in site_segments {
            if cursor == start && *segment_width == width {
                return Ok((tensor.clone(), batch_index.clone()));
            }
            cursor += segment_width;
        }

        // Misaligned fallback: read the covering individual columns via the
        // existing (unchanged) `column` method and re-stack them. Reachable
        // only when a request spans a segment boundary that isn't exactly
        // `start`/`width` -- see this method's doc comment.
        let block = (0..width)
            .map(|offset| self.column(site, start + offset))
            .collect::<Result<Vec<_>>>()?;
        let block_refs = block.iter().collect::<Vec<_>>();
        let batch_index = T::Index::new_link(width)?;
        let stacked = T::stack_along_new_index(&block_refs, batch_index.clone(), -1).map_err(|error| {
            anyhow::anyhow!(
                "contract_src: site {site} misaligned segment request [{start}, {}) failed: {error}",
                start + width
            )
        })?;
        Ok((stacked, batch_index))
    }
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib request_grows_a_fresh_segment
```

Expected: PASS.

- [ ] **Step 6: Write the failing cross-request reuse + ragged-boundary test**

Add to the same `mod tests` block:

```rust
#[test]
fn request_reuses_an_earlier_ragged_segment_without_recomputing_it() {
    let (a0, b0, a1, b1) = two_site_local(3);
    let local = vec![(&a0, &b0), (&a1, &b1)];
    let outputs = vec![vec![a0.indices()[0].clone()], vec![a1.indices()[0].clone()]];
    let mut probes = ProbeBank::new(
        outputs.iter().flat_map(|o| o.iter().cloned()).collect(),
        1,
        42,
    )
    .unwrap();
    // batch_size 3, first caller only needs width 4 -- forces a ragged
    // final segment of width 1 (segments end up [0,3) then [3,4)).
    let mut cache = PrefixCache::new(&local, &outputs, &mut probes, 3);
    let (_first, _) = cache.request(0, 0, 4).unwrap();
    assert_eq!(cache.segments[0].len(), 2, "expected a [0,3) segment plus a ragged [3,4) segment");
    assert_eq!(cache.segments[0][1].2, 1, "second segment should be the ragged width-1 remainder");

    // A second caller re-reading exactly the ragged [3,4) segment must hit
    // it directly (aligned request), not fall back to misaligned handling.
    let (_second, second_batch) = cache.request(0, 3, 1).unwrap();
    assert_eq!(second_batch.dim(), 1);
    assert_eq!(cache.segments[0].len(), 2, "re-reading an existing aligned segment must not grow a new one");
}
```

(This test asserts "no new segment was created" directly via
`cache.segments[0].len()` staying at 2 across the second `request` call —
no separate call counter needed.)

- [ ] **Step 7: Run test to verify it fails, then passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib request_reuses_an_earlier_ragged_segment
```

Expected: FAILs first only if `request`'s "does an existing segment already
cover this exact range" scan (Step 4's `request` body) has a bug; given
Step 4's implementation, this should pass immediately. If it doesn't,
re-check the `cursor == start && *segment_width == width` matching logic
against this test's exact request sequence before concluding there's a
design problem.

- [ ] **Step 8: Full crate test suite, fmt, clippy**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

Expected: all clean. `PrefixCache`'s old `prefixes`/`column`/`fresh_segment`/
`grow_one_segment` are still present and still used by `contract()`'s
existing (not-yet-migrated) closures, so no dead-code warnings should appear.

- [ ] **Step 9: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs
git commit -m "feat(treetn): add segment-based PrefixCache::request, alongside the existing per-column cache"
```

---

### Task 4: Wire `src_chain.rs`'s adaptive sites to the batch-native path, remove the old per-column machinery

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs`

**Interfaces:**
- Consumes: `factorize_probe_batches` (Task 2), `PrefixCache::request` (Task 3).
- Produces: nothing new for later tasks (this is the chain-path cutover).

- [ ] **Step 1: Re-read the interior-sites loop and last-site step in full**

Re-read `contract()`'s `for site in (1..last).rev()` loop and the `else`
branch just above it (the last-site `factorize_site_adaptive` call) in
`src_chain.rs` in full immediately before editing — this task replaces both
closures' bodies.

- [ ] **Step 2: Convert `factorize_site_adaptive` itself to the batch-native closure shape**

`factorize_site_adaptive`'s actual current body (confirmed while writing
this plan) is a thin wrapper: it builds `left` (`outputs` plus an optional
`right_cap`), calls `factorize_probe_columns(&left, initial_width,
maximum_width, src_options, label, make_column)` to get `(factor, cap)`,
then does one more step this plan must not lose — computing `environment`
via `contract_site_pair(operands.0, operands.1, &[&factor_conj,
right_environment?])`:

```rust
fn factorize_site_adaptive<T, F>(
    request: FactorizeSiteRequest<'_, T>,
    make_column: F,
) -> Result<(T, T::Index, T)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize) -> Result<T>,
{
    let FactorizeSiteRequest { outputs, right_cap, operands, right_environment, initial_width, maximum_width, src_options, label } = request;
    let mut left = outputs.to_vec();
    if let Some(right_cap) = right_cap {
        left.push(right_cap.clone());
    }
    let (factor, cap) = factorize_probe_columns(&left, initial_width, maximum_width, src_options, label, make_column)?;
    let factor_conj = factor.conj();
    let environment = if let Some(right_environment) = right_environment {
        contract_site_pair(operands.0, operands.1, &[&factor_conj, right_environment])
    } else {
        contract_site_pair(operands.0, operands.1, &[&factor_conj])
    }
    .map_err(|error| anyhow::anyhow!("contract_src: {label} environment failed: {error}"))?;
    Ok((factor, cap, environment))
}
```

Change only its closure type and its one internal call, preserving
everything else (including the post-QR `environment` computation) exactly:

```rust
fn factorize_site_adaptive<T, F>(
    request: FactorizeSiteRequest<'_, T>,
    make_batch: F,
) -> Result<(T, T::Index, T)>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    F: FnMut(usize, usize) -> Result<(T, T::Index)>,
{
    let FactorizeSiteRequest { outputs, right_cap, operands, right_environment, initial_width, maximum_width, src_options, label } = request;
    let mut left = outputs.to_vec();
    if let Some(right_cap) = right_cap {
        left.push(right_cap.clone());
    }
    let (factor, cap) = factorize_probe_batches(&left, initial_width, maximum_width, src_options, label, make_batch)?;
    let factor_conj = factor.conj();
    let environment = if let Some(right_environment) = right_environment {
        contract_site_pair(operands.0, operands.1, &[&factor_conj, right_environment])
    } else {
        contract_site_pair(operands.0, operands.1, &[&factor_conj])
    }
    .map_err(|error| anyhow::anyhow!("contract_src: {label} environment failed: {error}"))?;
    Ok((factor, cap, environment))
}
```

(Only the parameter name `make_column` -> `make_batch`, the `F` bound, and
the one `factorize_probe_columns` -> `factorize_probe_batches` call changed
— the destructure and the `environment` computation below it are copied
verbatim.)

- [ ] **Step 3: Replace the last-site and interior-sites call sites' closures**

Last site — find:

```rust
        factorize_site_adaptive(
            FactorizeSiteRequest {
                outputs: &outputs[last],
                right_cap: None,
                operands: local[last],
                right_environment: None,
                initial_width: last_initial_width,
                maximum_width: last_maximum_width,
                src_options: &sketch_options,
                label: "last-site",
            },
            |column| {
                let prefix = prefixes.column(last - 1, column)?;
                contract_prefix_with_site_pair(&prefix, local[last].0, local[last].1).map_err(
                    |error| anyhow::anyhow!("contract_src: last-site sketch failed: {error}"),
                )
            },
        )?
```

Replace the closure (keep the surrounding `factorize_site_adaptive(FactorizeSiteRequest { ... }, ...)?` call shape — only `make_column`'s body changes to the new two-argument form, now using `prefixes.request` instead of `prefixes.column` and batching both site contractions instead of doing them one column at a time):

```rust
        factorize_site_adaptive(
            FactorizeSiteRequest {
                outputs: &outputs[last],
                right_cap: None,
                operands: local[last],
                right_environment: None,
                initial_width: last_initial_width,
                maximum_width: last_maximum_width,
                src_options: &sketch_options,
                label: "last-site",
            },
            |start, width| {
                let (prefix, batch_index) = prefixes.request(last - 1, start, width)?;
                let after_a = contract_retaining(&[&prefix, local[last].0], &batch_index).map_err(
                    |error| anyhow::anyhow!("contract_src: last-site prefix-A contraction failed: {error}"),
                )?;
                contract_retaining(&[&after_a, local[last].1], &batch_index)
                    .map(|tensor| (tensor, batch_index))
                    .map_err(|error| anyhow::anyhow!("contract_src: last-site sketch failed: {error}"))
            },
        )?
```

Interior sites — find the whole `factorize_site_adaptive(FactorizeSiteRequest { ... }, |column| { ... })`
call inside `for site in (1..last).rev()` and replace its closure the same
way:

```rust
        factorize_site_adaptive(
            FactorizeSiteRequest {
                outputs: &outputs[site],
                right_cap: Some(right_cap),
                operands: local[site],
                right_environment: Some(&right_environment),
                initial_width: site_initial_width,
                maximum_width: site_max_width,
                src_options: &sketch_options,
                label: &label,
            },
            |start, width| {
                let (stacked, batch_index) = prefixes.request(site - 1, start, width)?;
                let after_a = contract_retaining(&[&stacked, local[site].0], &batch_index)
                    .map_err(|error| {
                        anyhow::anyhow!("contract_src: site {site} prefix-A contraction failed: {error}")
                    })?;
                let after_b = contract_retaining(&[&after_a, local[site].1], &batch_index)
                    .map_err(|error| {
                        anyhow::anyhow!("contract_src: site {site} prefix-B contraction failed: {error}")
                    })?;
                contract_retaining(&[&after_b, &right_environment], &batch_index)
                    .map(|tensor| (tensor, batch_index))
                    .map_err(|error| anyhow::anyhow!("contract_src: site {site} sketch failed: {error}"))
            },
        )?
```

Keep this call's existing `FactorizeSiteRequest { ... }` field values exactly
as they are today (only the trailing closure argument changes) — re-check
the current file for the exact field values before editing, since this
plan's earlier reading only quoted the closure body, not the full call
expression's `FactorizeSiteRequest` construction.

Delete the `lookahead_width`, `pending_columns: VecDeque<T>`, and
`next_column` local variables that today sit just above this call inside the
same loop iteration — they are no longer referenced by anything after this
replacement.

- [ ] **Step 4: Remove `PrefixCache`'s now-dead per-column machinery**

Delete `PrefixCache`'s `prefixes: Vec<Vec<T>>` field, its initialization in
`new`, and the `column`/`fresh_segment`/`grow_one_segment` methods —
confirm with `grep -n "\.column(\|\.fresh_segment(\|grow_one_segment"
crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` that nothing
outside `PrefixCache`'s own `request` fallback path (which this plan's Task
3 wrote to call `self.column(...)`, not the deleted field) still references
them — if `request`'s misaligned-fallback branch still calls
`self.column(...)`, that call must be rewritten in this step to read
directly from `self.segments[site]` instead (splitting/restacking the
covering *segments*, not a since-deleted per-column list). Concretely,
replace `request`'s fallback block with:

```rust
        // Misaligned fallback: split the covering segment(s) into
        // individual columns via `select_indices` and re-stack the
        // requested range. Reachable only when `[start, start+width)`
        // doesn't align with a single stored segment's boundaries -- see
        // this method's doc comment.
        let mut collected = Vec::with_capacity(width);
        let mut cursor = 0usize;
        for (tensor, batch_index, segment_width) in &self.segments[site] {
            let segment_start = cursor;
            let segment_end = cursor + segment_width;
            cursor = segment_end;
            let overlap_start = segment_start.max(start);
            let overlap_end = segment_end.min(start + width);
            if overlap_start >= overlap_end {
                continue;
            }
            for position in (overlap_start - segment_start)..(overlap_end - segment_start) {
                collected.push(tensor.select_indices(std::slice::from_ref(batch_index), &[position]).map_err(
                    |error| {
                        anyhow::anyhow!(
                            "contract_src: site {site} misaligned segment split at position {position} failed: {error}"
                        )
                    },
                )?);
            }
        }
        anyhow::ensure!(
            collected.len() == width,
            "contract_src: site {site} misaligned segment request [{start}, {}) only found {} of {} columns",
            start + width,
            collected.len(),
            width
        );
        let collected_refs = collected.iter().collect::<Vec<_>>();
        let batch_index = T::Index::new_link(width)?;
        let stacked = T::stack_along_new_index(&collected_refs, batch_index.clone(), -1).map_err(|error| {
            anyhow::anyhow!(
                "contract_src: site {site} misaligned segment request [{start}, {}) failed: {error}",
                start + width
            )
        })?;
        Ok((stacked, batch_index))
```

- [ ] **Step 5: Run the full existing dense-oracle/isometry test suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
```

Expected: PASS, including every pre-existing SRC test named in
`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs` (dense-oracle
comparisons, the isometry check, the endpoint-center chain tests). If any
fail, do not proceed to Step 6 until they pass — this is the primary
correctness gate for this task.

- [ ] **Step 6: Add the chain-path cross-site segment-reuse integration test**

Add to `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`
(alongside the other SRC integration tests — search for
`src_adaptive_matches_exact_contraction_on_a_small_chain` for the
established fixture-building style in this file):

```rust
#[test]
fn src_adaptive_chain_reuses_a_ragged_segment_across_sites_and_matches_dense_reference() {
    // A 5-site chain with a physical dimension small enough that an early
    // site's maximum_width caps below a full rank_increment step (forcing
    // a ragged segment), while a later site needs to grow past it.
    let (mpo, mps) = make_five_site_chain_pair(/* physical_dim */ 2, /* bond_dim */ 3, /* seed */ 21);
    let center = mpo.node_names().into_iter().min().unwrap();
    let exact = mpo.contract_naive(&mps).unwrap();

    let result = contract(
        &mpo,
        &mps,
        &center,
        ContractionOptions::src()
            .with_max_bond_dim(9)
            .with_src_options(
                SrcOptions::adaptive(1.0e-8, 9)
                    .with_min_rank(1)
                    .with_rank_increment(2)
                    .with_seed(7),
            ),
    )
    .unwrap();

    let dense = result.to_dense().unwrap();
    let rel_error = dense.sub(&exact).unwrap().maxabs().unwrap() / exact.maxabs().unwrap();
    assert!(rel_error < 1e-6, "relative error {rel_error} too large");
}
```

`make_five_site_chain_pair` does not exist yet — write it following the
exact pattern of this same test file's existing `make_three_node_chain_pair`
(or `benchmark_src.rs`'s `make_mpo_mps`, adapted to `IdxTensor`/`String`
node names matching this test file's conventions) — check
`make_three_node_chain_pair`'s real body in this file first and mirror its
random-tensor-construction style exactly (same RNG usage, same index
construction pattern) rather than inventing a new one, so this fixture
looks like it belongs next to the others.

- [ ] **Step 7: Run the new test, iterate until it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib src_adaptive_chain_reuses_a_ragged_segment
```

Expected: PASS. If the relative error is too large, first check whether the
chosen `bond_dim`/`max_bond_dim`/`rank_increment` combination actually
produces a ragged segment at all (add a temporary `eprintln!` inside
`PrefixCache::grow_segment` printing `(start, width)` to confirm, then
remove it) before suspecting a logic bug in `request`.

- [ ] **Step 8: fmt, clippy, full suite one more time**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

- [ ] **Step 9: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "perf(treetn): wire chain-path adaptive SRC to the batch-native probe/QR path"
```

---

### Task 5: `EnvironmentCache` segment-based storage (tree, adaptive path only), added alongside the existing cache

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`

**Interfaces:**
- Consumes: nothing new yet (parallel to Task 3).
- Produces (for Task 6):
  ```rust
  impl<'a, T, V> EnvironmentCache<'a, T, V> {
      fn request(&mut self, parent: &V, child: &V, start: usize, width: usize) -> Result<(T, T::Index)>;
  }
  ```

- [ ] **Step 1: Re-read `EnvironmentCache`'s current full definition**

Re-read `struct EnvironmentCache` through the end of its `impl` block
(`ensure_width`, `batch`, `column`) in `src_tree.rs` in full before changing
anything. **Do not touch `batch`** — it is the fixed-rank path and stays
exactly as-is (Global Constraints).

- [ ] **Step 2: Write the failing test**

Add a `#[cfg(test)] mod tests` block at the end of `src_tree.rs` (check
first whether one already exists). Build the two independent networks the
same way `make_star_pair()` in
`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs` does (two
independently-scaled copies of the same topology — that existing test
fixture is the closest, already-working example of a two-network `TreeTN`
pair for this module's tests), and derive `local`/`outputs` the same way
`contract()` itself does a few lines above `EnvironmentCache::new`'s call
site in this file: via `local_site_pairs`/`local_output_indices`, both
already `pub(super)` in `src_probe.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::super::src_probe::{local_output_indices, local_site_pairs, ProbeBank};
    use super::EnvironmentCache;
    use crate::treetn::TreeTN;
    use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

    fn three_node_path(offset: f64) -> TreeTN<IdxTensor, String> {
        let dim = 3usize;
        let ab = DynIndex::new_dyn(dim);
        let bc = DynIndex::new_dyn(dim);
        let a_out = DynIndex::new_dyn(dim);
        let b_out = DynIndex::new_dyn(dim);
        let c_out = DynIndex::new_dyn(dim);
        let a = IdxTensor::from_dense(
            vec![a_out, ab.clone()],
            (0..dim * dim).map(|i| offset + f64::from(i as i32) / 10.0).collect(),
        )
        .unwrap();
        let b = IdxTensor::from_dense(
            vec![ab, b_out, bc.clone()],
            (0..dim * dim * dim).map(|i| offset + f64::from(i as i32) / 11.0).collect(),
        )
        .unwrap();
        let c = IdxTensor::from_dense(
            vec![bc, c_out],
            (0..dim * dim).map(|i| offset + f64::from(i as i32) / 12.0).collect(),
        )
        .unwrap();
        TreeTN::from_tensors(vec![a, b, c], vec!["A".into(), "B".into(), "C".into()]).unwrap()
    }

    #[test]
    fn request_grows_a_fresh_segment_and_re_reads_an_aligned_one() {
        let tn_a = three_node_path(1.0);
        let tn_b = three_node_path(2.0);
        let mut nodes = tn_a.node_names();
        nodes.sort();
        let edges = tn_a
            .edges_to_canonicalize_by_names(&"B".to_string())
            .unwrap();

        let local_values = local_site_pairs(&tn_a, &tn_b, &nodes).unwrap();
        let local = nodes
            .iter()
            .cloned()
            .zip(local_values)
            .collect::<std::collections::HashMap<_, _>>();
        let outputs = nodes
            .iter()
            .map(|node| (node.clone(), local_output_indices(&tn_a, &tn_b, node).unwrap()))
            .collect::<std::collections::HashMap<_, _>>();
        let mut probe_indices = nodes
            .iter()
            .flat_map(|node| outputs[node].iter().cloned())
            .collect::<Vec<_>>();
        let mut probes = ProbeBank::new(std::mem::take(&mut probe_indices), 1, 99).unwrap();
        let mut cache = EnvironmentCache::new(&tn_a, &edges, &nodes, &local, &outputs, &mut probes);

        let (first, first_batch) = cache.request(&"B".to_string(), &"A".to_string(), 0, 3).unwrap();
        assert_eq!(first_batch.dim(), 3);

        let (again, again_batch) = cache.request(&"B".to_string(), &"A".to_string(), 0, 3).unwrap();
        assert_eq!(again_batch.dim(), first_batch.dim());
        assert_eq!(again.to_dense().unwrap(), first.to_dense().unwrap());
        assert_eq!(cache.segments.len(), 1, "re-reading an aligned segment must not grow a new one");
    }
}
```

If `EnvironmentCache::new`'s exact parameter order/types differ from this
plan's assumption (`tn`, `edges`, `nodes`, `local`, `outputs`, `probes`), or
if `edges_to_canonicalize_by_names`'s exact name differs, re-check
`EnvironmentCache::new`'s current definition and `contract()`'s own call to
it (both a few lines apart in this file, already read in Task 5 Step 1) and
adjust this test to match — this plan's earlier reading of `contract()`
confirms both names exist with this shape as of this session.

- [ ] **Step 3: Run test to verify it fails**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib request_grows_a_fresh_segment_and_re_reads
```

Expected: FAIL to compile (`EnvironmentCache::request` and the `segments`
field do not exist yet).

- [ ] **Step 4: Add the segment-based fields and `request` method to `EnvironmentCache`**

Add two new fields alongside the existing ones (`environments`,
`batched_environments` stay untouched -- they back `column`/`ensure_width`
and `batch` respectively, both left in place for now):

```rust
    segments: Vec<(usize, usize, T::Index, HashMap<(V, V), T>)>, // (start, width, batch index, per-edge environment)
    segment_total_width: usize,
```

Initialize both to empty/0 in `EnvironmentCache::new`. Add:

```rust
    /// Compute one new segment covering `[start, start + width)`'s
    /// per-edge environment tensors, without ever materializing a
    /// per-column (`width == 1`) representation.
    fn grow_segment(&mut self, start: usize, width: usize) -> Result<()> {
        self.probes.extend_to(start + width)?;
        let batch = T::Index::new_link(width)?;
        let probed = self
            .nodes
            .iter()
            .map(|node| {
                let (tensor_a, tensor_b) = self.local.get(node).copied().ok_or_else(|| {
                    anyhow::anyhow!("contract_src: local tensor is missing")
                })?;
                let site_outputs = self
                    .outputs
                    .get(node)
                    .ok_or_else(|| anyhow::anyhow!("contract_src: output list is missing"))?;
                Ok((
                    node.clone(),
                    probed_site_pair_batch_range(
                        tensor_a, tensor_b, site_outputs, self.probes, start, width, &batch,
                    )?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        // `directed_messages_batched` contracts every message via
        // `contract_retaining(&factors, batch)`, which explicitly keeps
        // `batch` in the result -- every tensor in `directed` carries this
        // exact `batch` index, so it is safe to store and hand back later
        // without re-deriving it from any one tensor's own index list.
        let directed = directed_messages_batched(self.tn, self.edges, &probed, &batch)?;
        self.segments.push((start, width, batch, directed));
        self.segment_total_width += width;
        Ok(())
    }

    /// Return the environment tensor for `(parent, child)` covering
    /// `[start, start + width)`, growing new segments first if needed. See
    /// `PrefixCache::request` (`src_chain.rs`) for the equivalent
    /// chain-path method and the same aligned-vs-misaligned reasoning.
    fn request(&mut self, parent: &V, child: &V, start: usize, width: usize) -> Result<(T, T::Index)> {
        while self.segment_total_width < start + width {
            let next_start = self.segment_total_width;
            self.grow_segment(next_start, width.min(start + width - next_start))?;
        }
        for (segment_start, segment_width, batch_index, environments) in &self.segments {
            if *segment_start == start && *segment_width == width {
                let environment = environments
                    .get(&(parent.clone(), child.clone()))
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "contract_src: cached segment environment is missing for {:?}->{:?}",
                            parent,
                            child
                        )
                    })?;
                return Ok((environment, batch_index.clone()));
            }
        }
        anyhow::bail!(
            "contract_src: no segment covers [{start}, {}) for {:?}->{:?} after growth",
            start + width,
            parent,
            child
        )
    }
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib request_grows_a_fresh_segment_and_re_reads
```

Expected: PASS.

- [ ] **Step 6: fmt, clippy, full suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

- [ ] **Step 7: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs
git commit -m "feat(treetn): add segment-based EnvironmentCache::request for the tree adaptive path"
```

---

### Task 6: Wire `src_tree.rs`'s adaptive branch to the batch-native path, remove the old per-column machinery

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs`

**Interfaces:**
- Consumes: `factorize_probe_batches` (Task 2), `EnvironmentCache::request` (Task 5).

- [ ] **Step 1: Re-read `contract()`'s adaptive branch in full**

In `src_tree.rs`'s `contract()`, re-read the
`if sketch_options.rtol.is_none() { ... } else { factorize_probe_columns(...) }`
branch (the adaptive/`else` arm) in full before editing.

- [ ] **Step 2: Replace the adaptive branch's `factorize_probe_columns` call**

Find:

```rust
                factorize_probe_columns(
                    &left_indices,
                    site_initial_width,
                    site_max_width,
                    &sketch_options,
                    &label,
                    |column| {
                        let environment = environment_cache.column(parent, child, column)?;
                        let mut factors = source_factors.clone();
                        factors.push(&environment);
                        T::contract(&factors).map_err(|error| {
                            anyhow::anyhow!(
                                "contract_src: tree sketch for {:?}->{:?} failed: {error}",
                                child,
                                parent
                            )
                        })
                    },
                )?
```

Replace with:

```rust
                factorize_probe_batches(
                    &left_indices,
                    site_initial_width,
                    site_max_width,
                    &sketch_options,
                    &label,
                    |start, width| {
                        let (environment, batch_index) =
                            environment_cache.request(parent, child, start, width)?;
                        let mut factors = source_factors.clone();
                        factors.push(&environment);
                        contract_retaining(&factors, &batch_index)
                            .map(|tensor| (tensor, batch_index))
                            .map_err(|error| {
                                anyhow::anyhow!(
                                    "contract_src: tree sketch for {:?}->{:?} failed: {error}",
                                    child,
                                    parent
                                )
                            })
                    },
                )?
```

Note the switch from `T::contract(&factors)` to `contract_retaining(&factors,
&batch_index)`: the old per-column path used a plain `T::contract` because a
single column carries no separate "batch" axis to retain, but the new
batched `environment` tensor does — dropping it would silently contract away
the batch dimension.

- [ ] **Step 3: Remove `EnvironmentCache`'s now-dead adaptive-only per-column machinery**

Delete `ensure_width` and `column` (the adaptive path's old methods) and the
`environments: Vec<HashMap<(V,V),T>>` field they used, along with `use
super::src_probe::{... local_output_indices ...}` adjustments if any import
becomes unused. **Do not delete `batch`, `batched_environments`, or
anything the fixed-rank branch (`sketch_options.rtol.is_none()`) uses** —
confirm with `grep -n "\.ensure_width(\|\.column(\|environments\b"
crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs` that only the
adaptive branch's now-replaced call site referenced the deleted members
before removing them.

- [ ] **Step 4: Run the full existing dense-oracle/isometry test suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
```

Expected: PASS, including the general-tree (star/branched) SRC tests.

- [ ] **Step 5: Add a tree-path adaptive dense-oracle integration test**

`make_star_pair()` (already defined in
`crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`, used by the
existing `src_fixed_traverses_a_branched_tree_without_dense_fallback` test)
builds a 3-node path `L`-`C`-`R` where `C` connects to both leaves — since
`C` is not a chain endpoint, requesting it as the contraction center routes
through `src_tree.rs`'s general path (confirmed by the existing fixed-rank
test using exactly this fixture and center). Add an adaptive counterpart
right after that existing test, exercising this task's rewired adaptive
branch end-to-end:

```rust
#[test]
fn src_adaptive_traverses_a_branched_tree_and_matches_dense_reference() {
    let (tn_a, tn_b) = make_star_pair();
    let exact = tn_a.contract_naive(&tn_b).unwrap();
    let options = ContractionOptions::src().with_max_bond_dim(4).with_src_options(
        SrcOptions::adaptive(1.0e-8, 4)
            .with_min_rank(1)
            .with_rank_increment(2)
            .with_seed(88),
    );
    let result = contract(&tn_a, &tn_b, &"C".to_string(), options).unwrap();
    let dense = result.to_dense().unwrap();
    let rel_error = dense.distance(&exact).unwrap() / exact.norm().unwrap();
    assert!(rel_error < 1e-6, "relative error {rel_error} too large");
}
```

Check `SrcOptions::adaptive`/`with_min_rank`/`with_rank_increment`'s exact
builder names against the existing
`src_fixed_traverses_a_branched_tree_without_dense_fallback` test's sibling
adaptive tests in this same file (search for `SrcOptions::adaptive(` — it is
already used elsewhere in this test module) before finalizing, and match
whatever `IdxTensor`'s dense-comparison helper is called elsewhere in this
file (`distance`/`norm`, or a `sub`/`maxabs` pair — Task 4 Step 6 uses
`sub`/`maxabs`; pick whichever this file's own existing SRC tests already
use for consistency rather than mixing both styles).

- [ ] **Step 6: Run the new test, iterate until it passes**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib src_adaptive_traverses_a_branched_tree_and_matches_dense_reference
```

Expected: PASS.

- [ ] **Step 7: fmt, clippy, full suite**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

- [ ] **Step 8: Commit**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "perf(treetn): wire tree-path adaptive SRC to the batch-native probe/QR path"
```

---

### Task 7: Delete the now-dead old growth loop, measure, and record results

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs`
- Create: `docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing further (terminal task).

- [ ] **Step 1: Confirm `factorize_probe_columns` has no remaining callers**

```bash
grep -rn "factorize_probe_columns\b" crates/tensor4all-treetn/src
```

Expected: only the function's own definition and its `#[cfg(test)]` module
(if any test still exercises it directly) remain. If a test still calls it
directly, either delete that test (if Task 3's/5's new tests already cover
the same ground) or leave `factorize_probe_columns` in place and stop this
task here, reporting the conflict rather than deleting a function something
still depends on.

- [ ] **Step 2: Delete `factorize_probe_columns` from `src_probe.rs`**

Remove the function and any now-unused imports it required.

- [ ] **Step 3: Full workspace-scoped test suite, fmt, clippy (both crates touched by this plan)**

```bash
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --lib
cargo test --manifest-path crates/tensor4all-core/Cargo.toml --doc
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib
cargo fmt --manifest-path crates/tensor4all-core/Cargo.toml -- --check
cargo fmt --manifest-path crates/tensor4all-treetn/Cargo.toml -- --check
cargo clippy --manifest-path crates/tensor4all-core/Cargo.toml --all-targets -- -D warnings
cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings
```

Expected: all clean.

- [ ] **Step 4: Commit the deletion**

```bash
git add crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs
git commit -m "chore(treetn): remove the now-dead per-column factorize_probe_columns"
```

- [ ] **Step 5: Measure before/after, matching this session's original methodology**

```bash
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml --example benchmark_src
for bd in 4 8 16 32; do
  RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    T4A_BENCH_SKIP_EXACT=1 \
    target/release/examples/benchmark_src 10 $bd 5 mpo-mps 3 2>&1 | grep -E "^config|^case"
done
```

Record the `src-adaptive` `per_run` numbers at each bond dimension. Compare
against this plan's spec's "Problem" table (bond 4: 37.735ms; bond 8:
54.610ms; bond 16: 60.314ms; bond 32: 74.955ms, all rust-side numbers from
before this plan's changes) and against the Python reference numbers already
recorded there.

Also re-run the `tree` mode (star topology) to confirm the general-tree
path's improvement independently of the chain path's:

```bash
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  T4A_BENCH_SKIP_EXACT=1 \
  target/release/examples/benchmark_src 10 8 5 tree 3 2>&1 | grep -E "^config|^case"
```

- [ ] **Step 6: Write the results worklog**

Create `docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md`
following this worktree's established worklog format (see
`docs/worklogs/2026-08-29-src-tree-path-performance.md` for the structure:
Summary, Context and sources, Measurement methodology, Results, Analysis,
Decision). Include:
- The exact before/after numbers from Step 5, at every bond dimension
  tested, both chain (`mpo-mps` mode) and tree (`tree` mode).
- The Python reference numbers from the design spec's "Problem" section,
  for the same before/after comparison the original investigation used.
- An honest accounting of whatever the actual improvement turned out to be
  — per this plan's Global Constraints and the design spec's explicit
  "Testing" item 3, there is no preset number to hit; report what happened,
  including if the improvement is smaller than the theoretical
  `select_indices`-call-count reduction would suggest (e.g. if `tenferro`'s
  per-call overhead turns out to dominate even the reduced call count, or if
  the general-tree path's remaining architectural overhead — the
  whole-tree-per-segment message pass itself, now less frequent but not
  eliminated — remains the larger term).

- [ ] **Step 7: Commit**

```bash
git add docs/worklogs/2026-08-30-src-adaptive-batch-probe-columns-results.md
git commit -m "docs(treetn): record before/after results for batch-native adaptive SRC"
```
