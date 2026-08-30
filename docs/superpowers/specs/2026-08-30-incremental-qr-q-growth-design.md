# Incremental QR: amortized-growth Q buffer (issue #696, Finding A, step 1 of 2)

## Problem

SRC's adaptive incremental QR (`crates/tensor4all-tensorbackend/src/incremental_qr.rs`) pays
O(final_rank²) total copy cost across an adaptive run instead of O(final_rank), because
`Matrix<T>` has no in-place growth API. Every `append()` call rebuilds `Q`, `R`, and
`R^{-†}` from scratch via `concatenate_columns`/`assemble_r`/`update_inverse_adjoint`,
each of which allocates a brand-new buffer sized to the full new total and copies *all*
of the previous state into it, not just the new increment. Filed as
[tensor4all-rs#696](https://github.com/tensor4all/tensor4all-rs/issues/696).

This step addresses only the `Q` factor's growth. `R`/`R^{-†}}` grow in both dimensions
as rank increases (unlike `Q`, which only grows in column count while its row count `m`
stays fixed), which is a fundamentally harder problem in column-major flat storage and is
deferred to a follow-up step, evaluated only if benchmarking after this step shows it
still matters.

## Reference precedent

The paper's own reference Python implementation
(`RandomMPOMPS-reference-20260827/code/tensornetwork/incrementalqr.py`) solves exactly
this problem for its `data`/`tau` buffers via manual capacity tracking (a `size` field,
doubled via `_resize` only when exceeded, with `append` writing new columns directly into
already-reserved space). Rust's `Vec<T>` already provides this — `len()` vs `capacity()`,
amortized-doubling growth built into `push`/`extend_from_slice` — so no new capacity-
tracking machinery needs to be hand-rolled the way the Python reference had to.

## Design

`Q` is `m × rank` where `m` (the original input's row count) is fixed for the lifetime of
one `IncrementalQr<T>` and only `rank` (column count) grows. In column-major layout,
appending new columns is appending to the *end* of the flat `data: Vec<T>` buffer — no
existing element needs to move. This is exactly the access pattern `Vec::extend_from_slice`
is already amortized-O(1)-per-element for.

**Add one new crate-internal method to `Matrix<T>`** (`matrix.rs`), next to the existing
`from_col_major_vec`/`as_col_major_slice` family:

```rust
impl<T: Clone> Matrix<T> {
    /// Appends `right`'s columns to the end of this matrix in place, in
    /// column-major order. Reuses existing Vec capacity via amortized-doubling
    /// growth (`Vec::extend_from_slice`) instead of always reallocating and
    /// copying every existing element, unlike constructing a fresh
    /// concatenated `Matrix` via `from_col_major_vec`.
    pub(crate) fn append_columns(&mut self, right: &Matrix<T>) -> Result<(), ...> {
        // validate right.nrows() == self.nrows, check for column-count overflow
        // self.data.extend_from_slice(right.as_col_major_slice());
        // self.ncols = new_ncols;
    }
}
```

Visibility is `pub(crate)`: `IncrementalQr<T>` lives in the same crate
(`tensor4all-tensorbackend`) as `Matrix<T>`, so this doesn't need to be part of `Matrix`'s
public API. This keeps the change additive and fully backward compatible — no existing
caller's behavior changes, no existing invariant (`data.len() == nrows*ncols`) is weakened
(it still holds exactly at every observable point; only *how* it's achieved changes for
this one new growth path).

**Use it in `commit_full_rank_block`** (`incremental_qr.rs:308-356`): replace

```rust
let q = concatenate_columns(&self.q, &appended_q)?;
...
self.q = q;
```

with an in-place update:

```rust
self.q.append_columns(&appended_q)?;
```

removing the need to reassign `self.q` from a freshly built matrix. `commit_dependent_column`
(the single-column, rank-deficient path, `incremental_qr.rs:358+`) should be checked for the
same pattern and updated identically if it also rebuilds `q` via concatenation.

`concatenate_columns` itself can be deleted once nothing calls it, *unless* something
else in the file also needs it — check before removing.

## What stays unchanged

- `R` and `R^{-†}` (`assemble_r`, `update_inverse_adjoint`) — untouched in this step.
- `Matrix<T>`'s public API and invariants — untouched; `append_columns` is additive and
  crate-internal.
- The actual QR math (`project_twice`, `factorize_backend`, residual factorization) —
  untouched; this is purely a bookkeeping/storage change for how `Q`'s accumulated state
  is merged after each append, not a change to the algorithm.

## Testing

- Existing `incremental_qr` test suite (14 tests, all currently passing) must continue to
  pass unchanged — they check reconstruction, orthogonality, rank-deficiency, and
  cross-check against direct backend QR, so they're a strong correctness safety net for a
  pure storage-strategy change.
- Add a differential/unit test on `Matrix::append_columns` directly: appending columns
  produces the same result as the old `concatenate_columns` free function for a few shapes
  (including edge cases: row mismatch error, column-count overflow, appending to a matrix
  with spare vs. exactly-full Vec capacity).
- No new test is needed to prove the *complexity* improvement — that's a benchmark
  concern (see Verification below), not a correctness one.

## Verification (before moving to the R/inverse_adjoint step)

Run `crates/tensor4all-treetn/examples/benchmark_src.rs`'s adaptive-mode sweep (the same
harness already used earlier this session) before and after this change, at a rank high
enough that the O(final_rank²) copy cost should be visible against the O(final_rank²)
compute cost it's layered on top of (e.g. the same n_sites=16, D=χ=128, target rank
sweep up to ~192 used earlier this session). Compare per-point wall-clock time. This
determines whether the Finding A follow-up step (R/inverse_adjoint) is still worth doing.
