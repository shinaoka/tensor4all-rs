# Incremental QR Q-Buffer Amortized Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `IncrementalQr<T>`'s `Q` factor grow via amortized-doubling in-place `Vec` extension instead of a full fresh-allocate-and-copy on every `append()` call, fixing the `Q`-specific half of tensor4all-rs#696 Finding A.

**Architecture:** Add a `pub(crate)` `Matrix::append_columns` method to `tensor4all-tensorbackend/src/matrix.rs` that extends the matrix's existing `Vec<T>` buffer in place (`Vec::extend_from_slice`, which reuses spare capacity via Rust's built-in amortized-doubling growth). Wire it into `IncrementalQr::commit_full_rank_block` (`incremental_qr.rs`), replacing the `concatenate_columns` free function, which always allocates a fresh buffer sized to the full new total and copies every old element into it. `R`/`R^{-†}` (which grow in both dimensions, unlike `Q`) are untouched — deferred to a follow-up step evaluated only if benchmarking here shows it still matters.

**Tech Stack:** Rust, `tensor4all-tensorbackend` crate (workspace member of `tensor4all-rs`).

**Spec:** `docs/superpowers/specs/2026-08-30-incremental-qr-q-growth-design.md`

## Global Constraints

- Do not change `Matrix<T>`'s public API surface or its `data.len() == nrows*ncols` invariant as observed by any existing caller — `append_columns` is additive and `pub(crate)`.
- Do not touch `R`/`R^{-†}` growth (`assemble_r`, `update_inverse_adjoint`) in this plan.
- `commit_full_rank_block` must remain atomic: on any error, `self` (the `IncrementalQr` receiver) must be left completely unchanged, matching its current behavior. Do not mutate `self.q`/`self.r`/`self.inverse_adjoint` until every fallible step has already succeeded.
- Build with `cargo check --manifest-path crates/tensor4all-tensorbackend/Cargo.toml` (and other crate-scoped commands) — never a workspace-wide build.
- The worktree is `/root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src` on branch `feature/treetn-src`. All file paths below are relative to that worktree root.

---

## Task 1: Capture the pre-fix performance baseline

**Files:** none (measurement only).

**Interfaces:** none.

- [ ] **Step 1: Run the adaptive-mode SRC benchmark sweep and record the output**

Run this exact sweep (same problem parameters used earlier this session for the wide bond-dimension scaling sweep: n_sites=16, input_bond=128, physical_dim=2, seed=7, mode=mpo-mps, rank_increment=3, final_svd=false, reps=2), once per target rank:

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml --example benchmark_src
for rank in 8 16 32 64 96 128 160 192; do
  RAYON_NUM_THREADS=1 T4A_BENCH_SKIP_EXACT=1 \
    ./target/release/examples/benchmark_src 16 128 2 mpo-mps 3 false "$rank" \
    | grep 'case=mpo-mps/src-adaptive'
done
```

Save the full output (all 8 lines, one per rank) to `/tmp/claude-0/-root-projects-gw-rs/872136a4-bcbd-4dc2-b4df-4f45f14c1f30/scratchpad/q-growth-baseline.txt`. Each captured line has the form `case=mpo-mps/src-adaptive reps=2 elapsed=X.XXXXXXs per_run=Y.YYYYYYs nodes=... edges=... max_bond=... relative_error=...` — use the `per_run=` field (elapsed divided by `reps`), not the raw `elapsed=` total, as the comparable number. This is the figure that must improve after the fix, since `src-adaptive` is the code path that calls `IncrementalQr::append` the most times (once per `rank_increment`-sized growth step, `rank_increment=3` here), making it the case most exposed to the O(final_rank²) copy bug in `commit_full_rank_block`.

- [ ] **Step 2: Note the baseline numbers in this plan file**

Edit this plan file (append a line under this task) recording the 8 `src-adaptive` `per_run=` times you captured, so Task 5's comparison has a fixed reference point even if the saved file is later cleaned up.

---

## Task 2: Add `Matrix::append_columns` with a failing test first

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/matrix.rs` (new method, insert into the existing `impl<T> Matrix<T>` block that starts at line 410, placed immediately after `into_col_major_vec` which ends at line 511)
- Test: `crates/tensor4all-tensorbackend/src/matrix/tests/mod.rs`

**Interfaces:**
- Produces: `Matrix::<T>::append_columns(&mut self, right: &Matrix<T>) -> std::result::Result<(), MatrixShapeError>` where `T: Clone`, `pub(crate)` visibility.

- [ ] **Step 1: Write the failing tests**

Append to `crates/tensor4all-tensorbackend/src/matrix/tests/mod.rs` (this file already starts with `use super::*;` and `use num_complex::Complex64;`, both already in scope for what follows):

```rust
#[test]
fn append_columns_reuses_existing_data_and_grows_ncols() {
    let mut left = Matrix::from_col_major_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]);
    let right = Matrix::from_col_major_vec(2, 1, vec![5.0_f64, 6.0]);
    left.append_columns(&right).unwrap();
    assert_eq!(left.nrows(), 2);
    assert_eq!(left.ncols(), 3);
    assert_eq!(left.as_col_major_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn append_columns_matches_building_a_fresh_concatenated_matrix() {
    let left = Matrix::from_col_major_vec(3, 2, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let right = Matrix::from_col_major_vec(3, 1, vec![7.0_f64, 8.0, 9.0]);

    let mut grown = left.clone();
    grown.append_columns(&right).unwrap();

    let mut expected_data = left.as_col_major_slice().to_vec();
    expected_data.extend_from_slice(right.as_col_major_slice());
    let expected = Matrix::try_from_col_major_vec(3, 3, expected_data).unwrap();

    assert_eq!(grown.nrows(), expected.nrows());
    assert_eq!(grown.ncols(), expected.ncols());
    assert_eq!(grown.as_col_major_slice(), expected.as_col_major_slice());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib matrix::tests::append_columns
```

Expected: compile error, `no method named 'append_columns' found for struct 'Matrix<f64>'`.

- [ ] **Step 3: Implement `Matrix::append_columns`**

In `crates/tensor4all-tensorbackend/src/matrix.rs`, insert immediately after the `into_col_major_vec` method (which currently ends at line 511 with the closing `}` before the doc comment for `borrow this matrix as an owned tenferro TypedTensor`):

```rust
    /// Appends `right`'s columns to the end of this matrix in place, in
    /// column-major order.
    ///
    /// Reuses existing spare `Vec` capacity via amortized-doubling growth
    /// (`Vec::extend_from_slice`) instead of always reallocating and copying
    /// every existing element the way building a fresh concatenated `Matrix`
    /// via [`Matrix::try_from_col_major_vec`] would. Column-major layout
    /// makes appending columns an append-to-the-end operation on the flat
    /// buffer, so no existing element moves.
    ///
    /// # Errors
    /// Returns [`MatrixShapeError::ShapeOverflow`] when the combined column
    /// count would overflow `usize`.
    pub(crate) fn append_columns(
        &mut self,
        right: &Matrix<T>,
    ) -> std::result::Result<(), MatrixShapeError>
    where
        T: Clone,
    {
        debug_assert_eq!(
            self.nrows(),
            right.nrows(),
            "append_columns requires matching row counts: {} vs {}",
            self.nrows(),
            right.nrows()
        );
        let new_ncols =
            self.ncols()
                .checked_add(right.ncols())
                .ok_or(MatrixShapeError::ShapeOverflow {
                    nrows: self.nrows(),
                    ncols: self.ncols().saturating_add(right.ncols()),
                })?;
        self.data.extend_from_slice(right.as_col_major_slice());
        self.ncols = new_ncols;
        Ok(())
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib matrix::tests::append_columns
```

Expected: `test result: ok. 2 passed`.

- [ ] **Step 5: Run the full `tensor4all-tensorbackend` test suite to check for regressions**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib
```

Expected: all tests pass (matches the pre-change pass count — this step only added tests, changed nothing else yet).

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/matrix.rs crates/tensor4all-tensorbackend/src/matrix/tests/mod.rs
git commit -m "$(cat <<'EOF'
perf(tensorbackend): add Matrix::append_columns for in-place growth

Adds a pub(crate) method that extends a matrix's existing column-major
Vec<T> buffer via extend_from_slice instead of allocating fresh,
reusing Rust's built-in amortized-doubling Vec growth. Not yet wired
into any caller.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3ZQEnU8Hn1pWQBAAogR93
EOF
)"
```

---

## Task 3: Wire `append_columns` into `commit_full_rank_block`

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/incremental_qr.rs:308-356` (`commit_full_rank_block`)

**Interfaces:**
- Consumes: `Matrix::<T>::append_columns` from Task 2.

- [ ] **Step 1: Replace the body of `commit_full_rank_block`**

The current body (lines 308-356) is:

```rust
    fn commit_full_rank_block(
        &mut self,
        projection: Matrix<T>,
        appended_q: Matrix<T>,
        appended_r: Matrix<T>,
    ) -> std::result::Result<(), BackendLinalgError> {
        let old_rank = self.q.ncols();
        let old_column_count = self.r.ncols();
        let appended_rank = appended_q.ncols();
        if projection.nrows() != old_rank
            || projection.ncols() != appended_rank
            || appended_r.nrows() != appended_rank
            || appended_r.ncols() != appended_rank
        {
            return Err(anyhow!(
                "incremental QR backend update returned incompatible blocks: projection {}x{}, Q' {}x{}, R'' {}x{}",
                projection.nrows(),
                projection.ncols(),
                appended_q.nrows(),
                appended_q.ncols(),
                appended_r.nrows(),
                appended_r.ncols()
            )
            .into());
        }

        let q = concatenate_columns(&self.q, &appended_q)?;
        let r = assemble_r(&self.r, &projection, &appended_r)?;
        let new_rank = q.ncols();
        let new_column_count = r.ncols();
        let inverse_adjoint = if new_rank == new_column_count {
            if old_rank == old_column_count {
                if let Some(previous) = self.inverse_adjoint.as_ref() {
                    Some(update_inverse_adjoint(previous, &projection, &appended_r)?)
                } else {
                    try_inverse_adjoint(&r)
                }
            } else {
                try_inverse_adjoint(&r)
            }
        } else {
            None
        };

        self.q = q;
        self.r = r;
        self.inverse_adjoint = inverse_adjoint;
        Ok(())
    }
```

Replace the body from `let q = concatenate_columns(...)` through `self.q = q;` (i.e. keep the validation block and the leading `let`s unchanged) with:

```rust
        let r = assemble_r(&self.r, &projection, &appended_r)?;
        let new_rank = old_rank
            .checked_add(appended_rank)
            .ok_or_else(|| anyhow!("incremental QR rank overflow"))?;
        let new_column_count = r.ncols();
        let inverse_adjoint = if new_rank == new_column_count {
            if old_rank == old_column_count {
                if let Some(previous) = self.inverse_adjoint.as_ref() {
                    Some(update_inverse_adjoint(previous, &projection, &appended_r)?)
                } else {
                    try_inverse_adjoint(&r)
                }
            } else {
                try_inverse_adjoint(&r)
            }
        } else {
            None
        };

        self.q
            .append_columns(&appended_q)
            .map_err(|error| anyhow!("incremental QR Q append failed: {error}"))?;
        self.r = r;
        self.inverse_adjoint = inverse_adjoint;
        Ok(())
    }
```

Note: `append_columns` returns `MatrixShapeError`, not `BackendLinalgError` (`commit_full_rank_block`'s return type), and there is no `From<MatrixShapeError> for BackendLinalgError` impl — only `From<anyhow::Error>` (see `backend.rs:310-314`). The `.map_err(|error| anyhow!(...))?` form above matches the exact convention already used a few lines earlier in the *original* `commit_dependent_column` for the same situation (`Matrix::try_zeros(...).map_err(|error| anyhow!("incremental QR R allocation failed: {error}"))?;`) — wrap into `anyhow::Error` inside `map_err`, then let `?`'s own `From<anyhow::Error>` conversion do the rest. Do not use a bare `?` directly on `append_columns`'s result — it will not compile.

Note why this ordering preserves atomicity (self is left fully unchanged on any error path): every fallible step (`assemble_r`, the `checked_add`, `update_inverse_adjoint`) now runs and is fully resolved into local variables (`r`, `new_rank`, `inverse_adjoint`) *before* `self.q.append_columns(...)` — the one call that mutates `self` — executes. `self.q.append_columns` can itself only fail on the same overflow condition already checked via `new_rank` above (`self.q.ncols() == old_rank` is still true at that point, unchanged since the top of the function), so by construction it cannot fail here; it is called last regardless, to keep `self` mutation fully confined to the end of the function rather than relying on that guarantee silently.

- [ ] **Step 2: Verify `concatenate_columns` now has zero callers**

```bash
grep -n "concatenate_columns" crates/tensor4all-tensorbackend/src/incremental_qr.rs
```

Expected: only the function definition itself (`fn concatenate_columns<T>(`) remains, no call sites. If any other call site exists, stop and re-examine before proceeding to Task 4 — do not delete a function that's still used.

- [ ] **Step 3: Run the full incremental_qr test suite**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib incremental_qr
```

Expected: all 14 existing tests plus the 2 from Task 2 pass (16 total). These tests check reconstruction, orthogonality, rank-deficiency, and cross-checks against direct backend QR — they are the correctness safety net for this change.

- [ ] **Step 4: Run the full `tensor4all-tensorbackend` test suite**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib
```

Expected: all tests pass, same count as Task 2 Step 5 plus nothing removed.

- [ ] **Step 5: Run the SRC/zipup contraction test suite in `tensor4all-treetn`**

```bash
cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib treetn::contraction
```

Expected: all 63 tests pass (this is the count already established earlier this session; SRC's adaptive/fixed contraction tests exercise `IncrementalQr::append` end-to-end through `factorize_probe_columns`, so this is the most direct regression check that `Q`'s accumulated values are still correct after the storage change).

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/incremental_qr.rs
git commit -m "$(cat <<'EOF'
perf(tensorbackend): grow Q in place instead of rebuilding on every append

commit_full_rank_block now extends the existing Q buffer via
Matrix::append_columns instead of calling concatenate_columns to build
a fresh matrix and copy all of Q's prior columns into it on every
IncrementalQr::append call. R/R^{-dagger} are unchanged in this
commit (they grow in both dimensions, unlike Q, and are deferred to a
follow-up per docs/superpowers/specs/2026-08-30-incremental-qr-q-growth-design.md).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3ZQEnU8Hn1pWQBAAogR93
EOF
)"
```

---

## Task 4: Remove the now-unused `concatenate_columns` function

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/incremental_qr.rs:702-731`

**Interfaces:** none (pure deletion; Task 3 Step 2 already confirmed no remaining callers).

- [ ] **Step 1: Delete the function**

Delete lines 702-731 (the entire `fn concatenate_columns<T>(...) { ... }` definition, from its opening `fn concatenate_columns<T>(` through its closing `}`).

- [ ] **Step 2: Verify the crate still builds**

```bash
cargo check --manifest-path crates/tensor4all-tensorbackend/Cargo.toml
```

Expected: clean build, no `unused function` warning (since it's fully deleted, not just unreferenced) and no errors.

- [ ] **Step 3: Run the full test suite once more**

```bash
cargo test --manifest-path crates/tensor4all-tensorbackend/Cargo.toml --lib
```

Expected: same pass count as Task 3 Step 4.

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/incremental_qr.rs
git commit -m "$(cat <<'EOF'
chore(tensorbackend): remove concatenate_columns, superseded by append_columns

Its only caller was replaced in the previous commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01C3ZQEnU8Hn1pWQBAAogR93
EOF
)"
```

---

## Task 5: Verify the performance improvement

**Files:** none (measurement only).

**Interfaces:** none.

- [ ] **Step 1: Re-run the exact same benchmark sweep as Task 1**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
cargo build --release --manifest-path crates/tensor4all-treetn/Cargo.toml --example benchmark_src
for rank in 8 16 32 64 96 128 160 192; do
  RAYON_NUM_THREADS=1 T4A_BENCH_SKIP_EXACT=1 \
    ./target/release/examples/benchmark_src 16 128 2 mpo-mps 3 false "$rank" \
    | grep 'case=mpo-mps/src-adaptive'
done
```

Save output to `/tmp/claude-0/-root-projects-gw-rs/872136a4-bcbd-4dc2-b4df-4f45f14c1f30/scratchpad/q-growth-after.txt`.

- [ ] **Step 2: Compare against the Task 1 baseline**

Diff the two files' elapsed times point by point (same 8 target ranks in the same order). Report the ratio (`baseline / after`) at each rank. Expect the improvement to grow with rank — at low ranks the fixed per-call overhead dominates and the ratio should be close to 1; at rank=192 (the top of the sweep, where `Q` accumulates the most columns before the run ends) the ratio should be visibly greater than 1, since that's where the O(final_rank²) vs. O(final_rank) gap is most pronounced.

- [ ] **Step 3: Record the comparison table in this plan file**

Append a results table under this task (target rank | baseline (s) | after (s) | ratio) so the outcome is preserved alongside the plan.

- [ ] **Step 4: Decide on the follow-up (`R`/`R^{-†}`) step**

If the ratio at rank=192 is small (say, under ~1.2x) even after this fix, that's a signal `Q`'s copy cost wasn't actually the dominant term in practice, and `R`/`R^{-†}`'s remaining O(n²) cost (or something else entirely) may be worth investigating before doing the harder `R` growth redesign. If the ratio is substantial, report the finding and ask the user whether to proceed with a `R`/`R^{-†}` design pass next, following the same brainstorming → spec → plan process as this task.

- [ ] **Step 5: Report results to the user**

Do not commit anything in this task — it's pure measurement and reporting. Summarize Task 1's baseline, Task 5's after numbers, and the recommendation from Step 4 back to the user.
