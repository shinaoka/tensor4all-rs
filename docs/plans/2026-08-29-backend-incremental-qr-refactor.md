# Backend-Native Incremental QR Refactor Implementation Plan

## Goal

Replace `IncrementalQr`'s scalar, element-indexed Householder implementation
with an independently derived backend-native incremental update while
preserving its public API, rank-deficiency behavior, error estimator, and SRC
consumer contract.

The implementation follows the block update derived in Appendix C.3 of
Camaño--Epperly--Tropp, arXiv:2504.06475:

```text
B  = Q† Y'
Z  = Y' - Q B
Z  = Q' R''
[Y Y'] = [Q Q'] [[R B], [0 R'']].
```

The paper's displayed `Z := Y - Q R'` is treated as a typographical error:
the dimensions and the following block factorization require `Y'`.

## Scope and constraints

- Keep the public `IncrementalQr` API unchanged: `new`, `from_factors`,
  `append`, `q`, `q_columns`, `rank`, `r`, and `error_estimate`.
- Independently derive the Rust structure from the paper. Do not translate the
  unlicensed reference Python or C++ implementation.
- Use `tensor4all-tensorbackend`'s existing `qr_backend` and `mat_mul`
  primitives for decompositions and dense products.
- Use two-pass block classical Gram--Schmidt (BCGS2): the second projection
  corrects loss of orthogonality from the first pass. The residual QR itself
  uses tenferro's Householder backend.
- Preserve the existing per-column residual threshold and rank-saturation
  behavior. The normal full-rank path remains blocked; a block whose residual
  QR indicates numerical rank loss falls back to ordered one-column updates so
  dependent columns can be skipped without losing later independent columns.
- Preserve the block update of `R^{-†}` used by the Appendix C estimator.
- Do not change tolerances in existing tests.
- Keep the pre-existing untracked benchmark examples uncommitted unless a
  later explicit decision promotes them.
- Do not push or open a pull request without user approval.

## Files

- Modify `crates/tensor4all-tensorbackend/src/incremental_qr.rs`: replace the
  private reflector representation and scalar kernels with explicit-Q,
  backend-native BCGS2 helpers.
- Create `crates/tensor4all-tensorbackend/src/incremental_qr/tests.rs`: move
  the existing unit tests out of the production file and add differential,
  near-dependent, complex, and source-contract coverage.
- Modify `docs/worklogs/2026-08-26-treetn-src-contraction.md` or create a
  focused `docs/worklogs/2026-08-29-backend-incremental-qr-refactor.md`:
  record derivation, baseline/candidate benchmark protocol, results, and
  remaining upstream limitation.
- Use the untracked
  `crates/tensor4all-treetn/examples/benchmark_src.rs` only as the local paired
  benchmark driver.

## Task 1: Establish regression tests before implementation

1. Move the existing `incremental_qr.rs` test module verbatim to
   `incremental_qr/tests.rs`, leaving `#[cfg(test)] mod tests;` in the owner.
2. Add a differential full-rank test that:
   - constructs a deterministic tall matrix in multiple appended blocks;
   - factorizes the complete matrix directly with `qr_backend`;
   - compares incremental and direct reconstruction residuals;
   - compares the two orthogonal projectors rather than raw signed/phased QR
     columns;
   - compares `error_estimate().error` and `.norm`.
3. Add a near-dependent append test that verifies reconstruction and
   `Q†Q = I` after several blocks. Cover both `f64` and `Complex64` through a
   shared generic helper where practical.
4. Add a source-contract test asserting that production
   `incremental_qr.rs` calls `qr_backend` and no longer defines
   `householder_factor`, `householder_vector`, `apply_reflector`,
   `apply_q_adjoint`, or `form_q`.
5. Run the focused test and confirm the source-contract assertion fails on the
   old implementation while the numerical differential test characterizes the
   existing behavior:

   ```bash
   cargo test --release -p tensor4all-tensorbackend incremental_qr -- --nocapture
   ```

## Task 2: Replace reflector state with explicit backend factors

1. Change the private representation to:

   ```rust
   pub struct IncrementalQr<T> {
       q: Matrix<T>,
       r: Matrix<T>,
       inverse_adjoint: Option<Matrix<T>>,
   }
   ```

2. Add a private `factorize_backend` helper that converts `Matrix<T>` to the
   existing typed tensor boundary, calls `qr_backend`, converts both outputs
   back to `Matrix<T>`, and maps conversion failures to
   `BackendLinalgError` with operation context.
3. Make `new` validate its existing preconditions and call
   `factorize_backend` once.
4. Make `from_factors` retain its current behavior for any compatible thin
   factors: backend-factorize the supplied `q` as `q = Q_h R_q`, then store
   `Q_h` and `R_q r`. Do not assume the caller's `q` is exactly orthonormal.
5. Make `q()` clone the stored explicit factor, `rank()` read `q.ncols()`, and
   `q_columns()` copy the requested contiguous column block after the existing
   checked-range validation.
6. Update rustdoc to describe explicit-Q state and backend QR accurately; keep
   all public examples runnable and assertion-bearing.
7. Run focused tests.

## Task 3: Implement BCGS2 append with rank-preserving fallback

1. Add private helpers for:
   - forming a conjugate transpose in column-major `Matrix` layout;
   - checked same-shape matrix addition/subtraction used only around backend
     products;
   - horizontally assembling Q blocks and assembling the block-triangular R.
2. Implement BCGS2 for an append block:

   ```text
   B1 = Q† Y'
   Z1 = Y' - Q B1
   B2 = Q† Z1
   Z2 = Z1 - Q B2
   B  = B1 + B2
   (Q', R'') = qr_backend(Z2)
   ```

3. Compute the existing residual threshold once from the original appended
   block. Inspect the diagonal of `R''` in append order:
   - if every diagonal magnitude exceeds the threshold, commit the whole
     block in one update;
   - otherwise discard the speculative factors and replay the original block
     one column at a time through the same BCGS2/backend-QR helper, skipping a
     column when its twice-projected residual norm is at or below the original
     block threshold.
4. For a skipped column, append its corrected projection coefficients to R
   without increasing Q's width. This preserves the current rectangular-R
   saturation signal.
5. For accepted columns, assemble `[Q Q']` and `[[R B], [0 R'']]` with checked
   dimensions and commit only after every backend operation succeeds.
6. Refactor `update_inverse_adjoint` to consume the projection block `B` and
   residual triangular block `R''` directly. Preserve the paper's formula and
   retain the full recomputation fallback for non-square prior state.
7. Run focused tests after the minimal implementation and fix only actual
   regressions; do not relax tolerances.

## Task 4: Validate SRC consumers

Run the owner-crate and TreeTN SRC tests in release mode:

```bash
cargo test --release -p tensor4all-tensorbackend incremental_qr -- --nocapture
cargo test --release -p tensor4all-treetn src_ -- --nocapture
```

Then run the complete tests for both affected crates:

```bash
cargo test --release -p tensor4all-tensorbackend
cargo test --release -p tensor4all-treetn
```

The adaptive chain and branched-tree tests must still honor rank caps, and the
fixed-rank tests must remain unchanged because they do not use
`IncrementalQr`.

## Task 5: Format, lint, docs, and repository checks

1. Incorporate the one fetched `origin/main` commit before treating results as
   final, resolving only genuine overlaps and preserving the user's untracked
   files.
2. Run:

   ```bash
   cargo fmt --all --check
   cargo clippy --release -p tensor4all-tensorbackend -p tensor4all-treetn --all-targets -- -D warnings
   cargo test --doc --release -p tensor4all-tensorbackend -p tensor4all-treetn
   python3 scripts/check-public-error-docs.py
   ```

3. Run any additional changed-file or mdBook checker required by the final
   diff. No public API name changes are expected, so tutorial edits should not
   be necessary; verify that assumption from the diff.

## Task 6: Paired adaptive SRC benchmark

Use the same worktree, release profile, benchmark source, and fixed thread
configuration for baseline and candidate:

```bash
RAYON_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
T4A_BENCH_SKIP_EXACT=1 \
cargo run --release -p tensor4all-treetn --example benchmark_src -- \
  10 4 3 mpo-mps 3 false
```

Baseline commit: `86d3cbf`.

Baseline host: AMD Ryzen 9 6900HX under WSL2, Rust 1.98.0. Recorded initial
per-run timings were Zip-up 0.019446 s, fixed SRC 0.014206 s, and adaptive SRC
0.040759 s. These three-repetition values are orientation data, not a
statistical performance claim.

After correctness validation, run the complete baseline/candidate case list
with more repetitions under the same settings. Report all results, including
regressions or inconclusive host-noise outcomes. The primary comparison is
adaptive SRC candidate versus adaptive SRC baseline; fixed SRC is a
non-regression/control case because it bypasses `IncrementalQr`.

## Task 7: Worklog and upstream follow-up

1. Record the independent derivation, the use of BCGS2 rather than the
   reference implementation's Householder update, validation commands, paired
   timing table, and remaining stability/device limitations in the focused
   worklog.
2. Prepare—but do not submit—an upstream tenferro issue draft requesting a
   backend/device-neutral incremental Householder QR abstraction. The draft
   should describe the required compact-reflector and reflector-application
   capabilities, affected CPU/CUDA/eager/traced/AD surfaces, and the
   tensor4all SRC use case without proposing a copied reference
   implementation.
3. Show the draft to the user and obtain separate explicit approval before
   creating any upstream issue.
