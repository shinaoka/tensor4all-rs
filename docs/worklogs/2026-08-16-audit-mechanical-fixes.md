# Audit mechanical fixes: issues #632, #635, and #636

## Session summary

Implement the mechanically specified audit fixes in one batch:

- #632: remove test-only state from the exported `t4a_tensor` ABI and mechanically verify the generated C header.
- #636: preserve the original thread-local diagnostic when `t4a_last_error_message` reports an undersized retrieval buffer.
- #635: validate numerical options before callbacks, allocation, no-op returns, or backend calls in TensorCI1, TensorCI2, TreeTCI, and the quantics Fourier constructor.

Initial base: `origin/main` at `c9ecb7f0d4eec4bbf6db51dc7648a4b2d84b4f34`.
Synchronized pre-PR base: `origin/main` at `39b1884b05133d1c4f195e296df238bc2b59b04d`.

## Code and documents read

- `AGENTS.md`, `REPOSITORY_RULES.md`, shared common/Rust/numerical rules
- `docs/CAPI_DESIGN.md`
- `.github/workflows/CI_rs.yml`
- `crates/tensor4all-capi/{src/lib.rs,src/types.rs,src/tests/mod.rs,src/tensor/tests/mod.rs,cbindgen.toml,include/tensor4all_capi.h}`
- `crates/tensor4all-tensorci/{src/error.rs,src/tensorci1.rs,src/tensorci2.rs}` and tests
- `crates/tensor4all-treetci/{src/error.rs,src/optimize.rs}` and tests
- `crates/tensor4all-quanticstransform/{src/error.rs,src/fourier.rs}` and tests
- CodeGraph caller/blast-radius reports for the edited public entries

## Design

### #632: keep test injection behind the opaque pointer

Keep `t4a_tensor` as a one-field opaque C wrapper in every build. Introduce a private Rust heap payload containing `InternalTensor` and, only under `cfg(test)`, the injected `TensorStorageError`. `_private` points to that private payload. `new`, `inner`, `Clone`, `Drop`, and test injection access the payload; no test-only field appears on the exported `#[repr(C)]` struct.

Add `scripts/check-capi-header.sh` that:

1. requires cbindgen 0.29.2 and asserts the generated `Generated with cbindgen:` version line;
2. regenerates the header to a temporary file;
3. diffs it against the committed header;
4. compiles a minimal include-only source as C11 and C++.

Run it in the maintenance-scripts CI job after `cargo install cbindgen --version 0.29.2 --locked`. Regenerate the committed header.

Rejected alternative: a test-only global/thread-local side table keyed by pointer. It adds lifecycle and cross-thread bookkeeping solely for tests; a private heap payload is smaller and keeps ownership local.

### #636: retrieval must not mutate the diagnostic

On an undersized `t4a_last_error_message` buffer, write the required length and return `T4A_BUFFER_TOO_SMALL` directly. Do not call the generic helper that replaces `LAST_ERROR`. Update the regression to retry with the reported size and assert byte-for-byte recovery of the original message. Clarify this preservation guarantee in both the function rustdoc (and therefore the generated C header) and `docs/CAPI_DESIGN.md`.

### #635: one mandatory validation seam per options type

Use crate-local typed configuration errors and small validation helpers:

- `TCI1Options`: finite nonnegative `tolerance` and `pivot_tolerance`; positive `max_iter`. Represent failures as a new `TCIError::InvalidConfiguration` variant.
- `TCI2Options`: finite nonnegative `tolerance` and `tol_margin_global_search`; positive `max_iter`, `ncheck_history`, and optional `max_bond_dim`. Keep `nsearch == 0` and `max_nglobal_pivot == 0` valid: existing public doctests use both to disable global search while still converging, so validation must preserve that contract. Represent failures as `TCIError::InvalidConfiguration`.
- `TreeTciOptions`: finite nonnegative `tolerance` in addition to existing checks; retain existing validation for iteration, bond dimension, and global-search margin. Represent failures as `TreeTciError::InvalidConfiguration` rather than an untyped operation error.
- `FourierOptions`: `sign` exactly `-1.0` or `1.0`, finite nonnegative `tolerance`, positive optional `max_bond_dim`, and positive `k`. Reuse `QuanticsTransformError::InvalidConfiguration`.

Call validation at every direct public entry before callbacks or allocation. Direct lower-level public optimization entry points and convenience constructors both validate, even when one normally delegates to the other. Validate the equivalent raw public parameters at `TensorCI1::add_pivot`, `TensorCI1::add_global_pivot`, `TensorCI2::sweep1site`, and `TensorCI2::make_canonical`: tolerances must be finite and nonnegative, and TensorCI2 raw entries require a positive bond dimension. `make_canonical` validates before its first delegated half-sweep so invalid input cannot invoke callbacks before the later parameterized sweep.

Tests use compact invalid-value tables covering negative, NaN, positive/negative infinity, and zero where prohibited. Add direct-entry regressions for `optimize_with_finder`, `integrate`, and TreeTCI `crossinterpolate2`, plus callback-before-validation checks for the raw-parameter entries. Pin zero tolerance as accepted for the TensorCI1 raw entries, and update every raw entry's `# Errors` documentation for `InvalidConfiguration`. Existing valid defaults and successful numerical paths remain unchanged. No tolerance is relaxed.

## Non-goals

- #633 inert options/input contract choices
- #634 PartitionedTT API redesign
- #637 generated Rust API inventory policy
- broad checked-arithmetic backlog #544
- unrelated C API output-slot semantics
- backward-compatibility shims; this repository is early-development

## Verification plan

Focused first:

```bash
cargo fmt --all -- --check
./scripts/check-capi-header.sh
cargo nextest run --release -p tensor4all-capi
cargo nextest run --release -p tensor4all-tensorci
cargo nextest run --release -p tensor4all-treetci
cargo nextest run --release -p tensor4all-quanticstransform
cargo test --doc --release -p tensor4all-capi -p tensor4all-tensorci -p tensor4all-treetci -p tensor4all-quanticstransform
```

Pre-PR gates:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo nextest run --release --workspace
cargo test --doc --release --workspace
./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
python3 scripts/check-coverage.py coverage.json
python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run
python3 scripts/test-repository-rules-review.py
python3 scripts/test-audit-library-panics.py
python3 scripts/audit-library-panics.py
python3 scripts/test-check-public-error-docs.py
python3 scripts/check-public-error-docs.py
python3 scripts/test-check-crate-boundaries.py
python3 scripts/check-crate-boundaries.py
```

Run every gate locally before creating the PR; focused tests do not substitute for coverage, docs, lint, or maintenance-script gates. Before merge, fetch `origin`, synchronize with current `origin/main`, rerun/monitor CI, and record reviewer-flash plus final reviewer-gpt verdicts.

## Remaining risks

- cbindgen output can vary across versions; pin 0.29.2 in CI and verify the version in the script.
- Validation may expose previously accepted invalid configurations; this is intentional under early-development policy.
- The full workspace test matrix and coverage job may reveal feature/path interactions not exercised by focused tests.

## #635 implementation and verification record

Implemented the reviewed validation seam without touching the pre-existing #632/#636 C API, header, CI, or maintenance-script changes:

- Added typed `TCIError::InvalidConfiguration` validation for TCI1/TCI2 options, including direct sweep, optimizer, and integration entry points.
- Added typed `TreeTciError::InvalidConfiguration` validation at both TreeTCI optimization entry points and the convenience constructor path.
- Added Fourier option validation using `QuanticsTransformError::InvalidConfiguration` at the public operator, `FTCore::new`, and private MPO construction paths.
- Preserved zero tolerance, `nsearch == 0`, and `max_nglobal_pivot == 0` as valid configurations.
- Added compact NaN/infinity/negative/zero regression tests and callback-before-validation checks; updated public `# Errors` documentation.

Verification performed:

```text
RED: cargo test --release -p tensor4all-tensorci --lib test_tci2_rejects_invalid_options_before_callback
     failed to compile because InvalidConfiguration was not yet implemented.
PASS: cargo fmt --all
PASS: cargo fmt --all -- --check
PASS: cargo nextest run --release -p tensor4all-tensorci (75 tests)
PASS: cargo nextest run --release -p tensor4all-treetci (57 tests)
PASS: cargo nextest run --release -p tensor4all-quanticstransform (170 tests)
PASS: cargo test --doc --release -p tensor4all-capi -p tensor4all-tensorci -p tensor4all-treetci -p tensor4all-quanticstransform (111 doctests)
PASS: python3 scripts/check-public-error-docs.py
PASS: git diff --check
```

Scope review: production/test changes are limited to `tensor4all-tensorci`, `tensor4all-treetci`, `tensor4all-quanticstransform`, and this appended worklog record; the existing #632/#636 files remain unchanged by #635.

## Review-gate record

- Pre-implementation design review for #632/#636: `reviewer-flash`, verdict **Correct-to-merge** before production edits.
- Post-implementation review for #632/#636: `reviewer-flash`, verdict **Correct-to-merge**.
- Pre-implementation design review for #635: `reviewer-flash`, verdict **Correct-to-merge** before production edits.
- Post-implementation review for the main #635 diff: `reviewer-flash`, verdict **Correct-to-merge**.
- Pre-implementation design review for the #635 raw-parameter/direct-entry addendum: `reviewer-gpt`, verdict **Correct-to-merge** before addendum edits.
- Post-implementation re-review for that addendum: `reviewer-flash`, verdict **Correct-to-merge**, with no Critical or Important findings.
- Final full-diff frontier confirmation: `reviewer-gpt` found one Important Fourier public error-contract mismatch; it was fixed with a typed `InvalidConfiguration` return and variant assertions, all local gates were rerun, and `reviewer-gpt` re-reviewed the complete staged diff with verdict **Correct-to-merge**.

## Synchronized pre-PR verification

After synchronizing the branch to `origin/main` at `39b1884b05133d1c4f195e296df238bc2b59b04d`, all required local gates passed:

```text
PASS: cargo fmt --all
PASS: cargo fmt --all -- --check
PASS: ./scripts/check-capi-header.sh (fresh cbindgen 0.29.2 header; C11 and C++ include compile)
PASS: cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
PASS: cargo nextest run --release --workspace (2815 passed, 14 skipped)
PASS: cargo test --doc --release --workspace (867 passed)
PASS: ./scripts/test-mdbook.sh (26 chapters)
PASS: cargo doc --workspace --no-deps
PASS: cargo llvm-cov --release --workspace --exclude tensor4all-hdf5 --json --output-path coverage.json
PASS: python3 scripts/check-coverage.py coverage.json (215/215 files)
PASS: python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run
PASS: python3 scripts/test-repository-rules-review.py (90 tests)
PASS: python3 scripts/test-audit-library-panics.py (11 tests)
PASS: python3 scripts/audit-library-panics.py (0 unbaselined findings; 0 stale entries)
PASS: python3 scripts/test-check-public-error-docs.py (15 tests)
PASS: python3 scripts/check-public-error-docs.py
PASS: python3 scripts/test-check-crate-boundaries.py (13 tests)
PASS: python3 scripts/check-crate-boundaries.py
PASS: git diff --check
```
