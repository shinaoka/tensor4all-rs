# Issue #566 Task 8 checkpoint / handoff

**Status:** Task 8 remains in progress. This is a WIP checkpoint, not a completion report.

## Branch and commit context

- Worktree: `/home/shinaoka/tensor4all/tensor4all-rs/.worktrees/issue-566-remediation`
- Branch: `audit/issue-566-remediation`
- Source HEAD when the API work was interrupted: `e1b0dfcc31d3` (`fix(scanner): close Task 8 review gaps`)
- Observed base: `origin/main` at `32852babf087`; merge-base was `ae655a9ec08a`
- Observed branch relation before this handoff commit: **ahead 45, behind 3** relative to `origin/main`
- No pull request exists and nothing has been pushed.
- The latest fully validated committed API checkpoint is `184e949bd620` (`fix(api): keep compact tensor metrics allocation-safe`).

The seven source/test files below were dirty at `e1b0dfc`; the scanner changes are already committed in `e1b0dfc` and are not part of this API checkpoint.

## Dirty API checkpoint

The interrupted checkpoint implements compact-support, allocation-safe tensor metrics and scaling:

- `crates/tensor4all-core/src/defaults/tensordynlen.rs`
  - Streams `isapprox` over compact payload support rather than the logical tensor domain, including aligned/permuted index layouts and unmatched support points.
  - Keeps exact comparisons exact and handles finite values, matching/mismatched infinities, and NaN errors during tolerant comparison.
  - Computes `norm`, `norm_squared`, and `maxabs` from compact payload values with an `f64` LASSQ accumulator rather than source-dtype `self * conj(self)` materialization.
  - Allows tracked scaling of materialized structured storage by converting only its compact payload and returning compact structured storage.
- `crates/tensor4all-tensorbackend/src/storage.rs`
  - Checks payload products, axis-class ranks, strides, and offsets before iteration or access.
  - Replaces the allocating/logical `scalar_at` path and removes the public offset-scanning accessor in favor of direct compact payload-coordinate lookup.
- `crates/tensor4all-core/tests/tensor_api_correction.rs`
  - Adds f32-extreme metric coverage and a large public structured-storage tracked-scaling regression.
- `crates/tensor4all-core/tests/tensor_comparison.rs`
  - Adds permuted/dense layout, large compact-support, and matching-infinity comparison coverage.
- `crates/tensor4all-core/tests/tensor_mask.rs`
- `crates/tensor4all-core/tests/tensor_native_ad.rs`
  - Update compact payload-coordinate callers for the storage accessor.
- `crates/tensor4all-tensorbackend/src/storage/tests/mod.rs`
  - Adds checked-product and direct compact-coordinate access regressions, including strided storage.

## Validation evidence from the interrupted API session

Focused checks recorded in the session logs:

- `cargo test --release -p tensor4all-core --test tensor_comparison --no-fail-fast`: **20 passed**.
- `cargo test --release -p tensor4all-core --test tensor_api_correction --no-fail-fast`: **14 passed**.
- `cargo test --release -p tensor4all-core --test tensor_mask --no-fail-fast`: **18 passed**.
- `cargo test --release -p tensor4all-core --test tensor_native_ad --no-fail-fast`: **12 passed**.
- `cargo test --release -p tensor4all-tensorbackend storage --no-fail-fast`: **77 passed**.
- `cargo test --release -p tensor4all-core --no-fail-fast`: **833 passed, 1 ignored**.
- `cargo test --release -p tensor4all-tensorbackend --no-fail-fast`: **313 passed, 2 ignored**.
- `cargo clippy --release -p tensor4all-core -p tensor4all-tensorbackend --all-targets -- -D warnings`: **no issues found**.
- `cargo test --doc --release -p tensor4all-core`: **179 passed**.
- `cargo test --doc --release -p tensor4all-tensorbackend`: **133 passed**.
- `cargo fmt --all -- --check`: passed.
- The large compact-support test was rerun successfully after replacing an exact floating-point `norm_squared` assertion with a `1e-8` residual check; review that tolerance before treating the API checkpoint as complete.

The final full-workspace command was started but interrupted at the workspace clippy stage:

```text
cargo fmt --all -- --check && cargo clippy --release --workspace --all-targets -- -D warnings && cargo nextest run --release --workspace && cargo test --doc --release --workspace && cargo doc --workspace --no-deps
```

Therefore full workspace clippy, nextest, doctests, mdBook, API generation, scanner validation, and independent API review are **not complete for this dirty checkpoint**. Do not infer completion from the focused results.

For comparison, committed checkpoint `184e949bd620` was the latest fully validated point: workspace nextest reported **2704/2704 passed, 14 skipped**, and doctest, mdBook, API, and scanner checks were reported green.

## Outstanding scanner review findings at `e1b0dfc`

These are review findings against the scanner commit and must not be silently closed:

1. **Critical:** dep-info sources are flattened across Make rules. Sources must be associated with the exact matching artifact output; an unrelated rule must not satisfy source validation for the selected compiler artifact.
2. **Important:** nested `macro_rules!` matchers can still be reported as assertions. Nested macro definitions must be excluded while scanning an outer transcriber.

The Make continuation separator finding was reported fixed. Scanner review packages are under:

```text
.superpowers/sdd/2026-08-08-issue-566-pr1-soundness-ci/
```

Relevant diffs include `review-ce24ca2..184e949.diff` for the prior API checkpoint and `review-184e949..e1b0dfc.diff` for the scanner checkpoint. The corresponding `.pi-subagents/` artifacts are local review evidence only and must remain untracked/excluded.

## Resume sequence

1. Start from the WIP handoff commit and read this file; preserve the seven API/test files as one reviewable checkpoint.
2. Keep `.pi-subagents/` excluded. Inspect the API diff and obtain an independent specification and correctness review, including the tolerance and compact-support mapping paths.
3. Run the focused core/backend release tests, focused clippy, and formatting check again after any confirmed corrections.
4. Resolve the two outstanding scanner findings and run the scanner self-test/audit.
5. Run the complete required validation for this API checkpoint: workspace clippy, release nextest, release doctests, `./scripts/test-mdbook.sh`, API dump/checks, and scanner checks. Do not call the checkpoint complete until all results are fresh and green.
6. Only after review and full validation update the Task 8 ledger/work log and decide whether the branch is ready for the PR1 boundary.

Task 8 is still in progress. Tasks 9–13, PR2–4, and the shared-rules prerequisite remain pending. There is no PR yet.

## Exclusion

`.pi-subagents/` is untracked session infrastructure and is intentionally excluded from the handoff commit; never add it with a broad `git add`.
