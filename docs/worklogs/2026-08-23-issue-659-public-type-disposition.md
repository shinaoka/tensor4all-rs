# Issue #659 public type disposition worklog

Date: 2026-08-23

## Scope

Resolve the incomplete #566 “Collapse duplicate types” claim without coupling the repository's two tensor-train stacks or introducing one over-constrained scalar trait.

## Contract read

- both tensor-train error enums and all workspace callers;
- core and tensorbackend scalar value implementations and conversions;
- core `Scalar`, MatrixLUCI's duplicate trait, and backend/storage/TT/ACI capability traits;
- crate manifests and the documented two-stack dependency boundary;
- #566 evidence ledger and active architecture/user guidance.

## Design and review

`docs/design/issue-659-public-type-disposition.md` records the exact disposition. Independent read-only pre-implementation review used `reviewer-flash-opencode-go`.

- Round 1: blocking caller-migration findings for downstream `#[from]` wrappers, MatrixLUCI internal/tests, and supertrait UFCS calls.
- Round 2 after incorporating those paths: **Correct-to-implement**.
- Focused amendment review: **Correct-to-implement** for `BackendScalar`, avoiding a new collision with core's `Scalar` trait.

## Implementation

- rename the positional error to `SimpleTensorTrainError` without a compatibility alias;
- retain itensorlike `TensorTrainError` for its distinct tree-based invariants;
- remove tensorbackend's `AnyScalar` alias and rename/export its value as `BackendScalar`;
- make the directly named `MatrixLuciScalar` extend core `Scalar`, deleting duplicate arithmetic declarations and implementations;
- migrate downstream error wrappers and scalar UFCS calls;
- document error and scalar-capability ownership;
- correct #566 ledger item 31 and its inaccurate final-audit claim.

## Verification

- focused release tests for all directly affected crates: 1750 passed;
- `cargo check --workspace --all-targets`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- `cargo nextest run --release --workspace`: 3188 passed, 17 skipped;
- `cargo test --doc --release --workspace`: 933 passed;
- `cargo doc --workspace --no-deps`;
- `cargo run -p xtask --release -- api-dump`;
- `scripts/test-mdbook.sh`;
- library-panic, public-error-doc, crate-boundary, and checker-test suites;
- exact stale-public-name/source searches from the approved design.

Independent post-implementation review was split into two exhaustive read-only slices after a full-diff timeout; both slices passed, and the final coverage synthesis returned **Correct-to-merge**. GitHub CI remains pending.
