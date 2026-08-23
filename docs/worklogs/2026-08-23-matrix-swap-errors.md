# Fallible matrix row and column swaps

## Summary

Issue #667 found that public `swap_rows` and `swap_cols` converted invalid caller indices into panics. Both now return `Result<(), MatrixShapeError>` and validate before mutation or the equal-index no-op.

## Evidence reviewed

- every workspace caller of `swap_rows` and `swap_cols`
- existing `MatrixShapeError` and matrix layout invariants
- related public matrix helpers with index/shape preconditions
- repository public-boundary and Rust indexing rules

## Decisions

- Extend the existing crate-local error with row/column out-of-bounds variants; do not add a second matrix-index error type.
- Preserve the in-place algorithms after one boundary validation pass.
- Validate `a == b` before returning, so an equal out-of-range pair is still rejected.
- Assert that every error path leaves the matrix unchanged.

## Adjacent audit

`submatrix` and `submatrix_argmax` still panic on invalid public indices/ranges, while compatibility constructors such as `from_vec2d` intentionally panic and already have fallible alternatives. Those surfaces are not silently changed in #667; the helpers without fallible alternatives should be tracked separately.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace` (3182 passed)
- `cargo test --doc --release --workspace` (932 passed)
- `cargo doc --workspace --no-deps`
- `python3 scripts/audit-library-panics.py`
- repository-rules review script tests and dry-run passed
