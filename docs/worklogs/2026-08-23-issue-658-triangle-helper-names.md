# Issue #658 triangle helper naming worklog

Date: 2026-08-23

## Scope

Correct private helper names and documentation in `quanticstransform::cumsum` so they describe the matrix emitted, without changing public operator behavior.

## Implementation

- the `y > x` helper is now `lower_triangle_tensor`;
- the `y < x` helper is now `upper_triangle_tensor`;
- `triangle_mpo` maps each public `TriangleType` directly to the same-named helper;
- cumulative-sum documentation consistently describes a strict lower-triangular prefix sum;
- helper test and integration failure vocabulary use the corrected names.

## Verification

- `cargo fmt --all -- --check`;
- focused quanticstransform clippy with warnings denied;
- focused release tests: 170 passed;
- focused release doctests: 45 passed;
- full workspace clippy with warnings denied;
- full release workspace tests: 3188 passed, 17 skipped;
- full release workspace doctests: 933 passed;
- workspace rustdoc and public API inventory;
- library-panic, public-error-doc, and crate-boundary audits.

Existing dense integration tests pin both public triangle matrices and the cumulative-sum/lower-triangle equivalence. Independent read-only post-implementation review returned **Correct-to-merge**; its two nonblocking stale-doc findings were fixed.
