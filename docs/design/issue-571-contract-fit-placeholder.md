# Design: #571 — make simplett `contract_fit` fail loudly

Status: proposed. Review gate: luna read-only design review before
implementation, then luna read-only diff review.

## Problem

`tensor4all-simplett::mpo::contract_fit` is presented as a variational
(DMRG-like) MPO fitting algorithm, but `update_two_site_core` is only a
placeholder that returns `Ok(true)` without changing the result. The initial
result is the naive contraction, so every call returns the naive result while
spending up to `max_sweeps` on dead environment-building and sweep loops. The
convergence check is also unfinished. This makes `Fit` silently claim a
contraction method that it does not implement.

The real TreeTN fitting implementation used by `tensor4all-itensorlike` and
the TreeTN C API is separate and is not changed by this issue.

## Decision

Make the simplett API fail before doing any contraction work:

1. Keep the existing length and shared physical-dimension validation so
   malformed inputs continue to produce their specific errors.
2. For every otherwise valid input, including an empty MPO, return a new typed
   `MPOError::Unsupported` explaining that variational MPO fitting is not
   implemented and suggesting `contract_naive` or `contract_zipup`.
3. Ignore the `initial` argument intentionally while Fit is unavailable; it
   remains in the signature as part of the future fitting API surface.
4. Remove the unreachable placeholder environment/update implementation and
   its tests. This removes the dead sweeps rather than retaining code that can
   never produce a valid fit.
5. Update `FitOptions`, `contract_fit`, `ContractionAlgorithm::Fit`, module
   docs, doctests, and dispatch tests to state and assert the unsupported
   behavior. `FitOptions` remains as the future configuration surface; its
   fields are not consumed while the algorithm is unavailable.

`ContractionAlgorithm::Fit` continues to dispatch through `contract_fit`, so
callers receive the same typed error whether they use the function directly or
the unified dispatcher. No TreeTN APIs or C API behavior are changed.

## Error and compatibility

This is an intentional behavior correction for a function that returned an
incorrectly labeled result. Valid calls that previously returned a naive result
will now return `MPOError::Unsupported`; callers must select `Naive` or `ZipUp`
until a real simplett fit implementation is added. Existing invalid-input
errors remain unchanged.

## Verification

- Unit tests cover the unsupported result for empty and non-empty valid MPOs,
  plus preservation of length/shared dimension validation.
- Dispatcher tests and the public doctest cover
  `ContractionAlgorithm::Fit` returning the same typed error.
- `cargo fmt --all`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --release -p tensor4all-simplett`
- `cargo nextest run --release --workspace`
- `cargo test --doc --release --workspace`
- `./scripts/test-mdbook.sh`
- `cargo doc --workspace --no-deps`
- repository-rules review and `git diff --check`

## Review verdicts

- Design (pre-implementation), luna: needs-fix (empty-input and unused-
  `initial` semantics) → fixed and re-reviewed: Correct-to-merge.
- Diff (post-implementation), luna: Correct-to-merge.
