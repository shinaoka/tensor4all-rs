# Small correctness issue batch (#653, #685)

## Summary

Fixed two independent low-risk correctness issues in one batch: deterministic
initial pivots for zero-at-origin QuanticsTCI tests, and scalar-identity norms
for empty `tensor4all-itensorlike::TensorTrain` values.

## Reviewed material

- `AGENTS.md`, `REPOSITORY_RULES.md`, root `README.md`, and workspace coding rules
- Shared common, Rust, testing, and numerical agent rules
- Generated API inventories for `tensor4all-itensorlike` and
  `tensor4all-quanticstci`
- `TensorTrain::{inner,norm_squared,norm_squared_fast_path}` and
  `TensorConstructionLike::scalar_one`
- Every QuanticsTCI test using `f(i, j) = i + j`, including the sibling test not
  named in #653

## Decisions

- Preserve the tested function in #653 and provide the known nonzero grid pivot
  `[1, 0]`. This removes dependence on unseeded random pivots without changing
  production interpolation behavior or expected values.
- Return one from both empty `TensorTrain` metric fast paths because this type's
  public `scalar_one` implementation explicitly defines the empty train as the
  multiplicative identity.
- Do not change `SimpleTensorTrain::inner_product`: its empty constructor erases
  the requested constant value and it has no equivalent explicit scalar-one
  contract, so changing that separate API requires a dedicated contract decision.

## Verification

- `cargo fmt --all -- --check`
- `cargo test -p tensor4all-itensorlike --release`
- `cargo test -p tensor4all-quanticstci --release`
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`
- `cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5`
- `cargo test --profile ci -p tensor4all-hdf5`
- `cargo test --doc --profile ci --workspace -j 8`
- `cargo doc --workspace --no-deps`
- `./scripts/test-mdbook.sh`
- `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run`
- `python3 scripts/test-repository-rules-review.py`
- `cargo run -p xtask --release -- api-dump`
- `git diff --check`

The first full itensorlike run found an older integration assertion that encoded
the buggy zero result. It was replaced by the focused scalar-one contract test,
and the full gate was rerun successfully.
