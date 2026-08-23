# Dense `IdxTensor` to `TensorTrain` TT-SVD

## Summary

Issue #665 exposed a discoverability gap: downstream users could compose TT-SVD from `factorize`, but `tensor4all-itensorlike` had no owning high-level entry point. This change adds `TensorTrain::from_dense` and a runnable guide.

## Evidence reviewed

- `tensor4all-itensorlike::TensorTrain` construction and canonical metadata
- `tensor4all-treetn::factorize_tensor_to_treetn_with`
- `tensor4all-core::{FactorizeOptions, SvdOptions, SvdTruncationPolicy}`
- dense layout, public-boundary, numerical, documentation, and memory rules

## Decisions

- Reuse the existing general TreeTN decomposition with a generated chain topology; do not duplicate the sequential factorization loop.
- Accept the existing `SvdOptions` rather than add another options type. Only the SVD policy and maximum bond dimension are relevant to TT-SVD.
- Require each full dense index exactly once and use the supplied slice as TT site order. Full index identity preserves prime/tag distinctions.
- Sweep left-to-right and record unitary canonical metadata with the final site as orthogonality center.
- Keep the path explicitly dense and document local-versus-global truncation error and workspace cost.

## Alternatives rejected

- A free function would be less discoverable than the constructor users already attempted.
- Accepting `FactorizeOptions` would expose irrelevant QR/LU/CI and canonical-direction settings.
- A new TT-SVD options wrapper would duplicate `SvdOptions`.

## Verification

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo nextest run --release --workspace` (3186 passed)
- `cargo test --doc --release --workspace` (933 passed)
- `cargo doc --workspace --no-deps`
- `scripts/test-mdbook.sh`
- source-blind external path-dependency example copied from the guide: release build and run passed
- repository-rules review script tests and dry-run passed

Coverage includes f64, Complex64, truncation, invalid site sets/options, canonical metadata, reconstruction residuals, and same-ID/different-prime index identity.
