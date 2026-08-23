# Contributing to tensor4all-rs

Thank you for contributing. tensor4all-rs is in early development: public APIs may change, and fixes should remove obsolete paths rather than add compatibility shims.

## Start with an issue

Use the repository issue forms so the scope and acceptance criteria are visible before substantial work begins:

- [Bug report](https://github.com/tensor4all/tensor4all-rs/issues/new?template=bug_report.yml): include a minimal reproducer, expected behavior, Rust version, and platform when relevant.
- [Feature request](https://github.com/tensor4all/tensor4all-rs/issues/new?template=feature_request.yml): explain the use case, proposed behavior, and acceptance criteria. Link the related Tensor4all.jl issue for cross-repository work.

Small documentation and typo fixes may go directly to a pull request. For nontrivial, public-API, numerical, performance, or cross-crate changes, comment on the issue before implementation so maintainers can confirm the scope and dependency boundary.

## Prepare the change

1. Branch from the current `origin/main`.
2. Read [`AGENTS.md`](AGENTS.md), [`REPOSITORY_RULES.md`](REPOSITORY_RULES.md), and the relevant [architecture guide](docs/book/src/architecture.md).
3. Keep source code and documentation in English.
4. Add the smallest test that would fail without the change. Numerical tests must check a meaningful value, identity, residual, or reconstruction error rather than only shapes or finiteness.
5. Update rustdoc, guides, examples, and generated/public API claims affected by the change.
6. Record nontrivial work in `docs/worklogs/` with the contract read, decisions, and verification evidence.

### API naming

Follow the repository vocabulary documented in the [architecture guide](docs/book/src/architecture.md#vocabulary-conventions):

- unsuffixed method for the ordinary operation;
- `_mut` for in-place mutation;
- `_into` when consuming `self`;
- `_batched` for batched input;
- `_owned` only when input ownership enables a distinct optimization.

Use `max_bond_dim: Option<usize>` for bond caps and `SvdTruncationPolicy` for SVD truncation. Dense flat buffers are column-major. Do not add scalar-specific Rust entry points when a generic API works; scalar-specific names belong at the C FFI boundary.

## Validate locally

Run focused changed-crate tests first, then the CI-equivalent repository gates. These require `cargo-nextest`, mdBook 0.5.2, the HDF5 development library, and (for C-header checks) cbindgen 0.29.2; see `.github/workflows/CI_rs.yml` for platform setup.

```bash
cargo fmt --all -- --check
cargo check -p tensor4all-tensorbackend --no-default-features --features explicit-context,tenferro-cpu-faer
cargo doc -p tensor4all-tensorbackend --no-deps --no-default-features --features explicit-context,tenferro-cpu-faer
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5
cargo test --profile ci -p tensor4all-hdf5
cargo test --doc --profile ci --workspace -j 8
TENSOR4ALL_CARGO_PROFILE=ci ./scripts/test-mdbook.sh
cargo doc --workspace --no-deps
./scripts/check-capi-header.sh
cargo run -p xtask --release -- api-dump
```

Also run maintenance checks relevant to changed public APIs or crate boundaries:

```bash
python3 scripts/audit-library-panics.py
python3 scripts/check-public-error-docs.py
python3 scripts/check-crate-boundaries.py
```

Do not commit `target/`, coverage output, generated API dumps, or other build artifacts.

## Open the pull request

- Link the issue with `Closes #…` when the PR fully resolves it.
- Explain the behavior change and why it belongs in the selected crate/layer.
- List the exact validation commands and results.
- Call out public API breaks, numerical tolerance changes, dense materialization, new dependencies, or cross-repository sequencing explicitly.
- Keep unrelated changes in separate PRs.

Required CI and the repository-rules review must pass. Maintainers normally squash-merge accepted PRs and delete the completed branch.
