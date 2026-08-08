# Default Release Debug-Info Reduction Design

## Goal

Make tensor4all-rs ordinary release builds generate no debug information while retaining an explicit full-debug release profile for debugger sessions and detailed C API source-level diagnostics.

## Current behavior

PR #581 already replaced full release debuginfo with `line-tables-only`, set local dev/test profiles to `debug = 0`, and added `[profile.release-debug]` with full debuginfo. This substantially addresses the original build-artifact problem while preserving default file-and-line backtraces. The remaining question is whether line tables still impose enough disk and build cost to justify the maintainer-approved ordinary release setting `debug = 0`.

## Design

Measure `line-tables-only` against `debug = 0` from the same immutable `origin/main` commit and fresh target directories. If the reduction is material, change only the ordinary release profile:

```toml
[profile.release]
debug = 0

[profile.release-debug]
inherits = "release"
debug = true
```

Keep `release-debug` as the explicit diagnostic path. Update the nearby profile comment and C API documentation so users who require file/line-rich Rust backtraces or debugger metadata build with `--profile release-debug`. Keep existing CI environment overrides unless they become incorrect; redundant `debug=0` overrides are harmless and avoid unrelated workflow churn.

## Measurement

Use fresh targets, the same toolchain and commit, disabled incremental compilation, and fixed Cargo jobs. Compare:

1. a cold release workspace test compile (`cargo test --workspace --release --no-run`), using an environment override to produce the `debug=0` candidate before source changes;
2. `tensor4all-capi` release library size and ELF `.debug_*` sections.

Record allocated bytes, wall time, maximum RSS, largest artifacts, and C API library size. Treat one cold timing sample as indicative; disk and ELF-section differences are deterministic acceptance evidence.

## Verification

- Parse the manifest with Cargo metadata/tree.
- Build ordinary release and confirm representative artifacts contain no `.debug_*` sections.
- Build `tensor4all-capi` with `--profile release-debug` and confirm debug sections remain.
- Run formatting, clippy, focused/full required tests, rustdoc, repository-rule review, and hosted PR checks.

## Compatibility and non-goals

Optimization, runtime numerical behavior, public APIs, ABI, feature selection, and release assertions remain unchanged. Ordinary release backtraces may lose source file/line detail; full diagnostic behavior remains available through `release-debug`. This change does not add shared worktree targets, automatic artifact GC, sccache, or dependencies.
