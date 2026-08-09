# Issue #566 PR 1: compiler-selected panic audit

## Scope

Task 8 replaces the assertion scanner's hand-written module/path/configuration
resolver. Raw panic-style findings remain compiler diagnostics. Assertion
findings now come from the exact workspace-local Rust files listed by rustc
Make dep-info for each production `cargo clippy --lib --bins` run (default
features and, when workspace features exist, `--all-features`).

## Decision

- Cargo metadata selects only production targets under `crates/`; compiler
  artifact `filenames` identify the matching dep-info beside each rlib,
  cdylib, binary, or proc-macro output. Missing artifact filenames, dep-info,
  malformed JSON, missing build-finished records, malformed Make rules, or
  missing local source files fail closed.
- Dep-info paths are Make-unescaped, canonicalized, deduplicated, and filtered
  to workspace-local `.rs` files. No module root, `#[path]`, feature, target,
  or host-cfg resolution remains in the assertion scanner.
- `PublicAssertionVisitor` is run once per canonical selected source file. It
  scans public free functions, public inherent methods, public trait defaults,
  and all trait-implementation methods for `assert!` and `debug_assert!`.
- Configuration filtering is intentionally three-valued: `test` is false,
  `not(test)` is true, and every other atom is unknown. Only definitely false
  item/file/statement/expression/match-arm/cfg_attr content is skipped; target
  and feature assertions remain conservatively included.
- The reviewed assertion baseline remains exactly 14 entries. Raw panic-style
  findings are never accepted in the baseline.

The rejected alternative was retaining source-module traversal and evaluating
Cargo feature/target/path attributes. rustc already records the selected file
set, so duplicating that resolver was both larger and less sound.

## Verification

- `cargo test --release -p library-panic-audit` — passed (10 tests).
- `cargo clippy --release -p library-panic-audit --all-targets -- -D warnings` — passed.
- `python3 scripts/test-audit-library-panics.py` — passed (6 fixture tests).
- `cargo clippy --workspace --all-targets -- -D warnings` — passed.
- `cargo nextest run --release --workspace --exclude tensor4all-hdf5` — passed (2641 tests, 10 skipped).
- `cargo test --release -p tensor4all-hdf5` — passed (49 tests, 4 ignored).
- `cargo test --doc --release --workspace` — passed (837 tests).
- `python3 scripts/audit-library-panics.py` — passed: 14 baseline matches,
  0 unbaselined findings, 0 stale entries.
- `cargo run -p api-dump --release -- . -o docs/api` — passed; regenerated
  `docs/api/library_panic_audit.md`.
- `cargo fmt --all -- --check`, Python bytecode compilation, and `git diff --check`
  — passed. `actionlint` was not installed in the environment, so workflow
  linting remains a local/CI follow-up.

The fixture covers a cdylib+rlib library, binary sharing a root/path module,
feature-selected `cfg_attr` modules, a target-architecture assertion, a
file-level `cfg(test)` source, statement/expression/match-arm `cfg(test)`,
nested `cfg_attr`, and deduplicated shared files.

## Residual risks

The scanner intentionally treats unknown configuration as production so it can
report false positives rather than miss target-specific assertions. It relies
on Cargo/rustc's documented Make dep-info format; malformed or unsupported
syntax fails closed instead of guessing.
