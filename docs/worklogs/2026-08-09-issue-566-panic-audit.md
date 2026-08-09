# Issue #566 compiler-backed panic audit

## Decision

The Task 8 raw panic scan delegates production reachability and name/macro
resolution to Cargo and Clippy. The audit selects workspace packages under
`crates/`, compiles library-like Cargo targets (`lib`, `rlib`, `dylib`,
`cdylib`, `staticlib`, and `proc-macro`) plus `bin` targets, and therefore
covers the `tensor4all-capi` `cdylib`/`rlib` target. It runs default features
and `--all-features` when a selected production package defines non-default
features, forcing exactly these diagnostics: `clippy::panic`,
`clippy::unreachable`, `clippy::unwrap_used`, and `clippy::expect_used`.

The Cargo JSON boundary is fail-closed. Invalid UTF-8, malformed/non-object
records, missing or non-string reasons, unknown reasons, unsuccessful or
missing `build-finished` records, unsupported targeted Clippy codes, or
incomplete production-target compiler-artifact coverage are configuration
errors. Clippy runs have a ten-minute timeout; the wrapper allows both feature
runs without the old combined 120-second limit. Local paths are canonicalized,
normalized, deduplicated, and sorted. Macro expansion definitions and external
dependency spans are not findings; the local expansion call site is used when
present.

`syn` remains only for the reviewed public `assert!`/`debug_assert!` baseline;
it does not resolve names, types, imports, or macros. Assertion traversal follows
target/root logical module context in its visited identity, evaluates default
and all-feature `cfg`/`cfg_attr` predicates structurally (`all`, `any`, and
`not`, including `feature` and `test`), and fails closed when an active path
attribute cannot be evaluated. `#[cfg(test)]` statements, nested items, and
expressions inside public bodies are skipped while production `cfg(not(test))`
code is retained. The reviewed baseline remains exactly the 14 existing matrix
assertions.

## Verification

- Rust unit tests cover exact-code filtering, synthetic expansion spans,
  outside-root/missing-span failures, baseline normalization, strict Cargo JSON,
  invalid UTF-8, build completion, and `cdylib`/`rlib` target coverage.
- The Python self-test builds the audit binary once and runs compiled fixtures
  covering aliases, local macro arguments, custom methods, safe and dormant
  macros, all four raw diagnostics (panic/unreachable/unwrap/expect),
  `#[cfg(test)]`, public/private assertions, feature-selected `cfg_attr` module
  paths, lib-module versus bin-root traversal of one canonical file, and a
  production `cdylib`/`rlib` plus binary target.
- The real audit reports 14 matched baseline entries, zero unbaselined findings,
  and zero stale entries for both default and all-feature Clippy runs.
- `docs/api/library_panic_audit.md` was regenerated from the tool source.

No Rust library or C API surface was changed.

## Remaining boundary

Direct panic tokens dormant inside an uninvoked `macro_rules!` transcriber are
outside the compiler diagnostic boundary. An invoked macro whose production
call site contains the forbidden expression is reported by Clippy; custom
`unwrap`/`expect` methods and safe local macros are not reported.
