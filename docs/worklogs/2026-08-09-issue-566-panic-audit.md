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
errors. Each Clippy pass has a ten-minute timeout; the Rust tool allows 25
minutes for both feature passes, the Python wrapper allows 10 minutes to build
plus 25 minutes to audit, and the CI step allows 40 minutes. Local paths are
canonicalized, normalized, deduplicated, and sorted. Macro expansion
definitions and external dependency spans are not findings; the local
expansion call site is used when present.

`syn` remains only for the reviewed public `assert!`/`debug_assert!` baseline;
it does not resolve names, types, imports, or macros. Assertion traversal follows
target/root logical module context in its visited identity, evaluates default
and all-feature `cfg`/`cfg_attr` predicates structurally (`all`, `any`, and
`not`, including `feature` and `test`), and fails closed when an active path
attribute cannot be evaluated. `#[cfg(test)]` statements, nested items, and
expressions inside public bodies are skipped while production `cfg(not(test))`
code is retained. The reviewed baseline remains exactly the 14 existing matrix
assertions.

## Final dep-info corrections

Each Cargo compiler-artifact JSON record is retained independently, including
its full filenames, target object, and profile object. Every candidate dep-info
file is parsed and accepted only when a Make-rule output matches a normalized
artifact filename; unrelated or stale heuristic `.d` files are ignored, and
every expected production target must have an accounted artifact. The parser
implements rustc's Make encoding (`$$`, escaped spaces/colons/backslashes,
continuations, and unescaped `#` path characters).

A dep-info `.rs` dependency is not necessarily a Rust module: it may be an
`include!` fragment or `include_str!`/`include_bytes!` data. Parseable files use
the public assertion visitor. Parse failures do not invalidate a successful
compiler build; they receive a conservative literal assertion token scan.
Public macro invocation arguments and `macro_rules!` transcribers are scanned
for literal `assert!`/`debug_assert!` calls without attempting name or hygiene
resolution.

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
