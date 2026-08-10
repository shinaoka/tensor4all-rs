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
its full filenames, target object, and profile object. Sources are associated
with the exact matching artifact output: only the Make rules whose outputs
include the artifact's filename contribute their `.rs` dependencies, so an
unrelated rule cannot satisfy an artifact's source validation (a stale file
that lists an artifact output with no sources of its own now fails closed).
Every candidate dep-info file is parsed and accepted only when a Make-rule
output matches a normalized artifact filename; unrelated or stale heuristic
`.d` files are ignored, and every expected production target must have an
accounted artifact. The parser implements rustc's Make encoding (`$$`, escaped
spaces/colons/backslashes, continuations, and unescaped `#` path characters).

A dep-info `.rs` dependency is not necessarily a Rust module: it may be an
`include!` fragment or `include_str!`/`include_bytes!` data. Parseable files use
the public assertion visitor. Parse failures do not invalidate a successful
compiler build; they receive a conservative literal assertion token scan.
Public macro invocation arguments and `macro_rules!` transcribers are scanned
for literal `assert!`/`debug_assert!` calls without attempting name or hygiene
resolution. Nested `macro_rules!` definitions inside a transcriber are
recognized: a nested name in matcher position is a pattern, not a call, so only
the nested rules' transcribers contribute findings.

## Verification

- Rust unit tests (17) cover exact-code filtering, synthetic expansion spans,
  outside-root/missing-span failures, baseline normalization, strict Cargo JSON,
  invalid UTF-8, build completion, `cdylib`/`rlib` target coverage, and the two
  scanner regressions added for the review findings: an artifact output whose
  own Make rule has no sources (sources only on an unrelated rule) fails
  closed, and nested `macro_rules!` matcher patterns are not reported.
- The Python self-test (8 tests) builds the audit binary once and runs compiled
  fixtures covering aliases, local macro arguments, custom methods, safe and
  dormant macros, all four raw diagnostics (panic/unreachable/unwrap/expect),
  `#[cfg(test)]`, public/private assertions, feature-selected `cfg_attr` module
  paths, lib-module versus bin-root traversal of one canonical file, a
  production `cdylib`/`rlib` plus binary target, and a nested-`macro_rules!`
  fixture whose matcher is excluded while its real transcriber is reported.
- The real audit reports 14 matched baseline entries, zero unbaselined findings,
  and zero stale entries for both default and all-feature Clippy runs.
- `docs/api/library_panic_audit.md` was regenerated from the tool source.

The panic-audit tool itself adds no Rust library or C API surface. Task 8 also
carries an API checkpoint (committed as `6add467` and its successors) that
makes `TensorDynLen` compact-support metrics, scaling, and comparison
allocation-safe (`tensordynlen.rs`, tensorbackend `storage.rs`); it was
validated by workspace nextest, doctests, mdBook, API dump, workspace clippy,
and the scanner self-test/audit.

## Task 8 API checkpoint reviews (2026-08-10)

Two independent reviews were run on the checkpoint. The first (session-default
DeepSeek V4 Flash) returned PASS with minor notes. A second review on a
different model family (GPT-5.6 Sol, `reviewer-gpt`) found three Important
findings, all fixed and re-confirmed:

1. Untracked `Materialized` scaling traversed unreferenced strided-gap backing
   entries via the whole-backing `Storage::scale` fast path. Fixed by routing
   all `Eager`/`Compact`/`Materialized` scaling through `scale_eager_payload`,
   which converts only the compact payload and returns compact storage
   (commit `53897d3`); the now-unused `TensorDynLenStorage::scale` method and
   `BackendScalar` import were removed.
2. Dense logical materialization silently returned an empty buffer when the
   logical-dim product overflowed. `StructuredStorage::logical_dense_col_major_vec`
   is now fallible and fails closed via `checked_logical_len`; the public
   `Storage::to_dense_*_col_major_vec` wrappers propagate the error
   (commits `53897d3`, `f5a72f3`).
3. New control-flow branches lacked regression coverage. Added: untracked
   scale of genuinely gapped storage collapsing to dense compact storage,
   dense logical-product overflow failing closed, unmatched support compared
   in both operand orders, exact-mode and structured-payload NaN rejection,
   and zero-vs-nonzero tolerant comparison.

Final validation after the fixes is green: workspace clippy `-D warnings`,
nextest 2722/2722 (+14 skipped), doctests 839, `cargo doc`, mdBook exit 0,
scanner self-test 8/8 and audit 0 unbaselined / 0 stale.

## Remaining boundary

Direct panic tokens dormant inside an uninvoked `macro_rules!` transcriber are
outside the compiler diagnostic boundary. An invoked macro whose production
call site contains the forbidden expression is reported by Clippy; custom
`unwrap`/`expect` methods and safe local macros are not reported.
