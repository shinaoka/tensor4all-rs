# Issue #566 compiler-backed panic audit

## Decision

The Task 8 raw panic scan now delegates production reachability and name/macro
resolution to Cargo and Clippy. The audit selects workspace packages under
`crates/`, compiles only `--lib --bins`, runs default features and
`--all-features` when a production package defines non-default features, and
forces exactly these diagnostics: `clippy::panic`, `clippy::unreachable`,
`clippy::unwrap_used`, and `clippy::expect_used`.

The JSON boundary is fail-closed. Cargo failure, malformed output, incomplete
target coverage, unsupported targeted Clippy codes, or a targeted diagnostic
without a local primary/expansion call-site span exits with configuration error
2. Local paths are canonicalized, normalized, deduplicated, and sorted. Macro
expansion definitions and external dependency spans are not findings; the
local expansion call site is used when present.

`syn` remains only for the reviewed public `assert!`/`debug_assert!` baseline.
It follows Cargo target roots and a small structural `mod` graph, excludes
`#[cfg(test)]`, and deliberately does not resolve imports, types, or macros.
The compiler boundary therefore does not claim to inspect dormant macro token
text or arbitrary metavariable expansions. The reviewed baseline remains the
14 existing matrix assertions.

## Verification

- Rust unit tests cover exact-code filtering, synthetic expansion spans,
  outside-root/missing-span failures, and baseline normalization.
- The Python self-test builds the audit binary once and runs one fixture audit;
  that fixture covers aliases, local macro arguments, custom methods, safe and
  dormant macros, `#[cfg(test)]`, public/private assertions, a feature-only
  production function, and a production binary.
- The real audit reports 14 matched baseline entries, zero unbaselined findings,
  and zero stale entries for both default and all-feature Clippy runs.

No Rust library or C API surface was changed.

## Remaining boundary

Direct panic tokens dormant inside an uninvoked `macro_rules!` transcriber are
outside the compiler diagnostic boundary. An invoked macro whose production
call site contains the forbidden expression is reported by Clippy; custom
`unwrap`/`expect` methods and safe local macros are not reported.
