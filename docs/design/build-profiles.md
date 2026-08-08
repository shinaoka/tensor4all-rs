# Build Profiles

## Goal

Keep ordinary tensor4all-rs development and verification builds small while retaining an explicit full-debug release profile for debugger sessions and detailed Rust diagnostics through Tensor4all.jl or the C API.

## Profile contract

Ordinary profiles generate no Rust debug information:

```toml
[profile.dev]
debug = 0

[profile.test]
debug = 0

[profile.release]
debug = 0
```

Debug assertions and overflow checks remain enabled for dev and test builds. Release optimization and runtime semantics are unchanged.

Full source-level release diagnostics use a separate profile:

```toml
[profile.release-debug]
inherits = "release"
debug = true
```

Build the C API for a Rust debugger or file-and-line-rich backtrace with:

```bash
cargo build --profile release-debug -p tensor4all-capi
```

Set `RUST_BACKTRACE=1` or `full` when running the Tensor4all.jl/C API caller. Keeping this output under `target/release-debug` avoids rebuilding or retaining full debuginfo in the ordinary release profile.

One-command dev/test debugger overrides remain available through `CARGO_PROFILE_DEV_DEBUG=2` and `CARGO_PROFILE_TEST_DEBUG=2`.

## Rationale

The repository normally runs tests, documentation examples, tutorials, and benchmarks in release mode. Full debuginfo or line tables are therefore multiplied across many independently linked test and example executables. A controlled full-workspace measurement found that changing ordinary release from line tables to `debug = 0` reduced allocated target output from 14,759,100,416 to 3,354,374,144 bytes (77.27%); see `docs/worklogs/2026-08-09-release-debug-info-reduction.md`.

Most development does not debug Rust through Tensor4all.jl or the C API and does not benefit from this metadata. The named profile preserves the diagnostic capability without imposing its storage cost on every release build.

## Compatibility and non-goals

Optimization, numerical behavior, public APIs, ABI, feature selection, and release assertions are unchanged. Ordinary release backtraces may lack Rust source file/line detail. This policy does not introduce shared worktree targets, automatic artifact garbage collection, sccache, or dependencies. Existing target directories retain historical variants until explicitly cleaned.
