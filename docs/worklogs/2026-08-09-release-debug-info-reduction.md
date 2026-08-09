# Ordinary Release Debug-Info Reduction

## Session summary

PR #581 replaced tensor4all-rs's full release debuginfo with line tables and added a full-debug `release-debug` profile. A controlled follow-up measurement found that line tables still occupied most of a full workspace release-test target: setting ordinary release to `debug = 0` reduced allocated output from 13.74 GiB to 3.12 GiB (77.27%). Ordinary dev, test, and release builds now omit debug information; source-level Tensor4all.jl/C API diagnostics remain available through `--profile release-debug`.

## Context reviewed

- `Cargo.toml` workspace profiles and history from PR #248 and PR #581.
- `AGENTS.md`, `README.md`, `REPOSITORY_RULES.md`, and shared tensor4all rules.
- C API error/backtrace implementation in `crates/tensor4all-capi/src/lib.rs`.
- Release-mode test, documentation, tutorial, benchmark, and CI commands.
- `.github/workflows/CI_rs.yml` and `CI_rs_selfhost.yml` profile overrides.
- tensor4all-rs issue #566 and tensor4all-agent-rules issue #9.

## Measurement contract

- Source basis: `origin/main` `bf6ee400e5903034f4b9f0b0da854f53b67e3cb8` (merged PR #581); measurement HEAD differed only by planning documents.
- Host: Linux 6.8, `x86_64-unknown-linux-gnu`.
- Rust: `rustc 1.97.1 (8bab26f4f 2026-07-14)`; Cargo 1.97.1.
- Four Cargo jobs, incremental disabled, empty `RUSTC_WRAPPER`, fresh target directories.
- One generated ignored `Cargo.lock` with SHA-256 `235bee9fc15386dfe12e1adbe2ca7e911547a657af0ecfb804d48ab2e15c51e3` was used unchanged for both builds, then removed. The repository does not track a lockfile, so the initially specified clean `--locked` command could not run until this controlled lock was generated.
- Sizes are allocated bytes from `du -s --block-size=1`.
- Times are one cold sample and contextual; allocated bytes and ELF sections are the primary evidence.

Commands differed only by target path and the candidate debug override:

```bash
env CARGO_TARGET_DIR=/tmp/tensor4all-target-<stage> \
  CARGO_BUILD_JOBS=4 CARGO_INCREMENTAL=0 RUSTC_WRAPPER= \
  [CARGO_PROFILE_RELEASE_DEBUG=0] \
  cargo test --locked --workspace --release --no-run
```

Normalized Cargo fingerprints matched across all 486 build units, including targets, features, dependencies, rustflags, and compile kind.

## Results

| Metric | Line tables (PR #581) | Debug zero | Reduction |
|---|---:|---:|---:|
| Allocated target bytes | 14,759,100,416 B (13.74 GiB) | 3,354,374,144 B (3.12 GiB) | 11,404,726,272 B (77.27%) |
| Wall time | 478.01 s | 456.98 s | 21.03 s (4.40%) |
| Maximum RSS | 1,720,984 KiB | 1,240,392 KiB | 480,592 KiB (27.93%) |
| Unique direct ELF files with `.debug_*` | 206 / 206 | 0 / 206 | 100% removed |

The C API shared library decreased from 118,446,592 to 23,072,104 logical bytes (80.52%). Its baseline contained 95,373,979 bytes of direct `.debug_*` sections, including 14,519,422 bytes of `.debug_line`; the debug-zero library contained none. Representative C API, TreeTN, and tutorial ELFs showed the same pattern.

Raw local evidence is under `/tmp/tensor4all-build-artifact-measurements/`; it is not committed.

## Design decision

Ordinary development does not need Rust source-level debug metadata when Tensor4all.jl and C API internals are not being debugged. The default profiles therefore use no debug information:

```toml
[profile.dev]
debug = 0

[profile.test]
debug = 0

[profile.release]
debug = 0
```

The existing diagnostic profile remains isolated so switching modes does not churn ordinary release artifacts:

```toml
[profile.release-debug]
inherits = "release"
debug = true
```

Users needing source-level Rust backtraces or a debugger through Tensor4all.jl/C API build `tensor4all-capi` with `--profile release-debug` and run the caller with `RUST_BACKTRACE=1`.

## Alternatives rejected

- Keep line tables by default: preserves file/line backtraces but retains 77% of avoidable target allocation in the measured workspace workload.
- Toggle `CARGO_PROFILE_RELEASE_DEBUG=2` in the ordinary release target: valid for one-off use but causes broad profile rebuild/cache churn; the named profile has explicit ownership.
- Add more CI-only environment overrides: does not reduce ordinary local artifacts and duplicates policy across commands.
- Add shared worktree targets, automatic artifact GC, sccache, or dependencies: unrelated to the profile root cause.

## Compatibility

Optimization, numerical behavior, public Rust APIs, C ABI, feature selection, assertions, and overflow behavior are unchanged. Ordinary backtraces may lack Rust source file/line detail. Full prior diagnostic behavior remains available through `release-debug`.

## Verification and remaining work

The paired full workspace compiles passed. The delivery phase verifies Cargo metadata/tree parsing, ordinary C API absence of `.debug_*`, diagnostic-profile presence of `.debug_*`, formatting, clippy, full release tests, doctests, rustdoc, mdBook snippets, independent review, and hosted CI.

Timing results are single samples. Existing `target/` directories retain historical variants until explicitly cleaned; the profile change reduces new artifacts but does not garbage-collect old ones.
