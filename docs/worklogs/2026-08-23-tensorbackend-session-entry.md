# Tensorbackend Session Entry Worklog

## Summary

Implemented the next issue #623 slice on top of the explicit CPU execution
context from #663. Plain concrete CPU operations now enter tenferro through one
crate-private `CpuExecutionContext` session method. CUDA remains for PR 2.

## Material reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, shared Rust/backend rules
- issue #623, PR #631, and issue #663 / PR #669
- `docs/design/tenferro-main-session-migration.md`
- `docs/design/explicit-cpu-execution-context.md`
- all tensorbackend `with_backend`, `with_default_backend`, and
  `with_backend_session` call sites
- pinned tenferro CPU session and execution-owner implementation

## Review gates

- Design: `docs/design/tensorbackend-session-entry.md`
- Pre-implementation reviewer: `reviewer-flash-opencode-go`
- Latest-main verdict: **Correct-to-merge**
- Implementation: `luna-implementer`, max thinking, with parent integration
- Post-implementation reviewer: `reviewer-flash-opencode-go`
- Final verdict: **Correct-to-merge** (no blocking findings)
- Final panic-baseline delta re-review: **Correct-to-merge**

The earlier process-global implementation was reviewed but invalidated before
PR creation when #663 merged. It is retained only on local backup branch
`backup/session-seam-pre-663`; no part of that obsolete ownership model is in
this diff.

## Decisions

- Added one crate-private `CpuExecutionContext::with_session` and one thin
  crate-private default-context adapter.
- Preserved public `with_backend`, `with_default_backend`, and all re-exports.
- Migrated concrete operations in `backend.rs`, `matrix.rs`,
  `tenferro_bridge.rs`, and explicit `LogicalTensor` reconstruction.
- Preserved per-context backend mutex/cache ownership, graph runtime ownership,
  eager AD ownership, and explicit-only feature isolation.
- Left two core benchmark call sites on raw backend/session access because they
  intentionally measure tenferro rather than implement production policy.
- Added no speculative placement enum, factory, or CUDA parameter.

## Nested-entry implementation

Tenferro may execute a session closure on a CPU worker thread. The helper checks
the execution-thread TLS before taking the context mutex and installs an RAII
guard inside the actual session closure. Recursive canonical entry therefore
fails before re-locking in debug and release builds; unwind restores the guard.
Raw backend re-entry and spawned-thread synchronous re-entry remain forbidden
by contract and are not claimed as detected.

## Changed files

- `crates/tensor4all-tensorbackend/src/context.rs`
- `crates/tensor4all-tensorbackend/src/backend.rs`
- `crates/tensor4all-tensorbackend/src/logical_tensor.rs`
- `crates/tensor4all-tensorbackend/src/matrix.rs`
- `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- `docs/design/index.md`
- `docs/design/tensorbackend-session-entry.md`
- `scripts/library-panics-baseline.json` (line-only shifts from `matrix.rs`)
- this worklog

## Verification

Completed during implementation/integration:

- release test binaries compiled with a 600-second build timeout
- release nested-entry test passed under a 30-second execution timeout
- release exact-value concrete-session test passed under a 30-second timeout
- `cargo fmt --all -- --check`: pass
- `cargo check -p tensor4all-tensorbackend --all-targets`: pass
- `cargo clippy -p tensor4all-tensorbackend --all-targets -- -D warnings`: pass
- explicit-only feature check: pass
- `cargo nextest run --release -p tensor4all-tensorbackend`: 194 passed, 2 skipped

- `cargo check --workspace --all-targets`: pass
- `cargo clippy --workspace --all-targets -- -D warnings`: pass
- `cargo nextest run --release --workspace --no-fail-fast`: 3161 passed, 16 skipped
- `cargo test --doc --release --workspace`: pass
- `cargo doc --workspace --no-deps`: pass (pre-existing warnings only)
- `./scripts/test-mdbook.sh`: pass
- repository-rules tests: 90 passed
- repository-rules dry run: pass
- library panic audit: 0 unbaselined, 0 stale
- changed public-error-doc audit: pass
- `git diff --check`: pass

## Coverage impact

No test or tolerance was removed or weakened. New tests cover explicit-context
canonical entry, exact matmul values, recursive entry before mutex re-lock, and
guard restoration after unwind. Existing independent-context concurrency and
explicit-only tests remain.

## Remaining issue #623 work

PR 2 adds optional CUDA dependencies/features, explicit upload/download, typed
mixed-placement rejection, one CUDA-resident TreeTN contraction, and separate
setup/transfer/steady-state benchmark reporting.
