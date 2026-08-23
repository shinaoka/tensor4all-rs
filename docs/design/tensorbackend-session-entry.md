# Tensorbackend Session Entry

## Status and scope

This document defines the next focused slice of
[tensor4all-rs issue #623](https://github.com/tensor4all/tensor4all-rs/issues/623)
after PR #631 and the explicit CPU context from issue #663. It centralizes
plain concrete CPU operations behind one `CpuExecutionContext`-owned session
entry and prepares that owner for later placement dispatch. It adds no CUDA
support.

## Goals

1. Add one crate-private concrete-session method on `CpuExecutionContext`.
2. Route every production direct concrete CPU operation in
   `tensor4all-tensorbackend` through that method, including logical
   reconstruction on an explicit context and global convenience operations
   through a thin default-context adapter.
3. Reject nested canonical session entry before attempting to re-lock a
   context backend, in debug and release builds, and restore the guard during
   unwind.
4. Preserve all public APIs, numerical behavior, explicit-context isolation,
   global-default feature gating, CPU-only builds, graph caches, and eager AD.
5. Leave CUDA dependencies, placement dispatch, transfer APIs, and GPU kernels
   to the next separately reviewed PR.

## Non-goals

This slice does not:

- add `tenferro-gpu`, CUDA features, upload/download, device values, or GPU CI;
- add a public session API or change `CpuExecutionContext::with_backend`;
- add a backend enum, factory, context registry, TLS context override, or
  one-variant placement type;
- wrap `Runtime` or `EagerRuntime` execution in another session; those owners
  enter their own sessions;
- remove the opt-in `global-defaults` compatibility surface;
- change C API or higher-level tensor/network method signatures.

## Canonical context-owned entry

`CpuExecutionContext` gains a crate-private method with this semantic shape:

```text
pub(crate) fn with_session<R: Send>(
    &self,
    f: impl FnOnce(&mut dyn BackendSession) -> R + Send,
) -> R
```

It is the only tensorbackend-owned entry for plain concrete operations. The
method:

1. checks the current execution thread for an active canonical session before
   locking `self.backend`;
2. locks the context-local backend with the same poisoned-lock recovery as
   `with_backend`;
3. calls `CpuBackend::with_backend_session` exactly once;
4. installs an RAII guard inside the actual session closure, which tenferro may
   execute on a worker thread;
5. runs the complete operation and restores the guard on return or unwind.

The guard is module-private and active in all build modes. It forbids nested
canonical sessions on one execution thread. It does not intercept raw
`with_backend` / `with_default_backend` re-entry from a session closure, or a
session closure that spawns another thread and synchronously re-enters; both
patterns remain forbidden because they can wait on a context mutex. The
regression test assumes the workspace's unwind-on-panic profile; a
`panic=abort` build aborts on this internal programming-contract violation.

The existing public `CpuExecutionContext::with_backend` remains unchanged for
caller-managed low-level integration. Production tensorbackend concrete
wrappers use `with_session`; graph/cache management may continue using raw
backend access because it owns runtime setup rather than a concrete operation.

## Global-default adapter

Under `global-defaults`, add one crate-private `with_default_session` that does
only:

```text
default_context().with_session(f)
```

It adds no second implementation and is not re-exported publicly. The existing
public `with_default_backend` remains unchanged for compatibility.

## Call-site migration

Migrate direct production concrete operations in:

- `crates/tensor4all-tensorbackend/src/backend.rs`;
- `crates/tensor4all-tensorbackend/src/matrix.rs`;
- `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`;
- `crates/tensor4all-tensorbackend/src/logical_tensor.rs`.

Global-default modules call `with_default_session`; `LogicalTensor`
reconstruction calls `self.with_session` so explicit-only builds remain free of
global references. No canonical session closure may call `with_backend`,
`with_default_backend`, `with_session`, `with_default_session`, or
`with_backend_session`.

The two `tensor4all-core` benchmark files intentionally use the preserved raw
`with_default_backend` plus tenferro session entry to measure tenferro and do
not implement production tensorbackend policy; they remain unchanged.

## Runtime-owner boundaries

- Plain concrete operations use `CpuExecutionContext::with_session`.
- Graph execution remains owned by each context's `GraphState` and `Runtime`.
- Eager AD remains owned by each context's `EagerRuntime`.

Separate `CpuExecutionContext` values keep separate backend mutexes and caches,
as established by #663. This PR must not reintroduce a process-global backend
outside the optional default context.

## Tests

Add focused tests proving:

1. an explicit context executes a representative concrete operation through
   the canonical session and returns exact values;
2. nested canonical entry fails with a stable programming-error message before
   mutex re-lock and does not hang;
3. after `catch_unwind`, the same context accepts a new non-nested session;
4. existing independent-context concurrency tests still pass;
5. explicit-only feature checks compile without `global-defaults`.

Local execution of the nested test uses an external command timeout as a final
fail-safe against regressions that deadlock before the assertion.

## CUDA follow-up boundary

The next PR extends this context-owned seam with optional CUDA dependencies,
placement inspection, a CUDA ordinal-0 runtime owner, typed mixed-placement
errors, explicit upload/download, one device-resident TreeTN contraction, and
separate setup/transfer/steady-state benchmark reporting. This PR does not
pre-build parameters that have no current caller.

## Review gate

- Reviewer: `reviewer-flash-opencode-go`
- Updated latest-main pre-implementation verdict: **Correct-to-merge**

The reviewer confirmed the explicit-context ownership, feature isolation,
complete production migration scope, thread-global nested-session guard,
runtime-owner split, public API preservation, and CUDA deferral.

The earlier process-global design review and implementation were invalidated
before PR creation when issue #663 merged. The old implementation is retained
only on backup branch `backup/session-seam-pre-663` and is not part of this PR.

## Verification

Run:

1. `cargo fmt --all -- --check`;
2. `cargo check --workspace --all-targets`;
3. `cargo clippy --workspace --all-targets -- -D warnings`;
4. `cargo check -p tensor4all-tensorbackend --no-default-features --features explicit-context,tenferro-cpu-faer`;
5. `cargo doc -p tensor4all-tensorbackend --no-deps --no-default-features --features explicit-context,tenferro-cpu-faer`;
6. `cargo nextest run --release -p tensor4all-tensorbackend`;
7. `cargo nextest run --release --workspace`;
8. `cargo test --doc --release --workspace`;
9. `cargo doc --workspace --no-deps`;
10. `python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run`;
11. `git diff --check`.

## Acceptance criteria

- One context-owned session method governs all production plain concrete CPU
  operations.
- Global convenience operations are thin adapters over the same method.
- Explicit reconstruction uses the supplied context and never a default.
- Nested canonical entry fails before context mutex re-lock in debug and
  release, and the guard restores on unwind.
- Public APIs, explicit-context isolation, graph/eager ownership, numerical
  behavior, and CPU-only builds are unchanged.
- No CUDA dependency, feature, transfer API, GPU implementation, compatibility
  shim, duplicated path, TODO, tolerance change, or hidden fallback is added.
