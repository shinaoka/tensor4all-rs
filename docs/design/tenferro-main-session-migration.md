# Tenferro Main Session-API Migration

## Status and scope

This document defines implementation-order item 2 of
[tensor4all-rs issue #623](https://github.com/tensor4all/tensor4all-rs/issues/623):
update the pinned tenferro revision to the completed canonical session API and
migrate tensor4all's existing CPU execution calls without changing tensor4all's
public eager execution model.

The final upstream revision is tenferro-rs
`b5a106be3133979d78832a0ca3f4d6b57613b3d7`. It includes the canonical
session API from [tenferro-rs #1680](https://github.com/tensor4all/tenferro-rs/issues/1680)
and the eager-AD fixes required by tensor4all validation:
[#1692](https://github.com/tensor4all/tenferro-rs/issues/1692),
[#1698](https://github.com/tensor4all/tenferro-rs/issues/1698), and
[#1700](https://github.com/tensor4all/tenferro-rs/issues/1700).

This is a dependency and call-site migration, not the complete Stage 1 design
from issue #623.

## Goals

1. Pin every tenferro workspace dependency to the same target revision.
2. Replace removed backend-taking concrete operation calls with tenferro's
   final receiver-first, session-explicit APIs.
3. Preserve tensor4all's current public backend-implicit behavior: existing
   tensor/network method signatures and the process-global CPU execution
   helpers remain unchanged.
4. Preserve eager AD, graph execution, dtype promotion, column-major layout,
   structured storage, and error propagation.
5. Leave the repository compiling and tested against the exact upstream
   revision that future issue #623 phases will build on.

## Non-goals

This migration does not:

- centralize all backend/session entry into the future placement-aware seam;
- add CUDA dependencies, features, upload/download, or placement dispatch;
- expose a session or backend in tensor4all public tensor/network APIs;
- remove `with_default_backend` or the current process-global CPU objects;
- introduce the Stage 2 breaking explicit-session API;
- add trace-region APIs, StableHLO, PJRT, or JAX integration;
- redesign host-backed `Storage` or `Matrix`;
- retain compatibility shims for removed tenferro APIs.

These exclusions keep the first implementation slice mechanical and make any
semantic regression attributable to the upstream API migration rather than to
new execution policy.

## Dependency update

The six workspace dependencies in the root `Cargo.toml` must move together:

- `tenferro-runtime` (workspace alias `tenferro`)
- `tenferro-ad`
- `tenferro-cpu`
- `tenferro-einsum`
- `tenferro-linalg`
- `tenferro-tensor`

Mixing revisions is prohibited because the session traits, extension surfaces,
runtime graph types, and tensor types form one release boundary.

No new dependency is required for this phase.

## Canonical call patterns

### Core concrete operations

Tenferro's final concrete surface is receiver-first through
`TensorSessionOpsExt` and accepts `&mut dyn BackendSession`.

Before the migration, tensor4all may call a method on the session/backend:

```rust
backend.with_backend_session(|session| session.convert(&tensor, dtype))
```

After the migration, the tensor is the receiver:

```rust
backend.with_backend_session(|session| tensor.convert(dtype, session))
```

The same rule applies to concrete operations. Session entry remains at the
existing tensor4all boundary; helpers inside that closure must not open another
session. Rank-2 matrix multiplication uses the receiver-first `matmul` method.
The session extension surface has no arbitrary batched-dot method, so
`batched_mat_mul_same_shape_owned` expresses `[m,k,b] × [k,n,b]` through the
existing compiled einsum runtime rather than reaching through to the low-level
`TensorDot` SPI or opening one session per batch.

### Linear algebra

The removed backend-taking one-shot linalg surface must be replaced with
`TensorLinalgExt` on the input tensor:

```rust
backend.with_backend_session(|session| tensor.svd(session))
backend.with_backend_session(|session| tensor.qr(session))
backend.with_backend_session(|session| tensor.full_piv_lu(session))
```

This applies to production wrappers and benchmark call sites. The low-level
`LinalgBackend` SPI is not the tensor4all concrete user surface and must not be
used as a compatibility substitute.

### Eager AD and graph execution

The existing `EagerRuntime` constructors, eager tensor extension methods,
`GraphCompiler`, and graph runtime calls should remain unchanged when they
compile against the new revision. If upstream signature drift requires an
adaptation, the change must preserve current ownership and execution semantics;
it must not introduce new global objects, cache owners, or fallback paths.
Tracked mixed-dtype tensor/scalar helpers must compose tenferro eager casts and
operations; materializing primals for native arithmetic would detach the AD
graph and is prohibited.

### Einsum extensions

Existing eager and traced einsum calls should remain on their current semantic
paths. Any compile-required rename or registration change must be the minimum
adaptation to the new upstream revision. This phase must not replace graph
execution with dense materialization or a detached local implementation.

## Expected ownership and behavior

- `tensor4all-tensorbackend` remains the owner of process-global CPU execution.
- Higher-level crates continue using tensorbackend wrappers rather than
  constructing `CpuBackend` directly.
- One tensor4all high-level operation may open one tenferro session, matching
  current behavior. Reusing one session across several public tensor4all calls
  belongs to issue #623 Stage 2.
- Backend errors retain their diagnostic source through existing tensor4all
  error conversion.
- No operation silently changes dtype, layout, placement, differentiability, or
  storage representation because of the migration.

## Files expected to change

The final compile diagnostics are authoritative, but the expected surface is:

- root `Cargo.toml` for the synchronized pins;
- `tensor4all-tensorbackend` concrete conversion, dot, SVD, QR, solve, and
  native bridge wrappers;
- direct benchmark-only tenferro linalg calls;
- imports of removed/renamed extension traits;
- tests or current user-facing documentation only where the pinned API makes
  an existing example stale.

Changes outside this surface require an explicit explanation in the work log.
Historical plans do not need rewriting unless they are presented as current
user guidance.

## Verification

Run release-mode checks where tests are executed:

1. `cargo fmt --all`
2. `cargo fmt --all -- --check`
3. `cargo check --workspace --all-targets`
4. `cargo clippy --workspace --all-targets -- -D warnings`
5. `cargo nextest run --release -p tensor4all-tensorbackend`
6. `cargo nextest run --release -p tensor4all-core`
7. `cargo check -p tensor4all-tcicore --benches`
8. `cargo test --doc --release --workspace`
9. `cargo nextest run --release --workspace`
10. `cargo doc --workspace --no-deps`

If a full-workspace command is blocked by an external system dependency, record
the exact command and error, but complete every unaffected targeted check.
Tests and numerical tolerances must not be weakened to make the migration pass.

## Acceptance criteria

- All six tenferro pins equal the target revision.
- No tensor4all call site relies on the removed one-shot backend-taking concrete
  API.
- Existing tensor4all public eager operation signatures are unchanged.
- Production concrete operations use receiver-first session APIs inside the
  existing CPU session boundary.
- No nested session entry, dense fallback, host round trip, or AD detachment is
  introduced.
- Tensorbackend, core, benchmark, doctest, clippy, and full-workspace gates pass
  or have a clearly identified external blocker.
- A curated work log records the design and diff review gates, files changed,
  verification evidence, and remaining issue #623 work.

## Follow-up boundary

After this migration lands, the next issue #623 item is to centralize session
entry behind a placement-aware tensorbackend seam. CUDA support then builds on
that seam as a separate reviewed phase. This migration intentionally does not
pre-build those abstractions.
