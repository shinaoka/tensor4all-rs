# Explicit CPU execution context

**Status:** Implemented for tensor4all-rs#663

## Goal

Make one caller-supplied `tenferro_cpu::CpuBackend` the root of plain tensor,
graph, and eager-AD execution without consulting process-global defaults.
Tensor4all does not define an executor abstraction or depend on a host
scheduler.

## Public boundary

`tensor4all_tensorbackend::CpuExecutionContext::from_backend` consumes the
caller-selected backend. `CpuBackend::clone` supplies the graph and eager
handles; those clones preserve the backend runtime identity and share only the
caller-selected executor/provider resources. Construction never calls
`CpuBackend::new`, `CpuContext::from_env`, or placement resolution. The context
owns:

- a backend handle for plain operations;
- a graph compiler, runtime, prepared-plan cache, and cloned backend handle;
- an eager runtime built from another cloned backend handle.

The context exposes explicit methods for borrowing the plain backend, compiling
and running graph programs, and cloning the eager runtime handle. These methods
are the canonical explicit integration surface. Existing higher-level
`IdxTensor`, matrix, and bridge functions remain opt-in global convenience APIs;
an explicit host adapter must call the context surface (and may wrap values in
its own application type) rather than those convenience functions. Graph IR
and `tenferro::CompiledGraph` remain backend-neutral; backend-prepared
executables, workspaces, and caches stay private to the context-owned runtime.

Separately created contexts never share a tensor4all mutex or cache. Backend
identity checks remain tenferro's responsibility and errors are preserved as
typed sources.

## Global convenience path

The existing process-global helpers remain only behind the opt-in
`global-defaults` Cargo feature. The normal workspace enables that feature for
compatibility. An integration can depend on `tensor4all-tensorbackend` with
`default-features = false` and select a CPU provider feature without compiling
any `DEFAULT_*`, `from_env`, or global backend mutex path. The public context is
re-exported from `tensor4all-tensorbackend`; no C API change is made.

Explicit methods do not call global helpers. The global wrappers are thin
adapters over the same context implementation, avoiding a second operation
implementation.

### Feature-gating inventory

`explicit-context` compiles `context` without its default submodule plus the new
`logical_tensor` module. Its public surface is
`CpuExecutionContext`, its typed construction/execution errors,
`LogicalTensor`, `LogicalTensorData`, and their typed error. This is the small
integration API for plain backend borrowing, graph compile/run/cache, eager
runtime access, and target-context reconstruction.

`global-defaults` additionally compiles the legacy `any_scalar`, `backend`,
`matrix`, `memory`, `storage`, `tenferro_bridge`, and `tensor_element` modules
and their current re-exports. `backend-tenferro` remains a compatibility alias
for `global-defaults`, and the crate's default features keep enabling it. This
coarse module boundary is deliberate: it leaves no legacy operation that can
reference a default in an explicit-only build and avoids scattered per-function
cfg gates.

CI checks both isolated code and docs:

```text
cargo check -p tensor4all-tensorbackend --no-default-features \
  --features explicit-context,tenferro-cpu-faer
cargo doc -p tensor4all-tensorbackend --no-deps --no-default-features \
  --features explicit-context,tenferro-cpu-faer
```

A source-isolated build proves graph and reconstruction code cannot reference
the default module; a runtime hit-counter test separately covers explicit plain
and eager initialization. Other tests exercise graph/cache, reconstruction,
parallel-context, and drop/fresh-cache behavior. The cross-context test creates eager tensors in two
separate `EagerRuntime` values and calls `EagerTensor::add`, which tenferro
rejects specifically with `tenferro_ad::Error::ContextMismatch`; no plain host
tensor is claimed to be context-bound.

## Logical transfer

`LogicalTensor` is a new owned, canonical column-major host snapshot containing
only shape and dtype-preserving scalar data for all tenferro CPU dtypes (`f32`,
`f64`, `i32`, `i64`, `bool`, `Complex32`, and `Complex64`). It is distinct from
`Storage`, whose scalar contract intentionally covers only `f64` and
`Complex64` structured payloads. It contains no backend, executor, runtime,
cache, admission token, pointer, or address identity. Reconstruction is a
method on `CpuExecutionContext`, making the receiving context mandatory; shape,
element-count, and dtype failures are typed bridge errors. The adapter owns
serialization and MPI transport.

## Errors and lifecycle

Construction and graph execution use crate-local typed errors with original
tenferro errors retained as sources. The context is `Send + Sync`. Dropping the
last context owner drops its backend handles, graph runtime/cache, and eager
runtime, but never shuts down a caller-owned executor retained elsewhere. A
fresh context has a fresh runtime/cache and cannot recover stale prepared
entries. Tests cover both properties.

## Non-goals

- Hataori, MPI, or serialization dependencies;
- a generic executor trait, TLS override, or process-global context registry;
- serializing compiled programs or provider workspaces;
- changing tenferro's caller-managed admission contract delivered by
  tenferro-rs#1716 / PR #1717; tensor4all updates its tenferro pin to merge
  commit `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45` and verifies the existing
  backend-session bridges against it.
