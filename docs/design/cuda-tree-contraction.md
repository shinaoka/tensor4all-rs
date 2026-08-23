# CUDA TreeTN Contraction Vertical Slice

## Status and scope

This document defines PR 2 of the two-PR GPU foundation for
[tensor4all-rs #623](https://github.com/tensor4all/tensor4all-rs/issues/623)
and [#553](https://github.com/tensor4all/tensor4all-rs/issues/553). PR #674
already centralized concrete CPU session entry on `CpuExecutionContext`.

This PR adds explicit single-GPU transfer and one device-resident TreeTN
contraction path. The supported algorithm is a CUDA-specific deterministic
edge walk over dense, untracked, same-dtype node tensors uploaded into one CUDA
eager runtime. It deliberately does not call the existing generic
`contract_to_tensor`, whose N-ary route currently enters the process-global CPU
graph runtime. It does not claim general GPU support for zip-up, fitting,
TCI/ACI, truncation, or host-backed matrix algorithms.

## Goals

1. Add optional `tenferro-gpu` plumbing behind `tenferro-cuda` features.
2. Add a caller-owned `CudaExecutionContext` for visible CUDA ordinal 0.
3. Add explicit `IdxTensor::upload_cuda` and `IdxTensor::download` boundaries.
4. Preserve full `Index` metadata and exact logical values across transfer.
5. Reject tracked, structured, foreign-runtime, host/CUDA-mixed, unsupported
   dtype, and unavailable-CUDA cases with typed errors; never fall back to CPU.
6. Add `TreeTN<IdxTensor>` upload/download helpers and a checked
   `contract_to_tensor_cuda` vertical slice whose intermediate pairwise
   contractions remain in one CUDA `EagerRuntime`.
7. Add a reproducible CPU/CUDA benchmark harness that reports context setup,
   upload, steady-state contraction, download, and CPU contraction separately.
8. Keep the default CPU-only build and public CPU behavior unchanged.

## Non-goals

- Multi-GPU, nonzero visible ordinals, sharding, streams exposed to callers, or
  device meshes.
- Automatic transfer, automatic backend selection, or CPU fallback.
- Tracked-value transfer or cross-runtime AD migration.
- Structured/diagonal payload upload; this first slice rejects it instead of
  silently densifying an unbounded tensor.
- CUDA SVD/QR/eigh/truncation, TreeTN×TreeTN zip-up/fitting, TCI, ACI, GSE,
  TDVP, DMRG, C API, or language-binding support.
- Making CUDA a default feature or requiring CUDA in ordinary CI.
- A generic public CPU/GPU context trait or backend enum.

## Feature plumbing

Add the pinned `tenferro-gpu` crate to root workspace dependencies. It is
optional only at the consuming crate boundary.

`tensor4all-tensorbackend` exposes:

```text
tenferro-cuda = [
    "explicit-context",
    "tenferro-cpu-faer",
    "dep:tenferro-gpu",
    "tenferro-gpu/cuda",
    "tenferro-ad/cuda",
    "tenferro-einsum/autodiff",
    "tenferro-einsum/cuda",
    "tenferro-linalg/autodiff",
    "tenferro-linalg/cuda",
]
```

Do not add `tenferro/cuda`: `tenferro-runtime` has no such feature.

`tensor4all-core/tenferro-cuda` includes its existing `backend-tenferro` and
`tenferro-cpu-faer` features (needed for host reconstruction through the
existing CPU default context), propagates to tensorbackend, and enables CUDA on its direct
`tenferro-ad`, `tenferro-einsum`, and `tenferro-linalg` dependencies.
`tensor4all-treetn/tenferro-cuda` propagates to core and tensorbackend. CUDA
remains absent from every default feature set.

## CUDA execution context

Under `tensor4all-tensorbackend::cuda`, add:

- `CudaExecutionContext`;
- `CudaExecutionContextError`;
- visible-ordinal-0 constants or accessors needed by tests.

The context owns:

- one `Mutex<CudaBackend>` created with `CudaDeviceId::from_ordinal(0)`;
- one lazy `Arc<EagerRuntime>` built from a clone of that backend, preserving
  CUDA runtime/allocation identity;
- no CPU backend and no graph runtime in this first slice.

Public operations:

- `CudaExecutionContext::new()` selects visible ordinal 0 and returns a typed
  device/runtime error when unavailable;
- `upload_cuda(&Tensor)` accepts host tensors only;
- `download(&Tensor)` accepts tensors resident on this CUDA context only;
- `eager_runtime()` returns the context-owned CUDA eager runtime;
- `synchronize()` provides an explicit timing boundary;
- placement validation helpers reject non-CUDA0 or incompatible inputs.

Transfers use `TensorDeviceTransfer` through one context-owned CUDA session.
No transfer occurs inside an arithmetic or contraction operation.

`CudaExecutionContextError` preserves tenferro/CUDA sources and has explicit
variants for initialization, transfer, placement mismatch, unsupported input,
and synchronization. Error messages identify the operation and remedy
(upload, download, use one context, or disable tracked values).

## IdxTensor transfer contract

Behind `tensor4all-core/tenferro-cuda`, add:

- `IdxTensorCudaError`;
- `IdxTensor::upload_cuda(&CudaExecutionContext)`;
- `IdxTensor::download(&CudaExecutionContext)`;
- `IdxTensor::validate_cuda_residency(&CudaExecutionContext)`, which checks
  device placement and `EagerTensor::ctx_id()` against the supplied context
  without host access.

`upload_cuda`:

1. validates deferred storage errors;
2. rejects `tracks_grad()` before transfer;
3. requires both logical-dense axis classes and `StorageKind::Dense`, rejecting
   diagonal/structured storage before materialization;
4. obtains the existing dense native value without scalar conversion;
5. uploads explicitly through the supplied CUDA context;
6. wraps the resident value in the context's CUDA `EagerRuntime`;
7. reconstructs `IdxTensor` with the original ordered full indices.

`download` performs the inverse explicit transfer, then reconstructs the host
value through the existing CPU eager context enabled by core's
`backend-tenferro` compatibility feature. It rejects host tensors, foreign CUDA
allocation domains, and tracked values. Both methods preserve f32/f64, c32/c64,
integer, and bool dtypes only where tenferro CUDA transfer supports them;
unsupported dtypes return typed errors.

The first slice does not preserve compact diagonal/structured representation
because it does not upload those inputs. Dense input remains dense.

## TreeTN vertical slice

Behind `tensor4all-treetn/tenferro-cuda`, add an `IdxTensor`-specific module
with:

- `CudaTreeTNError` containing the failing `NodeIndex` and typed source;
- `TreeTN<IdxTensor, V>::upload_cuda(&CudaExecutionContext)`;
- `TreeTN<IdxTensor, V>::download(&CudaExecutionContext)`;
- `TreeTN<IdxTensor, V>::contract_to_tensor_cuda(&CudaExecutionContext)`.

Upload/download clone the network metadata and replace node tensors only after
each replacement validates successfully. Failure returns an error without
mutating the source network.

`contract_to_tensor_cuda` first validates every node tensor:

- CUDA device memory on visible ordinal 0;
- `EagerTensor::ctx_id()` equal to the supplied context's eager-runtime id;
- untracked dense value and `StorageKind::Dense`;
- one common dtype across all nodes.

It must not delegate to generic `contract_to_tensor`: that method uses
`T::contract`, whose N-ary plan enters the default CPU graph runtime. Instead,
the CUDA method repeats only the small topology-level algorithm:

1. validate tree topology and choose the deterministic minimum-name root;
2. obtain the existing post-order edge schedule;
3. remove each child/parent pair from a local tensor map;
4. contract with the pairwise `TensorContractionLike::contract_pair` path;
5. validate each intermediate's CUDA residency/context, dense axis classes,
   and `StorageKind::Dense` before reinsertion;
6. permute the final tensor to canonical site-index order through the existing
   eager transpose path;
7. validate and return the still-resident CUDA result.

For same-dtype dense EagerTensor operands, pairwise contraction reaches
`EagerTensor::dot_general_with_conj` in the operands' own eager runtime. The
mixed-dtype fallback through `contract_native_tensor` is unreachable because
same dtype is validated before the first edge.

A mixed host/CUDA, foreign-context, or mixed-dtype TreeTN fails at the typed
validation boundary before contraction. It is never sent to the CPU graph
runtime and never falls back.

This is an explicitly dense full-network contraction and retains the existing
`contract_to_tensor` output-size cost. It is the first measurable vertical
slice, not the scalable TreeTN×TreeTN production contraction milestone.

## Benchmark harness

Add a feature-gated release example or bench target under
`tensor4all-treetn` that builds a deterministic dense chain with:

- endpoint physical dimension 2;
- configurable chain length and bond dimension;
- dense f64 node tensors with bounded deterministic values;
- small output size so benchmark memory is controlled while bond contractions
  are nontrivial.

The harness reports separate durations for:

1. host TreeTN setup;
2. CUDA context setup;
3. host-to-device upload;
4. CUDA warm-up (reported but excluded from steady state);
5. steady-state CUDA `contract_to_tensor_cuda` plus explicit synchronize;
6. device-to-host download;
7. steady-state CPU `contract_to_tensor`.

It prints configuration, iteration count, GPU name/runtime information when
available, and milliseconds per stage. Correctness is checked outside timed
loops by comparing the downloaded CUDA result against the CPU result using
exact index order and a reported max-absolute residual. The output tensor's
CUDA residency is asserted before download.

Suggested command:

```text
cargo run --release -p tensor4all-treetn \
  --example cuda_tree_contraction --features tenferro-cuda,tenferro-cpu-faer
```

## Tests

### Hardware-independent

- CPU-only default build has no `tenferro-gpu` activation.
- CUDA feature compile-checks for tensorbackend, core, and treetn.
- Public CUDA error/source chains and feature-gated API signatures compile.
- Mock/metadata tests cover host-vs-CUDA placement classification without
  requiring a CUDA device where practical.
- Mixed-placement, foreign-context, and mixed-dtype validation is tested before
  contraction execution.
- Structured and tracked transfer rejection is tested before CUDA access.
- A source-contract regression proves `contract_to_tensor_cuda` uses the
  pairwise edge walk and does not call generic `contract_to_tensor` or
  `T::contract`.

### CUDA hardware

On visible CUDA ordinal 0:

- context creation reports ordinal 0;
- dense f64 upload retains shape/dtype and produces device placement;
- download restores exact values and ordered full indices;
- foreign-context download returns a typed error;
- a two-node, a branched TreeTN, and a scalar-output closed tree upload,
  contract, and download match CPU;
- every checked intermediate and the output are CUDA-resident in the supplied
  eager context before download;
- mixed host/CUDA, foreign-context, and mixed-dtype node tensors return typed
  errors before the first edge contraction;
- no host access succeeds on the resident output; the output-residency check
  plus the source-contract regression proves no default CPU graph route ran;
- benchmark harness completes and reports all timing categories.

## Review gate

- Reviewer: `reviewer-flash-opencode-go`
- Round 1 verdict: **Needs changes** because generic `contract_to_tensor` routes
  through the default CPU graph runtime.
- Fix: define a same-dtype, pairwise eager edge walk; reject structured storage
  by `StorageKind`; propagate `backend-tenferro` for download reconstruction.
- Updated design verdict: **Correct-to-merge**. The reviewer verified that the
  pairwise same-dtype dense branches execute in the operands' CUDA eager
  runtime and that required TreeTN topology/schedule accessors are public.
- Implementation recon added `tenferro-cpu-faer` to tensorbackend/core CUDA
  features because `explicit-context`/`backend-tenferro` compile `CpuBackend`
  and download reconstruction uses the existing CPU default context; the
  no-default CUDA gates otherwise fail closed with tenferro-cpu's provider
  check. The reviewer confirmed this is the minimal provider choice.

## Verification

CPU/default gates:

```text
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release --workspace --no-fail-fast
cargo test --doc --release --workspace
cargo doc --workspace --no-deps
```

Feature and hardware gates:

```text
cargo check -p tensor4all-tensorbackend --no-default-features --features tenferro-cuda
cargo check -p tensor4all-core --no-default-features --features tenferro-cuda
cargo check -p tensor4all-treetn --no-default-features --features tenferro-cuda,tenferro-cpu-faer
cargo nextest run --release -p tensor4all-tensorbackend --features tenferro-cuda
cargo nextest run --release -p tensor4all-core --features tenferro-cuda
cargo nextest run --release -p tensor4all-treetn --features tenferro-cuda
cargo run --release -p tensor4all-treetn --example cuda_tree_contraction \
  --features tenferro-cuda,tenferro-cpu-faer
```

Also run repository-rules review, panic/public-error audits, mdBook tests, and
`git diff --check`.

## Acceptance criteria

- CUDA is optional and absent from CPU defaults.
- Visible ordinal 0 is the only supported CUDA device.
- Explicit upload/download preserve dense values, dtype, shape, and full Index
  metadata without hidden transfer.
- Tracked, structured, foreign, unsupported, and mixed-placement inputs return
  typed errors with no CPU fallback.
- One TreeTN full contraction remains CUDA-resident until explicit download and
  matches CPU numerically.
- Setup, upload, warm-up, steady-state contraction, download, and CPU timings
  are reported separately.
- CPU-only CI and all available CUDA feature/hardware gates pass.
