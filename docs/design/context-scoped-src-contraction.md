# Context-scoped SRC contraction

**Status:** Reviewed and approved for tensor4all-rs #720 and the first #623 Stage-2 migration

## Problem and baseline

`TreeTN` SRC currently creates Gaussian probes with `T::from_dense` and scalar
caps with `T::ones`. For `IdxTensor`, those context-free constructors enter the
process-global CPU eager runtime. Combining the resulting tensors with inputs
uploaded into a caller-selected `CudaExecutionContext` therefore crosses both
placement and eager-runtime identities.

The original NVIDIA A100 baseline reached a CPU extension from general SRC and
failed on a device-resident operand. The merged context-aware execution phases
removed that fallback. The final rank-revealing follow-up pins tenferro commit
`0457a2ed0aeea21b14f4297f7f4731e09b3a0507`, whose eager extension provides
fixed-output, traced-compatible RRQR on CPU-faer, CPU-BLAS/LAPACK, and native
CUDA.

## Invariants

- One caller-supplied execution context is the sole construction and validation
  authority for an SRC call.
- Both input trees, all probes/caps, every contraction/factorization/truncation
  output, and the returned tree belong to that exact context.
- Placement remains metadata on the underlying tenferro tensor; tensor4all
  adds no second device tag.
- There is no automatic host/device migration, CPU numerical fallback, backend
  construction, or process-global context lookup in the explicit path.
- Host-originated constructor data and bounded decision metadata readbacks are
  explicit context operations. They are documented and measured; they are not
  arithmetic fallbacks.
- Full `Index` identity, direction, prime level, tags, order, column-major
  payload semantics, dtype, storage class, eager AD identity, numerical
  policies, and source errors are preserved.

## Dependency-ordered PRs

### PR 1: tenferro eager extension target dispatch

Make eager general/N-ary einsum and linalg extension registration select the
validated owning runtime target. This is an upstream tenferro PR with its own
design and review gates.

### PR 2: tensor4all context foundation and tenferro pin

After PR 1 merges, update all seven tensor4all tenferro pins together. A
compatibility probe from tensor4all `origin/main` to tenferro `0190e43b` passed
both targeted CPU `--all-targets` and CUDA feature checks without source
changes, so a separate mechanical pin-only PR is unnecessary.

Add a reusable `TensorExecutionContext<T>` trait in `tensor4all-core`. It is
not SRC-specific. Its initial required operations are:

```text
validate_tensor(&self, operation, &T)
from_dense(&self, ordered_indices, column_major_host_data)
ones(&self, ordered_indices)
read_decision_data(&self, operation, &T)
```

The construction methods return `T` in the supplied context. The decision-read
method is an explicit, typed synchronization/readback boundary for small
algorithm-control payloads; it cannot be called by ordinary tensor arithmetic.
The interface remains generic and statically dispatched rather than storing a
trait object or execution context inside each tensor.

Implement the trait for `IdxTensor` with both `CpuExecutionContext` and
`CudaExecutionContext`. The implementations:

- validate eager runtime identity, placement, allocation domain, supported
  dtype, untracked status where required, dense axis classes, and storage kind;
- create a host native tensor from column-major data, then either adopt it into
  the supplied CPU eager runtime or explicitly upload it through the supplied
  CUDA context before eager wrapping;
- preserve the original ordered full indices;
- retain original tensorbackend/tenferro errors as typed sources.

No constructor derives a fresh tensor's home from another tensor. Inputs are
only checked against the caller's chosen context.

Extend `CudaExecutionContext` with `new_on(visible_ordinal)` and retain `new()`
as ordinal-0 construction only if it remains a documented convenience rather
than the canonical API. Context state and diagnostics record the selected
ordinal. No operation relies on thread-local current-device state. Allocation
and eager-runtime identity, not ordinal text alone, decide compatibility.

The explicit CPU implementation never consults the default context. Existing
CPU-global convenience APIs may remain at their current compatibility boundary,
but the new trait implementation and every explicit-context test are built and
checked without using them.

### PR 3: context-aware factorization and SRC decision primitives

Make the factorization operations required by SRC context-aware before changing
SRC itself:

- QR and column-pivoted RRQR for incremental probe factorization;
- SVD and device-resident slicing/truncation;
- the Appendix-C adaptive SRC error estimate;
- final-SVD factor absorption.

Eager QR/SVD/triangular-solve and elementwise/reduction operations execute in
the operand's owning runtime after the upstream dispatch fix. Identity/one
values are constructed through `TensorExecutionContext`.

Current `svd_truncated_inner` calls `EagerTensor::value()` and reads the full
singular-value vector on host to choose rank. Current
`IdxTensor::src_error_estimate` materializes the full QR factor into host
`Matrix` and performs the estimator on the CPU. Neither path is acceptable for
CUDA SRC.

Replace the adaptive estimator's CPU fallback with eager operations in the
same runtime: form the sketch factor's adjoint, solve the small square general
system there (the factor is restored from RRQR pivot order and is not generally
triangular), compute norm terms there, and read back only the bounded
scalar/vector decision payload needed by Rust control flow. SVD factors and slicing remain resident; only singular
values needed for rank selection/public diagnostics cross the explicit
`read_decision_data` boundary. Validate results after each factorization and
slice.

Record synchronization count and transferred decision bytes. This is the
unavoidable scalar/rank-selection readback anticipated by #623; it is not an
implicit tensor download and must never reconstruct an arithmetic operand on
CPU.

### PR 4: SRC API and algorithm migration

Make the canonical Rust SRC entry accept `&impl TensorExecutionContext<T>`.
Thread it through chain/tree scheduling, probe batches, caps, fixed/adaptive
factorization, optional final SVD, and result assembly. Validate both complete
input trees before RNG advancement or contraction. Mixed host/CUDA inputs and
foreign CUDA contexts fail at this boundary.

The API migration must leave no public CUDA-capable SRC route that omits a
context. Existing CPU convenience layers may explicitly pass their existing CPU
context, but generic SRC implementation code cannot call a global helper.
Context-free `T::from_dense`, `T::ones`, `T::scalar_one`, direct host matrix
conversion, and unscoped factorization are forbidden in production SRC files.

Fixed/adaptive semantics, deterministic seed prefixes, cap normalization,
chain/tree topology handling, final-SVD behavior, and existing CPU results stay
unchanged.

## Error model

Add non-exhaustive public errors with explicit variants for:

- host/device mismatch;
- foreign eager context or CUDA allocation domain;
- selected-device mismatch;
- unsupported dtype, storage, tracked value, or operation;
- context construction/transfer/synchronization failure;
- bounded decision readback failure;
- factorization and contraction failures.

Each wrapper stores the original lower-level error as `#[source]`. Validation
errors identify the operation and offending tree/node. No panic, string-only
flattening, or `anyhow` erasure may occur at the public boundary.

## Construction and transfer semantics

`from_dense` accepts host-originated column-major values. With a CUDA context,
its contract explicitly includes one host-to-device construction transfer. SRC
probe/cap construction therefore remains visible to instrumentation and is
reported separately from caller input upload and steady-state arithmetic.
There is no device-to-host conversion of probes, caps, intermediates, factors,
or results.

Decision readbacks are a separate category and are limited to:

- the scalar I64 rank returned by RRQR (permutation metadata stays resident);
- singular values needed to choose/return retained rank;
- final scalar estimates required by adaptive stopping.

A test counter rejects any other transfer during the algorithm.

## Tests

### CPU and hardware-independent

- Explicit CPU construction uses exactly the supplied eager runtime.
- Different CPU contexts are rejected.
- Column-major F64/C64 probe and cap values plus complete indices round-trip.
- CUDA features remain optional and CPU-only/explicit-only builds compile.
- Mixed placement, foreign context, unsupported dtype/storage/tracked inputs,
  and invalid options fail before RNG or arithmetic.
- A recording context proves every SRC constructor and validation hook is
  scoped; source-contract tests reject global-default calls in production SRC.
- Existing CPU fixed/adaptive and chain/tree result tests remain numerically
  unchanged.
- Runnable doctests demonstrate explicit CPU and feature-gated CUDA use with
  assertions.

### Real CUDA

For F64 and C64 where all required kernels are supported, cover the matrix:

- fixed and adaptive mode;
- chain and branched tree;
- `final_svd` false and true.

For every case:

- upload both inputs into one caller-owned context;
- validate every observed intermediate and the output against its exact eager
  runtime/allocation domain;
- compare the explicitly downloaded result with CPU using complete index order
  and a dense whole-result `maxabs` residual;
- assert no CPU arithmetic fallback and no unclassified transfer;
- verify mixed host/CUDA and two-context inputs return typed source chains before
  work begins.

Unsupported dtype/operation cases fail at entry and do not partially execute.

## Benchmark evidence

The release benchmark reports independently:

1. host fixture setup;
2. CPU/CUDA context setup;
3. explicit input upload;
4. probe/cap construction transfer;
5. warm-up/JIT;
6. synchronized steady-state SRC;
7. decision-readback synchronization/count/bytes;
8. explicit result download;
9. CPU steady-state reference.

Correctness and residency assertions are outside timed loops. Warm-up is never
included in the steady-state statistic.

## Sibling audit

Audit all touched construction/factorization helpers and all uses of
`T::from_dense`, `from_dense_any`, `ones`, `scalar_one`, host `Matrix`
materialization, `EagerTensor::value`, and `to_vec` reachable from SRC. Fix
shared seams needed by SRC in PRs 2-3. Record non-SRC callers in #623 rather
than adding local compatibility shims.

## Non-goals

- Distributed or partitioned execution of one contraction.
- Automatic device choice, sharding, or transfer.
- CUDA LU/CI or unsupported linalg emulation.
- Storing execution contexts inside `IdxTensor`.
- A process-global CUDA registry or thread-local current device.
- Migrating every tensor4all algorithm to explicit contexts in #720; the seam
  is reusable and #623 owns subsequent entry-point migrations.
- Changing AD rules, structured-storage semantics, or index equality.

## Rank-revealing QR follow-up

The transitional resident rank guard inspected successive diagonal entries of
non-pivoted QR and therefore could miss independent columns after an
interspersed dependent column. The follow-up replaces that loop with upstream
column-pivoted RRQR. Its scale-invariant SRC policy is
`rtol = 32 * f64::EPSILON * max(rows, columns)` and `atol = 0`; rank is the
strict leading RRQR diagonal prefix defined by tenferro. Only the scalar rank
is explicitly synchronized to host control flow.

RRQR returns `A[:, permutation] = Q R`. To retain the tensor4all factorization
contract without downloading permutation metadata, tensor4all computes
`right = Q_rank^H A` in the owning context. This restores the original sketch
column order. The resulting square right factor is generally non-triangular,
so the Appendix-C estimator uses a general solve for this small sketch matrix;
the legacy global-CPU `IncrementalQr` path and its triangular update remain
unchanged.

## Review and verification gates

Pre-implementation design review:

- Reviewer: `reviewer-flash` (DeepSeek family, read-only)
- Verdict: **Correct-to-implement**
- Evidence: the reviewer verified the current context-free probes/caps, full
  singular-value host read, host-Matrix adaptive estimator, proposed reusable
  trait and PR split, and found no blocking issue.

Each PR requires a recorded pre-implementation design verdict and a post-diff
verdict from a different model family, with fixes re-reviewed. Run repository
format/clippy, changed-crate release tests, CUDA-gated tests, workspace nextest
and HDF5, workspace doctests, rustdoc, mdBook, API inventory,
repository-rules review, coverage-impact audit, and clean diff/status checks.
Before merge, fetch `origin/main`, update the branch, rerun affected gates and
CI, verify exact head/check state, then squash-merge in dependency order.
