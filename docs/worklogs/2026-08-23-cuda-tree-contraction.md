# CUDA TreeTN Contraction Worklog

## Summary

Implemented PR 2 of the issue #623/#553 GPU foundation: optional CUDA feature
plumbing, a caller-owned visible-ordinal-0 CUDA context, explicit dense
IdxTensor transfer, typed placement/context errors, a pairwise device-resident
TreeTN full contraction, and a separated-timing benchmark harness.

## Material reviewed

- `AGENTS.md`, `REPOSITORY_RULES.md`, shared Rust/GPU/performance rules
- issues #623, #553, and merged PR #674
- `docs/design/tensorbackend-session-entry.md`
- pinned tenferro `a21a4c6` CUDA backend, transfer, eager-runtime, einsum, and
  linalg contracts
- IdxTensor storage/eager/context paths
- generic TreeTN `contract_to_tensor` and core n-ary/pairwise contraction paths

## Review gates

- Design: `docs/design/cuda-tree-contraction.md`
- Reviewer: `reviewer-flash-opencode-go`
- Round 1: **Needs changes** — generic `contract_to_tensor` enters the default
  CPU graph runtime
- Fixed design: direct deterministic pairwise edge walk, same-dtype validation,
  `StorageKind::Dense` rejection, and host-provider feature propagation
- Final pre-implementation verdict: **Correct-to-merge**
- Implementation: `luna-implementer`, max thinking, with parent integration
- Post-implementation reviewer: `reviewer-flash-opencode-go`
- Initial verdict: **Correct-to-merge** with six minor findings
- Findings fixed: redundant validation removed; single/empty tests added;
  per-method rustdoc completed; GPU name reported; metadata tests strengthened;
  root sentinel replaced.
- Final re-review verdict: **Correct-to-merge**
- Final panic-baseline delta review: **Correct-to-merge**

## Decisions

- CUDA is optional and absent from all default feature sets.
- `CudaExecutionContext` owns one ordinal-0 `CudaBackend` and one lazy eager
  runtime built from a backend clone, preserving allocation identity.
- Transfers are explicit; tracked and structured inputs fail before CUDA
  access. No automatic transfer or CPU fallback exists.
- Dense transfer preserves dtype, shape, ordered full indices, values, CUDA
  placement, and eager context identity.
- The CUDA TreeTN method duplicates only the topology-level edge schedule. It
  uses `contract_pair`; it does not call generic `contract_to_tensor` or
  `T::contract`, avoiding the CPU n-ary graph runtime.
- Every node, intermediate, and final result is validated against the supplied
  CUDA context and common dtype.
- CUDA feature activation includes faer CPU support because host download
  reconstruction uses the existing default CPU eager context.

## Changed surface

- root and tensorbackend/core/treetn Cargo feature plumbing
- `tensor4all-tensorbackend/src/cuda.rs`
- `tensor4all-core/src/defaults/cuda.rs` plus minimal IdxTensor accessors
- `tensor4all-treetn/src/cuda.rs`
- CUDA transfer and TreeTN integration tests
- `tensor4all-treetn/examples/cuda_tree_contraction.rs`
- design index and design document
- panic baseline line shift caused by the feature-gated IdxTensor accessors
- this worklog

## Focused verification

Hardware: NVIDIA A100 80GB PCIe, visible ordinal 0.

- default tensorbackend cargo tree: no `tenferro-gpu`
- CUDA tensorbackend cargo tree: `tenferro-gpu` present
- CUDA no-default checks: tensorbackend/core/treetn pass
- CUDA clippy `-D warnings`: tensorbackend/core/treetn pass
- tensorbackend CUDA nextest: 196 passed, 2 skipped
- core CUDA nextest: 823 passed, 2 skipped
- treetn CUDA nextest: 726 passed, 1 skipped
- focused TreeTN CUDA tests: 9 passed
- core dense transfer tests: 2 passed
- CPU workspace nextest: 3161 passed, 16 skipped
- CPU workspace and CUDA-feature doctests: pass
- workspace docs and mdBook tests: pass (pre-existing rustdoc warnings only)
- repository-rules tests: 90 passed; dry run: pass
- panic audit: 0 unbaselined, 0 stale
- changed public-error-doc audit and `git diff --check`: pass

Small benchmark run:

```text
config chain_length=4 bond_dim=2 iterations=2
device name="NVIDIA A100 80GB PCIe" visible_cuda_ordinal=0 runtime=cuda-eager
residual_max_abs=0.000000e0
host_setup_ms=4.718398
cuda_context_setup_ms=191.252571
upload_ms=339.436728
cuda_warmup_ms=2.374865
cuda_steady_sync_ms_total=0.936309 cuda_steady_sync_ms_per_iter=0.468154
download_ms=18.629205
cpu_steady_ms_total=5.981228 cpu_steady_ms_per_iter=2.990614
```

The benchmark is a functional vertical-slice measurement, not a general GPU
performance claim; setup and transfer dominate this intentionally small case.

## Coverage impact

No test or tolerance was removed or weakened. New coverage includes typed
tracked/structured/foreign/mixed-placement and mixed-dtype failures, exact
transfer round trips, two-node/branched/scalar-output contraction parity,
resident host-access rejection, pairwise source contract, and benchmark
correctness.

## Remaining issue #623 work

This PR does not support zip-up/fitting, SVD/QR/truncation, TCI/ACI, tracked
transfer, structured storage, C API/bindings, automatic placement, or multiple
GPUs. Those require separately reviewed vertical slices after this explicit
contract is stable.
