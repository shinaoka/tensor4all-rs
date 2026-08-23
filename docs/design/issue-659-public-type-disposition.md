# Issue #659: public type and scalar-capability disposition

## Status

Approved for implementation. Independent read-only review by `reviewer-flash-opencode-go` returned **Correct-to-implement** after the first-round caller-migration findings were incorporated; a focused amendment review also approved the collision-free `BackendScalar` name.

## Problem

Issue #566 recorded “Collapse duplicate types” as complete after only renaming the positional tensor-train type to `SimpleTensorTrain`. Three distinct gaps remain:

1. both tensor-train crates export an unrelated `TensorTrainError`;
2. `tensor4all-core::AnyScalar` coexists with tensorbackend's compatibility alias `AnyScalar = Scalar`;
3. core's general `Scalar` and MatrixLUCI's public `matrixluci::Scalar` duplicate the same arithmetic methods, while several purpose-specific scalar capability traits are undocumented as a family.

A single workspace-wide error enum or scalar supertrait would couple the two independent tensor-train stacks and force unrelated backend, storage, factorization, and algorithm capabilities onto every generic function. The fix must remove genuine naming and implementation duplication without creating that dependency inversion.

## Decision

### 1. Give the positional error its representation-specific name

Rename `tensor4all_simplett::SimpleTensorTrainError` to `SimpleTensorTrainError`. Keep `tensor4all_itensorlike::TensorTrainError` because its owning public type remains `TensorTrain`.

Do not leave a compatibility alias: the repository is pre-1.0, #566 explicitly rejected compatibility aliases for the corresponding tensor-train rename, and retaining the old alias would preserve the collision reported by #659.

The enums remain separate because their variants encode different invariants:

- positional core shapes, flat indices, and MatrixCI errors for `SimpleTensorTrain`;
- named indices, TreeTN structure, orthogonality, and `IdxTensor` factorization for itensorlike `TensorTrain`.

A shared enum would make each lower-level crate depend on the other stack or move stack-specific variants into an artificial facade, contradicting the documented two-stack architecture.

Migrate the complete workspace surface mechanically, including simplett rustdoc and internal imports plus the `#[from]` wrappers in `tensor4all-aci`, `tensor4all-interpolativeqtt`, and `tensor4all-tensorci`. Keep those wrapper variant names representation-specific as well.

### 2. Make core the only public `AnyScalar`

Remove tensorbackend's `pub type AnyScalar = Scalar` compatibility alias and rename the concrete backend value to `tensor4all_tensorbackend::BackendScalar`. Update tensorbackend signatures, tests, rustdoc, and core's explicit conversion boundary accordingly. The representation-specific name avoids replacing the `AnyScalar` collision with a new collision against the public core `Scalar` trait.

Keep `tensor4all_core::AnyScalar` as the sole `AnyScalar`: it is not the same value type. It can retain an eager rank-0 `IdxTensor`, AD tracking, gradients, and deferred operation errors, while `BackendScalar` is a compact untracked scalar value. Core continues converting explicitly to/from `BackendScalar`.

### 3. Remove the genuinely duplicated scalar-method trait

Rename the trait declaration `tensor4all_core::matrixluci::Scalar` to `MatrixLuciScalar` and re-export that name directly rather than through `pub use ...::Scalar as MatrixLuciScalar`.

Make `MatrixLuciScalar: tensor4all_core::Scalar + BackendLinalgScalar + MatrixSolveScalar + MatrixTriangularSolveScalar`. (`MatrixScalar` is already implied by core `Scalar`.) Remove its duplicate declarations and implementations of `conj`, `abs_sq`, `abs`, `abs_val`, `from_f64`, `is_nan`, and `epsilon`; those come from the core `Scalar` supertrait. Keep only the MatrixLUCI dispatch methods and their four concrete implementations.

Rename every internal `matrixluci::scalar::Scalar` import and update its tests. Only the two dispatch methods remain callable through `MatrixLuciScalar`; constructors and arithmetic methods must resolve through core `Scalar`. In particular, migrate the existing UFCS `MatrixLuciScalar::abs_val` calls in `tensor4all-aci` and `tensor4all-quanticstci` to core `Scalar` (or unambiguous method syntax), because Rust does not expose supertrait methods as associated items of the subtrait.

Keep the remaining purpose-specific traits, because they describe non-identical capability boundaries:

- core `Scalar`: common value arithmetic;
- tensorbackend `TensorElement`: conversion between supported Rust scalars and native tensors;
- tensorbackend `StorageScalar`: construction of compact storage representations;
- tensorbackend `MatrixScalar` and linalg traits: backend matrix kernels;
- simplett `TTScalar`: the conjunction needed by positional TT algorithms;
- ACI/TreeACI scalar traits: sealed algorithm-supported scalar sets and algorithm-only sampling/conversion behavior.

`CommonScalar` remains a documented alias of core `Scalar` for source readability at call sites that also use a local MatrixLUCI `Scalar` alias; it does not define a second trait or implementation.

## Documentation and historical correction

- Add the error/scalar ownership table to the architecture guide.
- Correct item 31 and the final-audit claim in `docs/superpowers/ledgers/2026-08-11-issue-566-pr4-ledger.md`; record what #595 actually completed and what #659 completes.
- Add a #659 worklog with the reviewed decision and verification evidence.
- After merge, comment on closed issue #566 with the correction and #659 merge evidence.

## Compatibility and behavior

This intentionally changes public Rust names but not numerical behavior, layouts, error variants, or error messages. No compatibility aliases are added. Compiler errors point downstream users to the representation-specific error name or backend `Scalar`.

## Verification

- source/API search: no exported `tensor4all_simplett::TensorTrainError`, no exported `tensor4all_tensorbackend::AnyScalar` or ambiguous backend value named `Scalar`, no public `tensor4all_core::matrixluci::Scalar`, no stale downstream `#[from]` path, and no `MatrixLuciScalar::abs_val` UFCS call;
- focused compile/tests and doctests for core, tensorbackend, simplett, itensorlike, ACI, TreeACI, TensorCI, TreeTCI, quanticstci, interpolativeqtt, and downstream workspace crates;
- public API inventory regeneration/check;
- formatting, clippy with warnings denied, release workspace tests, release workspace doctests, rustdoc, mdBook, crate-boundary/public-error/panic audits, and repository-rules review;
- successful PR checks and post-merge main CI.

## Non-goals

- one error enum shared by both tensor-train stacks;
- one scalar trait containing every storage, backend, factorization, AD, and algorithm capability;
- compatibility aliases that preserve the reported collisions;
- numerical algorithm changes.
