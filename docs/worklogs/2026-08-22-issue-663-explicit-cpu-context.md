# Issue #663: explicit CPU execution context

## Summary

Added a caller-supplied CPU context as the canonical tensorbackend integration
path for plain operations, graph compile/run/cache, eager AD, and logical tensor
reconstruction. Legacy process-global operations remain behind the opt-in
`global-defaults` feature.

## Reviewed inputs

- tensor4all-rs issue #663 and Hataori P0 design
- tenferro-rs issue #1716 and merged PR #1717
- `tensor4all-tensorbackend` context, bridge, matrix, backend, and storage APIs
- repository and shared Rust/performance/test rules

## Decisions

- Pin tenferro PR #1717 merge commit
  `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`.
- Consume one configured `CpuBackend` and clone its runtime identity for graph
  and eager ownership; never construct an implicit backend in explicit paths.
- Keep graph-prepared entries private to each context-owned `Runtime`.
- Use a coarse legacy-module feature boundary instead of scattered per-function
  cfg gates.
- Add `LogicalTensor` because existing `Storage` intentionally supports only
  structured `f64`/`Complex64`, while transfer must preserve every CPU dtype.
- Keep MPI, serialization, Hataori types, TLS overrides, and global registries
  out of tensor4all.

## Review gate

- Design: `docs/design/explicit-cpu-execution-context.md`
- Reviewer: `reviewer-flash-opencode-go` (read-only, DeepSeek family)
- Rounds: initial findings fixed; feature inventory and exact mismatch path
  clarified; final verdict **Correct-to-implement**.
- Post-implementation diff review: `reviewer-flash` verified the full diff;
  verdict **Correct-to-merge**. One Minor documentation overstatement was fixed.

## Verification

- explicit-only `cargo check` and `cargo doc`
- explicit-only release tests
- default tensorbackend release tests
- workspace `cargo check`
- formatting and clippy gates

## Remaining risk

The optional Hataori adapter still owns wire encoding and the joint MPI/domain
integration test. Tensor4all exposes no serialization format and makes no MPI
claim.
