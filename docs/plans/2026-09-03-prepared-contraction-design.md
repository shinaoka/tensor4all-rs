# Caller-owned prepared contraction design (#700)

## Decision

Add `tensor4all_core::PreparedContraction`, an explicitly caller-owned immutable
plan for repeatedly contracting `IdxTensor` operands with the same index and
axis-layout contract. It owns core label assignment and result ordering; no
thread-local/global core cache is added.

```rust
let plan = PreparedContraction::new(&[&a, &b, &c], options)?;
let result = plan.execute(&[&a2, &b2, &c2])?;
```

## Compatibility contract

Preparation records, per operand, the ordered full `DynIndex` values, explicit
dimensions, and axis classes. Execution requires:

- the same operand count;
- the same full indices in the same order (including ID, prime level, tags, and
  direction);
- the same explicit dimensions (dimension is validated separately because index
  equality alone is not the dimension contract);
- the same axis classes.

Values, dtype, storage payload, and gradient-tracking state may change. Full
index equality makes validation linear in total rank and prevents a later tensor
from introducing an unplanned contraction. Fresh SRC-generated index identities
are intentionally incompatible; the caller prepares a new plan when identity or
layout changes.

A mismatch returns the existing typed `IdxTensorError::ShapeMismatch` with
operation `prepared contraction`. Backend/materialization failures preserve their
existing source chain.

## Execution

- Zero operands fail during preparation; disconnected inputs retain generic
  contraction's rejection.
- One operand returns a clone.
- Connected binary/no-retain execution keeps the existing pairwise fast path.
- N-ary and retained dense execution reuses stored input/output labels and the
  backend's existing shape/dtype-aware `ConcreteEinsumPlan` cache/session.
- Structured and AD execution uses the same structured/AD-preserving path as the
  ordinary borrowed entry with stored labels and result metadata.
- Retained indices are resolved only during preparation; exact input identity
  validation proves they remain present during execution.

The type exposes no mutable plan internals, cache capacity, or compatibility
shim. `Debug` is metadata-only and `Clone` shares no hidden global state.

## SRC integration decision

Current SRC is generic over `TensorLike` and creates fresh batch/cap/link indices
as rank segments grow and on every top-level repetition. Exact caller-owned
plans therefore do not survive those boundaries, while a normalized plan would
have to revalidate the same contractability graph it is meant to avoid. Add no
SRC-specific reach-through. Measure the core repeated-contract benchmark and the
SRC #706 gates separately; adopt in SRC only if a future owner keeps exact index
identities stable and a fresh end-to-end gate passes.

## Tests and evidence

- Dense N-ary prepared execution matches ordinary contraction for repeated
  values and mixed dtype.
- Retained-index output order and values match.
- Diagonal/structured storage remains compact; tracked inputs preserve backward
  gradients.
- Operand count, dimension, full-index identity (including same-ID prime/tag
  variants), and axis-class mismatches return `ShapeMismatch` before backend
  execution.
- Existing binary, disconnected, and empty semantics remain covered.
- The ignored repeated-contraction benchmark compares ordinary versus prepared
  core calls after warm-up; promote only with a predeclared paired improvement
  and no correctness regression.
