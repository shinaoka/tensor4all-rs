# TreeTN square linsolve: MPO external-index handling

Source of the still-live rationale from the deleted `plan/linsolve-mpo.md`
(tensor4all-rs#566 Task 11 debris removal).

## Scope

`crates/tensor4all-treetn/src/linsolve/square` solves
`(a₀ + a₁ A) x = b` over tree tensor networks where input and output spaces
coincide (`V_in = V_out`).

## Conventions (implemented)

- The local problem is `⟨ref|H|x⟩` vs `⟨ref|b⟩` — the reference state is the
  conjugated bra and `b` is the ket. See the `ProjectedState` rustdoc in
  `crates/tensor4all-treetn/src/linsolve/square/projected_state.rs` and the
  implementation of `relative_linear_system_residual` in
  `crates/tensor4all-treetn/src/linsolve/square/mod.rs`.
- Operators with internal input/output indices are supported through explicit
  per-site `IndexMapping { true_index, internal_index }` maps; both input and
  output maps must be provided together.
- External-index mismatch between `init` and `rhs` is validated before the
  sweep (`validate_linsolve_inputs` in `square/mod.rs`); the local
  index-structure mismatch diagnostics live in
  `crates/tensor4all-treetn/src/linsolve/square/updater.rs`.
- One-site systems are rejected with an explicit error: the two-site sweep
  planner produces an empty plan on a one-node network, which would otherwise
  silently return the initial guess.

## Verification

- Mapped identity solve and solution reproduction:
  `crates/tensor4all-treetn/tests/linsolve.rs`
  (`test_square_linsolve_with_mappings_identity`, one-site rejection test).
- Matching/mismatching MPO-like structures:
  `crates/tensor4all-treetn/tests/linsolve_mpo_xb.rs`.

## Follow-up

- A real one-site local solve (rather than rejection) is not implemented.
