# TreeACI next phase: interpolation invariant, order-freedom experiment, injection restructure

Date: 2026-08-14
Branch: `codex/treeaci`
Status: approved design, ready for implementation planning

## Background

A theory audit of `~/treeaci/tree-aci-theory.tex` against the
`tensor4all-treeaci` implementation found the core mathematics sound: the
edge-cut bipartition, unique gluing, exact directed input frames, the reduction
to train ACI, the algebraic rank bound, the `2|E| - diam(T)` walk-length result,
and the complexity table all check out.

Three defects remain.

**B1 — no interpolation theorem.** The document proves only that every entry fed
to the local LU is a true sample of `Z`. It proves nothing about the TreeTN `Y`
that the algorithm actually produces. In particular `Y(sigma) = Z(sigma)` on
pivot configurations is never established, so the local error indicator
`epsilon_e` is formally disconnected from `Y`. The missing ingredient is
nestedness, which Assumption 4.2 explicitly permits to lapse.

**B2 — continuity is asserted as a mathematical necessity but is a consequence of
a representation choice.** The local ACI update never reads output cores; `M_e`
is built entirely from input frames. Therefore the output is a pure function of
the active pivot sets: nodes carry `Z` evaluations and edges carry
`P_e^{-1} = Z[I_{u->v}, I_{v->u}]^{-1}`. Each edge update fully rewrites both
endpoint cores from the current active sets, so consistency is preserved under
any edge order. The implementation instead stores a DMRG-style moving canonical
centre (`transaction.rs` `set_edge_ortho_towards` + `set_canonical_region`,
`local_update.rs` `left_orthogonal`, `schedule.rs` asserting the walk starts at
the centre), which absorbs `P_e^{-1}` into one side and creates the ordering
constraint. The perfect-binary-tree regression that motivated Proposition 6.2
is consistent with a canonicalization relabelling bug rather than an intrinsic
obstruction.

**B3 — global pivot injection can produce a singular pivot block.** Section 6.1
grows both orientations atomically and lets the already-represented side reuse
its record. That duplicates an element of `I_{u->v}`, giving `P_e` two identical
rows. The zero-padding convention masks this only until the next update, and if
a neighbouring edge is updated first it recomputes `T_u` against the enlarged
set while `P_e` is still the padded block-diagonal, so the represented function
jumps uncontrolled.

**B4 — the guard rebuilds input evaluators that can never change.**
`GuardEvaluators::new` constructs a fresh `TreeTNCachedEvaluator` for every input
on every guard invocation, although the inputs are immutable for the whole run.
`evaluate_inputs` also takes no split argument, so a floating-zone walk
re-contracts the entire tree for every coordinate it moves. Commit `c9ecb7f`
(#621) fixed exactly this in train ACI: input caches moved into
`ElementwiseProblem` and accumulate across sweeps and starting points, the
solution cache is created once per guard invocation because the solution is
fixed during the search, and walk batches are evaluated with the split at the
varying site so the shared side is contracted once. Measured effect there was
1.9 s to 5.2 ms at R = 20.

The progress notes filed this under "needs a bounded cross-call cache API in
TreeTN". Half of it does not: moving the input evaluators' ownership from
`GuardEvaluators` to `TreeAciState` is a pure lifetime change requiring no new
TreeTN surface. That half is in scope here; the cross-call bounded cache seam is
not.

## Reference baseline

TreeACI is aligned against the **current Rust `tensor4all-aci`**, not against
`AlternatingCrossInterpolation.jl`. The Rust port has since fixed a number of
bugs present in the Julia original — `convergence_criterion_like_julia` is
named "like" for that reason. Re-deriving TreeACI behaviour from the Julia
implementation would be a regression, and any future audit should compare
against train ACI's current state.

## Goals and non-goals

Goals:

- Establish the interpolation invariant in the theory document and make it
  machine-checkable in tests.
- Settle experimentally whether edge order affects correctness or only quality.
- Restructure global pivot injection so it cannot produce a singular `P_e`.
- Give the guard's input evaluators run-lifetime ownership and adopt #621's
  split-at-the-varying-node batch evaluation.

Non-goals for this phase:

- Rewriting the output representation (removing the moving canonical centre,
  externalizing `P_e^{-1}`). The B2 experiment decides whether this is worth
  doing; it is deliberately designed not to require it.
- A bounded cross-call cache API in TreeTN (the other half of the B4 story).
- Train-versus-TreeACI parity benchmarks.
- Parallel or discontinuous execution strategies.
- Replacing `build_random_output` with deterministic skeleton initialization.
- Any downstream integration.

## Already aligned with train ACI — do not re-litigate

Verified during the audit; no work needed:

- Convergence: TreeACI's `convergence_criterion` already mirrors #617 —
  guard-empty window, `RankLimited`, a cleanup pass under a binding
  `max_bond_dim`, and a `global_pivots_found` history.
- Option defaults match train ACI post-#619: `scale_tolerance: true`,
  `max_nglobal_pivots: 5`, `global_tolerance_margin: 10.0`, `min_sweeps: 2`,
  `max_sweeps: 20`.
- Diagnostics are ahead of train ACI. `TreeAciDiagnostics` already carries
  `evaluated_points`, `sample_arena_records`, `sample_arena_retained_bytes`, and
  `saturated_edges`. The oracle-evaluation-count assertions below use
  `evaluated_points` as-is; the candidate-size diagnostic hangs off the same
  struct rather than a new one.

## Design principle: the test suite is a performance canary

ACI is meant to be fast. A slow test signals an algorithmic defect, not merely
an expensive test. Consequently:

- All fixtures stay small (perfect binary tree of height 3, `d = 2`,
  single-digit rank caps).
- Every experiment test runs in the normal suite. Nothing is gated behind
  `#[ignore]` or an opt-in feature.
- Tests assert an upper bound on **oracle evaluation count**, not wall-clock
  time. Evaluation count is the real cost measure and is deterministic, so a
  regression that inflates candidate matrices or recomputes frames fails the
  suite instead of silently slowing it.

## Architecture

One new test-only module pair; everything else is a local change to existing
modules.

### `skeleton.rs` (new, `#[cfg(test)]`)

Single responsibility: reconstruct `Y` directly from the active pivot sets. It
does not read `state.output`, does not canonicalize, and does not use
`frames.rs` or `transaction.rs`. Independence is the point — it must be a
second opinion, not a restatement of the code under test.

```
skeleton_tensors(problem, arena, pivots, oracle) -> (Vec<T_v>, Vec<P_e_inv>)
skeleton_evaluate(tensors, sigma)                -> T
skeleton_dense(tensors)                          -> IdxTensor   // small trees only
```

`T_v` is `Z` evaluated on `Sigma_v x prod_{a in N(v)} I_{a->v}`; `P_e` is
`Z[I_{u->v}, I_{v->u}]`. The `oracle` closure evaluates `Z` at a global
configuration; tests supply it from dense inputs and the pointwise `f`.

Note this reuses the observation that `M_e[:, C_e]`, reshaped, *is* the
tree-TCI `T_u` tensor, because the `v` axis of the local candidate matrix is
exactly `I_{v->u}`.

### `validate.rs` (new, `#[cfg(test)]`)

Built on `skeleton.rs`:

```
check_nesting(problem, arena, pivots)            -> NestingReport
check_interpolation(problem, arena, pivots, oracle) -> f64
check_gauge_equivalence(state, oracle)           -> f64
```

- `check_nesting` reports, per directed edge, whether
  `I_{u->v} subset Sigma_u x prod_{a != v} I_{a->u}` holds against the *current*
  active sets.
- `check_interpolation` returns `max |Y_ref(sigma) - Z(sigma)|` over the pivot
  configurations. Under the B1 theorem this is machine precision.
- `check_gauge_equivalence` returns `max |Y_stored(sigma) - Y_ref(sigma)|`,
  establishing that the moving-centre form is a gauge of the skeleton. Because
  `build_random_output` is unchanged this phase, it is only meaningful after the
  first complete directional pass; tests must respect that.

### Pivot state restructure (B3)

`ActivePivotSets { ids: Vec<Vec<SampleId>> }` currently serves two distinct
roles. Split it:

| Type | Role | Constraint |
|---|---|---|
| `PivotPairs { per_edge: Vec<Vec<(SampleId, SampleId)>> }` | determines bond rank and `P_e` | must be a genuine cross; no duplicates on either side |
| `CandidateSets { per_directed: Vec<Vec<SampleId>> }` | supplies neighbouring edges' candidate spaces | need only contain the pivot projections; may be enlarged freely |

New invariant, checked by `validate.rs`: for every directed edge, the
projection of `PivotPairs` onto that orientation is a subset of
`CandidateSets` for that orientation.

Consequent changes:

- `local_update.rs`: candidate row/column spaces are built from
  `CandidateSets`; the `(row, col)` pairs selected by LUCI are written to
  `PivotPairs`, and both sides' samples are merged into the corresponding
  `CandidateSets`.
- `global_guard.rs`: `inject_global_pivots` writes `CandidateSets`, never
  `PivotPairs`.

### Correction, found during implementation

The original version of this section also called for deleting `pad_output_bonds`
and letting rank growth emerge from the next local update. **That is wrong for
the current representation**, and the attempt failed with
`dense payload length 16 does not match dims [2, 4, 1]`.

`transaction.rs::factor_indices` takes an output core's incoming axis dimensions
from `output.bond_index(edge)`, while the LUCI `left` factor has
`row_count = d_u * prod |candidates.ids[incoming]|` rows. Therefore:

> An output bond's dimension equals the **candidate-set size** on that cut, not
> the pivot rank.

Two consequences follow, both of which the pre-existing code was already
respecting for the right reason:

1. Enlarging a candidate set *requires* padding the output bond. `pad_output_bonds`
   stays.
2. Both directed candidate sets on a cut share one bond index, so they must grow
   together. Asymmetric growth is impossible here, and the duplicate id that
   lands on the already-represented side is the price. It is harmless once
   pivots are separated: a repeated candidate row is simply never selected
   twice by the rank-revealing step.

The actual B3 defect — a repeat reaching `P_e` and making it singular — is fully
resolved by the candidate/pivot split alone, because injection no longer touches
`PivotPairs`. Locked in by
`injection_leaves_pivot_pairs_untouched_and_keeps_bonds_in_step`.

This is the same underlying coupling as B2: the moving-centre representation ties
together things the skeleton view keeps separate — there the edge order, here the
candidate-set size and the bond dimension. Under a skeleton representation, where
node tensors are rebuilt from pivot sets, candidate enlargement would indeed be
free. Both couplings should dissolve together if the B2 experiment supports the
rewrite; neither is worth attacking piecemeal beforehand.
- Saturation changes meaning from "can this edge still grow" to "does this
  edge's candidate space already contain every algebraically possible
  configuration", which is the more honest statement.

### Precedent in train ACI

This split is not an invention. Train ACI already does it implicitly:
`add_global_pivots` (`tensor4all-aci/src/state.rs`) appends the injected row and
column to the **frames** — the candidate source — while the solution's bonds are
separately zero-padded and rewritten by the next sweep's rrLU. A duplicate row
in a frame is harmless there because it only ever acts as a redundant candidate,
and rrLU will not select it twice.

TreeACI's defect is therefore narrower than "duplication": it is that
`active.ids` serves both roles, so the duplicate leaks into the rank accounting.
`samples.rs` pushes the id onto the already-represented side anyway, then
`global_guard.rs` does `edge_ranks += added` and pads the bond. Making the split
explicit is the minimal change that restores train ACI's proven semantics.

### No eviction policy

Train ACI has no candidate eviction, and growth there is bounded by three gates
that TreeACI already implements with identical defaults: `max_nglobal_pivots`
(5 injections per guard run), per-edge algebraic saturation skipping, and
skipping a cut that already holds both the row and the column of the pivot.
Commit `b160bb7` (#619) shows that the fix for uncontrolled rank growth was
per-bond decisions and saturation skipping, not eviction.

This phase therefore adds **no** eviction policy. Instead it adds a diagnostic
reporting per-directed-edge candidate-set sizes, so that if monotone growth is
in fact a problem, the next phase can design a policy against measurements
rather than against a guess.

Also verified during this audit: TreeACI's option defaults already match train
ACI post-#619 (`scale_tolerance: true`, `max_nglobal_pivots: 5`,
`global_tolerance_margin: 10.0`, `min_sweeps: 2`, `max_sweeps: 20`). No drift to
correct.

### Guard evaluator ownership (B4)

Move the per-input `TreeTNCachedEvaluator`s out of `GuardEvaluators` and into
`TreeAciState`, constructed once when the state is initialized. The inputs are
immutable for the run, so the caches accumulate across sweeps and across
floating-zone starting points. The output evaluator stays per guard invocation,
matching #621's reasoning that the solution is fixed during a search but changes
between them.

Add a split argument to the batch evaluation path so a floating-zone walk that
varies one node contracts the shared side once and reuses it across the scan,
mirroring `TTCache::evaluate_many(points, split)`.

This is an ownership and signature change only. It introduces no new TreeTN
public API, and `evaluated_points` makes the improvement directly measurable.

## The B2 experiment

A test-only **pivot-only update** path: build the candidate matrix, run LUCI,
write back `PivotPairs`/`CandidateSets`, but skip `replace_edge_cores` and
canonicalization. Output quality is measured entirely through
`skeleton_evaluate`. This frees edge order without touching production code.

Controlled: global guard off, fixed seeds, fixed tolerance and rank cap.

Arms, on identical fixtures and seeds:

1. the current continuous minimum-retracing walk;
2. a fresh random permutation of all edges each sweep;
3. a fixed arbitrary order (edge index order).

Fixtures: path, Y tree, comb, perfect binary tree of height 3 (the fixture that
originally failed), degree-four star.

Metrics, recorded against **edge update count** rather than sweep count, since
the update count is precisely what is in dispute:

- `max |Y_ref(sigma) - Z(sigma)|` on a held-out random configuration set;
- the same on pivot configurations (the B1 interpolation error);
- final per-edge ranks;
- the nested fraction reported by `check_nesting`;
- oracle evaluation count.

Decision rule:

The quality criterion is **final per-edge rank at fixed tolerance**, not a ratio
of update counts. Commit `b160bb7` (#619) established that rank inflation at
fixed tolerance is the failure mode that matters for this algorithm, and it is a
sharper and less noisy signal than sweep counts.

| Outcome | Conclusion | Follow-up |
|---|---|---|
| arms 2 and 3 reach the tolerance within `max_sweeps` with every edge rank within +1 of arm 1 | order-freedom holds | next phase restructures the output representation; `MinimumRetracingWalk` becomes a heuristic |
| arms 2 and 3 reach the tolerance but inflate ranks beyond +1, or need more sweeps | order affects quality, not correctness | keep the continuous walk as default; parallelism is unblocked |
| arms 2 or 3 fail to converge or permute coordinates | the analysis is wrong | record the counterexample; continuity is promoted to a genuine necessary condition |

The pivot-configuration interpolation error is mandatory, not optional: a
coordinate permutation blows it up while the held-out error can still look
small. That is exactly the failure mode of the original regression, so the
experiment must be able to distinguish outcome 3.

Measured result from the test-only pivot/skeleton path:

| Fixture | Continuous walk `(updates, evaluations; ranks)` | Random permutation | Edge-index order | Max held-out / pivot error | Nested fraction |
|---|---:|---:|---:|---:|---:|
| Path | `(18, 180; [2,2,2])` | `(18, 180; [2,2,2])` | `(18, 180; [2,2,2])` | `1.25e-14 / 0` | `1.0` |
| Y tree | `(24, 356; [2,2,2])` | `(18, 268; [2,2,2])` | `(18, 268; [2,2,2])` | `2.49e-14 / 1.42e-14` | `1.0` |
| Comb | `(30, 484; [2,2,2,2])` | `(24, 396; [2,2,2,2])` | `(24, 396; [2,2,2,2])` | `2.84e-14 / 2.84e-14` | `1.0` |
| Perfect binary tree, height 3 | `(48, 876; [2,2,2,2,2,2])` | `(36, 704; [2,2,2,2,2,2])` | `(36, 700; [2,2,2,2,2,2])` | `1.17e-13 / 8.53e-14` | `1.0` |
| Degree-four star | `(36, 1044; [2,2,2,2])` | `(24, 700; [2,2,2,2])` | `(24, 700; [2,2,2,2])` | `7.11e-14 / 7.11e-14` | `1.0` |

Thus outcome 1 holds for this experiment: all three arms preserve the same
rank profile and nestedness, and both interpolation and held-out errors stay
at machine precision. The continuous walk is not needed for mathematical
correctness of this pure pivot/skeleton construction. It does retrace more
often and therefore evaluates more points on the branched fixtures; its
remaining role is an efficiency and representation constraint for the current
in-place moving-centre output path. This experiment deliberately does not
canonicalize or rewrite stored output cores, so it does not yet prove that a
generic moving-centre commit is order-free.

## Theory document changes (B1)

Edits to `~/treeaci/tree-aci-theory.tex`:

1. New definition in section 4: nestedness,
   `I_{u->v} subset Sigma_u x prod_{a != v} I_{a->u}`.
2. New theorem after section 5: if every directed pivot set is nested toward
   some root, then `Y = prod_v T_v prod_e P_e^{-1}` equals `Z` exactly on every
   pivot configuration. The proof is short once the document states that
   `M_e[:, C_e]` reshaped is the tree-TCI `T_u`, which places the construction
   inside the existing tree cross-interpolation framework.
3. Rewrite the discussion following Assumption 4.2: the schedule maintains
   nesting only in the direction of travel and the reverse pass restores the
   other half. This is isomorphic to train two-site TCI and is not a new defect,
   but the current text reads as though nesting were abandoned outright.
4. Rewrite section 6.1 for the candidate/pivot-pair split: injection touches
   only candidate sets; delete the zero-padding and per-cut atomic growth rules.

Proposition 6.2's wording is tightened to say what its proof shows — that
generic canonicalization destroys bond coordinate labels — rather than implying
discontinuity is itself the obstruction. Its conclusion is left standing until
the B2 experiment reports.

## Testing

New tests:

- `skeleton.rs` self-check: on small trees, `skeleton_dense` matches a dense
  full-tensor construction.
- `check_interpolation` reaches machine precision under the continuous walk
  (the empirical form of the B1 theorem).
- `check_gauge_equivalence` holds after the first complete directional pass.
- `check_nesting` sanity: fully nested after initialization; nested only in the
  direction of travel after one pass.
- B3: `P_e` is non-singular after injection.
- B3 ordering hazard: inject, then update a *neighbouring* edge before the
  injected edge, and confirm the result is still correct. This is the case the
  current zero-padding scheme mishandles.
- B3 diagnostic: candidate-set sizes are reported per directed edge, and the
  three existing growth gates (`max_nglobal_pivots`, algebraic saturation,
  already-represented skip) each demonstrably bind in a fixture.
- B2: the three arms as parameterized tests across all five fixtures.
- B4: a guard-enabled fixture asserts an upper bound on `evaluated_points` that
  the pre-change code would exceed, so the cache lifetime cannot silently
  regress.

Every one of these runs in the normal suite, with an oracle-evaluation-count
assertion attached.

New integration test file `tests/rank_scaling.rs`, mirroring train ACI's
`tests/rank_scaling.rs` from #619. TreeACI currently has only
`tests/public_api.rs`, and since the B2 verdict now turns on final ranks at
fixed tolerance, rank growth needs a guarded home of its own.

Regression obligations:

- `cargo nextest run --release -p tensor4all-treeaci` green.
- `cargo fmt --all`, focused clippy with `-D warnings`, focused rustdoc.
- Per `AGENTS.md`, after deleting `pad_output_bonds`, check whether its tests
  were the sole exerciser of any shared helper and add replacement coverage if
  so; attest to this in the PR body.

## Acceptance

The phase is complete when the theory document carries the nesting definition,
the interpolation theorem, and the rewritten section 6.1; the validator confirms
interpolation to machine precision under the continuous walk; the B2 experiment
has produced a recorded verdict under the decision rule above; injection no
longer constructs a singular `P_e` and `pad_output_bonds` is gone; the guard's
input evaluators live for the whole run and walk batches split at the varying
node, with the improvement shown in `evaluated_points`; and the full crate suite
passes within its evaluation-count budgets.
