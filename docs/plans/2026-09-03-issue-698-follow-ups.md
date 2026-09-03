# Issue #698 follow-up execution plan

## Objective

Close the low-bond SRC performance umbrella through small, independently
mergeable changes. Every optimization is measurement-gated; a failed need or
promotion gate closes the corresponding experiment without production code.

## Order and dependencies

```text
Item 1 binary dispatch (current branch)
  -> #705 fixed-rank semantics
  -> #706 valid benchmark gates
  -> #700 caller-owned prepared plan
  -> rebaseline
     -> #702 directed-message experiment
     -> #703 tree-cache alignment experiment
     -> #701 residual fixed-overhead profile/fix
          -> #704 flattened-batch primitive, only if still justified
  -> final cross-item audit
  -> close #698
```

`#702` and `#703` may be measured in parallel after the rebaseline, but their
implementations remain separate changes. No Cargo jobs share one target
directory concurrently.

## Phase 0: land item 1

**Scope:** current `fix/issue-698-small-bond` diff.

- Keep the connected binary/no-retain fast-path scope approved by the maintainer.
- Preserve generic connectivity and retained-index validation before dispatch.
- Keep SRC call sites generic and verify generic-vs-explicit timing.
- Run formatting, focused core/TreeTN tests, clippy, repository-rules review,
  and the recorded bond-32 A/B gate.

**Exit:** item 1 is merged; #698 links the merged PR and marks item 1 complete.

## Phase 1: settle semantics and measurement infrastructure

### #705 — fixed-rank SRC semantics

Decide clamp versus typed error versus exact-request behavior before benchmark
cases encode the current ambiguity. Record the mathematical support bound,
checked-arithmetic implementation point, public diagnostics, and boundary tests.

**Exit:** one documented canonical rank rule is merged and benchmark output can
report requested and effective ranks unambiguously.

### #706 — valid benchmark gates

Replace the invalid large-star case with bounded-center chain/tree cases. Define
hardware/provider lanes, memory formulas and guards, seeds, warm-up, repetitions,
statistics, host-noise rejection, correctness oracle, and promotion thresholds.
Keep the default local gate short; archive heavy studies outside the source tree.

**Exit:** every later issue has a reproducible current-main baseline and a
predeclared promotion/non-regression gate.

## Phase 2: remove reusable planning cost

### #700 — caller-owned prepared contraction plan

1. Profile current-main core/backend planning shares on #706 gates.
2. Write and review the public API design before implementation.
3. Implement explicit caller-owned preparation/execution; do not add an identity-
   keyed, thread-local, or global cache.
4. Test compatibility rejection/fallback for indices, shapes, dtype, layout,
   retained labels, structured storage, and AD.
5. Integrate with SRC only if its separate end-to-end gate passes.

**Exit:** repeated compatible calls demonstrably reuse core and backend planning,
public docs/doctests pass, and SRC either adopts the API with a passing gate or
records why it does not.

## Phase 3: rebaseline and run bounded experiments

After #700, rerun #706 and refresh the contraction section profile. Old shares
must not be used to justify new code.

### #702 — directed messages

Measure cost versus tree degree and bond size. If the path is material, compare
prefix/suffix partial contractions with the existing message reference. Promote
only with bounded memory, numerical parity, and a passing tree gate; otherwise
close with measurements.

### #703 — tree segment alignment

Measure runtime, cache reuse, over-generated columns/messages, and peak memory
for aligned versus ragged growth. Promote only if the primary tree gate passes
without a memory or other-topology regression; otherwise remove the candidate.

### #701 — residual fixed overhead

Profile first and select one ownership-level subpath. Preserve validation,
structured storage, AD, dtype promotion, layout, and source errors. Do not repeat
the four rejected experiments from #698 without new evidence.

**Exit:** each issue contains a PASS/FAIL/INCONCLUSIVE result and either one
independently mergeable fix or no production diff.

## Phase 4: last-resort batching

### #704 — flattened-batch primitive

Start only if the post-#701 profile still shows many small retained contractions
or QR appends as a material end-to-end cost. Review ownership and API design
before implementation. Prefer an existing core/backend batch seam; add no
application-specific C API and no per-column tensor construction.

**Exit:** a paired low-bond improvement passes with no high-bond, memory,
correctness, layout, or AD regression. Otherwise close as unnecessary.

## Per-change gates

- Branch from current `origin/main`; one issue and one independently mergeable
  diff per task.
- Record baseline/candidate commits and the full measurement protocol before
  candidate results are known.
- Do not relax numerical or coverage thresholds.
- Run focused release tests first, then formatting, clippy, doctests/docs when
  public APIs change, and repository-rules review.
- Any delegated implementation requires the repository's pre-design and
  post-diff cross-model review gate.
- Update the issue and its worklog with negative and inconclusive results, not
  only successful candidates.

## Final audit and closure

Re-run all #706 gates on one exact candidate commit. Audit specification,
correctness/AD/structured paths, performance, public API/docs, and available
CPU-provider/CUDA lanes. Close #698 only after every child issue is closed,
merged, or explicitly deferred with evidence and an owner.
