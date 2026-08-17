# Where TreeACI's per-evaluation cost goes

Work log for the measurement that decides the order of the #646 performance
work. The pre-registration below was written and fixed before the benchmark was
written or run; the result follows it, reported against the rule as registered.

## Pre-registration

Filed before writing or running the benchmark. Supersedes v1, which was
discarded: v1's own host-noise rule (two runs within 20% per case) failed on
four of twelve cases, so its results are void and are not used here — including
to set any threshold below. The thresholds in this version are derived from what
the two competing explanations predict, not from any observed number.

## What changed from v1 and why

Two defects, both in v1's design rather than its outcome:

1. **The host-noise rule was unmeetable on this host.** Hand-rolled timing under
   `cargo test` on a consumer laptop does not reach 20% run-to-run stability.
   Criterion — already used by `benches/index_key.rs` and
   `benches/cached_function.rs` in this repository — does warm-up, outlier
   detection and confidence intervals, and reports when it cannot separate two
   measurements. Noise handling moves to the tool.
2. **A two-point ratio straddled a possible crossover.** v1 compared bond 128
   against bond 2 and called the result "independent" or "scaling" from that
   ratio alone. If the cost is fixed below some bond and data-proportional above
   it, that ratio mixes both regimes and the verdict depends on where the
   endpoints happen to fall. This version fits a slope across the whole range
   instead, which detects a crossover rather than being confounded by one.

## Hypotheses and what each predicts

The question is which lever to pull first on the ~80x per-evaluation gap that
#646 review makes merge-blocking: reduce the number of message contractions (a
cross-call cache and a centre hint, as the review specifies), or reduce the cost
of one contraction.

- **H-fixed**: one `evaluate_batched` call costs a fixed amount per directed
  message, set by contraction planning and dispatch rather than by the data.
  Predicts cost independent of bond dimension: slope of `log(time)` against
  `log(bond)` near 0.
- **H-data**: the cost is the contraction arithmetic and the data movement
  around it. A message contraction over a bond of size `chi` moves at least
  `O(chi^2)` elements and multiplies `O(chi^2)`–`O(chi^3)`. Predicts slope
  near 2 or above.

These predictions come from the operations' arithmetic, not from any
measurement.

## What is measured

Criterion wall-clock time for one `TreeTNCachedEvaluator::evaluate_batched`
call, over a fixed batch.

## Cases

Complete list, fixed now. All are run; none is added, dropped or re-run after
results are seen.

- topology: chain of `n` nodes, `n` in {8, 32}
- bond dimension: {2, 4, 8, 16, 32, 64, 128}
- local dimension: 2
- batch: one coordinate scan — both values of the site at position `n / 2`, all
  other coordinates fixed
- contraction centre: pinned to the varying site, so centre placement is not a
  free variable
- scalar: `f64`

14 cases.

## Build and host

- Benchmark source: `bench_warm_call_vs_bond` in
  `crates/tensor4all-treetn/benches/cached_evaluator.rs`. The file already
  existed; this work adds that function.
- Baseline commit: `5785611c` on `treeaci-crate`, plus that benchmark.
- Build: `cargo bench -p tensor4all-treetn`, release profile.
- Criterion settings: default warm-up and sample size; no per-case tuning.
- Host: whatever this machine is at run time. Noise is handled by criterion's
  confidence intervals rather than by a quiet-host precondition, which this
  host has already failed once.

## Statistic and decision rule, fixed in advance

For each `n`, fit an ordinary least-squares slope `s` of `log(median time)`
against `log(bond)` across all seven bond values, and report criterion's 95%
confidence interval for each point.

- **`s < 0.5` for both `n`**: consistent with H-fixed and not with H-data.
  Conclusion: reduce the per-contraction cost first; #626's cache is deferred
  pending a second measurement of what remains.
- **`s > 1.5` for both `n`**: consistent with H-data. Conclusion: the cost is
  the work itself, so neither a cache nor cheaper dispatch is the first lever;
  report and reconsider from scratch.
- **`0.5 <= s <= 1.5`, or the two `n` disagree**: neither hypothesis is
  supported as stated. Report the slope and the per-point intervals, claim
  neither, and treat the ordering question as open.

Additionally, if criterion's 95% intervals for adjacent bond values overlap
across the whole range, the measurement cannot separate the cases at all and is
reported as inconclusive regardless of the fitted slope.

The result is recorded in every branch, including when it contradicts the
hypothesis that motivated the measurement. No case is re-run to move a result
across a boundary, and no threshold is revised after seeing results.

## What this does not establish

A synthetic chain with a two-point batch, not the downstream workload. It
attributes cost *within one call*; it does not establish what share of an
end-to-end run these calls represent. That share is already known from the A/B
comparison on #646 — essentially all of the gap is per-evaluation cost, with
operator-call counts within 0.74–1.02x — and is what establishes the need.

## Result

Criterion medians, `cargo bench -p tensor4all-treetn`, one coordinate scan of
two points, evaluator built outside the timed closure, centre pinned to the
varying site:

| bond | n = 8 | n = 32 |
|---|---|---|
| 2 | 1.220 ms | 5.296 ms |
| 4 | 1.272 ms | 6.152 ms |
| 8 | 1.291 ms | 5.680 ms |
| 16 | 1.303 ms | 5.575 ms |
| 32 | 1.547 ms | 6.628 ms |
| 64 | 2.139 ms | 9.255 ms |
| 128 | 4.293 ms | 18.523 ms |

Fitted slope of `log(median)` against `log(bond)` across all seven bond values:
**0.257 at n = 8, 0.244 at n = 32**.

The interval check does not trigger: criterion's 95% intervals separate cleanly
at the top of the range (n = 8, bond 64 is [2.111, 2.168] ms against bond 128 at
[4.241, 4.350] ms), so the measurement distinguishes the cases.

### Verdict against the registered rule

`s < 0.5` for both `n`, which is the first branch: **consistent with H-fixed and
not with H-data**. H-data predicts a slope of 2 or above from the `O(chi^2)`
elements a message contraction moves; the measurement is an order of magnitude
below that.

Registered conclusion: reduce the per-contraction cost first; the cross-call
cache proposed in the #646 review is deferred pending a second measurement of
what remains once the unit price falls.

For scale: at n = 32 and bond 2, one call takes 5.30 ms for 31 directed
messages, or roughly 171 us per message. At bond 2 a message contraction spans
tensors of a handful of elements, so essentially all of that is planning and
dispatch rather than arithmetic.

### Reported but not registered

Split across the range, the slope is 0.02 below bond 16 and 0.74 at bond 32 and
above. This sub-range fit was not part of the registered analysis and is
recorded as shape only; it is not used to support the verdict. Note that even
the upper segment stays well below H-data's predicted 2.

## What this changes about the #646 plan

The review specifies the fix as a per-call centre hint plus a bounded cross-call
message cache. Both reduce how *many* message contractions happen. This
measurement says the *unit price* of one contraction is what is anomalous, and
that it is nearly independent of the data.

The two are multiplicative factors of the same product, so the end state is the
same whichever is done first. The order matters for a different reason: the unit
price fix is internal to `compute_stacked_message` and adds no public API, while
the cache carries a permanent public surface and the full cache-ownership
contract. The cache's value can only be judged against a corrected unit price —
at 171 us per avoided contraction it looks decisive; at a tenth of that it may
not carry its own complexity.

The sequence adopted is therefore: commit this benchmark, implement the centre
hint the review asks for (its ceiling is separately measured at about 1.5-2x on
a coordinate scan, since only the messages on the path from the varying site to
the centre are recontracted), reduce the unit price, then re-measure and decide
on the cache with numbers rather than in advance.

## Discarded first attempt

An earlier version of this measurement used hand-rolled timers under
`cargo test`. Its own registered host-noise rule — two runs agreeing within 20%
per case — failed on four of twelve cases, so that run was discarded and its
numbers are not used anywhere, including to choose the thresholds above. The
failure is why noise handling moved to criterion.

An exploratory run before any pre-registration adjusted the bond-dimension cap
after seeing results, which is the practice the protocol exists to prevent. It
is recorded here as the origin of the hypothesis, not as evidence for it.

## A note on the existing benchmark

`bench_bond_dim_scaling` in the same file varies the same parameter but
constructs `TreeTNCachedEvaluator` inside the timed closure, so it measures
layout construction and centre search on every iteration. In the ACI path
evaluators live for a whole run, so that benchmark describes cold-start cost
rather than steady-state evaluation. Both are worth having; they answer
different questions.
