# TreeACI branch cost attribution (#732)

## Finding

At fixed local tensor size and query batch, the controlled Guard experiment
does not exhibit a universal positive degree penalty. The large remaining
degree-dependent cost in these fixtures is in candidate frames, predominantly
inside matrix-multiply calls, not in the separately measured core-slice setup.
This isolates a follow-up batching target; it is not a claim that all matrix
time is avoidable, or that the downstream GW discrepancy is resolved.

The prerequisite correctness fix is to distinguish input/output operands and
changing tensor shapes, measure messages exclusively of recursive children,
and report actual dimensions, phase-specific batch denominators and kernel
deltas. The previous diagnostics could not support that comparison. Numerical
contraction algorithms and resource limits are unchanged; no cache was added.

## Reproduction and complete evidence

Baseline: `1fc754a` (includes #733). Candidate implementation: `f4c9479`.
Subsequent report, tests and documentation edits do not change benchmark or
production calculations. See the [predeclared protocol and derivation](../../docs/worklogs/2026-09-06-treeaci-branch-cost.md)
and [runner instructions](../README.md#treeaci-branch-cost-attribution-732).

Three complete studies, each five alternating pairs and five timed repetitions
after one warm-up per process, were retained without selective retries:

1. Independent A/A noise study: 120 cases, identical baseline executable.
2. Baseline/candidate with diagnostics disabled: 120 cases.
3. Candidate diagnostics off/on: 432 cases (physical dimensions 2 and 3,
   degrees 2/3/4, four unequal-bond profiles, f64/Complex64, ACI, and cold/warm
   queries with batches 2/8/32/128).

The host was an AMD Ryzen 9 6900HX Microsoft VM, 16 logical CPUs, pinned to CPU
2. All six provider/Rayon thread variables were one. Builds used release,
`debug=0`, default CPU/faer backend, rustc 1.98.0, identical lockfile and fixture
source. No builds/tests ran concurrently with measurements. Frequency readings
were unavailable on this VM, so the frequency gate could not assess host
frequency changes; affinity is not CPU isolation.

Complete paired case summaries are in [wall.csv](2026-09-06-treeaci-branch-cost-wall.csv),
all 40 dimension-adjusted fits in [fits.csv](2026-09-06-treeaci-branch-cost-fits.csv),
and binary/source/lockfile hashes, host metadata, validity limits and raw-report
hashes in [protocol.json](2026-09-06-treeaci-branch-cost-protocol.json).
These compact artifacts are projections of the runner JSON, with no omitted
cases. Full raw repetitions, every per-node/per-phase observation, and every
fitted residual remain in the local ignored files `target/branch-noise-1.json`,
`target/branch-production-1.json`, and `target/branch-attribution-1.json`.
The commands below regenerate complete reports; use new output names to retain
old evidence:

```bash
python3 scripts/run-treeaci-branch-cost.py \
  --baseline target/branch-cost/baseline-final \
  --candidate target/branch-cost/candidate \
  --baseline-commit 1fc754a --candidate-commit f4c9479 \
  --pairs 5 --repeats 5 --cpu 2 --output target/branch-production-2.json
python3 scripts/run-treeaci-branch-cost.py \
  --baseline target/branch-cost/candidate \
  --candidate target/branch-cost/candidate-diagnostics \
  --baseline-commit f4c9479 --candidate-commit f4c9479 \
  --full --pairs 5 --repeats 5 --cpu 2 --output target/branch-attribution-2.json
```

## Wall time, numerical checks and gate decision

All three studies passed their predeclared numerical and validity checks.
Maximum within-side relative MAD was 0.10345 for A/A, 0.10701 for production,
and 0.11167 for attribution (limit 0.20). Maximum relative numerical error in
the full study was `2.30e-12`; every individual result passed its mode's bound.

| Comparison | Cases | Minimum / median / maximum paired ratio |
|---|---:|---:|
| A/A | 120 | 0.909 / 0.999 / 1.087 |
| Candidate/baseline, diagnostics off | 120 | 0.805 / 1.001 / 1.074 |
| Diagnostics on/off, ACI | 48 | 0.992 / 1.038 / 1.358 |
| Diagnostics on/off, cold query | 192 | 0.995 / 1.203 / 1.653 |
| Diagnostics on/off, warm query | 192 | 1.046 / 1.198 / 1.785 |

The table summarizes ratios across the complete case matrix; individual paired
confidence intervals are in the CSV. These are descriptive measurements, not
speedup or timing-nonregression claims. In particular, the largest A/A upper
95% bootstrap ratio bound was 1.587 despite modest MAD. The independent noise
study therefore did not justify a useful strict time bound, and no permissive
bound was chosen after observing the candidate. The runner implements an
explicit `--max-regression` gate for a future independently calibrated host;
tests verify that noise or a failed pair invalidates the entire experiment.
Deterministic attribution/shape/cache invariants are regression-tested now.

Diagnostics are an observer with substantial overhead for short queries. Do
not use instrumented wall times as production performance estimates.

## Dimension-adjusted residual

For each scalar/d/phase/operand/requested-batch/cache-state stratum, fit

    ns/point = intercept + beta*S + gamma3*[z=3] + gamma4*[z=4]
    S = d * product(incident bond dimensions).

Only hub rows are fitted; the fused leaves deliberately change their physical
dimensions. Each fit below contains 300 observations across four tensor sizes
and three degrees. Fits are descriptive: repeated rows are correlated and the
model does not remove aspect-ratio, route-threshold, or adaptive frame-batch
effects. In particular, a negative intercept is not a physical negative cost.

| d=2 stratum | beta (ns/element) | gamma3 (ns/point) | gamma4 (ns/point) | RMS residual (ns/point) |
|---|---:|---:|---:|---:|
| f64, cold Guard, caller batch 32 | 1.168 | -7.2 | -88.1 | 445.8 |
| Complex64, cold Guard, caller batch 32 | 1.637 | -432.9 | -451.7 | 668.6 |
| f64, ACI frame, input 0 | 4.512 | 4720.9 | 9767.4 | 3115.1 |
| Complex64, ACI frame, input 0 | 4.653 | 4453.3 | 9541.6 | 3095.8 |

The cold Guard fits have exactly 16 unique component assignments per batch and
100% misses throughout. The frame fits do not: mean batch sizes span 6 to 16,
and miss fractions span 0.5 to 1.0. Their degree coefficients must not be
interpreted as pure cache overhead. The complete d=3, other batch-size, warm,
and second-operand fits are retained in the CSV.

## Where the time goes at equal S

One cross-section of the complete experiment uses atomic bonds `[2,4,4,8]`,
d=2, hence S=512 at every degree. Actual hub bonds are `[2,128]`, `[2,4,32]`,
and `[2,4,4,8]`. Each topology represents the same dense function; the oracle
checks the tensor-product regrouping numerically.

Cold Guard, caller batch 32 (16 unique hub assignments), aggregated over all
25 timed repetitions:

| Scalar | z | ns/unique point | Setup % | Matmul % | Accumulation % |
|---|---:|---:|---:|---:|---:|
| f64 | 2 | 1914.9 | 13.6 | 45.6 | 0.3 |
| f64 | 3 | 1747.7 | 4.0 | 57.8 | 7.7 |
| f64 | 4 | 1782.0 | 9.7 | 49.5 | 7.8 |
| Complex64 | 2 | 2713.1 | 14.0 | 38.0 | 0.3 |
| Complex64 | 3 | 2158.2 | 4.0 | 67.4 | 5.4 |
| Complex64 | 4 | 1995.1 | 9.3 | 52.9 | 7.9 |

Every row dispatched two matrix multiplications per timed query. The remaining
phase fraction includes lookup, allocation, assignment handling and observer
work; these kernel buckets intentionally do not purport to cover every scalar
instruction. Across the entire warm-query matrix, new message matmul jobs were
zero, as expected from the cache-hit path.

For ACI frames at the same S, input 0:

| Scalar | z | ns/candidate | Mean batch | Miss % | Setup % | Matmul % | Matmul jobs/candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| f64 | 2 | 1332.2 | 6.0 | 75 | 0.0 | 46.8 | 0.125 |
| f64 | 3 | 6617.3 | 13.3 | 50 | 1.8 | 90.5 | 1.775 |
| f64 | 4 | 13042.8 | 16.0 | 100 | 1.0 | 92.6 | 4.500 |
| Complex64 | 2 | 1565.5 | 6.0 | 75 | 0.0 | 44.0 | 0.125 |
| Complex64 | 3 | 6801.1 | 13.3 | 50 | 1.1 | 91.6 | 1.775 |
| Complex64 | 4 | 13374.9 | 16.0 | 100 | 1.1 | 92.1 | 4.500 |

Here total ACI callback-point counts are 96, 192 and 256 respectively; topology
changes adaptive sampling even though the represented input function and S are
equal. Those counts are distinct from frame candidates and message assignments.

Source inspection explains the increased job count. For outgoing dimension a
and incoming dimensions b,c, `two_incoming_core_matrix_batched` makes c small
first-stage matmuls and one final matmul per physical group. Its arithmetic
cost is `O(abc*r1 + ac*r1*r2)`, but its c+1 backend calls also carry setup,
validation and allocation overhead. The generalized kernel starts with one
small matmul for every remaining incoming-bond combination; later stages use
the shared grouped-matmul facade. For q incoming dimensions b_0,...,b_(q-1),
the logical job count is `sum_(k=0..q-1) product_(j=k+1..q-1) b_j`. A grouped
API invocation counts each logical matrix job, not just the single API call.

Consequently, batching the first stage through the existing high-level matrix
facade is a concrete follow-up to investigate. The measurements locate time
inside those calls; they do not separately measure backend dispatch versus
arithmetic, and do not prove that a proposed batching change will improve it.
The changed cache-hit fractions and number of requested candidates also remain
part of the explanation. A new numerical cache is not justified by this study.

## Memory and limits

No numerical cache was introduced. Query records observe existing whole-
evaluator retained storage, including estimated map/message-capacity overhead,
and must not be summed across centers. Across cold/warm query cases, maxima
were 51,648 logical / 56,112 estimated owned message bytes and 49,152 logical /
49,536 estimated owned prepared-slice bytes. ACI Guard observations reached
88,512 logical / 106,312 estimated owned message bytes. These are observations
after queries, not peak RSS; backend allocator arenas and heap storage inside
generic node labels are excluded. Existing ACI frame/arena counters are also
retained in the raw reports. Observation rows are thread-local aggregates per
operand/node/shape and are released by resetting the diagnostics window.

The benchmark uses deliberately bounded small dense oracles and is not a GW
replay. The downstream Pi/W/Sigma fixtures and caller batch distribution must
be measured with this corrected interface before closing the broader #671
performance concern. Suggested related audit: compare the two-incoming and
generalized frame first-stage batching, then capture the downstream actual
shapes, batch sizes and miss fractions with the same per-phase report.
