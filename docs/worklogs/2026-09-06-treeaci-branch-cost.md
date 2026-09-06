# TreeACI branch cost attribution (#732)

## Baseline and cause analysis

Baseline: `1fc754a` (main, including #733). The #732 downstream timings use
`2205972` and cannot establish the remaining cost at this baseline.

The old diagnostics merge different Guard operands by node Debug label, replace
the shape of an accumulated record with the last observed shape, omit physical
dimensions and batch sizes, and measure recursive message calls inclusively.
Kernel counters are process-wide and cannot be attributed to a node. Consequently
they cannot support the dimension-adjusted comparison requested by #732.

For a node core A with physical dimension d and incident bond dimensions
chi_0,...,chi_(z-1), the stored element count is

    S = d * product(chi_e).

A directed message at fixed physical coordinate x is

    m[a_0,p] = sum_(a_1,...,a_(z-1))
                 A[x,a_0,...,a_(z-1)] * product_j M_j[a_j,p].

The leading scalar work per distinct assignment is proportional to S/d, not
maximum chi. Equal incident dimensions give chi^z; z=3 therefore legitimately
adds one chi relative to z=2. A grouped contraction can reuse slices across
points, and candidate frames can evaluate Cartesian products of child samples.
S is a dimension proxy, not an exact FLOP count. Batch size, cache state,
orientation and phase must be controlled or reported in the residual model.

More specifically, for outgoing dimension a, child dimensions b,c and P
assignments in one physical group, the branch message kernel packs an
`(a*c) x b` matrix, multiplies it by `b x P`, and contracts the remaining child
per point. Its leading costs are `O(abc)` setup, `O(abcP)` matmul, and
`O(acP)` accumulation. Setup amortizes as `abc/P`; a fixed backend-dispatch cost
also amortizes as `1/P`. Therefore the same S can have different per-point costs
without a cache bug.

Candidate frames are different: two children with r1,r2 candidate columns use
sequential Cartesian contractions costing `O(abc*r1 + ac*r1*r2)` per physical
coordinate. Dividing by r1*r2 gives `O(abc/r2 + ac)`. A maximum-chi-only or
full-point-only denominator loses these sample-count and aspect-ratio effects.
The fitted degree coefficients must consequently be interpreted alongside the
reported batch sizes, misses and kernel buckets, not as automatically avoidable
cost. These equations are direct derivations of the repository contractions;
no third-party implementation or additional algorithm is imported.

## Intended measurement

Repair attribution before changing any numerical kernel. Keep diagnostics behind
the existing opt-in feature; preserve operand and shape identity, exclusive
message time, separate query totals and frame timing, and thread-local kernel
deltas. No new numerical cache is proposed.

The release experiment will compare identical deterministic fixtures with
diagnostics disabled/enabled, with the same seed, tolerance and thread settings.
It will also pair chain and branch fixtures with identical local core buffers
where their dimensions permit reshaping; degree and unequal-bond sweeps report
the actual dimension product. Report every case, repetition and phase. Timing
gates will only be selected after an independent baseline noise study; the first
study is descriptive and does not claim an optimization speedup.

## Reading and verification

Read shared common/Rust performance, numerical and testing rules; repository
rules, README, PERFORMANCE_TIPS, API inventory, existing TreeACI design and
scaling benchmark; Guard/frame/evaluator diagnostics and their related tests.
Verification and measured results will be recorded below.

The diagnostics-enabled full test run exposed the existing process-global
counter test racing with parallel kernel tests (4 dispatches observed where its
own two groups require 2). The test now uses the thread-local kernel delta and
retains both branch and chain dispatch assertions. Related process-counter reads
were audited: the other production reads belong only to the explicitly
process-wide summary. Removed duplicate assertions did not uniquely exercise a
shared helper; the same contraction and counter-update paths remain exercised.

## Predeclared experiment protocol

Independent A/A noise study `target/branch-noise-1.json`: baseline `1fc754a`,
120 default cases, five alternating pairs, five timed repetitions after one
warm-up, CPU 2, all provider/Rayon thread settings one. Every numerical and
validity check passed. Maximum within-side relative MAD was 0.10345; median
paired ratios ranged from 0.90863 to 1.08738, but the largest upper bootstrap
95% ratio bound was 1.58659. This study does not support a useful strict timing
gate. No time bound is selected from candidate results, and no speedup or
timing-nonregression claim will be made. The runner provides an explicit gate
for a later independently calibrated host; attribution invariants are tested
deterministically now.

Before running the candidate, declare these complete descriptive experiments:

- Production-path comparison: baseline `1fc754a` versus the implementation
  commit containing this protocol, both diagnostics off, all 120 default
  cases, five alternating pairs, five timed repetitions per case.
- Attribution/observer-overhead comparison: the same candidate off versus on,
  all 432 full cases, five alternating pairs, five timed repetitions per case.
- Both use CPU 2 on this AMD Ryzen 9 6900HX Microsoft VM, release profile
  (`opt-level=3`, `debug=0`), rustc 1.98.0, default tenferro CPU/faer backend,
  seed 732, tolerance 1e-8, and all six runner thread settings one. Use identical
  fixture source and lockfile. JSON protocols record their SHA-256 values,
  binary hashes, commits, raw repeats, host load/frequency and all cases.
- Predeclared validity limits: relative MAD <= 0.20, one-minute load per
  logical CPU <= 1.5, observed per-process frequency ratio <= 1.5; any numerical,
  execution or validity failure makes the entire experiment inconclusive.
  No selective reruns or case deletion. No benchmark runs alongside builds/tests.

The original A/A executable is retained. Formatting the shared fixture before
the final baseline/candidate builds does not change its calculations; rebuild
both with the identical formatted source, and retain the new baseline under a
separate filename. Reports are written under ignored `target/` and the measured
results are summarized in a checked-in benchmark report.

Public-surface audit: README's experimental TreeACI claim remains accurate;
the shipped usage-skill references and live tutorials have no references to the
changed diagnostic signatures. Updated the design reference and benchmark
entry point; no tutorial numerical API changed.

## Measured result

Implementation commit: `f4c9479`. Both predeclared candidate experiments
completed with zero validity failures and no reruns. Production off/off paired
ratio median across 120 cases was 1.00139; the 432-case instrumented comparison
showed median observer ratios 1.03758 (ACI), 1.20282 (cold query), and 1.19805
(warm query). Maximum full-study relative numerical error was `2.30e-12`.
These are descriptive, not timing gates or a speedup claim.

At equal S=512, d=2 and caller batch 32, cold hub-message time was comparable
across degree, while ACI frame cost increased substantially. For input 0,
degree-3/4 frames spent approximately 90-93% of their measured time inside
matrix-multiply calls. Core-slice setup was about 1-2%, and batch sizes and
cache-miss fractions also changed. Source analysis identifies repeated small
first-stage matmuls as a follow-up target, not proof that all their time is
avoidable. No kernel optimization or new numerical cache is introduced.

See the [full report](../../benchmarks/results/2026-09-06-treeaci-branch-cost.md),
complete case CSV, all 40 fitted models, and protocol/hash JSON alongside it.
Every raw node observation and fit residual is retained locally under ignored
`target/branch-attribution-1.json`. The downstream GW stage fixtures were not
replayed; #671 remains a separate open performance question. Related-code audit
covered frame first-stage batching, the generalized grouped route, old counter
reads, C API evaluator option construction, and stale historical API documents.

## Verification

- Affected TreeACI/TreeTN release unit and integration suites passed with
  diagnostics both enabled and disabled. Existing manually ignored tests were
  not enabled; no tolerance was relaxed.
- Diagnostics-enabled affected-crate doctests passed (21 TreeACI, 169 TreeTN).
- Affected-crate release Clippy, including all targets and strict error/panic
  documentation warnings, passed.
- Python analysis/gate tests passed: known model coefficients, complete case
  matrices, paired-ratio direction, explicit gate pass/fail, and whole-run
  invalidation preserving failed/noisy pairs (seven tests).
- Formatting and the deterministic repository-rules review passed.
- Final cross-crate check: `cargo check --release --all-targets` and the release
  doctests are clean for every workspace crate that depends on the two changed
  crates (treetci, itensorlike, capi, partitionedtreetn, aci, quanticstransform,
  partitionedtt, quanticstci). Outside those crates and
  `benchmarks/rust/benchmark_treeaci_branch_cost.rs`, nothing references the
  diagnostics API, so the whole-workspace doctest run is left to CI. No
  `docs/book` page changed, so the mdBook wrapper was not rerun.
