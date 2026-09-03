# SRC benchmark gates design (#706)

## Goals

Provide one reproducible CPU gate runner for current-main/candidate comparisons,
replace the unbounded-degree star fixture, refuse oversized inputs/oracles before
allocation, and make unavailable backend lanes explicit.

## Benchmark fixture changes

- Keep deterministic 10-site MPO--MPS chains for low/high bond behavior.
- Replace `tree` mode's star with a rooted binary tree. Its maximum degree is 3,
  so each input core is bounded by `physical_dim^2 * bond_dim^3` rather than
  `physical_dim^2 * bond_dim^n` at the star center.
- Compute checked estimates before construction:
  - every chain/tree input core and total input bytes;
  - dense oracle output bytes (`2^n` values for MPO--MPS and `4^n` for
    MPO--MPO/tree);
  - maximum topology degree.
- Refuse runs above `T4A_BENCH_MAX_INPUT_BYTES` or
  `T4A_BENCH_MAX_DENSE_BYTES`; defaults are 512 MiB and 256 MiB.
- Warm each selected algorithm once before timed repetitions. Keep setup and
  dense-oracle time outside the measured section.
- Print machine-parseable requested/effective ranks, estimates, topology, seed,
  relative error, compile-time git commit, release/debug profile, enabled
  backend features, and backend selection. The runner rejects an expected
  commit/profile/feature mismatch and records each binary's SHA-256.

## Gate runner

Add a Python-stdlib runner under `scripts/` that accepts prebuilt baseline and
candidate binaries, alternates their order for each pair, fixes Rayon/BLAS
threads to one, applies a child-process address-space ceiling, captures host load
and every raw result, and writes JSON. GNU `time` records each child's peak RSS;
`RLIMIT_AS` is reported separately as virtual-memory enforcement, never as an
RSS measurement.

Each invocation declares before execution:

- suite and exact cases;
- pair/repetition counts;
- correctness tolerance;
- required improvement and allowed non-regression percentages;
- maximum paired-ratio and per-binary dispersion, peak RSS, and virtual memory;
- binary paths, expected embedded commit identifiers, SHA-256 digests, profile,
  enabled features, and backend.

For each case use paired ratios `r_i = candidate_ms / baseline_ms`. The point
estimate is `median(r_i)`; dispersion gates use
`MAD(r_i) / median(r_i)` and each binary's `MAD(time) / median(time)`. A
fixed-seed 10,000-resample paired bootstrap gives a percentile 95% confidence
interval for the median ratio. A promotion case passes only when the interval's
upper bound is at or below `1 - required_improvement`; a non-regression case
passes only when it is at or below `1 + allowed_regression`. Quick checks use at
least 5 pairs and full promotion studies at least 10.

Classification is `PASS`, `FAIL`, or `INCONCLUSIVE`, with no selective retries.
A baseline/shared timeout, memory failure, malformed output, identity mismatch,
or excessive baseline dispersion is `INCONCLUSIVE`. A candidate-only timeout,
RSS/address-space excess, malformed output, identity mismatch, correctness
failure, or excessive candidate dispersion after a valid baseline is `FAIL`.
Any numerical correctness failure is `FAIL` and takes precedence over an
unrelated `INCONCLUSIVE` case in the aggregate result.

## Suites

Every case uses physical dimension 2, SRC seed 1234, `final_svd=false`, and
requested max rank 32. Network-generation seeds are independently fixed and
reported: 7 for MPO--MPS chains, 11 for MPO--MPO chains, and 13 for binary
trees. Adaptive cases additionally fix `rtol=1e-4`, `atol=0`, `min_rank=2`, and
`rank_increment=3`.

Quick local suite:

1. chain MPO--MPS, 10 sites, MPO/MPS bond 32, fixed SRC;
2. chain MPO--MPS, 10 sites, MPO/MPS bond 32, adaptive SRC;
3. binary tree MPO--MPO, 7 nodes, bond 4, fixed SRC;
4. the same tree, adaptive SRC.

The full chain suite uses 10 sites, MPO/MPS bonds 4/8/16/32/64/128, and both
fixed/adaptive modes. The full tree suite is the Cartesian product of node
counts 3/7/10, bonds 2/4/8, and fixed/adaptive modes. Ten nodes keep the exact
`4^n` dense oracle below the default 256 MiB ceiling while still exercising
increasing tree size. All use the fixed settings
above and remain subject to preflight/process limits. Heavy raw records belong
in `tensor4all-benchmark`; the repository stores the protocol and curated
worklog only.

## Backend lanes

- Canonical required lane: default one-thread tenferro/faer CPU.
- Provider-injection lane: compile/run when the caller-configured TreeTN
  contraction surface can own an injected context; until then classify
  `UNSUPPORTED` with the missing seam, not as a CPU-faer pass.
- CUDA lane: compile the existing CUDA feature and run CUDA TreeTN smoke checks.
  SRC itself currently constructs CPU probes through `TensorLike` and has no
  device/context-owning API, so CUDA SRC is `UNSUPPORTED` rather than silently
  inferred from dense CUDA contraction. Record GPU/runtime diagnostics and the
  exact command needed once an SRC device-context seam exists.

Adding provider/device execution to SRC is not part of a benchmark-only change;
it requires its own reviewed public-API design.

## Tests

- Unit-test checked memory estimates, overflow, and binary-tree degree bounds.
- Unit-test runner parsing/statistics/classification with synthetic output.
- Smoke-run every quick topology with an exact oracle.
- Verify CUDA feature compilation and the existing CUDA smoke test on available
  hardware, while reporting SRC CUDA support separately.
