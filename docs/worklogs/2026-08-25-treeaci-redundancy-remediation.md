# TreeACI redundancy remediation

## Scope

This pass implements the confirmed, semantics-preserving redundant-work items
from issue #686 on top of the TreeACI audit baseline. Correctness issue #687
remains a separate boundary: the saved R=10, T=0.01 artifacts show a large
nblock `pi_rtau` rank, but they are not a post-change replay of the current
crate.

The implementation preserves candidate ordering, contraction/reduction order,
transaction rollback, convergence inputs, real/complex scalar behavior, and
the existing working-memory limits. No tolerance was changed.

## Changes

- Removed the unused directional-pass `ranks_before` snapshot and
  `PassReport::rank_changed`; `updated_edges` and all scheduling order remain
  unchanged.
- Hoisted the physical-offset calculation out of the incoming-sample loop in
  the single-incoming all-physical kernel. The temporary offset table is now
  included in the working-byte check.
- Added a nontrivial physical-axis-order oracle for both real and complex
  values, so the hoist is checked against the exact column-major mapping.
- Reused one computed Guard evaluation hint for the input and output
  evaluators in the floating-zone callback. Empty global-pivot injection now
  validates capacity lengths and returns before checkpointing or cloning state.
- Added exact-value and no-mutation tests for the two Guard changes.
- Moved owned row/column samples out of `LocalUpdateResult` during transaction
  commit, removing per-sample clones while preserving the staged commit and
  rollback path.
- Made `FrameBuilder` priming use a non-copying `ensure_computed` path when a
  memoized frame already exists. The ordinary `compute` API still returns an
  owned vector when its caller needs one.
- Carried candidate cache keys from the initial lookup through insertion,
  avoiding a second key reconstruction for batched one- and two-incoming
  candidates.
- Computed algebraic edge bounds once during state initialization and passed
  them into initial-rank selection instead of walking the prepared tree twice.

## Audit decisions

- The frame append path already uses `extend_from_slice`; git blame traces
  that fix to the earlier frame-overhead work, so no duplicate change was
  made.
- Reusing the Guard start-point evaluation is deliberately deferred. It can
  change floating-zone evaluation and false-convergence behavior, and must be
  gated by a low-temperature/high-R regression reproducing #687.
- Structural changes to phase traversal, scratch ownership, and diagnostic
  `updated_edges` representation are deferred until paired release profiling
  demonstrates a material cost and a numerical oracle is available. These are
  not treated as harmless syntax cleanup.

## Verification

- `cargo fmt --all -- --check`
- `git diff --check`
- `cargo clippy --release -p tensor4all-treeaci --all-targets -- -D warnings`
- `cargo test --release -p tensor4all-treeaci --no-fail-fast`: 136 unit tests,
  7 public API tests, 1 rank-scaling test, and 18 doctests passed; 2 opt-in
  tests were ignored.
- `cargo test --release -p tensor4all-treeaci --features diagnostics
  --no-fail-fast`: 139 unit tests, 7 public API tests, 1 rank-scaling test,
  and 18 doctests passed; 2 opt-in tests were ignored.
- The saved gw-rs artifacts were inspected without rerunning the pipeline.
  Their provenance is R=10, T=0.01, mu=0.5, with CTTN `pi_rtau` max bond 192
  and nblock `pi_rtau` max bond 764. The resulting Pi/W difference is retained
  as a correctness regression target, not presented as fixed by these local
  unit tests.

## Downstream replay

The local Cargo patch in `gw-rs/sgw/Cargo.toml` was then used to run the
current worktree at R=10 for T=0.01 and T=0.1. The available logs contain
successful nblock runs through `Sigma`; the CTTN T=0.1 run also completed, and
the currently retained CTTN T=0.01 log contains all stages through `Sigma`.

- At T=0.1, nblock TreeACI agrees with the same-run SimpleTT slices within
  0.42% for Pi, 0.47% for W, and 0.10% for Sigma when normalized by the
  reference maximum per row. Its G0 agrees with the analytic lattice formula
  with maximum absolute row errors below `7.1e-4`. The CTTN comparison is also
  small at this temperature.
- At T=0.01, nblock TreeACI still agrees with same-run SimpleTT within 0.64%
  for Pi, 0.56% for W, and 0.44% for Sigma. This is evidence against a
  wholesale corruption caused by the redundancy patch, but it is not a proof
  that SimpleTT is an independent correctness oracle.
- The low-temperature nblock--CTTN discrepancy remains: the largest relative
  Pi difference is about 23.7% at n=0 and the largest W difference about
  12.5%. The current nblock `pi_rtau` max bond is 796 versus 210 for CTTN,
  while both report final TreeACI errors near `1e-4` and `Converged`.
- The observed wall times were 200.7 s (nblock) and 591.4 s (CTTN) at T=0.01,
  and 35.0 s (nblock) and 36.3 s (CTTN) at T=0.1. These are single runs, so
  they are directional rather than a controlled benchmark. The T=0.1 nblock
  run is about 10% faster than its retained pre-remediation run; CTTN is about
  8% slower, dominated by G0 variance.

This replay does not clear the low-temperature false-convergence concern:
Guard start-point reuse remains untouched, and #687 still requires a focused
correctness diagnosis rather than a performance-only conclusion.

## Follow-up gate

Before changing Guard start-point reuse or declaring #687 resolved, replay the
same low-temperature/high-R case with the current branch and compare the
saved NPZ slices, convergence histories, ranks, and exact/reference checks.
