# TreeACI branched hot-path investigation

## Scope and invariants

This work targets the cost of native branched TreeACI and its Guard evaluator.
It must not change the sampled function, candidate enumeration, LUCI pivot
selection, truncation tolerances, accepted ranks, or sweep convergence rules.
Every optimized contraction is required to implement the same finite sums as
the scalar reference for both `f64` and `Complex64`.

For a degree-three node rooted through one edge, Guard computes

```text
m_p(a) = sum_(b,c) A(x_p, a, b, c) m1_p(b) m2_p(c).
```

For TreeACI's Cartesian candidate frames, the corresponding local map is

```text
F(a, i, j) = sum_(b,c) A(a, b, c) V1(b, i) V2(c, j).
```

Both expressions permit reordered, batched contractions without changing the
mathematical algorithm.  A future change to a hierarchical cross algorithm
would alter candidate selection and is deliberately outside this optimization.

## Baseline and acceptance gate

- Benchmark baseline commit: `009bacc8bfb93f930d9515b6382047f014dba700`.
  Before submission, the branch was rebased onto origin/main commit
  `5a5303228d0c634d3d5daff7dffed0717151405e` (`fix(tensorbackend): return
  typed matrix swap errors`, #679); that upstream change is outside this
  benchmarked patch.
- Branch: `optimize-treeaci-branched-hotpaths`
- Compiler: `rustc 1.98.0 (88d9e12ae 2026-08-18)`, LLVM 22.1.8
- Host: AMD Ryzen 9 6900HX, 8 cores / 16 threads, 16 MiB L3
- Explicit thread environment: `OPENBLAS_NUM_THREADS=1`; the synthetic test is
  run with `--test-threads=1`.
- Command (the explicit feature works around a pre-existing standalone feature
  resolution failure recorded below):

```text
cargo test --release -p tensor4all-treetn \
  --features tensor4all-core/backend-tenferro \
  diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk -- \
  --ignored --nocapture --test-threads=1
```

Initial result: chain 84.867 ms, comb 2.712 s, comb/chain 31.96x.

At the baseline commit, `cargo test --release -p tensor4all-treetn` fails before
running tests because `tensor4all-core` is built without
`tensor4all-tensorbackend/global-defaults`, although core imports that legacy
surface. Workspace feature unification can hide this. The optimization
baseline therefore enables `tensor4all-core/backend-tenferro` explicitly until
the feature wiring is repaired and independently tested.

A candidate is retained only if all deterministic real/complex contraction
oracles and crate release tests pass. Performance candidates must be measured
with at least three post-warm-up samples. The first target is a 20% or larger
median reduction of comb time without a greater than 5% median chain
regression. Downstream CTTN, NBlock TreeACI, and SimpleTT comparisons remain
the final workload-level gate; stage outputs must remain within the existing
tolerances and no tolerance may be relaxed.

The first candidate makes `IdxTensor::with_dense_slice` borrow an already
host-contiguous retained value and changes the chain and branch raw kernels to
keep the core inside that borrow.  It preserves the prior materializing
fallback for backend-resident or non-contiguous values.  Three post-build runs
were 47.755/47.689/46.923 ms for the chain and
647.546/652.474/656.382 ms for the comb.  The medians improve by 44.5% and
76.3%, respectively, and the median ratio falls to 13.68x.  The candidate
therefore passes the predeclared synthetic performance gate.

On the existing downstream T=0.01 checkpoints, two cached coordinate sweeps
improved from 2.898 s to 846.8 ms for CTTN G0 and from 166.9 ms to 45.8 ms for
CTTN W.  The matching NBlock measurements improved from 112.6 ms to 94.2 ms
and from 23.1 ms to 19.5 ms.  Thus the CTTN/NBlock ratios fell from 25.7x to
9.0x for G0 and from 7.22x to 2.36x for W.  CTTN G0 contains 3.81x as many
core elements as its NBlock counterpart, including one 5,803,320-element hub.

A TreeACI candidate changed the two-incoming Cartesian contraction from
`incoming_dim_2 + 1` matrix multiplications to two matrix multiplications with
one algebraically inert axis permutation between them.  A new complex oracle
covers all 24 placements of physical, outgoing, and both incoming axes.  The
existing realistic branch microbenchmark changed from a median 2.312 ms to
1.255 ms (1.84x faster); its scalar-reference time remained about 30.6 ms.

The candidate was rejected after the downstream T=1 gate.  NBlock reproduced
the old ranks, errors, and values, but CTTN became pivot-path sensitive to the
changed floating-point reduction: Pi evaluated points increased from 163,260
to 184,164 and Sigma from 129,424 to 186,750.  The combined Pi/W/Sigma time did
not improve, and five extracted rows moved slightly farther from SimpleTT
(relative L2 changed from 3.24e-4--3.79e-4 to 3.51e-4--4.23e-4).  The original
contraction order was restored.  The 24-axis complex oracle and a centralized
scratch-size calculation remain as maintainability and correctness coverage.

After restoration, a fresh full T=0.01 CTTN run could not serve as a valid
performance gate because its pre-TreeACI G0 stage had already regressed: it
took 381.6 s, produced maximum bond 412, and reported sanity value
`0.2222113919-0.0015513282i`, versus the trusted run's roughly 14.9 s, bond 118,
and `0.2211443913-0.0154387910i`.  The run was stopped during `g_rtau`; no
TreeACI or Guard stage had begun.  Its temporary checkpoint and log are under
`/tmp/treeaci-guard-fix-cttn-T0.01`.  Therefore only the direct evaluator
checkpoint comparison above is used for this patch's T=0.01 performance claim.

## Structural audit

- The schedule executes a minimal continuous tree walk and updates each
  directed edge once per forward/reverse pair; exhaustive labeled-tree tests
  cover this invariant.  Branching does not introduce a redundant sweep.
- TreeACI contains neither SVD nor zip-up.  Local rank revelation is LUCI;
  Guard is a separate bounded global-search phase.
- Input Guard evaluators persist across passes.  Output messages are correctly
  rebuilt after output mutation, so the optimization borrows only immutable
  payloads and does not widen cache lifetime.
- The three copies of the two-incoming scratch formula are now one checked
  helper, preventing the planner, committed frame store, and incremental frame
  builder from disagreeing about the resource limit.
- `frames.rs` remains large and the committed-store and builder paths have
  parallel control flow.  They share the numerical kernels and resource
  accounting; a file split alone would not improve runtime or correctness.
  Candidate small-vector storage and matrix-shaped return buffers remain
  possible follow-ups, but were not changed without a workload measurement.
- Release clippy over core, TreeTN, and TreeACI all targets is warning-free.
  Library paths contain no unchecked `unwrap`/`expect`; occurrences found by
  the audit are confined to test modules.
- A standalone default-feature TreeTN build was broken because its direct
  legacy tensorbackend calls did not enable `backend-tenferro`.  CPU-faer and
  provider-inject feature propagation now enables both the core and direct
  tensorbackend legacy surfaces explicitly.

The more radical theoretical alternative is hierarchical or dimension-tree
cross interpolation, which avoids forming an edge's full Cartesian candidate
matrix.  That changes the pivot space and convergence theory rather than merely
reordering an equal contraction.  It is not an admissible correctness-preserving
optimization for this repair and would require its own algorithm design and
validation project.
