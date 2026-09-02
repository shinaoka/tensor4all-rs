# Successive Randomized Compression Reimplementation Design

## Status

Accepted for implementation after the repository audit on 2026-08-26.

## Goal

Replace the existing TreeTN SRC prototype with an implementation whose chain
path follows Camaño, Epperly, and Tropp's Algorithm 1 and adaptive QR procedure
exactly, and whose tree path is an explicit, tested rooted-tree extension of
the same randomized QB invariant.

The reference paper is [Successive randomized compression: A randomized
algorithm for the compressed MPO--MPS product](https://arxiv.org/abs/2504.06475).
The cited author implementation is
[RandomMPOMPS](https://github.com/chriscamano/RandomMPOMPS).

## Audit findings

The paper contains four relevant algorithm descriptions:

1. Randomized QB: generate Omega, collect Y = A Omega, orthonormalize
   Y = Q R, and project with B = Q* A.
2. SRC Algorithm 1: generate Gaussian site matrices once; contract the
   left-to-right prefix tensors C; obtain Y^(n), then repeatedly obtain
   Y^(j) from the saved C^(j-1) and the already projected S^(j+1).
3. Adaptive SRC: append random columns and update the affected C tensors;
   append the corresponding columns of Y, estimate the residual and norm,
   and repeat until the tolerance is met.
4. Appendix C: update QR and G = R^(-dagger) when columns are appended.

The author code implements the same mechanisms in two variants:

| Paper mechanism | RandomMPOMPS evidence | Required Rust behavior |
|---|---|---|
| One shared random column sequence | random_contraction keeps envs and only creates indices from len(envs) to the requested width | One RNG and one append-only probe bank per contraction |
| Saved prefix work | envs[idx][j - 1] is reused at later right-to-left sites | Chain C prefixes are computed once and reused |
| BLAS-shaped sketching | reshape, @, and batched np.stack paths in contraction.py | Avoid one-hot and axpby construction in the production dense path |
| Adaptive append | random_contraction_inc appends only new sketch columns | Existing columns and environments are never regenerated |
| Incremental QR | IncrementalQR.append and the optimized LAPACK/C++ path | Use a numerically stable incremental QR implementation |
| Appendix C estimator | error_estimate reads the inverse-triangular QR representation | Compute R^(-dagger) norms with conjugate adjoints |

The previous Rust implementation violates the performance-critical parts of
this table. It resets the RNG and environment column vector inside every edge,
recomputes all directed tree messages for every column, constructs Gaussian
vectors as sums of one-hot tensors, constructs the sketch one column at a time
with tensor objects, and recomputes QR during adaptive expansion. It is
therefore removed rather than incrementally repaired.

## Non-negotiable invariants

### Randomness

- A contraction owns one seeded Gaussian stream.
- A probe column contains independent standard real Gaussian entries for each
  physical index used by the current cut.
- Probe columns are append-only. Requesting width p + delta preserves the
  first p columns bit-for-bit.
- A later cut consumes a prefix of the same probe sequence; it does not reset
  the seed or draw a new sequence.
- Complex tensor values use real Gaussian probes, matching the paper's
  experiments and the author code.

### Chain SRC

For a chain rooted at the left end, the implementation must map as follows:

~~~text
Omega^(1)...Omega^(n-1)       one Gaussian site matrix per unprocessed site
C^(1)                         contract Omega^(1), H^(1), psi^(1)
C^(i)                         extend C^(i-1) with Omega^(i), H^(i), psi^(i)
Y^(n)                         contract C^(n-1), H^(n), psi^(n)
QR(Y^(n))                     eta^(n), R^(n)
S^(n)                         eta^(n)* H^(n) psi^(n)
Y^(j)                         C^(j-1), H^(j), psi^(j), S^(j+1)
QR(Y^(j))                     eta^(j), R^(j)
S^(j)                         eta^(j)* H^(j) psi^(j) S^(j+1)
eta^(1)                       final contraction with S^(2)
~~~

The chain path must never form the exact uncompressed MPO--MPS product. A
fixed-rank path uses the requested oversampled width only when final SVD
rounding is enabled, with the paper's default
max(ceil(1.5 * target), target + 10). An adaptive path starts at min_rank,
appends rank_increment columns, and caps at the configured maximum or the
local product rank.

### TreeTN extension

The paper proves the chain algorithm, not a general tree algorithm. TreeTN
uses this explicit extension:

- Root the common topology at center and orient every edge child -> parent.
- A global probe column assigns one Gaussian vector to every surviving output
  physical index in the whole rooted tree.
- For that column, one tree message pass contracts the original MPO--MPS
  operands with the factorized product probes. The message parent -> child is
  the complement-side sketch environment for that cut.
- Store only the selected complement message for each edge. Do not retain all
  directed messages after the column pass.
- Each edge consumes a prefix of the shared environment columns. Edge widths
  may differ in fixed or adaptive mode, but their column k is always the same
  global probe column k.
- Process edges bottom-up. The local tensor at a node is its two original
  operand tensors plus already produced child projected tensors. Its sketch is
  the local tensor contracted with the cached complement environments.
- Sibling edge QBs are independent cuts of the original product and are
  computed from the same global probe bank. This is the declared tree
  generalization; it is not described as a theorem from the paper.
- Exact-recovery tests must show that full-width edge sketches reproduce the
  original contracted tensor on chains and branched trees. Approximate tree
  results are validated against dense references and per-edge rank/error
  invariants.

This extension is deliberately different from a sequential sibling sweep:
an already compressed sibling is part of the local tensor when its parent is
processed, while the environment for each child cut is the original
complement-side message. This gives a well-defined bottom-up hierarchical
randomized projection and permits one shared message pass per global probe
column.

### Numerical linear algebra

- QR is economy QR with m >= p and a square upper-triangular R.
- The adaptive estimator is sqrt((1/p) * sum_i ||g_i||^-2) where
  G = R^(-dagger).
- The norm estimate is ||R||_F / sqrt(p).
- Complex paths use Hermitian adjoints everywhere.
- Adaptive expansion updates the QR representation and G without
  refactorizing all prior columns.
- Rank-deficient sketches terminate at the numerically saturated rank; they do
  not invoke an invalid inverse-adjoint estimator.

## Architecture

The implementation is split into private SRC units rather than growing
contraction.rs with another monolithic helper:

- treetn/src/treetn/src_probe.rs: append-only Gaussian probe bank.
- treetn/src/treetn/src_chain.rs: paper-faithful chain prefix/sketch/cap
  sweep and its tests.
- treetn/src/treetn/src_tree.rs: rooted tree edge schedule, shared probe
  environment messages, and bottom-up edge projections.
- treetn/src/treetn/src_qr.rs: TreeTN-facing wrapper around backend incremental
  QR state and Appendix C estimates.
- tensor4all-core construction capability: an optional dense probe/stack seam
  with a generic error fallback, implemented first for IdxTensor.
- tensor4all-tensorbackend/src/incremental_qr.rs: column-major dense
  Householder incremental QR for supported backend scalar types.

The public SrcOptions and contraction dispatch remain stable unless a test
demonstrates that an option currently has ambiguous semantics. No low-level
backend types are exposed by TreeTN. Generic tensor types retain the existing
correctness fallback, while benchmarked IdxTensor production paths use the
dense backend seam.

## Error handling

- Public option validation remains in SrcOptions and returns existing
  TreeTNOperationError diagnostics through the current dispatcher.
- Internal failures include the edge, direction, site, and phase in their
  anyhow context.
- Invalid dimensions, empty probe widths, m < p, zero adaptive increments,
  non-finite tolerances, and unsupported dense capability are rejected before
  mutating result state.
- No library production path adds unwrap or expect.

## Verification gates

Each implementation stage has its own release-mode test gate:

1. Probe bank prefix stability and Gaussian column shape.
2. Chain C reuse and exact full-width contraction.
3. Tree complement messages against direct cut contraction.
4. Incremental QR reconstruction, orthogonality, estimator, complex input,
   and rank-deficient rejection.
5. Adaptive output ranks and dense correctness for chain and branched trees.
6. Full crate tests, workspace doctests, clippy, and the latest benchmark.

The benchmark must report separate environment, sketch, QR, projection, and
final-round timings. A speed claim is accepted only when the environment
work scales with global probe columns rather than edge times probe columns,
and when correctness remains within the existing benchmark gate.

## Explicit non-goals

- Symmetry-preserving SRC; the paper explicitly lists this as future work.
- Linear combinations of MPO--MPS products.
- A new public backend API beyond the minimum dense and incremental seams
  needed by TreeTN.
- Keeping the previous prototype's internal helpers or behavior.
