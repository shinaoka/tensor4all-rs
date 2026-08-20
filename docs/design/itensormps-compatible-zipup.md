# ITensorMPS-compatible chain zip-up contraction

## Status

Approved chain-schedule implementation, extended for implementation with
policy-aware automatic SVD versus Hermitian-eigen factorization. The final
staged diff was reviewed by `reviewer-flash-opencode-go`; after adding coverage
for topology-preserving scalar-node branches, the fresh verdict was
`Correct-to-merge`. A separately configurable coarse fit initializer remains
follow-up work.

## Reference

The chain schedule is inspired by `ITensorMPS.jl` v0.3.45, commit `794c97d`,
`src/mpo.jl`, method `ITensors.contract(::Algorithm"zipup", A::MPO,
B::AbstractMPS; ...)`. The Rust implementation does not copy or translate
upstream source code. It independently expresses the same algorithm through
existing TreeTN contraction, canonicalization, and factorization APIs. Record
this as algorithmic inspiration, not as a license-bearing code derivation.

The relevant upstream schedule is:

1. Orthogonalize both operands at the first sweep site.
2. Carry an intermediate `R` and contract `R * A[i] * B[i]`.
3. Factorize after each site through the third-to-last site.
4. Contract the final two sites into one block and factorize that block once.
5. Apply a final locally optimal truncation to a standalone zip-up result.

## Problem

The current TreeTN zip-up is a tree-general post-order implementation. On a
chain it differs from ITensorMPS.jl in three material ways:

- it does not orthogonalize both operands at the first sweep site;
- it factorizes the penultimate site before contracting the last site;
- it has no final locally optimal truncation sweep for standalone zip-up.

Its internal contraction already submits accumulated intermediates and both
site tensors to the FLOP-aware multi-tensor contraction planner. The planner,
not source expression order, owns the contraction order. This change must not
replace that planner with a hand-written pairwise contraction order.

## Scope

### Chain path

Detect a chain from topology degrees, independently of the requested center.
Use this path for unitary-form contractions with at least one surviving output
site index; other forms and pure-scalar contractions retain the general path.
Choose the requested center as the sweep endpoint when it is a chain endpoint.
For an interior requested center, choose a deterministic endpoint, sweep to the
opposite endpoint, then move the result center exactly to the requested node.
For that path:

1. Clone and orthogonalize each operand at the path's first node with exact QR.
2. Replace internal bond identities independently in the two operands.
3. Sweep toward the requested center, carrying the accumulated right factor.
4. At each ordinary site, jointly contract the accumulated factor and both site
   tensors through the existing multi-tensor contraction planner.
5. Use the caller's current SVD policy and maximum bond dimension for local
   factorization.
6. Contract the final two sites into one block, then factorize once with the
   right factor canonical, matching ITensorMPS.jl. This leaves the numerical
   center on the penultimate site.
7. Preserve the input topology when the path is used as a fit initializer. Move
   the numerical center from the penultimate site to the requested center with
   exact full-rank QR before returning the initializer.
8. For the standalone chain zip-up API, perform one final truncation sweep with
   the same policy and cap. The sweep finishes at the requested center and
   therefore also reconciles the final two-site gauge.
9. Keep pure-scalar contractions on the existing exact path. Canonicalizing a
   contraction with no surviving site indices only adds floating-point
   roundoff and provides no zip-up compression benefit.

### General-tree path

Keep the existing post-order tree-general implementation unchanged. There is no
single ITensorMPS sweep order for a branched tree, and inventing one is outside
this change.

### Fit initialization

`contract_fit` continues to use zip-up while preserving topology. The fit
initializer skips the standalone final truncation because the following
variational sweep is already locally optimal.

The low-rank random initializer is now the default starting state for the
public `ContractOptions::fit()` path in `tensor4all-itensorlike` — it starts
every bond at dimension 1 and lets the sweeps grow ranks per the SVD truncation
policy, so the exact `χ_A·χ_B` product is never materialized during
initialization. The zip-up initializer remains available explicitly for callers
who want the exact-start + refine behavior. No hard-coded tolerance
transformation is introduced here because `SvdTruncationPolicy` supports
relative or absolute scaling, values or squared values, and per-value or
discarded-tail rules.

### Decomposition choice

Add one method to `TensorFactorizationLike` for policy-aware automatic SVD or
Hermitian-eigen factorization. Its default implementation delegates to the
existing explicit factorization method, so generic tensor backends retain their
current behavior. `IdxTensor` overrides it with the automatic implementation.
The method accepts ordinary SVD `FactorizeOptions`; its default implementation
and the `IdxTensor` override both reject other algorithms with
`FactorizeError::InvalidOptions`. Explicit `FactorizeAlg::SVD` remains full SVD and therefore preserves
current behavior. QR keeps its separate options and is not used because it
cannot enforce a maximum retained bond dimension.

Do not add a new `FactorizeAlg` variant or C API enum value. Auto is a concrete
factorization operation used by zip-up, not a decomposition kind that every
fit, canonicalization, partial-contraction, C API, and full-rank dispatch site
must interpret.

The automatic path uses Gram-matrix Hermitian eigendecomposition only for
untracked `f64` and `Complex64` tensors when the policy gives a conservative
effective relative eigenvalue cutoff greater than `1e-12`:

- relative squared-value per-value: `threshold`;
- relative squared-value discarded-tail-sum: `threshold`;
- relative singular-value per-value: `threshold * threshold`;
- relative singular-value discarded-tail-sum: fall back to SVD;
- absolute policy: fall back to SVD;
- absent policy, zero cutoff, or max-rank-only truncation: fall back to SVD.

Tracked tensors fall back to SVD because the existing eigendecomposition API
returns eigenvalues as host scalars and cannot preserve derivatives through the
singular values.

The retained rank after eigendecomposition must exactly preserve every
`SvdTruncationPolicy` combination, including combinations that Auto currently
routes to SVD. For each nonnegative Gram eigenvalue `lambda`, evaluate the
policy on `sqrt(lambda)` for `SingularValueMeasure::Value` and on `lambda` for
`SquaredValue`, then reuse the existing relative or absolute, per-value or
discarded-tail rank logic, including minimum retained rank one even when every
nonzero mode is below cutoff. Apply `max_bond_dim` afterward. Return retained
`sqrt(lambda)` values in `FactorizeResult::singular_values`, matching SVD. Clamp
only negative eigenvalues attributable to roundoff; reject materially negative
eigenvalues.

For an unfolded `m` by `n` matrix `A`:

- when `m <= n`, diagonalize `A A^H`, retain `U`, and reconstruct
  `V^H = Sigma^-1 U^H A`;
- when `m > n`, diagonalize `A^H A`, retain `V`, and reconstruct
  `U = A V Sigma^-1`.

Build the smaller Gram tensor directly with native `IdxTensor` contraction;
do not materialize the rectangular tensor as a host `Matrix`. Sort eigenpairs
in descending order. Let `lambda_scale` be the largest absolute eigenvalue.
Reject any `lambda < -1e-12 * lambda_scale`; clamp eigenvalues in
`[-1e-12 * lambda_scale, 0)` to zero. All-zero matrices and retained zero
singular values use the SVD fallback, which preserves the established rank-one
and canonical-factor behavior without a special eigen basis.

Call Hermitian eigendecomposition with relative Hermitian tolerance `1e-12`.
Auto falls back to explicit SVD if Gram construction or eigendecomposition
fails. The deterministic policy gate decides whether eigen is attempted;
numerical failure never makes a contraction fail when SVD can complete.

The `1e-12` Auto boundary keeps requested Gram eigenvalues above the regime
where condition-number squaring and `O(epsilon * lambda_max)` Gram roundoff are
likely to dominate. Mathematical rank semantics match SVD, but spectra exactly
on a truncation boundary may select different ranks because the decompositions
round differently. Rank-equality tests therefore use spectra separated from
the cutoff and separately pin both sides of the strict Auto boundary.

Chain zip-up local factorizations call the automatic trait method. General-tree
zip-up, fit refinement sweeps, final standalone truncation, and explicit SVD
callers remain on their existing SVD paths.

## API impact

`TensorFactorizationLike` gains one documented automatic factorization method
with a default implementation. No enum or C ABI value changes. Existing
`contract_zipup`, `contract_zipup_with`, and `ContractionOptions::zipup` retain
their signatures.

## Correctness requirements

- Compare internal Hermitian-eigen factorization with explicit SVD for `f64` and
  `Complex64`, tall and wide matrices, both canonical directions, every policy
  combination, cap-only and zero-matrix edge cases. Assert retained-rank equality
  on spectra separated from each cutoff as well as reconstruction error.
- Verify Auto selects eigen only for the documented safe policy combinations and
  falls back for tracked tensors, unsupported policies, eigen numerical failure,
  and effective cutoffs at or below `1e-12`.
- Pin the strict Auto boundary for squared-value policies at `1e-12` and for
  singular-value per-value policies at threshold `1e-6`.
- Match a small dense reference exactly for untruncated `f64` and `Complex64`
  chains.
- Exercise one-, two-, and multi-site chains, including an interior requested
  center.
- Cover both MPO-like times MPO-like and MPO-like times MPS-like local site
  spaces.
- Verify truncated results obey the requested cap and a numerical error bound
  separately from the exact no-truncation reference.
- Verify standalone zip-up and fit initialization preserve the expected output
  site indices, chain topology, and requested canonical center.
- Keep a branched-tree regression to prove the existing fallback remains in
  use and numerically unchanged.
- Include a long-chain test whose intended path is cheap but accidental full
  dense materialization would fail quickly.

## Performance evidence

Measure on one pinned CPU core with Rayon, OpenMP, and BLAS set to one thread.
Setup, format conversion, and patch preparation stay outside timing. Record:

- total standalone zip-up time;
- fit zip-up initialization and fit sweep time separately;
- input, patch, and output bond dimensions;
- patch count and sampled numerical error.

The primary experiment lives in the separate
`tensor4all-benchmark-correctness` worktree. It contracts anisotropic Gaussian
MPOs at `N = 2048`, input maximum bond dimension about 256, patch cap 128, and
eight compatible patch contractions. The tensor4all-rs implementation worktree
does not own or commit those benchmark records.

## Deferred work

- A separately configurable coarse fit-initialization policy. (Implemented:
  see `FitContractionOptions` / the `FitInitializer`-style split between the
  topology-preserving zip-up initializer and the deterministic low-rank random
  initializer used by adaptive tolerance-driven fit. See
  `benchmarks/results/2026-08-19-adaptive-fit-lowrank-initializer.md`.)
- Rank-capped or rank-revealing QR suitable for patch-wise initialization.
- A distinct ITensor-inspired schedule for branched trees.
