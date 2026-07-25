# Adaptive interpolation for partitioned tensor trains

## Context read

- Repository and Rust contributor rules, including numerical, performance,
  documentation, testing, and provenance requirements.
- The public APIs of `tensor4all-partitionedtt`, `tensor4all-tensorci`,
  `tensor4all-simplett`, `tensor4all-core`, and `tensor4all-tensorbackend`.
- `adaptiveinterpolate`, `createpatch`, and `_globalpivots` from
  TCIAlgorithms.jl commit `e501032278c9dd41b46c5851d8238169c8d178c5`.

## Alternatives considered

- Post-hoc splitting of one global TT was rejected because it does not retry
  TCI independently on difficult subdomains.
- Parallel patch execution was deferred to keep callback requirements minimal
  and seeded behavior deterministic.
- Dense identity cores at fixed middle sites were rejected because their
  storage grows quadratically with the carried bond dimension.
- Treating a child with no compatible recycled pivots as zero was rejected:
  an empty sample set does not establish that a function vanishes.

## Chosen design

Durable algorithm and API decisions are recorded in
[`docs/design/adaptive-tci-interpolation.md`](../design/adaptive-tci-interpolation.md).

- `tensor4all-partitionedtt::adaptiveinterpolate` runs TCI2 on each patch's
  active sites and splits a nonconverged patch in an explicit complete order.
- Zero- and one-active-site leaves use exact evaluation because TCI2 requires
  at least two sites.
- Full-domain user pivots and opt-in recycled diagonal pivots are filtered for
  each patch, deduplicated, and replenished to a configurable target with a
  seeded generator.
- Fixed middle sites carry bonds with compact structured storage using axis
  classes `[0, 1, 0]`, avoiding dense rank-squared identity cores.
- Sampled-zero detection remains an explicit finite-sampling policy and is
  documented as risky for sparse functions without known nonzero pivots.
- TCI2 convergence now compares its already-normalized error history with the
  configured tolerance rather than a magnitude-rescaled absolute tolerance.
- TCI2 reports whether it converged, reached the bond cap, or exhausted its
  iteration budget; adaptive interpolation accepts only full convergence.
- The TCIAlgorithms.jl MIT notice is retained in the crate and in the
  repository provenance table.

## Verification

- Focused adaptive interpolation tests cover low-rank real and complex values,
  batch callbacks, forced splitting, exact one-site fallback, sampled-zero
  patches, compact fixed-site storage, invalid input, opt-in recycling, and
  the no-compatible-recycled-pivot regression.
- The TCI2 convergence criterion has a regression assertion for normalized
  errors under large function scaling.
- Full contributor checks are recorded in the pull request.

## Remaining risks

- Sampled-zero classification can miss isolated nonzero entries not represented
  by supplied or seeded candidates; callers interpolating sparse functions
  should provide pivots in known nonzero regions.
- Patch execution is intentionally sequential; parallel scheduling can be
  considered later with an explicit callback thread-safety API.
