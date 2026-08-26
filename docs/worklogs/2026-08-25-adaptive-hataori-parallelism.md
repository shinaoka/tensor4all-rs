# Adaptive Hataori parallelism work log

## Scope

Design and implement optional Hataori scheduling for adaptive TCI patches,
with default rank-local Rayon, optional MPI, physical one-pass child-cache
partitioning, and returned per-patch caches.

## Documents and code reviewed

- `docs/design/adaptive-tci-interpolation.md`
- `docs/design/adaptive-tci-parallel-execution.md`
- `crates/tensor4all-partitionedtt/src/adaptive_interpolation.rs`
- `crates/tensor4all-partitionedtt/src/adaptive_interpolation/tests.rs`
- `crates/tensor4all-partitionedtt/Cargo.toml`
- Hataori `Domain`, `LocalMode`, `map_in`, `pmap`, placement, and Cargo features
- Workspace and repository rules for public APIs, caches, performance, and tests

## Design decisions

- Hataori is an optional dependency enabled by the default
  `adaptive-hataori-rayon` feature; MPI is a separate default-off feature.
- Hataori receives explicit caller-owned domains. Rank-local patch scheduling
  uses `LocalMode::Outer`, allowing intentional nested Rayon work in the same
  pool.
- A split consumes its parent cache and moves every entry to exactly one child
  cache in one pass. Parent and sibling maps are not shared.
- Accepted caches are returned one-to-one with accepted `SubDomainTT`s.
- MPI processes breadth-first waves. Root broadcasts a fallible `WaveControl`
  only after all root post-processing and final validation, preventing rank
  divergence between collectives.
- All execution modes derive deterministic per-patch seeds from the root seed
  and stable patch path.

## Pre-implementation reviewer gate

Selected reviewer: `reviewer-flash` (read-only, high thinking).

### Round 1

Verdict: **Changes required**.

Findings:

1. Root post-wave reconstruction/final-validation errors were outside a
   collective and could deadlock or desynchronize MPI ranks.
2. The public `mpi::traits::Communicator` bound needed a direct optional
   `mpi = 0.8.1` dependency.
3. Sequential/Rayon seed equivalence and the intentional legacy RNG change
   needed to be explicit.
4. MPI domain, main-thread, and `MPI_THREAD_FUNNELED` requirements were missing.
5. Collective input, batch-cache, wire validation, cache stats, and MPI
   determinism tests were missing from the matrix.
6. The bare `PartitionedTT` return-type migration and documentation consumers
   were ambiguous.
7. `MissingPool`/`ForeignPool` entry restrictions were undocumented.

All findings were corrected in
`docs/design/adaptive-tci-parallel-execution.md`.

### Round 2

Verdict: **Correct-to-merge**.

The reviewer verified all seven corrections against current Hataori,
mpi-0.8.1, and tensor4all sources. It also reconfirmed that physical one-pass
cache splitting and accepted-cache return remain intact. Non-blocking
implementation notes: root post-processing must remain panic-free or abort the
MPI communicator on panic, and the exact stable integer seed mixer must be
pinned and documented before merge.

## Verification

Design-only checks completed so far:

- `git diff --check`
- design index link exists
- pre-implementation cross-model reviewer verdict recorded

## Implementation status

Implemented in `tensor4all-partitionedtt`:

- uniform `AdaptiveInterpolationResult<T>` and returned
  `AcceptedPatchCache<T>` values;
- patch-owned deduplicating scalar/batch evaluation cache;
- physical one-pass cache partitioning with moved, sibling-isolated maps;
- deterministic SplitMix64-derived per-path candidate and TCI seeds;
- sequential breadth-first waves and Hataori `LocalMode::Outer` Rayon waves;
- default `adaptive-hataori-rayon` and default-off
  `adaptive-hataori-mpi` Cargo features;
- private column-major TT-core MPI wire representation with checked shape
  products and payload lengths;
- collective validation, `pmap` waves, root-only reconstruction, fallible
  `WaveControl`, root-only result, and communicator abort on root panic;
- MPI f64 and Complex64 serialization plus a multi-rank smoke program.

The upstream-MPI build needed
`BINDGEN_EXTRA_CLANG_ARGS=-I/usr/lib/gcc/x86_64-linux-gnu/13/include` in this
checkout because clang did not find GCC's `stddef.h` automatically. This is an
environment workaround, not a source or CI requirement.

## Verification in progress

Passed so far:

- focused default release tests: 116 tests;
- focused MPI-feature release tests: 117 tests;
- default focused clippy;
- Hataori-absent feature graph and build with
  `--no-default-features --features tenferro-cpu-faer`;
- two-rank MPI smoke, including remote patch evaluation, accepted TT/cache
  return, completion-order determinism, Complex64 wire values, and collective
  invalid-input rejection;
- partitionedtt rustdoc tests.

Full workspace gates passed: workspace clippy, 2,740 nextest tests,
843 rustdoc tests, mdBook tests, rustdoc build, formatting, and repository-rules
dry run. The two-rank MPI smoke passed with exit status 0.

## Post-implementation reviewer gate

Selected reviewer: `reviewer-flash` (read-only, high thinking). It inspected the
full tracked and untracked worktree diff plus the pinned Hataori source.

Verdict: **Correct-to-merge**. No block or major findings.

Three minor findings were addressed before final validation:

1. The design now matches the implemented converged distributed-error category
   and diagnostic-message behavior.
2. The exact SplitMix64 path-seed formula is pinned in the design and source.
3. A direct callback-count test proves that a child hits its inherited cache but
   re-evaluates a sample retained only by its sibling.

## Origin-main synchronization

Before PR preparation, the branch was recreated from current `origin/main`
(`a15d30c57e86b478132de56137dd1ce930696bc7`). That base included the
`IdxTensor`/TreeTN ownership migration and the issue #598 adaptive-interpolation
regression. The implementation was integrated with the new owner types:
accepted simple TTs are converted through `tensor_train_to_treetn`, embedding
uses `TreeTN<IdxTensor, usize>`, fallible projector/subdomain construction is
preserved, `Option<usize>` bond caps are used, and the upstream issue #598 test
remains present. Focused, MPI, and full workspace gates were rerun after this
integration.

The synchronized full diff was re-reviewed by `reviewer-flash` (read-only,
high thinking). Verdict: **Correct-to-merge**, with no block or major findings.
Its one new minor defensive finding was fixed: MPI accepted-wire validation now
rejects incorrect core counts and per-core site dimensions before allocation or
TreeTN reconstruction, with direct malformed-wire coverage.

## Final verification

All required local gates passed after the final corrections and base sync:

- `cargo fmt --all -- --check` and `git diff --check`;
- `cargo clippy --workspace --all-targets -- -D warnings`;
- 135 default focused release tests;
- 136 MPI-feature focused release tests and MPI-feature clippy;
- Hataori-absent build and dependency graph with
  `--no-default-features --features tenferro-cpu-faer`;
- two-rank MPI smoke (`mpi_smoke_exit=0`), including remote evaluation,
  completion-order repeat, f64/Complex64 wire values, root-only result, and
  collective invalid-input failure;
- 3,195 workspace nextest tests (17 skipped by existing configuration);
- 936 workspace rustdoc tests;
- all mdBook tests;
- workspace rustdoc build (only pre-existing warnings in core and
  quanticstransform);
- repository-rules dry run and its 89 self-tests;
- public-surface grep found no stale bare adaptive result usage.

Hosted CI exposed two maintenance-gate issues and both were corrected:

1. its all-feature panic audit enabled `adaptive-hataori-mpi` on a runner
   without MPI, so the maintenance native-dependency step now installs OpenMPI
   development/runtime packages, matching the feature's documented build
   prerequisite;
2. the public-error-doc audit required concrete variants on
   `adaptiveinterpolate_mpi`, so its `# Errors` section now names placement,
   pmap, and converged distributed-error variants and their conditions.

Default lint, test, doctest, and coverage jobs do not enable the default-off MPI
feature.

Coverage impact attestation: the sequential queue implementation was replaced
by the common wave/patch processor, not simply deleted. Replacement tests cover
sequential, Rayon, MPI, exact 0/1-site paths, accepted/split/zero paths,
first/middle/last cache projection, inherited hit and sibling miss, batch
deduplication and error, wire validation, and collective failure. No coverage
threshold or numerical tolerance was relaxed.
