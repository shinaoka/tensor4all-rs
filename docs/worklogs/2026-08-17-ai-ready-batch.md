# AI-ready issue batch work log

## Session summary

Implemented a first integrated batch for the open AI-ready audit issues on `ai-ready-batch`, based on `origin/main` plus the preserved local documentation commit. The branch keeps the work in one reviewable stream so it can be split into the fewest coherent PRs after final audit.

## Decisions and changes

- **#637:** generated API output is no longer tracked. `cargo run -p xtask --release -- api-dump` clears and regenerates `target/api-dump/`, derives the expected public crate set from `cargo metadata`, rejects missing/extra/duplicate artifacts, and is run in CI. `AGENTS.md` and the usage skill now describe the same policy.
- **#550:** all native einsum owned, borrowed, and read paths share checked `usize` → `u32` label conversion; oversized input/output labels return errors before profiling or backend calls.
- **#633:** removed inert `TruncateOptions::site_range` and Qtci global-search fields/builders. Explicit array interpolation validates finite strictly increasing coordinates, preserves nonuniform interior coordinates through the discrete grid path, and retains the continuous discretized path for uniform arrays.
- **#634:** `Projector` construction/insertion is fallible and coordinate-checked; `SubDomainTT::new` rejects absent indices; partition insertion checks overlap transactionally; deterministic projector ordering/hash use full index metadata; unrestricted mutable data access was removed/restricted.
- **#548:** removed placeholder merged-bond metadata, rejected ambiguous multi-site automatic MPO mapping, corrected full-index `TensorIndex` documentation, and added a shared TreeTN linear-chain validator used by `TensorTrain::from_treetn`.
- **#547:** index replacement stages all changes before committing, raw public site-space mutation was restricted, and local update sweeps commit only after the full staged sweep succeeds.
- **#546:** global pivot search now validates pivots and propagates TT evaluation errors; TensorCI1 no longer converts evaluation failures to zero; batched quantics callbacks validate component lengths before indexing; matrix LU and MatrixACA helpers validate shape/pivot/mutation preconditions transactionally; itensorlike linsolve validates operand lengths; `CachedFunction` validates rank/ranges and batch callback lengths before cache mutation.
- **#543:** HDF5 schema versions reject negatives, storage types are exact, ITensor payload shape/length is checked before reading, and TreeTN C API pointer spans are checked before pointer arithmetic/raw slices.
- **#544:** direct-sum, block-tensor, rank-3 tensor, checked Matrix constructors, TreeTCI batch/materialization, ACI batch/frame, GSE fused-dimension, interpolation-core, and sweep-counter paths gained checked shape/stride/offset arithmetic and overflow regressions.
- **#545:** scalar/structured AD paths retain backend graphs where supported; LU/CI factorization now rejects tracked inputs before materialization with a typed unsupported-storage error, and the AD regression suite records the preserve-or-error contract.

## Verification performed

- Focused release tests passed for all changed crates during iteration.
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` passed.
- `python3 scripts/check-public-error-docs.py --changed-from origin/main` passed.
- `cargo test --doc --release --workspace` passed (861 doctests).
- `./scripts/test-mdbook.sh` passed.
- `cargo run -p xtask --release -- api-dump` passed and verified 14 public crate inventory files.

## Follow-up #544 slices

After the initial batch, the branch additionally checked:

- core factorization matrix lengths and Krylov capacities/projected matrices, with typed invalid-option errors and overflow regressions;
- ACI frame products, padded-core dimensions, local batch sizes, flat offsets, and reshape boundaries;
- itensorlike dense inner-product/norm products and saturating `LinsolveOptions` sweep construction;
- interpolative QTT basis, fused-core, dense-test-point, flat-index, inverse-matrix, and adaptive-core dimensions;
- TensorCI1 candidate matrices, Pi-set capacities, refresh propagation, conversion reshapes, and site dimensions;
- SimpleTT cache site-dimension products and a new fallible `try_tensor3_zeros` constructor for external shape input.

Current focused release evidence: tensor4all-core 789 tests (2 skipped), tensor4all-aci 90 tests (1 skipped), tensor4all-itensorlike 154 tests (5 skipped), tensor4all-interpolativeqtt 27 tests, tensor4all-simplett 252 tests, tensor4all-tensorci 79 tests; all passed. Strict workspace clippy also passed.

A stale guide callback was corrected (`CachedFunction::eval` now unwraps its documented valid-input result), and a probabilistically flaky interior-coordinate test now uses a nonzero-at-origin affine function while retaining its coordinate assertions.

## Final synchronized verification

The final branch is `ai-ready-batch` at `a180742` (ahead of `origin/main` at `2b11f7f`), with a clean worktree. The issue acceptance map is:

- **#543:** HDF5 schema/storage validation and TreeTN C API span checks reject invalid input before access.
- **#544:** checked arithmetic and overflow regressions cover core factorization/Krylov, ACI, TensorCI/TreeTCI, SimpleTT, interpolative QTT, tensor backend, block/direct-sum, and materialization paths; the final TreeTCI batch helper and SimpleTT fallible-shape tests are in commits `5e86dea` and `193c217`.
- **#545:** AD-preserving paths retain backend graphs where supported and reject unsupported tracked LU/CI materialization with typed errors.
- **#546:** pivot, callback, factorization, cache, matrix, linsolve, and batched-evaluation failures are validated and propagated transactionally.
- **#547/#548:** staged mutation, TreeTN topology/index invariants, canonical metadata, and public conversion contracts are covered by regression tests and docs.
- **#550:** einsum label conversion is checked before profiling/backend calls.
- **#633/#634:** removed inert options and hardened nonuniform interpolation, projector ordering/validation, and transactional partition insertion.
- **#637:** API inventory generation/verification is deterministic, CI-backed, and currently verifies all 14 public crate artifacts.

Final validation passed: release workspace nextest (**2,875 passed, 14 skipped**), release rustdoc (**862 passed**), coverage (**215/215 files**), strict workspace clippy, `cargo doc --workspace --no-deps`, public error-doc validation, API inventory, mdBook tests, crate-boundary audit, library panic audit, repository-rules review, formatting, and `git diff --check`. The doc build retains only the repository's existing rustdoc warnings.
