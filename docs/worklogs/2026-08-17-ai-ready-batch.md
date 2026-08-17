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
- **#544:** direct-sum, block-tensor, rank-3 tensor, TreeTCI batch/materialization, ACI batch/frame, GSE fused-dimension, interpolation-core, and sweep-counter paths gained checked shape/stride/offset arithmetic and overflow regressions.

## Verification performed

- Focused release tests passed for all changed crates during iteration.
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc` passed.
- `python3 scripts/check-public-error-docs.py --changed-from origin/main` passed.
- `cargo test --doc --release --workspace` passed (861 doctests).
- `./scripts/test-mdbook.sh` passed.
- `cargo run -p xtask --release -- api-dump` passed and verified 14 public crate inventory files.

## Remaining audit work

The branch still needs a fresh issue-by-issue acceptance audit, the remaining cited raw-arithmetic sweep for every #544/#543 path, AD reverse-gradient/unsupported-path coverage for #545, and the final repository-rules/CI audit. The full repository rules script currently also traverses pre-existing `.worktrees/` material; changed-file error-doc validation passes. Full release nextest, coverage, formatting, clippy, API inventory, rustdoc (before the latest CachedFunction-only edits), and mdBook have passed. A subsequent core rustdoc rerun was blocked by repeated linker `SIGBUS` resource failures after 220 doctests. Do not mark the goal complete or close issues until those requirements have fresh evidence.
