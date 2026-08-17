# Tree ACI bootstrap worklog

## Summary

Started the independent `tensor4all-treeaci` crate and implemented native
TreeTN preparation, recursive component samples, exact input frames, dense
edge-local LUCI, atomic output transactions, deterministic directional sweeps,
convergence, a floating-zone global guard, single-node direct evaluation, and
public elementwise plus simultaneous n-way Hadamard entry points.

## References read

- `AGENTS.md`, `README.md`, and `REPOSITORY_RULES.md`.
- Current shared common, Rust, performance, numerical, documentation, and
  provenance agent rules.
- Generated API references for `tensor4all-treetn` and `tensor4all-tcicore`.
- The existing train ACI batch/options/error contracts for semantic comparison;
  no train implementation is linked into or called by the new crate.
- The tree ACI derivation and implementation plan under
  `/Users/lingruicheng/treeaci/`.

## Decisions

- Keep tree ACI in a new crate depending downward on TreeTN and dense LUCI.
- Make TreeTN's train-conversion bridge an optional `simplett-bridge` feature.
  TreeTN keeps that feature by default, while treeaci disables default features
  and therefore has no simplett path in its Cargo dependency graph.
- Use the shortest continuous edge-covering walk. With an automatic root its
  length is `2|E| - diameter`; an explicit root fixes the start and chooses a
  farthest endpoint.
- Expose only the non-exhaustive strategy selector, initially containing
  `MinimumRetracingWalk`. Keep the unified `SweepPlan`, its validator, and planner
  implementation private until prepared TreeTN topology metadata consumes them;
  the temporary dead-code allowance is tied to this staged invariant.
- Do not publish a custom planner trait yet. Future built-in or user-defined
  planners must lower into the same validated plan rather than enter the
  executor through strategy-specific paths.
- Treat serial continuous walks as the reference. Discontinuous/parallel paths
  require sample-aware gauge transport or snapshot merging and remain deferred.
- Canonicalize node order by the full ordered node name and preserve each
  reference tensor's physical-axis order. Compare physical indices across
  inputs by full index equality, while deliberately permitting different input
  bond dimensions.
- Build both directed arcs and their incoming-branch lists in linear adjacency
  time. Reject invalid topology/options and conservative lower-bound allocation
  estimates before numerical callbacks are introduced.
- Keep recursive component samples immutable and separate them from replaceable
  active-ID sets. Global injection is proposal/commit: budget failure changes
  neither retained records nor the active generation.
- Key frames by `(input, directed edge, immutable sample ID)`. Materialize each
  input site core once in its own column-major tensor order, then contract exact
  recursive frames without a whole-TN dense conversion.
- Materialize only one budgeted local edge matrix, pass all input/point values to
  the callback in column-major `input + n_inputs * point` order, and factor via
  the tcicore owned dense LUCI entry point. A zero target uses the rank-one zero
  convention so later TreeTN materialization remains defined.
- Choose random-output ranks from the minimum input rank, algebraic cut bound,
  and configured cap. Explicit guesses retain their own valid ranks. Fresh
  output bonds preserve the input's full physical indices and tensor axis order;
  deterministic standard-normal cores are controlled by `rng_seed`.
- Bootstrap every directed component to its target rank in its own mixed-radix
  component space. Other directed sets may grow during projection and are then
  truncated deterministically, while immutable arena records remain retained.
- Never use generic TreeTN canonicalization to jump between live ACI paths. It
  changes bond gauges without remapping the active sample basis; a perfect
  binary-tree regression exposed an exact physical-coordinate permutation.
- Use the public `TreeTNCachedEvaluator` for guard batches. Persistent cross-scan
  cache state and statistics remain an upstream TreeTN API/performance gap.
- Grow global pivots atomically per undirected cut. A cut with headroom appends
  both directed active IDs even when one component projection is already
  represented, then zero-pads the output bond by the same amount. Cuts at the
  configured cap remain inactive while their recursive records may still be
  retained for nesting.

## Verification

- Baseline `cargo test --release -p tensor4all-treetn -p tensor4all-tcicore`:
  passed before edits.
- New crate: 52 unit tests, 7 public API/source-contract tests, and 18 doctests
  passed in release mode.
- Scheduler tests exhaust every labeled tree with 2 through 7 vertices and
  verify the `2|E| - diameter` optimum, explicit roots, exact reverse walks,
  train-path parity, Y, degree-four star, perfect binary, and comb cases.
- Validator corruption tests reject discontinuous walks and invalid reverse
  passes.
- Preparation tests cover path, Y, and degree-four trees; directed reverse and
  incoming-branch metadata; zero physical legs; multiple physical axes in
  tensor order; full-index mismatch; unequal input bond dimensions; empty and
  cyclic inputs; labeled topology mismatch; unknown roots; and core limits.
- Zero-dimensional physical indices cannot reach TreeACI through the public
  API because `IdxTensor` rejects them at construction; this upstream boundary
  is tested explicitly.
- Sample tests cover empty and duplicate seeds, every directed cut on path/Y/
  degree-four trees, projection of one global point to all cuts, retained older
  parent IDs, and arena-budget rejection.
- Exact-frame tests cover f64 and Complex64 on a two-node path and Y tree,
  zero-leg centers, two physical axes, and active-set replacement. Local-update
  tests check hand-derived input values, batched callback layout, callback error
  propagation, local matrix budgets, exact low-rank reconstruction, and the
  rank-one zero convention.
- State tests cover reproducible and seed-sensitive random initialization,
  algebraically capped ranks, active/frame consistency, initial-guess physical
  compatibility, rank and core budgets, preservation of the guessed dense
  tensor, and single-root canonicalization.
- `cargo tree -p tensor4all-treeaci --edges normal` contains none of the forbidden
  train-specific crates.
- The train ACI separated-two-peak regression was ported end to end: guard-off
  misses the remote feature while guard-on injects it and recovers both peaks
  within `1e-6`. Partial saturation tests prove capped cuts do not grow and an
  immediately following directional pass remains shape-consistent.
- Algebraic cut bounds are retained per edge and combined with the configured
  cap for guard injection, rank-limited termination, and saturation diagnostics.
- TreeTN's 405 library tests passed with its default bridge feature.
- Release checks passed for `tensor4all-quanticstci`,
  `tensor4all-partitionedtt`, and `tensor4all-quanticstransform`, which explicitly
  opt into the bridge.
- Strict clippy for the complete dependency traversal is currently blocked by
  45 pre-existing `clippy::nonminimal_bool` findings in `tensor4all-core` under
  Rust 1.92. The new crate passes strict clippy with `--no-deps`.
- `cargo doc --release -p tensor4all-treeaci --no-deps`, the public-error-doc
  audit, and the repository-rules worktree dry-run pass.
- Homebrew HDF5 2.2.0 is usable when the build is given
  `HDF5_DIR=/opt/homebrew/opt/hdf5`; `cargo check --release -p tensor4all-hdf5`
  and `cargo doc --workspace --no-deps` pass with that setting.
- Full `cargo test --release --workspace` reaches the test phase with that HDF5
  setting, then stops on nine existing `library-panic-audit` unit tests whose
  temporary-path expectations disagree with macOS `/private/var` canonical
  paths. No TreeACI test fails. `cargo-nextest` and the `mdbook` executable are
  not installed, so those two repository gates were not run or installed
  implicitly.

## Remaining risks

- Persistent cross-scan evaluator caching and path/train performance parity
  remain. The current public TreeTN cached evaluator reuses environments within
  one batch call but exposes neither a persistent cross-call message cache nor
  cache statistics; treeaci does not reach through that abstraction boundary.
- Current preparation checks exact known core sizes and conservative local
  matrix lower bounds. Rank-dependent candidate/frame estimates must be checked
  again at every generation once active pivot ranks exist.
- The path representation currently owns vectors per path. Before parallel
  execution and large-tree benchmarks, replace this with flat ranges if the
  measured metadata overhead is material.
