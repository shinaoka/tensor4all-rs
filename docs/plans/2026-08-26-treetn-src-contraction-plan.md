# TreeTN Successive Randomized Compression Implementation Plan

**Issue:** [tensor4all-rs #563](https://github.com/tensor4all/tensor4all-rs/issues/563)

**Goal:** Add successive randomized compression (SRC) as a fourth native
TreeTN contraction method, alongside naive, zip-up, and fit. The implementation
must support fixed-rank and adaptive-rank contraction, MPO-MPS application,
MPO-MPO contraction with factorized physical probes, and general tree
topologies.

**Primary reference:** C. Camaño, E. N. Epperly, and J. A. Tropp,
"Successive randomized compression: A randomized algorithm for the compressed
MPO-MPS product," Quantum 10, 2022 (2026),
[arXiv:2504.06475](https://arxiv.org/abs/2504.06475).

**Implementation policy:** Implement independently from the paper. The authors'
[RandomMPOMPS](https://github.com/chriscamano/RandomMPOMPS) repository may be
used to validate numerical behavior and parameter conventions, but its source
must not be translated line by line. The repository has no detected license and
contains no license file.

## Implementation status

The first implementation now lives in the TreeTN contraction path and exposes
`ContractionMethod::Src` with fixed-rank and adaptive-rank `SrcOptions`. It
uses two directed environment passes, factorized physical probes for MPO-MPO
products, optional final TreeTN SVD truncation, and CPU backend support for the
Appendix C estimator. Operator application, partial contraction, itensorlike,
the C API method enum, documentation, and the existing Gaussian benchmark
case are wired to the same TreeTN implementation.

The implementation intentionally keeps the report-bearing API proposed below
deferred: the ordinary contraction result remains a `TreeTN`, while benchmark
records and tests inspect the resulting topology, rank bounds, and numerical
residual. Per-edge diagnostics can be added after the algorithm and benchmark
interface have stabilized.

## Scope decisions

1. TreeTN is the owning implementation layer. Do not add an independent SRC
   implementation to `tensor4all-simplett`.
2. Add `ContractionMethod::Src` beside `Zipup`, `Fit`, and `Naive` in
   `tensor4all-treetn`.
3. Expose chain-oriented access through `tensor4all-itensorlike` after the
   TreeTN implementation is complete. SimpleTT compatibility, if needed, is an
   adapter through TreeTN rather than a second algorithm.
4. Implement MPO-MPO probing by attaching independent Gaussian vectors to each
   surviving physical output leg. Do not fuse two physical legs into a
   dimension-`d^2` production probe.
5. Preserve the input tree topology, including dimension-one links for
   scalar-only subtrees.
6. Use real standard-Gaussian probes embedded in the input scalar domain. This
   supports real and complex tensors and matches the reference experiments.
7. CPU correctness and performance come first. Do not advertise CUDA support
   until TreeTN QR, SVD, and contraction execution have a caller-owned CUDA
   context path.
8. Symmetry-preserving SRC is out of scope. Reject unsupported structured or
   block-sparse execution explicitly rather than silently densifying it.

## Existing seams to reuse

- `crates/tensor4all-treetn/src/treetn/contraction.rs`
  - topology validation and internal-index simulation;
  - deterministic leaf-to-center traversal;
  - TreeTN result construction and canonical metadata;
  - the generic contraction dispatcher.
- `crates/tensor4all-treetn/src/treetn/fit.rs`
  - directed-edge cache ownership pattern. The SRC messages are different and
    need a separate cache type.
- `crates/tensor4all-treetn/src/treetn/truncate.rs`
  - optional final SVD sweep on the already compressed output.
- `crates/tensor4all-treetn/src/operator/apply.rs`
  - operator/state index mapping and generic non-naive contraction route.
- `crates/tensor4all-treetn/src/treetn/partial_contraction.rs`
  - selected-index alignment for Hadamard and partial contractions.
- `crates/tensor4all-tensorbackend`
  - QR, matrix storage, and triangular solve ownership.

Do not reuse zip-up remainder tensors as SRC environments. Zip-up sees only the
already visited part of the network, while SRC requires a sketch of the entire
component on the opposite side of each cut.

## Mathematical design

### Rooting and exact product bonds

Choose the requested canonical center as the root. Replace the internal bond
indices of both operands with fresh IDs before any local contractions. For each
tree edge, the implicit exact product therefore carries one bond from the first
operand and one from the second operand.

For every local external index, classify it as one of:

- a contracted external index shared by the two operands;
- a surviving output index from the first operand;
- a surviving output index from the second operand.

Only surviving output indices receive random probes.

### Factorized product probes

For every surviving output index `i` and sketch column `k`, generate an
independent vector

```text
omega[i, k] ~ N(0, I_dim(i)).
```

The global column is the Khatri-Rao product of all local vectors. If an MPO-MPO
site has output indices `s` and `t`, its probe is

```text
Omega[s, t, k] = X[s, k] * Y[t, k].
```

The contraction order at an MPO-MPO site must be:

1. contract `conj(X[:, k])` into the first operand;
2. contract `conj(Y[:, k])` into the second operand;
3. contract the shared physical index between the operands;
4. contract incoming tree messages.

This avoids constructing either a fused random vector of length `d^2` or the
full local MPO product before probing.

### Directed sketch messages

For an oriented edge `u -> v`, remove the undirected edge `{u, v}` and let
`component(u | v)` denote the component containing `u`. Define
`E[u -> v, k]` as the contraction of:

- all local operand tensors in `component(u | v)`;
- all probes in that component for column `k`;
- all internal exact-product bonds in that component.

The message retains only the two exact product bonds crossing `{u, v}`.

Compute the messages with two deterministic passes:

1. postorder computes every child-to-parent message;
2. preorder computes every parent-to-child message using the already cached
   messages from the other neighbors.

The cache key is `(from_node, to_node)`. Each value owns columns indexed by the
global sketch-column number so adaptive expansion can append columns without
changing earlier random vectors.

### Successive QB compression

Process non-root nodes in postorder. Assume every child `c` of the current node
`v` has produced:

- an output isometry `Q[c]` stored as that child's result tensor;
- a projection cap `P[c]` connecting the new child bond to the two original
  operand bonds on edge `{c, v}`.

Form the effective local product `C_prime[v]` by contracting the two local
operand tensors with all child caps. It retains:

- all surviving output indices at `v`;
- one compressed bond for each child;
- the two exact operand bonds towards the parent.

For every active sketch column, form

```text
Y[v, k] = contract(C_prime[v], E[parent -> v, k]).
```

Stack the columns along a fresh sketch index. QR-factorize the matrix whose row
space is

```text
surviving output indices at v + compressed child bonds
```

and whose column space is the sketch index. Keep `Q[v]` as the result tensor and
form the cap

```text
P[v] = contract(conj(Q[v]), C_prime[v]).
```

At the root, contract the root's two operand tensors with all child caps. There
is no final QR at the root.

Every non-root result tensor is then isometric towards the root, so the result
can be marked as unitary canonical with the requested center.

### Chain reduction gate

For a chain rooted at its left endpoint:

- `E[parent -> child, k]` must equal the paper's forward environment column;
- the postorder compression runs from right to left;
- `P[v]` must equal the paper's projection cap;
- the root contraction must equal the paper's first-site completion.

Do not proceed to performance optimization until an internal chain test proves
these identities against a direct implementation of the paper equations on a
small deterministic input.

### Fixed-rank mode

For target rank `r`, use the paper's default oversampled sketch dimension

```text
p = max(ceil(1.5 * r), r + 10).
```

Cap `p` separately at each edge by the available row dimension and exact
product cut dimension. Run SRC at rank `p`, then reuse the existing TreeTN SVD
truncation sweep to reduce the output to rank `r`.

If final SVD is disabled, use the requested rank directly and return the QB
result without the oversampling-and-rounding step.

### Adaptive-rank mode

For a QR factorization `Y = Q R`, use Appendix C's estimator. Let

```text
G = R^(-adjoint).
```

Then compute

```text
error_estimate = sqrt(sum_i norm(row_i(G))^(-2) / p)
norm_estimate  = norm_frobenius(R) / sqrt(p).
```

Stop when

```text
error_estimate <= atol + rtol * norm_estimate
```

and the minimum rank has been reached. The defaults are minimum rank 2 and
rank increment 3. A finite maximum rank is mandatory.

For the robust adaptive preset, run the sketch at `0.1 * requested_rtol`, then
perform the existing SVD truncation using the requested tolerance.

The first correct implementation may recompute backend QR after appending each
block of columns, provided environment columns are cached and never recomputed.
Incremental Householder QR is a later optimization gate because neither
tensor4all nor the pinned tenferro API currently exposes it.

## Initial public API sketch

The following was the initial design sketch. The landed API uses a flatter
`SrcOptions` configuration because it composes naturally with the existing
`ContractionOptions` and `ApplyOptions` builders; the nested rank-selection and
report types remain future extensions rather than compatibility obligations.

Add the following types in the TreeTN contraction surface:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContractionMethod {
    Zipup,
    Fit,
    Naive,
    Src,
}

#[derive(Debug, Clone)]
pub struct SrcOptions {
    pub rank_selection: SrcRankSelection,
    pub seed: u64,
    pub final_svd: bool,
}

#[derive(Debug, Clone)]
pub enum SrcRankSelection {
    Fixed {
        target_rank: usize,
    },
    Adaptive {
        rtol: f64,
        atol: f64,
        min_rank: usize,
        rank_increment: usize,
        max_rank: usize,
    },
}
```

Expose constructors that produce valid configurations without requiring users
to coordinate unrelated fields manually:

```rust
ContractionOptions::src_fixed(target_rank)
ContractionOptions::src_adaptive(rtol, max_rank)
ApplyOptions::src_fixed(target_rank)
ApplyOptions::src_adaptive(rtol, max_rank)
```

The initial design also considered a report-bearing entry point for tests and
benchmarks:

```rust
pub struct SrcContractionResult<T, V> {
    pub network: TreeTN<T, V>,
    pub report: SrcContractionReport<V>,
}

pub struct SrcContractionReport<V> {
    pub edges: Vec<SrcEdgeReport<V>>,
    pub hit_rank_cap: bool,
}
```

Each edge report should include the selected sketch dimension, final output
bond dimension, last adaptive estimator when applicable, and whether the edge
stopped at its rank cap. Timing breakdowns should remain optional diagnostic
data rather than changing the ordinary contraction result.

All public types, fields, constructors, and functions require complete rustdoc,
runnable examples, assertions, and documented errors.

## Numerical and backend prerequisites

### Required first

Add an algorithm-neutral small-matrix helper owned by
`tensor4all-tensorbackend` that can compute the adaptive estimator from `R` for
f32, f64, Complex32, and Complex64. It must:

- preserve column-major layout;
- use the configured backend for triangular solve;
- implement the adjoint correctly for complex matrices;
- avoid explicitly forming a general matrix inverse when a triangular solve is
  sufficient;
- return a typed error for singular or malformed `R`.

The current triangular-solve API applies a plain transpose for complex inputs,
not a conjugate transpose. The helper must conjugate `R` explicitly before the
transpose solve, or the backend API must be extended to represent adjoint
solves directly.

### Profiling-dependent

The generic implementation can initially represent sketch columns as separate
tensors and assemble `Y` with one-hot sketch indices. Before accepting the
performance result, profile:

- random-vector construction;
- per-column contraction planning;
- sketch-column assembly;
- QR;
- cap projection;
- final SVD.

If column construction or assembly is material, add a reusable batch/stack
constructor at the tensor abstraction that owns index-labelled tensor
construction, with an optimized `IdxTensor` implementation. Do not add an
SRC-specific dense-buffer shortcut in `treetn`.

## File plan

### tensor4all-tensorbackend and tensor4all-core

- Modify `crates/tensor4all-tensorbackend/src/backend.rs` or add a focused
  small-QR helper module.
- Modify `crates/tensor4all-tensorbackend/src/lib.rs` for the public helper.
- Add backend tests for real and complex adaptive estimators.
- Modify tensor construction APIs only if profiling justifies a reusable stack
  or batched-column seam.

### tensor4all-treetn

- Create `crates/tensor4all-treetn/src/treetn/contraction_src.rs`.
- Modify `crates/tensor4all-treetn/src/treetn/mod.rs`.
- Modify `crates/tensor4all-treetn/src/treetn/contraction.rs`.
- Modify `crates/tensor4all-treetn/src/operator/apply.rs`.
- Modify `crates/tensor4all-treetn/src/algorithm.rs` to add `Src = 3`, or remove
  that duplicate public enum if its audit confirms it has no real consumer.
- Create `crates/tensor4all-treetn/src/treetn/contraction/tests/src.rs` and
  register it from the existing contraction test module.
- Extend operator apply and partial-contraction tests.
- Update `crates/tensor4all-treetn/src/lib.rs` and the crate README.

### tensor4all-itensorlike

- Modify `crates/tensor4all-itensorlike/src/options.rs`.
- Modify `crates/tensor4all-itensorlike/src/contract.rs`.
- Extend options and contraction tests.

### Documentation

- Update `docs/book/src/guides/mpo-mpo-contraction.md` with the SRC cost and
  tree generalization.
- Update `docs/book/src/guides/quantics.md` method-selection tables and examples.
- Update `docs/book/src/tutorials/gpu/cuda-tree-contraction.md` to state the
  current SRC placement limitation accurately.
- Check the root README and all public-surface descriptions for stale lists of
  contraction methods.

## Implementation tasks

### Task 1: Establish an isolated implementation branch

Create a worktree from current `origin/main`, not from the current
`optimize-treeaci-branched-hotpaths` branch.

```bash
git fetch origin
git worktree add ../tensor4all-rs-src -b feature/treetn-src origin/main
```

Generate the API inventory and read the relevant crate surfaces before edits:

```bash
cargo run -p xtask --release -- api-dump
```

### Task 2: Add backend estimator coverage and implementation

Write tests first for:

- a known real upper-triangular `R`;
- a complex `R` whose transpose and adjoint produce different answers;
- singular diagonal;
- non-square input;
- f32 and Complex32 promotion behavior if the helper promotes internally.

Implement the estimator through backend triangular solve and verify against a
small direct test oracle. Do not add a direct dependency on a tenferro crate
outside `tensor4all-tensorbackend`.

### Task 3: Define and validate SRC options

Add option tests covering:

- fixed rank zero;
- adaptive NaN, infinity, or negative tolerances;
- zero rank increment;
- zero maximum rank;
- minimum rank greater than maximum rank;
- deterministic default seed;
- final-SVD defaults;
- method dispatch and ignored-field prevention.

Invalid combinations must fail before single-node or zero-edge shortcuts.

### Task 4: Implement probe and index classification helpers

Test classification independently with:

- MPO-MPS indices;
- MPO-MPO indices;
- Hadamard diagonal pairing;
- multiple surviving indices on one node;
- same ID with different prime/tag metadata;
- scalar-output nodes.

Generate probes by stable node order, stable external-index order, and global
column number so the same seed is reproducible across runs.

### Task 5: Implement directed sketch-message construction

Start with a single fixed sketch dimension. Add tests for:

- two-node chain messages;
- three-node chain messages;
- a three-leaf branched tree;
- agreement of every directed message with a small dense component
  contraction;
- no mutation of input networks;
- deterministic cache expansion.

### Task 6: Implement fixed-rank successive QB

Implement leaf-to-root QR and projection caps. Cover:

- single node;
- one edge;
- long chain;
- branched tree;
- scalar-only subtree with topology preservation;
- exact low-rank recovery;
- capped approximate contraction;
- unitary canonical invariants towards the requested center.

Add the explicit chain-reduction test before enabling public dispatch.

### Task 7: Add factorized MPO-MPO regression coverage

Use a small test-only fused implementation as an oracle. Verify that the
production factorized path matches it for f64 and Complex64.

Add a structural regression that fails if production code contracts both
unprobed MPO site tensors into a fused `d^2` local product before attaching
the external probes.

### Task 8: Add adaptive expansion

Append sketch columns in deterministic blocks. Reuse all existing directed
messages and random vectors. At every expansion:

1. assemble the enlarged local sketch;
2. run backend QR;
3. evaluate the Appendix C estimator;
4. stop, expand, or report the hard cap.

Test early convergence, multiple expansions, exact-dimension termination,
rank-cap termination, and singular `R` handling.

### Task 9: Reuse final TreeTN truncation

Run final SVD only after the SRC output is fully assembled. Tests must verify:

- fixed oversampling returns the requested maximum bond dimension;
- adaptive oversampling meets the requested deterministic residual gate;
- disabling final SVD performs no truncation sweep;
- final SVD preserves the requested canonical center.

### Task 10: Wire generic contraction and operator application

Add `Src` to the dispatcher. Ensure `apply_linear_operator` routes SRC through
the generic contraction path and forwards all SRC options.

Test a nontrivial MPO-MPS apply against the local exact naive result after one
dense materialization. Include chain and branched operator/state topologies.

### Task 11: Wire selected contractions and itensorlike

Exercise SRC through:

- `partial_contract`;
- `hadamard`;
- itensorlike `TensorTrain::contract`;
- itensorlike MPO-MPO index ordering.

Do not add a SimpleTT implementation in this task.

### Task 12: Documentation and public-surface audit

Add runnable asserted examples for fixed and adaptive SRC. Document:

- rank and tolerance meanings;
- oversampling;
- deterministic seed behavior;
- memory cap behavior;
- the lack of symmetry preservation;
- current CPU-only limitation;
- the distinction from zip-up and fit.

Run API dump again and check all method lists for stale three-method claims.

## Test matrix

### Correctness

- f64 and Complex64.
- Single node, two-node chain, longer chain, Y-tree, and comb tree.
- MPO-MPS, MPO-MPO, Hadamard, scalar result, and multiple output legs per node.
- Exact recovery when every target rank reaches the exact cut rank.
- Dense whole-result residual for small approximate cases.
- Canonical/isometric edge invariants.

### Control flow and errors

- Every invalid option branch.
- Incompatible topologies.
- Empty networks.
- Unsupported tensor storage.
- Adaptive convergence before and after an increment.
- Rank cap reached with and without satisfying tolerance.
- Singular or rank-deficient sketch matrices.

### Reproducibility

- Same seed produces identical edge ranks and dense output.
- Adaptive expansion preserves the first `p` columns exactly.
- Different seeds meet the same residual acceptance gate without asserting
  elementwise equality.

### Test comparison policy

Materialize each whole result only once. Compare dense tensors using tensor
subtraction and `maxabs()` or a whole-tensor norm. Do not perform one complete
network contraction per output element.

Do not relax existing tolerances without explicit user approval.

## Benchmark plan

### In-repository focused benchmarks

Before optimizing, record a release, single-thread baseline for:

- chain length;
- input bond dimension;
- target/output bond dimension;
- physical dimension;
- branching degree.

Separate input construction from timed contraction. Record setup, sketch
environment, QR, projection, and final-SVD time outside or inside the main
metric as explicitly labelled submetrics.

### tensor4all-benchmark

Extend the existing `gaussian_mpo_contraction` case. Do not add a fourth
benchmark case. Add these arms:

- `global_zipup`;
- `global_fit`;
- `global_src_fixed`;
- `global_src_adaptive`;
- retain `patched_fit`.

Use the same cached inputs, seed, sampled finite-grid reference, and timing
boundary. Record:

- median wall time and individual runs;
- sampled relative-L2 error;
- input and output maximum bond dimensions;
- output parameter count;
- per-edge SRC sketch ranks;
- whether adaptive SRC reached its cap;
- peak intermediate elements or bytes when observable.

Every tensor4all-rs dependency pin in the benchmark repository must move to the
same revision. A pin update requires fresh measurements.

### gw-rs follow-up

Do not modify the currently dirty local gw-rs checkout as part of the initial
implementation. In a later downstream branch, add `src` to its apply-method
configuration and test:

- Fourier operator application on interleaved and nblock chains;
- Hadamard product on interleaved, nblock, and branched comb topologies;
- SRC with the existing explicit post-apply recompression;
- whether SRC reduces redundant raw-apply bonds enough to remove that extra
  sweep in a separately measured experiment.

gw-rs is a downstream realistic benchmark, not the primary correctness oracle.

## Performance acceptance gates

1. No full dense operand or output materialization in the production SRC path.
2. No fused dimension-`d^2` physical probe in MPO-MPO production code.
3. Fixed-rank hot path consists of tensor contractions, QR, projection, and an
   optional final SVD on the compressed output.
4. Cached environment columns are not recomputed during adaptive growth.
5. Measured chain scaling is consistent with the expected SRC regime before
   making performance claims.
6. Benchmark error is reported together with runtime. Faster but materially
   less accurate output is not accepted as a speedup.
7. Any batching or incremental-QR optimization is justified by a recorded
   profile and lives in the owning backend or tensor abstraction.

## Verification commands

Run focused release tests first:

```bash
cargo nextest run --release -p tensor4all-tensorbackend
cargo nextest run --release -p tensor4all-treetn contraction_src
cargo nextest run --release -p tensor4all-treetn operator::apply
cargo nextest run --release -p tensor4all-itensorlike contract
```

Then run the repository gates:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- \
  -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5
cargo test --profile ci -p tensor4all-hdf5
cargo test --doc --profile ci --workspace -j 8
cargo doc --workspace --no-deps
./scripts/test-mdbook.sh
python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run
python3 scripts/test-repository-rules-review.py
git diff --check
```

Before treating checks as final, fetch `origin` and verify the implementation
branch contains the current `origin/main`. Re-run affected checks after any
update from main.

## Completion criteria

- `ContractionMethod::Src` is available from TreeTN contraction options.
- Fixed and adaptive SRC pass chain and branched-tree correctness tests.
- MPO-MPS application and MPO-MPO factorized probing are both covered.
- The production MPO-MPO path never constructs a fused physical probe.
- The optional final SVD reuses TreeTN truncation.
- Adaptive rank growth terminates deterministically at tolerance, saturation,
  or the configured hard cap; per-edge reports are deferred as described in
  the implementation-status note above.
- itensorlike exposes SRC without a duplicate implementation.
- Public docs and runnable examples describe the real supported surface.
- tensor4all-benchmark contains reproducible zip-up, fit, and SRC comparisons
  in the existing Gaussian MPO-MPO case.
- The implementation is independently derived from the paper, with source
  provenance recorded and no unlicensed code copied from RandomMPOMPS.
- All required release tests, doctests, documentation builds, and repository
  rule checks pass.

## Explicit non-goals for the first implementation

- A separate SimpleTT SRC engine.
- Symmetry-preserving or block-sparse random sketches.
- CUDA execution or automatic CPU/GPU placement.
- Partitioned/patched SRC.
- Compressed sums of multiple MPO-MPS products from paper Section 3.6.
- A hand-written QR, SVD, matrix inverse, or LAPACK wrapper in `treetn`.

These can be proposed after the fixed and adaptive TreeTN implementation is
validated and benchmarked.
