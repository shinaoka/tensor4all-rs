# SRC Reimplementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Rebuild TreeTN SRC from the paper's randomized QB, shared-prefix, adaptive-column, and incremental-QR mechanisms, then add a tested rooted-tree extension.

**Architecture:** The chain path is an independent paper-faithful implementation of Algorithm 1. The tree path uses one append-only global Gaussian probe bank and one complement-message pass per probe column, then performs bottom-up edge projections. Dense probe/stack and incremental Householder QR capabilities live below TreeTN in the established core/backend layers.

**Tech Stack:** Rust, tensor4all-core::TensorLike, tensor4all-tensorbackend::Matrix, tenferro-backed column-major dense operations, rand, release-mode Cargo tests, and the existing tensor4all benchmark repository.

**Spec:** docs/superpowers/specs/2026-08-26-src-reimplementation-design.md

## Global Constraints

- Source code and committed documentation are English; collaboration updates are Traditional Chinese.
- The paper-faithful chain path must preserve one seeded append-only Gaussian probe sequence and saved C prefixes.
- TreeTN is a declared rooted-tree extension, not an unqualified claim that the paper proves general trees.
- Dense flat buffers are column-major and every batch/reshape test must assert that convention.
- Use tenferro-backed tensor4all-tensorbackend operations for dense linear algebra; do not add direct tenferro dependencies to TreeTN.
- Production library code must not add unwrap or expect.
- Every implementation change starts with a failing release-mode test and ends with a passing focused test before the next change.
- Run cargo fmt --all before any local commit; do not push or open a PR without user approval.

---

### Task 1: Reset the prototype and establish module boundaries

**Files:**

- Modify: crates/tensor4all-treetn/src/treetn/contraction.rs
- Modify: crates/tensor4all-treetn/src/treetn/mod.rs
- Test: crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs

**Interfaces:**

- Consumes: existing ContractionMethod::Src, SrcOptions, and TreeTN dispatcher.
- Produces: private src_probe, src_qr, src_chain, and src_tree modules; no old src_* helper is called by the dispatcher.

- [ ] Step 1: Add a public-dispatch regression test for a two-node IdxTensor TreeTN. It must compare the materialized SRC result with the dense reference once and assert the requested topology and canonical center.
- [ ] Step 2: Run the test before deletion.

Run:

~~~
cargo test -p tensor4all-treetn --release src_dispatch_preserves_public_contract
~~~

Expected: the existing prototype passes or the test exposes its current behavior.

- [ ] Step 3: Delete the old one-hot random-vector helper, per-edge RNG reset, per-edge environment-column allocation, full-message-per-column path, old sketch assembler, and old contract_src_fixed body. Keep option validation and dispatcher signatures.
- [ ] Step 4: Add private module declarations and a replacement contract_src_impl seam returning a contextual unsupported error.
- [ ] Step 5: Run the same focused test and confirm the expected failure is the missing replacement implementation.

~~~
cargo test -p tensor4all-treetn --release src_dispatch_preserves_public_contract
~~~

- [ ] Step 6: Format and compile.

~~~
cargo fmt --all
cargo check -p tensor4all-treetn --release
~~~

- [ ] Step 7: Commit the reset.

~~~
git add docs/superpowers/specs/2026-08-26-src-reimplementation-design.md docs/superpowers/plans/2026-08-26-src-reimplementation.md crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/mod.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "refactor(treetn): reset SRC prototype for paper-faithful rewrite"
~~~

### Task 2: Implement the append-only Gaussian probe bank

**Files:**

- Create: crates/tensor4all-treetn/src/treetn/src_probe.rs
- Modify: crates/tensor4all-treetn/src/treetn/mod.rs
- Test: crates/tensor4all-treetn/src/treetn/src_probe.rs

**Interfaces:**

- Consumes: TensorLike, IndexLike, StdRng, and deterministic site ordering.
- Produces: private SrcProbeBank<T> with new(seed, sites), ensure_width(width), column(site, column), and test-only value snapshots.

- [ ] Step 1: Write a failing test proving that a bank seeded with 17 and extended from width 2 to width 5 preserves the first two columns exactly. Also assert one vector per site and the expected dimensions.

~~~
cargo test -p tensor4all-treetn --release probe_bank_appending_preserves_existing_columns
~~~

Expected: FAIL because SrcProbeBank does not exist.

- [ ] Step 2: Implement one StdRng in SrcProbeBank::new. Generate real standard Gaussian values in deterministic site order. Never reset the RNG in ensure_width.
- [ ] Step 3: Run probe tests.

~~~
cargo test -p tensor4all-treetn --release probe_bank
~~~

- [ ] Step 4: If TensorLike cannot construct a dense vector in one operation, add an optional construction capability with a documented unsupported default, implement the dense path for IdxTensor, and retain a tested generic fallback. The probe bank must use the dense path when available.
- [ ] Step 5: Run core and TreeTN tests.

~~~
cargo test -p tensor4all-core --release tensor_like
cargo test -p tensor4all-treetn --release probe_bank
~~~

- [ ] Step 6: Commit.

~~~
git add crates/tensor4all-core crates/tensor4all-treetn/src/treetn/src_probe.rs crates/tensor4all-treetn/src/treetn/mod.rs
git commit -m "feat(src): add append-only Gaussian probe bank"
~~~

### Task 3: Implement backend incremental Householder QR

**Files:**

- Create: crates/tensor4all-tensorbackend/src/incremental_qr.rs
- Modify: crates/tensor4all-tensorbackend/src/lib.rs
- Test: crates/tensor4all-tensorbackend/src/incremental_qr.rs

**Interfaces:**

- Consumes: column-major Matrix<T>, existing QR and triangular-solve backend seams, and supported scalar types.
- Produces: IncrementalQr<T> with new(matrix), append(columns), error_estimate(), norm_estimate(), and q().

- [ ] Step 1: Write failing f64 and Complex64 tests. Construct an m by p matrix with m >= p, append a second block, compare Q R with one full QR reconstruction, assert Q*Q is identity, and assert finite estimators. Add a rank-deficient rejection test.

~~~
cargo test -p tensor4all-tensorbackend --release incremental_qr
~~~

Expected: FAIL because IncrementalQr does not exist.

- [ ] Step 2: Implement compact Householder reflectors, applying old reflectors to appended columns, factorizing only the bottom residual block, and storing the inverse-adjoint triangular factor needed by Appendix C.
- [ ] Step 3: Implement error_estimate as the row-norm formula for R^(-dagger), norm_estimate as ||R||_F / sqrt(p), and conjugate-adjoint handling for complex scalars.
- [ ] Step 4: Run focused tests.

~~~
cargo test -p tensor4all-tensorbackend --release incremental_qr -- --nocapture
cargo test -p tensor4all-tensorbackend --release src_error_estimate
~~~

- [ ] Step 5: Commit.

~~~
git add crates/tensor4all-tensorbackend/src/incremental_qr.rs crates/tensor4all-tensorbackend/src/lib.rs
git commit -m "feat(tensorbackend): add incremental Householder QR"
~~~

### Task 4: Reimplement paper-faithful chain SRC

**Files:**

- Create: crates/tensor4all-treetn/src/treetn/src_chain.rs
- Modify: crates/tensor4all-treetn/src/treetn/contraction.rs
- Test: crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs

**Interfaces:**

- Consumes: SrcProbeBank, IncrementalQr, existing TreeTN bond/index helpers, and optional final SVD.
- Produces: contract_src_chain returning Result<TreeTN<T, V>> and implementing C prefix reuse plus the right-to-left Y, QR, eta, S sweep.

- [ ] Step 1: Add failing three-site full-width chain exact-recovery and probe-prefix-reuse tests. Materialize SRC and the dense reference once, subtract, and assert maxabs below 1e-10. Use a test-only counter for prefix construction.
- [ ] Step 2: Run the focused tests and confirm they fail because the new chain implementation is absent.

~~~
cargo test -p tensor4all-treetn --release src_chain_full_width_exact_recovery
cargo test -p tensor4all-treetn --release src_chain_reuses_probe_prefix
~~~

- [ ] Step 3: Implement deterministic chain order and C^(1) through C^(n-1). Each prefix column is created once and every later Y^(j) reads a prefix slice.
- [ ] Step 4: Implement Y^(n), economy QR, S^(n), then the descending Y^(j), QR, S^(j) sweep and the final eta^(1) contraction. Never materialize the exact uncompressed product.
- [ ] Step 5: Implement fixed width, paper oversampling, adaptive min_rank and rank_increment, and IncrementalQr append.
- [ ] Step 6: Run chain and existing contraction tests.

~~~
cargo test -p tensor4all-treetn --release src_chain
cargo test -p tensor4all-treetn --release contraction
~~~

- [ ] Step 7: Commit.

~~~
git add crates/tensor4all-treetn/src/treetn/src_chain.rs crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "feat(treetn): reimplement paper-faithful chain SRC"
~~~

### Task 5: Implement shared complement environments for TreeTN

**Files:**

- Create: crates/tensor4all-treetn/src/treetn/src_tree.rs
- Modify: crates/tensor4all-treetn/src/treetn/contraction.rs
- Test: crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs

**Interfaces:**

- Consumes: rooted postorder edges, SrcProbeBank, original operand tensors, and generic TensorLike contractions.
- Produces: SrcTreeEnvironmentCache<T, V> with ensure_width(width) and edge_column(child, parent, column); each new width performs one global tree message pass.

- [ ] Step 1: Add failing direct-cut tests for a three-node chain and a four-node branched tree. Compare each cached parent-to-child message against a direct contraction of the exact complement component and its probes.
- [ ] Step 2: Run the test and verify it fails because the shared cache is absent.

~~~
cargo test -p tensor4all-treetn --release src_tree_environment_matches_direct_cut
~~~

- [ ] Step 3: Implement the rooted edge schedule using edges_to_canonicalize_by_names(center).
- [ ] Step 4: For each new global probe column, form the factorized probes in deterministic node/index order, execute the original-tree postorder and reverse message passes, and retain only the parent-to-child message for every edge.
- [ ] Step 5: Keep the cache and probe bank outside the edge loop. Edge widths read prefixes; no edge-level RNG or full-tree recomputation is allowed.
- [ ] Step 6: Run real and complex chain/star/branched environment tests.

~~~
cargo test -p tensor4all-treetn --release src_tree_environment
~~~

- [ ] Step 7: Commit.

~~~
git add crates/tensor4all-treetn/src/treetn/src_tree.rs crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "feat(treetn): share SRC complement environments across tree cuts"
~~~

### Task 6: Complete bottom-up TreeTN SRC

**Files:**

- Modify: crates/tensor4all-treetn/src/treetn/src_tree.rs
- Modify: crates/tensor4all-treetn/src/treetn/contraction.rs
- Modify: crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs

**Interfaces:**

- Consumes: SrcTreeEnvironmentCache, IncrementalQr, local projected-tensor accumulation, and result topology builders.
- Produces: contract_src_tree with fixed/adaptive ranks, final_svd, real/complex values, scalar subtrees, and canonical center handling.

- [ ] Step 1: Add failing branched numerical tests for fixed no-final-SVD, fixed oversampled final-SVD, adaptive, and complex inputs. Compare the full materialized result with the dense reference once.
- [ ] Step 2: Run the focused test and confirm a meaningful numerical or missing-implementation failure.

~~~
cargo test -p tensor4all-treetn --release src_branched_tree_matches_dense_reference
~~~

- [ ] Step 3: For each child-to-parent edge, contract the original node tensors with already produced child projected tensors, contract the local tensor with cached complement columns, form the sketch, factorize, estimate, project Q*local, and append the projected tensor to the parent accumulator.
- [ ] Step 4: Assemble one result tensor per node, connect each fresh bond once, preserve dimension-one scalar bridges, set orthogonality toward center, and apply final SVD only when enabled.
- [ ] Step 5: Run all SRC correctness and itensorlike contract tests.

~~~
cargo test -p tensor4all-treetn --release src_
cargo test -p tensor4all-itensorlike --release contract
~~~

- [ ] Step 6: Commit.

~~~
git add crates/tensor4all-treetn/src/treetn/src_tree.rs crates/tensor4all-treetn/src/treetn/contraction.rs crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs
git commit -m "feat(treetn): complete rooted-tree SRC contraction"
~~~

### Task 7: Replace object-heavy probe and sketch paths with dense batches

**Files:**

- Modify: crates/tensor4all-core/src/tensor_like.rs
- Modify: crates/tensor4all-core/src/defaults/idx_tensor.rs
- Modify: crates/tensor4all-tensorbackend/src/matrix.rs
- Modify: crates/tensor4all-treetn/src/treetn/src_chain.rs
- Modify: crates/tensor4all-treetn/src/treetn/src_tree.rs
- Test: corresponding core, backend, and TreeTN test modules

**Interfaces:**

- Consumes: green generic correctness paths from Tasks 4 through 6.
- Produces: column-major dense probe and sketch batching for IdxTensor, with explicit fallback/error behavior for unsupported tensor types.

- [ ] Step 1: Add a failing non-square batch test. Assert that two sampled columns use row + rows * column ordering and that batched output equals the per-column reference.
- [ ] Step 2: Run the focused tests.

~~~
cargo test -p tensor4all-core --release tensor_shape_packing
cargo test -p tensor4all-treetn --release src_batched_sketch_matches_reference
~~~

- [ ] Step 3: Implement dense probe construction and stack with the batch index appended as axis -1. Preserve existing index order.
- [ ] Step 4: Use existing backend batched matrix multiplication when shapes match. Do not introduce row-major round trips or direct tenferro dependencies to TreeTN.
- [ ] Step 5: Run core, backend, and SRC focused tests.

~~~
cargo test -p tensor4all-core --release
cargo test -p tensor4all-tensorbackend --release
cargo test -p tensor4all-treetn --release src_
~~~

- [ ] Step 6: Commit.

~~~
git add crates/tensor4all-core crates/tensor4all-tensorbackend crates/tensor4all-treetn/src/treetn/src_chain.rs crates/tensor4all-treetn/src/treetn/src_tree.rs
git commit -m "perf(src): use column-major dense probe and sketch batches"
~~~

### Task 8: Verify, document, and benchmark

**Files:**

- Modify: docs/book/src/guides/mpo-mpo-contraction.md
- Modify: crates/tensor4all-treetn/README.md
- Modify: benchmark repository only after checking its current origin/main

**Interfaces:**

- Consumes: all green implementation tasks and current benchmark resources.
- Produces: evidence-backed correctness and speed report with no unverified SRC claims.

- [ ] Step 1: Fetch and inspect the latest benchmark state.

~~~
git -C /root/projects/tensor4all-rust/tensor4all-benchmark fetch origin
git -C /root/projects/tensor4all-rust/tensor4all-benchmark status --short --branch
git -C /root/projects/tensor4all-rust/tensor4all-benchmark log -1 --oneline origin/main
~~~

- [ ] Step 2: Run release correctness gates.

~~~
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
cargo test -p tensor4all-treetn --release
cargo nextest run --release --workspace --exclude tensor4all-hdf5
cargo test --doc --release --workspace -j 8
cargo doc --workspace --no-deps
~~~

- [ ] Step 3: Run the current benchmark resources with fixed, adaptive, chain, branched, real, and complex cases. Record warmups, repeated timed samples, CPU/thread settings, output ranks, dense errors, and phase-level environment/sketch/QR/projection counters.
- [ ] Step 4: Compare against result/full-src-20260826/report.md. Investigate regressions before changing code or inputs.
- [ ] Step 5: Update docs to state that SRC is alongside naive, fit, and zip-up, distinguish the paper chain algorithm from the TreeTN extension, and omit symmetry-preservation claims.
- [ ] Step 6: Run repository rule checks and inspect the final worktree.

~~~
python3 scripts/test-repository-rules-review.py
python3 scripts/repository-rules-review.py --base main --worktree --dry-run
git diff --check
git status --short
~~~

- [ ] Step 7: Report exact evidence and wait for explicit approval before pushing or opening a PR.
