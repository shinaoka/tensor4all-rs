> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Work Log — 2026-08-12 treetci branching-node candidate handling (gw-rs#8)

Session summary: root-caused the branching-topology slowdown reported in
[lingrui96/gw-rs#8](https://github.com/lingrui96/gw-rs/issues/8) (Rust
`tensor4all-treetci` up to 11x slower than the TreeTCI.jl reference at R=10
for comb/CTTN topologies with a 3-way junction, while chain topologies stay
faster than Julia), landed the fix in treetci, and verified it against the
g0 benchmark.

Code and documents read:

- `crates/tensor4all-treetci/src/proposer.rs` (candidate de-dup)
- `crates/tensor4all-treetci/src/update.rs` (candidate-matrix evaluation)
- `crates/tensor4all-tcicore/src/matrixlu.rs` (rrLU pivot kernel)
- `TreeTCI.jl/src/pivotcandidateproposer.jl`, `simpletci_tensors.jl`,
  `simpletci_optimize.jl` (Julia reference; same cartesian candidate
  generation, hash-based `union`, buffer-reusing `_call`)
- `TensorCrossInterpolation.jl/src/matrixlu.jl` (Julia rrLU; same
  complete-pivoting algorithm)
- `gw-rs/g0/examples/cttn_profile.rs` (instrumented benchmark harness)

Reference implementations considered: TreeTCI.jl / TCI.jl (behavioral
reference), TCI.jl's `:rook` pivoting (rejected: changes pivot selection and
Julia's own default path uses full pivoting).

Decisions made:

1. `union_with_history` de-duplicates through a `HashSet` instead of a linear
   `Vec::contains` scan, preserving first-occurrence order (mirrors
   TreeTCI.jl's `Base.union`). At a branching vertex the candidate count is
   the *product* of two incident bond dimensions (`d * chi_1 * chi_2`, up to
   2.9e5 at R=10) rather than `d * chi` (~1.2e3) on a chain, so the quadratic
   term dominated the whole optimization.
2. `update_edge` hands `evaluate_candidate_matrix`'s column-major buffer
   directly to `Matrix::from_col_major_vec` (no `Matrix::zeros` +
   per-element `matrix[[r, c]] = v` fill).
3. `evaluate_candidate_matrix` writes global points straight into one
   contiguous `(n_sites, n_points)` buffer (one allocation per edge update)
   instead of `assemble_global_point` (one `Vec` per matrix entry) +
   `assemble_points_column_major` (one pack copy per entry), mirroring
   TreeTCI.jl's `_call` buffer reuse. Bipartition and candidate-length
   validation happens once up front; the two sides are validated separately
   (a length-based side inference could let a malformed candidate pass and
   silently leave part of its point at zero).

Alternatives rejected or deferred:

- rrLU SIMD rewrite: rejected after measurement. The hot loops
  (`submatrix_argmax_col_major`, `update_trailing_submatrix`) are
  memory-bandwidth-bound (~36 GB/s effective, single-core bandwidth) and
  already autovectorized by LLVM (release binary contains ~1.8e5 vector
  instructions). A manual re/im rewrite cannot beat memory bandwidth;
  expect ~1.0–1.3x. The real lever would be parallelizing the scan/update
  (rayon), which is a separate decision (dependency + determinism care).
- Changing candidate *order* to Julia's `Iterators.product` order (first key
  fastest vs Rust's last key fastest): rejected. Order only affects
  tie-breaking in the argmax, not the candidate *set*; changing it risks
  shifting convergence behavior without a performance benefit. PR claim is
  set/size equivalence with Julia, not trajectory equivalence.
- `materialize.rs` (same per-point pattern in `fill_tensor_values`): deferred.
  One-time per run, measured ~1% of runtime.

Verification performed:

- `cargo test --release -p tensor4all-tcicore -p tensor4all-treetci`: 108 +
  41 + integration tests pass, including new tests for exact batch point
  order with unequal candidate counts, rejection of malformed candidate
  lengths (unequal partitions, so the length-coincides-with-other-side case
  is exercised), rejection of out-of-range candidate coordinates, and
  first-occurrence de-dup order with duplicates in both `values` and history.
- g0 benchmark (gw-rs, cttn/tt at R=8/9/10, T=0.01, mu=0.5, maxbonddim 2000,
  maxiter 50, tol 1e-4), release build, default faer backend, single thread:
  - cttn R=10 converges identically to the reported reference (max bond 412,
    final error 9.8e-5) — same results as before the change.
  - Wall time: cttn R=10 191.5 s here vs 2697 s in the issue (pre-fix code,
    O(n^2) de-dup); R=9 40.6 s vs 131.5 s in the issue; R=8 A/B on the old
    base: 6.4 s (fixed) vs 14.2 s (O(n^2) de-dup restored). Chain `tt` R=10
    control: 61.7 s, unchanged by the fix.
  - Breakdown at R=10 (fixed): proposer 1.0 s (0.5%), f-eval 40 s (21%),
    rrLU + assembly + materialize 151 s (79%). The residual is dominated by
    the bandwidth-bound rrLU on the product-sized junction candidate matrices
    and by per-point f-eval overhead in the g0 closure itself
    (`site_to_physical` allocations — gw-rs side, not treetci).
  - Re-run after the review-fix validation hardening (local-dims bounds
    check): R=8 unchanged at 6.6 s; built-in proposer output is always
    in-range so the added checks are no-ops on the real path.
- Note: pre/post measurements ran on different bases (pre-fix runs on
  `ae655a9` + uncommitted worktree; post-fix runs on `origin/main` rebased +
  this change). The R=8 A/B isolates the de-dup effect on one base.

Pre-PR checks (AGENTS.md CI-equivalent list): `cargo fmt --all -- --check`
clean; `cargo clippy --workspace --all-targets -- -D warnings` clean;
`cargo nextest run --release --workspace` green; `cargo doc --workspace
--no-deps` builds.

Remaining risks:

- The residual junction overhead (rrLU bandwidth + f-eval per-point work in
  downstream closures) is structural and still present; matching Julia's
  1.47x junction overhead would need parallel rrLU and/or allocation-free
  downstream evaluators (gw-rs side).
- Candidate order differs from Julia's product order; tie-break-driven pivot
  trajectories may differ between the two implementations even though
  candidate sets and final accuracy match.
