# Backend-Native Incremental QR Refactor

## Summary

Commit `1dcd36d` replaces the adaptive SRC path's scalar, element-indexed
Householder implementation with a backend-native incremental update. The
public `IncrementalQr` API is unchanged. The state now stores explicit thin Q
and upper-trapezoidal R factors, applies two block Gram--Schmidt projection
passes through backend matrix multiplication, and factorizes only the appended
residual block through `qr_backend`.

The dependent-column policy is preserved. Full-rank appends stay blocked; a
residual block with a small QR diagonal falls back to ordered one-column
updates so dependent columns can be skipped without discarding later
independent columns. The Appendix C.3 block update of `R^{-dagger}` remains
incremental.

## Context and sources

- `docs/plans/2026-08-28-src-provenance-audit-report.md`, sections 1 and 12,
  plus the WS-backend derivations.
- `/root/projects/gw-rs/sgw/src/transform/ft.rs`: gw-rs constructs
  `SrcOptions::adaptive`, so this path is reachable whenever
  `SGW_APPLY_METHOD=src`; default gw-rs runs use Zip-up and do not invoke SRC.
- Camaño--Epperly--Tropp, arXiv:2504.06475, Appendix C.3, independently
  checked against `report.tex`.
- Existing tensor4all backend QR and matrix multiplication paths, and pinned
  tenferro QR implementations for CPU BLAS, faer, and CUDA.

No reference Python or C++ implementation was translated. The implementation
was derived from the paper's equations and validated against the pre-existing
Rust behavior and direct backend QR.

## Mathematical decision

For existing `Y = Q R` and appended columns `Y_new`, the update computes:

```text
B1 = Q^dagger Y_new
Z1 = Y_new - Q B1
B2 = Q^dagger Z1
Z2 = Z1 - Q B2
B  = B1 + B2
Z2 = Q_new R_new
```

and assembles:

```text
[Y Y_new] = [Q Q_new] [[R B], [0 R_new]].
```

The second projection is BCGS2 reorthogonalization. It addresses the paper's
warning that the displayed one-pass block Gram--Schmidt update is less stable
than the Householder update used by the reference implementation.

The paper's displayed `Z := Y - Q R'` is dimensionally inconsistent; the
following block factorization requires `Z := Y' - Q R'`.

## Alternatives rejected or deferred

- Recomputing QR on the entire accumulated matrix was rejected because it
  loses the adaptive algorithm's incremental efficiency.
- Keeping local scalar Householder reflectors was rejected because it
  duplicates LAPACK, prevents blocked backend execution, and was the audit's
  confirmed `HANDROLLED-DUPLICATE` finding.
- A true incremental Householder update is deferred to tenferro. Its current
  public surface does not expose compact reflector state, reflector
  application, or an append/update primitive.

## Tests and validation

The following passed after synchronizing the branch with `origin/main` at
merge commit `3ea2945`:

```text
cargo test --release -p tensor4all-tensorbackend incremental_qr -- --nocapture
  9 passed

cargo test --release -p tensor4all-treetn src_ -- --nocapture
  27 passed

cargo test --release -p tensor4all-tensorbackend
  208 unit tests and 138 doctests passed at the full-suite checkpoint

cargo test --release -p tensor4all-treetn
  483 unit tests/integration suites and 140 doctests passed

cargo test --doc --release -p tensor4all-tensorbackend
  148 doctests passed after the final rustdoc update

cargo clippy --release -p tensor4all-tensorbackend -p tensor4all-treetn \
  --all-targets -- -D warnings
  passed

cargo fmt -p tensor4all-tensorbackend -- --check
  passed

cargo run -p xtask --release -- api-dump
  complete API inventory verified

python3 scripts/check-public-error-docs.py \
  crates/tensor4all-tensorbackend/src/incremental_qr.rs
  passed
```

The repository-wide public-error-doc command still reports three pre-existing
findings in `crates/tensor4all-core/src/tensor_like.rs` at the
`contract_retaining_indices`, `from_dense_any`, and
`concatenate_along_new_index` APIs. This refactor does not touch those APIs.

New regression coverage includes:

- a differential comparison with direct `qr_backend` using reconstruction,
  orthogonal projectors, and both Appendix C error estimates;
- nearly dependent blocked columns and `Q^dagger Q` orthogonality;
- preservation of dependent-column rank saturation;
- complex Hermitian projection and inverse-adjoint behavior;
- a source-contract test forbidding the removed scalar reflector kernels.

## Paired benchmark

### Protocol

- Baseline: `86d3cbf` in a detached temporary worktree.
- Candidate: `1dcd36d`.
- Benchmark source: the same untracked
  `crates/tensor4all-treetn/examples/benchmark_src.rs` copied to both
  worktrees.
- Host: AMD Ryzen 9 6900HX, WSL2, Rust 1.98.0.
- Release profile; `RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
- Seed 1234, MPO--MPS mode, rank increment 3, final SVD disabled, 50
  repetitions per measured case.
- The exact-reference run was separate from timing and used the same seed.

### Results

| Max rank | Pair | Baseline adaptive, s/run | Candidate adaptive, s/run | Change |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 0.046153 | 0.045779 | -0.8% |
| 16 | 2 | 0.046311 | 0.047002 | +1.5% |
| 64 | 1 | 0.071713 | 0.070479 | -1.7% |
| 64 | 2 | 0.071624 | 0.070989 | -0.9% |

At max rank 16 the paired results straddle zero; no improvement is
demonstrated. At max rank 64 the candidate is consistently faster, averaging
about 1.3%, but the effect is small. Fixed SRC, which bypasses
`IncrementalQr`, stayed within the same narrow run-to-run range and served as
a control.

The exact-reference run produced adaptive SRC relative error `1.420e-18` for
both baseline and candidate.

Classification: the performance result is marginal. It confirms no material
adaptive SRC regression and suggests a small benefit at larger rank, but it
does not establish `incremental_qr.rs` as the cause of gw-rs's reported
slowness. The audit's F-4/F-5 tree-probe candidates remain higher-priority
profiling targets.

## Remaining limitations

- `IncrementalQr` still crosses the public host-backed `Matrix<T>` boundary.
  Its QR and products now use backend primitives, but this particular API does
  not retain device-resident tensors or an explicit CUDA execution context.
  The refactor removes scalar kernels and makes the algebra backend-owned; it
  does not by itself make the current SRC consumer end-to-end GPU-resident.
- BCGS2 is more stable than the paper's one-pass block Gram--Schmidt formula,
  but it is not the reference implementation's incremental Householder
  update.
- A tenferro follow-up should define a context/device-neutral incremental
  Householder or QR-append abstraction rather than exposing raw LAPACK calls
  downstream.
