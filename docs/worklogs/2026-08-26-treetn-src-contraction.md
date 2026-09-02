# TreeTN SRC implementation work log

## Scope

This work implements tensor4all-rs issue #563 as a TreeTN-native successive
randomized compression (SRC) contraction method. The implementation is placed
beside naive, zip-up, and fit in the TreeTN layer; SimpleTT does not receive a
second SRC engine.

## Inputs reviewed

- The issue discussion, including the follow-up direction to start from TreeTN.
- The SRC paper by Camaño, Epperly, and Tropp, including the factorized
  MPO-MPO probe construction and Appendix C stopping estimator.
- Existing TreeTN zip-up, fit, truncation, operator-apply, partial-contraction,
  canonicalization, and SimpleTT bridge code.
- tensor4all-benchmark's Gaussian MPO-MPO case and its current output records.
- The shared tensor4all agent rules, `README.md`, and `REPOSITORY_RULES.md`.

## Decisions and implementation

- `ContractionMethod::Src` and `SrcOptions` are public TreeTN APIs. `rtol =
  None` selects fixed rank; `Some(rtol)` selects adaptive growth with explicit
  maximum rank, minimum rank, increment, seed, and optional final SVD.
- The core SRC path does not run the optional final SVD by default. Callers
  must opt into the paper's oversampled final round with
  `with_final_svd(true)`; when a final tolerance policy is present, the sketch
  tolerance uses the paper's `0.1 * requested_tol` experiment convention.
- SRC roots at the requested canonical center, computes child-to-parent and
  parent-to-child directed environment messages, then performs leaf-to-root
  QR/projection caps. The result preserves the input topology and canonical
  metadata.
- MPO-MPO environments probe each surviving operand physical leg before the
  operand contraction. This retains the factorized Khatri-Rao probe and avoids
  constructing a fused random physical vector or an unprobed local physical
  product in the production SRC path.
- The Appendix C estimator is owned by `tensor4all-tensorbackend`; it builds
  the Hermitian adjoint explicitly for complex values. Incremental QR stores
  and updates `R^{-†}` with the Appendix C.3 block formula; the triangular
  solve is used only to initialize a state or handle a fallback. `IdxTensor`
  exposes this through the existing factorization abstraction. Singular
  adaptive sketches terminate safely and can be cleaned up by the optional
  final SVD.
- The MPO-MPO probe path now makes the paper's contraction order explicit: it
  partitions each local probe by operand, contracts X into A and Y into B,
  and only then contracts the shared physical/virtual legs. Batched probes
  retain one common sample axis through both operand contractions.
- The incremental QR backend now stores packed Householder reflectors and
  updates only the appended residual block. The packed state is carried
  through `FactorizeResult` as an internal core detail, so adaptive prefix
  growth resumes the existing reflectors instead of reconstructing them from
  Q/R factors.
- Adaptive prefix results reuse the already materialized Q columns and form
  only newly accepted Q columns after an append. The public Q/R fallback now
  resumes with `R_q R` (the R factor from QR of the supplied Q), preserving the
  represented product for externally constructed factorization results.
- Scalar-only branches retain the input TreeTN topology through explicit
  dimension-one bridge links, matching the topology-preservation contract.
- Operator application, partial contraction, itensorlike, and the C API method
  enum route through the same TreeTN implementation. CUDA and symmetry-aware
  SRC remain out of scope.
- The report-bearing API was deferred until diagnostics stabilize; ordinary
  contraction continues to return `TreeTN` and benchmark JSON records retain
  timing, error, and rank observations.

No source was copied from RandomMPOMPS. The implementation is independently
written from the paper; RandomMPOMPS was used only for conventions and
numerical cross-checking.

## Benchmark integration

The tensor4all-benchmark Gaussian MPO-MPO case now records global fit, global
zip-up, fixed SRC, adaptive SRC, and patched fit. While the tensor4all-rs SRC
implementation is uncommitted, the benchmark checkout uses sibling local path
dependencies and a `local-treetn-src` cache marker; after the implementation is
committed, those paths must be replaced with the resulting git revision before
sharing the benchmark checkout or running it in CI.

A smoke run with `BENCH_NS=2`, one run, and no warmups generated all five JSON
records. It demonstrated correctness and rank reporting, but it is not a
formal performance-gated baseline/candidate experiment; the timings should not
be used as a performance claim.

The complete benchmark profile `full-src-20260826` ran all maintained cases at
their default scales: six Fourier mode counts and Gaussian `N = 2, 8, 32, 128`.
With one warmup and three timed runs pinned to one AMD Ryzen 9 6900HX logical
CPU, it generated 60 records and passed the report validator. The largest
sampled error was `4.66e-6` overall and `2.11e-6` for the MPO-MPO case, below
the `1e-4` correctness gate.

For the MPO-MPO case, the paired global-fit speed ratios (fit time divided by
arm time) were:

| N | global zip-up | adaptive SRC | fixed SRC | patched fit |
|---:|---:|---:|---:|---:|
| 2 | 1.58x | 0.023x | 0.0053x | 0.54x |
| 8 | 1.50x | 0.023x | 0.0041x | 0.67x |
| 32 | 1.54x | 0.018x | 0.0030x | 0.74x |
| 128 | 1.60x | 0.025x | 0.0042x | 1.28x |

These are single-machine paired measurements, not a formal performance gate.
Those measurements predate the packed-state, factorized-probe, and incremental-Q
fixes. They are retained as historical diagnostics only and must not be used as
the current SRC speed result; the benchmark has to be rerun after this code is
committed.

The post-fix benchmark smoke attempt in this worktree reached dependency
compilation but did not enter measurement, so it produced no new timing data.

## Verification

Passed in the isolated `feature/treetn-src` worktree:

- `cargo test --release --workspace`
- `cargo test --doc --release --workspace`
- `cargo test --release -p tensor4all-capi --lib`
- `cargo test --release -p tensor4all-treetn --lib` (465 tests, including
  complex probes and scalar-only topology preservation)
- `cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc`
- `cargo doc --workspace --no-deps`
- `cargo run -p xtask --release -- api-dump`
- `./scripts/test-mdbook.sh`
- benchmark `cargo check --release --all-targets --offline --locked` and
  Gaussian MPO-MPO smoke run using the local SRC path dependencies
- `git diff --check`

Rust doctests, mdBook guide tests, and all other workspace checks pass.

## Remaining risks

- Results externally constructed through the public factorization constructor
  have no packed state; if such a result is supplied as an adaptive predecessor,
  the implementation falls back to one Q/R-to-Householder conversion. Native
  SRC adaptive results carry the packed state and do not use this fallback.
- Probe construction is generic and currently uses tensor one-hot/axpby
  assembly; a backend-native random-vector constructor may be worthwhile after
  a formal performance experiment.
- C API selection exposes the SRC enum but not adaptive option fields; it uses
  fixed defaults with the caller's maximum bond dimension. A future C API
  extension can add explicit SRC controls if the binding requires them.
