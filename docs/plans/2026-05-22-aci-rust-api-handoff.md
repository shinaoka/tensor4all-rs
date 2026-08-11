# ACI Rust API Handoff

Date: 2026-05-22

Branch: `codex/aci-rust-api`

## Scope

This branch ports the core public Rust API for Alternating Cross Interpolation
workflows and aligns the local update path with the upstream Julia algorithm.
The current work focuses on the elementwise two-input TensorTrain case,
MatrixLUCI factor construction, batch evaluation, cache-aware local update
setup, and benchmark instrumentation against
`AlternatingCrossInterpolation.jl`.

No PR has been created from this branch.

## Main Changes

- Added `crates/tensor4all-aci/README.md` with upstream attribution and the
  Ritter paper citation.
- Added repository rules that forbid explicit pivot inverses, local dense
  linear algebra kernels, and copy-pasted scalar-specific implementations where
  a generic helper or macro is appropriate.
- Added system BLAS feature forwarding through relevant crates:
  `tenferro-system-blas`.
- Reworked ACI local update setup:
  - precomputes local input factors per input instead of repeatedly evaluating
    full local TT entries;
  - batches local setup GEMMs when shapes match;
  - batches full local input materialization with `batched_mat_mul_same_shape`;
  - batches frame updates;
  - keeps test-only benchmark gates:
    `T4A_ACI_DISABLE_BATCHED_LOCAL_SETUP=1`,
    `T4A_ACI_DISABLE_BATCHED_MATERIALIZE=1`, and
    `T4A_ACI_DISABLE_BATCHED_FRAME=1`.
- Added cached input-core matrix views in `ElementwiseProblem`, so repeated
  local setup does not rebuild the same dense core matrix views.
- Added `tensor4all-tensorbackend::Matrix` conversion APIs:
  - `into_col_major_vec`;
  - `to_typed_tensor`;
  - `into_typed_tensor`;
  - `try_from_typed_tensor`;
  - `MatrixTensorConversionError`.
- Added owned matrix linalg wrappers:
  - `mat_mul_owned`;
  - `solve_matrix_owned`;
  - `triangular_solve_matrix_owned`.
- Added backend triangular solve wrappers and tests.
- Reworked MatrixLUCI dense factor construction to follow the Julia rrLU-based
  solve formulation:
  - no explicit pivot inverse;
  - uses backend triangular solves;
  - uses owned Matrix-to-TypedTensor conversion where input buffers can be
    consumed.
- Added MatrixLUCI Hilbert timing benchmark and Julia counterpart.
- Added ACI local update step timing benchmark and Julia counterpart.

## Tenferro Notes

Do not assume tenferro changes are present. Another agent was working in a
separate tenferro worktree. This branch currently uses the pinned tenferro API
and its existing `From<TypedTensor<_>> for Tensor` conversions.

A possible future tenferro cleanup is to add a macro-generated
`TensorScalar::into_dynamic` / typed view helper inside tenferro itself, but
that change is not required for this branch.

## Benchmarks Run

The machine was heavily loaded during the last benchmark pass:

- load average was about `48`;
- many unrelated `solve_bse_afm_hf_only_quench` processes were active;
- `mpstat` showed a few mostly idle cores, but measurements still varied.

Treat the following numbers as handoff context, not final performance claims.

### ACI Local Step, `chi = 32`

Command:

```bash
taskset -c 58 env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_CHIS=32 \
cargo test --release -p tensor4all-aci local_update_step_timing -- --ignored --nocapture
```

Result:

```text
rust,32,50,3,33,2.022448,0.004660,0.005720,1.027935,0.986077,0.017761,1.147757,0.118297,3.225273,0.217013,1.325973,8.073430,26,6.172906e-11
```

Julia command:

```bash
taskset -c 58 env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 \
julia benchmarks/julia/benchmark_aci_local_steps.jl --chis 32
```

Result:

```text
julia,32,50,3,33,1.7984784999999999,0.0,0.0,0.896193,0.9031885,0.0,1.036934,0.085758,1.6588165,0.07977200000000001,1.076152,5.7409,26,6.172906150931444e-11
```

Earlier in the same session, without CPU pinning, Rust produced `4.457842 ms`
total for `chi=32`, while Julia produced `6.0671765 ms`. Because the machine
load changed during the session, rerun on a quieter host before drawing
conclusions.

### MatrixLUCI Hilbert, size 64

Rust command:

```bash
taskset -c 58 env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=100 T4A_MATRIX_LUCI_SIZES=64 \
cargo test --release -p tensor4all-tcicore matrix_luci_hilbert_timing -- --ignored --nocapture
```

Result:

```text
rust,hilbert,64,100,true,0.108844,0.000000,0.046796,0.045056,0.193050,13,9.601802e-12,2.000000e0
rust,hilbert,64,100,false,0.104603,0.000000,0.045361,0.040711,0.187775,13,9.601802e-12,2.000000e0
```

Julia command:

```bash
taskset -c 58 env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=100 \
julia benchmarks/julia/benchmark_matrix_luci.jl --sizes 64
```

Result:

```text
julia,hilbert,64,100,true,0.18505549999999998,0.0,0.015735,0.008965,0.212001,13,9.601801849524244e-12,2.0
julia,hilbert,64,100,false,0.174785,0.0,0.008660000000000001,0.0145305,0.198306,13,9.601801849524244e-12,2.0
```

MatrixLUCI selection is close under CPU pinning, but factor construction still
differs by implementation route. The total is close enough that a quieter host
is needed before further tuning.

### ACI Local Step, System OpenBLAS Rerun

On the macOS handoff host, Rust was rebuilt against Homebrew system OpenBLAS
instead of the default faer backend. The test binary was verified with
`otool -L` to link `/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib`.

Command:

```bash
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_CHIS=16,32 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  local_update_step_timing -- --ignored --nocapture
```

Result after owned-conversion and rrLU flat-slice optimizations:

```text
rust,16,50,3,33,0.117666,0.000835,0.001292,0.055043,0.053250,0.006645,0.080397,0.027372,0.323687,0.004709,0.063438,0.614538,25,8.805722e-11
rust,32,50,3,33,0.211792,0.000668,0.001084,0.073544,0.131126,0.005478,0.094503,0.025479,0.323209,0.003999,0.100208,0.759833,26,6.172906e-11
```

The corresponding Julia run from the same host earlier in the session was:

```text
julia,16,50,3,33,0.1879575,0.0,0.0,0.086867,0.0966605,0.0,0.111789,0.0087505,0.3064995,0.0111035,0.1248325,0.755751,25,8.805721625343378e-11
julia,32,50,3,33,0.2998115,0.0,0.0,0.0964205,0.19694300000000003,0.0,0.138312,0.0093775,0.3468935,0.0110005,0.1685645,0.9774875,26,6.172906150931444e-11
```

This makes the Rust/Julia comparison use system OpenBLAS on both sides.

For full Rust test runs that include doctests, add
`RUSTDOCFLAGS="-L native=${OPENBLAS_ROOT}/lib"` as well. `RUSTFLAGS` handles
normal test binaries, while rustdoc needs its own library search path.

### ACI Local Step, L=16, chi <= 128

The earlier `L = 12` fixture clamps `chi = 128` because the central exact-rank
bound is `2^6 = 64`. A new `L = 16` run avoids that clamp (`2^8 = 256`) and
uses `fixed_sweeps = 3` so Rust and Julia both perform 45 local updates.

Saved result:
`benchmarks/results/2026-05-22-aci-local-step-l16-openblas.md`.

Summary:

```text
impl,chi,n_sites,repeats,min_iters,fixed_sweeps,n_sweeps,n_updates,total_ms,matrix_luci_ms,final_rank,final_error
rust,16,16,20,2,3,3,45,1.841535,1.259810,33,9.525310e-11
julia,16,16,20,2,3,3,45,2.115942,1.233481,33,9.525310e-11
rust,32,16,20,2,3,3,45,3.330730,2.209271,46,9.720931e-11
julia,32,16,20,2,3,3,45,4.085233,2.414964,46,9.720948e-11
rust,64,16,20,2,3,3,45,7.975667,4.708000,63,9.320186e-11
julia,64,16,20,2,3,3,45,9.584417,5.213416,63,9.930868e-11
rust,128,16,20,2,3,3,45,15.889415,7.987332,76,8.523082e-11
julia,128,16,20,2,3,3,45,17.230575,8.276169,76,8.961109e-11
```

The Rust implementation now uses a Julia-shaped convergence helper in
`tensor4all-aci`: `iteration < min_iters` rejects, `errors[iteration] >
tolerance` rejects, and `any(last(ranks, min_iters) .>
ranks[iteration-min_iters+1])` rejects. Rust's optional `scale_tolerance` mode
is disabled in this benchmark. Fixed sweeps are used because small
numerical/path differences can still make the native stopping sweep differ.

### MatrixLU Standalone Timing

Added standalone MatrixLU Hilbert runners:

- `benchmarks/rust/benchmark_matrix_lu.rs`
- `crates/tensor4all-tcicore/examples/benchmark_matrix_lu.rs`
- `benchmarks/julia/benchmark_matrix_lu.jl`

Saved result:
`benchmarks/results/2026-05-22-matrix-lu-hilbert.md`.

MatrixLU itself does not call BLAS. The Rust runner currently uses the default
feature set because the `tensor4all-tcicore` dev-dependency on
`tensor4all-tensorci` still forces `tenferro-cpu-faer`, which conflicts with a
system-BLAS example build.

## TODO: Similar Clean Optimizations

- Split MatrixLUCI timing into `rrlu_mut`, LU extraction, triangular solve,
  permutation, and public factor packaging. The current `matrix_luci_ms` bucket
  is now close to Julia, but the sub-buckets will show whether more work belongs
  in tcicore or tenferro.
- Make `tensor4all-tcicore` benchmark/dev-dependency backend selection
  feature-clean so standalone examples can be built with
  `--no-default-features --features tenferro-system-blas` without pulling in a
  conflicting `tensor4all-tensorci/tenferro-cpu-faer`.
- Move ACI local-step timing out of the ignored unit test once a minimal
  feature-gated `bench_support` API can expose the crate-private timing hooks.
  Target layout: `benchmarks/rust/benchmark_aci_local_steps.rs` plus a thin
  `crates/tensor4all-aci/examples/benchmark_aci_local_steps.rs` wrapper.
- Add a MatrixLUCI-owned factorization path that constructs ACI-facing left and
  right factors directly from the factorized column-major buffer. This can avoid
  building intermediate full `RrLU` L/U matrices when callers only need
  `MatrixLuciFactors`.
- Audit remaining hot nested loops that use `Matrix[[row, col]]` after their
  ranges are already known. Good candidates are `matrix_luci.rs`
  `apply_row_permutation`, `apply_col_permutation`, `identity_rect`, and the
  post-solve copy loops.
- Keep unsafe indexing localized behind small column-major helpers, as in
  `matrixlu.rs`; do not spread raw `get_unchecked` through algorithm code.
- Check `matrixaca.rs`, `matrixluci/factors.rs`, and default
  `AbstractMatrixCI` helper paths for the same pattern: validate once, then use
  flat column-major slice iteration.
- Consider borrowed/owned variants for repeated submatrix extraction and
  permutation helpers so callers that just created a `Matrix` can avoid extra
  clones while keeping the public API clear.

## Verification Run Before Commit

The latest local verification before commit included:

```bash
cargo fmt --all -- --check
cargo test --release -p tensor4all-tensorbackend --lib
cargo test --release -p tensor4all-tcicore
cargo test --release -p tensor4all-aci
cargo test --doc --release -p tensor4all-tensorbackend
cargo test --doc --release -p tensor4all-aci
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
RUSTDOCFLAGS="-L native=${OPENBLAS_ROOT}/lib" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas
```

Observed results:

- `tensor4all-tensorbackend`: 140 lib tests passed; 123 doc tests passed.
- `tensor4all-tcicore`: 102 unit tests passed, 1 ignored; 97 doc tests passed.
- `tensor4all-aci`: 74 unit tests passed, 1 ignored; 4 integration tests
  passed; 15 doc tests passed.
- `tensor4all-aci` with `tenferro-system-blas`: 74 unit tests passed,
  1 ignored; 4 integration tests passed; 15 doc tests passed.

## Recommended Next Steps

1. Re-run the Rust and Julia benchmark commands above on a quieter host with
   the same CPU affinity and one-thread settings.
2. Compare `chi = 16, 32, 64` scaling for ACI local updates; `chi = 32` alone
   is not enough to diagnose scaling.
3. If Rust remains slower, inspect:
   - MatrixLUCI selection timing;
   - frame update timing;
   - `core_update_ms`, where Rust was much slower in the noisy pinned run;
   - whether tenferro-owned linalg wrappers can avoid more typed/dynamic
     boundary overhead.
4. Audit the repository for remaining hand-written dense linalg in feature
   crates, as requested earlier.
5. Decide whether the test-only ACI benchmark gates should remain in-tree or
   move to a benchmark-specific configuration path.
