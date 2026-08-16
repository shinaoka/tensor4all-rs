# Benchmarks

This directory contains benchmark code for comparing Rust and Julia implementations.

## Structure

- `rust/`: shared Rust benchmark bodies. Prefer putting reusable benchmark
  source here, then include it from a thin crate example.
- `julia/`: Julia benchmark code using `ITensors.jl`, `ITensorMPS.jl`, or
  `AlternatingCrossInterpolation.jl`.
- `results/`: saved benchmark commands and representative local outputs.
- Crate-local `benches/`: Criterion microbenchmarks that are useful for Rust
  regression tracking but not meant as the cross-language source of truth.

Use ignored unit tests only when the benchmark needs crate-private state or
instrumentation. Once the required hooks can be exposed cleanly, move the
runner body into `benchmarks/rust/` and keep the crate-local entry point thin.

## Running Benchmarks

### Rust

```bash
cd crates/tensor4all-itensorlike
cargo run --release --example benchmark_contract
```

Projected local-operator apply:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_projected_apply --release -- 38 32 32 3 0
```

Prepared local linsolve:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_local_linsolve --release -- 38 32 32 1 10 30 0
```

Two-site TreeTN DMRG against a Pauli Heisenberg Hamiltonian
`sum_(i,j in E) X_i X_j + Y_i Y_j + Z_i Z_j` on chain and star topologies.
The benchmark compresses the summed MPO before timing, uses the repository
ITensors-compatible relative discarded squared singular-value tail cutoff
`1e-12`, and performs one untimed DMRG warm-up per topology. It reports runtime
and accuracy against a small dense exact reference. The star case uses a leaf
DMRG root in both Rust and Julia because ITensorNetworks.jl requires a leaf root
when converting an `OpSum` to a TTN. The Rust benchmark runs all requested
sweeps, without sweep-to-sweep energy early stopping:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_dmrg --release -- 8 4 3
```

Two-site TreeTN TDVP against the same Pauli Heisenberg Hamiltonian on chain and
star topologies. The benchmark evolves an alternating product state, uses
`order = 2`, `maxdim = 32`, ITensors-compatible relative discarded squared
singular-value tail cutoff `1e-12`, and a Hermitian Krylov exponential with
`krylovdim = 30` and `tol = 1e-12`. Both runners use the numeric cap `100`,
though Julia's KrylovKit `maxiter` and Rust's `max_time_splits` are different
control mechanisms. It performs one untimed TDVP warm-up per topology and
reports runtime plus L2 error against a small dense exact evolution reference:

```bash
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 cargo run -p tensor4all-treetn --example benchmark_tdvp --release -- 8 4 3 0.02
```

Append `all`, `chain`, or `star` to the Rust command to run both topologies or
only one topology in a fresh process.

Non-AD local tensor operations:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-core --example benchmark_tensor_ops --release -- 20000 6 2 2 6
```

TensorTrain-level operations against ITensorMPS:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-itensorlike --example benchmark_tt_ops --release -- --L 32 --zipup-L 10 --chis 4,8,16,32,64
```

PartitionedTT adaptive patching on a deterministic sum of randomly placed
anisotropic Gaussians. The benchmark compares the static sequential split order
against the exact child-parameter-gain ordering at the same global `rtol` and
`max_bond_dim`, and reports patch count, total TT core parameters, max patch
bond dimension, dense-reference relative error, and best elapsed time:

```bash
RAYON_NUM_THREADS=1 cargo run -p tensor4all-partitionedtt --example benchmark_patching --release -- --x 24 --y 16 --components 10 --max-bond-dim 3 --rtol 1e-8 --repeats 3
```

ACI elementwise TT chi scaling:

```bash
RAYON_NUM_THREADS=1 cargo bench -p tensor4all-aci --bench elementwise_scaling -- --sample-size 10
```

For Julia parity runs on macOS/Homebrew, build Rust against the same system
OpenBLAS backend instead of the default faer backend:

```bash
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
cargo bench -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  --bench elementwise_scaling -- --sample-size 10
```

ACI local-step bucket timing with system OpenBLAS:

```bash
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 T4A_STEP_TIMING_N_SITES=16 \
T4A_STEP_TIMING_FIXED_SWEEPS=3 \
T4A_STEP_TIMING_CHIS=16,32,64,128 \
cargo test --release -p tensor4all-aci \
  --no-default-features --features tenferro-system-blas \
  local_update_step_timing -- --ignored --nocapture
```

When running Rust doctests with `tenferro-system-blas`, also set
`RUSTDOCFLAGS="-L native=${OPENBLAS_ROOT}/lib"`. `RUSTFLAGS` is not enough for
rustdoc's final link step.

Optional long `chi = 32` case:

```bash
RAYON_NUM_THREADS=1 cargo bench -p tensor4all-aci --bench elementwise_scaling -- aci_elementwise_chi_scaling_long --sample-size 10
```

MatrixLUCI Hilbert step timing:

```bash
OPENBLAS_ROOT=${OPENBLAS_ROOT:-$(brew --prefix openblas)}
env \
RUSTFLAGS="-L native=${OPENBLAS_ROOT}/lib -l dylib=openblas" \
DYLD_LIBRARY_PATH="${OPENBLAS_ROOT}/lib:${DYLD_LIBRARY_PATH:-}" \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=20 T4A_MATRIX_LUCI_SIZES=16,32,64 \
cargo test --release -p tensor4all-core \
  --no-default-features --features tenferro-system-blas \
  matrix_luci_hilbert_timing -- --ignored --nocapture
```

MatrixLU standalone Hilbert timing:

```bash
env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LU_REPEATS=20 T4A_MATRIX_LU_SIZES=16,32,64,128 \
cargo run --release -p tensor4all-core --example benchmark_matrix_lu
```

MatrixLU itself does not call BLAS, so no system-BLAS feature is required for
this standalone runner.

Inspect Julia-dumped local linsolve inputs:

```bash
cargo run -p tensor4all-hdf5 --example inspect_mps_inputs --release -- benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5
```

### Julia

```bash
cd external/ITensorMPS.jl
julia benchmark/benchmark_contract.jl
```

Projected local-operator apply:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_projected_apply.jl 38 32 32 3 0
```

Prepared local linsolve:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_local_linsolve.jl 38 32 32 1 1 10
```

ITensorNetworks.jl two-site DMRG parity benchmark on matching chain and star
Pauli Heisenberg Hamiltonians:

```bash
export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench
julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations"]); Pkg.instantiate()'
BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
  benchmarks/julia/benchmark_dmrg_itensornetworks.jl 8 4 3
```

The temporary project keeps the local ITensorNetworks.jl checkout clean while
loading `TensorOperations.jl` so ITensorNetworks.jl can choose optimized
contraction sequences. The Julia runner performs one untimed DMRG warm-up per
topology before recording timings, so reported times exclude Julia compilation
and ITensorNetworks.jl first-call setup. Both runners use `cutoff = 1e-12` as a
relative discarded squared singular-value tail cutoff.

ITensorNetworks.jl two-site TDVP parity benchmark on matching chain and star
Pauli Heisenberg Hamiltonians:

```bash
export T4A_ITENSORNETWORKS_PATH=/home/shinaoka/tensor4all/ITensor/ITensorNetworks.jl
export T4A_ITN_BENCH_PROJECT=/tmp/t4a-itensornetworks-bench
julia --project=$T4A_ITN_BENCH_PROJECT -e 'using Pkg; Pkg.develop(path=ENV["T4A_ITENSORNETWORKS_PATH"]); Pkg.add(["Graphs", "ITensors", "TensorOperations", "KrylovKit"]); Pkg.instantiate()'
BLAS_NUM_THREADS=1 julia --project=$T4A_ITN_BENCH_PROJECT \
  benchmarks/julia/benchmark_tdvp_itensornetworks.jl 8 4 3 0.02
```

The Julia TDVP runner performs one untimed warm-up per topology, so reported
times exclude Julia compilation, ITensorNetworks.jl first-call setup, and
KrylovKit initialization. It uses `exponentiate_solver` with `ishermitian =
true`, `krylovdim = 30`, `tol = 1e-12`, and `maxiter = 100`. The Julia runner
asserts that the benchmark's selected leaf root matches ITensorNetworks.jl's
default TDVP sweep root for the chain and star graphs.

Non-AD local tensor operations:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tensor_ops.jl 20000 6 2 2 6
```

TensorTrain-level operations against tensor4all:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_tt_ops.jl --L 32 --zipup-L 10 --chis 4,8,16,32,64
```

ACI elementwise TT chi scaling:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/benchmark_aci_elementwise.jl --chis 2,4,8,16
```

ACI local-step bucket timing:

```bash
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_STEP_TIMING_REPEATS=50 \
julia benchmarks/julia/benchmark_aci_local_steps.jl --sites 16 --fixed-sweeps 3 --chis 16,32,64,128
```

MatrixLUCI Hilbert step timing:

```bash
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LUCI_REPEATS=20 \
julia benchmarks/julia/benchmark_matrix_luci.jl --sizes 16,32,64
```

MatrixLU standalone Hilbert timing:

```bash
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LU_REPEATS=20 \
julia benchmarks/julia/benchmark_matrix_lu.jl --sizes 16,32,64,128
```

Dump local linsolve inputs as ITensorMPS-compatible HDF5:

```bash
BLAS_NUM_THREADS=1 julia --project=benchmarks/julia benchmarks/julia/dump_local_linsolve_inputs.jl benchmarks/results/local_linsolve_inputs_N38_b32_o32.h5 38 32 32
```

## Benchmark Details

Both benchmarks perform MPO-MPO contraction using zip-up algorithm:
- Length: 10 sites
- Physical dimension: 2 (input and output per site)
- Bond dimension: 50
- Max rank: 50
- Includes orthogonalization time in measurements

The projected local-operator apply benchmarks isolate the local matvec hot path
used by two-site TreeTN/ITensor-style local solves. The Rust source of truth is
`benchmarks/rust/benchmark_projected_apply.rs`, included by the cargo example
target under `tensor4all-treetn`; the Julia counterpart is
`benchmarks/julia/benchmark_projected_apply.jl`.

The prepared local linsolve benchmarks construct the operator, right-hand side,
and initial state once, then time the local solve body. They also report local
GMRES/apply/RHS/factorization buckets. Use Julia `maxiter=1, krylovdim=10` as a
rough match to Rust's `krylov_maxiter=10` total-iteration cap; KrylovKit's
`maxiter=10, krylovdim=30` performs far more local operator applications.

The non-AD local tensor operation benchmarks isolate `inner`, `norm`, affine
addition, and explicit `conj`-then-contract on a small dense tensor. The default
shape `[6, 2, 2, 6]` mirrors the small two-site local tensors observed in the
QuanticsNEGF long-time local Krylov test, where dispatch/allocation overhead can
dominate floating-point work.

The TensorTrain-level operation benchmarks compare tensor4all's
`TensorTrain::inner`, strict direct-sum MPS addition, and prepared MPO×MPO zipup
contraction against ITensorMPS.jl. They use deterministic Complex64 fixtures and
print CSV-style rows with sample counts, min/median/mean/max milliseconds,
result max bond dimension, and a checksum. The Rust source of truth is
`benchmarks/rust/benchmark_tt_ops.rs`, included by
`tensor4all-itensorlike/examples/benchmark_tt_ops.rs`; the Julia counterpart is
`benchmarks/julia/benchmark_tt_ops.jl`.

The ACI elementwise benchmarks compare `tensor4all_aci::elementwise_batched`
with upstream `AlternatingCrossInterpolation.jl` on deterministic two-input TT
multiplication. The scaling axis is `chi`/`χ`, the maximum TT bond dimension of
the inputs and initial guess. The default smoke run covers `chi = 2, 4, 8, 16`.
`chi = 32` is a longer optional Rust case available through the
`aci_elementwise_chi_scaling_long` Criterion filter, and Julia can run the same
long case with `--chis 2,4,8,16,32`.

The ACI local-step benchmark isolates the local update buckets inside
elementwise ACI. Use `T4A_STEP_TIMING_N_SITES=16` and fixed sweeps when checking
`chi <= 128`; the old default `L = 12` has a central exact-rank bound of only
`64` for `local_dim = 2`, so it clamps `chi = 128`.

The MatrixLUCI Hilbert benchmarks isolate the matrix cross-interpolation step
used inside ACI local updates. They print CSV-style timing buckets for rrLU
pivot selection and factor construction on deterministic Hilbert matrices, with
both left- and right-orthogonal variants. Keep these benchmarks out of normal
test runs by using the ignored Rust test and the standalone Julia script.

The MatrixLU standalone Hilbert benchmarks isolate `rrlu_mut` and `rrlu`
without MatrixLUCI factor wrappers. The Rust source of truth is
`benchmarks/rust/benchmark_matrix_lu.rs`, included by
`tensor4all-core/examples/benchmark_matrix_lu.rs`; the Julia counterpart is
`benchmarks/julia/benchmark_matrix_lu.jl`.

`dump_local_linsolve_inputs.jl` writes the prepared local operator as
`operator_as_mps`, plus `rhs` and `init`, in one HDF5 file. The operator is a
Julia `MPO` stored through the `MPS` schema by saving its site tensors as
`MPS([H[i] for i in 1:length(H)])`; Rust reads all three groups with
`tensor4all_hdf5::load_mps`.
