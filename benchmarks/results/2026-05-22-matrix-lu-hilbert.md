# MatrixLU Hilbert Timing

Date: 2026-05-22

## Scope

This benchmark isolates rank-revealing LU (`rrlu_mut` and non-destructive
`rrlu`) on deterministic Hilbert matrices. It does not call BLAS, so the Rust
standalone runner uses the default feature set; system OpenBLAS is irrelevant
for this particular microbenchmark.

Thread settings:
`RAYON_NUM_THREADS=1`, `JULIA_NUM_THREADS=1`, `BLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.

## Commands

Rust:

```bash
env \
RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LU_REPEATS=20 T4A_MATRIX_LU_SIZES=16,32,64,128 \
cargo run --release -p tensor4all-tcicore --example benchmark_matrix_lu
```

Julia:

```bash
env \
JULIA_NUM_THREADS=1 BLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
T4A_MATRIX_LU_REPEATS=20 \
julia benchmarks/julia/benchmark_matrix_lu.jl --sizes 16,32,64,128
```

## Results

All timings are medians in milliseconds over 20 repeats.

| size | left orthogonal | Rust inplace | Julia inplace | Rust borrowed | Julia borrowed | rank | last error |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | true | 0.004042 | 0.002438 | 0.004125 | 0.002562 | 10 | 2.198484e-12 |
| 16 | false | 0.004041 | 0.002708 | 0.004084 | 0.002959 | 10 | 2.198484e-12 |
| 32 | true | 0.016188 | 0.011709 | 0.013104 | 0.011979 | 11 | 4.197675e-11 |
| 32 | false | 0.011917 | 0.013187 | 0.012021 | 0.013333 | 11 | 4.197675e-11 |
| 64 | true | 0.091688 | 0.062083 | 0.092771 | 0.063167 | 13 | 9.601802e-12 |
| 64 | false | 0.092000 | 0.062208 | 0.093105 | 0.064792 | 13 | 9.601802e-12 |
| 128 | true | 0.349042 | 0.288854 | 0.347458 | 0.315896 | 14 | 3.690140e-11 |
| 128 | false | 0.309084 | 0.287292 | 0.315875 | 0.287583 | 14 | 3.690140e-11 |

## Notes

Rust and Julia agree on rank and pivot error. On this Hilbert microbenchmark,
Julia's standalone rrLU is usually faster, especially at smaller sizes. This is
not identical to the ACI local MatrixLUCI bucket because the local matrices and
factor-wrapper work differ; use this benchmark to track rrLU-specific changes.

TODO: make the `tensor4all-tcicore` dev-dependency backend features easier to
select so examples can also be built with `--no-default-features --features
tenferro-system-blas` without pulling in `tensor4all-tensorci/tenferro-cpu-faer`.
