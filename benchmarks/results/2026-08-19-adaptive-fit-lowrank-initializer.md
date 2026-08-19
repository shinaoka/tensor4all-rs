# Adaptive low-rank FIT contraction vs zip-up / zip-up-init FIT (2026-08)

Single pinned CPU core (`RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`).

Command:

```sh
cargo run -p tensor4all-itensorlike --release \
  --example benchmark_contract_fit_adaptive -- 12 24 2 3
```

Setup: `A` is a generic random MPO with bond-24 content; `B` is the identity
operator padded to physical bond 24 (rank-1 content), so `A·B = A` needs
product rank equal to `A`'s bond while the naïve product bond would be
`χ_a·χ_b = 576`. `nsweeps=3`, SVD tolerance `1e-8`, `max_bond_dim = None`.

| method | time | maxbd (exact = 24) | rel. err |
|---|---|---|---|
| zipup exact | 50.2 ms | 24 | — |
| fit + zipup init (previous default) | 191.1 ms | 24 | 2.0e-8 |
| fit + low-rank init (new default) | 120.1 ms | 24 | 2.4e-8 |

Observations:

- The low-rank-initialized FIT is ~1.6× faster than the zip-up-initialized FIT
  at this size; the gap grows with bond dimension (χ=32: 227 ms vs 399 ms;
  χ=48: 1.32 s vs 2.43 s, ~1.8×).
- Both fit paths converge to the same accuracy; the zip-up-initialized path
  spends the difference inside the initializer, which materializes product-bond
  intermediates of dimension up to `χ_a·χ_b` before the sweeps run.
- The low-rank path starts with every bond at dimension 1 and grows them
  adaptively per the tolerance during the sweeps — no `χ_a·χ_b` intermediate is
  ever formed, and no `max_bond_dim` is required.
