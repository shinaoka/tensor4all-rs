# simplett MPO truncation semantics, zip-up cost, and compression optimality

## Summary

Three defects in `crates/tensor4all-simplett/src/mpo/` were found while
building the public benchmark repository `tensor4all/tensor4all-benchmark`,
on its 2D quantics Gaussian mixture MPO times MPO case. They are independent
but they all show up on the same measurement, so they are fixed together.

1. `factorize_svd` compared raw singular values against `options.tolerance`,
   an unnormalized absolute cutoff, unlike every other truncation surface in
   the repository.
2. `contract_zipup` evaluated a three-operand tensor contraction as one
   eight-deep scalar loop, which costs two extra bond-dimension factors.
3. `compress_mpo` ran a truncating left-to-right sweep with no prior
   canonicalization, so each local SVD truncated against a non-orthonormal
   right environment and kept the wrong subspace.

## Code and documents read

- `REPOSITORY_RULES.md`, `AGENTS.md`, and the shared rules in
  `tensor4all-agent-rules` (`common/repository.md`, `rust/numerical.md`).
- `crates/tensor4all-simplett/src/mpo/`: `factorize.rs`, `contract_zipup.rs`,
  `contract_naive.rs`, `contract_fit.rs`, `contraction.rs`, `environment.rs`,
  `types.rs`, `mpo.rs`.
- `crates/tensor4all-simplett/src/compression.rs` and
  `crates/tensor4all-core/src/truncation.rs` plus
  `crates/tensor4all-core/src/defaults/svd.rs`, as the two existing
  definitions of what a truncation tolerance means here.
- `crates/tensor4all-tensorbackend/src/backend.rs` for `svd_backend` and
  `qr_backend`.

## Decisions

### Relative truncation is the house semantics

`CompressionOptions::tolerance` documents "Singular values (or pivots) smaller
than `tolerance * sigma_max` are discarded", and
`SvdTruncationPolicy::new` defaults to `ThresholdScale::Relative`, which
`compute_retained_rank` implements against the largest measured value.
`factorize_svd` was the outlier, so it was changed rather than the other two.
`factorize_lu` already forwards `tolerance` as `RrLUOptions::rel_tol` with
`abs_tol` at zero, so it was already relative and is left alone; a doc comment
now records that.

The zero-matrix case is handled explicitly instead of falling out of the
comparison: with `sigma_max == 0` a relative cutoff is zero, which would retain
the full spectrum of zeros, so the sweep is skipped and the existing rank-one
floor applies.

### Pairwise contraction in zip-up, not a single scalar loop

The zip-up step needs

```
C[n, s1, s2, ra, rb] = sum_{la, lb, k} R[n, la, lb] A[la, s1, k, ra] B[lb, k, s2, rb]
```

Evaluating that sum directly is `O(n * la * lb * s1 * s2 * k * ra * rb)`. Two
pairwise `einsum` calls through `tensor4all-tensorbackend` cost
`O(n * la * lb * s1 * k * ra)` plus `O(n * s1 * ra * lb * k * s2 * rb)`, which
drops two bond-dimension factors and routes the arithmetic through GEMM instead
of a bounds-checked scalar loop. On the benchmark instance at bond dimension
77 that is a 900x wall-time reduction.

The matrix reshapes around the local factorization are now plain column-major
reshapes of the tenferro tensor rather than index-by-index copies. The row and
column orderings differ from the previous hand-written ones by a permutation,
which is immaterial because the same convention is used to pack and unpack.

### Canonicalize before truncating

Alternatives considered for `compress_mpo`:

- Left-to-right QR pass, then a truncating right-to-left sweep. Rejected: on a
  naive MPO product both boundaries carry un-reduced Kronecker bonds, and the
  left-to-right QR then factorizes matrices of shape `(left * 4) x right` with
  `left` at its largest, which was measured to be the expensive direction.
- Right-to-left QR pass, then the existing truncating left-to-right sweep.
  Chosen: the QR pass shrinks each bond to its exact rank as it moves left, so
  every later factorization is smaller than the one before it.

`right_canonicalize` is rank preserving. It only replaces a bond by the exact
rank of its matricization, so it is safe to run ahead of any truncation policy
and cannot itself lose accuracy.

Canonicalizing the two inputs of `contract_zipup` was also prototyped, since
that is the textbook prescription for zip-up. It was measured and reverted:
the error moved from 2.5e-5 to 1.3e-5 on one instance and not at all on
another, which is inside the instance-to-instance spread of the benchmark, and
it costs two full MPO clones. Zip-up stays a heuristic whose accuracy matches
the treetn engine's zip-up exactly, which is the right reference point.

## Verification

`cargo nextest run --release --workspace`: 2520 passed.
`cargo test --doc --release --workspace`: passed.
`cargo clippy --workspace --all-targets -- -D warnings`: clean.

New tests, each verified to fail with its fix reverted:

- `factorize::tests::test_factorize_svd_truncation_is_scale_invariant`
- `contract_naive::tests::compression_at_a_binding_rank_cap_attains_the_optimal_truncation`
- `contract_zipup::tests::test_contract_zipup_untruncated_matches_naive`
  (guards the new index conventions rather than a fix)

End-to-end, `tensor4all/tensor4all-benchmark`'s `mpo_mpo_quantics` at
`BENCH_RS=6,8 BENCH_RUNS=1`, with the benchmark's tensor4all dependencies
pointed at this branch. The `zipup_treetn` and `fit_treetn` arms run the treetn
engine and are unaffected by this change; they are the control.

| r | arm | before | after |
|---|-----|--------|-------|
| 6 | naive | 0.620 s, 1.05e-4, chi 53 | 0.027 s, 8.64e-9, chi 48 |
| 6 | zipup_simplett | 2.528 s, 1.05e-4, chi 53 | 0.015 s, 1.05e-4, chi 53 |
| 6 | zipup_treetn | 0.019 s, 1.05e-4, chi 53 | 0.020 s, 1.05e-4, chi 53 |
| 6 | fit_treetn | 0.051 s, 8.64e-9, chi 48 | 0.039 s, 8.64e-9, chi 48 |
| 8 | naive | 25.584 s, 2.01e-5, chi 76 | 0.590 s, 1.71e-8, chi 59 |
| 8 | zipup_simplett | 103.075 s, 2.01e-5, chi 76 | 0.119 s, 2.28e-5, chi 77 |
| 8 | zipup_treetn | 0.142 s, 2.01e-5, chi 76 | 0.138 s, 2.28e-5, chi 77 |
| 8 | fit_treetn | 0.231 s, 1.71e-8, chi 59 | 0.226 s, 1.71e-8, chi 59 |

The input rank of the generated instance is 76 or 77 depending on the run, so
the r = 8 error figures move by a few percent between runs. Comparisons are
only meaningful within a row against the treetn control in the same run.

## Remaining risks

- `contract_zipup` still truncates against a non-orthonormal environment, so
  its accuracy remains that of a plain zip-up. It is now exactly on par with
  the treetn zip-up arm, and the variational `contract_fit` path is the answer
  for callers who need better. Improving zip-up itself is out of scope here.
- Callers who relied on `FactorizeOptions::tolerance` being an absolute cutoff
  on a large-norm operator will now see smaller ranks. That is the intended
  correction, but it is a behavior change on a public field.
