# Resident SRC RRQR integration

## Summary

Replaces tensor4all's transitional non-pivoted resident QR diagonal guard with
tenferro's merged traced-compatible rank-revealing QR. All seven tenferro git
dependencies are pinned to merged commit
`0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.

## Code and documents reviewed

- `docs/design/context-scoped-src-contraction.md`
- tenferro `docs/design/rank-revealing-qr.md` and the merged RRQR public API
- `IdxTensor::factorize_probe_batch_incremental_in`,
  `resident_probe_batch_qr`, and `resident_src_error_estimate`
- `tensor4all-tensorbackend` Appendix-C estimator and matrix solve APIs
- chain/tree adaptive SRC factorization callers and CPU/CUDA integration tests

## Decisions

- Explicit contexts use upstream RRQR with
  `rtol = 32 * f64::EPSILON * max(rows, columns)` and `atol = 0`.
  This is scale-invariant and detects independent columns after interspersed
  dependence. The process-global CPU compatibility path remains unchanged.
- Only RRQR's scalar I64 rank is read through the caller-owned context.
  Q, R, permutation, and matrix payloads remain resident; permutation is never
  downloaded.
- Since tenferro defines `A[:, permutation] = Q R`, tensor4all restores its
  original-column reconstruction contract as `right = Q_rank^H A`, entirely in
  the owning runtime.
- The restored square factor is not generally triangular. A separate
  `src_error_estimate_general` uses the existing configured general matrix
  solve for host factors, while the CUDA estimator switches from triangular to
  general eager solve. The old triangular helper remains intact for
  `IncrementalQr` and its hot path and now explicitly rejects non-triangular
  input instead of silently applying triangular-solve semantics.

## Rejected alternatives

- Keeping the diagonal-prefix heuristic: fails for interspersed dependence.
- Downloading permutation metadata and reordering on CPU: violates residency
  and the rank-only readback contract.
- Returning pivoted sketch-column order: breaks `left * right` reconstruction
  semantics and CPU/CUDA reference parity.
- Fixed maximum rank plus masks: changes variable bond dimensions and is not
  required by RRQR's fixed output shapes.

## Verification performed

- Explicit CPU-faer and system-BLAS resident RRQR tests: real interspersed
  dependence, complex reconstruction, square estimator, and source contract.
- General estimator tests: real/complex column-restored factors plus empty,
  non-square, singular, and non-finite errors.
- CUDA hardware: complete context-factorization suite; real interspersed
  dependence; complex reconstruction and adaptive square estimator; complete
  fixed/adaptive chain/star/final-SVD SRC matrix; mixed/foreign rejection.
- CPU SRC-focused suite and context-seam tests.

## Review gate

- Design basis: `docs/design/context-scoped-src-contraction.md` plus tenferro's
  independently approved `docs/design/rank-revealing-qr.md`.
- Post-diff reviewer: `reviewer-flash` (DeepSeek family, read-only).
- Verdict: **Correct-to-merge**, no blocking findings. Follow-up review notes
  were fixed: legacy/global estimator routing was kept on the original
  triangular helper, complex CUDA estimator gained direct CPU numerical
  parity, and the triangular helper now rejects non-triangular factors. The
  final delta verdict was **APPROVE**, no blocking findings.

Full repository formatting, lint, doctest, mdBook, API inventory, repository
rules, coverage-impact, and CI/PR evidence are recorded before merge.
