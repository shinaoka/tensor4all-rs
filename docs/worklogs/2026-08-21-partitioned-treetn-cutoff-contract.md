# Corrective follow-up: local discarded-weight `cutoff` contract (#655)

Work log for the corrective PR that implements the authoritative maintainer
decisions in issue
[#655](https://github.com/tensor4all/tensor4all-rs/issues/655). The merged #661
PR implemented several superseded remedies (per-bond budget split, claimed
global `rtol` guarantee) and left reproducible boundary/validation bugs. This
PR replaces those remedies with the final maintainer contract and fixes the
remaining boundary bugs.

Supersedes the per-bond budget split and global-`rtol` claims recorded in
`2026-08-20-partitioned-treetn-followups.md` (banner added there). The
authoritative design record is `docs/design/partitioned-treetn.md`.

## Maintainer decisions implemented (issue #655)

1. Scalar partition truncation surface is local discarded-weight `cutoff`; the
   final whole-network norm error is best effort, never advertised as bounded.
2. Adaptive patching uses volume-proportional absolute local cutoffs
   `local_cutoff_p = cutoff * ||F||^2 * volume_p / total_volume`, applied
   whole (not per-edge) at every local SVD.
3. `max_bond_dim` is a hard cap taking precedence over `cutoff`.
4. `default().cutoff = 1e-24` — maintainer-confirmed behavior-parity
   translation of the superseded `rtol = 1e-12` (`cutoff = old_rtol^2`).
5. Same-identity/different-dimension indices are rejected at the public
   boundary with `SiteIndexMismatch`; masking/splitting use the canonical site
   index; deprecated `partitionedtt` gets only the same-identity/dimension
   correctness fix.
6. Zero-node subdomains are rejected with typed `Empty`; the empty partition
   collection keeps its documented semantics.
7. Shared core validator `validate_svd_truncation_options(max_bond_dim,
   policy)` root-guards direct SVD and is reused by TreeTN/partition/adaptive
   entry validators before every shortcut; C API `maxdim == 0 -> None` sentinel
   unchanged.
8. `add_with_patching` sums equal-key inputs; internal result construction
   collects/groups/sorts duplicate contributions, exact-adds each group,
   truncates once, and validates the whole partition once.
9. Removed the stale direct `tensor4all-tensorbackend` dependency from the
   design record (the Cargo.toml entry was already dropped in #661).

## Reproducer

`/tmp/pr661-repro` against the exact #661 commit confirmed three contract
failures before this PR: (a) dim-3 projector alias on a dim-2 site reaches
`mask_index` and returns a lower-level label-shape mismatch instead of
`SiteIndexMismatch`; (b) `TreeTN::truncate([], NaN policy)` returns `Ok`
(validation after the empty shortcut); (c) a two-node diagonal state with
squared weights `[1, 0.006, 0.006]` at `rtol = 0.1` shows aggregate relative
error `0.1089`, proving the #661 global-`rtol` guarantee false.

## Design gate

Corrected `docs/design/partitioned-treetn.md` (Adaptive patching truncation
convention, Dependencies, Review record) and added a supersession banner to
the old worklog before any source edit.

Read-only cross-model design review (`reviewer` agent on
`deepseek-v4-flash-284b:max`, fork context) returned **APPROVE** with no
blocking findings. Residual risk noted: the `1e-24` default was separately
confirmed by the maintainer before implementation.

## Verification

- Reproducer `/tmp/pr661-repro` now returns `SiteIndexMismatch` for the
  dim-3 projector alias, rejects the NaN empty-center truncate, and the
  two-node diagonal example trims to rank one under the whole local cutoff
  (best effort, no global-bound assertion).
- `cargo test --release --workspace` (excluding the HDF5-only members
  `tensor4all-hdf5` and `book-tests`, which cannot build in this sandbox
  because `hdf5-metno-sys` needs a native parallel-HDF5/MPI/OpenSSL
  toolchain): all pass, including the rewritten local-cutoff, known-spectrum,
  validations-table, empty-semantics, dimension-mismatch, and
  order-independence tests.
- Focused release tests for `tensor4all-core` (SVD root guard),
  `tensor4all-treetn` (empty-center/dispatch validation), and both partition
  crates pass.
- Release doctests for the four changed crates pass; full workspace release
  doctests pass.
- Default CPU and `--no-default-features --features tenferro-provider-inject`
  builds pass for `tensor4all-partitionedtreetn` and `tensor4all-treetn`.
- `cargo fmt --all -- --check` clean.
- `cargo clippy --workspace --all-targets -- -D warnings`: all violations are
  the pre-existing `nonminimal_bool` lints in the untouched
  `core/src/defaults/idx_tensor.rs` (present on base `main` @ `646ee3a` under
  the local stable 1.96 clippy); none originate from this diff. CI clippy is
  authoritative.
- `cargo run -p xtask --release -- api-dump` verifies every crate appears once;
  the partition API inventory shows `truncate_adaptive(..., cutoff, ...)` and
  core shows `validate_svd_truncation_options`.
- `scripts/check-public-error-docs.py`: ok. `scripts/repository-rules-review.py
  --base main --worktree --dry-run`: pass, no findings; its 90 self-tests
  pass.
- mdBook gate (`./scripts/test-mdbook.sh`): cannot run in this sandbox because
  `book-tests` compiles `tensor4all-hdf5`, whose native HDF5 build fails in
  this environment (no parallel-HDF5/MPI toolchain). The changed guide block
  in `partitioned-treetn.md` is byte-equivalent to the passing
  `add_with_patching` rustdoc doctest, so its compile validity is covered.
  CI runs the mdBook gate with HDF5 provisioned.

## Cross-model review

Read-only cross-model design review (`deepseek-v4-flash-284b:max`) approved
the corrected design record before source edits (`APPROVE`, no blockers).

Fresh read-only cross-model review of the complete final diff
(`deepseek-v4-flash-284b:max`, full candidate worktree vs pristine `main`
`646ee3a`): **APPROVE**, no blockers and no critical findings. Both hard
lines verified (no global whole-network error bound; whole local-cutoff reuse
without per-edge division). Minor non-blocking residuals recorded:

- pre-existing (not introduced here): direct `TreeTN::contract_zipup*` /
  `contract_fit` calls can still return before factorization-option
  validation on single-node / no-SVD short-cuts; the documented
  `contraction::contract` dispatch validates up front and covers all internal
  and partition callers. Optional hardening (hoisting the guard into
  `contract_zipup_impl`) is deferred.
- `add_with_patching`'s final `truncate_adaptive` re-derives its reference
  norm/volume from the post-split working partition; consistent with
  per-operation pinning, not a contract violation.

Both verdicts are recorded in this worklog and the PR body.
