> [!NOTE] Superseded (2026-08, #639)
> `tensor4all-tcicore` was dissolved into `tensor4all-core`; the crate no
> longer exists. References below describe the pre-dissolution state.

# Issue #566 PR 1: soundness fixes and blocking panic audit

This work log covers the full PR 1 branch (`audit/issue-566-remediation`),
Tasks 1–8 of the plan
`docs/superpowers/plans/2026-08-08-issue-566-pr1-soundness-ci.md` plus the
Task 13 integration round. Task 8's scanner decisions have their own entry in
`docs/worklogs/2026-08-09-issue-566-panic-audit.md`.

## Tasks 1–7 decisions and verification

### Task 1 — Matrix shape and indexing invariants

`checked_matrix_len` returns `Option<usize>` and the three constructors panic
with a named-axis message on overflow, instead of the plan's `-> usize`
signature. Functionally equivalent (the plan's variant would also panic); the
`Option` form avoids double-computing the checked product. The public
`Index`/`IndexMut` route through a single axis-checked `offset`, so `m[[3, 0]]`
on a 2x2 matrix is rejected per axis before linearization.

### Task 2 — RRLU invariants

`validate_col_major_matrix_len` runs once at `rrlu_mut` entry, before
max-rank shortcuts and before any unchecked kernel. `rrlu` reaches the same
validated path. Leaf unsafe blocks retain local invariant assertions as
documentation.

### Task 3 — Quanticstci coordinate errors

`evaluate_grid_point` propagates conversion failures with the offending
quantics index; batch evaluation collects with `?`; continuous and discrete
initial pivots use fallible collection (no `filter_map(...ok())`). No
`V::default()` coordinate fallback remains; the remaining `V::default()` calls
initialize permutation buffers only.

### Task 4 — C API checked arithmetic

`checked_dims_product` guards every dimension product; generic raw slices
reject byte lengths above `isize::MAX`; the c64 interleaved reader checks both
`n_complex * 2` and byte length before `from_raw_parts`. Null-pointer
precedence for valid positive lengths is preserved.

### Task 5 — HDF5 file-derived integers

All reader-side signed conversions use `usize::try_from`/`i32::try_from` with
dataset context; child-group membership is cross-checked before
`Vec::with_capacity`; MPS `length`, `llim`, and `rlim` are validated
semantically before allocation. The sibling sweep also made backend attribute
lookup constant-space (`H5Aexists`) so a corrupted attribute count cannot drive
large allocations.

### Task 6 — Quantics transform shift widths

`checked_multivar_dims` centralizes the `nvariables` power-of-two and squared
site-dimension arithmetic; `embed_single_var_mpo`, `shift_operator_multivar`,
flip, phase-rotation, and affine reuse it. No unchecked `1 << nvariables`
remains in the crate.

### Task 7 — Always-on release tests

The `test` job is restored in `.github/workflows/CI_rs.yml`, uses the `ci`
cargo profile introduced by main's #581/#586/#587 (inherits `release`; keeps
CI artifacts small), and is in `rollup-rs.needs`. The redundant
`CI_rs_selfhost.yml` was retired (commit authored by the repo owner; it only
ran a subset of the restored test job and its self-hosted runner was never
active).

## Task 13 integration round (2026-08-10)

### Synchronization with main

Merged `origin/main` (commits #581/#586/#587, release profile/debug-info
reduction) into the branch. `CI_rs.yml` now uses `--cargo-profile ci` /
`--profile ci` consistently and reports cargo artifact size; the restored
`test` job adopted the `ci` profile instead of `--release` so the disk-pressure
fix in #581–#587 is preserved. `CI_rs_selfhost.yml` deletion was kept (the
delete/modify conflict with main's profile update resolved in favor of the
deliberate retirement).

### Reviews (reviewer-gpt, GPT-5.6 Sol, read-only)

Spec-compliance review (Tasks 1–8 vs plan + issue #566 Phase 0/1): no
Blocking findings; all 144 changed files categorized; 143 traceable to the
plan, documented related-bug sweeps, or the Task 8 API cascade. Findings
closed this round:

1. **Important — Task 8 missing CI-executed fixtures** (comments/rustdoc
   exclusion, private-helper classification, stale baseline). Added three
   subprocess fixtures to `scripts/test-audit-library-panics.py` (11 tests
   total).
2. **Important — `CI_rs_selfhost.yml` deletion out of Task 7 scope.** Kept as
   a repo-owner-authored decision; recorded here for the parent.
3. **Minor — `checked_matrix_len -> Option<usize>` vs plan `-> usize`.**
   Recorded above; no code change.
4. **Minor — PR1 worklog omitted Tasks 1–7.** Fixed by this entry.

Independent correctness review (full branch): five Important findings, all
fixed and verified in commit `cbfe040`:

1. `blas_mat_mul` computed the output element count with unchecked `m * n`
   after the backend call. Now `m.checked_mul(n)` is validated before tensor
   conversion, and the expected length is compared after.
2. `quanticscrossinterpolate_discrete` panicked on an empty `size`. Now
   returns an error.
3. `read_indices_from_ptrs` allocated with an unbounded `rank` before the
   `isize::MAX / size_of` byte-length check used for scalar slices. Now applies
   the same bound first.
4. Structured/diagonal C constructors copied the whole payload before
   validating metadata. Now metadata is validated first via the new public
   `Storage::validate_structured_metadata` (refactored out of
   `StructuredStorage::new`, no duplicated validation), and the payload length
   must match before the copy. Diagonal constructors pre-check `diag_len`
   against the declared index dimension.
5. The panic audit recognized only `assert!`/`debug_assert!`. It now also
   recognizes `assert_eq!`, `assert_ne!`, `debug_assert_eq!`,
   `debug_assert_ne!`, and the reviewed baseline grew from 14 to 23 entries.
   The 9 new entries were reviewed: three `matrixluci/source.rs` sites were
   hardened first (their shape products were unchecked and could wrap), the
   rest are legitimate invariants on public paths. Private-helper assertions
   remain classified as outside the public-function/method surface per the
   plan's scanner design; the boundary is documented in
   `tools/library-panic-audit/src/audit.rs`.

## Residual risks / decisions for the parent

- The assertion scanner covers public functions/methods and trait impls, not
  private helpers. Extending it to all production functions would grow the
  reviewed baseline substantially; deferred as a PR 2 candidate.
- `actionlint` is not installed locally; workflow syntax is validated by
  YAML parsing only. CI runs the real workflow.

## Final validation evidence (after the review round)

- `cargo fmt --all -- --check` — clean.
- `cargo clippy --workspace --all-targets -- -D warnings` — clean.
- `cargo nextest run --release --workspace --exclude tensor4all-hdf5` —
  2689/2689 passed, 10 skipped.
- `cargo test --release -p tensor4all-hdf5` — passed (49 tests, 4 ignored).
- `cargo test --doc --release --workspace` — 840 passed.
- `./scripts/test-mdbook.sh` — exit 0; `cargo doc --workspace --no-deps` — 0 errors.
- `python3 scripts/test-audit-library-panics.py` — 11/11.
- `python3 scripts/audit-library-panics.py` — 0 unbaselined, 0 stale
  (baseline 23 entries).
- `python3 scripts/test-repository-rules-review.py` — 89 tests;
  `repository-rules-review.py --base origin/main --worktree --dry-run` — pass
  (after adding `.pi-subagents/` to `.gitignore`, it is session infrastructure
  and was tripping the sensitive-diff scan).
- `cargo llvm-cov` (debug, the CI gate) — **207/207 files pass** after adding
  tests for the `TensorVectorSpace` fallible paths: the Task 8 API cascade had
  dropped `core/src/tensor_like.rs` from 75.2% (base) to 70.0%; new NaN-input
  and trait-default `isapprox` tests (via `BlockTensor`) restored it to 80.6%.
- `cargo llvm-cov --release` — only `treetn/src/dmrg/mod.rs` (70.3%) is below
  default; the file is untouched by this branch (base debug = branch debug =
  77.2%), so this is a pre-existing release-only deficit for Task 12's
  rationale-threshold work. The audit tool files (`audit.rs`, `main.rs`) are
  covered by `_comment_tooling` thresholds: they are exercised deterministically
  by the subprocess self-tests, which llvm-cov cannot attribute.
- `.pi-subagents/` is untracked session infrastructure and now gitignored.

## Shared-rules alignment: coverage ownership (2026-08-10)

Maintainer decision: align local coverage requirements with the shared agent
rules (`tensor4all-agent-rules`, `common/docs-and-tests.md`). Coverage is the
canonical CI-owned gate; the local pre-PR gate is attestation-based, so local
llvm-cov runs are no longer required for PR validation, deletion PRs, or the
closure audit. The CI coverage job is authoritative. Revised: plan Task 12
Step 5 and Task 13 step 4, the design Validation Contract, and AGENTS.md.
Unchanged: CI release-mode coverage (Task 12), thresholds never lowered,
rationale clusters required for exceptional files. The local
always-`--release` test policy remains a documented repository-local stricter
override (justification: heavy numerical test runtime in debug builds); it is
a separate decision point for the #566 Phase 2 shared-rules adoption.

## Task 10 — runnable doctests and kryst removal (2026-08-10)

- Both prohibited `no_run` doctests in treetn were replaced with runnable
  examples carrying numerical assertions:
  - `partial_contract`: contracts [1,2] and [3,4] over the pair and asserts the
    scalar result is 11.0 via `contract_to_tensor().only().real()`.
  - `square_linsolve`: two-site identity MPO with explicit index mappings and a
    zero initial guess; asserts relative residual < 1e-8 and that the solution
    reproduces the RHS (Frobenius diff < 1e-6). A zero init (not the solution)
    is used so the assertions actually exercise the solver.
- One-site mapped systems silently returned the initial guess (residual 1.0)
  because the two-site sweep planner produces an empty plan for a one-node
  network. `square_linsolve` now rejects single-site inputs with an explicit
  error ("requires at least two sites"), with a regression test
  `test_square_linsolve_rejects_one_site_systems`. A real one-site local solve
  remains a follow-up. The plan's one-site example therefore could not be used
  verbatim; the two-site example mirrors the passing
  `test_square_linsolve_with_mappings_identity`.
- `kryst` removed from workspace and treetn manifests; stale "via kryst" and
  the kryst link in `linsolve/square/mod.rs` now name
  `tensor4all_core::krylov::gmres`. Cargo.lock is gitignored (regenerated
  locally by Cargo). No live dependency on `kryst` remains; the only remaining
  textual `kryst` mentions are intentional records in the worklog and the
  historical plan under `docs/superpowers/plans/**` (not shipped, excluded by
  the plan's own `:!docs/superpowers/specs/**`/`plans/**` grep convention).
- `cargo test --doc --release --workspace` — 840 passed; treetn linsolve
  35 tests pass; no `no_run`/`ignore` doctest fences remain in `.rs` (a
  historical planning doc under `docs/superpowers/plans/` still contains
  ```ignore fences; it is not shipped or compiled, out of scope).

## Task 12 — release-mode coverage and threshold rationale (2026-08-10)

- CI coverage job now runs `cargo llvm-cov --release --workspace --exclude
  tensor4all-hdf5` (release is the repository's normal verification mode; the
  debug-only `CARGO_PROFILE_DEV_DEBUG` env was removed). Coverage stays a
  CI-owned gate; the local pre-PR gate is attestation-based per the shared
  agent rules.
- `scripts/check-coverage.py` gained `--thresholds PATH` so the self-test can
  isolate fixtures; `scripts/test-check-coverage.py` (5 tests) proves default
  pass/fail, exact per-file override, missing-file fallback to default, and
  that top-level `_comment_*` keys are ignored for enforcement. Wired into
  `jobs.scripts`.
- `coverage-thresholds.json` gained rationale clusters (`_comment_tooling`,
  `_comment_xtask`, `_comment_test_utils`, `_comment_capi`,
  `_comment_expensive_algorithms`, `_comment_reference_crates`,
  `_comment_boundary_modules`, `_comment_dmrg_release_only`) without changing
  any number. The only release-mode deficit is `treetn/src/dmrg/mod.rs`
  (72.4% vs 75; debug is 77.2% — release compiles out debug_assert paths);
  pinned at 72 with a documented reason, consistent with PR 1's recorded
  pre-existing deficit.
- Local release measurement: 207/207 files pass with the new thresholds.

## PR 2 round summary (2026-08-10)

Branch `chore/issue-566-pr2-gates` completes Tasks 9-12 of the plan plus the
permanent review-bot LLM removal:

- Task 9: incremental public-error-doc + crate-boundary gates (ported from
  tenferro, repository-specific tests; tcicore->tensorci dev cycle removed by
  moving the end_to_end_chain_tci bench to tensorci; CI wiring with base-SHA
  resolution).
- Task 10: runnable doctests for partial_contract and square_linsolve; kryst
  removed; one-site linsolve now rejected instead of silently returning init.
- Task 11: debug.md, plan/, coverage-local.json, and the orphan
  internal/tenferro-internal-ad-linalg test deleted; still-live linsolve and
  contraction-API rationale migrated to docs/design/.
- Task 12: CI coverage switched to release; check-coverage.py gains
  --thresholds + 5-fixture self-test; _comment_* rationale clusters completed
  per the spec contract (incl. _comment_hdf5 and the release-only dmrg pin).
- review_bot.yml: external LLM review permanently removed; deterministic
  --dry-run review is the default; waive label preserved.

Per-task reviews ran reviewer-gpt (GPT-5.6 Sol) after Tasks 9, 10, 11, and a
combined round for Task 12 + review bot; all Blocking/Important findings were
fixed and re-verified (sub-table Cargo parser, multiline trait scope,
multiline changed signatures, stale tuples, dev-cycle dedup, non-solution
solver doctest, one-site rejection, durable design docs, rationale clusters).

Final validation (all green): fmt, clippy -D warnings, nextest release
2694/2694 (+10 skipped), hdf5 49, doctests 840, mdBook, cargo doc 0 errors,
all five python self-tests, crate-boundary + incremental public-error-doc +
repository-rules checks, and release coverage 207/207 (CI-owned gate; local
attestation recorded here). Note: the coverage measurement was performed once
locally to confirm the CI switch; the CI coverage job is the authoritative
gate going forward.

## PR 2 merge (2026-08-10)

PR #590 merged to main as `a6a51de`. CI was fully green before auto-merge:
Lint, Maintenance scripts (all five self-tests + incremental public-error-doc
+ crate-boundary + coverage-checker gates), Test (nextest release), Doctests,
Coverage (release-mode llvm-cov, 207/207 with the tutorial-code CI-variance
pin at 74), and the review-bot gate (deterministic-only; the
`rules-review:no-llm` label was used for this PR because main's review-bot
still ran the old LLM workflow until this merge). Post-merge verification:
`origin/main` carries the deterministic `review_bot.yml` (no DEEPSEEK refs),
the release coverage command, and the completed rationale clusters.

Remaining #566 work: PR 3 (typed errors/layering), PR 4 (API vocabulary),
PR 5 (performance + closure), and the shared-rules prerequisite
(tensor4all-agent-rules#6).
