# Issue #566 Task 13 integration round — reviews fixed, validated, awaiting PR decision

**Status:** Task 13 integration (sync, dual review, fixes, full validation) is
**done** on `audit/issue-566-remediation`. Branch synced with `origin/main`
(merged #581/#586/#587). **No PR created** (parent decision pending). The full
PR1 scope (Tasks 1–8) plus the Task 13 review/fix round are on this branch.

## Branch state

- Worktree: `/home/shinaoka/tensor4all/tensor4all-rs/.worktrees/issue-566-remediation`
- HEAD: `088f82e` (`test(core): restore tensor_like coverage from Task 8 API cascade`)
- Relative to `origin/main`: **ahead 57, behind 0** (fully synchronized; the
  3-behind commits #581/#586/#587 were merged as `af44e88`).
- Key commits since the Task 8 handoff:
  - `af44e88` — merge of `origin/main` (ci-profile workflow reconciliation;
    selfhost retirement kept; restored `test` job now uses `--cargo-profile ci`).
  - `cbfe040` — fix(audit): close full-branch review findings (5 Important
    correctness findings + baseline 14→23 + 3 new scanner fixtures).
  - `088f82e` — test(core): restore tensor_like coverage (75.2%→70.0%→80.6%).

## Reviews (reviewer-gpt, GPT-5.6 Sol, read-only; reports archived)

1. **Spec-compliance** (Tasks 1–8 vs plan + issue #566): no Blocking. Two
   Important findings resolved: missing CI-executed scanner fixtures (added
   comment/rustdoc exclusion, private-helper classification, stale-baseline
   subprocess tests → 11 tests), and the selfhost-workflow deletion (kept;
   repo-owner-authored decision, recorded in worklog). Two Minors resolved in
   the worklog (checked_matrix_len `Option` vs plan `usize`; Tasks 1–7
   evidence added).
2. **Independent correctness** (full branch): five Important findings, all
   fixed in `cbfe040`:
   - matrix multiplication output shape overflow validated before backend call;
   - empty discrete QTCI grid returns an error instead of panicking;
   - capi index-pointer arrays get the `isize::MAX / size_of` byte bound;
   - structured/diagonal capi constructors validate metadata and payload
     length before copying (new public `Storage::validate_structured_metadata`,
     refactored out of `StructuredStorage::new` — no duplicated validation);
   - scanner now recognizes `assert_eq`/`assert_ne`/`debug_assert_eq`/
     `debug_assert_ne`; 9 new reviewed baseline entries (3 matrixluci sites
     hardened with checked products first).

## Validation (all green after the final fixes)

- fmt/clippy `-D warnings` clean; nextest release 2689/2689 (+10 skipped);
  hdf5 cargo test passed; doctests 840; mdBook exit 0; cargo doc 0 errors.
- scanner self-test 11/11; audit 0 unbaselined / 0 stale (baseline 23).
- repository-rules-review (worktree, dry-run): pass — required adding
  `.pi-subagents/` to `.gitignore` (untracked session infra was tripping the
  sensitive-diff scan).
- debug llvm-cov (the CI gate): **207/207 files pass**. The Task 8 API
  cascade had dropped `core/src/tensor_like.rs` to 70.0%; NaN-input +
  BlockTensor trait-default tests restored it to 80.6%.
- release llvm-cov: only `treetn/src/dmrg/mod.rs` (70.3%) below default —
  untouched by the branch, pre-existing release-only deficit for Task 12's
  rationale-threshold work. The panic-audit tool files are pinned with a
  `_comment_tooling` rationale (they are exercised by subprocess self-tests
  llvm-cov cannot attribute).

## Remaining work / decisions

1. **Parent decision: create the PR.** Branch is synced and fully validated.
   Per repo convention: `git push -u origin audit/issue-566-remediation`, then
   `gh pr create --base main` referencing #566, squash auto-merge. PR body
   should mention the selfhost-workflow retirement and the private-helper
   scanner boundary as recorded decisions.
2. **Tasks 9–13 remainder** (after PR1): public-error-doc/crate-boundary gates,
   doctest/kryst cleanup, debris deletion, release-coverage switch (Task 12
   will formalize the `_comment_tooling`/dmrg rationale clusters), and the
   shared-rules prerequisite (`tensor4all-agent-rules#6`).
3. `.pi-subagents/` is now gitignored (never stage it; do not use broad
   `git add`).

## Context pointers

- Plan: `docs/superpowers/plans/2026-08-08-issue-566-pr1-soundness-ci.md`
- Worklog: `docs/worklogs/2026-08-08-issue-566-pr1-soundness-ci.md`
  (Tasks 1–7 evidence, review round, final validation numbers)
- Scanner worklog: `docs/worklogs/2026-08-09-issue-566-panic-audit.md`
- Review artifacts: `.pi-subagents/artifacts/` (local, gitignored)
