# Issue #566 Task 8 handoff — complete and pushed, awaiting PR decision

**Status:** Task 8 of PR1 (`docs/superpowers/plans/2026-08-08-issue-566-pr1-soundness-ci.md`) is **complete**. Branch pushed to `origin/audit/issue-566-remediation`. **No PR created** (parent decision pending).

## Branch and commit context

- Worktree: `/home/shinaoka/tensor4all/tensor4all-rs/.worktrees/issue-566-remediation`
- Branch: `audit/issue-566-remediation` (tracking `origin/audit/issue-566-remediation`)
- HEAD when this handoff is committed: `115cb63` (`docs(worklog): record GPT-5.6 Sol review round and its fixes`)
- Relative to `origin/main`: **ahead 53, behind 3**. The branch has NOT been synchronized with the current `origin/main` (integration/sync is Task 13 work, not Task 8).
- This branch carries the full PR1 scope (Tasks 1–8). Tasks 9–13, PR2–4, and the shared-rules prerequisite (`tensor4all-agent-rules#6`) remain pending.

## Task 8 deliverables (all committed on this branch)

The previous checkpoint `6add467` (WIP handoff commit) plus seven commits pushed since:

| Commit | Content |
|---|---|
| `e9fc131` | Scanner: dep-info sources resolved per Make rule against the exact artifact output (unrelated rule can no longer satisfy source validation); nested `macro_rules!` matcher patterns excluded while real nested transcribers are reported. Regression tests for both. |
| `de029b0` | API polish: rtol-mode structured/permuted/dense isapprox, diagonal-vs-dense collapsed support, NaN isapprox rejection, `norm_squared` doc fix. |
| `e53b2f5` | API reference regenerated (`docs/api/library_panic_audit.md`). |
| `2aef5c9` | Worklog: scanner gap closure + first review + validation. |
| `53897d3` | API: scale only compact payload (all `Materialized` scaling through `scale_eager_payload`, no whole-backing strided-gap traversal); dense overflow fail-closed in `to_dense_*_col_major_vec`; branch coverage tests. |
| `f5a72f3` | API: `StructuredStorage::logical_dense_col_major_vec` now fallible (fails closed via `checked_logical_len`); gapped-storage scale regression. |
| `115cb63` | Worklog: GPT-5.6 Sol review round. |

## Reviews

Two independent reviews of the API checkpoint; the second on a different model family:

1. **Session-default DeepSeek V4 Flash** (`reviewer` builtin): PASS; minor doc/coverage notes.
2. **`reviewer-gpt` (GPT-5.6 Sol, `openai-codex/gpt-5.6-sol:high`)** — created at `~/.pi/agent/agents/reviewer-gpt.md` (user-level agent file with `model:` frontmatter; this is how model selection is done in this environment since the `subagent*` tools here do not expose a `model` argument). It found three Important findings that were then fixed and re-verified:
   1. Untracked `Materialized` scale traversed unreferenced strided-gap backing entries → all scaling now converts only the compact payload (`53897d3`).
   2. Dense logical materialization returned `Ok(empty)` on `usize` overflow → `logical_dense_col_major_vec` fallible, wrappers propagate (`53897d3`, `f5a72f3`).
   3. Coverage gaps (unmatched support in both orders, exact-mode/structured NaN, zero-vs-nonzero, gapped-storage scale) → tests added (`53897d3`, `f5a72f3`).
   Scoped re-review: finding 1 CLOSED; findings 2/3 closed in the `f5a72f3` round.

## Validation (all green, re-run after the final fixes)

- `cargo clippy --release --workspace --all-targets -- -D warnings` — clean
- `cargo nextest run --release --workspace` — **2722/2722 passed, 14 skipped**
- `cargo test --doc --release --workspace` — 839 passed
- `cargo doc --workspace --no-deps` — 0 errors (pre-existing rustdoc warnings only)
- `./scripts/test-mdbook.sh` — exit 0
- `cargo run -p api-dump --release -- . -o docs/api` — regenerated (unrelated `tensor4all_aci.md` trailing-newline drift reverted; `library_panic_audit.md` committed)
- Scanner: `python3 scripts/test-audit-library-panics.py` — 8/8; `python3 scripts/audit-library-panics.py` — 0 unbaselined, 0 stale

Note: `cargo fmt --all -- --check` and focused core/backend tests were also green at every step.

## Remaining work / decisions before PR1

1. **Parent decision: create the PR.** Nothing is pushed beyond this branch; no PR exists. If approved, follow repo convention (`git push -u origin audit/issue-566-remediation`, then `gh pr create --base main` referencing #566, squash auto-merge).
2. **Task 13 prerequisites not done:** no final spec-compliance review against `origin/main...HEAD`, no final independent correctness review of the full branch, no coverage/llvm-cov run, no main synchronization (branch is behind 3). See Task 13 in the plan for the exact command list.
3. **Tasks 9–13** and the shared-rules prerequisite (`tensor4all-agent-rules#6`) remain pending.
4. `.pi-subagents/` is untracked session infrastructure and must stay excluded — never commit it with a broad `git add`.

## Context pointers

- Plan: `docs/superpowers/plans/2026-08-08-issue-566-pr1-soundness-ci.md`
- Design: `docs/superpowers/specs/2026-08-08-issue-566-remediation-design.md`
- Prior checkpoint: `docs/superpowers/handoffs/2026-08-09-issue-566-task-8-checkpoint.md` (committed in `6add467`)
- Worklog: `docs/worklogs/2026-08-09-issue-566-panic-audit.md`
- SDK ledger: `.superpowers/sdd/2026-08-08-issue-566-pr1-soundness-ci/progress.md` (local, gitignored)
- Review artifacts: `.pi-subagents/artifacts/` (local, untracked)
