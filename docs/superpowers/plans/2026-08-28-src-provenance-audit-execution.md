# SRC Provenance Audit Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Tasks 1-6 have no dependencies on each other and may be dispatched in parallel; Task 7 depends on all of Tasks 1-6.

**Goal:** Produce a from-scratch provenance audit report of every line the
`feature/treetn-src` branch added on top of `origin/main` (the SRC
implementation `gw-rs` pins), cross-checked against the paper, its reference
Python, and Hiroshi Shinaoka's issue comments — flagging hallucinated,
hand-rolled-duplicate, missing, or unverified-derivation code.

**Architecture:** Six independent workstream tasks each read their assigned
slice of the diff plus the spec's embedded Tier-1 sources and produce a
structured provenance table as a standalone markdown file. A seventh
synthesis task merges all six into the single deliverable report. No source
code changes happen at any point in this plan — every task's deliverable is
a markdown file.

**Tech Stack:** Markdown deliverables; `git`/`grep`/`wc` for structural
verification; no build or test suite is exercised because no Rust code is
touched.

**Spec:** `docs/plans/2026-08-28-src-provenance-audit.md` (this plan's
tasks implement it; every task's implementer must read the spec's Epistemics,
Verdict taxonomy, and Appendix A sections in full before starting — they are
not reproduced here).

## Global Constraints

- **Working directory for every task:** the `feature/treetn-src` worktree at
  `/root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src`,
  currently at commit `7d574d7` (spec commit, on top of the pinned `9e018d4`).
  Do not check out a different commit or branch.
- **No source code edits.** Every task's deliverable is a new markdown file
  under `docs/plans/`. If a task's implementer finds `crates/` modified when
  they start, stop and report it rather than continuing — see the stash note
  below.
- **Do not touch `stash@{0}`** on this worktree (`git stash list`) — it holds
  an unrelated, pre-existing efficiency investigation that was deliberately
  set aside so this audit runs against the clean pinned commit. Do not pop,
  drop, or apply it.
- **Tier 1 (only these close out a finding):** the paper
  (`/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`
  + PDF), the reference Python
  (`/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/*.py`
  + `.cpp`), and Hiroshi's comments (full text in the spec's Appendix A —
  read chronologically, respect the marked corrections, never cite a
  superseded statement).
- **Tier 2 (map only, never a reference):** the spec doc itself quoting
  `docs/plans/2026-08-26-treetn-src-contraction-plan.md` and
  `docs/PROVENANCE_AND_CITATION_POLICY.md`. Any claim borrowed from these
  must be re-traced to Tier 1 before being used as a pass/fail criterion;
  if it can't be, the finding is `PLAN-CLAIM-UNVERIFIED`, not a pass.
- **Verdict taxonomy — exactly these eleven tokens, no others:**
  `SOURCED-PAPER(...)`, `SOURCED-PYTHON(...)`, `SOURCED-COMMENT(...)`,
  `DERIVED-VERIFIED`, `SUSPECT-UNVERIFIED`, `HANDROLLED-DUPLICATE`,
  `MISSING-VS-SOURCE`, `SCOPE-DEVIATION`, `LICENSE-RISK`,
  `PLAN-CLAIM-UNVERIFIED`, `SOURCE-AMBIGUOUS`. Full definitions are in the
  spec's "Verdict taxonomy" section.
- **Output schema — every workstream file (Tasks 1-6) uses this exact
  structure** so Task 7 can merge them mechanically:

  ```markdown
  # <WS-ID> — <Workstream Name>

  **Files audited:**
  - `<exact path>`
  - `<exact path>`
  ...

  ## Provenance table

  | File | Code unit | Lines | Verdict | Citation / gap |
  |---|---|---|---|---|
  | `<path>` | `<fn/impl/section name>` | `<start>-<end>` | `<TOKEN(...)>` | `<citation text, or what's missing>` |

  ## Detailed derivations and flagged findings

  ### <Code unit name> — <VERDICT>
  <Full prose: derivation, quoted source text, or explanation of the gap.
  Mandatory for every DERIVED-VERIFIED, SUSPECT-UNVERIFIED,
  HANDROLLED-DUPLICATE, MISSING-VS-SOURCE, SCOPE-DEVIATION, LICENSE-RISK,
  PLAN-CLAIM-UNVERIFIED, and SOURCE-AMBIGUOUS row — a one-line table cell is
  not sufficient for these eight verdicts.>
  ```

  Every non-trivial code unit (function, impl block, or clearly identifiable
  algorithm step — not every single line) in the "Files audited" list gets a
  table row. Trivial plumbing (re-exports, `mod` declarations, derive
  attributes) may be summarized as one row per file instead of one row per
  item, but the file must still appear at least once.
- **No placeholders.** No `TBD`, `TODO`, "needs further investigation" as a
  final answer — if something is genuinely unresolved after real effort, the
  verdict is `SUSPECT-UNVERIFIED` or `SOURCE-AMBIGUOUS` with a concrete
  explanation of what's missing, not a placeholder.
- **Report only.** No task in this plan fixes anything it finds. Do not edit
  `crates/`.

---

## Task 1: WS-chain — the literal single-chain case

**Files:**
- Create: `docs/plans/audit-workstreams/ws-chain.md`
- Read (do not modify):
  `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` (736 lines)

**Interfaces:**
- Consumes: the spec's Epistemics, Verdict taxonomy, Appendix A (Hiroshi
  comments), and "WS-chain" sections — read the whole spec first.
- Produces: `docs/plans/audit-workstreams/ws-chain.md` in the Global
  Constraints output schema, listing `src_chain.rs` under "Files audited."
  Consumed by Task 7.

- [ ] **Step 1: Read the spec and the Tier-1 chain sources**

Read, in this order:
1. `docs/plans/2026-08-28-src-provenance-audit.md` in full (spec + Appendix A).
2. `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`
   §2.1 and Algorithm 1 (the chain SRC algorithm).
3. `/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/contraction.py`,
   `MPO.py`, `MPS.py` — the chain contraction path.
4. Appendix A's 2026-07-29 comment (QR-only claim) in the spec.

- [ ] **Step 2: Read the target file**

Read `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` in full
(736 lines — one or two `Read` calls).

- [ ] **Step 3: Build the provenance table**

For every function, impl block, and algorithm step in the file: identify
whether it maps to a specific paper equation/Algorithm-1 step
(`SOURCED-PAPER`), a specific Python function (`SOURCED-PYTHON`), neither
(`DERIVED-VERIFIED` if you can independently re-derive and confirm it,
`SUSPECT-UNVERIFIED` otherwise), or reimplements something
`tensor4all-tensorbackend` already exposes (`HANDROLLED-DUPLICATE`). Grep the
file for every SVD-related identifier (`svd`, `Svd`, `SVD`) and confirm each
call site is the one permitted final-truncation call, not inside the QR-only
hot path — any SVD call inside the per-site sketch/QR/projection loop is a
`SUSPECT-UNVERIFIED` or worse finding, and say so explicitly in the Detailed
findings section with the exact line number.

Write the file at `docs/plans/audit-workstreams/ws-chain.md` following the
Global Constraints schema exactly.

- [ ] **Step 4: Verify structural completeness**

Run:

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-chain.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
grep -q "src_chain.rs" "$FILE" && echo "covers src_chain.rs" || echo "FAIL: src_chain.rs not covered"
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -cE '\`(SOURCED-PAPER|SOURCED-PYTHON|SOURCED-COMMENT|DERIVED-VERIFIED|SUSPECT-UNVERIFIED|HANDROLLED-DUPLICATE|MISSING-VS-SOURCE|SCOPE-DEVIATION|LICENSE-RISK|PLAN-CLAIM-UNVERIFIED|SOURCE-AMBIGUOUS)' "$FILE"
```

Expected: "file exists", "covers src_chain.rs", "no placeholders", and a
verdict-token count greater than zero (should roughly track the number of
non-trivial code units in a 736-line file — a count of 1-2 means the table
is too coarse; go back and split it by function).

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-chain.md
git commit -m "docs(audit): WS-chain provenance table for src_chain.rs"
```

---

## Task 2: WS-tree-probe — tree generalization and MPO-MPO probing

**Files:**
- Create: `docs/plans/audit-workstreams/ws-tree-probe.md`
- Read (do not modify):
  `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs` (641 lines),
  `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs` (1171 lines,
  the largest file in the diff)

**Interfaces:**
- Consumes: spec Epistemics/taxonomy/Appendix A, spec "WS-tree-probe" section.
- Produces: `docs/plans/audit-workstreams/ws-tree-probe.md` in the output
  schema, listing both files under "Files audited." Consumed by Task 7.

- [ ] **Step 1: Read the spec and Tier-1 sources**

Read `docs/plans/2026-08-28-src-provenance-audit.md` in full, with particular
attention to the 2026-08-24 Hiroshi comment (factorized MPO-MPO probe order:
contract `conj(X[:,k])` into the first operand, `conj(Y[:,k])` into the
second, then the shared physical index, then incoming tree messages) and the
note that the tree generalization has no Tier-1 source at all (neither paper
nor comments cover trees), so it is necessarily `DERIVED-VERIFIED` or
`SUSPECT-UNVERIFIED` throughout.

- [ ] **Step 2: Read both target files**

Read `src_tree.rs` (641 lines) and `src_probe.rs` (1171 lines) in full.

- [ ] **Step 3: Build the provenance table, with full derivations**

For `src_probe.rs`: verify every MPO-MPO probe construction against the
exact contraction order from Hiroshi's comment; confirm the production path
never constructs a fused `d²`-dimensional physical probe anywhere (grep for
any place two physical legs get fused into one dimension before probing —
that is a `SCOPE-DEVIATION` if found, since Hiroshi's comment and the paper
both call for avoiding it).

For `src_tree.rs`: re-derive the directed sketch-message construction
(postorder child-to-parent pass, preorder parent-to-child pass, cache keyed
by `(from_node, to_node)`) from first principles, generalizing the paper's
chain forward/backward environment recursion to a tree — do not just check
that it matches the spec's or the implementation plan's description of it,
since those are Tier 2. Write out the full re-derivation in the "Detailed
derivations" section for every code unit marked `DERIVED-VERIFIED`; if the
re-derivation doesn't confirm the code, mark it `SUSPECT-UNVERIFIED` and
show exactly where the derivation breaks down.

Write the file at `docs/plans/audit-workstreams/ws-tree-probe.md`.

- [ ] **Step 4: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-tree-probe.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
grep -q "src_tree.rs" "$FILE" && echo "covers src_tree.rs" || echo "FAIL: src_tree.rs not covered"
grep -q "src_probe.rs" "$FILE" && echo "covers src_probe.rs" || echo "FAIL: src_probe.rs not covered"
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -c "DERIVED-VERIFIED" "$FILE"
```

Expected: both files covered, no placeholders, and every `DERIVED-VERIFIED`
row has a matching subsection under "Detailed derivations" — spot check by
grepping the code-unit names from a few table rows against the section
headers below.

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-tree-probe.md
git commit -m "docs(audit): WS-tree-probe provenance table for src_tree.rs and src_probe.rs"
```

---

## Task 3: WS-backend — numerics and the incremental-QR question

**Files:**
- Create: `docs/plans/audit-workstreams/ws-backend.md`
- Read (do not modify):
  `crates/tensor4all-tensorbackend/src/backend.rs`,
  `crates/tensor4all-tensorbackend/src/incremental_qr.rs`,
  `crates/tensor4all-tensorbackend/src/lib.rs`

**Interfaces:**
- Consumes: spec Epistemics/taxonomy/Appendix A, spec "WS-backend" section.
- Produces: `docs/plans/audit-workstreams/ws-backend.md`. Consumed by Task 7.

- [ ] **Step 1: Read the spec and Tier-1 sources**

Read the spec's "WS-backend" section and Appendix C reference. Read the
paper's Appendix C (adaptive-rank error estimator) in
`arxiv-source/report.tex`. Read
`/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/incrementalqr.py`
and `incrementalqr.cpp` in full — these are the real reference
implementations this Rust file must be checked against, not assumed
hallucinated on sight.

- [ ] **Step 2: Read the target files**

Read `backend.rs`, `incremental_qr.rs`, and `lib.rs` (the diff hunks — use
`git diff origin/main -- crates/tensor4all-tensorbackend/src/backend.rs
crates/tensor4all-tensorbackend/src/incremental_qr.rs
crates/tensor4all-tensorbackend/src/lib.rs` to see exactly what this branch
added, since `incremental_qr.rs` is a new file and `backend.rs`/`lib.rs`
have pre-existing content too).

- [ ] **Step 3: Build the provenance table**

Check the adaptive-rank estimator in `backend.rs` against paper Appendix C
directly — re-derive the estimator formula from the paper's equations
yourself rather than trusting any transcription of it elsewhere.

For `incremental_qr.rs`: determine, function by function, whether each part
is a legitimate verified port of `incrementalqr.py`/`.cpp`'s logic
(`SOURCED-PYTHON`, and check it isn't close enough to read as a translation
— `LICENSE-RISK` if it is), a `SUSPECT-UNVERIFIED`/hallucinated addition, or
a `SCOPE-DEVIATION` (the 2026-08-26 plan text calls incremental QR "a later
optimization gate" — if this file exists with no profiling justification
recorded anywhere in the branch, say so explicitly even if the logic itself
turns out to be sound).

Grep `backend.rs` and `incremental_qr.rs` for every SVD, matrix-inverse, or
hand-rolled decomposition and check each one against
`tensor4all-tensorbackend`'s and tenferro's existing public API — any
duplicate of existing backend functionality is `HANDROLLED-DUPLICATE`.

Write the file at `docs/plans/audit-workstreams/ws-backend.md`.

- [ ] **Step 4: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-backend.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
for f in backend.rs incremental_qr.rs lib.rs; do
  grep -q "$f" "$FILE" && echo "covers $f" || echo "FAIL: $f not covered"
done
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -c "SVD\|svd" "$FILE"
```

Expected: all three files covered, no placeholders, and at least one
mention of SVD handling (the audit's original motivating question) with an
explicit verdict attached.

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-backend.md
git commit -m "docs(audit): WS-backend provenance table for tensorbackend additions"
```

---

## Task 4: WS-core — tensor4all-core additions

**Files:**
- Create: `docs/plans/audit-workstreams/ws-core.md`
- Read (do not modify):
  `crates/tensor4all-core/src/defaults/idx_tensor.rs`,
  `crates/tensor4all-core/src/tensor_like.rs`,
  `crates/tensor4all-core/src/defaults/factorize.rs`,
  `crates/tensor4all-core/src/index_like.rs`,
  `crates/tensor4all-core/src/index_ops.rs`,
  `crates/tensor4all-core/src/defaults/index.rs`

**Interfaces:**
- Consumes: spec Epistemics/taxonomy/Appendix A, spec "WS-core" section.
- Produces: `docs/plans/audit-workstreams/ws-core.md`. Consumed by Task 7.

- [ ] **Step 1: Read the spec's WS-core section**

Note the priority flag: `factorize.rs` is called out as the most likely
place for a hand-rolled SVD/QR interface given its name — treat it as the
first file to check, not the last.

- [ ] **Step 2: Diff and read the target files**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git diff origin/main -- \
  crates/tensor4all-core/src/defaults/idx_tensor.rs \
  crates/tensor4all-core/src/tensor_like.rs \
  crates/tensor4all-core/src/defaults/factorize.rs \
  crates/tensor4all-core/src/index_like.rs \
  crates/tensor4all-core/src/index_ops.rs \
  crates/tensor4all-core/src/defaults/index.rs \
  > /tmp/ws-core.diff
```

Read `/tmp/ws-core.diff` (this isolates exactly what the branch added, since
all six files pre-exist on `origin/main` with unrelated content). Read
`factorize.rs` and `idx_tensor.rs` in full for context around the diff
hunks, not just the added lines in isolation.

- [ ] **Step 3: Build the provenance table**

For every added function in `factorize.rs`: does the paper or reference
Python call for this factorization at all? Does Hiroshi's QR-only claim
(2026-07-29 comment) rule out an SVD here? If the function duplicates
something `tensor4all-tensorbackend` already exposes, it's
`HANDROLLED-DUPLICATE` regardless of whether the math is correct.

For `idx_tensor.rs`/`tensor_like.rs`: the 2026-08-26 plan text gates these
additions behind profiling ("add a reusable batch/stack constructor... if
column construction or assembly is material"). Search the branch's commit
messages (`git log --oneline 9e018d4 ^origin/main`) and any docs/worklogs on
the branch for evidence that profiling was actually done before adding
these ~1000 lines. If you find none, mark the addition
`PLAN-CLAIM-UNVERIFIED` (the plan's own justification for the addition
doesn't trace to any recorded evidence) in addition to whatever provenance
verdict the code itself gets.

For `index_like.rs`/`index_ops.rs`/`defaults/index.rs`: check whether these
are mechanical (new index/ID kinds needed for SRC's "replace internal bond
indices with fresh IDs before local contractions" step) or introduce
unrelated capability.

- [ ] **Step 4: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-core.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
for f in idx_tensor.rs tensor_like.rs factorize.rs index_like.rs index_ops.rs defaults/index.rs; do
  grep -q "$(basename "$f")" "$FILE" && echo "covers $f" || echo "FAIL: $f not covered"
done
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -q "factorize.rs" "$FILE" && grep -A2 "factorize.rs" "$FILE" | grep -qE "SOURCED|SUSPECT|HANDROLLED|DERIVED" && echo "factorize.rs has a verdict" || echo "FAIL: factorize.rs not verdicted"
```

Expected: all six files covered, no placeholders, `factorize.rs` explicitly
verdicted.

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-core.md
git commit -m "docs(audit): WS-core provenance table for tensor4all-core additions"
```

---

## Task 5: WS-integration — dispatch, public API, and cross-cutting glue

**Files:**
- Create: `docs/plans/audit-workstreams/ws-integration.md`
- Read (do not modify):
  `crates/tensor4all-treetn/src/treetn/contraction.rs`,
  `crates/tensor4all-treetn/src/operator/apply.rs`,
  `crates/tensor4all-treetn/src/treetn/fit.rs`,
  `crates/tensor4all-treetn/src/treetn/swap.rs`,
  `crates/tensor4all-treetn/src/algorithm.rs`,
  `crates/tensor4all-itensorlike/src/options.rs`,
  `crates/tensor4all-itensorlike/src/contract.rs`,
  `crates/tensor4all-capi/src/treetn.rs`,
  `crates/tensor4all-capi/src/types.rs`,
  `crates/tensor4all-treetn/src/lib.rs`,
  `crates/tensor4all-treetn/src/prelude.rs`,
  `crates/tensor4all-treetn/README.md`,
  `crates/tensor4all-itensorlike/src/lib.rs`,
  `crates/tensor4all-itensorlike/src/prelude.rs`,
  `crates/tensor4all-capi/include/tensor4all_capi.h`

**Interfaces:**
- Consumes: spec Epistemics/taxonomy/Appendix A, spec "WS-integration"
  section (including the plumbing-file catch-all note).
- Produces: `docs/plans/audit-workstreams/ws-integration.md`. Consumed by
  Task 7.

- [ ] **Step 1: Read the spec's WS-integration section**

Note the three specific checks it names: the mandatory chain-reduction-gate
test's faithfulness (not just its existence), the `PrefixCache` trait ask
from Hiroshi's 2026-08-27 comments, and #691 leakage (interface-sketching /
parallel-SRC material that shouldn't be here since #691 is a separate
unimplemented proposal).

- [ ] **Step 2: Read the target files**

Read the main five files in full
(`contraction.rs`, `apply.rs`, `fit.rs`, `swap.rs`, `algorithm.rs`) and the
itensorlike/capi files' diff hunks via
`git diff origin/main -- crates/tensor4all-itensorlike/ crates/tensor4all-capi/`.
For the plumbing files, a `git diff origin/main -- <file>` is sufficient —
these don't need full-file reads.

- [ ] **Step 3: Build the provenance table**

Check the public API surface (`ContractionMethod::Src`, `SrcOptions`,
`SrcRankSelection`, `SrcContractionResult`, `SrcContractionReport`,
`SrcEdgeReport`, the `*_src_fixed`/`*_src_adaptive` constructors) against
what actually exists in `contraction.rs`/`apply.rs`/`algorithm.rs` — note
any mismatch as its own finding (this is a code-vs-plan comparison, which is
allowed; what's not allowed is treating the plan's proposed shapes as
correct without separately checking whether they make sense against Tier 1).

Locate the chain-reduction-gate test (likely in
`contraction/tests/mod.rs`, cross-reference with Task 6's file list) via the
call sites in `contraction.rs`, and independently assess — re-deriving from
the paper's chain equations, not from the plan's description of the gate —
whether it actually verifies what it claims to.

Grep the five main files for `PrefixCache`, `trait.*[Cc]ache`, or an
equivalent abstraction; if the cache is a hard-coded `Vec` built in a
forward loop with no trait boundary, that's a direct, checkable
`SCOPE-DEVIATION` against Hiroshi's explicit ask — quote the ask and the
code side by side in the Detailed findings section.

Grep for any code structure resembling "interface sketching," "Layer 2,"
sub-chain partitioning, or MPI/parallel-chain machinery — flag as
`SCOPE-DEVIATION` (#691 material) if found.

Give every plumbing file (`lib.rs`, `prelude.rs`, `README.md`, capi header)
at least a one-line row confirming it's a mechanical re-export/registration
change with no logic to verify, or flagging it if it's not.

- [ ] **Step 4: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-integration.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
for f in contraction.rs apply.rs fit.rs swap.rs algorithm.rs options.rs contract.rs; do
  grep -q "$f" "$FILE" && echo "covers $f" || echo "FAIL: $f not covered"
done
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -qi "PrefixCache\|cache trait" "$FILE" && echo "cache-trait ask addressed" || echo "FAIL: cache-trait ask not addressed"
```

Expected: all named files covered, no placeholders, cache-trait ask
explicitly addressed one way or the other.

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-integration.md
git commit -m "docs(audit): WS-integration provenance table for dispatch and public API"
```

---

## Task 6: WS-tests — test coverage and tolerance integrity

**Files:**
- Create: `docs/plans/audit-workstreams/ws-tests.md`
- Read (do not modify): every test file touched in the diff — enumerate them
  first with the command in Step 1 rather than trusting a fixed list, since
  this workstream's job is specifically to catch anything untracked
  elsewhere.

**Interfaces:**
- Consumes: spec Epistemics/taxonomy/Appendix A, spec "WS-tests" section,
  and the "Test matrix" section of `docs/plans/2026-08-26-treetn-src-contraction-plan.md`
  (read as Tier 2 — a map of what should exist, not proof that it does).
- Produces: `docs/plans/audit-workstreams/ws-tests.md`. Consumed by Task 7.

- [ ] **Step 1: Enumerate every test file touched in the diff**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git diff --stat origin/main -- crates/ | grep -i test
```

This should return (at minimum) `contraction/tests/mod.rs`,
`operator/apply/tests/mod.rs`, `partial_contraction/tests/mod.rs`,
`tensor_like/tests/mod.rs`, `backend/tests/mod.rs`, plus any smaller test
file additions in `itensorlike`/`capi`/`algorithm` not enumerated in the
spec's WS-integration catch-all. Use the actual command output as your file
list, not the list here — this step exists precisely so nothing is missed.

- [ ] **Step 2: Read the plan's Test matrix section and diff each test file**

Read the "Test matrix" section of
`docs/plans/2026-08-26-treetn-src-contraction-plan.md`. For each test file
from Step 1, run `git diff origin/main -- <file>` and read the added test
cases.

- [ ] **Step 3: Build the provenance table**

For each named category in the plan's Test matrix (correctness by dtype and
topology, control flow/errors, reproducibility, the explicit chain-reduction
test, the structural regression against fused-`d²` MPO-MPO probes): find the
actual test in the diff, or mark it `MISSING-VS-SOURCE` if absent — but
independently judge whether each present test is a real correctness check
against Tier 1 (does it compare against a dense/paper-equation oracle?) or
superficial (does it just check the code doesn't panic?), noting the
difference explicitly rather than crediting a shallow test as covering the
category.

Diff every test's numeric tolerance (`rtol`, `atol`, `epsilon`, or similar)
against `origin/main`'s pre-existing tests in the same files where such
tests already existed before this branch. Flag any loosened tolerance as
`SCOPE-DEVIATION` — the plan explicitly forbids this without user approval,
and this is a directly checkable diff, not a Tier-2 judgment call.

- [ ] **Step 4: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/audit-workstreams/ws-tests.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
grep -qi "tolerance" "$FILE" && echo "tolerance check addressed" || echo "FAIL: tolerance check not addressed"
grep -qi "chain.reduction" "$FILE" && echo "chain-reduction gate addressed" || echo "FAIL: chain-reduction gate not addressed"
```

Expected: no placeholders, tolerance and chain-reduction-gate checks both
present with explicit verdicts.

- [ ] **Step 5: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/audit-workstreams/ws-tests.md
git commit -m "docs(audit): WS-tests provenance table for test coverage and tolerances"
```

---

## Task 7: Synthesis — merge into the final audit report

**Files:**
- Create: `docs/plans/2026-08-28-src-provenance-audit-report.md`
- Read (do not modify): all six files from Tasks 1-6:
  `docs/plans/audit-workstreams/ws-{chain,tree-probe,backend,core,integration,tests}.md`

**Interfaces:**
- Consumes: the six workstream files (each in the Global Constraints output
  schema).
- Produces: `docs/plans/2026-08-28-src-provenance-audit-report.md` — the
  spec's deliverable. This is the terminal task; nothing consumes its
  output within this plan.

- [ ] **Step 1: Confirm all six inputs exist and are complete**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
for f in chain tree-probe backend core integration tests; do
  FILE="docs/plans/audit-workstreams/ws-$f.md"
  test -f "$FILE" && echo "$FILE present" || echo "BLOCKED: $FILE missing, cannot synthesize"
done
```

If any file is missing, stop — Task 7 cannot proceed until Tasks 1-6 are all
committed.

- [ ] **Step 2: Read all six files in full**

Read each of the six files completely before drafting the synthesis — the
merge requires seeing every table row to rank severity and catch overlaps.

- [ ] **Step 3: Draft the executive summary**

Rank every non-`SOURCED-*` finding (i.e. every `DERIVED-VERIFIED` needing
review, `SUSPECT-UNVERIFIED`, `HANDROLLED-DUPLICATE`, `MISSING-VS-SOURCE`,
`SCOPE-DEVIATION`, `LICENSE-RISK`, `PLAN-CLAIM-UNVERIFIED`, and
`SOURCE-AMBIGUOUS` row across all six files) into this order: confirmed-wrong
math first, then `HANDROLLED-DUPLICATE` items — for each one, check it
against the implementation plan's "Performance acceptance gates" section
(no full dense materialization, no fused `d²` probe, no SVD outside final
truncation, no recomputed cached environment columns — used here only as a
map of what to check, not as authority) and note whether it plausibly
explains the reported downstream slowness — then `MISSING-VS-SOURCE`, then
`SUSPECT-UNVERIFIED`, then `PLAN-CLAIM-UNVERIFIED`/`SOURCE-AMBIGUOUS`, then
citation-only gaps (rows that are fine but under-cited).

- [ ] **Step 4: Merge cross-workstream overlaps**

Where two workstream files discuss the same call site (e.g. WS-backend and
WS-tree-probe both touching an incremental-QR call inside `src_tree.rs`),
combine into a single entry in the final report rather than duplicating it,
and note in the entry which workstreams found it.

- [ ] **Step 5: Assemble the full report**

Structure:

```markdown
# SRC Provenance Audit Report

**Audits:** feature/treetn-src at commit 9e018d4 (spec commit 7d574d7 added on top)
**Spec:** docs/plans/2026-08-28-src-provenance-audit.md

## Executive summary

[ranked findings from Step 3, one paragraph or bullet per finding, each
citing which workstream file and file:line has the full detail]

## WS-chain: src_chain.rs

[full table + detailed findings from ws-chain.md]

## WS-tree-probe: src_tree.rs, src_probe.rs

[full table + detailed findings from ws-tree-probe.md]

## WS-backend: tensor4all-tensorbackend additions

[full table + detailed findings from ws-backend.md]

## WS-core: tensor4all-core additions

[full table + detailed findings from ws-core.md]

## WS-integration: dispatch and public API

[full table + detailed findings from ws-integration.md]

## WS-tests: test coverage and tolerance integrity

[full table + detailed findings from ws-tests.md]
```

Write this to `docs/plans/2026-08-28-src-provenance-audit-report.md`.

- [ ] **Step 6: Verify structural completeness**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
FILE=docs/plans/2026-08-28-src-provenance-audit-report.md
test -f "$FILE" && echo "file exists" || echo "FAIL: file missing"
grep -q "## Executive summary" "$FILE" && echo "has executive summary" || echo "FAIL: no executive summary"
for f in chain tree-probe backend core integration tests; do
  grep -qi "$f" "$FILE" && echo "references ws-$f" || echo "FAIL: ws-$f not referenced"
done
grep -qiE "TBD|TODO|to be determined|FIXME" "$FILE" && echo "FAIL: placeholder text found" || echo "no placeholders"
wc -l "$FILE"
```

Expected: file exists, executive summary present, all six workstreams
referenced, no placeholders, line count roughly the sum of the six inputs
(a much smaller count means content was dropped during merge, not
condensed — go back and check).

- [ ] **Step 7: Commit**

```bash
cd /root/projects/tensor4all-rust/tensor4all-rs/.worktrees/treetn-src
git add docs/plans/2026-08-28-src-provenance-audit-report.md
git commit -m "docs(audit): synthesize SRC provenance audit report from six workstreams"
git log --oneline -8
```
