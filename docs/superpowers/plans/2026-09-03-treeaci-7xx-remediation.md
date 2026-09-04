# TreeACI 7xx Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the actionable TreeACI audit findings in issues #707--#718 by preserving numerical and transaction semantics while removing confirmed evaluator, ownership, cache, and arbitrary-degree performance defects.

**Architecture:** Keep TreeACI above `tensor4all-tensorbackend`; repair each operation at its owning layer and expose no direct tenferro dependency to TreeACI or TreeTN. Establish a correctness/performance matrix first, then land cut-local transaction and ownership fixes, followed by evaluator cache reuse, typed batch evaluation, budgeted grouped GEMM, and arbitrary-degree batching. The tenferro eager-leaf optimization and rectangular rank-revealing LU remain independent upstream tracks.

**Tech Stack:** Rust 2021, `tensor4all-treeaci`, `tensor4all-treetn`, `tensor4all-core`, `tensor4all-tensorbackend`, tenferro-backed column-major matrices, release-mode Cargo tests, Criterion benchmarks, and the existing provenance/performance worklog.

**Spec:** `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md` (audit authority and issue #707--#718 evidence)

## Global Constraints

- Refresh `origin/main` and verify the implementation branch contains the current base before changing code; do not use the stale audit base silently.
- Keep source code and documentation in English; keep this plan and user communication in Chinese when appropriate.
- Do not add direct `tenferro-*` dependencies outside `tensor4all-tensorbackend`; downstream code uses high-level tensorbackend/core seams.
- Preserve column-major layout, candidate/sample order, pivot selection, reduction order, error propagation, and `max_working_bytes` behavior.
- Keep generic Rust APIs for `f32`, `f64`, `Complex32`, and `Complex64`; do not add scalar-suffixed public APIs.
- Do not relax numerical tolerances. Dense whole-result tests materialize once and compare through tensor/matrix operations rather than recontracting every element.
- Every production optimization gets a scalar-reference differential test and a release-mode measurement. Every changed failure path gets an explicit error/rollback test.
- `cargo fmt --all` is required before each task is considered complete. Library code must not add `unwrap()` or `expect()`.
- Each issue is reviewed and committed independently where practical; overlapping files may be developed in parallel worktrees but are rebased and merged serially.
- The current branch already contains the #707 audit ledger, the exact f32/Complex32 dispatch correction from #717, the raw-center topology work-count gate, and the exactly-two-incoming batching fix. New work must extend those artifacts rather than replace them.

## Source compliance and literature lookup

Before changing an algorithm, numerical invariant, convergence rule, rank
policy, or complexity claim, perform a source lookup and record the result in
the audit worklog. The lookup may use the network and must prefer primary
sources: the original paper/preprint, author-maintained implementation, or
official upstream specification. Search snippets, wikis, blogs, and an AI
summary are discovery aids only, not algorithmic authority.

The full source must be cloned/downloaded before it is used: do not rely on an
abstract, search result, citation database, or browser excerpt. For a paper
with a Git repository, clone the exact revision. For an arXiv-only paper,
download both the complete PDF and the complete TeX/source archive from the
version being cited into ignored `target/literature/`; this is the canonical
full-text clone for the audit. For an official specification, clone the
upstream repository at the relevant commit and inspect the complete linked
specification file, not just its landing-page summary. Read every page,
appendix, and algorithm/pseudocode block of each cited paper. Record the page
count and a page-range checklist in the worklog.

Every nontrivial claim gets all of the following in the worklog or plan row:
source URL/DOI, local archive/repository path, exact commit/version, SHA-256
of downloaded archives where applicable, accessed date, PDF page number and
printed page number when they differ, section/subsection heading, exact
equation number, algorithm/pseudocode name and line range, or paragraph
heading plus a short identifying text anchor when no equation/algorithm
number exists. Also record a short paraphrase of the supported claim and its
authority label. Implementation measurements and downstream behavior are
cited to the repository/worklog/harness that produced them, not retroactively
attributed to a paper.

An evidence row is incomplete if it says only “the paper says” or gives only a
URL. A source locator must identify the concrete page/equation/pseudocode/text
segment that was actually read. If the source has no such locator, record the
page and paragraph heading and explain why the finer locator does not exist.

Use the following labels without silently upgrading one into another:

- `Paper`: the linked primary source states the algorithm/invariant being
  claimed. This does not make implementation-specific choices paper-backed.
- `simplett ACI`: behavior inherited from the existing simplett/chain ACI
  implementation and its tests.
- `Hiroshi review`: an explicit review/design decision recorded by Hiroshi or
  in the linked review artifact; quote/paraphrase only what the artifact says.
- `Tree generalization — re-derived`: a new tree case derived in this plan from
  the cited chain identity plus explicit tree invariants and differential
  tests. It must not be presented as pseudocode from the paper.
- `[AI Supplied]`: an AI-generated hypothesis, decomposition, gate, or design
  suggestion. It is not evidence or a citation; keep the tag until a human
  review or a primary source plus tests promotes the claim to a stronger label.

The initial bibliography for this plan is:

- [Ritter, “Fast elementwise operations on tensor trains with alternating
  cross interpolation,” arXiv:2604.00037](https://arxiv.org/abs/2604.00037):
  source for the ACI algorithm's stated error-control and rank-dependent
  complexity claims only. Clone the cited version using
  `https://export.arxiv.org/e-print/2604.00037v2` and
  `https://arxiv.org/pdf/2604.00037v2` before citing a page or equation.
- [Núñez Fernández et al., “Learning tensor networks with tensor cross
  interpolation: new algorithms and libraries,” SciPost Phys. 18, 104
  (2025)](https://doi.org/10.21468/SciPostPhys.18.3.104): source for TCI,
  rank-revealing behavior, and the reported stability motivation for partially
  rank-revealing LU; it does not authorize tree-specific pseudocode. Clone the
  cited v3 source using `https://export.arxiv.org/e-print/2407.02454v3` and
  `https://arxiv.org/pdf/2407.02454v3` before citing a page or equation.
- [tenferro-rs specification](https://tensor4all.org/tenferro-rs/spec/index.html):
  official upstream contract for tensor semantics/backend boundaries. Clone
  `https://github.com/tensor4all/tenferro-rs.git` at the dependency revision
  being audited and record the exact child specification file/commit when
  asserting a tenferro capability.

If a source is unavailable, record that fact and block every source-dependent
algorithmic implementation claim rather than filling the gap from memory.
`[AI Supplied]` or `Tree generalization — re-derived` may describe a clearly
non-authoritative hypothesis or a separately derived experiment, but neither
can substitute for the missing source. Do not copy source code or extensive
text; preserve the source's license/citation requirements and link it from the
worklog.

## Acceptance gate contract

Every numbered implementation step below is closed only after its gates pass;
the task-level gate block is the step's exit checklist, not an optional final
smoke test. The order is `C` (correctness), `E` (efficiency), then `R`
(regression/integration):

- `C`: compare the complete result against the existing scalar/oracle path or
  the pre-change implementation, including all affected scalar kinds, cache
  states, layouts, error/rollback paths, and boundary degrees. Use one dense
  materialization plus a whole-result residual where applicable; do not accept
  pointwise-only spot checks.
- `BC` (benchmark correctness): a benchmark or performance harness is
  admissible only after the complete affected-crate correctness matrix has
  passed in release mode, including integration tests, error/rollback paths,
  and any exercised downstream ACI stage. A smoke benchmark or one passing
  micro-case may diagnose a failure, but it cannot close `C`, `BC`, or `E`.
- `E`: run baseline and candidate in paired release-mode measurements with the
  same input, seed, backend/provider, thread affinity, and memory limit. Record
  at least three measurements per side (five for noisy end-to-end runs), report
  medians and the observed noise floor, and keep the raw result files.
- `E` is ordered after `BC`: if the full correctness gate fails, discard the
  corresponding timing/resource result and do not claim an efficiency gain.
- `E` passes for either a wall-time improvement above the measured noise floor
  or a causal resource improvement in the target path (for example fewer
  allocations/copies, lower peak bytes, fewer reconstructed messages, or fewer
  evaluated points) with no target-path time regression. A small measurable
  improvement is a valid pass; a large improvement is not required. A result
  at or below the noise floor with no objective resource reduction is not an
  efficiency pass and must remain diagnostic/incomplete.
- `R`: run the changed-crate release tests plus the relevant workspace or
  downstream checks, and verify that unaffected chain/simplett behavior,
  convergence/order, memory limits, and error messages did not regress.
- `CI`: after the subissue commit is pushed, inspect the new required GitHub
  checks together with the immediately preceding run's failure record. The
  previous failure must either be absent in the new run or be reproduced,
  fixed, and explicitly re-verified; a pending check is not a pass. Record the
  old/new head SHAs, failed job name and log anchor, local reproduction command,
  and final conclusion in the worklog. This gate is independent of local
  release tests and cannot be closed from a smoke job or a green unrelated job.

The following secondary gates are required whenever the task touches the
corresponding behavior; otherwise the task must record `N/A` and explain why:

- `N` numerical stability/convergence: use near-degenerate, rank-deficient,
  ill-conditioned, zero/identity, and dtype-stress fixtures. Check complete
  residuals plus rank, pivots, orthogonality, truncation error, and convergence
  trajectory/iteration count. Never pass by relaxing tolerances.
- `M` metamorphic semantics: reorder/split/merge batches, preserve duplicates,
  change equivalent tree rooting/orientation or edge enumeration, and compare
  after inverse permutation. This is separate from one fixed oracle fixture.
- `F` fallback parity: compare optimized and scalar/fallback routes for every
  affected dtype, provider, non-contiguous/trace mode, unsupported capability,
  over-budget route, and error class.
- `I` invalidation/retention: mutate numerical inputs or generation, exercise
  failed transactions, clear/reuse caches, and run repeated warm cycles. No
  stale values are allowed and retained memory must plateau/release according
  to the documented policy.
- `D` determinism: repeat the same seed/config at least three times. Require
  exact output/rank/pivot/callback equality for deterministic routes, or record
  a declared ULP/residual envelope for explicitly nondeterministic providers.
- `S` scaling law: measure at least 1x/2x/4x problem sizes and fit counters
  and timing separately to the expected complexity. A single-size speedup
  cannot prove an asymptotic improvement.
- `P` provenance/observability: every algorithmic claim has a source label and
  a locally cloned full source with hash plus a concrete page/equation/
  algorithm/pseudocode/paragraph locator; every performance claim has raw
  counters, command, configuration, and baseline/candidate identity. Unknown
  or AI-originated claims remain tagged.

Before measuring, each task must name its primary metric and minimum detectable
effect (MDE) as “above the measured noise floor” or as an explicitly recorded
resource-counter delta. Do not invent a universal percentage threshold after
seeing the result. If a correctness gate fails, do not use a performance win to
waive it; if an efficiency gate fails, do not claim the corresponding issue is
fixed merely because a refactor is cleaner.

## Downstream and benchmark gates

Use the requested downstream ACI workflow when a change affects TreeACI
evaluation, message injection, guard/recovery behavior, branch contraction, or
the end-to-end stage cost. The requested checkout is `../../gw-rs/sgw` relative
to the repository root. Resolve it before use and never edit its current dirty
working tree. For a pre-merge local run, make a clean temporary copy from its
`HEAD` (or use a separate clean worktree), apply a non-committed Cargo local
path patch for the affected tensor4all crates, and then run the existing stage
harness. The local copy is required because its checked-in git dependency pin
may not point at the candidate commit.

The repeatable downstream sequence is:

```bash
tensor4all_root="$(dirname "$(realpath "$(git rev-parse --git-common-dir)")")"
sgw_root="$(realpath "$tensor4all_root/../../gw-rs/sgw")"
sgw_tmp="$(mktemp -d)"
git -C "$sgw_root" archive HEAD | tar -x -C "$sgw_tmp"
# In "$sgw_tmp", add a temporary [patch] section for the local tensor4all
# crates touched by this task and update its temporary Cargo.lock if needed;
# do not commit either change or alter "$sgw_root".
cd "$sgw_tmp"
sgw_T="1.0"
SGW_RUN_TAG=aci-gate SGW_ACI_GLOBAL_GUARD=1 \
  ./run_r10_nblock_treeaci_ab.sh "$sgw_T"
run_dir="$sgw_tmp/runs/R10_nblock_T1.0_mu0.5_aci-gate/treeaci"
cargo run --release --locked --features isolation-diagnostics \
  --bin isolate_aci_stage -- "$run_dir" pi_rtau
cargo run --release --locked --features isolation-diagnostics \
  --bin isolate_aci_stage -- "$run_dir" sigma_rtau
```

The downstream correctness gate compares the TreeACI stage outputs with the
same-run reference/simplett or hadamard result, including `pi_rtau` and
`sigma_rtau` when those stages are exercised. The downstream efficiency gate
uses the stage elapsed time plus diagnostics such as evaluated points, cache
hits/misses, copied bytes, and peak/retained bytes. Use identical `R`, `T`,
`mu`, `U`, layout, guard settings, thread count, CPU affinity, and run count;
report medians and preserve logs. A stage isolation run is mandatory for the
affected stage, while the full A/B script is required for changes whose risk
is not local to one stage.

The sibling `tensor4all-benchmark` checkout is available as a supplementary
gate only when its workload is semantically comparable. It currently provides
SimpleTT/chain ACI cases (`elementwise_fourier`,
`elementwise_gauss2d_patched`, and related reports), not a TreeACI benchmark.
Therefore it may catch a shared tensorbackend or dense-operator regression,
but it must not be used to claim a TreeACI improvement for #708, #709, #711,
or #713. If the changed code is not exercised by one of its maintained cases,
record “not applicable” and ignore it. When applicable, run it from a clean
copy or temporary local patch with the same release/single-thread settings,
and apply its existing accuracy/holdout report as `C`; use its timed region
only as an additional `E` signal, never as the sole TreeACI gate.

## Dependency and Parallelism Map

```text
Task 0 baseline
    ├── Task 1: #707 audit closure and dependency matrix
    ├── Task 2: #717 dtype/cache/topology matrix
    ├── Task 3: #715 transaction tests and cut-local frame delta
    ├── Task 4: #716 owned dense/factorization seams
    ├── Task 5: #714 TreeACI local-update ownership
    ├── Task 6: #711 TreeTN branch-slice preparation
    └── Task 9: #712 grouped GEMM facade

Task 2 + Task 3 ──> Task 7: #710 directed-component keys/layouts/accounting
Task 7 ────────────> Task 8: #708 warm edge-cut evaluator
Task 8 + typed seam ─> Task 10: #709 typed batch evaluator and Guard plans
Task 5 + Task 9 ───> Task 11: #713 arbitrary-degree batching

Task 12: #718 final gates runs throughout and closes after Tasks 3--11
Upstream parallel track: tenferro #1704; optional rectangular rank-revealing LU
```

The dependency graph is logical, not a requirement that all work be serial. Tasks 3--6 and Task 9 can be implemented in parallel after the baseline. #1704 does not block any correctness task; it can affect the final Guard performance ceiling only.

## Current Baseline and Out-of-Scope Upstream Work

- `2dfabd6` contains the curated production provenance ledger and the exact four-way dtype dispatch correction. Task 1 verifies the ledger; it does not repeat the audit.
- The existing two-incoming frame kernel is the reference for Task 11. Task 11 targets three-or-more incoming components and must retain the scalar path as its oracle until the generalized kernel is proven.
- Tenferro already supplies owned tensor construction, owned matrix extraction, and offset-based grouped GEMM at the pinned revision. The current tensor4all work is to use or wrap those facilities at the correct layer.
- Tenferro issue [#1704](https://github.com/tensor4all/tenferro-rs/issues/1704) is a parallel performance fix for already-contiguous eager leaves. Do not change the dependency pin merely to begin this plan.
- Rectangular complete-pivot/rank-revealing LU is an optional tenferro feature. `tensor4all-core` retains its internal rectangular rrLU fallback, so it is not a prerequisite for TreeACI issue closure.

---

### Task 0: Refresh the branch and capture the baseline

**Files:**
- Read: `Cargo.toml`, `README.md`, `REPOSITORY_RULES.md`, `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`
- Generate: `target/api-dump/` through `xtask` (ignored build output; never commit)
- Record: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: current `origin/main`, current audit branch, and the generated API inventory.
- Produces: a named baseline commit, release test counts, benchmark configuration, and a confirmed list of changed crates.

**Acceptance gates:**
- `C0`: the API dump and literature/source authority map are complete and the
  listed release baseline tests pass; the worklog contains dense/reference
  outputs for the baseline fixtures.
- `E0`: the baseline benchmark is reproducible with paired runs, recorded
  medians, noise floor, and primary metrics; later tasks use this measurement
  anchor rather than an uncalibrated one-off.
- `P0`: every algorithmic claim made before implementation is tagged with one
  of the allowed provenance labels and has a URL/DOI or repository location.
- `R0`: only the baseline record changes, `git diff --check` passes, and no
  unrelated user change is staged.

- [ ] **Step 1: Refresh remote metadata and inspect worktree state.**

  ```bash
  git fetch origin
  git status --short --branch
  git log --oneline --decorate -12
  git merge-base --is-ancestor origin/main HEAD
  ```

  If the final command fails, update the implementation branch from `origin/main` before starting any code task and rerun the command. Preserve unrelated user changes.

- [ ] **Step 2: Perform the literature and source lookup before implementation.**

  Clone/download the complete ACI and TCI sources, not only their abstracts,
  and clone the tenferro repository at the audited revision. Keep these
  artifacts under ignored `target/literature/` and inspect the PDF and source
  archive for every page, appendix, equation, algorithm, and pseudocode block.
  The following commands establish the required full-text artifacts:

  ```bash
  literature_root="target/literature"
  mkdir -p "$literature_root/aci-2604.00037v2" \
    "$literature_root/tci-2407.02454v3" "$literature_root/tenferro"
  curl --fail --location --retry 3 \
    https://arxiv.org/pdf/2604.00037v2 \
    --output "$literature_root/aci-2604.00037v2/paper.pdf"
  curl --fail --location --retry 3 \
    https://export.arxiv.org/e-print/2604.00037v2 \
    --output "$literature_root/aci-2604.00037v2/source.tar"
  tar -xf "$literature_root/aci-2604.00037v2/source.tar" \
    -C "$literature_root/aci-2604.00037v2"
  curl --fail --location --retry 3 \
    https://arxiv.org/pdf/2407.02454v3 \
    --output "$literature_root/tci-2407.02454v3/paper.pdf"
  curl --fail --location --retry 3 \
    https://export.arxiv.org/e-print/2407.02454v3 \
    --output "$literature_root/tci-2407.02454v3/source.tar"
  tar -xf "$literature_root/tci-2407.02454v3/source.tar" \
    -C "$literature_root/tci-2407.02454v3"
  git clone https://github.com/tensor4all/tenferro-rs.git \
    "$literature_root/tenferro/repository"
  tenferro_revision="$(awk -F'#' \
    '/^source = "git+.*tenferro-rs/ {gsub(/"/, "", $2); print $2; exit}' \
    Cargo.lock)"
  test -n "$tenferro_revision"
  git -C "$literature_root/tenferro/repository" checkout --detach "$tenferro_revision"
  git -C "$literature_root/tenferro/repository" rev-parse HEAD \
    | tee "$literature_root/tenferro/audited-revision.txt"
  pdfinfo "$literature_root/aci-2604.00037v2/paper.pdf" \
    | tee "$literature_root/aci-2604.00037v2/pdfinfo.txt"
  pdfinfo "$literature_root/tci-2407.02454v3/paper.pdf" \
    | tee "$literature_root/tci-2407.02454v3/pdfinfo.txt"
  pdftotext -layout "$literature_root/aci-2604.00037v2/paper.pdf" \
    "$literature_root/aci-2604.00037v2/paper.txt"
  pdftotext -layout "$literature_root/tci-2407.02454v3/paper.pdf" \
    "$literature_root/tci-2407.02454v3/paper.txt"
  sha256sum "$literature_root"/*/paper.pdf "$literature_root"/*/source.tar \
    > "$literature_root/sha256sums.txt"
  ```

  The tenferro clone is pinned to the exact dependency revision extracted from
  `Cargo.lock` before inspection; record that resolved commit hash in the
  worklog together with the package/source block that selected it.
  Read the complete extracted source and PDF text, then create a page checklist
  covering `1..N` for each paper. For every claim, fill an evidence row with
  PDF/printed page, section heading, equation number, algorithm/pseudocode name
  and line range, or paragraph heading plus a short text anchor. Record the
  exact source file and line range for tenferro claims. A URL without one of
  these concrete locators fails this gate.

  Use this exact evidence-row shape in
  `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`:

  ```text
  claim_id | authority_label | source_url_or_doi | local_archive_or_repo |
  source_commit_or_version | archive_sha256 | accessed_date |
  pdf_page | printed_page | section_or_subsection |
  equation_number_or_algorithm_and_line_range |
  paragraph_heading_and_short_anchor | supported_claim |
  validation_test_or_benchmark
  ```

  `pdf_page` and `printed_page` are both required when they differ. If the
  source has no equation or algorithm number, record the page, heading, and a
  short identifying text anchor instead; `N/A` is allowed only with that
  explanation. Do not paste a long quotation into the worklog.

  ```bash
  git grep -nE 'Paper|simplett ACI|Hiroshi review|Tree generalization|\[AI Supplied\]' -- docs/worklogs
  grep -nE 'Equation|Algorithm|pseudocode|Theorem|Lemma|Complexity|stability' \
    "$literature_root"/aci-2604.00037v2/paper.txt \
    "$literature_root"/tci-2407.02454v3/paper.txt
  git diff --check
  ```

  Gate: no new algorithmic assertion proceeds to code until the complete
  source archive has been read and its evidence row contains the concrete
  page/equation/algorithm/pseudocode/paragraph locator plus one allowed
  authority label. An unresolved locator blocks the claim; it is not silently
  converted into paper authority.

- [ ] **Step 3: Generate and inspect the API inventory.**

  ```bash
  cargo run -p xtask --release -- api-dump
  sed -n '1,260p' target/api-dump/tensor4all_treeaci.md
  sed -n '1,260p' target/api-dump/tensor4all_treetn.md
  sed -n '1,220p' target/api-dump/tensor4all_core.md
  sed -n '1,220p' target/api-dump/tensor4all_tensorbackend.md
  ```

  Confirm the current inventory contains `TreeTNCachedEvaluator`, `EvaluationHint`, `Matrix::try_from_col_major_vec`, `Matrix::into_col_major_vec`, `mat_mul_owned`, `matrix_luci_factors_from_matrix_owned`, and the existing dtype predicates before relying on them in later tasks.

- [ ] **Step 4: Run the release baseline.**

  ```bash
  cargo test --release -p tensor4all-aci -p tensor4all-treeaci --no-fail-fast
  cargo test --release -p tensor4all-treetn --no-fail-fast
  cargo test --release -p tensor4all-core --no-fail-fast
  cargo test --release -p tensor4all-tensorbackend --no-fail-fast
  ```

  Expected: all existing tests pass. Record commit, Rust/Cargo versions, CPU/provider/thread settings, profile, seeds, and test counts in the audit worklog.

- [ ] **Step 5: Commit only the baseline record if it changed.**

  ```bash
  git add docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "docs(treeaci): record 7xx remediation baseline"
  ```

---

### Task 1: Close #707 with an explicit owner-layer matrix

**Files:**
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`
- Read: `target/api-dump/tensor4all_treeaci.md`, `target/api-dump/tensor4all_treetn.md`, `target/api-dump/tensor4all_core.md`, `target/api-dump/tensor4all_tensorbackend.md`

**Interfaces:**
- Consumes: Task 0 API inventory and the existing line-range provenance ledger.
- Produces: one table mapping every #707 operation to its owning crate, existing seam, planned issue, tenferro status, and correctness/performance evidence.

**Acceptance gates:**
- `C1`: every #707 row has an owning layer, authority label, affected scalar
  kinds, and a concrete oracle or test reference; no operation is left as an
  unexplained “TreeACI issue”.
- `E1`: every performance-sensitive row names its baseline metric and planned
  counter/timing gate; no performance claim is accepted without evidence.
- `R1`: the ledger contains no stale API names or unresolved placeholders and
  its dependency decisions agree with the API dump and repository rules.
- Secondary gates: `P`; every matrix row records its exact source location or
  remains explicitly `[AI Supplied]`/`Tree generalization — re-derived`.

- [x] **Step 1: Add the owner-layer matrix without changing algorithm authority.**

  The table must include planning, key construction, cold/full-hit/partial-hit/miss/reconstruction, center contraction, dtype dispatch, raw/generic kernels, dense construction, matrix conversion, GEMM/grouped GEMM, LUCI/LU, slicing/views, cache accounting, and AD/trace preservation. Each row must use one of the existing provenance labels: `Paper`, `simplett ACI`, `Hiroshi review`, `Tree generalization — re-derived`, or `[AI Supplied]`.

- [x] **Step 2: Mark the dependency decisions explicitly.**

  Record that #712 uses the already-present tenferro grouped GEMM through a new tensorbackend facade, #716 fixes tensor4all-core/tensorbackend ownership seams, #709 is a TreeTN API seam with optional tenferro #1704 performance support, and rectangular LU is not on the critical path.

- [x] **Step 3: Validate the artifact and close the documentation task.**

  ```bash
  grep -nE '#707|#708|#709|#710|#711|#712|#713|#714|#715|#716|#717|#718|tenferro' docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git diff --check
  ```

  The final worklog must contain no unresolved placeholder and must state its audited commit and limitations.

- [x] **Step 4: Commit the audit closure.**

  ```bash
  git add docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "docs(treeaci): close dependency-side audit matrix"
  ```

---

### Task 2: Complete #717's dtype, cache, and topology regression matrix

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs` test module, only where the existing tests are the owning location
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`
- Modify: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs` for Guard-facing typed coverage
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: the existing `TreeTNCachedEvaluator::evaluate_batched_with_hint`, raw/generic dispatch, cache options, and current four-way dtype fix.
- Produces: generic fixture helpers and deterministic tests covering the full acceptance matrix; no new public API.

**Acceptance gates:**
- `C2`: f32, f64, Complex32, and Complex64 cold, warm, partial-hit,
  reordered/duplicate, capacity, clear/reuse, Y-comb, and unequal-bond cases
  match the ordinary evaluator within existing dtype-specific tolerances;
  mixed-dtype inputs preserve the ordinary evaluator's promotion behavior or
  return the same contextual typed error.
- `E2`: each affected dtype/topology shows either a median time win above
  noise or a measured reduction in work/allocations/cache misses; f64/c64 do
  not regress beyond the recorded noise floor.
- `R2`: all cache/error tests pass in release mode and no raw-kernel
  genericization is merged without the corresponding matrix evidence.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; record PASS or N/A with a
  reason for each in the worklog.

This task completes a regression matrix around the already-landed #717
dispatch correction, so a test-only result records `E2` as a baseline/no-
regression characterization rather than claiming a production speedup. A
future raw-kernel performance change requires its own before/after resource
gate and must not be smuggled into this coverage task.

- [x] **Step 1: Add reusable four-scalar-kind fixtures and ordinary-evaluator oracles.**

  Add one fixture builder parameterized by `f32`, `f64`, `Complex32`, and `Complex64`, plus a dense-result helper that materializes `TreeTN::evaluate` once. Add tests named:

  ```rust
  #[test]
  fn cached_evaluator_four_scalar_kinds_match_tree_evaluate() { /* cold and warm */ }

  #[test]
  fn cached_evaluator_reordered_duplicate_and_partial_hit_batches_match() { /* exact order */ }

  #[test]
  fn cached_evaluator_capacity_zero_over_budget_and_clear_reuse_match() { /* policy paths */ }

  #[test]
  fn cached_evaluator_path_y_comb_and_unequal_bond_layouts_match() { /* topology */ }
  ```

  Use declared dtype-appropriate tolerances and assert values/residuals, not merely shape or finiteness.

- [x] **Step 2: Add error-path coverage.**

  Cover invalid point shapes, invalid physical coordinates, mixed dtype conversion, unsupported raw dispatch, wide key construction, and working-memory overflow. Assert typed errors and verify no panic.

- [x] **Step 3: Run the matrix before any raw-kernel genericization.**

  ```bash
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo test --release -p tensor4all-treeaci global_guard --no-fail-fast
  cargo bench -p tensor4all-treetn --bench cached_evaluator -- treetn_cached_scalar_kind
```

  Cargo's `bench` subcommand selects the optimized bench profile and rejects
  `--release`; the focused filter above records the affected scalar kinds while
  avoiding unrelated high-rank legacy cases that can take minutes per sample.
  Record 32-bit timing separately. Do not add duplicated f32/Complex32 raw
  kernels unless this measurement and the end-to-end gate show a need.

- [x] **Step 4: Commit the coverage matrix.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treetn crates/tensor4all-treeaci/src/global_guard/tests/mod.rs docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "test(treetn): cover cached evaluator dtype and cache matrix"
  ```

---

### Task 3: Implement #715 cut-local frame growth and failure-atomic commits

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Modify: `crates/tensor4all-treeaci/src/transaction.rs`
- Modify: `crates/tensor4all-treeaci/src/state.rs` only for staged-state plumbing
- Modify: `crates/tensor4all-treeaci/src/global_guard.rs` for shared injection growth
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`
- Test: `crates/tensor4all-treeaci/src/state/tests/mod.rs`
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: append-only `SampleArena`, existing `InputFrameStore::extend`, `commit_edge_proposal`, and the current checkpoint/rollback mechanism.
- Produces: an internal cut-local extension seam with the shape `InputFrameStore::extend_new_samples(inputs, problem, arena, previous_counts) -> Result<Self>`, which reuses unchanged `Rc<DirectedFrame<T>>` records and computes only newly interned sample ranges.

**Acceptance gates:**
- `C3`: edge commit and Guard injection produce the same complete tensors,
  frame IDs, candidate order, pivots, and errors as the reference; injected
  failures leave the original state logically unchanged.
- `BC3`: before any timing is admissible, the complete release-mode TreeACI
  correctness matrix must pass: frames, transaction, state, global Guard,
  all crate unit/integration tests, and the affected downstream ACI stage when
  Guard/message injection is exercised. Focused tests and smoke benchmarks are
  diagnostic only and cannot satisfy this gate.
- `E3`: old-prefix copying and full-store extension calls decrease on the
  scaling fixture, and paired timing or allocation/byte counters show a
  measurable improvement above noise with no memory-budget regression. Run
  this gate only after `BC3` passes and retain the raw paired measurements.
- `R3`: cut-local and full-extension paths, zero/new/duplicate samples, long
  chains, branch degrees, and affected downstream ACI stages pass release
  checks. Run the downstream stage gate when Guard or message injection is
  exercised.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; rollback, invalidation,
  repeated-cycle retention, and source authority must be explicitly recorded.

- [x] **Step 1: Add red frame-growth differential tests.**

  Add `extend_new_samples_matches_full_rebuild_on_chain_and_branch` and `extend_new_samples_computes_only_new_ranges`. The tests must compare every affected and unaffected directed frame against a fresh `from_samples` store and assert that existing frame allocations remain shared where their sample range is unchanged.

- [x] **Step 2: Add red transaction rollback tests before changing commit order.**

  Add failure injection for operator failure, invalid factor shape, sample interning failure, frame budget overflow, and output metadata validation. Snapshot output tensors, graph/topology metadata, canonical region, sample arena, candidates, pivots, frames, and generation before the staged operation; assert byte/value/equality preservation after each error.

- [x] **Step 3: Implement cut-local frame staging.**

  Record `previous_counts` from the existing store, identify directed edges whose arena record count grew, allocate only the new frame ranges, and use dependency order only for those new records. Keep `records`, retained-byte accounting, memo capacity, and over-budget errors checked with `checked_*` arithmetic. Unchanged frame payloads must remain `Rc`-shared.

- [x] **Step 4: Stage output replacement before mutating state.**

  Factor indices, dense tensors, bond metadata, and all validation must complete before the commit mutation. Keep `SampleArena::checkpoint` around the complete fallible staging block. After staging succeeds, apply the already-validated edge replacement, candidate/pivot updates, frame swap, and generation update in a no-failure commit section. Do not clone the complete frame store or candidate sets for rollback after the last fallible operation.

- [x] **Step 5: Replace both edge-commit and Guard-injection full extensions.**

  `commit_edge_proposal` and `inject_global_pivots` must call the same cut-local extension seam. Empty input remains validated first and returns without changing generation or arena state. The operation must not duplicate the separate #686 random-start evaluation work.

- [x] **Step 6: Run focused and scaling tests.**

  ```bash
  cargo test --release -p tensor4all-treeaci frames --no-fail-fast
  cargo test --release -p tensor4all-treeaci transaction --no-fail-fast
  cargo test --release -p tensor4all-treeaci state --no-fail-fast
  cargo test --release -p tensor4all-treeaci global_guard --no-fail-fast
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  ```

  This full release matrix is the `BC3` gate; a smoke-only result cannot close
  it. Only after it passes, record paired old-prefix copies, new values,
  frame-extension calls, chain length, branch degree, and complete-ACI
  numerical/convergence parity for `E3`.

- [ ] **Step 7: Commit #715.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/transaction.rs crates/tensor4all-treeaci/src/state.rs crates/tensor4all-treeaci/src/global_guard.rs crates/tensor4all-treeaci/src/{frames,state,global_guard}/tests docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "perf(treeaci): make frame growth cut-local"
  ```

---

### Task 4: Implement #716 ownership-preserving dense and factorization seams

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/tensor_element.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-core/src/defaults/idx_tensor.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-core/src/matrix_luci.rs` only if an existing owned facade needs a checked conversion seam
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treeaci/src/transaction.rs`
- Test: `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs`
- Test: `crates/tensor4all-core/tests/linalg_factorize.rs` and `crates/tensor4all-core/tests/tensor_basic.rs`
- Test: existing TreeTN and TreeACI evaluator/frame tests

**Interfaces:**
- Consumes: tenferro's existing `from_vec_col_major`, `Matrix::try_from_col_major_vec`, `Matrix::into_col_major_vec`, and `matrix_luci_factors_from_matrix_owned`.
- Produces: `TensorElement::dense_native_tensor_from_col_major_owned(data: Vec<Self>, dims: &[usize])`, used by `IdxTensor::from_dense` without an intermediate slice copy; existing public `from_dense` semantics and column-major order remain unchanged.

**Acceptance gates:**
- `C4`: owned dense construction, matrix round trips, square/rectangular
  factorization, all four scalar kinds, and the leaf fallback match the
  pre-change values, shapes, column-major order, and errors.
- `E4`: the target conversion/factorization path shows a measurable reduction
  in copies/allocations or paired release time above noise, with no peak-byte or
  normal tensor-operation regression. Use the sibling benchmark only for a
  semantically comparable shared ACI/backend case.
- `R4`: tensorbackend/core release tests, public API inventory, and affected
  downstream output construction pass without exposing a direct tenferro
  dependency downstream.
- Secondary gates: `N`, `F`, `D`, `P`; include non-contiguous/trace fallback
  parity and the exact tenferro specification page used for each claim.

- [x] **Step 1: Add a failing owned-constructor test for all four scalar kinds.**

  Exercise `IdxTensor::from_dense` with `f32`, `f64`, `Complex32`, and `Complex64`, then read once with `with_dense_slice`/typed extraction and assert exact column-major values. Add shape mismatch and checked dimension-overflow tests for the new owned bridge path.

- [x] **Step 2: Add the owned trait/bridge path.**

  Implement the new trait method with `NativeTensor::from_vec_col_major(dims.to_vec(), data)`. Keep the existing borrowed method for callers that truly borrow. Change `IdxTensor::from_dense` to call the owned method after validating indices and payload length; do not add a TreeACI-local constructor.

- [x] **Step 3: Add red factorization ownership tests.**

  For rectangular and square LU/CI full-rank paths, compare factor reconstruction and reported rank against the existing reference for real and complex values. The test must exercise `try_from_col_major_vec`, owned rrLU/rrLUCI, and `into_col_major_vec` paths, including the rectangular internal rrLU fallback.

- [x] **Step 4: Replace the conversion round trips.**

  In `factorize_lu_with_options` and `factorize_ci_with_options`, construct matrices with `Matrix::try_from_col_major_vec`, pass them to the owned factorization facade, and extract outputs with `into_col_major_vec`. Remove the zero-fill nested copy and `matrix_to_vec` calls when ownership is available. Keep the existing rectangular fallback because tenferro `full_piv_lu` remains square-only.

- [x] **Step 5: Replace non-escaping `to_vec` calls with borrowed reads.**

  Convert raw leaf, leaf-center, tensor-backed leaf-center, internal-center, and TreeACI output-padding paths to `with_dense_slice`, retaining materialization fallback for non-contiguous/backend-resident tensors. Keep closures short enough that no borrowed slice escapes.

- [x] **Step 6: Verify ownership and numerical behavior.**

  ```bash
  cargo test --release -p tensor4all-tensorbackend --no-fail-fast
  cargo test --release -p tensor4all-core --test linalg_factorize --no-fail-fast
  cargo test --release -p tensor4all-core --test tensor_basic --no-fail-fast
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  ```

  Use allocation/copy counters or controlled before/after timings to show that the old allocation is reused; API compilation alone does not close the issue. Run the complete correctness matrix before the ignored paired-release measurement; a filtered smoke test is insufficient for this gate.

- [x] **Step 7: Commit #716.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-tensorbackend crates/tensor4all-core crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treeaci/src/transaction.rs
  git commit -m "perf(core): preserve ownership across dense factorization paths"
  ```

---

### Task 5: Implement #714 TreeACI local-update ownership and packed batches

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Modify: `crates/tensor4all-treeaci/src/local_update.rs`
- Modify: `crates/tensor4all-treeaci/src/state.rs` only if the prepared-core lifetime is stored there
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`
- Test: `crates/tensor4all-treeaci/src/local_update.rs` test module or the owning TreeACI integration test
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: `mat_mul_owned`, existing one/two-incoming frame kernels, `PreparedCore`, candidate cache semantics, and Task 3's cut-local frame owner.
- Produces: an internal packed candidate-frame batch with explicit `bond_dim`, candidate count, column-major `Vec<T>` payload, and candidate-order mapping; local update consumes it directly without `Vec<Vec<T>>` extraction and repacking.

**Acceptance gates:**
- `C5`: scalar and packed local updates match for real/complex values, all
  listed cache/budget/axis/bond cases, candidate order, and frame dimensions.
- `BC5`: the complete affected TreeACI release correctness matrix and the
  standalone ACI release matrix pass before the paired measurement is
  admissible; the ignored measurement is never the only correctness test.
- `E5`: paired chain and branch runs show a measurable reduction in extracted
  or repacked values, copies/allocations, peak bytes, or release time above
  noise; no target-path time regression is hidden by a microbenchmark-only win.
- `R5`: reduction order, ranks, pivot errors, dense residuals, and working-byte
  limits remain unchanged. Run the downstream ACI stage gate for affected
  branch/local-update workloads.
- Secondary gates: `N`, `M`, `F`, `I`, `S`, `P`; candidate-order invariance and
  packed-versus-scalar evidence must be retained with the benchmark records.

- [x] **Step 1: Add scalar-vs-packed differential tests.**

  Add real/complex tests for one-incoming, two-incoming, leaf, unequal-bond, alternate-axis, cache-hit, cache-miss, and over-budget paths. Compare the packed batch once against the scalar frame oracle and assert candidate order and frame dimensions.

- [x] **Step 2: Add an immutable oriented-core preparation test.**

  Instrument `prepare_cores`/oriented matrix construction and assert that one `(input, directed_edge)` identity is prepared once per `InputFrameStore` owner, while a changed sample set reuses the immutable core representation.

- [x] **Step 3: Route eligible products through owned GEMM.**

  Change the stored-frame one-incoming path, candidate-frame one-incoming path, and final local row-by-column product to consume `Matrix` values with `mat_mul_owned`. Do not change the GEMM decomposition or floating-point reduction order.

- [x] **Step 4: Replace the candidate cache value cycle.**

  Store packed candidate payloads keyed by the existing candidate identity. On a hit, return a shared packed view/owner in requested order; on a miss, retain or return the same packed allocation according to the existing budget policy. Charge payload capacity and metadata separately, and return an explicit over-budget uncached batch rather than cloning through nested vectors.

- [x] **Step 5: Keep the working-memory contract exact.**

  Charge simultaneously live row/column batches, GEMM output, candidate scratch, cache insertion, and local operator buffers with checked arithmetic. A cache optimization must never hide a duplicate RHS or exceed `max_working_bytes`.

- [x] **Step 6: Run correctness and paired ACI measurements.**

  ```bash
  cargo test --release -p tensor4all-treeaci frames --no-fail-fast
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  cargo test --release -p tensor4all-aci --no-fail-fast
  ```

  Re-run the 32-site two-input `chi=256` chain and a branched fixture. Record owned-vs-borrowed products, oriented-core packs, candidate-cache hits/misses, extracted values, repacked values, peak working bytes, output ranks, pivot errors, and dense residuals.

- [x] **Step 7: Commit #714.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/local_update.rs crates/tensor4all-treeaci/src/state.rs docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "perf(treeaci): keep local update batches owned and packed"
  ```

---

### Task 6: Implement #711 reusable TreeTN branch-slice preparation

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/mod.rs` only if an option is re-exported
- Modify: `crates/tensor4all-treetn/Cargo.toml` only if a test feature is required; do not add a new dependency
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`

**Interfaces:**
- Consumes: `with_dense_slice`, specialized raw branch contraction, immutable evaluator lifetime, and configured execution context.
- Produces: an internal `PreparedBranchSlice<T>` cache keyed by directed orientation, physical coordinate, and exact dtype; each entry owns a packed column-major matrix or records a budget refusal and uses the existing fallback.

**Acceptance gates:**
- `C6`: every axis arrangement, unequal-bond branch, dtype, cache state, and
  budget fallback matches the scalar/generic result and reports correct bytes.
- `E6`: repeated degree-three/higher branch setup shows a measurable prepared
  slice reuse win above noise or fewer child/slice copies, with no chain
  regression and no hidden full-network materialization.
- `R6`: evaluator release tests, cache policy, memory limits, and the affected
  downstream isolated ACI stage pass. A downstream run is required because
  branch setup is an end-to-end TreeACI hot path.
- `CI6`: after pushing the #711 commit, inspect the preceding PR CI run (in
  particular the earlier CI_rs / Maintenance failure) and the new required
  checks. Re-run the exact failed job command locally when available; if the
  old failure recurs, #711 remains open until the cause is fixed and the new
  run passes. Record the old/new run IDs, head SHAs, job/log anchors, and
  conclusion in the worklog; a pending remote check is not accepted as a pass.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; mark any provider/layout
  combination not exercised by the benchmark as N/A rather than extrapolating.

- [x] **Step 1: Add a scalar reference test for all axis arrangements.**

  Cover degree-three and higher branch nodes, unequal incident bond dimensions, physical axes in non-canonical order, `f64` and `Complex64`, and compare the prepared-slice path to the existing scalar/generic result with a declared tight tolerance.

- [x] **Step 2: Add budget and fallback tests.**

  Verify that a cache budget large enough retains one entry per required physical slice, a zero budget retains none, and an insufficient budget falls back without allocating a hidden full-network dense tensor. Assert reported retained/peak bytes.

- [x] **Step 3: Prepare immutable slices once.**

  Build the physical slice and axis permutation once per evaluator owner using `with_dense_slice`; reuse the resulting `Matrix` for all message groups with the same key. Preserve the existing loop and reduction order in the branch contraction.

- [x] **Step 4: Measure child gather separately.**

  Keep diagnostics for decode, child gather, prepared-slice lookup, and GEMM. Remove child copies only when the new owner representation makes them unnecessary; otherwise retain and report them.

- [x] **Step 5: Run the branch regression.**

  ```bash
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo bench --release -p tensor4all-treetn --bench cached_evaluator
  ```

  Compare paired chain/comb runs at bond 64, 128, and 256. Require no chain regression, bounded memory, matching outputs, and improvement in the repeated degree-three branch-slice setup cost.

- [x] **Step 6: Commit #711.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/benches/cached_evaluator.rs
  git commit -m "perf(treetn): reuse prepared branch core slices"
  ```

- [x] **Step 7: Run the remote CI regression gate after pushing #711.**

  ```bash
  git fetch origin
  old_run_id="<preceding PR CI run>"
  old_head_sha="<preceding PR head SHA>"
  gh pr checks 722
  gh run view "$old_run_id" --json headSha,conclusion,jobs
  gh run view "$old_run_id" --log-failed
  # After the new required checks finish:
  gh pr checks 722
  ```

  Compare the old and new required-check conclusions. Confirm that the
  previous CI_rs / Maintenance failure is absent or that its exact repair has
  a passing replacement job. Record the old/new run IDs, head SHAs, job name,
  log line, local reproduction, and final result in the worklog before marking
  `CI6` and Task 6 complete. A pending run is not a pass, but it does not block
  starting the next independent subissue; re-check it as a parallel CI gate.

---

### Task 7: Implement #710 directed-component keys, layouts, and cache accounting

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Read/use: `crates/tensor4all-core/src/index_key/`
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`
- Coordinate: issue #626 cache policy semantics

**Interfaces:**
- Consumes: checked `KeyBuilder`, directed-message cache keys, `PackedMessageCache`, and Task 2's policy matrix.
- Produces: one immutable `DirectedComponentPlan` and `MessageCacheLayout` per directed edge, plus documented owned/logical retained-byte accounting.

**Acceptance gates:**
- `C7`: composed and direct keys are identical for empty, duplicate, reordered,
  wide, nested, invalid, and overflow inputs; partial-hit/miss behavior and
  cache policy match the existing evaluator.
- `E7`: metadata construction/storage and lookup show the expected directed-
  component reduction or a measurable retained-byte/lookup-time improvement
  above noise; the 51.2 ns versus 29.1 ns primitive is not itself an
  evaluator-speedup claim.
- `R7`: cache hit ratios, eviction/over-budget semantics, numerical outputs,
  and path/Y/comb scaling remain stable in release tests.
- Secondary gates: `M`, `I`, `D`, `S`, `P`; stale-generation and repeated
  clear/reuse behavior must be measured separately from key microbenchmarks.

- [x] **Step 1: Add metadata-count regression tests.**

  For a 16-site chain visited at every center, assert that retained component layouts and physical position lists scale with the `2E` directed components rather than with center × component combinations. Test path, Y, comb, and unequal-bond trees.

- [x] **Step 2: Add key equivalence tests.**

  Build keys both from the existing direct assignment encoder and from checked local/child `KeyBuilder` composition. Assert exact equality for empty, singleton, duplicate, reordered, wide, and nested component assignments. Invalid dimensions and overflow must return errors.

- [x] **Step 3: Move layout ownership to directed components.**

  Replace center-indexed physical layout retention with `(from, to)` component plans. Keep numerical message caches keyed by direction. Rooting-specific traversal state may remain per center only when it controls traversal, not when it merely repeats a directed component layout.

- [x] **Step 4: Correct cache accounting.**

  Track packed payload capacity, key storage, map entries/buckets using a documented estimate, and all cache-owned vectors/metadata. Distinguish logical payload bytes from owned retained bytes in statistics. Preserve capacity-zero, over-budget, and uncached-miss semantics.

- [x] **Step 5: Run policy and scaling tests.**

  ```bash
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo test --release -p tensor4all-core --test common_basic --no-fail-fast
  cargo bench -p tensor4all-treetn --bench cached_evaluator
  ```

  Record key construction time as a primitive diagnostic only; do not claim the 51.2 ns versus 29.1 ns key result as an evaluator speedup.

- [ ] **Step 6: Commit #710.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/benches/cached_evaluator.rs
  git commit -m "perf(treetn): deduplicate directed evaluator metadata"
  ```

---

### Task 8: Implement #708 complete warm edge-cut evaluation

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: Task 7 directed-component plans, lazy cache lookup, `EvaluationHint`, current raw/generic messages, and the scalar `TreeTN::evaluate` oracle.
- Produces: an internal edge-cut assembly path that combines two directed component messages with an `O(points * chi_edge)` final dot-product work count on a full warm batch.

**Acceptance gates:**
- `C8`: cold, warm, partial, reordered, duplicate, miss, path, Y, comb, and
  unequal-bond batches match `TreeTN::evaluate` for real and complex values.
- `E8`: warm work count is exactly `points * chi_edge`, descendant work is zero
  on a parent hit, and paired warm end-to-end timing or cache/reconstruction
  counters improve above noise without a cold-path regression.
- `R8`: raw-center complexity, sweep ranks/residuals, Guard behavior, and
  affected downstream `pi_rtau`/`sigma_rtau` stages pass. The downstream
  isolated-stage gate is mandatory for this evaluator change.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; cold, warm, fallback,
  and stage-isolated provenance must all be represented in the report.

- [ ] **Step 1: Add red warm-hit work-count tests.**

  Add tests named `warm_edge_cut_skips_descendant_reconstruction` and `warm_edge_cut_assembly_scales_with_edge_bond`. The first must assert zero descendant calls when the requested parent message is already cached; the second must use a zero-sized test scalar or deterministic counter and assert exactly `points * chi_edge` final assembly visits.

- [ ] **Step 2: Add cold/partial/reordered differential tests.**

  Compare ordinary `TreeTN::evaluate` and cached evaluation for cold, full-hit, partial-hit, miss, duplicate-point, reordered-batch, and repeated-batch calls on path, Y, comb, and unequal-bond fixtures. Include real and complex values and preserve declared tolerances.

- [ ] **Step 3: Implement hit-before-recursion.**

  Request the exact directed component key first. If a parent message is a cache hit, return its packed columns without walking descendants or reconstructing unused child messages. On a miss, recurse only into dependencies needed by the missing keys and merge results in original point order.

- [ ] **Step 4: Implement edge-cut final assembly.**

  Select a valid cut associated with the hinted varying center, obtain both directed component messages, and perform the final bond contraction. Keep the current vertex-center raw kernel available for cold/fallback paths. The tree-cut identity must be documented in the worklog as `Tree generalization — re-derived`/`[AI Supplied]`, not attributed to the paper's absent tree pseudocode.

- [ ] **Step 5: Run paired warm/cold measurements.**

  ```bash
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo bench --release -p tensor4all-treetn --bench cached_evaluator
  cargo test --release -p tensor4all-treeaci global_guard --no-fail-fast
  ```

  Require an end-to-end default-Guard improvement, no predeclared cold regression, preserved raw-center `d * product(chi_e)` kernel coverage, and matching sweep ranks/residuals.

- [ ] **Step 6: Commit #708.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treetn/src/treetn/cached_evaluator.rs crates/tensor4all-treetn/benches/cached_evaluator.rs docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "perf(treetn): reuse complete warm edge-cut contractions"
  ```

---

### Task 9: Implement #712 budgeted shared-operand grouped GEMM facade

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/matrix.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/backend.rs` only for configured-provider dispatch
- Test: `crates/tensor4all-tensorbackend/src/matrix.rs` tests and `tests/bench_einsum_native.rs`
- Modify: `crates/tensor4all-treeaci/src/frames.rs` only when Task 11 adopts the facade
- Modify: `crates/tensor4all-tensorbackend/README.md` if the crate documents public matrix operations there

**Interfaces:**
- Consumes: tenferro's offset-based grouped GEMM capability through the tensorbackend-owned bridge.
- Produces: generic column-major `GroupedGemmJob` and `grouped_mat_mul_shared`/owned execution APIs whose public types do not expose tenferro descriptors or internals.

**Acceptance gates:**
- `C9`: all four scalar kinds, offsets, empty/singleton/shared operands,
  validation errors, provider dispatch, and working-budget cases match
  individual GEMMs and preserve column-major results.
- `E9`: the facade's microbenchmark shows a measurable shared-operand copy,
  allocation, peak-byte, or runtime improvement above noise. Production
  TreeACI adoption additionally requires an end-to-end branch win; if that
  need gate fails, retain only the tested facade and record adoption deferred.
- `R9`: tensorbackend tests pass, no direct tenferro dependency leaks, and any
  optional sibling benchmark result is reported only as shared-backend
  evidence, never as a TreeACI claim.
- Secondary gates: `F`, `D`, `S`, `P`; shared-operand benefit must be causal,
  and unsupported-provider/over-budget behavior must match the fallback.

- [ ] **Step 1: Define and test the public job contract.**

  The job descriptor must contain checked output, left, and right offsets plus `rows`, `contracted`, and `cols`. Document that jobs may share input spans, output spans must be disjoint unless an explicit reduction mode is added, all buffers are column-major, and validation occurs before backend execution.

- [ ] **Step 2: Add red validation tests.**

  Cover f32/f64/Complex32/Complex64, empty and singleton jobs, mismatched dimensions, out-of-bounds offsets, checked overflow, overlapping outputs, shared RHS/LHS spans, unsupported dtype/layout, zero working budget, and configured provider/thread behavior. Compare every valid job to an individual GEMM result.

- [ ] **Step 3: Implement the tensorbackend facade.**

  Validate all spans and the total peak working memory before entering the backend. Translate the validated generic jobs to tenferro grouped descriptors inside `tensor4all-tensorbackend`; preserve the caller-owned buffer/lifetime contract and return backend errors with context.

- [ ] **Step 4: Add the need-gated TreeACI adoption probe.**

  Keep the existing two-incoming decomposition and reduction order. Add a test/benchmark-only switch that replaces duplicated-RHS batching with shared-operand grouped jobs, then compare output, peak memory, and runtime. Promote the path only if the end-to-end branch gate wins without a chain regression.

- [ ] **Step 5: Verify and commit the facade separately from adoption.**

  ```bash
  cargo test --release -p tensor4all-tensorbackend --no-fail-fast
  cargo test --release -p tensor4all-treeaci frames --no-fail-fast
  cargo fmt --all
  git add crates/tensor4all-tensorbackend
  git commit -m "feat(tensorbackend): add budgeted shared grouped GEMM"
  ```

  If the TreeACI need gate fails, retain the tested facade and record that TreeACI adoption is deferred; do not force a slower path into production.

---

### Task 10: Implement #709 typed batch evaluation and reusable Guard plans

**Files:**
- Modify: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Modify: `crates/tensor4all-treetn/src/treetn/mod.rs` only for module/API organization
- Modify: `crates/tensor4all-treeaci/src/global_guard.rs`
- Modify: `crates/tensor4all-treeaci/src/scalar.rs` only for the typed conversion seam
- Test: `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`
- Test: `crates/tensor4all-treeaci/src/global_guard/tests/mod.rs`
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`

**Interfaces:**
- Consumes: Task 8 evaluator cache path and internal `CachedScalar` representation.
- Produces: generic `TreeTNCachedEvaluator::evaluate_batched_typed<T: TensorElement>(values, hint) -> Result<Vec<T>, TreeTNOperationError>`; the existing `evaluate_batched` remains an `AnyScalar` compatibility wrapper. Guard uses the typed method for all four scalar kinds.

**Acceptance gates:**
- `C10`: typed and compatibility results match for all four scalar kinds,
  ordering, duplicates, hints, errors, callbacks, and cache accounting; the
  typed path creates no rank-zero result tensors.
- `E10`: the typed Guard path shows a measurable reduction in short-lived
  assignment/result allocations or paired end-to-end time above noise, with no
  callback, memory, or cold-path regression. The scalar-wrapper microbenchmark
  alone cannot pass this gate.
- `R10`: TreeTN/TreeACI release tests and the affected downstream Guard stage
  pass; public rustdoc has a runnable asserted example and states shape,
  column-major, dtype, ownership, and error contracts.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; typed and compatibility
  routes must be compared after output mutation and cache invalidation.

- [ ] **Step 1: Add the typed API differential test.**

  Add `evaluate_batched_typed_matches_any_scalar_wrapper_for_all_scalar_kinds`, asserting exact/tight-tolerance equality, batch order, duplicate points, hint behavior, evaluated-point accounting, and dtype mismatch errors. The public method's rustdoc must describe shape, column-major layout, dtype requirements, errors, ownership, and a runnable asserted example.

- [ ] **Step 2: Refactor the internal result boundary.**

  Make the evaluator's internal raw/generic paths return `Vec<CachedScalar>` (or an equivalent lightweight typed carrier) and convert directly to `Vec<T>` in `evaluate_batched_typed`. Keep `AnyScalar` construction only in the compatibility wrapper. No rank-zero `IdxTensor` may be constructed per result on the typed path.

- [ ] **Step 3: Reuse immutable Guard plans across changing output values.**

  Separate immutable topology, directed-component plans, key layouts, and dtype-independent assignment metadata from numerical message caches. Rebuild/invalidate only numerical messages when output tensors change. Keep input evaluator reuse and output evaluator lifetime explicit in `find_global_pivots`/`GuardOutputEvaluator`.

- [ ] **Step 4: Reduce assignment allocations after the type/lifetime seam is stable.**

  Replace repeated cloned local-coordinate and child-assignment vectors with compact checked keys or borrowed slices where they do not escape. Preserve batch order and hint behavior. Add allocation counters for the two-point Guard workload and assert a material reduction from the current minimum 3,072 short-lived assignment vectors per 16-site/64-point call.

- [ ] **Step 5: Run typed Guard and end-to-end gates.**

  ```bash
  cargo test --release -p tensor4all-treetn treetn::cached_evaluator --no-fail-fast
  cargo test --release -p tensor4all-treeaci global_guard --no-fail-fast
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  cargo bench --release -p tensor4all-treetn --bench cached_evaluator
  ```

  Require no `AnyScalar`/rank-zero construction in the typed Guard path, unchanged callback counts and errors, and a paired end-to-end Guard improvement. The scalar-wrapping microbenchmark alone cannot close #709.

- [ ] **Step 6: Commit #709.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treetn/src/treetn crates/tensor4all-treetn/benches/cached_evaluator.rs crates/tensor4all-treeaci/src/global_guard.rs crates/tensor4all-treeaci/src/scalar.rs
  git commit -m "perf(treetn): add typed cached batch evaluation"
  ```

---

### Task 11: Implement #713 arbitrary-degree candidate-frame batching

**Files:**
- Modify: `crates/tensor4all-treeaci/src/frames.rs`
- Modify: `crates/tensor4all-treeaci/src/local_update.rs` only if the generalized packed batch changes its consumer
- Test: `crates/tensor4all-treeaci/src/frames/tests/mod.rs`
- Test: `crates/tensor4all-treeaci/tests/rank_scaling.rs` or a new focused integration test under `tests/`
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: the existing zero/one/two-incoming kernels, Task 5 packed candidate batch, and optionally Task 9 grouped GEMM through tensorbackend.
- Produces: an internal `incoming_batch_matrix(core, outgoing_axis, incoming_axes, physical_offset, frame_matrices) -> Result<PackedCandidateBatch<T>>` that handles degree 0, 1, 2, 3, and greater-than-3 incoming components with checked Cartesian dimensions.

**Acceptance gates:**
- `C11`: degree 0/1/2/3/4+, real/complex, unequal bonds, axis orders,
  duplicate/reordered candidates, cache states, and budget fallbacks match the
  scalar oracle, including candidate order and dense residuals.
- `E11`: the 3+-incoming scalar cliff is measurably reduced above noise, or
  intermediate allocations/bytes and evaluated work decrease with no target
  time regression. The full-cross cost and peak memory are reported even when
  the lazy/block redesign is deferred.
- `R11`: degree-two specialization, pivot order, ranks, reduction order,
  errors, and working-memory compliance remain unchanged. Run downstream ACI
  validation when its production tree exercises degree 3+.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; every unsupported or
  over-budget higher-degree route must have explicit fallback parity.

- [ ] **Step 1: Add scalar-reference tests before generalizing.**

  Cover degree 0, 1, 2, 3, and 4 incoming edges; real and complex data; unequal bond dimensions; nontrivial axis order; duplicate and reordered candidates; cache hit/miss; and working-memory limits. Compare once against the existing scalar `candidate_frame`/`compute` oracle and assert candidate/sample order.

- [ ] **Step 2: Implement the generalized intermediate layout.**

  Represent the Cartesian product of incoming candidate columns with checked prefix strides. Build only the required intermediate matrices, charge all simultaneously live buffers, and preserve the current scalar reduction order. Use the existing exactly-two-incoming kernel as the degree-two specialization and keep the scalar path as a fallback while higher-degree tests are red.

- [ ] **Step 3: Add degree-aware dispatch.**

  Route all degree ≥3 calls through the generalized path only after differential tests pass. If a provider or memory budget cannot support the intermediate, return the existing typed limit/error or use the scalar fallback according to the documented routing contract; never silently allocate the full tree.

- [ ] **Step 4: Separate the full-cross need gate.**

  Measure contraction time normalized by `d * product(chi_e)` and by candidate-product size. If complete edge-cross materialization remains dominant after removing the scalar cliff, record the result and stop this task at the measurement boundary. A lazy/block/pivot-search redesign requires a separate derivation and convergence review.

- [ ] **Step 5: Run full-degree verification.**

  ```bash
  cargo test --release -p tensor4all-treeaci frames --no-fail-fast
  cargo test --release -p tensor4all-treeaci --test rank_scaling --no-fail-fast
  cargo test --release -p tensor4all-treeaci --no-fail-fast
  ```

  Require no change in pivot selections, ranks, reduction order, errors, or dense residuals. Record the 3+-incoming speedup and peak memory in the audit worklog.

- [ ] **Step 6: Commit #713.**

  ```bash
  cargo fmt --all
  git add crates/tensor4all-treeaci/src/frames.rs crates/tensor4all-treeaci/src/local_update.rs crates/tensor4all-treeaci/src/frames/tests crates/tensor4all-treeaci/tests docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "perf(treeaci): batch arbitrary-degree frame contractions"
  ```

---

### Task 12: Define and enforce #718 scaling and end-to-end gates

**Files:**
- Create or modify: `crates/tensor4all-treeaci/benches/aci_scaling.rs`
- Modify: `crates/tensor4all-treetn/benches/cached_evaluator.rs`
- Modify: `crates/tensor4all-treeaci/Cargo.toml` only if the benchmark target requires explicit configuration
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`
- Modify: CI configuration only if a bounded smoke gate can run within repository CI limits; retain heavy studies as a documented runbook
- Read-only external validation when applicable: `../../gw-rs/sgw` and the
  sibling `tensor4all-benchmark` checkout; do not commit changes there

**Interfaces:**
- Consumes: all task-level counters and public behavior from Tasks 2--11.
- Produces: reproducible paired release benchmarks and deterministic complexity gates for TreeACI, Guard, cached evaluation, branch contraction, cache memory, callbacks, and working-memory policy.

**Acceptance gates:**
- `C12`: complete ACI parity covers TreeACI versus the established reference,
  all affected scalar kinds/topologies, sweeps, ranks, pivots, callbacks, and
  dense residuals; deterministic work-count gates are asserted independently
  of timing.
- `E12`: every claimed issue has a baseline/candidate median, noise floor, and
  primary metric. Each accepted optimization has either a measurable time win
  above noise or an objective causal resource reduction, including small wins;
  no issue is closed on a single favorable run.
- `R12`: run the affected downstream isolated stage and, where scope is
  end-to-end, the full `../../gw-rs/sgw` A/B workflow. Use
  the sibling benchmark only for comparable shared ACI/backend cases and
  mark TreeACI-inapplicable cases explicitly. Complete repository CI gates,
  docs, and rules review pass.
- Secondary gates: `N`, `M`, `F`, `I`, `D`, `S`, `P`; the final report must
  include PASS/N/A status and evidence links for every task-level gate.

- [ ] **Step 1: Add independent scaling fixtures.**

  Scale chain length, fixed input bond dimension, active candidate/output rank, coordination number, unequal incident bonds, and batch size independently. Include chain, exactly-two-incoming branch, three-or-more-incoming branch, cold evaluator, warm evaluator, and over-budget cases.

- [ ] **Step 2: Add deterministic complexity tests.**

  Keep the existing raw-center `d * product(chi_e)` gate. Add the Task 8 warm assembly `points * chi_edge` gate. Add candidate-product accounting for Task 11. Use counters rather than wall-clock assertions for exponents.

- [ ] **Step 3: Add complete-ACI parity measurements.**

  Preserve the default-Guard comparison against simplett ACI, including the existing bond-256 reference, callbacks/evaluated points, sweeps, maximum output rank, pivot errors, dense residuals, cache hits/misses/evictions, retained/peak bytes, and working-limit compliance.

- [ ] **Step 4: Record reproducibility metadata.**

  Every benchmark report must include baseline/candidate commits, hardware, CPU affinity, provider/thread settings, seeds, build profile, warm-up count, repetitions, statistic/confidence interval, noise gate, and threshold. A reported slow workload must either reproduce or be marked inconclusive with an owner and exact run command.

- [ ] **Step 5: Run the downstream ACI and applicable sibling benchmark gates.**

  For changes touching evaluator, message injection, Guard/recovery, branch
  contraction, or end-to-end stage cost, execute the clean-copy sequence in
  **Downstream and benchmark gates**. Run `isolate_aci_stage` for every
  affected `pi_rtau`/`sigma_rtau` stage and use the full
  `run_r10_nblock_treeaci_ab.sh` comparison whenever the change crosses a
  stage boundary. Assert complete output parity first, then compare paired
  medians for elapsed time, evaluated points, cache statistics, copied bytes,
  and peak/retained bytes.

  If the changed seam is shared by a maintained case in the sibling
  `tensor4all-benchmark` checkout, run the official single-core profile from a
  clean copy or temporary local patch, for example:

  ```bash
  tensor4all_root="$(dirname "$(realpath "$(git rev-parse --git-common-dir)")")"
  benchmark_root="$(realpath "$tensor4all_root/../tensor4all-benchmark")"
  cd "$benchmark_root"
  benchmark_profile="aci-shared-gate"
  BENCH_CPU_CORE=0 scripts/run_all.sh "$benchmark_profile"
  python3 scripts/report.py "result/$benchmark_profile"
  ```

  Use its existing accuracy/holdout report as a correctness signal and its
  timed region as supplementary shared-backend evidence. If the case does not
  exercise the changed code, write “not applicable”; do not use these
  SimpleTT/chain ACI runs as a TreeACI efficiency gate.

- [ ] **Step 6: Run the complete pre-PR verification.**

  ```bash
  cargo fmt --all
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets -- -D warnings -D clippy::missing_errors_doc -D clippy::missing_panics_doc
  cargo nextest run --cargo-profile ci --workspace --exclude tensor4all-hdf5
  cargo test --profile ci -p tensor4all-hdf5
  cargo test --doc --profile ci --workspace -j 8
  cargo doc --workspace --no-deps
  python3 scripts/repository-rules-review.py --base main --worktree --dry-run
  python3 scripts/test-repository-rules-review.py
  ```

- [ ] **Step 7: Commit the gate and close #718.**

  ```bash
  git add crates/tensor4all-treeaci/benches crates/tensor4all-treetn/benches docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md
  git commit -m "bench(treeaci): define 7xx scaling and performance gates"
  ```

---

### Task 13: Track the non-blocking tenferro upstream lane

**Files:**
- External coordination: `tensor4all/tenferro-rs#1704`
- Read-only dependency pin: workspace `Cargo.toml` tenferro revision
- Modify only if accepted upstream behavior is required: `Cargo.toml`, `Cargo.lock`, and affected bridge tests
- Modify: `docs/worklogs/2026-09-01-treeaci-provenance-performance-audit.md`

**Interfaces:**
- Consumes: Task 10 typed evaluation benchmark and tenferro's existing eager tensor construction API.
- Produces: an independently reviewed tenferro change that removes the unnecessary session/contiguous conversion for already-contiguous, non-traced eager leaves; no TreeACI direct dependency.

**Acceptance gates:**
- `C13`: the tenferro reproducer and tensor4all bridge produce identical
  values, shapes, dtype behavior, errors, and traced/non-contiguous fallback
  behavior before and after the upstream change.
- `E13`: the eager-leaf session/contiguous conversion or its allocation/byte
  counter improves measurably above noise; if not, keep the issue as evidence
  only and do not force a dependency update.
- `R13`: TreeACI remains correct and buildable at the pinned revision, the
  upstream change is reviewed independently, and any pin update is run through
  the full release/downstream gates. Rectangular LU stays a separate optional
  experiment with its own correctness and rank gate.
- Secondary gates: `N`, `F`, `D`, `P`; upstream claims, implementation claims,
  and AI-supplied hypotheses must remain separately labeled.

- [ ] **Step 1: Reproduce the lower-layer cost against the pinned revision.**

  Run the existing tenferro #1704 reproducer or add it in the tenferro repository. Measure empty session cost and `from_tensor_in` cost for a small already-contiguous tensor with tracing disabled. Record the pinned revision and provider.

- [ ] **Step 2: Keep TreeACI usable without the upstream change.**

  Confirm Task 10's typed path avoids per-result rank-zero construction. The TreeACI release tests and correctness gates must pass with the current pinned revision; do not add a local tenferro reach-through workaround.

- [ ] **Step 3: If upstream lands, update the pin in a separate reviewed change.**

  Re-run tensorbackend bridge tests, TreeTN evaluator tests, TreeACI Guard tests, and Task 12 performance gates after updating the revision. Earlier green results do not carry across a dependency update.

- [ ] **Step 4: Record rectangular LU separately.**

  Keep the `tensor4all-core` rectangular rrLU fallback and document the square-only tenferro `full_piv_lu` limitation. Open or update a separate tenferro feature request only if eliminating the fallback becomes an explicit TreeACI requirement.

---

## Final Self-Review Checklist

- [ ] Every #707--#718 acceptance item maps to a task and a release-mode test or benchmark.
- [ ] Every numbered step has a correctness gate, an efficiency gate, and a regression/integration gate; each efficiency gate records paired medians, noise, and its primary metric.
- [ ] Small efficiency improvements are accepted only when measurable above noise or supported by a causal resource-counter reduction; no single-run or microbenchmark-only claim closes an issue.
- [ ] Literature/source lookup happens before algorithmic implementation; every nontrivial claim records URL/DOI, version/date, exact location, supported claim, and authority label.
- [ ] `Paper`, `simplett ACI`, `Hiroshi review`, `Tree generalization — re-derived`, and `[AI Supplied]` are not conflated; `[AI Supplied]` is never treated as evidence.
- [ ] Every applicable secondary gate `N/M/F/I/D/S/P` is PASS or explicitly N/A with a reason and evidence location.
- [ ] No task introduces a direct downstream `tenferro-*` dependency.
- [ ] No task changes candidate order, pivot order, reduction order, dtype semantics, or tolerances without an explicit differential gate.
- [ ] All four scalar kinds appear in the relevant tests; branch and unequal-bond layouts are covered.
- [ ] Failure atomicity covers output, topology/canonical metadata, samples, candidates, pivots, frames, and generation.
- [ ] Cache accounting includes capacity and metadata, not just numerical payload.
- [ ] Warm evaluator complexity and raw-center topology complexity are measured as separate properties.
- [ ] Affected end-to-end changes run the clean-copy `../../gw-rs/sgw` stage/A-B gate without modifying its dirty checkout.
- [ ] `tensor4all-benchmark` is used only for semantically comparable shared ACI/backend coverage; TreeACI-specific issues are explicitly marked not applicable there when appropriate.
- [ ] The plan contains no unresolved placeholders or doctest escape hatches.
- [ ] Upstream tenferro work is clearly parallel and non-blocking for the tensor4all-rs mainline.
