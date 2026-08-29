# SRC Provenance Audit Report

**Audits:** feature/treetn-src at commit 9e018d4 (spec commit 7d574d7 added on top)
**Spec:** [`docs/plans/2026-08-28-src-provenance-audit.md`](2026-08-28-src-provenance-audit.md)

This report is the synthesis of six independent workstream audits
(`docs/plans/audit-workstreams/ws-{chain,tree-probe,backend,core,integration,tests}.md`),
each of which was written, reviewed by an independent scoped reviewer, and in
several cases corrected across one or two fix rounds. The per-workstream
sections below reproduce those documents in full — their provenance tables,
derivations, and non-findings — so this single file is the complete audit
record. The executive summary is the only part that is new: it ranks findings
across all six, resolves the overlaps where two workstreams looked at the same
code, and states the one question the audit could not answer.

## How to read the verdicts (taxonomy normalization)

The spec defines eleven verdict tokens. Five of the six workstreams
(`ws-chain`, `ws-core`, `ws-tree-probe`, `ws-integration`, `ws-tests`) record
trivial, no-derivation-owed units — imports, accessors, parameter-bundle
structs, re-export lines — as `DERIVED-VERIFIED` with a "(trivial plumbing)"
qualifier. `ws-backend` instead introduced a sentinel, `NO-FINDING`, for the
identical situation, and documented that choice inside its own file. Both
conventions were reviewed and are internally consistent, but they are **not the
same taxonomy**, so raw token counts are not comparable across files.

**Normalization used in this report:** `ws-backend`'s `NO-FINDING` rows are
treated as equivalent to the other workstreams' trivial-plumbing
`DERIVED-VERIFIED` rows. Both mean "inspected, nothing owed, no finding."
Neither is counted as a substantive finding in the ranking below. Where this
report speaks of "`DERIVED-VERIFIED` findings needing review," it means only
those rows carrying an actual derivation and an open question — never the
trivial-plumbing rows of either convention. `ws-backend`'s `F7` row is marked
`WITHDRAWN` by that workstream itself and is excluded from the ranking, as its
own document instructs.

## Executive summary

### 0. Confirmed-wrong math: none

Across all six workstreams, **no confirmed-wrong mathematics was found.** Every
derivation that was re-derived from Tier-1 sources held:

- The chain recursion in `src_chain.rs` matches paper §3.1–§3.5 / Algorithm 1
  step by step (WS-chain, provenance table and derivations).
- The from-scratch rooted-tree generalization in `src_tree.rs`, which no Tier-1
  source covers at all, was independently re-derived from the paper's chain
  forward/backward environment recursion and **holds**, including the
  non-obvious asymmetry between the processed side (`projected_children`, the
  tree analogue of `S^(j)`) and the unprocessed side (the probed complement
  message, the analogue of `C^(j-1)`) — WS-tree-probe, derivation D-1.
- The adaptive-rank estimator matches paper Appendix C `eq:err-est` and
  `eq:norm_est` exactly, the Appendix C.3 block inverse-adjoint update matches
  symbol for symbol including the sign and the zero block, and the complex
  Householder construction is unitary with real `τ ∈ [1,2]` — WS-backend,
  derivations D1, D2, D3, D5, D7, D8.
- The MPO-MPO factorized probe matches Hiroshi's 2026-08-24 comment **exactly**,
  including contraction order, and the production path never forms a fused
  `d²` physical object — WS-tree-probe, derivation D-8 and the fused-probe
  audit.
- The QR-only claim from Hiroshi's 2026-07-29 comment holds without exception:
  every SVD call site in the branch is a `final_svd`-gated `truncate_impl` on
  the fully-assembled output. `src_chain.rs` has exactly two (the adaptive and
  fixed twins of the same code), `src_tree.rs` has exactly one, and
  `backend.rs`/`incremental_qr.rs`/`src_probe.rs` have none.
- **The audit's original motivating suspicion — a hand-rolled SVD interface in
  a codepath the literature never calls for — is negative.** WS-core's
  high-priority check of `factorize.rs` (the file most likely to hide one)
  found **no new factorization logic at all**: every `factorize_*` function in
  it pre-exists unchanged on `origin/main`, and the entire diff is a mechanical
  struct-literal → `FactorizeResult::new(...)` call-site migration forced by a
  new private field. WS-backend found zero SVD calls, zero eigendecompositions,
  and zero general dense inverses in its three files.

### 1. `HANDROLLED-DUPLICATE` — and the open question that decides whether it explains the slowness

**F1 (WS-backend, high). `incremental_qr.rs` (+1005 lines, new) re-implements
LAPACK's `geqrf`/`larfg`/`larf`/`ormqr`/`orgqr` as element-indexed scalar Rust,
duplicating the backend QR the branch already has and already uses.** This is a
real, confirmed `HANDROLLED-DUPLICATE` against the implementation plan's
explicit ban on "a hand-written QR, SVD, matrix inverse, or LAPACK wrapper."
The mathematics is correct (WS-backend D2/D5/D8) and the reference-repository
citations in its header are accurate to the line — this file was written with
the sources open. The defect is scope and engineering, not hallucination.

Three qualifications matter, and none of them may be dropped:

1. **Scope: it is reachable only on the adaptive (`rtol.is_some()`) path.**
   WS-backend independently verified — and corrected an earlier revision of its
   own document that claimed otherwise — that fixed-rank SRC
   (`rtol.is_none()`) never calls `factorize_probe_columns` at all, and routes
   through `factorize_qr_full_rank` → `qr_with` → tenferro's real
   `Tensor::qr`. The `rtol.is_none()` branch inside `src_probe.rs:717` is a
   dead defensive guard in a function fixed-rank never enters.
   Narrowing the scope **strengthens** the duplication finding: a working
   backend QR path sits directly beside the hand-rolled one, so "no backend QR
   was available" is not a defence.
2. **It was built against its own plan's explicit deferral (F2, F3,
   `SCOPE-DEVIATION`).** The 2026-08-26 plan calls incremental Householder QR
   "a later optimization gate," and its performance gate #7 requires "a
   recorded profile." No profile exists anywhere on the branch: the one
   post-fix benchmark "reached dependency compilation but did not enter
   measurement." Separately, the hand-rolled path works on host `Matrix<T>`
   with per-element indexing, so it **cannot dispatch to a GPU backend at all**
   — the specific capability Hiroshi's 2026-07-29 comment says the QR-only hot
   path should unlock (#553).
3. **The timing evidence does NOT establish this as the cause of the reported
   downstream slowness.** The only measurements on the branch show fixed SRC at
   0.0030×–0.0053× of fit speed and adaptive SRC at 0.018×–0.025× — i.e. fixed
   SRC measured roughly **4×–6× slower** than adaptive SRC, and fixed SRC is
   the configuration that never touches the hand-rolled QR. The slowest
   measured configuration is the one *not* using `IncrementalQr`. An earlier
   revision of WS-backend read these ratios as corroborating F1; that inverted
   their meaning and was corrected.

**The audit's single most consequential open question, which none of the six
workstreams can answer: which SRC mode does `gw-rs` actually invoke —
fixed-rank (`rtol.is_none()`) or adaptive (`rtol.is_some()`)?** If `gw-rs` runs
SRC fixed-rank, F1 is a real code-quality, duplication, and scope finding but
is **irrelevant** to the "very slow downstream pipeline" symptom that motivated
this audit, and the cause lies in WS-tree-probe / WS-core territory (see F-4
and F-5 below). If it runs adaptive, F1 becomes a live candidate cause — and
even then the existing timing data is simply silent about it, since no
adaptive-vs-adaptive comparison with and without `IncrementalQr` exists.
`gw-rs` is outside every workstream's file list and no profiling was run in
this pass (explicitly out of scope per the spec).

**Recommended next step, ranked first among all recommendations: determine
`gw-rs`'s actual SRC invocation mode before concluding anything about the
performance symptom.** That single fact decides whether F1 is a performance
finding or purely a hygiene finding.

*Cross-workstream merge (item 4 of the synthesis brief):* WS-core examined the
same `IncrementalQr`/`src_error_estimate` surface from the consumer side and
explicitly deferred the duplication verdict to WS-backend rather than issuing a
second one. WS-core's own files (`idx_tensor.rs`, `tensor_like.rs`) contain no
hand-written linear algebra — every incremental-QR unit in them is a thin
dtype-dispatch/bookkeeping bridge. But WS-core also corrected an earlier draft
that had described `IncrementalQr`, `IncrementalQrScalar`, `SrcErrorEstimate`
and `src_error_estimate` as "pre-existing backend primitives": they are
**branch-new**, confirmed absent from `origin/main`. So this is **one finding,
not two**: the branch-new backend types carry WS-backend's
`HANDROLLED-DUPLICATE` verdict scoped to the adaptive path, and WS-core's
bridge code is the (clean) consumer of them.

**F-3 (WS-tree-probe, low). `standard_normal` in `src_probe.rs:126-130`
hand-rolls Box-Muller** where `rand_distr::StandardNormal` is already a
workspace dependency used for exactly this purpose in
`tensor4all-core/src/defaults/idx_tensor.rs:188`. Minor: the plan's prohibition
names QR/SVD/inverse/LAPACK, not RNG transforms. Two attached observations: it
discards the sine branch (2 uniforms per sample), and `ProbeBank` uses
`StdRng`, whose stream is not stable across `rand` releases, which qualifies
the module's own reproducibility claim.

### 2. Performance findings that are live candidates for the slowness symptom regardless of the open question

These two are in the tree path, which both fixed-rank and adaptive SRC use,
so — unlike F1 — their relevance does not depend on `gw-rs`'s mode.

**F-4 (WS-tree-probe, high, `SCOPE-DEVIATION` vs plan performance gate 1). The
tree path materializes the full probed local pair with all `2·deg(v)` virtual
bonds open at every node, before contracting any environment message.** For a
degree-2 node with `χ_A = χ_B = χ` that is a `χ⁴·l` object; for a degree-3
branch point, `χ⁶·l`; and they are all live simultaneously in one `HashMap`, so
peak memory is `O(n · χ^{2·deg} · l)`. This is precisely the intermediate that
`src_probe.rs`'s own comment at lines 424-429 says must be avoided, and that
`contract_prefix_with_probed_site_pair_batch_range` correctly avoids — the
chain path uses that function at every interior site; the tree path does not.
WS-tree-probe verified the `O(χ⁴)` claim in that comment by explicit index
counting (D-9): prefix-first peaks at `χ²d²l` against pair-first's `χ⁴l`, and
`d ≪ χ` in any regime where SRC is worth running. **An `O(χ⁴l)` per-site
intermediate where the cost model predicts `O(χ²dl)` is exactly the shape of
discrepancy the reported symptom describes.** Not a correctness bug — the
messages are contracted down afterwards and the tree tests pass.

**F-5 (WS-tree-probe, high, `SUSPECT-UNVERIFIED`).
`EnvironmentCache::batched_environments` is keyed by sketch width alone.**
Because `site_max_width` varies per edge (it depends on the child's row
dimension and the parent edge's cut dimension), each distinct width triggers a
fresh full directed-message sweep over all `2|E|` messages — each of which
itself rebuilds the F-4 probed local pairs. Worst case, `O(n²)` message
contractions where `O(n)` suffices. Note this makes the **fixed-rank** path the
worse of the two: the per-column adaptive path is keyed by column, grows
monotonically, and does not have the problem. That is the wrong way round, and
it is consistent with the timing data showing fixed SRC slower than adaptive.
A single `BatchedEnvironment` computed at `max_e(site_max_width)` would serve
every edge, since all batches start at probe column 0 and narrower batches are
prefixes of wider ones (D-3). No Tier-1 source and no derivation in the code or
plan justifies the width-keyed design.

**F4 and F8 (WS-backend, medium/low).** Within `incremental_qr.rs`, `append`
performs three full reallocate-and-copy passes with no capacity/doubling policy
(the reference pre-allocates `size = 2n` and doubles), giving `O(mχ̄²/3)` of
pure copy traffic under the paper's adaptive schedule; and `from_factors` burns
a complete `O(mp²)` Householder refactorization on a matrix already known to be
orthonormal — WS-backend's D7 proves the resulting `R_q` is provably a signed
identity, so the work recovers information the caller already had. Both inherit
F1's scope: adaptive path only.

**Against the plan's "Performance acceptance gates"** (used as a map, not as
authority, per the spec): gate 1 (no full dense materialization) is the one
F-4 lands on — arguably, since a per-node `χ^{2·deg}·l` object is not the full
network, but it is unambiguously the same class of intermediate the gate and
the file's own comments exist to prevent. Gate 2 (no fused `d²` probe) is
**honoured** — WS-tree-probe's grep sweep found no index-fusing operation
anywhere in the production path. Gate 3 (fixed-rank hot path = contractions,
QR, projection, optional final SVD) is met in shape everywhere, though on the
adaptive path the QR is not the backend's (F1/F3). Gate 4 (cached environment
columns not recomputed during adaptive growth) is **honoured** on the tree path
and pinned by a real test — F-5's repeated sweeps are a *batched-path*
key-coarseness problem, not a gate-4 violation.

### 3. `MISSING-VS-SOURCE`

All substantive `MISSING-VS-SOURCE` findings are in test coverage (WS-tests);
WS-tree-probe and WS-chain explicitly recorded none for the algorithm files
themselves. WS-backend records one against the reference implementations (F4,
the absent capacity/doubling policy, covered above).

Test-coverage gaps, from WS-tests §7:

- **No numerical dense-oracle correctness test for adaptive-rank contraction
  anywhere in the diff.** The adaptive tests check bond-dimension bounds and
  `validate_ortho_consistency()` only; the dense-oracle methodology used so
  well for fixed-rank is not applied to adaptive mode.
- **Complex64 coverage is a single 2-node fixed-rank chain test.** No
  Complex64 for branched/star topology, adaptive mode, or the
  `apply_linear_operator`/`partial_contract` entry points.
- **No dedicated MPO-MPO test in the twelve test files.** The only MPO-MPO
  checks live in `src_probe.rs`'s inline test module (WS-tree-probe's file) —
  they are genuine independent-oracle numerical checks against Hiroshi's `E_k`
  formula, but they assert values, not shapes, so they are not the *structural*
  regression guard the plan's category name promises.
- **No reproducibility tests at all** (same seed → identical output; adaptive
  expansion preserves the first `p` columns; different seeds meet the same
  residual gate).
- **No SRC-specific control-flow/error tests.** `src_chain.rs` has explicit
  `bail!`s for incompatible topologies and empty chains; the only tests
  exercising those paths dispatch through `Naive`/`Zipup`, never `Src`.
- **The "rank cap actually binding" branch is untested.** WS-tests read the
  fixture directly (§5i) and found the cap is set to exactly the fixture's own
  maximum exact rank, with a `<=` assertion — so the test cannot distinguish
  "cap reached and enforced" from "cap never binding."
- **No test pins `IncrementalQr`'s output against `qr_backend`'s** (WS-backend
  cross-workstream note 3). Given F1, a differential test against the backend
  QR is the obvious missing check, and its absence is why a scalar
  re-implementation of LAPACK could land unremarked.

### 4. `SUSPECT-UNVERIFIED`

- **F-1 (WS-tree-probe, medium — correctness risk).** The
  `.or_else(|| messages.get(&(parent, neighbor)))` fallback in both
  `directed_messages` functions (`src_tree.rs:562-565` and the identical
  `:624-627`) is **structurally unreachable** — WS-tree-probe proved the
  guarantee in D-1 step 2, and the code's own adjacent comment states it
  correctly. If it ever fired it would substitute a message flowing in the
  *opposite* direction, producing a silently wrong sketch that no shape check
  would catch, bypassing the correct `ok_or_else` error path. Defensive code
  whose only possible effect is to convert a loud failure into a quiet wrong
  answer.
- **F5, F6, F8–F12 (WS-backend).** The rank-deficiency skip policy and its
  `32·ε·max(m,k)·max(‖·‖_F,1)` tolerance have no basis in the paper, the
  Python, the C++, or any Hiroshi comment — the arithmetic around it is correct
  (D4), and the consumer compensates by short-circuiting before the estimator,
  but the end-to-end behaviour is right only by a convention split across two
  crates and two workstreams that nothing documents. Plus: a dead `norm_sq`
  accumulation recomputed later (F9), unreachable `f32`/`Complex32` impls
  (F10), an undocumented hard-error contract (F11), and two doc claims that do
  not hold — the `IncrementalQr` struct doc claiming "the same state layout
  used by the reference implementation" (F6), which **contradicts line 8 of the
  same file**, and `new`'s `# Errors` attributing failure to "the backend QR
  factorization" in a function that calls no backend QR (F12).
- **WS-tests, test-quality (not code-correctness).** `validate_ortho_consistency()`
  is called after nearly every SRC test as the apparent fulfilment of the
  "canonical/isometric edge invariants" category, but WS-tests read its
  implementation line by line and confirmed it checks only that the
  `ortho_towards`/`canonical_region` **bookkeeping metadata** is self-consistent
  — it never computes `QᴴQ` or touches tensor data. A grep of the whole diff
  for `is_unitary`/`is_isometric`/`isometry` returns zero hits: **no test
  anywhere performs a genuine numerical isometry check.** Also flagged: a test
  named `..._column_major_dense_payloads` whose full-buffer round-trip cannot
  discriminate column-major from row-major (§5g), and the Complex32 half of the
  single-precision estimator test, which asserts only `.is_finite()`.

### 5. `SCOPE-DEVIATION`

**The `PrefixCache` trait ask was not honored** — one finding, found
independently by WS-chain (which owns the struct's file) and WS-integration
(which owns the authoritative verdict). Hiroshi's 2026-08-27T12:56:57Z comment
asked to "put the cache behind a small trait (something like
`PrefixCache: fn extend(piece), fn get(k)`) instead of hard-coding a Vec built
in a forward loop"; the 15:38 correction retracted only the scan/parallelism
framing and explicitly reconfirmed "The implementation ask is unchanged: a
small cache trait, flat list first." What exists is `struct PrefixCache` with
`prefixes: Vec<Vec<T>>`, grown in a forward `while` loop — i.e. exactly the
pattern named as the thing to avoid — plus a second, differently-structured
concrete `BatchedPrefixCache`, selected by a runtime `if rtol.is_none()` branch,
with no shared interface between them or with any future policy.
`grep -rn "trait.*Cache"` over the whole diff: zero matches.

WS-integration flags the naming specifically: **adopting Hiroshi's exact
suggested type name for a structure that is architecturally the opposite of the
ask** makes it easy to grep for `PrefixCache`, conclude the ask was honored, and
stop reading. Low severity in substance — the "flat list first" implementation
Hiroshi sanctioned is what exists — but the abstraction boundary is absent, and
it is over-built relative to "flat list first" in having two concrete caches
instead of one.

**F-4 (WS-tree-probe)** also carries a `SCOPE-DEVIATION` verdict against plan
performance gate 1; see §2 above.

**No #691 scope creep anywhere.** WS-integration and WS-tree-probe both swept
for interface sketching, Layer 2, sub-chain partitioning, MPI, rayon, threads,
segment trees, and checkpointing: zero hits across the SRC diff.
`BatchedPrefixCache`'s segment logic chunks *sketch width* for GEMM batching
inside a still-fully-sequential sweep; it never materializes D×D transfer
operators, so it is also clean against the 2026-08-27T15:38 correction.
WS-tree-probe checked the same correction against `src_tree.rs`'s directed
messages and found them compliant: those are thin `(χ_A·χ_B) × l` boundary
states on the physical network's own topology, produced by ordinary edge
application — exactly what the correction says survives — not a scan
composition of the sequential cache.

### 6. `PLAN-CLAIM-UNVERIFIED` and `SOURCE-AMBIGUOUS`

**`SOURCE-AMBIGUOUS`: none.** No workstream found two Tier-1 statements in
conflict on the same sub-claim. The 12:56 → 15:38 supersession is an explicit
self-correction, handled as the spec directs.

**`PLAN-CLAIM-UNVERIFIED` (WS-core F1): the plan's own profiling gate was not
satisfied.** The 2026-08-26 plan gates the `idx_tensor.rs`/`tensor_like.rs`
additions behind a *component-level* profile ("random-vector construction;
per-column contraction planning; sketch-column assembly; QR; cap projection;
final SVD"). WS-core searched all six branch commits and every worklog: what
exists is an **end-to-end** wall-clock/error table, self-described as historical
and "not a formal performance gate," and the branch's own worklog admits the
formal experiment has not happened. This is a finding about the *plan's* unmet
precondition, not automatically a defect in the code — and WS-core's F2
substantially shrinks the risk it implies: `stack_along_new_index` and
`from_dense_any` **already existed as inherent methods on `origin/main`**, so
roughly half the added surface is pre-existing functionality being exposed
through a new trait seam, not fresh unprofiled work. The genuinely new,
unprofiled units are `try_contract_pairwise_retaining`/`contract_retaining_indices`,
`concatenate_along_new_index`, and the incremental-QR bridge.

The same unmet-profile pattern is WS-backend's F2 for `incremental_qr.rs`
(gate #7). Both point at the same missing artifact.

**The plan's proposed report/diagnostics API was never built** (WS-integration
check 1): `SrcRankSelection`, `SrcContractionResult`, `SrcContractionReport`,
`SrcEdgeReport`, `global_src_fixed`, `global_src_adaptive` have zero hits
repo-wide. WS-integration judged this a factual note about an aspirational plan
rather than a `PLAN-CLAIM-UNVERIFIED` — the plan misstates no Tier-1 fact, and
the smaller surface that exists (`SrcOptions` with `rtol: Option<f64>` encoding
the fixed/adaptive split) delivers exactly what the issue's opening post asked
for, consistently with how every other contraction method in the crate returns
a bare `TreeTN`.

**The plan's literal "chain reduction gate" does not exist** — found
independently by WS-integration (via dispatch-routing analysis) and WS-tests
(via grepping the test files and reading `validate_ortho_consistency`'s
implementation), merged here as one finding. The plan specifies four *named
intermediate identities* checked against a hand-written reference
implementation of the paper's equations: grepping for `chain reduction`,
`chain_reduction`, `forward environment`, `paper equation` across the whole
`contraction/` tree returns zero hits. What exists instead is a real,
non-vacuous end-to-end regression suite: six tests that build chain, branched
and complex fixtures, run SRC at a full probe cap with `final_svd: false`, and
assert the dense output matches `contract_naive` to `< 1e-8`. Both workstreams
independently derived why that is a genuine signal (a generic random sketch at
width ≥ exact rank spans the same column space almost surely, so QR-projecting
against it reproduces the exact answer up to floating point) — **and both noted
its limit: an end-to-end pass cannot distinguish "every step is individually
correct" from "an even number of compensating errors cancel," which is the
exact failure mode the four-identity decomposition exists to rule out.**
WS-integration additionally traced the dispatch to confirm these tests really
do reach `src_chain.rs`'s paper-faithful chain path rather than the general
tree fallback (the shared fixture's center is a degree-1 endpoint, so
`src_tree::contract`'s `chain.last() == Some(center)` delegation condition
holds).

### 7. Citation-only gaps

**The fabricated paper citation — verified independently three times.** In-code
module doc comments in `contraction.rs:11`, `src_chain.rs:4`, and
`src_tree.rs:4` state, verbatim in all three, that the implementation follows
"Algorithm 1, Sections 2.3--2.5" of Camaño–Epperly–Tropp. **WS-chain,
WS-tree-probe, and WS-integration each independently walked the actual
`\section`/`\subsection` structure of the paper's LaTeX source and each
confirmed the same thing: §2 ("Background") has exactly two subsections, §2.1
"Randomized QB approximation" and §2.2 "The Khatri–Rao product". There is no
§2.3, §2.4, or §2.5 anywhere in the paper.** The material actually being cited
is §3.1–§3.5. Three independent verifications against the primary source make
this an especially high-confidence finding; the identical wrong string in three
files rules out a typo and points to a templated citation that was never
checked. This is the audit's cleanest instance of the confident-but-wrong
citation pattern.

Two mitigating notes, recorded for honesty by WS-tree-probe: Hiroshi's
2026-07-29 comment cites "Sec. 2.1 and Algorithm 1," which would fit a
numbering where SRC is §2 (plausibly an earlier arXiv version) — but the
issue-opening post cites "Sec. 3.6" for linear combinations, which matches the
*local* numbering, so no single alternative version accommodates both. Against
the Tier-1 source this audit is instructed to use, the citation is wrong.
WS-integration additionally flags "Appendices C--D" in the same string:
Appendix C is a legitimate, verified citation for the adaptive-mode content,
but Appendix D is a pure operation-count table with nothing the dispatch code
implements, and reads as citation padding.

**The closed citation loop (WS-core F3).** Several new functions carry comments
saying a unit is "labelled `[AI-Supplied]` in the audit," without naming a
file. WS-core identified the referenced document as
`docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md` by
matching line ranges and labels — stating explicitly that this is its own
inference, not a quotation — and then checked that worklog's own provenance:
`git log --diff-filter=A` shows it was **added in the same commit (`9e018d4`)
as the code whose comments point at it.** Same session, same author, no
independent review in between. Per the spec's Tier-2 epistemics these in-code
citations carry no authority on their own; WS-core re-traced every claim they
make to Tier 1 directly, and the ones checked (Appendix C.3, the
`incrementalqr.py`/`.cpp` append contract, §C.2's recompute-or-update
semantics) held up. The *pattern* — code citing its own prior self-assessment
as though it were external verification — remains a concern for the rest of the
branch. Related: WS-backend flags that the same worklog's line citations into
`incremental_qr.rs` are stale throughout (it places `from_factors` at 113–178;
it is at 155–190), so a provenance table that cannot be indexed against the file
it audits should not be relied on as a map. WS-backend also **withdrew** an
earlier claim (its F7) that the `[AI-Supplied]` label promise was dangling —
the labels do exist, in the worklog the comment points at.

**Python line-range citations are inconsistent across the three SRC files** for
the same two functions: `random_contraction` cited as "82--353" and "133--353",
`random_contraction_inc` as "357--593" and "405--593", against actual
boundaries of 82–356 and 357–594. `src_probe.rs`'s citations are accurate;
`src_tree.rs`'s start line is 48 lines into the function.

**Not a citation gap: the reference citations that were checked and are
right.** WS-backend checked all four of `incremental_qr.rs`/`backend.rs`'s
external line citations against the reference repository —
`incrementalqr.py:90–151`, `incrementalqr.cpp:21–88`, `incrementalqr.cpp:106–119`,
and Appendix C/C.3 — and **all four are exactly right.** WS-integration fetched
issue comment 5396107820 live and confirmed body and timestamp match Appendix A
exactly.

### 8. `LICENSE-RISK` — assessed on complementary scope, resolved, no finding

Two workstreams looked at this from different angles and the question is
**closed, not open.** WS-integration flagged that `contraction.rs:15`'s own doc
comment self-describes "a line-by-line cross-check" against
`chriscamano/RandomMPOMPS` — a repository the spec establishes has no detected
license, and against which code reading as a line-by-line translation is a
finding in its own right. That self-description is exactly the trigger phrase
the spec names, so WS-integration flagged the wording and deferred the
body-level comparison to WS-chain, whose files carry the numerical logic.
WS-chain performed that comparison and found **no finding**: the reference
Python is raw NumPy `reshape`/`transpose`/`@` sequences with hand-computed axis
permutations, while the Rust works through named `Index`/`IndexLike` objects
and generic `TensorLike::contract` calls with no positional axis arithmetic at
all — different control flow, different data structures, no structural
correspondence that could read as a translation. WS-tree-probe reached the same
conclusion independently for its two files, and WS-backend likewise declined to
raise it for `incremental_qr.rs`, which stores the actual `R` plus a separate
`G = R^{-†}` where the reference overwrites its buffer with `trtri(R)` in
place — a genuinely different representation, and the code says so. What
matches across all of them is *parameter conventions* (real Gaussians, the
`min(maxdim, rows, cut)` width bound, the `err ≤ tol·norm` stopping rule),
which is validation, not copying. **Verdict: no `LICENSE-RISK` finding; the
doc comment's word choice is the only residue, and it should be reworded.**

### 9. Dead and premature code

Not taxonomy verdicts on their own, but recorded because they were found
consistently:

- **F-6 (WS-tree-probe).** Four `pub(super)` helpers in `src_probe.rs` —
  `site_probe`, `site_probe_batch`, `site_probe_batch_range`,
  `contract_prefix_with_probed_site_pair` — have **zero production call sites**;
  ~120 lines of documented API reachable only from the file's own tests. The
  aggravating detail: `site_probe` is the **only** function in the entire SRC
  surface that explicitly materializes the fused `∏d_i` (= `d²` for MPO-MPO)
  physical probe that Hiroshi's 2026-08-24 comment says never has to be formed.
  It is dead, so this is **not** an actual `SCOPE-DEVIATION` today — but the
  file ships, documents, and tests the exact construction Tier 1 rules out, in
  a form any future caller could pick up by name. Recommendation: delete rather
  than wire up.
- **WS-backend's sweep** found four instances of defensive code for
  structurally impossible cases, two premature abstractions (`IncrementalQrScalar`
  implemented for four scalars when only two are reachable), and one dead
  computation.
- **WS-tree-probe** notes two smaller instances in `src_tree.rs`: a discarded
  `local.get(node).ok_or_else(...)?` lookup that cannot fail, and a
  `site_initial_width` value computed on every fixed-rank call but consumed
  only in the adaptive branch.
- **WS-chain** records `contract` vs `contract_fixed` as a structural
  duplication that fits no taxonomy token — the Python encodes fixed-vs-adaptive
  as one function branching internally; the Rust has two top-level functions
  independently re-running the same five-phase schedule. Every phase of
  `contract_fixed` was checked and is faithful; the consequence for this audit
  is that the QR-only guarantee and the `PrefixCache` gap each had to be
  checked twice.

### 10. Branch hygiene: sync with `origin/main` before anything ships

**Independently caught by WS-tests and WS-integration, with consistent
conclusions after both were corrected in their fix rounds — stated once here.**
`git merge-base origin/main HEAD` is `72de8fb`, an older common ancestor, not
`origin/main`'s tip (`fd61f08`). `origin/main` independently picked up PR #693,
which restores the correct convention that an empty tensor train is the scalar
`1`; this branch forked before that landed and still carries the pre-fix `0.0`.
So `git diff origin/main` surfaces hunks in `tensortrain.rs`,
`tensortrain_inner.rs` and `quantics_tci/tests/mod.rs` that
`feature/treetn-src` never wrote.

**This is a staleness/hygiene issue, not a functional or merge-regression
risk.** WS-tests verified directly that
`git diff --stat 72de8fb..HEAD` over the three files #693 touched is **empty** —
the branch has made zero changes to any of them — so merging or rebasing simply
carries #693's fix along unmodified and cannot reintroduce the bug. The narrow,
accurate consequence: **the branch's own test results were produced against a
stale, pre-#693 base and should not be treated as final until it syncs.** Both
workstreams explicitly recommend the branch sync with `origin/main` before any
further work ships from it, and the plan's own "Verification commands" section
anticipates exactly this precondition. This report adopts that recommendation.

### 11. On the citation-precision residue

Each of the six workstreams went through a review loop that caught and either
fixed or explicitly deferred a number of minor issues: off-by-a-few line
numbers, imprecise wording, citations naming the right artifact at slightly the
wrong offset, and in a few cases claims that a fix round itself introduced and a
further round corrected. Those judged not to warrant another round are recorded
in the workstream files themselves and are not itemized here; they do not
change any verdict above.

### 12. Recommended next steps, in order

1. **Determine which SRC mode `gw-rs` invokes** (fixed vs adaptive). This is
   the highest-value single fact in the whole audit: it decides whether F1 is a
   performance finding or a hygiene finding, and it costs one grep of the
   caller.
2. **Sync the branch with `origin/main`** and re-run its test suite, per §10.
3. **Fix F-4 and F-5** — the tree path's local-pair materialization and the
   width-keyed batched cache. These are live candidates for the slowness
   symptom independent of the answer to step 1, and F-4's fix direction is
   already demonstrated by the chain path's own
   `contract_prefix_with_probed_site_pair_batch_range`.
4. **Decide `incremental_qr.rs`'s fate** in light of step 1's answer. Note that
   Appendix C.3's own block Gram–Schmidt formulation needs only `qr` and
   `mat_mul`, both of which the backend already provides, and would keep the
   hot path BLAS3 and GPU-dispatchable.
5. **Correct the fabricated "Sections 2.3--2.5" citation** in all three files,
   and reword `contraction.rs:15`'s "line-by-line cross-check" self-description.
6. **Close the highest-value test gaps**: a dense-oracle test for adaptive
   mode, a differential test of `IncrementalQr` against `qr_backend`, and a
   genuine numerical isometry check.
7. **Delete the dead `site_probe` family** rather than wiring it up.
8. **Remove or repair the `.or_else` fallback** in both `directed_messages`
   functions, so a missing message fails loudly instead of silently
   substituting a wrong-direction one.

---

The six workstream reports follow in full.



---

## WS-chain: src_chain.rs

*Source: [`docs/plans/audit-workstreams/ws-chain.md`](audit-workstreams/ws-chain.md), reproduced in full.*

### WS-chain — the literal single-chain case

**Files audited:**
- `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` (736 lines, full file)

**Tier-1 sources consulted:**
- Paper: `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`,
  §3 "Successive randomized compression" (`sec:alg`), specifically §3.1 "Step
  1: The last site" (`sec:Alg1`), §3.2 "Step 2: Second-to-last site"
  (`sec:Alg2`), §3.3 "Finishing up" (`sec:Alg3`), §3.4 "Optional step:
  Oversampling and final round" (`sec:final_round`), §3.5 "Summary: Pseudocode
  and time complexity" (`sec:pseudocode-complexity`, Algorithm 1,
  `\label{alg:rand-MPO--MPS}`, lines 640-672).
- Python reference: `/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/contraction.py`,
  `random_contraction` (def at line 82, main loop lines 133-355) and
  `random_contraction_inc` (def at line 357, main loop lines 405-593);
  `MPS.py` (`round`, line 170, and `rounding.py`'s `round_left`/`round_right`,
  lines 15-58, confirming the final-truncation SVD call site).
- Appendix A, spec `docs/plans/2026-08-28-src-provenance-audit.md`: the
  2026-07-29T07:13:51Z QR-only comment (current, not superseded — primary
  test for every SVD site below) and the 2026-08-27T12:56:57Z `PrefixCache`
  trait ask (status: partially superseded on the scan/parallelism claims by
  the 15:38 correction, but "the `PrefixCache` trait ask itself is not
  retracted and remains current").
- Verified independently (not trusted from the plan): the reference Python
  repository at `/root/projects/RandomMPOMPS-reference-20260827` has no
  `LICENSE*` file and no `.git`-tracked license text (checked directly with
  `find`), confirming the spec's "no detected license" premise before using
  it as a citation basis.

#### Provenance table

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_chain.rs` | Module doc comment (provenance claim) | 1-11 | `SOURCED-PAPER(Algorithm 1)` + flagged citation error | Algorithm and Python line-range citations check out (see Detailed findings), but the claimed paper location "Sections 2.3--2.5" is wrong — see flagged finding below. The doc's third claim (lines 9-11, "Prefix batching and the Q-column reuse optimization are derived implementation choices... labelled `[AI-Supplied]` in the audit") is also checked and holds: `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md:282` (`PrefixCache`/`BatchedPrefixCache` segment growth: "a Rust performance optimization not present in the author source: `[AI-Supplied]`") and `:313` (`incremental_qr.rs`'s `q_columns`: "Reusing old Q columns... is `[AI-Supplied]` optimization") both carry the exact `[AI-Supplied]` label the comment cites. Flag for Task 7: the code comment's provenance authority for this claim is itself a Tier-2 worklog, not a Tier-1 source (paper/Python/Hiroshi comment) — worth surfacing in the synthesis pass. |
| `src_chain.rs` | Imports | 13-27 | `DERIVED-VERIFIED` | Trivial, no source needed. Pulls `FactorizeAlg`, `SvdTruncationPolicy`, `TensorLike`, `IndexLike`, `Canonical` from `tensor4all_core` (not hand-rolled), and probe/prefix helpers from sibling `src_probe.rs` (audited under WS-tree-probe). |
| `src_chain.rs` | `fn contract` — signature, chain retrieval, topology/emptiness checks | 30-52 | `DERIVED-VERIFIED` | Engineering precondition-checking with no direct paper analog (the paper's MPO/MPS *are* linear chains by construction; extracting a chain from a general `TreeTN` and checking `same_topology` is Rust-side infrastructure). Trivially correct: refuses to proceed without a valid, non-empty, topology-matched chain. |
| `src_chain.rs` | `fn contract` — index desimilarization + local site pairs + `chain.len()==1` special case | 54-67 | `DERIVED-VERIFIED` | See "n=1 special case" derivation below. Python's `random_contraction` explicitly does *not* implement this case (`raise NotImplementedError`) — the Rust code adds a capability, not a citation gap. |
| `src_chain.rs` | `fn contract` — `outputs`, `cut_dimensions`, `probe_indices`, `last_output_dim`, `last_maximum_width`, `ProbeBank::new` | 69-91 | `SOURCED-PYTHON(contraction.py:138-150)` | Matches Python's per-site `prod_bond_dims`/`current_maxdim`/`current_sketchdim` computation; see `chain_cut_dimensions` row below for the exact correspondence. The probe/width-bound *logic itself* (`maximum_site_width`) is defined in `src_probe.rs`, out of this workstream's file list — noted, not re-verified here. |
| `src_chain.rs` | `fn contract` — dispatch to `contract_fixed` when `src_options.rtol.is_none()` | 92-104 | `SOURCED-PAPER(§3, fixed-χ̄ vs adaptive-χ̄ dichotomy)` | Matches the paper's two operating modes: §3.1-3.3 assume a supplied output bond dimension χ̄; Appendix (`sec:approx`, report.tex:1146, and `sec:adaptivity`, report.tex:1265) adds tolerance-driven adaptive determination. Python encodes the same dichotomy with one function guarded by `if outputdim is not None`; Rust splits it into two functions (`contract` vs `contract_fixed`) — see duplication note below. |
| `src_chain.rs` | `fn contract` — `sketch_options`, `PrefixCache::new` | 105-107 | `SCOPE-DEVIATION` | Cross-reference: this call site instantiates the `PrefixCache` struct; the verdict is the same one carried by the `PrefixCache` struct definition row below (`SCOPE-DEVIATION` — see "PrefixCache trait ask" finding below). |
| `src_chain.rs` | `fn contract` — adaptive last-site determination (`outputs[last].is_empty()` scalar branch + `factorize_site_adaptive` call) | 109-151 | `SOURCED-PAPER(§3.1, Algorithm 1 "Determine the last site")` + `SOURCED-PYTHON(contraction.py:205-212, 463-471)` + `DERIVED-VERIFIED` (scalar sub-branch) | See "Last-site sketch" and "Scalar-output boundary case" derivations below. |
| `src_chain.rs` | `fn contract` — main site loop `for site in (1..last).rev()` | 153-206 | `SOURCED-PAPER(§3.2/§3.3, Algorithm 1 "Determine sites η^(n-1),...,η^(2)")` + `SOURCED-PYTHON(contraction.py:213-238, random_contraction_inc:460-490)` | See "Interior-site sketch" derivation below. No SVD call anywhere in this loop (grep-verified, see SVD audit below). |
| `src_chain.rs` | `fn contract` — first-site determination | 208-210 | `SOURCED-PAPER(§3.3, Algorithm 1 "Determine the first site η^(1)", pseudocode line 669)` + `SOURCED-PYTHON(contraction.py:344-348)` | `contract_site_pair(local[0].0, local[0].1, &[&cap_environment])` matches η^(1)(a,b) = Σ H^(1)(a,c,d) ψ^(1)(d,e) S^(2)(e,d,b) — three tensors contracted (H, ψ, S^(2)), same as the pseudocode and the Python final block. |
| `src_chain.rs` | `fn contract` — result assembly (`TreeTN::new`, per-site `add_tensor`, `connect_result_edge`) | 212-221 | `DERIVED-VERIFIED` | Mechanical: the paper/Python return a flat MPS list; Rust must rebuild a graph (`TreeTN`) from the same per-site tensors and wire up the same chain edges. No new math — the edges connected are exactly the chain's original adjacency (`chain.windows(2)`, `chain[i-1]`-`chain[i]`), which reproduces the same 1-D topology. |
| `src_chain.rs` | `fn contract` — final SVD / canonical marking | 223-237 | `SOURCED-PAPER(§3.4, pseudocode line 670 "Optional: Run an MPS truncation algorithm")` + `SOURCED-COMMENT(#563, 2026-07-29T07:13:51Z)` | The one legitimate SVD call site in `contract`. See SVD audit below — gated by `src_options.final_svd`, applied only to the fully-assembled `result` (post per-site QR loop), matching Hiroshi's "acts on the already-compressed MPS" claim exactly. When `final_svd` is false, `mark_result_canonical` is used instead (no SVD at all). |
| `src_chain.rs` | `struct FixedContractionRequest` | 240-253 | `DERIVED-VERIFIED` | Trivial, no source needed. Parameter-bundle struct, no logic. |
| `src_chain.rs` | `fn contract_fixed` — signature/setup, `fixed_options`, `last_maximum_width`, `BatchedPrefixCache::new` | 255-289 | `SOURCED-PYTHON(contraction.py: outputdim-is-not-None branch, lines 110-117, 391-393)` | Mirrors the fixed-rank branch of the Python reference (`if outputdim is None: ... else: maxdim=mindim=sketchdim=outputdim`). |
| `src_chain.rs` | `fn contract_fixed` — last-site fixed determination (scalar branch + `BatchedPrefixCache::batch` + `factorize_fixed_batch`) | 290-324 | `SOURCED-PAPER(§3.1, Algorithm 1)` + `SOURCED-PYTHON(contraction.py:205-212)` + `DERIVED-VERIFIED` (scalar sub-branch, same derivation as the adaptive path) | Same math as the adaptive last-site branch, but the sketch is built as a single batch (`prefixes.batch(...)`, width fixed to `last_maximum_width` up front) rather than incrementally — see "Batched vs incremental sketch equivalence" derivation below. |
| `src_chain.rs` | `fn contract_fixed` — main site loop `for site in (1..last).rev()` | 326-374 | `SOURCED-PAPER(§3.2/§3.3, Algorithm 1)` + `SOURCED-PYTHON(contraction.py:213-238)` | Same tensors contracted as the adaptive path's interior-site loop (prefix batch, `local[site].0`/`.1`, `right_environment`), via `contract_prefix_with_probed_site_pair_batch_range` + `contract_pair` instead of the incremental `PrefixCache::column` + `T::contract` chain. Mathematically the same operation (see batching-equivalence derivation). |
| `src_chain.rs` | `fn contract_fixed` — first-site + result assembly + final SVD/canonical | 376-405 | `SOURCED-PAPER(§3.3, Algorithm 1 "Determine the first site η^(1)")` + `SOURCED-PYTHON(contraction.py:344-348)` + `DERIVED-VERIFIED` (result assembly) + `SOURCED-PAPER(§3.4, pseudocode line 670)` + `SOURCED-COMMENT(#563, 2026-07-29T07:13:51Z)` | Byte-for-byte structurally identical logic to lines 208-237 of `contract` (first-site contraction, `TreeTN` assembly, `final_svd`-gated truncation) — carries the same verdicts as those three rows above (first-site determination, result assembly, final SVD/canonical marking). This is the second (and only other) SVD call site in the file — see SVD audit below. |
| `src_chain.rs` | `fn chain_cut_dimensions` | 407-441 | `SOURCED-PYTHON(contraction.py:138-146)` | Computes, per internal chain edge, `dim_a(edge) * dim_b(edge)` (MPO bond dim × MPS bond dim). Combined with the call-site `.max()` of the two adjacent edges (line 163, 336), this is algebraically identical to Python's per-site `prod_bond_dims = max(H[j].shape[0]*psi[j].shape[0], H[j].shape[2]*psi[j].shape[2])` — see derivation below. The boundary case (`cut_dimensions.last()`, one edge only) matches Python's `j == n-1` branch (single bond, no `max`). |
| `src_chain.rs` | `fn factorize_fixed_batch` | 443-462 | `SOURCED-PAPER(§3, QR-only claim)` + confirms no `HANDROLLED-DUPLICATE` | Calls `sketch.factorize_full_rank(left_indices, FactorizeAlg::QR, FactorizeCanonical::Left)` — an explicit, named QR decomposition delegated to `tensor4all-core`'s typed factorization API, not a hand-rolled linear-algebra routine. This is the fixed-rank path's analog of the paper's step (ii)/(iii) QR-and-project; matches Python's `np.linalg.qr` call sites (contraction.py:244, 247) in kind (QR, not SVD). |
| `src_chain.rs` | `struct PrefixCache` | 464-473 | `SCOPE-DEVIATION` | See "PrefixCache trait ask" finding below — field `prefixes: Vec<Vec<T>>` is a concrete, hard-coded Vec of Vecs, not a trait, contradicting Hiroshi's still-current 2026-08-27T12:56:57Z ask. |
| `src_chain.rs` | `struct BatchedPrefixCache`, `struct PrefixBatchSegment` | 475-493 | `SCOPE-DEVIATION` | Same finding as `PrefixCache`: `BatchedPrefixCache` is likewise a concrete struct (fields `cached`, `segments: Vec<PrefixBatchSegment<T>>`), not behind any trait. |
| `src_chain.rs` | `impl BatchedPrefixCache::new` | 500-513 | `DERIVED-VERIFIED` | Trivial, no source needed. Field initialization only. |
| `src_chain.rs` | `impl BatchedPrefixCache::batch` | 515-597 | `SOURCED-PYTHON(concept: contraction.py's incrementally-grown `envs` list, lines 164-199)` + `DERIVED-VERIFIED` (segment/concatenate batching mechanism) | See "Batched vs incremental sketch equivalence" derivation below. |
| `src_chain.rs` | `impl PrefixCache::new` | 605-618 | `DERIVED-VERIFIED` | Trivial, no source needed. Field initialization only. |
| `src_chain.rs` | `impl PrefixCache::ensure_width` | 620-666 | `SOURCED-PYTHON(concept: contraction.py's `sketchincrement`-driven `envs` growth, random_contraction_inc lines 431-456)` + `DERIVED-VERIFIED` (batch-then-split mechanism) | Grows the cache in `batch_size` chunks, then splits each chunk into individual per-column tensors via `select_indices`. See derivation below. |
| `src_chain.rs` | `impl PrefixCache::column` | 668-680 | `SOURCED-PYTHON(contraction.py: `envs[idx][j-1]` column access pattern)` | Direct analog of indexing into the Python `envs` list; triggers `ensure_width` on demand instead of requiring the caller to pre-grow the cache. |
| `src_chain.rs` | `struct FactorizeSiteRequest` | 683-695 | `DERIVED-VERIFIED` | Trivial, no source needed. Parameter-bundle struct, no logic. |
| `src_chain.rs` | `fn factorize_site_adaptive` | 697-736 | `SOURCED-PAPER(§3, Algorithm 1 steps (ii)-(iii): orthonormalize + project)` + `SOURCED-PYTHON(contraction.py:280-313, random_contraction_inc:540-571)` | See "Per-site orthonormalize+project" derivation below. Delegates the actual QR/orthonormalization to `factorize_probe_columns` in `src_probe.rs` (out of this file, audited under WS-tree-probe) — this function's own logic is the conjugate-and-project (environment/cap update) step. No SVD call in this function (grep-verified). |

#### SVD audit (mandatory per Task 1 brief)

`grep -n -i "svd"` over the full file returns exactly these 14 lines, all
accounted for:

- Line 17: `use tensor4all_core::{..., SvdTruncationPolicy, ...}` — type import only.
- Line 34: `svd_policy: Option<SvdTruncationPolicy>` — parameter of `fn contract`.
- Line 95: `svd_policy,` — struct-field shorthand, passed into `FixedContractionRequest`.
- Line 102: `final_svd: src_options.final_svd,` — struct-field shorthand.
- Line 105: `let sketch_options = src_options.sketch_options(svd_policy.is_some());` — configuration derivation, not a decomposition call.
- Line 223: `if src_options.final_svd {` — **SVD call site #1** (guard).
- Line 226: `svd_policy,` — argument to `result.truncate_impl(...)` inside the guarded branch.
- Line 245: `svd_policy: Option<SvdTruncationPolicy>,` — field declaration in `FixedContractionRequest`.
- Line 252: `final_svd: bool,` — field declaration in `FixedContractionRequest`.
- Line 264: `svd_policy,` — destructured field, `contract_fixed`'s request unpacking.
- Line 271: `final_svd,` — destructured field, `contract_fixed`'s request unpacking.
- Line 274: `let fixed_options = SrcOptions::fixed().with_final_svd(final_svd);` — configuration.
- Line 390: `if final_svd {` — **SVD call site #2** (guard).
- Line 393: `svd_policy,` — argument to the second `result.truncate_impl(...)`.

**Conclusion: there are exactly two SVD-triggering call sites in this file**
(`result.truncate_impl(..., svd_policy, ...)` at line 224-229 inside the
`if src_options.final_svd` guard, and its `contract_fixed` twin at line
391-396 inside `if final_svd`). Both:
1. Fire only when the caller has opted into final rounding (`final_svd`/
   `src_options.final_svd`), matching the paper's §3.4 "optional step."
2. Operate on `result`, the already-fully-assembled `TreeTN` built from the
   per-site QR factors (i.e., the "already-compressed MPS" of Hiroshi's
   2026-07-29 comment) — not on any per-site sketch, prefix, or intermediate
   tensor inside the site loops.
3. Are the *only* place in the file an actual decomposition is requested;
   `factorize_fixed_batch` (line 443-462) and `factorize_probe_columns`
   (delegated to `src_probe.rs`) are explicitly `FactorizeAlg::QR`, never SVD.

No SVD call appears inside either site loop (`contract`'s loop at 153-206,
`contract_fixed`'s loop at 326-374) or inside any prefix/cache-construction
code (`PrefixCache`/`BatchedPrefixCache`, lines 464-681). **The QR-only hot
path claim from Hiroshi's 2026-07-29T07:13:51Z comment holds for this file
without exception.** This is a clean result on the audit's original
motivating suspicion, at least for `src_chain.rs` specifically — the
suspected hand-rolled-SVD-in-the-hot-path pattern is not present here.

#### Detailed derivations and flagged findings

##### Module doc comment — flagged citation error (AI-hallucination-signature: confident but wrong citation)

The header (lines 3-4) states: *"`contract` implements the right-to-left
schedule in Algorithm 1 and Sections 2.3--2.5 of Camaño--Epperly--Tropp."*
Checking directly against `report.tex`'s `\section`/`\subsection` structure:

```
\section{Introduction}                                    (§1)
\section{Background}                                       (§2)
  \subsection{Randomized QB approximation}                 (§2.1)
  \subsection{The Khatri--Rao product}                     (§2.2)
\section{Successive randomized compression}                (§3)
  \subsection{Step 1: The last site}                       (§3.1)
  \subsection{Step 2: Second-to-last site}                 (§3.2)
  \subsection{Finishing up}                                (§3.3)
  \subsection{Optional step: Oversampling and final round}  (§3.4)
  \subsection{Summary: Pseudocode and time complexity}      (§3.5, Algorithm 1)
```

Section 2 ("Background") has exactly two subsections, 2.1 and 2.2 — **there
is no §2.3, §2.4, or §2.5 anywhere in the paper.** The actual step-by-step
algorithm content the comment is trying to cite is entirely in §3
(specifically §3.1-§3.5, not §2.3-§2.5). This is a clean example of the
"confident-sounding comment that doesn't hold up when checked against the
paper text" hallucination pattern named in the audit brief: the section
numbers are fabricated (they don't exist), even though — as the rest of this
table shows — the *algorithm itself*, once checked against the correct
location (§3 and Algorithm 1), is faithfully implemented. The Algorithm-1
citation and the two Python line-range citations (`random_contraction`
133-353, `random_contraction_inc` 405-593) are both accurate on direct
inspection (verified above), so this is a narrow, single-fact citation error
rather than a sign the surrounding code is unverified — but it is exactly
the kind of unverifiable-on-its-face claim the audit exists to catch, and it
should be corrected to "§3.1-§3.5" (or simply "§3, Algorithm 1").

##### `fn contract` — signature, chain retrieval, topology/emptiness checks (lines 30-52) — DERIVED-VERIFIED

`chain = tn_a.chain_order(center)` retrieves the linear ordering of sites
through `center`, failing if `center` isn't part of a chain; `tn_a.same_topology(tn_b)`
checks that the two operand networks (`H` and `ψ`) share the same graph
structure; `chain.is_empty()` rejects a degenerate zero-site chain. None of
this has a direct paper or Python analog: the paper's MPO and MPS *are*
linear chains by construction (there is no general-tree case to guard
against), so `random_contraction`/`random_contraction_inc` never perform an
equivalent check. This is Rust-side infrastructure needed because `contract`
is reached from a general `TreeTN` API that must first establish it's
actually looking at a chain before the chain-only algorithm below can run.
The derivation is trivial: each check is a precondition that, if it fails,
correctly aborts before any numerically meaningful work is done (no
chain/topology mismatch/empty-chain state can produce a valid contraction),
and none of the checks discards or approximates anything — they are pure
early-return guards. No paper/Python equivalent is needed or expected.

##### n=1 special case (lines 57-67) — DERIVED-VERIFIED

For a length-1 chain, `H|psi>` reduces to contracting the single MPO tensor
`H^{(1)}` with the single MPS tensor `psi^{(1)}` over their shared physical
index — there is no internal bond to compress, so no sketching, QR, or
truncation is mathematically necessary; the "compressed" output *is* the
exact contraction. `contract_site_pair(local[0].0, local[0].1, &[])` performs
exactly this (no extra probe/cap tensors), and the subsequent
`canonicalize_impl(..., CanonicalForm::Unitary, ...)` puts the single
resulting tensor into isometric form for consistency with the rest of the
`TreeTN` canonical-form machinery (a single-tensor network is trivially
already "canonical" once any residual gauge freedom is fixed by a QR/
polar step). This is correct and adds no new numerical risk. Note: the
Python reference `random_contraction` explicitly raises
`NotImplementedError` for `n == 1` (contraction.py:100) — so this Rust
branch is a legitimate capability addition, not a deviation from a Python
behavior that should have been preserved.

##### Scalar-output boundary case (`outputs[last].is_empty()`, lines 118-129 and 290-301) — DERIVED-VERIFIED

When the last chain site has zero exposed physical indices (`outputs[last]`
empty — e.g. because this chain function is invoked on a sub-chain of a
larger tree where the boundary node carries no local physical leg), the
"matrix" being QB-decomposed at that site is conceptually `1 × (rest)`: its
row space has dimension 1. Any orthonormal basis for a 1-dimensional space is
a single unit vector, so the correct "Q" factor is the scalar 1 embedded in a
dimension-1 index. The code constructs exactly this: `cap =
T::Index::new_link(1)`, `factor = T::ones([cap])` (a rank-1 all-ones tensor
of shape `(1,)`, i.e. the scalar 1), and `environment = local_product
.outer_product(&factor)` where `local_product = contract_site_pair(H, psi,
&[])` is the direct (unsketched) contraction of the site's own H/psi pair.
Since `factor` is the scalar 1, `environment` is just `local_product`
re-indexed with an extra dummy dimension-1 leg so that its shape matches what
the next loop iteration expects (a "capped" tensor with the same index
structure as the general case). This is exact, not an approximation — no
information is discarded, matching the paper's promise that `QB`
decomposition of an already-full-rank-1 factor is trivially exact. Confirmed
correct by direct construction; no counterexample exists since the
degenerate case has a unique correct answer.

##### Last-site sketch (lines 109-149, adaptive; 290-322, fixed) — SOURCED-PAPER + SOURCED-PYTHON

Algorithm 1 pseudocode (report.tex:655): `Y^{(n)}(a,b) = Σ_{c,d,e}
C^{(n-1)}(a,c,d) H^{(n)}(b,c,e) ψ^{(n)}(d,e)`. The `make_column` closure at
lines 142-147 (`factorize_site_adaptive`'s adaptive caller) computes: `prefix
= prefixes.column(last-1, column)` (this is `C^{(n-1)}`, one sketch column),
then `contract_prefix_with_site_pair(&prefix, local[last].0, local[last].1)`
— contracting the prefix with `H^{(n)}` and `ψ^{(n)}`, exactly the three
tensors in the pseudocode's `Y^{(n)}` formula. This matches Python's
`j == n-1` sketch-formation branch (contraction.py:205-212 /
random_contraction_inc:463-471): `temp = envs[idx][j-1] @ psi[j]; temp =
H[j] @ temp` — same three-tensor contraction (env/`C`, ψ, H), but **a
different contraction order**: `contract_prefix_with_site_pair` (defined in
`src_probe.rs:358-366`) contracts `prefix` with `tensor_a` first and
`tensor_b` second, and here `tensor_a = local[last].0` is `H^{(n)}` while
`tensor_b = local[last].1` is `ψ^{(n)}` — i.e. Rust contracts prefix-then-H
first, then folds in ψ, whereas Python contracts `envs[idx][j-1] @ psi[j]`
(prefix-then-ψ) first, then folds in `H[j]`. Same tensor set, opposite order
of the last two contraction steps; mathematically value-equivalent (tensor
contraction is associative/commutative in the operands being combined here),
but the order-equivalence is not literal — cost/performance implications of
the differing order are not assessed by this workstream. The fixed-rank path (lines 303-314)
performs the identical contraction batched over all columns at once via
`contract_prefix_with_probed_site_pair_batch_range` rather than per-column —
see the batching-equivalence derivation below for why this is the same
computation.

##### Interior-site sketch (lines 153-206, adaptive; 326-374, fixed) — SOURCED-PAPER + SOURCED-PYTHON

Algorithm 1 pseudocode (report.tex:662): `Y^{(j)} = Σ_{d,e,f,g,h}
C^{(j-1)}(a,d,e) H^{(j)}(d,b,f,g) ψ^{(j)}(e,g,h) S^{(j+1)}(h,f,c)` — a
four-tensor contraction of the prefix `C^{(j-1)}`, the MPO site `H^{(j)}`,
the MPS site `ψ^{(j)}`, and the right-environment cap `S^{(j+1)}`. The
adaptive `make_column` closure (lines 186-201) computes exactly these four
contractions: `prefix = prefixes.column(site-1, column)`
(→ `C^{(j-1)}`), `after_a = T::contract(&[&prefix, local[site].0])` (→ with
`H^{(j)}`, A-side first), `after_b = T::contract(&[&after_a, local[site].1])`
(→ with `ψ^{(j)}`), `T::contract(&[&after_b, &right_environment])` (→ with
`S^{(j+1)}`, i.e. `cap_environment` from the previous iteration). Python's
`else` (interior-site) sketch-formation branch (random_contraction_inc:472-488,
contraction.py:218-235) builds the identical four-tensor product (`env`/`C`,
`psi`, `H`, `cap`) but in a different dependency order: `temp =
envs[idx][j-1] @ reshaped_psis2[j-1]` folds in ψ before H (environment/
MPS-side first), then `reshaped_H2[j-1] @ temp_reshaped` folds in H, then
the result is contracted against `cap`. As with the last-site case (above),
this is the same tensor set contracted in a different order — A-side (`H`)
first in Rust vs. environment/MPS-side (`ψ`) first in Python — which is
mathematically value-equivalent but not a literal order match; the
cost/performance implications of the differing order are not assessed by
this workstream. The fixed-rank path (lines 344-360) performs the same four-tensor contraction
batched via `contract_prefix_with_probed_site_pair_batch_range` +
`contract_pair`.

##### `fn contract` — result assembly (lines 212-221) — DERIVED-VERIFIED

After the site loop and first-site contraction produce one factored tensor
per chain site, this block rebuilds a `TreeTN`: `TreeTN::new()` creates an
empty graph, each site's tensor is added via `add_tensor`, and
`connect_result_edge` wires up an edge between each adjacent pair
(`chain.windows(2)`, i.e. `chain[i-1]`-`chain[i]`) — exactly the chain's
original adjacency. The paper and Python reference return a flat ordered
list of per-site tensors (an MPS); Rust's `TreeTN` representation requires
an explicit graph object, so this step is mechanical bookkeeping to
reconstruct the same 1-D topology as a first-class graph rather than an
implicit list ordering. No new math is introduced: the same per-site
tensors computed by the (already-verified) site loop/first-site/last-site
code are placed into the graph unchanged, and the edges added reproduce
exactly the original chain's linear adjacency, so the resulting `TreeTN`
represents the identical MPS the Python code would return as a list. No
paper/Python equivalent is needed since neither uses a graph data structure.

##### `factorize_site_adaptive` — per-site orthonormalize+project (lines 697-736) — SOURCED-PAPER + SOURCED-PYTHON

Algorithm 1 steps (ii)-(iii) (orthonormalize via QR, then project) are split
across two calls: `factorize_probe_columns(...)` (defined in `src_probe.rs`,
out of this file's scope — presumably performs the incremental/adaptive QR
that yields `(factor, cap)` = `(η^{(j)}, bond index of rank χ̄)`), then this
function computes `factor_conj = factor.conj()` and calls
`contract_site_pair(operands.0, operands.1, &[&factor_conj,
right_environment])` (or without `right_environment` for the last site).
This matches the pseudocode's `S^{(j)}` update (report.tex:665): `S^{(j)}(a,
b,c) = Σ conj(η^{(j)}(c,d,e)) H^{(j)}(b,d,f,g) ψ^{(j)}(a,g,h) S^{(j+1)}(h,f,e)`
— the same four tensors (`conj(η)`, `H`, `ψ`, `S^{(j+1)}`) are contracted
together; `contract_site_pair`'s index-typed contraction (rather than an
explicit einsum string) is expected to wire the correct shared indices,
which is not independently re-verifiable from `src_chain.rs` alone since
`contract_site_pair`'s body lives in `src_probe.rs` (WS-tree-probe's file).
Given that constraint, the verdict here is `SOURCED-PAPER`+`SOURCED-PYTHON`
at the level of "the correct four tensors are assembled in the correct
order," with the caveat that full index-level verification requires
WS-tree-probe's audit of `contract_site_pair`'s implementation.

##### `chain_cut_dimensions` — edge-vs-site bound refactor (lines 407-441, consumed at 163 and 336) — SOURCED-PYTHON, refactor verified equivalent

Python (contraction.py:138-146):
```python
if j == n - 1:
    prod_bond_dims = H[j].shape[0] * psi[j].shape[0]
else:
    prod_bond_dims = max(H[j].shape[0]*psi[j].shape[0], H[j].shape[2]*psi[j].shape[2])
```
i.e. per *site* `j`, take the max of (MPO-left-dim × MPS-left-dim) and
(MPO-right-dim × MPS-right-dim), or just the left pair at the boundary.
Rust's `chain_cut_dimensions` instead computes, once, per internal *edge* `e`
between chain sites `(left, right)`: `dim_a(e) * dim_b(e)` (the product of
the two networks' bond dimensions on that edge). Since site `j`'s "left"
bond is edge `(j-1,j)` and its "right" bond is edge `(j,j+1)`, `max(left-bond
value, right-bond value)` computed per-site is identical to `max(cut_dimensions[j-1],
cut_dimensions[j])` computed from the precomputed per-edge array — which is
exactly what `contract`/`contract_fixed` do at line 163/336
(`cut_dimensions[site-1].max(cut_dimensions[site])`). The boundary case
(`cut_dimensions.last()`, a single edge, no `.max()`) matches Python's
`j == n-1` branch (only the left/only bond). This is a correct, and more
efficient (edge values computed once instead of recomputed per adjacent
site), restructuring of the same formula — `DERIVED-VERIFIED` for the
refactor itself, `SOURCED-PYTHON` for the underlying formula.

##### Batched vs incremental sketch equivalence (`BatchedPrefixCache::batch`, lines 515-597; `PrefixCache::ensure_width`, lines 620-666) — DERIVED-VERIFIED

The Python reference grows its sketch one column at a time: `for idx in
range(len(envs), current_sketchdim): ... envs.append(env)` (contraction.py:164,
random_contraction_inc:431). Both Rust caches instead grow the sketch in
chunks and either (a) keep the chunk as a single wider tensor indexed by a
"batch" index (`BatchedPrefixCache`, used by the fixed-rank path, which knows
the final width up front) or (b) grow in `batch_size` chunks and then split
each chunk into individual per-column tensors via `select_indices`
(`PrefixCache`, used by the adaptive path, which needs individually
addressable columns for `factorize_probe_columns`'s incremental QR).

Correctness argument: each sketch column `k` is defined by contracting the
network against one fixed random probe vector/matrix `Ω^{(1)}_{:,k}, ...,
Ω^{(j)}_{:,k}` (Khatri-Rao structure, §2.2 background + §3's per-site
application). This contraction is linear and acts
independently per column `k` — the tensor-network contraction that produces
column `k`'s prefix value does not depend on any other column. Stacking `w`
independent per-column computations into one batched tensor contraction
(introducing a size-`w` "batch" index and running the same contraction once
against all `w` probe columns simultaneously) is mathematically the
elementwise-in-`k` application of a linear map — i.e. `f(Ω_{:,1}), ...,
f(Ω_{:,w})` computed one call as `f_batched([Ω_{:,1},...,Ω_{:,w}])` — which
is identical to doing it one column at a time and concatenating results,
*provided* no column's computation reads another column's intermediate state
(true here: each `prefixes.column`/segment only ever contracts probe/MPO/MPS
tensors, never mixes across the batch index until the final QR). This is a
standard batching optimization with no numerical effect, and both caches'
column-selection/segment-concatenation logic (`T::concatenate_along_new_index`,
`select_indices`) preserves the mapping from logical column index to sketch
value. Verdict: `DERIVED-VERIFIED` — the paper/Python do not describe this
batching (they are pure single-column Python loops), but the batching is a
provably equivalent reformulation of the same per-column linear map.

##### `PrefixCache`/`BatchedPrefixCache` — the `PrefixCache` trait ask (lines 464-493) — SCOPE-DEVIATION

Hiroshi's 2026-08-27T12:56:57Z comment on #563 (status: the scan/parallelism
framing is superseded by the 15:38 correction, but the spec's Appendix A
explicitly notes "The `PrefixCache` trait ask itself is not retracted and
remains current — WS-integration should check for it") makes one concrete,
current implementation request:

> "The only request: put the cache behind a small trait (something like
> `PrefixCache: fn extend(piece), fn get(k)`) instead of hard-coding a Vec
> built in a forward loop. Then flat list is the first implementation, and
> blocked/tree/checkpointed policies can be swapped in later without
> touching the SRC logic itself."

`src_chain.rs` defines `struct PrefixCache<'a, T> { ..., prefixes:
Vec<Vec<T>>, ... }` (line 464-473) with methods `ensure_width` (a forward
`while start < width { ... self.prefixes[site].push(...) }` loop, lines
620-666) and `column` (lines 668-680, matching the suggested `get(k)` shape).
This is, almost feature-for-feature, the exact pattern Hiroshi asked to be
placed *behind* a trait — including the literal name `PrefixCache` — but it
is implemented as a concrete struct with inherent methods, not a trait with
implementations. `BatchedPrefixCache` (line 475-485, used by the fixed-rank
path) has the same shape problem: a concrete struct (`cached`, `segments:
Vec<PrefixBatchSegment<T>>`) with no trait boundary. Neither struct is
generic over a swappable caching policy; `contract`/`contract_fixed` call
`PrefixCache::column`/`BatchedPrefixCache::batch` directly as inherent
methods, so a future "blocked" or "tree" cache policy (the kind of thing
Hiroshi's memory/depth trade-off table describes) could not be swapped in
"without touching the SRC logic itself" — it would require changing the
concrete type used at every call site in `src_chain.rs`.

This is a `SCOPE-DEVIATION` against a still-current, explicit, named
implementation ask (the taxonomy's own definition cites "the `PrefixCache`
trait ask" by name as its example). It is a low-severity deviation — the
"flat list first" implementation Hiroshi explicitly sanctioned is exactly
what exists — but the abstraction boundary he asked for is absent. Flagging
here because the struct lives in this file; **WS-integration owns the
authoritative synthesis verdict** for this finding (its brief explicitly
assigns it "Check whether the `PrefixCache` trait Hiroshi asked for ... was
honored, ignored, or over-built"), and Task 7 should merge this entry with
WS-integration's rather than double-count it.

##### `contract` vs `contract_fixed` — structural duplication (not a taxonomy verdict on its own, noted for completeness)

Python encodes fixed-vs-adaptive rank selection as one function
(`random_contraction`/`random_contraction_inc`) branching internally on
`outputdim is None`. Rust instead has two top-level functions (`contract`,
lines 30-238; `contract_fixed`, lines 255-405) that independently re-run
essentially the same five-phase schedule (last-site → site loop → first-site
→ assembly → optional final SVD) using different cache types
(`PrefixCache`/incremental vs `BatchedPrefixCache`/batched) and different
factorization entry points (`factorize_site_adaptive`/`factorize_probe_columns`
vs `factorize_fixed_batch`). Every phase of `contract_fixed` was checked
against the paper/Python above and is faithful; this is a code-structure
observation, not a provenance gap, and doesn't map cleanly onto any of the
eleven verdict tokens (it is closest to "unjustified" duplication rather
than "premature abstraction" — if anything it is an *absence* of shared
abstraction). Noted for the synthesis pass since it means the QR-only
guarantee and the `PrefixCache`-trait gap both had to be, and were, checked
twice (once per function) rather than once.

##### License-risk assessment — no finding

Compared line-by-line against the reasoning in `contract.py`'s
reshape/transpose/`@`-matmul style (raw NumPy, explicit index bookkeeping via
reshapes), `src_chain.rs`'s implementation is structurally independent: it
uses named `Index`/`IndexLike` objects and generic `TensorLike::contract`/
`contract_site_pair` calls rather than manual reshape-and-matmul sequences,
different control flow (closures passed into shared `factorize_site_adaptive`/
`factorize_probe_columns` helpers rather than inline per-branch code), and a
different data-flow structure (typed prefix caches vs. Python's flat `envs`
list of lists). This reads as an independent, index-typed reimplementation
validated against the reference's *numerical* behavior and per-step tensor
contractions, not a translation of its *code*. No `LICENSE-RISK` finding for
this file.

##### `HANDROLLED-DUPLICATE` — no finding

Every actual decomposition call in this file (`factorize_fixed_batch`'s
`sketch.factorize_full_rank(..., FactorizeAlg::QR, ...)`, and the two
`result.truncate_impl(..., svd_policy, ...)` final-truncation calls)
delegates to `tensor4all-core`'s typed factorization/truncation API (imported
at line 16-18: `Canonical as FactorizeCanonical, FactorizeAlg,
SvdTruncationPolicy`). No inline QR, SVD, matrix-inverse, or LAPACK-wrapper
code exists anywhere in `src_chain.rs` — the plan's explicit ban on
hand-rolled linear algebra in `treetn` is respected in this file.


---

## WS-tree-probe: src_tree.rs, src_probe.rs

*Source: [`docs/plans/audit-workstreams/ws-tree-probe.md`](audit-workstreams/ws-tree-probe.md), reproduced in full.*

### WS-tree-probe — the from-scratch tree generalization and MPO-MPO probing

**Files audited:**
- `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs` (641 lines, full file)
- `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs` (1171 lines, full file)

Every line of both files is accounted for by some row of the provenance table
below: the table's line ranges are exhaustive over all non-blank, semantically
meaningful lines of 1-641 and 1-1171 respectively. They are *not* literally
contiguous — roughly 50 single-line gaps fall on blank separator lines between
code units, and two further uncovered lines are non-blank but carry no
semantics: `src_probe.rs:124` (the `}` closing `impl ProbeBank`) and
`src_probe.rs:1171` (the `}` closing `mod tests`).
Rows marked `n/a (trivial plumbing)` cover only imports,
type aliases, parameter-bundle structs, and field-initialisation constructors.

**Tier-1 sources consulted:**

- Paper: `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`.
  Section structure verified directly from the LaTeX (not assumed):
  §1 Introduction (1.1-1.3), §2 Background with **exactly two** subsections
  — §2.1 "Randomized QB approximation" (`sec:randomized-qb`, line 230) and
  §2.2 "The Khatri--Rao product" (`sec:krp`, line 283) — §3 "Successive
  randomized compression" (`sec:alg`, line 351) with §3.1 `sec:Alg1`
  (line 381), §3.2 `sec:Alg2` (454), §3.3 `sec:Alg3` (594), §3.4
  `sec:final_round` (627), §3.5 `sec:pseudocode-complexity` (638, containing
  Algorithm 1 `alg:rand-MPO--MPS`, lines 640-672), §3.6 (706), §3.7 (719);
  Appendix A `app:khatri-rao-exact` (1102), Appendix B `app:src-exact-proof`
  (1125), Appendix C `sec:approx` "Implementing SRC with a tolerance" (1146)
  with `sec:error-estimation` (1152), `eq:err-est` (~1240), `eq:norm_est`
  (~1252), `sec:adaptivity` (1265), `app:qr-updating` (1288); Appendix D
  "Full operation counts" (1333). **There is no §2.3, §2.4 or §2.5 in this
  source** — see the flagged citation finding.
- Python reference:
  `/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/contraction.py`.
  Function boundaries verified directly: `random_contraction` def at line 82
  (body through 356), `random_contraction_inc` def at line 357 (body through
  594). Probe generation at line 432 (`np.random.randn(visible_dim)`),
  width bounds at 415-417 (`current_maxdim` at 415, `current_mindim` and
  `current_sketchdim` at 416-417), adaptive stopping at 509-522.
- Appendix A of the spec, read in chronological order. Governing comments for
  this workstream:
  - **2026-07-29T07:13:51Z** (status: current) — SRC's core loop is QR-only;
    an SVD appears only in one optional final truncation on the
    already-compressed output. Primary test for every SVD site in these files.
  - **2026-08-24T13:45:35Z** (status: current) — the factorized MPO-MPO probe
    `Ω_{s,t,k} = X_{s,k} Y_{t,k}`, giving
    `E_k = Σ_{s,t,u} X*_{s,k} A^{s,u} B^{u,t} Y*_{t,k}`, "so the fused
    (d^2)-dimensional physical tensor never has to be formed explicitly."
    Primary test for `src_probe.rs`'s probe construction.
  - **2026-08-27T15:38:47Z** (status: current, supersedes the 12:56 comment's
    scan/tree-parallelism framing) — "Any tree or scan composition ... must
    materialize interval transfer operators as D×D matrices ... edge-sequential
    contraction is essentially forced." Checked below against the tree cache;
    `src_tree.rs`'s directed messages are *spatial* tree messages on the
    physical network's own topology, not a scan-composition of the sequential
    cache, so this correction does not forbid them — but see finding F-4,
    where a related materialization problem does occur.
  - **2026-08-27T18:24:53Z** and both #691 comments (status: current, but
    describe proposals). No interface-sketching machinery found in either
    file — no `SCOPE-DEVIATION` on that axis.
- **No Tier-1 source covers the tree generalization at all.** The paper is
  chain-only ("a single pass from right-to-left", §3), the reference Python
  operates on `MPO`/`MPS` list types with integer site indices only, and no
  Hiroshi comment mentions trees. Everything in `src_tree.rs` that is not a
  literal restatement of the chain recursion is therefore `DERIVED-VERIFIED`
  or `SUSPECT-UNVERIFIED`, per the spec.

**Verification runs performed for this workstream:**
`cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --lib contraction`
→ `114 passed; 0 failed` (exit 0). Includes the four tree-path tests
`src_fixed_traverses_a_branched_tree_without_dense_fallback`,
`src_fixed_matches_naive_on_a_branched_tree_when_probe_cap_is_full`,
`src_adaptive_contracts_a_branched_tree_with_a_rank_cap`,
`src_preserves_scalar_only_subtrees_with_dimension_one_bridges`, plus
`partial_contract_src_uses_the_same_directed_tree_path`.

---

#### Executive summary of this workstream

The tree recursion in `src_tree.rs` is **mathematically correct** — I re-derived
it independently from the paper's chain forward/backward environment recursion
(derivation D-1 below) and it holds, including the non-obvious part: for a
rooted edge `(child, parent)`, the *processed* side of the cut is carried by
`projected_children` (conjugated-isometry bridges, the tree analogue of the
paper's `S^(j)`) while the *unprocessed* side is carried by the probed complement
message `M_{parent→child}` (the analogue of `C^(j-1)`), and the two are correctly
distinguished. The MPO-MPO probe construction in `src_probe.rs` matches Hiroshi's
2026-08-24 comment **exactly**, including contraction order, and the production
path never fuses the two physical legs.

Six findings are raised. Ranked:

| ID | Severity | Verdict | Summary |
|---|---|---|---|
| F-4 | **High (performance)** | `SCOPE-DEVIATION` vs plan gate 1 | The tree path materializes the full probed local pair with **all `2·deg(v)` virtual bonds open** at every node before contracting any environment message — the exact O(χ⁴) (chain) / O(χ^{2·deg}) (tree) intermediate that `src_probe.rs`'s own comment at lines 424-429 says must be avoided. This is a plausible direct cause of the reported downstream slowness. |
| F-5 | **High (performance)** | `SUSPECT-UNVERIFIED` | `EnvironmentCache::batched_environments` is keyed by *width alone*. Because `site_max_width` varies per edge, the fixed-rank tree path recomputes the entire 2·\|E\|-message directed sweep once per distinct width — up to O(n) full tree sweeps, i.e. O(n²) message contractions. |
| F-1 | Medium (correctness risk) | `SUSPECT-UNVERIFIED` | The `.or_else(\|\| messages.get(&(parent, neighbor)))` fallback in both `directed_messages` functions is **structurally unreachable** and, if it were ever reached, would silently substitute a message flowing in the *opposite* direction — a wrong sketch instead of an error. The code contradicts its own adjacent comment, which correctly states the guarantee that makes the fallback dead. |
| F-2 | Medium (citation) | flagged citation error | `src_tree.rs` line 3 cites "Algorithm 1 and Sections 2.3--2.5" of the paper. §2 of the Tier-1 source has only §2.1 and §2.2. Same error as flagged independently by WS-chain. Python line-range citations are also inconsistent across the three SRC files for the same functions. |
| F-3 | Low | `HANDROLLED-DUPLICATE` (minor) | `standard_normal` (src_probe.rs:126-130) hand-rolls Box-Muller although `rand_distr::StandardNormal` is a workspace dependency already used for exactly this purpose in `tensor4all-core/src/defaults/idx_tensor.rs:188` (the crate `tensor4all-treetn` depends on). Also uses `StdRng`, whose stream is not guaranteed stable across `rand` versions, while `rand_chacha` is already available. |
| F-6 | Low | dead code / premature abstraction | Four `pub(super)` helpers in `src_probe.rs` — `site_probe`, `site_probe_batch`, `site_probe_batch_range`, `contract_prefix_with_probed_site_pair` — have **zero production call sites**; they are reachable only from the file's own `#[cfg(test)]` module. Notably `site_probe` is the one function in the whole SRC surface that *does* materialise the `∏d_i` (= `d²` for MPO-MPO) physical probe explicitly. It is dead, so this is not an actual `SCOPE-DEVIATION`, but it is a latent one. |

No `LICENSE-RISK`, no `MISSING-VS-SOURCE`, no `SOURCE-AMBIGUOUS`, no `SVD` in
the hot path, and no fused `d²` probe in production.

---

#### Provenance table

##### `src_tree.rs` (641 lines)

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_tree.rs` | Module doc comment (provenance claim) | 1-10 | `SOURCED-PAPER(Algorithm 1)` + **flagged citation error (F-2)** | The Algorithm-1 attribution and the "no tree implementation exists in the author repository" claim both check out (verified: `contraction.py` has no tree/graph code; all SRC entry points take `MPO`/`MPS` list types). "Sections 2.3--2.5" does not exist in the Tier-1 paper source. The `random_contraction_inc` range "lines 405--593" understates the function (def at 357). The self-labelling `[AI-Supplied]` for rooting/message-passing/complement-environments is honest and matches this audit's own conclusion that no Tier-1 source covers them. |
| `src_tree.rs` | Imports | 12-23 | n/a (trivial plumbing) | All linear algebra comes from `tensor4all_core` (`TensorLike`, `IndexLike`, `SvdTruncationPolicy`) and sibling `src_probe.rs`. No hand-rolled numerics imported. |
| `src_tree.rs` | `type DirectedEnvironment`, `type BatchedEnvironment` | 25-29 | n/a (trivial plumbing) | Type aliases only. `DirectedEnvironment<T,V> = HashMap<(V,V),T>` is the "cache keyed by `(from_node, to_node)`" the spec asks about — key shape verified correct in D-1. |
| `src_tree.rs` | `fn contract` — signature and trait bounds | 31-44 | n/a (trivial plumbing) | `V: Ord` is genuinely required (the deterministic `nodes.sort()` at line 71). |
| `src_tree.rs` | `fn contract` — chain-delegation guard | 45-59 | `DERIVED-VERIFIED` (D-2) | Delegates to `src_chain::contract` **only** when the topology is a chain *and* the requested center is the terminal site. Verified against `TreeTN::chain_order` (contraction.rs:404-450): `chain.last() == Some(center)` holds iff `center` is a degree-1 endpoint (or n=1). The adjacent comment ("The chain recurrence produces a left-canonical sweep whose center is the final site. An interior requested center needs the rooted tree recurrence") is accurate. |
| `src_tree.rs` | `fn contract` — topology / center / non-empty validation | 60-71 | `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed. Engineering preconditions with no paper analogue (the paper's inputs are chains by construction). Note the ordering smell: `chain_order` is consulted *before* `same_topology(tn_b)` is checked, so the chain path is entered without this file having validated topology — `src_chain::contract` repeats the check itself (src_chain.rs:46-48), so this is harmless. |
| `src_tree.rs` | `fn contract` — rooted-edge extraction + connectedness check | 72-81 | `DERIVED-VERIFIED` (D-1, step 1) | `edges_to_canonicalize_by_names(center)` verified to return a genuine **postorder** child→parent listing: `NodeNameNetwork::edges_to_canonicalize(None, target)` (node_name_network.rs:430-440) calls `post_order_dfs_by_index(target)` and then `compute_parent_edges` emits `(node, parent(node))` in that order. The `edges.len() + 1 != nodes.len()` check enforces a connected tree, which the derivation relies on. |
| `src_tree.rs` | `fn contract` — `sim_internal_inds`, `sketch_options`, `local`, `outputs` | 83-103 | `SOURCED-PAPER(Appendix C.2 `sec:adaptivity`, last ¶)` for `sketch_options`; `DERIVED-VERIFIED (trivial plumbing)` for the rest — `sim_internal_inds`, `local` and `outputs` are index-renaming and per-node lookup bookkeeping with no mathematical content | `sketch_options(svd_policy.is_some())` tightens `rtol` to `0.1·rtol` when `final_svd` is on — exactly the paper's "set the relative tolerance to be 0.1 times the requested tolerance and run a final truncation with the requested tolerance" (`sec:adaptivity`, final ¶). Verified in `SrcOptions::sketch_options` (contraction.rs:1428-1437). **Dead defensive code:** line 95-97 does `local.get(node).ok_or_else(...)?;` and discards the result, but `local` was built by zipping the same `nodes` list two statements earlier — the lookup cannot fail. |
| `src_tree.rs` | `fn contract` — global probe-index collection + zero-dimension guard | 105-118 | `SOURCED-PAPER(§2.2, `sec:krp`)` + `DERIVED-VERIFIED` (D-3) | Collecting *all* sites' physical output indices into one `ProbeBank` with one global column counter is precisely the paper's Khatri-Rao reuse requirement — "we use a common set of random matrices Ω^(1),…,Ω^(n-1) across all steps of the algorithm" (§3.5, discussion of Theorem 3). `sort_indices_deterministic` makes the RNG consumption order reproducible. See D-3 for why the reuse property is preserved across *edges* in the tree, which is the non-trivial part. |
| `src_tree.rs` | `fn contract` — `ProbeBank::new(.., 1, seed)` + `EnvironmentCache::new` | 119-121 | see the `src_probe.rs` / `EnvironmentCache` rows | Initial width 1; grown on demand. |
| `src_tree.rs` | `fn contract` — result containers | 123-126 | n/a (trivial plumbing) | The comment "Rooted edges are in child-to-parent postorder. Every projected child bridge therefore exists before its parent source is assembled" is **accurate** (verified above). |
| `src_tree.rs` | `fn contract` — per-edge loop: `source_factors`, `edge_bonds`, `cut_dimension`, `left_indices` | 128-145 | `DERIVED-VERIFIED` (D-1, step 3) | `source_factors = [A_child, B_child] ++ projected_children[child]` is the tree analogue of the paper's `B^(j)`; `left_indices` (all indices appearing exactly once, minus the two parent bonds) is the tree analogue of the paper's row split "(bond to η^(j+1), physical j)". `cut_dimension = dim(bond_a)·dim(bond_b)` is a valid and tight rank bound (D-4). |
| `src_tree.rs` | `fn contract` — scalar-subtree branch (`left_indices.is_empty()`) | 146-157 | `DERIVED-VERIFIED` (D-5) | A dim-1 structural bridge with `factor = ones([cap])`, which is a legitimate 1×1 isometry, so the QB step is exact by inspection. Reachable and covered by `src_preserves_scalar_only_subtrees_with_dimension_one_bridges` (tests/mod.rs:890). |
| `src_tree.rs` | `fn contract` — width selection (`row_dim`, `maximum_site_width`, `initial_width`) | 158-169 | `SOURCED-PYTHON(contraction.py:415-417)` + `SOURCED-PAPER(sec:adaptivity)` | See the `maximum_site_width`/`initial_width` rows under `src_probe.rs`. **Dead value:** the `else { site_max_width }` arm at line 168 is reachable but its value is unused downstream — it executes on every fixed-rank call, yet `site_initial_width` is consumed only inside the `sketch_options.rtol.is_some()` branch at line 196. |
| `src_tree.rs` | `fn contract` — fixed-width batched sketch + QR | 170-196 | `SOURCED-PAPER(§3.1-§3.3, Algorithm 1 lines 5-6/10-11)` + `SOURCED-COMMENT(#563, 2026-07-29)` + `DERIVED-VERIFIED` (D-1, step 4) | `contract_retaining(source_factors ++ [environment], batch)` then `factorize_full_rank(left_indices, FactorizeAlg::QR, Canonical::Left)` — QR only, delegated to `tensor4all-core`'s typed factorization API. No SVD, no hand-rolled linear algebra. Row/column split matches Algorithm 1's `Y^(j)(a,b,c) = Σ_d η^(j)(d,b,c) R^(j)(a,d)` with `a` = sketch-column index. |
| `src_tree.rs` | `fn contract` — adaptive per-column path | 196-216 | `SOURCED-PAPER(Appendix C, `sec:adaptivity`)` + `DERIVED-VERIFIED` (D-1, step 4) | The `make_column` closure builds one sketch column from the cached per-column complement environment; column caching is in `EnvironmentCache::column`. Satisfies plan performance gate 4 ("cached environment columns are not recomputed during adaptive growth") — verified, each `(parent,child,column)` message is computed once and reused across the adaptive loop *and* across edges. |
| `src_tree.rs` | `fn contract` — projection `conj(factor) × source_factors` | 217-229 | `SOURCED-PAPER(§3.1/§3.2, "projection" step; Algorithm 1 lines 8/12)` + `DERIVED-VERIFIED` (D-1, step 5) | Exactly `B^(j-1) = (η^(j))^† B^(j)` generalized to a tree: `factor.conj()` shares `left_indices` with the source factors, so the contraction is the adjoint projection. The result's open indices are the two cut bonds plus the new cap — the same shape as the paper's `S^(j)(a,b,c)`. |
| `src_tree.rs` | `fn contract` — store factor + bridge | 231-236 | `DERIVED-VERIFIED` (D-1, step 6) | `projected` accumulates into `projected_children[parent]`; postorder guarantees availability. |
| `src_tree.rs` | `fn contract` — root/center assembly (`merge_projected`) | 238-247 | `SOURCED-PAPER(§3.3, Algorithm 1 line 14 "Determine the first site η^(1)")` + `DERIVED-VERIFIED` (D-1, step 7) | The paper's η^(1) = contract-down of `B^(1)` with one incoming `S^(2)`; the tree center absorbs *k* incoming bridges instead of one. This is the only genuinely new structural element versus the chain and it is the correct generalization (D-1, step 7). |
| `src_tree.rs` | `fn contract` — result `TreeTN` assembly | 249-258 | `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed. Mechanical rebuild; edges reconnected via `connect_result_edge` on the same rooted-edge list, reproducing the input topology exactly. |
| `src_tree.rs` | `fn contract` — final SVD gate / canonical marking | 260-271 | `SOURCED-PAPER(§3.4 `sec:final_round`, Algorithm 1 line 15)` + `SOURCED-COMMENT(#563, 2026-07-29T07:13:51Z)` | **This is the only SVD site in either audited file.** It is gated on `src_options.final_svd`, applied to the fully-assembled `result` after every per-edge QR, matching Hiroshi's "That SVD acts on the already-compressed MPS, so it is cheap." When `final_svd` is off there is no SVD anywhere. |
| `src_tree.rs` | `struct EnvironmentCache` + fields | 273-286 | `DERIVED-VERIFIED` (D-1, D-6) | `environments: Vec<HashMap<(V,V),T>>` indexed by probe column; `batched_environments: HashMap<usize, BatchedEnvironment<T,V>>` keyed by width. The `(from,to)` key shape the spec asks about is correct. **See F-5** on the width-only keying. |
| `src_tree.rs` | `EnvironmentCache::new` | 288-312 | n/a (trivial) | Field initialisation only. |
| `src_tree.rs` | `EnvironmentCache::ensure_width` | 314-352 | `DERIVED-VERIFIED` (D-6) + **F-4** | For each not-yet-computed column: probe every site, run the two-pass directed message sweep, keep only the `(parent, child)` complement messages. Correctly incremental (`for column in self.environments.len()..width`), so no recomputation. **F-4**: `probed_site_pair` materialises the full `2·deg(v)`-bond local pair at every node before any message is contracted. |
| `src_tree.rs` | `EnvironmentCache::batch` | 354-413 | `DERIVED-VERIFIED` (D-6) + **F-4, F-5** | Same sweep with a width-`w` batch index instead of one column. Always starts at probe column 0, so the width-`w1` and width-`w2` batches are *nested prefixes of the same Ω* — this is what preserves the paper's reuse requirement across edges with different widths (D-3). **F-5**: the whole sweep is recomputed for each distinct width. |
| `src_tree.rs` | `EnvironmentCache::column` | 415-429 | `DERIVED-VERIFIED` (D-6) | On-demand `ensure_width(column+1)` then lookup. |
| `src_tree.rs` | `fn merge_projected` | 431-442 | `DERIVED-VERIFIED` (D-1, step 7) | Contracts `A_center`, `B_center` and all incoming bridges. |
| `src_tree.rs` | `fn site_factors` | 444-455 | n/a (trivial) | Builds the borrow list; identical body to `merge_projected` minus the final contract. |
| `src_tree.rs` | `fn uncontracted_indices` | 457-475 | `DERIVED-VERIFIED` (D-1, step 3) | Counts external-index occurrences across the factor list, keeps those appearing exactly once and not in the excluded bond set. Correct because in a tree the child's own physicals and its grandchildren's caps each appear exactly once in `source_factors`, whereas the bonds to already-absorbed subtrees appear twice. `sort_indices_deterministic` fixes the row ordering, which matters for reproducibility of the QR. |
| `src_tree.rs` | `fn contract_factors` | 477-486 | n/a (trivial) | Single-element short-circuit + error context. |
| `src_tree.rs` | `fn edge_bonds` | 488-514 | `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed — a pair of lookups. Looks up the A-bond and B-bond of the `(child,parent)` edge in the two (sim'd) networks. These are the two legs crossing the cut, and their product is `cut_dimension`. |
| `src_tree.rs` | `fn directed_messages` (unbatched) | 516-578 | `DERIVED-VERIFIED` (D-1, step 2) + **F-1** | Two-pass belief-propagation-style sweep on a tree: upward (postorder) then downward (reverse postorder). Independently re-derived and confirmed correct, including the ordering guarantees. **F-1**: the `.or_else(\|\| messages.get(&(parent.clone(), neighbor.clone())))` at line 564 is dead and would be wrong if live. |
| `src_tree.rs` | `fn directed_messages_batched` | 580-641 | `DERIVED-VERIFIED` (D-1, step 2; D-7) + **F-1** | Identical structure with `contract_retaining(&factors, batch)` in place of `contract_factors`. Batched semantics verified to be per-column (diagonal in the retained index), not an outer product — see D-7. Same dead `.or_else` at line 626. Note this function lacks the two explanatory comments its unbatched twin carries (lines 527-528, 552-553), which is the *only* difference in documentation between two otherwise-parallel functions. |

##### `src_probe.rs` (1171 lines)

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_probe.rs` | Module doc comment (provenance claim) | 1-13 | `SOURCED-PAPER(§2.2 `sec:krp`)` + partially flagged (F-2) | "Section 2.2" for the `Ω^(1) ⊙ … ⊙ Ω^(n)` definition **is correct** — §2.2 is exactly `sec:krp`. "Algorithm 1" is correct. The Python line ranges `random_contraction` 82-353 / `random_contraction_inc` 357-593 are accurate to within 1-3 trailing lines (actual: 82-356 / 357-594). The "issue #563 comment 5396107820" attribution for the A/B probe partition matches Appendix A's 2026-08-24 comment. This header is materially more accurate than `src_tree.rs`'s. |
| `src_probe.rs` | Imports | 15-22 | n/a (trivial plumbing) | `rand::{Rng, SeedableRng}` — see F-3 on the absence of `rand_distr`. |
| `src_probe.rs` | `struct ProbeBank` + doc | 24-35 | `SOURCED-PAPER(§2.2)` + `DERIVED-VERIFIED` (D-3) | Doc claim "coefficients are stored as a column-major `dim × width` matrix" **verified accurate** against `extend_to`'s append order. Doc claim "an adaptive run observes exactly the same prefix as a fixed-width run with the same seed and index order" **verified accurate** (D-3) and tested (`probe_bank_extension_preserves_the_existing_prefix`). |
| `src_probe.rs` | `ProbeBank::new` | 37-70 | `SOURCED-PAPER(§2.2)` | Per-index Gaussian columns = the paper's per-site `Ω^(i) ∈ C^{d×χ̄}`. Rejects zero-dimensional and duplicated indices. Overflow-checked capacity. |
| `src_probe.rs` | `ProbeBank::width`, `::coefficients` | 72-80 | n/a (trivial) | Accessors. `coefficients` is used only by tests. |
| `src_probe.rs` | `ProbeBank::column` | 82-104 | `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed — a bounds-checked slice. Bounds-checked column slice out of the column-major buffer. Checked arithmetic. |
| `src_probe.rs` | `ProbeBank::extend_to` | 106-123 | `SOURCED-PAPER(Appendix C `sec:adaptivity`, step 4)` + `SOURCED-PYTHON(contraction.py:431-432)` | "If χ̄ is larger than the number of columns in Ω^(1:j-1), then append columns to Ω^(1:j-1) until it has χ̄ in total." Python does the same with `for idx in range(len(envs), current_sketchdim)`. The append order (outer loop over columns, inner loop over `self.indices` in fixed order) is what makes the prefix property hold — D-3. |
| `src_probe.rs` | `fn standard_normal` | 126-130 | `SOURCED-PAPER(§2.2: "standard (real or complex) normal entries")` + `SOURCED-PYTHON(contraction.py:432 `np.random.randn`)` for the *real* choice; **`HANDROLLED-DUPLICATE` (F-3)** for the implementation | The distribution is right and real Gaussians are explicitly sanctioned by both the paper and the reference implementation. But Box-Muller is hand-rolled where `rand_distr::StandardNormal` is a workspace dep already used at `tensor4all-core/src/defaults/idx_tensor.rs:188`. Also discards the sine branch (2 uniforms per sample). |
| `src_probe.rs` | `fn single_probe` | 132-145 | `SOURCED-COMMENT(#563, 2026-08-24)` | One rank-1 probe vector `X[:,k]` as a **one-index tensor**. This is the factorized form Hiroshi asked for. Real-valued (`AnyScalar::new_real`), so the comment's `conj(X)` and the code's un-conjugated `X` agree identically — see D-8. |
| `src_probe.rs` | `fn single_probe_batch` | 147-177 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed beyond `single_probe`'s: the batch variant of an already-derived function, emitting the same `X[:,k]` columns as a `(index, batch)` two-index tensor over a contiguous column range. The `first_column` parameter is genuinely exercised with non-zero values — `src_chain.rs` supplies it positionally as its own `start` argument (src_chain.rs:548, 560, 632, 644), so this is not premature generalization. |
| `src_probe.rs` | `fn site_probe` | 179-206 | **dead production code (F-6)**; latent `SCOPE-DEVIATION` if ever wired up | Builds the *outer product* of all of a site's probe vectors as a single dense tensor of `∏ dim` entries — for MPO-MPO that is the explicit `d²` object Hiroshi's 2026-08-24 comment says never has to be formed. Zero production call sites (verified by grep across `crates/tensor4all-treetn/src/`); referenced only by this file's own test module. |
| `src_probe.rs` | `fn site_probe_batch` | 208-221 | **dead production code (F-6)** | Thin `first_column = 0` wrapper over `site_probe_batch_range`. Zero production call sites. |
| `src_probe.rs` | `fn site_probe_batch_range` | 223-271 | **dead production code (F-6)** | Batched version of the same fused construction, plus a scalar-outputs branch returning `ones([batch])`. Zero production call sites. |
| `src_probe.rs` | `fn product_dim` | 273-279 | n/a (trivial) | Overflow-checked product of index dimensions. |
| `src_probe.rs` | `fn probed_site_pair` | 281-307 | `SOURCED-COMMENT(#563, 2026-08-24T13:45:35Z)` — **exact match** | See D-8. `A × X(s)`, then `B × Y(t)`, then contract the two operands over the shared physical `u` (and any shared virtual bonds). Never forms a `d²` object. The doc comment's claim ("the shared physical leg is contracted between the operands without first constructing a fused `d^2` local product") is **verified true**. Numerically pinned by `probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`, which asserts against a literal `Σ_{s,t,u} A[s,u] B[u,t] x[s] y[t]` oracle. |
| `src_probe.rs` | `fn probed_site_pair_batch_range` | 309-349 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED (trivial plumbing)` | Trivial beyond `probed_site_pair`, whose derivation (D-8) it inherits: the batch variant of an already-derived function, differing only in that the probes carry a batch axis and the joins go through `contract_retaining` (batched semantics derived separately in D-7). Same order, batched. The `outputs.is_empty()` branch (scalar sites) contracts the pair and broadcasts over the batch — correct, since a site with no physical output contributes a probe factor of 1 in every column. |
| `src_probe.rs` | `fn contract_prefix_with_site_pair` | 351-366 | `SOURCED-PYTHON(contraction.py:462-471)` | Comment claims the ordering "mirrors the reference `env @ psi[j]` then `H[j] @ ...`". **Verified true**: in `operator/apply.rs:405-409` the SRC dispatch passes `tn_a = transformed_state` (the MPS ψ) and `tn_b = full_operator.mpo()` (the MPO H), so `prefix × tensor_a` really is `env @ psi[j]`. Loose wording: it calls itself "the unbatched counterpart of `contract_prefix_with_probed_site_pair_batch_range`" when the exact unbatched-probed counterpart is `contract_prefix_with_probed_site_pair`. |
| `src_probe.rs` | `fn contract_prefix_with_probed_site_pair` | 368-396 | **dead production code (F-6)** | Zero production call sites; referenced only from this file's tests. Its INVARIANT comment is accurate about what it does. |
| `src_probe.rs` | `fn contract_prefix_with_probed_site_pair_batch_range` | 398-439 | `SOURCED-COMMENT(#563, 2026-08-24)` + `SOURCED-PYTHON(contraction.py:462-486)` + `DERIVED-VERIFIED` (D-9) | The **cost-correct** ordering: prefix into A, A-probes, then B, then B-probes. The in-code claim "Building the complete local MPO-MPO product first would expose both virtual bonds at once and costs O(chi^4) storage for a single probe block" is **verified true** by explicit index counting (D-9). Used only by `src_chain.rs`. This is precisely the ordering the tree path does *not* use — F-4. |
| `src_probe.rs` | `fn partition_probes` | 441-485 | `SOURCED-COMMENT(#563, 2026-08-24)` | Assigns each output index to the operand that carries it, and hard-errors on the two structurally-invalid cases (index in neither operand / in both). Unlike F-1's fallback, these errors are correct: an index in *both* operands would be a contracted leg, not an output, so `local_output_indices` would not have emitted it — but erroring is the right response, not silently guessing. |
| `src_probe.rs` | `fn contract_operand_with_probes` | 487-515 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` (D-7) | Unbatched branch contracts all probes at once; batched branch folds them in one at a time with `contract_retaining` so the batch axis stays diagonal. The `probes.is_empty()` early return handles the MPO-MPS case (only one operand carries an output index) correctly in both branches. |
| `src_probe.rs` | `fn contract_site_pair` | 517-528 | n/a (trivial) | Contract `A`, `B` and extra factors in one call; used by `src_chain.rs` only. |
| `src_probe.rs` | `fn contract_retaining` | 530-550 | `DERIVED-VERIFIED` (D-7) | Delegates to `T::contract_retaining_indices` (batched/diagonal semantics verified from its doctest at `tensor_like.rs:869-885`), then normalises the batch index to the trailing axis via `permuteinds`. The permutation branch is defensive but harmless. |
| `src_probe.rs` | `fn site_operands` | 552-575 | `DERIVED-VERIFIED (trivial plumbing)` — four `HashMap` lookups returning borrowed references, no arithmetic; **doc/code mismatch (AI-hallucination signature)** | Doc says "Contract corresponding tensors at every named site once." The function **contracts nothing** — it performs four lookups and returns a borrowed pair. Confident doc text describing behaviour the code does not implement. |
| `src_probe.rs` | `fn local_site_pairs` | 577-590 | n/a (trivial) | Map over `site_operands`. No doc comment. |
| `src_probe.rs` | `fn local_output_indices` | 592-643 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` (D-8) | Collects the bond indices of both networks at the node, then takes the **symmetric difference** of the two operands' non-bond externals. For MPO-MPO `A:(s,u)`, `B:(u,t)` this yields exactly `{s,t}` and drops the contracted `u` — which is what makes the factorized probe well-defined. For MPO-MPS `A:(s,u)`, `B:(u)` it yields `{s}`. Redundantly recomputes `site_operands` that the caller already has (minor). |
| `src_probe.rs` | `fn fixed_probe_width` | 645-660 | `SOURCED-PAPER(§3.4 `sec:final_round`)` — **verbatim** | `max(⌈1.5χ̄⌉, χ̄+10)` when oversampling, else `χ̄`, capped by the row dimension. The paper: "As a sensible default, we recommend χ̄' = max(⌈1.5χ̄⌉, χ̄+10)." Exact. |
| `src_probe.rs` | `fn maximum_site_width` | 662-674 | `SOURCED-PYTHON(contraction.py:415)` + `DERIVED-VERIFIED` (D-4) | `min(max_rank, row_dimension, cut_dimension)` is the same triple bound as the reference's `current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)`. The Rust uses the *exact* parent-edge cut instead of the reference's looser `max(left,right)` site bound — tighter and still valid (D-4). Pinned by `maximum_probe_width_respects_the_exact_product_cut_dimension`. |
| `src_probe.rs` | `fn initial_width` | 676-679 | `SOURCED-PAPER(Appendix C `sec:adaptivity`: χ̄_0)` + `SOURCED-PYTHON(contraction.py:416-417)` | `min_rank.min(max).max(1)`; `SrcOptions::default().min_rank == 2` matches the paper's stated experimental χ̄_0 = 2. |
| `src_probe.rs` | `fn factorize_probe_columns` | 681-742 | `SOURCED-PAPER(Appendix C, `eq:err-est` + `eq:norm_est` + `sec:adaptivity` steps 1-5)` + `DERIVED-VERIFIED` (rank-deficiency stop, D-10) | The stopping test `estimate.error <= atol + rtol·estimate.norm` is **exactly** the paper's `Êrr^(j) ≤ τ_abs + τ_rel · N̂orm`. I verified the estimator's own definition end-to-end (`backend.rs:490-540`): `error = sqrt((1/p)·Σ_i ‖g_i‖^{-2})` with `G = R^{-†}` = `eq:err-est`, and `norm = ‖R‖_F/√p` = `eq:norm_est`. Growth by `rank_increment` (default 3 = the paper's Δ_χ) capped at `maximum_width` = `sec:adaptivity` steps 3-5. The extra `factorized.rank < width` saturation rule is not in the paper or the reference — derived and confirmed sound (D-10), and it is correctly short-circuited *before* the estimator is called (which requires a square `R`). |
| `src_probe.rs` | `fn connect_result_edge` | 744-776 | `DERIVED-VERIFIED (trivial plumbing)` | Trivial, no source needed — a one-index set intersection whose uniqueness argument is given inline. Rediscovers the shared cap index via `index_ops::common_inds` and takes `.first()`. Verified that exactly one index can be shared between a child factor and its parent tensor (child factor holds its own physicals + grandchild caps + its cap; the parent holds only the cap of these). Mild fragility: the cap index is *known* at the QR site (`factorized.bond_index`) and discarded there (`let (factor, _cap) = ...`, src_tree.rs:171), then rediscovered by set intersection. Correct but avoidably indirect. |
| `src_probe.rs` | `fn mark_result_canonical` | 778-808 | `DERIVED-VERIFIED` (D-11) | Records `set_canonical_region([center])`, `set_edge_ortho_towards(edge, parent)` per rooted edge, and `CanonicalForm::Unitary`. Verified consistent: `factorize_full_rank(left_indices, QR, Canonical::Left)` yields a `Q` isometric from cap→left, i.e. isometric *towards the center* across each edge, and `CanonicalForm::Unitary` is documented in `algorithm.rs` as "Each tensor is isometric towards the orthogonality center. Uses QR decomposition internally." Skipping a redundant canonicalization sweep is a legitimate optimization, not a shortcut. Minor doc inaccuracy: "SRC constructs every non-center factor with a left-canonical QR" is false for the scalar-subtree branch, which uses `T::ones` — still isometric, so the recorded metadata remains true. |
| `src_probe.rs` | Test module header + imports | 810-819 | n/a (trivial plumbing) | |
| `src_probe.rs` | `probe_bank_extension_preserves_the_existing_prefix` | 821-852 | real correctness check | Pins the prefix property the Khatri-Rao reuse argument depends on (D-3). Compares an extended bank against a from-scratch bank of the final width. |
| `src_probe.rs` | `probe_bank_rejects_zero_width_and_zero_dimensional_indices` | 854-861 | real check (input validation) | |
| `src_probe.rs` | `site_probe_uses_column_major_tensor_product_order` | 863-878 | real check, but **tests dead code (F-6)** | Pins the layout convention of `site_probe`, which has no production call sites. |
| `src_probe.rs` | `probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg` | 880-917 | **the key Tier-1 test** | Asserts against a literal `Σ_{s,t,u} A[s,u]·B[u,t]·x[s]·y[t]` oracle written out in the test — i.e. directly against Hiroshi's `E_k` formula, not against the implementation. This is the single most load-bearing test in the file. |
| `src_probe.rs` | `batched_probed_site_pair_keeps_independent_mpo_probes_paired` | 919-976 | real check | Same oracle per column; also asserts the result carries *only* the batch index, which is what proves no fused physical object survives. |
| `src_probe.rs` | `prefix_probe_contraction_matches_local_product_but_uses_environment_first_order` | 978-1084 | real check | Establishes that the three prefix orderings agree numerically to 1e-10 with the naive `contract(prefix, local_pair)` reference. Covers the batched, unbatched-unprobed, and unbatched-probed variants. |
| `src_probe.rs` | `scalar_site_probe_batch_broadcasts_over_the_batch_axis` | 1086-1105 | real check | Covers the `outputs.is_empty()` branches. |
| `src_probe.rs` | `adaptive_factorization_requests_only_the_columns_it_needs` | 1107-1136 | real check (laziness) | Asserts `requested == 1`, i.e. the adaptive loop does not over-fetch columns. This is the test backing plan gate 4. |
| `src_probe.rs` | `maximum_probe_width_respects_the_exact_product_cut_dimension` | 1138-1145 | real check | Pins D-4's cut bound. |
| `src_probe.rs` | `final_svd_adaptive_sketch_uses_the_paper_safety_factor` | 1147-1155 | real check against Tier 1 | Asserts `rtol 1e-6 → 1e-7`, which is the paper's "0.1 times the requested tolerance" (`sec:adaptivity`). |
| `src_probe.rs` | `final_svd_without_a_tolerance_policy_keeps_the_requested_sketch_tolerance` | 1157-1162 | real check | |
| `src_probe.rs` | `adaptive_src_defaults_to_the_requested_tolerance_without_final_round` | 1164-1170 | real check | |

---

#### SVD / hand-rolled-linear-algebra audit

`grep -n -i "svd"` over both files returns:

- `src_tree.rs`: line 16 (`use ... SvdTruncationPolicy`), 36 (parameter), 54
  (pass-through to `src_chain`), 85 (`sketch_options(svd_policy.is_some())`),
  260 (`if src_options.final_svd`), 263 (pass-through to `truncate_impl`).
  **Exactly one SVD call site: `result.truncate_impl(...)` at line 261**, gated
  on `final_svd`, applied to the fully-assembled output.
- `src_probe.rs`: lines 649/651/672 are the `final_svd: bool` oversampling
  *flag*, not an SVD call; the remainder are test names. **No SVD call.**

Factorization calls in these files are `factorize_full_rank(..., FactorizeAlg::QR, ...)`
(src_tree.rs:184-188) and `T::factorize_probe_columns_incremental` (src_probe.rs:709),
whose default implementation is a stacked QR (`tensor_like.rs:1066-1070`). Both
delegate to `tensor4all-core`'s typed factorization API. **No hand-written QR,
SVD, matrix inverse, or LAPACK wrapper exists in either file.** The only
hand-rolled numerical routine anywhere is `standard_normal` (F-3), which is a
random-number transform, not linear algebra.

This satisfies Hiroshi's 2026-07-29 QR-only claim without exception.

#### Fused-`d²`-probe audit

`grep -niE "combiner|fuse|fused|reshape|merge_ind|combine_ind|group_ind"` over
`src_tree.rs`, `src_probe.rs` and `src_chain.rs` returns **only two hits, both
comments asserting the absence of fusion** (src_probe.rs:285 and :401). There is
no index-fusing operation anywhere in the SRC production path.

The only place a `∏ d_i`-sized physical probe object is constructed is
`site_probe`/`site_probe_batch_range` (src_probe.rs:179-271), which have **zero
production call sites** — F-6. Every production probe flows through
`single_probe`/`single_probe_batch` → `partition_probes` →
`contract_operand_with_probes`, which keeps `X` and `Y` as separate one-index
tensors contracted into separate operands.

**Verdict: no `SCOPE-DEVIATION` on the fused-probe axis.** The production path
honours Hiroshi's 2026-08-24 ask.

---

#### Detailed derivations and flagged findings

##### D-1 — Full re-derivation of the rooted-tree SRC recurrence from the paper's chain recursion

This is the central derivation. It is performed from the paper's equations, not
from `docs/plans/2026-08-26-treetn-src-contraction-plan.md` (Tier 2) and not
from the code's own comments.

**What the paper actually does (chain).** From §3.1-§3.3 and Algorithm 1:

Let the network be `H|ψ⟩` on sites `1..n`. Define `B^(n) := H|ψ⟩`. The algorithm
sweeps `j = n, n-1, …, 2` and at each step performs a randomized QB approximation
of `B^(j)` viewed as a matrix whose **rows** are `(bond to η^(j+1), physical j)`
and whose **columns** are the physicals `1..j-1`:

1. *Collect.* `Y^(j) = B^(j) · Ω^(1:j-1)` where `Ω^(1:j-1) = Ω^(1) ⊙ … ⊙ Ω^(j-1)`.
   Algorithm 1 line 10 writes this out as
   `Y^(j) = Σ C^(j-1)(a,d,e) H^(j)(d,b,f,g) ψ^(j)(e,g,h) S^(j+1)(h,f,c)`.
2. *Orthonormalize.* `Y^(j) = η^(j) · R^(j)` by QR (Algorithm 1 line 11).
3. *Project.* `B^(j-1) = (η^(j))^† B^(j)`; the conjugate is absorbed into the
   local tensors to give `S^(j)` (Algorithm 1 line 13).

Two distinct kinds of environment appear in step 1 and they are **not**
interchangeable:

- `C^(j-1)`: the accumulated contraction of sites `1..j-1`, with **every one of
  those sites' physical legs contracted against the corresponding `Ω^(i)`
  column**. This is the *unprocessed* side.
- `S^(j+1)`: the accumulated contraction of sites `j+1..n`, with those sites'
  physical legs contracted against **`conj(η^(i))`, the already-determined
  isometries** — not against probes. This is the *processed* side.

This asymmetry is the whole content of the recursion. Any tree generalization
must reproduce it.

**Step 1 — rooting.** Choose the requested `center` as the root. `edges` =
`edges_to_canonicalize_by_names(center)`, verified above to be a postorder
child→parent listing of the `n-1` tree edges. Removing edge `(c,p)` splits the
tree into `S(c)` (the subtree rooted at `c`, containing `c`) and its complement
`S̄(c) = V \ S(c)` (containing `p` and the root).

Define the tree analogue of `B`: after all edges strictly inside `S(c)` have been
processed, the network restricted to `S(c)` is
`B_c := ( ∏_{g ∈ S(c)\{c\}} conj(η_g) ) · ( ∏_{v ∈ S(c)} A_v B_v )`,
with open indices = `c`'s own physicals, the caps of `c`'s children, and the two
bonds `(bond_a, bond_b)` crossing the cut.

**Step 2 — the complement environment.** For the QB at edge `(c,p)`, the columns
of the unfolding are indexed by the physicals of *all* nodes in `S̄(c)`. Sketching
those columns with the Khatri-Rao matrix `⊙_{v ∈ S̄(c)} Ω^(v)` means: contract
every node in `S̄(c)` with its own probe vector for column `k`, then contract the
whole complement down. Because `S̄(c)` is itself a tree hanging off `p`, that
contraction is exactly the message
`M_{p→c}^{(k)} = probed_k(p) · ∏_{u ∈ N(p)\{c}} M_{u→p}^{(k)}`
with the recursion terminating at leaves. This is the standard tree-message
recursion and it is what `directed_messages` computes.

The correctness of the *ordering* in `directed_messages` is a separate claim,
which I verified:

- *Upward pass* (`for (child,parent) in edges`, lines 529-550): computes
  `M_{c→p}` from `probed(c)` and `M_{u→c}` for `u ∈ N(c)\{p}`. In a rooted tree
  those `u` are exactly `c`'s children, whose edges precede `c`'s edge in
  postorder. ✔ available.
- *Downward pass* (`for (child,parent) in edges.iter().rev()`, lines 554-576):
  computes `M_{p→c}` from `probed(p)` and `M_{u→p}` for `u ∈ N(p)\{c}`. Those
  `u` are either other children of `p` (upward messages, all computed in pass 1)
  or `p`'s own parent `gp`. `M_{gp→p}` is produced when the edge `(p,gp)` is
  visited in the reverse pass. Since postorder places `(p,gp)` **after** `(c,p)`,
  the reversed order places it **before**. ✔ available.

So both passes are well-founded and every message the algorithm needs exists when
it is looked up. The messages are exactly the sketched complement environments.
**This confirms the `(from_node, to_node)`-keyed directed cache the spec asked
about.** The key must be directed, because `M_{p→c} ≠ M_{c→p}`.

**Step 3 — the row space.** `left_indices = uncontracted_indices(source_factors,
{bond_a, bond_b})`: indices appearing exactly once across `[A_c, B_c, bridges]`,
minus the two cut bonds. Enumerating: `c`'s physicals appear once (in `A_c` or
`B_c`); a grandchild cap appears once (only in that grandchild's bridge); the
bonds between `c` and an already-processed child appear twice (once in
`A_c`/`B_c`, once in the bridge) and are therefore excluded automatically; the
two cut bonds appear once each and are excluded explicitly. Result: exactly
`c`'s physicals + `c`'s children's caps. **This is precisely the tree analogue of
the paper's row split `(bond to η^(j+1), physical j)`** — with `k` incoming caps
instead of one. ✔

**Step 4 — collect + orthonormalize.**
`Y_c = contract(B_c , M_{p→c})`, contracting over the two cut bonds. Open indices
= `left_indices` (+ the batch/column index). QR with `left_indices` as rows gives
`Y_c = η_c · R_c` with `η_c` isometric from cap→rows. This is Algorithm 1
lines 10-11 verbatim, modulo the row space being `k+1`-fold instead of 2-fold. ✔

**Step 5 — project.**
`projected_c = conj(η_c) · [A_c, B_c, bridges]`, contracting over `left_indices`.
Open indices: the two cut bonds + the new cap. This is `B^(j-1) = (η^(j))^† B^(j)`
with the conjugate absorbed into the local tensors — the same object as the
paper's `S^(j)(a,b,c)`, which likewise carries two incoming bonds and one cap. ✔

**Step 6 — propagation.** `projected_c` is pushed into `projected_children[p]`.
When `p`'s own edge is processed, `source_factors` for `p` is
`[A_p, B_p] ++ projected_children[p]`, which by induction equals `B_p` as defined
in step 1. ✔ The induction is well-founded exactly because `edges` is postorder.

**Step 7 — the center.** After all `n-1` edges,
`root = A_center · B_center · ∏_{c ∈ children(center)} projected_c`. In the chain
this is `η^(1) := B^(1)` contracted down (Algorithm 1 line 14, §3.3's "we take the
`B^(1)` tensor network and contract it down"). In the tree the center absorbs `k`
bridges instead of one. Since every bridge is `conj(η)` applied to its subtree,
the product `∏_v (result tensor)_v` reconstructs
`( ∏_{v≠center} η_v ) · ( ∏_{v≠center} conj(η_v) ) · ( ∏_v A_v B_v )`
= `P · (A·B)` where `P` is the product of the per-edge projectors — exactly the
chain's telescoping identity `H|ψ⟩ = η^(n)·η^(n-1)·…·η^(2)·B^(1)`, extended to a
tree by the fact that the projectors act on disjoint cuts. ✔

**Step 8 — the chain reduces correctly.** Instantiate on the chain `1-2-3-4-5`
with center `3`. Postorder edges: `(1,2),(2,3),(5,4),(4,3)` (up to the DFS's
branch order). Edge `(1,2)`: source = `[A_1,B_1]`, environment = probed
contraction of `{2,3,4,5}`, rows = node 1's physicals. That is **exactly paper
Step 1** (§3.1) with the sweep started from the other end. Edge `(2,3)`: source =
`[A_2,B_2,bridge_1]` = `conj(η_1)·(local 1)(local 2)`, environment = probed
`{3,4,5}`, rows = node 2's physicals + `η_1`'s cap. That is **exactly paper
Step 2** (§3.2), `B^(n-1)` unfolded with `(bond, physical)` as rows and the
remaining physicals sketched by the Khatri-Rao prefix. The two branches `1→2→3`
and `5→4→3` are two independent instances of the chain recursion meeting at the
center, which is the correct generalization of the paper's "left-canonical sweep
whose center is the final site" to a two-sided sweep. ✔

**Conclusion of D-1: the tree recurrence is correct.** Verdict
`DERIVED-VERIFIED` for the whole `contract` body and both `directed_messages`
functions, with the exceptions raised as F-1, F-4 and F-5.

##### D-2 — The chain-delegation guard

`chain_order(center)` (contraction.rs:404-450) returns `None` unless the graph is
a path (exactly two degree-1 nodes, `edge_count == node_count - 1`). If `center`
is one of the two endpoints, the path is oriented so that `center` is `end`,
hence `chain.last() == Some(center)`; otherwise `center` is interior and the last
element is the other endpoint. So `src_chain::contract` is used exactly when the
paper's own one-sided recursion applies verbatim, and the rooted-tree recursion
otherwise. Correct, and the in-code comment describing this is accurate.

##### D-3 — Khatri-Rao reuse is preserved across tree edges

The paper's Theorem 3 (§3.5) depends on the *same* `Ω^(i)` being reused at every
step; §3.5 explicitly notes "the fact that we use a common set of random matrices
Ω^(1),…,Ω^(n-1) across all steps of the algorithm makes the result not entirely
trivial."

In `src_tree.rs` there is exactly one `ProbeBank`, keyed by physical index, with
one global column counter. `ProbeBank::extend_to` appends columns in a fixed
order (outer loop over new columns, inner loop over `self.indices` in the order
supplied at construction, which `contract` fixes with
`sort_indices_deterministic`). Therefore the first `w` columns of the bank are
identical no matter what final width is reached — the property tested by
`probe_bank_extension_preserves_the_existing_prefix`.

The subtle part is *across edges*. Different edges get different
`site_max_width`. Both the batched path (`EnvironmentCache::batch`, which always
calls `probed_site_pair_batch_range(..., first_column = 0, width, ...)`) and the
per-column path (`EnvironmentCache::column(.., column)`) draw from columns
`0..w_e` of the same bank. Hence for two edges with widths `w1 ≤ w2`, the
sketches use nested prefixes of one Khatri-Rao matrix. This is the exact analogue
of the paper's `Ω^(1:j-1)` being a prefix of `Ω^(1:n-1)`. ✔ **Verified.**

(If `first_column` had been advanced per edge — as the `first_column` parameter
would allow — the reuse property would break and Theorem 3's argument would not
carry over. It is not advanced. Good.)

##### D-4 — `cut_dimension` is a valid and tight rank bound

At edge `(c,p)`, the matrix being QB-approximated is `B_c` unfolded with
`left_indices` as rows and the complement's physicals as columns. The network
factorizes through the two bonds `bond_a`, `bond_b` crossing the cut, so the
unfolding has the form `L · R` where the inner dimension is
`dim(bond_a)·dim(bond_b)`. Hence
`rank ≤ min( ∏ dim(left_indices), dim(bond_a)·dim(bond_b) )`,
which is exactly `min(row_dimension, cut_dimension)` in `maximum_site_width`.

The reference Python uses the looser
`prod_bond_dims = max(H[j].shape[0]*psi[j].shape[0], H[j].shape[2]*psi[j].shape[2])`
(contraction.py:413, the `else` arm of the if/else block spanning 410-413) —
the max over both adjacent bonds, because in the chain
sweep it does not track which side is the cut. The Rust's per-edge cut is the
correct tightening for a rooted tree, where each edge is unambiguously *the* cut.
✔ `DERIVED-VERIFIED`, and consistent with `SOURCED-PYTHON` on the triple-min
structure.

##### D-5 — The scalar-subtree branch

`left_indices.is_empty()` requires the child to have no physical outputs *and*
no children (any grandchild cap would appear in `left_indices`). Then `B_c` has
only the two cut bonds open, and the "matrix" being factorized is `1 × N`. Any
rank-1 isometry is exact; the code uses `factor = ones([cap])` with
`dim(cap) = 1`, i.e. the 1×1 matrix `[1]`, which is unitary. `projected =
source ⊗ factor` pushes the entire scalar weight into the parent, so
`factor · projected = source`. Exact, not approximate. ✔

Reachability: confirmed by `src_preserves_scalar_only_subtrees_with_dimension_one_bridges`
(contraction/tests/mod.rs:890). The branch is live, not speculative.

`mark_result_canonical`'s doc claim that "SRC constructs every non-center factor
with a left-canonical QR" is inaccurate for this branch (`ones`, not QR), but the
*metadata* it records stays true because `[1]` is isometric.

##### D-6 — `EnvironmentCache`: correct, but see F-4/F-5

`ensure_width` is correctly incremental (`for column in
self.environments.len()..width`), and `column` triggers it on demand. So during
adaptive growth each `(parent,child,column)` message is computed exactly once and
shared across all edges — plan performance gate 4 is honoured on the tree path.
The `batch` path is separately memoized. Correctness: ✔. Cost: see F-4 and F-5.

One structural note in favour of the code: `ensure_width` computes the full
`directed_messages` map (both directions on every edge, `2|E|` messages) but
retains only the `|E|` `(parent,child)` complement messages. Since every rooted
edge needs its complement message, and the two-pass recursion needs the
child→parent messages as intermediates, this is the minimal work, not waste.

##### D-7 — Batched contraction really is per-column (not an outer product)

`contract_retaining` delegates to `T::contract_retaining_indices(tensors,
[batch])`. I verified the semantics from the trait's own doctest
(`tensor_like.rs:869-885`): for `left[b,c]`, `right[b,c]` it produces
`result[b] = Σ_c left[b,c]·right[b,c]` (asserted values `[14, 26]`), i.e. the
retained index is a **batch/diagonal** axis, not an outer-product axis. The
`IdxTensor` implementation routes 2-operand calls through
`try_contract_pairwise_retaining` and n-operand calls through
`contract_with_options(..).with_retain_indices(..)` (idx_tensor.rs:5981-5992).

This matters at three places where **both** operands carry the batch index:
`contract_retaining(&[&probed_a, &probed_b], batch)` in
`probed_site_pair_batch_range`; `contract_operand_with_probes`'s batched loop
after the first probe; and `directed_messages_batched`'s message contractions.
If the semantics were outer-product, every batched result would be wrong by a
factor of the width and the shapes would blow up. They are not. ✔

##### D-8 — The MPO-MPO factorized probe matches Hiroshi's 2026-08-24 comment exactly

Hiroshi's stated construction:

> `Ω_{s,t,k} = X_{s,k} Y_{t,k}` … `E_k = Σ_{s,t,u} X*_{s,k} A^{s,u} B^{u,t} Y*_{t,k}`
> … contracting `conj(X[:,k])` into the first operand, `conj(Y[:,k])` into the
> second, then the shared physical index, then incoming tree messages.

The code path, step by step:

1. `local_output_indices` returns the symmetric difference of the two operands'
   non-bond externals = `{s, t}` (the shared `u` is dropped, since it appears in
   both and is therefore the contracted leg).
2. `single_probe` / `single_probe_batch` builds `X[:,k]` and `Y[:,k]` as
   **separate one-index tensors**.
3. `partition_probes` assigns `X` to the operand containing `s` and `Y` to the
   operand containing `t`, hard-erroring if an output index is in neither or in
   both.
4. `contract_operand_with_probes(tensor_a, [X], …)` → `A·X`;
   `contract_operand_with_probes(tensor_b, [Y], …)` → `B·Y`.
   **"conj(X) into the first operand, conj(Y) into the second"** ✔
5. `T::contract(&[&probed_a, &probed_b])` contracts over the shared physical
   index `u` (and any shared virtual legs). **"then the shared physical index"** ✔
6. In `EnvironmentCache::ensure_width` / `::batch`, the resulting per-site
   objects are then fed to `directed_messages*`, which contract the incoming
   tree messages. **"then incoming tree messages"** ✔

Order, factorization, and the no-`d²` requirement all match. The fused
`Ω_{s,t,k}` is never built in the production path.

**On the conjugation.** Hiroshi writes `X*` and `Y*`; the code contracts `X` and
`Y` un-conjugated. This is *not* a discrepancy here, because the probe entries
are constructed as `AnyScalar::new_real(...)` from `f64` samples of
`standard_normal`, so `conj(X) = X` identically. The paper's Algorithm 1 likewise
writes `Ω^(1)(a,d)` with no conjugate (line 642), and the reference Python uses
`np.random.randn` (real) at contraction.py:432. Consistent across all three
Tier-1 sources. **However**, this is a latent trap: if anyone later switches the
probe bank to complex Gaussians (which §2.2 permits), the missing conjugate
becomes a real bug, and nothing in the code documents that dependency. Worth a
note in the fix pass; not a finding under this audit's taxonomy.

`probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`
(src_probe.rs:880-917) pins this against a hand-written oracle of exactly
Hiroshi's `E_k` formula, which is the right kind of test.

##### D-9 — The O(χ⁴) claim in `contract_prefix_with_probed_site_pair_batch_range` is true

The in-code comment (lines 424-429) claims that building the complete local
MPO-MPO product before folding in the prefix "would expose both virtual bonds at
once and costs O(chi^4) storage for a single probe block." Index counting, with
`χ_A = χ_B = χ` and sketch width `l`:

- *Pair-first:* `probed_site_pair(A,B)` has indices
  `(left_a, right_a, left_b, right_b)` → `χ⁴` entries, times `l` = `χ⁴ l`.
- *Prefix-first:* `prefix(left_a, left_b, batch)` × `A` → `(left_b, right_a, u,
  s, batch)` = `χ² d² l` (this is the peak); probe `s` away → `(left_b, right_a,
  u, batch)` = `χ² d l`; × `B` → `(right_a, right_b, t, batch)` = `χ² d l`; probe
  `t` away → `(right_a, right_b, batch)` = `χ² l`.

Peak intermediate `χ² d² l` vs `χ⁴ l`. Since `d ≪ χ` in any regime where SRC is
worth running, `χ² d² l ≪ χ⁴ l`. The comment is **accurate**. ✔ This
function is the correct implementation. The problem is that the tree path does
not use it — F-4.

##### D-10 — The `factorized.rank < width` early stop is sound

Neither the paper nor the reference Python has this rule; it is an addition.
Derivation: the sketch is `Y = A·Ω` with `Ω` a width-`p` Khatri-Rao matrix. If
`rank(Y) < p`, the `p` sketch columns are linearly dependent. For a Khatri-Rao
matrix with independent standard-normal factors, Theorem 2's argument
(`app:khatri-rao-exact`) gives that with probability one this happens only when
`rank(A) < p`, and in that case `range(A·Ω) = range(A)`, so the QB approximation
`Q Q^† A = A` is **exact**. Stopping is therefore correct with probability one —
the same probability-one qualifier the paper itself uses for Theorem 2/3. ✔
`DERIVED-VERIFIED`.

Ordering detail worth crediting: because `factorized.right` (the `R` factor) is
`rank × width` and is *not square* when `rank < width`, calling
`src_error_estimate()` on it would fail ("SRC estimator requires a square R",
backend.rs:433). The code's `if src_options.rtol.is_none() || saturated { true }
else { … estimate … }` short-circuits `||` before the estimator runs. This is
correct and non-obvious.

**Residual caveat (not a finding, but a dependency):** the soundness of this rule
rests entirely on `FactorizeResult::rank` using a *conservative* numerical rank
tolerance. A loose tolerance would let the loop stop early without ever
consulting the error estimator, silently under-approximating. That determination
lives in `tensor4all-core`/`tensor4all-tensorbackend` and belongs to WS-core /
WS-backend; flagging the cross-workstream dependency here.

##### D-11 — `mark_result_canonical` records a true orientation

`factorize_full_rank(left_indices, FactorizeAlg::QR, Canonical::Left)` yields
`Q` with `left_indices` as rows and orthonormal columns, i.e.
`Σ_{left} conj(Q[left,c]) Q[left,c'] = δ_{cc'}`. Contracting `Q` with its
conjugate over `left_indices` gives the identity on the cap index — so each
non-center factor is isometric when contracted *away from* the center, which is
what `set_edge_ortho_towards(edge, Some(parent))` asserts. `CanonicalForm::Unitary`
is documented in `algorithm.rs` as "Each tensor is isometric towards the
orthogonality center. Uses QR decomposition internally." ✔ Consistent.

Skipping the generic `TreeTN` canonicalization sweep (which would re-QR factors
that are already `Q`) is a legitimate optimization, correctly justified in the
doc comment. This is the *opposite* of a hallucination signature: a claim that
checks out.

---

#### AI-hallucination-signature findings

##### F-1 — Unreachable defensive fallback that would be silently wrong if reached (`SUSPECT-UNVERIFIED`)

**Pattern: unnecessary defensive code for a case that structurally cannot occur,
combined with a fallback that contradicts its own adjacent comment.**

`src_tree.rs:562-565` (and the identical `:624-627` in the batched twin):

```rust
let message = messages
    .get(&(neighbor.clone(), parent.clone()))
    .or_else(|| messages.get(&(parent.clone(), neighbor.clone())))
```

The comment three lines above (line 552-553) reads:

> // Downward pass: the reverse postorder guarantees that the parent-side
> // message is available before the next child is visited.

The comment is **correct** — I proved it in D-1 step 2: for a rooted edge
`(c,p)`, every `u ∈ N(p)\{c}` is either a child of `p` (upward message computed
in pass 1) or `p`'s parent (downward message computed earlier in the reversed
loop, because postorder puts ancestor edges later). So `messages.get(&(neighbor,
parent))` **always** succeeds and the `.or_else` arm is dead.

But the guarantee makes the fallback worse than useless. `M_{p→u}` is the message
flowing in the *opposite* direction from `M_{u→p}`; it covers the complement of
`u`'s subtree rather than `u`'s subtree. Substituting one for the other would
produce a silently wrong sketch — no shape error is guaranteed to catch it, since
both messages carry the same two bond indices. The code's own `ok_or_else` error
path ("side message is missing for …") is the correct behaviour and the
`.or_else` bypasses it.

This is a textbook hallucination signature: plausible-looking defensive code
whose only effect, if it ever fired, would be to convert a loud failure into a
quiet wrong answer — added despite an adjacent comment correctly explaining why
it can't fire. Verdict `SUSPECT-UNVERIFIED` on the fallback specifically (the
surrounding function is `DERIVED-VERIFIED`).

##### F-2 — Fabricated / inconsistent source citations (flagged citation error)

**Pattern: confident, specific-looking citation that does not resolve in the
Tier-1 source.**

`src_tree.rs:3-4`:

> //! Provenance: the per-edge sketch/QR/projection pattern is derived from
> //! Algorithm 1 and Sections 2.3--2.5 of Camaño--Epperly--Tropp,

§2 of `report.tex` contains exactly two subsections: §2.1 `sec:randomized-qb`
(line 230) and §2.2 `sec:krp` (line 283). **There is no §2.3, §2.4 or §2.5.**
The material actually being cited is §3.1-§3.3 (`sec:Alg1`/`sec:Alg2`/`sec:Alg3`).
WS-chain flagged the identical string in `src_chain.rs` independently, so this is
a copied-and-propagated error, not a one-off typo.

Mitigating possibility, stated for honesty: Hiroshi's 2026-07-29 comment cites
"Sec. 2.1 and Algorithm 1" for the QR-only core loop, which fits a numbering
where SRC is §2 rather than §3 — plausibly an earlier arXiv version. But the
issue-opening post cites "Sec. 3.6" for linear combinations, which matches the
*local* numbering, so the two do not both fit one alternative version. Against
the Tier-1 source this audit is instructed to use, the citation is wrong and
cannot be resolved. Flagged, not silently accepted.

Secondary inconsistency: the same two Python functions are cited with three
different line ranges across the three SRC files —
`random_contraction` as "82--353" (src_probe.rs) and "133--353" (src_chain.rs);
`random_contraction_inc` as "357--593" (src_probe.rs) and "405--593"
(src_tree.rs). Actual boundaries are 82-356 and 357-594. `src_probe.rs`'s
citations are accurate; `src_tree.rs`'s start line is 48 lines into the function.

##### F-3 — Hand-rolled Gaussian sampler duplicating existing repository functionality (`HANDROLLED-DUPLICATE`, minor)

`src_probe.rs:126-130` implements Box-Muller by hand:

```rust
let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
let u2 = rng.random::<f64>();
(-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
```

`rand_distr = "0.5"` is a workspace dependency (root `Cargo.toml:78`), and
`tensor4all-core` — which `tensor4all-treetn` depends on — already uses
`rand_distr::StandardNormal` for exactly this purpose in
`defaults/idx_tensor.rs:188` (`RandomScalar for f64`), with a `Complex64` variant
at :193. `tensor4all-treetn/Cargo.toml` pulls `rand` but not `rand_distr`, so the
duplication is a missing one-line dependency, not an architectural constraint.

Two secondary observations on the same function:

1. It discards the sine branch, consuming two uniforms per normal (2× the RNG
   work of `StandardNormal`, which is a Ziggurat implementation).
2. `ProbeBank` uses `rand::rngs::StdRng::seed_from_u64`. `StdRng`'s underlying
   algorithm is explicitly *not* guaranteed stable across `rand` releases, so the
   "deterministic seeding" the module doc advertises is only within-build
   determinism. `rand_chacha` (a reproducible-by-contract generator) is already a
   dev-dependency of this crate. This does not affect correctness but does affect
   the reproducibility claim in `src_probe.rs:1-13`.

The plan's prohibition is specifically on "a hand-written QR, SVD, matrix
inverse, or LAPACK wrapper in `treetn`", which this is not — hence "minor".

##### F-4 — The tree path materializes the full `2·deg(v)`-bond local pair at every node (`SCOPE-DEVIATION` vs plan performance gate 1; likely performance cause)

**Pattern: math that is correct but whose *cost structure* contradicts what the
same file's own comments say the code is designed to avoid.**

`src_probe.rs:424-429` states the design rule:

> // Building the complete local MPO-MPO product first would expose both virtual
> // bonds at once and costs O(chi^4) storage for a single probe block.

`contract_prefix_with_probed_site_pair_batch_range` honours this (D-9), and
`src_chain.rs` uses it at every interior site.

`src_tree.rs` does not. Both `EnvironmentCache::ensure_width` (lines 320-334) and
`EnvironmentCache::batch` (lines 373-398) build, **for every node in the tree**,
`probed_site_pair(_batch_range)(tensor_a, tensor_b, …)` — the complete local
MPO-MPO product with the probes contracted but *all* virtual bonds still open —
and store the whole map before `directed_messages*` contracts a single message
into any of them.

For a node of degree `k` with `χ_A = χ_B = χ`, that object has `2k` open bond
indices: `χ^{2k}` entries, times the sketch width `l`. Concretely:

| node degree | probed local pair size | chain equivalent |
|---|---|---|
| 1 (leaf) | `χ² l` | fine |
| 2 (chain interior) | `χ⁴ l` | **the exact object the comment forbids** |
| 3 (branch point) | `χ⁶ l` | — |

And they are all live simultaneously: the `probed` `HashMap` holds one per node,
so peak memory is `O(n · χ^{2·deg} · l)` before any message contraction happens.

This is not a correctness bug — `directed_messages` contracts them down
afterwards and the results are right (the tree tests pass). It is a cost bug, and
it lands squarely on the audit's motivating symptom: *"gw-rs's downstream
pipeline is very slow when it uses SRC — slower than the algorithm's own cost
model would predict."* An `O(χ⁴ l)` per-site intermediate where the cost model
predicts `O(χ² d l)` is exactly that shape of discrepancy.

Cross-referenced against the plan's "Performance acceptance gates" (Tier 2, used
here as a map to check, not as authority): gate 1, "No full dense operand or
output materialization in the production SRC path." The tree path materializes a
dense `χ^{2·deg}·l` local object per node. Whether that counts as "full dense
materialization" is arguable — it is not the full network — but it is
unambiguously the same class of intermediate the plan and the file's own comments
are trying to prevent, and it is O(χ²) larger than the chain path's peak.

The fix direction (out of scope for this report) is the same trick the chain
already uses: fold incoming messages into the operands *before* joining `A` and
`B`, so no node ever exposes both operands' full bond sets at once.

##### F-5 — Batched environment cache keyed by width alone, causing repeated full tree sweeps (`SUSPECT-UNVERIFIED`)

`src_tree.rs:285`:

```rust
batched_environments: HashMap<usize, BatchedEnvironment<T, V>>,
```

`EnvironmentCache::batch(parent, child, width)` looks up by `width` only; on a
miss it recomputes **the entire directed-message sweep for every node and every
edge** (lines 371-411) and caches the whole map under that width.

In `contract`, the width is computed per edge:

```rust
let site_max_width = maximum_site_width(max_bond_dim, row_dim, cut_dimension, &sketch_options);
```

`row_dim` (the child's physicals × its children's caps) and `cut_dimension`
(the parent edge's `χ_A·χ_B`) both vary across edges in any non-uniform network.
Each distinct value triggers a fresh `O(|E|)`-message sweep, each of which itself
builds the `n` probed local pairs of F-4. Worst case — all `n-1` widths distinct —
this is `O(n²)` message contractions and `O(n²)` probed-pair constructions where
`O(n)` suffices.

The per-column path (`ensure_width`/`column`) does not have this problem: it is
keyed by column and grows monotonically, so the fixed-rank path is the *worse* of
the two. That is the wrong way round — the fixed-rank path is supposed to be the
fast one.

Note that a single `BatchedEnvironment` computed at `max_e(site_max_width)` would
serve every edge, since all batches start at probe column 0 and narrower batches
are prefixes of wider ones (D-3). The caching key is simply too coarse in the
wrong direction.

No Tier-1 source speaks to this (it is a tree-only construct), and no derivation
in the code or the plan justifies the width-keyed design. `SUSPECT-UNVERIFIED`.

##### F-6 — Four `pub(super)` helpers with no production call sites (dead code / premature abstraction)

Verified by grepping all of `crates/tensor4all-treetn/src/` and subtracting
`src_probe.rs`'s own `#[cfg(test)]` module:

| Function | Lines | Production call sites |
|---|---|---|
| `site_probe` | 179-206 | 0 |
| `site_probe_batch` | 208-221 | 0 |
| `site_probe_batch_range` | 223-271 | 0 |
| `contract_prefix_with_probed_site_pair` | 368-396 | 0 |

That is ~120 lines of `pub(super)` API — with full doc comments and, in the last
case, an `// INVARIANT:` annotation — reachable only from the tests in the same
file. `src_chain.rs` imports neither; `src_tree.rs` imports neither.

Two aggravating details:

1. `site_probe` (and its batch variants) is the **only** function in the SRC
   surface that explicitly materializes the `∏ d_i` physical probe object — for
   MPO-MPO, the fused `d²` outer product `Ω_{s,t,k} = X_{s,k} Y_{t,k}` that
   Hiroshi's 2026-08-24 comment says "never has to be formed explicitly". It is
   dead, so this is **not** an actual `SCOPE-DEVIATION` today. But it is a loaded
   gun: the file ships, documents, and tests the exact construction the Tier-1
   comment rules out, in a form any future caller could pick up by name.
2. `site_probe_uses_column_major_tensor_product_order` (lines 863-878) is a
   genuine, careful test — of dead code. It contributes coverage numbers without
   testing anything the production path executes.

Matching the audit's stated pattern list: *premature or unjustified abstraction*.
Verdict: not a taxonomy verdict on its own (it is neither wrong math nor a
duplicate of an existing API); reported as a finding for the fix pass, with a
recommendation to delete rather than to wire up.

---

#### Non-findings, recorded explicitly

These were checked and came back clean; recording them so the synthesis pass can
state coverage rather than silence.

- **`LICENSE-RISK`: none.** The reference Python (`random_contraction`,
  `random_contraction_inc`) is written as explicit NumPy `reshape`/`transpose`/
  `@` sequences over raw arrays with hand-computed axis permutations. Both
  audited Rust files work through index-labelled tensors (`T::contract`,
  `contract_retaining_indices`) with no positional axis arithmetic at all. There
  is no structural correspondence that could read as a translation. The
  *parameter conventions* (real Gaussians, `min(maxdim, rows, cut)` width bound,
  `err ≤ tol·norm` stopping) match, which is validation, not copying.
- **`MISSING-VS-SOURCE`: none for these files.** Everything Algorithm 1 specifies
  for the per-site step (collect / QR / project, adaptive growth, oversampling
  factor, leave-one-out estimator, norm estimator) is present. The paper's §3.6
  "linear combinations of MPO-MPS products" extension is absent, but the issue's
  own "Proposed scope" lists it as a nice-to-have, not a requirement, and it is
  out of scope for these two files.
- **`SCOPE-DEVIATION` on #691 interface sketching: none.** No segment/interface
  projection machinery, no sub-TT partitioning, no `l'` interface sketch width
  anywhere in either file.
- **`SCOPE-DEVIATION` on the 2026-08-27T15:38 correction: none.** That comment
  forbids scan/tree *composition of the sequential cache's boundary operators*
  (`O(D³)` per composition with `D = χ²`). `src_tree.rs`'s directed messages are
  not that: they are messages on the physical network's own tree topology, each
  one a thin `(χ_A·χ_B) × l` boundary state produced by ordinary edge
  application, exactly the "thin boundary states … with no operator
  materialization" the correction says survives. The tree's `O(n)` message depth
  is intrinsic to the network's shape, not a parallelization scheme layered on a
  chain. (F-4's problem is a *local* intermediate, not a composed transfer
  operator.)
- **`SOURCE-AMBIGUOUS`: none.** No two Tier-1 statements conflict on anything
  these files depend on. The 12:56 → 15:38 supersession affects only
  `PrefixCache`/parallelism, which lives in `src_chain.rs` (WS-chain) and
  `contraction.rs` (WS-integration).
- **`PrefixCache` trait ask:** not applicable to these files. `EnvironmentCache`
  is a tree-specific complement-environment cache, not the chain prefix product
  Hiroshi's ask names. WS-chain owns that finding.
- **Plan performance gate 4** ("Cached environment columns are not recomputed
  during adaptive growth"): **honoured** on the tree path — verified in D-6 and
  pinned by `adaptive_factorization_requests_only_the_columns_it_needs`.
- **Plan performance gate 2** ("No fused dimension-`d²` physical probe in MPO-MPO
  production code"): **honoured** — see the fused-probe audit above.
- **Plan performance gate 3** ("Fixed-rank hot path consists of tensor
  contractions, QR, projection, and an optional final SVD on the compressed
  output"): **honoured** — see the SVD audit above.

#### Cross-workstream dependencies flagged for the synthesis pass

1. `factorize_probe_columns` (src_probe.rs:709) calls
   `T::factorize_probe_columns_incremental`, whose `FactorizeResult::rank`
   tolerance determines whether D-10's early-stop rule is safe. Owned by WS-core
   (`tensor_like.rs`, `idx_tensor.rs`) / WS-backend (`incremental_qr.rs`).
2. `factorize_probe_columns` calls `src_error_estimate()`; I verified its formula
   against paper `eq:err-est`/`eq:norm_est` (backend.rs:490-540) because the
   stopping rule's provenance depends on it, but WS-backend owns the verdict on
   that file — including the `src_inverse_adjoint` triangular solve and whether
   it duplicates existing backend functionality.
3. `contract_retaining_indices` and `stack_along_new_index` are branch-new
   `tensor4all-core` APIs that both audited files depend on. WS-core owns whether
   their addition was justified; this workstream confirms they are genuinely used
   in production (not speculative) and that their batched semantics are correct
   (D-7).
4. F-4 and F-5 are performance findings that should be merged with WS-chain's
   batching analysis in the synthesis pass's "does this explain the downstream
   slowness" section.


---

## WS-backend: tensor4all-tensorbackend additions

*Source: [`docs/plans/audit-workstreams/ws-backend.md`](audit-workstreams/ws-backend.md), reproduced in full.*

### WS-backend — numerics and the incremental-QR question

Workstream of the [SRC Provenance Audit](2026-08-28-src-provenance-audit.md).
Report only; no code changes. Branch `feature/treetn-src`, diff base
`origin/main`.

**Files audited (Tier-2 artifacts under audit, read-only):**

| File | Added | Scope of this workstream |
| --- | ---: | --- |
| `crates/tensor4all-tensorbackend/src/backend.rs` | +191 | added hunk only (import line 7; body lines 354–541) |
| `crates/tensor4all-tensorbackend/src/incremental_qr.rs` | +1005 | whole file (new) |
| `crates/tensor4all-tensorbackend/src/lib.rs` | +12/−4 | added hunks only (lines 25–27, 49–55, 62–63) |

`crates/tensor4all-tensorbackend/src/backend/tests/mod.rs` (+69) is WS-tests'
file; it is referenced here only as corroborating evidence for the estimator,
not tabled.

---

#### Executive summary

1. **The audit's motivating suspicion — an unjustified SVD interface — is
   NEGATIVE in this workstream.** There is no `svd` call, no eigendecomposition,
   and no hand-rolled decomposition-by-diagonalization anywhere in the three
   added hunks. The only pre-existing `svd_backend` in `backend.rs` is outside
   the diff and untouched. Verdict `SOURCED-COMMENT(#563, 2026-07-29)` —
   compliant.
2. **The real finding is the opposite shape of the same problem: on the
   *adaptive* SRC path, the QR that Hiroshi's 2026-07-29 comment says must be a
   BLAS3 backend kernel has been replaced by a hand-written scalar Householder
   implementation.** `incremental_qr.rs` re-implements LAPACK
   `geqrf`/`larfg`/`larf`/`ormqr`/`orgqr` in element-indexed safe Rust.
   `HANDROLLED-DUPLICATE`, **scoped to the adaptive (`rtol.is_some()`) path**
   — see item 3 for the scope, which is narrower than an earlier revision of
   this document claimed.
3. **Scope correction: `IncrementalQr` is *not* on every SRC QR path.** The
   **fixed-rank** path (`rtol.is_none()`) bypasses it entirely and already
   routes through the real backend QR:

   ```
   src_chain.rs:92  → contract_fixed (:255) → factorize_fixed_batch (:443)
                    → factorize_full_rank(.., FactorizeAlg::QR, ..) (:453)
   src_tree.rs:171  → same pattern (:184)
        both        → factorize.rs:448 / :475 → factorize_qr_full_rank (:581)
                    → factorize_qr_with_options (:595) → qr_with (qr.rs:248)
                    → matrix_inner.qr() via tenferro_linalg::EagerTensorLinalgExt
                      (qr.rs:259, import at qr.rs:11)
   ```

   `IncrementalQr` is reached **only** from the adaptive path
   (`rtol.is_some()`), through `factorize_probe_columns` (`src_probe.rs:682`),
   which is entered from `src_tree.rs:196`'s else-branch or from
   `src_chain.rs:720` (`factorize_site_adaptive`). The `rtol.is_none()` branch
   *inside* `src_probe.rs:717` is a dead defensive guard on a function the
   fixed-rank path never calls — it is not evidence of a fixed-rank route into
   `IncrementalQr`, and an earlier revision of this document wrongly read it as
   one.

   Narrowing the scope does not weaken F1; it **strengthens** it as evidence of
   duplication. The branch already has, and already uses, a working
   backend/tenferro QR path for fixed-rank SRC. The adaptive path
   re-implements that same factorization by hand in scalar Rust instead of
   reusing the path sitting next to it. "There was no backend QR available" is
   therefore not an available defence for the adaptive path.
4. **The timing evidence does *not* support "IncrementalQr explains the
   downstream slowness."** The only measurements on the branch
   (`docs/worklogs/2026-08-26-treetn-src-contraction.md:90–99`) show fixed SRC
   at `0.0030×–0.0053×` of fit speed and adaptive SRC at `0.018×–0.025×` —
   i.e. **fixed SRC is roughly 4×–6× slower than adaptive SRC**, and fixed SRC
   is the configuration that never touches `IncrementalQr` at all. The slowest
   configuration measured is the one *not* using the hand-rolled QR. Full
   treatment in [D6](#d6--what-the-available-timing-data-does-and-does-not-say).
5. **Open question handed to Task 7 (blocking for relevance):** *which mode —
   fixed (`rtol.is_none()`) or adaptive (`rtol.is_some()`) — does `gw-rs`
   actually invoke SRC in?* If `gw-rs` runs SRC fixed-rank, F1 is a real
   engineering and scope finding but is **irrelevant** to the reported
   slowness. If it runs adaptive, F1 becomes a live candidate cause. This
   workstream cannot answer it — `gw-rs` is outside its file list.
6. **`incremental_qr.rs` exists against an explicit deferral and with no
   recorded profile.** The 2026-08-26 plan, line 247: *"Incremental Householder
   QR is a later optimization gate because neither tensor4all nor the pinned
   tenferro API currently exposes it."* Its performance gate #7 (line 655)
   requires *"a recorded profile."* The 2026-08-26 worklog states the post-fix
   benchmark *"did not enter measurement, so it produced no new timing data."*
   No profile exists anywhere in the branch. `SCOPE-DEVIATION`.
7. **The mathematics is correct.** Both Appendix C estimator formulas, the
   Appendix C.3 block inverse-adjoint update, and the complex Householder
   construction were re-derived from scratch in
   [Detailed derivations](#detailed-derivations-and-flagged-findings) and all
   hold exactly. No confirmed-wrong math.
8. **`incremental_qr.rs` is not a translation of the reference Python/C++.**
   It stores the actual `R` plus a separate `G = R^{-†}`, where the reference
   overwrites its buffer with `trtri(R)` in place. Different representation,
   different storage, hand-written arithmetic where the reference calls LAPACK.
   `LICENSE-RISK` is **low / not raised**.
9. Three in-code doc claims do not hold up: one directly contradicts another
   comment in the same file (F6), one overstates what the backend split avoids
   (D3's F-note), and `new`'s `# Errors` section attributes a failure to "the
   backend QR factorization" in a function that calls no backend QR (F12).
   Details in
   [AI-hallucination signature sweep](#ai-hallucination-signature-sweep).
   **Withdrawn in fix round 1:** the earlier claim that the module doc's
   `[AI-Supplied]` label promise is dangling (old F7). It is not — the labels
   exist, in the audit worklog the comment points at.

Findings by severity. Every `Verdict` cell is one of the spec's eleven taxonomy
tokens, with the single exception of the withdrawn F7 row, which is marked
`WITHDRAWN` and carries no verdict by design — Task 7 should drop it rather
than merge it. Everything else merges mechanically.

| # | Severity | Verdict | Item |
| --- | --- | --- | --- |
| F1 | High | `HANDROLLED-DUPLICATE` | Hand-rolled Householder QR core duplicates `qr_backend`/tenferro `Tensor::qr`; scalar, host-only, un-GPU-able. **Scope: the adaptive (`rtol.is_some()`) SRC path only** — the fixed-rank path already routes through tenferro's QR (exec-summary item 3) |
| F2 | High | `SCOPE-DEVIATION` | Whole file built despite plan's "later optimization gate"; gate #7 profile requirement unmet |
| F3 | High | `SCOPE-DEVIATION` | Contradicts current Tier-1 comment #563 2026-07-29 ("hot path is GEMM + QR… should map well onto the existing dense backends and onto GPU execution (relevant to #553)") — on the adaptive path; the fixed-rank path complies |
| F4 | Medium | `MISSING-VS-SOURCE` | No capacity/`_resize` doubling policy; `append` reallocates and copies both factors every call |
| F5 | Medium | `SUSPECT-UNVERIFIED` | Rank-deficiency skip policy and its `32·ε·max(m,k)·max(‖·‖_F,1)` tolerance have no basis in paper, Python, C++, or Hiroshi |
| F6 | Medium | `SUSPECT-UNVERIFIED` | (doc claim) `IncrementalQr` struct doc: "the same state layout used by the reference implementation" — false, and contradicts the module doc 80 lines above; the claimed source basis does not survive checking |
| F7 | — | **WITHDRAWN** | Was: "module doc promises `[AI-Supplied]` labels that appear nowhere in the file." **False — withdrawn in fix round 1.** `grep -rn "AI-Supplied" --include=*.rs` returns **9** hits across five crates, including `incremental_qr.rs:10` itself, and each refers the reader to "the audit" / "the audit worklog" — i.e. `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`, which carries 76 `[AI-Supplied]` occurrences overall and matching `[AI-Supplied]` labels for this file at lines 307–318. The doc comment is accurate. No finding |
| F8 | Low | `SUSPECT-UNVERIFIED` | (performance) `from_factors` performs a full `O(m p²)` refactorization of an already-orthonormal `Q`; no analogue in the reference and no recorded justification |
| F9 | Low | `SUSPECT-UNVERIFIED` | (dead computation) `src_inverse_adjoint` accumulates a Frobenius `norm_sq` that is validated and then discarded; the identical sum is recomputed in `src_error_estimate_from_inverse_adjoint` |
| F10 | Low | `SUSPECT-UNVERIFIED` | (dead code / premature generality) `IncrementalQrScalar` impls for `f32`/`Complex32` are unreachable — `IncrementalQrState` (core) has only `F64`/`C64` variants |
| F11 | Info | `SUSPECT-UNVERIFIED` | (undocumented contract) `IncrementalQr::error_estimate` hard-errors on a rank-deficient state; the sole caller guards it, but the `# Errors` section does not document it |
| F12 | Medium | `SUSPECT-UNVERIFIED` | (doc claim, new in fix round 1) `new`'s `# Errors` (`incremental_qr.rs:202–203`) says it errors when "the backend QR factorization fails" — `new` calls no backend QR; it calls the file's own `householder_factor` (`:232`). Third instance of the confident-comment-describing-unimplemented-behaviour pattern, and the one most directly entangled with F1 |

---

#### Provenance table

Verdicts per the spec's taxonomy. Line numbers are **post-merge** (current
file contents), not diff offsets.

**Verdict-cell convention (tightened in fix round 1).** Every `Verdict` cell
below contains either one of the spec's eleven taxonomy tokens or the single
sentinel `NO-FINDING`, which is *not* a taxonomy verdict and means "this unit
was inspected and carries no provenance finding — trivial plumbing, an
accessor, or a shape guard." Every `NO-FINDING` cell carries a one-line
justification for why no derivation is owed. Task 7 can drop `NO-FINDING` rows
wholesale. `DERIVED-VERIFIED` is now used **only** where a derivation actually
exists, either inline in the cell or in a named `D`-subsection that the cell
links to, as the spec's taxonomy requires ("Must include the derivation").

##### `crates/tensor4all-tensorbackend/src/backend.rs` (added hunk)

| Lines | Unit | Verdict | Basis |
| --- | --- | --- | --- |
| 7 | `use num_complex::{…, ComplexFloat}` import change | `NO-FINDING` | Import line; no algorithmic content, so no derivation is owed. Needed for `T::conj()`/`T::one()` on the generic scalar; `ComplexFloat` is a real `num-complex` trait (verified, not invented) |
| 354–376 | `SrcErrorEstimate` struct, doc, doctest | `SOURCED-PAPER(App. C §"Error estimation" Eq. (err-est); §"Norm estimation" Eq. (norm_est))` | Two-field carrier for the paper's two adaptive quantities. Doc's "Both values use the sketch width as their normalization factor" is accurate: both are `·/χ̄` under the square. Doctest `R=[2]` → `err = norm = 2` verified by hand |
| 378–408 | `src_error_estimate` doc block (provenance + `# Errors` + doctest) | `SOURCED-PAPER` / `SOURCED-PYTHON(incrementalqr.cpp:106–119)` | Cited C++ line range **verified accurate** — `get_error_estimate` is exactly `incrementalqr.cpp:106–119`. The doc's own disclaimer ("not a literal port of the author's inverse-`R` storage") is correct and is why `LICENSE-RISK` is not raised |
| 409–417 | `src_error_estimate` body | `SOURCED-PAPER(App. C Eq. (err-est), Eq. (norm_est))` | Re-derived in D1 below; matches the paper exactly. Two-line delegation |
| 419–424 | `src_inverse_adjoint` doc | `SOURCED-PAPER(App. C.3)` | Claims the block formula lives in App. C.3 — verified (`\subsection{Final optimization: Updating the \QR factorization}`, `report.tex:1288`). Overstated on one point, see F-note in derivation D3 |
| 425–447 | `src_inverse_adjoint` shape guards + validation loop | `NO-FINDING` (guards) + `SUSPECT-UNVERIFIED` (**F9**, the dead `norm_sq`) | Guards are input validation, not algorithm — no derivation is owed: square/non-empty checks plus a nonzero-diagonal pre-check, legitimate because tenferro's `triangular_solve` gives no singularity signal, so `R` must be screened first. Separately, the `norm_sq` accumulated in the same loop is validated and then never used — dead, and unsourced (**F9**) |
| 449–459 | build `R†`, build `I` | `SOURCED-PAPER(App. C, "inverting the small p×p matrix R†")` | Explicit Hermitian adjoint (`.conj()`), correct for complex; the paper prescribes the adjoint, not the transpose |
| 461–478 | `triangular_solve_matrix(&adjoint, &identity, true, true, false, false)` | `SOURCED-PAPER`; **not** `HANDROLLED-DUPLICATE` | Flags decoded against `backend.rs:1098` signature `(a, b, left_side, lower, transpose_a, unit_diagonal)`: solves `R† X = I` with `R†` treated as lower-triangular — correct, since `R` upper ⇒ `R†` lower. Delegated to the tenferro backend; **no hand-rolled inverse**. tenferro exposes no `trtri`, so this is not a duplicate of anything existing |
| 480–486 | `src_error_estimate_from_inverse_adjoint` doc | `NO-FINDING` | Doc comment only; accurate as written (the split does let `IncrementalQr` reuse its incrementally updated `G`), so no derivation is owed. Its sibling doc at 419–424 *is* flagged — see D3's F-note |
| 487–503 | shape/non-empty guards | `NO-FINDING` (guards) | Input validation, not algorithm — no derivation is owed. Recorded only because the `nrows != ncols` guard is the mechanism behind F11's undocumented hard-error |
| 505–512 | `‖R‖_F²` accumulation | `SOURCED-PAPER(Eq. (norm_est))` | Duplicate of the sum already done at 435–437 when reached via `src_error_estimate` (F9) |
| 513–526 | per-column `‖g_i‖⁻²` accumulation | `SOURCED-PAPER(Eq. (err-est))` / `SOURCED-PYTHON(incrementalqr.py:158, incrementalqr.cpp:111–117)` | Re-derived in D1; the column-of-`G` ↔ row-of-`R^{-1}` correspondence with the Python's `axis=1` row norms is exact. The `column_norm_sq == 0.0` branch is structurally unreachable (a successful solve on a nonsingular `R` gives an invertible, hence zero-column-free, `G`) |
| 528–541 | normalize by `p = ncols`, finiteness check, construct result | `SOURCED-PAPER` | `Err̂ = (p⁻¹ Σ‖g_i‖⁻²)^{1/2}`, `Norm̂ = p^{-1/2}‖R‖_F` — both exact |

**Whole-file verdict for the `backend.rs` hunk: `SOURCED-PAPER`, verified.**
This is the cleanest unit in the workstream: the formulas are right, the
citations are right down to the C++ line range, the divergence from the
reference's storage is disclosed rather than hidden, and the one inverse it
forms is the small `p×p` inverse the paper explicitly asks for, delegated to
the backend.

##### `crates/tensor4all-tensorbackend/src/incremental_qr.rs` (new file, 1005 lines)

**Whole-file verdicts: `SCOPE-DEVIATION` (F2, F3) + `HANDROLLED-DUPLICATE`
(F1). `LICENSE-RISK`: not raised.**

| Lines | Unit | Verdict | Basis |
| --- | --- | --- | --- |
| 1–10 | module doc / provenance header | `SOURCED-PYTHON(incrementalqr.py:90–151, incrementalqr.cpp:21–88)` — refs **verified** | `incrementalqr.py` `_setup`/`append` really are lines 90–151; `incrementalqr.cpp` `setup`/`add_cols` really are lines 21–88. Both citations check out exactly. The `[AI-Supplied]` label promise on lines 9–10 was flagged in an earlier revision (old F7) and is **withdrawn in fix round 1**: the labels do exist, in `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md:307–318`, which is the artifact the comment points at. Separate, still-standing Tier-2 hygiene note: that worklog's line citations into this file are stale (it puts `from_factors` at 113–178; it is at 155–190) — see cross-workstream note 4 |
| 12–19 | imports | `NO-FINDING` | Import list; no algorithmic content, so no derivation is owed. All named items verified to exist: `src_error_estimate`, `src_error_estimate_from_inverse_adjoint`, `src_inverse_adjoint`, `mat_mul`, `Matrix`, `MatrixScalar`, `MatrixTriangularSolveScalar`, `BackendLinalgScalar`, `BackendLinalgError`. No invented names |
| 21 | `type HouseholderFactorization<T>` alias | `NO-FINDING` | Single-use 3-tuple type alias, cosmetic; no derivation is owed |
| 23–43 | `IncrementalQrScalar` trait + doc | `NO-FINDING` | Trait declaration — method signatures only, no algorithm, so no derivation is owed. Both methods are load-bearing downstream (`conjugate` for complex `Q^†`, `from_real` to build `β = −phase·‖x‖`; the arithmetic that needs them is derived in D2). Cosmetic note: the supertrait list is redundant, since `MatrixTriangularSolveScalar: BackendLinalgScalar + MatrixScalar` already |
| 45–83 | four scalar impls | `SUSPECT-UNVERIFIED` (**F10**) | `f32`/`Complex32` impls have no reachable caller: `tensor_like.rs:485` `IncrementalQrState` carries only `F64`/`C64`. Generality built for a case that does not exist |
| 85–112 | `IncrementalQr` struct doc + doctest | `SUSPECT-UNVERIFIED` (**F6** — doc claim does not hold) | "This is the same state layout used by the reference implementation's incremental QR path, expressed in safe Rust" is false and **contradicts this file's own line 8** ("actual-R storage … derived or engineering choices"). Reference layout: one `m × size` buffer holding packed reflectors *and* `trtri(R)` in place, one `tau` vector, a growable `size` with doubling. This file: three separate allocations (`reflectors`, `r`, `tau`) plus an optional fourth (`inverse_adjoint`), storing `R` **not** `R^{-1}`, with no capacity concept |
| 113–122 | struct fields | `NO-FINDING` | Field declarations; the encoding decision they carry (`Option<Matrix<T>>` for `G`, "None ⇒ rank-deficient/rectangular") is the [AI-Supplied] rank policy already flagged as **F5** and analyzed in D4 — no separate derivation is owed here |
| 124–154 | `from_factors` doc + doctest | `NO-FINDING` | Doc comment and doctest only; both accurate as written (the doctest's arithmetic is checked as part of D7). No derivation owed for the prose |
| 155–179 | `from_factors` validation | `NO-FINDING` (guards) | Input validation, not algorithm — no derivation is owed. Non-empty / thin / `R.nrows == Q.ncols` / `R.ncols ≥ Q.ncols` are each necessary for the identity derived in D7 to typecheck dimensionally |
| 180–190 | `from_factors` body | `DERIVED-VERIFIED` (derivation in [D7](#d7--from_factors-refactorization-identity-and-its-redundancy-f8)); **F8** | Re-derived in D7: `q = Q_h R_q` ⇒ `q·r_in = Q_h·(R_q r_in)`, so storing `R := R_q r_in` is exact, and `R_q` is provably a signed identity — which is why the refactorization is redundant work (**F8**). **No analogue in the reference** (`IncrementalQR.__init__` only accepts raw data, never `Q,R`) |
| 192–218 | `new` doc + doctest | `NO-FINDING` | Doc comment and doctest only. The `# Errors` prose in this same block is separately flagged as **F12** (see the findings table) — that is a doc-claim finding, not a derivation gap |
| 219–230 | `new` shape guards | `NO-FINDING` (guards) | Non-empty and thin (`m ≥ n`) shape checks; input validation with no algorithmic content, so no derivation is owed |
| 232–240 | `new` body | `SOURCED-PYTHON(incrementalqr.py:90–97 / .cpp:21–44)` **in intent**, `HANDROLLED-DUPLICATE` **in implementation** | The reference's `_setup` is `geqrf` + `trtri`. This calls the file's own `householder_factor` instead of `qr_backend(&input.to_typed_tensor())`, which exists at `backend.rs:921` and is re-exported at `lib.rs:50`. **Scope correction (fix round 1):** an earlier revision claimed `new` is "unconditionally on the fixed-rank path too, so no SRC QR in this branch ever reaches tenferro's QR." That is **false and withdrawn** — the fixed-rank path never calls `factorize_probe_columns` at all and reaches tenferro's `Tensor::qr` through `factorize_qr_full_rank`; see exec-summary item 3 for the full call graph. `new` is reached only from the adaptive path |
| 242–274 | `append` doc + doctest | `NO-FINDING` | Doc comment and doctest only; accurate as written, and the block update it describes is derived in D3. No derivation owed for the prose |
| 275–301 | `append` shape/overflow guards | `NO-FINDING` (guards), with defensive-code note | Input validation, not algorithm — no derivation is owed. Noted only for the hallucination sweep: the `checked_add` overflow guard is on a quantity already bounded by `self.reflectors.nrows()` (an allocated dimension), hence structurally unreachable |
| 303–304 | `apply_q_adjoint` on the new block | `SOURCED-PYTHON(incrementalqr.py:135–138 `ormqr('L','C',…)`, .cpp:56–66)` / `SOURCED-PAPER(App. C.3, "orthogonalize the new columns against Q")` | Step 1 of the paper's block update, faithfully |
| 305–314 | residual tolerance | `SUSPECT-UNVERIFIED` (**F5**) | `32·ε·max(m,k)·max(‖Y'‖_F, 1)` has no basis in the paper, the Python, the C++, or any Hiroshi comment. It is a **block-wide** threshold applied per column, and the `max(·,1.0)` clamp makes it absolute rather than relative for small-norm blocks. Neither reference implementation does any rank detection — LAPACK `geqrf` simply yields a tiny diagonal |
| 315–320 | reallocate + copy old reflectors | `MISSING-VS-SOURCE` (**F4**) | The reference pre-allocates `size = 2n` and doubles via `_resize` (`incrementalqr.py:99–112`); the C++ takes a caller-sized buffer. Here every `append` allocates a fresh `m × (p+k)` matrix and copies the old one. With the paper's adaptive schedule (`Δχ = 3` up to `χ̄`) that is `~χ̄/3` reallocations of `O(mχ̄)` each ⇒ `O(mχ̄²/3)` pure copy traffic |
| 321–351 | per-column Householder loop with rank skipping | `SOURCED-PAPER(App. C.3)` for the factorization; `SUSPECT-UNVERIFIED` for the skip (**F5**) | Verified correct in D4: for a skipped column the un-annihilated tail is retained in `R` with magnitude ≤ tol, so reconstruction stays within tolerance |
| 353–367 | rebuild block-triangular `R` | `SOURCED-PAPER(App. C.3 `[[R, R'],[0, R'']]`)` / `SOURCED-PYTHON(.py:142–149, .cpp:78–87)` | Structurally the same `2×2` block layout. Note the sign difference is only because the reference stores the *inverse*: `.cpp:86–87` computes `inv(R)_12 = -inv(R_11) R_12 inv(R_22)` while this code stores `R_12` directly |
| 369–387 | inverse-adjoint update branch | `SOURCED-PAPER(App. C.3 `G'` block formula)` | Correct guard structure: uses the `O(k³ + k p²)` block update only when the state is and stays square full-rank, else falls back to a fresh solve, else `None` |
| 389–399 | truncate reflector storage to `rank`, commit state | `MISSING-VS-SOURCE` (contributes to **F4**) | Third full reallocation-and-copy in one `append`, where the reference resizes with a doubling policy |
| 402–419 | `q()` + doc/doctest | `DERIVED-VERIFIED` (delegates to `form_q`; ordering derived in [D5](#d5--reflector-application-order)); `HANDROLLED-DUPLICATE` of LAPACK `orgqr` (`.cpp:90–104`) | Thin wrapper over `form_q`, whose reflector ordering `Q[:, :p] = H_1⋯H_p·[I_p;0]` is derived in D5. `O(m p²)` recomputation on every call; the reference's `get_q` runs `orgqr` once and closes the state |
| 421–440 | `rank()` + doc/doctest | `NO-FINDING` | Trivial accessor returning a stored `usize`; no algorithmic content, so no derivation is owed |
| 442–474 | `q_columns` doc + doctest | `NO-FINDING` | Doc comment and doctest only. Doctest re-derived by hand: `Q` col 1 of `[e₁ e₂]` comes out as `−e₂`, and the assertion uses `.abs()`, so it is sign-robust — evidence it was executed, not fabricated. The doc's "optimization" framing is qualified in the body row below |
| 475–501 | `q_columns` body | `DERIVED-VERIFIED` (derivation in [D5](#d5--reflector-application-order)); no reference analogue | Correct: seeding `e_{start..start+count}` and applying the full reflector product in reverse yields exactly those columns of `Q`. The saving is in the `count` dimension only — the loop still applies **all** `tau.len()` reflectors, of which the ones with index `> start+count` are provable no-ops. So it is `O(m·p·count)`, not the `O(m·(p−start)·count)` the "optimization" framing implies |
| 503–520 | `r()` + doc/doctest | `NO-FINDING` | Trivial accessor returning a borrow of the stored `R`; no derivation is owed. Recorded because it marks the divergence from the reference, which never exposes `R` at all — it overwrites the buffer with `trtri(R)` |
| 522–546 | `error_estimate` doc + doctest | `SUSPECT-UNVERIFIED` (**F11**) | `# Errors` says "singular or contains invalid values"; it omits the rank-deficient/rectangular case, which is the one the `Option` field was introduced for |
| 547–555 | `error_estimate` body | `SOURCED-PAPER(App. C Eq. (err-est), (norm_est))` | Correct dispatch: updated `G` when available, full solve otherwise. **Contract check:** the sole caller, `src_probe.rs:716–730`, computes `saturated = factorized.rank < width` and short-circuits before calling, so the hard-error path is unreachable in production. Safe as used, fragile as designed |
| 558–563 | `try_inverse_adjoint` | `NO-FINDING`, with a note | Three-line delegation to `src_inverse_adjoint` (whose math is derived in D1) plus an `.ok()`; no algorithmic content of its own, so no derivation is owed. Note: `.ok()` discards the backend error, so a genuine tenferro solve failure is indistinguishable from expected rank deficiency |
| 565–614 | `update_inverse_adjoint` | `SOURCED-PAPER(App. C.3, `G' = [[G,0],[-(R'')^{-†}(R')^† G, (R'')^{-†}]]`)` | **Re-derived symbol by symbol in D3 — exact match, including the zero top-right block and the sign on the bottom-left.** The `debug_assert_eq!(old_rank, old_column_count)` restates the caller's own branch condition |
| 616–662 | `householder_factor` | `HANDROLLED-DUPLICATE` (**F1**) of `qr_backend` (`backend.rs:921`) / tenferro `TensorLinalgExt::qr`; math `DERIVED-VERIFIED` (derivation in [D8](#d8--the-packed-householder-factorization-loop-householder_factor--apply_reflector)) | Packed-reflector layout (unit implicit on the diagonal, `v` tail below, `R` above) is LAPACK's `geqrf` convention, re-implemented. Note the asymmetry with `append`: `householder_factor` has **no** rank-deficiency policy at all, so `new` and `append` disagree on what a rank-deficient input means |
| 664–711 | `householder_vector` | `DERIVED-VERIFIED` (full derivation in [D2](#d2--complex-householder-reflector-re-derived)); `HANDROLLED-DUPLICATE` of LAPACK `larfg` | Complex-correct, uses the numerically stable sign (`β = −phase·‖x‖`, so `δ = α − β` never cancels), and `τ` comes out real in `[1,2]` as it must. Handles `α = 0` and `‖x‖ = 0` |
| 713–736 | `apply_reflector` | `DERIVED-VERIFIED` (derivation in [D8](#d8--the-packed-householder-factorization-loop-householder_factor--apply_reflector)); `HANDROLLED-DUPLICATE` of LAPACK `larf` | `H = I − τ v v^†` applied as rank-1 update. Scalar loop with bounds-checked `Matrix::index` per element (`matrix.rs:1011–1024`) — Level-2 BLAS work at interpreted-loop density |
| 738–748 | `apply_q_adjoint` | `DERIVED-VERIFIED` (derivation in [D5](#d5--reflector-application-order)); `HANDROLLED-DUPLICATE` of LAPACK `ormqr` (`.cpp:61–66`) | Order verified in D5: each `H_i` is Hermitian (τ real), so `Q^† = H_p⋯H_1`, matching the ascending loop. Allocates a fresh `Vec` per reflector inside the loop |
| 750–766 | `form_q` | `DERIVED-VERIFIED` (derivation in [D5](#d5--reflector-application-order)); `HANDROLLED-DUPLICATE` of LAPACK `orgqr` (`.cpp:90–104`) | Reverse order verified in D5: `Q[:, :p] = H_1⋯H_p·[I_p; 0]`. Same per-reflector `Vec` allocation |
| 768–1005 | `mod tests` (6 tests) | `NO-FINDING` here — verdict deferred to WS-tests by the spec's file assignment | No provenance verdict is owed by *this* workstream. Recorded for coverage completeness. They are substantive (reconstruction, orthonormality, `R^†G = I` for real and complex, dependent-column rank preservation, shape rejection, incremental-vs-full estimate agreement), and the complex test exercises the Hermitian path specifically. Gap noted for WS-tests: no test of `IncrementalQr` output against `qr_backend`, i.e. nothing pins the hand-rolled QR to the backend QR it replaces |

##### `crates/tensor4all-tensorbackend/src/lib.rs` (added hunks)

| Lines | Unit | Verdict | Basis |
| --- | --- | --- | --- |
| 25–27 | `#[cfg(feature = "global-defaults")] mod incremental_qr;` + doc comment | `SCOPE-DEVIATION` (inherited from the module; see F2) | One `mod` line, so no derivation of its own is owed; it inherits the module's verdict because it is what makes the deferred module exist in the build. Feature gate matches `mod backend;` and `mod matrix;`; correct, since `IncrementalQrScalar` depends on `MatrixTriangularSolveScalar` |
| 49–55 | `pub use backend::{…}` list edited to add `src_error_estimate`, `SrcErrorEstimate` | `NO-FINDING` | Re-export list edit; no algorithmic content, so no derivation is owed. Re-export only; remaining churn on these lines is `rustfmt` re-wrapping of an unchanged list |
| 62–63 | `pub use incremental_qr::{IncrementalQr, IncrementalQrScalar};` + doc | `SCOPE-DEVIATION` (inherited; see F1, F2) | A re-export line, so no derivation is owed, but it is not neutral plumbing. Makes a scalar-loop QR part of the crate's **public** API surface. If F1 is fixed by routing through `qr_backend`, this is a breaking-change surface, and `IncrementalQrScalar` (F10) exposes two dead scalar impls |

---

#### Detailed derivations and flagged findings

Paper references are to `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`.
Appendix C is `\section{Implementing SRC with a tolerance}` (`\label{sec:approx}`,
line 1146) — the third appendix, after `app:khatri-rao-exact` and
`app:src-exact-proof`.

##### D1 — Appendix C error and norm estimators, re-derived

Paper, Eq. `\label{eq:err-est}` (report.tex:1238):

```
G^(j) = (R^(j))^{-†} = [ g_1 … g_χ̄ ],    Err̂^(j) = ( (1/χ̄) Σ_{i=1..χ̄} ‖g_i‖^{-2} )^{1/2}
```

Paper, Eq. `\label{eq:norm_est}` (report.tex:1252):

```
Norm̂ = (1/√χ̄) ‖Y^(j)‖_F = (1/√χ̄) ‖R^(j)‖_F
```

Code path, independently:

- `src_inverse_adjoint` (`backend.rs:449–478`) builds `adjoint[r,c] = conj(R[c,r])`,
  i.e. `R^†`, then solves `R† X = I` with `left_side=true, lower=true`
  (correct: `R` upper ⇒ `R†` lower). So `X = (R†)^{-1} = R^{-†} = G`. ✔ matches
  the paper's `G` definition, adjoint and not transpose.
- `src_error_estimate_from_inverse_adjoint` (`backend.rs:513–526`) sums
  `column_norm_sq = Σ_row |G[row,col]|²` = `‖g_col‖²`, accumulating
  `Σ_col 1/‖g_col‖²`. ✔
- Line 529–530: `error_sq = that / ncols`, `ncols = p = χ̄`; line 538
  `error = sqrt(error_sq)`. ✔ **exactly Eq. (err-est).**
- Lines 505–512 sum `|R_ij|²` over all entries = `‖R‖_F²`; line 530
  divides by `χ̄`; line 539 takes the root ⇒ `‖R‖_F/√χ̄`. ✔ **exactly
  Eq. (norm_est).**

Cross-check against the reference, which stores `R^{-1}` in place rather than
`R`. `incrementalqr.py:158`:

```python
np.sqrt(sum(np.linalg.norm(np.triu(self.data[:n,:n]), axis=1) ** -2) / n)
```

`self.data[:n,:n]` is `trtri(R) = R^{-1}` (set at `.py:97` / `.cpp:43`), and
`axis=1` takes **row** norms. Since `G = R^{-†} = (R^{-1})^†`, column `i` of
`G` is the conjugate of row `i` of `R^{-1}`, so `‖g_i‖ = ‖(R^{-1})[i,:]‖`. The
two expressions are identical. `incrementalqr.cpp:111–117` computes the same
row norms over `col ∈ [row, n)` only, exploiting upper-triangularity. ✔

**Verdict D1: `SOURCED-PAPER(App. C, Eq. (err-est) and Eq. (norm_est))`,
corroborated by `SOURCED-PYTHON(incrementalqr.py:158, incrementalqr.cpp:106–119)`.
Correct. No `LICENSE-RISK`: the representations genuinely differ (actual `R` +
backend solve vs. in-place `trtri`), and the code says so.**

Independent numeric spot-check against `backend/tests/mod.rs:7–18`, done by
hand rather than by reading the test: `R = [[2,1],[0,3]]` ⇒
`R^{-1} = [[1/2, −1/6],[0, 1/3]]` ⇒ `G = [[1/2, 0],[−1/6, 1/3]]` ⇒
`‖g_1‖² = 1/4 + 1/36`, `‖g_2‖² = 1/9` ⇒ `Err̂ = √(½(3.6 + 9)) = √6.3`;
`‖R‖_F² = 14` ⇒ `Norm̂ = √7`. The test's oracle matches, and it is a genuine
independent oracle (closed-form, not a re-run of the implementation).

##### D2 — Complex Householder reflector, re-derived

`householder_vector` (`incremental_qr.rs:664–711`) builds, for
`x = data[start.., column]` with `α = x_0`, `‖x‖ = norm`:

```
phase = α/|α|  (1 if α = 0);  β = −phase·norm;  δ = α − β
v = [1, x_2/δ, …, x_m/δ];     τ = (β − α)/β
```

Claim: `H = I − τ v v^†` is unitary and `H x = β e_1`.

*Step 1 — `v^† x`.* `v^† x = α + (1/δ̄)·Σ_{i≥2}|x_i|² = α + (norm² − |α|²)/δ̄`.
With `α = phase·|α|` and `β = −phase·norm`, `δ = phase(|α| + norm)`, so
`δ̄ = phasē(|α| + norm)`. Then

```
(norm² − |α|²)/δ̄ = (norm − |α|)(norm + |α|) / [phasē (|α| + norm)]
                  = (norm − |α|)·phase          [since 1/phasē = phase]
```

Hence `v^† x = phase·|α| + phase·norm − phase·|α| = phase·norm = −β`.

*Step 2 — `H x`.* `H x = x − τ v(v^† x) = x + τβ v`.
Component 0: `α + τβ = α + (β − α) = β`. ✔
Component `i ≥ 2`: `x_i + τβ·x_i/δ = x_i(1 + (β−α)/(α−β)) = 0`. ✔

*Step 3 — unitarity.* `τ = 1 − α/β`, and `α/β = phase|α| / (−phase·norm) =
−|α|/norm`, so **`τ = 1 + |α|/norm` is real and lies in `[1,2]`**.
`|δ|² = (|α| + norm)²`, so `‖v‖² = 1 + (norm² − |α|²)/(norm+|α|)² =
2·norm/(norm + |α|)`. Then

```
τ²‖v‖² = ((norm+|α|)/norm)² · 2norm/(norm+|α|) = 2(norm+|α|)/norm = 2τ = τ + τ̄
```

which is exactly the condition `H^† H = I`. ✔ `τ` real also gives `H^† = H`,
used in D5.

**Verdict D2: `DERIVED-VERIFIED`.** This is the LAPACK `zlarfg` convention
including the stable sign choice. Not a port — the reference delegates all of
this to LAPACK and contains no such arithmetic. But see F1: this *is* LAPACK,
retyped.

##### D3 — Appendix C.3 block inverse-adjoint update, re-derived

Paper (report.tex:1320–1327):

```
G' = [[R, R'],[0, R'']]^{-†} = [[ R^{-†},                      0        ],
                                [ −(R'')^{-†}(R')^† R^{-†},  (R'')^{-†} ]]
```

`update_inverse_adjoint` (`incremental_qr.rs:565–614`), with `p = old_rank`,
`k = appended_column_count`, `transformed = Q_old^† Y'` post-reflectors:

| Code | Symbol |
| --- | --- |
| `b_adjoint[c,r] = conj(transformed[r,c])`, `r<p`, `c<k` | `(R')^†` (`k×p`) |
| `c[r,c] = transformed[p+r, c]`, `r,c<k` | `R''` (`k×k`, upper-triangular by construction) |
| `c_inverse_adjoint = src_inverse_adjoint(&c)` | `(R'')^{-†}` |
| `coupling = b_adjoint · previous` | `(R')^† G` |
| `lower = c_inverse_adjoint · coupling` | `(R'')^{-†}(R')^† G` |
| `updated[0..p, 0..p] = previous` | top-left `= G` ✔ |
| `updated[p.., p..] = c_inverse_adjoint` | bottom-right `= (R'')^{-†}` ✔ |
| `updated[p+c, r] = −lower[c, r]` | bottom-left `= −(R'')^{-†}(R')^† G` ✔ |
| top-right left at `zeros` | `0` ✔ |

**Verdict D3: `SOURCED-PAPER(App. C.3)`, exact.** Every block, including the
sign and the zero block, matches.

Two supporting checks:

- *`R''` really is upper-triangular.* The guard at `incremental_qr.rs:369–371`
  admits this branch only when `rank == new_column_count` **and**
  `old_rank == old_column_count`, i.e. all `k` appended columns were accepted.
  In that case the `col`-th reflector sits at row `old_rank + col` and
  `apply_reflector` touches columns `col..k` only, so
  `transformed[old_rank + r, c] = 0` for `r > c`. ✔
- *F-note on the doc.* `src_inverse_adjoint`'s doc (`backend.rs:419–424`) says
  the split lets incremental QR "update it with the block formula … **instead
  of solving the same triangular system after every appended sketch block**."
  `update_inverse_adjoint` does still call `src_inverse_adjoint` — on the `k×k`
  block `C = R''`, at `incremental_qr.rs:592`. The claim is true only if "the
  same triangular system" is read as "the same `p×p` system." The paper itself
  is explicit that `(R'')^†` must be computed, so the *code* is right; the
  *comment* overstates. Low severity, recorded for completeness.

##### D4 — the rank-skip policy is sound but sourceless (F5)

`append`'s loop (`incremental_qr.rs:323–351`) skips a column whose residual
below the current rank falls under `residual_tolerance`. Correctness check: for
a skipped column `col`, rows `rank..m` of `transformed[:, col]` are never
annihilated, and `R` is filled from rows `0..rank` only
(`incremental_qr.rs:363–367`), so the discarded mass is exactly that residual,
bounded by `residual_tolerance`. Reconstruction error therefore stays within
tolerance. For accepted columns the discarded mass is exactly zero. ✔

But the *policy* has no source. Neither `incrementalqr.py` nor
`incrementalqr.cpp` does any rank detection — LAPACK `geqrf` simply produces a
small diagonal entry, and the reference's `trtri` would then produce a large
`R^{-1}`, which the leave-one-out estimator interprets (correctly, per
Theorem "Leave-one-out error estimation") as a *small* error. In other words,
**the reference lets rank deficiency flow into the estimator as a near-zero
error, which is the paper's own convergence signal.** This code instead
diverts it into a rectangular `R` for which the estimator is undefined
(`incremental_qr.rs:385–386` sets `inverse_adjoint = None`;
`error_estimate` then hard-errors at `backend.rs:490–494`).

The consumer compensates: `src_probe.rs:716` sets
`saturated = factorized.rank < width` and stops without calling the estimator.
So the end-to-end behaviour is right. But it is right by a compensating
convention split across two crates and two workstreams, and neither the
`# Errors` docs nor the module doc says so. Also note the tolerance itself is
block-wide (`‖Y'‖_F` of the whole appended block) applied per column, with a
`max(·, 1.0)` clamp that turns it absolute for small-norm blocks.

**Verdict D4: `SUSPECT-UNVERIFIED`** for the policy and its constant;
the arithmetic around it is correct.

##### D5 — reflector application order

`apply_q_adjoint` iterates `reflector in 0..tau.len()` ascending; `form_q` and
`q_columns` iterate `(0..tau.len()).rev()`. With LAPACK's convention
`A = QR`, `Q = H_1 H_2 ⋯ H_p`, and with each `H_i` Hermitian (D2 step 3, `τ`
real):

- `Q^† = H_p^† ⋯ H_1^† = H_p ⋯ H_1`. Applying ascending computes
  `H_p(⋯(H_1 x))` = `Q^† x`. ✔
- `Q·[I_p;0] = H_1(⋯(H_p·[I_p;0]))`. Applying descending gives exactly that. ✔

`q_columns` additionally seeds `e_{start+c}` and still runs all `p` reflectors;
those with index `j > start+c` see an all-zero tail at the time they run
(descending order guarantees it), so they are exact no-ops — correct, but
`O(m·p·count)` where `O(m·(p−start)·count)` was available.

**Verdict D5: `DERIVED-VERIFIED`.**

##### D7 — from_factors refactorization identity and its redundancy (F8)

Added in fix round 1 to discharge the `DERIVED-VERIFIED` cell at
`incremental_qr.rs:180–190`, which previously asserted the result without
showing it.

`from_factors(q, r_in)` takes `q ∈ ℂ^{m×p}` (documented as orthonormal columns)
and `r_in ∈ ℂ^{p×n}` with `n ≥ p`, and stores state
`(reflectors, tau, R)` where `(reflectors, tau, R_q) = householder_factor(q)`
and `R := R_q · r_in`.

*Claim 1 — the state represents the same matrix.* Let `Q_h` be the `m×p`
orthonormal factor implied by `(reflectors, tau)` (D5 gives
`Q_h = H_1⋯H_p·[I_p;0]`). `householder_factor` produces `q = Q_h R_q` exactly
(D8). Therefore

```
q · r_in = (Q_h R_q) r_in = Q_h (R_q r_in) = Q_h · R
```

so the constructed state reproduces the caller's product `q·r_in` with no
approximation. ✔ Dimensionally: `R_q` is `p×p`, `r_in` is `p×n`, so `R` is
`p×n` with `n ≥ p`, matching the rectangular-`R` convention the struct uses
elsewhere. ✔

*Claim 2 — `R_q` is a signed identity, so the work is redundant.* If `q` has
orthonormal columns then `q^† q = I_p`. From `q = Q_h R_q` with `Q_h^† Q_h = I_p`,

```
I_p = q^† q = R_q^† Q_h^† Q_h R_q = R_q^† R_q
```

so `R_q` is a `p×p` **unitary** matrix. It is also upper-triangular by
construction. A unitary upper-triangular matrix is diagonal (comparing column
norms bottom-up: `‖R_q e_1‖ = |R_q[0,0]| = 1` forces the rest of column 1 —
already zero by triangularity — and inductively each column has a single unit
entry), and each diagonal entry has modulus 1. In the real case that is exactly
`diag(±1)`; in the complex case a unit-modulus phase, which is the sign
convention D2 fixes as `β = −phase·‖x‖`.

So `R := R_q r_in` is, up to a per-row unit phase, just `r_in` — and the
`O(m p²)` Householder factorization that produced `R_q` recovered information
the caller already had. That is **F8**: the one operation an incremental QR
exists to avoid, performed on input already known to be orthonormal. A
sign/phase-fixing pass costs `O(mp)`, or the reflector state could be seeded
directly. The reference has no analogue at all — `IncrementalQR.__init__`
(`incrementalqr.py:79–97`) accepts raw data only, never a `(Q,R)` pair — so
there is no source to check this against; it is a branch-local addition.

**Verdict D7: `DERIVED-VERIFIED` for the identity (exact, no approximation);
`SUSPECT-UNVERIFIED` (F8) for the cost, which the derivation shows is
avoidable and which no comment or worklog justifies.**

##### D8 — the packed Householder factorization loop (householder_factor + apply_reflector)

Added in fix round 1 to discharge the `DERIVED-VERIFIED` cells at
`incremental_qr.rs:616–662` (`householder_factor`) and `713–736`
(`apply_reflector`). D2 derives one reflector; this derives the loop that
composes them and the storage layout it writes into.

*`apply_reflector` (`:713–736`).* For a reflector `(v, τ)` anchored at row
`start`, the code computes, for each column `c` of the working matrix,
`w_c = Σ_{i≥0} conj(v_i)·A[start+i, c]` and then
`A[start+i, c] -= τ · v_i · w_c`. In matrix form on the trailing block
`A_s := A[start.., :]` that is

```
A_s ← A_s − τ v (v^† A_s) = (I − τ v v^†) A_s = H A_s
```

which is exactly the rank-1 form D2 proves unitary. Rows above `start` are
untouched, which is correct: `H` acts as the identity there. ✔ This is LAPACK
`larf` with no blocking — `H A_s` is computed as one `gemv` + one `ger` worth of
arithmetic per reflector, written as scalar loops with a bounds-checked
`Matrix::index` per element (F1's density argument).

*The loop (`:616–662`).* For `col = 0..p` the code builds `(v_col, τ_col)` from
the current `data[col.., col]` via `householder_vector`, then applies it to
columns `col..n`. By D2, `H_col·data[col.., col] = β_col e_1`, i.e. the
sub-diagonal of column `col` is annihilated and stays annihilated: subsequent
reflectors are anchored at rows `> col` and, by the `A_s` restriction above,
touch only rows `col+1..` of columns `col+1..`. Hence after the loop
`data[0..p, 0..n]` is upper-triangular and

```
H_p ⋯ H_1 · A = R    ⇒    A = H_1 ⋯ H_p · R = Q_h R
```

using `H_i^† = H_i` (τ real, D2 step 3). That is the `A = QR` claim, and the
`Q_h = H_1⋯H_p·[I_p;0]` ordering it implies is exactly what D5 verifies
`form_q` and `q_columns` compute. ✔

*Storage layout.* `v_col[0] = 1` implicitly (never stored; `householder_vector`
normalizes to a leading 1, D2), the tail `v_col[1..]` is written into
`reflectors[col+1.., col]` — the space the annihilated sub-diagonal just
vacated — and `R` occupies the upper triangle. `tau[col]` is stored separately.
This is precisely LAPACK's `geqrf` packed convention, reproduced. The
correctness of the packing follows from the annihilation argument above: the
sub-diagonal entries it overwrites are provably zero at that point, so nothing
of `R` is lost. ✔

*Asymmetry worth recording.* `householder_factor` applies **no**
rank-deficiency test, while `append`'s loop (D4, F5) skips columns under a
`32·ε·…` threshold. So `new` and `append` disagree about what a rank-deficient
input means: `new` accepts it silently and produces a tiny diagonal entry (the
LAPACK behaviour, and the reference's), `append` diverts it into a rectangular
`R` with `inverse_adjoint = None`. Both are individually defensible; the
inconsistency between them is not documented anywhere in the file.

**Verdict D8: `DERIVED-VERIFIED` for the mathematics and the packing;
`HANDROLLED-DUPLICATE` (F1) for the fact that this is `geqrf` + `larf`
retyped in scalar Rust.**

##### D6 — what the available timing data does and does not say

Cross-referenced against the 2026-08-26 plan's "Performance acceptance gates"
(treated as a map, per Tier-2 rules) and against Tier-1 comment #563
2026-07-29.

Tier-1, #563, shinaoka, 2026-07-29 (status: current, not superseded):

> The hot path is GEMM + QR, i.e. pure BLAS3-friendly kernels, which should map
> well onto the existing dense backends and onto GPU execution (relevant to
> #553).

What the branch actually does on that hot path:

1. **The adaptive SRC path's QR goes through `IncrementalQr`.** Call chain:
   `src_tree.rs:196` / `src_chain.rs:720` (adaptive, `rtol.is_some()`) →
   `factorize_probe_columns` (`src_probe.rs:682`) → `src_probe.rs:709` →
   `TensorFactorizationLike::factorize_probe_columns_incremental` →
   `idx_tensor.rs:5813/5833` → `IncrementalQr::from_factors` / `::new`.
   **Scope correction (fix round 1):** this does **not** hold for the
   fixed-rank case. An earlier revision read the `rtol.is_none()` branch at
   `src_probe.rs:717` as proof that fixed-rank SRC also lands here; it is a
   dead defensive guard inside a function the fixed-rank path never calls.
   Fixed-rank SRC routes through `factorize_qr_full_rank` to tenferro's real
   `Tensor::qr` — full call graph in exec-summary item 3.
2. `IncrementalQr` works on host `Matrix<T>` with per-element
   `Index<[usize;2]>` (`matrix.rs:1011–1024`), not on `Tensor`/`TypedTensor`.
   It therefore **cannot dispatch to a GPU backend at all**, which is the
   specific capability the 2026-07-29 comment says SRC should unlock for #553.
3. Density: `apply_reflector` is a scalar rank-1 update; `householder_factor`
   is `O(m p²)` of those; `apply_q_adjoint`, `form_q` and `q_columns` each
   allocate a fresh `Vec` **per reflector, inside the loop** (lines 743–745,
   759–761, 495–497). A blocked LAPACK `dgeqrf` runs near machine peak; an
   unvectorized, bounds-checked scalar loop typically runs one to two orders of
   magnitude below it. Same asymptotics, very different constant.
4. Per-`append` overhead beyond the factorization: three full
   reallocate-and-copy passes (lines 315–320, 353–367, 389–395), no capacity
   doubling (F4).
5. `from_factors` (F8) burns a complete `O(m p²)` refactorization on a matrix
   already known to be orthonormal.

Against the plan's gates: gate 3 ("Fixed-rank hot path consists of tensor
contractions, QR, projection, and an optional final SVD") is met in *shape*
but the QR is not the backend's. Gate 7 ("Any batching or incremental-QR
optimization is justified by a recorded profile and lives in the owning
backend or tensor abstraction") is **half met**: the code does live in the
owning backend crate, but no recorded profile exists. `docs/worklogs/2026-08-26-treetn-src-contraction.md:103–104`
states the post-fix benchmark *"reached dependency compilation but did not
enter measurement, so it produced no new timing data."* The only numbers in
the branch are the pre-fix MPO–MPO ratios at
`docs/worklogs/2026-08-26-treetn-src-contraction.md:90–99` — adaptive SRC at
0.018×–0.025× and fixed SRC at 0.0030×–0.0053× of fit speed, i.e. SRC running
**40×–330× slower than the method it is meant to beat**. Those are disclaimed
as historical, but they are the only measurements that exist.

**What those numbers do *not* support (corrected in fix round 1).** An earlier
revision of this section was titled "why this is a plausible cause of the
downstream slowness" and read the timings as pointing the same direction as F1.
They do not. Lower ratio = slower, so `0.0030×–0.0053×` (fixed) is roughly
**4×–6× slower** than `0.018×–0.025×` (adaptive). Combined with the scope
correction in item 1 — fixed-rank SRC never touches `IncrementalQr` — the only
timing data on the branch says the **slowest measured configuration is the one
that does not use the hand-rolled QR**. That is evidence *against*
`IncrementalQr` being the dominant cost, or at minimum evidence that some other
factor dominates both configurations. Treating these ratios as corroboration
for F1 inverted their meaning.

**Open question, handed to Task 7 (blocking for F1's relevance):** which mode
does `gw-rs` invoke SRC in? If fixed-rank, F1 is a real scope and engineering
finding but is **irrelevant** to the reported slowness, and the cause lies in
WS-tree-probe / WS-core territory. If adaptive, F1 becomes a live candidate and
the timing data above is simply silent about it (no adaptive-vs-adaptive
comparison with and without `IncrementalQr` exists). This workstream cannot
resolve it — `gw-rs` is outside its file list — and no profiling was run in this
pass (explicitly out of scope per the spec).

**What D6 does establish, independent of the timing question:** the branch
replaced a backend/GPU-capable QR with a host scalar loop on the adaptive path
that the Tier-1 comment singles out as the hot path, while an already-working
backend QR path sits next to it serving fixed-rank SRC, and shipped that
without the recorded profile its own plan's gate #7 required. That is F1/F2/F3,
and none of it depends on the ratios above.

##### AI-hallucination signature sweep

Per the additional requirement, checked actively rather than for plausibility.
Result by pattern:

| Pattern | Found? | Instances |
| --- | --- | --- |
| **Confident comment describing behavior the code doesn't implement** | **Yes, 3** (third added in fix round 1) | (c) **F12**, `incremental_qr.rs:202–203`: `new`'s `# Errors` says it returns an error when "the backend QR factorization fails", but `new` calls no backend QR — it calls this file's own `householder_factor` (`:232`). The comment describes the implementation the plan deferred, not the one that shipped, which makes it the instance most directly entangled with F1. (a) **F6**, `incremental_qr.rs:90–91`: "This is the same state layout used by the reference implementation's incremental QR path" — it is not (see the F6 row), and it **contradicts line 8 of the same file**, which correctly calls the actual-`R` storage a derived choice. Internal self-contradiction inside one file is the strongest single signal in this workstream. (b) D3's F-note, `backend.rs:422–423`: "instead of solving the same triangular system after every appended sketch block" — a `k×k` triangular system *is* solved on every append |
| **Dangling reference to a labeling/convention not present** | **No — earlier "yes" withdrawn in fix round 1** | Old **F7** claimed `incremental_qr.rs:9–10` ("the audit labels choices without an external basis `[AI-Supplied]`") was a dangling promise, on the strength of a `grep -rn "AI-Supplied" --include=*.rs` said to return zero hits. Re-run, it returns **9** hits, one of which is line 10 of this very file; and the comment does not promise in-code labels — it points at "the audit", i.e. `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`, which carries `[AI-Supplied]` labels for this file at 307–318. The claim was wrong and is withdrawn. The *separate* observation that that worklog's line citations are stale (it places `from_factors` at 113–178; it is at 155–190, and subsequent ranges are off similarly) survives, as a Tier-2 hygiene note about the worklog — see cross-workstream note 4 |
| **Invented function/API names** | **No** | Every external name used was checked against the actual crate source. `triangular_solve_matrix` (`backend.rs:1098`), `qr_backend` (`backend.rs:921`), `mat_mul` (`matrix.rs:1517`), `matrix_abs_sq` (`matrix.rs:1468`, pre-existing — `matrix.rs` is untouched by this diff), `MatrixTriangularSolveScalar` (`backend.rs:330`), `ComplexFloat` (real `num-complex` trait). tenferro's linalg surface (`tenferro-linalg/src/tensor_ext.rs`) was checked directly: it exposes `svd`, `qr`, `lu`, `triangular_solve` and **no** reflector-level primitives (`geqrf`/`ormqr`/`orgqr`/`trtri` are absent), which is what makes a from-scratch Householder implementation *possible* to justify — though Appendix C.3's block Gram–Schmidt variant needs only `qr` + `mat_mul`, both of which do exist |
| **Fabricated source citations** | **No — the opposite** | All four external line citations were checked and **all four are exactly right**: `incrementalqr.py:90–151` (`_setup` at 90, `append` ending 151), `incrementalqr.cpp:21–88` (`setup` at 21, `add_cols` ending 88), `incrementalqr.cpp:106–119` (`get_error_estimate`), and Appendix C / C.3 for the estimator and the block update. Worth stating plainly: whoever wrote these headers had the reference files open |
| **Unnecessary defensive code for structurally impossible cases** | **Yes, 4** | (a) three `checked_add` overflow guards (`incremental_qr.rs:291–293, 354–356, 480–482`) on quantities already bounded by allocated matrix dimensions; (b) `column_norm_sq == 0.0` (`backend.rs:518`) — a successful triangular solve on a screened-nonsingular `R` yields an invertible `G`, which has no zero column; (c) `debug_assert_eq!(old_rank, old_column_count)` (`incremental_qr.rs:575`) restating the caller's own `if`; (d) doubled finiteness validation of the same `R` (`backend.rs:439–447` then `500–512`) |
| **Premature / unjustified abstraction** | **Yes, 2** | **F10**: `IncrementalQrScalar` implemented for four scalars when only two are reachable (`tensor_like.rs:485–487`). Plus the single-use `HouseholderFactorization<T>` alias and two redundant supertrait bounds (`incremental_qr.rs:36`) |
| **Dead computation** | **Yes, 1** | **F9**: `src_inverse_adjoint` (`backend.rs:435–447`) accumulates `‖R‖_F²`, validates it, discards it; `src_error_estimate_from_inverse_adjoint` (`backend.rs:505–512`) recomputes the identical `O(p²)` sum on the same matrix |
| **Correct-looking notation that is subtly wrong under re-derivation** | **No** | D1, D2, D3, D5 were each re-derived from the paper's LaTeX without consulting any transcription. All four hold exactly, including the complex-adjoint-vs-transpose distinction (which is the classic place this fails), the `−` sign on `G'`'s bottom-left block, the `1/p` vs `1/√p` split between the two estimators, and the reflector ordering |
| **In-code comment claiming to match the paper that doesn't hold up** | **No** | Every paper claim in the two files was checked against `report.tex` and holds: Eq. (err-est) ✔, Eq. (norm_est) ✔, App. C.3 block formula ✔, "Appendix C.3" as the location of the update formula ✔ |

Net read: this file was written **with the sources in hand** — the citations
are accurate to the line, the math is right, and the divergences from the
reference are mostly disclosed. The hallucination-shaped defects are the
smaller kind (a self-contradicting summary sentence, an `# Errors` section
naming a backend QR the function never calls, defensive branches for impossible
states) rather than
invented APIs or fabricated math. **The serious problems here are not
hallucination — they are scope and engineering: a hand-rolled scalar LAPACK
sitting on the hot path, shipped past its own plan's deferral and profiling
gate.**

##### SVD / matrix-inverse / hand-rolled-decomposition sweep

Required explicitly by the brief. Result:

| Query | `backend.rs` added hunk | `incremental_qr.rs` | `lib.rs` added hunk |
| --- | --- | --- | --- |
| SVD call | **none** | **none** | none (the `svd_backend` name in the re-export list is pre-existing and merely re-wrapped by `rustfmt`) |
| Eigendecomposition | none | none | none |
| General dense inverse | none | none | none |
| Triangular inverse | 1 — `src_inverse_adjoint`, delegated to `triangular_solve_matrix` | 2 call sites, both into that same backend helper (`try_inverse_adjoint`, `update_inverse_adjoint`) | n/a |
| Hand-rolled decomposition | none | **yes — `householder_factor` + `householder_vector` + `apply_reflector` + `apply_q_adjoint` + `form_q`** | n/a |

- **SVD verdict: `SOURCED-COMMENT(#563, 2026-07-29)` — compliant.** Hiroshi's
  2026-07-29 comment (Tier-1, current, not superseded) states SRC's core loop
  is QR-only and an SVD is permitted only in the optional final truncation on
  the already-compressed output. These three files contain zero SVD calls, so
  the audit's original motivating suspicion does **not** land here. The
  permitted final-truncation site and any other SVD in the branch belong to
  WS-chain / WS-integration / WS-core (`factorize.rs` is flagged high-priority
  in the spec for exactly this).
- **Matrix-inverse verdict: not `HANDROLLED-DUPLICATE`.** `src_inverse_adjoint`
  forms an explicit `p×p` inverse, but (i) Appendix C prescribes precisely
  that — *"computing this estimate just requires inverting the small `p×p`
  matrix `R†`"* — (ii) it is confined to the sketch-width matrix, and (iii) it
  is delegated to the tenferro-backed `triangular_solve_matrix`, not
  hand-rolled. tenferro exposes no `trtri`-equivalent, so this duplicates
  nothing that already exists.
- **Hand-rolled-decomposition verdict: `HANDROLLED-DUPLICATE` (F1).** Five
  functions re-implement LAPACK primitives. The initial factorization in `new`
  duplicates `qr_backend` (`backend.rs:921`) / tenferro
  `TensorLinalgExt::qr` outright. The reflector-level primitives
  (`larfg`/`larf`/`ormqr`/`orgqr`) have no tenferro equivalent, so *some*
  hand-written code would be needed for a true reflector-reusing incremental
  QR — but Appendix C.3's own block Gram–Schmidt formulation
  (`Z := Y' − Q(Q^†Y')`, then `QR(Z)`) needs only `qr` and `mat_mul`, both of
  which the backend already provides, and would have kept the hot path on
  BLAS3 and GPU-dispatchable. The paper prefers Householder for numerical
  stability, which is a legitimate argument for this design — but it is an
  argument that had to be weighed against the plan's own deferral (F2) and the
  Tier-1 BLAS3/GPU requirement (F3), and no record of that weighing exists in
  the branch.

##### Line-coverage ledger

Confirming every added line maps to a verdict above.

| File | Range | Covered by |
| --- | --- | --- |
| `backend.rs` | 7 | table row "import change" |
| `backend.rs` | 354–376 | `SrcErrorEstimate` row |
| `backend.rs` | 378–417 | `src_error_estimate` doc + body rows |
| `backend.rs` | 419–478 | `src_inverse_adjoint` doc / guards / adjoint+identity / solve rows (4 rows) |
| `backend.rs` | 480–541 | `src_error_estimate_from_inverse_adjoint` doc / guards / norm / column / normalize rows (5 rows) |
| `incremental_qr.rs` | 1–10, 12–19, 21, 23–43, 45–83 | header/imports/alias/trait/impls rows |
| `incremental_qr.rs` | 85–122 | struct doc + fields rows |
| `incremental_qr.rs` | 124–190 | `from_factors` doc / validation / body rows |
| `incremental_qr.rs` | 192–240 | `new` doc / guards / body rows |
| `incremental_qr.rs` | 242–400 | `append`, split across 8 rows (doc, guards, `Q^†` apply, tolerance, realloc, loop, `R` rebuild, `G` branch, commit) |
| `incremental_qr.rs` | 402–440 | `q`, `rank` rows |
| `incremental_qr.rs` | 442–501 | `q_columns` doc + body rows |
| `incremental_qr.rs` | 503–555 | `r`, `error_estimate` doc + body rows |
| `incremental_qr.rs` | 558–614 | `try_inverse_adjoint`, `update_inverse_adjoint` rows |
| `incremental_qr.rs` | 616–766 | `householder_factor`, `householder_vector`, `apply_reflector`, `apply_q_adjoint`, `form_q` rows |
| `incremental_qr.rs` | 768–1005 | `mod tests` row (recorded; verdict deferred to WS-tests by the spec's file assignment) |
| `lib.rs` | 25–27, 49–55, 62–63 | three plumbing rows |

No added line of real logic is unaccounted for. The only rows summarized in
bulk are import lists, the type alias, trivial accessors, and the `lib.rs`
re-export edits.

---

#### Cross-workstream notes (for the Task 7 synthesis)

1. **Overlap with WS-core.** `IncrementalQr`'s only consumer is
   `crates/tensor4all-core/src/defaults/idx_tensor.rs:5755–5900`
   (`incremental_probe_factorize_typed`). F1's performance argument and F8's
   `from_factors` cost are properly *joint* findings with WS-core; resolve
   them into one entry rather than two.
2. **Overlap with WS-tree-probe.** The estimator contract (F11 / D4) is
   completed by `src_probe.rs:716–730`, whose stopping rule
   `estimate.error <= atol + rtol * estimate.norm` is an exact match for the
   paper's Appendix C.2 step 2 (`Err̂ ≤ τ_abs + τ_rel · Norm̂`). That is
   independent corroboration that `backend.rs`'s two outputs are consumed as
   the paper prescribes — worth stating in the synthesis, since it is the one
   place the backend/probe boundary lines up cleanly.
3. **For WS-tests.** No test anywhere pins `IncrementalQr`'s output against
   `qr_backend`'s. Given F1, a differential test against the backend QR is the
   obvious missing check, and its absence is why a scalar re-implementation of
   LAPACK could land unremarked.
4. **Tier-2 hygiene, for whoever audits the worklogs.**
   `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md:303–318`
   is a prior self-audit of this very file whose line citations are stale
   throughout (`incremental_qr.rs:113-178` for `from_factors`, actually
   155–190; `225-363` for `append`, actually 275–400; and so on). Its verdicts
   are broadly consistent with this workstream's, but a provenance table that
   cannot be indexed against the file it audits should not be relied on as a
   map. Flagging as `PLAN-CLAIM-UNVERIFIED`-adjacent, about the *worklog*, not
   the code.


---

## WS-core: tensor4all-core additions

*Source: [`docs/plans/audit-workstreams/ws-core.md`](audit-workstreams/ws-core.md), reproduced in full.*

### WS-core — tensor4all-core additions

Workstream of the [SRC Provenance Audit](2026-08-28-src-provenance-audit.md).
Report only; no code changes. Branch `feature/treetn-src`, diff base
`origin/main`.

**Files audited:**
- `crates/tensor4all-core/src/defaults/idx_tensor.rs` (+533/-0)
- `crates/tensor4all-core/src/tensor_like.rs` (+510/-0)
- `crates/tensor4all-core/src/defaults/factorize.rs` (+46/-34)
- `crates/tensor4all-core/src/index_like.rs` (+28/-0)
- `crates/tensor4all-core/src/index_ops.rs` (+4/-0)
- `crates/tensor4all-core/src/defaults/index.rs` (+4/-0)

Diffed against `origin/main` (`git diff origin/main -- <files>`, captured to
`/tmp/ws-core.diff`: 1308 lines, 1091 insertions / 34 deletions total across
these six files). Every hunk in that diff is accounted for in the provenance
table below; no hunk was skipped.

**Branch history consulted:** `git log --oneline origin/main..HEAD` shows six
commits on the branch (`cd3724b`, `e69b6ea`, `4e56730`, `4167f3c`, `9e018d4`,
`7d574d7`). Of these, only `cd3724b` (initial seams, +37/+4/+28/+4/+35 across
these six files) and `9e018d4` ("align SRC with paper and reference",
+46/+496/+475/+192-tests) touch the WS-core files. `9e018d4`'s commit message
contains only an **end-to-end** benchmark table (fit/zip-up/adaptive-SRC/
fixed-SRC wall-clock and error at N=2,8,32,128) plus a `cargo test`/`clippy`/
`fmt` pass summary — no per-component profile of "random-vector construction /
per-column contraction planning / sketch-column assembly" as the plan's gate
demands (see Finding F1 below).

#### Priority check: `factorize.rs` (the file named for factorization)

**Verdict: no new factorization logic exists in this file.** Read in full
(931 lines) and diffed: every one of its ~20 `factorize_*`/`matrix_*`
functions (SVD, QR, LU, CI/cross-interpolation, Gram) pre-exists unchanged on
`origin/main`. The only change across all 6 hunks (diff lines 5-100) is that
every `Ok(FactorizeResult { left, right, bond_index, singular_values, rank
})` struct-literal construction is replaced by `Ok(FactorizeResult::new(left,
right, bond_index, singular_values, rank))`, because `tensor_like.rs` (below)
turns `FactorizeResult` into a struct with a new private field
(`incremental_qr_state`) that a struct literal can no longer construct from
outside the module. **This directly answers the audit's top-priority
suspicion for this file: unfounded.** The real QR/incremental-QR surface
lives in `tensor4all-tensorbackend::IncrementalQr` (WS-backend's
`incremental_qr.rs`, +1005 new lines — out of WS-core's file list) and in the
`tensor_like.rs`/`idx_tensor.rs` bridge below, not in `factorize.rs`. Row and
full derivation below.

#### Provenance table

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `factorize.rs` | `factorize_gram`, `factorize_svd_with_options` (×2 arms), `factorize_qr_with_options`, `factorize_lu_with_options`, `factorize_ci_with_options`: struct-literal → `FactorizeResult::new(...)` call-site rewrite | 5-100 (diff) | `DERIVED-VERIFIED` | Mechanical signature migration only; no SVD/QR/linear algebra added, changed, or hidden — see "Priority check" above and detailed row below. |
| `index_like.rs` | `IndexLike::new_link(dim) -> anyhow::Result<Self>` trait method + doctest | 219-244 (current) / 703-733 (diff) | `DERIVED-VERIFIED` | Trivial identity/metadata operation, matches SRC's fresh-bond-index need — see below. |
| `defaults/index.rs` | `DynIndex::new_link` impl | 523-527 (current) / 684-698 (diff) | `DERIVED-VERIFIED` | One-line delegation to a pre-existing inherent constructor — see below. |
| `index_like.rs` / `index_ops.rs` | Test-double `new_link` impls in `mod tests` (test fixtures only) | 734-744, 749-759 (diff) | `DERIVED-VERIFIED` | Test-only fixture code, no production behavior — see below. |
| `idx_tensor.rs` | `IdxTensor::try_contract_pairwise_retaining` (batched `dot_general`, retained/batch axes, generic fallback) | 3648-3785 | `DERIVED-VERIFIED` + `PLAN-CLAIM-UNVERIFIED` (F1) | Independently re-derived batching strategy; verified against `tenferro-tensor/src/backend.rs:106-190`. Not `HANDROLLED-DUPLICATE` (reuses pre-existing `dot_general_with_conj`). See below. |
| `idx_tensor.rs` | `impl TensorFactorizationLike for IdxTensor { factorize_probe_columns_incremental, src_error_estimate }`, `IncrementalQrStateScalar` (+f64/Complex64 impls), `incremental_probe_factorize_typed`, `probe_columns_matrix` | 5643-5906 (approx.; diff lines 275-573) | `SOURCED-PYTHON(incrementalqr.py:114, incrementalqr.cpp:46)` + `SOURCED-PAPER(Appendix C.3, report.tex:1288-1332)` + `[AI-Supplied]` bridge plumbing | Append contract and R^{-†} block-update role verified real in `report.tex`; dtype dispatch/state bookkeeping independently traced. **Not** independently numerically tested by WS-core. See below for the narrowed API-name check (I4). |
| `idx_tensor.rs` | `impl TensorContractionLike for IdxTensor { contract_retaining_indices }` (2-tensor fast-path dispatch) | 5978-5996 | `DERIVED-VERIFIED` (trivial) | One-line dispatch, mechanical — see below. |
| `idx_tensor.rs` | `impl TensorConstructionLike for IdxTensor { from_dense_any, stack_along_new_index }` (trait forwarders) | 6041-6055 | `DERIVED-VERIFIED` (trivial) | Pure forwarders to pre-existing native inherent methods (Finding F2) — see below. |
| `idx_tensor.rs` | `impl TensorConstructionLike for IdxTensor { concatenate_along_new_index }` (native override) | 6056-6125 | `DERIVED-VERIFIED` | No prior native implementation existed; delegates to pre-existing `EagerTensor::concatenate`. See below. |
| `tensor_like.rs` | `FactorizeResult::new`, `with_incremental_qr_state`, `incremental_qr_state`, `IncrementalQrState` enum | 478-538 (diff 772-837) | `DERIVED-VERIFIED` (trivial) | New private field + constructor/accessor, no algorithmic content — see below. |
| `tensor_like.rs` | `TensorContractionLike::contract_retaining_indices` trait method + generic default | 839-901 (diff 840-905) | `DERIVED-VERIFIED` | Matches an established repo convention for unsupported-storage defaults — see below. |
| `tensor_like.rs` | `TensorFactorizationLike::factorize_probe_columns_incremental` trait method + generic default | 1001-1111 (diff 906-1013) | `SOURCED-PAPER(§C.2 step 5, report.tex:1265-1287)` | "Recompute from scratch" branch of the paper's adaptive loop — see below. |
| `tensor_like.rs` | `TensorFactorizationLike::src_error_estimate` trait method + generic default | 1113-1122 | `DERIVED-VERIFIED` (trivial) | Delegates real numerics to `tensor4all_tensorbackend::src_error_estimate`; that function's correctness is WS-backend's audit territory, out of WS-core scope — see below. |
| `tensor_like.rs` | `TensorConstructionLike::from_dense_any` trait method + generic default | 1171-1249 (diff 1017-1089) | `DERIVED-VERIFIED` | Definitional dense-from-column-major construction, holds by construction — see below. |
| `tensor_like.rs` | `TensorConstructionLike::stack_along_new_index` trait method + generic default | 1249-1313 (diff 1091-1189) | `DERIVED-VERIFIED` | Definitional outer-product-with-onehot stacking, holds by construction — see below. |
| `tensor_like.rs` | `TensorConstructionLike::concatenate_along_new_index` trait method + generic default | 1313-1409 (diff 1191-1304) | `DERIVED-VERIFIED` | Generic slice-and-restack counterpart of the native override — see below. |

#### Detailed derivations and flagged findings

##### Finding F1 — `PLAN-CLAIM-UNVERIFIED`: the plan's profiling gate was not satisfied

The 2026-08-26 plan (`docs/plans/2026-08-26-treetn-src-contraction-plan.md:341-357`,
itself Tier-2) says: *"The generic implementation can initially represent
sketch columns as separate tensors... Before accepting the performance
result, profile: random-vector construction; per-column contraction
planning; sketch-column assembly; QR; cap projection; final SVD. If column
construction or assembly is material, add a reusable batch/stack constructor
at the tensor abstraction..."* — i.e. the plan's own precondition for these
additions is a **component-level profile**, not an end-to-end benchmark.

Searched `git log --oneline origin/main..HEAD` (6 commits) and every
`docs/worklogs/*.md` and `docs/design/*.md` file touched or referenced by
the branch. Found:
- `9e018d4`'s commit message: an end-to-end wall-clock/error table (fit vs.
  zip-up vs. adaptive/fixed SRC), not a component breakdown.
- `docs/worklogs/2026-08-26-treetn-src-contraction.md:84-97`: a "complete
  benchmark profile" that is also end-to-end (60 records of total contraction
  time/error/rank), explicitly flagged in its own text as historical and "not
  a formal performance gate."
- The same worklog's own "Remaining risks" section (line ~136): *"Probe
  construction is generic and currently uses tensor one-hot/axpby assembly; a
  backend-native random-vector constructor may be worthwhile **after a
  formal performance experiment**."* — the branch's own author-facing
  worklog **admits the formal performance experiment the plan requires has
  not happened**, and that the generic (unoptimized) fallback path is still
  what ships for random-vector construction.

**Verdict: `PLAN-CLAIM-UNVERIFIED`** for the plan's own gating claim — no
recorded profiling evidence exists anywhere on the branch showing that
"column construction or assembly" was actually material before these ~1000
lines were added. This applies in addition to the per-unit verdicts below,
per the task brief's instruction. Note this is a finding about the *plan's*
unmet precondition, not automatically a defect in the *code* — see F2 for
why the actual code impact turns out to be smaller than "1000 speculative
lines" suggests.

##### Finding F2 — the "reusable batch/stack constructor... with an optimized `IdxTensor` implementation" the plan asks for **already existed before this branch**

Checked whether `IdxTensor`'s "native" implementations of `stack_along_new_index`
and `from_dense_any` are new work introduced to satisfy the plan's gate.
They are not:

```
$ git show origin/main:crates/tensor4all-core/src/defaults/idx_tensor.rs | grep -n "fn from_dense_any\|fn stack_along_new_index"
2472:    pub fn stack_along_new_index(
5785:    pub fn from_dense_any(
```

Both `pub fn stack_along_new_index` and `pub fn from_dense_any` are
pre-existing **inherent** methods on `IdxTensor`, present on `origin/main`
already, unrelated to SRC (also used pre-branch by `gse.rs` and
`treetn/cached_evaluator.rs`, neither of which is an SRC file). What this
branch actually adds in `idx_tensor.rs` for these two (diff lines 599-612) is
only a thin trait-impl forwarder — `impl TensorConstructionLike for IdxTensor
{ fn stack_along_new_index(...) { IdxTensor::stack_along_new_index(...) } }`
— wiring the pre-existing, presumably-already-exercised native methods into
a *new generic trait surface* so that SRC code written against `T:
TensorConstructionLike` (not hardcoded to `IdxTensor`) can call them
polymorphically. This substantially changes the risk picture from "~1000
speculative unprofiled lines of new dense-tensor-construction code" to: the
genuinely new code is (a) the trait abstraction itself, (b) its generic
(intentionally slow, correctness-only) default implementations for backends
without a native override, and (c) two units that are genuinely new even at
the `IdxTensor` native level — `concatenate_along_new_index` and
`try_contract_pairwise_retaining`/`contract_retaining_indices` (see below).
Both F1 and F2 stand together: F1 is not retracted (there is still no
profiling evidence, and the genuinely-new units in (c) still lack it), but
F2 shows the addition is smaller and lower-risk than its line count implies.

##### Finding F3 — in-code "Provenance:" comments self-cite a Tier-2, same-process document; treat their claims as unverified until independently re-traced

Several of the new functions carry comments referencing "the audit" as the
basis for an `[AI-Supplied]` label:

- `idx_tensor.rs:3651`: *"...is labelled `[AI-Supplied]` in the SRC audit."*
- `idx_tensor.rs:5696`: *"...are labelled `[AI-Supplied]` in the audit."*
- `tensor_like.rs:846`: *"...are tensor4all-specific `[AI-Supplied]` plumbing"*
  (no audit reference at this specific site).
- `tensor_like.rs:1010`: *"...is labelled `[AI-Supplied]` in the audit."*

**These comments do not name a specific file.** None of them says
`docs/worklogs/2026-08-27-...md` or any other path — they only say "in the
audit" / "in the SRC audit". An earlier draft of this finding stated that
these comments "cite `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`"
as if that were a quoted fact; that overstates what the comments themselves
say.

**This workstream identifies** the referenced audit as
`docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`, **as
an inference from cross-referencing, not a quotation of the comments**: that
worklog's own table carries matching `[AI-Supplied]`/`[Derived]`/`[Repo]`
classifications for the exact same line ranges these comments annotate —
worklog line 325 covers `tensor_like.rs:842-899` (matching the
`contract_retaining_indices` comment at `tensor_like.rs:846`), worklog line
326 covers `tensor_like.rs:1005-1072` (matching the comment at
`tensor_like.rs:1010`), worklog line 330 covers `idx_tensor.rs:3648-3785`
(matching the comment at `idx_tensor.rs:3651`), and worklog line 331 covers
`idx_tensor.rs:5647-5906` (matching the comment at `idx_tensor.rs:5696`). The
line-range and label match is precise enough that the identification is
well-founded, but it remains this workstream's own inference, not something
the code comments themselves assert.

Checked the worklog's own provenance in the commit history, using commit
authorship rather than filesystem mtime (mtime is a checkout artifact, not
authoritative project history):

```
$ git log --diff-filter=A --format="%H %ai %s" -- docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md
9e018d474cd42ba20b79a32c6e3001a5692d3ba 2026-08-27 14:06:05 +0200 feat(treetn): align SRC with paper and reference
```

The worklog was **added in the same commit** (`9e018d4`) as the code whose
comments this workstream infers cite it — not merely written shortly before
it. (An earlier draft of this finding compared the worklog's filesystem
`mtime` — `ls -la` reported `Aug 27 14:05`, one minute before the commit
timestamp — to the commit time and concluded "written one minute before";
that comparison is strictly weaker evidence, since `mtime` reflects the
local checkout state, not authorship history, and can drift on any clone,
rebase, or checkout.) Same commit, same author, no independent review in
between — if anything this is **stronger** evidence for the closed-citation-
loop concern than a one-minute gap would have been: the citing code and the
cited worklog were authored and landed as a single atomic unit, with no
external checkpoint between writing the self-assessment and writing the code
comments that (per this workstream's inference) point back to it.

Per this audit's own epistemics (Tier 2: *"any in-code comment that claims a
derivation are themselves products of the same AI-assisted process under
suspicion... [it is] a map... [n]ever cite them as the thing a piece of code
is checked against"*), these in-code citations are **not** evidence on their
own authority. This is a distinct hallucination-adjacent pattern from
"invented API" or "padding": the code cites its own prior self-assessment
(via an unnamed but identifiable "the audit") as though that constituted
external verification — a closed citation loop. It does not by itself make
the underlying `[Derived]`/`[AI-Supplied]` claims wrong (several were
independently re-verified against Tier 1 below and did hold up), but the
citation style itself should not be trusted, and every claim it makes was
re-checked against Tier 1 directly rather than accepted on the worklog's
authority, per the rows below.

##### `factorize.rs` — call-site migration (diff lines 5-100) — DERIVED-VERIFIED

Verified by direct comparison: each of the ~20 call sites passes the
identical five values in the identical order to `FactorizeResult::new(...)`
that it previously used in the struct literal; no field is dropped,
reordered, or defaulted differently. No SVD/QR/hand-rolled linear algebra was
added, changed, or hidden here.

##### `index_like.rs::IndexLike::new_link` (lines 219-244) — DERIVED-VERIFIED

"Create a fresh undirected link index of a given dimension" has no
interesting math; it's an identity/metadata operation. **Mechanical, matches
SRC's stated need**: confirmed by usage —
`crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:570` calls
`T::Index::new_link(width)` to mint the fresh batch index for the
prefix-cache's combined sketch columns, i.e. exactly the "replace/create
bond-like indices with fresh IDs before/for local contractions" pattern
WS-core was asked to check for. Not unrelated capability.

##### `defaults/index.rs::DynIndex::new_link` (lines 523-527) — DERIVED-VERIFIED

One-line delegation to an already-existing inherent constructor
(`Index::new_link` pre-dates this branch); only the trait-level plumbing is
new.

##### Test-double `new_link` fixtures (`index_like.rs::mod tests`, `index_ops.rs::mod tests`) — DERIVED-VERIFIED

Test-only fixture code required to keep the two hand-written `IndexLike` test
doubles compiling against the now-larger trait; no production behavior.

##### `idx_tensor.rs::try_contract_pairwise_retaining` (lines 3648-3785) — DERIVED-VERIFIED + PLAN-CLAIM-UNVERIFIED (F1)

No paper/Python/Hiroshi source specifies this exact batching strategy
(Hiroshi's comments address chain-level caching, not intra-step column
batching). Re-derived independently: batches N independent probe-column
contractions into one `dot_general_with_conj` call using
`DotGeneralConfig{lhs_batch_dims, rhs_batch_dims}`. Verified against
`tenferro-tensor/src/backend.rs:106-190` (real, pre-existing dependency): a
`dot_general` batch axis is contracted per-batch-index independently and the
result is `[lhs_free, rhs_free, batch]` in that order — exactly what the
function's own comment claims and exactly what its
`current_indices`/`desired_indices` permutation logic assumes. This makes the
batched call mathematically identical to running one `contract()` per
retained-index value and stacking — i.e. correct — while being one GEMM
instead of N. Does **not** duplicate/hand-roll linear algebra: it reuses
`dot_general_with_conj` (pre-existing tenferro primitive) and
`common_ind_positions`/`is_contractable` (pre-existing `tensor4all-core`
helpers), so `HANDROLLED-DUPLICATE` does not apply. Guard clauses (empty
`retained_indices`, structured-payload short-circuit, dtype mismatch,
absent/non-contractable retained index) are all real possible caller states
for a `pub(crate)` helper reachable through the public
`contract_retaining_indices` trait method with caller-supplied
`retained_indices` — not padding for a structurally-impossible case.

##### `idx_tensor.rs` incremental-QR bridge (lines 5643-5906, diff lines 275-573) — SOURCED-PYTHON + SOURCED-PAPER + `[AI-Supplied]` plumbing

`impl TensorFactorizationLike for IdxTensor { factorize_probe_columns_incremental,
src_error_estimate }`, `IncrementalQrStateScalar` (+ f64/Complex64 impls),
`incremental_probe_factorize_typed`, `probe_columns_matrix`.
`SOURCED-PYTHON(incrementalqr.py:114 IncrementalQR.append, incrementalqr.cpp:46
add_cols)` for the resume/append *contract*;
`SOURCED-PAPER(Appendix C.3 "Final optimization: Updating the QR
factorization", report.tex:1288-1332)` for the role of the update (confirmed
real — see verification below); dtype dispatch / private-state-enum plumbing
/ index bookkeeping is tensor4all-specific `[AI-Supplied]`, independently
reviewed here rather than accepted from the self-audit citation (Finding F3).

Verified `report.tex` actually has an Appendix C with exactly the structure
claimed: `\appendix` (line 1100) → §A `khatri-rao-exact` (1102) → §B
`src-exact-proof` (1125) → **§C `Implementing SRC with a tolerance`** (1146,
subsections C.1 "Error estimation" 1152, C.2 "Adaptive determination of bond
dimension" 1265, **C.3 "Final optimization: Updating the QR factorization"**
1288) → §D "Full operation counts" (1333). §C.3's block formula (`G' =
[[G,0],[-(R'')^{-†}(R')^†G, (R'')^{-†}]]` for `G := R^{-†}`) is exactly the
"R^{-†} block formula" the code/worklog claim it implements — this citation
holds up under direct re-check, it is not fabricated. §C.2 step 5 ("Recompute
(or update, see App. C.3) the QR factorization of Y^(j)") matches this
function's "resume-or-recompute" branch structure. Traced the bridge logic
by hand (dtype branch via `is_f64()`/`is_c64()` →
`incremental_probe_factorize_typed::<f64/Complex64>`; `S::resume` recovers
packed Householder state or falls back to reconstructing
`IncrementalQr::from_factors` from the previous dense Q/R; rank-decrease is
rejected as an error; new `Q` columns are fetched only for the appended range
via `state.q_columns(previous_rank, appended_rank)` and concatenated onto the
previous `Q` via the new `concatenate_along_new_index`) — internally
consistent with the append contract and no arithmetic error found on
inspection. **Not independently numerically tested by WS-core** (no `cargo
test` run beyond `cargo check`, which passed) — cross-reference WS-tests for
whether `tensor_like/tests/mod.rs` and `backend/tests/mod.rs` actually
exercise incremental-append correctness against ground truth, not just
shape/API coverage.

**Narrowed API-name check (see Finding I4 / Summary).** The 12 API names used
here split into two groups for invented-name-checking purposes.
`Matrix::{from_col_major_vec,as_col_major_slice}` pre-date this branch on
`origin/main` (confirmed there directly, not just at HEAD) — grepping and
finding these is genuine evidence against invention. `IncrementalQr`,
`IncrementalQrScalar`, `IncrementalQr::{new,from_factors,append,q_columns,r}`,
`SrcErrorEstimate`, and `src_error_estimate` exist **only** in files this
same branch added (`incremental_qr.rs` is a new file; the `src_error_estimate`
free function is a new hunk in `backend.rs`) — grepping HEAD and finding them
only confirms internal self-consistency (the crate compiles against its own
new API), not that the names weren't invented in this same session. This
check cannot rule out invention for the branch-new group.

##### `idx_tensor.rs::contract_retaining_indices` dispatch (lines 5978-5996) — DERIVED-VERIFIED (trivial)

One-line dispatch: 2 tensors → `try_contract_pairwise_retaining`; else →
generic `contract_with_options`. Mechanical.

##### `idx_tensor.rs` construction forwarders (lines 6041-6055) — DERIVED-VERIFIED (trivial)

`impl TensorConstructionLike for IdxTensor { from_dense_any,
stack_along_new_index }`. Pure forwarders to pre-existing native inherent
methods, confirmed pre-dating this branch on `origin/main` (Finding F2). No
new logic.

##### `idx_tensor.rs::concatenate_along_new_index` native override (lines 6056-6125) — DERIVED-VERIFIED

No prior native implementation existed for this one (confirmed: absent from
`origin/main`). Derivation: validates all operand tensors share the same
external-index order away from one designated concatenation axis, sums
source-index dimensions, then delegates to `EagerTensor::concatenate(&inners,
first_axis)` — a real, pre-existing `tenferro-ad` primitive
(`tenferro-ad/src/eager_ops.rs:964`, signature `(tensors: &[&Self], axis:
usize) -> Result<Self>` matches the call site exactly). This is literally
what "concatenate along an axis" means; no invented math. Confirmed **not
over-generalized**: called with more than 2 operands at
`crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:588`
(`T::concatenate_along_new_index(&segment_prefixes, &source_indices, batch)`,
where `segment_prefixes` ranges over an arbitrary number of adaptive-batching
segments), so the N-ary signature is exercised by real SRC control flow, not
padding beyond an actual 2-way append use case.

##### `tensor_like.rs::FactorizeResult` incremental-state plumbing (lines 478-538, diff 772-837) — DERIVED-VERIFIED (trivial)

`FactorizeResult::new`, `with_incremental_qr_state`, `incremental_qr_state`,
`IncrementalQrState` enum. New private field + constructor/accessor to carry
opaque incremental-QR state across the new `FactorizeResult`; exactly what
`factorize.rs`'s refactor and `idx_tensor.rs`'s incremental bridge require.
No algorithmic content.

##### `tensor_like.rs::TensorContractionLike::contract_retaining_indices` default (lines 839-901, diff 840-905) — DERIVED-VERIFIED

Trait method + generic default (falls back to `contract` when
`retained_indices` empty; else returns an "unsupported" error). Confirmed
this "return an error unless a backend overrides" default pattern is an
established repo convention, not invented for this PR:
`FactorizeError::UnsupportedStorage` already exists pre-branch
(`tensor_like.rs:76` on `origin/main`) and other trait methods in this same
file already document the "fully generic (monomorphic)... does not
support..." pattern pre-branch.

##### `tensor_like.rs::TensorFactorizationLike::factorize_probe_columns_incremental` default (lines 1001-1111, diff 906-1013) — SOURCED-PAPER(§C.2 step 5, report.tex:1265-1287)

Trait method + generic default (stack all columns, one from-scratch QR). The
"recompute the QR factorization of the full prefix from scratch" fallback
semantics: the paper's own adaptive loop explicitly allows "recompute (or
update, see App. C.3)"; the from-scratch default is the "recompute" branch,
faithfully implemented as a plain `stack_along_new_index` +
`factorize_full_rank(..., QR, Left)`. Trait/default plumbing itself is
`[AI-Supplied]`, independently reviewed and found correct.

##### `tensor_like.rs::TensorFactorizationLike::src_error_estimate` default (lines 1113-1122) — DERIVED-VERIFIED (trivial)

Trait method + generic default (`Err(UnsupportedStorage(...))`). Default just
delegates the real numerics to `tensor4all_tensorbackend::src_error_estimate`
(verified real, `backend.rs:409`) or errors if unsupported. No math is
performed in this trait default itself. **Its correctness is WS-backend's
audit territory, out of WS-core scope** — this row already reflects the
correct division of labor between the two workstreams; see the Summary for
how the C1 fix aligns the rest of this document with this row rather than
contradicting it.

##### `tensor_like.rs::TensorConstructionLike::from_dense_any` default (lines 1171-1249, diff 1017-1089) — DERIVED-VERIFIED

Trait method + generic default (one-hot/axpby dense-from-column-major
fallback). Re-derived independently: builds `Σ_i data[i] · e_i` via `onehot`
+ `scale` + `axpby` over a mixed-radix (column-major) index decomposition of
the linear position — this is definitionally what "construct a dense tensor
from a column-major payload" means; no paper-specific content, holds by
construction. Skips zero-valued entries (an optimization, not a correctness
issue — verified the `is_zero()` skip does not change the result since it's
additive identity).

##### `tensor_like.rs::TensorConstructionLike::stack_along_new_index` default (lines 1249-1313, diff 1091-1189) — DERIVED-VERIFIED

Trait method + generic default (outer-product-with-onehot-batch-vector
fallback). Re-derived: stacking `T_1,...,T_n` along a fresh axis of size `n`
equals `Σ_k T_k ⊗ e_k` (outer product with the k-th standard basis vector of
the new index), then optionally permuted to the caller's requested axis
position — holds by construction; matches the standard definition of a
stack/concatenate-along-new-axis operation.

##### `tensor_like.rs::TensorConstructionLike::concatenate_along_new_index` default (lines 1313-1409, diff 1191-1304) — DERIVED-VERIFIED

Trait method + generic default (validate, then slice-and-restack via
`select_indices` + `stack_along_new_index`). Generic (backend-agnostic)
counterpart of the native override above; re-derived the same way — splits
each operand into per-column slices via the pre-existing `select_indices`
trait method (confirmed pre-existing, not part of this diff) and restacks
via the just-derived `stack_along_new_index`. Internally consistent with the
native `idx_tensor.rs` override's semantics (both preserve tensor order and
require identical index order away from the concatenated axis).

#### Verification

```
$ cargo check --manifest-path crates/tensor4all-core/Cargo.toml
    Finished `dev` profile [unoptimized] target(s) in 24.93s
```

`tensor4all-core` (with `tensor4all-tensorbackend` as a dependency) compiles
cleanly against these changes. No `cargo test`/doctest run was performed by
WS-core (out of scope for a report-only audit pass and delegated to
WS-tests, which independently judges test-category correctness against Tier
1); the `cargo check` above only confirms the API surface and type-level
correctness of the bridge, not the numerical correctness of the incremental
QR append path.

#### Summary

- **Priority check discharged**: `factorize.rs` contains no new
  factorization logic and no hand-rolled SVD/QR — the audit's founding
  suspicion does not live in this file. (See top section.)
- **No hallucinated/invented API names found among the 8 symbols that
  pre-date this branch.** `dot_general_with_conj`, `DotGeneralConfig`,
  `common_ind_positions`, `is_contractable`, `select_indices`,
  `ensure_shape_packing_preserves_ad`, `from_col_major_vec`,
  `as_col_major_slice` were grepped directly against `origin/main` (not just
  HEAD) and confirmed to exist there with matching signatures — genuine
  evidence against invention for these eight. **The other 4 symbols used by
  the incremental-QR bridge cannot be checked this way**: `IncrementalQr`,
  `IncrementalQrScalar`, `SrcErrorEstimate`, and `src_error_estimate` exist
  only in files this same branch added, so grepping HEAD and finding them
  only proves the crate compiles against its own new API, not that the names
  weren't invented in this same session. An earlier draft of this summary
  treated all 12 names as equally strong evidence against invention; that
  was circular for these four and has been corrected here (Finding I4).
- **No `HANDROLLED-DUPLICATE` finding within WS-core's own files — this is
  not the same as a clean bill of health for the backend code WS-core's
  files call into.** The retained-contraction bridge
  (`try_contract_pairwise_retaining`) and the trait-forwarder code in
  WS-core's files genuinely reuse pre-existing primitives:
  `dot_general_with_conj`/`DotGeneralConfig` (pre-existing tenferro),
  `EagerTensor::concatenate` (pre-existing `tenferro-ad`),
  `common_ind_positions`/`is_contractable`/`select_indices` (pre-existing
  `tensor4all-core`) — none of that is reimplemented linear algebra.
  However, `IncrementalQr`, `IncrementalQrScalar`, `SrcErrorEstimate`, and
  `src_error_estimate` are **not** pre-existing backend primitives: they are
  **branch-new**, confirmed absent from `origin/main`
  (`git cat-file -e origin/main:crates/tensor4all-tensorbackend/src/incremental_qr.rs`
  fails; `git show origin/main:.../backend.rs | grep -c "fn src_error_estimate"`
  returns `0`). An earlier draft of this summary described these four as
  "pre-existing backend primitives" the new code merely "reuses" — that is
  wrong and has been corrected here (Finding C1). WS-core's own files
  (`idx_tensor.rs`, `tensor_like.rs`) do not themselves contain hand-written
  linear algebra: every incremental-QR/error-estimate code unit in them is a
  thin dtype-dispatch/bookkeeping bridge that calls into
  `tensor4all-tensorbackend`'s branch-new `IncrementalQr`/`src_error_estimate`
  API (see the per-unit row above), never reimplementing the numerics
  locally. Whether *that backend code itself* duplicates something is
  WS-backend's audit territory, not WS-core's — and WS-backend's audit
  (`docs/plans/audit-workstreams/ws-backend.md`, finding F1) already found a
  real one: `incremental_qr.rs` re-implements LAPACK
  `geqrf`/`larfg`/`larf`/`ormqr`/`orgqr` in scalar Rust instead of routing
  through the backend's existing `qr_backend`/tenferro QR path, a
  `HANDROLLED-DUPLICATE` **scoped to the adaptive (`rtol.is_some()`) SRC path
  only** — the fixed-rank path already routes through tenferro's QR and never
  reaches `IncrementalQr` at all. This matches the `src_error_estimate`
  per-unit row in this document, which already correctly deferred
  backend-numerics correctness to WS-backend's scope rather than asserting a
  no-duplication conclusion for it — the corrected summary here now aligns
  with that row instead of contradicting it.
- **`PLAN-CLAIM-UNVERIFIED` (Finding F1)**: the plan's own profiling
  precondition for these additions was not met — no component-level profile
  exists anywhere on the branch, and the branch's own worklog admits the
  "formal performance experiment" is still outstanding. This is a finding
  about the *plan*, per the taxonomy; it does not on its own indict the code,
  and Finding F2 shows roughly half the added surface (the `from_dense_any`/
  `stack_along_new_index` native paths) was already-existing functionality
  being exposed through a new trait seam, not fresh unprofiled work. The
  genuinely new, unprofiled units are `try_contract_pairwise_retaining`/
  `contract_retaining_indices` and `concatenate_along_new_index`
  (native + generic default) plus the incremental-QR bridge.
- **Self-citation pattern (Finding F3)**: several new functions' doc/
  provenance comments say a code unit is "labelled `[AI-Supplied]` in the
  audit," without naming a specific file. This workstream infers (from
  matching line ranges and labels, not from the comments' own text) that the
  referenced document is
  `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`,
  which was added in the **same commit** (`9e018d4`) as the code that
  references it — same session, same author, no independent review in
  between. Every claim those comments make was independently re-traced to
  Tier 1 above rather than accepted on the worklog's authority; the ones
  checked (Appendix C.3, the `incrementalqr.py`/`.cpp` append contract,
  §C.2's recompute-or-update semantics) held up. The citation *pattern
  itself* is still a concern for the rest of the branch (other workstreams
  should not accept `[Derived]`/`[AI-Supplied]`/`[Repo]` labels from that
  worklog at face value either).
- No `SUSPECT-UNVERIFIED`, `MISSING-VS-SOURCE`, `SCOPE-DEVIATION`,
  `LICENSE-RISK`, or `SOURCE-AMBIGUOUS` findings in WS-core's file list.


---

## WS-integration: dispatch and public API

*Source: [`docs/plans/audit-workstreams/ws-integration.md`](audit-workstreams/ws-integration.md), reproduced in full.*

### WS-integration — dispatch, public API, and cross-cutting glue

**Files audited:**
- `crates/tensor4all-treetn/src/treetn/contraction.rs` (diff only: SRC
  dispatch/options additions, 2108 lines total)
- `crates/tensor4all-treetn/src/operator/apply.rs` (full file, SRC-related
  hunks)
- `crates/tensor4all-treetn/src/treetn/fit.rs` (diff only)
- `crates/tensor4all-treetn/src/treetn/swap.rs` (diff only)
- `crates/tensor4all-treetn/src/algorithm.rs` (diff only)
- `crates/tensor4all-itensorlike/src/options.rs` (full file, 527 lines)
- `crates/tensor4all-itensorlike/src/contract.rs` (full file, 162 lines)
- `crates/tensor4all-capi/src/treetn.rs` (diff only)
- `crates/tensor4all-capi/src/types.rs` (diff only)
- `crates/tensor4all-capi/include/tensor4all_capi.h` (diff only)
- `crates/tensor4all-treetn/src/lib.rs`, `crates/tensor4all-treetn/src/prelude.rs`,
  `crates/tensor4all-treetn/README.md` (plumbing, diff only)
- `crates/tensor4all-itensorlike/src/lib.rs`,
  `crates/tensor4all-itensorlike/src/prelude.rs` (plumbing, diff only —
  `tensor4all-itensorlike/README.md` has no diff against `origin/main` and is
  not touched by this branch; see M4 note below)
- `crates/tensor4all-itensorlike/src/tensortrain.rs` (added per Fix round 1;
  in the diff, unclaimed by any other workstream — see the dedicated section
  below)

**Scope:** `crates/tensor4all-treetn/src/treetn/contraction.rs` (diff only:
SRC dispatch/options additions),
`crates/tensor4all-treetn/src/operator/apply.rs`,
`crates/tensor4all-treetn/src/treetn/fit.rs`,
`crates/tensor4all-treetn/src/treetn/swap.rs`,
`crates/tensor4all-treetn/src/algorithm.rs`,
`crates/tensor4all-itensorlike/src/options.rs`,
`crates/tensor4all-itensorlike/src/contract.rs`,
`crates/tensor4all-capi/src/treetn.rs`, `crates/tensor4all-capi/src/types.rs`,
plus the plumbing files listed in the spec's WS-integration section
(`lib.rs`/`prelude.rs` in `tensor4all-treetn` and `tensor4all-itensorlike`,
`tensor4all-treetn/README.md` — the only one of the two crates' `README.md`
files this branch actually touches (see M4 note below) —,
`crates/tensor4all-capi/include/tensor4all_capi.h`).
`src_chain.rs`/`src_tree.rs`/`src_probe.rs` bodies are WS-chain /
WS-tree-probe territory; they are referenced here only where the
`PrefixCache`/chain-reduction-gate checks required a look past the `mod`
declarations added in `contraction.rs`.

Verdict taxonomy per
[`docs/plans/2026-08-28-src-provenance-audit.md`](2026-08-28-src-provenance-audit.md).
Every diff hunk in the five main files plus `options.rs`/`contract.rs` is
covered below; plumbing files get one summary row each per the spec's
catch-all allowance.

#### Three named checks (spec's WS-integration section)

##### 1. Public API surface vs. the plan's proposed shapes

The Tier-2 plan (`docs/plans/2026-08-26-treetn-src-contraction-plan.md`)
proposes `SrcRankSelection` (enum), `SrcContractionResult<T,V>`,
`SrcContractionReport<V>`, `SrcEdgeReport<V>` (lines 270, 276, 304, 306,
309-310), and top-level `global_src_fixed`/`global_src_adaptive` functions
(lines 612-613).

**None of these exist anywhere in the crate** (repo-wide grep, zero hits).
What actually exists is a much simpler surface: `ContractionMethod::Src`
(unit variant, no payload), `SrcOptions` (plain struct: `rtol`, `atol`,
`min_rank`, `rank_increment`, `max_rank`, `final_svd`, `seed`),
`ContractionOptions::src()`/`.with_src_options(...)`, and the mirrored
`ApplyOptions::src()`/`.with_src_options(...)` and
`ContractOptions::src()`/`.with_src_options(...)`/`.src_options()` in
`tensor4all-itensorlike`. `contract()` still returns a plain
`TreeTN<T, V>`, exactly like the `Zipup`/`Fit`/`Naive` arms — no per-edge
diagnostics/report object is produced or threaded through anywhere in the
five main files.

The `src_fixed`/`src_adaptive` name fragments that do exist are **test
function names** in `contraction/tests/mod.rs`
(`src_fixed_matches_exact_contraction_when_probe_cap_is_full`,
`src_adaptive_contracts_and_honors_rank_cap`, etc.), not the
`global_src_fixed`/`global_src_adaptive` constructors the plan proposed.

**Finding (code-vs-plan, not itself a taxonomy verdict on the code):** the
implementation is a simplification of the plan's proposed API, not a
mismatch that makes the code wrong — `SrcOptions`'s `rtol: Option<f64>`
switch cleanly encodes the plan's fixed/adaptive `SrcRankSelection` split
without a separate enum, and the option's own `validate()` (see table)
enforces the same fixed/adaptive constraints the plan's enum would have.
The *report/diagnostics* surface (`SrcContractionResult`/`Report`/`EdgeReport`)
is simply not built at all — every other contraction method in this crate
also returns a bare `TreeTN`, so this is consistent with existing
conventions rather than a gap unique to SRC. Verdict: the actual builder
surface is `SOURCED-COMMENT(#563, 2026-07-29T07:13:51Z + opening post)` —
it delivers exactly the "fixed output rank and adaptive rank selection"
the opening post asked for, via a smaller surface than Tier 2 sketched.
The plan's report-object proposal is simply unimplemented; that is a
factual note about the *plan* being aspirational here, not a
`PLAN-CLAIM-UNVERIFIED` (the plan doesn't misstate any Tier-1 fact — it
just proposes API shapes that were never built).

##### 2. The mandatory "chain reduction gate" test

Spec text (`2026-08-26-treetn-src-contraction-plan.md:190-201`) defines the
gate as an **internal chain test that independently proves four specific
identities** against "a direct implementation of the paper equations on a
small deterministic input": `E[parent→child,k]` equals the paper's forward
environment column, postorder compression runs right-to-left, `P[v]`
equals the paper's projection cap, and the root contraction equals the
paper's first-site completion.

Grepped `contraction.rs`, `contraction/tests/mod.rs`, and the whole
`contraction/` module tree for `chain reduction`, `chain_reduction`,
`forward environment`, `forward_environment`, and `paper equation`: **zero
hits anywhere.** No test with that name, and no test structured as an
independent re-implementation of the four paper identities, exists.

What exists instead (`contraction/tests/mod.rs:518-1000`, all calling the
public `contract()` dispatcher, so genuinely exercised through the same
path `contraction.rs` wires up): six `src_fixed_*`/`src_adaptive_*`/
`src_complex_*` tests (`tests/mod.rs:531, 560, 586, 728, 829, 997` — the
dense-output-vs-`tn_a.contract_naive(&tn_b)` residual-assertion lines) that
build a chain (or branched-tree, or complex-`f64`) input pair, run
`ContractionOptions::src()` at `final_svd: false` with the probe cap large
enough to be lossless, and assert the **dense output**
matches `tn_a.contract_naive(&tn_b)` to `< 1e-8`, plus
`validate_ortho_consistency()`/canonical-center/topology assertions. This
is an end-to-end numerical oracle test (SRC output vs. an independent,
pre-existing exact contraction path), not a check of the four named
intermediate identities.

**Independent re-derivation of whether this is a meaningful gate at all**
(not trusting the plan's description, and not trusting the worklog): when
the sketch width at every cut is at or above the exact rank needed there
(the "probe cap is full" condition each test title states), a randomized
QB sketch generically has the same column space as the exact operator
(probability 1 over the continuous Gaussian draw), so `QR`-projecting
against it reproduces the exact contraction bit-for-bit up to floating
point error. A pass at `< 1e-8` residual with this setup is therefore a
real, non-vacuous correctness signal for the whole `src_chain.rs`
pipeline's *end-to-end* composition (sketch → QR → projection → cache
reuse → assembly), including cases a purely-unit-level identity check
could miss (integration bugs between the four steps). It does **not**,
however, independently verify each of the four sub-identities the plan's
gate specifically named — a compensating pair of sign/ordering errors in
two of the four steps could in principle cancel and still pass this test.

**Independently checked the dispatch routing, not just trusted `contract()`
calls what it looks like it calls:** `contraction.rs`'s `Src` arm always
calls `src_tree::contract(...)`, never `src_chain::contract` directly.
`src_tree::contract` (line ~44 of `src_tree.rs`) only delegates to
`src_chain::contract` when `tn_a.chain_order(center)` succeeds **and**
`chain.last() == Some(center)`. Traced `chain_order`'s endpoint-selection
logic (`contraction.rs:404-454`, real source lines — corrected from an
earlier diff-offset citation):

```rust
let (start, end) = if center == &endpoints[0].1 {
    (endpoints[1].0, endpoints[0].0)
} else {
    (endpoints[0].0, endpoints[1].0)
};
```

`endpoints` is sorted alphabetically before this branch runs, so
`endpoints[0]` is always the alphabetically-first degree-1 endpoint. This
`if`/`else` does **not** decide *whether* `chain_order` returns
`Some`/`None` at all — that decision is made earlier and is purely
topological (the `node_count`, `graph.edge_count() != node_count - 1`, and
`endpoints.len() != 2` guards), independent of `center`. `chain_order`
returns `Some(path)` for **any** valid chain topology, whether `center` is
one of the two degree-1 endpoints or an interior node. What the `if`/`else`
picks is only the path's *orientation* between the two already-known
endpoints: if `center` equals the alphabetically-first endpoint
(`center == &endpoints[0].1`), it flips `start`/`end` so the walk ends at
`endpoints[0]` (i.e. at `center`); otherwise — which covers both the case
where `center` is the alphabetically-*second* endpoint **and** the case
where `center` is an interior node whose name matches neither endpoint —
the walk runs `endpoints[0] -> endpoints[1]` and ends at `endpoints[1]`. So
when `center` is interior, `chain_order` still returns `Some(path)`, but
that path ends at `endpoints[1]`, not at `center` — the function has no way
to know or care that `center` wasn't one of the two endpoints it found.

The real discriminator for whether `src_tree::contract` delegates to
`src_chain::contract` is therefore not `chain_order`'s return value alone
but `src_tree.rs`'s separate follow-up check, `chain.last() == Some(center)`
(`src_tree.rs:49`, with the rationale spelled out in the comment at
`src_tree.rs:46-48`): it rejects exactly the interior-center case just
described, where `chain_order` succeeded but the path it returned doesn't
actually terminate at `center`. For the shared fixture
`make_three_node_chain_pair()` (nodes `"A"`-`"B"`-`"C"`, center `"C"`), the
graph is a 3-node chain, `center` `"C"` is one of its two degree-1
endpoints, so `chain_order` returns `Some([...])` ending at `"C"` and the
delegation condition holds — these tests do genuinely exercise
`src_chain.rs`'s paper-faithful chain path, not the general tree fallback.
(Confirmed by direct trace of the real branch logic, not the earlier
mis-stated "not alphabetically first" framing — the fixture's conclusion
was already correct, only the reasoning needed fixing.)

**Verdict:** the plan's literal "chain reduction gate" (identity-level,
against a hand-written reference of the paper's equations) **does not
exist** — this is a genuine gap relative to the Tier-2 precondition. A
different, real, and non-trivial end-to-end regression gate exists in its
place (`DERIVED-VERIFIED`, derivation above) and is wired through the same
public dispatch path this workstream audits. This overlaps with WS-tests'
formal file ownership (`contraction/tests/mod.rs`); flagging here because
the *routing* proof (which path the tests hit) is dispatch-logic-specific
to this workstream's files.

##### 3. `PrefixCache` trait ask (2026-08-27 comments) — honored, ignored, or over-built?

Grepped the five main WS-integration files
(`contraction.rs`, `apply.rs`, `fit.rs`, `swap.rs`, `algorithm.rs`) for
`PrefixCache`, `trait.*[Cc]ache`, `Cache.*trait`: **zero hits.** The cache
does not live in any WS-integration file at all — `mod src_chain;` in
`contraction.rs` only names the module; the cache types live inside
`src_chain.rs` (WS-chain's file). Followed the reference anyway because
Step 4's verification requires this ask to be "explicitly addressed one
way or the other," and because `contraction.rs`'s own module doc-comment
(see finding below) makes provenance claims about this exact code.

Found in `src_chain.rs`: two concrete structs, `PrefixCache` (line 464,
`prefixes: Vec<Vec<T>>`, grown in `ensure_width`'s `while start < width`
forward loop, lines 620-624) and `BatchedPrefixCache` (line 475,
`segments: Vec<PrefixBatchSegment<T>>`, grown in `batch`'s
`if width > self.generated_width` forward branch, lines 537-567).
`grep -rn "^trait \|trait.*Cache"` across the whole diff: **zero matches.
No trait named `PrefixCache` or anything cache-shaped exists anywhere in
the branch.**

Hiroshi's exact ask (2026-08-27T12:56:57Z, status: not retracted by the
15:38 correction — only the scan/tree-parallelism *framing* was retracted,
reconfirmed unchanged by the 15:38 comment's closing line "The
implementation ask is unchanged: a small cache trait, flat list first"):

> put the cache behind a small trait (something like `PrefixCache: fn
> extend(piece), fn get(k)`) instead of hard-coding a Vec built in a
> forward loop

Side by side:

| Ask | What exists |
| --- | --- |
| `trait PrefixCache { fn extend(piece); fn get(k); }` | No trait; `struct PrefixCache` and `struct BatchedPrefixCache`, both concrete |
| flat list first | `PrefixCache.prefixes: Vec<Vec<T>>` — a flat, hard-coded `Vec`, grown in a forward `while` loop |
| ...blocked/tree/checkpointed policies swappable later without touching SRC logic | Two separate concrete cache implementations already exist (`PrefixCache` for adaptive mode, `BatchedPrefixCache` for fixed mode, chosen by an `if src_options.rtol.is_none()` branch in `contract()`), each with its own segment-batching/reassembly logic, with no shared trait boundary between them or between them and any future policy |

**Verdict: `SCOPE-DEVIATION`, confirmed and direct.** The trait-boundary ask
was not honored — the code is exactly the hard-coded-Vec-in-a-forward-loop
pattern Hiroshi's comment named as the thing to avoid. It is also
over-built relative to "flat list first": rather than one flat-list cache
behind a trait, there are two independent concrete cache structs
(`PrefixCache`/`BatchedPrefixCache`) with internal segment-batching logic,
selected by a runtime branch, sharing no common interface — more surface
area than "flat list first," just not (see check 4) the specific tree/scan/
MPI machinery of #691. See the hallucination-signature note below: reusing
Hiroshi's suggested type name for a struct that is architecturally the
opposite of what was asked is worth calling out on its own.

##### 4. #691 leakage (interface sketching / Layer 2 / sub-chain partitioning / MPI)

Grepped the five main files and the whole `contraction/` module tree for
`interface.sketch`, `Layer 2`, `sub.chain`/`sub_chain`, `MPI`,
`parallel.*chain`/`chain.*parallel`, `rayon`, `std::thread`, `segment.tree`,
`checkpoint` (case-insensitive): **zero hits** anywhere in the SRC diff.
`BatchedPrefixCache`'s segment/batch logic (check 3) chunks *sketch width*
for GEMM batching within the still-fully-sequential chain sweep — it never
materializes D×D boundary/transfer operators, never partitions the *chain*
itself into independently-processed sub-chains, and has no
threading/MPI/parallel-scan machinery. **Verdict: clean — no #691
material found in WS-integration's files or the adjacent cache code.**
`BatchedPrefixCache`'s extra complexity is still worth flagging (check 3)
as unrequested elaboration, just not as #691-specific scope creep.

#### Provenance table (every diff hunk)

##### `contraction.rs` (SRC-related hunks only; +369 total)

**Line numbers below are real source-file line numbers, verified by directly
reading the current `contraction.rs` (2108 lines total) — not git-diff hunk
offsets. An earlier version of this table used diff-line offsets that were
off by roughly +1300 for every SRC-specific unit; every row was
re-verified against the file as it stands.**

| Unit | Lines (real, verified) | Verdict | Notes |
| --- | --- | --- | --- |
| Module doc-comment SRC provenance block | 11-24 | `SUSPECT-UNVERIFIED` (citation) | Cites "Algorithm 1, Sections 2.3--2.5, and Appendices C--D." **The paper's Section 2 ("Background") has only two subsections (2.1 randomized QB approximation, 2.2 Khatri-Rao product) — there is no 2.3, 2.4, or 2.5.** Verified against `report.tex`'s actual `\section`/`\subsection` list. Algorithm 1 (the pseudocode box, `\begin{algorithm}` at report.tex:640) and the step-by-step exposition (Step 1/Step 2/Finishing up/Optional oversampling) are in **Section 3** (`\S`3.1-3.4), not Section 2. Appendix C ("Implementing SRC with a tolerance") is genuinely the adaptive-mode source and checks out; Appendix D ("Full operation counts") is a pure cost/complexity table with nothing the dispatch code in this file implements — its inclusion reads as citation padding. See Detailed findings below: this exact wrong "Sections 2.3--2.5" string is copy-pasted verbatim into `src_chain.rs` and `src_tree.rs` too. |
| Module doc: reference-Python function/file citation (`random_contraction`, `random_contraction_inc`, `incrementalqr.py::IncrementalQR`, `incrementalqr.cpp::{setup,add_cols,get_error_estimate}`) | 15-19 | `SOURCED-PYTHON` (citation-only check) | Spot-checked: `IncrementalQR` class and `setup`/`add_cols`/`get_error_estimate` genuinely exist in the reference repo's `incrementalqr.py`/`incrementalqr.cpp` at matching names. `random_contraction`/`random_contraction_inc` genuinely exist as top-level `def`s in `contraction.py`. This citation is accurate (full line-range verification of `src_chain.rs`'s own body is WS-chain's job). **See also the I5/LICENSE-RISK note below on line 15's own self-description as a "line-by-line cross-check."** |
| Module doc: issue #563 comment 5396107820 citation | 21-22 | `SOURCED-COMMENT(#563, 2026-08-24T13:45:35Z)` verified | Fetched the comment live via `gh api repos/tensor4all/tensor4all-rs/issues/comments/5396107820`; body and timestamp match Appendix A's transcription exactly. Accurate citation, not fabricated. |
| Module doc: "labelled `[AI-Supplied]` in the audit worklog" | 22-24 | `SOURCED-COMMENT`-adjacent, verified | `docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md` exists and does label the rooted-tree recurrence `[AI-Supplied]` (lines 137, 292-293, 434 of that worklog). Note: that worklog is itself a Tier-2 artifact (an earlier AI-generated provenance pass) per the spec's epistemics section — its own `[Derived]` math claims are not independently re-verified here, only the fact that this doc-comment's *pointer* to it is accurate. |
| `mod src_chain; mod src_probe; mod src_tree;` | 47-49 | `DERIVED-VERIFIED` (trivial plumbing) | Mechanical module wiring; content of those modules is WS-chain/WS-tree-probe's scope. |
| `ContractionMethod::Src` variant + doc-example | 1341-1368 | `DERIVED-VERIFIED` (trivial plumbing), verified | Doc-test (`with_max_bond_dim(8)`) exercises real, existing methods; not fabricated API. |
| `SrcOptions` struct + field docs | 1370-1411 | `SOURCED-PAPER` / `SOURCED-COMMENT` (field semantics), `DERIVED-VERIFIED` (trivial plumbing; struct shape) | `rtol=None ⇒ fixed / Some ⇒ adaptive` matches the opening post's "fixed output rank and adaptive rank selection" ask; `final_svd` default matches Hiroshi's 2026-07-29 QR-only-hot-path / optional-final-SVD description. |
| `impl Default for SrcOptions` | 1413-1425 | `DERIVED-VERIFIED` (trivial plumbing) | `min_rank: 2, rank_increment: 3, final_svd: false, seed: 0` — reasonable, undocumented-in-paper policy defaults; consistent with `final_svd: false` matching "core loop is QR-only" framing. |
| `sketch_options` (private, oversampling tolerance tightening) | 1428-1436 | `SOURCED-PAPER(report.tex:1286)` verified | Independently grepped `report.tex`: line 1286 reads *"To implement with oversampling, we set the relative tolerance to be 0.1 times the requested tolerance and run a final truncation with the requested tolerance."* The code's `options.rtol = Some(0.1 * rtol)`, gated on `self.final_svd && final_truncation_has_tolerance`, matches this exactly — gating on `final_svd` is the only way this makes sense (no final truncation ⇒ no round to "run with the requested tolerance" ⇒ applying 0.1× would leave the result under-converged with no fix-up step). Confirmed against the paper directly, not the plan's transcription. |
| `SrcOptions::fixed()` | 1438-1451 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `SrcOptions::adaptive(rtol, max_rank)` | 1453-1474 | `SOURCED-COMMENT`(opening post: "adaptive rank selection") | — |
| `with_rtol`/`with_atol`/`with_min_rank`/`with_rank_increment`/`with_max_rank`/`with_final_svd`/`with_seed` builders | 1476-1579 | `DERIVED-VERIFIED` (trivial plumbing), each doc-tested | Straightforward field setters; no logic to independently verify beyond the doc-tests, which all pass their own assertions inline. |
| `SrcOptions::validate()` | 1581-1649 | `DERIVED-VERIFIED` | No direct paper/Python/Hiroshi source names this exact validation policy (it is API-level input-hygiene, not algorithm math), so this is necessarily derived. Re-derived: fixed mode needs a finite target rank (`output_max_bond_dim`) since there's no stopping test to determine one; adaptive mode needs `rtol` finite/non-negative and a finite `max_rank` (from `self.max_rank` or the ambient `max_bond_dim`) since the adaptive loop in `src_chain.rs` must have a hard stop; `min_rank ≤ max_rank` is required or the loop could never satisfy its own precondition; `max_rank > output_max_bond_dim` is only safe when `final_svd` will cut it back down. Each check maps to a real downstream invariant. Holds up. See the dedicated derivation subsection below (Detailed derivations and flagged findings). |
| `ContractionOptions.src_options` field + `Default` update | 1686-1687, 1702 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ContractionOptions::src()` / `.with_src_options(...)` | 1726-1755 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `contract()`: `SrcOptions::validate` call before dispatch | 1940-1945 | `DERIVED-VERIFIED` (trivial plumbing) | Correctly runs *after* the existing `validate_svd_truncation_options` call (1938-1939), before any method-specific branch — consistent with how the other methods' preconditions are already checked. |
| `contract()`: `ContractionMethod::Src` dispatch arm | 1987-2001 | `DERIVED-VERIFIED` (trivial plumbing / routing) | `output_rank = max_bond_dim.or(src_options.max_rank)` then calls `src_tree::contract(...)`. Traced (see check 2 above) that this always enters `src_tree::contract`, which conditionally re-delegates to `src_chain::contract` for genuine chain+endpoint-center cases. No dispatch bug found. |

##### `apply.rs` (+44/-2)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Module doc "ZipUp, Fit, SRC, or local exact naive apply" | ~9-11 | `DERIVED-VERIFIED` (trivial plumbing) | Accurate: `Src` is now a real dispatchable method here (see below), not a doc-only mention. |
| `use ...SrcOptions` import; `ApplyOptions` doc default note | ~85-95, ~107-127 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ApplyOptions.src_options` field + `Default` update | 150-166 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ApplyOptions::src()` | 196-211 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ApplyOptions::with_src_options(...)` | 237-254 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `contraction_options` struct literal: `src_options: options.src_options` | ~394-397 (context read, exact hunk line 400 of diff) | `DERIVED-VERIFIED` (trivial plumbing), verified | Traced: this is the internal `ContractionOptions` built inside `apply_linear_operator` and passed to the shared `contract()` in `contraction.rs`. Confirms `ApplyOptions::src()` genuinely reaches the SRC dispatch arm — this is real plumbing, not a dead/unused field. Matches the opening post's "Possibly extend to TreeTN apply later" scope item. |

##### `fit.rs` (+1/-7) and `swap.rs` (+2/-14)

| File | Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- | --- |
| `fit.rs` | `FactorizeResult { left, right, bond_index: dummy_left, singular_values: None, rank: 1 }` → `FactorizeResult::new(left, right, dummy_left, None, 1)` | 723 | `DERIVED-VERIFIED` (trivial plumbing), mechanical | Verified `FactorizeResult::new` genuinely exists at `tensor4all-core/src/tensor_like.rs:517` (corrected from an earlier ~487 citation) with a matching 5-argument signature `(left, right, bond_index, singular_values, rank)`. The struct gained a new **private** field `incremental_qr_state: Option<IncrementalQrState>` (WS-core's `IncrementalQr` addition), which is why the old positional struct-literal syntax stopped compiling outside its defining module — `::new(...)` sets that field to `None`. This is forced, cross-crate compile-fallout plumbing, not new logic and not a hallucinated API — confirmed the constructor is real and does what the call sites need. |
| `swap.rs` | Same replacement, two identical call sites | 59, 81 | `DERIVED-VERIFIED` (trivial plumbing), mechanical | Same `FactorizeResult::new` constructor, same forced compile-fallout reasoning as the `fit.rs` row above. |

##### `algorithm.rs` (+7)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Doc comment `T4A_CONTRACT_SRC = 3` | 36 | `DERIVED-VERIFIED` (trivial plumbing) | Matches the enum addition below and the capi header. |
| `ContractionAlgorithm::Src = 3` variant + doc | 66-67 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `from_i32` match arm `3 => Some(Self::Src)` | 79 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `name` match arm `Self::Src => "src"` | 95 | `DERIVED-VERIFIED` (trivial plumbing) | Corrected method name: the match is inside `pub fn name(&self) -> &'static str`, not an `as_str` method — no `as_str` exists anywhere in this file. |

##### `tensor4all-itensorlike/src/options.rs` (+55)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `pub use tensor4all_treetn::contraction::SrcOptions;` | 4 | `DERIVED-VERIFIED` (trivial plumbing) | Re-export, no new logic. |
| `ContractMethod::Src` variant | 104-105 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ContractOptions.src_options` field + `Default` update | 132, 143 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ContractOptions::src()` | 191-206 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ContractOptions::with_src_options(...)` | 260-275 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ContractOptions::src_options()` getter | 321-335 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |

##### `tensor4all-itensorlike/src/contract.rs` (+4/-1)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `ContractMethod::Src => ContractionMethod::Src` match arm | 70 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `.with_src_options(options.src_options().clone())` chained onto `treetn_options` | 77 | `DERIVED-VERIFIED` (trivial plumbing), verified | Confirms `tensor4all-itensorlike`'s `ContractOptions::src()` genuinely threads through to `tensor4all-treetn`'s `ContractionOptions` — not a dead/unused field on this side either. |

##### `tensor4all-capi/src/types.rs` (+3)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `t4a_contract_method::Src = 3` | 611 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `From<ContractionMethod>`/`From<t4a_contract_method>` match arms | 620, 631 | `DERIVED-VERIFIED` (trivial plumbing) | Bidirectional conversion is complete and consistent with `algorithm.rs`'s `ContractionAlgorithm::Src = 3`. |

##### `tensor4all-capi/src/treetn.rs` (+9) and `include/tensor4all_capi.h` (+12)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Doc-comment additions on `t4a_treetn_contract` (`treetn.rs:1606-1608`, `tensor4all_capi.h:1045-1046`), `t4a_treetn_partial_contract` (`treetn.rs:1717-1719`, `tensor4all_capi.h:1231-1232`), `t4a_treetn_apply_operator_chain` (`treetn.rs:1848-1849`, `tensor4all_capi.h:1005-1006`) — "`maxdim` must be nonzero...fixed-rank...Adaptive SRC controls are currently available through the Rust API only" | see cell | `DERIVED-VERIFIED` (trivial plumbing), verified accurate | `t4a_treetn_apply_operator_chain`'s internal `ApplyOptions`-equivalent struct literal sets `src_options: Default::default()` (fixed-rank, seed 0) with no C-ABI parameter to override it — the doc's "Rust API only" claim for adaptive controls is factually correct, not aspirational. |

##### Plumbing files (mechanical re-export/registration only, one row each)

| File | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `tensor4all-treetn/src/lib.rs` | 63 | `DERIVED-VERIFIED` (trivial plumbing) | `pub use treetn::contraction::SrcOptions;` re-export. No logic. |
| `tensor4all-treetn/src/prelude.rs` | 23 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the prelude re-export list. No logic. |
| `tensor4all-treetn/README.md` | 3-4 | `DERIVED-VERIFIED` (trivial plumbing) | One sentence: "naive/zip-up/fit/SRC contraction." Descriptive only. |
| `tensor4all-itensorlike/src/lib.rs` | 15 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the crate's `pub use options::{...}` list. No logic. |
| `tensor4all-itensorlike/src/prelude.rs` | 24 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the prelude re-export list. No logic. |
| `tensor4all-itensorlike/README.md` | n/a | N/A — not touched | `git diff origin/main -- crates/tensor4all-itensorlike/README.md` is empty. This file is **not** part of the SRC diff, unlike `tensor4all-treetn/README.md` above (see M4 fix note: an earlier prose pass in this document incorrectly implied both crates' `README.md` were touched). Row kept for completeness since the spec's plumbing-file list names it. |

##### `tensor4all-itensorlike/src/tensortrain.rs` (added per Fix round 1 — previously uncovered)

This file is in the `feature/treetn-src` diff against `origin/main`, is a
non-test source file in a crate this workstream already owns
(`tensor4all-itensorlike`), and was not claimed by any other workstream's
file list. It belongs in this table.

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `TensorTrain::inner()` — empty-tensor-train return value | 1370-1382 (empty case: 1380-1382) | `N/A — rebase-lag artifact, not attributable to feature/treetn-src` | Changes the empty-TT `inner()` result from `1.0` to `0.0` (`return Ok(AnyScalar::new_real(0.0));` at line 1381). |
| `TensorTrain::norm_squared_fast_path()` — empty-tensor-train return value | 1591-1593 | `N/A — rebase-lag artifact, not attributable to feature/treetn-src` | Changes the empty-TT fast-path result from `Some(1.0)` to `Some(0.0)` (`return Ok(Some(0.0));` at line 1592). |

**Not independently re-derived here** — WS-tests already diagnosed this
exact pair of hunks (`docs/plans/audit-workstreams/ws-tests.md`, "§0"
section around lines 46-111, and table rows 20-21 around lines 136-137) and
that diagnosis is adopted by reference rather than duplicated: `git
merge-base origin/main HEAD` (`72de8fb`) is an older common ancestor, not
`origin/main`'s own tip (`fd61f08`); `origin/main` independently picked up
PR `#693` (`fd61f08`), which *restores* the correct convention (empty
tensor train is the mathematical scalar `1`, matching `scalar_one()`'s
`// Empty tensor train represents scalar 1` comment at
`tensortrain.rs:2224-2226`) after this SRC branch had already forked from
an older commit that still had `0.0`. `git diff origin/main` surfaces these
two hunks only because the diff base moved out from under this branch, not
because `feature/treetn-src` itself touched or regressed this logic.
WS-tests' own process flag applies equally here, but **not** as a
merge-regression risk: WS-tests explicitly verified that `git diff --stat
72de8fb..HEAD -- crates/tensor4all-itensorlike/src/tensortrain.rs
crates/tensor4all-itensorlike/tests/tensortrain_inner.rs
crates/tensor4all-quanticstci/src/quantics_tci/tests/mod.rs` is **empty** —
`feature/treetn-src` has made zero changes to any of the three files
`#693`/`fd61f08` touched, relative to the merge base. There is nothing on
this branch for a merge or rebase to overwrite in those files, so merging
`feature/treetn-src` as-is simply carries `#693`'s fix along unmodified and
does **not** reintroduce the `0.0` bug. The accurate framing, matching
WS-tests' own conclusion, is about staleness, not regression risk: the
branch's own SRC-related test checks (and its `.inner()`/`scalar_one()`
behavior generally) were run against a stale, pre-`#693` base, so results
derived from running this branch's own test suite as-is should not be
treated as final until it syncs with `origin/main` — but the sync itself
(rebase or merge) carries no risk of reintroducing the `#693` bug. No
taxonomy verdict token cleanly fits an artifact that predates the branch's
own diff base moving, hence `N/A` rather than e.g.
`SUSPECT-UNVERIFIED`/`MISSING-VS-SOURCE` — this mirrors WS-tests' own
choice of token for the same underlying fact.

#### Detailed derivations and flagged findings

##### Finding 1 — Fabricated section citation, copy-pasted across three files (`SUSPECT-UNVERIFIED`, AI-hallucination pattern: *confident-sounding doc text describing something the code/paper doesn't have*)

`contraction.rs:11`, `src_chain.rs:4`, and `src_tree.rs:4` all state, verbatim,
that the SRC implementation "follows Algorithm 1, Sections 2.3--2.5, and
Appendices C--D" of arXiv:2504.06475. Independently walked every
`\section`/`\subsection` in the local `report.tex`:

```
1  Introduction
2  Background            2.1 Randomized QB approximation   2.2 Khatri-Rao product
3  Successive randomized compression (Algorithm 1 is here, \begin{algorithm} at line 640)
     3.1 Step 1: last site   3.2 Step 2: second-to-last site
     3.3 Finishing up        3.4 Optional: oversampling and final round
     3.5 Pseudocode/complexity  3.6 Linear combinations  3.7 Physical symmetries
4  Existing MPO-MPS product methods
5  Application: unitary time evolution
A  Proof of Theorem 2 (Khatri-Rao)
B  Proof of Theorem 3 (SRC-exact)
C  Implementing SRC with a tolerance   (error estimation, adaptive rank, QR updating)
D  Full operation counts
```

Section 2 ("Background") has exactly two subsections, 2.1 and 2.2 — there
is no 2.3, 2.4, or 2.5 anywhere in the paper. The actual algorithmic
content the doc-comment is presumably trying to cite (Algorithm 1, the
step-by-step last-site/second-to-last-site/finishing-up/oversampling
exposition) lives in **Section 3**, subsections 3.1-3.4. This is not a
close paraphrase or an off-by-one section-numbering convention issue — 2.3
through 2.5 simply do not exist. The same exact wrong string appears
identically in three separate files' module doc-comments, which rules out
a one-off typo and points to a templated/copy-pasted citation that was
never checked against the source it names. Appendix C is a legitimate,
verified citation for the adaptive-mode content (see `sketch_options`
above); Appendix D ("Full operation counts," a pure cost table) has no
content this dispatch code implements and reads as citation padding rather
than a real source. This is exactly the "confident-sounding doc text
describing behavior the code doesn't implement" hallucination pattern
called out in the audit brief — here it's a citation rather than a
behavior claim, but the mechanism (specific, authoritative-sounding, and
wrong) is the same.

##### Finding 2 — `PrefixCache` name reused for the opposite of what was asked (`SCOPE-DEVIATION`, AI-hallucination pattern: *a comment/name claiming alignment with a request that doesn't hold up when checked*)

See check 3 above for the full side-by-side. Worth stating explicitly as
its own finding because of the naming: Hiroshi's comment suggested the
trait be "something like `PrefixCache: fn extend(piece), fn get(k)`." The
code that resulted has a type literally named `PrefixCache` — but it is a
concrete `struct` wrapping `Vec<Vec<T>>`, not a `trait`, and there is a
second, differently-structured concrete cache (`BatchedPrefixCache`)
alongside it with no shared interface. Adopting the exact suggested name
for a structure that is architecturally the opposite of the ask (a fixed
concrete implementation instead of a swappable trait boundary) is worth
flagging on its own — it is easy to grep for "PrefixCache" and conclude
the ask was honored without reading further, which is exactly the
"comment claiming alignment that doesn't hold up when checked" pattern the
brief asks to hunt for.

##### Finding 3 — plan's report/diagnostics API surface never built (code-vs-plan comparison, see check 1)

Not a hallucination in the code itself (nothing invented pretends to be
`SrcContractionResult`/`SrcRankSelection`/etc.) — flagged here only
because Step 3 of the task brief explicitly calls for this comparison.
The simpler surface that does exist is fully consistent with Tier-1
sources (the opening post and the 2026-07-29 QR-only comment); this is a
scope note about the Tier-2 plan's aspirational API sketch, not a defect.

##### Finding 4 — chain-reduction-gate: real test suite, but not the gate as specified (see check 2)

Restated here as a hallucination-adjacent note: nothing in the code or
comments *claims* "this is the chain reduction gate" — there is no false
claim to debunk. The gap is that the Tier-2 plan's specific precondition
was silently not built as specified, and no comment anywhere says so. This
is a coverage gap for a reader relying on the plan, not a fabricated claim
in the code.

##### Finding 5 — `LICENSE-RISK` self-description in the module doc-comment (I5, this workstream's own file)

`contraction.rs:15` reads: "The author implementation used for a
line-by-line cross-check is `chriscamano/RandomMPOMPS`,
`code/tensornetwork/contraction.py`, ..." (full sentence: lines 15-19, see
the module-doc-comment row of the provenance table above). Per the spec's
Tier-1 epistemics section, the reference Python repository has no detected
license, and **code that reads as a line-by-line translation against it is
a `LICENSE-RISK` finding in its own right, separate from whether the
citation is accurate.** The citation itself checks out (the named functions
and files genuinely exist at those names — see the provenance table row),
but the doc-comment's own chosen phrase, "line-by-line cross-check," is the
kind of self-description the spec specifically calls out as a trigger: it
describes the author's *methodology* against the unlicensed reference in
terms ("line-by-line") that, if literally true of the resulting code
(rather than just the validation process), would itself be the
`LICENSE-RISK` condition.

This workstream's own files (`contraction.rs`'s dispatch/options code
audited above) do not implement the SRC algorithm's numerical body — that
logic lives in `src_chain.rs`/`src_tree.rs`, which are WS-chain's scope.
WS-chain's provenance report (`docs/plans/audit-workstreams/ws-chain.md`,
"License-risk assessment — no finding" section) already performs the
detailed body-level comparison against `contraction.py`'s reshape/matmul
style and concludes the Rust implementation is structurally independent
(named `Index`/`IndexLike` objects and generic `TensorLike::contract`
helpers vs. Python's raw NumPy reshape-and-matmul sequences), finding no
`LICENSE-RISK`. This workstream defers to that body-level check rather than
duplicating it — flagging here only because the *doc-comment's own choice
of words* ("line-by-line cross-check") is a WS-integration file
(`contraction.rs`'s module doc), and a reader who trusted the phrase at
face value without checking WS-chain's independent-reimplementation finding
could reasonably conclude a `LICENSE-RISK` exists where WS-chain's
line-by-line comparison shows it does not. No open `LICENSE-RISK` finding
for WS-integration's own files (they contain no reference-Python-derived
numerical logic to compare); this is a citation-wording note pointing to
WS-chain's finding, not a second independent verdict.

##### `SrcOptions::validate()` — DERIVED-VERIFIED

(I3: this subsection was missing for a table row already graded
`DERIVED-VERIFIED`; the derivation itself is unchanged from the provenance
table's Notes cell, only promoted to its own subsection here per the
sibling-workstream convention.)

`SrcOptions::validate(&self, output_max_bond_dim: Option<usize>)`
(`contraction.rs:1599-1649`) is API-level input hygiene for the public
`SrcOptions` builder — no paper equation or Python function corresponds to
it (the reference Python takes its rank/tolerance parameters as plain
function arguments with no equivalent up-front validation pass), so it is
necessarily `DERIVED-VERIFIED` rather than `SOURCED-*`.

**What it validates, and why each check is correct:**
- `atol` finite and non-negative (1600-1602): a negative or non-finite
  absolute tolerance can never be satisfied by any real residual norm, so
  rejecting it up front avoids a downstream adaptive loop that could never
  terminate on its stopping test.
- `min_rank >= 1` and `rank_increment >= 1` (1603-1608): both values are
  loop-step sizes in the adaptive sketch-growth loop in `src_chain.rs`; a
  zero value would either start the sketch with no columns or make the
  `while` loop that grows it non-progressing (infinite loop).
- Fixed mode (`rtol.is_none()`, 1611-1621): `atol`/`max_rank` are rejected
  as set (they only mean something relative to a stopping *test*, which
  fixed mode has none of, by construction), and `output_max_bond_dim` is
  required — fixed-rank SRC has no stopping test at all, so the target rank
  can only come from the caller-supplied cap.
- Adaptive mode (`Some(rtol)`, 1622-1646): `rtol` must be finite and
  non-negative for the same reason as `atol` above; a finite `max_rank` is
  required (from `self.max_rank` or the ambient `output_max_bond_dim`)
  because the adaptive column-growth loop in `src_chain.rs` needs a hard
  upper bound to guarantee termination even if the tolerance is never met;
  `min_rank <= max_rank` is required because the loop's own starting
  precondition (start at `min_rank`, grow toward `max_rank`) is otherwise
  unsatisfiable; `max_rank > output_max_bond_dim` is only permitted when
  `final_svd` is enabled, because only the optional final-SVD truncation
  sweep can bring an over-wide adaptive sketch back down to the caller's
  requested output rank — without it, exceeding the output cap would leave
  a result wider than what the caller asked for.

Every branch traces to a real downstream invariant in `src_chain.rs`'s
adaptive/fixed loops rather than guarding a state the type system already
rules out (see "No unnecessary defensive code found," below, for the
broader reachability check). The derivation holds up under re-derivation:
`DERIVED-VERIFIED`.

#### No invented APIs found in WS-integration's files

Explicitly checked (not just assumed) every function/type call in the
diff hunks above against its actual definition: `FactorizeResult::new`
(real, matching signature, `tensor4all-core/src/tensor_like.rs`),
`chain_order` (real, `contraction.rs`, semantics traced above),
`src_tree::contract`/`src_chain::contract` (real, correct signatures),
`SvdTruncationPolicy`/`validate_svd_truncation_options` (pre-existing,
unchanged call site). No grep for a plausible-sounding but non-existent
function name in these files turned up a hit that wasn't backed by a real
definition.

#### No unnecessary defensive code found for structurally-impossible cases

Reviewed `SrcOptions::validate()` line by line (see table): every branch
guards a state that is genuinely reachable through the public builder API
(e.g. `atol != 0.0` while `rtol.is_none()` is reachable because `atol` and
`rtol` are independent public fields with independent setters) — none of
the checks are dead guards against states the type system already rules
out.


---

## WS-tests: test coverage and tolerance integrity

*Source: [`docs/plans/audit-workstreams/ws-tests.md`](audit-workstreams/ws-tests.md), reproduced in full.*

### WS-tests — test coverage and tolerance integrity

**Scope:** every test file touched in the `feature/treetn-src` diff vs.
`origin/main`, enumerated with

```bash
git diff --stat origin/main -- crates/ | grep -i test
```

which returns exactly 12 files (confirmed against the unfiltered
`git diff --name-only origin/main` file list — no test file is missed by
the `test` keyword filter):

1. `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs` (+472/-4)
2. `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs` (+43)
3. `crates/tensor4all-treetn/src/treetn/partial_contraction/tests/mod.rs` (+43)
4. `crates/tensor4all-core/src/tensor_like/tests/mod.rs` (+192)
5. `crates/tensor4all-tensorbackend/src/backend/tests/mod.rs` (+69)
6. `crates/tensor4all-capi/src/types/tests/mod.rs` (+1)
7. `crates/tensor4all-core/src/defaults/contract/tests/mod.rs` (+5)
8. `crates/tensor4all-itensorlike/src/contract/tests/mod.rs` (+30)
9. `crates/tensor4all-itensorlike/src/options/tests/mod.rs` (+12)
10. `crates/tensor4all-treetn/src/algorithm/tests/mod.rs` (+2)
11. `crates/tensor4all-itensorlike/tests/tensortrain_inner.rs` (+6/-5, see
    §0 — **not SRC branch work**)
12. `crates/tensor4all-quanticstci/src/quantics_tci/tests/mod.rs` (+3/-4,
    see §0 — **not SRC branch work**)

Verdict taxonomy per
[`docs/plans/2026-08-28-src-provenance-audit.md`](2026-08-28-src-provenance-audit.md).
Tier-2 map consulted: the "Test matrix" section of
`docs/plans/2026-08-26-treetn-src-contraction-plan.md` (lines 553-587),
used only as a checklist of what *should* exist per the plan, never as
ground truth for whether the code is correct.

**Out of formal scope but noted where relevant:** inline `#[cfg(test)] mod
tests { ... }` blocks embedded directly in non-test-named source files
(`src_probe.rs:810-1171`, `incremental_qr.rs`'s tail, `idx_tensor.rs`'s
tail) are not matched by the Step-1 filename filter and belong to
WS-tree-probe / WS-backend / WS-core's file lists, not this workstream's.
Where one of those inline blocks is the *only* test bearing on a Test-matrix
category owned by this table (the fused-`d²`-probe structural-regression
category), it is discussed in §6 for completeness, clearly marked as
outside this workstream's file ownership.

#### §0. Methodological finding: two of the twelve files are not SRC work at all

Before building the provenance table, `git log <range> -- <file>` was run
per file against both `72de8fb..HEAD` (commits unique to this branch) and
`72de8fb..fd61f08` (commits unique to `origin/main`'s tip, where `72de8fb`
is `git merge-base origin/main HEAD`):

```
tensortrain_inner.rs:              branch-only commits: (none)
                                    main-only commits:   fd61f08 "Fix small tensor-train correctness issues (#693)"
quantics_tci/tests/mod.rs:         branch-only commits: (none)
                                    main-only commits:   fd61f08 "Fix small tensor-train correctness issues (#693)"
```

Every other file in the list has a genuine branch-only commit (`cd3724b
feat(treetn): add successive randomized compression`, `9e018d4 feat(treetn):
align SRC with paper and reference`, or `4e56730 refactor(treetn): reset SRC
implementation boundary`).

`origin/main` is **not** an ancestor-only relationship here — `git
merge-base origin/main HEAD` is `72de8fb`, an older common ancestor, not
`origin/main`'s own tip (`fd61f08`). The two branches diverged at `72de8fb`;
`origin/main` then picked up an unrelated bugfix (`fd61f08`, PR #693) that
this `feature/treetn-src` branch has not merged/rebased onto. Direct
confirmation via `git show <rev>:<path>`:

- `git show 72de8fb:.../tensortrain_inner.rs` (common ancestor) already has
  `TensorTrain::new(vec![]).inner(...) == 0.0` for the empty-TT case.
- `git show origin/main:.../tensortrain.rs` has `is_empty() =>
  AnyScalar::new_real(1.0)` (and `norm_squared_fast_path` returns `Some(1.0)`)
  — `fd61f08`'s fix, restoring the mathematical convention documented at
  `tensortrain.rs:2224-2226`, `scalar_one() { // Empty tensor train
  represents scalar 1; Self::new(vec![]) }`, i.e. `TensorTrain::new(vec![])`
  and `scalar_one()` are the *same object* and the empty product should be 1.
- `git show HEAD:.../tensortrain.rs` (this branch) still has the pre-fix
  `0.0` for both functions, and `tensortrain_inner.rs` on this branch still
  asserts `0.0` — because this branch forked before `fd61f08` landed and has
  not absorbed it.

**This is not an SRC-authored regression or a hallucinated test edit.** It
is a stale-branch/rebase-lag artifact: `git diff origin/main` picks it up
only because the diff base has moved. Both files get `N/A — rebase-lag
artifact, not attributable to feature/treetn-src` in the table below rather
than a provenance verdict. **Process flag:** the plan's own "Verification
commands" section (`2026-08-26-treetn-src-contraction-plan.md:686-688`)
anticipates exactly this scenario ("fetch `origin` and verify the
implementation branch contains the current `origin/main`... re-run affected
checks after any update from main") — this branch currently fails that
precondition for at least this one commit. **This does not, however, create
a merge-time regression risk**: `git diff --stat 72de8fb..HEAD --
crates/tensor4all-itensorlike/src/tensortrain.rs
crates/tensor4all-itensorlike/tests/tensortrain_inner.rs
crates/tensor4all-quanticstci/src/quantics_tci/tests/mod.rs` is **empty** —
`feature/treetn-src` has made zero changes to any of the three files
`#693`/`fd61f08` touched, relative to the merge base `72de8fb`. There is
nothing on this branch for a merge or rebase to overwrite in those files;
pulling `origin/main` in (or merging this branch onto `origin/main`) simply
carries `#693`'s fix along unmodified — no code path reintroduces the `0.0`
bug. The accurate framing is narrower than "would regress on merge": **the
branch's own SRC-related test checks (and its `.inner()`/`scalar_one()`
behavior generally) were run against a stale, pre-`#693` base**, so results
derived from running this branch's own test suite as-is should not be
treated as final until it syncs with `origin/main` — but merging or rebasing
it carries no risk of reintroducing the `#693` bug. This should be resolved
by rebase/merge before the branch ships, independent of anything else this
audit found.

#### §1. Provenance table

| # | File | Test-matrix category covered | Present? | Real (Tier-1-comparable) or superficial? | Verdict |
|---|---|---|---|---|---|
| 1 | `contraction/tests/mod.rs` | Correctness: f64, single/two/longer chain, Y-tree/branched-tree, MPO-MPS, scalar sites, exact recovery at full probe cap | `src_fixed_matches_exact_contraction_when_probe_cap_is_full`, `src_fixed_handles_scalar_sites_in_a_chain`, `src_dispatch_preserves_public_contract`, `src_fixed_traverses_a_branched_tree_without_dense_fallback`, `src_fixed_matches_naive_on_a_branched_tree_when_probe_cap_is_full`, `src_preserves_a_scalar_leaf_on_a_branched_tree`, `src_preserves_scalar_only_subtrees_with_dimension_one_bridges` | Yes | **Real for 5 of 7.** `src_fixed_matches_exact_contraction_when_probe_cap_is_full`, `src_fixed_handles_scalar_sites_in_a_chain`, `src_dispatch_preserves_public_contract`, `src_fixed_matches_naive_on_a_branched_tree_when_probe_cap_is_full`, and `src_preserves_a_scalar_leaf_on_a_branched_tree` compare SRC's `to_dense()` output against `tn_a.contract_naive(&tn_b)` (a Tier-1-legitimate dense oracle, independent code path) via `sub().maxabs()`/`.distance()`. The other 2 — `src_fixed_traverses_a_branched_tree_without_dense_fallback` and `src_preserves_scalar_only_subtrees_with_dimension_one_bridges` — assert **only structural properties** (`node_count()`/`edge_count()`, a per-edge bond-dim upper bound, `validate_ortho_consistency()`); neither performs any numerical comparison against an oracle. | `DERIVED-VERIFIED` for 5/7 (end-to-end correctness, methodology sound); the other 2 are structural-only and make no numerical claim to verify |
| 2 | `contraction/tests/mod.rs` | Correctness: Complex64 | `src_complex_chain_matches_naive_when_probe_cap_is_full` | Partial | Real (same dense-oracle methodology) but **only for the simplest topology** (2-node chain, fixed rank). No Complex64 SRC test for branched/comb topology, adaptive mode, or MPO-MPO factorized probe. | `MISSING-VS-SOURCE` (partial — see §5a) |
| 3 | `contraction/tests/mod.rs` | Correctness: adaptive rank, rank-cap behavior | `src_adaptive_contracts_and_honors_rank_cap`, `src_adaptive_uses_the_minimum_rank_when_the_estimate_is_already_small`, `src_adaptive_contracts_a_branched_tree_with_a_rank_cap` | Yes | Real, but weaker than the fixed-rank tests: these check `bond_index(edge).dim() <= cap` / `== min_rank` and `validate_ortho_consistency()`, **not** a dense-oracle residual bound. No adaptive-mode test compares against `contract_naive`. | `MISSING-VS-SOURCE` (a numerical dense-oracle check for adaptive-mode correctness is present nowhere in the diff; the structural checks here don't substitute for it) |
| 4 | `contraction/tests/mod.rs` | Correctness: options/builder round-trips (`SrcOptions`, `ContractionOptions::src()`) | `test_src_options_cover_fixed_and_adaptive_modes`, `test_src_options_reject_invalid_adaptive_parameters` | Yes | Real for what it claims (field/validation round-trips) — not a numerical-correctness test, doesn't claim to be one. | `DERIVED-VERIFIED` (trivial plumbing) |
| 5 | `contraction/tests/mod.rs` | Control flow/errors: "rank cap reached with and without satisfying tolerance" | `src_adaptive_contracts_and_honors_rank_cap` + `src_adaptive_uses_the_minimum_rank_when_the_estimate_is_already_small` | Yes | Partial — see §5i. `src_adaptive_uses_the_minimum_rank_when_the_estimate_is_already_small` genuinely covers the "tolerance satisfied before the cap" branch. `src_adaptive_contracts_and_honors_rank_cap` sets `.with_max_bond_dim(4)` and `SrcOptions::adaptive(1.0e-8, 4)` on `make_three_node_chain_pair()`, whose bond dimensions are themselves 4 (per-node dim-2 physical/shared legs) — i.e. the cap is set exactly to the fixture's own exact cut rank, and the assertion only checks `dim() <= 4`, never `== 4`. That test cannot distinguish "the cap was actually reached and enforced" from "the cap was never binding because the exact rank never needed it." The "cap reached, tolerance not yet satisfied" branch (cap strictly less than exact rank, forcing a truncated/non-exact result) is untested anywhere in the diff. | `DERIVED-VERIFIED` for the tolerance-satisfied-early branch; `MISSING-VS-SOURCE` for the cap-actually-binding branch |
| 6 | `contraction/tests/mod.rs` | Control flow/errors: incompatible topologies, empty networks, unsupported storage, adaptive convergence before/after an increment (SRC-specific) | — | **Absent** | `src_chain.rs` has explicit `same_topology`/`chain.is_empty()` checks (verified by reading the source), but no test in this file exercises them for the `Src` method — only pre-existing `test_contract_naive_topology_mismatch`/`test_contract_zipup_topology_mismatch`/`test_contract_to_tensor_empty_error` exist, and none dispatch through `ContractionMethod::Src`. | `MISSING-VS-SOURCE` |
| 7 | `contraction/tests/mod.rs` | Reproducibility: same seed → identical output; adaptive expansion preserves first *p* columns; different seeds meet the same residual gate | — | **Absent** | Grepped the file for `seed` outside of `.with_seed(...)` call sites used purely to make a single run deterministic; no test constructs two runs and compares them. | `MISSING-VS-SOURCE` |
| 8 | `contraction/tests/mod.rs` | The plan's "chain reduction gate" (four named identities: `E[parent→child,k]`, postorder direction, `P[v]`, root-completion) | — | **Absent as specified** | Grepped this file (and the whole `contraction/` tree) for `chain reduction`, `chain_reduction`, `forward environment`, `paper equation`: zero hits, independently confirming WS-integration's finding on the same question (`ws-integration.md` §"2. The mandatory chain reduction gate test"). What exists in its place is the end-to-end dense-oracle tests in row 1, run through the real public-dispatch path (traced by WS-integration to confirm they hit `src_chain.rs`, not a tree fallback) — a real but different, non-vacuous regression gate, not the plan's literal identity-level gate. See §5b for this workstream's independent framing (the same conclusion, reached from the test-file side). | `MISSING-VS-SOURCE` relative to the Tier-2 plan text; the substitute end-to-end gate is `DERIVED-VERIFIED` |
| 9 | `contraction/tests/mod.rs` | Canonical/isometric edge invariants | `validate_ortho_consistency()` called after nearly every SRC test (8+ call sites) | Present, but **superficial relative to its implied claim** | `validate_ortho_consistency` (defined in `contraction.rs:1194-1316`, a pre-existing, unmodified helper) checks only that the `ortho_towards`/`canonical_region` *bookkeeping metadata* is internally self-consistent (right edges present, right recorded direction, connected region) — it never computes or checks `Qᴴ Q ≈ I` or any other numerical property of the actual tensor data. No test anywhere in the diff (this file, or the inline `#[cfg(test)]` blocks in `src_chain.rs`/`src_probe.rs`/`src_tree.rs`) performs a genuine numerical isometry check. See §5c — flagged as a hallucination-signature pattern. | `SUSPECT-UNVERIFIED` (isometry claim specifically — see §5c) |
| 10 | `contraction/tests/mod.rs` | Structural regression: no fused-`d²` MPO-MPO probe in production path | — (none in this file) | **Absent in this file**; a related but distinct test exists in `src_probe.rs` (out of file scope, see §6) | This file has no MPO-MPO-specific test; the two `contract`-level tests closest to MPO-MPO (`src_fixed_matches_exact_contraction_when_probe_cap_is_full` etc.) use MPO-MPS-shaped fixtures (single output leg per tensor), not MPO-MPO (two output legs per tensor, testing the factorized-probe path specifically). | `MISSING-VS-SOURCE` (in this file); partially covered elsewhere, see §6 |
| 11 | `operator/apply/tests/mod.rs` | Correctness: SRC via `apply_linear_operator` | `apply_linear_operator_src_preserves_a_two_site_identity` | Yes | Real — compares against `state.to_dense()` for an identity operator (the correct oracle value is exactly the unmodified state, an independently derivable ground truth, not merely "doesn't panic"). One test only; no branched-tree, adaptive-mode, Complex64, or non-identity-operator SRC-via-apply test. | `DERIVED-VERIFIED`, narrow coverage — see §5a |
| 12 | `partial_contraction/tests/mod.rs` | Correctness: SRC through the partial-contraction/directed-tree-path entry point | `partial_contract_src_uses_the_same_directed_tree_path` | Yes | Real — compares against `ContractionOptions::new(ContractionMethod::Naive)` (dense oracle) via `.distance()`, not a self-referential check. One test only; no error-path or Complex64 coverage for this entry point. | `DERIVED-VERIFIED`, narrow coverage |
| 13 | `tensor_like/tests/mod.rs` | `TensorConstructionLike`/`TensorFactorizationLike` additions: batch/stack/concatenate constructors, incremental probe-column factorization | `tensor_construction_supports_column_major_dense_payloads`, `tensor_construction_supports_stacking_a_batch_axis`, `tensor_construction_concatenates_existing_batch_blocks`, `tensor_factorization_supports_incremental_probe_prefixes`, `tensor_factorization_preserves_multi_axis_probe_row_order` | Yes | The two `tensor_factorization_*` tests are **real, non-self-referential** checks: expected values are the literal input column data written directly in the test body (an independent oracle — the reconstructed `left.contract_pair(right)` must reproduce the exact original columns), not obtained by running the code once and hard-coding its output. The `tensor_construction_supports_column_major_dense_payloads` test is **weaker than its name claims** — see §5g. | `SUSPECT-UNVERIFIED` (isolated to the column-major-naming claim — 4/5 tests in this row are `DERIVED-VERIFIED`; the 5th, `tensor_construction_supports_column_major_dense_payloads`, does not verify the specific convention its name asserts, see §5g) |
| 14 | `backend/tests/mod.rs` | Adaptive-rank estimator (`src_error_estimate`) correctness, dtype coverage, control flow (singular/non-square R) | `src_error_estimate_matches_real_upper_triangular_oracle`, `src_error_estimate_uses_conjugate_adjoint_for_complex_r`, `src_error_estimate_rejects_singular_and_non_square_r`, `src_error_estimate_supports_single_precision_scalars` | Yes | **Real, and a positive counter-example to worry about, with one exception noted.** The first two tests hand-derive the expected `error`/`norm` values in the test body via explicit closed-form 2×2-triangular-inverse algebra (`g00 = 1/r00`, `g11 = 1/r11(.conj())`, `g10 = -(r01(.conj()) * g00 * g11)`, then column norms), not obtained by calling `src_error_estimate` once and hard-coding the result — the opposite of a self-fulfilling test. `src_error_estimate_supports_single_precision_scalars` is **not uniformly this strong**: its f32 half repeats the same hand-derived closed-form check (`(estimate32.error - 6.3_f64.sqrt()).abs() < 1.0e-5`), but its Complex32 half only asserts `estimatec32.error.is_finite()` / `.norm.is_finite()` — a pure no-panic/no-NaN smoke check with no numerical oracle at all. Whether the derived *formula itself* matches paper Appendix C is WS-backend's question, not re-litigated here. | `DERIVED-VERIFIED` (test methodology) for the real-, Complex64-, and f32-half tests; the Complex32 half of `src_error_estimate_supports_single_precision_scalars` is `SUSPECT-UNVERIFIED` (smoke-only, no derived expected value) — underlying-formula correctness is WS-backend's finding |
| 15 | `capi/src/types/tests/mod.rs` | FFI round-trip for `ContractionMethod::Src` | `test_contract_method_roundtrip` (list extended by one variant) | Yes | Real for what it is (enum round-trip through the C ABI), trivial. | `DERIVED-VERIFIED` (trivial plumbing) |
| 16 | `core/src/defaults/contract/tests/mod.rs` | `TensorContractionLike::contract_retaining_indices` trait-level entry point | One assertion block appended to an existing test, checking the trait method against the same expected dense values already computed in that test | Yes | Real — reuses the same pre-computed `expected` vector the pre-existing (non-trait) call already validated, so the trait wrapper is checked against the same independently-reasoned expected values, not against its own output. | `DERIVED-VERIFIED` |
| 17 | `itensorlike/src/contract/tests/mod.rs` | Correctness: SRC through `itensorlike`'s `TensorTrain`-level `contract()` | `test_contract_src_two_sites` | Yes | Real — uses the file's existing `assert_matches_naive` helper (dense-oracle comparison, same pattern as the file's pre-existing zipup/fit tests), not a bespoke looser check. | `DERIVED-VERIFIED`, single-topology coverage only |
| 18 | `itensorlike/src/options/tests/mod.rs` | `ContractOptions::src()`/`SrcOptions` plumbing at the itensorlike layer | `test_contract_options_methods` (extended), `test_contract_options_src_controls` | Yes | Real for what it claims (builder/accessor round-trip), trivial. | `DERIVED-VERIFIED` (trivial plumbing) |
| 19 | `treetn/src/algorithm/tests/mod.rs` | `ContractionAlgorithm::Src` enum round-trip/name | `test_contraction_algorithm_roundtrip`, `test_contraction_algorithm_name` (both extended by one variant) | Yes | Real, trivial. | `DERIVED-VERIFIED` (trivial plumbing) |
| 20 | `itensorlike/tests/tensortrain_inner.rs` | n/a | n/a | n/a | **Not SRC branch work** — see §0. | `N/A — rebase-lag artifact` |
| 21 | `quanticstci/src/quantics_tci/tests/mod.rs` | n/a | n/a | n/a | **Not SRC branch work** — see §0. | `N/A — rebase-lag artifact` |

#### §2. Tolerance diff against `origin/main`

Checked every one of the 10 genuinely-SRC-touched test files for **removed
or modified** tolerance lines (`git diff origin/main -- <file> | grep -E
"^-.*([0-9]e-[0-9]|rtol|atol|epsilon|< 1e|< 1\.0e)"`): **zero hits across
all 10 files.** No pre-existing test's numeric tolerance was edited,
loosened, or removed anywhere in this diff. Strictly, this means the plan's
explicit rule ("Do not relax existing tolerances without explicit user
approval") is **not violated** by the letter — nothing pre-existing was
touched.

However, per the audit brief's explicit instruction to compare new
tolerances against same-file precedent even where nothing existing was
edited, one directly-checkable pattern is worth flagging in
`contraction/tests/mod.rs` (see §5d for the full write-up and for why this
does **not** generalize to the other two files below): new SRC tests
claiming "exact recovery when the probe cap is full" — the same correctness
claim the pre-existing zip-up/naive tests in the same file already make —
consistently use a bound one order of magnitude looser than the same-file
precedent:

| File | Pre-existing "exact" tolerance (same category) | New SRC "exact/full-probe-cap" tolerance |
|---|---|---|
| `contraction/tests/mod.rs` | `1e-9` (`zipup_chain_matches_naive_without_truncation`, `zipup_complex_chain_matches_naive_without_truncation`, ×5 occurrences total, `origin/main` lines 332, 468, 527, 742, 758) | `1.0e-8` (all 6 `src_*_matches_*`/`src_dispatch_*`/`src_complex_*` full-probe-cap tests) — one order of magnitude looser, not two |
| `operator/apply/tests/mod.rs` | `1e-10` (`assert_identity_application` helper, the file's actual dominant/canonical precedent for this claim, reused by ≥4 same-form identity-application tests) | `1.0e-10` (`apply_linear_operator_src_preserves_a_two_site_identity`) — **matches the file's canonical precedent, no discrepancy here** |
| `partial_contraction/tests/mod.rs` | `1.0e-10` (`test_partial_contract_fit_inserts_dummy_links_...`, `fit`-path) | `1.0e-10` (`partial_contract_src_uses_the_same_directed_tree_path`) — **matches**, no discrepancy here |

An earlier draft of this table cited `1.0e-12` as the `operator/apply/tests/mod.rs`
precedent (from `apply_linear_operator_to_numbered_tags_binds_state_indices_in_tag_order`,
a *different*-shaped assertion than the identity-application family) and
generalized from it to "every new SRC test." That citation was not the
file's dominant precedent for the identity-application claim the new SRC
test actually makes; `assert_identity_application` (used by the SRC test's
own peers) is, and the SRC test matches it exactly. See §5d.

#### §3. Tests that are real, not superficial (positive findings)

To avoid a one-sided report: the dense-oracle-comparison pattern
(`contract_naive`/`ContractionMethod::Naive`/`assert_matches_naive` +
`.sub().maxabs()`/`.distance()`) is used consistently and correctly across
the numerical-comparison tests in rows 1, 2, 11, 12, 17 of §1 — 5 of row 1's
7 tests (the other 2 are structural-only, see row 1), and all of rows 2, 11,
12, 17 — this is genuinely Tier-1-comparable methodology (an independent,
pre-existing exact contraction path as oracle), not a rubber-stamp "doesn't
panic" check. The backend estimator tests (row 14) go further and hand-derive
the expected numbers algebraically in the test body rather than
round-tripping through the function under test. These are the audit's clean
results.

#### §4. Cross-workstream note

WS-integration's file list (`contraction.rs`, `apply.rs`, `fit.rs`,
`swap.rs`, `algorithm.rs`, `options.rs`, `contract.rs`, capi files) overlaps
this workstream's on the "chain reduction gate" question (§1 row 8):
WS-integration reached the same conclusion — the plan's literal gate does
not exist, and a real end-to-end dense-oracle test suite stands in its
place — via dispatch-routing analysis (confirming which tests actually
exercise `src_chain.rs` vs. the tree fallback). This workstream reached the
same conclusion independently via grepping the test files themselves and
reading `validate_ortho_consistency`'s implementation. The two should be
merged into one entry in the Task-7 synthesis rather than double-counted.

#### §5. Detailed derivations and flagged findings

##### 5a. Dtype/topology/product-type coverage is much thinner than the Test-matrix cross-product implies

The plan's Test matrix asks for "f64 and Complex64" × "single node, two-node
chain, longer chain, Y-tree, and comb tree" × "MPO-MPS, MPO-MPO, Hadamard,
scalar result, and multiple output legs per node." Actual coverage in the
diff:

- f64: 2-node chain, 3-node chain, star/Y-tree (`make_star_pair`), 4-node
  branched tree (`make_branched_pair`), scalar-leaf-on-branch, and a
  dimension-1-bridge degenerate case. Reasonable topology breadth for f64.
- Complex64: **exactly one** test (`src_complex_chain_matches_naive_when_probe_cap_is_full`,
  2-node chain only). No Complex64 coverage for branched/star topology,
  adaptive-rank mode, or `apply_linear_operator`/`partial_contract` entry
  points (`operator/apply/tests/mod.rs` has zero `Complex64` usage anywhere
  in the whole file, not just for SRC — a pre-existing gap this branch does
  not close).
- MPO-MPO (two output legs per node, exercising the factorized-probe path
  from Hiroshi's 2026-08-24 comment) has **no dedicated test in any of the
  12 files in this workstream's scope** — the closest thing,
  `probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`,
  lives in `src_probe.rs`'s inline test module, outside this file list (see
  §6).
- "Multiple output legs per node" beyond the single MPO-MPO case above: not
  found.

This is `MISSING-VS-SOURCE` relative to the Tier-2 test matrix — a map of
what should exist, not proof the missing cases are actually broken, but a
real, checkable gap in what was actually written.

##### 5b. The "chain reduction gate," from the test-file side (independent corroboration of WS-integration)

Re-derived independently (not trusting WS-integration's framing, though the
conclusion matches): a test asserting `‖SRC(A,B) − contract_naive(A,B)‖_∞ <
1e-8` when the probe/sketch width at every cut is set to the exact cut rank
(the "probe cap is full" condition every relevant test title states
explicitly) is a real end-to-end correctness signal, because a generic
(continuous-distribution) random sketch at width ≥ exact rank spans the same
column space as the exact factor almost surely, so QR-projecting against it
reproduces the exact answer up to floating-point error — this is not a
vacuous check. But it is not what the plan's Chain reduction gate section
literally specifies: four *named intermediate identities* (`E[parent→child,k]`,
postorder direction, `P[v]`, root completion), each checked independently
against a hand-written reference implementation of the paper's equations.
An end-to-end pass cannot distinguish "every step is individually correct"
from "an even number of compensating errors happen to cancel at the
whole-chain level" — the exact failure mode the plan's four-identity
decomposition exists to rule out. Verdict: `MISSING-VS-SOURCE` against the
plan's literal text; the substitute test suite that exists is
`DERIVED-VERIFIED` as its own (weaker) thing.

##### 5c. `validate_ortho_consistency()` does not check numerical isometry — hallucination-signature: name/usage implies a property the assertion doesn't verify

Read `validate_ortho_consistency`'s full implementation
(`crates/tensor4all-treetn/src/treetn/contraction.rs:1194-1316`, a
pre-existing, unmodified helper). Its logic, verified line by line:

1. If `canonical_region` is empty, require `ortho_towards` is also empty.
2. Otherwise, check `canonical_region` forms a connected subtree
   (`is_connected_subset`).
3. Compute `expected_edges` from `edges_to_canonicalize_to_region` and check
   that every expected bond has an `ortho_towards` entry pointing in the
   expected direction (a `HashMap<Index, NodeName>` lookup/comparison).
4. Check no bond *inside* the canonical region unexpectedly has an
   `ortho_towards` entry.

Every check operates on `self.ortho_towards` / `self.canonical_region` —
metadata fields recording what the code *claims* is canonical and in which
direction — never on the tensor data itself. There is no `Qᴴ Q`, no
contraction-against-conjugate, no norm check, anywhere in this function.
Grepped the whole diff (`crates/`) for `is_unitary`, `is_isometric`,
`isometry`, `check_isometr`, and adjoint-contraction patterns: zero hits.
**No test anywhere in the diff — in any of the 12 files here, or in the
inline `#[cfg(test)]` blocks in `src_chain.rs`/`src_tree.rs`/`src_probe.rs`
— performs a genuine numerical isometry check of an SRC output tensor.**

`validate_ortho_consistency()` is called after nearly every non-trivial SRC
test (`contraction/tests/mod.rs` lines 323, 380, 561, 589, 612, 704, 729,
758, 834, 1050, 1070) as the tests' apparent fulfillment of the Test
matrix's "Canonical/isometric edge invariants" correctness category. A
reader relying on the test names/the presence of these calls (e.g. as
"proof" the SRC output tensors are properly orthonormal/isometric, which is
a real mathematical property Algorithm 1's QR-based construction is
supposed to guarantee) would be misled: **passing `validate_ortho_consistency()`
proves the bookkeeping metadata is self-consistent; it proves nothing about
whether the actual QR factors the code produced are numerically orthonormal.**
A bug that produced non-isometric tensors while still correctly labeling
`ortho_towards` would pass every one of these calls. This matches the
audit's named hallucination-signature pattern #1 ("a test... asserts a
shape [here: bookkeeping consistency] ... when its name/comment claims to
verify a numerical correctness property") — though the fault here is at the
level of what the *Test matrix category* is implicitly credited with by
reusing this pre-existing helper, not a fabricated assertion inside any
individual test. Given the dense-oracle tests in the same functions (§3)
already bound the *whole-output* error, an SRC output that were badly
non-isometric would likely also fail the dense comparison in most cases —
so this is a coverage gap in what the isometry claim specifically verifies,
not necessarily evidence of an actual isometry bug. Flagging as
`SUSPECT-UNVERIFIED` for the isometry claim specifically (not for SRC
correctness overall, which rows 1/2/11/12/17 in §1 do support).

##### 5d. One verified tolerance-gap instance in `contraction/tests/mod.rs`; the cross-file "every new SRC test" generalization does not hold

An earlier draft of this finding cited one precedent
(`apply_linear_operator_to_numbered_tags_binds_state_indices_in_tag_order`'s
`1.0e-12`) in `operator/apply/tests/mod.rs` and generalized from it to
"every new SRC test" being one-to-two orders of magnitude looser than
same-file precedent, across all three files in §2's table. That
generalization does not survive re-checking: the `1.0e-12` citation was not
the file's dominant precedent for the specific claim the new SRC test
makes. The actual dominant, directly-reused precedent for "identity operator
application reproduces the input state" in that file is the shared
`assert_identity_application()` helper (`operator/apply/tests/mod.rs:611`,
`< 1e-10`), also matched independently by same-form tests at lines 911, 926,
and 952. The new SRC test, `apply_linear_operator_src_preserves_a_two_site_identity`,
asserts `< 1.0e-10` — it **matches** the file's canonical precedent exactly.
No finding there. `partial_contraction/tests/mod.rs`'s SRC test likewise
matches its file's precedent (§2). So of the three files in §2's table, two
show no discrepancy at all.

The one instance that does survive independent re-verification is in
`contraction/tests/mod.rs`: `src_complex_chain_matches_naive_when_probe_cap_is_full`
(`< 1e-8`) sits immediately adjacent to `zipup_complex_chain_matches_naive_without_truncation`
(`< 1e-9`), and both are built from the same fixture and the same
`contract_naive` oracle — the cleanest possible same-fixture, same-oracle,
adjacent-test comparison in the diff, and it shows a genuine
one-order-of-magnitude gap. The same file's other 5 new SRC full-probe-cap
tests (all `1.0e-8`) are likewise looser than the file's 5 pre-existing
"exact" assertions (all `1e-9`, `origin/main` lines 332, 468, 527, 742, 758),
though those pairs are not all drawn from an identical fixture the way the
complex-chain pair is, so they corroborate the same direction of the pattern
without being as tightly controlled a comparison individually. In either
framing this is a one-order-of-magnitude gap (`1e-9` to `1e-8`), not the
"one to two orders of magnitude" the earlier draft claimed — the second
order of magnitude came entirely from the now-refuted `operator/apply`
citation.

No pre-existing tolerance was edited (§2), so this does not literally match
`SCOPE-DEVIATION`'s definition ("contradicts an explicit plan decision" via
editing an existing assertion). But the brief explicitly asks this
workstream to flag "suspiciously loose tolerances relative to what similar
origin/main tests already use in the same files," and this one instance is
directly checkable. No comment, commit message, or plan text anywhere in the
diff explains the gap. That said, there is an a-priori, plan-consistent
reason a gap of this size *could* be legitimate rather than anomalous: SRC's
QR-plus-randomized-probe construction is a randomized-sketch method, while
zip-up's truncation is a deterministic SVD path — a randomized-sketch method
could plausibly warrant a systematically looser default tolerance than a
deterministic decomposition on the same fixture, purely from sketch-induced
floating-point noise on top of the QR step. This is an untested-but-plausible
explanation, not a confirmed one: nothing in the diff states or derives it,
and it was not checked here by actually tightening the assertion (that would
require running the test suite with a modified assertion, which conflicts
with this workstream's "read, do not modify" scope). Flagged for the
synthesis pass / WS-chain/WS-backend's numerical audits to resolve with an
actual residual measurement, since they have license to run and inspect the
numerics directly.

##### 5e. Missing reproducibility coverage

Grepped all 12 files for tests constructing two SRC runs and comparing
them: none exist. The plan's three reproducibility categories (same seed →
identical edge ranks/dense output; adaptive expansion preserves the first
*p* columns exactly; different seeds meet the same residual gate without
elementwise equality) are entirely absent from the diff. Every SRC test
that specifies a seed (`.with_seed(...)`) does so only to make its own
single run deterministic, not to compare two runs. `MISSING-VS-SOURCE`
relative to the Tier-2 map.

##### 5f. Missing SRC-specific control-flow/error coverage

`src_chain.rs` contains explicit `anyhow::bail!` calls for incompatible
topologies (`"contract_src: networks have incompatible topologies"`) and
empty chains (`"contract_src: empty chain"`) — read directly in the source.
No test in any of the 12 files constructs a topology-mismatched or
empty-network pair and dispatches it through `ContractionOptions::src()`;
the only topology-mismatch/empty-network tests present
(`test_contract_naive_topology_mismatch`, `test_contract_zipup_topology_mismatch`,
`test_contract_to_tensor_empty_error`) are pre-existing and exercise the
`Naive`/`Zipup` methods, not `Src`. "Unsupported tensor storage" and
"adaptive convergence before vs. after an increment" (a test capturing
state exactly at an adaptive rank-increment boundary) have no SRC-specific
test either. `MISSING-VS-SOURCE` for these four sub-categories; "rank cap
reached with/without satisfying tolerance" and "singular or rank-deficient
sketch matrices" (via `src_error_estimate_rejects_singular_and_non_square_r`,
though only at the backend-estimator unit level, not the full SRC-contraction
level) are adequately covered (§1 rows 5, 14).

##### 5g. `tensor_construction_supports_column_major_dense_payloads` — test name claims more than the assertion checks

`tensor_like.rs:1174-1183`'s doc comment defines `from_dense_any`'s
contract explicitly: *"Construct a tensor from a column-major dense
payload... `data` — Dense values in column-major order."* The test
(`tensor_like/tests/mod.rs`, new) builds a 2×3 tensor from
`[1.0..6.0]` and asserts `tensor.to_vec::<f64>() == vec![1.0..6.0]` — a
full-buffer round-trip equality check. This does not discriminate
column-major from row-major construction: since the dimensions are
non-square (2×3), a genuine test of the column-major *convention*
specifically (as opposed to mere round-trip self-consistency) would need to
query an individual element at a known `(row, col)` position and check it
against the value column-major indexing predicts at that position — e.g.
element `(1, 0)` should be `data[1]` under column-major but `data[3]` under
row-major for a 2×3 tensor. As written, the test would pass identically if
`from_dense_any` and `to_vec` both silently used row-major (or any other
consistent) convention internally, since it only checks that whatever
convention is used round-trips through itself. This matches the audit's
hallucination-signature pattern of a test name asserting a specific
property that the assertion body does not actually discriminate. Distinct
from — and does not call into question — WS-core's independent
re-derivation of the *implementation*'s correctness (`ws-core.md`, `DERIVED-VERIFIED`
for `from_dense_any`'s mixed-radix column-major construction, checked by
reading the code directly, not via this test). Low severity: this is a test
quality gap, not evidence the underlying feature is wrong.

##### 5h. Trivial-plumbing `DERIVED-VERIFIED` rows (§1 rows 4, 15, 16, 18, 19)

Five `DERIVED-VERIFIED` rows in §1 do not need an individual derivation
because they verify field/enum/builder round-trips, not numerical or
algorithmic properties — there is nothing to re-derive beyond reading the
assertion:

- Row 4 (`test_src_options_cover_fixed_and_adaptive_modes`,
  `test_src_options_reject_invalid_adaptive_parameters`): checks that
  `SrcOptions`'s builder methods set/reject the fields they claim to,
  compared directly against the literal values passed in. A round-trip
  through plain struct fields; nothing beyond the assertion to derive.
- Row 15 (`test_contract_method_roundtrip`): the `ContractionMethod::Src`
  C-ABI enum variant round-trips through the FFI boundary and back to the
  same variant. Enum-identity check; no numerics involved.
- Row 16 (`TensorContractionLike::contract_retaining_indices` assertion):
  reuses the *same* pre-computed `expected` dense vector an adjacent,
  pre-existing non-trait call in the same test already validated
  independently — the trait wrapper is checked against that already-derived
  value, not its own output, so there is no separate expected-value
  derivation for this row to add.
- Row 18 (`test_contract_options_methods`, `test_contract_options_src_controls`):
  `ContractOptions::src()`/`SrcOptions` accessor/builder round-trip at the
  `itensorlike` layer, same shape as row 4.
- Row 19 (`test_contraction_algorithm_roundtrip`, `test_contraction_algorithm_name`):
  `ContractionAlgorithm::Src` enum round-trip and display-name check, same
  shape as row 15.

None of these rows make a numerical-correctness claim, so `DERIVED-VERIFIED`
requires no more derivation than confirming the assertion checks the value
it claims to — done by reading each test directly (§1).

##### 5i. Row 5 derivation: `src_adaptive_contracts_and_honors_rank_cap` cannot show the cap actually binds

`src_adaptive_uses_the_minimum_rank_when_the_estimate_is_already_small` sets
`SrcOptions::adaptive(1.0e6, 4).with_min_rank(1).with_rank_increment(1)` —
an effectively-infinite tolerance (`1.0e6`) guarantees the adaptive loop
accepts the minimum rank on its first check, so `dim() == 1` on every edge
genuinely demonstrates the "tolerance satisfied before the cap is reached"
branch. This half is `DERIVED-VERIFIED`.

`src_adaptive_contracts_and_honors_rank_cap` sets
`.with_max_bond_dim(4)` and `SrcOptions::adaptive(1.0e-8, 4).with_min_rank(2).with_rank_increment(2)`
on `make_three_node_chain_pair()`. Read that fixture directly
(`contraction/tests/mod.rs:438-486`): every node has physical/shared legs of
dimension 2, and the internal bonds of each three-node chain are themselves
dimension-2, so the true cut rank at the contraction bond this test measures
is bounded by 4 (2×2) — i.e. the cap (`4`) is set to exactly the fixture's
own maximum possible exact rank, not below it. The assertion is
`bond_index(edge).dim() <= 4` — a `<=`, never checked against `== 4` or
against any independent computation of what the *actual* converged rank is.
A run in which the adaptive loop's tolerance check is satisfied at rank 2
(never approaching the cap at all) would pass this assertion exactly as
well as a run where the cap genuinely truncates the result. The test
therefore cannot distinguish "the cap was reached and enforced" from "the
cap was never binding." The category this row claims to cover — "rank cap
reached with and without satisfying tolerance" — needs a fixture whose true
exact rank exceeds the cap, forcing a demonstrably non-exact, cap-truncated
result; no such fixture/cap combination exists anywhere in the diff. Hence
`MISSING-VS-SOURCE` for that branch specifically (§1 row 5).

##### 5j. Row 11 derivation: `apply_linear_operator_src_preserves_a_two_site_identity`

The operator built by `build_bonded_identity_operator` is the identity map
on the two-site Hilbert space (each physical leg mapped to itself with no
mixing) — reading the helper confirms it constructs delta tensors on the
matching input/output leg pairs, with no other action. Mathematically,
applying an identity operator to any state must reproduce that state exactly
(up to floating point/algorithm noise): `L|ψ⟩ = |ψ⟩` when `L = I`. The test
computes `result.to_dense()` (SRC-contraction output) and compares it
against `state.to_dense()` — the literal input, not a value obtained by
running the code under test — via `.distance() < 1.0e-10`. This is an
independently-derivable ground truth (the identity operator's output is
knowable without running SRC at all), so the comparison is a real,
non-self-referential correctness check, not a shape/no-panic stand-in.
Confirms `DERIVED-VERIFIED` for §1 row 11; the row's stated coverage gap
(single test, no branched-tree/adaptive/Complex64/non-identity-operator
variant) stands as documented in §5a.

##### 5k. Row 12 derivation: `partial_contract_src_uses_the_same_directed_tree_path`

Compares SRC's output through the partial-contraction/directed-tree-path
entry point against `ContractionOptions::new(ContractionMethod::Naive)` on
the same input network via `.distance()` — `Naive` is the same
independent, pre-existing dense-oracle code path used throughout row 1, just
invoked through the partial-contraction entry point rather than the
top-level `contract()` function. Same derivation as row 1's numerical tests:
a generic random sketch at width ≥ exact rank spans the same column space as
the exact factor almost surely (§5b), so the two paths should agree up to
floating-point error when the probe cap is full, which is what this test's
fixture sets up. `DERIVED-VERIFIED`; single-topology coverage only, as
documented in the row.

##### 5l. Row 14 derivation, and its one exception

`src_error_estimate_matches_real_upper_triangular_oracle` and
`src_error_estimate_uses_conjugate_adjoint_for_complex_r` hand-compute the
expected `error`/`norm` from a 2×2 upper-triangular `R` via closed-form
triangular-inverse algebra written directly in the test body (`g00 = 1/r00`,
`g11 = 1/r11(.conj())`, `g10 = -(r01(.conj()) · g00 · g11)`, then explicit
column-norm arithmetic) — an independent derivation, not a round-trip
through `src_error_estimate` itself. `src_error_estimate_supports_single_precision_scalars`'s
f32 half repeats this pattern (`(estimate32.error - 6.3_f64.sqrt()).abs() <
1.0e-5`, `6.3` being the hand-computed expected value for that fixture) — also
`DERIVED-VERIFIED`. Its Complex32 half, however, only asserts
`estimatec32.error.is_finite()` and `estimatec32.norm.is_finite()` — no
derived expected value, no comparison against any oracle, just a
no-panic/no-NaN smoke check (§1 row 14, I4). `src_error_estimate_rejects_singular_and_non_square_r`
is a control-flow test (expects `Err(..)` on singular/non-square input), not
a numerical-correctness derivation, and is not re-litigated here.

#### §6. Supplementary note: fused-`d²`-probe structural regression test (outside this workstream's file scope)

The Test matrix's "structural regression against fused `d²` MPO-MPO probes"
category has no test in any of the 12 files this workstream owns (§1 row
10). The closest match found anywhere in the diff is
`probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`
and `batched_probed_site_pair_keeps_independent_mpo_probes_paired`, both in
`src_probe.rs`'s inline `#[cfg(test)] mod tests` (lines 810-1171 of that
file) — **not** matched by the Step-1 filename filter (`src_probe.rs` has
no `test` in its path) and formally owned by WS-tree-probe. Read for
completeness since it bears directly on a Test-matrix category this
workstream is scoped to judge: both tests hand-compute the expected value
via an explicit triple sum directly against the raw dense tensor data
(`Σ_u A[s,u]·B[u,t]·X[s]·Y[t]`, matching Hiroshi's 2026-08-24 factorized-probe
formula), which is a genuine independent-oracle numerical check — not
self-referential. However, note the category name ("**structural**
regression") promises something these tests do not provide: neither
asserts anything about intermediate tensor *shapes* (e.g. that no
intermediate ever reaches dimension `d²`) — they check the final numerical
value of `probed_site_pair`'s output only. A production code path could in
principle materialize a fused-`d²` intermediate internally and still
produce the numerically correct final answer these tests check; nothing
here would catch that. So: real numerical correctness test for the
factorized-probe formula, but not the shape/structural regression guard the
category name implies. WS-tree-probe should reconcile this in its own
table; noted here only so this workstream's table doesn't wrongly credit
its own file list with a category it does not, in fact, cover.

#### §7. Summary of MISSING/flagged items for Task 7 synthesis

- `MISSING-VS-SOURCE`: Complex64 × branched-topology / adaptive-mode / MPO-MPO
  coverage (§5a); the plan's literal chain-reduction-gate identities (§5b,
  corroborates WS-integration); reproducibility tests (§5e); SRC-specific
  topology-mismatch / empty-network / unsupported-storage / adaptive-boundary
  control-flow tests (§5f); dedicated MPO-MPO structural test in this
  workstream's own file list (§1 row 10, partially mitigated by §6's
  out-of-scope find); a numerical dense-oracle correctness test for
  adaptive-rank contraction (§1 row 3 — the present tests only check
  structural bond-dim bounds); the "rank cap actually binding, tolerance not
  yet satisfied" branch of the rank-cap control-flow category (§1 row 5, §5i
  — the cited test's cap equals the fixture's exact rank, so it can never be
  shown to bind).
- `SUSPECT-UNVERIFIED` (test-quality, not code-correctness): the
  "canonical/isometric edge invariants" category is credited via a
  bookkeeping-only check with no numerical isometry verification anywhere
  in the diff (§5c); the `column_major`-named test that doesn't discriminate
  the convention it names (§1 row 13, §5g); the Complex32 half of
  `src_error_estimate_supports_single_precision_scalars`, which asserts only
  `.is_finite()` with no derived expected value (§1 row 14).
- Flagged, not formally taxonomized: one verified tolerance-gap instance —
  `src_complex_chain_matches_naive_when_probe_cap_is_full` (`1e-8`) vs. its
  adjacent, same-fixture, same-oracle precedent
  `zipup_complex_chain_matches_naive_without_truncation` (`1e-9`), one order
  of magnitude, unexplained in any comment, with the same direction (not as
  tightly matched per-pair) across the rest of `contraction/tests/mod.rs`'s
  new SRC full-probe-cap tests (§2, §5d). A cross-file "every new SRC test
  is systematically looser" version of this claim does not hold:
  `operator/apply/tests/mod.rs`'s and `partial_contraction/tests/mod.rs`'s
  new SRC tests both match their file's actual dominant precedent exactly
  (§2, §5d).
- Process flag, not a code/test provenance finding: `tensortrain_inner.rs`
  and `quantics_tci/tests/mod.rs` are rebase-lag noise from `origin/main`'s
  independent `#693` fix, not SRC work. The branch has zero diff on the
  three files `#693` touched relative to the merge base, so merging or
  rebasing creates **no regression risk** — but the branch's own test
  results were run against a stale, pre-`#693` base and should not be
  treated as final until it syncs with `origin/main` (§0).
- Positive findings, for balance: the dense-oracle-comparison methodology
  used throughout the fixed-rank correctness tests is genuinely
  Tier-1-comparable (§3); the backend estimator tests hand-derive expected
  values algebraically rather than round-tripping through the function
  under test (§1 row 14) — the opposite of a self-fulfilling test.
