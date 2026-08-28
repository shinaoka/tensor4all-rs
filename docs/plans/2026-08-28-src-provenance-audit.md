# SRC Provenance Audit — Design

**Companion to:** [`2026-08-26-treetn-src-contraction-plan.md`](2026-08-26-treetn-src-contraction-plan.md)
(the implementation plan for `ContractionMethod::Src`, issue
[#563](https://github.com/tensor4all/tensor4all-rs/issues/563)).

## Why this audit exists

The SRC implementation on `feature/treetn-src` (commit `9e018d4`, the exact
revision `gw-rs` pins) is suspected of containing AI-hallucinated content: code
that was never derived from, or checked against, any legitimate source. The
strongest concrete symptom is a suspected hand-rolled SVD interface in a
codepath where the primary reference literature never calls for one. A second
symptom is that `gw-rs`'s downstream pipeline is very slow when it uses SRC —
slower than the algorithm's own cost model would predict, which is consistent
with hand-rolled or dense-materializing code standing in for what should be a
thin QR-only hot path.

This document defines a from-scratch provenance audit of every line the
`feature/treetn-src` branch added (6,223 insertions across 37 files, relative
to `origin/main`). The audit produces a report only. No code changes happen in
this pass — see [Deliverable](#deliverable).

## Epistemics: what counts as evidence

### Tier 1 — legitimate ground truth

Only these three sources can close out a finding as sourced or verified:

1. **The paper and its pseudocode.** C. Camaño, E. N. Epperly, J. A. Tropp,
   "Successive randomized compression: A randomized algorithm for the
   compressed MPO-MPS product," arXiv:2504.06475. Local copy:
   `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`
   (LaTeX source, grep-able — cite by section/equation/Algorithm number) and
   `CET26-Successive-Randomized-quantum.pdf` (for figures).
2. **The paper's reference Python implementation.**
   `/root/projects/RandomMPOMPS-reference-20260827/code/tensornetwork/`:
   `MPO.py`, `MPS.py`, `contraction.py`, `incrementalqr.py` +
   `incrementalqr.cpp`, `linalg.py`, `misc.py`, `rounding.py`, `stopping.py`.
   Per the implementation plan's own policy, this repository has no detected
   license: it may validate numerical behavior and parameter conventions, but
   Rust code that reads as a line-by-line translation is a finding in its own
   right (`LICENSE-RISK`, below), not a clean citation.
3. **Hiroshi Shinaoka's (`shinaoka`) comments** on issues
   [#563](https://github.com/tensor4all/tensor4all-rs/issues/563) and
   [#691](https://github.com/tensor4all/tensor4all-rs/issues/691). Full text
   is reproduced in [Appendix A](#appendix-a-hiroshis-comments-verbatim-chronological)
   so no workstream needs live network/`gh` access. **Read the appendix
   sequentially, not as an unordered bag of facts** — see the next
   subsection.

**Handling corrections within Tier 1 itself.** Hiroshi's comment thread
contains at least one explicit self-correction (the 2026-08-27 15:38 comment,
headed "Correction to the scan claim above," revises the cost/parallelism
claims made in the 12:56 comment nine hours earlier), and at least one
partial revision that sharpens rather than retracts an earlier comment (the
19:13 cost-accounting comment on #691 upgrades the role of interface
sketching relative to the 19:07 comment). **Only the most current,
uncorrected statement on a given sub-claim is Tier-1 truth.** If a workstream
finds two comments that conflict on the same sub-claim and neither explicitly
supersedes the other, do not pick one — report it as `SOURCE-AMBIGUOUS` and
let the user resolve it.

### Tier 2 — artifacts under audit, not references

`docs/plans/2026-08-26-treetn-src-contraction-plan.md`,
`docs/PROVENANCE_AND_CITATION_POLICY.md`, and any in-code comment that claims
a derivation are themselves products of the same AI-assisted process under
suspicion. Use them only as a **map** of what the code claims to do and where
to look. Never cite them as the thing a piece of code is checked against.
Every mathematical or scope claim borrowed from them — the Appendix-C
estimator formula, the oversampling rule `p = max(ceil(1.5r), r+10)`, the
"chain reduction gate" definition, the proposed API shape, the non-goals
list, the file plan — must be independently re-traced to Tier 1 *before* it
is used as a pass/fail criterion for code. If a Tier-2 claim cannot be traced
to Tier 1, or contradicts Tier 1, that is a finding
(`PLAN-CLAIM-UNVERIFIED`), and code that faithfully implements an
unverified plan claim is not thereby excused.

## Verdict taxonomy

Apply per code unit (function, impl block, or identifiable algorithm step):

| Verdict | Meaning |
| --- | --- |
| `SOURCED-PAPER(§/eq/Algorithm N)` | Directly matches a specific paper location. |
| `SOURCED-PYTHON(file:line)` | Directly matches specific reference-Python logic (validated, not translated — see `LICENSE-RISK`). |
| `SOURCED-COMMENT(issue#, date)` | Directly specified by a current (non-superseded) Hiroshi comment. |
| `DERIVED-VERIFIED` | No direct source exists (e.g. the tree generalization, which neither the paper nor Hiroshi's comments cover); the audit re-derives the math independently in the report, and it holds. Must include the derivation. |
| `SUSPECT-UNVERIFIED` | No direct source, and either no derivation is shown anywhere or the shown derivation does not hold up under re-derivation. |
| `HANDROLLED-DUPLICATE` | Reimplements something `tensor4all-tensorbackend`/tenferro/`tensor4all-core` already exposes. The implementation plan explicitly forbids "a hand-written QR, SVD, matrix inverse, or LAPACK wrapper in `treetn`." |
| `MISSING-VS-SOURCE` | The paper or Python has a capability this Rust code lacks. |
| `SCOPE-DEVIATION` | Contradicts an explicit plan decision or explicit Hiroshi ask (e.g. the `PrefixCache`-as-trait ask, the MPO-MPO factorized-probe contraction order, anything resembling #691's not-yet-implemented interface sketching leaking into the #563 implementation). |
| `LICENSE-RISK` | Logic close enough to the reference Python to read as a translation rather than an independent, validated implementation. |
| `PLAN-CLAIM-UNVERIFIED` | A Tier-2 claim used as a checklist item does not trace back to Tier 1, or contradicts it. Report this about the *plan*, separately from whether the *code* matches the plan. |
| `SOURCE-AMBIGUOUS` | Two Tier-1 statements conflict on the same sub-claim and neither explicitly supersedes the other. Do not resolve by guessing; report both and flag for the user. |

## Workstreams

Six parallel, independent passes. Each returns a provenance table in the
taxonomy above, covering every non-trivial code unit in its file list —
report only, no fixes.

### WS-chain — the literal single-chain case

Files: `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs` (736
lines).

Check against paper §2.1 / Algorithm 1, the paper's QR-only claim
(reinforced by Hiroshi's 2026-07-29 comment: SRC's core loop never forms an
SVD of any large matrix; SVD appears only in one optional final-truncation
step, on the already-compressed output), and the Python chain path in
`contraction.py`/`MPO.py`/`MPS.py`. This is the most directly falsifiable
workstream — there is a real paper algorithm and a real reference
implementation to check literally every step against. Grep the whole file
for any SVD call and confirm each one is the single permitted final-rounding
site, not the QR-only hot path.

### WS-tree-probe — the from-scratch tree generalization and MPO-MPO probing

Files: `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs` (641
lines), `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs` (1171
lines, the largest file in the diff).

The tree generalization is covered by neither the paper (chain-only) nor
Hiroshi's comments, so it is necessarily `DERIVED-VERIFIED` or
`SUSPECT-UNVERIFIED` — this workstream carries the heaviest derivation
burden. Cross-check probe construction specifically against Hiroshi's
2026-08-24 comment: the exact factorized contraction order
`Ω[s,t,k] = X[s,k]·Y[t,k]`, contracting `conj(X[:,k])` into the first
operand, `conj(Y[:,k])` into the second, then the shared physical index,
then incoming tree messages — and confirm the production path never
constructs a fused `d²`-dimensional physical probe (an explicit
`SCOPE-DEVIATION` if it does). Re-derive the directed sketch-message
construction (postorder/preorder passes, cache keyed by `(from_node,
to_node)`) independently against the paper's forward-environment recursion
generalized to a tree, rather than trusting the plan doc's description of
it.

### WS-backend — numerics and the incremental-QR question

Files: `crates/tensor4all-tensorbackend/src/backend.rs` (+191),
`crates/tensor4all-tensorbackend/src/incremental_qr.rs` (+1005, new file),
`crates/tensor4all-tensorbackend/src/lib.rs`.

Check the adaptive-rank estimator against paper Appendix C directly (not
against the plan's transcription of it). For `incremental_qr.rs`: the
reference Python repository has its own `incrementalqr.py` **and**
`incrementalqr.cpp` — so this file is not automatically hallucinated the way
its mere existence might suggest (the 2026-08-26 plan text calls incremental
QR "a later optimization gate," which makes its presence here worth
explaining either way). Determine which is true: a legitimate, verified port
of real reference logic, a `SUSPECT-UNVERIFIED`/`LICENSE-RISK` translation,
or a `SCOPE-DEVIATION` (built before the plan said it should be). Separately,
grep this whole file and `backend.rs` for any SVD, matrix-inverse, or
hand-rolled linear-algebra routine that duplicates existing
`tensor4all-tensorbackend`/tenferro functionality (`HANDROLLED-DUPLICATE`).

### WS-core — tensor4all-core additions

Files: `crates/tensor4all-core/src/defaults/idx_tensor.rs` (+533),
`crates/tensor4all-core/src/tensor_like.rs` (+510),
`crates/tensor4all-core/src/defaults/factorize.rs` (+46),
`crates/tensor4all-core/src/index_like.rs` (+28),
`crates/tensor4all-core/src/index_ops.rs` (+4),
`crates/tensor4all-core/src/defaults/index.rs` (+4).

The plan gates the `idx_tensor.rs`/`tensor_like.rs` additions behind
profiling ("add a reusable batch/stack constructor... if column
construction or assembly is material"). Determine whether that profiling
exists anywhere in the branch's history or docs, or whether this is
speculative addition. Check for functionality that duplicates something
already present elsewhere in `tensor4all-core`. **Treat `factorize.rs` as
high priority**: a file named for factorization is the most likely place
for a hand-rolled SVD/QR interface to hide, which is the audit's original
motivating suspicion — check every function in it against Tier 1 (does the
paper/Python call for this factorization at all, and does Hiroshi's
QR-only claim rule out an SVD here) before accepting it as legitimate.

### WS-integration — dispatch, public API, and cross-cutting glue

Files: `crates/tensor4all-treetn/src/treetn/contraction.rs` (+369),
`crates/tensor4all-treetn/src/operator/apply.rs` (+44),
`crates/tensor4all-treetn/src/treetn/fit.rs`,
`crates/tensor4all-treetn/src/treetn/swap.rs`,
`crates/tensor4all-treetn/src/algorithm.rs`,
`crates/tensor4all-itensorlike/src/{options.rs,contract.rs}`,
`crates/tensor4all-capi/src/{treetn.rs,types.rs}`.

Verify the mandatory "chain reduction gate" test (the plan's own stated
precondition for enabling public dispatch — itself only usable here as a
map, re-verify it's a real, faithful implementation of the paper's chain
identities, not just that a test with that name exists and passes) is
present and actually checks what it claims. Check whether the `PrefixCache`
trait Hiroshi asked for (2026-08-27 comments: "put the cache behind a small
trait... instead of hard-coding a Vec built in a forward loop") was honored,
ignored, or over-built into something more elaborate than "flat list first."
Check for any code resembling #691's not-yet-implemented interface-sketching
proposal (`SCOPE-DEVIATION` if found — #691 is a future proposal, not part
of #563).

Also sweep every remaining plumbing file touched by the diff not already
covered by another workstream — re-export/registration changes such as
`crates/tensor4all-treetn/src/{lib.rs,prelude.rs,README.md}`,
`crates/tensor4all-itensorlike/src/{lib.rs,prelude.rs}`,
`crates/tensor4all-capi/include/tensor4all_capi.h`. These are low-risk but
must still get at least a one-line verdict each so the audit can state it
covered the full 37-file diff, not a subset.

### WS-tests — test coverage and tolerance integrity

Files: `crates/tensor4all-treetn/src/treetn/contraction/tests/mod.rs`
(+472), `crates/tensor4all-treetn/src/operator/apply/tests/mod.rs` (+43),
`crates/tensor4all-treetn/src/treetn/partial_contraction/tests/mod.rs`
(+43), `crates/tensor4all-core/src/tensor_like/tests/mod.rs` (+192),
`crates/tensor4all-tensorbackend/src/backend/tests/mod.rs` (+69), and any
other test file touched in the diff.

Compare against the 2026-08-26 plan's "Test matrix" section item by item —
but as a map of what *should* exist per Tier 2, not as ground truth for what
*must* exist; independently judge whether each named test category is a
real correctness check against Tier 1 or a superficial one. Flag any test
tolerance that appears loosened relative to `origin/main`'s existing tests
(the plan explicitly forbids this without explicit user approval).

## Synthesis pass

After all six workstreams return, merge into one report:

1. **Executive summary**, findings ranked by severity: confirmed-wrong math
   → `HANDROLLED-DUPLICATE` items plausibly explaining the downstream
   slowness → `MISSING-VS-SOURCE` → `SUSPECT-UNVERIFIED` derivations →
   `PLAN-CLAIM-UNVERIFIED` / `SOURCE-AMBIGUOUS` → citation-only gaps.
2. For every `HANDROLLED-DUPLICATE` or confirmed-wrong finding, note whether
   it plausibly explains the "very slow in the downstream pipeline" symptom
   — cross-reference against the implementation plan's own "Performance
   acceptance gates" section (no full dense materialization, no fused `d²`
   probe, no SVD outside final truncation, no recomputed cached environment
   columns), again treating that section as a map to check, not as
   authoritative on its own.
3. Full per-workstream provenance tables.
4. Resolve cross-workstream overlaps (e.g. WS-backend and WS-tree-probe may
   both discuss the same call site) into one entry.

## Deliverable

A single report, `docs/plans/2026-08-28-src-provenance-audit-report.md`, in
this repository. This design document defines the audit; the report is its
output and does not exist yet. No source code changes happen in this pass —
fix planning is a separate follow-up scoped after the report is reviewed.

## Out of scope for this pass

- No code changes to `feature/treetn-src`.
- No gw-rs profiling (correctness/provenance is being verified before any
  performance work, since fixing hallucinated or hand-rolled code is likely
  to change the performance picture on its own).
- No new branch — the report is a new file, so this work happens in place on
  `feature/treetn-src`.

## Appendix A: Hiroshi's comments, verbatim, chronological

### Issue #563 — opening post (author: the issue creator, for context)

> ## Summary
>
> Implement the successive randomized compression (SRC) algorithm of Camaño,
> Epperly, and Tropp as an additional contraction method, alongside the
> existing `contract_naive`, `contract_zipup`, and `contract_fit` in
> `crates/tensor4all-simplett/src/mpo/`.
>
> Reference: C. Camaño, E. N. Epperly, and J. A. Tropp, *Successive
> randomized compression: A randomized algorithm for the compressed MPO-MPS
> product*, Quantum 10, 2022 (2026), arXiv:2504.06475.
>
> ## Why
>
> SRC computes the compressed MPO-MPS (and hence MPO-MPO) product with a
> sequence of randomized QB approximations, sketching the full environment
> instead of truncating locally. Compared to what we currently have:
>
> | Method | Cost (simplified, Table 1 of the paper) | Notes |
> |---|---|---|
> | naive contract-then-compress | O(n D^3 chi^3) | near-optimal accuracy, slow |
> | zip-up | O(n D chi chibar^2) | single-shot, but truncations only see a partial environment |
> | fit | O(n D chi chibar^2) per sweep | iterative; can converge slowly or fail (e.g. long-range MPOs, Fig. 2 of the paper) |
> | SRC | O(n D chi chibar^2) | single-shot, near-optimal accuracy using the full environment |
>
> So the asymptotic cost matches zip-up and one fitting sweep, while giving
> near-optimal truncation accuracy in a single pass and avoiding the
> convergence failure modes of fitting. The paper reports speedups up to
> 181x over direct multiplication, and it also describes an extension to
> compressed linear combinations of MPO-MPS products (their Sec. 3.6), which
> would be useful beyond plain contraction.
>
> ## Proposed scope
>
> - Add `ContractMethod::Src` (naming TBD) to the options in
>   `tensor4all-itensorlike` and an implementation `contract_src.rs` in
>   `tensor4all-simplett`, for both MPO-MPS and MPO-MPO products.
> - Support both fixed output rank and adaptive rank selection (the paper
>   discusses an adaptive variant).
> - Benchmark against `contract_zipup` and `contract_fit` in `benchmarks/`
>   (same setup as `benchmarks/results/2026-05-19-tt-ops.md`).
> - Possibly extend to TreeTN `apply` later; the initial implementation can
>   be TT-only.
>
> ## Caveats
>
> - SRC as published is not symmetry-preserving: the sketched intermediates
>   have no block-sparsity structure (Sec. 3.5 of the paper). This is
>   irrelevant for the quantics use cases but worth documenting.
> - Needs a source of Gaussian (or structured) random sketches; a structured
>   sketch could reduce constants.

### #563, shinaoka, 2026-07-29T07:13:51Z

> One implementation-relevant detail from the paper: the core loop of SRC is
> QR-only. Each site performs (i) a sketch contraction with a Khatri-Rao
> structured Gaussian test matrix, (ii) a QR decomposition of the sketched
> matrix for orthonormalization, and (iii) a projection (Sec. 2.1 and
> Algorithm 1). It produces the same approximation as a randomized SVD but
> never forms an SVD of any large matrix.
>
> An SVD appears only in an optional final step: the sketch runs with a
> slightly oversampled rank, and a standard SVD-based MPS truncation is
> applied at the end to cut the output down to the target bond dimension or
> to select the rank adaptively from a tolerance. That SVD acts on the
> already-compressed MPS, so it is cheap.
>
> Consequences for the implementation:
>
> - The hot path is GEMM + QR, i.e. pure BLAS3-friendly kernels, which
>   should map well onto the existing dense backends and onto GPU execution
>   (relevant to #553).
> - The final truncation can reuse the existing SVD-based MPS compression
>   code, so `contract_src` only needs the sketched QB sweep plus a call
>   into the existing truncation routine.

**Status: current, not superseded.** This is the primary test for any SVD
usage found in the branch.

### #563, shinaoka, 2026-08-24T13:45:35Z

> For the MPO-MPO case, it may be worth exploiting the operator structure
> rather than simply fusing the two physical legs and treating the product
> as an MPS with local dimension (d^2).
>
> For the product `C^{s,t} = Σ_u A^{s,u} B^{u,t}`, a natural structured
> sketch is to factorize the random probe on the two external physical legs,
> `Ω_{s,t,k} = X_{s,k} Y_{t,k}`.
>
> Then the local sketched contraction becomes
> `E_k = Σ_{s,t,u} X*_{s,k} A^{s,u} B^{u,t} Y*_{t,k}`, so the fused
> (d^2)-dimensional physical tensor never has to be formed explicitly.
>
> Conceptually, this is an operator-space rank-1 probe, i.e. sampling matrix
> elements of the form `⟨x_k| C |y_k⟩`. The chain-direction SRC algorithm
> itself can remain essentially unchanged; only the local sketch contraction
> is specialized to the MPO-MPO structure.
>
> This would give us two possible implementations:
>
> 1. a simple reference implementation obtained by fusing (s,t) and reusing
>    the MPO-MPS SRC machinery;
> 2. a dedicated MPO-MPO path using the factorized physical sketch above.
>
> The latter should reduce constants and avoid unnecessary (d^2)
> intermediates. It is also a more natural formulation if MPO-MPO
> contraction is treated as a first-class operation rather than as MPS
> compression after index fusion.

**Status: current, not superseded.** This is the exact contraction order
WS-tree-probe must check `src_probe.rs` against.

### #563, shinaoka, 2026-08-27T12:56:57Z — "Future parallelization: SRC's sequential cache is a prefix product..."

> Notes from a design discussion, recorded here so the first implementation
> can keep the door open. Nothing below needs to be implemented now; the ask
> is only an abstraction boundary (last section).
>
> **Where the sequential dependency lives.** As published, SRC is a single
> edge-to-edge pass: processing site k requires the accumulated sketch
> contraction of sites 1..k-1 (the cache), built as
> `M_1 -> M_2 = M_1 * (site-2 piece) -> M_3 = M_2 * (site-3 piece) -> ...`.
> The per-site pieces are small matrices and the accumulation is an
> associative product. So the cache column is a prefix product, structurally
> identical to a cumulative sum. Consequences: (1) the published algorithm
> has parallel depth O(n) along the chain (only the per-step GEMMs
> parallelize); (2) because the operation is associative, the whole family
> of scan/blocking/checkpointing techniques applies — as far as we can tell
> a parallel-scan formulation of sketched TT compression has not been
> published.
>
> **The storage/depth/memory trade-off family.** Let n be the number of
> sites and m the size of one cached unit. The options form one family
> parametrized by how many tree levels are stored:
>
> | scheme | stored memory | parallel depth | recompute overhead |
> |---|---|---|---|
> | flat list (all M_k) | n m | n | none |
> | 2-level blocking (sqrt-n blocks) | ~2 sqrt(n) m | sqrt(n) | ~2x |
> | r-level blocking | ~r n^(1/r) m | n^(1/r) | ~r x |
> | full segment tree | 2 n m | log n | none |
> | binomial checkpointing (REVOLVE) | O(log n) m | n | O(log n) x |
>
> The 2-level scheme is exactly the standard GPU scan implementation
> (block-local pass, then a scan over block aggregates), so it is the
> natural default when this ever moves to GPU: construction depth drops
> from n to sqrt(n), and the checkpoint structure is read-only at query
> time, hence trivially shared across a batch of independent contractions.
>
> Worked example for scale: n = 2^20 units of a 20x20 complex matrix
> (6.4 KB each): flat list ~6.7 GB, 2-level blocking ~13 MB at 2x recompute,
> log-level checkpointing ~130 KB at ~20x recompute. Recompute is a chain of
> small GEMMs, so on modern hardware memory capacity is the scarce resource
> and the blocked schemes win.
>
> **Same primitive appears elsewhere.** The "segment tree of an associative
> matrix product over an ordered sequence" is already a proven pattern in
> adjacent codes and in this project's roadmap: CT-HYB trace caches
> (TRIQS/cthyb balanced tree, lazy skip lists, point update in O(log n));
> dyadic evaluation of time-ordered transfer operators (range query in
> O(log n)); parallel-scan construction for SRC caches (full prefix in
> O(log n) depth). Three use sites, one data structure, three access
> patterns (point update / range query / parallel scan). This suggests a
> shared primitive crate-level utility rather than an SRC-local solution.
>
> **Concrete ask for the first implementation.** None of the above needs to
> ship with `contract_src`. The only request: put the cache behind a small
> trait (something like `PrefixCache: fn extend(piece), fn get(k)`) instead
> of hard-coding a Vec built in a forward loop. Then flat list is the first
> implementation, and blocked/tree/checkpointed policies can be swapped in
> later without touching the SRC logic itself.

**Status: SUPERSEDED in part** by the 15:38 comment below (the
tree/scan-parallelism framing was wrong for large-chi TT caches — "edge-
sequential contraction is essentially forced"). **The `PrefixCache` trait
ask itself is not retracted** and remains current — WS-integration should
check for it.

### #563, shinaoka, 2026-08-27T15:38:47Z — "Correction to the scan claim above: tree composition inflates the work by the boundary dimension"

> An important correction to my previous comment, following further
> discussion.
>
> The sequential edge pass never materializes the boundary operators: it
> applies each site's tensors to a thin block (boundary dimension
> `D = chi_A chi_W` times the sketch width l), at matrix-times-thin-block
> cost per site. Any tree or scan composition, by contrast, must
> materialize interval transfer operators as D×D matrices and multiply them
> pairwise, at O(D^3) per composition. With `D = chi^2` that is a `chi^6`
> step against the `chi^3`-class sequential pass. So for generic large-chi
> MPO-MPS caches, parallel depth via scan is bought at a work penalty of
> roughly a factor D, which is not acceptable; edge-sequential contraction
> is essentially forced.
>
> The scan family is appropriate only when the associative elements are
> small or structured:
>
> - CT-HYB trace trees work because the elements are atomic-space matrices
>   (small, block-diagonal).
> - Dyadic products of state-space transfer matrices work because chi_l is
>   small (tens).
> - ML state-space models (S4, Mamba) run on parallel scans precisely
>   because they restrict the transition matrices to diagonal form so that
>   composition is cheap. Same constraint, same reason.
>
> What survives unchanged is the memory side: blocking and binomial
> checkpointing store thin boundary states and recompute within blocks by
> ordinary edge application, with no operator materialization. So the
> trade-off table in the previous comment should be read as: the memory
> column stands for generic TT caches; the parallel-depth column applies
> only when the cached elements are small or structured. For SRC itself at
> large chi, the realistic parallelism is per-step GEMM parallelism and
> batching over independent contractions.
>
> The implementation ask is unchanged: a small cache trait, flat list
> first, blocked or checkpointed policies later for memory (not for depth).

**Status: current — this is the authoritative statement on chain-cache
parallelism, superseding the 12:56 comment's scan/tree framing.** Any code
or derivation that materializes tree/scan-composed boundary operators for
SRC's own cache should be checked against this correction, not the earlier
comment.

### #563, shinaoka, 2026-08-27T18:24:53Z

> Cross-reference on the parallelization question discussed above:
> chain-parallel contraction does exist in-house, for the fit algorithm:
> Fodera, Ritter, Shinaoka, and von Delft, arXiv:2606.23274 (sub-TT
> partitioning with MPI, inverse-canonical vs site-canonical boundary
> gauges, plus randomized projections that bring 2-site updates down to
> 1-site cost).
>
> This clarifies the division of labor between contract methods: one-pass
> algorithms (SRC, RSI, zip-up) are near-optimal in a single sweep but
> structurally hard to parallelize along the chain (composing boundary
> operators inflates the work by the boundary dimension), whereas the
> iterative fit tolerates sub-chain partitioning because subsequent sweeps
> heal the boundary inconsistencies. So SRC is the right default for
> small-to-medium chi and batched workloads, and parallel fit (2606.23274)
> is the tool for single large-chi MPO-MPO contractions. Both fit naturally
> behind the same ContractMethod interface.

**Status: current.** Confirms SRC (this issue) and parallel-fit
(2606.23274) are deliberately separate tools — not evidence that SRC itself
should grow chain-parallel machinery.

### Issue #691 — opening framing (title: "Parallel SRC via interface sketching (combine arXiv:2606.23274 sub-TT partitioning with SRC)")

This is a **separate, unimplemented future proposal**, opened after #563.
Nothing in it describes work that should already exist in the
`feature/treetn-src` branch; any code resembling it there is scope creep,
not fulfillment of #563.

### #691, shinaoka, 2026-08-27T19:07:24Z

> Two literature notes relevant to Layer 2 (interface sketching):
>
> 1. Al Daas, Ballard, Grigori, Aguilar, Saibaba, and Verma, *Adaptive
>    randomized tensor train rounding using Khatri-Rao products*,
>    arXiv:2511.03598. Provides an adaptive randomized rounding scheme with
>    a rigorous error estimator, directly relevant to choosing the
>    interface sketch width l' adaptively.
> 2. Status check on adjacent work (RSI, arXiv:2602.17974, Feb 2026): the
>    RSI authors' stated future directions are improved sketching and
>    nonlinear elementwise mappings, not MPO-MPO/MPO-MPS contraction or
>    parallelization. SRC (arXiv:2504.06475) remains the only sketched
>    MPO-MPS contraction, and it is sequential. So chain-parallel sketched
>    contraction with interface projections appears open in the published
>    literature.

**Status: current, but describes a proposal, not an implementation
requirement for #563.**

### #691, shinaoka, 2026-08-27T19:13:30Z — "Cost accounting: the sketch pass is the dominant term, which upgrades the role of interface sketching"

> A correction and sharpening of the discussion above, after counting
> operations for the two-sided sketching construction applied to an
> MPO-MPS network input.
>
> **Work.** With MPO bond chi_W, MPS bond chi, physical dimension d, length
> n, sketch width l: sketch pass ~ `n l chi chi_W (chi d + chi_W d^2)`, the
> same order as one zip-up pass, i.e. the dominant cost of the whole
> contraction; core determination (pseudo-inverses) ~ `n l^3 d`, smaller by
> roughly a factor chi_W. So for network inputs it is wrong to say the
> expensive part parallelizes and only a cheap tail stays sequential: the
> sequential sketch pass IS the dominant term.
>
> **What sketching does improve about the sequential part.** (1) The l
> sketch columns are mutually independent, so the pass is embarrassingly
> parallel across columns; only the depth stays O(n). (2) The remaining
> sequential unit per site is a single small GEMM, with no QR or SVD inside
> the chain — lighter than SRC's chain, which interleaves QR and
> projections.
>
> **Consequence.** Breaking the O(n) depth requires cutting the sketch pass
> itself into segments (the interface projections of Layer 2). The open
> variance-vs-N_p question is the central technical risk of the whole
> proposal.

**Status: current, and itself explicitly labeled as revising the framing of
the 19:07 comment on the same thread.** Also describes a proposal, not an
implementation requirement for #563.

---

*This appendix is a verbatim transcription for offline reference. If it and
the live GitHub issue ever disagree, the live issue is authoritative — check
`gh issue view 563 --repo tensor4all/tensor4all-rs --comments` and the
equivalent for #691 if there is any doubt about currency.*
