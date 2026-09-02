# WS-integration — dispatch, public API, and cross-cutting glue

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
[`docs/plans/2026-08-28-src-provenance-audit.md`](../2026-08-28-src-provenance-audit.md).
Every diff hunk in the five main files plus `options.rs`/`contract.rs` is
covered below; plumbing files get one summary row each per the spec's
catch-all allowance.

## Three named checks (spec's WS-integration section)

### 1. Public API surface vs. the plan's proposed shapes

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

### 2. The mandatory "chain reduction gate" test

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

### 3. `PrefixCache` trait ask (2026-08-27 comments) — honored, ignored, or over-built?

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

### 4. #691 leakage (interface sketching / Layer 2 / sub-chain partitioning / MPI)

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

## Provenance table (every diff hunk)

### `contraction.rs` (SRC-related hunks only; +369 total)

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

### `apply.rs` (+44/-2)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Module doc "ZipUp, Fit, SRC, or local exact naive apply" | ~9-11 | `DERIVED-VERIFIED` (trivial plumbing) | Accurate: `Src` is now a real dispatchable method here (see below), not a doc-only mention. |
| `use ...SrcOptions` import; `ApplyOptions` doc default note | ~85-95, ~107-127 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ApplyOptions.src_options` field + `Default` update | 150-166 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ApplyOptions::src()` | 196-211 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ApplyOptions::with_src_options(...)` | 237-254 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `contraction_options` struct literal: `src_options: options.src_options` | ~394-397 (context read, exact hunk line 400 of diff) | `DERIVED-VERIFIED` (trivial plumbing), verified | Traced: this is the internal `ContractionOptions` built inside `apply_linear_operator` and passed to the shared `contract()` in `contraction.rs`. Confirms `ApplyOptions::src()` genuinely reaches the SRC dispatch arm — this is real plumbing, not a dead/unused field. Matches the opening post's "Possibly extend to TreeTN apply later" scope item. |

### `fit.rs` (+1/-7) and `swap.rs` (+2/-14)

| File | Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- | --- |
| `fit.rs` | `FactorizeResult { left, right, bond_index: dummy_left, singular_values: None, rank: 1 }` → `FactorizeResult::new(left, right, dummy_left, None, 1)` | 723 | `DERIVED-VERIFIED` (trivial plumbing), mechanical | Verified `FactorizeResult::new` genuinely exists at `tensor4all-core/src/tensor_like.rs:517` (corrected from an earlier ~487 citation) with a matching 5-argument signature `(left, right, bond_index, singular_values, rank)`. The struct gained a new **private** field `incremental_qr_state: Option<IncrementalQrState>` (WS-core's `IncrementalQr` addition), which is why the old positional struct-literal syntax stopped compiling outside its defining module — `::new(...)` sets that field to `None`. This is forced, cross-crate compile-fallout plumbing, not new logic and not a hallucinated API — confirmed the constructor is real and does what the call sites need. |
| `swap.rs` | Same replacement, two identical call sites | 59, 81 | `DERIVED-VERIFIED` (trivial plumbing), mechanical | Same `FactorizeResult::new` constructor, same forced compile-fallout reasoning as the `fit.rs` row above. |

### `algorithm.rs` (+7)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Doc comment `T4A_CONTRACT_SRC = 3` | 36 | `DERIVED-VERIFIED` (trivial plumbing) | Matches the enum addition below and the capi header. |
| `ContractionAlgorithm::Src = 3` variant + doc | 66-67 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `from_i32` match arm `3 => Some(Self::Src)` | 79 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `name` match arm `Self::Src => "src"` | 95 | `DERIVED-VERIFIED` (trivial plumbing) | Corrected method name: the match is inside `pub fn name(&self) -> &'static str`, not an `as_str` method — no `as_str` exists anywhere in this file. |

### `tensor4all-itensorlike/src/options.rs` (+55)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `pub use tensor4all_treetn::contraction::SrcOptions;` | 4 | `DERIVED-VERIFIED` (trivial plumbing) | Re-export, no new logic. |
| `ContractMethod::Src` variant | 104-105 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ContractOptions.src_options` field + `Default` update | 132, 143 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `ContractOptions::src()` | 191-206 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ContractOptions::with_src_options(...)` | 260-275 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |
| `ContractOptions::src_options()` getter | 321-335 | `DERIVED-VERIFIED` (trivial plumbing), doc-tested | — |

### `tensor4all-itensorlike/src/contract.rs` (+4/-1)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `ContractMethod::Src => ContractionMethod::Src` match arm | 70 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `.with_src_options(options.src_options().clone())` chained onto `treetn_options` | 77 | `DERIVED-VERIFIED` (trivial plumbing), verified | Confirms `tensor4all-itensorlike`'s `ContractOptions::src()` genuinely threads through to `tensor4all-treetn`'s `ContractionOptions` — not a dead/unused field on this side either. |

### `tensor4all-capi/src/types.rs` (+3)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `t4a_contract_method::Src = 3` | 611 | `DERIVED-VERIFIED` (trivial plumbing) | — |
| `From<ContractionMethod>`/`From<t4a_contract_method>` match arms | 620, 631 | `DERIVED-VERIFIED` (trivial plumbing) | Bidirectional conversion is complete and consistent with `algorithm.rs`'s `ContractionAlgorithm::Src = 3`. |

### `tensor4all-capi/src/treetn.rs` (+9) and `include/tensor4all_capi.h` (+12)

| Unit | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| Doc-comment additions on `t4a_treetn_contract` (`treetn.rs:1606-1608`, `tensor4all_capi.h:1045-1046`), `t4a_treetn_partial_contract` (`treetn.rs:1717-1719`, `tensor4all_capi.h:1231-1232`), `t4a_treetn_apply_operator_chain` (`treetn.rs:1848-1849`, `tensor4all_capi.h:1005-1006`) — "`maxdim` must be nonzero...fixed-rank...Adaptive SRC controls are currently available through the Rust API only" | see cell | `DERIVED-VERIFIED` (trivial plumbing), verified accurate | `t4a_treetn_apply_operator_chain`'s internal `ApplyOptions`-equivalent struct literal sets `src_options: Default::default()` (fixed-rank, seed 0) with no C-ABI parameter to override it — the doc's "Rust API only" claim for adaptive controls is factually correct, not aspirational. |

### Plumbing files (mechanical re-export/registration only, one row each)

| File | Lines | Verdict | Notes |
| --- | --- | --- | --- |
| `tensor4all-treetn/src/lib.rs` | 63 | `DERIVED-VERIFIED` (trivial plumbing) | `pub use treetn::contraction::SrcOptions;` re-export. No logic. |
| `tensor4all-treetn/src/prelude.rs` | 23 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the prelude re-export list. No logic. |
| `tensor4all-treetn/README.md` | 3-4 | `DERIVED-VERIFIED` (trivial plumbing) | One sentence: "naive/zip-up/fit/SRC contraction." Descriptive only. |
| `tensor4all-itensorlike/src/lib.rs` | 15 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the crate's `pub use options::{...}` list. No logic. |
| `tensor4all-itensorlike/src/prelude.rs` | 24 | `DERIVED-VERIFIED` (trivial plumbing) | Adds `SrcOptions` to the prelude re-export list. No logic. |
| `tensor4all-itensorlike/README.md` | n/a | N/A — not touched | `git diff origin/main -- crates/tensor4all-itensorlike/README.md` is empty. This file is **not** part of the SRC diff, unlike `tensor4all-treetn/README.md` above (see M4 fix note: an earlier prose pass in this document incorrectly implied both crates' `README.md` were touched). Row kept for completeness since the spec's plumbing-file list names it. |

### `tensor4all-itensorlike/src/tensortrain.rs` (added per Fix round 1 — previously uncovered)

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

## Detailed derivations and flagged findings

### Finding 1 — Fabricated section citation, copy-pasted across three files (`SUSPECT-UNVERIFIED`, AI-hallucination pattern: *confident-sounding doc text describing something the code/paper doesn't have*)

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

### Finding 2 — `PrefixCache` name reused for the opposite of what was asked (`SCOPE-DEVIATION`, AI-hallucination pattern: *a comment/name claiming alignment with a request that doesn't hold up when checked*)

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

### Finding 3 — plan's report/diagnostics API surface never built (code-vs-plan comparison, see check 1)

Not a hallucination in the code itself (nothing invented pretends to be
`SrcContractionResult`/`SrcRankSelection`/etc.) — flagged here only
because Step 3 of the task brief explicitly calls for this comparison.
The simpler surface that does exist is fully consistent with Tier-1
sources (the opening post and the 2026-07-29 QR-only comment); this is a
scope note about the Tier-2 plan's aspirational API sketch, not a defect.

### Finding 4 — chain-reduction-gate: real test suite, but not the gate as specified (see check 2)

Restated here as a hallucination-adjacent note: nothing in the code or
comments *claims* "this is the chain reduction gate" — there is no false
claim to debunk. The gap is that the Tier-2 plan's specific precondition
was silently not built as specified, and no comment anywhere says so. This
is a coverage gap for a reader relying on the plan, not a fabricated claim
in the code.

### Finding 5 — `LICENSE-RISK` self-description in the module doc-comment (I5, this workstream's own file)

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

### `SrcOptions::validate()` — DERIVED-VERIFIED

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

## No invented APIs found in WS-integration's files

Explicitly checked (not just assumed) every function/type call in the
diff hunks above against its actual definition: `FactorizeResult::new`
(real, matching signature, `tensor4all-core/src/tensor_like.rs`),
`chain_order` (real, `contraction.rs`, semantics traced above),
`src_tree::contract`/`src_chain::contract` (real, correct signatures),
`SvdTruncationPolicy`/`validate_svd_truncation_options` (pre-existing,
unchanged call site). No grep for a plausible-sounding but non-existent
function name in these files turned up a hit that wasn't backed by a real
definition.

## No unnecessary defensive code found for structurally-impossible cases

Reviewed `SrcOptions::validate()` line by line (see table): every branch
guards a state that is genuinely reachable through the public builder API
(e.g. `atol != 0.0` while `rtol.is_none()` is reachable because `atol` and
`rtol` are independent public fields with independent setters) — none of
the checks are dead guards against states the type system already rules
out.
