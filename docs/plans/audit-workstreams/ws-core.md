# WS-core — tensor4all-core additions

Scope: `crates/tensor4all-core/src/defaults/idx_tensor.rs` (+533/-0),
`crates/tensor4all-core/src/tensor_like.rs` (+510/-0),
`crates/tensor4all-core/src/defaults/factorize.rs` (+46/-34),
`crates/tensor4all-core/src/index_like.rs` (+28/-0),
`crates/tensor4all-core/src/index_ops.rs` (+4/-0),
`crates/tensor4all-core/src/defaults/index.rs` (+4/-0), diffed against
`origin/main` (`git diff origin/main -- <files>`, captured to
`/tmp/ws-core.diff`, 1308 lines, 6 hunks-worth of files, 1091
insertions / 34 deletions total). Every hunk in that diff is accounted
for below; no hunk was skipped.

Provenance/history used: `git log --oneline origin/main..HEAD` shows six
commits on the branch (`cd3724b`, `e69b6ea`, `4e56730`, `4167f3c`,
`9e018d4`, `7d574d7`). Of these, only `cd3724b` (initial seams, +37/+4/+28/+4/+35
across these six files) and `9e018d4` ("align SRC with paper and reference",
+46/+496/+475/+192-tests) touch the WS-core files. `9e018d4`'s commit message
contains only an **end-to-end** benchmark table (fit/zip-up/adaptive-SRC/fixed-SRC
wall-clock and error at N=2,8,32,128) plus a `cargo test`/`clippy`/`fmt` pass
summary — no per-component profile of "random-vector construction / per-column
contraction planning / sketch-column assembly" as the plan's gate demands (see
Finding F1 below).

## Priority check: `factorize.rs` (the file named for factorization)

**Verdict: no new factorization logic exists in this file.** Read in full
(931 lines) and diffed: every one of its ~20 `factorize_*`/`matrix_*`
functions (SVD, QR, LU, CI/cross-interpolation, Gram) pre-exists unchanged on
`origin/main`. The only change across all 6 hunks (lines 5-100 of the diff)
is that every `Ok(FactorizeResult { left, right, bond_index,
singular_values, rank })` struct-literal construction is replaced by
`Ok(FactorizeResult::new(left, right, bond_index, singular_values, rank))`,
because `tensor_like.rs` (below) turns `FactorizeResult` into a struct with a
new private field (`incremental_qr_state`) that a struct literal can no
longer construct from outside the module.

| Unit | Lines (diff) | Verdict |
| --- | --- | --- |
| `factorize_gram`, `factorize_svd_with_options` (×2 arms), `factorize_qr_with_options`, `factorize_lu_with_options`, `factorize_ci_with_options`: struct-literal → `FactorizeResult::new(...)` call-site rewrite | 5-100 (all of factorize.rs's diff) | `DERIVED-VERIFIED` — mechanical signature migration only. Verified by direct comparison: each call site passes the identical five values in the identical order to `::new()` that it previously used in the struct literal; no field is dropped, reordered, or defaulted differently. No SVD/QR/hand-rolled linear algebra was added, changed, or hidden here. **This directly answers the audit's top-priority suspicion for this file: unfounded.** The real QR/incremental-QR surface lives in `tensor4all-tensorbackend::IncrementalQr` (WS-backend's `incremental_qr.rs`, +1005 new lines — out of WS-core's file list) and in the `tensor_like.rs`/`idx_tensor.rs` bridge below, not in `factorize.rs`. |

## `index_like.rs` / `index_ops.rs` / `defaults/index.rs` — mechanical

| Unit | Lines (diff) | Verdict |
| --- | --- | --- |
| `IndexLike::new_link(dim) -> anyhow::Result<Self>` trait method + doctest (index_like.rs:219-244) | 703-733 | `DERIVED-VERIFIED` — trivial: "create a fresh undirected link index of a given dimension" has no interesting math; it's an identity/metadata operation. **Mechanical, matches SRC's stated need**: confirmed by usage — `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:570` calls `T::Index::new_link(width)` to mint the fresh batch index for the prefix-cache's combined sketch columns, i.e. exactly the "replace/create bond-like indices with fresh IDs before/for local contractions" pattern WS-core was asked to check for. Not unrelated capability. |
| `DynIndex::new_link` impl (defaults/index.rs:523-527), delegates to pre-existing `Index::new_link` | 684-698 | `DERIVED-VERIFIED` — one-line delegation to an already-existing inherent constructor (`Index::new_link` pre-dates this branch; only the trait-level plumbing is new). |
| Test-double `new_link` impls in `index_like.rs::mod tests` and `index_ops.rs::mod tests` (test fixtures only) | 734-744, 749-759 | `DERIVED-VERIFIED` — test-only fixture code required to keep the two hand-written `IndexLike` test doubles compiling against the now-larger trait; no production behavior. |

## `factorize.rs` and index files: structural completeness

All hunks in these three small files are accounted for above; nothing was
summarized away.

## `tensor_like.rs` (+510) and `idx_tensor.rs` (+533) — the SRC tensor-abstraction additions

### Finding F1 — `PLAN-CLAIM-UNVERIFIED`: the plan's profiling gate was not satisfied

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

### Finding F2 — the "reusable batch/stack constructor... with an optimized `IdxTensor` implementation" the plan asks for **already existed before this branch**

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

### Finding F3 — in-code "Provenance:" comments self-cite a Tier-2, same-process document; treat their claims as unverified until independently re-traced

Several of the new functions carry comments of the form *"labelled
`[AI-Supplied]` in the SRC audit"* or *"is `[Repo]`/`[AI-Supplied]`"*
(e.g. `idx_tensor.rs` diff lines 124-127, 326-329; `tensor_like.rs` diff
lines 846-848, 913-916). These cite
`docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md`.
Checked its provenance:

```
$ git log --format="%H %ai %s" 9e018d4 -1
9e018d4... 2026-08-27 14:06:05 +0200 feat(treetn): align SRC with paper and reference
$ ls -la docs/worklogs/2026-08-27-treetn-src-provenance-and-derivation-audit.md
-rw-r--r-- 1 root root 41712 Aug 27 14:05 ...
```

The self-audit worklog was written **one minute before** the commit whose
code comments cite it — same session, same author, no independent review in
between. Per this audit's own epistemics (Tier 2: *"any in-code comment that
claims a derivation are themselves products of the same AI-assisted process
under suspicion... [it is] a map... [n]ever cite them as the thing a piece of
code is checked against"*), these in-code citations are **not** evidence.
This is a distinct hallucination-adjacent pattern from "invented API" or
"padding": the code cites its own prior self-assessment as if that
constituted external verification — a closed citation loop. It does not by
itself make the underlying `[Derived]`/`[AI-Supplied]` claims wrong (several
were independently re-verified against Tier 1 below and did hold up), but
the citation style itself should not be trusted, and every claim it makes
was re-checked against Tier 1 directly rather than accepted on the worklog's
authority, per the rows below.

### Per-unit table

| Unit | File:lines (current file) | Verdict | Notes / derivation |
| --- | --- | --- | --- |
| `IdxTensor::try_contract_pairwise_retaining` (batched dot_general with retained/batch axes, fallback to generic `contract_with_options`) | idx_tensor.rs:3648-3785 | `DERIVED-VERIFIED` + `PLAN-CLAIM-UNVERIFIED` (F1) | No paper/Python/Hiroshi source specifies this exact batching strategy (Hiroshi's comments address chain-level caching, not intra-step column batching). Re-derived independently: batches N independent probe-column contractions into one `dot_general_with_conj` call using `DotGeneralConfig{lhs_batch_dims, rhs_batch_dims}`. Verified against `tenferro-tensor/src/backend.rs:106-190` (real, pre-existing dependency): a `dot_general` batch axis is contracted per-batch-index independently and the result is `[lhs_free, rhs_free, batch]` in that order — exactly what the function's own comment claims and exactly what its `current_indices`/`desired_indices` permutation logic assumes. This makes the batched call mathematically identical to running one `contract()` per retained-index value and stacking — i.e. correct — while being one GEMM instead of N. Does **not** duplicate/hand-roll linear algebra: it reuses `dot_general_with_conj` (pre-existing tenferro primitive) and `common_ind_positions`/`is_contractable` (pre-existing `tensor4all-core` helpers), so `HANDROLLED-DUPLICATE` does not apply. Guard clauses (empty `retained_indices`, structured-payload short-circuit, dtype mismatch, absent/non-contractable retained index) are all real possible caller states for a `pub(crate)` helper reachable through the public `contract_retaining_indices` trait method with caller-supplied `retained_indices` — not padding for a structurally-impossible case. |
| `impl TensorFactorizationLike for IdxTensor { factorize_probe_columns_incremental, src_error_estimate }`, `IncrementalQrStateScalar` (+ f64/Complex64 impls), `incremental_probe_factorize_typed`, `probe_columns_matrix` | idx_tensor.rs:5643-5906 (approx., all of diff lines 275-573) | `SOURCED-PYTHON(incrementalqr.py:114 IncrementalQR.append, incrementalqr.cpp:46 add_cols)` for the resume/append *contract*; `SOURCED-PAPER(Appendix C.3 "Final optimization: Updating the QR factorization", report.tex:1288-1332)` for the role of the update (confirmed real — see verification below); dtype dispatch / private-state-enum plumbing / index bookkeeping is tensor4all-specific `[AI-Supplied]`, independently reviewed here rather than accepted from the self-audit citation | Verified `report.tex` actually has an Appendix C with exactly the structure claimed: `\appendix` (line 1100) → §A `khatri-rao-exact` (1102) → §B `src-exact-proof` (1125) → **§C `Implementing SRC with a tolerance`** (1146, subsections C.1 "Error estimation" 1152, C.2 "Adaptive determination of bond dimension" 1265, **C.3 "Final optimization: Updating the QR factorization"** 1288) → §D "Full operation counts" (1333). §C.3's block formula (`G' = [[G,0],[-(R'')^{-†}(R')^†G, (R'')^{-†}]]` for `G := R^{-†}`) is exactly the "R^{-†} block formula" the code/worklog claim it implements — this citation holds up under direct re-check, it is not fabricated. §C.2 step 5 ("Recompute (or update, see App. C.3) the QR factorization of Y^(j)") matches this function's "resume-or-recompute" branch structure. Traced the bridge logic by hand (dtype branch via `is_f64()`/`is_c64()` → `incremental_probe_factorize_typed::<f64/Complex64>`; `S::resume` recovers packed Householder state or falls back to reconstructing `IncrementalQr::from_factors` from the previous dense Q/R; rank-decrease is rejected as an error; new `Q` columns are fetched only for the appended range via `state.q_columns(previous_rank, appended_rank)` and concatenated onto the previous `Q` via the new `concatenate_along_new_index`) — internally consistent with the append contract and no arithmetic error found on inspection. **Not independently numerically tested by WS-core** (no cargo test run beyond `cargo check`, which passed) — cross-reference WS-tests for whether `tensor_like/tests/mod.rs` and `backend/tests/mod.rs` actually exercise incremental-append correctness against ground truth, not just shape/API coverage. All API names used (`IncrementalQr`, `IncrementalQrScalar`, `IncrementalQr::{new,from_factors,append,q_columns,r}`, `Matrix::{from_col_major_vec,as_col_major_slice}`, `SrcErrorEstimate`, `src_error_estimate`) were grepped and confirmed to exist verbatim in `tensor4all-tensorbackend/src/{incremental_qr.rs,matrix.rs,backend.rs}` — no invented API names found. |
| `impl TensorContractionLike for IdxTensor { contract_retaining_indices }` (2-tensor fast path dispatch) | idx_tensor.rs:5978-5996 | `DERIVED-VERIFIED` (trivial) | One-line dispatch: 2 tensors → `try_contract_pairwise_retaining`; else → generic `contract_with_options`. Mechanical. |
| `impl TensorConstructionLike for IdxTensor { from_dense_any, stack_along_new_index }` (trait forwarders) | idx_tensor.rs:6041-6055 | `DERIVED-VERIFIED` (trivial) | Pure forwarders to pre-existing native inherent methods, confirmed pre-dating this branch on `origin/main` (see Finding F2). No new logic. |
| `impl TensorConstructionLike for IdxTensor { concatenate_along_new_index }` (native override, per-tensor slice validation + `EagerTensor::concatenate`) | idx_tensor.rs:6056-6125 | `DERIVED-VERIFIED` | No prior native implementation existed for this one (confirmed: absent from `origin/main`). Derivation: validates all operand tensors share the same external-index order away from one designated concatenation axis, sums source-index dimensions, then delegates to `EagerTensor::concatenate(&inners, first_axis)` — a real, pre-existing `tenferro-ad` primitive (`tenferro-ad/src/eager_ops.rs:964`, signature `(tensors: &[&Self], axis: usize) -> Result<Self>` matches the call site exactly). This is literally what "concatenate along an axis" means; no invented math. Confirmed **not over-generalized**: called with more than 2 operands at `crates/tensor4all-treetn/src/treetn/contraction/src_chain.rs:588` (`T::concatenate_along_new_index(&segment_prefixes, &source_indices, batch)`, where `segment_prefixes` ranges over an arbitrary number of adaptive-batching segments), so the N-ary signature is exercised by real SRC control flow, not padding beyond an actual 2-way append use case. |
| `FactorizeResult::new`, `with_incremental_qr_state`, `incremental_qr_state`, `IncrementalQrState` enum | tensor_like.rs:478-538 (diff lines 772-837) | `DERIVED-VERIFIED` (trivial) | New private field + constructor/accessor to carry opaque incremental-QR state across the new `FactorizeResult`; exactly what `factorize.rs`'s refactor and `idx_tensor.rs`'s incremental bridge require. No algorithmic content. |
| `TensorContractionLike::contract_retaining_indices` trait method + generic default (falls back to `contract` when `retained_indices` empty; else returns an "unsupported" error) | tensor_like.rs:839-901 (diff lines 840-905) | `DERIVED-VERIFIED` | Confirmed this "return an error unless a backend overrides" default pattern is an established repo convention, not invented for this PR: `FactorizeError::UnsupportedStorage` already exists pre-branch (`tensor_like.rs:76` on `origin/main`) and other trait methods in this same file already document the "fully generic (monomorphic)... does not support..." pattern pre-branch. |
| `TensorFactorizationLike::factorize_probe_columns_incremental` trait method + generic default (stack all columns, one from-scratch QR) | tensor_like.rs:1001-1111 (diff lines 906-1013) | `SOURCED-PAPER(§C.2 step 5, report.tex:1265-1287)` for the "recompute the QR factorization of the full prefix from scratch" fallback semantics — the paper's own adaptive loop explicitly allows "recompute (or update, see App. C.3)"; the from-scratch default is the "recompute" branch, faithfully implemented as a plain `stack_along_new_index` + `factorize_full_rank(..., QR, Left)`. Trait/default plumbing itself is `[AI-Supplied]`, independently reviewed and found correct. |
| `TensorFactorizationLike::src_error_estimate` trait method + generic default (`Err(UnsupportedStorage(...))`) | tensor_like.rs:1113-1122 | `DERIVED-VERIFIED` (trivial) | Default just delegates the real numerics to `tensor4all_tensorbackend::src_error_estimate` (verified real, `backend.rs:409` — its correctness is WS-backend's audit territory, out of WS-core scope) or errors if unsupported. No math performed in this trait default itself. |
| `TensorConstructionLike::from_dense_any` trait method + generic default (one-hot/axpby dense-from-column-major fallback) | tensor_like.rs:1171-1249 (diff lines 1017-1089) | `DERIVED-VERIFIED` | Re-derived independently: builds `Σ_i data[i] · e_i` via `onehot` + `scale` + `axpby` over a mixed-radix (column-major) index decomposition of the linear position — this is definitionally what "construct a dense tensor from a column-major payload" means; no paper-specific content, holds by construction. Skips zero-valued entries (an optimization, not a correctness issue — verified the `is_zero()` skip does not change the result since it's additive identity). |
| `TensorConstructionLike::stack_along_new_index` trait method + generic default (outer-product-with-onehot-batch-vector fallback) | tensor_like.rs:1249-1313 (diff lines 1091-1189) | `DERIVED-VERIFIED` | Re-derived: stacking `T_1,...,T_n` along a fresh axis of size `n` equals `Σ_k T_k ⊗ e_k` (outer product with the k-th standard basis vector of the new index), then optionally permuted to the caller's requested axis position — holds by construction; matches the standard definition of a stack/concatenate-along-new-axis operation. |
| `TensorConstructionLike::concatenate_along_new_index` trait method + generic default (validate, then slice-and-restack via `select_indices` + `stack_along_new_index`) | tensor_like.rs:1313-1409 (diff lines 1191-1304) | `DERIVED-VERIFIED` | Generic (backend-agnostic) counterpart of the native override above; re-derived the same way — splits each operand into per-column slices via the pre-existing `select_indices` trait method (confirmed pre-existing, not part of this diff) and restacks via the just-derived `stack_along_new_index`. Internally consistent with the native `idx_tensor.rs` override's semantics (both preserve tensor order and require identical index order away from the concatenated axis). |

## Verification

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

## Summary

- **Priority check discharged**: `factorize.rs` contains no new
  factorization logic and no hand-rolled SVD/QR — the audit's founding
  suspicion does not live in this file. (See top section.)
- **No hallucinated/invented API names found.** Every backend symbol used by
  the new code (`IncrementalQr`, `IncrementalQrScalar`, `SrcErrorEstimate`,
  `src_error_estimate`, `Matrix::{from_col_major_vec,as_col_major_slice}`,
  `EagerTensor::concatenate`, `dot_general_with_conj`, `DotGeneralConfig`,
  `common_ind_positions`, `is_contractable`, `select_indices`,
  `ensure_shape_packing_preserves_ad`) was grepped against the real
  `tensor4all-tensorbackend`/`tensor4all-core`/`tenferro-*` crate sources and
  confirmed to exist with matching signatures.
- **No `HANDROLLED-DUPLICATE` found in WS-core's files.** The new retained-
  contraction and incremental-QR bridge code reuses pre-existing backend
  primitives (`dot_general_with_conj`, `EagerTensor::concatenate`,
  `tensor4all_tensorbackend::IncrementalQr`/`src_error_estimate`) rather than
  reimplementing linear algebra.
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
  provenance comments cite a same-session, same-process self-audit worklog
  as though it were external verification. Every claim those comments make
  was independently re-traced to Tier 1 above rather than accepted on the
  worklog's authority; the ones checked (Appendix C.3, the
  `incrementalqr.py`/`.cpp` append contract, §C.2's recompute-or-update
  semantics) held up. The citation *pattern itself* is still a concern for
  the rest of the branch (other workstreams should not accept `[Derived]`/
  `[AI-Supplied]`/`[Repo]` labels from that worklog at face value either).
- No `SUSPECT-UNVERIFIED`, `MISSING-VS-SOURCE`, `SCOPE-DEVIATION`,
  `LICENSE-RISK`, or `SOURCE-AMBIGUOUS` findings in WS-core's file list.
