# WS-backend — numerics and the incremental-QR question

Workstream of the [SRC Provenance Audit](../2026-08-28-src-provenance-audit.md).
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

## Executive summary

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

## Provenance table

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

### `crates/tensor4all-tensorbackend/src/backend.rs` (added hunk)

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

### `crates/tensor4all-tensorbackend/src/incremental_qr.rs` (new file, 1005 lines)

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

### `crates/tensor4all-tensorbackend/src/lib.rs` (added hunks)

| Lines | Unit | Verdict | Basis |
| --- | --- | --- | --- |
| 25–27 | `#[cfg(feature = "global-defaults")] mod incremental_qr;` + doc comment | `SCOPE-DEVIATION` (inherited from the module; see F2) | One `mod` line, so no derivation of its own is owed; it inherits the module's verdict because it is what makes the deferred module exist in the build. Feature gate matches `mod backend;` and `mod matrix;`; correct, since `IncrementalQrScalar` depends on `MatrixTriangularSolveScalar` |
| 49–55 | `pub use backend::{…}` list edited to add `src_error_estimate`, `SrcErrorEstimate` | `NO-FINDING` | Re-export list edit; no algorithmic content, so no derivation is owed. Re-export only; remaining churn on these lines is `rustfmt` re-wrapping of an unchanged list |
| 62–63 | `pub use incremental_qr::{IncrementalQr, IncrementalQrScalar};` + doc | `SCOPE-DEVIATION` (inherited; see F1, F2) | A re-export line, so no derivation is owed, but it is not neutral plumbing. Makes a scalar-loop QR part of the crate's **public** API surface. If F1 is fixed by routing through `qr_backend`, this is a breaking-change surface, and `IncrementalQrScalar` (F10) exposes two dead scalar impls |

---

## Detailed derivations and flagged findings

Paper references are to `/root/projects/RandomMPOMPS-reference-20260827/arxiv-source/report.tex`.
Appendix C is `\section{Implementing SRC with a tolerance}` (`\label{sec:approx}`,
line 1146) — the third appendix, after `app:khatri-rao-exact` and
`app:src-exact-proof`.

### D1 — Appendix C error and norm estimators, re-derived

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

### D2 — Complex Householder reflector, re-derived

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

### D3 — Appendix C.3 block inverse-adjoint update, re-derived

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

### D4 — the rank-skip policy is sound but sourceless (F5)

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

### D5 — reflector application order

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

### D7 — from_factors refactorization identity and its redundancy (F8)

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

### D8 — the packed Householder factorization loop (householder_factor + apply_reflector)

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

### D6 — what the available timing data does and does not say

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

### AI-hallucination signature sweep

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

### SVD / matrix-inverse / hand-rolled-decomposition sweep

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

### Line-coverage ledger

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

## Cross-workstream notes (for the Task 7 synthesis)

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
