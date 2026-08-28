# WS-chain — the literal single-chain case

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

## Provenance table

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_chain.rs` | Module doc comment (provenance claim) | 1-11 | `SOURCED-PAPER(Algorithm 1)` + flagged citation error | Algorithm and Python line-range citations check out (see Detailed findings), but the claimed paper location "Sections 2.3--2.5" is wrong — see flagged finding below. |
| `src_chain.rs` | Imports | 13-27 | n/a (trivial plumbing) | Pulls `FactorizeAlg`, `SvdTruncationPolicy`, `TensorLike`, `IndexLike`, `Canonical` from `tensor4all_core` (not hand-rolled), and probe/prefix helpers from sibling `src_probe.rs` (audited under WS-tree-probe). |
| `src_chain.rs` | `fn contract` — signature, chain retrieval, topology/emptiness checks | 30-52 | `DERIVED-VERIFIED` | Engineering precondition-checking with no direct paper analog (the paper's MPO/MPS *are* linear chains by construction; extracting a chain from a general `TreeTN` and checking `same_topology` is Rust-side infrastructure). Trivially correct: refuses to proceed without a valid, non-empty, topology-matched chain. |
| `src_chain.rs` | `fn contract` — index desimilarization + local site pairs + `chain.len()==1` special case | 54-67 | `DERIVED-VERIFIED` | See "n=1 special case" derivation below. Python's `random_contraction` explicitly does *not* implement this case (`raise NotImplementedError`) — the Rust code adds a capability, not a citation gap. |
| `src_chain.rs` | `fn contract` — `outputs`, `cut_dimensions`, `probe_indices`, `last_output_dim`, `last_maximum_width`, `ProbeBank::new` | 69-91 | `SOURCED-PYTHON(contraction.py:138-150)` | Matches Python's per-site `prod_bond_dims`/`current_maxdim`/`current_sketchdim` computation; see `chain_cut_dimensions` row below for the exact correspondence. The probe/width-bound *logic itself* (`maximum_site_width`) is defined in `src_probe.rs`, out of this workstream's file list — noted, not re-verified here. |
| `src_chain.rs` | `fn contract` — dispatch to `contract_fixed` when `src_options.rtol.is_none()` | 92-104 | `SOURCED-PAPER(§3, fixed-χ̄ vs adaptive-χ̄ dichotomy)` | Matches the paper's two operating modes: §3.1-3.3 assume a supplied output bond dimension χ̄; §3.6/Appendix (`sec:approx`) adds tolerance-driven adaptive determination. Python encodes the same dichotomy with one function guarded by `if outputdim is not None`; Rust splits it into two functions (`contract` vs `contract_fixed`) — see duplication note below. |
| `src_chain.rs` | `fn contract` — `sketch_options`, `PrefixCache::new` | 105-107 | see struct rows below | Cross-reference only; verdict lives on the `PrefixCache` struct definition (SCOPE-DEVIATION, see below). |
| `src_chain.rs` | `fn contract` — adaptive last-site determination (`outputs[last].is_empty()` scalar branch + `factorize_site_adaptive` call) | 109-151 | `SOURCED-PAPER(§3.1, Algorithm 1 "Determine the last site")` + `SOURCED-PYTHON(contraction.py:205-212, 463-471)` + `DERIVED-VERIFIED` (scalar sub-branch) | See "Last-site sketch" and "Scalar-output boundary case" derivations below. |
| `src_chain.rs` | `fn contract` — main site loop `for site in (1..last).rev()` | 153-206 | `SOURCED-PAPER(§3.2/§3.3, Algorithm 1 "Determine sites η^(n-1),...,η^(2)")` + `SOURCED-PYTHON(contraction.py:213-238, random_contraction_inc:460-490)` | See "Interior-site sketch" derivation below. No SVD call anywhere in this loop (grep-verified, see SVD audit below). |
| `src_chain.rs` | `fn contract` — first-site determination | 208-210 | `SOURCED-PAPER(§3.3, Algorithm 1 "Determine the first site η^(1)", pseudocode line 669)` + `SOURCED-PYTHON(contraction.py:344-348)` | `contract_site_pair(local[0].0, local[0].1, &[&cap_environment])` matches η^(1)(a,b) = Σ H^(1)(a,c,d) ψ^(1)(d,e) S^(2)(e,d,b) — three tensors contracted (H, ψ, S^(2)), same as the pseudocode and the Python final block. |
| `src_chain.rs` | `fn contract` — result assembly (`TreeTN::new`, per-site `add_tensor`, `connect_result_edge`) | 212-221 | `DERIVED-VERIFIED` | Mechanical: the paper/Python return a flat MPS list; Rust must rebuild a graph (`TreeTN`) from the same per-site tensors and wire up the same chain edges. No new math — the edges connected are exactly the chain's original adjacency (`chain.windows(2)`, `chain[i-1]`-`chain[i]`), which reproduces the same 1-D topology. |
| `src_chain.rs` | `fn contract` — final SVD / canonical marking | 223-237 | `SOURCED-PAPER(§3.4, pseudocode line 670 "Optional: Run an MPS truncation algorithm")` + `SOURCED-COMMENT(#563, 2026-07-29T07:13:51Z)` | The one legitimate SVD call site in `contract`. See SVD audit below — gated by `src_options.final_svd`, applied only to the fully-assembled `result` (post per-site QR loop), matching Hiroshi's "acts on the already-compressed MPS" claim exactly. When `final_svd` is false, `mark_result_canonical` is used instead (no SVD at all). |
| `src_chain.rs` | `struct FixedContractionRequest` | 240-253 | n/a (trivial plumbing) | Parameter-bundle struct, no logic. |
| `src_chain.rs` | `fn contract_fixed` — signature/setup, `fixed_options`, `last_maximum_width`, `BatchedPrefixCache::new` | 255-289 | `SOURCED-PYTHON(contraction.py: outputdim-is-not-None branch, lines 110-117, 391-393)` | Mirrors the fixed-rank branch of the Python reference (`if outputdim is None: ... else: maxdim=mindim=sketchdim=outputdim`). |
| `src_chain.rs` | `fn contract_fixed` — last-site fixed determination (scalar branch + `BatchedPrefixCache::batch` + `factorize_fixed_batch`) | 290-324 | `SOURCED-PAPER(§3.1, Algorithm 1)` + `SOURCED-PYTHON(contraction.py:205-212)` + `DERIVED-VERIFIED` (scalar sub-branch, same derivation as the adaptive path) | Same math as the adaptive last-site branch, but the sketch is built as a single batch (`prefixes.batch(...)`, width fixed to `last_maximum_width` up front) rather than incrementally — see "Batched vs incremental sketch equivalence" derivation below. |
| `src_chain.rs` | `fn contract_fixed` — main site loop `for site in (1..last).rev()` | 326-374 | `SOURCED-PAPER(§3.2/§3.3, Algorithm 1)` + `SOURCED-PYTHON(contraction.py:213-238)` | Same tensors contracted as the adaptive path's interior-site loop (prefix batch, `local[site].0`/`.1`, `right_environment`), via `contract_prefix_with_probed_site_pair_batch_range` + `contract_pair` instead of the incremental `PrefixCache::column` + `T::contract` chain. Mathematically the same operation (see batching-equivalence derivation). |
| `src_chain.rs` | `fn contract_fixed` — first-site + result assembly + final SVD/canonical | 376-405 | Same verdicts as the corresponding `contract` rows | Byte-for-byte structurally identical logic to lines 208-237 of `contract` (first-site contraction, `TreeTN` assembly, `final_svd`-gated truncation). This is the second (and only other) SVD call site in the file — see SVD audit below. |
| `src_chain.rs` | `fn chain_cut_dimensions` | 407-441 | `SOURCED-PYTHON(contraction.py:138-146)` | Computes, per internal chain edge, `dim_a(edge) * dim_b(edge)` (MPO bond dim × MPS bond dim). Combined with the call-site `.max()` of the two adjacent edges (line 163, 336), this is algebraically identical to Python's per-site `prod_bond_dims = max(H[j].shape[0]*psi[j].shape[0], H[j].shape[2]*psi[j].shape[2])` — see derivation below. The boundary case (`cut_dimensions.last()`, one edge only) matches Python's `j == n-1` branch (single bond, no `max`). |
| `src_chain.rs` | `fn factorize_fixed_batch` | 443-462 | `SOURCED-PAPER(§3, QR-only claim)` + confirms no `HANDROLLED-DUPLICATE` | Calls `sketch.factorize_full_rank(left_indices, FactorizeAlg::QR, FactorizeCanonical::Left)` — an explicit, named QR decomposition delegated to `tensor4all-core`'s typed factorization API, not a hand-rolled linear-algebra routine. This is the fixed-rank path's analog of the paper's step (ii)/(iii) QR-and-project; matches Python's `np.linalg.qr` call sites (contraction.py:244, 247) in kind (QR, not SVD). |
| `src_chain.rs` | `struct PrefixCache` | 464-473 | `SCOPE-DEVIATION` | See "PrefixCache trait ask" finding below — field `prefixes: Vec<Vec<T>>` is a concrete, hard-coded Vec of Vecs, not a trait, contradicting Hiroshi's still-current 2026-08-27T12:56:57Z ask. |
| `src_chain.rs` | `struct BatchedPrefixCache`, `struct PrefixBatchSegment` | 475-493 | `SCOPE-DEVIATION` | Same finding as `PrefixCache`: `BatchedPrefixCache` is likewise a concrete struct (fields `cached`, `segments: Vec<PrefixBatchSegment<T>>`), not behind any trait. |
| `src_chain.rs` | `impl BatchedPrefixCache::new` | 500-513 | n/a (trivial) | Field initialization only. |
| `src_chain.rs` | `impl BatchedPrefixCache::batch` | 515-597 | `SOURCED-PYTHON(concept: contraction.py's incrementally-grown `envs` list, lines 164-199)` + `DERIVED-VERIFIED` (segment/concatenate batching mechanism) | See "Batched vs incremental sketch equivalence" derivation below. |
| `src_chain.rs` | `impl PrefixCache::new` | 605-618 | n/a (trivial) | Field initialization only. |
| `src_chain.rs` | `impl PrefixCache::ensure_width` | 620-666 | `SOURCED-PYTHON(concept: contraction.py's `sketchincrement`-driven `envs` growth, random_contraction_inc lines 431-456)` + `DERIVED-VERIFIED` (batch-then-split mechanism) | Grows the cache in `batch_size` chunks, then splits each chunk into individual per-column tensors via `select_indices`. See derivation below. |
| `src_chain.rs` | `impl PrefixCache::column` | 668-680 | `SOURCED-PYTHON(contraction.py: `envs[idx][j-1]` column access pattern)` | Direct analog of indexing into the Python `envs` list; triggers `ensure_width` on demand instead of requiring the caller to pre-grow the cache. |
| `src_chain.rs` | `struct FactorizeSiteRequest` | 683-695 | n/a (trivial plumbing) | Parameter-bundle struct, no logic. |
| `src_chain.rs` | `fn factorize_site_adaptive` | 697-736 | `SOURCED-PAPER(§3, Algorithm 1 steps (ii)-(iii): orthonormalize + project)` + `SOURCED-PYTHON(contraction.py:280-313, random_contraction_inc:540-571)` | See "Per-site orthonormalize+project" derivation below. Delegates the actual QR/orthonormalization to `factorize_probe_columns` in `src_probe.rs` (out of this file, audited under WS-tree-probe) — this function's own logic is the conjugate-and-project (environment/cap update) step. No SVD call in this function (grep-verified). |

## SVD audit (mandatory per Task 1 brief)

`grep -n -i "svd"` over the full file returns exactly these 12 lines, all
accounted for:

- Line 17: `use tensor4all_core::{..., SvdTruncationPolicy, ...}` — type import only.
- Line 34: `svd_policy: Option<SvdTruncationPolicy>` — parameter of `fn contract`.
- Line 95: `svd_policy,` — struct-field shorthand, passed into `FixedContractionRequest`.
- Line 102: `final_svd: src_options.final_svd,` — struct-field shorthand.
- Line 105: `let sketch_options = src_options.sketch_options(svd_policy.is_some());` — configuration derivation, not a decomposition call.
- Line 223: `if src_options.final_svd {` — **SVD call site #1** (guard).
- Line 226: `svd_policy,` — argument to `result.truncate_impl(...)` inside the guarded branch.
- Line 245, 252, 264, 271: parameter/struct-field declarations in `FixedContractionRequest`/`contract_fixed`'s destructure, threading the same policy through.
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

## Detailed derivations and flagged findings

### Module doc comment — flagged citation error (AI-hallucination-signature: confident but wrong citation)

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

### n=1 special case (lines 57-67) — DERIVED-VERIFIED

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

### Scalar-output boundary case (`outputs[last].is_empty()`, lines 118-129 and 290-301) — DERIVED-VERIFIED

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

### Last-site sketch (lines 109-149, adaptive; 290-322, fixed) — SOURCED-PAPER + SOURCED-PYTHON

Algorithm 1 pseudocode (report.tex:655): `Y^{(n)}(a,b) = Σ_{c,d,e}
C^{(n-1)}(a,c,d) H^{(n)}(b,c,e) ψ^{(n)}(d,e)`. The `make_column` closure at
lines 142-147 (`factorize_site_adaptive`'s adaptive caller) computes: `prefix
= prefixes.column(last-1, column)` (this is `C^{(n-1)}`, one sketch column),
then `contract_prefix_with_site_pair(&prefix, local[last].0, local[last].1)`
— contracting the prefix with `H^{(n)}` and `ψ^{(n)}`, exactly the three
tensors in the pseudocode's `Y^{(n)}` formula. This matches Python's
`j == n-1` sketch-formation branch (contraction.py:205-212 /
random_contraction_inc:463-471): `temp = envs[idx][j-1] @ psi[j]; temp =
H[j] @ temp` — same three-tensor contraction (env/`C`, ψ, H), same order of
operations (prefix into ψ first, then H). The fixed-rank path (lines 303-314)
performs the identical contraction batched over all columns at once via
`contract_prefix_with_probed_site_pair_batch_range` rather than per-column —
see the batching-equivalence derivation below for why this is the same
computation.

### Interior-site sketch (lines 153-206, adaptive; 326-374, fixed) — SOURCED-PAPER + SOURCED-PYTHON

Algorithm 1 pseudocode (report.tex:662): `Y^{(j)} = Σ_{d,e,f,g,h}
C^{(j-1)}(a,d,e) H^{(j)}(d,b,f,g) ψ^{(j)}(e,g,h) S^{(j+1)}(h,f,c)` — a
four-tensor contraction of the prefix `C^{(j-1)}`, the MPO site `H^{(j)}`,
the MPS site `ψ^{(j)}`, and the right-environment cap `S^{(j+1)}`. The
adaptive `make_column` closure (lines 186-201) computes exactly these four
contractions in the same order: `prefix = prefixes.column(site-1, column)`
(→ `C^{(j-1)}`), `after_a = T::contract(&[&prefix, local[site].0])` (→ with
`H^{(j)}`), `after_b = T::contract(&[&after_a, local[site].1])` (→ with
`ψ^{(j)}`), `T::contract(&[&after_b, &right_environment])` (→ with
`S^{(j+1)}`, i.e. `cap_environment` from the previous iteration). This
matches Python's `else` (interior-site) sketch-formation branch
(random_contraction_inc:472-488), which builds the identical four-tensor
product (`env`/`C`, `psi`, `H`, `cap`) in the same dependency order. The
fixed-rank path (lines 344-360) performs the same four-tensor contraction
batched via `contract_prefix_with_probed_site_pair_batch_range` +
`contract_pair`.

### `factorize_site_adaptive` — per-site orthonormalize+project (lines 697-736) — SOURCED-PAPER + SOURCED-PYTHON

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

### `chain_cut_dimensions` — edge-vs-site bound refactor (lines 407-441, consumed at 163 and 336) — SOURCED-PYTHON, refactor verified equivalent

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

### Batched vs incremental sketch equivalence (`BatchedPrefixCache::batch`, lines 515-597; `PrefixCache::ensure_width`, lines 620-666) — DERIVED-VERIFIED

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
Ω^{(j)}_{:,k}` (Khatri-Rao structure, §2.2/2.3 — sorry, §2.2 background +
§3's per-site application). This contraction is linear and acts
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

### `PrefixCache`/`BatchedPrefixCache` — the `PrefixCache` trait ask (lines 464-493) — SCOPE-DEVIATION

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

### `contract` vs `contract_fixed` — structural duplication (not a taxonomy verdict on its own, noted for completeness)

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

### License-risk assessment — no finding

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

### `HANDROLLED-DUPLICATE` — no finding

Every actual decomposition call in this file (`factorize_fixed_batch`'s
`sketch.factorize_full_rank(..., FactorizeAlg::QR, ...)`, and the two
`result.truncate_impl(..., svd_policy, ...)` final-truncation calls)
delegates to `tensor4all-core`'s typed factorization/truncation API (imported
at line 16-18: `Canonical as FactorizeCanonical, FactorizeAlg,
SvdTruncationPolicy`). No inline QR, SVD, matrix-inverse, or LAPACK-wrapper
code exists anywhere in `src_chain.rs` — the plan's explicit ban on
hand-rolled linear algebra in `treetn` is respected in this file.
