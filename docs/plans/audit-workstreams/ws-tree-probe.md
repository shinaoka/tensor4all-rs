# WS-tree-probe — the from-scratch tree generalization and MPO-MPO probing

**Files audited:**
- `crates/tensor4all-treetn/src/treetn/contraction/src_tree.rs` (641 lines, full file)
- `crates/tensor4all-treetn/src/treetn/contraction/src_probe.rs` (1171 lines, full file)

Every line of both files is accounted for by some row of the provenance table
below: the table's line ranges are contiguous and exhaustive over 1-641 and
1-1171 respectively. Rows marked `n/a (trivial plumbing)` cover only imports,
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
  594). Probe generation at line 431 (`np.random.randn(visible_dim)`),
  width bounds at 416-418, adaptive stopping at 509-522.
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

## Executive summary of this workstream

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

## Provenance table

### `src_tree.rs` (641 lines)

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_tree.rs` | Module doc comment (provenance claim) | 1-10 | `SOURCED-PAPER(Algorithm 1)` + **flagged citation error (F-2)** | The Algorithm-1 attribution and the "no tree implementation exists in the author repository" claim both check out (verified: `contraction.py` has no tree/graph code; all SRC entry points take `MPO`/`MPS` list types). "Sections 2.3--2.5" does not exist in the Tier-1 paper source. The `random_contraction_inc` range "lines 405--593" understates the function (def at 357). The self-labelling `[AI-Supplied]` for rooting/message-passing/complement-environments is honest and matches this audit's own conclusion that no Tier-1 source covers them. |
| `src_tree.rs` | Imports | 12-23 | n/a (trivial plumbing) | All linear algebra comes from `tensor4all_core` (`TensorLike`, `IndexLike`, `SvdTruncationPolicy`) and sibling `src_probe.rs`. No hand-rolled numerics imported. |
| `src_tree.rs` | `type DirectedEnvironment`, `type BatchedEnvironment` | 25-29 | n/a (trivial plumbing) | Type aliases only. `DirectedEnvironment<T,V> = HashMap<(V,V),T>` is the "cache keyed by `(from_node, to_node)`" the spec asks about — key shape verified correct in D-1. |
| `src_tree.rs` | `fn contract` — signature and trait bounds | 31-44 | n/a (trivial plumbing) | `V: Ord` is genuinely required (the deterministic `nodes.sort()` at line 71). |
| `src_tree.rs` | `fn contract` — chain-delegation guard | 45-59 | `DERIVED-VERIFIED` (D-2) | Delegates to `src_chain::contract` **only** when the topology is a chain *and* the requested center is the terminal site. Verified against `TreeTN::chain_order` (contraction.rs:404-450): `chain.last() == Some(center)` holds iff `center` is a degree-1 endpoint (or n=1). The adjacent comment ("The chain recurrence produces a left-canonical sweep whose center is the final site. An interior requested center needs the rooted tree recurrence") is accurate. |
| `src_tree.rs` | `fn contract` — topology / center / non-empty validation | 60-71 | `DERIVED-VERIFIED` | Engineering preconditions with no paper analogue (the paper's inputs are chains by construction). Note the ordering smell: `chain_order` is consulted *before* `same_topology(tn_b)` is checked, so the chain path is entered without this file having validated topology — `src_chain::contract` repeats the check itself (src_chain.rs:46-48), so this is harmless. |
| `src_tree.rs` | `fn contract` — rooted-edge extraction + connectedness check | 72-81 | `DERIVED-VERIFIED` (D-1, step 1) | `edges_to_canonicalize_by_names(center)` verified to return a genuine **postorder** child→parent listing: `NodeNameNetwork::edges_to_canonicalize(None, target)` (node_name_network.rs:430-440) calls `post_order_dfs_by_index(target)` and then `compute_parent_edges` emits `(node, parent(node))` in that order. The `edges.len() + 1 != nodes.len()` check enforces a connected tree, which the derivation relies on. |
| `src_tree.rs` | `fn contract` — `sim_internal_inds`, `sketch_options`, `local`, `outputs` | 83-103 | `SOURCED-PAPER(§3.4, `sec:adaptivity` last ¶)` for `sketch_options`; `DERIVED-VERIFIED` for the rest | `sketch_options(svd_policy.is_some())` tightens `rtol` to `0.1·rtol` when `final_svd` is on — exactly the paper's "set the relative tolerance to be 0.1 times the requested tolerance and run a final truncation with the requested tolerance" (`sec:adaptivity`, final ¶). Verified in `SrcOptions::sketch_options` (contraction.rs:1428-1437). **Dead defensive code:** line 95-97 does `local.get(node).ok_or_else(...)?;` and discards the result, but `local` was built by zipping the same `nodes` list two statements earlier — the lookup cannot fail. |
| `src_tree.rs` | `fn contract` — global probe-index collection + zero-dimension guard | 105-118 | `SOURCED-PAPER(§2.2, `sec:krp`)` + `DERIVED-VERIFIED` (D-3) | Collecting *all* sites' physical output indices into one `ProbeBank` with one global column counter is precisely the paper's Khatri-Rao reuse requirement — "we use a common set of random matrices Ω^(1),…,Ω^(n-1) across all steps of the algorithm" (§3.5, discussion of Theorem 3). `sort_indices_deterministic` makes the RNG consumption order reproducible. See D-3 for why the reuse property is preserved across *edges* in the tree, which is the non-trivial part. |
| `src_tree.rs` | `fn contract` — `ProbeBank::new(.., 1, seed)` + `EnvironmentCache::new` | 119-121 | see the `src_probe.rs` / `EnvironmentCache` rows | Initial width 1; grown on demand. |
| `src_tree.rs` | `fn contract` — result containers | 123-126 | n/a (trivial plumbing) | The comment "Rooted edges are in child-to-parent postorder. Every projected child bridge therefore exists before its parent source is assembled" is **accurate** (verified above). |
| `src_tree.rs` | `fn contract` — per-edge loop: `source_factors`, `edge_bonds`, `cut_dimension`, `left_indices` | 128-145 | `DERIVED-VERIFIED` (D-1, step 3) | `source_factors = [A_child, B_child] ++ projected_children[child]` is the tree analogue of the paper's `B^(j)`; `left_indices` (all indices appearing exactly once, minus the two parent bonds) is the tree analogue of the paper's row split "(bond to η^(j+1), physical j)". `cut_dimension = dim(bond_a)·dim(bond_b)` is a valid and tight rank bound (D-4). |
| `src_tree.rs` | `fn contract` — scalar-subtree branch (`left_indices.is_empty()`) | 146-157 | `DERIVED-VERIFIED` (D-5) | A dim-1 structural bridge with `factor = ones([cap])`, which is a legitimate 1×1 isometry, so the QB step is exact by inspection. Reachable and covered by `src_preserves_scalar_only_subtrees_with_dimension_one_bridges` (tests/mod.rs:890). |
| `src_tree.rs` | `fn contract` — width selection (`row_dim`, `maximum_site_width`, `initial_width`) | 158-169 | `SOURCED-PYTHON(contraction.py:416-418)` + `SOURCED-PAPER(sec:adaptivity)` | See the `maximum_site_width`/`initial_width` rows under `src_probe.rs`. **Dead code:** the `else { site_max_width }` arm at line 168 is unreachable — `site_initial_width` is consumed only inside the `sketch_options.rtol.is_some()` branch at line 196. |
| `src_tree.rs` | `fn contract` — fixed-width batched sketch + QR | 170-196 | `SOURCED-PAPER(§3.1-§3.3, Algorithm 1 lines 5-6/10-11)` + `SOURCED-COMMENT(#563, 2026-07-29)` + `DERIVED-VERIFIED` (D-1, step 4) | `contract_retaining(source_factors ++ [environment], batch)` then `factorize_full_rank(left_indices, FactorizeAlg::QR, Canonical::Left)` — QR only, delegated to `tensor4all-core`'s typed factorization API. No SVD, no hand-rolled linear algebra. Row/column split matches Algorithm 1's `Y^(j)(a,b,c) = Σ_d η^(j)(d,b,c) R^(j)(a,d)` with `a` = sketch-column index. |
| `src_tree.rs` | `fn contract` — adaptive per-column path | 196-216 | `SOURCED-PAPER(Appendix C, `sec:adaptivity`)` + `DERIVED-VERIFIED` (D-1, step 4) | The `make_column` closure builds one sketch column from the cached per-column complement environment; column caching is in `EnvironmentCache::column`. Satisfies plan performance gate 4 ("cached environment columns are not recomputed during adaptive growth") — verified, each `(parent,child,column)` message is computed once and reused across the adaptive loop *and* across edges. |
| `src_tree.rs` | `fn contract` — projection `conj(factor) × source_factors` | 217-229 | `SOURCED-PAPER(§3.1/§3.2, "projection" step; Algorithm 1 lines 8/12)` + `DERIVED-VERIFIED` (D-1, step 5) | Exactly `B^(j-1) = (η^(j))^† B^(j)` generalized to a tree: `factor.conj()` shares `left_indices` with the source factors, so the contraction is the adjoint projection. The result's open indices are the two cut bonds plus the new cap — the same shape as the paper's `S^(j)(a,b,c)`. |
| `src_tree.rs` | `fn contract` — store factor + bridge | 231-236 | `DERIVED-VERIFIED` (D-1, step 6) | `projected` accumulates into `projected_children[parent]`; postorder guarantees availability. |
| `src_tree.rs` | `fn contract` — root/center assembly (`merge_projected`) | 238-247 | `SOURCED-PAPER(§3.3, Algorithm 1 line 14 "Determine the first site η^(1)")` + `DERIVED-VERIFIED` (D-1, step 7) | The paper's η^(1) = contract-down of `B^(1)` with one incoming `S^(2)`; the tree center absorbs *k* incoming bridges instead of one. This is the only genuinely new structural element versus the chain and it is the correct generalization (D-1, step 7). |
| `src_tree.rs` | `fn contract` — result `TreeTN` assembly | 249-258 | `DERIVED-VERIFIED` | Mechanical rebuild; edges reconnected via `connect_result_edge` on the same rooted-edge list, reproducing the input topology exactly. |
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
| `src_tree.rs` | `fn edge_bonds` | 488-514 | `DERIVED-VERIFIED` | Looks up the A-bond and B-bond of the `(child,parent)` edge in the two (sim'd) networks. These are the two legs crossing the cut, and their product is `cut_dimension`. |
| `src_tree.rs` | `fn directed_messages` (unbatched) | 516-578 | `DERIVED-VERIFIED` (D-1, step 2) + **F-1** | Two-pass belief-propagation-style sweep on a tree: upward (postorder) then downward (reverse postorder). Independently re-derived and confirmed correct, including the ordering guarantees. **F-1**: the `.or_else(\|\| messages.get(&(parent.clone(), neighbor.clone())))` at line 564 is dead and would be wrong if live. |
| `src_tree.rs` | `fn directed_messages_batched` | 580-641 | `DERIVED-VERIFIED` (D-1, step 2; D-7) + **F-1** | Identical structure with `contract_retaining(&factors, batch)` in place of `contract_factors`. Batched semantics verified to be per-column (diagonal in the retained index), not an outer product — see D-7. Same dead `.or_else` at line 626. Note this function lacks the two explanatory comments its unbatched twin carries (lines 527-528, 552-553), which is the *only* difference in documentation between two otherwise-parallel functions. |

### `src_probe.rs` (1171 lines)

| File | Code unit | Lines | Verdict | Citation / gap |
|---|---|---|---|---|
| `src_probe.rs` | Module doc comment (provenance claim) | 1-13 | `SOURCED-PAPER(§2.2 `sec:krp`)` + partially flagged (F-2) | "Section 2.2" for the `Ω^(1) ⊙ … ⊙ Ω^(n)` definition **is correct** — §2.2 is exactly `sec:krp`. "Algorithm 1" is correct. The Python line ranges `random_contraction` 82-353 / `random_contraction_inc` 357-593 are accurate to within 1-3 trailing lines (actual: 82-356 / 357-594). The "issue #563 comment 5396107820" attribution for the A/B probe partition matches Appendix A's 2026-08-24 comment. This header is materially more accurate than `src_tree.rs`'s. |
| `src_probe.rs` | Imports | 15-22 | n/a (trivial plumbing) | `rand::{Rng, SeedableRng}` — see F-3 on the absence of `rand_distr`. |
| `src_probe.rs` | `struct ProbeBank` + doc | 24-35 | `SOURCED-PAPER(§2.2)` + `DERIVED-VERIFIED` (D-3) | Doc claim "coefficients are stored as a column-major `dim × width` matrix" **verified accurate** against `extend_to`'s append order. Doc claim "an adaptive run observes exactly the same prefix as a fixed-width run with the same seed and index order" **verified accurate** (D-3) and tested (`probe_bank_extension_preserves_the_existing_prefix`). |
| `src_probe.rs` | `ProbeBank::new` | 37-70 | `SOURCED-PAPER(§2.2)` | Per-index Gaussian columns = the paper's per-site `Ω^(i) ∈ C^{d×χ̄}`. Rejects zero-dimensional and duplicated indices. Overflow-checked capacity. |
| `src_probe.rs` | `ProbeBank::width`, `::coefficients` | 72-80 | n/a (trivial) | Accessors. `coefficients` is used only by tests. |
| `src_probe.rs` | `ProbeBank::column` | 82-104 | `DERIVED-VERIFIED` | Bounds-checked column slice out of the column-major buffer. Checked arithmetic. |
| `src_probe.rs` | `ProbeBank::extend_to` | 106-123 | `SOURCED-PAPER(Appendix C `sec:adaptivity`, step 4)` + `SOURCED-PYTHON(contraction.py:430-431)` | "If χ̄ is larger than the number of columns in Ω^(1:j-1), then append columns to Ω^(1:j-1) until it has χ̄ in total." Python does the same with `for idx in range(len(envs), current_sketchdim)`. The append order (outer loop over columns, inner loop over `self.indices` in fixed order) is what makes the prefix property hold — D-3. |
| `src_probe.rs` | `fn standard_normal` | 126-130 | `SOURCED-PAPER(§2.2: "standard (real or complex) normal entries")` + `SOURCED-PYTHON(contraction.py:431 `np.random.randn`)` for the *real* choice; **`HANDROLLED-DUPLICATE` (F-3)** for the implementation | The distribution is right and real Gaussians are explicitly sanctioned by both the paper and the reference implementation. But Box-Muller is hand-rolled where `rand_distr::StandardNormal` is a workspace dep already used at `tensor4all-core/src/defaults/idx_tensor.rs:188`. Also discards the sine branch (2 uniforms per sample). |
| `src_probe.rs` | `fn single_probe` | 132-145 | `SOURCED-COMMENT(#563, 2026-08-24)` | One rank-1 probe vector `X[:,k]` as a **one-index tensor**. This is the factorized form Hiroshi asked for. Real-valued (`AnyScalar::new_real`), so the comment's `conj(X)` and the code's un-conjugated `X` agree identically — see D-8. |
| `src_probe.rs` | `fn single_probe_batch` | 147-177 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` | Same, as a `(index, batch)` two-index tensor over a contiguous column range. `first_column` is genuinely used with non-zero values by `src_chain.rs` (lines 548, 560, 632, 644), so this is not premature generalization. |
| `src_probe.rs` | `fn site_probe` | 179-206 | **dead production code (F-6)**; latent `SCOPE-DEVIATION` if ever wired up | Builds the *outer product* of all of a site's probe vectors as a single dense tensor of `∏ dim` entries — for MPO-MPO that is the explicit `d²` object Hiroshi's 2026-08-24 comment says never has to be formed. Zero production call sites (verified by grep across `crates/tensor4all-treetn/src/`); referenced only by this file's own test module. |
| `src_probe.rs` | `fn site_probe_batch` | 208-221 | **dead production code (F-6)** | Thin `first_column = 0` wrapper over `site_probe_batch_range`. Zero production call sites. |
| `src_probe.rs` | `fn site_probe_batch_range` | 223-271 | **dead production code (F-6)** | Batched version of the same fused construction, plus a scalar-outputs branch returning `ones([batch])`. Zero production call sites. |
| `src_probe.rs` | `fn product_dim` | 273-279 | n/a (trivial) | Overflow-checked product of index dimensions. |
| `src_probe.rs` | `fn probed_site_pair` | 281-307 | `SOURCED-COMMENT(#563, 2026-08-24T13:45:35Z)` — **exact match** | See D-8. `A × X(s)`, then `B × Y(t)`, then contract the two operands over the shared physical `u` (and any shared virtual bonds). Never forms a `d²` object. The doc comment's claim ("the shared physical leg is contracted between the operands without first constructing a fused `d^2` local product") is **verified true**. Numerically pinned by `probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`, which asserts against a literal `Σ_{s,t,u} A[s,u] B[u,t] x[s] y[t]` oracle. |
| `src_probe.rs` | `fn probed_site_pair_batch_range` | 309-349 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` | Same order, batched. The `outputs.is_empty()` branch (scalar sites) contracts the pair and broadcasts over the batch — correct, since a site with no physical output contributes a probe factor of 1 in every column. |
| `src_probe.rs` | `fn contract_prefix_with_site_pair` | 351-366 | `SOURCED-PYTHON(contraction.py:462-471)` | Comment claims the ordering "mirrors the reference `env @ psi[j]` then `H[j] @ ...`". **Verified true**: in `operator/apply.rs:405-409` the SRC dispatch passes `tn_a = transformed_state` (the MPS ψ) and `tn_b = full_operator.mpo()` (the MPO H), so `prefix × tensor_a` really is `env @ psi[j]`. Loose wording: it calls itself "the unbatched counterpart of `contract_prefix_with_probed_site_pair_batch_range`" when the exact unbatched-probed counterpart is `contract_prefix_with_probed_site_pair`. |
| `src_probe.rs` | `fn contract_prefix_with_probed_site_pair` | 368-396 | **dead production code (F-6)** | Zero production call sites; referenced only from this file's tests. Its INVARIANT comment is accurate about what it does. |
| `src_probe.rs` | `fn contract_prefix_with_probed_site_pair_batch_range` | 398-439 | `SOURCED-COMMENT(#563, 2026-08-24)` + `SOURCED-PYTHON(contraction.py:462-486)` + `DERIVED-VERIFIED` (D-9) | The **cost-correct** ordering: prefix into A, A-probes, then B, then B-probes. The in-code claim "Building the complete local MPO-MPO product first would expose both virtual bonds at once and costs O(chi^4) storage for a single probe block" is **verified true** by explicit index counting (D-9). Used only by `src_chain.rs`. This is precisely the ordering the tree path does *not* use — F-4. |
| `src_probe.rs` | `fn partition_probes` | 441-485 | `SOURCED-COMMENT(#563, 2026-08-24)` | Assigns each output index to the operand that carries it, and hard-errors on the two structurally-invalid cases (index in neither operand / in both). Unlike F-1's fallback, these errors are correct: an index in *both* operands would be a contracted leg, not an output, so `local_output_indices` would not have emitted it — but erroring is the right response, not silently guessing. |
| `src_probe.rs` | `fn contract_operand_with_probes` | 487-515 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` (D-7) | Unbatched branch contracts all probes at once; batched branch folds them in one at a time with `contract_retaining` so the batch axis stays diagonal. The `probes.is_empty()` early return handles the MPO-MPS case (only one operand carries an output index) correctly in both branches. |
| `src_probe.rs` | `fn contract_site_pair` | 517-528 | n/a (trivial) | Contract `A`, `B` and extra factors in one call; used by `src_chain.rs` only. |
| `src_probe.rs` | `fn contract_retaining` | 530-550 | `DERIVED-VERIFIED` (D-7) | Delegates to `T::contract_retaining_indices` (batched/diagonal semantics verified from its doctest at `tensor_like.rs:869-885`), then normalises the batch index to the trailing axis via `permuteinds`. The permutation branch is defensive but harmless. |
| `src_probe.rs` | `fn site_operands` | 552-575 | `DERIVED-VERIFIED`; **doc/code mismatch (AI-hallucination signature)** | Doc says "Contract corresponding tensors at every named site once." The function **contracts nothing** — it performs four lookups and returns a borrowed pair. Confident doc text describing behaviour the code does not implement. |
| `src_probe.rs` | `fn local_site_pairs` | 577-590 | n/a (trivial) | Map over `site_operands`. No doc comment. |
| `src_probe.rs` | `fn local_output_indices` | 592-643 | `SOURCED-COMMENT(#563, 2026-08-24)` + `DERIVED-VERIFIED` (D-8) | Collects the bond indices of both networks at the node, then takes the **symmetric difference** of the two operands' non-bond externals. For MPO-MPO `A:(s,u)`, `B:(u,t)` this yields exactly `{s,t}` and drops the contracted `u` — which is what makes the factorized probe well-defined. For MPO-MPS `A:(s,u)`, `B:(u)` it yields `{s}`. Redundantly recomputes `site_operands` that the caller already has (minor). |
| `src_probe.rs` | `fn fixed_probe_width` | 645-660 | `SOURCED-PAPER(§3.4 `sec:final_round`)` — **verbatim** | `max(⌈1.5χ̄⌉, χ̄+10)` when oversampling, else `χ̄`, capped by the row dimension. The paper: "As a sensible default, we recommend χ̄' = max(⌈1.5χ̄⌉, χ̄+10)." Exact. |
| `src_probe.rs` | `fn maximum_site_width` | 662-674 | `SOURCED-PYTHON(contraction.py:416)` + `DERIVED-VERIFIED` (D-4) | `min(max_rank, row_dimension, cut_dimension)` is the same triple bound as the reference's `current_maxdim = min(prod_bond_dims, maxdim, visible_dim * cap_dim)`. The Rust uses the *exact* parent-edge cut instead of the reference's looser `max(left,right)` site bound — tighter and still valid (D-4). Pinned by `maximum_probe_width_respects_the_exact_product_cut_dimension`. |
| `src_probe.rs` | `fn initial_width` | 676-679 | `SOURCED-PAPER(Appendix C `sec:adaptivity`: χ̄_0)` + `SOURCED-PYTHON(contraction.py:417-418)` | `min_rank.min(max).max(1)`; `SrcOptions::default().min_rank == 2` matches the paper's stated experimental χ̄_0 = 2. |
| `src_probe.rs` | `fn factorize_probe_columns` | 681-742 | `SOURCED-PAPER(Appendix C, `eq:err-est` + `eq:norm_est` + `sec:adaptivity` steps 1-5)` + `DERIVED-VERIFIED` (rank-deficiency stop, D-10) | The stopping test `estimate.error <= atol + rtol·estimate.norm` is **exactly** the paper's `Êrr^(j) ≤ τ_abs + τ_rel · N̂orm`. I verified the estimator's own definition end-to-end (`backend.rs:490-540`): `error = sqrt((1/p)·Σ_i ‖g_i‖^{-2})` with `G = R^{-†}` = `eq:err-est`, and `norm = ‖R‖_F/√p` = `eq:norm_est`. Growth by `rank_increment` (default 3 = the paper's Δ_χ) capped at `maximum_width` = `sec:adaptivity` steps 3-5. The extra `factorized.rank < width` saturation rule is not in the paper or the reference — derived and confirmed sound (D-10), and it is correctly short-circuited *before* the estimator is called (which requires a square `R`). |
| `src_probe.rs` | `fn connect_result_edge` | 744-776 | `DERIVED-VERIFIED` | Rediscovers the shared cap index via `index_ops::common_inds` and takes `.first()`. Verified that exactly one index can be shared between a child factor and its parent tensor (child factor holds its own physicals + grandchild caps + its cap; the parent holds only the cap of these). Mild fragility: the cap index is *known* at the QR site (`factorized.bond_index`) and discarded there (`let (factor, _cap) = ...`, src_tree.rs:171), then rediscovered by set intersection. Correct but avoidably indirect. |
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

## SVD / hand-rolled-linear-algebra audit

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

## Fused-`d²`-probe audit

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

## Detailed derivations and flagged findings

### D-1 — Full re-derivation of the rooted-tree SRC recurrence from the paper's chain recursion

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

### D-2 — The chain-delegation guard

`chain_order(center)` (contraction.rs:404-450) returns `None` unless the graph is
a path (exactly two degree-1 nodes, `edge_count == node_count - 1`). If `center`
is one of the two endpoints, the path is oriented so that `center` is `end`,
hence `chain.last() == Some(center)`; otherwise `center` is interior and the last
element is the other endpoint. So `src_chain::contract` is used exactly when the
paper's own one-sided recursion applies verbatim, and the rooted-tree recursion
otherwise. Correct, and the in-code comment describing this is accurate.

### D-3 — Khatri-Rao reuse is preserved across tree edges

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

### D-4 — `cut_dimension` is a valid and tight rank bound

At edge `(c,p)`, the matrix being QB-approximated is `B_c` unfolded with
`left_indices` as rows and the complement's physicals as columns. The network
factorizes through the two bonds `bond_a`, `bond_b` crossing the cut, so the
unfolding has the form `L · R` where the inner dimension is
`dim(bond_a)·dim(bond_b)`. Hence
`rank ≤ min( ∏ dim(left_indices), dim(bond_a)·dim(bond_b) )`,
which is exactly `min(row_dimension, cut_dimension)` in `maximum_site_width`.

The reference Python uses the looser
`prod_bond_dims = max(H[j].shape[0]*psi[j].shape[0], H[j].shape[2]*psi[j].shape[2])`
(contraction.py:413-415) — the max over both adjacent bonds, because in the chain
sweep it does not track which side is the cut. The Rust's per-edge cut is the
correct tightening for a rooted tree, where each edge is unambiguously *the* cut.
✔ `DERIVED-VERIFIED`, and consistent with `SOURCED-PYTHON` on the triple-min
structure.

### D-5 — The scalar-subtree branch

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

### D-6 — `EnvironmentCache`: correct, but see F-4/F-5

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

### D-7 — Batched contraction really is per-column (not an outer product)

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

### D-8 — The MPO-MPO factorized probe matches Hiroshi's 2026-08-24 comment exactly

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
`np.random.randn` (real) at contraction.py:431. Consistent across all three
Tier-1 sources. **However**, this is a latent trap: if anyone later switches the
probe bank to complex Gaussians (which §2.2 permits), the missing conjugate
becomes a real bug, and nothing in the code documents that dependency. Worth a
note in the fix pass; not a finding under this audit's taxonomy.

`probed_site_pair_contracts_mpo_mpo_outputs_before_pairing_the_physical_leg`
(src_probe.rs:880-917) pins this against a hand-written oracle of exactly
Hiroshi's `E_k` formula, which is the right kind of test.

### D-9 — The O(χ⁴) claim in `contract_prefix_with_probed_site_pair_batch_range` is true

The in-code comment (lines 424-429) claims that building the complete local
MPO-MPO product before folding in the prefix "would expose both virtual bonds at
once and costs O(chi^4) storage for a single probe block." Index counting, with
`χ_A = χ_B = χ` and sketch width `l`:

- *Pair-first:* `probed_site_pair(A,B)` has indices
  `(left_a, right_a, left_b, right_b)` → `χ⁴` entries, times `l` = `χ⁴ l`.
- *Prefix-first:* `prefix(left_a, left_b, batch)` × `A` → `(left_b, right_a, u,
  s, batch)`; probe `s` away → `(left_b, right_a, u, batch)` = `χ² d l`; × `B` →
  `(right_a, right_b, t, batch)`; probe `t` away → `(right_a, right_b, batch)` =
  `χ² l`.

Peak intermediate `χ² d l` vs `χ⁴ l`. The comment is **accurate**. ✔ This
function is the correct implementation. The problem is that the tree path does
not use it — F-4.

### D-10 — The `factorized.rank < width` early stop is sound

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

### D-11 — `mark_result_canonical` records a true orientation

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

## AI-hallucination-signature findings

### F-1 — Unreachable defensive fallback that would be silently wrong if reached (`SUSPECT-UNVERIFIED`)

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

### F-2 — Fabricated / inconsistent source citations (flagged citation error)

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

### F-3 — Hand-rolled Gaussian sampler duplicating existing repository functionality (`HANDROLLED-DUPLICATE`, minor)

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

### F-4 — The tree path materializes the full `2·deg(v)`-bond local pair at every node (`SCOPE-DEVIATION` vs plan performance gate 1; likely performance cause)

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

### F-5 — Batched environment cache keyed by width alone, causing repeated full tree sweeps (`SUSPECT-UNVERIFIED`)

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

### F-6 — Four `pub(super)` helpers with no production call sites (dead code / premature abstraction)

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

## Non-findings, recorded explicitly

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

## Cross-workstream dependencies flagged for the synthesis pass

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
