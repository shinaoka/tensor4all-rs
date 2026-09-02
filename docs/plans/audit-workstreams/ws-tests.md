# WS-tests — test coverage and tolerance integrity

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
[`docs/plans/2026-08-28-src-provenance-audit.md`](../2026-08-28-src-provenance-audit.md).
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

## §0. Methodological finding: two of the twelve files are not SRC work at all

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

## §1. Provenance table

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

## §2. Tolerance diff against `origin/main`

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

## §3. Tests that are real, not superficial (positive findings)

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

## §4. Cross-workstream note

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

## §5. Detailed derivations and flagged findings

### 5a. Dtype/topology/product-type coverage is much thinner than the Test-matrix cross-product implies

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

### 5b. The "chain reduction gate," from the test-file side (independent corroboration of WS-integration)

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

### 5c. `validate_ortho_consistency()` does not check numerical isometry — hallucination-signature: name/usage implies a property the assertion doesn't verify

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

### 5d. One verified tolerance-gap instance in `contraction/tests/mod.rs`; the cross-file "every new SRC test" generalization does not hold

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

### 5e. Missing reproducibility coverage

Grepped all 12 files for tests constructing two SRC runs and comparing
them: none exist. The plan's three reproducibility categories (same seed →
identical edge ranks/dense output; adaptive expansion preserves the first
*p* columns exactly; different seeds meet the same residual gate without
elementwise equality) are entirely absent from the diff. Every SRC test
that specifies a seed (`.with_seed(...)`) does so only to make its own
single run deterministic, not to compare two runs. `MISSING-VS-SOURCE`
relative to the Tier-2 map.

### 5f. Missing SRC-specific control-flow/error coverage

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

### 5g. `tensor_construction_supports_column_major_dense_payloads` — test name claims more than the assertion checks

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

### 5h. Trivial-plumbing `DERIVED-VERIFIED` rows (§1 rows 4, 15, 16, 18, 19)

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

### 5i. Row 5 derivation: `src_adaptive_contracts_and_honors_rank_cap` cannot show the cap actually binds

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

### 5j. Row 11 derivation: `apply_linear_operator_src_preserves_a_two_site_identity`

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

### 5k. Row 12 derivation: `partial_contract_src_uses_the_same_directed_tree_path`

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

### 5l. Row 14 derivation, and its one exception

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

## §6. Supplementary note: fused-`d²`-probe structural regression test (outside this workstream's file scope)

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

## §7. Summary of MISSING/flagged items for Task 7 synthesis

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
