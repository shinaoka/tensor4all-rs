# Issue #566 — PR 4 Evidence Ledger and Final Audit

Date: 2026-08-11
Basis: `origin/main` at 69a24e7 (2026-08-02 audit), remediated across PR #589,
#590, #593, #595, and this PR.

## Merged remediation pull requests

| PR | Scope | Merge commit |
|----|-------|--------------|
| #589 / #590 | Phase 0 soundness, Phase 1 CI gates, review-bot LLM removal | `a6a51de` (PR2 rollup) |
| #593 | PR 3 typed errors: public `anyhow::Result` → zero, layering, test hygiene | `5376317` |
| #595 | PR 3 vocabulary/stack seam/architecture (this program's PR 3) | `7d263fa` |
| this PR | PR 4: performance evidence, ledger, final audit | — |

## Phase 0 — soundness and CI hotfixes

1. **Re-enable always-on workspace tests** — DONE (PR #589: release tests restored on all CI paths).
2. **Checked shape products in `Matrix` constructors** — DONE (PR #589: `checked_mul` in `matrix.rs` constructors).
3. **Promote `debug_assert` to real validation on public `rrlu` path** — DONE (PR #589).
4. **Fix silent data corruption in quanticstci** (failed origcoord conversion) — DONE (PR #589: error propagation).
5. **`Matrix` `Index`/`IndexMut` out-of-range aliasing** — DONE (PR #589: row bound check).
6. **capi checked arithmetic** (`from_raw_parts`, dim products) — DONE (PR #589).
7. **hdf5 file-derived integers** — DONE (PR #589: validated conversion, no panic/OOM).
8. **quanticstransform shift overflow** (`1usize << nvariables`) — DONE (PR #589).

## Phase 1 — enforcement gates and housekeeping

9. **Wire `scripts/audit-library-panics.py` into CI** with baseline — DONE (PR #589; baseline regenerated in PR #595).
10. **clippy `missing_errors_doc` / `missing_panics_doc` gate** — DONE (PR #589; `check-public-error-docs.py` ported).
11. **Crate-boundary checker** — DONE (PR #589; `check-crate-boundaries.py`).
12. **Fix the two prohibited `no_run` doctests** — DONE (PR #589).
13. **Delete committed debris** (`debug.md`, `plan/`, orphan test, `coverage-local.json`) — DONE (PR #589).
14. **Remove unused `kryst` dependency + stale GMRES claim** — DONE (PR #589).
15. **Coverage in release + rationale for threshold clusters** — DONE (PR #589/#590; `coverage-thresholds.json` documented).
16. **repository-rules-review bot** — DONE (PR #568 merged, LLM removed in #590; label `rules-review:no-llm`).

## Phase 2 — rules adoption by reference

17. **Adopt `rules/common/agent-consumers.md` by reference** — DONE: downstream-usage skill (`skills/use-tensor4all-rs/`), remedy clauses in error messages (#593 `# Remedies` rustdoc), verified `llms.txt` (PR #595; all 10 links verified).
18. **Replace overlapping REPOSITORY_RULES.md sections with references** — DONE (this PR: header now routes to `tensor4all-agent-rules`; file keeps tensor4all-rs-specific rules; `AGENTS.md` references shared rules by URL + sibling checkout).
19. **Propose tenferro assets for the shared repo / adopt locally** — DONE locally: Performance-Gated Experiment Protocol applied in this PR (measure first, implement only material shares); work-log PR-body requirement followed; final cross-phase audit performed (this ledger).
20. **Per-repo vocabulary decisions** — DONE (PR #595: unsuffixed/`_mut`/`_into`/`_batched`; retire `_inplace`/`_in_place`/`scaled`; `_owned` kept only for input-ownership optimization; `max_bond_dim: Option<usize>`; `SvdTruncationPolicy`; documented in `CONTRIBUTING.md` + `docs/book/src/architecture.md`).
21. **Add CONTRIBUTING.md + external-contribution intake** — DONE (PR #595).

## Phase 3 — violations of rules this repository already states

22. **tensorbackend single tenferro route** — DONE (PR #593: sanctioned exceptions documented in `check-crate-boundaries.py`).
23. **`#[doc(hidden)] pub` FullPivLuScalar reach-through** — DONE (PR #593: moved to tensorbackend, `BackendLinalgError`).
24. **capi Index Identity rule** (`t4a_index_new_with_id`/`t4a_index_id`) — DONE (PR #593: serialization semantics documented; union-find plumbing made private).
25. **Test comparison rule (per-element re-evaluation loops)** — DONE (PR #593: remaining loops are data construction / point evaluation / finite differences; tensor comparisons use `.sub().maxabs()`/`.distance()`).
26. **anyhow migration, ordered by leverage** — DONE (PR #593: zero public `anyhow::Result` across the workspace).
27. **Remove `evaluate_at` compatibility alias** — DONE (PR #595).
28. **Graph Algorithms rule** — DONE (PR #593: deduplicated `transform.rs` DFS onto `post_order_dfs_by_index`, justifications added).
29. **Work-log discipline** — DONE: work logs maintained in `docs/superpowers/`; this ledger is the PR 4 record; PR bodies link work logs.

## Phase 4 — API unification

30. **Resolve the four semantic traps** — DONE (PR #595: `TreeTN::scale` split into `scale`/`scale_mut`; `evaluate` unsuffixed + `_batched`; `truncate`/`compress` documented; `norm` documented).
31. **Collapse duplicate types** — DONE (PR #595: simplett positional type renamed `SimpleTensorTrain`; naming resolution documented; no compat aliases).
32. **Julia-style concatenated names to snake_case** — DONE (PR #595: `link_indices`, `site_indices`, `max_bond_dim`, `full_tensor`, `min_dim`, `replace_indices`, …).
33. **Unify densification / inner-product / canonicalization / options vocabulary** — DONE (PR #595: `inner_product` everywhere; `full_tensor` vs `to_dense` documented; options `max_bond_dim`/`SvdTruncationPolicy` unified; `canonicalize` vs `orthogonalize` documented per stack).
34. **Remove scalar-suffixed Rust entry points outside capi** — DONE (PR #595: `native_tensor_primal_to_diag<T>`/`dense_col_major<T>` generics; `as_slice_f64/c64` removed; `as_`/`is_` dtype accessors documented as exempt).

## Phase 5 — structural decisions

35. **Record the two-stack decision and manage the seam** — DONE (PR #595: `docs/book/src/architecture.md` two-stack/no-facade diagram; `treetn::simplett_bridge` sanctioned as the only crossing; partitionedtt + quanticstci routed through it; MPO→LinearOperator exception documented; stack default stated).
36. **Fix the inverted layer (core depends on tcicore)** — DONE (PR #593: CI-factorization default seam direction documented in `core/defaults/factorize.rs`; tcicore independent/acyclic; boundary script enforces).
37. **Shrink treetn's surface** — DONE (PR #595: all 14 submodules private; root re-exports are the public surface; dead exports removed; no speculative crate split).
38. **Re-export or wrap tenferro types simplett's constructors take** — DONE (this PR: `pub use tenferro_tensor::TypedTensor` re-exported from `tensor4all-simplett` with doc example; downstream users no longer hand-pin a tenferro rev).
39. **Document the no-facade decision and layer diagram** — DONE (PR #595: architecture.md).

## Performance candidates — end-to-end measurement

All candidates measured in release builds on this machine (perf unavailable:
`perf_event_paranoid=4`; instrumentation timing used). Harness:
`benchmarks/rust/issue566_perf_candidates*.rs` (examples, reproducible).

| # | Candidate | Workload | Measurement | Share | Decision |
|---|-----------|----------|-------------|-------|----------|
| 1 | `TreeTN::evaluate` rebuilds evaluator per call | chain TT 32 sites/bond16, batch 200×20 and per-point 4000 | rebuild vs reused: 1.9% (batch), 0.8% (per-point) | 0.8–2% | immaterial — no change |
| 2 | capi cached-evaluator rebuild per FFI call | 16 sites/bond8, 32-pt batch, 2000 calls | 6.6% (small) → 2.0% (128 sites/bond64) | 2–7%, shrinks with size | immaterial at scale — evaluation dominates; no FFI handle change |
| 3 | whole-TT evaluation in global pivot search | `floating_zone`, 16–128 sites | `tt.evaluate` = 93.7–100% of `floating_zone` time | evaluation itself dominates | resolved by evidence: per-candidate re-evaluation is the intrinsic cost; sequential pivot updates do not batch into `TTCache` |
| 4 | SimpleTT per-site allocation / slice copy | chain TT 64/128 sites, 1000 points | `TTCache::evaluate_many` is 25× (64 sites) / 69–107× (128 sites) faster than per-point `evaluate` | cache reuse is the fix | resolved by existing abstraction: use `TTCache` for multi-point evaluation; zero-init removal (push) measured 40% *slower* — reverted |
| 5 | `contract_profile_enabled()` re-reads env twice per `contract` | `env::var` ×1e6 (77–84 ns/call); `contract(4×4)` = 52.8 µs/call | 2 × env::var = 169 ns = **0.32%** of a contract call | 0.32% | immaterial — no change |

## Final audit

- 39/39 Phase 0–5 checklist items have a disposition; all required behavior
  items are DONE (no item is deferred, narrowed, or "good enough").
- Residual risk: none for required behavior. Performance candidates 3/4 are
  resolved by recorded evidence and the existing `TTCache` abstraction, per
  the Performance-Gated Experiment Protocol ("immaterial share is resolved by
  the recorded evidence rather than speculative code").
- Validation: `cargo fmt --check`, clippy `-D warnings -D missing_errors_doc
  -D missing_panics_doc`, `cargo nextest run --release --workspace` (+hdf5),
  doctests, `./scripts/test-mdbook.sh`, `check-public-error-docs.py`,
  `check-crate-boundaries.py`, `audit-library-panics.py`, coverage
  (CI-owned gate, 210/210 files pass).
