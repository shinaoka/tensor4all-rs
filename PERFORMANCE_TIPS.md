# Performance Audit Catalog

Failure modes where tensor4all-rs code is correct but wastefully slow because
the obvious API was used instead of the right one. Read it during
performance-sensitive implementation (TT evaluation, interpolation, caches,
contraction hot paths) and in every review touching those areas. Findings are
rule violations, not measured regressions: a fix still follows the measurement
discipline of the shared rules. Generic performance rules live in
`tensor4all-agent-rules` `rules/common/performance.md` and
`rules/rust/performance.md`; this file does not repeat them. Dense
materialization, index identity, and layout rules live in
`REPOSITORY_RULES.md` and `AGENTS.md`.

Ported from Tensor4all.jl `rules/tensor4all-usage-audit.md` (FM1 to FM12).

## Audit procedure

1. Scope: the pending diff by default; `full` or a path means every `.rs` file
   under it. Skip generated code and `target/`.
2. Run each section's "Detect" grep over the scope. Hits are candidates, not
   findings.
3. Read each candidate in context. Keep only sites on a path repeated over many
   index sets, sweeps, or calls where the "Fix" API applies. Record justified
   exceptions.
4. Classify each finding by section. Check that the proposed fix is not itself a
   violation (for example, replacing a pointwise loop with `to_dense()`).
5. Report `file:line`, violated section, evidence, remediation; most severe
   first. Do not edit code; do not claim a speedup without measurement.

## Pointwise TT Readout In A Loop

Contract: repeated `evaluate(&idx)` on a tensor train recomputes the full
left-to-right contraction per point. Any path that evaluates one TT at many
indices shares left/right environments. Incident: Tensor4all.jl ReFrequenTT,
2026-09-01, batch readout was 1353x faster at rank 31.

Detect: `rg -n '\.evaluate\(' crates` inside `for`, `map`, or `iter` over
index sets; `QuanticsTensorCI2::evaluate` in a loop.

Fix: `tensor4all_simplett::TTCache` (`TTCache::new(&tt)`, then
`evaluate_many(&indices, split)`), or `evaluate_left`/`evaluate_right` for
one-sided scans. Trees: `tensor4all_treetn::TreeTNCachedEvaluator` with
`evaluate_batched`. For `QuanticsTensorCI2`, wrap `tensor_train()` in a
`TTCache`.

## Uncached Target Function Across Sweeps

Contract: TCI sweeps re-request the same function values across pivot searches
and runs. An expensive target passed raw to `crossinterpolate2` or
`quanticscrossinterpolate` recomputes them; related runs with separate caches
recompute across runs.

Detect: `rg -n 'crossinterpolate2\(|quanticscrossinterpolate\(' crates`; check
whether the closure calls a raw function or a `CachedFunction`.

Fix: `tensor4all_core::CachedFunction::new(f, &local_dims)` and `eval`; share
one instance across related runs; inspect `num_evals`, `num_cache_hits`,
`cache_hit_ratio`.

## Wrapper That Disables Batching

Contract: batch entry points exist but fire only when the caller hands them the
batch. `tensor4all_tensorci::crossinterpolate2` takes `batched_f: Option<B>`;
`tensor4all_treetci::crossinterpolate2` takes `F: Fn(GlobalIndexBatch) ->
Result<Vec<T>>`; `CachedFunction::with_batch` and `eval_batch` exist. A closure
`|idx| cf.eval(idx)` or a per-element loop inside the batch closure silently
turns batching into pointwise evaluation.

Detect: `rg -n 'crossinterpolate2\([^)]*None' crates`; batch closures whose
body loops `.eval(` or `.evaluate(`; `CachedFunction` values captured by a
plain `Fn(&[usize])` closure.

Fix: pass a batch function; inside batch closures call `eval_batch` or
`evaluate_many`; add behavior by wrapping the batch path, not by closing over
the pointwise one.

## Full Quantics Grid Sweeps

Contract: a quantics grid has `2^R` points; iterating all of them in a
production or validation path defeats the representation. Bounded sampled
checks or structural checks replace exhaustive sweeps.

Detect: `rg -n '1usize << |1 << r|pow\(2|2usize\.pow|iproduct!|grid\.len\(\)'
crates` inside loops that evaluate a TT or grid function.

Fix: random sampled points via `ChaCha8Rng`, batch readout of a bounded index
set (`TTCache::evaluate_many`), or a structural invariant. Keep exhaustive
sweeps to tests with small `R`.

## Per-Call Reconstruction Inside Loops

Contract: grids, TCIs, caches, evaluators, and backends are built once and
reused. Construction inside a loop or per-point helper pays constant overhead
per call and gives a zero cache hit rate.

Detect: `rg -n 'TTCache::new\(|CachedFunction::new\(|TreeTNCachedEvaluator::new\(|DiscretizedGrid::|InherentDiscreteGrid::|CpuExecutionContext::from_backend\(|with_default_backend\('
crates` inside `for`, `map`, or closures called per point.

Fix: hoist construction out of the loop; reuse a `CachedEvaluatorPlan` via
`TreeTNCachedEvaluator::with_plan` when tensors change but topology does not;
build one `CpuExecutionContext::from_backend` and reuse it, as `AGENTS.md`
requires.

## Owned Vector Cache Keys

Contract: a persistent cache keyed by `Vec<usize>` hashes and compares the
whole vector on every lookup and clones it on every insert. Multi-indices are
encoded as mixed-radix flat integers with width chosen from the index space.

Detect: `rg -n 'HashMap<Vec<usize>|HashMap<MultiIndex|BTreeMap<Vec<usize>'
crates` on long-lived caches.

Fix: `CachedFunction` (`u64`, `u128`, then `U256`/`U512`/`U1024` via the
`CacheKey` trait, `with_key_type` to force a width) or `TTCache`, which encode
keys internally. Do not hand-roll a multi-index cache.

## Silent Slow-Path Fallback

Contract: a fast path (batch readout, structured op, backend call) that falls
back to pointwise or dense behavior on error hides the failure and pins
production on the slow path.

Detect: `rg -n 'unwrap_or_else\(|Err\(_\) =>|if let Err\(_\)' crates` where the
fallback branch re-implements the operation pointwise or densely; feature-gated
fallbacks with no log.

Fix: return the typed error, or route to a named reference path (`*_dense`,
`*_reference`) that the caller opts into and that logs the degradation.

## Unpinned Timing Claims

Contract: timing comparisons without pinned thread counts are irreproducible
(oversubscription, drift). Every quoted number states its thread settings.

Detect: `rg -n 'Instant::now\(|criterion|cargo bench' crates benchmarks`
without the thread environment.

Fix: run as in `benchmarks/README.md` (`RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`), record thread
counts for scaling runs, and follow the Evidence rules in
`rules/common/performance.md`.

Not ported: FM4 (dense materialization) is `REPOSITORY_RULES.md` "No Hidden
Dense Materialization In Production Paths"; FM8 to FM10 (grid conventions,
FFI memory order, tolerance semantics) are convention rules in `AGENTS.md`
and `skills/use-tensor4all-rs/`.
