# Fixed-rank SRC rank semantics design (#705)

## Decision

Treat `ContractionOptions::max_bond_dim` as an upper bound, not a promise that
every output bond has that exact dimension. Fixed-rank SRC uses, at each cut,

```text
effective_sketch_width = min(requested_max_bond_dim,
                             local_row_dimension,
                             opposite_cut_support)
```

In fixed mode only, enabling `final_svd` first oversamples the requested width
and then applies the same local-row and cut-support bounds. Adaptive mode keeps
its separate `SrcOptions::max_rank` cap and is unchanged by this decision. The
realized output bond may be smaller than the requested maximum.

## Alternatives

- **Reject above-support requests:** rejected because `max_bond_dim` is a global
  cap while support varies by cut; callers should not need to precompute every
  internal cut merely to provide a safe cap.
- **Honor the exact request:** impossible without adding zero/redundant columns
  beyond mathematical support and would preserve the benchmark's avoidable work.
- **Clamp to support:** selected; it matches maximum-rank vocabulary, existing
  `maximum_site_width` behavior, and other truncation APIs.

## Contract owner and implementation

- `maximum_site_width` remains the single owner of per-cut clamping.
- `chain_cut_dimensions` and the corresponding rooted-tree edge dimensions
  compute opposite-side support with checked multiplication.
- Public `SrcOptions`/`ContractionOptions` docs state cap semantics explicitly.
- `benchmark_src` reports both `requested_max_rank` and the realized
  `effective_max_bond`, avoiding oversized-request ambiguity.
- No compatibility shim or new public API is needed.

## Tests

Use deterministic small fixed-SRC chains and branched trees whose support is
known. For each topology cover:

1. requested rank below support: realized bonds do not exceed the request;
2. requested rank equal to support: exact fixed contraction is recovered;
3. requested rank above support: result matches the support-sized request and
   does not create larger bonds;
4. fixed mode with `final_svd` still cannot exceed local or cut support;
5. existing zero/overflow validation remains unchanged.

Compare small results through one dense materialization and a whole-result
residual. Do not relax tolerances. Adaptive semantics and tests are not changed
by #705.
