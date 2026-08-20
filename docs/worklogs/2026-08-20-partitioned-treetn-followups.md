# PartitionedTreeTN follow-up fixes (#655)

## Summary

Addressed the adversarial review of #651 (implementing #648) tracked in issue
#655. Fixed two correctness bugs (global `rtol` violation in adaptive
truncation, and equal-index-identity-with-different-dimension acceptance),
tightened option validation, made `add_with_patching` duplicate-key semantics
match its name, and pinned the contraction output-overlap limitation with a
regression test.

## Inputs reviewed

- `crates/tensor4all-partitionedtreetn/src/{patching,partitioned_tree_tn,subdomain_tree_tn,projector,error}.rs`
- `crates/tensor4all-treetn/src/treetn/{truncate,localupdate,mod,ops}.rs`,
  `crates/tensor4all-treetn/src/options.rs`,
  `crates/tensor4all-core/src/{truncation,defaults/index}.rs`,
  `crates/tensor4all-treetn/src/site_index_network.rs`
- Existing integration tests and module tests in the partitioned-treeTN crate.
- Shared agent rules: `common/repository.md`, `common/docs-and-tests.md`,
  `rust/numerical.md` (fetched online; sibling checkout absent).
- `docs/design/partitioned-treetn.md`.

## Reference reproduction

Before the fix, a single unprojected 4-site chain
`|0000> + a|1000> + a|0111>` with `a = 0.095` truncated by
`truncate_adaptive(..., rtol = 0.1)` accumulated the squared tails of several
independent per-bond SVD truncations and produced a measured global relative
error of `0.133 > 0.1`.

## Decisions

- **Per-bond budget split in `truncate_subdomain_with_budget`**: the patch
  squared budget is divided by the number of internal TreeTN edges before the
  truncation sweep, because the two-site sweep runs one SVD per internal bond
  and each SVD would otherwise reuse the whole patch budget. This bounds the
  sum of all local discards by the patch budget and preserves single-bond and
  single-node patches (which keep their full budget). Empirically the sweep is
  idempotent on already-truncated bonds, so dividing by the edge count (not by
  twice the edge count) gives the guarantee with minimal loss of compression.
  `add_with_patching`'s repeated `budget_truncate_for_split_decision` and
  `split_child_parameter_count` truncations inherit the same bound because they
  share this helper.
- **Dimension-compatible site identities**: `DynIndex` equality/hash exclude
  the dimension, so `ensure_same_tree_structure` and
  `validate_contraction_site_assignment` now additionally require equal
  dimensions for every shared index identity and return `SiteIndexMismatch`
  otherwise. Constructors and algebra can no longer accept a patch that shares
  a logical identity with a different dimension, which previously let
  `from_subdomains` silently discard the earlier patch.
- **Consistent empty semantics**: zero-node TreeTNs are rejected by
  `validate_data` (all `SubDomainTreeTN` construction and revalidation paths)
  with a typed `InvalidTopology` source. An empty `PartitionedTreeTN` remains a
  valid zero-norm object whose algebra (`add`, `contract`, `to_treetn`)
  requires operands and returns `Empty`.
- **`add_with_patching` adds equal keys**: input patches sharing an equal
  projector key are summed by strict subdomain addition before patching,
  matching the function name and converting the previous silent
  last-write-wins `[0,1]` into the expected `[1,1]`.
- **Option thresholds validated before shortcuts**:
  `validate_truncation_options` and `validate_contraction_options` now reject
  non-finite or negative `SvdTruncationPolicy` thresholds, closing the
  NaN-through-shortcut gap.
- **Incremental insertion without full revalidation**: `add` and `contract`
  build results with a private `insert_prevalidated` that checks only
  projector-key overlap, avoiding a full partition rescan (validate invariants
  + pairwise checks) per inserted patch.
- **Contraction output overlap pinned as a documented limitation**: two valid
  disjoint input partitions can contract into overlapping output regions
  (`{a=0}` and `{b=0}` are distinct keys that intersect in the full site
  space). The operation keeps rejecting these with `OverlappingProjectors` and
  the contract docs now state this limitation explicitly and point to refining
  the output space. This mirrors the deferred "common-refinement" item already
  recorded in `docs/design/partitioned-treetn.md`.

## Rejected alternatives

- Splitting the patch budget by twice the edge count to guard against
  hypothetical sweep double-visits: the sweep re-truncates each bond
  idempotently, and the two-times divisor meaningfully over-compresses the
  common single-bond patch for no measured benefit.
- Auto-refining overlapping contraction outputs: deferring to the existing
  common-refinement deferral item.
- Adding a public `from_subdomains_with_patching` wrapper instead of fixing
  `add_with_patching`: the fix keeps the descriptive name honest.

## Verification

- New regression tests, each confirmed to fail without its fix and pass with
  it:
  - `truncate_adaptive_*` and `add_with_patching_*` global-error tests measure
    the materialized dense relative error and require `<= rtol` on a multi-edge
    patch that previously measured `0.133`.
  - dimension-mismatch, non-finite-threshold, zero-node, duplicate-key
    `add_with_patching`, and contraction-overlap tests in
    `tests/partitioned_tree_tn.rs` / `tests/patching.rs`.
- `cargo test --release -p tensor4all-partitionedtreetn`, release doctests,
  and the existing `partitioned_tree_tn`/`patching`/`subdomain_tree_tn`
  suites pass.
- No public API names changed; only docs, validation behavior, the adaptive
  budget split, `add_with_patching` duplicate handling, and an internal
  insertion path changed.

## Remaining risks

- The per-bond budget split reduces truncation aggressiveness for multi-edge
  patches relative to the previous (over-aggressive) behavior; adaptive
  patching compensates by splitting over-cap patches. If a benchmark shows real
  parameter-count regression, revisit an exact global-budget TreeTN truncation
  rather than the uniform split.
- `docs/design/partitioned-treetn.md` no longer contains "formal review
  pending"; that text was already replaced by #654 before this change.
