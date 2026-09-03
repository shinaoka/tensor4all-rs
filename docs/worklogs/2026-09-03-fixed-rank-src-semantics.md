# Fixed-rank SRC rank semantics (#705)

## Decision and implementation

`max_bond_dim` is documented as an upper bound. Fixed SRC continues to clamp
each requested sketch width to the local row dimension and exact opposite-cut
support in `maximum_site_width`; no redundant rank-padding or new public API was
added. Adaptive `SrcOptions::max_rank` semantics are unchanged.

The reviewed design is
`docs/plans/2026-09-03-fixed-rank-src-semantics-design.md`. After adding explicit
chain/tree and fixed/adaptive distinctions, the independent reviewer verdict was
`Correct-to-merge`.

`benchmark_src` now prints `requested_max_rank` and `effective_max_bond`, so an
oversized benchmark request cannot be mistaken for realized work.

## Coverage

A shared deterministic test exercises both a chain and branched tree with a
requested rank below, equal to, and above cut support, plus the oversampled
`final_svd` fixed path. Full-support cases are compared to one dense oracle and
all realized bonds are checked against both request and support. The private
width-selection test separately pins row and cut clamping.

## Verification

- `cargo test --release -p tensor4all-treetn src_fixed_ --lib`: 4 passed.
- `cargo test --release -p tensor4all-treetn maximum_probe_width_respects_rank_row_and_cut_bounds --lib`: 1 passed.
- `cargo run --release -p tensor4all-treetn --example benchmark_src -- 3 2 1 mpo-mps 1 false 16`: passed; reported requested rank 16 and effective SRC bond 4 with relative error `6.525e-16`.
- No tolerance changed.
