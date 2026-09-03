# Caller-owned prepared contraction (#700)

## Summary

Added public `PreparedContraction`, which owns core index matching, internal
labels, retained-label output order, and result axis metadata for repeated N-ary
or retained-index `IdxTensor` contractions. Execution requires the same operand count, ordered
full indices, explicit dimensions, and axis classes; values, dtype, payload, and
gradient state may change. Compatibility failures return
`IdxTensorError::ShapeMismatch` before backend execution.

The implementation preserves ordinary single clone, binary pairwise, N-ary
native, retained, structured, and AD dispatch. Binary calls without retained
indices deliberately keep the faster pairwise path and do not reuse the stored
N-ary labels; their support is semantic parity, not a reuse claim. Backend shape/dtype/label plans
continue through the existing backend-owned cache/session. No new global or
thread-local core cache was added.

SRC was not wired to the concrete API: SRC is generic over `TensorLike` and
creates fresh batch/cap/link identities across growth and top-level runs. An
exact caller-owned plan therefore cannot survive those boundaries, while a
normalized validator would repeat the planner's contractability work. The SRC
non-regression gate was measured separately.

The design and final implementation received independent `Correct-to-merge`
verdicts. All review findings were fixed, including explicit dimension
validation, structured mixed-dtype/AD coverage, missing-retain and zero-execute
coverage, precise error docs, and metadata-summary `Debug`.

## Performance

Predeclared repeated-core gate: release, one thread, five fresh processes; each
process compares 2,000 calls, best of three, and requires at least 5% paired
median improvement. Generic/prepared ratios were `1.17x`, `1.15x`, `1.14x`,
`1.16x`, and `1.14x`; correctness was asserted before timing.

The first #706 SRC quick non-regression run (5 pairs, 3 repetitions) failed
because one very short tree-fixed baseline sample was 8.2% below its median.
A complete 10-pair/10-repetition rerun still had a within-limit +3.05% point
estimate but a 6.20% CI upper bound. The final predeclared stronger complete
rerun used 10 pairs and 30 repetitions with the unchanged 5% threshold and
10% dispersion limit; all four cases passed:

- chain fixed ratio `1.0126`, CI `[0.9715, 1.0368]`;
- chain adaptive `0.9747`, CI `[0.9680, 0.9997]`;
- tree fixed `0.9854`, CI `[0.9709, 1.0203]`;
- tree adaptive `0.9885`, CI `[0.9725, 1.0177]`.

## Verification

- `cargo test --release -p tensor4all-core --test prepared_contraction`: 8 passed.
- `cargo test --doc --release -p tensor4all-core PreparedContraction`: 3 passed after final review fixes.
- Ignored repeated-contraction benchmark: 5/5 performance pairs passed.
- #706 quick SRC non-regression: final 10-pair/30-repetition run passed.
- No tolerance changed.
