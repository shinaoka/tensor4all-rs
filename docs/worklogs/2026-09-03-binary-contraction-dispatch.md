# Binary contraction dispatch for issue #698

## Summary

The generic borrowed contraction entry now routes connected two-tensor calls
without retained indices through the existing pairwise implementation. SRC's
statically binary call sites were returned to `TensorLike::contract`, so new
generic algorithm code receives the fast path automatically.

This closes item 1 of [issue #698](https://github.com/tensor4all/tensor4all-rs/issues/698).
The prepared-plan API and the other independent umbrella items remain deferred.

## Sources and decision

- Baseline: `origin/main` at `b9636123`.
- Read `docs/worklogs/2026-08-31-src-performance-remediation.md`, the issue,
  `contract_with_options_impl`, the pairwise implementation, SRC binary call
  sites, and the existing N-ary/pairwise equivalence tests.
- Connectivity and retained-index validation remain before dispatch, preserving
  the generic entry's rejection of disconnected networks and retained-label
  semantics.
- Restricting dispatch to tensors with trivial axis classes did not recover the
  SRC path because some dense SRC intermediates retain structured axis metadata.
  The pairwise implementation already owns structured and AD-preserving paths,
  so every connected binary no-retain call uses that canonical seam.

## Performance gate

Predeclared gate: 10-site MPO--MPS, input bond and maximum rank 32,
`rank_increment=3`, release profile, 20 repetitions, fixed
`RAYON_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`; generic dispatch must be within 5% of explicit
`contract_pair`.

Five alternating process pairs used binaries built from the same candidate,
with only the SRC call sites differing:

| Method | Explicit median ms/run | Generic median ms/run | Ratio-of-medians change | Median paired change |
|---|---:|---:|---:|---:|
| SRC fixed | 31.284 | 31.684 | +1.28% | +0.62% |
| SRC adaptive | 30.627 | 30.298 | -1.07% | +0.06% |

Both cases pass the 5% gate. A one-repetition dense-oracle run of adaptive SRC
reported relative error `1.755e-23`.

## Verification

- `cargo test --release -p tensor4all-core --lib`: 442 passed, 1 ignored.
- `cargo test --release -p tensor4all-core --test tensor_contract_nary_pair_equivalence`: 4 passed.
- `cargo test --release -p tensor4all-core --test ad_integration`: 6 passed.
- `cargo test --release -p tensor4all-core --test tensor_contraction`: 28 passed.
- `cargo test --release -p tensor4all-core --test tensor_diag`: 25 passed.
- `cargo test --release -p tensor4all-treetn 'treetn::contraction::' --lib`: 75 passed.
- No tolerance was changed.
