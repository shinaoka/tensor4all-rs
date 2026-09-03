# SRC benchmark gates (#706)

## Summary

The invalid unbounded-degree star fixture was replaced by a deterministic
maximum-degree-3 binary tree. `benchmark_src` now performs checked input/oracle
memory preflight before allocation, warms each selected algorithm once, and
emits machine-readable build, configuration, memory, timing, rank, seed, and
correctness records.

`scripts/run-src-benchmark-gates.py` provides candidate smoke and paired
baseline/candidate modes with fixed one-thread settings, binary identity checks,
SHA-256 records, separate RLIMIT_AS/peak-RSS controls, alternating order,
median/MAD statistics, deterministic bootstrap intervals, and explicit
PASS/FAIL/INCONCLUSIVE classification. Its design and final diff received
`Correct-to-merge` verdicts after all review findings were fixed.

## Backend lanes

- Default tenferro/faer CPU is the required SRC lane.
- Provider-injection compiles, but SRC has no caller-owned execution-context
  seam; provider-specific SRC timing is recorded as unsupported rather than
  inferred from a feature build.
- CUDA hardware was available (NVIDIA A100 80GB). Existing resident TreeTN CUDA
  tests pass, but SRC itself remains unsupported on CUDA because generic probe
  construction and factorization are CPU/context-free. The command and missing
  seam are documented in `benchmarks/README.md`.

## Verification

- `python3 scripts/test-run-src-benchmark-gates.py`: 7 passed.
- `cargo test --release -p tensor4all-treetn --example benchmark_src`: 3 passed.
- Quick candidate smoke (4 exact cases, one warm-up + one timed run each): PASS.
  - chain bond-32 fixed: relative error `6.562e-26`;
  - chain bond-32 adaptive: `1.698e-23`;
  - binary-tree n=7 bond-4 fixed: `3.719e-18`;
  - binary-tree n=7 bond-4 adaptive: `3.845e-18`.
- One-byte input limit rejected the tree fixture before allocation (exit 101,
  estimated 10,240 bytes).
- `cargo check -p tensor4all-treetn --no-default-features --features simplett-bridge,tenferro-provider-inject`: passed.
- `cargo test --release -p tensor4all-treetn --features tenferro-cuda --test cuda_tree_contraction -- --nocapture`: 9 passed on A100.
