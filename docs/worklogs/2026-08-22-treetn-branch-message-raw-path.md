# Closing TreeTNCachedEvaluator's branch-point gap (tensor4all-rs#671 follow-up)

Work log. Continues from `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`,
which fixed a branch-point batching gap in `tensor4all-treeaci/frames.rs`. A
downstream re-run of gw-rs's real R=10 comb pipeline (`aci_global_guard=true`,
the same config issue #671 used) showed that fix alone left `pi_rtau`'s wall
time essentially unchanged (103.6s, matching the pre-fix baseline). This
session traced that to a second, independent, structurally analogous bug in a
different crate: whenever `aci_global_guard=true`, global pivot search
(`crates/tensor4all-treeaci/src/global_guard.rs`) evaluates candidate points
through `TreeTNCachedEvaluator` (`tensor4all-treetn`), not through
`frames.rs` at all -- so the first fix never touched this path.

## Root cause

`TreeTNCachedEvaluator::get_or_compute_node_message` dispatches to a raw
`Vec<f64>`-based fast path (`try_compute_chain_message_raw`,
`grouped_chain_message_contraction`'s BLAS "chain kernel") only for a node
with exactly one child in the rooted message-passing tree -- the same
"one-incoming/one-child gets batched, branch doesn't" shape as the
`frames.rs` bug, independently implemented in this crate for message-passing
evaluation rather than local-update candidate contraction. A node with two
children falls through to the generic `compute_stacked_message`
(`IdxTensor` + `contract_with_options`).

A pure chain or `SimpleTensorTrain`/`TTCache`-based train never exercises the
multi-child branch at all: rooting a chain anywhere gives every non-root node
at most one child (the root's own combination goes through a separate
one-shot "centre contraction" step, not this per-node recursion), and
`SimpleTensorTrain`'s data structure cannot represent a branch in the first
place. Tree ACI on a genuinely branching topology (gw-rs's 3-arm comb, one
degree-3 hub) is the first caller that visits this code path at scale.

## Confirming the bottleneck before fixing it

A temporary `#[ignore]`d diagnostic
(`diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk`, kept
in `crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`'s test module,
mirroring the file's own `message_cache_wall_time_on_realistic_floating_zone_walk`
benchmark) measured a 16-site chain (bond=128) against a same-size comb tree
(one degree-3 hub, three 5-site arms, bond=128), both under the real
`NSEARCH=5`, `MAX_SWEEPS=100` floating-zone-walk call pattern
`find_global_pivots` actually uses: **83.7ms vs 2.16s -- 25.84x**, at a scale
close to the real gw-rs workload's observed bond (~90).

## Fix

Added (mirroring the chain kernel's shape, but not its exact math -- see
"Why this isn't a cartesian-product fix" below):

- `BranchContractionSpec` + `grouped_branch_message_contraction`
  (`crates/tensor4all-treetn/src/treetn/cached_evaluator.rs`): contracts a
  branch node's raw tensor data against two children's raw message columns
  in two steps per physical-value group -- one shared-matrix `mat_mul_owned`
  call folds in the first child (the node's tensor slice at that physical
  value is the same for every point in the group, so every point's child-1
  column batches into one matmul), then a vectorized accumulate loop over
  `child_dim_2` folds in the second child.
- `try_compute_branch_message_raw` / `_complex_raw`, wired into
  `get_or_compute_node_message` between the existing chain attempt and the
  `compute_stacked_message` fallback, for both dispatch branches.

**Why this isn't a cartesian-product fix.** The `tensor4all-treeaci/frames.rs`
fix batched a full cartesian product of candidates in one shot, because that
crate's candidate lists are exactly that. This evaluator's `points` are
independent per-point assignments instead -- point `p` names one specific
`(child1_assignment, child2_assignment)` pair, not every combination -- so
only the first child's contraction reduces to a single BLAS call per group;
the second child's contraction cannot share a matrix across points the same
way and is folded in via a plain vectorized loop instead.

## A regression, caught before it shipped

The first version of `grouped_branch_message_contraction` copied the chain
kernel's `point_count < 2 * MIN_GROUP_POINTS` gate verbatim. That gate is
wrong for the branch case: a branch step's per-point work is
`O(parent_dim * child_dim_1 * child_dim_2)`, an extra bond-dimension factor
over the chain kernel's `O(parent_dim * child_dim)`, so at realistic bond
dimensions even a single point already has enough work to justify BLAS.
Re-running the diagnostic after wiring the fix in showed **44.24x** --
*worse* than the 25.84x baseline -- because `floating_zone_walk`'s typical
1-2-point batches always fell below the gate and always took the naive
`scalar_branch_message_contraction` fallback, which is slower per flop than
whatever `compute_stacked_message` was already doing. The fix: drop the
point/group-count gate entirely and key `scalar_work >= BRANCH_BLAS_WORK_THRESHOLD`
alone, since a branch step's larger per-point cost means total work already
captures whether BLAS is worth it. Re-measuring after that correction gave
29.26x -- better, but still not clearly beating the 25.84x baseline on this
same-bond-for-both synthetic diagnostic.

**That synthetic result was misleading, and real downstream data caught it.**
The 25.84x/29.26x/44.24x numbers above all use bond=128 for *both* chain and
comb, chosen to isolate topology's effect at equal bond. But gw-rs's actual
saved run logs (`runs/R10_cttn_T0.1_mu0.5/cttn/state/run.log` vs
`runs/R10_nblock_T0.1_mu0.5/treeaci/state/run.log`, both pinned to this
session's `frames.rs`-only-fixed commit) show NBlock's own max bond (163) is
*larger* than CTTN's (90) -- the opposite of the synthetic diagnostic's
equal-bond setup -- and inspecting the actual `04_Pi.h5` checkpoints
(`tensor4all_hdf5::load_treetn`, per-node `IdxTensor::dims()`) showed CTTN's
largest tensor (hub, `[13, 7, 2, 13]` = 2366 elements) is barely larger than
NBlock's largest (`[34, 2, 31]` = 2108 elements) -- nowhere near what a
bond=128-for-both proxy would suggest. The synthetic benchmark's uniform-bond
assumption does not represent the real workload's shape (asymmetric bond
growth, small/irregular floating-zone batch sizes, message-cache reuse
patterns across a full 9-sweep run), so its negative-looking result should
not have been trusted as the final word -- only a real downstream measurement
settles it.

## Real downstream validation

Ran gw-rs's actual `sgw` binary (R=10, comb topology, `aci_global_guard=true`,
`max_bond_dim=4096`, identical config to the saved baseline and to issue
#671's original reproduction) against this fix, via a temporary local
`[patch]` override in `gw-rs/sgw/Cargo.toml` pointing at this session's local
tensor4all-rs checkout (not yet pushed/merged -- see Follow-up). Compared
against the same saved baseline (frames.rs fix only, no treetn fix):

| stage | baseline elapsed | with this fix | bond (baseline -> fixed) | speedup |
|---|---:|---:|---|---:|
| pi_rtau | 103.55s | 33.7s | 90 -> 91 | **3.07x** |
| W | 8.38s | 5.5s | 42 -> 43 | **1.52x** |
| sigma_rtau | 72.77s | 23.1s | 81 -> 81 (identical) | **3.15x** |
| **treeaci stages total** | **184.7s** | **62.3s** | | **2.96x** |

Bond dimensions and truncation errors match the baseline closely (sigma_rtau's
bond is exactly identical), so this is a genuine wall-time win at matched
accuracy, not a "converged to a cheaper answer" artifact.

## Verification

- TDD throughout, same discipline as the `frames.rs` fix: every new
  function/branch had a failing test confirmed red before implementation.
- New tests: `grouped_branch_contraction_matches_direct_reference_for_real_values`
  (isolated primitive, hand-computed reference), large-group real/complex
  variants forcing the actual BLAS path against `scalar_branch_message_contraction`,
  `raw_branch_message_matches_generic_contraction` /
  `raw_complex_branch_message_matches_generic_contraction` (manual
  plan/messages-level parity against `compute_stacked_message`, using a new
  `complex_star_tree` fixture), and
  `cached_evaluator_matches_tree_evaluate_on_star_tree_with_fixed_leaf_center`
  (end-to-end through the public `evaluate_batched` API -- the existing
  star-tree test fixed centre to the hub itself and never reached the branch
  dispatch at all).
- `cargo fmt --all -- --check` clean.
- `cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings`
  clean (one fix needed: a `needless_range_loop` in the diagnostic's comb-tree
  builder).
- `cargo test --release --manifest-path crates/tensor4all-treetn/Cargo.toml --no-fail-fast`:
  all green.

## Follow-up

- This fix is validated locally (both crate-level tests and the real gw-rs
  downstream pipeline) but not yet pushed to `origin/main`. gw-rs's
  `sgw/Cargo.toml` currently carries a temporary `[patch]` section pointing
  at this local checkout for that validation; remove it once this PR merges
  and gw-rs's existing `branch = "main"` git dependencies pick up the merged
  commit on their own.
- Nodes with 3+ children (degree >= 5) remain on the scalar/generic path,
  same scope decision as the `frames.rs` fix and for the same reason: not
  needed for #671's reported topology, and a general N-child version would
  need an inter-step buffer transpose this 2-child case avoids.
- The synthetic same-bond diagnostic's numbers (25.84x / 29.26x / 44.24x)
  are kept as a regression-style measurement tool
  (`diagnostic_chain_vs_comb_wall_time_on_realistic_floating_zone_walk`,
  `#[ignore]`d), but should be read as "does branching cost something at
  matched bond" evidence, not as a proxy for real downstream wall time --
  this session's own experience with it argues for that caveat explicitly.
