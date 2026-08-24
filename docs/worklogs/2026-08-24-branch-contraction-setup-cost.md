# Branch contraction setup cost (tensor4all-rs#671 follow-up)

Work log. Continues from `docs/worklogs/2026-08-22-treetn-branch-message-raw-path.md`
(which added `grouped_branch_message_contraction`, the branch kernel this
entry modifies) and from the new `diagnostics`/`branch_diagnostics` feature
(`docs/superpowers/specs/2026-08-24-treeaci-branch-diagnostics-design.md`).

## What the new per-phase counters found

`grouped_branch_message_contraction`/`grouped_chain_message_contraction`
were instrumented with feature-gated counters splitting branch calls into
`setup_ns` (building the `left` intermediate before the BLAS call),
`matmul_ns`, and `accumulate_ns`, plus BLAS-vs-scalar call/point counts for
both kernels (`contraction_diagnostics` in `cached_evaluator.rs`). A real R=10
Comb `pi_rtau` product (T=0.01, `aci_global_guard=true`, same config as
issue #671's reproduction, replayed via gw-rs's `isolate_aci_stage` harness
from an existing checkpoint) showed:

```
branch: blas_calls=11392  blas_points=26049   (2.3 points/call average)
        setup_ns=117.8s   matmul_ns=25.6s     accumulate_ns=1.25s
chain:  blas_calls=2219   blas_points=408186  (184 points/call average)
        contract_ns=1.98s
```

`setup_ns` -- the scalar triple-nested loop building the `left` intermediate
matrix, independent of how many points share the group -- is 81% of the
branch kernel's own time, and larger than the entire chain kernel's total
time for the same call. Branch's average of 2.3 points per BLAS call (vs
chain's 184) means this fixed setup cost is essentially never amortized.

## Two follow-up questions, and why the answer wasn't "add a min-group gate"

The obvious fix -- mirror the chain kernel's `CHAIN_BLAS_MIN_GROUP_POINTS`
gate on branch, falling back to `scalar_branch_message_contraction` for
small groups -- was already tried and rejected during
`2026-08-22-treetn-branch-message-raw-path.md`'s own development (see that
log's "A regression, caught before it shipped" section): it regressed a
synthetic benchmark by ~1.7x, because `scalar_branch_message_contraction` is
slower per FLOP than the BLAS path even at 1-2 points, once bond dimensions
are realistic. So this investigation asked two different questions instead:

**Could Guard's own search hand branch nodes larger batches?** Traced
`find_global_pivots` (`tensor4all-treeaci/src/global_guard.rs`) through
`floating_zone_walk` (`tensor4all-core/src/floating_zone.rs`) into
`TreeTNCachedEvaluator::evaluate_batched_with_hint`. The top-level batch size
is already the widest the coordinate-descent walk can offer (every candidate
value of the one site currently being swept). Chain's much larger effective
batch comes from a level below: each node's batch gets re-grouped by that
node's *own* physical value before dispatch, and only shatters into
near-singleton groups for whichever node happens to *own* the site currently
being swept -- the comb's single hub is disproportionately likely to be that
node relative to its many chain-arm neighbors. This is intrinsic to how the
search visits the tree, not a batching oversight; restructuring it would
change search semantics and is out of scope here.

**Is the setup loop itself doing more work than it needs to?** Yes, partly.
The original loop rebuilt a `[usize; 4]` `axis_values` array and recomputed
the full 4-term stride dot product from scratch on every one of the up to
`parent_dim * child_dim_1 * child_dim_2` `(c1, c2, parent)` iterations, even
though three of those four terms are loop-invariant at each nesting level.

## Fix

Rewrote the `left`-matrix construction loop in
`grouped_branch_message_contraction` to hoist the per-`c1`/per-`c2` partial
sums out of their respective loop levels and turn the innermost `parent`
loop's index arithmetic into a running accumulator (`flat += strides[parent_axis]`
per step) instead of a fresh multiply-add. Mathematically identical to the
original per-point computation -- no change to which elements are read or
where they are written, verified against the existing branch-kernel
correctness tests (`grouped_branch_contraction_matches_direct_reference_for_real_values`,
`grouped_branch_contraction_large_{real,complex}_groups_match_scalar_reference`,
`raw_{,complex_}branch_message_matches_generic_contraction`), all passing
unchanged.

## Measured effect

Same checkpoint, same call, before/after:

| | before | after |
|---|---:|---:|
| `setup_ns` | 117.8s | 90.5s (**-23%**) |
| `matmul_ns` | 25.6s | 26.8s (noise) |
| `pi_rtau` total | 299.8s | 275.6s (**-8%**) |
| max_bond / final_err | 228 / 9.478e-5 | 228 / 9.478e-5 (unchanged) |

A real, measured win at matched accuracy, but a partial one: `setup_ns` is
still 76% of the branch kernel's own time after this change. The arithmetic
was not the dominant cost -- the per-element write `left[left_offset] =
raw[flat]` itself, at a `flat` stride pattern that is not contiguous across
the innermost loop, is the more likely remaining bottleneck (cache-unfriendly
scalar writes at this scale: up to ~1.35M elements for the hub's peak
observed dims `[247, 20, 273]`). Not pursued further in this session.

## Verification

- `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release branch`:
  all 7 matching tests pass, including every branch-kernel correctness test
  listed above.
- `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release`
  (default and `--features diagnostics`): full suite green, no change from
  before this fix.
- `cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings`
  clean, default and `--features diagnostics`.
- Downstream: re-ran the same real R=10 Comb `pi_rtau` checkpoint via gw-rs's
  `isolate_aci_stage` harness before and after; results in the table above.

## Follow-up

- The remaining `setup_ns` cost (still the majority of branch's own time) is
  a plausible next target, but would need profiling at the memory-access
  level (or a differently-shaped write pattern) rather than more arithmetic
  hoisting, which is now largely exhausted.
- This fix, the `diagnostics`/`branch_diagnostics` feature, and gw-rs's
  `isolate_aci_stage` harness are all still local to this session's branches
  (`optimize-treeaci-branched-hotpaths` here, `aci-stage-isolation-harness`
  in gw-rs) and not yet pushed.
