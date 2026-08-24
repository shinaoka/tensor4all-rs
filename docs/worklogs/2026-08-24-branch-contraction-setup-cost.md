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
scalar reads at this scale: up to ~1.35M elements for the hub's peak
observed dims `[247, 20, 273]`).

## Second fix: contiguous-read fast path

`raw`'s strides are `[1, dims[0], dims[0]*dims[1], dims[0]*dims[1]*dims[2]]`
(column-major, axis 0 fastest). `flat` steps by `strides[parent_axis]` in the
innermost loop, which is contiguous (stride 1) only when `parent_axis`
happens to land on axis 0 for that particular node's tensor -- data-dependent
per node, not a fixed convention. Added a fast path: when
`strides[parent_axis] == 1`, read the whole `parent_dim`-length run as one
slice and `copy_from_slice` it into `left`, instead of one bounds-checked
element at a time; falls back to the existing per-element loop otherwise.
Same values, same write offsets either way.

No existing branch-kernel test happened to exercise `strides[parent_axis] ==
1` -- every fixture put `physical_axis` at stride 1 instead. Added
`grouped_branch_contraction_matches_scalar_reference_when_parent_axis_is_contiguous`
(same shape as the existing large-groups test, with `parent_axis`/`physical_axis`
swapped) before relying on the new path.

Measured, same checkpoint, same call, on top of the first fix:

| | first fix | + contiguous-read fast path |
|---|---:|---:|
| `setup_ns` | 90.5s | 73.8s (**-18%**, -37% cumulative from 117.8s) |
| `pi_rtau` total | 275.6s | 262.8s (**-5%**, -12.3% cumulative from 299.8s) |
| max_bond / final_err | 228 / 9.478e-5 | 228 / 9.478e-5 (unchanged) |

`setup_ns` is now 72% of the branch kernel's own time (down from 81% before
either fix). The fast path only fires for whichever nodes happen to have
`parent_axis` on the fast tensor axis; most calls in this particular run
still take the general per-element path.

## Verification

- `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release branch`:
  all matching tests pass, including every branch-kernel correctness test
  listed above and the new contiguous-axis test.
- `cargo test --manifest-path crates/tensor4all-treetn/Cargo.toml --release`
  (default and `--features diagnostics`): full suite green (453/457 passed,
  1 ignored), no change from before either fix.
- `cargo clippy --manifest-path crates/tensor4all-treetn/Cargo.toml --all-targets -- -D warnings`
  clean, default and `--features diagnostics`, after both fixes.
- Downstream: re-ran the same real R=10 Comb `pi_rtau` checkpoint via gw-rs's
  `isolate_aci_stage` harness after each fix; results in the tables above.

## Third fix: measure which axis is actually fast before generalizing further

Before extending the fast path further, checked whether a prior attempt at
this specific angle already existed and failed (the min-group-points gate
above is a different, already-known rejection). None found in this branch's
worklogs or git history for the memory-access-pattern angle specifically.

Rather than guessing which of `child_axis_1`/`child_axis_2` to add a fast
path for, added scratch counters (`BRANCH_FAST_AXIS_IS_{PARENT,CHILD1,CHILD2,PHYSICAL}`,
not wired into `summary()`'s stable output) tallying which axis is actually
stride-1 once per BLAS call, and re-ran the same checkpoint. Result, out of
11392 branch BLAS calls: `parent=3986` (35%, already covered),
`child1=0` (never), `child2=5824` (51%, uncovered), `physical=1582` (14%, no
axis available to help). `child_axis_2`, not `parent_axis`, is the single
most common fast axis -- and `child_axis_1` needs no fast path at all.

Added a fast path for `strides[child_axis_2] == 1`: read the contiguous
`child_dim_2` run per `(c1, parent)` as one slice, then scatter it into
`left` at its `parent_dim` write-stride. `child_axis_2` is not `left`'s
contiguous write axis (`parent` is), so unlike the `parent_axis` fast path,
this trades a strided read for a strided write rather than eliminating both.
New test `grouped_branch_contraction_matches_scalar_reference_when_child_axis_2_is_contiguous`
(again, no existing fixture put `child_axis_2` at stride 1) exercises the
scatter-write path specifically.

Measured, same checkpoint, same call, on top of the first two fixes:

| | + contiguous-read (parent) | + contiguous-read (child_axis_2) |
|---|---:|---:|
| `setup_ns` | 73.8s | 69.0s (**-6.4%**, -41.4% cumulative from 117.8s) |
| `pi_rtau` total | 262.8s | 257.8s (**-1.9%**, -14.0% cumulative from 299.8s) |
| max_bond / final_err | 228 / 9.478e-5 | 228 / 9.478e-5 (unchanged) |

Despite covering more calls (51% vs. 35%), this fix's wall-time win is
smaller than the `parent_axis` fast path's -- consistent with the
scatter-write cost eating most of the contiguous-read benefit. Three fixes in
a row now show shrinking `setup_ns` reductions (23% -> 18% -> 6.4%):
diminishing returns, stopping this optimization thread here rather than
continuing to chase the remaining `setup_ns` (still 68% of the branch
kernel's own time after all three fixes).

## Follow-up

- Further `setup_ns` reduction from here would need either a genuinely
  different data layout for `raw` (out of scope: it is this node's `IdxTensor`
  storage, shared with every other consumer, not something this contraction
  alone can choose), or accepting the general path's scattered
  reads-and-writes as inherent to this contraction's shape for the ~14% of
  calls where `physical_axis` is the fast axis (no loop-axis fast path can
  help there) and whatever fraction still falls through the general path for
  other reasons. Not pursued further in this session.
- This fix, the `diagnostics`/`branch_diagnostics` feature, and gw-rs's
  `isolate_aci_stage` harness are all still local to this session's branches
  (`optimize-treeaci-branched-hotpaths` here, `aci-stage-isolation-harness`
  in gw-rs) and not yet pushed.
