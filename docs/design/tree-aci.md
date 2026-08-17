# Tree Alternating Cross Interpolation

`tensor4all-treeaci` implements Alternating Cross Interpolation directly on
tree tensor networks. It depends on `tensor4all-treetn` for the representation
and point evaluation, on `tensor4all-core` for dense matrix LUCI, and on the
core/backend crates for indices and numerical storage. It must not use the
train-specific ACI state or conversion bridges.

The implementation follows these invariants:

- Removing one edge of a validated tree produces two disjoint components.
  Component samples therefore glue to exactly one global physical assignment.
- Directed component samples are immutable records. A record stores its local
  physical coordinate and immutable sample IDs for incoming branches. Active
  pivot sets may change between generations without changing older records.
- Per-input frames are keyed by directed edge and immutable sample ID. A frame
  is the exact contraction of one input component with its physical assignment,
  leaving only the cut bond open.
- The conformance path materializes only a checked, edge-local, column-major
  matrix and calls the same owned dense LUCI contract as train ACI. It never
  materializes the full tensor represented by an input TreeTN.
- A deterministic minimum-retracing walk keeps consecutive local updates on
  adjacent edges. With an automatic root it starts at a diameter endpoint and
  has the graph-theoretically optimal length `2|E| - diameter`; the reverse pass
  is its exact orientation-reversing inverse.
- Traversal selection is a public, non-exhaustive strategy enum, while every
  strategy lowers into one internal `SweepPlan`. A common validator checks edge
  coverage, orientation, walk continuity, and reverse-pass semantics before
  execution. Only `MinimumRetracingWalk` is implemented initially; no custom
  planner trait is public while frame-generation semantics are still evolving.
- Serial execution is the reference semantics. Parallel phases, if enabled
  later, use snapshot-isolated proposals and deterministic commit-time sample
  ID remapping.

For a path topology, `diameter = |E|`, so forward and reverse passes reduce to
the two train sweeps without extra edge visits. Branch edges outside the chosen
diameter are visited in both directions. This retracing is required by the
current LUCI gauge invariant: a bond axis is ordered by its active component
samples, while a generic TreeTN canonicalization can change that basis without
updating the sample order.

Discontinuous minimum path covers and parallel paths remain mathematically
possible, but require sample-aware gauge transport or snapshot/merge semantics.
They must not be enabled by simply canonicalizing between paths: a binary-tree
regression showed that doing so can permute physical values while reporting tiny
local residuals.

Candidate products at a node scale with the product of incoming branch ranks.
The public preparation boundary therefore uses checked arithmetic and
caller-visible limits for candidate rows and columns, local matrix elements,
core and frame elements, arena retention, and working bytes. Low maximum degree
does not remove this requirement, but it keeps the rank exponent bounded in the
intended workloads.

## Caches

TreeACI owns two caches. Neither outlives one `tree_elementwise` call, which is
why neither exposes a clear method: the run boundary is the release point, and
a caller holds no handle that could accumulate across runs. A cache that does
outlive a call -- a cross-call directed-message cache on `TreeTNCachedEvaluator`,
for instance -- would need a real clear path as well as the bounds below.

| | sample arena | directed-frame cache |
|---|---|---|
| owner | `TreeAciState` | `TreeAciState` |
| lifetime | one run | one run, rebuilt on pivot injection |
| contents | immutable component-sample records and their deduplication keys | one exact component contraction per input per directed edge |
| aggregate bound | `max_sample_arena_bytes`, 256 MiB | `max_frame_bytes`, 256 MiB |
| per-entry bound | -- | `max_frame_elements`, `2^24` |
| accounting | `sample_arena_records`, `sample_arena_retained_bytes` | `frame_records`, `frame_retained_bytes` |

Both aggregate bounds are checked before each allocation rather than after a
batch of them, so an over-budget run is refused instead of first reaching the
peak it was configured to avoid. Retained bytes are each cache's own payload
accounting -- the elements it holds times the scalar width -- not an allocator
or process measurement.

The per-entry and aggregate frame bounds are independent and neither implies the
other: the cache keeps one frame per input per directed edge, so a per-frame
ceiling sized to admit one frame still admits all of them.

Both caches report through `TreeAciDiagnostics`, which is the aggregate stats
surface for the run: a caller sizing either bound reads both from one place, in
the same units.

The crate now exposes experimental native TreeTN elementwise and simultaneous
n-way Hadamard entry points. Persistent cross-scan guard caching and train-path
performance parity are not complete, so it is not yet a drop-in train ACI
replacement.

This document records the durable repository architecture. The staged
implementation history, including the edge-order experiment and its verdict,
is in `docs/superpowers/plans/2026-08-14-treeaci-next-phase.md` and the
accompanying spec; the invariants those phases established are stated above
rather than left to the plan.
