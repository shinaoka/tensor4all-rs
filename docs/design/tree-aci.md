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
- A deterministic minimum-retracing forward walk keeps consecutive local
  updates on adjacent edges. With an automatic root it starts at a diameter
  endpoint and has the graph-theoretically optimal open-walk length
  `2|E| - |P|`, where `P` is the selected spine. The return pass traverses only
  `P` in reverse order; the complete forward/return round therefore has
  exactly `2|E|` directed edge updates and is an Euler tour of the bidirected
  tree.
- Traversal selection is a public, non-exhaustive strategy enum, while every
  strategy lowers into one internal `SweepPlan`. A common validator checks edge
  references, orientation, walk continuity, complete-round coverage, and the
  declared forward/return semantics before execution. Only
  `MinimumRetracingWalk` is implemented initially; no custom planner trait is
  public while frame-generation semantics are still evolving.
- Serial execution is the reference semantics. Parallel phases, if enabled
  later, use snapshot-isolated proposals and deterministic commit-time sample
  ID remapping.

For a path topology, `diameter = |E|`, so the forward and spine-only return
passes reduce to the two train sweeps without extra edge visits. In a branched
tree, branch edges outside the chosen diameter are visited in both directions
during the forward excursions, while the return visits only the diameter
spine. Thus every edge is used once in each direction per complete round;
the previous full inverse would have repeated the branch excursions. Visiting
branch edges in both directions during the forward walk remains required by
the current LUCI gauge invariant: a bond axis is ordered by its active
component samples, while a generic TreeTN canonicalization can change that
basis without updating the sample order.

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

Initialization deliberately leaves both generated outputs and explicit initial
guesses numerically uncanonicalized until the first directional pass. Bootstrap
uses only validated bond dimensions and component coordinates; it does not read
the guess's gauge. The first pass replaces every core with CI factors and then
canonicalizes the lower-rank result. Canonicalizing a high-rank explicit guess
before that pass is redundant and makes setup scale with a gauge that ACI
immediately discards.

For a node with one incoming component, all local physical slices are stacked
into one `(outgoing × local) × incoming` matrix. One backend multiplication then
contracts every physical coordinate and incoming sample column. This avoids a
separate small BLAS dispatch for every local coordinate in both bootstrap-frame
construction and local candidate evaluation.

Guard injection is capacity-based, not merely enabled/disabled per edge. Each
cut carries its remaining distance to the configured/algebraic rank limit, and
is removed from the active projection mask as soon as that capacity is used.
Thus one Guard scan offering several pivots cannot overshoot a cut that had
only one rank available.

## Caches

TreeACI owns three cache families. None outlives one `tree_elementwise` call;
the run boundary releases all of them.

| | sample arena | input/candidate frames | evaluator messages |
|---|---|---|---|
| owner | `TreeAciState` | `TreeAciState` | guard input/output evaluators |
| lifetime | one run | one run, incrementally extended after sample growth | input evaluators: one run; output evaluator: one guard scan |
| contents | immutable component-sample records and deduplication keys | exact directed component contractions plus optional candidate contractions | directed subtree messages reused across floating-zone batches |
| aggregate bound | `max_sample_arena_bytes`, 256 MiB | `max_frame_bytes`, 256 MiB | `message_cache_max_bytes`, 256 MiB shared across all evaluators |
| per-entry bound | -- | `max_frame_elements`, `2^24` | -- |
| accounting | `sample_arena_records`, `sample_arena_retained_bytes` | `frame_records`, `frame_retained_bytes` | bounded but not currently exposed in diagnostics |

Mandatory arena and directed-frame bounds are checked before allocation, so an
over-budget run is refused instead of first reaching the configured peak.
Candidate frames and evaluator messages are optional accelerators: when their
budget is exhausted, evaluation continues without retaining new entries. If
base directed frames grow into space occupied by candidate frames, the
candidate cache is reclaimed first. Retained bytes are logical payload
accounting, not allocator or process measurements.

The per-entry and aggregate frame bounds are independent and neither implies the
other: the cache keeps one frame per input per directed edge, so a per-frame
ceiling sized to admit one frame still admits all of them.

The two state-owned cache families report through `TreeAciDiagnostics`, in the
same logical-byte units. Evaluator messages obey their aggregate option bound
but do not yet have a diagnostic counter.

The crate exposes native TreeTN elementwise and simultaneous n-way Hadamard
entry points. Input message caches now persist across guard scans, while the
output cache is rebuilt per scan because the approximating output changes after
each sweep. Performance parity remains workload-dependent, so it is not yet a
drop-in train ACI replacement.

This document records the durable repository architecture. The staged
implementation history, including the edge-order experiment and its verdict,
is in `docs/superpowers/plans/2026-08-14-treeaci-next-phase.md` and the
accompanying spec; the invariants those phases established are stated above
rather than left to the plan.
