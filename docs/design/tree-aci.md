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
| per-entry bound | -- | `max_frame_elements`, unset by default and then a quarter of `max_working_bytes` (`2^24` f64 elements at the 512 MiB default) | -- |
| accounting | `sample_arena_records`, `sample_arena_retained_bytes` | `frame_records`, `frame_retained_bytes` | opt-in `branch_diagnostics::snapshot()`, `query_cache` |

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

The element ceilings (`max_local_matrix_elements`, `max_core_elements`,
`max_frame_elements`) and the working-byte budget measure the same physical
resource, so the ceilings are not independent constants: each is `None` by
default and then derived as a quarter of `max_working_bytes`, expressed in
elements of the run's scalar type and resolved once, in `prepare_problem`, for
every later enforcement site. Raising the budget raises them with it. Setting
one explicitly pins it, and a pinned ceiling no longer follows the budget in
either direction. The retention budgets (`max_frame_bytes`,
`max_sample_arena_bytes`, `message_cache_max_bytes`) are separate and do not
follow the working budget, because they bound what survives between local
updates rather than what one update may allocate. Within one preparation the
byte budget is checked before any ceiling derived from it, so an impossible
budget is reported as `working bytes` rather than as a derived ceiling.

The two state-owned cache families report through `TreeAciDiagnostics`, in the
same logical-byte units. With the `diagnostics` feature, `query_cache` reports
message and prepared-slice entries, logical payload and owned-byte estimates.
These are whole-evaluator high-water observations attached to query-center
records, so do not sum them across nodes or output shapes. Estimates include
message vector capacity and hash-table buckets, but exclude backend allocator
arenas and dynamic allocations inside generic node labels.

## Per-node performance attribution

Enable `tensor4all-treeaci/diagnostics`, call `branch_diagnostics::reset()`
before a measurement window and read `snapshot()` afterward. Sorted records
distinguish `input:N:node` and `output:node`, and retain separate actual physical
and incident-bond shapes when output ranks change. `local_elements` is the
checked proxy `d * product(bond_dims)`, not a FLOP count. Message batches count
unique subtree assignments, frame batches count candidate samples, and query
batches count caller-supplied full points. Their denominators are not
interchangeable.

`guard_ns` excludes recursive child messages; `frame_ns` includes candidate
contraction and final packing. `query_ns` is inclusive of the whole evaluator
query and must not be added to message/frame time. Kernel snapshots are
thread-local, so setup, matrix multiplication, accumulation, gathering, and
prepared-slice hits/misses/refusals belong to the measured node. Scalar and
other uninstrumented work remains in the enclosing phase time. The legacy
`contraction_summary()` is process-wide and unsuitable for per-node attribution.

The observer stores aggregate rows rather than per-point histories. Its memory
scales with distinct operand/node/shapes in the chosen window; resetting releases
those rows. It adds no numerical cache and does not change contraction dispatch.
Instrumentation and cache-accounting overhead are included in measured wall
time; use a feature-disabled build for production timing, and measure observer
overhead separately. The reproducible experiment and dimension-adjusted model
are described in `benchmarks/README.md` (issue #732).

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
