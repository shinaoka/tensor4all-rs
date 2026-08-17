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

The crate now exposes experimental native TreeTN elementwise and simultaneous
n-way Hadamard entry points. Persistent cross-scan guard caching and train-path
performance parity are not complete, so it is not yet a drop-in train ACI
replacement.

The full mathematical derivation and staged implementation plan are maintained
in `/Users/lingruicheng/treeaci/` during development. This document records only
the durable repository architecture.
