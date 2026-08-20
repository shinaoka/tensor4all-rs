# TreeACI Spine-Only Return Schedule

## Status

Design draft for review. This document describes the scheduler change only;
it does not change the implementation.

## Problem

`tensor4all-treeaci` currently constructs a continuous open walk from one
selected spine endpoint to the other. The walk visits every off-spine branch
edge once in each direction so that it can return to the spine, while each
spine edge is visited once. The reverse pass is currently the exact reversal
of the entire forward walk. Consequently, a full forward/reverse round visits

- each spine edge twice, once in each direction; and
- each off-spine edge four times, twice in each direction.

For a tree with `E` edges and selected spine length `D`, the current round has
`2 * (2E - D) = 4E - 2D` edge updates. The repeated off-spine visits are a
candidate performance bug: they are not needed to achieve once-per-direction
coverage over a complete round.

## Terminology and mathematical basis

Use `walk`, not strict graph-theoretic `path`, for a sequence that may revisit
vertices. A finite tree `T` can be converted to a directed multigraph by
replacing every undirected edge `{u, v}` with arcs `u -> v` and `v -> u`.
Every vertex then has equal in-degree and out-degree, and the underlying graph
is connected. The directed Euler-circuit theorem therefore gives a closed walk
that uses every directed arc exactly once. In the original tree this means
that every edge is traversed exactly once in each direction, in exactly
`2|E|` steps. This is the standard tree Euler-tour / DFS-contour traversal.

The proposed schedule is a particular Euler tour assembled from the existing
walk:

1. Keep the current forward walk unchanged. It starts at the selected spine
   endpoint, performs each off-spine branch as an out-and-back excursion, and
   ends at the opposite spine endpoint.
2. Replace the current full reverse with only the reverse of the selected
   spine. It starts where the forward walk ends and returns to the original
   start.

The concatenation is continuous and uses every directed edge exactly once:
off-spine edges get both orientations during the forward excursions, while
spine edges get their forward orientation during the forward walk and their
reverse orientation during the return. Its length is

```text
(D + 2(E - D)) + D = 2E,
```

which is the lower bound imposed by the two required orientations of every
edge. Thus the proposed complete round is update-optimal under the
once-per-direction coverage requirement.

## Proposed design

### Sweep-plan representation

Keep the existing `forward` path construction and retain the selected spine
as an explicit ordered edge path (or an equivalent internal representation).
Construct `reverse` directly from that spine in reverse orientation; do not
derive it by reversing all forward steps.

The public traversal strategy remains `MinimumRetracingWalk` for now. Its
meaning becomes “minimum open forward walk plus spine-only return,” rather than
“forward walk plus its exact inverse.” No train conversion or train-specific
state is introduced.

### Validation

Split the current validation obligations:

- validate each path's edge references, orientations, continuity, and phase
  vertex-disjointness;
- validate the forward walk's expected coverage: spine edges occur once and
  off-spine edges occur twice, with both orientations for off-spine edges;
- validate the return path is the reverse of the selected spine and starts at
  the forward endpoint;
- validate the concatenated forward + return round has every directed edge
  exactly once and is closed at the starting node.

The validator must no longer require `reverse == reverse_pass(forward)` or
require the return pass by itself to visit every edge. A return pass that
contains only the spine is intentional.

### Scheduler semantics

Keep the existing `max_sweeps`, `min_sweeps`, and alternating forward/reverse
control flow for API compatibility. A forward pass remains the full current
walk; a reverse pass becomes the spine-only return. A pair of passes is the
coverage-complete round. The scheduler must continue to check that each path
starts at the current canonical center; the proposed return starts at the
forward endpoint, so the existing continuous-center invariant should remain
applicable.

Diagnostics and convergence history must continue to count passes as they do
today. The implementation must audit any logic that assumes every individual
pass updates every edge, especially `updated_edges`, global-pivot cleanup, and
rank/error convergence checks.

## Correctness requirements

The implementation is acceptable only if all of the following hold:

1. Every complete forward/reverse round updates each directed edge exactly
   once.
2. Every scheduled path is connected and starts at the canonical center that
   precedes it.
3. The forward walk remains unchanged for all tested topologies.
4. A chain produces the same edge-update counts as train ACI: one pass in each
   direction and two updates per edge per complete round.
5. Small star, comb, balanced, and irregular trees produce numerically valid
   TreeACI results: finite values, acceptable residuals, and no new rank or
   convergence failures.
6. Existing branch-point and frame/sample correctness tests remain passing.

## Verification plan

### Structural tests

Extend the path-cover tests to assert directed-edge coverage rather than only
the old exact-inverse property. Cover:

- a chain, where forward and return each visit every edge once;
- a star, where the forward walk visits non-spine spokes out-and-back and the
  return visits only the spine;
- comb and balanced trees, where branches occur at multiple spine positions;
- all labelled trees through the existing small exhaustive bound.

For every case, assert that the concatenated round is a closed walk and that
each `(edge, from, to)` directed occurrence appears exactly once.

### Numerical correctness

Run existing TreeACI unit/integration tests and add small deterministic cases
that compare the proposed schedule against the current schedule or an
independent dense/reference evaluation. Check values, residuals, edge ranks,
and termination behavior. Include cases with and without global pivots and
with a rank cap, because those paths may depend on per-pass edge coverage.

### Performance

Instrument or count scheduled edge updates before timing. For a tree with `E`
edges and spine length `D`, the structural count must change from `4E - 2D`
per current round to `2E`. Then benchmark representative star, comb, and
balanced trees at increasing bond dimensions. The existing chain benchmark
should remain a parity check, not evidence of branch-retracing cost.

## Alternatives considered

### Full Euler tour as a new single pass

This also achieves `2E` updates, but it changes the meaning of a sweep more
substantially and risks changing convergence and gauge-update ordering. The
spine-only return schedule achieves the same complete-round bound while
preserving the existing forward walk and pass alternation.

### Keep the exact reverse of the full forward walk

This preserves current semantics but retains the avoidable `4E - 2D` round
cost. It is the current behavior and is not recommended after the structural
coverage argument above.

### Split into independent branch paths

This could avoid retracing, but it would require snapshot/merge or
sample-aware gauge transport because paths would share vertices. It is a much
larger semantic change than the serial spine-only return and is out of scope.

## Documentation changes after implementation

Update `docs/design/tree-aci.md` to distinguish:

- the graph-theoretically optimal length `2E - D` of the current open forward
  walk;
- the `2E` lower bound and Euler-tour interpretation of a complete round;
- the fact that the return is spine-only, not the exact inverse of every
  forward excursion.

The document must retain the LUCI gauge rationale for visiting branch edges in
both directions during the forward walk, while no longer implying that those
branch edges must be repeated in the return pass.

## Non-goals

- Do not convert TreeTN inputs to `SimpleTensorTrain`.
- Do not change `SampleArena`, frame materialization, or local contraction
  kernels as part of this scheduler change.
- Do not introduce a public custom planner trait.
- Do not claim train-performance parity until the branch and chain benchmarks
  and numerical checks have been rerun.
