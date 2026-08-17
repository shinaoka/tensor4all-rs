# Partitioned TreeTN migration

## Status

Proposed. The pre-implementation review gate is pending.

## Goal

Add `tensor4all-partitionedtreetn`, a TreeTN-native successor to
`tensor4all-partitionedtt`. Keep the TT crate available but deprecated during a
migration window. The new crate must support branched tree topologies rather
than wrapping a linear chain in a TreeTN facade.

## Compatibility window

`tensor4all-partitionedtt` remains buildable and behavior-frozen while users
migrate. Mark the crate deprecated with a crate-level Rust deprecation attribute
and point its README and crate docs to `tensor4all-partitionedtreetn`. Only
correctness and security fixes should land in the deprecated crate.

The provisional removal date is **2026-12-31**, and removal must also wait until
the new crate passes the migrated TT tests plus the TreeTN-specific test matrix
below. The date can be extended explicitly, but the old crate must not become a
permanent compatibility layer.

## Scope and source layout

Create the new crate by copying the existing partition/projector implementation,
then replace TT-specific storage and algorithms in the copy. Temporary source
duplication is intentional: sharing code now would couple the deprecated API to
the new generic TreeTN API and make removal harder.

Use these public names:

- crate: `tensor4all-partitionedtreetn`
- `SubDomainTreeTN<V = usize>`
- `PartitionedTreeTN<V = usize>`
- `PartitionedTreeTNError`
- `AdaptiveInterpolateOptions`
- `PatchingOptions`
- `PatchSplitStrategy`

`Projector` remains full-index-based and is copied without changing its identity
semantics. Do not add aliases for the old TT names in the new crate.

## Subdomain model

`SubDomainTreeTN<V>` stores:

- `TreeTN<IdxTensor, V>` data,
- a `Projector`,
- one validated canonical/truncation center `V`, and
- the internal absolute squared truncation budget used by adaptive patching.

The center is explicit state rather than a hidden minimum-node policy. The
constructor validates that the center exists and that every projected full
index is an external site index with an in-bounds coordinate.

The public API provides TreeTN-specific vocabulary:

- `from_treetn`, `data`, `into_data`, `center`,
- `node_count`, `is_empty`, `all_indices`, `site_index_network`,
- `max_bond_dim`, `project`, `norm`, `norm_squared`, `truncate`, `contract`, and
  `inner`.

Projection retains every site index and masks non-selected coordinates with
`IdxTensor::mask_index`. Rebuild the TreeTN from its local tensors after masking
so stale canonical and orthogonality metadata cannot survive tensor changes.
This is local tensor work only; it must not materialize the full network.

## Partition invariant

A `PartitionedTreeTN<V>` is a map from mutually disjoint projectors to
subdomains. In addition to projector disjointness, all subdomains in one value
must have:

- exactly the same named tree topology,
- exactly the same full site-index set at each node, and
- the same stored center.

Constructor, insertion, and append operations validate these invariants before
mutation. Projectors continue to use full index equality, including prime level
and tags.

`to_treetn` deterministically sums projected patches with TreeTN direct-sum
addition. It is the replacement for `to_tensor_train`.

## TreeTN operations

Use `tensor4all-treetn` public operations directly:

- strict `TreeTN::add` for patch addition,
- `TreeTN::truncate_mut([center], TruncationOptions)` for truncation,
- `tensor4all_treetn::contraction::contract` for contraction,
- `TreeTN::inner` for inner products, and
- `TreeTN::link_dims` for the maximum bond dimension.

Contraction first applies each patch projector, contracts at the stored center,
and keeps projector entries only for surviving external indices. Pairwise
results with the same output projector are combined by strict TreeTN addition
and truncated with the contraction SVD policy and rank cap.

No production path may call `to_dense`, `contract_to_tensor`, or a dense
reference contraction unless the caller explicitly selected and bounded the
TreeTN dense-reference method.

## Adaptive TreeTCI

The new `adaptiveinterpolate` uses `tensor4all-treetci`, not
`tensor4all-tensorci` or a TT-to-TreeTN conversion.

Its topology input is a `TreeTciGraph`; ordered `site_indices[i]` belongs to
TreeTCI vertex `i`. The first implementation supports exactly one external site
index per TreeTCI vertex. General `PartitionedTreeTN` storage and algebra may
still hold multiple external indices per node.

For a projected patch, keep the full TreeTCI graph and replace each fixed
vertex's local dimension by one. The batch evaluator maps that local zero back
to the fixed full-domain coordinate. This preserves connectivity even when
removing fixed vertices would split a branched tree.

After TreeTCI materialization:

- active generated site indices are relabeled to the caller's full indices;
- a dimension-one fixed site is locally contracted with a one-hot bridge to the
  caller's full-dimensional index; and
- the TreeTN is rebuilt from the resulting local tensors.

The one-hot bridge is bounded by one site dimension and does not materialize a
full network. Zero-active and one-active patches use exact rank-one TreeTNs.
Sampled-zero detection remains an explicit finite-sampling policy.

Adaptive acceptance must distinguish convergence from max-iteration and
rank-cap termination. Extend `tensor4all-treetci` with a typed termination value
in its optimization/run result and use that value here; do not infer convergence
from only the last error sample. A patch is accepted only for `Converged` with
an accepted error at or below tolerance. Rejected patches split on the first
remaining full index in `patch_order`.

The initial port does not recycle internal TreeTCI pivots between parent and
child patches because the TreeTCI public result does not expose a stable global
pivot set. The deprecated TT crate retains that optional behavior. Add TreeTCI
pivot recycling only after a dedicated public pivot-provenance contract exists.

## Patching

Copy adaptive patching into the new crate and replace TT operations with the
TreeTN methods above. `ExactParameterGain` counts local tensor payload sizes by
iterating nodes; products and sums use checked arithmetic. Split candidates are
full external indices and remain independent of tree traversal order.

Volume-proportional truncation keeps the existing absolute squared-tail budget
semantics. TreeTN `TruncationOptions` and `SvdTruncationPolicy` replace
itensorlike options.

## Errors

Use a crate-local `thiserror` enum. Preserve typed sources from
`TreeTNOperationError`, `TreeTciError`, tensor storage, and tensor construction.
Validation errors for topology, center, projector indices, coordinates, and
options remain structured where callers need their payloads. Every public
`Result` API documents concrete failure conditions.

## Dependencies and provenance

The new crate depends on `tensor4all-core`, `tensor4all-tensorbackend`,
`tensor4all-treetn`, and `tensor4all-treetci`; it does not depend on
`tensor4all-itensorlike`, `tensor4all-simplett`, or `tensor4all-tensorci`.
Provider features propagate through all direct tensor4all dependencies.

Copied adaptive partition logic remains derived from TCIAlgorithms.jl. Copy the
existing derivation notices and `LICENSE-TCIALGORITHMS-MIT`, and update the
repository provenance/citation policy after maintainer confirmation. Do not
rewrite historical worklogs merely to change the crate name.

## Documentation surface

Update the workspace crate list, architecture guide, design index, `llms.txt`,
usage skill references, coverage thresholds, and the live adaptive interpolation
design. Add a crate README included by crate docs and runnable asserted rustdoc
examples for every public item. The old README and crate docs must state the
migration target and removal window.

## Verification

Minimum focused matrix:

- `f64` and `Complex64`;
- one node, chain, and branched tree;
- multiple external indices on one general TreeTN node;
- fixed indices at leaves and internal vertices;
- same-ID indices differing by prime level or tags;
- invalid center, topology, projector index, coordinate, and options;
- transactional insert/append failures;
- strict addition, contraction, truncation, and deterministic `to_treetn`;
- TreeTCI adaptive interpolation with zero, one, and several active sites;
- convergence, rank-cap, and max-iteration split paths;
- checked volume and parameter-count overflow;
- a long cheap TreeTN regression that would fail under accidental full dense
  materialization;
- small dense references materialized once and compared as whole tensors;
- default CPU and `tenferro-provider-inject` feature builds;
- release doctests and mdBook tests.

Run the repository's focused crate checks during development, then the complete
pre-PR format, clippy, release workspace tests, rustdoc, mdBook, API dump, and
repository-rules review gates. Coverage is CI-owned; record an explicit coverage
impact attestation in the worklog/PR body.

## Deferred work

- TreeTCI pivot recycling with a stable public pivot-provenance API.
- Adaptive interpolation with multiple logical site indices on one TreeTCI
  vertex.
- Removing `tensor4all-partitionedtt` before the compatibility window closes.
- Extracting shared projector code; deletion of the old crate is the preferred
  deduplication step.

## Review record

Pending.
