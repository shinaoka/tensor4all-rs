# Partitioned TreeTN migration

## Status

Proposed and revised after the 2026-08-17 readiness comment on tracking issue
[#648](https://github.com/tensor4all/tensor4all-rs/issues/648). The formal
cross-model pre-implementation review gate is still pending.

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

`Projector` remains full-index-based. Do not add aliases for the old TT names in
the new crate.

## Prerequisites

1. Resolve [#634](https://github.com/tensor4all/tensor4all-rs/issues/634) in the
   deprecated crate before copying `Projector`. The corrected implementation
   must use one canonical entry ordering consistent with `DynIndex::Eq` and
   `DynIndex::Hash` (currently ID, tags, and prime level) for both projector
   hashing and deterministic comparison. Equal projectors must hash identically
   regardless of insertion order or `HashMap` seed, and the deterministic
   comparator may return `Equal` only for fully equal projectors. The fix also
   supplies fallible coordinate validation and transactional partition mutation;
   the new crate copies that validated implementation rather than the buggy
   `origin/main` version.
2. Track the typed TreeTCI termination result as a separate, independently
   mergeable `tensor4all-treetci` issue and land it before implementing adaptive
   interpolation here. Add its issue number to this document and #648 once it
   exists. The required termination contract is specified below.

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

`norm` and `norm_squared` take `&mut self`, canonicalize the stored TreeTN to the
stored center, and delegate to the corresponding TreeTN method. This makes the
canonicalization mutation and cost explicit and avoids a hidden whole-network
clone. `PartitionedTreeTN::norm` likewise takes `&mut self`; `inner` remains
non-mutating.

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
rank-cap termination. The prerequisite TreeTCI issue replaces the bare result
tuple with a typed run result carrying `Converged`, `MaxBondDim`, or
`MaxIterations`; this crate must not infer termination from only the last error
sample. When convergence and rank saturation become true in the same iteration,
`Converged` takes precedence if the complete convergence criterion is satisfied;
otherwise the result is `MaxBondDim`. Exhausting the iteration budget without
either condition yields `MaxIterations`.

A patch is accepted only for `Converged` with an accepted error at or below
tolerance. Rejected patches split on the first remaining full index in
`patch_order`.

The initial port does not recycle internal TreeTCI pivots between parent and
child patches because the TreeTCI public result does not expose a stable global
pivot set. The deprecated TT crate retains that optional behavior. Add TreeTCI
pivot recycling only after a dedicated public pivot-provenance contract exists.

## Patching

Copy adaptive patching into the new crate and replace TT operations with the
TreeTN methods above. `ExactParameterGain` counts local tensor payload sizes by
iterating nodes; products and sums use checked arithmetic. Split candidates are
full external indices and remain independent of tree traversal order.

Volume-proportional truncation keeps the absolute squared-tail budget semantics
established by closed issue
[#554](https://github.com/tensor4all/tensor4all-rs/issues/554). TreeTN
`TruncationOptions` and `SvdTruncationPolicy` replace itensorlike options.

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
- equal projectors hashing identically across insertion orders and map seeds;
- projector comparator equality if and only if full projector equality;
- same-ID/different-metadata projectors used successfully as `HashMap` keys;
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

- 2026-08-17 issue readiness comment: conditional approval after making #634 an
  explicit prerequisite, splitting the TreeTCI termination API into its own
  issue, fixing norm receiver semantics, and adding cross-links. These changes
  are recorded above.
- Formal cross-model reviewer and verdict: pending. The issue comment does not
  by itself clear the repository's delegated-implementation review gate.
