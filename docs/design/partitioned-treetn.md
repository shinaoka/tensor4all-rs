# Partitioned TreeTN migration

## Status

Approved after maintainer decisions on tracking issue
[#648](https://github.com/tensor4all/tensor4all-rs/issues/648). The rebased
cross-model pre-implementation review gate passed on commit `bda6a58f`; this
record is the implementation contract for the migration.

## Goal

Add `tensor4all-partitionedtreetn`, a TreeTN-native successor to
`tensor4all-partitionedtt`. Keep the TT crate available but deprecated during a
migration window. The new crate supports arbitrary named tree topologies,
including branched trees and multiple external site indices per node.

Adaptive interpolation is not part of this crate or this migration task.

## Compatibility window

`tensor4all-partitionedtt` remains buildable and behavior-frozen while users
migrate. Its README and crate docs point to `tensor4all-partitionedtreetn` and
mark it deprecated. Do not add a crate-level `#![deprecated]` attribute or
compiler warning. Only correctness and security fixes should land in the old
crate.

No fixed removal date is declared. Removing the old crate requires a separate
maintainer decision after the successor reaches parity and downstream users
have migrated.

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
- `Projector`
- `PatchingOptions`
- `PatchSplitStrategy`

Both partition types are generic only over the node-name type `V` and store
`TreeTN<IdxTensor, V>`. Making the tensor type generic is out of scope because
projection depends on `IdxTensor::mask_index` and the TT predecessor also uses
the dynamic `IdxTensor` boundary.

Do not add aliases for the old TT names in the new crate.

## Prerequisite

Resolve [#634](https://github.com/tensor4all/tensor4all-rs/issues/634) in the
deprecated crate before copying `Projector`. The corrected implementation must
use one canonical entry ordering consistent with `DynIndex::Eq` and
`DynIndex::Hash` (currently ID, tags, and prime level) for both projector
hashing and deterministic comparison. Equal projectors must hash identically
regardless of insertion order or `HashMap` seed, and the deterministic
comparator may return `Equal` only for fully equal projectors.

The #634 fix also supplies fallible coordinate validation and transactional
partition mutation. The new crate copies that validated implementation rather
than the buggy `origin/main` version.

## Subdomain model

`SubDomainTreeTN<V>` stores:

- `TreeTN<IdxTensor, V>` data,
- a `Projector`, and
- the internal absolute squared truncation budget used by adaptive patching.

It does not store a canonical or truncation center. A center is operation policy,
not part of the represented partition. Operations that require one accept an
explicit `&V`, following the underlying TreeTN API.

Construction is fallible and eagerly applies the projector. The stored TreeTN
retains every full site index, but values outside the projected coordinates are
masked to zero with `IdxTensor::mask_index`. Rebuild the TreeTN from its local
tensors after masking so stale canonical and orthogonality metadata cannot
survive tensor changes. This is local tensor work only and must not materialize
the full network.

This eager invariant is authoritative: `data()` always returns an already
projected TreeTN. Norms, inner products, truncation budgets, contraction, and
summation operate directly on that representation and must not apply a second
lazy projection. `project` applies only newly requested compatible restrictions
and returns another eagerly masked value.

The public API follows the TT predecessor where tree topology does not require
an extra choice:

- `from_treetn`, `data`, `into_data`,
- `node_count`, `is_empty`, `all_indices`, `site_index_network`,
- `max_bond_dim`, `project`, `norm`, `norm_squared`, `truncate`, `contract`, and
  `inner`.

`norm` and `norm_squared` keep `&self` receiver semantics for parity with the TT
crate. They clone the TreeTN wrapper and canonicalize the clone internally;
rustdoc must state this cost. `truncate` and `contract` take an explicit center
because arbitrary trees have no TT-like rightmost site.

## Partition invariant

A `PartitionedTreeTN<V>` is a map from mutually disjoint projectors to eagerly
masked subdomains. All subdomains in one value must have:

- exactly the same named tree topology,
- exactly the same full site-index set at each node, and
- one homogeneous `IdxTensor` dtype across every node and patch.

Same topology and dimensions with different site-index identities are not
enough; automatic reindexing is prohibited because `Projector` keys use full
indices. Mixed `f64`/`Complex64` partitions are rejected before no-op shortcuts
or TreeTN algebra.

Constructor, insertion, and append operations validate all invariants before
mutation. Failure leaves the original partition unchanged.

`to_treetn` deterministically sums the already projected patches with strict
TreeTN direct-sum addition. It is the replacement for `to_tensor_train`.

## Addition and contraction

Patch-wise addition preserves the TT predecessor's restricted contract:

- identical projector keys may be added;
- a patch absent from one operand is treated as zero; and
- different overlapping projector layouts are rejected rather than refined.

Matching patches use strict `TreeTN::add` and are truncated at the caller's
explicit center. Addition requires exact named topology, site-index assignment,
and dtype compatibility.

Contraction likewise requires the two partitions to use the same named topology
in v1; it does not call `restructure_to` or a mismatched-topology dense fallback.
For each compatible projector pair it:

1. contracts the already masked TreeTNs with
   `tensor4all_treetn::contraction::contract` at the explicit center;
2. keeps projector entries only for surviving external full indices; and
3. combines duplicate output projectors with strict TreeTN addition and the
   requested truncation policy.

`contract_adaptive` adds the stronger project-first rule inherited from
`PartitionedMPSs.jl`. A contraction or duplicate-projector sum that reaches the
patch cap is only a probe and is discarded. The next child projector is chosen,
each original operand or addend is projected to that child, and contraction or
addition is retried recursively. It is incorrect to cap-truncate a parent sum
and split that already lossy value afterward. Equality with the cap counts as
saturation because a capped probe cannot distinguish an exact rank from a
truncated one. Each newly projected operand is first compressed without a bond
cap at a dtype-scaled numerical-rank threshold (`64 * epsilon`). This removes
projection-created null bond space without spending the caller's truncation
budget; user-requested approximation remains confined to contraction and
post-addition truncation.

`SubDomainTreeTN::inner` uses the eagerly masked stored data directly. No
projector-compatible region may be inferred or lazily remasked during inner
product evaluation.

No production path may call `to_dense`, `contract_to_tensor`, or a dense
reference contraction unless the caller explicitly selected and bounded a
TreeTN dense-reference method.

## Adaptive patching

The TreeTN-general patch algebra remains in this crate:

- `add_with_patching`,
- `truncate_adaptive`,
- `contract_adaptive`,
- `PatchingOptions`, and
- `PatchSplitStrategy`.

Each operation requiring truncation or contraction accepts an explicit center.
The public signatures are:

```text
add_with_patching<V>(Vec<SubDomainTreeTN<V>>, &V, &PatchingOptions)
    -> Result<PartitionedTreeTN<V>>
truncate_adaptive<V>(&PartitionedTreeTN<V>, &V, f64, Option<usize>)
    -> Result<PartitionedTreeTN<V>>
contract_adaptive<V>(
    &PartitionedTreeTN<V>,
    &PartitionedTreeTN<V>,
    &V,
    &ContractionOptions,
    &PatchingOptions,
) -> Result<PartitionedTreeTN<V>>
```

`PatchingOptions` contains `rtol`, `max_bond_dim`, a partial full-index
`patch_order`, and `split_strategy`. `PatchSplitStrategy::Sequential` follows
that order; `ExactParameterGain` evaluates all available candidates. Split
candidates are full external indices and remain independent of tree traversal
order. Multiple external indices on one node are supported.

`ExactParameterGain` preserves the TT predecessor's meaning: after forming and
budget-truncating each candidate's children, count the checked sum of each local
tensor's logical element count (the product of its dimensions). Do not use
backend payload length, storage bytes, structured-storage compression, or AD
state as the metric.

Volume-proportional truncation keeps the absolute squared-tail budget semantics
established by closed issue
[#554](https://github.com/tensor4all/tensor4all-rs/issues/554). TreeTN
`TruncationOptions` and `SvdTruncationPolicy` replace itensorlike options.

## Adaptive interpolation ownership

`adaptiveinterpolate`, `AdaptiveInterpolateOptions`, TreeTCI termination changes,
pivot recycling, sampled-zero inference, and TreeTCI checked-arithmetic work are
out of scope. The new crate does not depend on `tensor4all-treetci`.

The existing unmerged branch
`feat/treetci-adaptive-patching@85df576` remains a separate TreeTCI work item.
This migration neither reimplements it nor decides its public result type,
zero-detection policy, or merge disposition.

## Errors

Use a crate-local `thiserror` enum. Preserve typed sources from
`TreeTNOperationError`, tensor storage, and tensor construction. Validation
errors for topology, dtype, center, projector indices, coordinates, and options
remain structured where callers need their payloads. Volume and logical
parameter-count overflow return typed crate errors rather than wrapping.
Every public `Result` API documents concrete failure conditions.

## Dependencies and provenance

The new crate depends on `tensor4all-core`, `tensor4all-tensorbackend`, and
`tensor4all-treetn`. It does not depend on `tensor4all-itensorlike`,
`tensor4all-simplett`, `tensor4all-tensorci`, or `tensor4all-treetci`. Provider
features propagate through all direct tensor4all dependencies.

Preserve and document the partition representation's relationship to
[PartitionedMPSs.jl](https://github.com/tensor4all/PartitionedMPSs.jl) and the
scientific credit for “Adaptive Patching for Tensor Train Computations.” Because
`adaptive_interpolation.rs` is not copied, the new crate does not claim a code
derivation from TCIAlgorithms.jl and does not copy
`LICENSE-TCIALGORITHMS-MIT`. The deprecated TT crate retains its existing
TCIAlgorithms derivation notice and license.

## Documentation surface

Update the workspace crate list, architecture guide, design index, `llms.txt`,
usage skill references, provenance policy, and live partitioned-tensor guides.
Add the new crate to coverage enforcement; do not lower coverage thresholds
without separate approval.

Add a crate README included by crate docs and runnable asserted rustdoc examples
for public types, traits, and functions. Document public fields and enum variants
without requiring repetitive standalone examples for each field or variant. The
old README and crate docs state the migration target but no fixed removal date.

## Verification

Minimum focused matrix:

- `f64` and `Complex64` homogeneous partitions;
- mixed-dtype rejection before construction, append, addition, contraction, and
  summation, with no mutation on failure;
- one node, chain, and branched tree;
- multiple external indices on one TreeTN node;
- fixed indices at leaves and internal vertices;
- eager masking tests where unprojected values are much larger than projected
  values, covering norm, inner, truncation budgets, contraction, and summation;
- same-ID indices differing by prime level or tags;
- equal projectors hashing identically across insertion orders and map seeds;
- projector comparator equality if and only if full projector equality;
- same-ID/different-metadata projectors used successfully as `HashMap` keys;
- invalid center, topology, projector index, coordinate, and options;
- transactional construction, insertion, and append failures;
- identical-layout addition plus rejection of different overlapping layouts;
- strict same-topology contraction and deterministic `to_treetn`;
- explicit-center truncate, contract, and patching paths, including invalid
  centers;
- logical parameter-count tests using structured masked tensors;
- checked volume and logical parameter-count overflow;
- a safe long cheap TreeTN regression that fails quickly under accidental full
  dense materialization without intentionally exhausting CI memory;
- small dense references materialized once and compared as whole tensors;
- default CPU build and
  `--no-default-features --features tenferro-provider-inject` build;
- workspace clippy, release doctests, and mdBook tests.

Run the repository's focused crate checks during development, then the complete
pre-PR format, clippy, release workspace tests, rustdoc, mdBook, API dump, and
repository-rules review gates. Coverage is CI-owned; record an explicit coverage
impact attestation in the worklog/PR body.

## Deferred work

- Adaptive interpolation and the disposition of
  `feat/treetci-adaptive-patching`.
- Common-refinement addition for different overlapping projector layouts.
- Contraction between different named topologies.
- Generic tensor storage beyond `IdxTensor`.
- Removing `tensor4all-partitionedtt` after a separate maintainer decision.
- Extracting shared projector code; deletion of the old crate is the preferred
  deduplication step.

## Review record

Maintainer decisions recorded in this revision:

- adaptive interpolation is excluded; TreeTN-general adaptive patching remains;
- subdomain construction eagerly masks stored data;
- center is not stored and is passed explicitly to operations that require it;
- addition rejects different overlapping projector layouts;
- `ExactParameterGain` counts logical tensor elements;
- deprecation is documentation-only with no fixed removal date;
- node names are generic, tensor storage is `IdxTensor`, dtype is homogeneous,
  and multiple site indices per node are supported;
- patch topology/site identities are exact, and contraction requires matching
  named topology; and
- provenance includes PartitionedMPSs.jl and the adaptive patching paper, but not
  TCIAlgorithms-derived code or license.

The 2026-08-17 issue reviews informed these decisions. The final cross-model
design review used `reviewer-flash-opencode-go` and recorded the verdict
`Correct-to-merge`. The same reviewer inspected the final combined staged diff;
after its requested scalar-node path tests were added, the fresh verdict was
also `Correct-to-merge`.
