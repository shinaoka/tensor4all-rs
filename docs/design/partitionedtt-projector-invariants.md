# PartitionedTT projector invariant repair (#634)

## Status

Pre-implementation design for issue #634. This is an independently mergeable
correctness fix and must land before the TreeTN-native successor in #648 copies
`Projector`.

## Scope

Repair projector identity, coordinate validation, and transactional partition
mutation in `tensor4all-partitionedtt`. Breaking cleanup is accepted because the
repository is in early development.

In scope:

- one deterministic ordering over the fields used by `DynIndex::Eq` and
  `DynIndex::Hash`;
- fallible projector construction and insertion;
- rejection of projector indices absent from a tensor train;
- transactional, disjoint partition construction and mutation;
- removal or restriction of unrestricted mutable subdomain access; and
- contract/property tests and runnable rustdoc for every changed public API.

Out of scope: C API or language bindings, adaptive interpolation behavior,
backend/device behavior, dtype or AD policy, numerical tolerances, dependencies,
features, and broader index representation changes.

## Existing contract and defect

`DynIndex` equality and hashing use `id`, `tags`, and `plev`; `dim` is shape
metadata and is deliberately excluded. `Projector` currently stores a
`HashMap<DynIndex, usize>` but hashes entries after sorting only by ID. Two equal
maps can therefore hash differently when equal-ID entries differ by tags or
prime level. Partition output ordering similarly compares only `(id, value)`,
so unequal projectors can compare equal.

`Projector::from_pairs` and `Projector::insert` currently accept coordinates
outside `0..index.dim()`. `SubDomainTT::new` silently filters absent projector
indices. `PartitionedTT::insert` can bypass disjointness checks, and public
`get_mut` permits arbitrary mutation after a projector has become a map key.

## Canonical projector entry order

Define one private comparator for `DynIndex` identity with this lexicographic
key:

1. `DynIndex::id`;
2. the canonical sorted tag sequence; and
3. `DynIndex::plev`.

The key must not include `dim`, because `DynIndex::Eq` and `Hash` do not include
it. Comparator equality must be equivalent to `DynIndex` equality.

Define one private canonical-entry helper that sorts projector entries by that
identity comparator and then by projected coordinate. Reuse it for:

- `Projector::Hash`, hashing each full `DynIndex` and coordinate in canonical
  order; and
- the deterministic total comparator used to order partition results.

The projector comparator is lexicographic over canonical entries, including
length. It returns `Equal` if and only if `Projector::eq` is true. Public
`PartialOrd` retains its existing subset semantics and does not become this
total order.

`dim` remains attached shape metadata. A successful update of an already equal
identity must not leave an old key object paired with a coordinate validated
only against a different incoming dimension: validate before mutation and
replace the stored key/value together. Applying a projector to a tensor train
validates the coordinate against the matched tensor-train index dimension.

## Fallible projector APIs

Change:

```rust
pub fn Projector::from_pairs(
    pairs: impl IntoIterator<Item = (DynIndex, usize)>,
) -> Result<Projector>

pub fn Projector::insert(
    &mut self,
    index: DynIndex,
    value: usize,
) -> Result<()>
```

A coordinate is valid exactly when `value < index.dim()`. Failure returns the
existing typed `PartitionedTTError::ProjectorCoordinateOutOfBounds` and leaves
the projector unchanged.

Remove `FromIterator<(DynIndex, usize)> for Projector`: `FromIterator` cannot
report validation errors, and an infallible compatibility path would violate
the invariant. Update every production caller, test, benchmark, example, and
doctest to propagate or explicitly assert the result.

Operations that only retain or combine entries already validated by public
construction (`common_restriction`, `filter_indices`, and compatible
intersection) remain infallible. `filter_indices` is the explicit filtering API;
construction never filters silently.

## SubDomainTT validation

Change `SubDomainTT::new(data, projector)` to return `Result<Self>`.

Before commit, collect the tensor train's site indices and validate every
projector entry:

- a matching full `DynIndex` identity (`id`, `tags`, and `plev`) must exist;
- the coordinate must be in range for the matched tensor-train index; and
- no projector entry is silently removed or clamped.

Use `ProjectorIndexNotFound` for an absent full identity and
`ProjectorCoordinateOutOfBounds` for an invalid coordinate. `from_tt` remains
infallible because its projector is empty. `project` validates the merged
projector before constructing its result.

The stored `TensorTrain`, `Projector`, and truncation budget remain private.
Existing public accessors remain immutable.

## PartitionedTT transactional mutation

All insertion paths route through one validation rule:

- every stored subdomain is internally coherent by `SubDomainTT::new`;
- different projector keys are pairwise disjoint;
- inserting an exactly equal projector may replace its value; and
- a different compatible/overlapping projector is rejected with
  `OverlappingProjectors`.

`PartitionedTT::insert` performs all checks before mutating the map. On failure,
length, keys, and values are unchanged.

`from_subdomains`, `from_subdomain`, `append`, and `append_subdomains` either
preflight the complete candidate set or build a validated temporary partition
before commit. They never partially insert before reporting an error.

Restrict `PartitionedTT::get_mut` to crate-private use (or remove it if no
internal caller remains). Do not add a public unchecked replacement or
compatibility shim. Internal mutation must not change the private projector/data
relationship; public replacement goes through `insert` with a validated
`SubDomainTT`.

## Error and documentation contract

Keep errors in `PartitionedTTError`. Public changed functions document:

- arguments and zero-based coordinate constraints;
- return values;
- exact error variants and transactional behavior; and
- runnable examples with assertions and explicit `unwrap`/error checks.

No public item uses `ignore` or `no_run`. No old infallible aliases are retained.

## Verification

Add focused tests covering every distinct path:

1. equal projectors hash identically across insertion orders and independently
   created `HashMap` seeds;
2. canonical comparator equality is equivalent to projector equality;
3. same-ID indices differing by tags or prime level remain distinct and sort
   deterministically;
4. coordinates `0` and `dim - 1` succeed while `dim` and larger fail;
5. failed `Projector::insert` leaves the projector byte-for-contract unchanged;
6. duplicate equal identities update the stored key/value coherently;
7. `SubDomainTT::new` rejects missing full identities, including same-ID
   tag/prime variants, and validates against the tensor-train dimension;
8. exact-projector partition replacement succeeds;
9. overlapping partition insertion, append, and bulk construction fail without
   changing the receiver; and
10. disjoint insertion and append remain successful.

Use release mode for the changed crate's tests as required locally. Before the
PR, run and record:

```text
cargo fmt --all
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo nextest run --release -p tensor4all-partitionedtt
cargo test --doc --release --workspace
cargo nextest run --release --workspace
cargo doc --workspace --no-deps
./scripts/test-mdbook.sh
python3 scripts/repository-rules-review.py --base origin/main --worktree --dry-run
python3 scripts/test-repository-rules-review.py
```

Run the repository API inventory command if it exists on the synchronized base.
The current `origin/main` xtask does not yet expose the `api-dump` subcommand, so
absence of that command is recorded rather than worked around by editing
generated output.

Coverage is CI-owned. The PR must attest that removed/changed paths were checked
for coverage impact and that replacement tests exercise every invariant branch;
coverage thresholds and numerical tolerances are not changed.

## Review and integration order

1. Obtain a recorded `Correct-to-merge` verdict on this document from the
   selected read-only opencode-go DeepSeek V4 Flash reviewer.
2. Delegate implementation in this dedicated worktree to GPT-5.6 Luna.
3. Run local gates and inspect the full diff.
4. Obtain a recorded post-implementation full-diff verdict from the same
   reviewer and fix/re-review all blocking findings.
5. Synchronize with current `origin/main`, rerun affected gates, push, open the
   PR, monitor CI, and merge only after all required checks pass.
6. Rebase/recreate #648 from the merged #634 commit; do not copy the buggy
   `Projector` from an earlier revision.
