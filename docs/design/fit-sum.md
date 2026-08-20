# Variational fitting of TreeTN sums

## Status

Issue: [#660](https://github.com/tensor4all/tensor4all-rs/issues/660)

Pre-implementation review:

- Round 1: `CHANGES-REQUIRED` by `reviewer-flash-opencode-go`. Required clarification of site-only environment contractions, a single-node path, the public options export, and same-ID/prime coverage.
- Round 2: `CORRECT-TO-IMPLEMENT` by `reviewer-flash-opencode-go`. The reviewer confirmed every blocking Round-1 finding was resolved.

## Contract

Add this crate-root API:

```rust
pub fn fit_sum<T, V>(
    targets: &[TreeTN<T, V>],
    initial: &TreeTN<T, V>,
    center: &V,
    options: FitOptions,
) -> Result<TreeTN<T, V>, TreeTNOperationError>
```

`initial` is required and fixes the output topology, site-index identities, and starting bond spaces. Crate root re-exports the existing private-module `FitContractionOptions` as `FitOptions`; the implementation type and contraction dispatcher remain unchanged. The unrelated `tensor4all-simplett::FitOptions` remains crate-qualified and unchanged. The options control full sweeps, maximum bond dimension, factorization/truncation policy, and convergence.

Inputs are compatible when every nonempty target has the same named topology as the nonempty `initial` and, at each node, the same number and dimensions of site indices. Target site indices are deterministically relabeled to those of `initial` with the existing `reindex_site_space_like` operation. When a node owns multiple equal-dimensional indices, correspondence is the existing full-index deterministic sort order (including prime level/tags), not caller insertion order. Target bond identities and dimensions need not match `initial`.

Before mutating the result, validate the nonempty target list, nonempty networks, center membership, topology/site-space compatibility, SVD/factorization options, finite nonnegative convergence tolerance, and initial target-environment construction. Tensor scalar/backend failures receive `fit_sum` and target-index context. Unsupported scalar combinations fail; combinations supported by the tensor backend (including its normal real-to-complex promotion) remain compatible. Because all arguments use one tensor type `T`, a distinct Rust scalar type cannot be mixed at this API boundary; the integration test therefore documents the backend's real-to-complex promotion contract rather than adding a type-specific downcast.

## Algorithm

Reuse the existing two-site local-update machinery rather than construct an exact direct-sum TreeTN.

1. Validate options, `initial`, `center`, and every target; relabel target site spaces to `initial`.
2. Validate options first, including rejection of `max_bond_dim == 0`, even on short-circuit paths. If `nfullsweeps == 0`, return an unchanged clone of validated `initial` (including for a single-node topology). Otherwise clone and canonicalize `initial` to `center` without truncation.
3. For a single-node topology with a positive sweep count, directly sum the target node tensors with tensor-level `axpby`, apply the configured factorization-independent semantics, and return that one-node TreeTN. This path does not use an empty sweep plan.
4. Keep one directed-edge environment cache per target. For target `g_i`, `env_i[(from,to)]` contracts the `from`-side subtree of `g_i` with `conj(psi)` **only over corresponding site indices**. On the cut edge it leaves the target bond and variational bond as two distinct open legs; their identities and dimensions are never paired or required to match. Nodes with no site indices still remain connected through their target/variational bond legs and child environments.
5. Before the first mutation, build the initial environments with target-index context. At each two-site step, independently contract each target's two local tensors with its environments. The open physical legs are the `initial` site indices; accumulate the resulting local tensors one at a time with `TensorVectorSpace::axpby`.
6. Factorize the summed local optimum with the existing policy and replace the same two tensors/bond in `psi`; invalidate every target cache through the existing invalidation path.
7. Repeat the existing Euler-tour sweep and convergence loop. Profiling labels the shared machinery neutrally and identifies the `contract_fit` or `fit_sum` entry point.

The memory retained for targets and environments is linear in the number of terms, but no tensor-network bond dimension is the sum of all target bond dimensions. Local target contributions are accumulated one at a time, so no exact direct-sum network or full dense target is materialized.

## Refactoring boundary

Keep `FitEnvironment` and `FitUpdater` public behavior intact for contraction fitting. Add a private target-factor environment helper and extract the common two-site update body so:

- contraction fitting supplies one target term with local factors `[A, B]`;
- sum fitting supplies one target term per input with local factors `[g_i]`.

Do not duplicate the factorization/replacement logic and do not add a chain-specific implementation.

## Verification

- Small dense-reference tests for `f64` and `Complex64` sums, plus one-node exact summation and zero-sweep identity behavior.
- A normal chain and a branched tree whose internal node has no site index; assert that target and variational cut-bond legs remain distinct there.
- A fixed-rank test comparing the dense objective before and after fitting.
- Empty target list/network, empty initial network, missing center, named-topology mismatch, per-node site-count/dimension mismatch, and incompatible scalar/backend diagnostics.
- Reindexing regression with same-ID indices that differ by prime level/tags, pinning the full-index deterministic correspondence.
- A long cheap chain asserting bounded output bonds, guarding against exact direct-sum or dense materialization.
- Runnable public rustdoc with a known numerical result.
- Existing contraction-fit tests remain green.
