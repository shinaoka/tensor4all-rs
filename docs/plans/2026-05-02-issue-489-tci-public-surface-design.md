# Issue #489 TCI Public Surface Design

## Spec

Issue: [#489](https://github.com/tensor4all/tensor4all-rs/issues/489)

TCI-related crates expose too many low-level implementation details through
crate-root re-exports. The goal is to make each crate advertise only the API it
owns, while internal implementations keep direct imports from their real
dependencies.

Immediate scope:

- `tensor4all-tensorci` should not re-export foundational `tensor4all-tcicore`
  cache/index/scalar types unless they are part of the TensorCI user contract.
- `tensor4all-tcicore` should not publicly expose low-level MatrixLUCI kernels
  and pivot internals as top-level API.
- Similar re-export chains in `tensor4all-quanticstci` and
  `tensor4all-partitionedtt` should be audited and trimmed if they are
  accidental rather than intentional.

Decision:

- Close #489 only with a real public-surface cleanup, not with a top-level
  `pub use` shuffle. `DenseLuKernel`, `LazyBlockRookKernel`, `PivotKernel`, and
  `PivotSelectionCore` must disappear from public signatures as well as from
  crate-root re-exports.
- Keep #489 focused on `tensor4all-tensorci` and `tensor4all-tcicore`.
  `tensor4all-quanticstci` and `tensor4all-partitionedtt` re-exports are
  audited in this PR, but trimmed only if the change is mechanical. Otherwise
  record a follow-up issue or PR note.

Acceptance criteria:

- Crate-root re-exports are limited to user-facing types and functions owned by
  each crate.
- No public function or impl in workspace user-facing crates requires callers
  to name `DenseLuKernel`, `LazyBlockRookKernel`, or `PivotKernel` in a where
  clause.
- Workspace code compiles after switching internal imports to direct dependency
  paths.
- Public docs and examples import types from their owning crates instead of
  relying on transitive re-exports.
- `docs/api` no longer lists removed implementation types under the downstream
  crate roots after regeneration.

Non-goals:

- Do not remove modules or types that are still used internally.
- Do not redesign MatrixLUCI, TCI2, or quantics algorithms.
- Do not add compatibility `pub use` shims or `doc(hidden)` aliases unless a
  type remains genuinely necessary for a public signature.

## Design

Use a strict ownership rule:

- A crate may publicly export the domain types it defines and expects users to
  name.
- A crate may publicly export dependency types only when those types appear in a
  public signature that cannot reasonably be hidden.
- Otherwise callers should import dependency types from the dependency crate
  directly.

### `tensor4all-tensorci`

Current root-level re-export:

```rust
pub use tensor4all_tcicore::{
    CacheKey, CacheKeyError, CachedFunction, IndexInt, IndexSet,
    LocalIndex, MultiIndex, Scalar,
};
```

Remove this block unless a specific item appears in the public TensorCI API and
must be documented there. Public examples should import `IndexSet` or
`MultiIndex` from `tensor4all-tcicore` if those types are actually required by
the example.

`MultiIndex`, `IndexSet`, and `Scalar` may still appear in TensorCI public
signatures because the current algorithm is parameterized by those `tcicore`
concepts. That is acceptable for this PR. The important boundary is that users
must import those types from `tensor4all-tcicore`, the crate that owns them, not
through `tensor4all-tensorci`.

### `tensor4all-tcicore`

Current MatrixLUCI re-exports include both user-facing algorithm types and
kernel internals:

```rust
CandidateMatrixSource, CrossFactors, DenseLuKernel, DenseMatrixSource,
DenseOwnedMatrix, LazyBlockRookKernel, LazyMatrixSource, MatrixLuciError,
PivotKernel, PivotKernelOptions, PivotSelectionCore
```

Keep public high-level MatrixCI types:

- `MatrixLUCI`
- `MatrixACA`
- `RrLU`, `RrLUOptions`, `rrlu`, `rrlu_mut`
- `Matrix`, `from_vec2d`
- `AbstractMatrixCI`
- `MatrixLuciScalar` if generic callers still need a scalar capability bound

Do not keep low-level MatrixLUCI substrate types public:

- kernel dispatch: `DenseLuKernel`, `LazyBlockRookKernel`, `PivotKernel`
- source plumbing: `CandidateMatrixSource`, `DenseMatrixSource`,
  `LazyMatrixSource`
- internal factor/result storage: `DenseOwnedMatrix`, `CrossFactors`,
  `PivotSelectionCore`
- kernel-only options/errors: `PivotKernelOptions`, `MatrixLuciError`,
  `MatrixLuciResult`

The current workspace leaks these types through public where clauses such as:

```rust
DenseLuKernel: PivotKernel<T>,
LazyBlockRookKernel: PivotKernel<T>,
```

Those bounds must be removed from public signatures. Replace them with a
high-level scalar capability bound, normally `T: MatrixLuciScalar`, and move the
kernel dispatch behind `tensor4all-tcicore` facade functions.

Add a small public facade only if downstream crates need lazy/dense pivot
factorization outside `MatrixLUCI::from_matrix`. The facade should return a
domain-level result, not the low-level `CrossFactors` and `PivotSelectionCore`
types. A shape like this is acceptable:

```rust
pub struct MatrixLuciFactors<T> {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub pivot_errors: Vec<f64>,
    pub rank: usize,
    pub cols_times_pivot_inv: Matrix<T>,
    pub pivot_inv_times_rows: Matrix<T>,
}
```

The exact name can change, but the result type must expose what TensorCI needs
without exposing how MatrixLUCI selected pivots internally. Dense and lazy
construction functions can live in `tensor4all-tcicore` and use private
`matrixluci` kernel implementations.

### Related Crates

Audit these root-level re-exports:

- `tensor4all-quanticstci`: `tensor4all_simplett::{AbstractTensorTrain,
  TensorTrain}` and `tensor4all_treetci::{DefaultProposer, TreeTciGraph,
  TreeTciOptions}`
- `tensor4all-partitionedtt`: `tensor4all_core::{DynIndex, TensorDynLen}` and
  `tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions}`

For each item, decide whether the downstream crate owns the public concept. If
not, remove the re-export and update examples to import from the owning crate.
For this PR, the decision is:

- `tensor4all-quanticstci`: do not block #489 on removing these re-exports.
  They are convenience exports of higher-level user-facing dependencies, not
  the low-level kernel leak that makes #489 risky.
- `tensor4all-partitionedtt`: do not block #489 on removing these re-exports.
  They should be cleaned up separately with examples/docs because removing them
  is user-facing import churn.
- Record both as follow-up cleanup unless the implementation diff is trivial
  after the `tcicore` facade lands.

## Files

- Modify `crates/tensor4all-tensorci/src/lib.rs`.
- Modify `crates/tensor4all-tcicore/src/lib.rs`.
- Modify `crates/tensor4all-tcicore/src/matrix_luci.rs` and the
  `crates/tensor4all-tcicore/src/matrixluci/` module boundary so kernel types
  are private implementation details.
- Modify public functions in `tensor4all-core`, `tensor4all-simplett`,
  `tensor4all-tensorci`, and `tensor4all-quanticstci` that currently expose
  `DenseLuKernel: PivotKernel<T>` or `LazyBlockRookKernel: PivotKernel<T>`
  bounds.
- Audit `crates/tensor4all-quanticstci/src/lib.rs` and
  `crates/tensor4all-partitionedtt/src/lib.rs`; trim only if trivial, otherwise
  record follow-up.
- Update tests, examples, rustdoc snippets, and mdBook imports that relied on
  removed paths.
- Regenerate `docs/api`.

## Testing

Start with API and import checks:

```bash
cargo check --release -p tensor4all-tcicore
cargo check --release -p tensor4all-tensorci
cargo check --release -p tensor4all-core
cargo check --release -p tensor4all-simplett
cargo check --release -p tensor4all-quanticstci
cargo check --release -p tensor4all-partitionedtt
```

Then verify examples and documentation:

```bash
cargo test --doc --release -p tensor4all-tcicore
cargo test --doc --release -p tensor4all-tensorci
cargo test --doc --release -p tensor4all-quanticstci
cargo test --doc --release -p tensor4all-partitionedtt
cargo run -p api-dump --release -- . -o docs/api
```

If public examples change in `docs/book`, also run:

```bash
./scripts/test-mdbook.sh
```

## PR Handling

This chunk can close #489 only if:

- `tensor4all-tensorci` no longer re-exports `tcicore` foundational types from
  its crate root;
- MatrixLUCI kernel/source/factor internals are not public through
  `tensor4all-tcicore` crate-root or public workspace signatures;
- `docs/api` no longer advertises the low-level kernel path as user-facing API;
- related `quanticstci` and `partitionedtt` convenience re-exports are audited
  and either trimmed or explicitly recorded as follow-up.

If implementation stops at deleting root `pub use` lines while public where
clauses still mention `DenseLuKernel` or `PivotKernel`, do not close #489.
