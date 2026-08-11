# Structured Tensor Storage Design (Issue #434)

## Goal

Make structured tensor storage a first-class `TensorDynLen` representation, then
expose that representation through the C API for Tensor4all.jl.

Issue [#434](https://github.com/tensor4all/tensor4all-rs/issues/434) asks for a
C API path that can construct and inspect tensors from compact payload metadata:
payload data, payload dimensions, payload strides, and logical-axis equivalence
classes (`axis_classes`). The design should preserve compact diagonal and copy
tensors instead of expanding them to dense storage during construction.

## Context

The Rust storage layer already has the right low-level representation:

- `tensor4all_tensorbackend::StructuredStorage<T>` stores compact payload data,
  `payload_dims`, `strides`, and `axis_classes`.
- `Storage::new_structured(...)` constructs explicit structured storage for
  `f64` and `Complex64`.
- `Storage::from_diag_col_major(...)` constructs diagonal storage using a rank-1
  payload and repeated axis classes.
- `TensorDynLen::is_diag()` and the multi-tensor contraction path already use
  diagonal metadata to treat diagonal axes as shared hyperedges.

The current mismatch is in `TensorDynLen`. It stores a dense/native eager tensor
as the primary payload and keeps only `axis_classes` as side metadata:

```rust
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    pub(crate) inner: Arc<EagerTensor<CpuBackend>>,
    pub(crate) axis_classes: Vec<usize>,
}
```

`TensorDynLen::from_storage(...)` therefore calls through
`storage_to_native_tensor(...)`, which materializes structured storage into a
dense tenferro tensor. This preserves mathematical values, but it does not make
structured storage the tensor's primary representation.

The earlier `tenferro-rs` implementation had the desired model:

```rust
pub struct StructuredTensor<T> {
    payload: Tensor<T>,
    logical_dims: Vec<usize>,
    axis_classes: Vec<usize>,
}
```

and its dynamic tensor enum stored `StructuredTensor<T>` variants directly. The
same design principle should be applied here using the existing
`StructuredStorage<T>` / `Storage` type instead of reintroducing a second
structured tensor abstraction.

## Non-Goals

- Do not preserve the current `TensorDynLen` dense-native primary payload model.
  The repository is in early development, so this can be a hard cut.
- Do not make all tensor operations structured-native in one change. Operations
  that cannot yet operate on compact storage may materialize explicitly and
  return dense storage.
- Do not add scalar-specific Rust APIs beyond the C API boundary. Rust library
  code should remain generic where possible.
- Do not redesign tenferro-rs in this issue. Tensor4all-rs can use its own
  `Storage` as the structured source of truth and call tenferro only when dense
  execution is needed.

## Data Model

Change `TensorDynLen` so structured storage is the canonical payload:

```rust
pub struct TensorDynLen {
    indices: Vec<DynIndex>,
    storage: Arc<Storage>,
    eager_cache: OnceLock<Arc<EagerTensor<CpuBackend>>>,
}
```

`storage` is the source of truth. It stores dense, diagonal, and general
structured layouts uniformly through `StructuredStorage<T>`.

`eager_cache` is optional and derived. It is populated only when an operation
requires tenferro's native/eager representation, such as SVD, QR, AD-enabled
einsum, or dense readback. The cache must never be required to answer structured
metadata queries.

If `OnceLock` is awkward because of clone semantics or existing `Arc`
expectations, use:

```rust
pub struct TensorDynLen {
    indices: Vec<DynIndex>,
    storage: Arc<Storage>,
    eager_cache: Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>>,
}
```

The important invariant is that `storage` is authoritative and `eager_cache` is
discardable.

## Storage Accessors

Expose structured metadata through safe public `Storage` accessors rather than
making the internal `StorageRepr` public.

Add:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    Dense,
    Structured,
    Diagonal,
}

impl Storage {
    pub fn storage_kind(&self) -> StorageKind;
    pub fn logical_dims(&self) -> Vec<usize>;
    pub fn logical_rank(&self) -> usize;
    pub fn payload_dims(&self) -> &[usize];
    pub fn payload_strides(&self) -> &[isize];
    pub fn axis_classes(&self) -> &[usize];
    pub fn payload_len(&self) -> usize;
    pub fn payload_f64_col_major_vec(&self) -> Result<Vec<f64>, String>;
    pub fn payload_c64_col_major_vec(&self) -> Result<Vec<Complex64>, String>;
}
```

`payload_*_col_major_vec` returns the compact payload in payload column-major
order. It does not materialize logical dense tensor values.

Keep existing dense logical readback methods:

- `to_dense_f64_col_major_vec(logical_dims)`
- `to_dense_c64_col_major_vec(logical_dims)`

Those methods are explicit materialization paths.

## Validation Rules

Structured validation should be centralized in the storage layer and reused by
`TensorDynLen` and the C API.

Required invariants:

- `axis_classes.len() == logical_rank`.
- `axis_classes` is canonical first-appearance form.
- `payload_dims.len() == number_of_axis_classes`.
- `strides.len() == payload_dims.len()`.
- Every logical axis maps to a valid payload class.
- Every logical axis in the same class has the same dimension.
- `payload_dims[class_id] == logical_dim(axis)` for all axes in that class.
- Payload length matches the storage length required by `payload_dims` and
  `strides`.
- For diagonal convenience constructors, logical rank should be at least 2 for
  non-scalar diagonal tensors, and all logical dimensions must be equal.

`StructuredStorage::new(...)` already validates most payload metadata. Add a
`Storage` or `TensorDynLen` helper for validating logical index dimensions
against the storage's `axis_classes`.

## Tensor Construction

### Dense

`TensorDynLen::from_dense(indices, data)` should construct dense `Storage` and
then call `from_storage`:

```rust
let dims = expected_dims_from_indices(&indices);
let storage = Storage::from_dense_col_major(data, &dims)?;
TensorDynLen::from_storage(indices, Arc::new(storage))
```

### Diagonal

`TensorDynLen::from_diag(indices, data)` should stop using
`diag_native_tensor_from_col_major(...)`. It should construct compact diagonal
storage directly:

```rust
validate_diag_payload_len(data.len(), &dims)?;
let storage = Storage::from_diag_col_major(data, indices.len())?;
TensorDynLen::from_storage(indices, Arc::new(storage))
```

`TensorDynLen::from_diag_any(...)` and `copy_tensor(...)` should use this same
path.

### General Structured

Add a Rust constructor for explicit structured storage:

```rust
pub fn from_structured_storage(
    indices: Vec<DynIndex>,
    storage: Arc<Storage>,
) -> Result<Self>
```

This may initially be an alias for `from_storage`, but its name makes the
intended public surface clear for callers that already have compact payload
metadata.

`from_storage` must:

- validate unique indices,
- validate index dimensions against `storage.logical_dims()`,
- preserve `storage.axis_classes()`,
- not call `storage_to_native_tensor(...)`.

## Materialization

Add an explicit private materialization method:

```rust
fn materialized_eager(&self) -> Result<Arc<EagerTensor<CpuBackend>>>;
fn materialized_native(&self) -> Result<&NativeTensor>;
```

The method builds a dense native tensor from `self.storage` only when needed.
All existing code paths that currently call `self.as_native()` should be
classified:

- metadata-only paths should use `self.storage`;
- structured-preserving paths should operate on payload storage;
- dense execution paths should call `materialized_eager()` explicitly.

`as_native()` should either become private and materializing, or be renamed to
make the densification obvious. Avoid a public API that silently hides the
representation change.

## Operation Policy

### Must Preserve Structured Storage

These operations must not materialize or densify:

- `dims()`
- `indices()`
- `is_diag()`
- `is_complex()`
- `is_f64()`
- `storage()`
- `to_storage()`
- C API metadata and compact payload readback

### Should Preserve Structured Storage

These operations can be implemented directly on compact payloads:

- `permute(...)`
- `permute_indices(...)`
- `replaceind(...)`
- `replace_indices(...)`
- `conj()`
- `scale(...)`
- same-layout `add(...)`, `try_sub(...)`, and `axpby(...)`

If two tensors have identical `axis_classes`, payload dimensions, payload
strides, scalar type, and logical dimensions, elementwise operations can operate
on payload values and keep the layout. If layouts differ, the initial
implementation may materialize both operands and return dense storage.

### May Materialize Explicitly

These operations may materialize to dense/native storage in the first
implementation:

- `to_vec::<T>()`
- `as_slice_f64()`
- `as_slice_c64()`
- `sum()` if no compact reduction helper exists yet
- `maxabs()` if no compact reduction helper exists yet
- SVD / QR / factorize
- general dense tensor contraction cases
- AD-enabled eager operations

When an operation materializes, the returned tensor should have dense
`Storage`, unless a structured-preserving result is deliberately reconstructed.

### Contraction

The existing `contract_multi_impl` already builds diagonal-aware hyperedges
through `build_diag_union(...)`. Preserve that behavior, but avoid materializing
diagonal tensors as full logical dense operands.

For pure diagonal/copy tensors:

- pass the compact rank-1 payload to the einsum backend,
- map all repeated logical axes to the same internal ID,
- keep the output dense unless a structured output layout is easy to prove.

For general repeated `axis_classes` such as `[0, 0, 1]`, extend the same idea
from pure diagonal groups to multiple equivalence classes:

- each tensor operand should expose payload rank equal to the number of classes;
- logical axis IDs that share a class should map to the same operand axis ID;
- output indices remain logical indices selected from non-contracted IDs.

This makes copy tensors and partial-copy tensors first-class rather than a
special `is_diag()` only case.

## C API Design

Add a C-visible storage-kind enum:

```rust
#[repr(C)]
pub enum t4a_tensor_storage_kind {
    T4A_TENSOR_STORAGE_DENSE = 0,
    T4A_TENSOR_STORAGE_STRUCTURED = 1,
    T4A_TENSOR_STORAGE_DIAGONAL = 2,
}
```

Add constructors:

```c
t4a_tensor_new_structured_f64(
    size_t rank,
    const t4a_index *const *index_ptrs,
    const double *payload,
    size_t payload_len,
    const size_t *payload_dims,
    size_t payload_rank,
    const ptrdiff_t *strides,
    size_t strides_len,
    const size_t *axis_classes,
    size_t axis_classes_len,
    t4a_tensor **out);

t4a_tensor_new_structured_c64(... interleaved payload ...);

t4a_tensor_new_diag_f64(...);
t4a_tensor_new_diag_c64(...);
```

Use `isize` on the Rust side and the corresponding generated C type from
`cbindgen` for strides. If `ptrdiff_t` generation is not stable with cbindgen,
use `intptr_t` explicitly in the Rust signature type and document it.

Add metadata/readback functions:

```c
t4a_tensor_storage_kind(...)
t4a_tensor_payload_rank(...)
t4a_tensor_payload_dims(...)
t4a_tensor_payload_strides(...)
t4a_tensor_axis_classes(...)
t4a_tensor_payload_len(...)
t4a_tensor_copy_payload_f64(...)
t4a_tensor_copy_payload_c64(...)
```

Existing dense readback functions keep their current behavior:

- `t4a_tensor_copy_dense_f64(...)`
- `t4a_tensor_copy_dense_c64(...)`

Those functions explicitly return logical dense values and may materialize.

## Error Handling

All new C API functions must use `run_catching` / `run_status` patterns that
preserve error details through `t4a_last_error_message`.

Do not introduce new patterns that discard error details, such as:

```rust
Err(_) => T4A_INTERNAL_ERROR
```

Validation errors should return `T4A_INVALID_ARGUMENT` with messages that name
the inconsistent field:

- `payload_len mismatch`
- `payload rank ... does not match axis_classes`
- `axis class ... has inconsistent logical dimensions`
- `payload dim mismatch for class ...`
- `data pointer is null`
- `index_ptrs[i] is null`

## Testing Strategy

Use TDD and keep each behavior small.

### Core Storage Tests

- `Storage::storage_kind()` distinguishes dense, structured, and diagonal.
- `Storage` accessors return payload metadata without materializing dense values.
- `payload_f64_col_major_vec()` and `payload_c64_col_major_vec()` reject wrong
  dtypes with clear errors.
- Structured validation rejects non-canonical `axis_classes`, payload-rank
  mismatch, inconsistent class dimensions, stride-rank mismatch, and payload
  length mismatch.

### TensorDynLen Tests

- `TensorDynLen::from_diag(...)` returns `is_diag() == true`.
- `TensorDynLen::from_diag(...).storage().payload_len() == diagonal_len`.
- `TensorDynLen::from_storage(...)` preserves general `axis_classes` such as
  `[0, 0, 1]`.
- `to_storage()` returns the same structured metadata and payload length without
  dense expansion.
- `to_vec::<f64>()` and `to_vec::<Complex64>()` still return correct logical
  dense column-major values.
- `permute(...)` preserves structured layout where possible.
- `scale(...)` and `conj()` preserve compact payload layout.
- same-layout `add/sub/axpby` preserve compact payload layout.
- layout-mismatched elementwise operations materialize and return dense storage
  if structured preservation is not implemented yet.

### Contraction Tests

- Contracting a C API-created diagonal tensor uses compact diagonal metadata and
  matches the dense result.
- Partial copy tensor with `axis_classes = [0, 0, 1]` contracts correctly
  against dense tensors.
- Mixed diagonal and dense contractions match dense reference tensors using
  whole-tensor dense comparison, not element-by-element recontraction.

### C API Tests

- `t4a_tensor_new_diag_f64` and `t4a_tensor_new_diag_c64` create tensors with
  diagonal storage kind and compact payload length.
- `t4a_tensor_new_structured_f64` and `t4a_tensor_new_structured_c64` round-trip
  payload dimensions, strides, axis classes, payload length, dtype, and compact
  payload.
- Metadata query functions support query-then-fill buffer conventions and return
  `T4A_BUFFER_TOO_SMALL` when buffers are short.
- Dense readback from structured tensors returns correct logical dense values.
- Validation failures preserve error messages through `t4a_last_error_message`.

## Migration Plan

1. Add `StorageKind` and public `Storage` structured metadata/payload accessors.
2. Refactor `TensorDynLen` to store `Arc<Storage>` as the source of truth.
3. Rewire constructors (`from_dense`, `from_diag`, `from_storage`,
   `from_dense_any`, `from_diag_any`, `copy_tensor`) through storage.
4. Add explicit dense/eager materialization helpers and update internal
   `as_native()` call sites.
5. Preserve structured layouts for metadata-only and simple payload operations.
6. Update contraction to consume compact payloads for diagonal and general
   repeated-axis-class operands.
7. Add C API structured constructors and compact metadata/readback functions.
8. Regenerate `crates/tensor4all-capi/include/tensor4all_capi.h`.
9. Update `docs/CAPI_DESIGN.md` and any README or mdBook references that still
   describe tensors as dense-only.

## Risks

### AD and Eager Cache Semantics

Current gradient APIs rely on `EagerTensor`. A storage-first tensor can still
materialize to eager for AD, but once AD tracking is enabled, mutations to
storage and eager cache must not diverge. The conservative rule is:

- plain tensors use `storage` as source of truth;
- AD-enabled tensors materialize once and operations that require AD return
  dense storage unless structured preservation is explicitly implemented.

### Performance Regression for Dense Workloads

Dense tensors will still be represented as structured storage with dense
`axis_classes = [0, 1, ...]`. Dense operations should fast-path contiguous dense
storage and populate the eager cache lazily to avoid repeated conversions.

### Contracting General Axis Classes

Pure diagonal tensors are already handled by `is_diag()`. General axis classes
such as `[0, 0, 1]` require extending the internal ID mapping beyond the pure
diagonal special case. This should be implemented after storage preservation is
tested, not mixed into the first constructor refactor.

## Open Questions

- Should `TensorDynLen::as_native()` remain as a private materializing helper,
  or should it be renamed to make dense materialization explicit?
- Should `StorageKind::Structured` include dense non-contiguous payloads, or
  should storage kind distinguish `DenseContiguous`, `DenseStrided`,
  `Structured`, and `Diagonal`?
- Should scalar/rank-0 tensors be represented as dense storage or as structured
  storage with empty `axis_classes` and scalar payload? The current
  `StructuredStorage::from_diag_col_major(..., 0)` behavior needs review before
  making rank-0 C API guarantees.
- Should C API strides accept negative values now? `StructuredStorage::new`
  currently uses `isize` strides, but negative-stride semantics and required
  storage length need explicit tests before advertising support.

## Success Criteria

- `TensorDynLen::from_diag(...)` preserves compact diagonal storage.
- `TensorDynLen::from_storage(...)` preserves arbitrary canonical
  `axis_classes`.
- `TensorDynLen::to_storage()` does not densify structured tensors.
- C API can construct and inspect structured tensors, including compact payload
  readback, without dense materialization.
- Existing dense constructors and dense readback continue to work.
- Diagonal/copy tensor contraction remains correct and uses structured metadata.
