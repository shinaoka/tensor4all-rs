# EagerTensor-Based Migration — Design Document

Date: 2026-04-13
Repositories: tenferro-rs, tensor4all-rs

---

## 1. Problem Statement

tensor4all-rs depends on an old tenferro-rs revision (`c4e18845`) where
`tenferro::Tensor` was a rich type with AD, structure info, element access,
and operations. The new tenferro-rs has a clean separation:

```
Old:  Tensor = data + AD + structure + ops (one type does everything)

New:  TypedTensor<T> = data + shape only
      Tensor         = enum{F32|F64|C32|C64} + basic ops (via TensorBackend)
      EagerTensor    = Arc<Tensor> + AD (backward, grad)
```

Additionally, several tenferro crates were merged/removed:

| Old crate | New location |
|-----------|-------------|
| `tenferro-linalg` | `tenferro-tensor` (TypedTensor::svd/qr methods) |
| `tenferro-prims` | `tenferro-tensor::cpu` (CpuContext, CpuBackend) |
| `tenferro-tensor-compute` | `tenferro-einsum` (typed_eager_einsum) |

Key symbol mapping:

| Old symbol | New symbol |
|-----------|-----------|
| `tenferro::ScalarType` | `tenferro_tensor::DType` |
| `tensor.scalar_type()` | `tensor.dtype()` |
| `tensor.dims()` | `tensor.shape()` |
| `tensor.ndim()` | `tensor.shape().len()` |
| `tensor.try_get::<T>(&idx)` | `tensor.as_slice::<T>()` + index |
| `tensor.try_to_vec::<T>()` | `tensor.as_slice::<T>().map(\|s\| s.to_vec())` |
| `tensor.is_dense()` | always true (no diagonal in new Tensor) |
| `tensor.to_dense()` | identity (already dense) |
| `Tensor::from_slice(data, dims, order)` | `Tensor::new(shape, data)` |
| `TypedTensor::from_slice(data, dims, order)` | `TypedTensor::from_vec(shape, data)` |
| `NativeTensor::from(typed)` | `Tensor::from(typed)` (unchanged) |
| `tensor.requires_grad()` | `EagerTensor::tracks_grad()` |
| `tensor.zero_grad()` | `EagerTensor::clear_grad()` |
| `tensor.backward()` | moved to `EagerTensor::backward()` |
| `tensor.backward_with_seed(seed)` | removed; initial migration exposes scalar-loss `backward()` only |
| `tensor.grad()` | `EagerTensor::grad()` (accumulated until cleared) |
| `tenferro_linalg::LinalgScalar` | removed (use `TensorScalar`) |
| `tenferro_linalg::svd(ctx, tensor)` | `TypedTensor::svd(&self, ctx)` |
| `tenferro_linalg::qr(ctx, tensor)` | `TypedTensor::qr(&self, ctx)` |
| `tenferro_prims::CpuContext` | `tenferro_tensor::cpu::CpuContext` |
| `tenferro_tensor_compute::einsum` | `tenferro_einsum::typed_eager_einsum` |
| `tenferro_prims::print_and_reset_contract_profile` | removed |

Target migration baseline: re-pin to a tenferro-rs revision at or after the
2026-04-13 merge that introduced the final eager AD surface used here:
`EagerTensor::tracks_grad()`, `EagerTensor::clear_grad()`,
`EagerContext::clear_grads()`, Send+Sync eager tensors, and PyTorch-style
gradient accumulation in `EagerTensor::backward()`.

## 2. Design Principles

1. **Eager execution** -- every operation produces a concrete result immediately.
2. **AD is optional** -- `EagerTensor` with `requires_grad=false` (default)
   has zero overhead. Only primal computation runs.
3. **Selective public compatibility** -- preserve
   `tensor4all_core::AnyScalar` as the public dynamic scalar entry point, but
   do not preserve legacy AD method names. AD-facing APIs adopt the eager
   naming (`enable_grad`, `tracks_grad`, `clear_grad`, ...) instead of legacy
   shims such as `set_requires_grad` or `zero_grad`.
4. **Two-tier tensor types:**
   - `TensorDynLen` wraps `EagerTensor` (index labels, dynamic dtype,
     optional AD, diagonal structure)
   - SimpleTT `Tensor<T,N>` wraps `TypedTensor<T>` (no index labels,
     static dtype, no AD)
5. **Paired thread-local execution objects** -- tensor4all-rs uses one
   thread-local `CpuBackend` for `TypedTensor` operations and one
   thread-local `EagerContext<CpuBackend>` for eager graphs. They are paired
   by thread, but they are not the same backend instance.
6. **Explicit gradient lifecycle** -- `backward()` accumulates gradients;
   callers clear them explicitly with clear APIs. The migration must not
   reintroduce implicit gradient clearing.
7. **Bottom-up migration** -- migrate by crate dependency order:
   tensorbackend -> tcicore -> core -> simplett -> higher crates -> capi.

## 3. Migration by Crate (Bottom-Up)

### 3.1 tensorbackend

The bridge layer between tenferro and tensor4all-rs. Four files need changes:

#### 3.1.1 Cargo.toml

Remove: `tenferro-prims`, `tenferro-linalg`.
Add: `tenferro` (for EagerTensor, EagerContext).
Keep: `tenferro-tensor`, `tenferro-einsum`, `tenferro-algebra`.

#### 3.1.2 any_scalar.rs (backend AnyScalar)

`tensor4all_tensorbackend::AnyScalar` wraps a rank-0 `tenferro::Tensor` for
dynamic scalar values. It remains a backend helper for storage conversion and
primal scalar arithmetic; it is not the final public compatibility layer for
`tensor4all_core::AnyScalar`.

**Changes:**
- `ScalarType` -> `DType`, `.scalar_type()` -> `.dtype()`
- `NativeTensor::from_slice(&[value], &[])` -> `Tensor::new(vec![], vec![value])`
- `.try_get::<T>(&[])` -> `.as_slice::<T>().unwrap()[0]`
- `.dims()` -> `.shape()`
- Remove AD methods from backend `AnyScalar` entirely:
  `requires_grad`, `set_requires_grad`, `zero_grad`, `backward`,
  `backward_with_seed`, `grad`, and `detach`.
  Core-level AD is reintroduced later through a core-owned
  `tensor4all_core::AnyScalar` wrapper around rank-0 `TensorDynLen`.
  Backend `AnyScalar` becomes a pure primal-value type.

**Resulting backend AnyScalar:**
```rust
pub struct Scalar {
    native: Tensor,  // rank-0, always primal
}
```

No AD methods. Construction, real/imag/abs/arithmetic stay the same
but updated for the new Tensor API. This type remains useful for
`Storage`/backend interop even after `tensor4all_core::AnyScalar` stops
re-exporting it directly.

#### 3.1.3 tensor_element.rs (TensorElement trait)

Trait for type-dispatched tensor construction and extraction.

**Changes:**
- `TypedTensor::<T>::from_slice(data, dims, MemoryOrder::ColumnMajor)`
  -> `TypedTensor::<T>::from_vec(shape, data)` (data must be in
  column-major order already; TypedTensor stores data as-is)
- `NativeTensor::from(typed)` -> `Tensor::from(typed)` (same)
- `.is_dense()` -> remove (always true in new Tensor)
- `.try_to_vec::<T>()` -> `.as_slice::<T>().map(|s| s.to_vec())`
- `.to_dense()` -> remove (no-op)
- `.ndim()` -> `.shape().len()`
- `.dims()` -> `.shape()`
- `.try_get::<T>(&index)` -> index into `.as_slice::<T>()`

#### 3.1.4 backend.rs (SVD/QR backend)

**Changes:**
- Remove `tenferro_linalg` dependency entirely.
- `LinalgScalar` -> `TensorScalar` (from `tenferro_tensor`)
- `KernelLinalgScalar` -> remove
- `BackendLinalgScalar` -> simplify to just `TensorScalar` bound
- `svd_backend()`: `tenferro_linalg::svd(ctx, tensor)` ->
  `TypedTensor::svd(&self, ctx: &mut CpuBackend)`
- `qr_backend()`: `tenferro_linalg::qr(ctx, tensor)` ->
  `TypedTensor::qr(&self, ctx: &mut CpuBackend)`
- `with_tenferro_ctx(...)` -> `with_default_backend(...)` (shared context)
- `SvdResult<T>`: singular values type uses `T::Real`
  (from `TensorScalar::Real` associated type)

#### 3.1.5 tenferro_bridge.rs

Most functions become thin wrappers or are removed. The bridge was needed
because the old Tensor had a complex API. The new Tensor is simple enough
for direct use.

**Keep (updated):**
- `storage_to_native_tensor()` -- conversion from Storage to Tensor
- `native_tensor_primal_to_storage()` -- conversion from Tensor to Storage
- `dense_native_tensor_from_col_major()` -- construction helper
- `diag_native_tensor_from_col_major()` -- construction helper
- `native_tensor_primal_to_dense_*_col_major()` -- extraction helpers
- `native_tensor_primal_to_diag_*()` -- extraction helpers
- Profiling infrastructure (einsum profiling)

**Remove (replaced by EagerTensor/TypedTensor methods):**
- `with_tenferro_ctx()` -- replaced by shared context
- `contract_native_tensor()` -- replaced by `eager_einsum_ad`
- `outer_product_native_tensor()` -- replaced by `eager_einsum_ad`
- `svd_native_tensor()` -- replaced by `EagerTensor::svd()`
- `qr_native_tensor()` -- replaced by `EagerTensor::qr()`
- `scale_native_tensor()` -- replaced by `EagerTensor::mul()`
- `axpby_native_tensor()` -- replaced by EagerTensor arithmetic
- `permute_native_tensor()` -- replaced by `EagerTensor::transpose()`
- `reshape_col_major_native_tensor()` -- replaced by `EagerTensor::reshape()`
- `conj_native_tensor()` -- replaced by `EagerTensor::conj()`
- `sum_native_tensor()` -- replaced by `EagerTensor::reduce_sum()`
- `tangent_native_tensor()` -- AD moved to EagerTensor
- All `*_storage_native()` variants -- can be reimplemented via
  Storage -> Tensor -> operation -> Storage if still needed

**Update (API changes):**
- `CpuContext::new(n)` -> `CpuContext::with_threads(n)`
- `CpuContext::default_num_threads()` -> `tenferro_tensor::cpu::available_parallelism()`
- All `dims()` -> `shape()`, `scalar_type()` -> `dtype()`
- `Tensor::from_slice()` -> `Tensor::new()`

#### 3.1.6 lib.rs

- Remove `print_and_reset_contract_profile` re-export (symbol removed)
- Remove re-exports of deleted bridge functions
- Add re-export of context module

#### 3.1.7 New: context.rs (paired thread-local execution objects)

```rust
use std::sync::Arc;
use tenferro_ad::EagerRuntime;
use tensor4all_tensorbackend::EagerContextError;

pub fn default_eager_ctx()
    -> std::result::Result<Arc<EagerRuntime>, EagerContextError>
```

The runtime is process-global and its backends share the process-global CPU
context, but registration can fail. Callers must propagate the typed
`EagerContextError` rather than assuming initialization is infallible.
This is intentionally *not* documented as "one shared backend instance."
`TypedTensor` APIs need a mutable `CpuBackend`, while `EagerTensor` owns its
backend through `EagerRuntime`. The migration should keep that distinction
explicit in both code and docs.

### 3.2 tcicore

#### 3.2.1 Cargo.toml

Remove: `tenferro-tensor-compute`.
Add: `tenferro-einsum`, `tenferro-tensor`, `tenferro-algebra`.

#### 3.2.2 matrix.rs

- `tenferro_tensor_compute::einsum::<Standard<T>, CpuBackend>(...)` ->
  `tenferro_einsum::typed_eager_einsum(&mut backend, inputs, subscripts)`
- Use `with_default_backend()` for the CpuBackend.

### 3.3 core (TensorDynLen)

#### 3.3.1 Cargo.toml

Remove: `tenferro-prims` (dev-dep).
Add: `tenferro` (for `EagerTensor`, `EagerContext`, and eager einsum).

#### 3.3.2 tensordynlen.rs

**Struct change:**
```rust
// Old:
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    native: Arc<NativeTensor>,
}

// New:
pub struct TensorDynLen {
    pub indices: Vec<DynIndex>,
    inner: EagerTensor<CpuBackend>,
    axis_classes: Vec<usize>,
}
```

**Data access pattern:**
```rust
// Old: &*self.native  or  self.native.as_ref()
// New: self.inner.data()  (returns &Tensor)
```

**Construction pattern:**
```rust
let ctx = tensor4all_tensorbackend::default_eager_ctx()?;
let inner = EagerTensor::from_tensor_in(native_tensor, ctx);
```

**Core primal operations use eager APIs directly:**
- contraction: `tenferro::eager_einsum::eager_einsum_ad(&[...], subscripts)`
- factorization staging: reshape/transposition on `EagerTensor`
- SVD/QR on reshaped payload tensors
- permutation: `transpose`
- scaling and elementwise ops: eager arithmetic
- norms/reductions: eager reductions over the primal payload

#### 3.3.3 any_scalar.rs (core-owned AnyScalar compatibility wrapper)

`tensor4all_core::AnyScalar` should stop re-exporting
`tensor4all_tensorbackend::AnyScalar`. Because `tensor4all-core` depends on
`tensor4all-tensorbackend`, the public scalar compatibility layer must live in
core and wrap rank-0 `TensorDynLen`; otherwise reintroducing AD would require a
reverse crate dependency.

```rust
pub struct AnyScalar {
    inner: TensorDynLen,  // invariant: rank 0
}
```

Required surface:
- Preserve the public type name `tensor4all_core::AnyScalar`.
- Preserve scalar constructors and primal helpers such as `new_real`,
  `new_complex`, `from_value`, `real`, `imag`, `abs`, `is_real`,
  `is_complex`, `is_zero`, `conj`, and arithmetic.
- Reintroduce AD only on this core-owned wrapper, using the eager names:
  - `enable_grad(self) -> Self`
  - `tracks_grad(&self) -> bool`
  - `grad(&self) -> Result<Option<AnyScalar>>`
  - `clear_grad(&self) -> Result<()>`
  - `backward(&self) -> Result<()>`
  - `detach(&self) -> Self`
- Add tensor interop helpers:
  - `try_from_tensor(TensorDynLen) -> Result<Self>`
  - `as_tensor(&self) -> &TensorDynLen`
  - `into_tensor(self) -> TensorDynLen`

Recommended usage contract:

```rust
fn affine(x: &AnyScalar, y: &AnyScalar) -> AnyScalar {
    x * y + x
}

let x = AnyScalar::new_real(2.0).enable_grad();
let y = AnyScalar::new_real(3.0).enable_grad();
let loss = &x * &y + &x;
loss.backward()?;
let dx = x.grad()?.unwrap();
```

Ownership and handle rules:
- For named owned values, the documented arithmetic contract is borrow-based:
  `&x * &y + &x`. The migration does not promise `x * y + x` for owned
  variables.
- Function parameters should normally be `&AnyScalar`; inside those functions
  `x * y + x` is acceptable because the bindings are already references.
- `Clone` is a shared-handle clone of the same tracked rank-0 tensor. It does
  not detach and it does not deep-copy the payload.
- Dropping one handle only destroys that handle. It does not clear gradients on
  other live clones, and there is no value-returning `close()` expression in
  the public API.

#### 3.3.4 defaults/svd.rs and defaults/qr.rs

These public entry points must be migrated explicitly. They currently depend on
legacy `ScalarType` dispatch and therefore belong in the main migration plan,
not in a later cleanup pass.

**Changes:**
- Replace `tenferro::ScalarType` dispatch with the new dtype path
  (`tenferro_tensor::DType` or local typed helpers).
- Keep the tensor4all public entry points stable while routing the payload
  through the new eager/typed backend path.
- Audit any shape/materialization assumptions while moving from old
  `NativeTensor` helpers to the new eager payload accessors.

#### 3.3.5 AD surface on TensorDynLen

The migrated wrapper should align with the final tenferro eager API names
instead of preserving ambiguous legacy names.

```rust
impl TensorDynLen {
    pub fn enable_grad(self) -> Self { ... }
    pub fn tracks_grad(&self) -> bool { ... }
    pub fn grad(&self) -> Result<Option<TensorDynLen>> { ... }
    pub fn clear_grad(&self) -> Result<()> { ... }
    pub fn backward(&self) -> Result<()> { ... }   // scalar loss only
    pub fn detach(&self) -> Self { ... }
}
```

Design rules:
- `enable_grad(self)` rebuilds a tracked eager leaf in the default eager context.
- `tracks_grad()` mirrors tenferro naming. It means the tensor participates in
  gradient tracking, not that an accumulated gradient value is currently present.
- `backward()` uses tenferro's accumulation semantics. Repeated calls add into
  the existing stored gradient until the caller clears it explicitly.
- `clear_grad()` is the explicit reset operation on the current tensor.
- `backward_with_seed(...)` is intentionally dropped from the public wrapper.
  Initial migration supports scalar-loss reverse mode only.
- `tensor4all_core::AnyScalar` mirrors the same naming and accumulation rules
  on top of a rank-0 `TensorDynLen` wrapper.

**Diagonal structure (`axis_classes`):**
- Dense tensors: `axis_classes = [0, 1, 2, ...]`
- Diagonal: `axis_classes = [0, 0]`
- Payload tensor has one axis per equivalence class
- Initial migration can keep payloads dense while preserving
  `axis_classes` metadata in tensor4all-rs

#### 3.3.6 tests/tensor_native_ad.rs and new AD integration coverage

- Replace legacy AD tests that rely on `set_requires_grad`, `zero_grad`, and
  seeded backward.
- Add new tests for:
  - `enable_grad` / `tracks_grad`
  - `grad`
  - repeated `backward()` accumulation
  - `clear_grad()`
  - `detach()`
- Extend `tensor_any_scalar.rs` to cover:
  - borrow-based named-value arithmetic (`&x * &y + &x`)
  - function-by-reference usage
  - rank-0 tensor conversion helpers
  - shared-handle `Clone` semantics for tracked scalars
- Add compile-time Send+Sync assertions for `EagerTensor<CpuBackend>` and
  `TensorDynLen`.

### 3.4 simplett

#### 3.4.1 Cargo.toml

Remove: `tenferro-linalg`, `tenferro-tensor-compute`.
Add: `tenferro-einsum`.

#### 3.4.2 einsum_helper.rs

Replace the old helper path:
- remove `EinsumScalar` trait and macro
- remove per-call `CpuContext::new(1)`
- use `with_default_backend()` + `typed_eager_einsum()`
- remove row-major helper code that depended on old tensor APIs

#### 3.4.3 compression.rs

- `LinalgScalar` -> `TensorScalar`
- `svd_backend()` -> `TypedTensor::svd(&self, ctx)` via `with_default_backend()`
- remove `MemoryOrder` / `contiguous()` calls
- `TypedTensor::from_slice()` -> `TypedTensor::from_vec()`

#### 3.4.4 tensor.rs, types.rs, contraction.rs, vidal.rs, mpo/*.rs

The migration scope must explicitly include the simplett files that still use
legacy tenferro tensor constructors, layout helpers, or linalg traits:

- `src/tensor.rs`
- `src/types.rs`
- `src/contraction.rs`
- `src/vidal.rs`
- `src/mpo/environment.rs`
- `src/mpo/factorize.rs`
- `src/mpo/types.rs`

Expected changes:
- remove `from_row_major_slice`, `MemoryOrder`, `contiguous()`, `buffer()`,
  and `is_unique()`
- use explicit row-major <-> column-major relayout helpers where needed
- replace old linalg trait bounds with `TensorScalar` or local typed helpers
- keep layout conversion visible and named rather than hidden behind blind
  `from_slice -> from_vec` renames

### 3.5 Higher-level crates (treetn, treetci, quanticstci, etc.)

Should compile without changes once lower crates are fixed, unless they
directly import removed symbols. Quick grep and fix pass.

#### 3.5.1 treetn

- `tenferro-prims` dev-dep -> `tenferro-tensor`
- `tenferro_prims::CpuContext::new(1)` -> `tenferro_tensor::cpu::CpuContext::with_threads(1)`

### 3.6 capi

The C API uses opaque handles to TensorDynLen. Since TensorDynLen's public
API is preserved, the C API should compile without changes.

**Thread safety:** With Arc/Mutex in EagerTensor (tenferro-rs change),
`TensorDynLen` remains `Send + Sync`, so `t4a_tensor` safety is preserved.

## 4. Implementation Order

Strict bottom-up. Each stage must compile and pass its crate-local tests
before the next stage starts.

```text
Step 1: tensorbackend foundations + repin
Step 2: tcicore typed einsum migration
Step 3: core primal migration (TensorDynLen storage + contract/factorize + defaults/svd.rs + defaults/qr.rs)
Step 4: core AD surface + core AnyScalar compatibility wrapper
Step 5: simplett typed-tensor + layout migration
Step 6: higher crates, benches, and Send+Sync / FFI-facing invariants
Step 7: full workspace verification and cleanup
```

## 5. Testing Strategy

Each stage has explicit verification:

1. **tensorbackend**:
   `cargo check -p tensor4all-tensorbackend --all-targets`
   `cargo nextest run --release -p tensor4all-tensorbackend`
2. **tcicore**:
   `cargo check -p tensor4all-tcicore --all-targets`
   `cargo nextest run --release -p tensor4all-tcicore`
3. **core primal + AD + AnyScalar**:
   `cargo check -p tensor4all-core --all-targets`
   `cargo nextest run --release -p tensor4all-core`
   plus dedicated accumulation/reset tests for the new AD surface and
   `tensor4all_core::AnyScalar`
4. **simplett**:
   `cargo check -p tensor4all-simplett --all-targets`
   `cargo nextest run --release -p tensor4all-simplett`
5. **Workspace sweep**:
   `cargo check --workspace --all-targets`
   `cargo nextest run --release --workspace`
6. **Final verification**:
   `cargo fmt --all`
   `cargo clippy --workspace --all-targets`
   `cargo test --doc --release --workspace`
   `./scripts/test-mdbook.sh`
   `cargo doc --workspace --no-deps`

## 6. Risks

1. **Memory order:** `TypedTensor::from_vec()` stores data as-is. Every
   migrated call site must make layout explicit.
2. **Diagonal tensors:** the new tenferro `Tensor` is always dense. tensor4all
   must preserve diagonal semantics in its own metadata (`axis_classes`,
   `Storage`) rather than expecting structural tensor support upstream.
3. **Gradient lifecycle confusion:** eager backward now accumulates like
   PyTorch. The tensor4all wrappers (`TensorDynLen` and core `AnyScalar`) must
   expose explicit clear APIs and tests so users do not infer legacy overwrite
   semantics.
4. **Execution-context wording:** tensor4all-rs must not claim that eager and
   typed operations share one backend instance unless that is demonstrably true.
5. **Performance:** `typed_eager_einsum` and typed linalg still copy through
   tenferro's current internal representations. These are upstream costs, not
   migration-specific regressions.
