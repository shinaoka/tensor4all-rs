# Structured Tensor Storage Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make structured tensor payload metadata first-class in `TensorDynLen` and expose construction/readback through the C API for issue #434.

**Architecture:** `Storage` remains the structured source of truth for dense, diagonal, and general structured payloads. `TensorDynLen` stores `Arc<Storage>` as its canonical payload and lazily materializes a tenferro `EagerTensor` only for operations that need native execution or AD tracking. The C API constructs tensors through the same storage path and adds compact payload metadata/readback functions alongside existing dense logical readback.

**Tech Stack:** Rust workspace, `tensor4all-tensorbackend`, `tensor4all-core`, `tensor4all-capi`, tenferro eager tensors, `anyhow`, `num_complex::Complex64`, `cbindgen`, cargo release tests.

---

## Prerequisites

- Read `README.md` and `docs/plans/2026-04-22-issue-434-structured-tensor-design.md`.
- Regenerate API docs before source exploration if public API context is stale:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

- Keep source code and docs in English.
- Do not add new C API `catch_unwind` / `Err(_) => T4A_INTERNAL_ERROR` patterns that discard error details. Use `run_catching`, `run_status`, `capi_error`, or existing `unwrap_catch` helpers.
- Run tests in release mode.

## Task 1: Expose Storage Metadata Accessors

**Files:**
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Test: `crates/tensor4all-tensorbackend/src/storage/tests/mod.rs`

**Step 1: Write the failing storage metadata tests**

Add these tests near the existing storage type inspection tests:

```rust
#[test]
fn storage_kind_and_metadata_accessors_cover_dense_diag_and_structured() {
    let dense = Storage::from_dense_col_major(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(dense.storage_kind(), StorageKind::Dense);
    assert_eq!(dense.logical_dims(), vec![2, 2]);
    assert_eq!(dense.logical_rank(), 2);
    assert_eq!(dense.payload_dims(), &[2, 2]);
    assert_eq!(dense.payload_strides(), &[1, 2]);
    assert_eq!(dense.axis_classes(), &[0, 1]);
    assert_eq!(dense.payload_len(), 4);
    assert_eq!(dense.payload_f64_col_major_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let diag = Storage::from_diag_col_major(vec![10.0_f64, 20.0], 2).unwrap();
    assert_eq!(diag.storage_kind(), StorageKind::Diagonal);
    assert_eq!(diag.logical_dims(), vec![2, 2]);
    assert_eq!(diag.payload_dims(), &[2]);
    assert_eq!(diag.axis_classes(), &[0, 0]);
    assert_eq!(diag.payload_f64_col_major_vec().unwrap(), vec![10.0, 20.0]);

    let structured = Storage::new_structured(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        vec![1, 2],
        vec![0, 1, 0],
    )
    .unwrap();
    assert_eq!(structured.storage_kind(), StorageKind::Structured);
    assert_eq!(structured.logical_dims(), vec![2, 3, 2]);
    assert_eq!(structured.payload_dims(), &[2, 3]);
    assert_eq!(structured.payload_strides(), &[1, 2]);
    assert_eq!(structured.axis_classes(), &[0, 1, 0]);
}

#[test]
fn storage_payload_c64_readback_is_interpreted_as_payload_not_logical_dense() {
    let data = vec![Complex64::new(1.0, -1.0), Complex64::new(2.0, -2.0)];
    let storage = Storage::from_diag_col_major(data.clone(), 2).unwrap();
    assert_eq!(storage.storage_kind(), StorageKind::Diagonal);
    assert_eq!(storage.payload_c64_col_major_vec().unwrap(), data);
    assert!(storage.payload_f64_col_major_vec().unwrap_err().contains("expected f64"));
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend storage_kind_and_metadata_accessors_cover_dense_diag_and_structured
cargo test --release -p tensor4all-tensorbackend storage_payload_c64_readback_is_interpreted_as_payload_not_logical_dense
```

Expected: FAIL because `StorageKind` and the new accessor methods do not exist.

**Step 3: Add `StorageKind` and public accessors**

In `crates/tensor4all-tensorbackend/src/storage.rs`, add this public enum near `Storage`:

```rust
/// Classifies the compact layout used by [`Storage`].
///
/// `Dense` means every logical axis has its own payload axis. `Diagonal` means
/// all logical axes share one payload axis. `Structured` covers all remaining
/// axis-equivalence layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageKind {
    /// Logical dense payload layout.
    Dense,
    /// Diagonal or copy-tensor payload layout.
    Diagonal,
    /// General structured payload layout with repeated axis classes.
    Structured,
}
```

Then add methods in `impl Storage` after `is_diag` or before scalar-kind helpers:

```rust
pub fn storage_kind(&self) -> StorageKind {
    if self.is_dense() {
        StorageKind::Dense
    } else if self.is_diag() {
        StorageKind::Diagonal
    } else {
        StorageKind::Structured
    }
}

pub fn logical_dims(&self) -> Vec<usize> {
    match &self.0 {
        StorageRepr::F64(value) => value.logical_dims(),
        StorageRepr::C64(value) => value.logical_dims(),
    }
}

pub fn logical_rank(&self) -> usize {
    match &self.0 {
        StorageRepr::F64(value) => value.logical_rank(),
        StorageRepr::C64(value) => value.logical_rank(),
    }
}

pub fn payload_dims(&self) -> &[usize] {
    match &self.0 {
        StorageRepr::F64(value) => value.payload_dims(),
        StorageRepr::C64(value) => value.payload_dims(),
    }
}

pub fn payload_strides(&self) -> &[isize] {
    match &self.0 {
        StorageRepr::F64(value) => value.strides(),
        StorageRepr::C64(value) => value.strides(),
    }
}

pub fn axis_classes(&self) -> &[usize] {
    match &self.0 {
        StorageRepr::F64(value) => value.axis_classes(),
        StorageRepr::C64(value) => value.axis_classes(),
    }
}

pub fn payload_len(&self) -> usize {
    self.len()
}

pub fn payload_f64_col_major_vec(&self) -> Result<Vec<f64>, String> {
    match &self.0 {
        StorageRepr::F64(value) => Ok(value.payload_col_major_vec()),
        StorageRepr::C64(_) => Err("expected f64 storage when copying f64 payload".to_string()),
    }
}

pub fn payload_c64_col_major_vec(&self) -> Result<Vec<Complex64>, String> {
    match &self.0 {
        StorageRepr::C64(value) => Ok(value.payload_col_major_vec()),
        StorageRepr::F64(_) => {
            Err("expected Complex64 storage when copying c64 payload".to_string())
        }
    }
}
```

Add rustdoc comments and runnable examples for every public item to satisfy `#![warn(missing_docs)]`.

**Step 4: Re-export `StorageKind`**

In `crates/tensor4all-tensorbackend/src/lib.rs`, include `StorageKind` in the storage export list.

In `crates/tensor4all-core/src/lib.rs`, include `StorageKind` in both the `storage` module re-export and the top-level `pub use storage::{...}` line.

**Step 5: Run tests to verify they pass**

Run:

```bash
cargo test --release -p tensor4all-tensorbackend storage_kind_and_metadata_accessors_cover_dense_diag_and_structured
cargo test --release -p tensor4all-tensorbackend storage_payload_c64_readback_is_interpreted_as_payload_not_logical_dense
cargo test --doc --release -p tensor4all-tensorbackend StorageKind
```

Expected: PASS.

**Step 6: Commit**

```bash
git add crates/tensor4all-tensorbackend/src/storage.rs crates/tensor4all-tensorbackend/src/lib.rs crates/tensor4all-core/src/lib.rs crates/tensor4all-tensorbackend/src/storage/tests/mod.rs
git commit -m "feat(storage): expose structured metadata accessors"
```

## Task 2: Add TensorDynLen Storage-First Regression Tests

**Files:**
- Modify: `crates/tensor4all-core/tests/tensor_basic.rs`
- Modify: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Write failing `TensorDynLen::from_storage` tests**

In `crates/tensor4all-core/tests/tensor_basic.rs`, add:

```rust
#[test]
fn tensor_from_structured_storage_preserves_compact_payload() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let storage = Arc::new(
        Storage::new_structured(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1, 0],
        )
        .unwrap(),
    );

    let tensor = TensorDynLen::from_storage(vec![i, j, k], Arc::clone(&storage)).unwrap();
    let snapshot = tensor.storage();

    assert_eq!(snapshot.storage_kind(), tensor4all_core::StorageKind::Structured);
    assert_eq!(snapshot.payload_dims(), &[2, 3]);
    assert_eq!(snapshot.payload_strides(), &[1, 2]);
    assert_eq!(snapshot.axis_classes(), &[0, 1, 0]);
    assert_eq!(
        snapshot.payload_f64_col_major_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(tensor.dims(), vec![2, 3, 2]);
}

#[test]
fn tensor_from_structured_storage_rejects_index_dim_mismatch() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(4);
    let storage = Arc::new(
        Storage::new_structured(
            vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![1, 2],
            vec![0, 1],
        )
        .unwrap(),
    );

    let err = TensorDynLen::from_storage(vec![i, j], storage).unwrap_err();
    assert!(err.to_string().contains("storage logical dims"));
}
```

**Step 2: Write failing diagonal preservation tests**

Update existing diagonal tests in `crates/tensor4all-core/tests/tensor_diag.rs` that currently assert `!tensor.is_diag()`. The new expected behavior is compact diagonal preservation.

Change:

```rust
assert!(!tensor.is_diag());
```

to:

```rust
assert!(tensor.is_diag());
assert_eq!(tensor.storage().storage_kind(), tensor4all_core::StorageKind::Diagonal);
```

Do this for:
- `test_diag_tensor_creation`
- `test_diag_tensor_scale_preserves_diagonal_values`
- `test_diag_tensor_contract_diag_diag_partial` only if the result remains diagonal after the implementation task; if the operation still materializes, keep value assertions and add a separate preservation test instead.
- `test_diag_tensor_rank3`

Add a focused test:

```rust
#[test]
fn from_diag_storage_roundtrip_uses_payload_not_dense_logical_values() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let tensor = TensorDynLen::from_diag(vec![i, j], vec![1.0_f64, 2.0, 3.0]).unwrap();
    let storage = tensor.storage();

    assert_eq!(storage.storage_kind(), tensor4all_core::StorageKind::Diagonal);
    assert_eq!(storage.payload_dims(), &[3]);
    assert_eq!(storage.axis_classes(), &[0, 0]);
    assert_eq!(storage.payload_f64_col_major_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(
        tensor.to_vec::<f64>().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    );
}
```

**Step 3: Run tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-core --test tensor_basic tensor_from_structured_storage_preserves_compact_payload
cargo test --release -p tensor4all-core --test tensor_basic tensor_from_structured_storage_rejects_index_dim_mismatch
cargo test --release -p tensor4all-core --test tensor_diag from_diag_storage_roundtrip_uses_payload_not_dense_logical_values
```

Expected: FAIL because `TensorDynLen::from_storage` densifies and `from_diag` builds native dense storage.

**Step 4: Commit failing tests**

```bash
git add crates/tensor4all-core/tests/tensor_basic.rs crates/tensor4all-core/tests/tensor_diag.rs
git commit -m "test(core): cover structured tensor storage preservation"
```

## Task 3: Convert TensorDynLen To Storage-First Construction

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_basic.rs`
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`

**Step 1: Change the struct fields**

Replace the existing fields:

```rust
pub(crate) inner: Arc<EagerTensor<CpuBackend>>,
pub(crate) axis_classes: Vec<usize>,
```

with:

```rust
pub(crate) storage: Arc<Storage>,
pub(crate) eager_cache: Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>>,
```

Add imports:

```rust
use std::sync::{Arc, OnceLock};
```

Keep `indices` public for the current API surface.

**Step 2: Add private cache/materialization helpers**

Add helpers near `seed_native_payload`:

```rust
fn empty_eager_cache() -> Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>> {
    Arc::new(OnceLock::new())
}

fn eager_cache_with(inner: EagerTensor<CpuBackend>) -> Arc<OnceLock<Arc<EagerTensor<CpuBackend>>>> {
    let cache = Arc::new(OnceLock::new());
    let _ = cache.set(Arc::new(inner));
    cache
}

fn storage_from_native_with_axis_classes(
    native: &NativeTensor,
    axis_classes: &[usize],
    logical_rank: usize,
) -> Result<Storage> {
    if Self::is_diag_axis_classes(axis_classes) {
        match native.dtype() {
            DType::F32 | DType::F64 => {
                Storage::from_diag_col_major(native_tensor_primal_to_diag_f64(native)?, logical_rank)
            }
            DType::C32 | DType::C64 => {
                Storage::from_diag_col_major(native_tensor_primal_to_diag_c64(native)?, logical_rank)
            }
        }
    } else {
        native_tensor_primal_to_storage(native)
    }
}

fn validate_storage_matches_indices(indices: &[DynIndex], storage: &Storage) -> Result<()> {
    let dims = Self::expected_dims_from_indices(indices);
    let storage_dims = storage.logical_dims();
    if storage_dims != dims {
        return Err(anyhow::anyhow!(
            "storage logical dims {:?} do not match indices dims {:?}",
            storage_dims,
            dims
        ));
    }
    if storage.is_diag() {
        Self::validate_diag_dims(&dims)?;
    }
    Ok(())
}

fn materialized_inner(&self) -> &EagerTensor<CpuBackend> {
    self.eager_cache
        .get_or_init(|| {
            let native = Self::seed_native_payload(self.storage.as_ref(), &self.dims())
                .unwrap_or_else(|err| panic!("TensorDynLen materialization failed: {err}"));
            Arc::new(EagerTensor::from_tensor_in(native, default_eager_ctx()))
        })
        .as_ref()
}
```

If clippy rejects `panic!` in the cache path, split callers into a fallible `try_materialized_inner` and keep `as_native` as the only panic wrapper. The invariant is that materialization is infallible after `from_storage` validation.

**Step 3: Update constructors**

Update `from_storage` so it no longer calls `storage_to_native_tensor`:

```rust
pub fn from_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
    Self::validate_indices(&indices);
    Self::validate_storage_matches_indices(&indices, storage.as_ref())?;
    Ok(Self {
        indices,
        storage,
        eager_cache: Self::empty_eager_cache(),
    })
}
```

Add the explicit alias:

```rust
pub fn from_structured_storage(indices: Vec<DynIndex>, storage: Arc<Storage>) -> Result<Self> {
    Self::from_storage(indices, storage)
}
```

Update `from_inner_with_axis_classes`:

```rust
let storage =
    Self::storage_from_native_with_axis_classes(inner.data(), &axis_classes, indices.len())?;
Ok(Self {
    indices,
    storage: Arc::new(storage),
    eager_cache: Self::eager_cache_with(inner),
})
```

Update `from_dense` to construct `Storage::from_dense_col_major(data, &dims)?` and call `from_storage`.

Update `from_diag` to construct `Storage::from_diag_col_major(data, dims.len())?` and call `from_storage`. Keep `validate_diag_payload_len` before construction.

**Step 4: Update storage and scalar metadata accessors**

Change:

```rust
pub fn to_storage(&self) -> Result<Arc<Storage>> { ... }
pub fn storage(&self) -> Arc<Storage> { ... }
pub fn is_diag(&self) -> bool { ... }
pub fn is_f64(&self) -> bool { ... }
pub fn is_complex(&self) -> bool { ... }
```

to:

```rust
pub fn to_storage(&self) -> Result<Arc<Storage>> {
    Ok(Arc::clone(&self.storage))
}

pub fn storage(&self) -> Arc<Storage> {
    Arc::clone(&self.storage)
}

pub fn is_diag(&self) -> bool {
    self.storage.is_diag()
}

pub fn is_f64(&self) -> bool {
    self.storage.is_f64()
}

pub fn is_complex(&self) -> bool {
    self.storage.is_complex()
}
```

**Step 5: Replace direct `inner` and `axis_classes` reads enough to compile**

Use the helpers:

- `self.inner` -> `self.materialized_inner()`
- `other.inner` -> `other.materialized_inner()`
- `self.as_native()` can remain and should call `self.materialized_inner().data()`
- `self.axis_classes` -> `self.storage.axis_classes()`
- clone-preserving constructors should clone `storage` and `eager_cache`

For `replaceind` and `replace_indices`, return:

```rust
Self {
    indices: new_indices,
    storage: Arc::clone(&self.storage),
    eager_cache: Arc::clone(&self.eager_cache),
}
```

For `enable_grad`, keep the same storage and seed the cache with a tracked eager leaf:

```rust
let native = self.as_native().clone();
Self {
    indices: self.indices,
    storage: self.storage,
    eager_cache: Self::eager_cache_with(EagerTensor::requires_grad_in(native, default_eager_ctx())),
}
```

For `detach`, use the detached eager tensor and rebuild through `from_inner_with_axis_classes` using `self.storage.axis_classes().to_vec()`.

**Step 6: Run targeted core tests**

Run:

```bash
cargo test --release -p tensor4all-core --test tensor_basic tensor_from_structured_storage_preserves_compact_payload
cargo test --release -p tensor4all-core --test tensor_basic tensor_from_structured_storage_rejects_index_dim_mismatch
cargo test --release -p tensor4all-core --test tensor_diag from_diag_storage_roundtrip_uses_payload_not_dense_logical_values
cargo test --release -p tensor4all-core --test tensor_basic test_tensor_shared_storage
```

Expected: PASS.

**Step 7: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_basic.rs crates/tensor4all-core/tests/tensor_diag.rs
git commit -m "feat(core): make TensorDynLen storage first"
```

## Task 4: Preserve Compact Storage Through Simple Tensor Operations

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_diag.rs`
- Test: `crates/tensor4all-core/tests/tensor_basic.rs`

**Step 1: Add failing preservation tests**

Add to `tensor_diag.rs`:

```rust
#[test]
fn diag_permute_scale_conj_and_replaceind_preserve_payload_metadata() {
    let i = Index::new_dyn(3);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(3);
    let tensor = TensorDynLen::from_diag(
        vec![i.clone(), j.clone(), k.clone()],
        vec![1.0_f64, -2.0, 4.0],
    )
    .unwrap();

    let permuted = tensor.permute(&[2, 0, 1]);
    assert!(permuted.is_diag());
    assert_eq!(permuted.storage().axis_classes(), &[0, 0, 0]);
    assert_eq!(permuted.storage().payload_f64_col_major_vec().unwrap(), vec![1.0, -2.0, 4.0]);

    let scaled = permuted.scale(AnyScalar::new_real(2.0)).unwrap();
    assert!(scaled.is_diag());
    assert_eq!(scaled.storage().payload_f64_col_major_vec().unwrap(), vec![2.0, -4.0, 8.0]);

    let replaced = scaled.replaceind(&k, &Index::new_dyn(3));
    assert!(replaced.is_diag());
    assert_eq!(replaced.storage().payload_f64_col_major_vec().unwrap(), vec![2.0, -4.0, 8.0]);

    let conjugated = replaced.conj();
    assert!(conjugated.is_diag());
    assert_eq!(conjugated.storage().payload_f64_col_major_vec().unwrap(), vec![2.0, -4.0, 8.0]);
}
```

Add to `tensor_basic.rs`:

```rust
#[test]
fn same_layout_axpby_preserves_structured_metadata() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let k = Index::new_dyn(2);
    let make = |offset| {
        TensorDynLen::from_storage(
            vec![i.clone(), j.clone(), k.clone()],
            Arc::new(
                Storage::new_structured(
                    vec![1.0 + offset, 2.0 + offset, 3.0 + offset, 4.0 + offset, 5.0 + offset, 6.0 + offset],
                    vec![2, 3],
                    vec![1, 2],
                    vec![0, 1, 0],
                )
                .unwrap(),
            ),
        )
        .unwrap()
    };

    let a = make(0.0);
    let b = make(10.0);
    let result = a.axpby(AnyScalar::new_real(2.0), &b, AnyScalar::new_real(-1.0)).unwrap();
    let storage = result.storage();

    assert_eq!(storage.storage_kind(), tensor4all_core::StorageKind::Structured);
    assert_eq!(storage.axis_classes(), &[0, 1, 0]);
    assert_eq!(storage.payload_f64_col_major_vec().unwrap(), vec![-9.0, -8.0, -7.0, -6.0, -5.0, -4.0]);
}
```

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-core --test tensor_diag diag_permute_scale_conj_and_replaceind_preserve_payload_metadata
cargo test --release -p tensor4all-core --test tensor_basic same_layout_axpby_preserves_structured_metadata
```

Expected: FAIL where operations materialize dense native results or drop axis metadata.

**Step 3: Implement storage-preserving operation branches**

Update these methods to use storage methods when AD tracking is not active:

- `permute_indices`
- `permute`
- `replaceind`
- `replace_indices`
- `conj`
- `scale`
- `axpby`

Rules:

- If `self.tracks_grad()` or another operand/scalar tracks grad, use the existing eager/native path.
- If operation is metadata-only (`replaceind`, `replace_indices`), clone storage/cache.
- For `permute`, use `self.storage.permute_storage(&self.dims(), perm)`.
- For `conj`, use `self.storage.conj()`.
- For `scale`, use `self.storage.scale(&scalar.to_backend_scalar() equivalent)` only through existing `Storage::scale(&tensor4all_tensorbackend::AnyScalar)` conversion already available. If scalar conversion is awkward, use `scalar.to_backend_scalar()` only at the backend boundary and add a small private conversion helper.
- For `axpby`, only preserve structured storage when:
  - both tensors have identical index order after alignment,
  - both storages have identical `payload_dims`, `payload_strides`, and `axis_classes`,
  - no involved tensor tracks grad.
  Otherwise materialize through the existing eager/native path.

Construct storage-preserving results through `TensorDynLen::from_storage(...)`.

**Step 4: Run tests to verify they pass**

Run:

```bash
cargo test --release -p tensor4all-core --test tensor_diag diag_permute_scale_conj_and_replaceind_preserve_payload_metadata
cargo test --release -p tensor4all-core --test tensor_basic same_layout_axpby_preserves_structured_metadata
cargo test --release -p tensor4all-core --test tensor_diag
cargo test --release -p tensor4all-core --test tensor_basic
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_diag.rs crates/tensor4all-core/tests/tensor_basic.rs
git commit -m "feat(core): preserve structured storage in simple tensor ops"
```

## Task 5: Keep Dense Execution Paths Correct

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Test: `crates/tensor4all-core/tests/tensor_contraction.rs`
- Test: `crates/tensor4all-core/tests/tensor_native_ad.rs`
- Test: `crates/tensor4all-core/tests/linalg_svd.rs`
- Test: `crates/tensor4all-core/tests/linalg_qr.rs`

**Step 1: Add focused fallback tests**

Add one test near existing contraction tests:

```rust
#[test]
fn structured_tensor_contract_materializes_to_correct_dense_result() {
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(2);
    let k = Index::new_dyn(2);
    let diag = TensorDynLen::from_diag(vec![i.clone(), j.clone()], vec![2.0_f64, 3.0]).unwrap();
    let dense = TensorDynLen::from_dense(vec![j, k.clone()], vec![5.0, 7.0, 11.0, 13.0]).unwrap();

    let result = diag.contract(&dense);

    let expected = TensorDynLen::from_dense(vec![i, k], vec![10.0, 21.0, 22.0, 39.0]).unwrap();
    assert!(result.distance(&expected).unwrap() < 1e-12);
}
```

If an equivalent test already exists, update it to assert `diag.is_diag()` before contraction and keep the dense whole-result comparison.

**Step 2: Run fallback-heavy tests**

Run:

```bash
cargo test --release -p tensor4all-core --test tensor_contraction structured_tensor_contract_materializes_to_correct_dense_result
cargo test --release -p tensor4all-core --test tensor_native_ad
cargo test --release -p tensor4all-core --test linalg_svd
cargo test --release -p tensor4all-core --test linalg_qr
```

Expected: PASS. If AD tests fail, inspect whether a storage-preserving branch skipped eager tracking. Fix by routing tracked tensors through eager/native code.

**Step 3: Replace remaining direct field assumptions**

Search:

```bash
rg -n "\\.inner|axis_classes" crates/tensor4all-core/src crates/tensor4all-core/tests
```

Allowed in `tensordynlen.rs`:
- struct field declarations
- cache helper internals
- comments/doc examples
- explicit `self.storage.axis_classes()` calls

Disallowed:
- downstream direct field access from other modules
- operations comparing stale copied axis metadata rather than storage metadata

**Step 4: Run the core crate**

Run:

```bash
cargo test --release -p tensor4all-core
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-core/tests/tensor_contraction.rs crates/tensor4all-core/tests/tensor_native_ad.rs crates/tensor4all-core/tests/linalg_svd.rs crates/tensor4all-core/tests/linalg_qr.rs
git commit -m "fix(core): keep native fallbacks correct with storage-first tensors"
```

## Task 6: Add C API Structured Tensor Surface

**Files:**
- Modify: `crates/tensor4all-capi/src/types.rs`
- Modify: `crates/tensor4all-capi/src/tensor.rs`
- Test: `crates/tensor4all-capi/src/tensor/tests/mod.rs`

**Step 1: Add failing C API tests**

In `crates/tensor4all-capi/src/tensor/tests/mod.rs`, add local helpers:

```rust
fn read_payload_dims(tensor: *const t4a_tensor) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(t4a_tensor_payload_dims(tensor, std::ptr::null_mut(), 0, &mut len), T4A_SUCCESS);
    let mut out = vec![0usize; len];
    assert_eq!(t4a_tensor_payload_dims(tensor, out.as_mut_ptr(), out.len(), &mut len), T4A_SUCCESS);
    out
}

fn read_axis_classes(tensor: *const t4a_tensor) -> Vec<usize> {
    let mut len = 0usize;
    assert_eq!(t4a_tensor_axis_classes(tensor, std::ptr::null_mut(), 0, &mut len), T4A_SUCCESS);
    let mut out = vec![0usize; len];
    assert_eq!(t4a_tensor_axis_classes(tensor, out.as_mut_ptr(), out.len(), &mut len), T4A_SUCCESS);
    out
}

fn read_payload_f64(tensor: *const t4a_tensor) -> Vec<f64> {
    let mut len = 0usize;
    assert_eq!(t4a_tensor_copy_payload_f64(tensor, std::ptr::null_mut(), 0, &mut len), T4A_SUCCESS);
    let mut out = vec![0.0; len];
    assert_eq!(t4a_tensor_copy_payload_f64(tensor, out.as_mut_ptr(), out.len(), &mut len), T4A_SUCCESS);
    out
}
```

Add tests:

```rust
#[test]
fn test_tensor_new_diag_f64_exposes_compact_payload() {
    let i = new_index(3);
    let j = new_index(3);
    let indices = [i as *const t4a_index, j as *const t4a_index];
    let mut tensor = std::ptr::null_mut();

    assert_eq!(
        t4a_tensor_new_diag_f64(2, indices.as_ptr(), [1.0, 2.0, 3.0].as_ptr(), 3, &mut tensor),
        T4A_SUCCESS
    );

    let mut kind = t4a_storage_kind::Dense;
    assert_eq!(t4a_tensor_storage_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_storage_kind::Diagonal);
    assert_eq!(read_payload_dims(tensor), vec![3]);
    assert_eq!(read_axis_classes(tensor), vec![0, 0]);
    assert_eq!(read_payload_f64(tensor), vec![1.0, 2.0, 3.0]);
    assert_eq!(read_dense_f64(tensor), vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
}

#[test]
fn test_tensor_new_structured_f64_roundtrips_metadata() {
    let i = new_index(2);
    let j = new_index(3);
    let k = new_index(2);
    let indices = [i as *const t4a_index, j as *const t4a_index, k as *const t4a_index];
    let payload = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let payload_dims = [2usize, 3usize];
    let payload_strides = [1isize, 2isize];
    let axis_classes = [0usize, 1usize, 0usize];
    let mut tensor = std::ptr::null_mut();

    assert_eq!(
        t4a_tensor_new_structured_f64(
            3,
            indices.as_ptr(),
            payload.as_ptr(),
            payload.len(),
            payload_dims.as_ptr(),
            payload_dims.len(),
            payload_strides.as_ptr(),
            payload_strides.len(),
            axis_classes.as_ptr(),
            axis_classes.len(),
            &mut tensor,
        ),
        T4A_SUCCESS
    );

    let mut kind = t4a_storage_kind::Dense;
    assert_eq!(t4a_tensor_storage_kind(tensor, &mut kind), T4A_SUCCESS);
    assert_eq!(kind, t4a_storage_kind::Structured);
    assert_eq!(read_payload_dims(tensor), vec![2, 3]);
    assert_eq!(read_axis_classes(tensor), vec![0, 1, 0]);
    assert_eq!(read_payload_f64(tensor), payload);

    t4a_tensor_release(tensor);
    t4a_index_release(i);
    t4a_index_release(j);
    t4a_index_release(k);
}
```

Also add a buffer-too-small test for one metadata function and an invalid-axis test that checks `t4a_last_error_message` contains `axis_classes`.

**Step 2: Run tests to verify they fail**

Run:

```bash
cargo test --release -p tensor4all-capi --lib tensor_new_diag_f64_exposes_compact_payload
cargo test --release -p tensor4all-capi --lib tensor_new_structured_f64_roundtrips_metadata
```

Expected: FAIL because the C symbols and enum do not exist.

**Step 3: Add C-facing enum**

In `crates/tensor4all-capi/src/types.rs`, add:

```rust
/// Compact tensor storage layout exposed by the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum t4a_storage_kind {
    /// Dense logical payload.
    Dense = 0,
    /// General structured payload.
    Structured = 1,
    /// Diagonal or copy-tensor payload.
    Diagonal = 2,
}

impl From<tensor4all_core::StorageKind> for t4a_storage_kind {
    fn from(kind: tensor4all_core::StorageKind) -> Self {
        match kind {
            tensor4all_core::StorageKind::Dense => Self::Dense,
            tensor4all_core::StorageKind::Structured => Self::Structured,
            tensor4all_core::StorageKind::Diagonal => Self::Diagonal,
        }
    }
}
```

**Step 4: Add C API helpers in `tensor.rs`**

Import `Storage`:

```rust
use tensor4all_core::{qr_with, svd_with, QrOptions, Storage, SvdOptions, SvdTruncationPolicy};
```

Add helpers:

```rust
fn read_usize_slice<'a>(
    ptr: *const usize,
    len: usize,
    name: &str,
) -> Result<&'a [usize], (StatusCode, String)> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{name} is null")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

fn read_isize_slice<'a>(
    ptr: *const isize,
    len: usize,
    name: &str,
) -> Result<&'a [isize], (StatusCode, String)> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(capi_error(T4A_NULL_POINTER, format!("{name} is null")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}
```

Add generic copy helper for metadata buffers if useful:

```rust
unsafe fn copy_slice_to_out<T: Copy>(
    values: &[T],
    buf: *mut T,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode {
    *out_len = values.len();
    if buf.is_null() {
        return T4A_SUCCESS;
    }
    if buf_len < values.len() {
        return T4A_BUFFER_TOO_SMALL;
    }
    std::ptr::copy_nonoverlapping(values.as_ptr(), buf, values.len());
    T4A_SUCCESS
}
```

**Step 5: Add metadata/readback functions**

Add these exported functions using `run_status` and `require_tensor`:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_storage_kind(
    ptr: *const t4a_tensor,
    out_kind: *mut t4a_storage_kind,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_rank(
    ptr: *const t4a_tensor,
    out_rank: *mut usize,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_dims(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_strides(
    ptr: *const t4a_tensor,
    buf: *mut isize,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_axis_classes(
    ptr: *const t4a_tensor,
    buf: *mut usize,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_payload_len(
    ptr: *const t4a_tensor,
    out_len: *mut usize,
) -> StatusCode { ... }
```

Each function should read `let storage = tensor.inner().storage();` and copy from the storage accessor. Return `T4A_NULL_POINTER` with a useful last-error message through `run_status` if output pointers are null.

Add payload copy functions:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_payload_f64(
    ptr: *const t4a_tensor,
    buf: *mut f64,
    buf_len: usize,
    out_len: *mut usize,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_copy_payload_c64(
    ptr: *const t4a_tensor,
    buf_interleaved: *mut f64,
    n_complex: usize,
    out_len: *mut usize,
) -> StatusCode { ... }
```

For `copy_payload_c64`, `out_len` is the number of complex values, matching `t4a_tensor_copy_dense_c64`.

**Step 6: Add structured constructors**

Add:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_structured_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data: *const f64,
    data_len: usize,
    payload_dims: *const usize,
    payload_rank: usize,
    payload_strides: *const isize,
    strides_len: usize,
    axis_classes: *const usize,
    axis_classes_len: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode { ... }
```

and the `c64` variant with interleaved data:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_structured_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data_interleaved: *const f64,
    n_complex: usize,
    payload_dims: *const usize,
    payload_rank: usize,
    payload_strides: *const isize,
    strides_len: usize,
    axis_classes: *const usize,
    axis_classes_len: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode { ... }
```

Validation rules:
- `axis_classes_len == rank`, else `T4A_INVALID_ARGUMENT`.
- `strides_len == payload_rank`, else `T4A_INVALID_ARGUMENT`.
- Nonzero `data_len` / `n_complex` requires a non-null data pointer.
- Nonzero `payload_rank` requires non-null dims and strides pointers.
- Nonzero `axis_classes_len` requires non-null axis classes pointer.
- Build `Storage::new_structured(...)`.
- Build `InternalTensor::from_structured_storage(indices, Arc::new(storage))`.
- Map all construction errors to `T4A_INVALID_ARGUMENT` with `capi_error`.

Add diagonal constructors:

```rust
#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_diag_f64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data: *const f64,
    data_len: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode { ... }

#[unsafe(no_mangle)]
pub extern "C" fn t4a_tensor_new_diag_c64(
    rank: usize,
    index_ptrs: *const *const t4a_index,
    data_interleaved: *const f64,
    n_complex: usize,
    out: *mut *mut t4a_tensor,
) -> StatusCode { ... }
```

These should call `InternalTensor::from_diag`.

**Step 7: Run C API tests**

Run:

```bash
cargo test --release -p tensor4all-capi --lib tensor_new_diag_f64_exposes_compact_payload
cargo test --release -p tensor4all-capi --lib tensor_new_structured_f64_roundtrips_metadata
cargo test --release -p tensor4all-capi --lib tensor
```

Expected: PASS.

**Step 8: Commit**

```bash
git add crates/tensor4all-capi/src/types.rs crates/tensor4all-capi/src/tensor.rs crates/tensor4all-capi/src/tensor/tests/mod.rs
git commit -m "feat(capi): expose structured tensor payloads"
```

## Task 7: Regenerate C Header And Public Docs

**Files:**
- Modify: `crates/tensor4all-capi/include/tensor4all_capi.h`
- Modify: `crates/tensor4all-capi/README.md`
- Modify: `crates/tensor4all-core/src/defaults/tensordynlen.rs`
- Modify: `crates/tensor4all-tensorbackend/src/storage.rs`
- Modify: `docs/api/*.md`

**Step 1: Update stale rustdoc and README text**

Update `TensorDynLen` rustdoc from "dynamic-rank dense tensor" to structured-storage wording. The key data layout note should say dense logical readback remains column-major, while compact storage may be dense, diagonal, or structured.

Update `crates/tensor4all-capi/README.md` feature list from:

```markdown
- **Tensor API**: Dense `f64` / `Complex64` construction, export, and contraction
```

to:

```markdown
- **Tensor API**: Dense and structured `f64` / `Complex64` construction, export, and contraction
```

Add a short C example for `t4a_tensor_new_diag_f64` or structured metadata readback if the README stays concise.

**Step 2: Regenerate the C header**

Run:

```bash
mkdir -p crates/tensor4all-capi/include
cbindgen crates/tensor4all-capi \
  --config crates/tensor4all-capi/cbindgen.toml \
  --output crates/tensor4all-capi/include/tensor4all_capi.h
```

Expected: header contains `t4a_storage_kind`, `t4a_tensor_new_structured_f64`, `t4a_tensor_copy_payload_f64`, and metadata functions.

Verify:

```bash
rg -n "t4a_storage_kind|t4a_tensor_new_structured|t4a_tensor_copy_payload|t4a_tensor_axis_classes" crates/tensor4all-capi/include/tensor4all_capi.h
```

**Step 3: Regenerate API docs**

Run:

```bash
cargo run -p api-dump --release -- . -o docs/api
```

Expected: docs update for `StorageKind`, storage accessors, `TensorDynLen::from_structured_storage`, and new C API symbols if the dump covers C API exports.

**Step 4: Run doc tests**

Run:

```bash
cargo test --doc --release -p tensor4all-tensorbackend
cargo test --doc --release -p tensor4all-core
cargo test --doc --release -p tensor4all-capi
```

Expected: PASS.

**Step 5: Commit**

```bash
git add crates/tensor4all-capi/include/tensor4all_capi.h crates/tensor4all-capi/README.md crates/tensor4all-core/src/defaults/tensordynlen.rs crates/tensor4all-tensorbackend/src/storage.rs docs/api
git commit -m "docs: document structured tensor storage API"
```

## Task 8: Final Verification

**Files:**
- No new source changes expected unless verification finds defects.

**Step 1: Format**

Run:

```bash
cargo fmt --all
```

Expected: no output or only formatted files.

**Step 2: Lint**

Run:

```bash
cargo clippy --workspace
```

Expected: PASS with no warnings that need code changes.

**Step 3: Run targeted release tests**

Run:

```bash
cargo nextest run --release -p tensor4all-tensorbackend
cargo nextest run --release -p tensor4all-core
cargo nextest run --release -p tensor4all-capi
```

Expected: PASS.

**Step 4: Run full release suite if time allows**

Run:

```bash
cargo nextest run --release --workspace
```

Expected: PASS. If this is too slow, record that targeted crate tests passed and full workspace was not run.

**Step 5: Build rustdoc**

Run:

```bash
cargo doc --workspace --no-deps
```

Expected: PASS.

**Step 6: Check worktree**

Run:

```bash
git status --short --branch
```

Expected: only intentional commits ahead of the base branch, no uncommitted files.

**Step 7: Do not push without user approval**

Stop after local commits and verification. Ask before pushing or opening a PR.

## Implementation Notes

- `Storage` metadata is the contract C API consumers need. Avoid exposing `StorageRepr`.
- Existing dense C readback (`t4a_tensor_copy_dense_*`) remains logical dense materialization. New payload readback (`t4a_tensor_copy_payload_*`) returns compact payload values.
- `TensorDynLen::storage()` must be cheap and must not materialize dense native tensors.
- AD correctness takes priority over compact preservation in tracked operations. If a tensor tracks gradients, route through eager/native code and rebuild storage from the resulting primal snapshot.
- When comparing dense whole results in tests, materialize once and compare with tensor subtraction plus `maxabs()`.
- If a bug is found in one C API readback helper, inspect the dense readback functions for the same null/buffer/error handling issue before continuing.
