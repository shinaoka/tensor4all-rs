# EagerTensor Migration Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate tensor4all-rs from the legacy tenferro API to the new split API (`Tensor`, `TypedTensor<T>`, `EagerTensor`) while removing deprecated AD shims, adopting explicit accumulation/reset semantics for eager gradients, and restoring `tensor4all_core::AnyScalar` as a core-owned rank-0 `IdxTensor` wrapper.

**Architecture:** Execute strictly bottom-up by actual crate dependencies: update the workspace pin and tensorbackend first, then migrate tcicore, then migrate tensor4all-core primal operations and public SVD/QR entry points, then reintroduce the renamed AD surface on `IdxTensor` plus a core-owned `AnyScalar` compatibility wrapper, then migrate simplett's typed-tensor paths and memory-layout handling, and finally sweep higher crates, tests, benches, and FFI invariants.

**Tech Stack:** Rust, tenferro, tenferro-tensor, tenferro-einsum, tenferro-algebra, cargo-nextest, mdBook

**Primary references:**
- `docs/design/symbolic-shape-and-tenferro-migration.md`
- `docs/design/review-eagertensor-migration-v2.md`

**Non-goals:**
- Preserve deprecated AD method names on `AnyScalar` or `IdxTensor`.
- Keep removed tenferro re-exports such as `print_and_reset_contract_profile` alive via compatibility shims.
- Perform blind `from_slice -> from_vec` renames without auditing memory order at each call site.
- Reintroduce seeded backward or implicit gradient clearing semantics that are absent from the target tenferro API.

**Execution rules:**
- Run `cargo fmt --all` before every commit.
- Run at least `cargo check -p <crate> --all-targets` for the crate touched in each task.
- Do not relax numerical tolerances without explicit user approval.
- If a required tenferro API is missing, stop, land that API upstream, repin, and then resume. Do not reach into private fields or add local hacks around missing upstream behavior.

---

## Task 1: Re-pin tenferro and migrate tensorbackend foundations

**Files:**
- Modify: `Cargo.toml`
- Modify: `Cargo.lock`
- Modify: `crates/tensor4all-tensorbackend/Cargo.toml`
- Create: `crates/tensor4all-tensorbackend/src/context.rs`
- Modify: `crates/tensor4all-tensorbackend/src/lib.rs`
- Modify: `crates/tensor4all-tensorbackend/src/any_scalar.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tensor_element.rs`
- Modify: `crates/tensor4all-tensorbackend/src/backend.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge.rs`
- Modify: `crates/tensor4all-tensorbackend/src/tenferro_bridge/tests/mod.rs`
- Modify: `crates/tensor4all-tensorbackend/tests/bench_einsum_native.rs`

- [ ] **Step 1: Update the workspace pin and direct dependencies**

Update all tenferro workspace dependencies in `Cargo.toml` from the old revision to a tenferro-rs revision at or after the 2026-04-13 eager-AD merge (the revision that includes `tracks_grad`, `clear_grad`, `clear_grads`, Send+Sync eager tensors, and accumulating `backward()`). In `crates/tensor4all-tensorbackend/Cargo.toml`, remove obsolete direct dependencies (`tenferro-prims`, `tenferro-linalg`) where no longer needed and add `tenferro.workspace = true`.

- [ ] **Step 2: Capture the expected tensorbackend breakage on the new pin**

Run:

```bash
cargo check -p tensor4all-tensorbackend --all-targets
```

Expected: FAIL with unresolved imports or method errors around `ScalarType`, `tenferro_linalg`, `tenferro_prims`, old tensor accessors (`dims`, `scalar_type`, `try_get`, `try_to_vec`), old runtime helpers, and AnyScalar AD methods.

- [ ] **Step 3: Resolve the execution-context model and implement tensorbackend migration**

Make the following changes in one coherent pass:

- Create `src/context.rs` and expose thread-local execution helpers.
- Verify whether the pinned tenferro API can make `TypedTensor` ops and `EagerTensor` ops share the exact same `CpuBackend` instance.
- If exact single-instance sharing is supported, implement it and document that invariant.
- If exact single-instance sharing is not supported, implement paired thread-local execution objects (`with_default_backend`, `default_eager_ctx`) and update the doc comments to avoid claiming a single shared backend instance.
- Remove AD behavior from backend `AnyScalar` entirely. Delete deprecated methods such as `requires_grad`, `set_requires_grad`, `zero_grad`, `backward_with_seed`, `grad`, and `detach`. The public `tensor4all_core::AnyScalar` compatibility layer is reintroduced later in Task 4 as a core-owned rank-0 `IdxTensor` wrapper.
- Update `tensor_element.rs` for `DType`, `shape()`, `as_slice()`, and explicit column-major `TypedTensor::from_vec(...)` construction.
- Update `backend.rs` to use `TensorScalar` plus `TypedTensor::{svd, qr}` through `with_default_backend(...)`.
- Simplify `tenferro_bridge.rs` to the conversion and profiling helpers still needed by downstream crates, and update it to `Tensor::new(...)`, `dtype()`, and `shape()`.
- Update tests and benches to stop depending on removed runtime/profile APIs from old tenferro.

- [ ] **Step 4: Verify tensorbackend**

Run:

```bash
cargo fmt --all
cargo check -p tensor4all-tensorbackend --all-targets
cargo nextest run --release -p tensor4all-tensorbackend
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/tensor4all-tensorbackend/
git commit -m "refactor: migrate tensorbackend to new tenferro foundations"
```

---

## Task 2: Migrate tcicore to typed_eager_einsum and audited layout conversions

**Files:**
- Modify: `crates/tensor4all-tcicore/Cargo.toml`
- Modify: `crates/tensor4all-tcicore/src/matrix.rs`

- [ ] **Step 1: Remove the old compute dependency and confirm the failure mode**

Update `crates/tensor4all-tcicore/Cargo.toml` to remove `tenferro-tensor-compute` and add any now-required tenferro crates.

Run:

```bash
cargo check -p tensor4all-tcicore --all-targets
```

Expected: FAIL around `tenferro_tensor_compute::einsum`, `MemoryOrder`, and old tensor materialization helpers.

- [ ] **Step 2: Replace the einsum path**

In `src/matrix.rs`, replace the old compute call with `tenferro_einsum::typed_eager_einsum(...)` using `tensor4all_tensorbackend::with_default_backend(...)`.

- [ ] **Step 3: Fix row-major assumptions explicitly**

Audit every tensor construction and extraction in `src/matrix.rs`.

- When source data is row-major, relayout it intentionally before `TypedTensor::from_vec(...)`.
- When extracting dense results, materialize once and convert layout once.
- Do not rely on `MemoryOrder`, `contiguous(...)`, or `buffer()` APIs that do not exist on the new tenferro path.

- [ ] **Step 4: Verify tcicore**

Run:

```bash
cargo fmt --all
cargo check -p tensor4all-tcicore --all-targets
cargo nextest run --release -p tensor4all-tcicore
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-tcicore/
git commit -m "refactor: migrate tcicore einsum to new tenferro API"
```

---

## Task 3: Migrate tensor4all-core primal tensor operations and public factorization entry points

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/idx_tensor.rs`
- Modify: `crates/tensor4all-core/src/defaults/contract.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-core/src/defaults/svd.rs`
- Modify: `crates/tensor4all-core/src/defaults/qr.rs`
- Modify: `crates/tensor4all-core/tests/tensor_native_ad.rs`

- [ ] **Step 1: Capture the current core failures**

Run:

```bash
cargo check -p tensor4all-core --all-targets
```

Expected: FAIL around old `ScalarType` dispatch, old `IdxTensor` internals (`Arc<NativeTensor>`), and the removed pre-migration AD methods.

- [ ] **Step 2: Change `IdxTensor` to the new storage model**

In `idx_tensor.rs`:

- Replace `native: Arc<NativeTensor>` with `inner: EagerTensor<CpuBackend>`.
- Add `axis_classes: Vec<usize>` and initialize it in every constructor and result-building path.
- Keep diagonal bookkeeping in tensor4all-core rather than expecting structural tensor support from new tenferro.
- Update all primal data accessors and conversions to read from `self.inner.data()`.

- [ ] **Step 3: Remove deprecated AD surface during the primal migration**

Delete the obsolete tensor-level AD methods that cannot be supported on the new API surface:

- `requires_grad(&self) -> bool`
- `set_requires_grad(...)`
- `zero_grad(...)`
- `backward_with_seed(...)`

Because AD-method backward compatibility is not required, do not leave compatibility wrappers behind. Update or temporarily replace the old test coverage in `tests/tensor_native_ad.rs` so that `--all-targets` compiles cleanly before the new AD surface and core-owned `AnyScalar` wrapper are added in Task 4.

- [ ] **Step 4: Migrate primal ops and public SVD/QR entry points**

Make the core operations use the new primitives directly:

- Contraction and outer-product paths should use `tenferro::eager_einsum::eager_einsum_ad(...)`.
- Factorization should reshape through `EagerTensor` and call `svd()` / `qr()` directly.
- `defaults/svd.rs` and `defaults/qr.rs` must stop dispatching on `tenferro::ScalarType` and instead use the new dtype path (`DType` or internal typed helpers).

- [ ] **Step 5: Verify tensor4all-core primal behavior**

Run:

```bash
cargo fmt --all
cargo check -p tensor4all-core --all-targets
cargo nextest run --release -p tensor4all-core
```

Expected: PASS, with AD tests either removed or rewritten to target the new API names only.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-core/
git commit -m "refactor: migrate core primal tensor ops to EagerTensor"
```

---

## Task 4: Add the new IdxTensor AD surface and restore core AnyScalar compatibility

**Files:**
- Modify: `crates/tensor4all-core/src/defaults/idx_tensor.rs`
- Create: `crates/tensor4all-core/src/any_scalar.rs`
- Modify: `crates/tensor4all-core/src/lib.rs`
- Modify: `crates/tensor4all-core/tests/tensor_native_ad.rs`
- Modify: `crates/tensor4all-core/tests/tensor_any_scalar.rs`
- Create: `crates/tensor4all-core/tests/ad_integration.rs`

- [ ] **Step 1: Confirm the required tenferro AD API exists**

Before changing tensor4all-core, confirm on the pinned tenferro revision that the following are available as public APIs:

- create grad-tracking tensors
- run scalar-loss `backward()`
- fetch accumulated `grad()`
- clear a single tensor's accumulated gradient via `clear_grad()`
- detach from the graph
- query whether a tensor tracks gradients via `tracks_grad()`

If the pinned revision does not include this public API surface, stop here and repin before changing tensor4all-core. Do not inspect private fields from tensor4all-rs.

- [ ] **Step 2: Write the failing tests for the new API and the new AnyScalar contract**

Replace old AD test expectations with the new surface:

- `enable_grad(self) -> Self`
- `tracks_grad(&self) -> bool`
- `grad(&self) -> Result<Option<IdxTensor>>`
- `clear_grad(&self) -> Result<()>`
- `backward(&self) -> Result<()>`
- `detach(&self) -> Self`

Also add explicit failing tests for:

- repeated `backward()` accumulation
- `clear_grad()` resetting the accumulated gradient
- `tracks_grad()` on tracked vs detached tensors
- core `AnyScalar` rank-0 conversion and forwarding to `IdxTensor`
- borrow-based named-value arithmetic such as `&x * &y + &x`
- function APIs that take `&AnyScalar`
- shared-handle `Clone` behavior for tracked scalars

Run:

```bash
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo nextest run --release -p tensor4all-core --test tensor_any_scalar
cargo nextest run --release -p tensor4all-core --test ad_integration
```

Expected: FAIL before implementation.

- [ ] **Step 3: Implement the new AD surface and core-owned AnyScalar without reviving removed APIs**

In `idx_tensor.rs` and the new `any_scalar.rs`:

- Implement `enable_grad`, `tracks_grad`, `grad`, `clear_grad`, `backward`, and `detach`.
- Do not reintroduce `zero_grad` or `backward_with_seed`.
- Preserve `axis_classes` and index metadata across gradient and detach paths.
- Ensure scalar-loss backward is the only supported public entry point.
- Preserve tenferro's accumulation semantics instead of clearing gradients implicitly inside the wrapper.
- Replace the `tensor4all-core` re-export of backend `AnyScalar` with a core-owned wrapper around rank-0 `IdxTensor`.
- Preserve the public type name `tensor4all_core::AnyScalar` plus primal helpers such as `new_real`, `new_complex`, `from_value`, `real`, `imag`, `abs`, `is_real`, `is_complex`, `is_zero`, and arithmetic.
- Implement the documented borrow-based operator contract for named values. Function APIs should take `&AnyScalar`; the plan does not require owned-variable `x * y + x`.
- Make `Clone` a shared-handle clone of the same tracked scalar. Do not add a value-returning `close()` API.

- [ ] **Step 4: Verify core AD**

Run:

```bash
cargo fmt --all
cargo check -p tensor4all-core --all-targets
cargo nextest run --release -p tensor4all-core --test tensor_native_ad
cargo nextest run --release -p tensor4all-core --test tensor_any_scalar
cargo nextest run --release -p tensor4all-core --test ad_integration
cargo nextest run --release -p tensor4all-core
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-core/
git commit -m "feat: add EagerTensor AD surface for IdxTensor and AnyScalar"
```

---

## Task 5: Migrate simplett to TypedTensor plus explicit layout helpers

**Files:**
- Modify: `crates/tensor4all-simplett/Cargo.toml`
- Modify: `crates/tensor4all-simplett/src/einsum_helper.rs`
- Modify: `crates/tensor4all-simplett/src/compression.rs`
- Modify: `crates/tensor4all-simplett/src/contraction.rs`
- Modify: `crates/tensor4all-simplett/src/tensor.rs`
- Modify: `crates/tensor4all-simplett/src/types.rs`
- Modify: `crates/tensor4all-simplett/src/vidal.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/environment.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/factorize.rs`
- Modify: `crates/tensor4all-simplett/src/mpo/types.rs`
- Modify: `crates/tensor4all-simplett/src/contraction/tests/mod.rs`

- [ ] **Step 1: Confirm the simplett failure set**

Run:

```bash
cargo check -p tensor4all-simplett --all-targets
```

Expected: FAIL around `tenferro-linalg`, `tenferro-tensor-compute`, `from_row_major_slice`, `MemoryOrder`, `contiguous(...)`, `buffer()`, `is_unique()`, and old trait bounds such as `KeepCountScalar`.

- [ ] **Step 2: Update dependencies and common helpers**

In `Cargo.toml`, remove obsolete tenferro crates and add `tenferro-einsum` if needed. Replace the old `einsum_helper.rs` implementation with a `typed_eager_einsum(...)` path via `with_default_backend(...)`.

- [ ] **Step 3: Introduce explicit layout conversion helpers**

Create or reuse local helpers so layout handling is visible and audited:

- Convert row-major source slices to column-major buffers before `TypedTensor::from_vec(...)`.
- When returning row-major dense data to existing simplett code paths, convert once in a named helper instead of repeatedly materializing via removed tenferro APIs.
- Prefer algorithm changes that avoid row-major round-trips entirely when practical, especially in `contraction.rs`.

- [ ] **Step 4: Update all known old-API call sites**

Migrate the files identified by the review and grep sweep:

- `src/tensor.rs`
- `src/types.rs`
- `src/contraction.rs`
- `src/compression.rs`
- `src/vidal.rs`
- `src/mpo/factorize.rs`
- `src/mpo/environment.rs`
- `src/mpo/types.rs`
- `src/contraction/tests/mod.rs`

Do not stop after `einsum_helper.rs` and `compression.rs`; the old plan was incomplete here.

- [ ] **Step 5: Verify simplett**

Run:

```bash
cargo fmt --all
cargo check -p tensor4all-simplett --all-targets
cargo nextest run --release -p tensor4all-simplett
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-simplett/
git commit -m "refactor: migrate simplett to TypedTensor and explicit layouts"
```

---

## Task 6: Sweep higher crates, benches, and FFI-facing invariants

**Files:**
- Modify: `crates/tensor4all-treetn/tests/ad_treetn.rs`
- Modify: any workspace files found by the sweep below
- Create: `crates/tensor4all-core/tests/send_sync.rs`

- [ ] **Step 1: Run a workspace-wide grep sweep for legacy APIs**

Run:

```bash
rg -n "ScalarType|MemoryOrder|contiguous\\(|buffer\\(|from_row_major_slice|from_slice\\(|set_default_runtime|RuntimeContext::Cpu|print_and_reset_contract_profile|requires_grad\\(|set_requires_grad|zero_grad\\(|backward_with_seed" crates
```

Expected: only intentional compatibility uses remain. Every production hit must either be migrated in this task or explained in a code comment with a strong reason.

- [ ] **Step 2: Fix remaining direct uses in higher crates and tests**

At minimum, update:

- `crates/tensor4all-treetn/tests/ad_treetn.rs`
- any remaining bench/test files that still rely on removed tenferro runtime or profiling APIs

If the grep reveals additional production files outside the earlier tasks, migrate them now instead of adding TODOs.

- [ ] **Step 3: Add Send + Sync compile-time assertions**

Create `crates/tensor4all-core/tests/send_sync.rs` with static assertions for:

- `tenferro::EagerTensor<tenferro::CpuBackend>`
- `tensor4all_core::defaults::IdxTensor`

If the FFI handle types in `tensor4all-capi` expose a similarly testable public type, add an assertion there too.

- [ ] **Step 4: Verify workspace compilation and tests**

Run:

```bash
cargo fmt --all
cargo check --workspace --all-targets
cargo nextest run --release --workspace
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/
git commit -m "test: finish migration sweep across higher crates and invariants"
```

---

## Task 7: Final verification, rustdoc, and mdBook checks

**Files:**
- Modify: any files required to fix verification failures

- [ ] **Step 1: Run formatting and linting**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets
```

Expected: PASS.

- [ ] **Step 2: Run the full compile and test matrix**

```bash
cargo check --workspace --all-targets
cargo nextest run --release --workspace
```

Expected: PASS.

- [ ] **Step 3: Run rustdoc and mdBook verification**

```bash
cargo test --doc --release --workspace
mdbook test docs/book
cargo doc --workspace --no-deps
```

Expected: PASS.

- [ ] **Step 4: Commit the final cleanup**

```bash
git add -A
git commit -m "chore: finalize EagerTensor migration"
```

---

## Dependency Graph

```text
Task 1: tensorbackend foundations + workspace pin
  |
  v
Task 2: tcicore
  |
  v
Task 3: core primal migration
  |
  v
Task 4: core AD surface + core AnyScalar compatibility
  |
  v
Task 5: simplett
  |
  v
Task 6: higher crates + benches + FFI invariants
  |
  v
Task 7: full verification
```

## Stop Conditions

Stop and update the plan before proceeding if any of the following happen:

- The pinned tenferro revision is older than the eager-AD merge required by this plan or otherwise lacks `tracks_grad()`, `clear_grad()`, `clear_grads()`, or accumulating `backward()`.
- The context model cannot honestly support the design's paired execution-object wording; in that case, fix the design text before continuing.
- A remaining row-major call site cannot be migrated safely without changing algorithm structure; in that case, document the exact layout invariant and update the design before continuing.

## Parallelism Guidance

Do not reuse the old parallel split. `tensor4all-simplett` depends on both `tensor4all-core` and `tensor4all-tcicore`, so Tasks 2 through 5 are sequential. The only safe parallel work is small verification or grep-based cleanup that does not edit overlapping files after Task 5 is green.
