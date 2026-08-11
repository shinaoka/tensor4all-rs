# tensor4all-tcicore Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `tensor4all-tcicore` crate by absorbing `matrixci` and extracting `CachedFunction`/`IndexSet` from `tensor4all-tensorci`, providing shared TCI infrastructure for both tensorci and treetn.

**Architecture:** Copy source files from matrixci and tensorci into a new crate, update all import paths in downstream crates, then delete the originals. The `matrixci` crate is fully absorbed. No backward compatibility needed (early development).

**Tech Stack:** Rust 2021, Cargo workspace. Use `cargo fmt --all`, `cargo clippy --workspace`, `cargo nextest run --release`.

**Important:** This plan should be executed on a worktree branched from the `cached-function-wide-key` branch (which has the wide-key CachedFunction implementation).

---

### Task 1: Create `tensor4all-tcicore` crate skeleton

**Files:**
- Create: `crates/tensor4all-tcicore/Cargo.toml`
- Create: `crates/tensor4all-tcicore/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Create Cargo.toml**

Create `crates/tensor4all-tcicore/Cargo.toml`:

```toml
[package]
name = "tensor4all-tcicore"
description = "TCI core infrastructure: matrix CI, cached function, index sets"
version.workspace = true
edition.workspace = true
authors = ["tensor4all contributors"]
license.workspace = true
repository.workspace = true

[dependencies]
num-complex.workspace = true
num-traits.workspace = true
anyhow.workspace = true
thiserror.workspace = true
rand.workspace = true
paste.workspace = true
bnum.workspace = true
tenferro-tensor-compute.workspace = true

[dev-dependencies]
approx.workspace = true
criterion.workspace = true
faer.workspace = true
rand_chacha.workspace = true
uint.workspace = true

[[bench]]
name = "rrlu_bench"
harness = false

[[bench]]
name = "cached_function"
harness = false
```

- [ ] **Step 2: Create placeholder lib.rs**

Create `crates/tensor4all-tcicore/src/lib.rs`:

```rust
#![warn(missing_docs)]
//! TCI Core infrastructure
//!
//! Shared foundation for tensor cross interpolation algorithms:
//! - Matrix CI: [`MatrixLUCI`], [`MatrixACA`], [`RrLU`]
//! - [`CachedFunction`]: Thread-safe cached function evaluation with wide key support
//! - [`IndexSet`]: Bidirectional index set for pivot management
```

- [ ] **Step 3: Add to workspace members**

In the root `Cargo.toml`, add `"crates/tensor4all-tcicore"` to the workspace members list, alongside the existing `"crates/matrixci"` entry (keep matrixci for now; we'll remove it later).

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p tensor4all-tcicore`
Expected: success (empty crate)

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-tcicore/Cargo.toml crates/tensor4all-tcicore/src/lib.rs Cargo.toml
git commit -m "feat: create tensor4all-tcicore crate skeleton"
```

---

### Task 2: Move matrixci source files into tcicore

**Files:**
- Copy from: `crates/matrixci/src/` → `crates/tensor4all-tcicore/src/`
- Copy from: `crates/matrixci/benches/` → `crates/tensor4all-tcicore/benches/`

Files to copy:
- `scalar.rs` + `scalar/tests/mod.rs`
- `util.rs` + `util/tests/mod.rs` → rename to `matrix.rs` + `matrix/tests/mod.rs`
- `error.rs`
- `traits.rs`
- `matrixlu.rs` + `matrixlu/tests/mod.rs`
- `matrixluci.rs` + `matrixluci/tests/mod.rs`
- `matrixaca.rs` + `matrixaca/tests/mod.rs`
- `benches/rrlu_bench.rs`

- [ ] **Step 1: Copy all matrixci source files**

```bash
# Copy source files
cp crates/matrixci/src/scalar.rs crates/tensor4all-tcicore/src/
cp -r crates/matrixci/src/scalar/ crates/tensor4all-tcicore/src/
cp crates/matrixci/src/error.rs crates/tensor4all-tcicore/src/
cp crates/matrixci/src/traits.rs crates/tensor4all-tcicore/src/
cp crates/matrixci/src/matrixlu.rs crates/tensor4all-tcicore/src/
cp -r crates/matrixci/src/matrixlu/ crates/tensor4all-tcicore/src/
cp crates/matrixci/src/matrixluci.rs crates/tensor4all-tcicore/src/
cp -r crates/matrixci/src/matrixluci/ crates/tensor4all-tcicore/src/
cp crates/matrixci/src/matrixaca.rs crates/tensor4all-tcicore/src/
cp -r crates/matrixci/src/matrixaca/ crates/tensor4all-tcicore/src/

# Copy util.rs as matrix.rs (rename)
cp crates/matrixci/src/util.rs crates/tensor4all-tcicore/src/matrix.rs
mkdir -p crates/tensor4all-tcicore/src/matrix/tests
cp crates/matrixci/src/util/tests/mod.rs crates/tensor4all-tcicore/src/matrix/tests/mod.rs

# Copy benchmarks
mkdir -p crates/tensor4all-tcicore/benches
cp crates/matrixci/benches/rrlu_bench.rs crates/tensor4all-tcicore/benches/
```

- [ ] **Step 2: Update internal imports in copied files**

In all files under `crates/tensor4all-tcicore/src/`, replace:
- `crate::util::` → `crate::matrix::`
- `crate::util` → `crate::matrix` (in `use` statements)
- Keep `crate::error::`, `crate::scalar::`, `crate::traits::` etc. as-is (same module names)

In `crates/tensor4all-tcicore/src/matrix.rs`:
- The module was previously named `util`, so `mod tests;` at the bottom should become `mod tests;` (unchanged, since the file is now `matrix.rs` and tests are at `matrix/tests/mod.rs`)

In `crates/tensor4all-tcicore/benches/rrlu_bench.rs`, replace:
- `use matrixci::` → `use tensor4all_tcicore::`
- `use matrixci::util::` → `use tensor4all_tcicore::matrix::`

- [ ] **Step 3: Update lib.rs with module declarations and re-exports**

Replace `crates/tensor4all-tcicore/src/lib.rs`:

```rust
#![warn(missing_docs)]
//! TCI Core infrastructure
//!
//! Shared foundation for tensor cross interpolation algorithms:
//! - Matrix CI: [`MatrixLUCI`], [`MatrixACA`], [`RrLU`]
//! - [`CachedFunction`]: Thread-safe cached function evaluation with wide key support
//! - [`IndexSet`]: Bidirectional index set for pivot management
//!
//! # Example
//!
//! ```
//! use tensor4all_tcicore::{MatrixLUCI, AbstractMatrixCI, from_vec2d};
//!
//! let m = from_vec2d(vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ]);
//!
//! let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
//! println!("Rank: {}", ci.rank());
//! ```

pub mod error;
pub mod matrix;
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod scalar;
pub mod traits;

// Re-export main types
pub use error::{MatrixCIError, Result};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_mut, RrLU, RrLUOptions};
pub use matrixluci::MatrixLUCI;
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
pub use matrix::{from_vec2d, Matrix};
```

- [ ] **Step 4: Verify tcicore compiles and tests pass**

Run: `cargo check -p tensor4all-tcicore && cargo nextest run --release -p tensor4all-tcicore`
Expected: all 37 matrixci tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-tcicore/
git commit -m "feat(tcicore): move matrixci source files into tensor4all-tcicore"
```

---

### Task 3: Move CachedFunction from tensorci into tcicore

**Files:**
- Copy from: `crates/tensor4all-tensorci/src/cached_function/` → `crates/tensor4all-tcicore/src/cached_function/`

- [ ] **Step 1: Copy cached_function module**

```bash
cp -r crates/tensor4all-tensorci/src/cached_function/ crates/tensor4all-tcicore/src/cached_function/
# This includes: mod.rs (or cached_function.rs), cache_key.rs, index_int.rs, error.rs, tests/mod.rs
```

Note: If the source is structured as `cached_function.rs` + `cached_function/` directory, copy the `.rs` file as `cached_function/mod.rs` in the destination, plus all subdirectories.

- [ ] **Step 2: Update internal imports in copied files**

The cached_function module should be self-contained. No matrixci imports expected. Verify there are no `use crate::` references that need updating. If there are references to other tensorci modules, remove them (cached_function should be independent).

- [ ] **Step 3: Add module declaration and re-exports to lib.rs**

Add to `crates/tensor4all-tcicore/src/lib.rs`:

```rust
pub mod cached_function;

// Re-export cached function types
pub use cached_function::CachedFunction;
pub use cached_function::cache_key::CacheKey;
pub use cached_function::index_int::IndexInt;
pub use cached_function::error::CacheKeyError;
```

- [ ] **Step 4: Verify tests pass**

Run: `cargo nextest run --release -p tensor4all-tcicore`
Expected: matrixci tests (37) + cached_function tests (25) all pass

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-tcicore/src/cached_function/ crates/tensor4all-tcicore/src/lib.rs
git commit -m "feat(tcicore): move CachedFunction from tensorci into tcicore"
```

---

### Task 4: Move IndexSet from tensorci into tcicore

**Files:**
- Copy from: `crates/tensor4all-tensorci/src/indexset.rs` → `crates/tensor4all-tcicore/src/indexset.rs`
- Copy tests if they exist as a subdirectory

- [ ] **Step 1: Copy indexset module**

```bash
cp crates/tensor4all-tensorci/src/indexset.rs crates/tensor4all-tcicore/src/
# If tests exist:
cp -r crates/tensor4all-tensorci/src/indexset/ crates/tensor4all-tcicore/src/ 2>/dev/null || true
```

- [ ] **Step 2: Add module declaration and re-exports to lib.rs**

Add to `crates/tensor4all-tcicore/src/lib.rs`:

```rust
pub mod indexset;

pub use indexset::{IndexSet, LocalIndex, MultiIndex};
```

- [ ] **Step 3: Verify tests pass**

Run: `cargo nextest run --release -p tensor4all-tcicore`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-tcicore/src/indexset.rs crates/tensor4all-tcicore/src/lib.rs
git commit -m "feat(tcicore): move IndexSet from tensorci into tcicore"
```

---

### Task 5: Move cached_function benchmark into tcicore

**Files:**
- Copy from: `crates/tensor4all-tensorci/benches/cached_function.rs` → `crates/tensor4all-tcicore/benches/cached_function.rs`

- [ ] **Step 1: Copy benchmark file**

```bash
cp crates/tensor4all-tensorci/benches/cached_function.rs crates/tensor4all-tcicore/benches/
```

- [ ] **Step 2: Update imports in benchmark**

In `crates/tensor4all-tcicore/benches/cached_function.rs`, the file uses `bnum` types directly (no matrixci or tensorci imports), so it should work as-is. Verify and fix any import issues.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p tensor4all-tcicore --benches`
Expected: success

- [ ] **Step 4: Commit**

```bash
git add crates/tensor4all-tcicore/benches/cached_function.rs
git commit -m "feat(tcicore): move cached_function benchmark into tcicore"
```

---

### Task 6: Update tensor4all-core to depend on tcicore

**Files:**
- Modify: `crates/tensor4all-core/Cargo.toml`
- Modify: `crates/tensor4all-core/src/scalar.rs`
- Modify: `crates/tensor4all-core/src/defaults/factorize.rs`
- Modify: `crates/tensor4all-core/src/tensor_like.rs`

- [ ] **Step 1: Update Cargo.toml**

In `crates/tensor4all-core/Cargo.toml`, replace:
- `matrixci = { path = "../matrixci" }` → `tensor4all-tcicore = { path = "../tensor4all-tcicore" }`

- [ ] **Step 2: Update imports in scalar.rs**

Replace: `pub use matrixci::Scalar as CommonScalar;`
With: `pub use tensor4all_tcicore::Scalar as CommonScalar;`

- [ ] **Step 3: Update imports in defaults/factorize.rs**

Replace all `matrixci::` with `tensor4all_tcicore::` and `matrixci::util::` with `tensor4all_tcicore::matrix::`:

- `use matrixci::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};` → `use tensor4all_tcicore::{rrlu, AbstractMatrixCI, MatrixLUCI, RrLUOptions, Scalar as MatrixScalar};`
- `matrixci::Matrix<T>` → `tensor4all_tcicore::Matrix<T>`
- `matrixci::util::zeros(` → `tensor4all_tcicore::Matrix::zeros(`
- `matrixci::util::nrows(` → `Matrix::nrows()`
- `matrixci::util::ncols(` → `Matrix::ncols()`

- [ ] **Step 4: Update error type in tensor_like.rs**

Replace: `matrixci::MatrixCIError` → `tensor4all_tcicore::MatrixCIError`

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p tensor4all-core`
Expected: success

- [ ] **Step 6: Commit**

```bash
git add crates/tensor4all-core/
git commit -m "refactor(core): migrate matrixci dependency to tensor4all-tcicore"
```

---

### Task 7: Update tensor4all-simplett to depend on tcicore

**Files:**
- Modify: `crates/tensor4all-simplett/Cargo.toml`
- Modify: All `.rs` files that import `matrixci`

Affected source files:
- `src/canonical.rs`
- `src/compression.rs`
- `src/vidal.rs`
- `src/contraction.rs`
- `src/tensortrain.rs`
- `src/error.rs`
- `src/mpo/error.rs`
- `src/mpo/factorize.rs`
- `src/tensortrain/tests/mod.rs`
- `src/contraction/tests/mod.rs`

- [ ] **Step 1: Update Cargo.toml**

Replace: `matrixci = { path = "../matrixci" }` → `tensor4all-tcicore = { path = "../tensor4all-tcicore" }`

- [ ] **Step 2: Global search-and-replace imports**

In all files listed above, replace:
- `use matrixci::util::` → `use tensor4all_tcicore::matrix::`
- `use matrixci::` → `use tensor4all_tcicore::`
- `matrixci::util::` → `tensor4all_tcicore::matrix::`
- `matrixci::Matrix` → `tensor4all_tcicore::Matrix`
- `matrixci::Scalar` → `tensor4all_tcicore::Scalar`
- `matrixci::MatrixCIError` → `tensor4all_tcicore::MatrixCIError`

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p tensor4all-simplett`
Expected: success

- [ ] **Step 4: Run tests**

Run: `cargo nextest run --release -p tensor4all-simplett`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add crates/tensor4all-simplett/
git commit -m "refactor(simplett): migrate matrixci dependency to tensor4all-tcicore"
```

---

### Task 8: Update tensor4all-tensorci to depend on tcicore and remove moved modules

**Files:**
- Modify: `crates/tensor4all-tensorci/Cargo.toml`
- Modify: `crates/tensor4all-tensorci/src/lib.rs`
- Modify: All `.rs` files that import `matrixci` or `crate::indexset` or `crate::cached_function`
- Delete: `crates/tensor4all-tensorci/src/cached_function/`
- Delete: `crates/tensor4all-tensorci/src/indexset.rs`
- Delete: `crates/tensor4all-tensorci/benches/cached_function.rs`

Affected source files:
- `src/lib.rs`
- `src/tensorci1.rs`
- `src/tensorci2.rs`
- `src/globalpivot.rs`
- `src/globalsearch.rs`
- `src/optfirstpivot.rs`
- `src/integration.rs`
- `src/error.rs`
- `src/tensorci2/tests/mod.rs`

- [ ] **Step 1: Update Cargo.toml**

Replace: `matrixci = { path = "../matrixci" }` → `tensor4all-tcicore = { path = "../tensor4all-tcicore" }`

Also remove `bnum.workspace = true` from `[dependencies]` if cached_function is no longer in this crate.

- [ ] **Step 2: Delete moved modules**

```bash
rm -rf crates/tensor4all-tensorci/src/cached_function/
rm -f crates/tensor4all-tensorci/src/cached_function.rs
rm -f crates/tensor4all-tensorci/src/indexset.rs
rm -rf crates/tensor4all-tensorci/src/indexset/
rm -f crates/tensor4all-tensorci/benches/cached_function.rs
```

- [ ] **Step 3: Update lib.rs**

Remove module declarations for deleted modules and update re-exports:

```rust
// Remove these lines:
// pub mod cached_function;
// pub mod indexset;

// Replace re-exports with tcicore re-exports:
pub use tensor4all_tcicore::{
    CachedFunction, CacheKey, CacheKeyError, IndexInt,
    IndexSet, LocalIndex, MultiIndex,
    Scalar,
};
```

- [ ] **Step 4: Update imports in all remaining source files**

Replace:
- `use matrixci::` → `use tensor4all_tcicore::`
- `use matrixci::util::` → `use tensor4all_tcicore::matrix::`
- `matrixci::util::zeros` → `tensor4all_tcicore::Matrix::zeros`
- `matrixci::rrlu` → `tensor4all_tcicore::rrlu`
- `matrixci::matrixlu::solve_lu` → `tensor4all_tcicore::matrixlu::solve_lu`
- `matrixci::MatrixCIError` → `tensor4all_tcicore::MatrixCIError`
- `use crate::indexset::` → `use tensor4all_tcicore::`  (for MultiIndex, IndexSet)
- `use crate::cached_function::` → `use tensor4all_tcicore::cached_function::`  (if any internal reference)

- [ ] **Step 5: Remove cached_function bench entry from Cargo.toml**

Remove from `crates/tensor4all-tensorci/Cargo.toml`:
```toml
[[bench]]
name = "cached_function"
harness = false
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check -p tensor4all-tensorci`
Expected: success

- [ ] **Step 7: Run tests**

Run: `cargo nextest run --release -p tensor4all-tensorci`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add crates/tensor4all-tensorci/
git commit -m "refactor(tensorci): migrate to tensor4all-tcicore, remove moved modules"
```

---

### Task 9: Delete matrixci crate

**Files:**
- Delete: `crates/matrixci/` (entire directory)
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Remove matrixci from workspace members**

In root `Cargo.toml`, remove `"crates/matrixci"` from the members list.

- [ ] **Step 2: Delete the crate**

```bash
rm -rf crates/matrixci/
```

- [ ] **Step 3: Verify workspace compiles**

Run: `cargo check --workspace`
Expected: success

- [ ] **Step 4: Run full test suite**

Run: `cargo nextest run --release --workspace`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove matrixci crate (absorbed into tensor4all-tcicore)"
```

---

### Task 10: Final validation and cleanup

**Files:** Verification only, plus any formatting fixes.

- [ ] **Step 1: Run formatting**

Run: `cargo fmt --all`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: clean

- [ ] **Step 3: Run full test suite**

Run: `cargo nextest run --release --workspace`
Expected: all pass

- [ ] **Step 4: Verify no stale matrixci references remain**

Run: `grep -r "matrixci" crates/ --include="*.rs" --include="*.toml" | grep -v target/ | grep -v tensor4all-tcicore/`
Expected: no matches (except possibly comments/docs that reference the old name, which are acceptable)

- [ ] **Step 5: Commit any cleanup**

```bash
git add -A
git commit -m "chore: final cleanup after tcicore extraction"
```
