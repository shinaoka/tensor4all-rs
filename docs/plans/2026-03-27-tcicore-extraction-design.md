# tensor4all-tcicore: TCI Core Extraction

## Problem

`matrixci`, `CachedFunction`, and `IndexSet` are foundational to both TCI (TensorCI1/TCI2) and the upcoming treeTCI in `tensor4all-treetn`. Currently:

- `matrixci` is a standalone crate
- `CachedFunction` and `IndexSet` live inside `tensor4all-tensorci`
- `tensor4all-treetn` cannot use any of these without depending on `tensor4all-tensorci`

## Design

### New Crate: `tensor4all-tcicore`

A single crate combining all TCI-shared infrastructure. The `matrixci` crate is absorbed (deleted).

### Module Structure

```
crates/tensor4all-tcicore/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── error.rs              # MatrixCIError (name retained for clarity)
│   ├── scalar.rs             # Scalar trait, scalar_tests! macro
│   ├── matrix.rs             # Matrix<T>, mat_mul, zeros, from_vec2d, etc.
│   ├── traits.rs             # AbstractMatrixCI trait
│   ├── matrixlu.rs           # RrLU, rrlu, rrlu_mut, RrLUOptions
│   ├── matrixluci.rs         # MatrixLUCI
│   ├── matrixaca.rs          # MatrixACA
│   ├── cached_function/      # CachedFunction (wide key, thread-safe, batch eval)
│   │   ├── mod.rs
│   │   ├── cache_key.rs      # CacheKey trait
│   │   ├── index_int.rs      # IndexInt trait
│   │   ├── error.rs          # CacheKeyError
│   │   └── tests/mod.rs
│   ├── indexset.rs            # IndexSet, MultiIndex, LocalIndex
│   └── (module)/tests/mod.rs # Per-module tests (existing pattern)
├── benches/
│   ├── rrlu_bench.rs         # From matrixci
│   └── cached_function.rs   # From tensorci
```

### Source Migration

| Source | Destination |
|--------|-------------|
| `crates/matrixci/src/*` | `crates/tensor4all-tcicore/src/` (all modules) |
| `crates/matrixci/benches/` | `crates/tensor4all-tcicore/benches/` |
| `crates/tensor4all-tensorci/src/cached_function/` | `crates/tensor4all-tcicore/src/cached_function/` |
| `crates/tensor4all-tensorci/src/indexset.rs` | `crates/tensor4all-tcicore/src/indexset.rs` |
| `crates/tensor4all-tensorci/benches/cached_function.rs` | `crates/tensor4all-tcicore/benches/` |

### Deletions

- `crates/matrixci/` — entire directory (absorbed into tcicore)
- `crates/tensor4all-tensorci/src/cached_function/` — moved to tcicore
- `crates/tensor4all-tensorci/src/indexset.rs` — moved to tcicore
- `crates/tensor4all-tensorci/benches/cached_function.rs` — moved to tcicore

### Dependencies

`tensor4all-tcicore` Cargo.toml dependencies (union of matrixci + cached_function needs):

```toml
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
```

### Downstream Crate Changes

No backward compatibility is maintained (early development stage).

| Crate | Change |
|-------|--------|
| `tensor4all-core` | `matrixci` → `tensor4all-tcicore` in Cargo.toml and imports |
| `tensor4all-simplett` | `matrixci` → `tensor4all-tcicore` in Cargo.toml and imports |
| `tensor4all-tensorci` | `matrixci` → `tensor4all-tcicore`, remove moved modules, update imports |
| `tensor4all-capi` | `matrixci` → `tensor4all-tcicore` in imports |
| `tensor4all-quanticstci` | Update if it references matrixci |
| Workspace `Cargo.toml` | Remove `matrixci` member, add `tensor4all-tcicore` |

### Error Types

- `MatrixCIError` — retained as-is (name is semantically clear for matrix CI errors)
- `CacheKeyError` — retained in `cached_function::error` submodule

### Public API

`tensor4all-tcicore` exports:

```rust
// Matrix CI
pub use error::{MatrixCIError, Result};
pub use scalar::Scalar;
pub use matrix::{Matrix, from_vec2d};
pub use traits::AbstractMatrixCI;
pub use matrixlu::{rrlu, rrlu_mut, RrLU, RrLUOptions};
pub use matrixluci::MatrixLUCI;
pub use matrixaca::MatrixACA;

// Cached function
pub use cached_function::{CachedFunction, CacheKey, IndexInt, CacheKeyError};

// Index set
pub use indexset::{IndexSet, MultiIndex, LocalIndex};
```

### Tests

All existing tests move with their source files:
- matrixci tests: 37 tests (matrixlu, matrixluci, matrixaca, scalar, util)
- cached_function tests: 25 tests (wide key, thread safety, batch, IndexInt)
- indexset tests: moved from tensorci

### Out of Scope

- Adding `tensor4all-tcicore` as a dependency of `tensor4all-treetn` (future task)
- `eval_batch_tensor` implementation (separate task)
