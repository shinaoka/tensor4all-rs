# tensor4all-rs

A Rust implementation of tensor networks, inspired by ITensors.jl and QSpace v4.
The API is designed to be largely compatible with ITensors.jl, with the goal of enabling easy conversion between the two libraries.

This library is experimental and is planned to focus primarily on QTT (Quantics Tensor Train) and TCI (Tensor Cross Interpolation) in the near future. Abelian and non-Abelian symmetries are not in the initial scope, but the design is extensible to support them in the future.

## Overview

tensor4all-rs provides a type-safe, efficient implementation of tensor networks with support for quantum number symmetries. The design is inspired by both ITensors.jl (Julia) and QSpace v4 (MATLAB/C++), which represent the same mathematical concept: block-sparse tensors organized by quantum numbers.

## Key Features

- **Type-safe Index system**: Generic `Index<Id, Symm, Tags = DefaultTagSet>` type supporting both runtime and compile-time identities
- **Tag support**: Index tags with configurable capacity via `Tags` type parameter (default: `DefaultTagSet` with max 4 tags, each max 16 characters)
- **Quantum number symmetries**: Support for Abelian (U(1), Z_n) and non-Abelian (SU(2), SU(N)) symmetries (planned)
- **Thread-safe ID generation**: UInt128 random IDs using thread-local RNG for extremely low collision probability
- **Flexible tensor types**: Both dynamic-rank and static-rank tensor variants
- **Copy-on-write storage**: Efficient memory management for tensor networks
- **Multiple storage backends**: DenseF64, DenseC64, DiagF64, and DiagC64 storage types
- **Linear algebra operations**: SVD and QR decompositions with configurable truncation tolerance, supporting both FAER and LAPACK backends
- **Modular architecture**: Separated into `tensor4all-index` (core index/tag utilities), `tensor4all-tensor` (tensor and storage implementations), and `tensor4all-linalg` (linear algebra operations: SVD, QR)

## Design Philosophy

### Design Principles

tensor4all-rs is designed with the following principles in mind:

- **Modular architecture for fast development cycles**: The library is split into independent crates (`tensor4all-index`, `tensor4all-tensor`, and `tensor4all-linalg`) to enable rapid AI-assisted code generation and testing. Each module can be developed, tested, and compiled independently, minimizing iteration time during development.

- **Compile-time error detection**: The design leverages Rust's type system to catch errors at compile time rather than runtime:
  - Generic type parameters (`Index<Id, Symm, Tags>`) enable compile-time validation of index compatibility
  - Static-rank tensors (`TensorStaticLen<N, Id, T, Symm>`) catch rank mismatches at compile time
  - Type-safe storage variants prevent incorrect storage type usage
  - Compile-time identity types (ZST markers) enable static analysis of index relationships

- **Extensibility without breaking changes**: The generic design allows adding new features (e.g., quantum number symmetries) without breaking existing code, through trait implementations and type parameters.

- **Performance through zero-cost abstractions**: Rust's zero-cost abstractions ensure that the type-safe, generic design does not incur runtime overhead compared to more direct implementations.

### Comparison with Existing Libraries

| Concept | QSpace v4 | ITensors.jl | tensor4all-rs |
|---------|-----------|-------------|---------------|
| **Tensor with QNs** | `QSpace` | `ITensor` | `TensorDynLen<Id, T, Symm>` / `TensorStaticLen<N, Id, T, Symm>` |
| **Index** | Quantum number labels in `QIDX` | `Index{QNBlocks}` | `Index<Id, Symm, Tags = DefaultTagSet>` |
| **Storage** | `DATA` (array of blocks) | `NDTensors.BlockSparse` | `Storage` enum (DenseF64, DenseC64, DiagF64, DiagC64) |
| **Language** | MATLAB/C++ | Julia | Rust |

### Index Design

The `Index` type is parameterized by identity type `Id`, symmetry type `Symm`, and tag type `Tags`:

```rust
pub struct Index<Id, Symm = NoSymmSpace, Tags = DefaultTagSet> {
    pub id: Id,
    pub symm: Symm,
    pub tags: Tags,
}
```

**Identity Types**:
- `DynId` (u128): Runtime identity with thread-local random ID generation
- ZST marker types: Compile-time-known identity for static analysis

**Symmetry Types**:
- `NoSymmSpace`: No symmetry (corresponds to `Index{Int}` in ITensors.jl)
- `QNSpace` (planned): Quantum number spaces (corresponds to `Index{QNBlocks}`)

**Tags**:
- Configurable via `Tags` type parameter (default: `DefaultTagSet`)
- `DefaultTagSet = TagSet<4, 16>` (max 4 tags, each max 16 characters)
- Tags are stored in `TagSet` using `SmallString` for efficient storage

### ID Generation

tensor4all-rs uses **UInt128 with thread-local RNG** for ID generation:

- Each thread has its own RNG instance, seeded from OS entropy
- Collision probability for 10^5 indices: ~1.5 × 10^-29 (effectively zero)
- Thread-safe without global synchronization
- Similar to ITensors.jl's task-local RNG but with UInt128 for better collision resistance

**Collision Probability** (Birthday Paradox):

For `n` randomly generated UInt128 IDs:
```
P(collision) ≈ 1 - exp(-n² / (2 × 2^128))
```

Examples:
- **n = 10^5 (100,000 indices)**: P(collision) ≈ **1.5 × 10^-29** (negligible)
- **n = 10^6 (1 million indices)**: P(collision) ≈ **1.5 × 10^-27** (negligible)
- **n = 10^9 (1 billion indices)**: P(collision) ≈ **1.5 × 10^-21** (negligible)

### Tensor Types

Two tensor variants for different use cases:

1. **Dynamic rank**: `TensorDynLen<Id, T, Symm = NoSymmSpace>`
   - Rank determined at runtime
   - Uses `Vec<Index>` and `Vec<usize>` for indices and dimensions

2. **Static rank**: `TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace>`
   - Rank determined at compile time
   - Uses arrays `[Index; N]` and `[usize; N]` for indices and dimensions

### Storage

Tensor data is shared via `Arc<Storage>` with copy-on-write (COW) semantics:
- If uniquely owned, mutate in place
- If shared, clone then mutate

**Storage Types**:
- `DenseF64`: Dense storage for `f64` elements (wraps `DenseStorageF64`)
- `DenseC64`: Dense storage for `Complex64` elements (wraps `DenseStorageC64`)
- `DiagF64`: Diagonal storage for `f64` elements (wraps `DiagStorageF64`) - stores only diagonal elements
- `DiagC64`: Diagonal storage for `Complex64` elements (wraps `DiagStorageC64`) - stores only diagonal elements

**Storage Architecture**:
- Storage types are implemented as newtype structs (`DenseStorageF64`, `DenseStorageC64`, `DiagStorageF64`, `DiagStorageC64`)
- The `Storage` enum wraps these newtypes, providing a unified interface
- Heavy operations (permutation, contraction, conversion) are implemented as methods on the storage newtypes
- This design keeps `match` blocks short and organizes logic by storage type

### Element Types

- **Static element type**: Use concrete types like `f64` or `Complex64`
- **Dynamic element type**: Use `AnyScalar` enum for runtime type dispatch

## Type Correspondence

| ITensors.jl | tensor4all-rs |
|-------------|--------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `Index<Id, QNSpace>` (future) |
| `Index(id, dim, ...)` | `Index::new_with_size(id, dim)` |
| `Index(dim)` | `Index::new_dyn(dim)` |
| `ITensor` | `TensorDynLen<Id, T, Symm>` or `TensorStaticLen<N, Id, T, Symm>` |
| `NDTensors.Dense` | `Storage::DenseF64` or `Storage::DenseC64` |
| `NDTensors.Diag` | `Storage::DiagF64` or `Storage::DiagC64` |

## Truncation Tolerance Comparison

tensor4all-rs and ITensors.jl use different conventions for truncation tolerance:

| Library | Parameter | Semantics | Guarantee |
|---------|-----------|----------|-----------|
| **tensor4all-rs** | `rtol` | Relative Frobenius error tolerance | \|A - A_approx\|_F / \|A\|_F ≤ rtol |
| **ITensors.jl** | `cutoff` | Squared relative error (discarded weight ratio) | Σ_{discarded} σ²_i / Σ_i σ²_i ≤ cutoff |

**Conversion**: ITensors.jl's `cutoff` corresponds to `rtol²` in tensor4all-rs:
- To achieve the same truncation behavior, use `rtol = sqrt(cutoff)` in tensor4all-rs
- For example, ITensors `cutoff=1e-20` (for ~10 digit accuracy) corresponds to `rtol=1e-10` in tensor4all-rs

**Default values**:
- tensor4all-rs: `rtol = 1e-12` (near machine precision)
- ITensors.jl: `cutoff = 1e-16` (default, corresponds to `rtol ≈ 1e-8`)

## Project Structure

tensor4all-rs is organized as a Cargo workspace with three main crates:

- **`tensor4all-index`**: Core index, tag, and small string utilities
  - `Index<Id, Symm, Tags>`: Generic index type
  - `TagSet`: Index tag management
  - `SmallString`: Efficient small string storage
  - `common_inds`: Find common indices between tensors

- **`tensor4all-tensor`**: Tensor and storage implementations
  - `TensorDynLen<Id, T, Symm>`: Dynamic-rank tensors
  - `TensorStaticLen<N, Id, T, Symm>`: Static-rank tensors
  - `Storage`: Storage backend enum
  - Storage newtypes: `DenseStorageF64`, `DenseStorageC64`, `DiagStorageF64`, `DiagStorageC64`

- **`tensor4all-linalg`**: Linear algebra operations for tensor networks
  - `svd`: Singular Value Decomposition with truncation control
  - `qr`: QR decomposition with truncation control
  - Backend support: FAER (default) and LAPACK (optional)
  - Configurable relative tolerance (`rtol`) for truncation

## Usage Example

### Basic Tensor Creation

```rust
use tensor4all_index::index::{DefaultIndex as Index, DynId};
use tensor4all_tensor::{Storage, TensorDynLen};
use std::sync::Arc;

// Create indices
let i = Index::new_dyn(2);  // Index with dimension 2, auto-generated ID
let j = Index::new_dyn(3);  // Index with dimension 3, auto-generated ID

// Create dense storage
let storage = Arc::new(Storage::new_dense_f64(6));  // Capacity for 2×3=6 elements

// Create tensor
let indices = vec![i, j];
let dims = vec![2, 3];
let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(indices, dims, storage);
```

### Diagonal Tensor Creation

```rust
use tensor4all_index::index::{DefaultIndex as Index, DynId};
use tensor4all_tensor::diag_tensor_dyn_len;

// Create a 3×3 diagonal tensor
let i = Index::new_dyn(3);
let j = Index::new_dyn(3);
let diag_data = vec![1.0, 2.0, 3.0];

let tensor = diag_tensor_dyn_len(vec![i, j], diag_data);
```

### Tensor Contraction

```rust
use tensor4all_index::index::{DefaultIndex as Index, DynId};
use tensor4all_tensor::{Storage, TensorDynLen};
use tensor4all_tensor::storage::DenseStorageF64;
use std::sync::Arc;

let i = Index::new_dyn(2);
let j = Index::new_dyn(3);
let k = Index::new_dyn(4);

// Create tensor A[i, j]
let storage_a = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 6]));
let tensor_a = TensorDynLen::new(
    vec![i.clone(), j.clone()],
    vec![2, 3],
    Arc::new(storage_a)
);

// Create tensor B[j, k]
let storage_b = Storage::DenseF64(DenseStorageF64::from_vec(vec![1.0; 12]));
let tensor_b = TensorDynLen::new(
    vec![j.clone(), k.clone()],
    vec![3, 4],
    Arc::new(storage_b)
);

// Contract along j: result is C[i, k]
let result = tensor_a.contract(&tensor_b);
```

## ITensors.jl ID Generation Algorithm

ITensors.jl uses **random ID generation** for Index objects:

```julia
const IDType = UInt64

const _INDEX_ID_RNG_KEY = :ITensors_index_id_rng_bLeTZeEsme4bG3vD
index_id_rng() = get!(task_local_storage(), _INDEX_ID_RNG_KEY, Xoshiro())::Xoshiro

function Index(dim::Number; tags="", plev=0, dir=Neither)
  return Index(rand(index_id_rng(), IDType), dim, dir, tags, plev)
end
```

**Key characteristics**:
- **Random generation**: Uses `rand(index_id_rng(), UInt64)` to generate random 64-bit IDs
- **Task-local RNG**: Each Julia task has its own `Xoshiro` random number generator stored in task-local storage
- **Collision probability**: Extremely low due to the large ID space (2^64 ≈ 1.84 × 10^19)
- **Reproducibility**: Can be controlled by seeding the task-local RNG if needed

**Comparison with tensor4all-rs**:
- **ITensors.jl**: Random IDs (UInt64 from Xoshiro RNG, task-local)
- **tensor4all-rs**: Random IDs (UInt128 from thread-local RNG)

The UInt128 approach provides significantly better collision resistance than UInt64 while maintaining the benefits of random ID generation (better hash distribution, no ID reuse issues).

## Future Extensions

- **Quantum Number Space**: Support for quantum number symmetries
- **Arrow/Direction**: Index direction encoding for non-Abelian symmetries
- **Non-Abelian Support**: Clebsch-Gordan coefficients for non-Abelian symmetries

## TODO

- **Optimize DiagTensor × DenseTensor contraction**: Currently, DiagTensor is converted to DenseTensor before contraction, which is inefficient. This can be optimized by implementing Block Matrix × Block Matrix contraction, as DiagTensor × DenseTensor is a special case of block matrix multiplication.

## Acknowledgments

This implementation is inspired by **ITensors.jl** (https://github.com/ITensor/ITensors.jl).
We have borrowed API design concepts and function names for compatibility, but the implementation
is independently written in Rust.

**Note**: This library is experimental and not intended for production use. If you use this code
in research and publish a paper, please cite:

> We used tensor4all-rs (https://github.com/tensor4all/tensor4all-rs), inspired by ITensors.jl.

If you cite ITensors.jl directly, please use:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", arXiv:2007.14822 (2020)

## References

- ITensors.jl: https://github.com/ITensor/ITensors.jl
- ITensors.jl paper: M. Fishman, S. R. White, E. M. Stoudenmire, arXiv:2007.14822 (2020)
- QSpace v4.0 Documentation: `qspace-v4-pub/Docu/user-guide.pdf`
- QSpace Source: `qspace-v4-pub/Source/QSpace.hh`, `qspace-v4-pub/Source/wbindex.hh`
- Original QSpace paper: A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- X-symbols paper: A. Weichselbaum, Phys. Rev. Research **2**, 023385 (2020)

## License

This project is dual-licensed under either:

- **MIT License** (see [LICENSE-MIT](LICENSE-MIT))
- **Apache License 2.0** (see [LICENSE-APACHE](LICENSE-APACHE))

at your option.
