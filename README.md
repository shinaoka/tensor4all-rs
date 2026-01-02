# tensor4all-rs

A Rust implementation of tensor networks with quantum number symmetries, inspired by ITensors.jl and QSpace v4.

## Overview

tensor4all-rs provides a type-safe, efficient implementation of tensor networks with support for quantum number symmetries. The design is inspired by both ITensors.jl (Julia) and QSpace v4 (MATLAB/C++), which represent the same mathematical concept: block-sparse tensors organized by quantum numbers.

## Key Features

- **Type-safe Index system**: Generic `Index<Id, Symm>` type supporting both runtime and compile-time identities
- **Tag support**: Index tags with configurable capacity (default: max 4 tags, each max 16 characters)
- **Quantum number symmetries**: Support for Abelian (U(1), Z_n) and non-Abelian (SU(2), SU(N)) symmetries (planned)
- **Thread-safe ID generation**: Sequential UInt64 IDs using AtomicU64 for deterministic ID generation
- **Flexible tensor types**: Both dynamic-rank and static-rank tensor variants
- **Copy-on-write storage**: Efficient memory management for tensor networks
- **Multiple storage backends**: DenseF64 and DenseC64 storage types

## Design Philosophy

### Comparison with Existing Libraries

| Concept | QSpace v4 | ITensors.jl | tensor4all-rs |
|---------|-----------|-------------|---------------|
| **Tensor with QNs** | `QSpace` | `ITensor` | `TensorDynLen<Id, T, Symm>` / `TensorStaticLen<N, Id, T, Symm>` |
| **Index** | Quantum number labels in `QIDX` | `Index{QNBlocks}` | `Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN>` |
| **Storage** | `DATA` (array of blocks) | `NDTensors.BlockSparse` | `Storage` enum (DenseF64, DenseC64) |
| **Language** | MATLAB/C++ | Julia | Rust |

### Index Design

The `Index` type is parameterized by identity type `Id`, symmetry type `Symm`, and tag capacity:

```rust
pub struct Index<Id, Symm = NoSymmSpace, const MAX_TAGS: usize = 4, const MAX_TAG_LEN: usize = 16> {
    pub id: Id,
    pub symm: Symm,
    pub tags: TagSet<MAX_TAGS, MAX_TAG_LEN>,
}
```

**Identity Types**:
- `DynId` (u64): Runtime identity with sequential ID generation using `AtomicU64`
- ZST marker types: Compile-time-known identity for static analysis

**Symmetry Types**:
- `NoSymmSpace`: No symmetry (corresponds to `Index{Int}` in ITensors.jl)
- `QNSpace` (planned): Quantum number spaces (corresponds to `Index{QNBlocks}`)

**Tags**:
- Configurable tag capacity via const generics
- Default: max 4 tags, each max 16 characters
- Tags are stored in `TagSet` using `SmallString` for efficient storage

### ID Generation

tensor4all-rs uses **sequential UInt64 ID generation** with `AtomicU64`:

- Thread-safe sequential ID generation starting from 1
- Deterministic and predictable ID assignment
- Simple and efficient implementation
- No collision risk for practical use cases

**Note**: The current implementation uses sequential IDs. Future versions may switch to random ID generation with thread-local RNG for better hash distribution (similar to ITensors.jl's approach).

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
- `DenseF64`: Dense storage for `f64` elements
- `DenseC64`: Dense storage for `Complex64` elements

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

## Usage Example

```rust
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_core::storage::Storage;
use tensor4all_core::tensor::TensorDynLen;
use std::sync::Arc;

// Create indices
let i = Index::new_dyn(2);  // Index with dimension 2, auto-generated ID
let j = Index::new_dyn(3);  // Index with dimension 3, auto-generated ID

// Create storage
let storage = Arc::new(Storage::new_dense_f64(6));  // Capacity for 2×3=6 elements

// Create tensor
let indices = vec![i, j];
let dims = vec![2, 3];
let tensor: TensorDynLen<DynId, f64> = TensorDynLen::new(indices, dims, storage);
```

## Future Extensions

- **Quantum Number Space**: Support for quantum number symmetries
- **Arrow/Direction**: Index direction encoding for non-Abelian symmetries
- **Non-Abelian Support**: Clebsch-Gordan coefficients for non-Abelian symmetries
- **Random ID generation**: Thread-local RNG for better hash distribution (similar to ITensors.jl)

## References

- ITensors.jl: https://github.com/ITensor/ITensors.jl
- QSpace v4.0 Documentation: `qspace-v4-pub/Docu/user-guide.pdf`
- QSpace Source: `qspace-v4-pub/Source/QSpace.hh`, `qspace-v4-pub/Source/wbindex.hh`
- Original QSpace paper: A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- X-symbols paper: A. Weichselbaum, Phys. Rev. Research **2**, 023385 (2020)

## License

MIT OR Apache-2.0
