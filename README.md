# tensor4all-rs

A Rust implementation of tensor networks with quantum number symmetries, inspired by ITensors.jl and QSpace v4.

## Overview

tensor4all-rs provides a type-safe, efficient implementation of tensor networks with support for quantum number symmetries. The design is inspired by both ITensors.jl (Julia) and QSpace v4 (MATLAB/C++), which represent the same mathematical concept: block-sparse tensors organized by quantum numbers.

## Key Features

- **Type-safe Index system**: Generic `Index<Id, Symm>` type supporting both runtime and compile-time identities
- **Quantum number symmetries**: Support for Abelian (U(1), Z_n) and non-Abelian (SU(2), SU(N)) symmetries (planned)
- **Thread-safe ID generation**: UInt128 random IDs using thread-local RNG for extremely low collision probability
- **Flexible tensor types**: Both dynamic-rank and static-rank tensor variants
- **Copy-on-write storage**: Efficient memory management for tensor networks

## Design Philosophy

### Comparison with Existing Libraries

| Concept | QSpace v4 | ITensors.jl | tensor4all-rs |
|---------|-----------|-------------|---------------|
| **Tensor with QNs** | `QSpace` | `ITensor` | `Tensor<Id, T, Symm>` |
| **Index** | Quantum number labels in `QIDX` | `Index{QNBlocks}` | `Index<Id, Symm>` |
| **Storage** | `DATA` (array of blocks) | `NDTensors.BlockSparse` | `Storage` enum |
| **Language** | MATLAB/C++ | Julia | Rust |

### Index Design

The `Index` type is parameterized by both identity type `Id` and symmetry type `Symm`:

```rust
pub struct Index<Id, Symm = NoSymmSpace, const MAX_TAGS: usize = 4, const MAX_TAG_LEN: usize = 16> {
    pub id: Id,
    pub symm: Symm,
    pub tags: TagSet<MAX_TAGS, MAX_TAG_LEN>,
}
```

**Identity Types**:
- `DynId` (u128): Runtime identity with thread-local random ID generation
- ZST marker types: Compile-time-known identity for static analysis

**Symmetry Types**:
- `NoSymmSpace`: No symmetry (corresponds to `Index{Int}` in ITensors.jl)
- `QNSpace` (planned): Quantum number spaces (corresponds to `Index{QNBlocks}`)

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
2. **Static rank**: `TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace>`

### Storage

Tensor data is shared via `Arc<Storage>` with copy-on-write (COW) semantics:
- If uniquely owned, mutate in place
- If shared, clone then mutate

## Type Correspondence

| ITensors.jl | tensor4all-rs |
|-------------|--------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `Index<Id, QNSpace>` (future) |
| `Index(id, dim, ...)` | `Index::new_with_size(id, dim)` |
| `Index(dim)` | `Index::new_dyn(dim)` |

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

## References

- ITensors.jl: https://github.com/ITensor/ITensors.jl
- QSpace v4.0 Documentation: `qspace-v4-pub/Docu/user-guide.pdf`
- QSpace Source: `qspace-v4-pub/Source/QSpace.hh`, `qspace-v4-pub/Source/wbindex.hh`
- Original QSpace paper: A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- X-symbols paper: A. Weichselbaum, Phys. Rev. Research **2**, 023385 (2020)

## License

MIT OR Apache-2.0
