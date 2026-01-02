# tensor4all-rs Design Document

This document describes the design of tensor4all-rs, a Rust implementation of tensor networks with quantum number symmetries, inspired by ITensors.jl and QSpace v4.

---

## Part 1: Existing Libraries - ITensors.jl and QSpace v4

### Overview

**ITensors.jl** (Julia) and **QSpace v4** (MATLAB/C++) represent the **same concept**: tensors with quantum number symmetries (block-sparse tensors). They are different implementations of the same mathematical object.

| Concept | QSpace v4 | ITensors.jl |
|---------|-----------|-------------|
| **Tensor with QNs** | `QSpace` | `ITensor` |
| **Index** | Quantum number labels in `QIDX` | `Index{QNBlocks}` |
| **Storage** | `DATA` (array of blocks) | `NDTensors.BlockSparse` |
| **Language** | MATLAB/C++ | Julia |

Both represent:
- Block-sparse tensors organized by quantum numbers
- Support for Abelian (U(1), Z_n) and non-Abelian (SU(2), SU(N)) symmetries
- Efficient storage and computation using symmetry constraints

### ITensors.jl Index System

#### Index Structure

```julia
struct Index{T}
  id::IDType          # UInt64 - unique identifier
  space::T            # Space information (Int or QNBlocks)
  dir::Arrow          # Direction (In, Out, Neither)
  tags::TagSet         # Tag set
  plev::Int            # Prime level (index versioning)
  function Index{T}(id, space::T, dir::Arrow, tags, plev) where {T}
    return new{T}(id, space, dir, tags, plev)
  end
end
```

#### No Symmetry Case

When there is no symmetry, `T = Int` and `space::Int` is simply the dimension:

```julia
i = Index(2)  # Index{Int}
# i.space = 2  (Int type)
# dim(i) = i.space = 2
```

#### QN Index Case

When there are quantum numbers, `T = QNBlocks`:

```julia
const QNBlock = Pair{QN, Int64}
const QNBlocks = Vector{QNBlock}
const QNIndex = Index{QNBlocks}

i = Index([
    QN("Sz", -1) => 2,  # Block 1: dimension 2
    QN("Sz",  0) => 4,  # Block 2: dimension 4
    QN("Sz", +1) => 2,  # Block 3: dimension 2
])
# i.space = [QN("Sz", -1) => 2, QN("Sz", 0) => 4, QN("Sz", +1) => 2]
# dim(i) = sum(blockdim for (_, blockdim) in i.space) = 2 + 4 + 2 = 8
```

**Key Point**: The total dimension of a QN Index is the **sum of all block dimensions**.

#### Index Dimension Calculation

```julia
function dim(qnblocks::QNBlocks)
  dimtot = 0
  for (_, blockdim) in qnblocks
    dimtot += blockdim
  end
  return dimtot
end
```

### QSpace v4 Index System

#### QSpace Class Structure

```cpp
template <class TQ, class TD>
class QSpace {
    wbMatrix<TQ> QIDX;              // Quantum number index matrix
    wbvector<wbarray<TD>*> DATA;    // Data blocks array
    wbMatrix<CRef<TQ>> CGR;         // Clebsch-Gordan coefficients (for non-Abelian)
    QVec qtype;                     // Symmetry type information
    unsigned QDIM;                   // Quantum number dimension
    iTags itags;                     // Index tags (string labels)
    QS_TYPES otype;                  // Object type (operator, A-matrix, etc.)
};
```

#### Key Components

1. **QIDX (Quantum Number Index Matrix)**
   - Each row represents a quantum number label for a tensor block
   - `QIDX.dim1` = number of blocks, `QIDX.dim2` = `rank * QDIM`
   - For a rank-`r` tensor, each row contains `r` quantum number vectors

2. **itags (Index Tags)**
   - String labels for each tensor leg
   - Compact 8-byte representation per tag
   - Conjugation flag (`*`) encodes arrow direction (In/Out)

3. **Arrow (Direction) System**
   - Encoded in the conjugation flag of `itag_`
   - `+` (no `*`): Out direction
   - `-` (with `*`): In direction
   - Critical for non-Abelian symmetries

4. **CGR (Clebsch-Gordan Coefficients)**
   - Stores CGC for non-Abelian symmetries
   - Only present for non-Abelian cases; empty for pure Abelian

#### Index Dimension

Same as ITensors.jl: **total dimension = sum of all block dimensions**.

### Comparison: ITensors.jl vs QSpace v4

| Feature | QSpace v4 | ITensors.jl |
|---------|-----------|-------------|
| Index ID | Not used | `id::UInt64` |
| Quantum Numbers | `QIDX` matrix | `space::QNBlocks` (`Vector{Pair{QN,Int}}`) |
| Arrow | Conjugation flag in `itag_` | `dir::Arrow` field |
| Tags | `itags` (string labels) | `tags::TagSet` |
| Prime Level | Not used | `plev::Int` |

**Key Differences**:
- QSpace uses conjugation flag in tags for arrow, ITensors uses separate `dir` field
- QSpace stores quantum numbers in matrix form, ITensors uses `Vector{Pair{QN,Int}}`
- QSpace doesn't use prime levels (index versioning), ITensors does

### Design Principles

1. **Compact Representation**: Efficient memory usage for large tensor networks
2. **Arrow Direction Encoding**: Preserved during tensor operations
3. **Quantum Number Blocking**: Each tensor block labeled by quantum number vector
4. **Non-Abelian Support**: Clebsch-Gordan coefficients for non-Abelian symmetries

### ITensors.jl ID Generation Algorithm

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

**Collision Probability Analysis** (Birthday Paradox):

For `n` randomly generated IDs, the probability of at least one collision is approximately:

```
P(collision) ≈ 1 - exp(-n² / (2 × 2^b))
```

where `b` is the number of bits (64 for UInt64, 128 for UInt128).

**UInt64 (64-bit IDs)**:
- **n = 10^5 (100,000 indices)**: P(collision) ≈ **2.7 × 10^-10** (0.000000027%)
- **n = 10^6 (1 million indices)**: P(collision) ≈ **2.7 × 10^-8** (0.0000027%)
- **n = 10^9 (1 billion indices)**: P(collision) ≈ **2.7 × 10^-2** (2.7%)
- Collision probability becomes significant (>50%) at approximately **√(2^64 × ln(2)) ≈ 5.4 × 10^9** IDs

**UInt128 (128-bit IDs)**:
- **n = 10^5 (100,000 indices)**: P(collision) ≈ **1.5 × 10^-29** (negligible)
- **n = 10^6 (1 million indices)**: P(collision) ≈ **1.5 × 10^-27** (negligible)
- **n = 10^9 (1 billion indices)**: P(collision) ≈ **1.5 × 10^-21** (negligible)
- Collision probability becomes significant (>50%) at approximately **√(2^128 × ln(2)) ≈ 1.5 × 10^19** IDs

**Comparison**: UInt128 provides approximately **2^64 ≈ 1.8 × 10^19** times lower collision probability than UInt64 for the same number of IDs. For 10^5 indices, UInt128 reduces the collision probability from 2.7 × 10^-10 to 1.5 × 10^-29, making collisions effectively impossible in practice.

**Comparison with tensor4all-rs**:
- **ITensors.jl**: Random IDs (UInt64 from Xoshiro RNG, task-local)
- **tensor4all-rs**: Random IDs (UInt128 from thread-local RNG)

**tensor4all-rs implementation**:
- Uses `u128` (UInt128) for even lower collision probability
- Thread-local random number generator (similar to ITensors.jl's task-local RNG)
- Each thread has its own RNG instance, providing thread-safe ID generation without global synchronization
- Collision probability for 10^5 indices: ~1.5 × 10^-29 (effectively zero)

The UInt128 approach provides significantly better collision resistance than UInt64 while maintaining the benefits of random ID generation (better hash distribution, no ID reuse issues).

---

## Part 2: Rust Design for tensor4all-rs

### Goals

- Keep compatibility with **ITensors.jl-style dynamic identity** (runtime IDs)
- Also support **strong compile-time checks** where possible (e.g. fixed tensor rank)
- Support **quantum number symmetries** via type parameter (like `Index{T}` in ITensors.jl)
- Keep **index sizes dynamic** (computed from symmetry information)

### Index Design

#### Core Structure

The `Index` type is parameterized by both identity type `Id` and symmetry type `Symm`:

```rust
/// Trait for symmetry information (quantum number space).
///
/// This corresponds to the `T` parameter in ITensors.jl's `Index{T}`,
/// where `T = Int` for no symmetry and `T = QNBlocks` for quantum numbers.
pub trait Symmetry: Clone + PartialEq + Eq + std::hash::Hash {
    /// Return the total dimension of the space.
    ///
    /// For no symmetry, this is just the dimension.
    /// For quantum number spaces, this is the sum of all block dimensions.
    fn total_dim(&self) -> usize;
}

/// No symmetry space (corresponds to ITensors.jl's `Index{Int}`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoSymmSpace {
    dim: usize,
}

impl Symmetry for NoSymmSpace {
    fn total_dim(&self) -> usize {
        self.dim
    }
}

/// Index with generic identity type `Id` and symmetry type `Symm`.
///
/// - `Id = DynId` for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Symm = NoSymmSpace` for no symmetry (default, corresponds to `Index{Int}`)
/// - `Symm = QNSpace` (future) for quantum numbers (corresponds to `Index{QNBlocks}`)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index<Id, Symm = NoSymmSpace> {
    pub id: Id,
    pub symm: Symm,
}

impl<Id, Symm: Symmetry> Index<Id, Symm> {
    pub fn new(id: Id, symm: Symm) -> Self {
        Self { id, symm }
    }

    /// Get the total dimension (size) of the index.
    pub fn size(&self) -> usize {
        self.symm.total_dim()
    }
}
```

#### Identity Types

```rust
/// Runtime ID for ITensors-like dynamic identity.
///
/// Uses UInt128 for extremely low collision probability (see design.md for analysis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u128);

thread_local! {
    /// Thread-local random number generator for ID generation.
    ///
    /// Each thread has its own RNG, similar to ITensors.jl's task-local RNG.
    /// This provides thread-safe ID generation without global synchronization.
    static ID_RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::thread_rng());
}

/// Generate a unique random ID for dynamic indices (thread-safe).
///
/// Uses thread-local random number generator to generate UInt128 IDs,
/// providing extremely low collision probability.
pub fn generate_id() -> u128 {
    ID_RNG.with(|rng| rng.borrow_mut().gen())
}
```

#### Convenience Constructors

```rust
impl<Id> Index<Id, NoSymmSpace> {
    /// Create a new index with no symmetry from dimension.
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
        }
    }
}

impl Index<DynId, NoSymmSpace> {
    /// Create a new index with a generated dynamic ID and no symmetry.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
        }
    }
}
```

### Type Correspondence

| ITensors.jl | tensor4all-rs |
|-------------|--------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `Index{QNBlocks}` | `Index<Id, QNSpace>` (future) |
| `Index(id, dim, ...)` | `Index::new_with_size(id, dim)` |
| `Index(dim)` | `Index::new_dyn(dim)` |

### Storage

```rust
#[derive(Debug, Clone)]
pub enum Storage {
    DenseF64(Vec<f64>),
    DenseC64(Vec<Complex64>),
}
```

### Tensor Types

Two tensor variants for different use cases:

1. **Dynamic rank**: `TensorDynLen<Id, T, Symm = NoSymmSpace>`
2. **Static rank**: `TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace>`

```rust
pub struct TensorDynLen<Id, T, Symm = NoSymmSpace> {
    pub indices: Vec<Index<Id, Symm>>,
    pub dims: Vec<usize>,
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}

pub struct TensorStaticLen<const N: usize, Id, T, Symm = NoSymmSpace> {
    pub indices: [Index<Id, Symm>; N],
    pub dims: [usize; N],
    pub storage: Arc<Storage>,
    _phantom: std::marker::PhantomData<T>,
}
```

### Element Types

- **Static element type**: Pick a concrete `T` and matching storage strategy
- **Dynamic element type**: Use `AnyScalar` and dispatch by matching on `Storage`

### Tensor Networks and Shared Storage

Tensor network objects (like an `MPS`) are highly dynamic and repeatedly update local tensors.

We use **shared tensor storage without exclusive ownership**:

- Tensor data is shared via `Arc<Storage>` (thread-safe shared ownership)
- Updates use **copy-on-write (COW)** via `Arc::make_mut(&mut arc_storage)`:
  - If uniquely owned, mutate in place
  - If shared, clone then mutate

### Future Extensions

#### Quantum Number Space

```rust
pub struct QNSpace {
    blocks: Vec<QNBlock>,  // Vec<(QN, usize)>
}

impl Symmetry for QNSpace {
    fn total_dim(&self) -> usize {
        self.blocks.iter().map(|(_, dim)| dim).sum()
    }
}
```

#### Arrow/Direction

Two design options:

**Option A**: Store arrow as separate field (like ITensors.jl)
```rust
pub struct Index<Id, Symm> {
    pub id: Id,
    pub symm: Symm,
    pub arrow: Arrow,  // In, Out, Neither
}
```

**Option B**: Encode in symmetry metadata (like QSpace)
- Arrow could be part of `Symm` type
- Or stored in a wrapper `Leg<Index, Arrow>`

#### Non-Abelian Support

```rust
pub struct NonAbelianSymm {
    sectors: Vec<Sector>,
    cgr: Option<CGRMatrix>,  // Clebsch-Gordan coefficients
}
```

### Design Decisions

1. **Type Parameter for Symmetry**: Following ITensors.jl's `Index{T}` pattern, we use `Index<Id, Symm>` where `Symm` defaults to `NoSymmSpace`
2. **No Backward Compatibility**: Clean break from old `Index<Id>` design
3. **Size as Method**: `size()` method computes from `symm.total_dim()` rather than storing separately
4. **Extensibility**: Easy to add `QNSpace`, `IrrepSpace`, etc. by implementing `Symmetry` trait

---

## References

- ITensors.jl: https://github.com/ITensor/ITensors.jl
- QSpace v4.0 Documentation: `qspace-v4-pub/Docu/user-guide.pdf`
- QSpace Source: `qspace-v4-pub/Source/QSpace.hh`, `qspace-v4-pub/Source/wbindex.hh`
- Original QSpace paper: A. Weichselbaum, Annals of Physics **327**, 2972 (2012)
- X-symbols paper: A. Weichselbaum, Phys. Rev. Research **2**, 023385 (2020)
