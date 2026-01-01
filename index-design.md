# Index and IdxTensor Design (inspired by ITensors.jl)

This document describes the design of `Index` and `IdxTensor` types for Rust, inspired by ITensors.jl's `Index` and `ITensor` design.

## Rust Goal

We want to support **both**:

- **Compatibility with ITensors.jl's dynamic index model** (runtime identity and matching), and
- **Strong compile-time checks enabled by Rust's static type system** (especially for structured tensor networks where index wiring/order is known ahead of time).

## ITensors.jl Design Summary

### Index Structure

The `Index` type in ITensors.jl has the following structure:

```julia
struct Index{T}
  id::IDType          # UInt64 - unique identifier
  space::T            # dimension or QN space structure
  dir::Arrow          # direction: In, Out, or Neither
  tags::TagSet        # set of fixed-length string tags
  plev::Int           # prime level (integer)
end
```

**Key Properties:**

1. **ID-based Identity**: Each `Index` has a unique `id` (UInt64) generated randomly when created. This `id` is the primary way to identify that two indices are copies of the same original index.

2. **Equality**: Two indices are equal if and only if:
   - They have the same `id`
   - They have the same `plev` (prime level)
   - They have the same `tags`

3. **Space**: The `space` field represents the dimension (for simple indices) or quantum number structure (for QN indices). For simple indices, `space` is just an integer dimension.

4. **Direction**: The `dir` field indicates the arrow direction (`In`, `Out`, or `Neither`), which is important for quantum number conservation in tensor contractions.

5. **Tags**: A `TagSet` is a collection of fixed-length string tags:
   - **TagSet Structure**: `TagSet = GenericTagSet{IntTag,4}` - can hold up to 4 tags
   - **Tag Format**: Each tag is a `SmallString` with maximum length of 15 characters
     - Stored as `SVector{16, UInt16}` (16 UInt16 values including null terminator)
     - Can be cast to/from `UInt256` for efficient storage and comparison
   - **Storage**: Tags are stored as `UInt256` integers (cast from SmallString) in a `SVector{4, UInt256}`
   - **Use Cases**: Tags help identify and filter indices:
     - Identifying indices when printing
     - Filtering indices for operations (e.g., priming indices with certain tags)
     - Organizing indices in tensor networks
   - **Note**: The fixed-length design (15 chars max per tag, 4 tags max) is optimized for performance and memory efficiency

6. **Prime Level**: An integer that can be incremented/decremented. Indices with different prime levels are considered different even if they have the same `id`. This is useful for distinguishing indices in time evolution or other operations where you need multiple copies of the same index.

**Key Operations:**

- `copy(i::Index)`: Creates a copy with the same `id`, `space`, `dir`, `tags`, and `plev`
- `sim(i::Index)`: Creates a similar index (same properties) but with a new `id`
- `prime(i::Index, plinc::Int)`: Increments prime level
- `setprime(i::Index, plev::Int)`: Sets prime level
- `addtags(i::Index, ts)`: Adds tags
- `removetags(i::Index, ts)`: Removes tags
- `dag(i::Index)`: Reverses direction (for QN indices)

### ITensor Structure

The `ITensor` type in ITensors.jl has the following structure:

```julia
mutable struct ITensor
  tensor  # internal NDTensors.Tensor
end
```

**Key Properties:**

1. **Index-Order Independent Interface**: The ITensor interface is independent of the memory layout. You don't need to know the ordering of indices - only which indices an ITensor has.

2. **Automatic Index Matching**: Operations like contraction and addition automatically handle any memory permutations. Indices are matched by their `id`, `plev`, and `tags`.

3. **Index Access**: 
   - `inds(T::ITensor)`: Returns the indices as a Tuple
   - `ind(T::ITensor, i::Int)`: Gets the Index at position `i`
   - Indexing with `IndexVal` pairs: `T[i => 1, j => 2]` works regardless of the internal order

4. **Storage Abstraction**: The internal `tensor` field can have different storage types:
   - `Dense`: Dense storage
   - `Diag`: Diagonal sparse storage
   - `BlockSparse`: Block sparse storage (for QN tensors)
   - `EmptyStorage`: Unallocated storage

**Key Operations:**

- `ITensor(inds...)`: Construct from indices
- `T[i => val, j => val]`: Get/set elements using IndexVal pairs
- `A * B`: Contract tensors (automatically matches indices)
- `A + B`: Add tensors (automatically handles permutation)
- `prime(T, plinc)`: Prime indices
- `addtags(T, tags)`: Add tags to indices
- `permute(T, inds)`: Permute to match given index order

**Index Matching Logic:**

When contracting or adding ITensors:
1. Indices are matched by `id`, `plev`, and `tags`
2. The internal storage can be in any order
3. The library automatically permutes as needed
4. For QN indices, directions must be opposite for contraction

## Rust Design Considerations

### Index Design

For Rust, we need to consider:

1. **ID Generation**: 
   - Use a thread-local or global RNG for generating unique IDs
   - Consider using `u64` for IDs (same as ITensors.jl)
   - Need to ensure thread-safety if using shared RNG

2. **Equality and Hashing**:
   - Implement `PartialEq` based on `(id, plev, tags)`
   - Implement `Hash` based on `(id, plev, tags)` for use in hash maps/sets
   - Consider using `Eq` if we can guarantee uniqueness

3. **Space Representation**:
   - For simple indices: `usize` dimension
   - For QN indices: More complex structure (can be added later)
   - Use generics or enums to handle both cases

4. **Tags**:
   - Use a `TagSet` type (could be `HashSet<String>` or a more optimized structure)
   - Consider using `SmallVec` or similar for small tag sets

5. **Prime Level**:
   - Simple `i32` or `i64` field
   - Consider using `NonZeroI32` if we want to optimize for zero prime level

6. **Direction**:
   - Enum: `In`, `Out`, `Neither`
   - For QN indices, this is important for flux conservation

### IdxTensor Design

For Rust, we need to consider:

1. **Index Storage**:
   - Store indices as a collection (Vec or similar)
   - Need efficient lookup by Index (for matching)
   - Consider using `IndexMap` or similar for O(1) lookup

2. **Storage Abstraction**:
   - Use trait objects or generics for different storage types
   - Consider using `enum` for storage variants if the number is small
   - Need to handle dense, sparse, diagonal, etc.

3. **Index Matching**:
   - Need efficient algorithm to match indices between tensors
   - Consider building a permutation map when needed
   - Cache permutation maps if operations are repeated

4. **Element Access**:
   - Support indexing with `(Index, usize)` pairs
   - Need to handle permutation internally
   - Consider using `IndexVal` type alias: `type IndexVal = (Index, usize)`

5. **Ownership and Borrowing**:
   - Consider using `Rc<Index>` or `Arc<Index>` to share indices
   - Or use indices by value and clone when needed
   - Need to balance performance vs. memory usage

6. **Type Safety**:
   - Consider using type-level programming for compile-time checks
   - But need to balance with runtime flexibility
   - Consider using const generics for fixed-rank tensors

### Key Differences from Julia

1. **Ownership Model**: Rust's ownership system means we need to carefully consider how indices are shared between tensors. Options:
   - `Rc<Index>` or `Arc<Index>` for shared ownership
   - Clone indices when needed
   - Use indices by value and accept cloning

2. **Type System**: Rust's type system is more strict. We may want to:
   - Use generics for element types
   - Use traits for storage backends
   - Consider const generics for compile-time optimizations

3. **Error Handling**: Use `Result` types for operations that can fail (e.g., dimension mismatches, index not found)

4. **Performance**: 
   - Consider using `SmallVec` for small index sets
   - Use `IndexMap` for O(1) index lookup
   - Consider using `ndarray` or similar for storage

5. **Thread Safety**: 
   - If using shared RNG for ID generation, need `Arc<Mutex<...>>` or similar
   - Consider using thread-local storage for RNG

## 2. Rust Design for QTT/TCI Use Cases

### 2.1 Design Philosophy

For QTT (Quantics Tensor Train) and TCI (Tensor Cross Interpolation) use cases, we can simplify the Index design by removing features that are not needed:

- **No QN (Quantum Numbers)**: QTT/TCI don't require quantum number conservation
- **No Direction**: Arrow direction is only needed for QN flux conservation
- **No Prime Level**: Prime levels are confusing and unnecessary for QTT/TCI use cases

This simplification allows us to focus on:
- **ID-based matching**: Core feature for automatic index matching in contractions
- **Size information**: Dimension of the index
- **Optional tags**: For filtering and identification (can be simplified or removed)

In addition, for the typed-index approach (e.g. `Tensor<(A, B, D)>`), we can make the **index labels and their order** compile-time, while keeping the **per-label dimensions** dynamic at runtime (stored in a `dims` vector/array aligned with the type-level order). In that approach, a "fixed-size index" type is not required initially.

### 2.2 Common Index Trait

All Index types implement a common trait for unified operations:

```rust
pub trait Index {
    fn size(&self) -> usize;
    fn tags(&self) -> &TagSet;
}

// Dynamic/runtime ID is only available for runtime Index types
pub trait IndexWithDynId: Index {
    fn id(&self) -> u64;
}
```

### 2.3 Three Index Types

We propose three Index types with different trade-offs between flexibility and compile-time guarantees:

#### 2.3.1 DynSizeIndex (Dynamic Size Index)

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DynSizeIndex {
    id: u64,      // unique identifier
    size: usize,  // dimension (runtime value)
    tags: TagSet, // set of tags for identification (ITensor compatible)
}

impl DynSizeIndex {
    pub fn new(size: usize) -> Self {
        Self {
            id: generate_id(),  // thread-safe ID generation
            size,
            tags: TagSet::empty(),
        }
    }
    
    pub fn with_tags(size: usize, tags: TagSet) -> Self {
        Self {
            id: generate_id(),
            size,
            tags,
        }
    }
}

impl Index for DynSizeIndex {
    fn size(&self) -> usize { self.size }
    fn tags(&self) -> &TagSet { &self.tags }
}

impl IndexWithDynId for DynSizeIndex {
    fn id(&self) -> u64 { self.id }
}
```

**Characteristics:**
- Most flexible: size determined at runtime
- Almost compatible with ITensors.jl's Index (without QN, dir, plev)
- Good for dynamic tensor networks
- Permutation/contraction order determined at runtime

#### 2.3.2 StaticIndex (Static Index)

```rust
// Type-level marker trait (the marker *type* is the identity)
pub trait IndexId: 'static + Send + Sync {}

// StaticIndex with type-level identity
// Note: We only need equality/inequality (and hashing for lookup). No ordering is required.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StaticIndex<I: IndexId, const N: usize> {
    _phantom: PhantomData<I>,
}

impl<I: IndexId, const N: usize> StaticIndex<I, N> {
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    
    pub const fn size_const() -> usize { N }
}

impl<I: IndexId, const N: usize> Index for StaticIndex<I, N> {
    fn size(&self) -> usize { N }
    fn tags(&self) -> &TagSet {
        // StaticIndex doesn't have tags - return empty TagSet
        static EMPTY: TagSet = TagSet::empty();
        &EMPTY
    }
}

// Conversion from StaticIndex to DynSizeIndex (always succeeds)
impl<I: IndexId, const N: usize> From<StaticIndex<I, N>> for DynSizeIndex {
    fn from(_index: StaticIndex<I, N>) -> Self {
        DynSizeIndex {
            id: generate_id(),
            size: N,
            tags: TagSet::empty(),
        }
    }
}

// Example usage:
pub struct I1;
impl IndexId for I1 {}

pub struct I2;
impl IndexId for I2 {}

// Usage: StaticIndex<I1, 8>, StaticIndex<I2, 16>
```

**Characteristics:**
- ID is encoded at the type level, preventing accidental duplication
- No runtime ID storage needed (identity is the marker type `I`)
- Size is a compile-time constant
- Zero runtime overhead
- Permutation/contraction order can be determined at compile time
- Enables compile-time optimizations for tensor operations
- Most restrictive but highest performance potential
- Type-level ID ensures uniqueness at the type level (distinct marker types are distinct indices)
- **Note**: runtime ID storage is only needed for `DynSizeIndex`. `StaticIndex` has no runtime ID storage.

### 2.4 Design Discussion

#### Use Cases for Each Type

1. **DynSizeIndex**: 
   - General-purpose tensor operations
   - When tensor dimensions are not known at compile time
   - Prototyping and flexible algorithms

2. **StaticIndex**:
   - High-performance tensor operations
   - When the entire tensor network structure is known at compile time
   - Enables compile-time contraction order optimization

#### Tensor Type Design

With these Index types, we can design corresponding tensor types:

```rust
// Dynamic tensor - most flexible
pub struct DynIdxTensor {
    indices: Vec<DynSizeIndex>,
    storage: Storage,
}

// Static tensor - full compile-time optimization
// Indices are encoded in the type as a tuple.
// Per-index dimensions are stored in `dims`, aligned with the type-level order.
// The rank (number of indices) is a const generic parameter `N`.
pub struct StaticIdxTensor<const N: usize, Idxs, T, S>
where
    Idxs: StaticIndices,
{
    // NOTE: `N` should match the tuple arity of `Idxs` (enforce via a trait if needed).
    dims: [usize; N],
    storage: S,
    _phantom: PhantomData<(Idxs, T)>,
}
```

Why do we keep two tensor representations?

- **ITensors.jl compatibility**: `DynIdxTensor` (with `DynSizeIndex`) mirrors the ITensors.jl philosophy where identity (id/tags) is runtime data. This is useful when indices are created dynamically and matching is done by runtime metadata.
- **Compile-time consistency checks for structured networks (e.g. MPO/MPS/DMRG)**: in many MPO contractions inside DMRG you repeatedly contract small tensors (order 3–4 is common) where the index labels and their order are fixed by the algorithm. `StaticIdxTensor` encodes the label order in the type (e.g. `Tensor<(A, B, D)>`), enabling compile-time validation of index wiring/order. Per-label dimensions remain runtime values (`dims`), but the *shape interface* is type-checked.

#### Key Questions

1. **Index type homogeneity**: Tensors use homogeneous Index types (all indices must be the same type). This simplifies the implementation and enables better optimizations.

3. **Contraction API**: How should contraction work with different Index types?
   - Runtime matching for DynSizeIndex
   - Compile-time matching for StaticIndex?

4. **Migration path**: We provide conversions between types:
   - `DynSizeIndex` → `StaticIndex<I, N>`: this is a *re-interpretation* into a different identity system (type-level). If you need a stable mapping between runtime IDs and marker types, maintain an explicit registry/mapping at runtime.
   - Conversions preserve tags when possible (note: `StaticIndex` has no tags).

5. **Tag support**: Tags are supported in `DynSizeIndex` for ITensor compatibility. `StaticIndex` doesn't need tags since everything is determined at compile time.

6. **ID support**: Runtime/dynamic ID storage is only needed for `DynSizeIndex` via `IndexWithDynId`. `StaticIndex` has no numeric ID; its identity is the marker type `I`.

## Open Questions

1. **Index Sharing**: Should we use `Rc<Index>` or clone indices? What's the performance trade-off?

2. **Storage Backend**: Should we use `ndarray`, custom storage, or trait objects?

3. **Error Types**: What error types should we use? Custom enum or use `anyhow`/`thiserror`?

4. **Generic Element Types**: Should `IdxTensor` be generic over element type (`T: Float`)?

5. **QN Support**: Should we design for QN indices from the start, or add later?

6. **Thread Safety**: How should we handle ID generation in multi-threaded contexts?

7. **Serialization**: Should we support serialization? What format?

8. **Performance vs. Flexibility**: How much compile-time checking vs. runtime flexibility?


