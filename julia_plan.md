# Julia wrapper plan for `tensor4all` (Rust) via a local C-API (no JLL)

## Goal (first milestone)

Create a Julia package (`Tensor4all.jl`, module `Tensor4all`) that **builds the local Rust workspace** and accesses **tensor4all** through a **C ABI** (a `cdylib`), without using a JLL.

Milestone 1 scope:

- Julia can **create** a `tensor4all` `Index` and **own/release** it safely.
- Provide basic accessors: **dimension**, **tags**, **id**.
- Add **weak dependency** on `ITensors.jl` and enable **bidirectional conversion** between:
  - Rust-backed `Tensor4all.Index` (Julia wrapper type)
  - `ITensors.Index`

Non-goals for milestone 1:

- Wrapping tensors, contractions, TT/TCI, etc.
- Packaging and distributing binaries (no JLL, no BinaryBuilder).
- Supporting QN/symmetry indices (start with `NoSymmSpace` / `Index{Int}` equivalent).

## Repository layout (chosen)

Inside the `tensor4all-rs` repository (keep the Julia wrapper co-located with the Rust ABI for rapid co-evolution):

- `Tensor4all.jl/` (Julia package; brand name is **tensor4all**)
  - `Project.toml`
  - `src/Tensor4all.jl`
  - `src/C_API.jl` (ccall declarations + dlopen logic)
  - `deps/build.jl` (local build: `cargo build --release`, copy dylib into `deps/`)
  - `deps/` (built artifacts, not committed)
  - `ext/Tensor4allITensorsExt.jl` (ITensors conversions; loaded only when ITensors is present)
  - `test/runtests.jl`
- Add a new Rust crate for the C ABI:
  - `tensor4all-capi/` (new workspace member)

Rationale: This mirrors the development workflow used by `extern/SparseIR.jl`:
Julia’s `deps/build.jl` builds a local Rust library and copies `lib*.{dylib,so,dll}` into `deps/`,
then Julia loads that path via `Libdl`.

## Rust side: C-API crate (`tensor4all-capi`)

### Reference implementation: `extern/sparse-ir-rs/sparse-ir-capi`

Use `extern/sparse-ir-rs/sparse-ir-capi` as the primary reference for the ABI boundary design:

- Opaque C handles are `#[repr(C)] struct t { _private: *const c_void }`
- Ownership/lifecycle is handled by allocating internal Rust objects on the heap and exposing only an opaque pointer.
- Provide explicit `*_release`, `*_clone`, and `*_is_assigned` functions.
- Use `std::panic::catch_unwind` to prevent Rust panics from crossing the FFI boundary.
- Prefer simple `StatusCode` (C int) return values for error handling.

We will follow the same patterns for `tensor4all-capi` to keep the implementation consistent and auditable.

### Crate type

Create `tensor4all-capi` with:

- `crate-type = ["cdylib"]`
- dependency on `tensor4all-core-common` (and maybe `tensor4all` if we want a re-export surface)

### ABI strategy

Expose an **opaque pointer** type for `Index`:

- The C-API will allocate a `Box<DefaultIndex<DynId>>` and return it as `*mut c_void`.
- Julia stores it as `Ptr{Cvoid}` and calls a `*_release` function via a finalizer.

Follow the same operational style as `libsparseir` C-API:

- Explicit `*_clone` and `*_release`
- Functions returning status codes and/or null pointers
- No Rust panics crossing the FFI boundary (convert to error codes)

### Minimal API (milestone 1)

Naming sketch (exact names TBD):

- Lifecycle:
  - `t4a_index_new(dim: usize) -> *mut t4a_index`
  - `t4a_index_clone(src: *const t4a_index) -> *mut t4a_index`
  - `t4a_index_release(ptr: *mut t4a_index) -> void`

- Accessors:
  - `t4a_index_dim(ptr, out_dim: *mut usize) -> int32`
  - `t4a_index_id_u128(ptr, out_hi: *mut u64, out_lo: *mut u64) -> int32`
  - `t4a_index_get_tags(ptr, buf: *mut u8, buf_len: usize, out_len: *mut usize) -> int32`
    - If `buf == NULL`, write required length to `out_len`.
  - Optional:
    - `t4a_index_add_tag(ptr, cstr: *const c_char) -> int32`
    - `t4a_index_set_tags_csv(ptr, cstr: *const c_char) -> int32`

Implementation notes (based on `sparse-ir-capi`):

- Provide `t4a_index_release/clone/is_assigned` with a small macro (like `impl_opaque_type_common!`).
- Convert Rust-side tag parsing errors (`TagSetError`) into a single negative status code.
- Never allocate and return Rust-owned strings directly across the boundary; use a caller-provided buffer pattern.

### Tag encoding

Rust `DefaultTagSet` is a set of up to 4 tags (max 16 chars each).
Julia/ITensors tags are typically strings. We will standardize on:

- C-API returns tags as a comma-separated UTF-8 string (e.g. `"Site,Link"`).
- C-API accepts tags as:
  - a single tag (`add_tag`)
  - or a comma-separated list (`set_tags_csv`)

If ITensors tags exceed the Rust capacity, **conversion must error** (strict).

## Julia side: `Tensor4all.jl` package

### Loading the native library

Implement `deps/build.jl` in the same spirit as `SparseIR.jl/deps/build.jl`:

- build the Rust workspace library with:
  - `cargo build --release -p tensor4all-capi`
- locate the output:
  - `tensor4all-rs/target/release/libtensor4all_capi.$(Libdl.dlext)`
- copy into:
  - `Tensor4all.jl/deps/libtensor4all_capi.$(Libdl.dlext)`

At runtime:

- `src/C_API.jl` resolves the library path in `deps/` and calls `ccall((:symbol, libpath), ...)`.
- We do not rely on `*_jll`.

### Julia wrapper types

Model after `SparseIR.C_API`:

- `struct t4a_index; _private::Ptr{Cvoid}; end` (opaque C object handle)
- `struct Index; ptr::Ptr{Cvoid}; end` (public Julia wrapper)
  - `finalizer` calls `t4a_index_release`.
  - Define `Base.unsafe_convert`/`Base.cconvert` as needed.

High-level constructors and accessors:

- `Index(dim::Integer; tags::AbstractString="")`
- `dim(i::Index)::Int`
- `tags(i::Index)::String`
- `id(i::Index)::UInt128` (or `(UInt64, UInt64)` as a stable internal representation)

### Error handling

Prefer “status code + Julia exception”:

- C-API returns an `int32` status code (`0` success, negative for errors).
- Julia checks and throws `ErrorException` or a small custom `Tensor4allError`.

## ITensors weak dependency + conversions

### Project.toml

Use Julia’s weak dependency mechanism:

- Add `ITensors` under `[weakdeps]`
- Add an extension under `[extensions]`:
  - `Tensor4allITensorsExt = ["ITensors"]`

### Conversion direction: `Tensor4all.Index` → `ITensors.Index`

We need a policy to map:

- dim: straightforward
- tags: straightforward (comma-separated ↔ ITensors tag string)
- id: **Rust uses UInt128**, ITensors uses **UInt64**

Chosen policy (recommended for interop):

- Use the lower 64 bits of the Rust id as ITensors id.
- Optionally embed the upper 64 bits in a reserved tag, e.g. `T4AIDHI=0x...`, for debugging.

This yields deterministic matching in ITensors for indices created in Rust/Julia.

### Conversion direction: `ITensors.Index` → `Tensor4all.Index`

Chosen policy:

- If the reserved tag with the upper bits exists, reconstruct the full UInt128.
- Otherwise: set upper bits to 0 and use ITensors UInt64 as the lower half.

## Testing

### Rust

- Add unit tests in `tensor4all-capi` that:
  - create/release index
  - verify dimension and tag roundtrip
  - verify id getters

### Julia

Use `Pkg.test()` for the Julia wrapper package.

Minimal Julia tests:

- `Index(2)` constructs, `dim` returns 2
- tags set/get roundtrip
- `id` returns `UInt128` and is stable per instance
- When ITensors is available:
  - `it = ITensors.Index(idx)` (or `convert`)
  - `idx2 = Tensor4all.Index(it)`
  - `dim` and tags are consistent

## CI considerations (later)

- For a Rust repo, CI would need Rust + Julia if we test Julia wrapper here.
- Alternatively, we can keep the Julia wrapper in a separate Julia repo and depend on a local checkout during development.

This plan focuses on local development first.

## Milestone 2: `TensorDynLen<DynId, NoSymmSpace>` ↔ `ITensors.ITensor`

### Scope (initial)

Implement bidirectional conversion between:

- Rust-backed `TensorDynLen<DynId, NoSymmSpace>` (exposed to Julia via C-API)
- `ITensors.ITensor` (Julia object)

Initial supported subset (to keep the surface minimal and robust):

- Symmetry: **NoSymmSpace only**
- Storage: start with **DenseF64**; optionally add **DenseC64** next
- Rank: any (dynamic rank)
- Tags: strict overflow error (same policy as `Index`)
- Anything not supported must return an error (no implicit densification unless explicitly allowed)

### Why this needs extra C-API

`TensorDynLen` in Rust stores:

- `indices: Vec<Index<Id, Symm>>`
- `dims: Vec<usize>`
- `storage: Arc<Storage>` where `Storage` can be Dense/Diag and f64/Complex64

To construct an `ITensors.ITensor` in Julia (and vice versa), Julia needs access to:

- the ordered list of indices (convertible to `ITensors.Index`)
- the dimension vector
- the storage kind and the contiguous raw data buffer in a well-defined memory order

Therefore milestone 2 requires extending the Rust C-API beyond indices.

### Proposed tensor C-API surface (design patterned after `sparse-ir-capi`)

Opaque handle:

- `t4a_tensor` as `#[repr(C)] struct { _private: *const c_void }`
- lifecycle functions `t4a_tensor_release/clone/is_assigned` (macro-generated)

Accessors (read-only first):

- `t4a_tensor_get_rank(ptr, out_rank)`
- `t4a_tensor_get_dims(ptr, out_dims /* len=rank */)`
- `t4a_tensor_get_indices(ptr, out_index_ptrs /* len=rank */)`
  - returns cloned `t4a_index` handles in the same order as the tensor
- `t4a_tensor_get_storage_kind(ptr, out_kind)` where kind is an enum:
  - DenseF64, DenseC64, DiagF64, DiagC64
- `t4a_tensor_get_data_f64(ptr, buf, buf_len, out_len)`
  - caller-buffer pattern; data is always in **row-major** order
- `t4a_tensor_get_data_c64(ptr, buf, buf_len, out_len)` with `Complex64 { re, im }`

Constructors (needed for ITensor → Rust direction):

- `t4a_tensor_new_dense_f64(rank, index_ptrs, dims, data, status) -> *mut t4a_tensor`
- `t4a_tensor_new_dense_c64(...)`

### Memory order convention

Rust side always uses **row-major** order (consistent with `ndarray`/`mdarray`).
Julia side handles conversion to/from column-major when crossing the FFI boundary.

Rationale:
- Simplifies the C-API (no `order` parameter needed)
- Rust internals remain consistent with standard Rust array libraries
- Conversion burden is placed on Julia, which has efficient `permutedims`

### Memory contiguity guarantee

**Critical**: Before passing any array data across FFI, ensure contiguous memory layout.

ITensor's internal storage may be a view or have non-standard strides after operations
like `permutedims`. Always materialize to a contiguous `Array` before FFI calls.

This is a common FFI bug pattern. See `SparseIR.jl` for reference implementation.

#### Helper functions

```julia
# Check if array is column-major contiguous (from SparseIR.jl pattern)
function _is_column_major_contiguous(A::AbstractArray)
    strides(A) == cumprod((1, size(A)...)[1:(end-1)])
end

# Ensure array is contiguous, copying if necessary
function _ensure_contiguous(A::AbstractArray{T,N}) where {T,N}
    if _is_column_major_contiguous(A)
        return A
    end
    # Materialize to contiguous Array
    return Array{T,N}(A)
end
```

#### Common pitfalls

1. `permutedims` often returns a **view**, not a copy
2. `reshape` may return a view with non-standard strides
3. ITensor slicing operations may produce non-contiguous views
4. Always call `_ensure_contiguous` immediately before `ccall`

### Julia conversion design

#### Rust tensor → `ITensors.ITensor`

1. Call tensor C-API to obtain:
   - `rank`, `dims`, `indices`
   - `storage_kind`
   - raw data in **row-major** order
2. Convert each `Tensor4all.Index` → `ITensors.Index` using the already-defined id/tag policy.
3. Convert row-major → column-major:
   ```julia
   # Data from Rust is row-major, need to convert to column-major for Julia
   # For dims (d1, d2, d3): row-major data has dims reversed in column-major view
   arr = reshape(rust_data, reverse(dims)...)
   arr = permutedims(arr, reverse(1:length(dims)))
   ```
4. Construct `ITensors.ITensor(arr, inds...)`.

#### `ITensors.ITensor` → Rust tensor

1. Extract indices and dims from `ITensors.ITensor`.
2. Convert each `ITensors.Index` → `Tensor4all.Index`:
   - strict tag overflow error
   - id mapping policy (lower 64 bits, optional upper bits tag)
3. Extract dense data with contiguity guarantee:
   - require that the ITensor is dense (error if not; no implicit densification)
   ```julia
   # Get array in canonical index order
   arr = Array(itensor, inds...)

   # CRITICAL: ensure contiguous before any further operations
   arr = _ensure_contiguous(arr)

   # Convert column-major → row-major for Rust
   arr = permutedims(arr, reverse(1:ndims(arr)))

   # CRITICAL: permutedims may return a view, ensure contiguous again!
   arr = _ensure_contiguous(arr)

   # Now safe to pass to Rust
   rust_data = vec(arr)
   ```
4. Call `t4a_tensor_new_dense_*` with row-major data.

### Open question (for milestone 2)

Decision: for `ITensor → Rust` conversion, **error if the ITensor is not dense**.

Rationale: avoid hidden performance cliffs and unintended allocations. If densification is desired,
it should be an explicit user action (and can be added as an opt-in helper later).

## Questions / decisions needed

All initial design decisions are now fixed:

- **Location**: `tensor4all-rs/Tensor4all.jl/`
- **Brand name**: tensor4all (lowercase in docs), module: `Tensor4all`
- **Tags**: strict error on overflow
- **ID mapping**: lower 64 bits for ITensors id; optional upper bits tag; inverse reconstruct when possible
- **Julia version**: Julia ≥ 1.9


