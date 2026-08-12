# Conventions

This page collects important conventions that apply across the entire tensor4all-rs codebase.

## Dense Layout (Column-Major)

tensor4all-rs uses **column-major** (Fortran order) dense linearization internally. Flat dense
buffers, `reshape`/`flatten` semantics, the C API, and the HDF5 layer are all defined in terms
of column-major ordering.

This matches Julia, ITensors.jl, and tenferro-rs. When exchanging dense data with NumPy, use
`order="F"` when you need explicit control over flattening or reshaping.

## Indexing

- Sites are **0-indexed** in Rust (unlike ITensors.jl, which is 1-indexed).
- **Exception**: `tensor4all-quanticstci` grid indices are **1-indexed**, following the Julia
  convention for compatibility with QuanticsTCI.jl.

## Truncation Tolerance

tensor4all-rs uses `rtol` (relative tolerance). ITensors.jl uses `cutoff`. The conversion is:

```text
rtol = sqrt(cutoff)
```

| Library | Parameter | Conversion |
|---------|-----------|------------|
| tensor4all-rs | `rtol` | — |
| ITensors.jl | `cutoff` | `rtol = √cutoff` |

**Example**: ITensors.jl `cutoff=1e-10` corresponds to `rtol=1e-5` in tensor4all-rs.

## Bond-Dimension Cap

tensor4all-rs uses one spelling and one type for the bond-dimension cap across
all crates: **`max_bond_dim: Option<usize>`** (`None` = unlimited). No
`usize::MAX` sentinel is used.

| Library | Parameter | tensor4all-rs |
|---------|-----------|---------------|
| tensor4all-rs | `max_bond_dim: Option<usize>` | — |
| ITensors.jl | `maxdim` | `max_bond_dim: Some(d)` |
| QuanticsTCI.jl / TCI | `maxbonddim` | `max_bond_dim: Some(d)` |
| (historical) | `max_rank` | `max_bond_dim: Option<usize>` |

`maxdim` is the closest ITensors.jl cousin of `max_bond_dim` (bond dimension
is the unambiguous tensor-network term; "rank" is overloaded in TCI
context).

## ITensors.jl Type Correspondence

| ITensors.jl | tensor4all-rs |
|-------------|---------------|
| `Index{Int}` | `Index<Id, NoSymmSpace>` |
| `ITensor` | `IdxTensor` |
| `Dense` | eager dense payload; `Storage` snapshot for `f64`/`Complex64` |
| `Diag` | compact `Storage` for `f64`/`Complex64`, eager diagonal payload for `f32`/`Complex32` |
| `A * B` | `a.contract(&b)` |

## Scalar Types

`IdxTensor` supports four scalar types:

- `f32` — single-precision real
- `f64` — double-precision real
- `Complex32` — single-precision complex
- `Complex64` — double-precision complex (from the `num-complex` crate)

Generic APIs handle all four types. Compact `Storage` snapshots are limited to
`f64`/`Complex64`; 32-bit tensors retain eager payloads rather than silently
promoting their values. Prefer generic code over scalar-specific variants
(`*_f64` / `*_c64`) in library and test code. The C API uses scalar-specific
names at the FFI boundary where generic dispatch is not available.
