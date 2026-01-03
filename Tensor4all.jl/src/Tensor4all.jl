"""
    Tensor4all

Julia wrapper for the tensor4all Rust library.

Provides tensor network types compatible with ITensors.jl, backed by efficient
Rust implementations.

# Basic Usage

```julia
using Tensor4all

# Create an index with dimension 5
i = Index(5)

# Create an index with tags
j = Index(3; tags="Site,n=1")

# Access properties
dim(i)   # dimension
id(i)    # unique ID (UInt128)
tags(i)  # tags as comma-separated string
```

# ITensors.jl Integration

When ITensors.jl is loaded, bidirectional conversion is available:

```julia
using Tensor4all
using ITensors

# Tensor4all.Index → ITensors.Index
t4a_idx = Tensor4all.Index(4; tags="Site")
it_idx = ITensors.Index(t4a_idx)

# ITensors.Index → Tensor4all.Index
it_idx2 = ITensors.Index(3, "Link")
t4a_idx2 = Tensor4all.Index(it_idx2)
```
"""
module Tensor4all

include("C_API.jl")

# Re-export public API
export Index, dim, tags, id, hastag

"""
    Index

A tensor index with dimension, ID, and tags.

Wraps a Rust `DefaultIndex<DynId, NoSymmSpace>` which corresponds to
ITensors.jl's `Index{Int}` (no quantum number symmetry).

# Constructors

- `Index(dim::Integer)` - Create index with dimension
- `Index(dim::Integer; tags::AbstractString)` - Create with tags
- `Index(dim::Integer, id::UInt128; tags::AbstractString)` - Create with specific ID

# Properties

- `dim(i::Index)` - Get the dimension
- `id(i::Index)` - Get the unique ID as UInt128
- `tags(i::Index)` - Get tags as comma-separated string
- `hastag(i::Index, tag::AbstractString)` - Check if index has a tag
"""
mutable struct Index
    ptr::Ptr{Cvoid}

    function Index(ptr::Ptr{Cvoid})
        if ptr == C_NULL
            error("Failed to create Index (null pointer from C API)")
        end
        idx = new(ptr)
        finalizer(idx) do x
            C_API.t4a_index_release(x.ptr)
        end
        return idx
    end
end

# Constructors
function Index(dim::Integer; tags::AbstractString="")
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    if isempty(tags)
        ptr = C_API.t4a_index_new(dim)
    else
        ptr = C_API.t4a_index_new_with_tags(dim, tags)
    end
    return Index(ptr)
end

function Index(dim::Integer, id::UInt128; tags::AbstractString="")
    dim > 0 || throw(ArgumentError("Index dimension must be positive, got $dim"))
    id_hi = UInt64(id >> 64)
    id_lo = UInt64(id & 0xFFFFFFFFFFFFFFFF)
    ptr = C_API.t4a_index_new_with_id(dim, id_hi, id_lo, tags)
    return Index(ptr)
end

# Accessors
"""
    dim(i::Index) -> Int

Get the dimension of an index.
"""
function dim(i::Index)
    d = Ref{Csize_t}(0)
    status = C_API.t4a_index_dim(i.ptr, d)
    C_API.check_status(status)
    return Int(d[])
end

"""
    id(i::Index) -> UInt128

Get the unique ID of an index.
"""
function id(i::Index)
    hi = Ref{UInt64}(0)
    lo = Ref{UInt64}(0)
    status = C_API.t4a_index_id_u128(i.ptr, hi, lo)
    C_API.check_status(status)
    return (UInt128(hi[]) << 64) | UInt128(lo[])
end

"""
    tags(i::Index) -> String

Get the tags of an index as a comma-separated string.
"""
function tags(i::Index)
    # First query required length
    len = Ref{Csize_t}(0)
    status = C_API.t4a_index_get_tags(i.ptr, C_NULL, 0, len)
    C_API.check_status(status)

    if len[] <= 1
        return ""
    end

    # Allocate buffer and get tags
    buf = Vector{UInt8}(undef, len[])
    status = C_API.t4a_index_get_tags(i.ptr, buf, len[], len)
    C_API.check_status(status)

    # Convert to string (excluding null terminator)
    return String(buf[1:end-1])
end

"""
    hastag(i::Index, tag::AbstractString) -> Bool

Check if an index has a specific tag.
"""
function hastag(i::Index, tag::AbstractString)
    result = C_API.t4a_index_has_tag(i.ptr, tag)
    result < 0 && error("Error checking tag")
    return result == 1
end

# Show method
function Base.show(io::IO, i::Index)
    d = dim(i)
    t = tags(i)
    id_val = id(i)
    id_short = string(id_val, base=16)[end-7:end]  # Last 8 hex digits
    if isempty(t)
        print(io, "(dim=$d|id=...$id_short)")
    else
        print(io, "(dim=$d|id=...$id_short|\"$t\")")
    end
end

function Base.show(io::IO, ::MIME"text/plain", i::Index)
    println(io, "Tensor4all.Index")
    println(io, "  dim: ", dim(i))
    println(io, "  id:  ", string(id(i), base=16))
    t = tags(i)
    if !isempty(t)
        println(io, "  tags: ", t)
    end
end

# Clone
function Base.copy(i::Index)
    ptr = C_API.t4a_index_clone(i.ptr)
    return Index(ptr)
end

# Equality based on ID (same as Rust side)
function Base.:(==)(i1::Index, i2::Index)
    return id(i1) == id(i2)
end

function Base.hash(i::Index, h::UInt)
    return hash(id(i), h)
end

end # module
