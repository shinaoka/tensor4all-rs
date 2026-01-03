"""
    Tensor4allITensorsExt

Extension module providing bidirectional conversion between
Tensor4all.Index and ITensors.Index.

## ID Mapping Policy

Rust uses UInt128 IDs, ITensors uses UInt64:
- Tensor4all → ITensors: Use lower 64 bits as ITensors ID
- ITensors → Tensor4all: Set upper 64 bits to 0, use ITensors ID as lower 64 bits

This ensures deterministic matching for indices created in either system.
"""
module Tensor4allITensorsExt

using Tensor4all
using ITensors

# ============================================================================
# Tensor4all.Index → ITensors.Index
# ============================================================================

"""
    ITensors.Index(idx::Tensor4all.Index)

Convert a Tensor4all.Index to an ITensors.Index.

The lower 64 bits of the Tensor4all ID are used as the ITensors ID.
Tags are preserved.
"""
function ITensors.Index(idx::Tensor4all.Index)
    d = Tensor4all.dim(idx)
    t = Tensor4all.tags(idx)

    # Use lower 64 bits of the 128-bit ID
    id128 = Tensor4all.id(idx)
    id64 = UInt64(id128 & 0xFFFFFFFFFFFFFFFF)

    # Create ITensors.Index with explicit ID using full constructor
    # Index(id, space, dir, tags, plev)
    tagset = isempty(t) ? ITensors.TagSet("") : ITensors.TagSet(t)
    return ITensors.Index(id64, d, ITensors.Neither, tagset, 0)
end

# ============================================================================
# ITensors.Index → Tensor4all.Index
# ============================================================================

"""
    Tensor4all.Index(idx::ITensors.Index)

Convert an ITensors.Index to a Tensor4all.Index.

The ITensors ID (UInt64) becomes the lower 64 bits of the Tensor4all ID,
with upper 64 bits set to 0. Tags are preserved.

Note: Tags that exceed Rust limits (max 4 tags, max 16 chars each) will
cause an error.
"""
function Tensor4all.Index(idx::ITensors.Index)
    d = ITensors.dim(idx)

    # Get ITensors ID and extend to UInt128
    id64 = ITensors.id(idx)
    id128 = UInt128(id64)

    # Get tags as comma-separated string
    tag_set = ITensors.tags(idx)
    tags_str = _tags_to_string(tag_set)

    return Tensor4all.Index(d, id128; tags=tags_str)
end

"""
Convert ITensors TagSet to comma-separated string.
"""
function _tags_to_string(ts::ITensors.TagSet)
    n = length(ts)
    if n == 0
        return ""
    end

    tag_strings = String[]
    for i in 1:n
        push!(tag_strings, string(ts[i]))
    end
    return join(tag_strings, ",")
end

# ============================================================================
# Conversion functions
# ============================================================================

"""
    Base.convert(::Type{ITensors.Index}, idx::Tensor4all.Index)

Enable `convert(ITensors.Index, t4a_idx)`.
"""
Base.convert(::Type{ITensors.Index}, idx::Tensor4all.Index) = ITensors.Index(idx)

"""
    Base.convert(::Type{Tensor4all.Index}, idx::ITensors.Index)

Enable `convert(Tensor4all.Index, it_idx)`.
"""
Base.convert(::Type{Tensor4all.Index}, idx::ITensors.Index) = Tensor4all.Index(idx)

end # module
