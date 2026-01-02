## Summary

This PR adds TagSet and SmallString implementations to tensor4all-rs, similar to ITensors.jl.

## Changes

- **SmallString<const MAX_LEN>**: Stack-allocated fixed-capacity string with explicit length
  - Uses `[char; MAX_LEN]` for storage
  - Supports UTF-8 arbitrary characters
  - Returns `Result::Err` on overflow

- **TagSet<const MAX_TAGS, const MAX_TAG_LEN>**: Fixed-capacity tag set with sorted order
  - Tags are always maintained in sorted order (like ITensors.jl)
  - Whitespace is ignored during parsing
  - Binary search for fast tag lookup

- **Index integration**: Added TagSet support to Index
  - Equality comparison uses `id` only (tags are ignored)
  - Hash implementation uses `id` only
  - Added `Copy` trait when components are `Copy`

## Testing

- All 21 tests pass
- Comprehensive test coverage for SmallString, TagSet, and Index integration
- Unicode character support tested

## Default Types

- `Tag = SmallString<16>` (matching ITensors.jl)
- `DefaultTagSet = TagSet<4, 16>` (matching ITensors.jl)
- `DefaultIndex<Id, Symm> = Index<Id, Symm, 4, 16>`

