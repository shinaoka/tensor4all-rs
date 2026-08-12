# Changelog

## Unreleased — 0-indexed grid indices (breaking)

Adopts the 0-indexed convention of `quanticsgrids` 0.2.0 (tensor4all/tensor4all-rs#584):
the five `±1` conversions at the TCI boundary are gone and the public surface is
0-indexed.

**Porting note for QuanticsTCI.jl scripts: subtract 1 from grid indices at the
call boundary.**

- Interpolation callbacks (`quanticscrossinterpolate_discrete`) now receive
  0-indexed grid indices as `&[usize]` instead of 1-indexed `&[i64]`.
- `QuanticsTensorCI2::evaluate` takes 0-indexed grid indices (`&[usize]`).
- `initial_pivots` are 0-indexed `Vec<Vec<usize>>`.
- `cachedata()` keys are 0-indexed quantics indices (`Vec<usize>`).
