# Remaining Issues: TCI Rust Port

This document tracks remaining work for the Rust port of T4AMatrixCI.jl, T4ATensorTrain.jl, and T4ATensorCI.jl.

## Current Status

| Crate | Tests | Status |
|-------|-------|--------|
| tensor4all-matrixci | 19 | Complete |
| tensor4all-tensortrain | 20 | Core complete |
| tensor4all-tensorci | 21 | Core complete |

## tensor4all-tensortrain

### Medium Priority

- [ ] **TTCache** - Caching structure for repeated tensor train contractions
  - Reference: `T4ATensorTrain.jl/src/cache.jl`
  - Used to avoid redundant computations in TCI algorithms

### Low Priority

- [ ] **VidalTensorTrain** - Vidal canonical form with diagonal bond matrices (Γ-Λ form)
  - Reference: `T4ATensorTrain.jl/src/vidal.jl`
  - Useful for certain algorithms but not essential for basic TCI

- [ ] **SiteTensorTrain** - Site-canonical (mixed-canonical) form
  - Reference: `T4ATensorTrain.jl/src/sitetensortrain.jl`
  - Orthogonality center at a specific site

- [ ] **InverseTensorTrain** - Inverse representation for efficient division
  - Reference: `T4ATensorTrain.jl/src/inverse.jl`

- [ ] **4-leg tensor support (Tensor4)** - For MPO-MPS contraction
  - Julia uses `TensorTrain{T,4}` for MPO operations
  - Current Rust implementation uses 3-leg tensors (MPS only)
  - Would enable `contract_naive` and `contract_zipup` for MPO-MPS

## tensor4all-tensorci

### Medium Priority

- [ ] **GlobalPivotFinder** - Global pivot search strategies
  - Reference: `T4ATensorCI.jl/src/globalpivot.jl`
  - Random search and rook pivoting for finding good global pivots
  - Important for robustness of TCI2

- [ ] **Integration utilities** - Weighted sums and integration
  - Reference: `T4ATensorCI.jl/src/integration.jl`
  - `integrate(tci, weights)` for numerical integration
  - Useful for applications in physics/chemistry

### Low Priority

- [ ] **More sweep strategies** - Additional pivot selection heuristics
  - Full matrix search vs. partial (rook) search
  - Adaptive tolerance strategies

## Optional Enhancements

- [ ] **Complex number testing** - Types are generic but not tested with `Complex64`
- [ ] **SVD compression** - Currently falls back to LU; true SVD would be more accurate
  - Requires LAPACK bindings or pure-Rust SVD implementation
- [ ] **Benchmarks** - Performance comparison with Julia implementation
- [ ] **Documentation** - Rustdoc examples and usage guides

## Notes

- The core TCI functionality (TensorCI1, TensorCI2) is complete and usable
- TensorCI2 uses explicit `batched_f: Option<B>` parameter instead of inheritance
- All 60 tests pass, clippy passes with no warnings
