# Design: rrlu_mut Performance Optimization

**Date:** 2026-02-18
**Issue:** [#229](https://github.com/tensor4all/tensor4all-rs/issues/229)

## Context

The `rrlu_mut` function in `matrixci` has significant performance issues:

1. **Row/column swaps clone the entire matrix** — O(m*n) allocation per swap, 2 swaps per pivot
2. **Vec allocation per pivot iteration** — `(k..nr).collect()` creates heap-allocated vectors each iteration
3. No SIMD, parallelism, or cache optimization (unlike faer)

## Approach: Incremental Optimization

Fix obvious performance bugs first, then benchmark against faer to decide if further optimization is needed.

## Phase 1: Fix Performance Bugs

### 1.1 In-place Row/Column Swaps

Change `swap_rows`/`swap_cols` in `util.rs` from cloning (`&Matrix<T> -> Matrix<T>`) to truly in-place (`&mut Matrix<T>`):

```rust
// Before: clones entire matrix
pub fn swap_rows<T: Clone + Zero>(m: &Matrix<T>, a: usize, b: usize) -> Matrix<T>

// After: zero allocation, element-wise swap
pub fn swap_rows<T>(m: &mut Matrix<T>, a: usize, b: usize)
```

Uses `Vec::swap` on the underlying data for each element in the row/column.

**Impact:** Eliminates 2 * rank matrix clones per decomposition. For 1000x1000 rank-100, saves ~200 full matrix copies.

### 1.2 Eliminate Vec Allocations in Pivot Search

Change `submatrix_argmax` to accept `Range<usize>` instead of `&[usize]`:

```rust
// Before
pub fn submatrix_argmax<T: Scalar>(a: &Matrix<T>, rows: &[usize], cols: &[usize]) -> (usize, usize, T)

// After
pub fn submatrix_argmax<T: Scalar>(a: &Matrix<T>, rows: Range<usize>, cols: Range<usize>) -> (usize, usize, T)
```

All callers (matrixlu.rs, matrixci.rs, matrixaca.rs) use contiguous ranges, so this is safe.

**Impact:** Eliminates 2 * rank heap allocations per decomposition.

## Phase 2: Benchmarks

Add criterion benchmarks comparing `rrlu_mut` vs faer's full-pivoting LU at sizes 10, 50, 100, 500, 1000.

- **faer added as dev-dependency only** (not a library dependency)
- Fair comparison: `rrlu_mut` with `max_rank = min(m,n)` (full decomposition)
- Fixed random seed for reproducibility

## Phase 3: Decide on Further Optimization

Based on benchmark data:
- If gap vs faer is acceptable (<3x), stop
- If significant, consider SIMD (via faer/pulp) or replacing core loop

## Files Changed

| File | Change |
|------|--------|
| `crates/matrixci/src/util.rs` | In-place swaps, Range-based argmax |
| `crates/matrixci/src/matrixlu.rs` | Update callers |
| `crates/matrixci/src/matrixci.rs` | Update submatrix_argmax call |
| `crates/matrixci/src/matrixaca.rs` | Update submatrix_argmax call |
| `crates/matrixci/Cargo.toml` | Add criterion + faer dev-deps |
| `crates/matrixci/benches/rrlu_bench.rs` | New benchmark file |

## Testing

All 14 existing tests must pass unchanged. No new correctness tests needed — changes are purely performance (same algorithm, same results).
