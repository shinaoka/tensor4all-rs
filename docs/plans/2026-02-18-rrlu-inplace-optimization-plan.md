# rrlu_mut Performance Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate allocation-heavy performance bugs in `rrlu_mut` and add criterion benchmarks comparing against faer.

**Architecture:** Make `swap_rows`/`swap_cols` truly in-place (zero allocation), change `submatrix_argmax` to accept `Range<usize>` instead of `&[usize]`, and add a benchmark suite comparing our implementation against faer's full-pivoting LU.

**Tech Stack:** Rust, criterion (benchmarks), faer (benchmark comparison), rand_chacha (deterministic RNG for benchmarks)

---

### Task 1: Make swap_rows/swap_cols truly in-place

**Files:**
- Modify: `crates/matrixci/src/util.rs:157-182`

**Step 1: Rewrite swap_rows to operate in-place**

Replace lines 156-168 in `crates/matrixci/src/util.rs`:

```rust
/// Swap two rows in a matrix in-place
pub fn swap_rows<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    for j in 0..m.ncols {
        let idx_a = a * m.ncols + j;
        let idx_b = b * m.ncols + j;
        m.data.swap(idx_a, idx_b);
    }
}
```

**Step 2: Rewrite swap_cols to operate in-place**

Replace lines 170-182 in `crates/matrixci/src/util.rs`:

```rust
/// Swap two columns in a matrix in-place
pub fn swap_cols<T>(m: &mut Matrix<T>, a: usize, b: usize) {
    if a == b {
        return;
    }
    for i in 0..m.nrows {
        let idx_a = i * m.ncols + a;
        let idx_b = i * m.ncols + b;
        m.data.swap(idx_a, idx_b);
    }
}
```

**Step 3: Update callers in matrixlu.rs**

In `crates/matrixci/src/matrixlu.rs`, change lines 217-224 from:

```rust
        if pivot_row != k {
            *a = swap_rows(a, k, pivot_row);
            lu.row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            *a = swap_cols(a, k, pivot_col);
            lu.col_permutation.swap(k, pivot_col);
        }
```

to:

```rust
        if pivot_row != k {
            swap_rows(a, k, pivot_row);
            lu.row_permutation.swap(k, pivot_row);
        }
        if pivot_col != k {
            swap_cols(a, k, pivot_col);
            lu.col_permutation.swap(k, pivot_col);
        }
```

**Step 4: Run existing tests to verify correctness**

Run: `cargo nextest run --release -p matrixci`
Expected: All 14 tests pass with identical results.

**Step 5: Commit**

```bash
git add crates/matrixci/src/util.rs crates/matrixci/src/matrixlu.rs
git commit -m "perf(matrixci): make swap_rows/swap_cols truly in-place

Eliminates full matrix clones on every row/column swap.
Previously each swap allocated O(m*n), now zero allocation.

Part of #229"
```

---

### Task 2: Change submatrix_argmax to accept Range<usize>

**Files:**
- Modify: `crates/matrixci/src/util.rs:291-315` (function definition + test)
- Modify: `crates/matrixci/src/matrixlu.rs:186-195` (caller)
- Modify: `crates/matrixci/src/matrixci.rs:310-312` (caller)
- Modify: `crates/matrixci/src/matrixaca.rs:186-188` (caller)

**Step 1: Rewrite submatrix_argmax to accept Range<usize>**

Replace lines 290-315 in `crates/matrixci/src/util.rs`:

```rust
/// Find the position of maximum absolute value in a submatrix defined by row/column ranges
pub fn submatrix_argmax<T: Scalar>(
    a: &Matrix<T>,
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> (usize, usize, T) {
    assert!(!rows.is_empty(), "rows must not be empty");
    assert!(!cols.is_empty(), "cols must not be empty");

    let mut max_val: f64 = a[[rows.start, cols.start]].abs_sq();
    let mut max_row = rows.start;
    let mut max_col = cols.start;

    for r in rows {
        for c in cols.clone() {
            let val: f64 = a[[r, c]].abs_sq();
            if val > max_val {
                max_val = val;
                max_row = r;
                max_col = c;
            }
        }
    }

    (max_row, max_col, a[[max_row, max_col]])
}
```

Note: Add `use std::ops::Range;` to the imports at the top of `util.rs` (line 8), or use fully qualified `std::ops::Range<usize>` in the signature. Since `std::ops::Range` is not currently imported, use the fully qualified path to minimize diff.

**Step 2: Update caller in matrixlu.rs**

In `crates/matrixci/src/matrixlu.rs`, change lines 186-195 from:

```rust
        let k = lu.n_pivot;
        let rows: Vec<usize> = (k..nr).collect();
        let cols: Vec<usize> = (k..nc).collect();

        if rows.is_empty() || cols.is_empty() {
            break;
        }

        // Find pivot with maximum absolute value in submatrix
        let (pivot_row, pivot_col, pivot_val) = submatrix_argmax(a, &rows, &cols);
```

to:

```rust
        let k = lu.n_pivot;

        if k >= nr || k >= nc {
            break;
        }

        // Find pivot with maximum absolute value in submatrix
        let (pivot_row, pivot_col, pivot_val) = submatrix_argmax(a, k..nr, k..nc);
```

**Step 3: Update caller in matrixci.rs**

In `crates/matrixci/src/matrixci.rs`, change lines 309-312 from:

```rust
    // Find initial pivot (maximum absolute value)
    let rows: Vec<usize> = (0..nrows(a)).collect();
    let cols: Vec<usize> = (0..ncols(a)).collect();
    let (first_i, first_j, _) = submatrix_argmax(a, &rows, &cols);
```

to:

```rust
    // Find initial pivot (maximum absolute value)
    let (first_i, first_j, _) = submatrix_argmax(a, 0..nrows(a), 0..ncols(a));
```

**Step 4: Update caller in matrixaca.rs**

In `crates/matrixci/src/matrixaca.rs`, change lines 185-188 from:

```rust
            // Find global maximum
            let rows: Vec<usize> = (0..self.nrows()).collect();
            let cols: Vec<usize> = (0..self.ncols()).collect();
            let (i, j, _) = submatrix_argmax(a, &rows, &cols);
```

to:

```rust
            // Find global maximum
            let (i, j, _) = submatrix_argmax(a, 0..self.nrows(), 0..self.ncols());
```

**Step 5: Update test in util.rs**

In `crates/matrixci/src/util.rs`, change the test (lines 396-407) from:

```rust
    #[test]
    fn test_submatrix_argmax() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let rows: Vec<usize> = vec![0, 1, 2];
        let cols: Vec<usize> = vec![0, 1, 2];
        let (r, c, _) = submatrix_argmax(&m, &rows, &cols);
        assert_eq!((r, c), (2, 2));
    }
```

to:

```rust
    #[test]
    fn test_submatrix_argmax() {
        let m = from_vec2d(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ]);

        let (r, c, _) = submatrix_argmax(&m, 0..3, 0..3);
        assert_eq!((r, c), (2, 2));
    }
```

**Step 6: Run tests**

Run: `cargo nextest run --release -p matrixci`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add crates/matrixci/src/util.rs crates/matrixci/src/matrixlu.rs crates/matrixci/src/matrixci.rs crates/matrixci/src/matrixaca.rs
git commit -m "perf(matrixci): use Range<usize> in submatrix_argmax

Eliminates Vec<usize> allocation per pivot iteration.

Part of #229"
```

---

### Task 3: Add criterion benchmarks

**Files:**
- Modify: `crates/matrixci/Cargo.toml`
- Create: `crates/matrixci/benches/rrlu_bench.rs`

**Step 1: Update Cargo.toml with dev-dependencies**

Add to `crates/matrixci/Cargo.toml`:

```toml
[dev-dependencies]
approx.workspace = true
criterion.workspace = true
faer.workspace = true
rand_chacha.workspace = true

[[bench]]
name = "rrlu_bench"
harness = false
```

**Step 2: Create benchmark file**

Create `crates/matrixci/benches/rrlu_bench.rs`:

```rust
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::prelude::*;
use matrixci::util::{from_vec2d, Matrix};
use matrixci::{rrlu_mut, RrLUOptions};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Generate a random f64 matrix as matrixci::Matrix
fn random_matrix(n: usize, m: usize, seed: u64) -> Matrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..m).map(|_| rng.random::<f64>()).collect())
        .collect();
    from_vec2d(data)
}

/// Generate a random faer Mat<f64>
fn random_faer_matrix(n: usize, m: usize, seed: u64) -> Mat<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    Mat::from_fn(n, m, |_, _| rng.random::<f64>())
}

fn bench_rrlu(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrlu_full_rank");

    for &size in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("rrlu_mut", size), &size, |b, &n| {
            b.iter_batched(
                || random_matrix(n, n, 42),
                |mut m| {
                    let opts = RrLUOptions {
                        max_rank: n,
                        rel_tol: 0.0,
                        abs_tol: 0.0,
                        ..Default::default()
                    };
                    rrlu_mut(&mut m, Some(opts)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("faer_lu_fullpiv", size), &size, |b, &n| {
            b.iter_batched(
                || random_faer_matrix(n, n, 42),
                |m| {
                    m.full_piv_lu();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(benches, bench_rrlu);
criterion_main!(benches);
```

**Step 3: Verify benchmark compiles and runs**

Run: `cargo bench -p matrixci --bench rrlu_bench -- --sample-size 10`
Expected: Benchmarks run successfully, output shows timing for each size.

**Step 4: Commit**

```bash
git add crates/matrixci/Cargo.toml crates/matrixci/benches/rrlu_bench.rs
git commit -m "bench(matrixci): add criterion benchmarks for rrlu vs faer

Compares rrlu_mut against faer's full-pivoting LU at
matrix sizes 10, 50, 100, 500, 1000.

Part of #229"
```

---

### Task 4: Run full test suite and format

**Files:** None (verification only)

**Step 1: Format all code**

Run: `cargo fmt --all`

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: No new warnings.

**Step 3: Run full test suite**

Run: `cargo nextest run --release --workspace`
Expected: All tests pass.

**Step 4: Run benchmarks and record results**

Run: `cargo bench -p matrixci --bench rrlu_bench -- --sample-size 10`
Expected: Results printed showing relative performance of rrlu_mut vs faer.

**Step 5: Commit any formatting fixes if needed**

```bash
git add -A && git commit -m "style: format code"
```
