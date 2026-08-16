use super::*;
use crate::rrlu;
use std::time::{Duration, Instant};
use tensor4all_tensorbackend::{from_vec2d, mat_mul};

#[test]
fn test_matrixluci_from_matrix() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.nrows(), 3);
    assert_eq!(luci.ncols(), 3);
    assert_eq!(luci.rank(), 3);
}

#[test]
fn matrix_luci_factors_from_matrix_owned_matches_borrowed() {
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 10.0],
    ]);

    let borrowed = matrix_luci_factors_from_matrix(&m, None).unwrap();
    let owned = matrix_luci_factors_from_matrix_owned(m, None).unwrap();

    assert_eq!(owned.rank, borrowed.rank);
    assert_eq!(owned.row_indices, borrowed.row_indices);
    assert_eq!(owned.col_indices, borrowed.col_indices);
    assert_eq!(
        owned.left.as_col_major_slice(),
        borrowed.left.as_col_major_slice()
    );
    assert_eq!(
        owned.right.as_col_major_slice(),
        borrowed.right.as_col_major_slice()
    );
}

#[test]
fn test_matrixluci_reconstruct() {
    let m = from_vec2d(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    let approx = luci.to_matrix();

    for i in 0..2 {
        for j in 0..2 {
            let diff = (m[[i, j]] - approx[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): {}",
                i,
                j,
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank2_iplusj_left_orthogonal() {
    // Pi matrix for f(i,j) = i + j on 4x4 grid
    let m = from_vec2d(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let opts = RrLUOptions {
        left_orthogonal: true,
        ..Default::default()
    };
    let luci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();
    assert_eq!(luci.rank(), 2);

    // Check left() * right() = Pi
    let left = luci.left();
    let right = luci.right();
    let reconstructed = mat_mul(&left, &right).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): expected {} got {} (diff {})",
                i,
                j,
                m[[i, j]],
                reconstructed[[i, j]],
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank2_iplusj_right_orthogonal() {
    let m = from_vec2d(vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0, 4.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![3.0, 4.0, 5.0, 6.0],
    ]);

    let opts = RrLUOptions {
        left_orthogonal: false,
        ..Default::default()
    };
    let luci = MatrixLUCI::from_matrix(&m, Some(opts)).unwrap();
    assert_eq!(luci.rank(), 2);

    let reconstructed = mat_mul(&luci.left(), &luci.right()).unwrap();
    for i in 0..4 {
        for j in 0..4 {
            let diff = (m[[i, j]] - reconstructed[[i, j]]).abs();
            assert!(
                diff < 1e-10,
                "Reconstruction error at ({}, {}): expected {} got {} (diff {})",
                i,
                j,
                m[[i, j]],
                reconstructed[[i, j]],
                diff
            );
        }
    }
}

#[test]
fn test_matrixluci_rank_deficient() {
    // Rank-1 matrix
    let m = from_vec2d(vec![
        vec![1.0, 2.0, 3.0],
        vec![2.0, 4.0, 6.0],
        vec![3.0, 6.0, 9.0],
    ]);

    let luci = MatrixLUCI::from_matrix(&m, None).unwrap();
    assert_eq!(luci.rank(), 1);
}

#[derive(Clone, Copy, Default)]
struct MatrixLuciHilbertTiming {
    selection: Duration,
    gather: Duration,
    left_factor: Duration,
    right_factor: Duration,
    rank: usize,
    last_error: f64,
    checksum: f64,
}

impl MatrixLuciHilbertTiming {
    fn total(self) -> Duration {
        self.selection + self.gather + self.left_factor + self.right_factor
    }
}

fn hilbert_matrix(size: usize) -> Matrix<f64> {
    let mut data = vec![0.0; size * size];
    for col in 0..size {
        for row in 0..size {
            data[row + size * col] = 1.0 / ((row + col + 1) as f64);
        }
    }
    Matrix::from_col_major_vec(size, size, data)
}

fn timed_hilbert_matrix_luci_once(size: usize, left_orthogonal: bool) -> MatrixLuciHilbertTiming {
    let matrix = hilbert_matrix(size);
    let options = RrLUOptions {
        max_bond_dim: usize::MAX,
        rel_tol: 0.0,
        abs_tol: 1.0e-10,
        left_orthogonal,
    };

    let mut timing = MatrixLuciHilbertTiming::default();

    let start = Instant::now();
    let lu = rrlu(&matrix, Some(options)).unwrap();
    timing.selection = start.elapsed();
    timing.rank = lu.npivots();
    timing.last_error = lu.pivot_errors().last().copied().unwrap_or(0.0);

    let start = Instant::now();
    let left = if left_orthogonal {
        rrlu_cols_times_pivot_solve(&lu).unwrap()
    } else {
        rrlu_colmatrix(&lu).unwrap()
    };
    timing.left_factor = start.elapsed();

    let start = Instant::now();
    let right = if left_orthogonal {
        rrlu_rowmatrix(&lu).unwrap()
    } else {
        rrlu_pivot_solve_times_rows(&lu).unwrap()
    };
    timing.right_factor = start.elapsed();

    let left_checksum = if left.nrows() > 0 && left.ncols() > 0 {
        left[[0, 0]].abs()
    } else {
        0.0
    };
    let right_checksum = if right.nrows() > 0 && right.ncols() > 0 {
        right[[0, 0]].abs()
    } else {
        0.0
    };
    timing.checksum = left_checksum + right_checksum;
    timing
}

fn timing_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn timing_median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[mid - 1] + values[mid])
    } else {
        values[mid]
    }
}

#[test]
#[ignore]
fn matrix_luci_hilbert_timing() {
    let repeats = std::env::var("T4A_MATRIX_LUCI_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(20);
    let sizes = std::env::var("T4A_MATRIX_LUCI_SIZES")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec![4, 8, 16, 32, 64]);

    println!(
        "impl,matrix,size,repeats,left_orthogonal,selection_ms,gather_ms,left_factor_ms,right_factor_ms,total_ms,rank,last_error,checksum"
    );
    for size in sizes {
        for left_orthogonal in [true, false] {
            let runs = (0..repeats)
                .map(|_| timed_hilbert_matrix_luci_once(size, left_orthogonal))
                .collect::<Vec<_>>();
            let first = runs[0];
            println!(
                "rust,hilbert,{size},{repeats},{left_orthogonal},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6e},{:.6e}",
                timing_median(runs.iter().map(|run| timing_ms(run.selection)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.gather)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.left_factor)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.right_factor)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.total())).collect()),
                first.rank,
                first.last_error,
                first.checksum,
            );
        }
    }
}
