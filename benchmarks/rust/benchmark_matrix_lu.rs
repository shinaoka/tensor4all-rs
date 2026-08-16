use std::time::{Duration, Instant};

use tensor4all_core::{rrlu, rrlu_mut, RrLUOptions};
use tensor4all_tensorbackend::Matrix;

#[derive(Clone, Copy, Default)]
struct MatrixLuTiming {
    inplace: Duration,
    borrowed: Duration,
    rank: usize,
    last_error: f64,
    checksum: f64,
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

fn parse_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_env_usize_list(name: &str, default: Vec<usize>) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|item| item.parse::<usize>().unwrap())
                .collect::<Vec<_>>()
        })
        .unwrap_or(default)
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

fn timed_hilbert_matrix_lu_once(size: usize, left_orthogonal: bool) -> MatrixLuTiming {
    let matrix = hilbert_matrix(size);
    let options = RrLUOptions {
        max_bond_dim: usize::MAX,
        rel_tol: 0.0,
        abs_tol: 1.0e-10,
        left_orthogonal,
    };

    let mut matrix_for_inplace = matrix.clone();
    let start = Instant::now();
    let lu_inplace = rrlu_mut(&mut matrix_for_inplace, Some(options.clone())).unwrap();
    let inplace = start.elapsed();

    let start = Instant::now();
    let lu_borrowed = rrlu(&matrix, Some(options)).unwrap();
    let borrowed = start.elapsed();

    let left = lu_inplace.left(false);
    let right = lu_inplace.right(false);
    let checksum = if left.nrows() > 0 && left.ncols() > 0 {
        left[[0, 0]].abs()
    } else {
        0.0
    } + if right.nrows() > 0 && right.ncols() > 0 {
        right[[0, 0]].abs()
    } else {
        0.0
    };

    MatrixLuTiming {
        inplace,
        borrowed,
        rank: lu_borrowed.npivots(),
        last_error: lu_borrowed.last_pivot_error(),
        checksum,
    }
}

fn main() {
    let repeats = parse_env_usize("T4A_MATRIX_LU_REPEATS", 20);
    let sizes = parse_env_usize_list("T4A_MATRIX_LU_SIZES", vec![16, 32, 64, 128]);

    println!(
        "impl,matrix,size,repeats,left_orthogonal,inplace_ms,borrowed_ms,rank,last_error,checksum"
    );
    for size in sizes {
        for left_orthogonal in [true, false] {
            let runs = (0..repeats)
                .map(|_| timed_hilbert_matrix_lu_once(size, left_orthogonal))
                .collect::<Vec<_>>();
            let first = runs[0];
            println!(
                "rust,hilbert,{size},{repeats},{left_orthogonal},{:.6},{:.6},{},{:.6e},{:.6e}",
                timing_median(runs.iter().map(|run| timing_ms(run.inplace)).collect()),
                timing_median(runs.iter().map(|run| timing_ms(run.borrowed)).collect()),
                first.rank,
                first.last_error,
                first.checksum,
            );
        }
    }
}
