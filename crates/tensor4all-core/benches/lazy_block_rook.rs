use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix, RrLUOptions,
};
use tensor4all_tensorbackend::Matrix;

fn random_column_major(nrows: usize, ncols: usize, seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0.0; nrows * ncols];
    for col in 0..ncols {
        for row in 0..nrows {
            data[row + nrows * col] = rng.random::<f64>();
        }
    }
    data
}

fn fill_from_dense(data: &[f64], nrows: usize, rows: &[usize], cols: &[usize], out: &mut [f64]) {
    for (j, &col) in cols.iter().enumerate() {
        for (i, &row) in rows.iter().enumerate() {
            out[i + rows.len() * j] = data[row + nrows * col];
        }
    }
}

fn expensive_entry(row: usize, col: usize) -> f64 {
    let mut value = ((row + 1) as f64) * 0.137 + ((col + 1) as f64) * 0.173;
    for _ in 0..4 {
        value = (value.sin() + value.cos()).abs() + 0.25;
    }
    value
}

fn materialize_expensive(n: usize) -> Vec<f64> {
    let mut data = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            data[row + n * col] = expensive_entry(row, col);
        }
    }
    data
}

fn bench_lazy_block_rook(c: &mut Criterion) {
    let options = RrLUOptions {
        max_bond_dim: 8,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    };

    let mut cheap_group = c.benchmark_group("lazy_block_rook_cheap_callback");
    for &size in &[32usize, 64, 100, 128] {
        let data = random_column_major(size, size, 100 + size as u64);

        cheap_group.bench_with_input(
            BenchmarkId::new("materialize_then_dense", size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let materialized = data.clone();
                    let mut matrix = Matrix::zeros(n, n);
                    for col in 0..n {
                        for row in 0..n {
                            matrix[[row, col]] = materialized[row + n * col];
                        }
                    }
                    black_box(
                        matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap(),
                    );
                });
            },
        );

        cheap_group.bench_with_input(BenchmarkId::new("lazy_rook", size), &size, |b, &n| {
            b.iter(|| {
                black_box(
                    matrix_luci_factors_from_blocks(
                        n,
                        n,
                        |rows, cols, out: &mut [f64]| {
                            fill_from_dense(&data, n, rows, cols, out);
                        },
                        options.clone(),
                    )
                    .unwrap(),
                );
            });
        });
    }
    cheap_group.finish();

    let mut expensive_group = c.benchmark_group("lazy_block_rook_expensive_callback");
    for &size in &[32usize, 64, 100] {
        expensive_group.bench_with_input(
            BenchmarkId::new("materialize_then_dense", size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let materialized = materialize_expensive(n);
                    let mut matrix = Matrix::zeros(n, n);
                    for col in 0..n {
                        for row in 0..n {
                            matrix[[row, col]] = materialized[row + n * col];
                        }
                    }
                    black_box(
                        matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap(),
                    );
                });
            },
        );

        expensive_group.bench_with_input(BenchmarkId::new("lazy_rook", size), &size, |b, &n| {
            b.iter(|| {
                black_box(
                    matrix_luci_factors_from_blocks(
                        n,
                        n,
                        |rows, cols, out: &mut [f64]| {
                            for (j, &col) in cols.iter().enumerate() {
                                for (i, &row) in rows.iter().enumerate() {
                                    out[i + rows.len() * j] = expensive_entry(row, col);
                                }
                            }
                        },
                        options.clone(),
                    )
                    .unwrap(),
                );
            });
        });
    }
    expensive_group.finish();
}

criterion_group!(benches, bench_lazy_block_rook);
criterion_main!(benches);
