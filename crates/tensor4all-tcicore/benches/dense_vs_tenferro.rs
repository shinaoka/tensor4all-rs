use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{BackendSessionHost, Tensor};
use tensor4all_tcicore::{matrix_luci_factors_from_matrix, RrLUOptions};
use tensor4all_tensorbackend::{with_default_backend, Matrix};

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

fn bench_dense_vs_tenferro(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_no_truncation_vs_tenferro");
    let options = RrLUOptions {
        max_bond_dim: usize::MAX,
        rel_tol: 0.0,
        abs_tol: 0.0,
        left_orthogonal: true,
    };

    for &size in &[32usize, 64, 100, 128] {
        let data = random_column_major(size, size, 42);

        group.bench_with_input(BenchmarkId::new("matrixluci", size), &size, |b, &n| {
            b.iter(|| {
                let mut matrix = Matrix::zeros(n, n);
                for col in 0..n {
                    for row in 0..n {
                        matrix[[row, col]] = data[row + n * col];
                    }
                }
                black_box(matrix_luci_factors_from_matrix(&matrix, Some(options.clone())).unwrap());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("tenferro_full_piv_lu", size),
            &size,
            |b, &n| {
                b.iter(|| {
                    let mat = Tensor::from_vec_col_major(vec![n, n], data.clone()).unwrap();
                    black_box(
                        with_default_backend(|backend| {
                            backend.with_backend_session(|session| mat.full_piv_lu(session))
                        })
                        .unwrap(),
                    );
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_dense_vs_tenferro);
criterion_main!(benches);
