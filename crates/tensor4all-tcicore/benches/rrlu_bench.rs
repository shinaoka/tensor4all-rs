use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tenferro_linalg::TensorLinalgExt;
use tenferro_tensor::{BackendSessionHost, Tensor};
use tensor4all_tcicore::{rrlu_mut, RrLUOptions};
use tensor4all_tensorbackend::{from_vec2d, with_default_backend, Matrix};

/// Generate a random f64 backend matrix.
fn random_matrix(n: usize, m: usize, seed: u64) -> Matrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..m).map(|_| rng.random::<f64>()).collect())
        .collect();
    from_vec2d(data)
}

/// Generate a random column-major tensor for the configured tensor backend.
fn random_tenferro_matrix(n: usize, m: usize, seed: u64) -> Tensor {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut data = vec![0.0; n * m];
    for col in 0..m {
        for row in 0..n {
            data[row + n * col] = rng.random::<f64>();
        }
    }
    Tensor::from_vec_col_major(vec![n, m], data).unwrap()
}

fn bench_rrlu(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrlu_full_rank");

    for &size in &[10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("rrlu_mut", size), &size, |b, &n| {
            b.iter_batched(
                || random_matrix(n, n, 42),
                |mut m| {
                    let opts = RrLUOptions {
                        max_bond_dim: n,
                        rel_tol: 0.0,
                        abs_tol: 0.0,
                        ..Default::default()
                    };
                    rrlu_mut(&mut m, Some(opts)).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_with_input(
            BenchmarkId::new("tenferro_full_piv_lu", size),
            &size,
            |b, &n| {
                b.iter_batched(
                    || random_tenferro_matrix(n, n, 42),
                    |m| {
                        with_default_backend(|backend| {
                            backend.with_backend_session(|session| m.full_piv_lu(session))
                        })
                        .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_rrlu);
criterion_main!(benches);
