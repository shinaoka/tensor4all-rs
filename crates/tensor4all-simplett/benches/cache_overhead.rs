//! Benchmark to measure overhead of TTCache::evaluate_many
//!
//! Tests robustness of auto_split heuristic with different data split positions.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tensor4all_simplett::cache::TTCache;
use tensor4all_simplett::tensortrain::SimpleTensorTrain;
use tensor4all_simplett::types::{tensor3_zeros, MultiIndex, Tensor3, Tensor3Ops};

/// Generate TCI-like indices with a specific split position
fn generate_tci_like_indices(
    n_left: usize,
    n_right: usize,
    n_sites: usize,
    local_dim: usize,
    split: usize,
    seed: u64,
) -> Vec<MultiIndex> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let left_parts: Vec<Vec<usize>> = (0..n_left)
        .map(|_| (0..split).map(|_| rng.random_range(0..local_dim)).collect())
        .collect();

    let right_parts: Vec<Vec<usize>> = (0..n_right)
        .map(|_| {
            (0..n_sites - split)
                .map(|_| rng.random_range(0..local_dim))
                .collect()
        })
        .collect();

    let mut indices = Vec::with_capacity(n_left * n_right);
    for left in &left_parts {
        for right in &right_parts {
            let mut idx = left.clone();
            idx.extend(right.iter().cloned());
            indices.push(idx);
        }
    }
    indices
}

/// Create a TT with specified bond dimension
fn create_tt_with_bond_dim(
    n_sites: usize,
    local_dim: usize,
    bond_dim: usize,
) -> SimpleTensorTrain<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(n_sites);

    for i in 0..n_sites {
        let left_dim = if i == 0 { 1 } else { bond_dim };
        let right_dim = if i == n_sites - 1 { 1 } else { bond_dim };

        let mut t: Tensor3<f64> = tensor3_zeros(left_dim, local_dim, right_dim);
        for l in 0..left_dim {
            for s in 0..local_dim {
                for r in 0..right_dim {
                    t.set3(l, s, r, rng.random::<f64>());
                }
            }
        }
        tensors.push(t);
    }

    SimpleTensorTrain::new(tensors).unwrap()
}

/// Benchmark split position robustness
///
/// Tests auto_split vs fixed correct/wrong positions for given data split.
fn bench_split_robustness(
    c: &mut Criterion,
    group_name: &str,
    data_split: usize, // Where data is actually split
    correct_split: usize,
    wrong_split: usize,
) {
    const N_SITES: usize = 100;
    const LOCAL_DIM: usize = 2;
    const N_LEFT: usize = 50;
    const N_RIGHT: usize = 50;
    const BOND_DIM: usize = 50;

    let tt = create_tt_with_bond_dim(N_SITES, LOCAL_DIM, BOND_DIM);
    let indices = generate_tci_like_indices(N_LEFT, N_RIGHT, N_SITES, LOCAL_DIM, data_split, 123);

    let mut group = c.benchmark_group(group_name);
    group.sample_size(20);

    // Auto split (should find the correct position)
    group.bench_function("auto", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache.evaluate_many(black_box(&indices), None).unwrap()
        })
    });

    // Correct split position
    group.bench_function("fixed_correct", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache
                .evaluate_many(black_box(&indices), Some(correct_split))
                .unwrap()
        })
    });

    // Wrong split position
    group.bench_function("fixed_wrong", |b| {
        b.iter(|| {
            let mut cache = TTCache::new(&tt);
            cache
                .evaluate_many(black_box(&indices), Some(wrong_split))
                .unwrap()
        })
    });

    group.finish();
}

/// Data split at 1/2 (middle): correct=1/2, wrong=1/4
fn bench_data_split_half(c: &mut Criterion) {
    bench_split_robustness(c, "data_split_1_2", 50, 50, 25);
}

/// Data split at 1/4: correct=1/4, wrong=1/2
fn bench_data_split_quarter(c: &mut Criterion) {
    bench_split_robustness(c, "data_split_1_4", 25, 25, 50);
}

/// Overhead measurement with rank-1 TT (pure overhead, minimal computation)
fn bench_overhead(c: &mut Criterion) {
    const N_SITES: usize = 100;
    const LOCAL_DIM: usize = 2;

    let tt = SimpleTensorTrain::<f64>::constant(&vec![LOCAL_DIM; N_SITES], 1.0);

    let mut group = c.benchmark_group("overhead");
    group.sample_size(20);

    for (n_left, n_right) in [(10, 10), (50, 50), (100, 100)] {
        let indices =
            generate_tci_like_indices(n_left, n_right, N_SITES, LOCAL_DIM, N_SITES / 2, 42);
        let n_total = indices.len();

        group.bench_with_input(
            BenchmarkId::new(format!("{}x{}", n_left, n_right), n_total),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut cache = TTCache::new(&tt);
                    cache.evaluate_many(black_box(indices), None).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_overhead,
    bench_data_split_half,
    bench_data_split_quarter
);
criterion_main!(benches);
