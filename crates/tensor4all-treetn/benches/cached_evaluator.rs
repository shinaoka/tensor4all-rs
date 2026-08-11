//! Benchmark cached TreeTN batch evaluation against TTCache and uncached TreeTN evaluation.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::ColMajorArrayRef;
use tensor4all_simplett::{tensor3_zeros, MultiIndex, TTCache, Tensor3, Tensor3Ops, TensorTrain};
use tensor4all_treetn::{tensor_train_to_treetn, CachedEvaluatorOptions, TreeTNCachedEvaluator};

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
            let mut index = left.clone();
            index.extend(right.iter().copied());
            indices.push(index);
        }
    }
    indices
}

fn create_tt_with_bond_dim(n_sites: usize, local_dim: usize, bond_dim: usize) -> TensorTrain<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut tensors: Vec<Tensor3<f64>> = Vec::with_capacity(n_sites);

    for site in 0..n_sites {
        let left_dim = if site == 0 { 1 } else { bond_dim };
        let right_dim = if site == n_sites - 1 { 1 } else { bond_dim };
        let mut tensor = tensor3_zeros(left_dim, local_dim, right_dim);
        for left in 0..left_dim {
            for local in 0..local_dim {
                for right in 0..right_dim {
                    tensor.set3(left, local, right, rng.random::<f64>());
                }
            }
        }
        tensors.push(tensor);
    }

    TensorTrain::new(tensors).unwrap()
}

fn multi_indices_to_col_major(indices: &[MultiIndex], n_sites: usize) -> Vec<usize> {
    let mut values = vec![0usize; n_sites * indices.len()];
    for (point, index) in indices.iter().enumerate() {
        for (site, value) in index.iter().copied().enumerate() {
            values[site + n_sites * point] = value;
        }
    }
    values
}

fn bench_chain_size_scaling(c: &mut Criterion) {
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 16;
    const N_LEFT: usize = 20;
    const N_RIGHT: usize = 20;

    let mut group = c.benchmark_group("treetn_cached_chain_size");
    group.sample_size(10);

    for n_sites in [16usize, 32, 64, 128] {
        let tt = create_tt_with_bond_dim(n_sites, LOCAL_DIM, BOND_DIM);
        let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();
        let indices = generate_tci_like_indices(
            N_LEFT,
            N_RIGHT,
            n_sites,
            LOCAL_DIM,
            n_sites / 2,
            n_sites as u64,
        );
        let values = multi_indices_to_col_major(&indices, n_sites);
        let shape = [n_sites, indices.len()];

        group.bench_with_input(
            BenchmarkId::new("ttcache", n_sites),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut cache = TTCache::new(&tt);
                    cache.evaluate_many(black_box(indices), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_cached", n_sites),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    let mut evaluator = TreeTNCachedEvaluator::new(
                        &tree,
                        &site_indices,
                        CachedEvaluatorOptions::<usize>::default(),
                    )
                    .unwrap();
                    evaluator.evaluate_batched(points).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_uncached", n_sites),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    tree.evaluate(&site_indices, points).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_size_scaling(c: &mut Criterion) {
    const N_SITES: usize = 64;
    const LOCAL_DIM: usize = 2;
    const BOND_DIM: usize = 16;

    let tt = create_tt_with_bond_dim(N_SITES, LOCAL_DIM, BOND_DIM);
    let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();
    let mut group = c.benchmark_group("treetn_cached_batch_size");
    group.sample_size(10);

    for (n_left, n_right) in [(10usize, 10usize), (20, 20), (40, 40)] {
        let indices = generate_tci_like_indices(
            n_left,
            n_right,
            N_SITES,
            LOCAL_DIM,
            N_SITES / 2,
            (n_left * n_right) as u64,
        );
        let values = multi_indices_to_col_major(&indices, N_SITES);
        let shape = [N_SITES, indices.len()];
        let label = format!("{}x{}", n_left, n_right);

        group.bench_with_input(
            BenchmarkId::new("ttcache", &label),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut cache = TTCache::new(&tt);
                    cache.evaluate_many(black_box(indices), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_cached", &label),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    let mut evaluator = TreeTNCachedEvaluator::new(
                        &tree,
                        &site_indices,
                        CachedEvaluatorOptions::<usize>::default(),
                    )
                    .unwrap();
                    evaluator.evaluate_batched(points).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_uncached", &label),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    tree.evaluate(&site_indices, points).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_bond_dim_scaling(c: &mut Criterion) {
    const N_SITES: usize = 128;
    const LOCAL_DIM: usize = 2;
    const N_LEFT: usize = 10;
    const N_RIGHT: usize = 10;

    let indices = generate_tci_like_indices(N_LEFT, N_RIGHT, N_SITES, LOCAL_DIM, N_SITES / 2, 2026);
    let values = multi_indices_to_col_major(&indices, N_SITES);
    let shape = [N_SITES, indices.len()];

    let mut group = c.benchmark_group("treetn_cached_bond_dim");
    group.sample_size(10);

    for bond_dim in [4usize, 8, 16, 32, 64] {
        let tt = create_tt_with_bond_dim(N_SITES, LOCAL_DIM, bond_dim);
        let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();

        group.bench_with_input(
            BenchmarkId::new("ttcache", bond_dim),
            &indices,
            |b, indices| {
                b.iter(|| {
                    let mut cache = TTCache::new(&tt);
                    cache.evaluate_many(black_box(indices), None).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_cached", bond_dim),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    let mut evaluator = TreeTNCachedEvaluator::new(
                        &tree,
                        &site_indices,
                        CachedEvaluatorOptions::<usize>::default(),
                    )
                    .unwrap();
                    evaluator.evaluate_batched(points).unwrap()
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chain_size_scaling,
    bench_batch_size_scaling,
    bench_bond_dim_scaling
);
criterion_main!(benches);
