//! Benchmark cached TreeTN batch evaluation against TTCache and uncached TreeTN evaluation.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use num_complex::{Complex32, Complex64};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, TensorElement};
use tensor4all_simplett::{
    tensor3_zeros, MultiIndex, SimpleTensorTrain, TTCache, Tensor3, Tensor3Ops,
};
use tensor4all_treetn::{
    tensor_train_to_treetn, CachedEvaluatorOptions, TreeTN, TreeTNCachedEvaluator,
};

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

fn create_tt_with_bond_dim(
    n_sites: usize,
    local_dim: usize,
    bond_dim: usize,
) -> SimpleTensorTrain<f64> {
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

    SimpleTensorTrain::new(tensors).unwrap()
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

fn create_uniform_three_leaf_star(
    local_dim: usize,
    bond_dim: usize,
) -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let physical = (0..4)
        .map(|_| DynIndex::new_dyn(local_dim))
        .collect::<Vec<_>>();
    let bonds = (0..3)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect::<Vec<_>>();
    let hub_len = local_dim * bond_dim * bond_dim * bond_dim;
    let hub = IdxTensor::from_dense(
        vec![
            physical[0].clone(),
            bonds[0].clone(),
            bonds[1].clone(),
            bonds[2].clone(),
        ],
        vec![1.0_f64; hub_len],
    )
    .unwrap();
    let mut tensors = vec![hub];
    for leaf in 0..3 {
        tensors.push(
            IdxTensor::from_dense(
                vec![bonds[leaf].clone(), physical[leaf + 1].clone()],
                vec![1.0_f64; bond_dim * local_dim],
            )
            .unwrap(),
        );
    }
    (
        TreeTN::from_tensors(tensors, vec![0, 1, 2, 3]).unwrap(),
        physical,
    )
}

trait BenchmarkScalar: TensorElement {
    fn from_parts(real: f64, imag: f64) -> Self;
}

impl BenchmarkScalar for f32 {
    fn from_parts(real: f64, _imag: f64) -> Self {
        real as f32
    }
}

impl BenchmarkScalar for f64 {
    fn from_parts(real: f64, _imag: f64) -> Self {
        real
    }
}

impl BenchmarkScalar for Complex32 {
    fn from_parts(real: f64, imag: f64) -> Self {
        Self::new(real as f32, imag as f32)
    }
}

impl BenchmarkScalar for Complex64 {
    fn from_parts(real: f64, imag: f64) -> Self {
        Self::new(real, imag)
    }
}

fn create_typed_benchmark_chain<T: BenchmarkScalar>() -> (TreeTN<IdxTensor, usize>, Vec<DynIndex>) {
    let s0 = DynIndex::new_dyn(3);
    let b01 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let b12 = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(3);
    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), b01.clone()],
        (0..6)
            .map(|value| T::from_parts(value as f64 * 0.25 - 0.5, 0.125))
            .collect(),
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![b01, s1.clone(), b12.clone()],
        (0..12)
            .map(|value| T::from_parts(value as f64 * 0.125, -0.25))
            .collect(),
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![b12, s2.clone()],
        (0..9)
            .map(|value| T::from_parts(1.0 - value as f64 * 0.1, 0.0625))
            .collect(),
    )
    .unwrap();
    (
        TreeTN::from_tensors(vec![t0, t1, t2], vec![0, 1, 2]).unwrap(),
        vec![s0, s1, s2],
    )
}

fn bench_typed_scalar_kind<T: BenchmarkScalar>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    label: &str,
) {
    let (tree, indices) = create_typed_benchmark_chain::<T>();
    let values = (0..16)
        .flat_map(|point| [point % 3, (point / 2) % 2, (point + 1) % 3])
        .collect::<Vec<_>>();
    let shape = [3usize, 16usize];
    group.bench_function(BenchmarkId::new("cold", label), |b| {
        b.iter_batched_ref(
            || {
                TreeTNCachedEvaluator::new(
                    &tree,
                    &indices,
                    CachedEvaluatorOptions {
                        center: Some(1),
                        ..Default::default()
                    },
                )
                .unwrap()
            },
            |evaluator| {
                let points = ColMajorArrayRef::new(black_box(&values), &shape).unwrap();
                evaluator.evaluate_batched(points).unwrap()
            },
            BatchSize::SmallInput,
        )
    });

    let mut evaluator = TreeTNCachedEvaluator::new(
        &tree,
        &indices,
        CachedEvaluatorOptions {
            center: Some(1),
            ..Default::default()
        },
    )
    .unwrap();
    let points = ColMajorArrayRef::new(&values, &shape).unwrap();
    evaluator.evaluate_batched(points).unwrap();
    group.bench_function(BenchmarkId::new("warm", label), |b| {
        b.iter(|| {
            let points = ColMajorArrayRef::new(black_box(&values), &shape).unwrap();
            evaluator.evaluate_batched(points).unwrap()
        })
    });
}

/// Scalar-kind cold/warm baseline for #717. This is intentionally a paired
/// measurement fixture: the same topology, assignments, center, and batch are
/// used for f32, f64, Complex32, and Complex64, while the evaluator keeps its
/// existing raw-vs-generic dispatch policy.
fn bench_cached_evaluator_scalar_kinds(c: &mut Criterion) {
    let mut group = c.benchmark_group("treetn_cached_scalar_kind");
    group.sample_size(10);
    bench_typed_scalar_kind::<f32>(&mut group, "f32");
    bench_typed_scalar_kind::<f64>(&mut group, "f64");
    bench_typed_scalar_kind::<Complex32>(&mut group, "c32");
    bench_typed_scalar_kind::<Complex64>(&mut group, "c64");
    group.finish();
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

/// Does one warm `evaluate_batched` call cost more as the bond dimension grows?
///
/// Registered measurement. Two explanations of TreeACI's per-evaluation cost
/// predict different answers: a fixed per-message planning and dispatch cost
/// predicts no growth, while the contraction arithmetic and the data movement
/// around it predict growth of at least `O(chi^2)`. The fitted slope of
/// `log(time)` against `log(bond)` separates them.
///
/// Differences from `bench_bond_dim_scaling`, which varies the same parameter
/// but answers a different question: the evaluator is built **outside** the
/// timed closure here, because in the ACI path evaluators live for the whole
/// run and rebuilding one per call would time layout construction and centre
/// search instead of steady-state evaluation. The batch is a floating-zone
/// coordinate scan rather than scattered TCI-like points, and the contraction
/// centre is pinned to the varying site so centre placement is not a free
/// variable.
fn bench_warm_call_vs_bond(c: &mut Criterion) {
    const LOCAL_DIM: usize = 2;

    let mut group = c.benchmark_group("treetn_warm_call_vs_bond");
    for n_sites in [8usize, 32] {
        let varying = n_sites / 2;
        // One coordinate scan: both values of the varying site, rest fixed.
        let mut values = vec![0usize; n_sites * 2];
        for point in 0..2 {
            for site in 0..n_sites {
                values[site + n_sites * point] = usize::from(site != varying || point == 1);
            }
        }
        let shape = [n_sites, 2];
        let indices = (0..2)
            .map(|point| {
                (0..n_sites)
                    .map(|site| values[site + n_sites * point])
                    .collect::<MultiIndex>()
            })
            .collect::<Vec<_>>();

        for bond_dim in [2usize, 4, 8, 16, 32, 64, 128, 256] {
            let tt = create_tt_with_bond_dim(n_sites, LOCAL_DIM, bond_dim);
            let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();
            let mut tt_cache = TTCache::new(&tt);
            tt_cache.evaluate_many(&indices, Some(varying)).unwrap();
            let mut evaluator = TreeTNCachedEvaluator::new(
                &tree,
                &site_indices,
                CachedEvaluatorOptions::<usize> {
                    center: Some(varying),
                    ..CachedEvaluatorOptions::default()
                },
            )
            .unwrap();
            let points = ColMajorArrayRef::new(&values, &shape).unwrap();
            evaluator.evaluate_batched(points).unwrap();

            // Hiroshi's #646 review asks for the same coordinate batches and
            // fixed center/split when comparing persistent-cache behavior.
            group.bench_with_input(
                BenchmarkId::new(format!("ttcache_n{n_sites}"), bond_dim),
                &indices,
                |b, indices| {
                    b.iter(|| {
                        tt_cache
                            .evaluate_many(black_box(indices), Some(varying))
                            .unwrap()
                    })
                },
            );

            group.bench_with_input(
                BenchmarkId::new(format!("n{n_sites}"), bond_dim),
                &values,
                |b, values| {
                    b.iter(|| {
                        let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                        evaluator.evaluate_batched(points).unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}

/// Direct chain-evaluator parity protocol requested by Hiroshi Shinaoka in
/// <https://github.com/tensor4all/tensor4all-rs/pull/646#issuecomment-5316892012>.
///
/// Both evaluators see the same tensors, coordinate batch, scalar type, and a
/// fixed midpoint. The APIs cannot express identical contraction objects:
/// `TTCache` splits on a bond while `TreeTNCachedEvaluator` centers on a node.
/// That semantic difference is retained and reported because it controls how
/// much of a warm contraction each cache can reuse. `iter_batched_ref`
/// constructs a fresh evaluator outside each timed region, so the sample
/// measures one cold-cache evaluation without charging either implementation
/// for construction. The representative bond dimensions are the review's
/// requested 64, 128, and 256, rather than the small-rank cases that can hide
/// scaling defects.
fn bench_hiroshi_chain_evaluator_parity(c: &mut Criterion) {
    const N_SITES: usize = 16;
    const LOCAL_DIM: usize = 2;
    const N_LEFT: usize = 8;
    const N_RIGHT: usize = 8;
    const SPLIT: usize = N_SITES / 2;

    let indices = generate_tci_like_indices(N_LEFT, N_RIGHT, N_SITES, LOCAL_DIM, SPLIT, 646);
    let values = multi_indices_to_col_major(&indices, N_SITES);
    let shape = [N_SITES, indices.len()];
    let mut group = c.benchmark_group("hiroshi_chain_evaluator_parity");
    group.sample_size(10);

    for bond_dim in [64usize, 128, 256] {
        let tt = create_tt_with_bond_dim(N_SITES, LOCAL_DIM, bond_dim);
        let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();

        // Validate numerical parity once, outside the timed measurements.
        let mut tt_check = TTCache::new(&tt);
        let expected = tt_check.evaluate_many(&indices, Some(SPLIT)).unwrap();
        let mut tree_check = TreeTNCachedEvaluator::new(
            &tree,
            &site_indices,
            CachedEvaluatorOptions::<usize> {
                center: Some(SPLIT),
                ..CachedEvaluatorOptions::default()
            },
        )
        .unwrap();
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let actual = tree_check.evaluate_batched(points).unwrap();
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().zip(&expected).all(|(actual, expected)| {
            let actual = actual.real();
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 1.0e-12 * actual.abs().max(expected.abs()).max(1.0)
        }));

        group.bench_with_input(
            BenchmarkId::new("ttcache_cold", bond_dim),
            &indices,
            |b, indices| {
                b.iter_batched_ref(
                    || TTCache::new(&tt),
                    |cache| {
                        cache
                            .evaluate_many(black_box(indices), Some(SPLIT))
                            .unwrap()
                    },
                    BatchSize::LargeInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("treetn_cold", bond_dim),
            &values,
            |b, values| {
                b.iter_batched_ref(
                    || {
                        TreeTNCachedEvaluator::new(
                            &tree,
                            &site_indices,
                            CachedEvaluatorOptions::<usize> {
                                center: Some(SPLIT),
                                ..CachedEvaluatorOptions::default()
                            },
                        )
                        .unwrap()
                    },
                    |evaluator| {
                        let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                        evaluator.evaluate_batched(points).unwrap()
                    },
                    BatchSize::LargeInput,
                )
            },
        );

        // The same batch after both persistent evaluators have cached every
        // reusable environment/message, corresponding to the review's
        // follow-up request to separate repeated-cache behavior from cold
        // contraction work.
        let mut tt_warm = TTCache::new(&tt);
        tt_warm.evaluate_many(&indices, Some(SPLIT)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("ttcache_warm", bond_dim),
            &indices,
            |b, indices| {
                b.iter(|| {
                    tt_warm
                        .evaluate_many(black_box(indices), Some(SPLIT))
                        .unwrap()
                })
            },
        );

        let mut tree_warm = TreeTNCachedEvaluator::new(
            &tree,
            &site_indices,
            CachedEvaluatorOptions::<usize> {
                center: Some(SPLIT),
                ..CachedEvaluatorOptions::default()
            },
        )
        .unwrap();
        tree_warm.evaluate_batched(points).unwrap();
        group.bench_with_input(
            BenchmarkId::new("treetn_warm", bond_dim),
            &values,
            |b, values| {
                b.iter(|| {
                    let points = ColMajorArrayRef::new(black_box(values), &shape).unwrap();
                    tree_warm.evaluate_batched(points).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Bond-dimension scaling at a center node of coordination two versus three.
///
/// Hiroshi's issue #671 comment derives the topology-required local tensor
/// size as `d * product(incident bond dimensions)`, or `d * chi^z` for uniform
/// bonds: <https://github.com/tensor4all/tensor4all-rs/issues/671#issuecomment-5391376991>.
/// The benchmark IDs include that exact work proxy, making it possible to
/// distinguish its unavoidable extra factor of `chi` from regressions in
/// cache lookup, packing, or repeated environment construction.
fn bench_warm_center_coordination_vs_bond(c: &mut Criterion) {
    const LOCAL_DIM: usize = 2;
    let mut group = c.benchmark_group("treetn_warm_center_coordination_vs_bond");
    group.sample_size(10);

    for bond_dim in [4usize, 8, 16, 32, 64] {
        let chain_tt = create_tt_with_bond_dim(3, LOCAL_DIM, bond_dim);
        let (chain, chain_indices) = tensor_train_to_treetn(&chain_tt).unwrap();
        let chain_values = [0usize, 0, 0, 1, 0, 0];
        let chain_shape = [3usize, 2usize];
        let mut chain_evaluator = TreeTNCachedEvaluator::new(
            &chain,
            &chain_indices,
            CachedEvaluatorOptions::<usize> {
                center: Some(1),
                ..CachedEvaluatorOptions::default()
            },
        )
        .unwrap();
        chain_evaluator
            .evaluate_batched(ColMajorArrayRef::new(&chain_values, &chain_shape).unwrap())
            .unwrap();
        let chain_local_elements = LOCAL_DIM * bond_dim * bond_dim;
        group.bench_function(
            BenchmarkId::new(
                format!("z2_local_elements_{chain_local_elements}"),
                bond_dim,
            ),
            |b| {
                b.iter(|| {
                    let points =
                        ColMajorArrayRef::new(black_box(&chain_values), &chain_shape).unwrap();
                    chain_evaluator.evaluate_batched(points).unwrap()
                })
            },
        );

        let (star, star_indices) = create_uniform_three_leaf_star(LOCAL_DIM, bond_dim);
        let star_values = [0usize, 0, 0, 0, 1, 0, 0, 0];
        let star_shape = [4usize, 2usize];
        let mut star_evaluator = TreeTNCachedEvaluator::new(
            &star,
            &star_indices,
            CachedEvaluatorOptions::<usize> {
                center: Some(0),
                ..CachedEvaluatorOptions::default()
            },
        )
        .unwrap();
        star_evaluator
            .evaluate_batched(ColMajorArrayRef::new(&star_values, &star_shape).unwrap())
            .unwrap();
        let star_local_elements = LOCAL_DIM * bond_dim * bond_dim * bond_dim;
        group.bench_function(
            BenchmarkId::new(format!("z3_local_elements_{star_local_elements}"), bond_dim),
            |b| {
                b.iter(|| {
                    let points =
                        ColMajorArrayRef::new(black_box(&star_values), &star_shape).unwrap();
                    star_evaluator.evaluate_batched(points).unwrap()
                })
            },
        );
    }
    group.finish();
}

/// Measures the #711 owner-cache effect while deliberately disabling message
/// retention. Each iteration therefore recomputes the branch contraction, but
/// the prepared evaluator reuses the immutable physical slices it owns.
///
/// The paired cases use the same TreeTN, centre, assignments, and tenferro
/// backend settings. The only changed variable is the prepared-slice payload
/// budget, so the result isolates slice setup rather than claiming that a
/// warm message cache made the branch faster.
fn bench_prepared_branch_slice_reuse(c: &mut Criterion) {
    const LOCAL_DIM: usize = 2;
    let mut group = c.benchmark_group("treetn_prepared_branch_slice_reuse");
    group.sample_size(10);

    for bond_dim in [64usize, 128, 256] {
        let (tree, indices) = create_uniform_three_leaf_star(LOCAL_DIM, bond_dim);
        let point_count = 8;
        let values = (0..point_count)
            .flat_map(|point| [point % 2, 0, (point / 2) % 2, (point / 4) % 2])
            .collect::<Vec<_>>();
        let shape = [4usize, point_count];
        let points = ColMajorArrayRef::new(&values, &shape).unwrap();
        let common = |branch_slice_cache_max_bytes| CachedEvaluatorOptions {
            center: Some(1usize),
            message_cache_max_bytes: 0,
            branch_slice_cache_max_bytes,
            ..CachedEvaluatorOptions::default()
        };

        let mut prepared = TreeTNCachedEvaluator::new(&tree, &indices, common(usize::MAX)).unwrap();
        prepared.evaluate_batched(points).unwrap();
        group.bench_function(BenchmarkId::new("prepared", bond_dim), |b| {
            b.iter(|| {
                let points = ColMajorArrayRef::new(black_box(&values), &shape).unwrap();
                prepared.evaluate_batched(points).unwrap()
            })
        });

        let mut repacked = TreeTNCachedEvaluator::new(&tree, &indices, common(0)).unwrap();
        repacked.evaluate_batched(points).unwrap();
        group.bench_function(BenchmarkId::new("repacked", bond_dim), |b| {
            b.iter(|| {
                let points = ColMajorArrayRef::new(black_box(&values), &shape).unwrap();
                repacked.evaluate_batched(points).unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chain_size_scaling,
    bench_batch_size_scaling,
    bench_bond_dim_scaling,
    bench_warm_call_vs_bond,
    bench_hiroshi_chain_evaluator_parity,
    bench_warm_center_coordination_vs_bond,
    bench_cached_evaluator_scalar_kinds,
    bench_prepared_branch_slice_reuse
);
criterion_main!(benches);
