//! Benchmark matrix for the index-key encoder (issue #628).
//!
//! Measures indexer construction, encoding, tree-key composition, hashing, and
//! `HashMap` hit/miss/insert across widths straddling every fixed-width arm and
//! the fixed-to-dynamic boundary, for both many binary dimensions and fewer
//! large-radix ones.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher, RandomState};

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tensor4all_core::index_key::{FlatIndexer, IndexKey, KeyBuilder};

/// Target widths in bits, straddling every fixed/dynamic boundary.
///
/// The ladder now ends at `U512`, so 512/513 is the crossing to watch; 1024 and
/// 1025 are kept because #628 named them and they show the limb path's scaling.
const WIDTHS: [u64; 9] = [64, 128, 256, 512, 513, 1024, 1025, 2048, 4096];

/// Entries inserted into the maps under test.
const ENTRIES: usize = 1024;

/// `(label, dims)` for a target width: many binary dimensions, or fewer
/// large-radix ones. Radix 256 uses 8 bits per dimension.
fn profiles(width: u64) -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("binary", vec![2usize; width as usize]),
        ("radix256", vec![256usize; (width as usize).div_ceil(8)]),
    ]
}

/// Mixed-radix expansion of `entry` over `dims`, so distinct entries give
/// distinct multi-indices as long as `entry < prod(dims)`.
fn index_for(entry: usize, dims: &[usize]) -> Vec<usize> {
    let mut idx = vec![0usize; dims.len()];
    let mut rest = entry;
    for (slot, &dim) in idx.iter_mut().zip(dims) {
        *slot = rest % dim;
        rest /= dim;
        if rest == 0 {
            break;
        }
    }
    idx
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/construct");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            group.bench_with_input(BenchmarkId::new(label, width), &dims, |b, dims| {
                b.iter(|| FlatIndexer::try_new(black_box(dims)).unwrap())
            });
        }
    }
    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/encode");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let idx: Vec<usize> = dims.iter().map(|d| d - 1).collect();
            group.bench_with_input(BenchmarkId::new(label, width), &idx, |b, idx| {
                b.iter(|| indexer.encode(black_box(idx)).unwrap())
            });
        }
    }
    group.finish();
}

fn bench_compose(c: &mut Criterion) {
    let mut group = c.benchmark_group("index_key/compose");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            // Split into a local part and three children, as a degree-three node.
            let chunk = dims.len().div_ceil(4).max(1);
            let parts: Vec<Vec<usize>> = dims.chunks(chunk).map(<[usize]>::to_vec).collect();
            let keys: Vec<(IndexKey, u64)> = parts
                .iter()
                .map(|part| {
                    let indexer = FlatIndexer::try_new(part).unwrap();
                    let idx: Vec<usize> = part.iter().map(|d| d - 1).collect();
                    (indexer.encode(&idx).unwrap(), indexer.width_bits())
                })
                .collect();
            let total: u64 = keys.iter().map(|(_, w)| *w).sum();
            group.bench_with_input(BenchmarkId::new(label, width), &keys, |b, keys| {
                b.iter(|| {
                    let mut builder = KeyBuilder::with_capacity_bits(total).unwrap();
                    for (key, key_width) in keys {
                        builder.push(black_box(key), *key_width).unwrap();
                    }
                    builder.finish()
                })
            });
        }
    }
    group.finish();
}

fn bench_hash(c: &mut Criterion) {
    let state = RandomState::new();
    let mut group = c.benchmark_group("index_key/hash");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let idx: Vec<usize> = dims.iter().map(|d| d - 1).collect();
            let key = indexer.encode(&idx).unwrap();
            group.bench_with_input(BenchmarkId::new(label, width), &key, |b, key| {
                b.iter(|| {
                    let mut hasher = state.build_hasher();
                    std::hash::Hash::hash(black_box(key), &mut hasher);
                    hasher.finish()
                })
            });
        }
    }
    group.finish();
}

fn bench_map(c: &mut Criterion) {
    for (op, present) in [("hit", true), ("miss", false)] {
        let mut group = c.benchmark_group(format!("index_key/map_{op}"));
        for width in WIDTHS {
            for (label, dims) in profiles(width) {
                let indexer = FlatIndexer::try_new(&dims).unwrap();
                let mut map: HashMap<IndexKey, usize> = HashMap::with_capacity(ENTRIES);
                for entry in 0..ENTRIES {
                    map.insert(indexer.encode(&index_for(entry, &dims)).unwrap(), entry);
                }
                assert_eq!(
                    map.len(),
                    ENTRIES,
                    "{label}/{width}: keys were not distinct"
                );
                let probe = if present {
                    indexer.encode(&index_for(0, &dims)).unwrap()
                } else {
                    indexer.encode(&index_for(ENTRIES, &dims)).unwrap()
                };
                assert_eq!(map.contains_key(&probe), present);
                group.bench_with_input(BenchmarkId::new(label, width), &probe, |b, probe| {
                    b.iter(|| map.get(black_box(probe)).copied())
                });
            }
        }
        group.finish();
    }

    let mut group = c.benchmark_group("index_key/map_insert");
    for width in WIDTHS {
        for (label, dims) in profiles(width) {
            let indexer = FlatIndexer::try_new(&dims).unwrap();
            let keys: Vec<IndexKey> = (0..ENTRIES)
                .map(|entry| indexer.encode(&index_for(entry, &dims)).unwrap())
                .collect();
            group.bench_with_input(BenchmarkId::new(label, width), &keys, |b, keys| {
                b.iter(|| {
                    let mut map: HashMap<IndexKey, usize> = HashMap::with_capacity(ENTRIES);
                    for (entry, key) in keys.iter().enumerate() {
                        map.insert(black_box(key).clone(), entry);
                    }
                    map.len()
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_construction,
    bench_encode,
    bench_compose,
    bench_hash,
    bench_map
);
criterion_main!(benches);
