//! Benchmark: Compare different key types for cache

use criterion::{criterion_group, criterion_main, Criterion};
use std::collections::HashMap;

use bnum::types::{U1024, U256, U512};

// Runtime-selectable cache key
#[allow(dead_code)]
enum CacheKey {
    U64(u64),
    U128(u128),
    U256(U256),
    U512(U512),
    U1024(U1024),
}

// Runtime-selectable cache
enum DynCache {
    U64 {
        cache: HashMap<u64, f64>,
        coeffs: Vec<u64>,
    },
    U128 {
        cache: HashMap<u128, f64>,
        coeffs: Vec<u128>,
    },
    U256 {
        cache: HashMap<U256, f64>,
        coeffs: Vec<U256>,
    },
    U512 {
        cache: HashMap<U512, f64>,
        coeffs: Vec<U512>,
    },
    U1024 {
        cache: HashMap<U1024, f64>,
        coeffs: Vec<U1024>,
    },
}

impl DynCache {
    fn new_u64(l: usize) -> Self {
        let coeffs: Vec<u64> = (0..l).map(|i| 1u64 << i).collect();
        Self::U64 {
            cache: HashMap::new(),
            coeffs,
        }
    }
    fn new_u128(l: usize) -> Self {
        let coeffs: Vec<u128> = (0..l).map(|i| 1u128 << i).collect();
        Self::U128 {
            cache: HashMap::new(),
            coeffs,
        }
    }
    fn new_u256(l: usize) -> Self {
        let coeffs: Vec<U256> = (0..l).map(|i| U256::ONE << i).collect();
        Self::U256 {
            cache: HashMap::new(),
            coeffs,
        }
    }
    fn new_u512(l: usize) -> Self {
        let coeffs: Vec<U512> = (0..l).map(|i| U512::ONE << i).collect();
        Self::U512 {
            cache: HashMap::new(),
            coeffs,
        }
    }
    fn new_u1024(l: usize) -> Self {
        let coeffs: Vec<U1024> = (0..l).map(|i| U1024::ONE << i).collect();
        Self::U1024 {
            cache: HashMap::new(),
            coeffs,
        }
    }

    fn insert(&mut self, idx: &[u64], val: f64) {
        match self {
            Self::U64 { cache, coeffs } => {
                cache.insert(flat_index_u64(idx, coeffs), val);
            }
            Self::U128 { cache, coeffs } => {
                cache.insert(flat_index_u128(idx, coeffs), val);
            }
            Self::U256 { cache, coeffs } => {
                cache.insert(flat_index_u256(idx, coeffs), val);
            }
            Self::U512 { cache, coeffs } => {
                cache.insert(flat_index_u512(idx, coeffs), val);
            }
            Self::U1024 { cache, coeffs } => {
                cache.insert(flat_index_u1024(idx, coeffs), val);
            }
        }
    }

    fn get(&self, idx: &[u64]) -> Option<f64> {
        match self {
            Self::U64 { cache, coeffs } => cache.get(&flat_index_u64(idx, coeffs)).copied(),
            Self::U128 { cache, coeffs } => cache.get(&flat_index_u128(idx, coeffs)).copied(),
            Self::U256 { cache, coeffs } => cache.get(&flat_index_u256(idx, coeffs)).copied(),
            Self::U512 { cache, coeffs } => cache.get(&flat_index_u512(idx, coeffs)).copied(),
            Self::U1024 { cache, coeffs } => cache.get(&flat_index_u1024(idx, coeffs)).copied(),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U64 { cache, .. } => cache.len(),
            Self::U128 { cache, .. } => cache.len(),
            Self::U256 { cache, .. } => cache.len(),
            Self::U512 { cache, .. } => cache.len(),
            Self::U1024 { cache, .. } => cache.len(),
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::U64 { cache, .. } => cache.capacity(),
            Self::U128 { cache, .. } => cache.capacity(),
            Self::U256 { cache, .. } => cache.capacity(),
            Self::U512 { cache, .. } => cache.capacity(),
            Self::U1024 { cache, .. } => cache.capacity(),
        }
    }

    fn key_size(&self) -> usize {
        match self {
            Self::U64 { .. } => 8,
            Self::U128 { .. } => 16,
            Self::U256 { .. } => 32,
            Self::U512 { .. } => 64,
            Self::U1024 { .. } => 128,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::U64 { .. } => "u64",
            Self::U128 { .. } => "u128",
            Self::U256 { .. } => "U256",
            Self::U512 { .. } => "U512",
            Self::U1024 { .. } => "U1024",
        }
    }

    /// Select appropriate key type based on max bits needed
    #[allow(dead_code)]
    fn select(l: usize, bits_per_dim: usize) -> Self {
        let total_bits = l * bits_per_dim;
        if total_bits <= 64 {
            Self::new_u64(l)
        } else if total_bits <= 128 {
            Self::new_u128(l)
        } else if total_bits <= 256 {
            Self::new_u256(l)
        } else if total_bits <= 512 {
            Self::new_u512(l)
        } else {
            Self::new_u1024(l)
        }
    }
}

fn bench_all_key_types(c: &mut Criterion) {
    let l = 40usize;
    let n = 10000usize;

    let indices: Vec<Vec<u64>> = (0..n)
        .map(|i| (0..l).map(|j| ((i >> j) & 1) as u64).collect())
        .collect();

    // Build all caches
    let mut caches: Vec<DynCache> = vec![
        DynCache::new_u64(l),
        DynCache::new_u128(l),
        DynCache::new_u256(l),
        DynCache::new_u512(l),
        DynCache::new_u1024(l),
    ];

    for cache in &mut caches {
        for idx in &indices {
            cache.insert(idx, 1.0);
        }
    }

    // Memory info
    println!("\n=== Memory ({} entries, L={}) ===", caches[0].len(), l);
    for cache in &caches {
        let key_size = cache.key_size();
        let mem = cache.capacity() * (key_size + 8 + 1); // key + f64 + control
        println!(
            "{:>6}: {:>6.1} KB (key={}B)",
            cache.name(),
            mem as f64 / 1024.0,
            key_size
        );
    }
    println!();

    // Benchmarks
    for cache in &caches {
        let name = cache.name();
        c.bench_function(name, |b| {
            b.iter(|| indices.iter().filter_map(|idx| cache.get(idx)).sum::<f64>())
        });
    }
}

fn flat_index_u64(index: &[u64], coeffs: &[u64]) -> u64 {
    index.iter().zip(coeffs).map(|(&i, &c)| c * i).sum()
}

fn flat_index_u128(index: &[u64], coeffs: &[u128]) -> u128 {
    index.iter().zip(coeffs).map(|(&i, &c)| c * i as u128).sum()
}

fn flat_index_u256(index: &[u64], coeffs: &[U256]) -> U256 {
    index
        .iter()
        .zip(coeffs)
        .map(|(&i, &c)| c * U256::from(i))
        .fold(U256::ZERO, |a, b| a + b)
}

fn flat_index_u512(index: &[u64], coeffs: &[U512]) -> U512 {
    index
        .iter()
        .zip(coeffs)
        .map(|(&i, &c)| c * U512::from(i))
        .fold(U512::ZERO, |a, b| a + b)
}

fn flat_index_u1024(index: &[u64], coeffs: &[U1024]) -> U1024 {
    index
        .iter()
        .zip(coeffs)
        .map(|(&i, &c)| c * U1024::from(i))
        .fold(U1024::ZERO, |a, b| a + b)
}

criterion_group!(benches, bench_all_key_types);
criterion_main!(benches);
