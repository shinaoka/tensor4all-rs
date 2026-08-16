use super::error::CacheKeyError;
use super::*;
use bnum::types::{U2048, U256};
use std::sync::Arc;
use std::thread;

#[test]
fn test_cache_key_u64_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(u64::BITS_COUNT, 64);
    assert_eq!(u64::ZERO, 0u64);
    assert_eq!(u64::ONE, 1u64);
    assert_eq!(u64::from_usize(42), 42u64);
    assert_eq!(u64::ONE.checked_mul(u64::from_usize(10)), Some(10u64));
    assert_eq!(u64::MAX.checked_mul(u64::from_usize(2)), None);
}

#[test]
fn test_cache_key_u128_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(u128::BITS_COUNT, 128);
    assert_eq!(
        u128::from_usize(100).checked_mul(u128::from_usize(3)),
        Some(300u128)
    );
}

#[test]
fn test_cache_key_u256_basics() {
    use super::cache_key::CacheKey;

    assert_eq!(U256::BITS_COUNT, 256);
    let big = U256::from_usize(usize::MAX);
    assert!(big.checked_mul(U256::from_usize(2)).is_some());
}

#[test]
fn test_index_int_all_types() {
    use super::index_int::IndexInt;

    assert_eq!(42u8.to_usize(), 42);
    assert_eq!(1000u16.to_usize(), 1000);
    assert_eq!(100_000u32.to_usize(), 100_000);
    assert_eq!(999usize.to_usize(), 999);
}

#[test]
fn test_compute_coeffs_u64() {
    let coeffs = super::compute_coeffs::<u64>(&[2, 3, 4]).unwrap();
    assert_eq!(coeffs, vec![1u64, 2, 6]);
}

#[test]
fn test_compute_coeffs_overflow() {
    let result = super::compute_coeffs::<u64>(&[2; 100]);
    assert!(result.is_err());
    match result.unwrap_err() {
        CacheKeyError::Overflow { .. } => {}
        other => panic!("expected overflow error, got {other:?}"),
    }
}

#[test]
fn test_total_bits_calculation() {
    assert_eq!(super::total_bits(&[2, 2, 2]), 3);
    assert_eq!(super::total_bits(&[4, 4]), 4);
    assert_eq!(super::total_bits(&[1, 2, 1]), 1);
    assert_eq!(super::total_bits(&[256]), 8);
}

#[test]
fn test_cached_function_basic() {
    let local_dims = vec![2, 3, 4];
    let cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims).unwrap();

    assert_eq!(cf.eval(&[0, 1, 2]), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 0);

    // Second call should use cache
    assert_eq!(cf.eval(&[0, 1, 2]), 3);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 1);

    // Different index
    assert_eq!(cf.eval(&[1, 2, 3]), 6);
    assert_eq!(cf.num_evals(), 2);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_auto_key_selection_small() {
    // Small space: should use u64
    let local_dims = vec![2; 30]; // 2^30 < 2^64
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "u64");
}

#[test]
fn test_auto_key_selection_large() {
    // Large space: should use u128
    let local_dims = vec![2; 100]; // 2^100 > 2^64
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "u128");
}

#[test]
fn test_auto_key_selection_u256() {
    let local_dims = vec![2; 200];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U256");
}

#[test]
fn test_auto_key_selection_u512() {
    let local_dims = vec![2; 300];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U512");
}

#[test]
fn test_auto_key_selection_u1024() {
    let local_dims = vec![2; 600];
    let cf = CachedFunction::new(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "U1024");
}

#[test]
fn test_overflow_error() {
    let local_dims = vec![2; 1025];
    let result = CachedFunction::new(|_: &[usize]| 0.0, &local_dims);
    assert!(result.is_err());
}

impl cache_key::CacheKey for U2048 {
    const BITS_COUNT: u32 = 2048;
    const ZERO: Self = U2048::ZERO;
    const ONE: Self = U2048::ONE;

    fn from_usize(v: usize) -> Self {
        U2048::from(v as u64)
    }

    fn checked_mul(self, rhs: Self) -> Option<Self> {
        self.checked_mul(rhs)
    }

    fn wrapping_add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

#[test]
fn test_custom_key_type_u2048() {
    let local_dims = vec![2; 1025];
    assert!(CachedFunction::new(|_: &[usize]| 0.0, &local_dims).is_err());

    let cf = CachedFunction::with_key_type::<U2048>(|_: &[usize]| 0.0, &local_dims).unwrap();
    assert_eq!(cf.key_type(), "custom");

    let idx_zeros = vec![0usize; 1025];
    let idx_ones = vec![1usize; 1025];
    assert_eq!(cf.eval(&idx_zeros), 0.0);
    assert_eq!(cf.eval(&idx_ones), 0.0);
    assert_eq!(cf.num_evals(), 2);

    assert_eq!(cf.eval(&idx_zeros), 0.0);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_eval_batch_no_batch_func() {
    let local_dims = vec![2, 3];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] * 10 + idx[1], &local_dims).unwrap();

    cf.eval(&[0, 1]);
    assert_eq!(cf.num_evals(), 1);

    let indices = vec![vec![0, 1], vec![1, 2], vec![0, 0]];
    let results = cf.eval_batch(&indices);
    assert_eq!(results, vec![1, 12, 0]);
    assert_eq!(cf.num_evals(), 3);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_eval_batch_with_batch_func() {
    let local_dims = vec![2, 3];
    let single_f = |idx: &[usize]| idx[0] * 10 + idx[1];
    let batch_f = |indices: &[Vec<usize>]| -> Vec<usize> {
        indices.iter().map(|idx| idx[0] * 10 + idx[1]).collect()
    };
    let cf = CachedFunction::with_batch(single_f, batch_f, &local_dims).unwrap();

    cf.eval(&[1, 0]);

    let indices = vec![vec![1, 0], vec![0, 2], vec![1, 1]];
    let results = cf.eval_batch(&indices);
    assert_eq!(results, vec![10, 2, 11]);
    assert_eq!(cf.num_cache_hits(), 1);
    assert_eq!(cf.num_evals(), 3);
}

#[test]
fn test_eval_batch_empty() {
    let local_dims = vec![2, 3];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0], &local_dims).unwrap();
    let results = cf.eval_batch(&[]);
    assert!(results.is_empty());
}

#[test]
fn test_thread_safety() {
    let local_dims = vec![10, 10];
    let cf =
        Arc::new(CachedFunction::new(|idx: &[usize]| idx[0] * 100 + idx[1], &local_dims).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cf = Arc::clone(&cf);
            thread::spawn(move || {
                for i in 0..10 {
                    let idx = vec![(t * 2 + i) % 10, (t + i) % 10];
                    let val = cf.eval(&idx);
                    assert_eq!(val, idx[0] * 100 + idx[1]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    assert!(cf.cache_size() > 0);
    assert_eq!(cf.num_evals() + cf.num_cache_hits(), 40);
}

#[test]
fn test_thread_safety_batch() {
    let local_dims = vec![5, 5];
    let cf = Arc::new(CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims).unwrap());

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let cf = Arc::clone(&cf);
            thread::spawn(move || {
                let indices: Vec<Vec<usize>> = (0..5).map(|i| vec![(t + i) % 5, i % 5]).collect();
                let results = cf.eval_batch(&indices);
                for (i, idx) in indices.iter().enumerate() {
                    assert_eq!(results[i], idx[0] + idx[1]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_index_int_u8_cached_function() {
    let local_dims = vec![2; 8];
    let cf = CachedFunction::new(
        |idx: &[u8]| idx.iter().map(|&i| i as usize).sum::<usize>(),
        &local_dims,
    )
    .unwrap();
    assert_eq!(cf.key_type(), "u64");

    assert_eq!(cf.eval(&[0u8, 1, 0, 1, 0, 1, 0, 1]), 4);
    assert_eq!(cf.eval(&[0u8, 1, 0, 1, 0, 1, 0, 1]), 4);
    assert_eq!(cf.num_evals(), 1);
    assert_eq!(cf.num_cache_hits(), 1);
}

#[test]
fn test_cached_function_clear() {
    let local_dims = vec![10, 10];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &local_dims).unwrap();
    cf.eval(&[1, 2]);
    cf.eval(&[3, 4]);
    assert_eq!(cf.cache_size(), 2);

    cf.clear_cache();
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_local_dims() {
    let local_dims = vec![2, 3, 4];
    let cf = CachedFunction::new(|_: &[usize]| 0, &local_dims).unwrap();
    assert_eq!(cf.local_dims(), &[2, 3, 4]);
    assert_eq!(cf.num_sites(), 3);
}

#[test]
fn test_u128_cache_operations() {
    // Use enough dimensions to force u128 key type
    let local_dims = vec![2; 100];
    let cf = CachedFunction::new(|idx: &[usize]| idx.iter().sum::<usize>(), &local_dims).unwrap();
    assert_eq!(cf.key_type(), "u128");

    let idx = vec![0; 100];
    assert_eq!(cf.eval(&idx), 0);
    assert_eq!(cf.num_evals(), 1);
    assert!(!cf.is_cached(&vec![1; 100]));
    assert!(cf.is_cached(&idx));

    // Cache hit
    assert_eq!(cf.eval(&idx), 0);
    assert_eq!(cf.num_cache_hits(), 1);
    assert_eq!(cf.cache_size(), 1);

    // Clear
    cf.clear_cache();
    assert_eq!(cf.cache_size(), 0);
}

#[test]
fn test_eval_no_cache_and_stats() {
    let local_dims = vec![2, 3];
    let cf = CachedFunction::new(|idx: &[usize]| idx[0] * 10 + idx[1], &local_dims).unwrap();

    // eval_no_cache does not populate cache or affect stats
    assert_eq!(cf.eval_no_cache(&[1, 2]), 12);
    assert_eq!(cf.num_evals(), 0);
    assert_eq!(cf.total_calls(), 0);
    assert_eq!(cf.cache_hit_ratio(), 0.0);

    // Now eval to populate cache
    cf.eval(&[1, 2]);
    cf.eval(&[1, 2]); // cache hit
    assert_eq!(cf.total_calls(), 2);
    assert_eq!(cf.cache_hit_ratio(), 0.5);
    assert!(cf.is_cached(&[1, 2]));
}
