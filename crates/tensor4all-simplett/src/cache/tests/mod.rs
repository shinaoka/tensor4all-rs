use super::*;
use crate::tensortrain::SimpleTensorTrain;
use crate::types::tensor3_zeros;

#[test]
fn test_ttcache_evaluate() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);

    // Evaluate at various indices
    let val = cache.evaluate(&[0, 0, 0]).unwrap();
    assert!((val - 2.0).abs() < 1e-10);

    let val = cache.evaluate(&[1, 2, 1]).unwrap();
    assert!((val - 2.0).abs() < 1e-10);
}

#[test]
fn test_ttcache_caching() {
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 2.0);
    t0.set3(0, 1, 1, 1.0);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 1);
    for l in 0..2 {
        for s in 0..3 {
            t1.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1]).unwrap();
    let mut cache = TTCache::new(&tt);

    // First evaluation
    let val1 = cache.evaluate(&[0, 1]).unwrap();

    // Should be cached now
    assert!(!cache.cache_left[0].is_empty());

    // Second evaluation should use cache
    let val2 = cache.evaluate(&[0, 1]).unwrap();
    assert!((val1 - val2).abs() < 1e-10);
}

#[test]
fn test_ttcache_clear() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    // Populate cache
    let _ = cache.evaluate(&[0, 0]);
    assert!(!cache.cache_left[0].is_empty());

    // Clear cache
    cache.clear_cache();
    assert!(cache.cache_left[0].is_empty());
    assert!(cache.cache_right[0].is_empty());
}

#[test]
fn test_ttcache_evaluate_many() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);

    let indices = vec![vec![0, 0, 0], vec![0, 1, 0], vec![1, 2, 1], vec![0, 0, 1]];

    let results = cache.evaluate_many(&indices, None).unwrap();

    // All values should be 2.0 for a constant TT
    assert_eq!(results.len(), 4);
    for val in &results {
        assert!((val - 2.0).abs() < 1e-10);
    }
}

#[test]
fn test_ttcache_evaluate_many_matches_single() {
    // Create a non-trivial TT
    let mut t0: Tensor3<f64> = tensor3_zeros(1, 2, 2);
    t0.set3(0, 0, 0, 1.0);
    t0.set3(0, 0, 1, 0.5);
    t0.set3(0, 1, 0, 2.0);
    t0.set3(0, 1, 1, 1.0);

    let mut t1: Tensor3<f64> = tensor3_zeros(2, 3, 2);
    for l in 0..2 {
        for s in 0..3 {
            for r in 0..2 {
                t1.set3(l, s, r, ((l + s + r) as f64) * 0.5 + 0.1);
            }
        }
    }

    let mut t2: Tensor3<f64> = tensor3_zeros(2, 2, 1);
    for l in 0..2 {
        for s in 0..2 {
            t2.set3(l, s, 0, (l + s + 1) as f64);
        }
    }

    let tt = SimpleTensorTrain::new(vec![t0, t1, t2]).unwrap();
    let mut cache = TTCache::new(&tt);

    // Generate all indices
    let mut indices = Vec::new();
    for i0 in 0..2 {
        for i1 in 0..3 {
            for i2 in 0..2 {
                indices.push(vec![i0, i1, i2]);
            }
        }
    }

    // Evaluate using evaluate_many
    let batch_results = cache.evaluate_many(&indices, None).unwrap();

    // Compare with single evaluations
    for (idx, batch_val) in indices.iter().zip(batch_results.iter()) {
        let single_val = cache.evaluate(idx).unwrap();
        assert!(
            (batch_val - single_val).abs() < 1e-10,
            "Mismatch at {:?}: batch={}, single={}",
            idx,
            batch_val,
            single_val
        );
    }
}

#[test]
fn test_ttcache_evaluate_many_cache_efficiency() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
    let mut cache = TTCache::new(&tt);

    // Indices with shared prefixes/suffixes
    let indices = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 0],
        vec![0, 0, 1, 1],
        vec![1, 1, 0, 0],
        vec![1, 1, 0, 1],
    ];

    let results = cache.evaluate_many(&indices, None).unwrap();
    assert_eq!(results.len(), 6);

    // Check that caches are populated
    // The optimal split should create fewer unique entries than indices.len()
    let total_left_cached: usize = cache.cache_left.iter().map(|c| c.len()).sum();
    let total_right_cached: usize = cache.cache_right.iter().map(|c| c.len()).sum();

    // With shared prefixes/suffixes, we should have fewer cached entries
    // than if we computed each index independently
    assert!(total_left_cached + total_right_cached > 0);
}

#[test]
fn test_ttcache_evaluate_many_empty() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    let results = cache.evaluate_many(&[], None).unwrap();
    assert!(results.is_empty());
}

#[test]
fn test_ttcache_evaluate_wrong_length() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);
    // Too few indices
    assert!(cache.evaluate(&[0]).is_err());
    // Too many indices
    assert!(cache.evaluate(&[0, 0, 0]).is_err());
}

#[test]
fn test_ttcache_partial_environment_wrong_length_errors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    assert!(cache.evaluate_left(&[0, 0, 0]).is_err());
    assert!(cache.evaluate_right(&[0, 0, 0]).is_err());
}

#[test]
fn test_ttcache_evaluate_many_invalid_split() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);
    let indices = vec![vec![0, 0]];
    // split=0 is invalid
    assert!(cache.evaluate_many(&indices, Some(0)).is_err());
    // split > n is invalid
    assert!(cache.evaluate_many(&indices, Some(10)).is_err());
}

#[test]
fn test_ttcache_evaluate_many_wrong_index_length_errors() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    let mut cache = TTCache::new(&tt);

    assert!(cache.evaluate_many(&[vec![0]], Some(1)).is_err());
    assert!(cache.evaluate_many(&[vec![0, 0, 0]], Some(1)).is_err());
}

#[test]
fn test_ttcache_evaluate_many_with_explicit_split() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 2.0);
    let mut cache = TTCache::new(&tt);
    let indices = vec![vec![0, 0, 0], vec![1, 2, 1]];
    let results = cache.evaluate_many(&indices, Some(1)).unwrap();
    assert_eq!(results.len(), 2);
    for r in &results {
        assert!((*r - 2.0).abs() < 1e-10);
    }
}

#[test]
fn test_with_site_dims_mismatch_length() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    // Wrong number of site_dims entries
    assert!(TTCache::with_site_dims(&tt, vec![vec![2]]).is_err());
}

#[test]
fn test_with_site_dims_mismatch_product() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    // Product doesn't match site dim
    assert!(TTCache::with_site_dims(&tt, vec![vec![3], vec![3]]).is_err());
}

#[test]
fn find_split_heuristic() {
    let tt = SimpleTensorTrain::<f64>::constant(&[2, 2, 2, 2], 1.0);
    let cache = TTCache::new(&tt);

    // Indices where split=1 is optimal:
    // Heuristic checks 1/4=1, 1/2=2, 3/4=3
    // split=1: unique_left=2, unique_right=2, total=4
    // split=2: unique_left=4, unique_right=1, total=5
    // split=3: unique_left=4, unique_right=1, total=5
    let indices = vec![
        vec![0, 0, 0, 0],
        vec![0, 1, 0, 0],
        vec![1, 0, 0, 0],
        vec![1, 1, 0, 0],
    ];

    let split = cache.find_split_heuristic(&indices);
    assert_eq!(split, 1);

    // Indices where split=3 is optimal:
    // split=1: unique_left=1, unique_right=4, total=5
    // split=2: unique_left=1, unique_right=4, total=5
    // split=3: unique_left=2, unique_right=2, total=4
    let indices2 = vec![
        vec![0, 0, 0, 0],
        vec![0, 0, 0, 1],
        vec![0, 0, 1, 0],
        vec![0, 0, 1, 1],
    ];

    let split2 = cache.find_split_heuristic(&indices2);
    assert_eq!(split2, 3);
}
