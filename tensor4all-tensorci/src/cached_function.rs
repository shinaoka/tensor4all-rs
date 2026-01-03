//! Cached function wrapper for expensive function evaluations

use std::collections::HashMap;
use std::hash::Hash;

/// A wrapper that caches function evaluations
///
/// This is useful when the function to be interpolated is expensive to evaluate.
#[derive(Debug)]
pub struct CachedFunction<K, V, F>
where
    K: Clone + Eq + Hash,
    V: Clone,
    F: Fn(&K) -> V,
{
    /// The underlying function
    func: F,
    /// Cache of evaluated values
    cache: HashMap<K, V>,
    /// Number of evaluations
    num_evals: usize,
    /// Number of cache hits
    num_cache_hits: usize,
}

impl<K, V, F> CachedFunction<K, V, F>
where
    K: Clone + Eq + Hash,
    V: Clone,
    F: Fn(&K) -> V,
{
    /// Create a new cached function wrapper
    pub fn new(func: F) -> Self {
        Self {
            func,
            cache: HashMap::new(),
            num_evals: 0,
            num_cache_hits: 0,
        }
    }

    /// Evaluate the function at a given key, using cache if available
    pub fn eval(&mut self, key: &K) -> V {
        if let Some(value) = self.cache.get(key) {
            self.num_cache_hits += 1;
            return value.clone();
        }

        self.num_evals += 1;
        let value = (self.func)(key);
        self.cache.insert(key.clone(), value.clone());
        value
    }

    /// Evaluate the function at a given key, bypassing the cache
    pub fn eval_no_cache(&self, key: &K) -> V {
        (self.func)(key)
    }

    /// Get the number of actual function evaluations
    pub fn num_evals(&self) -> usize {
        self.num_evals
    }

    /// Get the number of cache hits
    pub fn num_cache_hits(&self) -> usize {
        self.num_cache_hits
    }

    /// Get the total number of calls (evals + cache hits)
    pub fn total_calls(&self) -> usize {
        self.num_evals + self.num_cache_hits
    }

    /// Get the cache hit ratio
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.num_cache_hits as f64 / total as f64
        }
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the number of cached entries
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Check if a key is cached
    pub fn is_cached(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    /// Get a cached value without counting as a hit
    pub fn get_cached(&self, key: &K) -> Option<&V> {
        self.cache.get(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_function_basic() {
        let mut cf = CachedFunction::new(|x: &i32| x * x);

        assert_eq!(cf.eval(&5), 25);
        assert_eq!(cf.num_evals(), 1);
        assert_eq!(cf.num_cache_hits(), 0);

        // Second call should use cache
        assert_eq!(cf.eval(&5), 25);
        assert_eq!(cf.num_evals(), 1);
        assert_eq!(cf.num_cache_hits(), 1);

        // Different key
        assert_eq!(cf.eval(&3), 9);
        assert_eq!(cf.num_evals(), 2);
        assert_eq!(cf.num_cache_hits(), 1);
    }

    #[test]
    fn test_cached_function_vec_key() {
        let mut cf = CachedFunction::new(|x: &Vec<usize>| x.iter().sum::<usize>());

        assert_eq!(cf.eval(&vec![1, 2, 3]), 6);
        assert_eq!(cf.eval(&vec![1, 2, 3]), 6);
        assert_eq!(cf.num_evals(), 1);
        assert_eq!(cf.num_cache_hits(), 1);
    }

    #[test]
    fn test_cached_function_clear() {
        let mut cf = CachedFunction::new(|x: &i32| *x);
        cf.eval(&1);
        cf.eval(&2);
        assert_eq!(cf.cache_size(), 2);

        cf.clear_cache();
        assert_eq!(cf.cache_size(), 0);
    }
}
