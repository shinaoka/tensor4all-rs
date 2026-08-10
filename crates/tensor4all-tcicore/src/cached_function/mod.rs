//! Cached function wrapper for expensive function evaluations.
//!
//! [`CachedFunction`] wraps a user-supplied function `Fn(&[I]) -> V` with
//! thread-safe memoization. On first call for a given multi-index, the
//! function is evaluated and the result is cached; subsequent calls return
//! the cached value.
//!
//! The internal cache key type is automatically selected based on the total
//! index space size (up to 1024 bits by default). For larger index spaces,
//! use [`CachedFunction::with_key_type`] with a custom [`CacheKey`]
//! implementation.
//!
//! # Examples
//!
//! ```
//! use tensor4all_tcicore::CachedFunction;
//!
//! let cf = CachedFunction::new(
//!     |idx: &[usize]| idx[0] + idx[1],
//!     &[3, 4],
//! ).unwrap();
//!
//! assert_eq!(cf.eval(&[1, 2]), 3);
//! assert_eq!(cf.num_evals(), 1);
//! assert_eq!(cf.eval(&[1, 2]), 3);
//! assert_eq!(cf.num_cache_hits(), 1);
//! ```

pub mod cache_key;
pub mod error;
pub mod index_int;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use bnum::types::{U1024, U256, U512};

use cache_key::CacheKey;
use index_int::IndexInt;

/// Compute total bits needed to represent the index space.
pub(crate) fn total_bits(local_dims: &[usize]) -> u32 {
    local_dims
        .iter()
        .map(|&d| {
            if d <= 1 {
                0
            } else {
                ((d - 1) as u64).ilog2() + 1
            }
        })
        .sum()
}

/// Compute mixed-radix coefficients for flat index computation.
///
/// Returns `Err(CacheKeyError::Overflow)` if the index space overflows key type `K`.
pub(crate) fn compute_coeffs<K: CacheKey>(
    local_dims: &[usize],
) -> Result<Vec<K>, error::CacheKeyError> {
    let bits = total_bits(local_dims);
    if bits > K::BITS_COUNT {
        return Err(error::CacheKeyError::Overflow {
            total_bits: bits,
            max_bits: K::BITS_COUNT,
            key_type: std::any::type_name::<K>(),
        });
    }

    let mut coeffs = Vec::with_capacity(local_dims.len());
    let mut prod = K::ONE;
    for &d in local_dims {
        coeffs.push(prod.clone());
        let dim = K::from_usize(d);
        prod = prod
            .checked_mul(dim)
            .ok_or_else(|| error::CacheKeyError::Overflow {
                total_bits: bits,
                max_bits: K::BITS_COUNT,
                key_type: std::any::type_name::<K>(),
            })?;
    }

    Ok(coeffs)
}

fn read_lock<T>(lock: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    lock.read().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn write_lock<T>(lock: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    lock.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Compute flat index from multi-index and coefficients.
fn flat_index<K: CacheKey, I: IndexInt>(idx: &[I], coeffs: &[K]) -> K {
    idx.iter().zip(coeffs).fold(K::ZERO, |acc, (&i, c)| {
        acc.wrapping_add(
            c.clone()
                .checked_mul(K::from_usize(i.to_usize()))
                .unwrap_or(K::ZERO),
        )
    })
}

/// Internal cache with automatically selected key type.
enum InnerCache<V> {
    U64 {
        cache: RwLock<HashMap<u64, V>>,
        coeffs: Vec<u64>,
    },
    U128 {
        cache: RwLock<HashMap<u128, V>>,
        coeffs: Vec<u128>,
    },
    U256 {
        cache: RwLock<HashMap<U256, V>>,
        coeffs: Vec<U256>,
    },
    U512 {
        cache: RwLock<HashMap<U512, V>>,
        coeffs: Vec<U512>,
    },
    U1024 {
        cache: RwLock<HashMap<U1024, V>>,
        coeffs: Vec<U1024>,
    },
}

impl<V: Clone + Send + Sync> InnerCache<V> {
    /// Create a new cache, automatically selecting the key type.
    fn new(local_dims: &[usize]) -> Result<Self, error::CacheKeyError> {
        let bits = total_bits(local_dims);
        if bits <= 64 {
            Ok(Self::U64 {
                cache: RwLock::new(HashMap::new()),
                coeffs: compute_coeffs::<u64>(local_dims)?,
            })
        } else if bits <= 128 {
            Ok(Self::U128 {
                cache: RwLock::new(HashMap::new()),
                coeffs: compute_coeffs::<u128>(local_dims)?,
            })
        } else if bits <= 256 {
            Ok(Self::U256 {
                cache: RwLock::new(HashMap::new()),
                coeffs: compute_coeffs::<U256>(local_dims)?,
            })
        } else if bits <= 512 {
            Ok(Self::U512 {
                cache: RwLock::new(HashMap::new()),
                coeffs: compute_coeffs::<U512>(local_dims)?,
            })
        } else if bits <= 1024 {
            Ok(Self::U1024 {
                cache: RwLock::new(HashMap::new()),
                coeffs: compute_coeffs::<U1024>(local_dims)?,
            })
        } else {
            Err(error::CacheKeyError::Overflow {
                total_bits: bits,
                max_bits: 1024,
                key_type: "auto",
            })
        }
    }

    fn get<I: IndexInt>(&self, idx: &[I]) -> Option<V> {
        match self {
            Self::U64 { cache, coeffs } => read_lock(cache).get(&flat_index(idx, coeffs)).cloned(),
            Self::U128 { cache, coeffs } => read_lock(cache).get(&flat_index(idx, coeffs)).cloned(),
            Self::U256 { cache, coeffs } => read_lock(cache).get(&flat_index(idx, coeffs)).cloned(),
            Self::U512 { cache, coeffs } => read_lock(cache).get(&flat_index(idx, coeffs)).cloned(),
            Self::U1024 { cache, coeffs } => {
                read_lock(cache).get(&flat_index(idx, coeffs)).cloned()
            }
        }
    }

    fn insert<I: IndexInt>(&self, idx: &[I], value: V) {
        match self {
            Self::U64 { cache, coeffs } => {
                write_lock(cache).insert(flat_index(idx, coeffs), value);
            }
            Self::U128 { cache, coeffs } => {
                write_lock(cache).insert(flat_index(idx, coeffs), value);
            }
            Self::U256 { cache, coeffs } => {
                write_lock(cache).insert(flat_index(idx, coeffs), value);
            }
            Self::U512 { cache, coeffs } => {
                write_lock(cache).insert(flat_index(idx, coeffs), value);
            }
            Self::U1024 { cache, coeffs } => {
                write_lock(cache).insert(flat_index(idx, coeffs), value);
            }
        }
    }

    fn contains<I: IndexInt>(&self, idx: &[I]) -> bool {
        match self {
            Self::U64 { cache, coeffs } => read_lock(cache).contains_key(&flat_index(idx, coeffs)),
            Self::U128 { cache, coeffs } => read_lock(cache).contains_key(&flat_index(idx, coeffs)),
            Self::U256 { cache, coeffs } => read_lock(cache).contains_key(&flat_index(idx, coeffs)),
            Self::U512 { cache, coeffs } => read_lock(cache).contains_key(&flat_index(idx, coeffs)),
            Self::U1024 { cache, coeffs } => {
                read_lock(cache).contains_key(&flat_index(idx, coeffs))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U64 { cache, .. } => read_lock(cache).len(),
            Self::U128 { cache, .. } => read_lock(cache).len(),
            Self::U256 { cache, .. } => read_lock(cache).len(),
            Self::U512 { cache, .. } => read_lock(cache).len(),
            Self::U1024 { cache, .. } => read_lock(cache).len(),
        }
    }

    fn clear(&self) {
        match self {
            Self::U64 { cache, .. } => write_lock(cache).clear(),
            Self::U128 { cache, .. } => write_lock(cache).clear(),
            Self::U256 { cache, .. } => write_lock(cache).clear(),
            Self::U512 { cache, .. } => write_lock(cache).clear(),
            Self::U1024 { cache, .. } => write_lock(cache).clear(),
        }
    }

    fn key_type_name(&self) -> &'static str {
        match self {
            Self::U64 { .. } => "u64",
            Self::U128 { .. } => "u128",
            Self::U256 { .. } => "U256",
            Self::U512 { .. } => "U512",
            Self::U1024 { .. } => "U1024",
        }
    }
}

/// Type-erased cache interface for custom key types.
trait DynCache<V>: Send + Sync {
    fn get(&self, idx: &[usize]) -> Option<V>;
    fn insert(&self, idx: &[usize], value: V);
    fn contains(&self, idx: &[usize]) -> bool;
    fn len(&self) -> usize;
    fn clear(&self);
}

/// Generic cache for user-specified key types.
struct GenericCache<K: CacheKey, V> {
    cache: RwLock<HashMap<K, V>>,
    coeffs: Vec<K>,
}

impl<K: CacheKey, V: Clone + Send + Sync> GenericCache<K, V> {
    fn new(local_dims: &[usize]) -> Result<Self, error::CacheKeyError> {
        Ok(Self {
            cache: RwLock::new(HashMap::new()),
            coeffs: compute_coeffs::<K>(local_dims)?,
        })
    }
}

impl<K: CacheKey, V: Clone + Send + Sync> DynCache<V> for GenericCache<K, V> {
    fn get(&self, idx: &[usize]) -> Option<V> {
        let key = flat_index::<K, usize>(idx, &self.coeffs);
        read_lock(&self.cache).get(&key).cloned()
    }

    fn insert(&self, idx: &[usize], value: V) {
        let key = flat_index::<K, usize>(idx, &self.coeffs);
        write_lock(&self.cache).insert(key, value);
    }

    fn contains(&self, idx: &[usize]) -> bool {
        let key = flat_index::<K, usize>(idx, &self.coeffs);
        read_lock(&self.cache).contains_key(&key)
    }

    fn len(&self) -> usize {
        read_lock(&self.cache).len()
    }

    fn clear(&self) {
        write_lock(&self.cache).clear();
    }
}

/// Internal backend: auto-selected enum or custom type-erased cache.
enum CacheBackend<V: Clone + Send + Sync + 'static> {
    Auto(InnerCache<V>),
    Custom(Box<dyn DynCache<V>>),
}

impl<V: Clone + Send + Sync + 'static> CacheBackend<V> {
    fn get<I: IndexInt>(&self, idx: &[I]) -> Option<V> {
        match self {
            Self::Auto(inner) => inner.get(idx),
            Self::Custom(cache) => {
                let usize_idx: Vec<usize> = idx.iter().map(|&i| i.to_usize()).collect();
                cache.get(&usize_idx)
            }
        }
    }

    fn insert<I: IndexInt>(&self, idx: &[I], value: V) {
        match self {
            Self::Auto(inner) => inner.insert(idx, value),
            Self::Custom(cache) => {
                let usize_idx: Vec<usize> = idx.iter().map(|&i| i.to_usize()).collect();
                cache.insert(&usize_idx, value);
            }
        }
    }

    fn contains<I: IndexInt>(&self, idx: &[I]) -> bool {
        match self {
            Self::Auto(inner) => inner.contains(idx),
            Self::Custom(cache) => {
                let usize_idx: Vec<usize> = idx.iter().map(|&i| i.to_usize()).collect();
                cache.contains(&usize_idx)
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Auto(inner) => inner.len(),
            Self::Custom(cache) => cache.len(),
        }
    }

    fn clear(&self) {
        match self {
            Self::Auto(inner) => inner.clear(),
            Self::Custom(cache) => cache.clear(),
        }
    }

    fn key_type_name(&self) -> &'static str {
        match self {
            Self::Auto(inner) => inner.key_type_name(),
            Self::Custom(_) => "custom",
        }
    }
}

type BatchFunc<I, V> = dyn Fn(&[Vec<I>]) -> Vec<V> + Send + Sync;

/// A wrapper that caches function evaluations for multi-index inputs.
///
/// Thread-safe: all methods take `&self`. Multiple threads can call `eval`
/// concurrently.
///
/// # Type parameters
///
/// - `V` - cached value type
/// - `F` - single-evaluation function `Fn(&[I]) -> V`
/// - `I` - index element type (default `usize`); use `u8` for quantics
///
/// # Examples
///
/// ```
/// use tensor4all_tcicore::CachedFunction;
///
/// // Cache a 2-site function with local dimensions [3, 4]
/// let cf = CachedFunction::new(
///     |idx: &[usize]| (idx[0] * 4 + idx[1]) as f64,
///     &[3, 4],
/// ).unwrap();
///
/// // First call evaluates and caches
/// let v00 = cf.eval(&[0, 0]);
/// assert_eq!(v00, 0.0);
/// assert_eq!(cf.num_evals(), 1);
/// assert_eq!(cf.num_cache_hits(), 0);
///
/// // Second call uses cache
/// let v00_again = cf.eval(&[0, 0]);
/// assert_eq!(v00_again, 0.0);
/// assert_eq!(cf.num_cache_hits(), 1);
///
/// let v12 = cf.eval(&[1, 2]);
/// assert_eq!(v12, 6.0); // 1*4 + 2
/// ```
pub struct CachedFunction<V, F, I = usize>
where
    I: IndexInt,
    V: Clone + Send + Sync + 'static,
    F: Fn(&[I]) -> V + Send + Sync,
{
    func: F,
    batch_func: Option<Box<BatchFunc<I, V>>>,
    cache: CacheBackend<V>,
    local_dims: Vec<usize>,
    num_evals: AtomicUsize,
    num_cache_hits: AtomicUsize,
    _phantom: std::marker::PhantomData<I>,
}

impl<V, F, I> CachedFunction<V, F, I>
where
    I: IndexInt,
    V: Clone + Send + Sync + 'static,
    F: Fn(&[I]) -> V + Send + Sync,
{
    /// Create a new cached function with automatic key selection (up to 1024 bits).
    ///
    /// # Errors
    ///
    /// Returns an error when the cache dimensions are invalid (a shape mismatch)
    /// /// or the construction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0] + idx[1], &[2, 3]).unwrap();
    /// assert_eq!(cf.eval(&[1, 2]), 3);
    /// assert_eq!(cf.num_sites(), 2);
    /// assert_eq!(cf.local_dims(), &[2, 3]);
    /// ```
    pub fn new(func: F, local_dims: &[usize]) -> Result<Self, error::CacheKeyError> {
        Ok(Self {
            func,
            batch_func: None,
            cache: CacheBackend::Auto(InnerCache::new(local_dims)?),
            local_dims: local_dims.to_vec(),
            num_evals: AtomicUsize::new(0),
            num_cache_hits: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create with a batch function for efficient multi-point evaluation.
    ///
    /// The batch function is used for cache misses during [`eval_batch`](Self::eval_batch)
    /// calls, enabling amortized cost when evaluating many indices at once
    /// (e.g., batch FFI calls or vectorized computations).
    ///
    /// # Errors
    ///
    /// Returns an error when the batch configuration is invalid (an
    /// /// invalid-configuration failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::with_batch(
    ///     |idx: &[usize]| idx[0] * 10 + idx[1],
    ///     |indices: &[Vec<usize>]| indices.iter().map(|idx| idx[0] * 10 + idx[1]).collect(),
    ///     &[3, 4],
    /// ).unwrap();
    ///
    /// let results = cf.eval_batch(&[vec![0, 1], vec![2, 3]]);
    /// assert_eq!(results, vec![1, 23]);
    /// assert_eq!(cf.num_evals(), 2);
    /// ```
    pub fn with_batch<B>(
        func: F,
        batch_func: B,
        local_dims: &[usize],
    ) -> Result<Self, error::CacheKeyError>
    where
        B: Fn(&[Vec<I>]) -> Vec<V> + Send + Sync + 'static,
    {
        Ok(Self {
            func,
            batch_func: Some(Box::new(batch_func)),
            cache: CacheBackend::Auto(InnerCache::new(local_dims)?),
            local_dims: local_dims.to_vec(),
            num_evals: AtomicUsize::new(0),
            num_cache_hits: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create with an explicit key type for index spaces larger than 1024 bits.
    ///
    /// # Errors
    ///
    /// Returns an error when the key type configuration is invalid (an
    /// /// invalid-configuration failure).
    ///
    /// # Example
    ///
    /// ```
    /// use bnum::types::U2048;
    /// use tensor4all_tcicore::{CacheKey, CachedFunction};
    ///
    /// #[derive(Clone, Hash, PartialEq, Eq)]
    /// struct U2048Key(U2048);
    ///
    /// impl CacheKey for U2048Key {
    ///     const BITS_COUNT: u32 = 2048;
    ///     const ZERO: Self = Self(U2048::ZERO);
    ///     const ONE: Self = Self(U2048::ONE);
    ///
    ///     fn from_usize(v: usize) -> Self {
    ///         Self(U2048::from(v as u64))
    ///     }
    ///
    ///     fn checked_mul(self, rhs: Self) -> Option<Self> {
    ///         self.0.checked_mul(rhs.0).map(Self)
    ///     }
    ///
    ///     fn wrapping_add(self, rhs: Self) -> Self {
    ///         Self(self.0.wrapping_add(rhs.0))
    ///     }
    /// }
    ///
    /// let local_dims = vec![2usize; 1025];
    /// let cf = CachedFunction::with_key_type::<U2048Key>(
    ///     |idx: &[usize]| idx.iter().sum::<usize>(),
    ///     &local_dims,
    /// ).unwrap();
    /// let zeros = vec![0usize; 1025];
    ///
    /// assert_eq!(cf.eval(&zeros), 0);
    /// assert_eq!(cf.key_type(), "custom");
    /// ```
    pub fn with_key_type<K: CacheKey>(
        func: F,
        local_dims: &[usize],
    ) -> Result<Self, error::CacheKeyError> {
        Ok(Self {
            func,
            batch_func: None,
            cache: CacheBackend::Custom(Box::new(GenericCache::<K, V>::new(local_dims)?)),
            local_dims: local_dims.to_vec(),
            num_evals: AtomicUsize::new(0),
            num_cache_hits: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create with explicit key type and batch function.
    ///
    /// Combines [`with_key_type`](Self::with_key_type) and
    /// [`with_batch`](Self::with_batch) for index spaces larger than 1024
    /// bits that also benefit from batch evaluation.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration is invalid (an
    /// /// invalid-configuration failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// // Use u128 key type with batch support
    /// let cf = CachedFunction::with_key_type_and_batch::<u128, _>(
    ///     |idx: &[usize]| idx.iter().sum::<usize>(),
    ///     |indices: &[Vec<usize>]| indices.iter().map(|idx| idx.iter().sum()).collect(),
    ///     &[2, 3, 4],
    /// ).unwrap();
    ///
    /// let results = cf.eval_batch(&[vec![0, 0, 0], vec![1, 2, 3]]);
    /// assert_eq!(results, vec![0, 6]);
    /// ```
    pub fn with_key_type_and_batch<K: CacheKey, B>(
        func: F,
        batch_func: B,
        local_dims: &[usize],
    ) -> Result<Self, error::CacheKeyError>
    where
        B: Fn(&[Vec<I>]) -> Vec<V> + Send + Sync + 'static,
    {
        Ok(Self {
            func,
            batch_func: Some(Box::new(batch_func)),
            cache: CacheBackend::Custom(Box::new(GenericCache::<K, V>::new(local_dims)?)),
            local_dims: local_dims.to_vec(),
            num_evals: AtomicUsize::new(0),
            num_cache_hits: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Evaluate at a given index, using cache if available.
    ///
    /// On the first call for a given index, the wrapped function is invoked and
    /// the result is cached. Subsequent calls with the same index return the
    /// cached value. This method is thread-safe.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0] * idx[1], &[5, 5]).unwrap();
    /// assert_eq!(cf.eval(&[3, 4]), 12);
    /// assert_eq!(cf.num_evals(), 1);
    ///
    /// // Cache hit
    /// assert_eq!(cf.eval(&[3, 4]), 12);
    /// assert_eq!(cf.num_evals(), 1);
    /// assert_eq!(cf.num_cache_hits(), 1);
    /// ```
    pub fn eval(&self, idx: &[I]) -> V {
        if let Some(value) = self.cache.get(idx) {
            self.num_cache_hits.fetch_add(1, Ordering::Relaxed);
            return value;
        }

        self.num_evals.fetch_add(1, Ordering::Relaxed);
        let value = (self.func)(idx);
        self.cache.insert(idx, value.clone());
        value
    }

    /// Evaluate bypassing the cache.
    ///
    /// The result is neither read from nor stored in the cache, and
    /// evaluation counters are not updated. Useful for verification or
    /// when the caller intentionally wants a fresh evaluation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0] + 1, &[4]).unwrap();
    /// assert_eq!(cf.eval_no_cache(&[2]), 3);
    /// assert_eq!(cf.cache_size(), 0);
    /// assert_eq!(cf.num_evals(), 0);
    /// ```
    pub fn eval_no_cache(&self, idx: &[I]) -> V {
        (self.func)(idx)
    }

    /// Evaluate at multiple indices. Uses batch function for cache misses if available.
    ///
    /// Returns results in the same order as the input indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0] * 2 + idx[1], &[2, 2]).unwrap();
    /// let results = cf.eval_batch(&[vec![0, 0], vec![0, 1], vec![1, 0]]);
    /// assert_eq!(results, vec![0, 1, 2]);
    /// ```
    pub fn eval_batch(&self, indices: &[Vec<I>]) -> Vec<V> {
        if indices.is_empty() {
            return Vec::new();
        }

        let mut results: Vec<Option<V>> = Vec::with_capacity(indices.len());
        let mut miss_positions: Vec<usize> = Vec::new();
        let mut miss_indices: Vec<Vec<I>> = Vec::new();

        for (pos, idx) in indices.iter().enumerate() {
            if let Some(value) = self.cache.get(idx) {
                self.num_cache_hits.fetch_add(1, Ordering::Relaxed);
                results.push(Some(value));
            } else {
                results.push(None);
                miss_positions.push(pos);
                miss_indices.push(idx.clone());
            }
        }

        if miss_indices.is_empty() {
            return results.into_iter().flatten().collect();
        }

        self.num_evals
            .fetch_add(miss_indices.len(), Ordering::Relaxed);
        let miss_values = if let Some(batch_func) = self.batch_func.as_ref() {
            batch_func(&miss_indices)
        } else {
            miss_indices.iter().map(|idx| (self.func)(idx)).collect()
        };

        for (i, pos) in miss_positions.iter().enumerate() {
            self.cache.insert(&miss_indices[i], miss_values[i].clone());
            results[*pos] = Some(miss_values[i].clone());
        }

        results.into_iter().flatten().collect()
    }

    /// Get the local dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| 0, &[3, 4, 5]).unwrap();
    /// assert_eq!(cf.local_dims(), &[3, 4, 5]);
    /// ```
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Get the number of sites (length of the multi-index).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| 0, &[2, 3]).unwrap();
    /// assert_eq!(cf.num_sites(), 2);
    /// ```
    pub fn num_sites(&self) -> usize {
        self.local_dims.len()
    }

    /// Get the number of function evaluations (cache misses).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// cf.eval(&[0]);
    /// cf.eval(&[1]);
    /// cf.eval(&[0]); // cache hit, not a new eval
    /// assert_eq!(cf.num_evals(), 2);
    /// ```
    pub fn num_evals(&self) -> usize {
        self.num_evals.load(Ordering::Relaxed)
    }

    /// Get the number of cache hits.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// cf.eval(&[0]);
    /// assert_eq!(cf.num_cache_hits(), 0);
    /// cf.eval(&[0]);
    /// assert_eq!(cf.num_cache_hits(), 1);
    /// ```
    pub fn num_cache_hits(&self) -> usize {
        self.num_cache_hits.load(Ordering::Relaxed)
    }

    /// Get total calls (evaluations + cache hits).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// cf.eval(&[0]);
    /// cf.eval(&[1]);
    /// cf.eval(&[0]); // cache hit
    /// assert_eq!(cf.total_calls(), 3);
    /// assert_eq!(cf.total_calls(), cf.num_evals() + cf.num_cache_hits());
    /// ```
    pub fn total_calls(&self) -> usize {
        self.num_evals() + self.num_cache_hits()
    }

    /// Get cache hit ratio (0.0 when no calls have been made).
    ///
    /// Returns `num_cache_hits() / total_calls()` as a value in `[0.0, 1.0]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// assert_eq!(cf.cache_hit_ratio(), 0.0); // no calls yet
    ///
    /// cf.eval(&[0]);
    /// cf.eval(&[0]); // cache hit
    /// assert!((cf.cache_hit_ratio() - 0.5).abs() < 1e-10);
    /// ```
    pub fn cache_hit_ratio(&self) -> f64 {
        let total = self.total_calls();
        if total == 0 {
            0.0
        } else {
            self.num_cache_hits() as f64 / total as f64
        }
    }

    /// Clear the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// cf.eval(&[2]);
    /// assert_eq!(cf.cache_size(), 1);
    /// cf.clear_cache();
    /// assert_eq!(cf.cache_size(), 0);
    /// ```
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Number of cached entries.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// assert_eq!(cf.cache_size(), 0);
    /// cf.eval(&[0]);
    /// cf.eval(&[1]);
    /// assert_eq!(cf.cache_size(), 2);
    /// cf.eval(&[0]); // cache hit, no new entry
    /// assert_eq!(cf.cache_size(), 2);
    /// ```
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Check if an index is cached.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// let cf = CachedFunction::new(|idx: &[usize]| idx[0], &[4]).unwrap();
    /// assert!(!cf.is_cached(&[1]));
    /// cf.eval(&[1]);
    /// assert!(cf.is_cached(&[1]));
    /// ```
    pub fn is_cached(&self, idx: &[I]) -> bool {
        self.cache.contains(idx)
    }

    /// Internal key type name (for debugging).
    ///
    /// Returns `"u64"`, `"u128"`, `"U256"`, `"U512"`, `"U1024"` for
    /// automatically selected types, or `"custom"` when constructed with
    /// [`with_key_type`](Self::with_key_type).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tcicore::CachedFunction;
    ///
    /// // Small index space uses u64
    /// let cf = CachedFunction::new(|idx: &[usize]| 0, &[2, 3]).unwrap();
    /// assert_eq!(cf.key_type(), "u64");
    /// ```
    pub fn key_type(&self) -> &'static str {
        self.cache.key_type_name()
    }
}

#[cfg(test)]
mod tests;
