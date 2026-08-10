//! Contraction struct with caching for efficient MPO evaluation
//!
//! The Contraction struct wraps two MPOs and provides efficient
//! evaluation with memoization of left and right environments.

use std::collections::HashMap;

use super::error::{MPOError, Result};
use super::factorize::SVDScalar;
use super::matrix2_zeros;
use super::mpo::MPO;
use super::types::Tensor4Ops;
use super::Matrix2;

/// Options for contraction operations
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Relative truncation tolerance forwarded to
    /// [`FactorizeOptions::tolerance`](super::factorize::FactorizeOptions::tolerance).
    ///
    /// Singular values smaller than `tolerance * sigma_max` are discarded at
    /// each bond, where `sigma_max` is the largest singular value of that
    /// bond's matrix. The threshold is invariant under a global rescaling of
    /// the operators being contracted.
    pub tolerance: f64,
    /// Maximum bond dimension after contraction
    pub max_bond_dim: usize,
    /// Factorization method for compression
    pub factorize_method: super::factorize::FactorizeMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            factorize_method: super::factorize::FactorizeMethod::SVD,
        }
    }
}

/// Contraction of two MPOs with caching
/// This struct efficiently computes the product of two MPOs by
/// caching left and right environments for reuse.
pub struct Contraction<T: SVDScalar>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    /// First MPO
    mpo_a: MPO<T>,
    /// Second MPO
    mpo_b: MPO<T>,
    /// Cache for left environments, keyed by index sets
    left_cache: HashMap<Vec<(usize, usize)>, Matrix2<T>>,
    /// Cache for right environments, keyed by index sets
    right_cache: HashMap<Vec<(usize, usize)>, Matrix2<T>>,
    /// Optional transformation function applied to result
    transform_fn: Option<Box<dyn Fn(T) -> T + Send + Sync>>,
    /// Site dimensions for both MPOs [(s1_a, s2_a, s1_b, s2_b), ...]
    site_dims: Vec<(usize, usize, usize, usize)>,
}

impl<T: SVDScalar> Contraction<T>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    /// Create a new Contraction from two MPOs
    /// # Errors
    ///
    /// Returns an error when the two MPOs differ in length (a shape mismatch)
    /// /// or the contraction setup fails.
    ///
    pub fn new(mpo_a: MPO<T>, mpo_b: MPO<T>) -> Result<Self> {
        if mpo_a.len() != mpo_b.len() {
            return Err(MPOError::LengthMismatch {
                expected: mpo_a.len(),
                got: mpo_b.len(),
            });
        }

        // Validate shared dimensions
        for i in 0..mpo_a.len() {
            let (_s1_a, s2_a) = mpo_a.site_dim(i);
            let (s1_b, _s2_b) = mpo_b.site_dim(i);
            if s2_a != s1_b {
                return Err(MPOError::SharedDimensionMismatch {
                    site: i,
                    dim_a: s2_a,
                    dim_b: s1_b,
                });
            }
        }

        let site_dims: Vec<_> = (0..mpo_a.len())
            .map(|i| {
                let (s1_a, s2_a) = mpo_a.site_dim(i);
                let (s1_b, s2_b) = mpo_b.site_dim(i);
                (s1_a, s2_a, s1_b, s2_b)
            })
            .collect();

        Ok(Self {
            mpo_a,
            mpo_b,
            left_cache: HashMap::new(),
            right_cache: HashMap::new(),
            transform_fn: None,
            site_dims,
        })
    }

    /// Create a new Contraction with a transformation function
    /// # Errors
    ///
    /// Returns an error when the two MPOs differ in length (a shape mismatch)
    /// /// or the contraction setup fails.
    ///
    pub fn with_transform<F>(mpo_a: MPO<T>, mpo_b: MPO<T>, f: F) -> Result<Self>
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        let mut contraction = Self::new(mpo_a, mpo_b)?;
        contraction.transform_fn = Some(Box::new(f));
        Ok(contraction)
    }

    /// Get the number of sites
    pub fn len(&self) -> usize {
        self.mpo_a.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.mpo_a.is_empty()
    }

    /// Get site dimensions for the contracted result
    ///
    /// Returns (s1_result, s2_result) at each site where:
    /// - s1_result = s1_a (first physical index of A)
    /// - s2_result = s2_b (second physical index of B)
    pub fn result_site_dims(&self) -> Vec<(usize, usize)> {
        self.site_dims
            .iter()
            .map(|&(s1_a, _, _, s2_b)| (s1_a, s2_b))
            .collect()
    }

    /// Clear all cached environments
    pub fn clear_cache(&mut self) {
        self.left_cache.clear();
        self.right_cache.clear();
    }

    /// Evaluate the contraction at a specific set of indices
    ///
    /// indices should be [(i1, j1), (i2, j2), ...] where:
    /// - i_k is the index for s1 of MPO A at site k
    /// - j_k is the index for s2 of MPO B at site k
    /// # Errors
    ///
    /// Returns an error when the contraction evaluation fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn evaluate(&mut self, indices: &[(usize, usize)]) -> Result<T> {
        if indices.len() != self.len() {
            return Err(MPOError::InvalidOperation {
                message: format!("Expected {} index pairs, got {}", self.len(), indices.len()),
            });
        }

        if self.is_empty() {
            return Err(MPOError::Empty);
        }

        // Contract from left to right
        let first_a = self.mpo_a.site_tensor(0);
        let first_b = self.mpo_b.site_tensor(0);
        let (i0, j0) = indices[0];

        // Sum over shared index k
        let mut current: Matrix2<T> = matrix2_zeros(first_a.right_dim(), first_b.right_dim());
        for k in 0..first_a.site_dim_2() {
            for ra in 0..first_a.right_dim() {
                for rb in 0..first_b.right_dim() {
                    current[[ra, rb]] = current[[ra, rb]]
                        + *first_a.get4(0, i0, k, ra) * *first_b.get4(0, k, j0, rb);
                }
            }
        }

        // Contract through remaining sites
        #[allow(clippy::needless_range_loop)]
        for site in 1..self.len() {
            let a = self.mpo_a.site_tensor(site);
            let b = self.mpo_b.site_tensor(site);
            let (i_k, j_k) = indices[site];

            let mut new_current: Matrix2<T> = matrix2_zeros(a.right_dim(), b.right_dim());

            for la in 0..a.left_dim() {
                for lb in 0..b.left_dim() {
                    let c_val = current[[la, lb]];
                    for k in 0..a.site_dim_2() {
                        for ra in 0..a.right_dim() {
                            for rb in 0..b.right_dim() {
                                new_current[[ra, rb]] = new_current[[ra, rb]]
                                    + c_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                            }
                        }
                    }
                }
            }

            current = new_current;
        }

        let result = current[[0, 0]];

        // Apply transformation if present
        let result = if let Some(ref f) = self.transform_fn {
            f(result)
        } else {
            result
        };

        Ok(result)
    }

    /// Evaluate the left environment up to site n (exclusive)
    ///
    /// Returns L\[n\] = product of sites 0..n
    /// # Errors
    ///
    /// Returns an error when the left evaluation fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn evaluate_left(&mut self, n: usize, indices: &[(usize, usize)]) -> Result<Matrix2<T>> {
        if n > self.len() {
            return Err(MPOError::InvalidOperation {
                message: format!("Site {} is out of range [0, {}]", n, self.len()),
            });
        }

        if n == 0 {
            let mut env: Matrix2<T> = matrix2_zeros(1, 1);
            env[[0, 0]] = T::one();
            return Ok(env);
        }

        // Check cache
        let key: Vec<(usize, usize)> = indices[..n].to_vec();
        if let Some(cached) = self.left_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let prev_env = self.evaluate_left(n - 1, indices)?;
        let a = self.mpo_a.site_tensor(n - 1);
        let b = self.mpo_b.site_tensor(n - 1);
        let (i_k, j_k) = indices[n - 1];

        let mut new_env: Matrix2<T> = matrix2_zeros(a.right_dim(), b.right_dim());

        for la in 0..a.left_dim() {
            for lb in 0..b.left_dim() {
                let l_val = prev_env[[la, lb]];
                for k in 0..a.site_dim_2() {
                    for ra in 0..a.right_dim() {
                        for rb in 0..b.right_dim() {
                            new_env[[ra, rb]] = new_env[[ra, rb]]
                                + l_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                        }
                    }
                }
            }
        }

        // Cache the result
        self.left_cache.insert(key, new_env.clone());

        Ok(new_env)
    }

    /// Evaluate the right environment from site n (exclusive) to the end
    ///
    /// Returns R\[n\] = product of sites n..L
    /// # Errors
    ///
    /// Returns an error when the right evaluation fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn evaluate_right(&mut self, n: usize, indices: &[(usize, usize)]) -> Result<Matrix2<T>> {
        let len = self.len();
        if n > len {
            return Err(MPOError::InvalidOperation {
                message: format!("Site {} is out of range [0, {}]", n, len),
            });
        }

        if n == len {
            let mut env: Matrix2<T> = matrix2_zeros(1, 1);
            env[[0, 0]] = T::one();
            return Ok(env);
        }

        // Check cache
        let key: Vec<(usize, usize)> = indices[n..].to_vec();
        if let Some(cached) = self.right_cache.get(&key) {
            return Ok(cached.clone());
        }

        // Compute recursively
        let prev_env = self.evaluate_right(n + 1, indices)?;
        let a = self.mpo_a.site_tensor(n);
        let b = self.mpo_b.site_tensor(n);
        let (i_k, j_k) = indices[n];

        let mut new_env: Matrix2<T> = matrix2_zeros(a.left_dim(), b.left_dim());

        for ra in 0..a.right_dim() {
            for rb in 0..b.right_dim() {
                let r_val = prev_env[[ra, rb]];
                for k in 0..a.site_dim_2() {
                    for la in 0..a.left_dim() {
                        for lb in 0..b.left_dim() {
                            new_env[[la, lb]] = new_env[[la, lb]]
                                + r_val * *a.get4(la, i_k, k, ra) * *b.get4(lb, k, j_k, rb);
                        }
                    }
                }
            }
        }

        // Cache the result
        self.right_cache.insert(key, new_env.clone());

        Ok(new_env)
    }
}

#[cfg(test)]
mod tests;
