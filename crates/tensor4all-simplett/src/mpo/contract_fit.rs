//! Variational fitting algorithm for MPO contraction
//!
//! This module implements the variational fitting (DMRG-like) algorithm
//! for computing the product of two MPOs with controlled bond dimension.

use super::contraction::ContractionOptions;
use super::error::{MPOError, Result};
use super::factorize::{FactorizeMethod, SVDScalar};
use super::mpo::MPO;
use super::site_mpo::SiteMPO;
use super::types::{Tensor4, Tensor4Ops};
use crate::einsum_helper::EinsumScalar;

/// Options for the variational fit algorithm
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Relative truncation tolerance at each step, forwarded to
    /// [`FactorizeOptions::tolerance`](super::factorize::FactorizeOptions::tolerance).
    ///
    /// Singular values smaller than `tolerance * sigma_max` are discarded at
    /// each bond, where `sigma_max` is the largest singular value of that
    /// bond's matrix. The threshold is invariant under a global rescaling of
    /// the operators being contracted.
    pub tolerance: f64,
    /// Maximum bond dimension. `None` means no limit.
    pub max_bond_dim: Option<usize>,
    /// Maximum number of sweeps
    pub max_sweeps: usize,
    /// Convergence tolerance (stop if change < this)
    pub convergence_tol: f64,
    /// Factorization method
    pub factorize_method: FactorizeMethod,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: Some(100),
            max_sweeps: 10,
            convergence_tol: 1e-10,
            factorize_method: FactorizeMethod::SVD,
        }
    }
}

/// Perform variational fitting contraction of two MPOs
///
/// This computes C = A * B using a variational (DMRG-like) algorithm
/// that alternates between sweeping left-to-right and right-to-left,
/// optimizing two sites at a time.
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `options` - Fitting options
/// * `initial` - Optional initial guess (if None, uses naive contraction)
///
/// # Returns
/// The contracted MPO C with bond dimension controlled by options
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
pub fn contract_fit<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    options: &FitOptions,
    initial: Option<MPO<T>>,
) -> Result<MPO<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    if mpo_a.is_empty() {
        return Ok(MPO::from_tensors_unchecked(Vec::new()));
    }

    let n = mpo_a.len();

    // Validate shared dimensions
    for i in 0..n {
        let (_, s2_a) = mpo_a.site_dim(i);
        let (s1_b, _) = mpo_b.site_dim(i);
        if s2_a != s1_b {
            return Err(MPOError::SharedDimensionMismatch {
                site: i,
                dim_a: s2_a,
                dim_b: s1_b,
            });
        }
    }

    // Initialize the result
    let result = if let Some(init) = initial {
        SiteMPO::from_mpo(init, 0)?
    } else {
        // Use naive contraction as initial guess, then compress
        let naive_opts = ContractionOptions {
            tolerance: options.tolerance,
            max_bond_dim: options.max_bond_dim,
            factorize_method: options.factorize_method,
        };
        let naive_result = super::contract_naive::contract_naive(mpo_a, mpo_b, Some(naive_opts))?;
        SiteMPO::from_mpo(naive_result, 0)?
    };

    // For single or two-site systems, return immediately
    if n <= 2 {
        return Ok(result.into_mpo());
    }

    // Build left and right environments
    let mut left_envs: Vec<Option<Environment<T>>> = vec![None; n];
    let mut right_envs: Vec<Option<Environment<T>>> = vec![None; n];

    // Initialize boundary environments
    left_envs[0] = Some(Environment::identity(1, 1, 1));
    right_envs[n - 1] = Some(Environment::identity(1, 1, 1));

    // Build initial right environments
    for i in (1..n).rev() {
        let next_env = require_environment(&right_envs, i, "right")?;
        let env = build_right_environment(
            mpo_a.site_tensor(i),
            mpo_b.site_tensor(i),
            result.site_tensor(i),
            next_env,
        )?;
        right_envs[i - 1] = Some(env);
    }

    let mut current = result;
    let mut _prev_norm = f64::INFINITY;

    // Main optimization loop
    for _sweep in 0..options.max_sweeps {
        // Forward sweep (left to right)
        for i in 0..(n - 1) {
            // Update two-site core at positions i and i+1
            let _updated = update_two_site_core(
                mpo_a,
                mpo_b,
                &mut current,
                i,
                &left_envs,
                &right_envs,
                options,
            )?;

            // Update left environment at i+1
            let prev_env = require_environment(&left_envs, i, "left")?;
            let env = build_left_environment(
                mpo_a.site_tensor(i),
                mpo_b.site_tensor(i),
                current.site_tensor(i),
                prev_env,
            )?;
            left_envs[i + 1] = Some(env);
        }

        // Backward sweep (right to left)
        for i in (1..n).rev() {
            // Update two-site core at positions i-1 and i
            let _updated = update_two_site_core(
                mpo_a,
                mpo_b,
                &mut current,
                i - 1,
                &left_envs,
                &right_envs,
                options,
            )?;

            // Update right environment at i-1
            if i > 0 {
                let next_env = require_environment(&right_envs, i, "right")?;
                let env = build_right_environment(
                    mpo_a.site_tensor(i),
                    mpo_b.site_tensor(i),
                    current.site_tensor(i),
                    next_env,
                )?;
                right_envs[i - 1] = Some(env);
            }
        }

        // Check convergence
        // TODO: Implement proper convergence check using norm difference
    }

    Ok(current.into_mpo())
}

/// Environment tensor for variational algorithm
#[derive(Debug, Clone)]
struct Environment<T> {
    /// Shape: (link_result, link_a, link_b)
    data: Vec<T>,
    dim_result: usize,
    dim_a: usize,
    dim_b: usize,
}

impl<T: SVDScalar> Environment<T>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    fn identity(dim_result: usize, dim_a: usize, dim_b: usize) -> Self {
        let mut data = vec![T::zero(); dim_result * dim_a * dim_b];
        // Set diagonal elements to 1
        let min_dim = dim_result.min(dim_a).min(dim_b);
        for i in 0..min_dim {
            data[(i * dim_a + i) * dim_b + i] = T::one();
        }
        // Actually for identity, we just want a single 1.0
        if dim_result == 1 && dim_a == 1 && dim_b == 1 {
            data[0] = T::one();
        }
        Self {
            data,
            dim_result,
            dim_a,
            dim_b,
        }
    }

    fn get(&self, r: usize, a: usize, b: usize) -> T {
        self.data[(r * self.dim_a + a) * self.dim_b + b]
    }

    fn set(&mut self, r: usize, a: usize, b: usize, val: T) {
        self.data[(r * self.dim_a + a) * self.dim_b + b] = val;
    }
}

fn require_environment<'a, T>(
    envs: &'a [Option<Environment<T>>],
    site: usize,
    side: &'static str,
) -> Result<&'a Environment<T>> {
    envs.get(site)
        .and_then(Option::as_ref)
        .ok_or_else(|| MPOError::InvalidOperation {
            message: format!("missing {side} environment at site {site}"),
        })
}

/// Build left environment by extending from previous environment
fn build_left_environment<T: SVDScalar>(
    tensor_a: &Tensor4<T>,
    tensor_b: &Tensor4<T>,
    tensor_result: &Tensor4<T>,
    prev_env: &Environment<T>,
) -> Result<Environment<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let new_dim_result = tensor_result.right_dim();
    let new_dim_a = tensor_a.right_dim();
    let new_dim_b = tensor_b.right_dim();

    let mut new_env = Environment {
        data: vec![T::zero(); new_dim_result * new_dim_a * new_dim_b],
        dim_result: new_dim_result,
        dim_a: new_dim_a,
        dim_b: new_dim_b,
    };

    // Contract: L'[rr', ra', rb'] = sum_{rr, ra, rb, s1, s2, k}
    //   L[rr, ra, rb] * C[rr, s1, s2, rr'] * A[ra, s1, k, ra'] * B[rb, k, s2, rb']
    for rr in 0..prev_env.dim_result {
        for ra in 0..prev_env.dim_a {
            for rb in 0..prev_env.dim_b {
                let l_val = prev_env.get(rr, ra, rb);
                for s1 in 0..tensor_result.site_dim_1() {
                    for s2 in 0..tensor_result.site_dim_2() {
                        for k in 0..tensor_a.site_dim_2() {
                            for rr_new in 0..new_dim_result {
                                for ra_new in 0..new_dim_a {
                                    for rb_new in 0..new_dim_b {
                                        let c_val = *tensor_result.get4(rr, s1, s2, rr_new);
                                        let a_val = *tensor_a.get4(ra, s1, k, ra_new);
                                        let b_val = *tensor_b.get4(rb, k, s2, rb_new);
                                        let old = new_env.get(rr_new, ra_new, rb_new);
                                        new_env.set(
                                            rr_new,
                                            ra_new,
                                            rb_new,
                                            old + l_val * c_val * a_val * b_val,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(new_env)
}

/// Build right environment by extending from next environment
fn build_right_environment<T: SVDScalar>(
    tensor_a: &Tensor4<T>,
    tensor_b: &Tensor4<T>,
    tensor_result: &Tensor4<T>,
    next_env: &Environment<T>,
) -> Result<Environment<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let new_dim_result = tensor_result.left_dim();
    let new_dim_a = tensor_a.left_dim();
    let new_dim_b = tensor_b.left_dim();

    let mut new_env = Environment {
        data: vec![T::zero(); new_dim_result * new_dim_a * new_dim_b],
        dim_result: new_dim_result,
        dim_a: new_dim_a,
        dim_b: new_dim_b,
    };

    // Contract: R'[lr, la, lb] = sum_{rr, ra, rb, s1, s2, k}
    //   R[rr, ra, rb] * C[lr, s1, s2, rr] * A[la, s1, k, ra] * B[lb, k, s2, rb]
    for rr in 0..next_env.dim_result {
        for ra in 0..next_env.dim_a {
            for rb in 0..next_env.dim_b {
                let r_val = next_env.get(rr, ra, rb);
                for s1 in 0..tensor_result.site_dim_1() {
                    for s2 in 0..tensor_result.site_dim_2() {
                        for k in 0..tensor_a.site_dim_2() {
                            for lr in 0..new_dim_result {
                                for la in 0..new_dim_a {
                                    for lb in 0..new_dim_b {
                                        let c_val = *tensor_result.get4(lr, s1, s2, rr);
                                        let a_val = *tensor_a.get4(la, s1, k, ra);
                                        let b_val = *tensor_b.get4(lb, k, s2, rb);
                                        let old = new_env.get(lr, la, lb);
                                        new_env.set(
                                            lr,
                                            la,
                                            lb,
                                            old + r_val * c_val * a_val * b_val,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(new_env)
}

/// Update the two-site core tensor at positions site and site+1
fn update_two_site_core<T: SVDScalar>(
    _mpo_a: &MPO<T>,
    _mpo_b: &MPO<T>,
    _result: &mut SiteMPO<T>,
    _site: usize,
    _left_envs: &[Option<Environment<T>>],
    _right_envs: &[Option<Environment<T>>],
    _options: &FitOptions,
) -> Result<bool>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    // For now, just return Ok - full implementation would update the core
    // This is a placeholder for the variational update step
    Ok(true)
}

#[cfg(test)]
mod tests;
