//! Environment computation for MPO contractions
//!
//! This module provides functions for computing left and right environments
//! used in variational algorithms and efficient MPO evaluation.

use crate::einsum_helper::{einsum_tensors, typed_tensor_reshape, EinsumScalar};

use super::error::{MPOError, Result};
use super::factorize::SVDScalar;
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use super::{matrix2_zeros, Matrix2};

/// Wrap a helper-layer error (einsum, reshape, backend linear algebra) into an
/// `MPOError` with the caller's context preserved for the C API and bindings.
pub(crate) fn mpo_helper_error(context: &str, err: impl std::fmt::Display) -> MPOError {
    MPOError::InvalidOperation {
        message: format!("{context}: {err}"),
    }
}

/// Contract two 4D site tensors over their shared physical index
///
/// Given two 4D tensors:
/// - A: (left_a, s1_a, s2_a, right_a) where s2_a is the shared index
/// - B: (left_b, s1_b, s2_b, right_b) where s1_b is the shared index
///
/// Contracts over s2_a = s1_b to produce:
/// - C: (left_a * left_b, s1_a, s2_b, right_a * right_b)
///
/// This is the Rust equivalent of `_contractsitetensors` from Julia.
/// # Errors
///
/// Returns an error when the site tensors are incompatible (a shape or index
/// /// mismatch) or the contraction fails (a backend failure).
///
pub fn contract_site_tensors<T: SVDScalar + EinsumScalar>(
    a: &Tensor4<T>,
    b: &Tensor4<T>,
) -> Result<Tensor4<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    // Check that shared dimensions match
    if a.site_dim_2() != b.site_dim_1() {
        return Err(MPOError::SharedDimensionMismatch {
            site: 0,
            dim_a: a.site_dim_2(),
            dim_b: b.site_dim_1(),
        });
    }

    let left_a = a.left_dim();
    let s1_a = a.site_dim_1();
    let right_a = a.right_dim();

    let left_b = b.left_dim();
    let s2_b = b.site_dim_2();
    let right_b = b.right_dim();

    // Result dimensions
    let new_left = left_a * left_b;
    let new_s1 = s1_a;
    let new_s2 = s2_b;
    let new_right = right_a * right_b;

    // Arrange the open bond indices as (left_b, left_a, ..., right_b, right_a)
    // so the column-major reshape preserves the existing la * left_b + lb and
    // ra * right_b + rb indexing.
    //
    // TODO: Remove this materialization once tenferro supports reshaping
    // layout-compatible strided views directly.
    // Tracking issue: https://github.com/tensor4all/tenferro-rs/issues/575
    let contracted = einsum_tensors("askr,bktq->bastqr", &[a.as_inner(), b.as_inner()])
        .map_err(|err| mpo_helper_error("Failed to contract MPO site tensors", err))?;
    let reshaped = typed_tensor_reshape(&contracted, &[new_left, new_s1, new_s2, new_right])
        .map_err(|err| mpo_helper_error("Failed to reshape contracted MPO site tensor", err))?;

    Tensor4::from_tenferro(reshaped)
        .map_err(|err| mpo_helper_error("Contracted MPO site tensor has invalid rank", err))
}

/// Compute the left environment at site i for MPO contraction
///
/// The left environment L\[i\] represents the contraction of all sites 0..i
/// for the product of two MPOs A and B.
///
/// L\[i\] has shape (left_a_i, left_b_i) representing the accumulated
/// contraction from the left.
/// # Errors
///
/// Returns an error when the environment construction fails (a shape or index
/// /// mismatch, or a backend failure).
///
pub fn left_environment<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    site: usize,
    cache: &mut Vec<Option<Matrix2<T>>>,
) -> Result<Matrix2<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    // Base case: left of site 0 is just [[1]]
    if site == 0 {
        let mut env: Matrix2<T> = matrix2_zeros(1, 1);
        env[[0, 0]] = T::one();
        return Ok(env);
    }

    // Check cache
    if let Some(Some(cached)) = cache.get(site - 1) {
        return Ok(cached.clone());
    }

    // Recursively compute from the left
    let prev_env = left_environment(mpo_a, mpo_b, site - 1, cache)?;
    let a = mpo_a.site_tensor(site - 1);
    let b = mpo_b.site_tensor(site - 1);

    // Contract: L[i-1] with A[i-1] and B[i-1]
    // L[i-1]: (left_a, left_b)
    // A[i-1]: (left_a, s1_a, s2_a, right_a)
    // B[i-1]: (left_b, s1_b, s2_b, right_b)
    //
    // Sum over left_a, left_b, s1_a (=s1_b), s2_a (=s2_b) to get:
    // L[i]: (right_a, right_b)

    // Check shared dimensions
    if a.site_dim_1() != b.site_dim_1() || a.site_dim_2() != b.site_dim_2() {
        return Err(MPOError::SharedDimensionMismatch {
            site: site - 1,
            dim_a: a.site_dim_1(),
            dim_b: b.site_dim_1(),
        });
    }

    let new_env_tensor = einsum_tensors(
        "ab,asdr,bsdt->rt",
        &[prev_env.as_inner(), a.as_inner(), b.as_inner()],
    )
    .map_err(|err| mpo_helper_error("Failed to compute left MPO environment", err))?;
    let new_env = Matrix2::from_tenferro(new_env_tensor)
        .map_err(|err| mpo_helper_error("Left MPO environment has invalid rank", err))?;

    // Update cache
    while cache.len() < site {
        cache.push(None);
    }
    cache[site - 1] = Some(new_env.clone());

    Ok(new_env)
}

/// Compute the right environment at site i for MPO contraction
///
/// The right environment R\[i\] represents the contraction of all sites i+1..L
/// for the product of two MPOs A and B.
///
/// R\[i\] has shape (right_a_i, right_b_i) representing the accumulated
/// contraction from the right.
/// # Errors
///
/// Returns an error when the environment construction fails (a shape or index
/// /// mismatch, or a backend failure).
///
pub fn right_environment<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    site: usize,
    cache: &mut Vec<Option<Matrix2<T>>>,
) -> Result<Matrix2<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo_a.len() != mpo_b.len() {
        return Err(MPOError::LengthMismatch {
            expected: mpo_a.len(),
            got: mpo_b.len(),
        });
    }

    let n = mpo_a.len();

    // Base case: right of last site is just [[1]]
    if site == n - 1 {
        let mut env: Matrix2<T> = matrix2_zeros(1, 1);
        env[[0, 0]] = T::one();
        return Ok(env);
    }

    // Check cache
    let cache_idx = n - site - 2;
    if let Some(Some(cached)) = cache.get(cache_idx) {
        return Ok(cached.clone());
    }

    // Recursively compute from the right
    let prev_env = right_environment(mpo_a, mpo_b, site + 1, cache)?;
    let a = mpo_a.site_tensor(site + 1);
    let b = mpo_b.site_tensor(site + 1);

    // Contract: R[i+1] with A[i+1] and B[i+1]
    // R[i+1]: (right_a, right_b)
    // A[i+1]: (left_a, s1_a, s2_a, right_a)
    // B[i+1]: (left_b, s1_b, s2_b, right_b)
    //
    // Sum over right_a, right_b, s1_a (=s1_b), s2_a (=s2_b) to get:
    // R[i]: (left_a, left_b)

    // Check shared dimensions
    if a.site_dim_1() != b.site_dim_1() || a.site_dim_2() != b.site_dim_2() {
        return Err(MPOError::SharedDimensionMismatch {
            site: site + 1,
            dim_a: a.site_dim_1(),
            dim_b: b.site_dim_1(),
        });
    }

    let new_env_tensor = einsum_tensors(
        "rt,asdr,bsdt->ab",
        &[prev_env.as_inner(), a.as_inner(), b.as_inner()],
    )
    .map_err(|err| mpo_helper_error("Failed to compute right MPO environment", err))?;
    let new_env = Matrix2::from_tenferro(new_env_tensor)
        .map_err(|err| mpo_helper_error("Right MPO environment has invalid rank", err))?;

    // Update cache
    while cache.len() <= cache_idx {
        cache.push(None);
    }
    cache[cache_idx] = Some(new_env.clone());

    Ok(new_env)
}

#[cfg(test)]
mod tests;
