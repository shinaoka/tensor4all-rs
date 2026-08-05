//! Naive MPO contraction algorithm
//!
//! Contracts two MPOs by directly multiplying site tensors,
//! optionally followed by compression.

use super::canonical::right_canonicalize;
use super::contraction::ContractionOptions;
use super::environment::contract_site_tensors;
use super::error::{MPOError, Result};
use super::factorize::{factorize, FactorizeOptions, SVDScalar};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use super::Matrix2;
use crate::einsum_helper::{einsum_tensors, typed_tensor_reshape, EinsumScalar};

/// Perform naive contraction of two MPOs
///
/// This computes C = A * B where the contraction is over the shared
/// physical index (s2 of A contracts with s1 of B).
///
/// The naive algorithm:
/// 1. Contract each pair of site tensors
/// 2. Optionally compress the result
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `options` - Optional compression options
///
/// # Returns
/// The contracted MPO C with dimensions:
/// - s1: from A
/// - s2: from B
/// - bond dimensions: product of input bond dimensions (before compression)
pub fn contract_naive<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    options: Option<ContractionOptions>,
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

    // Contract each pair of site tensors
    let mut tensors: Vec<Tensor4<T>> = Vec::with_capacity(n);

    for i in 0..n {
        let a = mpo_a.site_tensor(i);
        let b = mpo_b.site_tensor(i);

        // Contract over shared index: a.s2 = b.s1
        // Result has shape:
        // (left_a * left_b, s1_a, s2_b, right_a * right_b)
        let contracted = contract_site_tensors(a, b)?;
        tensors.push(contracted);
    }

    let mut result = MPO::from_tensors_unchecked(tensors);

    // Apply compression if options are provided
    if let Some(opts) = options {
        compress_mpo(&mut result, &opts)?;
    }

    Ok(result)
}

/// Compress an MPO using the specified options
///
/// The MPO is first brought into right-canonical form with rank-preserving QR
/// factorizations. Only then is the truncating left-to-right sweep run, so
/// every local SVD truncation sees an orthonormal environment on both sides
/// and discards the singular values that are actually smallest globally.
fn compress_mpo<T: SVDScalar + EinsumScalar>(
    mpo: &mut MPO<T>,
    options: &ContractionOptions,
) -> Result<()>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    if mpo.len() <= 1 {
        return Ok(());
    }

    let factorize_opts = FactorizeOptions {
        method: options.factorize_method,
        tolerance: options.tolerance,
        max_rank: options.max_bond_dim,
        left_orthogonal: true,
        ..Default::default()
    };

    right_canonicalize(mpo)?;

    // Sweep left to right, factorizing each bond
    for i in 0..(mpo.len() - 1) {
        let tensor = mpo.site_tensor(i);
        let left_dim = tensor.left_dim();
        let s1 = tensor.site_dim_1();
        let s2 = tensor.site_dim_2();
        let right_dim = tensor.right_dim();

        // Column-major data of `(left, s1, s2, right)` groups the leading
        // three axes contiguously, so this reshape to `(left * s1 * s2, right)`
        // is a pure metadata change.
        let rows = left_dim * s1 * s2;
        let mat = typed_tensor_reshape(tensor.as_inner(), &[rows, right_dim])
            .map_err(|err| naive_error("Failed to reshape MPO site tensor", err))?;
        let mat: Matrix2<T> = Matrix2::from_tenferro_unchecked(mat);

        let fact_result = factorize(&mat, &factorize_opts)?;
        let new_rank = fact_result.rank;

        let new_tensor =
            typed_tensor_reshape(fact_result.left.as_inner(), &[left_dim, s1, s2, new_rank])
                .map_err(|err| naive_error("Failed to reshape compressed MPO site tensor", err))?;
        let new_tensor = Tensor4::from_tenferro(new_tensor)
            .map_err(|err| naive_error("Compressed MPO site has invalid rank", err))?;

        // Multiply the right factor into the next tensor:
        // R[new_rank, right_dim] @ next[right_dim, s1, s2, next_right]
        let next_tensor = mpo.site_tensor(i + 1);
        let new_next = einsum_tensors(
            "lk,ksqr->lsqr",
            &[fact_result.right.as_inner(), next_tensor.as_inner()],
        )
        .map_err(|err| naive_error("Failed to absorb right factor into MPO site", err))?;
        let new_next = Tensor4::from_tenferro(new_next)
            .map_err(|err| naive_error("Compressed MPO site has invalid rank", err))?;

        *mpo.site_tensor_mut(i) = new_tensor;
        *mpo.site_tensor_mut(i + 1) = new_next;
    }

    Ok(())
}

fn naive_error(context: &str, err: impl std::fmt::Display) -> MPOError {
    MPOError::InvalidOperation {
        message: format!("{context}: {err}"),
    }
}

#[cfg(test)]
mod tests;
