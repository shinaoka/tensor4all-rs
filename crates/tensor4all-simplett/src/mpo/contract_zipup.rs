//! Zip-up MPO contraction algorithm
//!
//! Contracts two MPOs with on-the-fly compression at each step.
//! This is more memory-efficient than naive contraction followed by compression.

use crate::einsum_helper::{einsum_tensors, typed_tensor_reshape, EinsumScalar};

use super::contraction::ContractionOptions;
use super::environment::mpo_helper_error;
use super::error::{MPOError, Result};
use super::factorize::{factorize, FactorizeOptions, SVDScalar};
use super::mpo::MPO;
use super::types::{Tensor4, Tensor4Ops};
use super::Matrix2;
use tenferro_tensor::TypedTensor;

/// Perform zip-up contraction of two MPOs
///
/// This computes C = A * B where the contraction is over the shared
/// physical index (s2 of A contracts with s1 of B), with on-the-fly
/// compression at each step.
///
/// The zip-up algorithm:
/// 1. Start from the left with a remainder tensor R = \[\[1\]\]
/// 2. At each site:
///    a. Contract R with A\[i\] and B\[i\]
///    b. Reshape to matrix
///    c. Factorize into left and right factors
///    d. Store left factor as result tensor
///    e. Use right factor as new remainder R
/// 3. At the last site, just store the contracted tensor
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `options` - Contraction options (tolerance, max_bond_dim, method)
///
/// # Returns
/// The contracted and compressed MPO C
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
pub fn contract_zipup<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    options: &ContractionOptions,
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

    // Remainder tensor R[new_link, link_a, link_b], starting as the 1x1x1
    // scalar one.
    let mut remainder: TypedTensor<T> =
        TypedTensor::from_vec_col_major(vec![1, 1, 1], vec![T::one()]);

    let mut result_tensors: Vec<Tensor4<T>> = Vec::with_capacity(n);

    let factorize_opts = FactorizeOptions {
        method: options.factorize_method,
        tolerance: options.tolerance,
        max_rank: options.max_bond_dim,
        left_orthogonal: true,
        ..Default::default()
    };

    for i in 0..n {
        let a = mpo_a.site_tensor(i);
        let b = mpo_b.site_tensor(i);

        let c_s1 = a.site_dim_1();
        let c_s2 = b.site_dim_2();
        let c_right_a = a.right_dim();
        let c_right_b = b.right_dim();
        let c_new_link = remainder.shape()[0];

        // Contract R with A and then with B:
        //   C[new_link, s1, s2, right_a, right_b]
        //   = sum_{link_a, link_b, k} R[new_link, link_a, link_b]
        //     * A[link_a, s1, k, right_a] * B[link_b, k, s2, right_b]
        //
        // The two contractions are done pairwise on purpose. Evaluating the
        // three-operand sum directly costs
        // O(new_link * link_a * link_b * s1 * s2 * k * right_a * right_b),
        // which is two bond dimensions worse than the pairwise route and was
        // the dominant cost of this function.
        let ra = einsum_tensors("nab,askc->nbskc", &[&remainder, a.as_inner()])
            .map_err(|err| mpo_helper_error("Failed to contract remainder with MPO A", err))?;
        let c = einsum_tensors("nbskc,bktd->nstcd", &[&ra, b.as_inner()])
            .map_err(|err| mpo_helper_error("Failed to contract remainder with MPO B", err))?;

        if i == n - 1 {
            // Last site: the trailing bonds are both 1, so this is a reshape.
            let last = typed_tensor_reshape(&c, &[c_new_link, c_s1, c_s2, 1])
                .map_err(|err| mpo_helper_error("Failed to reshape final zip-up site", err))?;
            result_tensors.push(
                Tensor4::from_tenferro(last)
                    .map_err(|err| mpo_helper_error("Final zip-up site has invalid rank", err))?,
            );
            continue;
        }

        // Column-major data of `(new_link, s1, s2, right_a, right_b)` groups
        // the leading three and the trailing two axes contiguously, so both
        // reshapes below are pure metadata changes.
        let rows = c_new_link * c_s1 * c_s2;
        let cols = c_right_a * c_right_b;
        let c_mat = typed_tensor_reshape(&c, &[rows, cols])
            .map_err(|err| mpo_helper_error("Failed to reshape zip-up site", err))?;
        let c_mat: Matrix2<T> = Matrix2::from_tenferro_unchecked(c_mat);

        let fact_result = factorize(&c_mat, &factorize_opts)?;
        let new_bond_dim = fact_result.rank;

        let site = typed_tensor_reshape(
            fact_result.left.as_inner(),
            &[c_new_link, c_s1, c_s2, new_bond_dim],
        )
        .map_err(|err| mpo_helper_error("Failed to reshape zip-up left factor", err))?;
        result_tensors.push(
            Tensor4::from_tenferro(site)
                .map_err(|err| mpo_helper_error("Zip-up site has invalid rank", err))?,
        );

        remainder = typed_tensor_reshape(
            fact_result.right.as_inner(),
            &[new_bond_dim, c_right_a, c_right_b],
        )
        .map_err(|err| mpo_helper_error("Failed to reshape zip-up right factor", err))?;
    }

    Ok(MPO::from_tensors_unchecked(result_tensors))
}

#[cfg(test)]
mod tests;
