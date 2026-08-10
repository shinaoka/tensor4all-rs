//! Algorithm dispatch for MPO contraction
//!
//! Provides a unified `contract` function that dispatches to the appropriate
//! algorithm based on [`ContractionAlgorithm`].

/// Algorithm for MPO contraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionAlgorithm {
    /// Naive contraction (exact but memory-intensive)
    #[default]
    Naive,
    /// Zip-up contraction with on-the-fly compression
    ZipUp,
    /// Variational fitting algorithm
    Fit,
}

use super::contract_fit::{contract_fit, FitOptions};
use super::contract_naive::contract_naive;
use super::contract_zipup::contract_zipup;
use super::contraction::ContractionOptions;
use super::error::Result;
use super::factorize::SVDScalar;
use super::mpo::MPO;
use crate::einsum_helper::EinsumScalar;

/// Unified contraction function with algorithm dispatch
///
/// Contracts two MPOs using the specified algorithm.
///
/// # Arguments
/// * `mpo_a` - First MPO
/// * `mpo_b` - Second MPO
/// * `algorithm` - Which algorithm to use
/// * `options` - Contraction options (tolerance, max_bond_dim, etc.)
///
/// # Returns
/// The contracted MPO
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Example
///
/// ```
/// use tensor4all_simplett::mpo::{contract, MPO, ContractionOptions, ContractionAlgorithm};
///
/// let mpo_a = MPO::<f64>::identity(&[2, 2]).unwrap();
/// let mpo_b = MPO::<f64>::identity(&[2, 2]).unwrap();
/// let options = ContractionOptions::default();
///
/// // Use naive algorithm
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Naive, &options).unwrap();
/// assert_eq!(result.site_dims(), vec![(2, 2), (2, 2)]);
///
/// // Use zip-up algorithm for memory efficiency
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::ZipUp, &options).unwrap();
/// assert_eq!(result.len(), 2);
///
/// // Use variational fitting for controlled bond dimension
/// let result = contract(&mpo_a, &mpo_b, ContractionAlgorithm::Fit, &options).unwrap();
/// assert_eq!(result.len(), 2);
/// ```
pub fn contract<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    algorithm: ContractionAlgorithm,
    options: &ContractionOptions,
) -> Result<MPO<T>>
where
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    match algorithm {
        ContractionAlgorithm::Naive => contract_naive(mpo_a, mpo_b, Some(options.clone())),
        ContractionAlgorithm::ZipUp => contract_zipup(mpo_a, mpo_b, options),
        ContractionAlgorithm::Fit => {
            let fit_options = FitOptions {
                tolerance: options.tolerance,
                max_bond_dim: options.max_bond_dim,
                factorize_method: options.factorize_method,
                ..Default::default()
            };
            contract_fit(mpo_a, mpo_b, &fit_options, None)
        }
    }
}

#[cfg(test)]
mod tests;
