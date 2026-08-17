//! Variational fitting algorithm for MPO contraction.
//!
//! The variational update is not implemented for the simplett MPO type yet.
//! [`contract_fit`] therefore reports [`MPOError::Unsupported`] instead of
//! returning a naive contraction under the misleading `Fit` label.

use super::error::{MPOError, Result};
use super::factorize::{FactorizeMethod, SVDScalar};
use super::mpo::MPO;
use crate::einsum_helper::EinsumScalar;

/// Configuration reserved for the simplett variational MPO fitting algorithm.
///
/// [`contract_fit`] currently returns [`MPOError::Unsupported`] for valid
/// inputs, so these fields have no effect until the variational update is
/// implemented. Use [`super::contract_naive::contract_naive`] or
/// [`super::contract_zipup::contract_zipup`] for an available contraction.
#[derive(Debug, Clone)]
pub struct FitOptions {
    /// Relative truncation tolerance at each step, forwarded to
    /// [`FactorizeOptions::tolerance`](super::factorize::FactorizeOptions::tolerance)
    /// once fitting is implemented.
    pub tolerance: f64,
    /// Maximum bond dimension. `None` means no limit once fitting is
    /// implemented.
    pub max_bond_dim: Option<usize>,
    /// Maximum number of sweeps once fitting is implemented.
    pub max_sweeps: usize,
    /// Convergence tolerance once fitting is implemented.
    pub convergence_tol: f64,
    /// Factorization method once fitting is implemented.
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

/// Attempt variational fitting of the product of two MPOs.
///
/// The simplett variational update is not implemented. This function fails
/// loudly rather than returning the naive contraction as if it were a fitted
/// result.
///
/// # Arguments
/// * `mpo_a` - First MPO.
/// * `mpo_b` - Second MPO.
/// * `options` - Reserved fitting options; currently ignored after validation.
/// * `initial` - Reserved initial guess; currently ignored after validation.
///
/// # Errors
///
/// Returns [`MPOError::LengthMismatch`] or
/// [`MPOError::SharedDimensionMismatch`] for invalid inputs. For every valid
/// input, including an empty MPO, returns [`MPOError::Unsupported`] until the
/// variational update is implemented.
pub fn contract_fit<T: SVDScalar + EinsumScalar>(
    mpo_a: &MPO<T>,
    mpo_b: &MPO<T>,
    _options: &FitOptions,
    _initial: Option<MPO<T>>,
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

    for i in 0..mpo_a.len() {
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

    Err(MPOError::Unsupported {
        message: "simplett variational MPO fitting is not implemented; use contract_naive or contract_zipup instead".to_owned(),
    })
}

#[cfg(test)]
mod tests;
