//! Backend dispatch helpers for linear algebra operations.
//!
//! This module provides centralized backend selection for SVD and QR operations,
//! reducing code duplication between decomposition implementations.

use anyhow::Result;
use mdarray::{DSlice, DTensor};
use mdarray_linalg::qr::QR;
use mdarray_linalg::svd::{SVDDecomp, SVD};

#[cfg(feature = "backend-faer")]
use mdarray_linalg_faer::Faer;

#[cfg(feature = "backend-lapack")]
use mdarray_linalg_lapack::Lapack;

/// Compute SVD decomposition using the selected backend.
///
/// # Arguments
/// * `a` - Input matrix slice (mutable, may be modified by backend)
///
/// # Returns
/// SVD decomposition result from mdarray-linalg
pub(crate) fn svd_backend<T>(a: &mut DSlice<T, 2>) -> Result<SVDDecomp<T>>
where
    T: num_complex::ComplexFloat
        + faer_traits::ComplexField
        + Default
        + From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    let decomp = {
        #[cfg(feature = "backend-faer")]
        {
            let bd = Faer;
            bd.svd(a)
        }
        #[cfg(feature = "backend-lapack")]
        {
            let bd = Lapack::new();
            bd.svd(a)
        }
        #[cfg(not(any(feature = "backend-faer", feature = "backend-lapack")))]
        {
            compile_error!(
                "At least one backend feature must be enabled (backend-faer or backend-lapack)"
            );
        }
    }
    .map_err(|e| anyhow::anyhow!("SVD computation failed: {}", e))?;

    Ok(decomp)
}

/// Compute QR decomposition using the selected backend.
///
/// # Arguments
/// * `a` - Input matrix slice (mutable, may be modified by backend)
///
/// # Returns
/// Tuple `(Q, R)` where Q is m×m and R is m×n (full QR)
pub(crate) fn qr_backend<T>(a: &mut DSlice<T, 2>) -> (DTensor<T, 2>, DTensor<T, 2>)
where
    T: num_complex::ComplexFloat
        + Default
        + faer_traits::ComplexField
        + From<<T as num_complex::ComplexFloat>::Real>
        + 'static,
    <T as num_complex::ComplexFloat>::Real: Into<f64>,
{
    #[cfg(feature = "backend-faer")]
    {
        let bd = Faer;
        bd.qr(a)
    }
    #[cfg(feature = "backend-lapack")]
    {
        let bd = Lapack::new();
        bd.qr(a)
    }
    #[cfg(not(any(feature = "backend-faer", feature = "backend-lapack")))]
    {
        compile_error!(
            "At least one backend feature must be enabled (backend-faer or backend-lapack)"
        );
    }
}
