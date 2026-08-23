//! Scalar capability for MatrixLUCI implementations.

use num_complex::{Complex32, Complex64};

use crate::error::Result;
use crate::matrix_luci::MatrixLuciFactors;
use crate::matrixlu::RrLUOptions;
use tensor4all_tensorbackend::{
    BackendLinalgScalar, Matrix, MatrixSolveScalar, MatrixTriangularSolveScalar,
};

/// Scalar types supported by MatrixLUCI factorization.
///
/// Common arithmetic comes from [`crate::Scalar`]; this trait adds only the
/// backend solve capabilities and MatrixLUCI dispatch used by the factorizer.
pub trait MatrixLuciScalar:
    crate::Scalar + BackendLinalgScalar + MatrixSolveScalar + MatrixTriangularSolveScalar
{
    #[doc(hidden)]
    fn matrix_luci_factors_from_matrix(
        a: &Matrix<Self>,
        options: RrLUOptions,
    ) -> Result<MatrixLuciFactors<Self>>
    where
        Self: Sized;

    #[doc(hidden)]
    fn matrix_luci_factors_from_blocks<F>(
        nrows: usize,
        ncols: usize,
        fill_block: F,
        options: RrLUOptions,
    ) -> Result<MatrixLuciFactors<Self>>
    where
        F: Fn(&[usize], &[usize], &mut [Self]),
        Self: Sized;
}

macro_rules! impl_matrix_luci_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl MatrixLuciScalar for $ty {
                fn matrix_luci_factors_from_matrix(
                    a: &Matrix<Self>,
                    options: RrLUOptions,
                ) -> Result<MatrixLuciFactors<Self>> {
                    crate::matrix_luci::dense_matrix_luci_factors_from_matrix(a, options)
                }

                fn matrix_luci_factors_from_blocks<F>(
                    nrows: usize,
                    ncols: usize,
                    fill_block: F,
                    options: RrLUOptions,
                ) -> Result<MatrixLuciFactors<Self>>
                where
                    F: Fn(&[usize], &[usize], &mut [Self]),
                {
                    crate::matrix_luci::lazy_matrix_luci_factors_from_blocks(
                        nrows, ncols, fill_block, options,
                    )
                }
            }
        )*
    };
}

impl_matrix_luci_scalar!(f64, f32, Complex64, Complex32);

#[cfg(test)]
mod tests;
