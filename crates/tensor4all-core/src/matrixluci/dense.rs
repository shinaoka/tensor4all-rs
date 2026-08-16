//! Dense pivot-kernel implementations.

use crate::matrixluci::kernel::PivotKernel;
use crate::matrixluci::scalar::Scalar;
use crate::matrixluci::source::{materialize_source, CandidateMatrixSource};
use crate::matrixluci::{MatrixLuciError, PivotKernelOptions, PivotSelectionCore, Result};
use crate::scalar::Scalar as LegacyScalar;
use crate::{rrlu, RrLUOptions};
use num_complex::{Complex32, Complex64};
use tenferro_tensor::TensorScalar;
use tensor4all_tensorbackend::{full_piv_lu_matrix, Matrix};

/// Dense full-pivoting LU kernel backed by the configured tensor backend.
///
/// Materializes the source matrix and performs dense pivot selection. Square
/// matrices use tenferro's complete-pivoting LU; rectangular matrices fall
/// back to the internal rrLU implementation until tenferro exposes rectangular
/// complete pivoting.
#[derive(Default)]
pub struct DenseLuKernel;

impl DenseLuKernel {
    fn is_no_truncation(options: &PivotKernelOptions, full_rank: usize) -> bool {
        options.max_bond_dim >= full_rank && options.rel_tol == 0.0 && options.abs_tol == 0.0
    }

    fn compute_pivot_errors(
        diag_abs: &[f64],
        nrows: usize,
        ncols: usize,
        options: &PivotKernelOptions,
    ) -> Vec<f64> {
        let full_rank = nrows.min(ncols);
        if full_rank == 0 {
            return vec![0.0];
        }

        if Self::is_no_truncation(options, full_rank) {
            let mut pivot_errors = Vec::with_capacity(full_rank + 1);
            for &pivot_abs in diag_abs.iter().take(full_rank) {
                if pivot_abs < f64::EPSILON {
                    if pivot_errors.is_empty() {
                        pivot_errors.push(pivot_abs);
                    } else {
                        pivot_errors.push(0.0);
                    }
                    return pivot_errors;
                }
                pivot_errors.push(pivot_abs);
            }
            pivot_errors.push(0.0);
            return pivot_errors;
        }

        let max_bond_dim = options.max_bond_dim.min(full_rank);
        let mut accepted = Vec::new();
        let mut max_error = 0.0f64;
        let mut last_error = f64::NAN;
        let mut rank = 0usize;

        while rank < max_bond_dim {
            let pivot_abs = diag_abs.get(rank).copied().unwrap_or(0.0);
            last_error = pivot_abs;

            if rank > 0 && (pivot_abs < options.rel_tol * max_error || pivot_abs < options.abs_tol)
            {
                break;
            }

            if pivot_abs < f64::EPSILON {
                if rank == 0 {
                    last_error = pivot_abs;
                }
                break;
            }

            max_error = max_error.max(pivot_abs);
            accepted.push(pivot_abs);
            rank += 1;
        }

        if rank >= full_rank {
            last_error = 0.0;
        } else if rank == max_bond_dim && rank > 0 {
            // Preserve the legacy tcicore-compatible semantics for max_bond_dim stopping.
            last_error = accepted[rank - 1];
        }

        accepted.push(last_error);
        accepted
    }

    fn factorize_with_rrlu<T: Scalar + LegacyScalar>(
        data: &[T],
        nrows: usize,
        ncols: usize,
        options: &PivotKernelOptions,
    ) -> Result<PivotSelectionCore> {
        let lu_options = RrLUOptions {
            max_bond_dim: options.max_bond_dim,
            rel_tol: options.rel_tol,
            abs_tol: options.abs_tol,
            left_orthogonal: options.left_orthogonal,
        };
        let matrix = Matrix::from_col_major_vec(nrows, ncols, data.to_vec());
        let lu =
            rrlu(&matrix, Some(lu_options)).map_err(|err| MatrixLuciError::InvalidArgument {
                message: format!("dense rrLU factorization failed: {err}"),
            })?;

        Ok(PivotSelectionCore {
            row_indices: lu.row_indices(),
            col_indices: lu.col_indices(),
            pivot_errors: lu.pivot_errors(),
            rank: lu.npivots(),
        })
    }

    fn permutation_indices_from_matrix<T: Scalar>(
        data: &[T],
        n: usize,
        name: &str,
    ) -> Result<Vec<usize>> {
        let mut indices = Vec::with_capacity(n);
        for row in 0..n {
            let mut selected = None;
            for col in 0..n {
                if data[row + n * col].abs_val() > 0.5 && selected.replace(col).is_some() {
                    return Err(MatrixLuciError::InvalidArgument {
                        message: format!("{name} permutation row {row} has multiple entries"),
                    });
                }
            }
            indices.push(selected.ok_or_else(|| MatrixLuciError::InvalidArgument {
                message: format!("{name} permutation row {row} has no entry"),
            })?);
        }
        Ok(indices)
    }

    fn factorize_square_with_tenferro<T: Scalar + TensorScalar>(
        data: &[T],
        n: usize,
        options: &PivotKernelOptions,
    ) -> Result<PivotSelectionCore> {
        let matrix = Matrix::from_col_major_vec(n, n, data.to_vec());
        let decomp =
            full_piv_lu_matrix(&matrix).map_err(|err| MatrixLuciError::InvalidArgument {
                message: format!("tenferro full_piv_lu failed: {err}"),
            })?;

        let u_values = decomp.u.as_col_major_slice();
        let diag_abs = (0..n)
            .map(|i| u_values[i + n * i].abs_val())
            .collect::<Vec<_>>();
        let pivot_errors = Self::compute_pivot_errors(&diag_abs, n, n, options);
        let rank = pivot_errors.len().saturating_sub(1);

        let row_perm =
            Self::permutation_indices_from_matrix(decomp.p.as_col_major_slice(), n, "P")?;
        let col_perm =
            Self::permutation_indices_from_matrix(decomp.q.as_col_major_slice(), n, "Q")?;

        Ok(PivotSelectionCore {
            row_indices: row_perm[..rank].to_vec(),
            col_indices: col_perm[..rank].to_vec(),
            pivot_errors,
            rank,
        })
    }
}

macro_rules! impl_dense_kernel {
    ($t:ty) => {
        impl PivotKernel<$t> for DenseLuKernel {
            fn factorize<S: CandidateMatrixSource<$t>>(
                &self,
                source: &S,
                options: &PivotKernelOptions,
            ) -> Result<PivotSelectionCore> {
                let run = |data: &[$t]| -> Result<PivotSelectionCore> {
                    let nrows = source.nrows();
                    let ncols = source.ncols();
                    if nrows == ncols {
                        DenseLuKernel::factorize_square_with_tenferro(data, nrows, options)
                    } else {
                        DenseLuKernel::factorize_with_rrlu(data, nrows, ncols, options)
                    }
                };

                if let Some(data) = source.dense_column_major_slice() {
                    run(data)
                } else {
                    let materialized = materialize_source(source);
                    run(materialized.as_col_major_slice())
                }
            }
        }
    };
}

impl_dense_kernel!(f32);
impl_dense_kernel!(f64);
impl_dense_kernel!(Complex32);
impl_dense_kernel!(Complex64);

#[cfg(test)]
mod tests;
