//! Contraction operations for tensor trains
//!
//! This module provides various ways to combine tensor trains:
//! - `dot`: Inner product (returns scalar)

use crate::compression::CompressionMethod;
use crate::einsum_helper::{
    einsum_tensors, tensor_to_col_major_vec, typed_tensor_from_col_major_slice, EinsumScalar,
};
use crate::error::{Result, TensorTrainError};
use crate::tensortrain::SimpleTensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3Ops;
use tensor4all_tcicore::Scalar;
use tensor4all_tensorbackend::Matrix;

fn contraction_helper_error(context: &str, err: impl std::fmt::Display) -> TensorTrainError {
    TensorTrainError::InvalidOperation {
        message: format!("{context}: {err}"),
    }
}

/// Options for MPO-MPO contraction with on-the-fly compression.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::ContractionOptions;
///
/// let opts = ContractionOptions::default();
/// assert!((opts.tolerance - 1e-12).abs() < 1e-15);
/// assert_eq!(opts.max_bond_dim, usize::MAX);
/// ```
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Relative truncation tolerance during contraction.
    pub tolerance: f64,
    /// Hard upper bound on bond dimension.
    pub max_bond_dim: usize,
    /// Decomposition method for intermediate compression.
    pub method: CompressionMethod,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_bond_dim: usize::MAX,
            method: CompressionMethod::LU,
        }
    }
}

impl<T: TTScalar + Scalar + Default + EinsumScalar> SimpleTensorTrain<T> {
    /// Inner product (dot product) of two tensor trains.
    ///
    /// Computes `sum_i self[i] * other[i]` by contracting the site tensors
    /// from left to right. Both tensor trains must have the same length and
    /// matching site dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the contraction or operation fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    ///
    /// // dot = sum_ij a[i,j]*b[i,j] = 1*2 * 2*3 = 12
    /// let d = a.dot(&b).unwrap();
    /// assert!((d - 12.0).abs() < 1e-10);
    /// ```
    pub fn dot(&self, other: &Self) -> Result<T> {
        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute dot product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(T::zero());
        }

        let n = self.len();

        // Start with contraction of first site
        // result[ra, rb] = sum_s a[0, s, ra] * b[0, s, rb]
        let a0 = self.site_tensor(0);
        let b0 = other.site_tensor(0);

        if a0.site_dim() != b0.site_dim() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Site dimensions mismatch at site 0: {} vs {}",
                    a0.site_dim(),
                    b0.site_dim()
                ),
            });
        }

        let mut result = Matrix::from_col_major_vec(
            a0.right_dim(),
            b0.right_dim(),
            tensor_to_col_major_vec(
                &einsum_tensors("asr,ast->rt", &[a0.as_inner(), b0.as_inner()]).map_err(|err| {
                    contraction_helper_error("Failed to contract first dot site", err)
                })?,
            ),
        );

        // Contract through remaining sites
        for i in 1..n {
            let a = self.site_tensor(i);
            let b = other.site_tensor(i);

            if a.site_dim() != b.site_dim() {
                return Err(TensorTrainError::InvalidOperation {
                    message: format!(
                        "Site dimensions mismatch at site {}: {} vs {}",
                        i,
                        a.site_dim(),
                        b.site_dim()
                    ),
                });
            }

            let result_tf = typed_tensor_from_col_major_slice(
                result.as_col_major_slice(),
                &[result.nrows(), result.ncols()],
            )
            .map_err(|err| {
                contraction_helper_error("Failed to prepare intermediate dot environment", err)
            })?;

            result = Matrix::from_col_major_vec(
                a.right_dim(),
                b.right_dim(),
                tensor_to_col_major_vec(
                    &einsum_tensors("ij,isk,jsl->kl", &[&result_tf, a.as_inner(), b.as_inner()])
                        .map_err(|err| {
                            contraction_helper_error(
                                &format!("Failed to contract dot site {i}"),
                                err,
                            )
                        })?,
                ),
            );
        }

        // Final result should be 1x1
        Ok(result[[0, 0]])
    }
}

/// Free-function wrapper for [`SimpleTensorTrain::dot`].
///
/// # Errors
///
/// Returns an error when the contraction or operation fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, contraction::dot};
///
/// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);
/// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
/// let d = dot(&a, &b).unwrap();
/// // 3*4 * 2*3 = 72
/// assert!((d - 72.0).abs() < 1e-10);
/// ```
pub fn dot<T: TTScalar + Scalar + Default + EinsumScalar>(
    a: &SimpleTensorTrain<T>,
    b: &SimpleTensorTrain<T>,
) -> Result<T> {
    a.dot(b)
}

#[cfg(test)]
mod tests;
