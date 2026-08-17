//! Core types for MPO (Matrix Product Operator) operations
//!
//! MPO site tensors are 4D tensors with shape (left_bond, site_dim_1, site_dim_2, right_bond).
//! - `left_bond`: Bond dimension connecting to the left neighbor
//! - `site_dim_1`: Physical index (e.g., row index for operators)
//! - `site_dim_2`: Physical index (e.g., column index, often contracted)
//! - `right_bond`: Bond dimension connecting to the right neighbor

use crate::tensor::Tensor;
pub use crate::tensor::Tensor4;
use tenferro_tensor::{TensorScalar, TypedTensor as TfTensor};

use super::error::{MPOError, Result};

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// Helper functions for Tensor4 operations
pub trait Tensor4Ops<T: Clone + Default> {
    /// Get the left (bond) dimension
    fn left_dim(&self) -> usize;

    /// Get the first site (physical) dimension
    fn site_dim_1(&self) -> usize;

    /// Get the second site (physical) dimension
    fn site_dim_2(&self) -> usize;

    /// Get the right (bond) dimension
    fn right_dim(&self) -> usize;

    /// Get element at (left, site1, site2, right)
    fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> &T;

    /// Get mutable element at (left, site1, site2, right)
    fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> &mut T;

    /// Set element at (left, site1, site2, right)
    fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T);

    /// Get a slice for fixed site indices as a flat column-major
    /// `(left_dim, right_dim)` matrix.
    fn slice_site(&self, s1: usize, s2: usize) -> Vec<T>;

    /// Reshape this tensor to a matrix (left_dim * site_dim_1 * site_dim_2, right_dim)
    fn as_left_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim, site_dim_1 * site_dim_2 * right_dim)
    fn as_right_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape this tensor to a matrix (left_dim * site_dim_1, site_dim_2 * right_dim)
    fn as_center_matrix(&self) -> (Vec<T>, usize, usize);
}

impl<T: Clone + Default + TensorScalar> Tensor4Ops<T> for Tensor4<T> {
    fn left_dim(&self) -> usize {
        self.dim(0)
    }

    fn site_dim_1(&self) -> usize {
        self.dim(1)
    }

    fn site_dim_2(&self) -> usize {
        self.dim(2)
    }

    fn right_dim(&self) -> usize {
        self.dim(3)
    }

    fn get4(&self, l: usize, s1: usize, s2: usize, r: usize) -> &T {
        &self[[l, s1, s2, r]]
    }

    fn get4_mut(&mut self, l: usize, s1: usize, s2: usize, r: usize) -> &mut T {
        &mut self[[l, s1, s2, r]]
    }

    fn set4(&mut self, l: usize, s1: usize, s2: usize, r: usize, value: T) {
        self[[l, s1, s2, r]] = value;
    }

    fn slice_site(&self, s1: usize, s2: usize) -> Vec<T> {
        let left_dim = self.left_dim();
        let right_dim = self.right_dim();
        let mut result = Vec::with_capacity(left_dim * right_dim);
        for r in 0..right_dim {
            for l in 0..left_dim {
                result.push(self[[l, s1, s2, r]]);
            }
        }
        result
    }

    fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim_1 * site_dim_2;
        let cols = right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for r in 0..right_dim {
            for l in 0..left_dim {
                for s1 in 0..site_dim_1 {
                    for s2 in 0..site_dim_2 {
                        result.push(self[[l, s1, s2, r]]);
                    }
                }
            }
        }
        (result, rows, cols)
    }

    fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim;
        let cols = site_dim_1 * site_dim_2 * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for s1 in 0..site_dim_1 {
            for s2 in 0..site_dim_2 {
                for r in 0..right_dim {
                    for l in 0..left_dim {
                        result.push(self[[l, s1, s2, r]]);
                    }
                }
            }
        }
        (result, rows, cols)
    }

    fn as_center_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim_1 = self.site_dim_1();
        let site_dim_2 = self.site_dim_2();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim_1;
        let cols = site_dim_2 * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for s2 in 0..site_dim_2 {
            for r in 0..right_dim {
                for l in 0..left_dim {
                    for s1 in 0..site_dim_1 {
                        result.push(self[[l, s1, s2, r]]);
                    }
                }
            }
        }
        (result, rows, cols)
    }
}

/// Create a zero-filled Tensor4
pub fn tensor4_zeros<T: Clone + Default + TensorScalar>(
    left_dim: usize,
    site_dim_1: usize,
    site_dim_2: usize,
    right_dim: usize,
) -> Tensor4<T> {
    Tensor::from_elem([left_dim, site_dim_1, site_dim_2, right_dim], T::default())
}

/// Create a Tensor4 from flat data (column-major order)
/// # Errors
///
/// Returns an error when the construction or conversion fails (a shape or
/// /// index mismatch, or a backend failure).
///
pub fn tensor4_from_data<T: TensorScalar>(
    data: Vec<T>,
    left_dim: usize,
    site_dim_1: usize,
    site_dim_2: usize,
    right_dim: usize,
) -> Result<Tensor4<T>> {
    let expected = left_dim
        .checked_mul(site_dim_1)
        .and_then(|value| value.checked_mul(site_dim_2))
        .and_then(|value| value.checked_mul(right_dim))
        .ok_or_else(|| MPOError::InvalidOperation {
            message: "rank-4 tensor shape product overflowed usize".to_string(),
        })?;
    if data.len() != expected {
        return Err(MPOError::DataLengthMismatch {
            expected,
            got: data.len(),
        });
    }
    let dims = [left_dim, site_dim_1, site_dim_2, right_dim];
    let inner = TfTensor::from_vec_col_major(dims.to_vec(), data).map_err(|error| {
        MPOError::InvalidOperation {
            message: format!("rank-4 tensor construction failed: {error}"),
        }
    })?;
    Ok(Tensor::from_tenferro_unchecked(inner))
}

#[cfg(test)]
mod tests;
