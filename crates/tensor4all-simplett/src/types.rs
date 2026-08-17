//! Core types for tensor train operations

use crate::error::{Result, TensorTrainError};
use crate::tensor::Tensor;
pub use crate::tensor::Tensor3;
use tenferro_tensor::{TensorScalar, TypedTensor as TfTensor};

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// Convenience accessors for rank-3 core tensors with shape
/// `(left_bond, site_dim, right_bond)`.
///
/// These methods give named access to dimensions and elements using the
/// tensor train convention where axis 0 is the left bond, axis 1 is the
/// physical (site) index, and axis 2 is the right bond.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{Tensor3Ops, tensor3_zeros};
///
/// let mut t = tensor3_zeros::<f64>(2, 3, 4);
/// assert_eq!(t.left_dim(), 2);
/// assert_eq!(t.site_dim(), 3);
/// assert_eq!(t.right_dim(), 4);
///
/// t.set3(1, 2, 3, 42.0);
/// assert_eq!(*t.get3(1, 2, 3), 42.0);
/// ```
pub trait Tensor3Ops<T: Clone + Default> {
    /// Left (bond) dimension (axis 0).
    fn left_dim(&self) -> usize;

    /// Physical (site) dimension (axis 1).
    fn site_dim(&self) -> usize;

    /// Right (bond) dimension (axis 2).
    fn right_dim(&self) -> usize;

    /// Borrow the element at `(left, site, right)`.
    fn get3(&self, l: usize, s: usize, r: usize) -> &T;

    /// Mutably borrow the element at `(left, site, right)`.
    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T;

    /// Set the element at `(left, site, right)` to `value`.
    fn set3(&mut self, l: usize, s: usize, r: usize, value: T);

    /// Extract the `(left_dim, right_dim)` matrix for a fixed site index `s`
    /// as a flat column-major vector.
    fn slice_site(&self, s: usize) -> Vec<T>;

    /// Reshape to a `(left_dim * site_dim, right_dim)` matrix.
    fn as_left_matrix(&self) -> (Vec<T>, usize, usize);

    /// Reshape to a `(left_dim, site_dim * right_dim)` matrix.
    fn as_right_matrix(&self) -> (Vec<T>, usize, usize);
}

impl<T: Clone + Default + TensorScalar> Tensor3Ops<T> for Tensor3<T> {
    fn left_dim(&self) -> usize {
        self.dim(0)
    }

    fn site_dim(&self) -> usize {
        self.dim(1)
    }

    fn right_dim(&self) -> usize {
        self.dim(2)
    }

    fn get3(&self, l: usize, s: usize, r: usize) -> &T {
        &self[[l, s, r]]
    }

    fn get3_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T {
        &mut self[[l, s, r]]
    }

    fn set3(&mut self, l: usize, s: usize, r: usize, value: T) {
        self[[l, s, r]] = value;
    }

    fn slice_site(&self, s: usize) -> Vec<T> {
        let left_dim = self.left_dim();
        let right_dim = self.right_dim();
        let mut result = Vec::with_capacity(left_dim * right_dim);
        for r in 0..right_dim {
            for l in 0..left_dim {
                result.push(self[[l, s, r]]);
            }
        }
        result
    }

    fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim * site_dim;
        let cols = right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for r in 0..right_dim {
            for l in 0..left_dim {
                for s in 0..site_dim {
                    result.push(self[[l, s, r]]);
                }
            }
        }
        (result, rows, cols)
    }

    fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let left_dim = self.left_dim();
        let site_dim = self.site_dim();
        let right_dim = self.right_dim();
        let rows = left_dim;
        let cols = site_dim * right_dim;
        let mut result = Vec::with_capacity(rows * cols);
        for s in 0..site_dim {
            for r in 0..right_dim {
                for l in 0..left_dim {
                    result.push(self[[l, s, r]]);
                }
            }
        }
        (result, rows, cols)
    }
}

/// Create a zero-filled rank-3 tensor with shape `(left_dim, site_dim, right_dim)`.
///
/// # Panics
///
/// Panics when the shape overflows `usize` or allocation fails. Use
/// [`try_tensor3_zeros`] when dimensions come from external or untrusted input.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_zeros, Tensor3Ops};
///
/// let t = tensor3_zeros::<f64>(2, 3, 4);
/// assert_eq!(t.left_dim(), 2);
/// assert_eq!(t.site_dim(), 3);
/// assert_eq!(t.right_dim(), 4);
/// assert_eq!(*t.get3(0, 0, 0), 0.0);
/// ```
pub fn tensor3_zeros<T: Clone + Default + TensorScalar>(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Tensor3<T> {
    Tensor::from_elem([left_dim, site_dim, right_dim], T::default())
}

/// Fallibly create a zero-filled rank-3 tensor.
///
/// # Errors
///
/// Returns [`TensorTrainError::InvalidOperation`] when the shape product
/// overflows `usize`, allocation fails, or the backend rejects the shape.
///
/// # Examples
/// ```
/// use tensor4all_simplett::{try_tensor3_zeros, Tensor3Ops};
///
/// let t = try_tensor3_zeros::<f64>(2, 3, 4)?;
/// assert_eq!((t.left_dim(), t.site_dim(), t.right_dim()), (2, 3, 4));
/// # Ok::<(), tensor4all_simplett::TensorTrainError>(())
/// ```
pub fn try_tensor3_zeros<T: Clone + Default + TensorScalar>(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    Tensor::try_from_elem([left_dim, site_dim, right_dim], T::default())
}

/// Create a rank-3 tensor from flat data in **column-major** order.
///
/// # Errors
///
/// Returns an error when the construction or conversion fails (a shape or
/// /// index mismatch, or a backend failure).
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{tensor3_from_data, Tensor3Ops};
///
/// // 1 x 2 x 1 tensor, column-major data: [10.0, 20.0]
/// let t = tensor3_from_data(vec![10.0, 20.0], 1, 2, 1).unwrap();
/// assert_eq!(*t.get3(0, 0, 0), 10.0);
/// assert_eq!(*t.get3(0, 1, 0), 20.0);
/// ```
pub fn tensor3_from_data<T: TensorScalar>(
    data: Vec<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
) -> Result<Tensor3<T>> {
    let expected = left_dim
        .checked_mul(site_dim)
        .and_then(|value| value.checked_mul(right_dim))
        .ok_or_else(|| TensorTrainError::InvalidOperation {
            message: "rank-3 tensor shape product overflowed usize".to_string(),
        })?;
    if data.len() != expected {
        return Err(TensorTrainError::DataLengthMismatch {
            expected,
            got: data.len(),
        });
    }
    let dims = [left_dim, site_dim, right_dim];
    let inner = TfTensor::from_vec_col_major(dims.to_vec(), data).map_err(|error| {
        TensorTrainError::InvalidOperation {
            message: format!("rank-3 tensor construction failed: {error}"),
        }
    })?;
    Ok(Tensor::from_tenferro_unchecked(inner))
}

#[cfg(test)]
mod tests;
