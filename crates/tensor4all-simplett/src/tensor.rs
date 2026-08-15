//! Fixed-rank tensor type backed by `tenferro_tensor::TypedTensor<T>`.
//!
//! This wrapper preserves a compile-time rank while delegating storage and
//! indexing to tenferro's typed dense tensor implementation.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

use tenferro_tensor::{TensorScalar, TypedTensor as TfTensor};

use crate::error::{Result, TensorTrainError};

/// Rank-N host-backed tensor backed by `tenferro_tensor::TypedTensor<T>`.
#[derive(Debug)]
pub struct Tensor<T: TensorScalar, const N: usize> {
    inner: TfTensor<T>,
    dims: [usize; N],
}

/// Iterator over tensor elements in column-major order.
pub struct TensorIter<'a, T: TensorScalar, const N: usize> {
    tensor: &'a TfTensor<T>,
    dims: [usize; N],
    next: usize,
    len: usize,
}

/// Mutable iterator over tensor elements in column-major order.
pub struct TensorIterMut<'a, T: TensorScalar, const N: usize> {
    tensor: *mut TfTensor<T>,
    dims: [usize; N],
    next: usize,
    len: usize,
    _marker: PhantomData<&'a mut TfTensor<T>>,
}

/// 2D tensor (matrix).
pub type Tensor2<T> = Tensor<T, 2>;

/// 3D tensor.
pub type Tensor3<T> = Tensor<T, 3>;

/// 4D tensor.
pub type Tensor4<T> = Tensor<T, 4>;

// Extract a result whose error branch means the host-backed wrapper invariant failed.
fn require_invariant<T, E: std::fmt::Display>(
    result: std::result::Result<T, E>,
    context: &str,
) -> T {
    let valid = result.is_ok();
    if let Err(error) = &result {
        assert!(valid, "{context}: {error}");
    }
    match result {
        Ok(value) => value,
        Err(_) => loop {
            std::hint::spin_loop();
        },
    }
}

fn col_major_index_from_linear<const N: usize>(mut linear: usize, dims: &[usize; N]) -> [usize; N] {
    let mut idx = [0usize; N];
    for axis in 0..N {
        let dim = dims[axis];
        if dim == 0 {
            return idx;
        }
        idx[axis] = linear % dim;
        linear /= dim;
    }
    idx
}

fn col_major_data_to_tensor<T: TensorScalar, const N: usize>(
    dims: [usize; N],
    data: Vec<T>,
) -> Result<Tensor<T, N>> {
    let inner = TfTensor::from_vec_col_major(dims.to_vec(), data).map_err(|error| {
        TensorTrainError::InvalidOperation {
            message: format!("invalid tensor shape or data: {error}"),
        }
    })?;
    Ok(Tensor::from_tenferro_unchecked(inner))
}

impl<T: TensorScalar, const N: usize> Clone for Tensor<T, N> {
    fn clone(&self) -> Self {
        // INVARIANT: constructors validate host accessibility, and mutable inner
        // access documents that callers must preserve this wrapper invariant.
        Self {
            inner: require_invariant(
                self.inner.duplicate(),
                "host-backed tensor duplication failed",
            ),
            dims: self.dims,
        }
    }
}

impl<T: TensorScalar + PartialEq, const N: usize> PartialEq for Tensor<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.dims() == other.dims() && self.iter().eq(other.iter())
    }
}

impl<T: TensorScalar + Eq, const N: usize> Eq for Tensor<T, N> {}

impl<'a, T: TensorScalar, const N: usize> Iterator for TensorIter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }

        let idx = col_major_index_from_linear(self.next, &self.dims);
        self.next += 1;
        Some(require_invariant(
            self.tensor.get(&idx[..]),
            "tensor index failed",
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a, T: TensorScalar, const N: usize> ExactSizeIterator for TensorIter<'a, T, N> {}

impl<'a, T: TensorScalar, const N: usize> Iterator for TensorIterMut<'a, T, N> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.len {
            return None;
        }

        let idx = col_major_index_from_linear(self.next, &self.dims);
        self.next += 1;

        // Safety: each logical index is visited at most once, so returned
        // mutable references never alias each other.
        let tensor = unsafe { &mut *self.tensor };
        let elem = require_invariant(tensor.get_mut(&idx[..]), "tensor index failed");
        let ptr = elem as *mut T;
        Some(unsafe { &mut *ptr })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a, T: TensorScalar, const N: usize> ExactSizeIterator for TensorIterMut<'a, T, N> {}

impl<T: TensorScalar, const N: usize> Tensor<T, N> {
    pub(crate) fn from_tenferro_unchecked(tensor: TfTensor<T>) -> Self {
        debug_assert_eq!(
            tensor.shape().len(),
            N,
            "tensor rank mismatch: expected rank {N}, got {}",
            tensor.shape().len()
        );
        // INVARIANT: crate-private callers construct or receive CPU host tensors.
        debug_assert!(
            tensor.host_data().is_ok(),
            "simplett tensors must remain host-backed"
        );
        let mut dims = [0usize; N];
        for (dim, value) in dims.iter_mut().zip(tensor.shape().iter().copied()) {
            *dim = value;
        }
        Self {
            inner: tensor,
            dims,
        }
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.inner.n_elements()
    }

    /// Whether the tensor is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Dimension along `axis`.
    pub fn dim(&self, axis: usize) -> usize {
        self.dims()[axis]
    }

    /// All dimensions.
    pub fn dims(&self) -> &[usize; N] {
        &self.dims
    }

    /// Export the host-backed tensor as a column-major flat vector.
    ///
    /// # Panics
    ///
    /// Panics if crate-internal code has violated the wrapper's host-backed
    /// storage invariant.
    pub fn to_col_major_vec(&self) -> Vec<T> {
        // INVARIANT: public and crate-private constructors validate that a
        // simplett tensor is host-backed, and `as_inner_mut` requires callers
        // to preserve that invariant.
        require_invariant(
            self.inner.host_data(),
            "host-backed tensor data is unavailable",
        )
        .to_vec()
    }

    /// Iterate over all elements in column-major order.
    pub fn iter(&self) -> TensorIter<'_, T, N> {
        TensorIter {
            tensor: &self.inner,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
        }
    }

    /// Iterate mutably over all elements in column-major order.
    pub fn iter_mut(&mut self) -> TensorIterMut<'_, T, N> {
        TensorIterMut {
            tensor: &mut self.inner,
            dims: *self.dims(),
            next: 0,
            len: self.len(),
            _marker: PhantomData,
        }
    }

    /// Borrow the wrapped tenferro tensor.
    pub fn as_inner(&self) -> &TfTensor<T> {
        &self.inner
    }

    /// Consume this wrapper and return the inner tenferro tensor.
    pub fn into_inner(self) -> TfTensor<T> {
        self.inner
    }

    /// Try to wrap an existing tenferro tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor rank does not match `N` or its data is
    /// not host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::{Tensor2, Tensor3};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let tensor = Tensor2::try_from_tenferro(rank_2)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    ///
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert!(Tensor3::try_from_tenferro(rank_2).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn try_from_tenferro(tensor: TfTensor<T>) -> Result<Self> {
        Self::from_tenferro(tensor)
    }

    /// Wrap an existing tenferro tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor rank does not match `N` or its data is
    /// not host-accessible.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::{Tensor2, Tensor3};
    /// use tenferro_tensor::TypedTensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// let tensor = Tensor2::from_tenferro(rank_2)?;
    /// assert_eq!(tensor.dims(), &[2, 2]);
    ///
    /// let rank_2 = TypedTensor::from_vec_col_major(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    /// assert!(Tensor3::from_tenferro(rank_2).is_err());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_tenferro(tensor: TfTensor<T>) -> Result<Self> {
        if tensor.shape().len() != N {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "tensor rank mismatch: expected rank {N}, got {}",
                    tensor.shape().len()
                ),
            });
        }
        tensor
            .host_data()
            .map_err(|error| TensorTrainError::InvalidOperation {
                message: format!("simplett tensors must be host-backed: {error}"),
            })?;
        let mut dims = [0usize; N];
        dims.copy_from_slice(tensor.shape());
        Ok(Self {
            inner: tensor,
            dims,
        })
    }

    /// Try to create a tensor by applying `f` to each multi-index in column-major order.
    ///
    /// # Errors
    ///
    /// Returns [`TensorTrainError::InvalidOperation`] when the shape product
    /// overflows, allocation cannot reserve the required capacity, or tenferro
    /// rejects the generated shape and data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::Tensor2;
    ///
    /// let tensor = Tensor2::try_from_fn([2, 2], |[i, j]| (i + 2 * j) as f64)?;
    /// assert_eq!(tensor.to_col_major_vec(), vec![0.0, 1.0, 2.0, 3.0]);
    /// # Ok::<(), tensor4all_simplett::TensorTrainError>(())
    /// ```
    pub fn try_from_fn(dims: [usize; N], mut f: impl FnMut([usize; N]) -> T) -> Result<Self> {
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or_else(|| TensorTrainError::InvalidOperation {
                message: format!("tensor shape product overflows usize: {dims:?}"),
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(total)
            .map_err(|error| TensorTrainError::InvalidOperation {
                message: format!("cannot allocate tensor shape {dims:?}: {error}"),
            })?;

        for linear in 0..total {
            data.push(f(col_major_index_from_linear(linear, &dims)));
        }

        col_major_data_to_tensor(dims, data)
    }

    /// Create a tensor by applying `f` to each multi-index in column-major order.
    ///
    /// # Panics
    ///
    /// Panics if the shape product overflows, allocation fails, or tenferro
    /// rejects the shape/data pair. Use [`Self::try_from_fn`] when dimensions
    /// are not already validated.
    pub fn from_fn(dims: [usize; N], f: impl FnMut([usize; N]) -> T) -> Self {
        require_invariant(Self::try_from_fn(dims, f), "invalid tensor shape")
    }
}

impl<T: TensorScalar, const N: usize> Tensor<T, N> {
    /// Try to create a tensor filled with `value`.
    ///
    /// # Errors
    ///
    /// Returns [`TensorTrainError::InvalidOperation`] when the shape product
    /// overflows, allocation cannot reserve the required capacity, or tenferro
    /// rejects the generated shape and data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::tensor::Tensor2;
    ///
    /// let tensor = Tensor2::try_from_elem([2, 3], 4.0_f64)?;
    /// assert_eq!(tensor.to_col_major_vec(), vec![4.0; 6]);
    /// # Ok::<(), tensor4all_simplett::TensorTrainError>(())
    /// ```
    pub fn try_from_elem(dims: [usize; N], value: T) -> Result<Self> {
        let total = dims
            .iter()
            .try_fold(1usize, |total, &dim| total.checked_mul(dim))
            .ok_or_else(|| TensorTrainError::InvalidOperation {
                message: format!("tensor shape product overflows usize: {dims:?}"),
            })?;
        let mut data = Vec::new();
        data.try_reserve_exact(total)
            .map_err(|error| TensorTrainError::InvalidOperation {
                message: format!("cannot allocate tensor shape {dims:?}: {error}"),
            })?;
        data.resize(total, value);
        col_major_data_to_tensor(dims, data)
    }

    /// Create a tensor filled with `value`.
    ///
    /// # Panics
    ///
    /// Panics if the shape product overflows, allocation fails, or tenferro
    /// rejects the shape/data pair. Use [`Self::try_from_elem`] when dimensions
    /// are not already validated.
    pub fn from_elem(dims: [usize; N], value: T) -> Self {
        require_invariant(Self::try_from_elem(dims, value), "invalid tensor shape")
    }
}

impl<T: TensorScalar, const N: usize> Index<[usize; N]> for Tensor<T, N> {
    type Output = T;

    fn index(&self, idx: [usize; N]) -> &T {
        require_invariant(self.inner.get(&idx[..]), "tensor index failed")
    }
}

impl<T: TensorScalar, const N: usize> IndexMut<[usize; N]> for Tensor<T, N> {
    fn index_mut(&mut self, idx: [usize; N]) -> &mut T {
        require_invariant(self.inner.get_mut(&idx[..]), "tensor index failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::einsum_helper::{tensor_to_col_major_vec, typed_tensor_from_col_major_slice};

    #[test]
    fn test_tensor2_from_elem() {
        let t: Tensor2<f64> = Tensor2::from_elem([3, 4], 0.0);
        assert_eq!(t.len(), 12);
        assert_eq!(t.dim(0), 3);
        assert_eq!(t.dim(1), 4);
        assert_eq!(t[[0, 0]], 0.0);
    }

    #[test]
    fn fallible_constructors_reject_invalid_capacity_before_evaluating_elements() {
        let mut called = false;
        let overflow_fn = Tensor2::<f64>::try_from_fn([usize::MAX, 2], |_| {
            called = true;
            0.0
        });
        let overflow_elem = Tensor2::try_from_elem([usize::MAX, 2], 0.0_f64);
        let capacity_fn = Tensor2::<f64>::try_from_fn([usize::MAX, 1], |_| {
            called = true;
            0.0
        });
        let capacity_elem = Tensor2::try_from_elem([usize::MAX, 1], 0.0_f64);

        assert!(overflow_fn.is_err());
        assert!(overflow_elem.is_err());
        assert!(capacity_fn.is_err());
        assert!(capacity_elem.is_err());
        assert!(!called);
    }

    #[test]
    fn test_tensor2_indexing() {
        let mut t: Tensor2<f64> = Tensor2::from_elem([2, 3], 0.0);
        t[[0, 0]] = 1.0;
        t[[0, 1]] = 2.0;
        t[[0, 2]] = 3.0;
        t[[1, 0]] = 4.0;
        t[[1, 1]] = 5.0;
        t[[1, 2]] = 6.0;
        assert_eq!(t[[0, 0]], 1.0);
        assert_eq!(t[[1, 2]], 6.0);
        assert_eq!(t.to_col_major_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_tensor3_from_fn() {
        let t: Tensor3<f64> =
            Tensor3::from_fn([2, 3, 4], |[i, j, k]| (i * 100 + j * 10 + k) as f64);
        assert_eq!(t[[0, 0, 0]], 0.0);
        assert_eq!(t[[1, 2, 3]], 123.0);
        assert_eq!(t[[0, 1, 2]], 12.0);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.len(), 24);
    }

    #[test]
    fn test_tensor4_from_fn() {
        let t: Tensor4<f64> = Tensor4::from_fn([2, 3, 4, 5], |[i, j, k, l]| {
            (i * 1000 + j * 100 + k * 10 + l) as f64
        });
        assert_eq!(t[[1, 2, 3, 4]], 1234.0);
        assert_eq!(t.dim(0), 2);
        assert_eq!(t.dim(1), 3);
        assert_eq!(t.dim(2), 4);
        assert_eq!(t.dim(3), 5);
        assert_eq!(t.len(), 120);
    }

    #[test]
    fn test_iter() {
        let t: Tensor2<f64> = Tensor2::from_fn([2, 3], |[i, j]| (i * 3 + j) as f64);
        let collected: Vec<f64> = t.iter().copied().collect();
        assert_eq!(collected, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn test_dims() {
        let t: Tensor3<f64> = Tensor3::from_elem([2, 3, 4], 1.0);
        assert_eq!(t.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_from_tenferro_roundtrip() {
        let inner =
            typed_tensor_from_col_major_slice(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();
        let tensor = Tensor2::from_tenferro(inner).unwrap();

        assert_eq!(tensor.dims(), &[2, 3]);
        assert_eq!(
            tensor.to_col_major_vec(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        let inner_again = tensor.clone().into_inner();
        assert_eq!(
            tensor_to_col_major_vec(&inner_again).unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn test_from_tenferro_rank_mismatch_errors() {
        let inner = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
        let err = Tensor3::from_tenferro(inner).unwrap_err();

        assert!(err.to_string().contains("expected rank 3"));
        assert!(err.to_string().contains("got 2"));
    }

    #[test]
    fn test_try_from_tenferro_rank_mismatch_errors() {
        let inner = typed_tensor_from_col_major_slice(&[1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
        let err = Tensor3::try_from_tenferro(inner).unwrap_err();

        assert!(err.to_string().contains("expected rank 3"));
        assert!(err.to_string().contains("got 2"));
    }

    #[test]
    fn test_clone_allows_index_mut() {
        let original: Tensor2<f64> = Tensor2::from_fn([2, 2], |[i, j]| (i * 2 + j) as f64);
        let mut cloned = original.clone();
        cloned[[1, 0]] = 99.0;

        assert_eq!(original[[1, 0]], 2.0);
        assert_eq!(cloned[[1, 0]], 99.0);
    }
}
