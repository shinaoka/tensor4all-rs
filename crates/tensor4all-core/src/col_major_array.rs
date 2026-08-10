//! N-dimensional column-major array types.
//!
//! Column-major layout: the element at multi-index `[i0, i1, i2, ...]` is stored
//! at flat offset `i0 + shape[0] * (i1 + shape[1] * (i2 + ...))`.
//!
//! Three flavors are provided:
//! - [`ColMajorArrayRef`] — borrowed data and shape (read-only)
//! - [`ColMajorArrayMut`] — mutably borrowed data, borrowed shape
//! - [`ColMajorArray`] — fully owned data and shape

/// Errors that can occur when constructing or modifying a column-major array.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum ColMajorArrayError {
    /// The length of the data does not match the product of the shape dimensions.
    #[error("Shape mismatch: shape {shape:?} requires {expected} elements, but got {actual}")]
    ShapeMismatch {
        /// The requested shape.
        shape: Vec<usize>,
        /// Number of elements implied by the shape.
        expected: usize,
        /// Number of elements actually provided.
        actual: usize,
    },

    /// The column length does not match `nrows`.
    #[error("Column length mismatch: expected {expected} elements, but got {actual}")]
    ColumnLengthMismatch {
        /// Expected number of rows.
        expected: usize,
        /// Actual number of elements in the column.
        actual: usize,
    },

    /// A 2D operation was called on an array that is not 2-dimensional.
    #[error("Expected a 2D array, but ndim = {ndim}")]
    Not2D {
        /// The actual number of dimensions.
        ndim: usize,
    },

    /// The product of shape dimensions overflows `usize`.
    #[error("Shape product overflow: shape {shape:?} overflows usize")]
    ShapeOverflow {
        /// The shape that caused the overflow.
        shape: Vec<usize>,
    },

    /// Incrementing the column count would overflow `usize`.
    #[error("Column count overflow")]
    ColumnCountOverflow,
}

// ---------------------------------------------------------------------------
// Helper: compute the total number of elements from a shape
// ---------------------------------------------------------------------------

fn checked_shape_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .copied()
        .try_fold(1usize, |acc, d| acc.checked_mul(d))
}

/// Compute the flat offset for a column-major multi-index, using checked
/// arithmetic. Returns `None` if any index is out of bounds or on overflow.
fn flat_offset(shape: &[usize], index: &[usize]) -> Option<usize> {
    if index.len() != shape.len() {
        return None;
    }
    // Traverse from the last axis to the first:
    //   offset = i_{n-1}
    //   offset = i_{n-2} + shape[n-2] * offset  -- but we build from the back
    // Actually, column-major: offset = i0 + s0*(i1 + s1*(i2 + ...))
    // Evaluate right-to-left (Horner-like):
    let mut offset: usize = 0;
    for (idx, dim) in index.iter().zip(shape.iter()).rev() {
        if *idx >= *dim {
            return None;
        }
        offset = offset.checked_mul(*dim)?.checked_add(*idx)?;
    }
    Some(offset)
}

// ===========================================================================
// ColMajorArrayRef
// ===========================================================================

/// A borrowed, read-only view of an N-dimensional column-major array.
#[derive(Debug, Clone, Copy)]
pub struct ColMajorArrayRef<'a, T> {
    data: &'a [T],
    shape: &'a [usize],
}

impl<'a, T> ColMajorArrayRef<'a, T> {
    /// Create a new borrowed array view.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn new(data: &'a [T], shape: &'a [usize]) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice.
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get(off)
    }
}

// ===========================================================================
// ColMajorArrayMut
// ===========================================================================

/// A mutably borrowed view of an N-dimensional column-major array.
#[derive(Debug)]
pub struct ColMajorArrayMut<'a, T> {
    data: &'a mut [T],
    shape: &'a [usize],
}

impl<'a, T> ColMajorArrayMut<'a, T> {
    /// Create a new mutable borrowed array view.
    ///
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn new(data: &'a mut [T], shape: &'a [usize]) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.to_vec(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape: shape.to_vec(),
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice (read-only).
    pub fn data(&self) -> &[T] {
        self.data
    }

    /// Flat (contiguous) data slice (mutable).
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get(off)
    }

    /// Get a mutable reference to the element at the given multi-index, or
    /// `None` if out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let off = flat_offset(self.shape, index)?;
        self.data.get_mut(off)
    }
}

// ===========================================================================
// ColMajorArray (owned)
// ===========================================================================

/// A fully owned N-dimensional column-major array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColMajorArray<T> {
    data: Vec<T>,
    shape: Vec<usize>,
}

impl<T> ColMajorArray<T> {
    /// Create a new owned array from data and shape.
    ///
    /// Returns an error if `data.len()` does not equal the product of the
    /// shape dimensions.
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Result<Self, ColMajorArrayError> {
        let expected =
            checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
                shape: shape.clone(),
            })?;
        if data.len() != expected {
            return Err(ColMajorArrayError::ShapeMismatch {
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { data, shape })
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Shape of the array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the array is empty (zero elements).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Flat (contiguous) data slice (read-only).
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Flat (contiguous) data slice (mutable).
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Get a reference to the element at the given multi-index, or `None` if
    /// out of bounds.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        let off = flat_offset(&self.shape, index)?;
        self.data.get(off)
    }

    /// Get a mutable reference to the element at the given multi-index, or
    /// `None` if out of bounds.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        let off = flat_offset(&self.shape, index)?;
        self.data.get_mut(off)
    }

    /// Consume the array and return the underlying data vector.
    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Borrow as a [`ColMajorArrayRef`].
    pub fn as_ref(&self) -> ColMajorArrayRef<'_, T> {
        ColMajorArrayRef {
            data: &self.data,
            shape: &self.shape,
        }
    }

    /// Borrow as a [`ColMajorArrayMut`].
    pub fn as_mut(&mut self) -> ColMajorArrayMut<'_, T> {
        ColMajorArrayMut {
            data: &mut self.data,
            shape: &self.shape,
        }
    }

    // -- 2D helpers ---------------------------------------------------------

    /// Number of rows, or `None` when the array is not 2D.
    pub fn nrows(&self) -> Option<usize> {
        if self.ndim() == 2 {
            Some(self.shape[0])
        } else {
            None
        }
    }

    /// Number of columns, or `None` when the array is not 2D.
    pub fn ncols(&self) -> Option<usize> {
        if self.ndim() == 2 {
            Some(self.shape[1])
        } else {
            None
        }
    }

    /// Return a slice for column `j` of a 2D array, or `None` if `j` is out
    /// of range or if the array is not 2D.
    pub fn column(&self, j: usize) -> Option<&[T]> {
        if self.ndim() != 2 {
            return None;
        }
        let nrows = self.shape[0];
        if j >= self.shape[1] {
            return None;
        }
        let start = nrows.checked_mul(j)?;
        let end = start.checked_add(nrows)?;
        Some(&self.data[start..end])
    }

    /// Append a column to a 2D array.
    ///
    /// The `col` slice must have length equal to `nrows()`. This extends
    /// the internal data and increments `shape[1]`.
    ///
    /// Returns an error if the array is not 2D or if the column length does
    /// not match `nrows()`.
    /// # Errors
    ///
    /// Returns an error when the column length does not match the row count (a
    /// /// shape mismatch).
    ///
    pub fn push_column(&mut self, col: &[T]) -> Result<(), ColMajorArrayError>
    where
        T: Clone,
    {
        if self.ndim() != 2 {
            return Err(ColMajorArrayError::Not2D { ndim: self.ndim() });
        }
        let nrows = self.shape[0];
        if col.len() != nrows {
            return Err(ColMajorArrayError::ColumnLengthMismatch {
                expected: nrows,
                actual: col.len(),
            });
        }
        self.data.extend_from_slice(col);
        self.shape[1] = self.shape[1]
            .checked_add(1)
            .ok_or(ColMajorArrayError::ColumnCountOverflow)?;
        Ok(())
    }
}

// -- Factories (require trait bounds on T) ----------------------------------

impl<T: Clone> ColMajorArray<T> {
    /// Create an array filled with a given value.
    ///
    /// Returns an error if the product of shape dimensions overflows `usize`.
    /// # Errors
    ///
    /// Returns an error when the dimensions are invalid (a shape mismatch) or the
    /// /// fill fails.
    ///
    pub fn filled(shape: Vec<usize>, value: T) -> Result<Self, ColMajorArrayError> {
        let n = checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            data: vec![value; n],
            shape,
        })
    }
}

impl<T: Default + Clone> ColMajorArray<T> {
    /// Create an array filled with [`Default::default()`] (e.g., zeros for
    /// numeric types).
    ///
    /// Returns an error if the product of shape dimensions overflows `usize`.
    /// # Errors
    ///
    /// Returns an error when the dimensions are invalid (a shape mismatch).
    ///
    pub fn zeros(shape: Vec<usize>) -> Result<Self, ColMajorArrayError> {
        let n = checked_shape_numel(&shape).ok_or_else(|| ColMajorArrayError::ShapeOverflow {
            shape: shape.clone(),
        })?;
        Ok(Self {
            data: vec![T::default(); n],
            shape,
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- 1D creation + get --------------------------------------------------

    #[test]
    fn test_1d_creation_and_get() {
        let arr = ColMajorArray::new(vec![10, 20, 30], vec![3]).unwrap();
        assert_eq!(arr.ndim(), 1);
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr.len(), 3);
        assert!(!arr.is_empty());

        assert_eq!(arr.get(&[0]), Some(&10));
        assert_eq!(arr.get(&[1]), Some(&20));
        assert_eq!(arr.get(&[2]), Some(&30));
    }

    // -- 2D creation + get --------------------------------------------------

    #[test]
    fn test_2d_creation_and_get() {
        // 2x3 matrix in column-major:
        // Column 0: [1, 2], Column 1: [3, 4], Column 2: [5, 6]
        // Flat: [1, 2, 3, 4, 5, 6]
        let arr = ColMajorArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.len(), 6);

        // (row, col)
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[1, 0]), Some(&2));
        assert_eq!(arr.get(&[0, 1]), Some(&3));
        assert_eq!(arr.get(&[1, 1]), Some(&4));
        assert_eq!(arr.get(&[0, 2]), Some(&5));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    // -- 3D creation + get --------------------------------------------------

    #[test]
    fn test_3d_creation_and_get() {
        // Shape [2, 3, 2]: total 12 elements
        let data: Vec<i32> = (0..12).collect();
        let arr = ColMajorArray::new(data.clone(), vec![2, 3, 2]).unwrap();
        assert_eq!(arr.ndim(), 3);
        assert_eq!(arr.len(), 12);

        // Verify column-major offset: i0 + 2*(i1 + 3*i2)
        for i2 in 0..2 {
            for i1 in 0..3 {
                for i0 in 0..2 {
                    let expected_offset = i0 + 2 * (i1 + 3 * i2);
                    assert_eq!(
                        arr.get(&[i0, i1, i2]),
                        Some(&(expected_offset as i32)),
                        "Mismatch at [{i0}, {i1}, {i2}]"
                    );
                }
            }
        }
    }

    // -- Column-major order verification (2D) -------------------------------

    #[test]
    fn test_column_major_order_2d() {
        let nrows = 3;
        let ncols = 4;
        let data: Vec<i32> = (0..(nrows * ncols) as i32).collect();
        let arr = ColMajorArray::new(data.clone(), vec![nrows, ncols]).unwrap();

        // In column-major, data[i + nrows * j] == arr[(i, j)]
        for j in 0..ncols {
            for i in 0..nrows {
                assert_eq!(arr.get(&[i, j]), Some(&data[i + nrows * j]));
            }
        }
    }

    // -- get_mut ------------------------------------------------------------

    #[test]
    fn test_get_mut() {
        let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        if let Some(v) = arr.get_mut(&[1, 0]) {
            *v = 42;
        }
        assert_eq!(arr.get(&[1, 0]), Some(&42));
        // Other elements unchanged
        assert_eq!(arr.get(&[0, 0]), Some(&1));
        assert_eq!(arr.get(&[0, 1]), Some(&3));
        assert_eq!(arr.get(&[1, 1]), Some(&4));
    }

    // -- push_column --------------------------------------------------------

    #[test]
    fn test_push_column() {
        let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        assert_eq!(arr.ncols(), Some(2));

        arr.push_column(&[5, 6]).unwrap();
        assert_eq!(arr.ncols(), Some(3));
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.len(), 6);
        assert_eq!(arr.get(&[0, 2]), Some(&5));
        assert_eq!(arr.get(&[1, 2]), Some(&6));
    }

    #[test]
    fn test_push_column_wrong_length() {
        let mut arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        let err = arr.push_column(&[5, 6, 7]).unwrap_err();
        assert_eq!(
            err,
            ColMajorArrayError::ColumnLengthMismatch {
                expected: 2,
                actual: 3,
            }
        );
    }

    #[test]
    fn test_push_column_not_2d() {
        let mut arr = ColMajorArray::new(vec![1, 2, 3], vec![3]).unwrap();
        let err = arr.push_column(&[4]).unwrap_err();
        assert_eq!(err, ColMajorArrayError::Not2D { ndim: 1 });
    }

    // -- column() slice access ----------------------------------------------

    #[test]
    fn test_column_access() {
        let arr = ColMajorArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]).unwrap();
        assert_eq!(arr.column(0), Some([1, 2].as_slice()));
        assert_eq!(arr.column(1), Some([3, 4].as_slice()));
        assert_eq!(arr.column(2), Some([5, 6].as_slice()));
        assert_eq!(arr.column(3), None); // out of bounds
    }

    // -- zeros, filled ------------------------------------------------------

    #[test]
    fn test_zeros() {
        let arr: ColMajorArray<f64> = ColMajorArray::zeros(vec![3, 2]).unwrap();
        assert_eq!(arr.len(), 6);
        assert!(arr.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_filled() {
        let arr = ColMajorArray::filled(vec![2, 3], 7i32).unwrap();
        assert_eq!(arr.len(), 6);
        assert!(arr.data().iter().all(|&v| v == 7));
    }

    // -- Shape mismatch error -----------------------------------------------

    #[test]
    fn test_shape_mismatch() {
        let result = ColMajorArray::new(vec![1, 2, 3], vec![2, 2]);
        assert_eq!(
            result.unwrap_err(),
            ColMajorArrayError::ShapeMismatch {
                shape: vec![2, 2],
                expected: 4,
                actual: 3,
            }
        );
    }

    // -- Out-of-bounds -> None ----------------------------------------------

    #[test]
    fn test_out_of_bounds() {
        let arr = ColMajorArray::new(vec![1, 2, 3, 4], vec![2, 2]).unwrap();
        // Index out of range
        assert_eq!(arr.get(&[2, 0]), None);
        assert_eq!(arr.get(&[0, 2]), None);
        // Wrong number of indices
        assert_eq!(arr.get(&[0]), None);
        assert_eq!(arr.get(&[0, 0, 0]), None);
    }

    // -- Ref and Mut views --------------------------------------------------

    #[test]
    fn test_as_ref() {
        let arr = ColMajorArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        let view = arr.as_ref();
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.get(&[1, 1]), Some(&40));
        assert_eq!(view.data(), arr.data());
    }

    #[test]
    fn test_as_mut() {
        let mut arr = ColMajorArray::new(vec![10, 20, 30, 40], vec![2, 2]).unwrap();
        {
            let mut view = arr.as_mut();
            if let Some(v) = view.get_mut(&[0, 1]) {
                *v = 99;
            }
        }
        assert_eq!(arr.get(&[0, 1]), Some(&99));
    }

    // -- into_data ----------------------------------------------------------

    #[test]
    fn test_into_data() {
        let arr = ColMajorArray::new(vec![1, 2, 3], vec![3]).unwrap();
        let data = arr.into_data();
        assert_eq!(data, vec![1, 2, 3]);
    }

    // -- Empty arrays -------------------------------------------------------

    #[test]
    fn test_empty_array() {
        let arr: ColMajorArray<i32> = ColMajorArray::new(vec![], vec![0]).unwrap();
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.ndim(), 1);
        assert_eq!(arr.nrows(), None);
        assert_eq!(arr.ncols(), None);
    }

    #[test]
    fn test_empty_2d_array() {
        let arr: ColMajorArray<i32> = ColMajorArray::new(vec![], vec![3, 0]).unwrap();
        assert!(arr.is_empty());
        assert_eq!(arr.len(), 0);
        assert_eq!(arr.nrows(), Some(3));
        assert_eq!(arr.ncols(), Some(0));
    }

    // -- ColMajorArrayRef construction --------------------------------------

    #[test]
    fn test_ref_new() {
        let data = [1, 2, 3, 4, 5, 6];
        let shape = [2, 3];
        let view = ColMajorArrayRef::new(&data, &shape).unwrap();
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.len(), 6);
        assert_eq!(view.get(&[1, 2]), Some(&6));
    }

    // -- ColMajorArrayMut construction --------------------------------------

    #[test]
    fn test_mut_new() {
        let mut data = [1, 2, 3, 4, 5, 6];
        let shape = [2, 3];
        let mut view = ColMajorArrayMut::new(&mut data, &shape).unwrap();
        assert_eq!(view.ndim(), 2);
        assert_eq!(view.len(), 6);
        *view.get_mut(&[0, 0]).unwrap() = 100;
        assert_eq!(view.get(&[0, 0]), Some(&100));
    }

    // -- Overflow detection ---------------------------------------------------

    #[test]
    fn test_new_rejects_overflow_shape() {
        let result = ColMajorArray::<u8>::new(vec![], vec![usize::MAX, 2]);
        assert!(
            matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
            "expected ShapeOverflow, got {:?}",
            result
        );
    }

    #[test]
    fn test_filled_rejects_overflow_shape() {
        let result = ColMajorArray::filled(vec![usize::MAX, 2], 0u8);
        assert!(
            matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
            "expected ShapeOverflow, got {:?}",
            result
        );
    }

    #[test]
    fn test_zeros_rejects_overflow_shape() {
        let result = ColMajorArray::<u8>::zeros(vec![usize::MAX, 2]);
        assert!(
            matches!(result, Err(ColMajorArrayError::ShapeOverflow { .. })),
            "expected ShapeOverflow, got {:?}",
            result
        );
    }
}
