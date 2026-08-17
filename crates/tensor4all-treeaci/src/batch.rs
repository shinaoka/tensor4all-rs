//! Borrowed column-major batches passed to elementwise operators.

use crate::{Result, TreeAciError};

/// A borrowed batch of tree ACI operator inputs.
///
/// Values form an `n_inputs` by `n_points` column-major matrix. Input is the
/// first (fast) axis, so `(input, point)` is stored at
/// `input + n_inputs * point`.
///
/// Related types: [`TreeAciError`] reports invalid shapes and coordinates.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeElementwiseBatch;
///
/// let values = [10, 20, 11, 21];
/// let batch = TreeElementwiseBatch::new(&values, 2, 2)?;
/// assert_eq!(batch.get(0, 1)?, 11);
/// assert_eq!(batch.get(1, 1)?, 21);
/// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct TreeElementwiseBatch<'a, T> {
    values: &'a [T],
    n_inputs: usize,
    n_points: usize,
}

impl<'a, T> TreeElementwiseBatch<'a, T> {
    /// Creates a column-major borrowed batch.
    ///
    /// `values` contains operator inputs, `n_inputs` is the number of input
    /// trees, and `n_points` is the number of sampled global assignments. Both
    /// axes must be nonempty, and `values.len()` must equal their checked
    /// product.
    ///
    /// Returns a view borrowing `values`; no values are copied.
    ///
    /// # Errors
    ///
    /// Returns [`TreeAciError::EmptyBatchAxis`] when either axis is zero,
    /// [`TreeAciError::SizeOverflow`] when their product does not fit in
    /// `usize`, or [`TreeAciError::LengthMismatch`] when the slice length does
    /// not match that product.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeElementwiseBatch;
    ///
    /// let values = [1.0, 2.0, 3.0, 4.0];
    /// let batch = TreeElementwiseBatch::new(&values, 2, 2)?;
    /// assert_eq!(batch.as_col_major_slice(), &values);
    /// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
    /// ```
    pub fn new(values: &'a [T], n_inputs: usize, n_points: usize) -> Result<Self> {
        if n_inputs == 0 {
            return Err(TreeAciError::EmptyBatchAxis { axis: "input" });
        }
        if n_points == 0 {
            return Err(TreeAciError::EmptyBatchAxis { axis: "point" });
        }
        let expected = n_inputs
            .checked_mul(n_points)
            .ok_or(TreeAciError::SizeOverflow { context: "batch" })?;
        if values.len() != expected {
            return Err(TreeAciError::LengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        Ok(Self {
            values,
            n_inputs,
            n_points,
        })
    }

    /// Returns the number of input trees per point.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeElementwiseBatch;
    ///
    /// let batch = TreeElementwiseBatch::new(&[1, 2], 2, 1)?;
    /// assert_eq!(batch.n_inputs(), 2);
    /// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
    /// ```
    pub fn n_inputs(&self) -> usize {
        self.n_inputs
    }

    /// Returns the number of sampled global assignments.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeElementwiseBatch;
    ///
    /// let batch = TreeElementwiseBatch::new(&[1, 2], 1, 2)?;
    /// assert_eq!(batch.n_points(), 2);
    /// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
    /// ```
    pub fn n_points(&self) -> usize {
        self.n_points
    }

    /// Returns the copied value at `(input, point)`.
    ///
    /// `input` addresses the fast column-major axis and `point` addresses the
    /// slow axis. The returned value is copied from the borrowed slice.
    ///
    /// # Errors
    ///
    /// Returns [`TreeAciError::BatchIndexOutOfBounds`] when either coordinate
    /// lies outside its axis.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeElementwiseBatch;
    ///
    /// let batch = TreeElementwiseBatch::new(&[10, 20, 11, 21], 2, 2)?;
    /// assert_eq!(batch.get(1, 0)?, 20);
    /// assert_eq!(batch.get(0, 1)?, 11);
    /// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
    /// ```
    pub fn get(&self, input: usize, point: usize) -> Result<T>
    where
        T: Copy,
    {
        if input >= self.n_inputs {
            return Err(TreeAciError::BatchIndexOutOfBounds {
                axis: "input",
                index: input,
                len: self.n_inputs,
            });
        }
        if point >= self.n_points {
            return Err(TreeAciError::BatchIndexOutOfBounds {
                axis: "point",
                index: point,
                len: self.n_points,
            });
        }
        Ok(self.values[input + self.n_inputs * point])
    }

    /// Returns the underlying column-major slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeElementwiseBatch;
    ///
    /// let values = [1, 2, 3, 4];
    /// let batch = TreeElementwiseBatch::new(&values, 2, 2)?;
    /// assert_eq!(batch.as_col_major_slice(), values.as_slice());
    /// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
    /// ```
    pub fn as_col_major_slice(&self) -> &'a [T] {
        self.values
    }
}
