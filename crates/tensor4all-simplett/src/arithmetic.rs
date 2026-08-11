//! Arithmetic operations for tensor trains

use crate::error::Result;
use crate::tensortrain::SimpleTensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::{tensor3_zeros, Tensor3Ops};

impl<T: TTScalar> SimpleTensorTrain<T> {
    /// Add two tensor trains element-wise: `result[i] = self[i] + other[i]`.
    ///
    /// The result has bond dimension equal to the **sum** of the input bond
    /// dimensions. Call [`compress`](crate::SimpleTensorTrain::compress) afterward
    /// to reduce the bond dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    /// let c = a.add(&b).unwrap();
    ///
    /// // Every entry = 1 + 2 = 3
    /// assert!((c.evaluate(&[0, 0]).unwrap() - 3.0).abs() < 1e-12);
    /// // Bond dim = 1 + 1 = 2
    /// assert_eq!(c.rank(), 2);
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self> {
        use crate::error::TensorTrainError;

        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot add tensor trains of different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(other.clone());
        }

        let n = self.len();
        let mut tensors = Vec::with_capacity(n);

        for i in 0..n {
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

            let site_dim = a.site_dim();

            if i == 0 && i == n - 1 {
                // Single-site TTs must keep both bond dimensions at 1.
                let mut new_tensor = tensor3_zeros(1, site_dim, 1);
                for s in 0..site_dim {
                    new_tensor.set3(0, s, 0, *a.get3(0, s, 0) + *b.get3(0, s, 0));
                }
                tensors.push(new_tensor);
            } else if i == 0 {
                // First tensor: [A | B] horizontally
                let new_right_dim = a.right_dim() + b.right_dim();
                let mut new_tensor = tensor3_zeros(1, site_dim, new_right_dim);

                for s in 0..site_dim {
                    for r in 0..a.right_dim() {
                        new_tensor.set3(0, s, r, *a.get3(0, s, r));
                    }
                    for r in 0..b.right_dim() {
                        new_tensor.set3(0, s, a.right_dim() + r, *b.get3(0, s, r));
                    }
                }
                tensors.push(new_tensor);
            } else if i == n - 1 {
                // Last tensor: [A; B] vertically
                let new_left_dim = a.left_dim() + b.left_dim();
                let mut new_tensor = tensor3_zeros(new_left_dim, site_dim, 1);

                for l in 0..a.left_dim() {
                    for s in 0..site_dim {
                        new_tensor.set3(l, s, 0, *a.get3(l, s, 0));
                    }
                }
                for l in 0..b.left_dim() {
                    for s in 0..site_dim {
                        new_tensor.set3(a.left_dim() + l, s, 0, *b.get3(l, s, 0));
                    }
                }
                tensors.push(new_tensor);
            } else {
                // Middle tensors: block diagonal [A 0; 0 B]
                let new_left_dim = a.left_dim() + b.left_dim();
                let new_right_dim = a.right_dim() + b.right_dim();
                let mut new_tensor = tensor3_zeros(new_left_dim, site_dim, new_right_dim);

                // Copy A block
                for l in 0..a.left_dim() {
                    for s in 0..site_dim {
                        for r in 0..a.right_dim() {
                            new_tensor.set3(l, s, r, *a.get3(l, s, r));
                        }
                    }
                }
                // Copy B block
                for l in 0..b.left_dim() {
                    for s in 0..site_dim {
                        for r in 0..b.right_dim() {
                            new_tensor.set3(
                                a.left_dim() + l,
                                s,
                                a.right_dim() + r,
                                *b.get3(l, s, r),
                            );
                        }
                    }
                }
                tensors.push(new_tensor);
            }
        }

        Ok(SimpleTensorTrain::from_tensors_unchecked(tensors))
    }

    /// Subtract element-wise: `result[i] = self[i] - other[i]`.
    ///
    /// Equivalent to `self.add(&other.negate())`.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 5.0);
    /// let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
    /// let c = a.sub(&b).unwrap();
    /// assert!((c.evaluate(&[0, 0]).unwrap() - 3.0).abs() < 1e-12);
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let neg_other = other.scale(-T::one());
        self.add(&neg_other)
    }

    /// Negate every entry: `result[i] = -self[i]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain};
    ///
    /// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 7.0);
    /// let neg = tt.negate();
    /// assert!((neg.evaluate(&[0, 0]).unwrap() + 7.0).abs() < 1e-12);
    /// ```
    pub fn negate(&self) -> Self {
        self.scale(-T::one())
    }
}

impl<T: TTScalar> std::ops::Add for SimpleTensorTrain<T> {
    type Output = Result<Self>;

    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn add(self, other: Self) -> Self::Output {
        SimpleTensorTrain::add(&self, &other)
    }
}

impl<T: TTScalar> std::ops::Add for &SimpleTensorTrain<T> {
    type Output = Result<SimpleTensorTrain<T>>;

    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn add(self, other: Self) -> Self::Output {
        SimpleTensorTrain::add(self, other)
    }
}

impl<T: TTScalar> std::ops::Sub for SimpleTensorTrain<T> {
    type Output = Result<Self>;

    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn sub(self, other: Self) -> Self::Output {
        SimpleTensorTrain::sub(&self, &other)
    }
}

impl<T: TTScalar> std::ops::Sub for &SimpleTensorTrain<T> {
    type Output = Result<SimpleTensorTrain<T>>;

    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn sub(self, other: Self) -> Self::Output {
        SimpleTensorTrain::sub(self, other)
    }
}

impl<T: TTScalar> std::ops::Neg for SimpleTensorTrain<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<T: TTScalar> std::ops::Neg for &SimpleTensorTrain<T> {
    type Output = SimpleTensorTrain<T>;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<T: TTScalar> std::ops::Mul<T> for SimpleTensorTrain<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        self.scale(scalar)
    }
}

impl<T: TTScalar> std::ops::Mul<T> for &SimpleTensorTrain<T> {
    type Output = SimpleTensorTrain<T>;

    fn mul(self, scalar: T) -> Self::Output {
        self.scale(scalar)
    }
}

#[cfg(test)]
mod tests;
