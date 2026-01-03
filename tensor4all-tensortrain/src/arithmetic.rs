//! Arithmetic operations for tensor trains

use crate::error::Result;
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3;

impl<T: TTScalar> TensorTrain<T> {
    /// Add two tensor trains element-wise
    ///
    /// The result has bond dimension equal to the sum of the input bond dimensions.
    /// Use `compress` to reduce the bond dimension afterward.
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

            if i == 0 {
                // First tensor: [A | B] horizontally
                let new_right_dim = a.right_dim() + b.right_dim();
                let mut new_tensor = Tensor3::zeros(1, site_dim, new_right_dim);

                for s in 0..site_dim {
                    for r in 0..a.right_dim() {
                        new_tensor.set(0, s, r, *a.get(0, s, r));
                    }
                    for r in 0..b.right_dim() {
                        new_tensor.set(0, s, a.right_dim() + r, *b.get(0, s, r));
                    }
                }
                tensors.push(new_tensor);
            } else if i == n - 1 {
                // Last tensor: [A; B] vertically
                let new_left_dim = a.left_dim() + b.left_dim();
                let mut new_tensor = Tensor3::zeros(new_left_dim, site_dim, 1);

                for l in 0..a.left_dim() {
                    for s in 0..site_dim {
                        new_tensor.set(l, s, 0, *a.get(l, s, 0));
                    }
                }
                for l in 0..b.left_dim() {
                    for s in 0..site_dim {
                        new_tensor.set(a.left_dim() + l, s, 0, *b.get(l, s, 0));
                    }
                }
                tensors.push(new_tensor);
            } else {
                // Middle tensors: block diagonal [A 0; 0 B]
                let new_left_dim = a.left_dim() + b.left_dim();
                let new_right_dim = a.right_dim() + b.right_dim();
                let mut new_tensor = Tensor3::zeros(new_left_dim, site_dim, new_right_dim);

                // Copy A block
                for l in 0..a.left_dim() {
                    for s in 0..site_dim {
                        for r in 0..a.right_dim() {
                            new_tensor.set(l, s, r, *a.get(l, s, r));
                        }
                    }
                }
                // Copy B block
                for l in 0..b.left_dim() {
                    for s in 0..site_dim {
                        for r in 0..b.right_dim() {
                            new_tensor.set(
                                a.left_dim() + l,
                                s,
                                a.right_dim() + r,
                                *b.get(l, s, r),
                            );
                        }
                    }
                }
                tensors.push(new_tensor);
            }
        }

        Ok(TensorTrain::from_tensors_unchecked(tensors))
    }

    /// Subtract another tensor train from this one
    pub fn sub(&self, other: &Self) -> Result<Self> {
        let neg_other = other.scaled(-T::one());
        self.add(&neg_other)
    }

    /// Negate the tensor train (multiply by -1)
    pub fn negate(&self) -> Self {
        self.scaled(-T::one())
    }
}

impl<T: TTScalar> std::ops::Add for TensorTrain<T> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Self::Output {
        TensorTrain::add(&self, &other)
    }
}

impl<T: TTScalar> std::ops::Add for &TensorTrain<T> {
    type Output = Result<TensorTrain<T>>;

    fn add(self, other: Self) -> Self::Output {
        TensorTrain::add(self, other)
    }
}

impl<T: TTScalar> std::ops::Sub for TensorTrain<T> {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Self::Output {
        TensorTrain::sub(&self, &other)
    }
}

impl<T: TTScalar> std::ops::Sub for &TensorTrain<T> {
    type Output = Result<TensorTrain<T>>;

    fn sub(self, other: Self) -> Self::Output {
        TensorTrain::sub(self, other)
    }
}

impl<T: TTScalar> std::ops::Neg for TensorTrain<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<T: TTScalar> std::ops::Neg for &TensorTrain<T> {
    type Output = TensorTrain<T>;

    fn neg(self) -> Self::Output {
        self.negate()
    }
}

impl<T: TTScalar> std::ops::Mul<T> for TensorTrain<T> {
    type Output = Self;

    fn mul(self, scalar: T) -> Self::Output {
        self.scaled(scalar)
    }
}

impl<T: TTScalar> std::ops::Mul<T> for &TensorTrain<T> {
    type Output = TensorTrain<T>;

    fn mul(self, scalar: T) -> Self::Output {
        self.scaled(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_constant_tensors() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 1.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 2.0);

        let result = tt1.add(&tt2).unwrap();

        // Sum should be (1.0 + 2.0) * 2 * 3 = 18.0
        assert!((result.sum() - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_sub_constant_tensors() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 5.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 2.0);

        let result = tt1.sub(&tt2).unwrap();

        // Sum should be (5.0 - 2.0) * 2 * 3 = 18.0
        assert!((result.sum() - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_negate() {
        let tt = TensorTrain::<f64>::constant(&[2, 2], 3.0);
        let neg_tt = tt.negate();

        assert!((neg_tt.sum() + tt.sum()).abs() < 1e-10);
    }

    #[test]
    fn test_add_operator() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 2], 1.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 2], 1.0);

        let result = (&tt1 + &tt2).unwrap();

        // Sum should be 2.0 * 2 * 2 = 8.0
        assert!((result.sum() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_preserves_evaluation() {
        // Create two simple tensor trains
        let mut t0_a = Tensor3::<f64>::zeros(1, 2, 1);
        t0_a.set(0, 0, 0, 1.0);
        t0_a.set(0, 1, 0, 2.0);

        let mut t1_a = Tensor3::<f64>::zeros(1, 2, 1);
        t1_a.set(0, 0, 0, 3.0);
        t1_a.set(0, 1, 0, 4.0);

        let tt_a = TensorTrain::new(vec![t0_a, t1_a]).unwrap();

        let mut t0_b = Tensor3::<f64>::zeros(1, 2, 1);
        t0_b.set(0, 0, 0, 0.5);
        t0_b.set(0, 1, 0, 1.5);

        let mut t1_b = Tensor3::<f64>::zeros(1, 2, 1);
        t1_b.set(0, 0, 0, 2.5);
        t1_b.set(0, 1, 0, 3.5);

        let tt_b = TensorTrain::new(vec![t0_b, t1_b]).unwrap();

        let result = tt_a.add(&tt_b).unwrap();

        // Test some evaluations
        // result([0, 0]) = a([0,0]) + b([0,0]) = 1*3 + 0.5*2.5 = 3 + 1.25 = 4.25
        let val_00 = result.evaluate(&[0, 0]).unwrap();
        let expected_00 =
            tt_a.evaluate(&[0, 0]).unwrap() + tt_b.evaluate(&[0, 0]).unwrap();
        assert!((val_00 - expected_00).abs() < 1e-10);

        // result([1, 1]) = a([1,1]) + b([1,1]) = 2*4 + 1.5*3.5 = 8 + 5.25 = 13.25
        let val_11 = result.evaluate(&[1, 1]).unwrap();
        let expected_11 =
            tt_a.evaluate(&[1, 1]).unwrap() + tt_b.evaluate(&[1, 1]).unwrap();
        assert!((val_11 - expected_11).abs() < 1e-10);
    }
}
