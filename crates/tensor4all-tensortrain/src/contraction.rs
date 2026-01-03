//! Contraction operations for tensor trains
//!
//! This module provides various ways to combine tensor trains:
//! - `hadamard`: Element-wise (Hadamard) product
//! - `dot`: Inner product (returns scalar)
//! - `hadamard_zipup`: Hadamard product with on-the-fly compression

use crate::compression::{CompressionMethod, CompressionOptions};
use crate::error::{Result, TensorTrainError};
use crate::tensortrain::TensorTrain;
use crate::traits::{AbstractTensorTrain, TTScalar};
use crate::types::Tensor3;
use tensor4all_matrixci::util::{nrows, ncols, zeros, Matrix, Scalar};
use tensor4all_matrixci::{AbstractMatrixCI, MatrixLUCI, RrLUOptions};

/// Options for contraction with compression
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Tolerance for truncation
    pub tolerance: f64,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Compression method (LU or CI)
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

impl<T: TTScalar + Scalar + Default> TensorTrain<T> {
    /// Compute the Hadamard (element-wise) product of two tensor trains
    ///
    /// For each index i = (i1, i2, ..., iL):
    ///   result[i] = self[i] * other[i]
    ///
    /// The resulting bond dimension is the product of the input bond dimensions.
    /// Use `compress` afterward to reduce the bond dimension.
    pub fn hadamard(&self, other: &Self) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute Hadamard product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(Self::from_tensors_unchecked(Vec::new()));
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
            let new_left_dim = a.left_dim() * b.left_dim();
            let new_right_dim = a.right_dim() * b.right_dim();

            let mut new_tensor = Tensor3::zeros(new_left_dim, site_dim, new_right_dim);

            // Kronecker product on bond indices
            // new[la*lb, s, ra*rb] = a[la, s, ra] * b[lb, s, rb]
            for la in 0..a.left_dim() {
                for lb in 0..b.left_dim() {
                    for s in 0..site_dim {
                        for ra in 0..a.right_dim() {
                            for rb in 0..b.right_dim() {
                                let new_l = la * b.left_dim() + lb;
                                let new_r = ra * b.right_dim() + rb;
                                let val = *a.get(la, s, ra) * *b.get(lb, s, rb);
                                new_tensor.set(new_l, s, new_r, val);
                            }
                        }
                    }
                }
            }

            tensors.push(new_tensor);
        }

        Ok(TensorTrain::from_tensors_unchecked(tensors))
    }

    /// Compute the inner product (dot product) of two tensor trains
    ///
    /// Returns: sum over all indices i of self[i] * other[i]
    ///
    /// This is equivalent to hadamard(self, other).sum() but more efficient.
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

        let mut result: Matrix<T> = zeros(a0.right_dim(), b0.right_dim());
        for s in 0..a0.site_dim() {
            for ra in 0..a0.right_dim() {
                for rb in 0..b0.right_dim() {
                    result[[ra, rb]] = result[[ra, rb]] + *a0.get(0, s, ra) * *b0.get(0, s, rb);
                }
            }
        }

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

            // new_result[ra', rb'] = sum_{la, lb, s} result[la, lb] * a[la, s, ra'] * b[lb, s, rb']
            let mut new_result: Matrix<T> = zeros(a.right_dim(), b.right_dim());

            for la in 0..a.left_dim() {
                for lb in 0..b.left_dim() {
                    let r_val = result[[la, lb]];
                    for s in 0..a.site_dim() {
                        for ra in 0..a.right_dim() {
                            for rb in 0..b.right_dim() {
                                new_result[[ra, rb]] = new_result[[ra, rb]]
                                    + r_val * *a.get(la, s, ra) * *b.get(lb, s, rb);
                            }
                        }
                    }
                }
            }

            result = new_result;
        }

        // Final result should be 1x1
        Ok(result[[0, 0]])
    }

    /// Compute Hadamard product with on-the-fly compression (zip-up algorithm)
    ///
    /// This is more efficient than hadamard() followed by compress() for
    /// tensor trains with large bond dimensions, as it compresses at each step.
    ///
    /// Reference: https://tensornetwork.org/mps/algorithms/zip_up_mpo/
    pub fn hadamard_zipup(&self, other: &Self, options: &ContractionOptions) -> Result<Self> {
        if self.len() != other.len() {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Cannot compute Hadamard product of tensor trains with different lengths: {} vs {}",
                    self.len(),
                    other.len()
                ),
            });
        }

        if self.is_empty() {
            return Ok(Self::from_tensors_unchecked(Vec::new()));
        }

        let n = self.len();

        // R tensor carries the remainder from the previous factorization
        // R[link_new, link_a, link_b]
        let mut r_tensor: Tensor3<T> = Tensor3::zeros(1, 1, 1);
        r_tensor.set(0, 0, 0, T::one());

        let mut result_tensors = Vec::with_capacity(n);

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

            // Contract R with A and B:
            // C[link_new, s, link_a', link_b'] = sum_{link_a, link_b} R[link_new, link_a, link_b]
            //                                   * A[link_a, s, link_a'] * B[link_b, s, link_b']
            let link_new = r_tensor.left_dim();
            let link_a_next = a.right_dim();
            let link_b_next = b.right_dim();

            let mut c_tensor = Tensor3::<T>::zeros(
                link_new * site_dim,
                link_a_next,
                link_b_next,
            );

            for ln in 0..r_tensor.left_dim() {
                for la in 0..a.left_dim() {
                    for lb in 0..b.left_dim() {
                        let r_val = *r_tensor.get(ln, la, lb);
                        for s in 0..site_dim {
                            for ra in 0..link_a_next {
                                for rb in 0..link_b_next {
                                    let val = r_val * *a.get(la, s, ra) * *b.get(lb, s, rb);
                                    let row = ln * site_dim + s;
                                    let old_val = *c_tensor.get(row, ra, rb);
                                    c_tensor.set(row, ra, rb, old_val + val);
                                }
                            }
                        }
                    }
                }
            }

            if i == n - 1 {
                // Last site: just reshape and store
                let mut result_tensor = Tensor3::zeros(link_new, site_dim, 1);
                for ln in 0..link_new {
                    for s in 0..site_dim {
                        // Sum over all link_a', link_b' (should be 1x1)
                        result_tensor.set(ln, s, 0, *c_tensor.get(ln * site_dim + s, 0, 0));
                    }
                }
                result_tensors.push(result_tensor);
            } else {
                // Factorize C into left and right parts
                // C is (link_new * site_dim) x (link_a' * link_b')
                let rows = link_new * site_dim;
                let cols = link_a_next * link_b_next;

                let mut c_mat: Matrix<T> = zeros(rows, cols);
                for row in 0..rows {
                    for ra in 0..link_a_next {
                        for rb in 0..link_b_next {
                            c_mat[[row, ra * link_b_next + rb]] = *c_tensor.get(row, ra, rb);
                        }
                    }
                }

                // Apply LU-based factorization
                let lu_options = RrLUOptions {
                    max_rank: options.max_bond_dim,
                    rel_tol: options.tolerance,
                    abs_tol: 0.0,
                    left_orthogonal: true,
                };

                let luci = MatrixLUCI::from_matrix(&c_mat, Some(lu_options));
                let left = luci.left();
                let right = luci.right();
                let new_bond_dim = luci.rank().max(1);

                // Store left factor as site tensor: (link_new, site_dim, new_bond_dim)
                let mut result_tensor = Tensor3::zeros(link_new, site_dim, new_bond_dim);
                for ln in 0..link_new {
                    for s in 0..site_dim {
                        for r in 0..new_bond_dim {
                            let row = ln * site_dim + s;
                            if row < nrows(&left) && r < ncols(&left) {
                                result_tensor.set(ln, s, r, left[[row, r]]);
                            }
                        }
                    }
                }
                result_tensors.push(result_tensor);

                // Update R tensor for next iteration: (new_bond_dim, link_a', link_b')
                r_tensor = Tensor3::zeros(new_bond_dim, link_a_next, link_b_next);
                for l in 0..new_bond_dim {
                    for ra in 0..link_a_next {
                        for rb in 0..link_b_next {
                            let col = ra * link_b_next + rb;
                            if l < nrows(&right) && col < ncols(&right) {
                                r_tensor.set(l, ra, rb, right[[l, col]]);
                            }
                        }
                    }
                }
            }
        }

        Ok(TensorTrain::from_tensors_unchecked(result_tensors))
    }

    /// Compute Hadamard product followed by compression
    ///
    /// Equivalent to self.hadamard(other)?.compress(options) but may be
    /// more efficient for large bond dimensions.
    pub fn hadamard_compressed(
        &self,
        other: &Self,
        options: &CompressionOptions,
    ) -> Result<Self> {
        let mut result = self.hadamard(other)?;
        result.compress(options)?;
        Ok(result)
    }
}

/// Convenience function to compute Hadamard product
pub fn hadamard<T: TTScalar + Scalar + Default>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
) -> Result<TensorTrain<T>> {
    a.hadamard(b)
}

/// Convenience function to compute dot product
pub fn dot<T: TTScalar + Scalar + Default>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
) -> Result<T> {
    a.dot(b)
}

/// Convenience function for Hadamard with zip-up compression
pub fn hadamard_zipup<T: TTScalar + Scalar + Default>(
    a: &TensorTrain<T>,
    b: &TensorTrain<T>,
    options: &ContractionOptions,
) -> Result<TensorTrain<T>> {
    a.hadamard_zipup(b, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard_constant() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);

        let result = tt1.hadamard(&tt2).unwrap();

        // Each element should be 2.0 * 3.0 = 6.0
        // Sum should be 6.0 * 2 * 3 = 36.0
        assert!((result.sum() - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_preserves_evaluation() {
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
        t1_b.set(0, 0, 0, 2.0);
        t1_b.set(0, 1, 0, 3.0);

        let tt_b = TensorTrain::new(vec![t0_b, t1_b]).unwrap();

        let result = tt_a.hadamard(&tt_b).unwrap();

        // Test evaluations
        // result([0, 0]) = a([0,0]) * b([0,0]) = (1*3) * (0.5*2) = 3 * 1 = 3
        let val_00 = result.evaluate(&[0, 0]).unwrap();
        let expected_00 = tt_a.evaluate(&[0, 0]).unwrap() * tt_b.evaluate(&[0, 0]).unwrap();
        assert!((val_00 - expected_00).abs() < 1e-10);

        // result([1, 1]) = a([1,1]) * b([1,1]) = (2*4) * (1.5*3) = 8 * 4.5 = 36
        let val_11 = result.evaluate(&[1, 1]).unwrap();
        let expected_11 = tt_a.evaluate(&[1, 1]).unwrap() * tt_b.evaluate(&[1, 1]).unwrap();
        assert!((val_11 - expected_11).abs() < 1e-10);
    }

    #[test]
    fn test_dot_constant() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);

        let result = tt1.dot(&tt2).unwrap();

        // Each element product is 2.0 * 3.0 = 6.0
        // Sum over 2*3=6 elements: 6.0 * 6 = 36.0
        assert!((result - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_equals_hadamard_sum() {
        let mut t0_a = Tensor3::<f64>::zeros(1, 3, 2);
        for s in 0..3 {
            for r in 0..2 {
                t0_a.set(0, s, r, (s + r + 1) as f64);
            }
        }

        let mut t1_a = Tensor3::<f64>::zeros(2, 2, 1);
        for l in 0..2 {
            for s in 0..2 {
                t1_a.set(l, s, 0, (l + s + 1) as f64);
            }
        }

        let tt_a = TensorTrain::new(vec![t0_a.clone(), t1_a.clone()]).unwrap();

        let mut t0_b = Tensor3::<f64>::zeros(1, 3, 1);
        for s in 0..3 {
            t0_b.set(0, s, 0, (s + 1) as f64 * 0.5);
        }

        let mut t1_b = Tensor3::<f64>::zeros(1, 2, 1);
        for s in 0..2 {
            t1_b.set(0, s, 0, (s + 2) as f64 * 0.3);
        }

        let tt_b = TensorTrain::new(vec![t0_b, t1_b]).unwrap();

        let hadamard_result = tt_a.hadamard(&tt_b).unwrap();
        let hadamard_sum = hadamard_result.sum();

        let dot_result = tt_a.dot(&tt_b).unwrap();

        assert!(
            (hadamard_sum - dot_result).abs() < 1e-10,
            "hadamard.sum() = {}, dot = {}",
            hadamard_sum,
            dot_result
        );
    }

    #[test]
    fn test_hadamard_zipup_constant() {
        let tt1 = TensorTrain::<f64>::constant(&[2, 3], 2.0);
        let tt2 = TensorTrain::<f64>::constant(&[2, 3], 3.0);

        let options = ContractionOptions::default();
        let result = tt1.hadamard_zipup(&tt2, &options).unwrap();

        // Each element should be 2.0 * 3.0 = 6.0
        // Sum should be 6.0 * 2 * 3 = 36.0
        assert!((result.sum() - 36.0).abs() < 1e-10);
    }

    #[test]
    fn test_hadamard_zipup_preserves_values() {
        let mut t0_a = Tensor3::<f64>::zeros(1, 2, 2);
        t0_a.set(0, 0, 0, 1.0);
        t0_a.set(0, 0, 1, 0.5);
        t0_a.set(0, 1, 0, 2.0);
        t0_a.set(0, 1, 1, 1.0);

        let mut t1_a = Tensor3::<f64>::zeros(2, 2, 1);
        t1_a.set(0, 0, 0, 1.0);
        t1_a.set(0, 1, 0, 2.0);
        t1_a.set(1, 0, 0, 0.5);
        t1_a.set(1, 1, 0, 1.5);

        let tt_a = TensorTrain::new(vec![t0_a, t1_a]).unwrap();

        let tt_b = TensorTrain::<f64>::constant(&[2, 2], 2.0);

        let options = ContractionOptions::default();
        let zipup_result = tt_a.hadamard_zipup(&tt_b, &options).unwrap();
        let naive_result = tt_a.hadamard(&tt_b).unwrap();

        // Check that values match
        for i in 0..2 {
            for j in 0..2 {
                let zipup_val = zipup_result.evaluate(&[i, j]).unwrap();
                let naive_val = naive_result.evaluate(&[i, j]).unwrap();
                assert!(
                    (zipup_val - naive_val).abs() < 1e-10,
                    "Mismatch at [{}, {}]: zipup={}, naive={}",
                    i,
                    j,
                    zipup_val,
                    naive_val
                );
            }
        }
    }

    #[test]
    fn test_hadamard_zipup_compresses() {
        // Create tensor trains with higher bond dimensions
        let mut t0_a = Tensor3::<f64>::zeros(1, 2, 3);
        for s in 0..2 {
            for r in 0..3 {
                t0_a.set(0, s, r, 1.0);
            }
        }

        let mut t1_a = Tensor3::<f64>::zeros(3, 2, 1);
        for l in 0..3 {
            for s in 0..2 {
                t1_a.set(l, s, 0, 1.0);
            }
        }

        let tt_a = TensorTrain::new(vec![t0_a, t1_a]).unwrap();
        let tt_b = tt_a.clone();

        // Naive Hadamard would give bond dim 3*3=9
        let naive_result = tt_a.hadamard(&tt_b).unwrap();
        assert_eq!(naive_result.link_dims(), vec![9]);

        // Zipup should compress
        let options = ContractionOptions {
            tolerance: 1e-12,
            max_bond_dim: 5,
            method: CompressionMethod::LU,
        };
        let zipup_result = tt_a.hadamard_zipup(&tt_b, &options).unwrap();

        // Check bond dimension is reduced
        assert!(
            zipup_result.link_dims()[0] <= 5,
            "Expected bond dim <= 5, got {}",
            zipup_result.link_dims()[0]
        );

        // Values should still be close
        assert!(
            (zipup_result.sum() - naive_result.sum()).abs() < 1e-8,
            "Sum mismatch: zipup={}, naive={}",
            zipup_result.sum(),
            naive_result.sum()
        );
    }
}
