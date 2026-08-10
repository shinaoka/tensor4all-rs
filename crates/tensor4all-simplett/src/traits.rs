//! Abstract traits for tensor train objects

use crate::error::Result;
use crate::types::{LocalIndex, Tensor3, Tensor3Ops};
use tenferro_tensor::TensorScalar;

/// Scalar trait bound shared by all simplett tensor types.
///
/// Combines [`tensor4all_core::CommonScalar`] (arithmetic, conversion) with
/// [`tenferro_tensor::TensorScalar`] (backend compatibility). Both `f64` and
/// `Complex64` implement this trait.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::TTScalar;
///
/// // f64 satisfies TTScalar
/// fn uses_ttscalar<T: TTScalar>(x: T, y: T) -> T { x + y }
///
/// let result = uses_ttscalar(1.0_f64, 2.0_f64);
/// assert!((result - 3.0).abs() < 1e-15);
///
/// // Complex64 also satisfies TTScalar
/// use num_complex::Complex64;
/// let c = uses_ttscalar(Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0));
/// assert!((c.re - 1.0).abs() < 1e-15);
/// assert!((c.im - 1.0).abs() < 1e-15);
/// ```
pub trait TTScalar: tensor4all_core::CommonScalar + TensorScalar {}

impl<T> TTScalar for T where T: tensor4all_core::CommonScalar + TensorScalar {}

/// Common interface implemented by all tensor train representations.
///
/// Provides read-only access to site tensors plus derived operations:
/// [`evaluate`](Self::evaluate), [`sum`](Self::sum),
/// [`norm`](Self::norm), and [`log_norm`](Self::log_norm).
///
/// # Implementors
///
/// - [`TensorTrain`](crate::TensorTrain) -- primary container
/// - [`SiteTensorTrain`](crate::SiteTensorTrain) -- center-canonical form
/// - [`VidalTensorTrain`](crate::VidalTensorTrain) -- Vidal form (after
///
///   conversion to `TensorTrain`)
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
///
/// // TensorTrain implements AbstractTensorTrain.
/// let tt = TensorTrain::<f64>::constant(&[2, 3, 4], 1.0);
///
/// // Query structure
/// assert_eq!(tt.len(), 3);
/// assert!(!tt.is_empty());
/// assert_eq!(tt.site_dims(), vec![2, 3, 4]);
/// assert_eq!(tt.site_dim(1), 3);
/// assert_eq!(tt.link_dims(), vec![1, 1]);
///
/// // Evaluate, sum, and norm
/// let val = tt.evaluate(&[0, 0, 0]).unwrap();
/// assert!((val - 1.0).abs() < 1e-12);
///
/// let s = tt.sum();
/// assert!((s - 24.0).abs() < 1e-10);
///
/// let n = tt.norm();
/// assert!((n - 24.0_f64.sqrt()).abs() < 1e-10);
/// ```
pub trait AbstractTensorTrain<T: TTScalar>: Sized {
    /// Number of sites (core tensors) in the tensor train.
    fn len(&self) -> usize;

    /// Returns `true` if the tensor train has zero sites.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the rank-3 core tensor at site `i`.
    fn site_tensor(&self, i: usize) -> &Tensor3<T>;

    /// Borrow all core tensors as a slice.
    fn site_tensors(&self) -> &[Tensor3<T>];

    /// Bond dimensions at every link (length = `len() - 1`).
    fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.site_tensor(i).left_dim())
            .collect()
    }

    /// Bond dimension at the link between site `i` and site `i+1`.
    fn link_dim(&self, i: usize) -> usize {
        self.site_tensor(i + 1).left_dim()
    }

    /// Physical (site) dimensions for every site.
    fn site_dims(&self) -> Vec<usize> {
        (0..self.len())
            .map(|i| self.site_tensor(i).site_dim())
            .collect()
    }

    /// Physical (site) dimension at site `i`.
    fn site_dim(&self, i: usize) -> usize {
        self.site_tensor(i).site_dim()
    }

    /// Maximum bond dimension across all links.
    fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Evaluate the tensor train at a given index set
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
    ///
    /// // Constant TT: all values are 5.0
    /// let tt = TensorTrain::<f64>::constant(&[3, 4], 5.0);
    ///
    /// let val = tt.evaluate(&[1, 2]).unwrap();
    /// assert!((val - 5.0).abs() < 1e-12);
    ///
    /// // Wrong number of indices returns an error
    /// assert!(tt.evaluate(&[0]).is_err());
    /// ```
    fn evaluate(&self, indices: &[LocalIndex]) -> Result<T> {
        use crate::error::TensorTrainError;

        if indices.len() != self.len() {
            return Err(TensorTrainError::IndexLengthMismatch {
                expected: self.len(),
                got: indices.len(),
            });
        }

        if self.is_empty() {
            return Err(TensorTrainError::Empty);
        }

        // Start with the first tensor slice
        let first = self.site_tensor(0);
        let idx0 = indices[0];
        if idx0 >= first.site_dim() {
            return Err(TensorTrainError::IndexOutOfBounds {
                site: 0,
                index: idx0,
                max: first.site_dim(),
            });
        }

        // Vector of size right_dim
        let mut current: Vec<T> = first.slice_site(idx0);

        // Contract with remaining tensors
        for (site, &idx) in indices.iter().enumerate().skip(1) {
            let tensor = self.site_tensor(site);
            if idx >= tensor.site_dim() {
                return Err(TensorTrainError::IndexOutOfBounds {
                    site,
                    index: idx,
                    max: tensor.site_dim(),
                });
            }

            let slice = tensor.slice_site(idx);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // Contract: current (of size left_dim) with slice (left_dim x right_dim)
            let mut next = vec![T::zero(); right_dim];
            for r in 0..right_dim {
                let mut sum = T::zero();
                for l in 0..left_dim {
                    sum = sum + current[l] * slice[l + left_dim * r];
                }
                next[r] = sum;
            }
            current = next;
        }

        // Should have a single element
        if current.len() != 1 {
            return Err(TensorTrainError::InvalidOperation {
                message: format!(
                    "Final contraction resulted in {} elements, expected 1",
                    current.len()
                ),
            });
        }

        Ok(current[0])
    }

    /// Sum over all indices of the tensor train
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
    ///
    /// // Constant TT with value 2.0 over 3×4 grid: sum = 2.0 * 3 * 4 = 24.0
    /// let tt = TensorTrain::<f64>::constant(&[3, 4], 2.0);
    /// let s = tt.sum();
    /// assert!((s - 24.0).abs() < 1e-10);
    ///
    /// // Zero TT sums to 0.0
    /// let zero_tt = TensorTrain::<f64>::zeros(&[2, 3]);
    /// assert!((zero_tt.sum() - 0.0).abs() < 1e-12);
    /// ```
    #[allow(clippy::needless_range_loop)]
    fn sum(&self) -> T {
        if self.is_empty() {
            return T::zero();
        }

        // Start with sum over first tensor
        let first = self.site_tensor(0);
        let mut current = vec![T::zero(); first.right_dim()];
        for s in 0..first.site_dim() {
            for r in 0..first.right_dim() {
                current[r] = current[r] + *first.get3(0, s, r);
            }
        }

        // Contract with sums of remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // Sum over site index
            let mut site_sum = vec![T::zero(); left_dim * right_dim];
            for l in 0..left_dim {
                for s in 0..tensor.site_dim() {
                    for r in 0..right_dim {
                        site_sum[l * right_dim + r] =
                            site_sum[l * right_dim + r] + *tensor.get3(l, s, r);
                    }
                }
            }

            // Contract with current
            let mut next = vec![T::zero(); right_dim];
            for r in 0..right_dim {
                let mut sum = T::zero();
                for l in 0..left_dim {
                    sum = sum + current[l] * site_sum[l * right_dim + r];
                }
                next[r] = sum;
            }
            current = next;
        }

        current[0]
    }

    /// Squared Frobenius norm: `sum_i |T[i]|^2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
    ///
    /// // Constant TT: T[i,j] = 2.0 on a 3x4 grid
    /// let tt = TensorTrain::<f64>::constant(&[3, 4], 2.0);
    /// // norm^2 = 2^2 * 3 * 4 = 48
    /// assert!((tt.norm2() - 48.0).abs() < 1e-10);
    /// ```
    fn norm2(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }

        // Contract tt with its conjugate
        // result[la, la_conj, ra, ra_conj] at each step
        let first = self.site_tensor(0);
        let right_dim = first.right_dim();

        // current[ra, ra_conj] = sum_s first[0, s, ra] * conj(first[0, s, ra_conj])
        let mut current = vec![T::zero(); right_dim * right_dim];
        for s in 0..first.site_dim() {
            for ra in 0..right_dim {
                for ra_c in 0..right_dim {
                    let idx = ra * right_dim + ra_c;
                    current[idx] =
                        current[idx] + *first.get3(0, s, ra) * first.get3(0, s, ra_c).conj();
                }
            }
        }

        // Contract through remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // new_current[ra, ra_conj] = sum_{la, la_conj, s}
            //     current[la, la_conj] * tensor[la, s, ra] * conj(tensor[la_conj, s, ra_conj])
            let mut new_current = vec![T::zero(); right_dim * right_dim];

            for la in 0..left_dim {
                for la_c in 0..left_dim {
                    let c_val = current[la * left_dim + la_c];
                    for s in 0..tensor.site_dim() {
                        for ra in 0..right_dim {
                            for ra_c in 0..right_dim {
                                let idx = ra * right_dim + ra_c;
                                new_current[idx] = new_current[idx]
                                    + c_val
                                        * *tensor.get3(la, s, ra)
                                        * tensor.get3(la_c, s, ra_c).conj();
                            }
                        }
                    }
                }
            }

            current = new_current;
        }

        // Final result should be a single element (1x1)
        current[0].abs_sq().sqrt()
    }

    /// Frobenius norm: `sqrt(sum_i |T[i]|^2)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
    ///
    /// let tt = TensorTrain::<f64>::constant(&[3, 4], 2.0);
    /// // norm = sqrt(48) ~ 6.928
    /// assert!((tt.norm() - 48.0_f64.sqrt()).abs() < 1e-10);
    /// ```
    fn norm(&self) -> f64 {
        self.norm2().sqrt()
    }

    /// Logarithm of the Frobenius norm: `ln(norm())`.
    ///
    /// This is more numerically stable than `norm().ln()` for tensor trains
    /// with very large or very small norms, because it normalizes at each
    /// contraction step to avoid overflow/underflow.
    ///
    /// Returns `f64::NEG_INFINITY` for zero tensor trains.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{TensorTrain, AbstractTensorTrain};
    ///
    /// let tt = TensorTrain::<f64>::constant(&[3, 4], 2.0);
    /// let log_n = tt.log_norm();
    /// assert!((log_n - tt.norm().ln()).abs() < 1e-10);
    ///
    /// // Zero TT returns negative infinity
    /// let zero_tt = TensorTrain::<f64>::zeros(&[2, 3]);
    /// assert_eq!(zero_tt.log_norm(), f64::NEG_INFINITY);
    /// ```
    fn log_norm(&self) -> f64 {
        if self.is_empty() {
            return f64::NEG_INFINITY;
        }

        // Contract tt with its conjugate, normalizing at each step
        // to avoid overflow/underflow
        let first = self.site_tensor(0);
        let right_dim = first.right_dim();

        // current[ra, ra_conj] = sum_s first[0, s, ra] * conj(first[0, s, ra_conj])
        let mut current = vec![T::zero(); right_dim * right_dim];
        for s in 0..first.site_dim() {
            for ra in 0..right_dim {
                for ra_c in 0..right_dim {
                    let idx = ra * right_dim + ra_c;
                    current[idx] =
                        current[idx] + *first.get3(0, s, ra) * first.get3(0, s, ra_c).conj();
                }
            }
        }

        // Normalize and accumulate log scale
        let mut log_scale = 0.0;
        let scale = current
            .iter()
            .map(|x| x.abs_sq())
            .fold(0.0, f64::max)
            .sqrt();
        if scale > 0.0 {
            log_scale += scale.ln();
            let inv_scale = T::one() / T::from_f64(scale);
            for val in &mut current {
                *val = *val * inv_scale;
            }
        } else if self.len() == 1 {
            // Single site with zero norm
            return f64::NEG_INFINITY;
        }

        // Contract through remaining tensors
        for site in 1..self.len() {
            let tensor = self.site_tensor(site);
            let left_dim = tensor.left_dim();
            let right_dim = tensor.right_dim();

            // new_current[ra, ra_conj] = sum_{la, la_conj, s}
            //     current[la, la_conj] * tensor[la, s, ra] * conj(tensor[la_conj, s, ra_conj])
            let mut new_current = vec![T::zero(); right_dim * right_dim];

            for la in 0..left_dim {
                for la_c in 0..left_dim {
                    let c_val = current[la * left_dim + la_c];
                    for s in 0..tensor.site_dim() {
                        for ra in 0..right_dim {
                            for ra_c in 0..right_dim {
                                let idx = ra * right_dim + ra_c;
                                new_current[idx] = new_current[idx]
                                    + c_val
                                        * *tensor.get3(la, s, ra)
                                        * tensor.get3(la_c, s, ra_c).conj();
                            }
                        }
                    }
                }
            }

            // Normalize and accumulate log scale
            let scale = new_current
                .iter()
                .map(|x| x.abs_sq())
                .fold(0.0, f64::max)
                .sqrt();
            if scale > 0.0 {
                log_scale += scale.ln();
                let inv_scale = T::one() / T::from_f64(scale);
                for val in &mut new_current {
                    *val = *val * inv_scale;
                }
            }

            current = new_current;
        }

        // Final result: norm = sqrt(norm2), where norm2 = final_val * cumulative_scale
        // log(norm) = 0.5 * log(norm2) = 0.5 * (log(final_val) + log_scale)
        let final_val = current[0].abs_sq().sqrt();
        if final_val > 0.0 {
            0.5 * (final_val.ln() + log_scale)
        } else {
            f64::NEG_INFINITY
        }
    }
}
