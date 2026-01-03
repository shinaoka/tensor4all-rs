//! Abstract traits for tensor train objects

use crate::error::Result;
use crate::types::{LocalIndex, Tensor3};
use num_traits::{One, Zero};

/// Scalar trait for tensor train elements
pub trait TTScalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + 'static
{
    /// Conjugate
    fn conj(self) -> Self;

    /// Absolute value squared
    fn abs_sq(self) -> f64;
}

impl TTScalar for f64 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        self * self
    }
}

impl TTScalar for f32 {
    fn conj(self) -> Self {
        self
    }

    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }
}

impl TTScalar for num_complex::Complex64 {
    fn conj(self) -> Self {
        num_complex::Complex64::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }
}

impl TTScalar for num_complex::Complex32 {
    fn conj(self) -> Self {
        num_complex::Complex32::conj(&self)
    }

    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }
}

/// Abstract trait for tensor train objects
///
/// A tensor train (also known as MPS) represents a high-dimensional tensor
/// as a product of low-rank tensors.
pub trait AbstractTensorTrain<T: TTScalar>: Sized {
    /// Number of sites (tensors) in the tensor train
    fn len(&self) -> usize;

    /// Check if the tensor train is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the site tensor at position i
    fn site_tensor(&self, i: usize) -> &Tensor3<T>;

    /// Get all site tensors
    fn site_tensors(&self) -> &[Tensor3<T>];

    /// Bond dimensions along the links between tensors
    /// Returns a vector of length L-1 where L is the number of sites
    fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.site_tensor(i).left_dim())
            .collect()
    }

    /// Bond dimension at the link between tensor i and i+1
    fn link_dim(&self, i: usize) -> usize {
        self.site_tensor(i + 1).left_dim()
    }

    /// Site dimensions (physical dimensions) for each tensor
    fn site_dims(&self) -> Vec<usize> {
        (0..self.len())
            .map(|i| self.site_tensor(i).site_dim())
            .collect()
    }

    /// Site dimension at position i
    fn site_dim(&self, i: usize) -> usize {
        self.site_tensor(i).site_dim()
    }

    /// Maximum bond dimension (rank) of the tensor train
    fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Evaluate the tensor train at a given index set
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
                    sum = sum + current[l] * slice[l * right_dim + r];
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
                current[r] = current[r] + *first.get(0, s, r);
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
                            site_sum[l * right_dim + r] + *tensor.get(l, s, r);
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
}
