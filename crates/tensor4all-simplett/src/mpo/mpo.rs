//! MPO (Matrix Product Operator) implementation
//!
//! An MPO represents a high-dimensional operator as a product of
//! 4D tensors with shape (left_bond, site_dim_1, site_dim_2, right_bond).

use super::error::{MPOError, Result};
use super::types::{tensor4_zeros, LocalIndex, Tensor4, Tensor4Ops};
use crate::einsum_helper::{
    einsum_tensors, row_vector_times_matrix, tensor_to_col_major_vec, EinsumScalar,
};
use crate::traits::TTScalar;

/// Matrix Product Operator representation
/// An MPO represents a high-dimensional operator as a product of
/// lower-dimensional tensors:
/// O[i1, j1, i2, j2, ..., iL, jL] = A1[i1, j1] * A2[i2, j2] * ... * AL[iL, jL]
/// where each Ak[ik, jk] is a matrix of shape (rk-1, rk) with physical indices ik, jk.
#[derive(Debug, Clone)]
pub struct MPO<T: TTScalar> {
    /// The tensors that make up the MPO
    /// Each tensor has shape (left_bond, site_dim_1, site_dim_2, right_bond)
    tensors: Vec<Tensor4<T>>,
}

impl<T: TTScalar> MPO<T> {
    /// Create a new MPO from a list of 4D tensors
    ///
    /// Each tensor should have shape (left_bond, site_dim_1, site_dim_2, right_bond)
    /// where the right_bond of tensor i equals the left_bond of tensor i+1.
    /// # Errors
    ///
    /// Returns an error when the MPO dimensions are invalid (a shape mismatch) or
    /// /// the construction fails.
    ///
    pub fn new(tensors: Vec<Tensor4<T>>) -> Result<Self> {
        // Validate dimensions
        for i in 0..tensors.len().saturating_sub(1) {
            if tensors[i].right_dim() != tensors[i + 1].left_dim() {
                return Err(MPOError::BondDimensionMismatch {
                    site: i,
                    left_right: tensors[i].right_dim(),
                    right_left: tensors[i + 1].left_dim(),
                });
            }
        }

        // First tensor should have left_dim = 1
        if !tensors.is_empty() && tensors[0].left_dim() != 1 {
            return Err(MPOError::InvalidBoundary);
        }

        // Last tensor should have right_dim = 1
        if let Some(last) = tensors.last() {
            if last.right_dim() != 1 {
                return Err(MPOError::InvalidBoundary);
            }
        }

        Ok(Self { tensors })
    }

    /// Create an MPO from tensors without dimension validation
    /// (for internal use when dimensions are known to be correct)
    pub(crate) fn from_tensors_unchecked(tensors: Vec<Tensor4<T>>) -> Self {
        Self { tensors }
    }

    /// Create an MPO representing the zero operator
    pub fn zeros(site_dims: &[(usize, usize)]) -> Self {
        let tensors: Vec<Tensor4<T>> = site_dims
            .iter()
            .map(|&(d1, d2)| tensor4_zeros(1, d1, d2, 1))
            .collect();
        Self { tensors }
    }

    /// Create an MPO representing a constant operator
    ///
    /// Each element O[i1, j1, i2, j2, ..., iL, jL] = value
    pub fn constant(site_dims: &[(usize, usize)], value: T) -> Self {
        if site_dims.is_empty() {
            return Self {
                tensors: Vec::new(),
            };
        }

        let n = site_dims.len();
        let mut tensors = Vec::with_capacity(n);

        // First tensor: all ones
        let (d1, d2) = site_dims[0];
        let mut first = tensor4_zeros(1, d1, d2, 1);
        for s1 in 0..d1 {
            for s2 in 0..d2 {
                first.set4(0, s1, s2, 0, T::one());
            }
        }
        tensors.push(first);

        // Middle tensors: all ones (only if n > 2)
        if n > 2 {
            for &(d1, d2) in &site_dims[1..n - 1] {
                let mut tensor = tensor4_zeros(1, d1, d2, 1);
                for s1 in 0..d1 {
                    for s2 in 0..d2 {
                        tensor.set4(0, s1, s2, 0, T::one());
                    }
                }
                tensors.push(tensor);
            }
        }

        // Last tensor: multiply by value
        if n > 1 {
            let (d1, d2) = site_dims[n - 1];
            let mut last = tensor4_zeros(1, d1, d2, 1);
            for s1 in 0..d1 {
                for s2 in 0..d2 {
                    last.set4(0, s1, s2, 0, value);
                }
            }
            tensors.push(last);
        } else {
            // Single site: multiply the first (and only) tensor by value
            let (d1, d2) = site_dims[0];
            for s1 in 0..d1 {
                for s2 in 0..d2 {
                    let current = *tensors[0].get4(0, s1, s2, 0);
                    tensors[0].set4(0, s1, s2, 0, current * value);
                }
            }
        }

        Self { tensors }
    }

    /// Create an identity MPO (only when site_dim_1 == site_dim_2 at each site)
    ///
    /// The identity operator: O[i1, j1, ...] = delta(i1, j1) * delta(i2, j2) * ...
    /// # Errors
    ///
    /// Returns an error when the identity MPO construction fails (a shape
    /// /// mismatch or backend failure).
    ///
    pub fn identity(site_dims: &[usize]) -> Result<Self> {
        if site_dims.is_empty() {
            return Ok(Self {
                tensors: Vec::new(),
            });
        }

        let mut tensors = Vec::with_capacity(site_dims.len());

        for &d in site_dims {
            let mut tensor = tensor4_zeros(1, d, d, 1);
            for s in 0..d {
                tensor.set4(0, s, s, 0, T::one());
            }
            tensors.push(tensor);
        }

        Ok(Self { tensors })
    }

    /// Number of sites (tensors) in the MPO
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Check if the MPO is empty
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Get the site tensor at position i
    pub fn site_tensor(&self, i: usize) -> &Tensor4<T> {
        &self.tensors[i]
    }

    /// Get mutable reference to the site tensor at position i
    pub fn site_tensor_mut(&mut self, i: usize) -> &mut Tensor4<T> {
        &mut self.tensors[i]
    }

    /// Get all site tensors
    pub fn site_tensors(&self) -> &[Tensor4<T>] {
        &self.tensors
    }

    /// Get mutable access to the site tensors
    pub fn site_tensors_mut(&mut self) -> &mut [Tensor4<T>] {
        &mut self.tensors
    }

    /// Bond dimensions along the links between tensors
    /// Returns a vector of length L-1 where L is the number of sites
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        (1..self.len())
            .map(|i| self.tensors[i].left_dim())
            .collect()
    }

    /// Bond dimension at the link between tensor i and i+1
    pub fn link_dim(&self, i: usize) -> usize {
        self.tensors[i + 1].left_dim()
    }

    /// Site dimensions (physical dimensions) for each tensor
    /// Returns a vector of (site_dim_1, site_dim_2) tuples
    pub fn site_dims(&self) -> Vec<(usize, usize)> {
        self.tensors
            .iter()
            .map(|t| (t.site_dim_1(), t.site_dim_2()))
            .collect()
    }

    /// Site dimensions at position i
    pub fn site_dim(&self, i: usize) -> (usize, usize) {
        (self.tensors[i].site_dim_1(), self.tensors[i].site_dim_2())
    }

    /// Maximum bond dimension (rank) of the MPO
    pub fn rank(&self) -> usize {
        let lds = self.link_dims();
        if lds.is_empty() {
            1
        } else {
            *lds.iter().max().unwrap_or(&1)
        }
    }

    /// Evaluate the MPO at a given index set
    ///
    /// indices should have length 2*L where L is the number of sites
    /// alternating between site_dim_1 and site_dim_2 indices:
    /// [i1, j1, i2, j2, ..., iL, jL]
    /// # Errors
    ///
    /// Returns an error when the MPO evaluation fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn evaluate(&self, indices: &[LocalIndex]) -> Result<T>
    where
        T: EinsumScalar,
    {
        if indices.len() != 2 * self.len() {
            return Err(MPOError::InvalidOperation {
                message: format!(
                    "Expected {} indices (2*{}), got {}",
                    2 * self.len(),
                    self.len(),
                    indices.len()
                ),
            });
        }

        if self.is_empty() {
            return Err(MPOError::Empty);
        }

        // Start with the first tensor slice
        let first = &self.tensors[0];
        let i1 = indices[0];
        let j1 = indices[1];
        if i1 >= first.site_dim_1() || j1 >= first.site_dim_2() {
            return Err(MPOError::IndexOutOfBounds {
                site: 0,
                index: i1.max(j1),
                max: first.site_dim_1().max(first.site_dim_2()),
            });
        }

        // Vector of size right_dim
        let mut current: Vec<T> = first.slice_site(i1, j1);

        // Contract with remaining tensors
        for site in 1..self.len() {
            let tensor = &self.tensors[site];
            let i_k = indices[2 * site];
            let j_k = indices[2 * site + 1];
            if i_k >= tensor.site_dim_1() || j_k >= tensor.site_dim_2() {
                return Err(MPOError::IndexOutOfBounds {
                    site,
                    index: i_k.max(j_k),
                    max: tensor.site_dim_1().max(tensor.site_dim_2()),
                });
            }

            let slice = tensor.slice_site(i_k, j_k);
            current =
                row_vector_times_matrix(&current, &slice, tensor.left_dim(), tensor.right_dim())
                    .map_err(|err| MPOError::InvalidOperation {
                        message: format!("Failed to evaluate MPO at site {site}: {err}"),
                    })?;
        }

        // Should have a single element
        if current.len() != 1 {
            return Err(MPOError::InvalidOperation {
                message: format!(
                    "Final contraction resulted in {} elements, expected 1",
                    current.len()
                ),
            });
        }

        Ok(current[0])
    }

    /// Sum over all indices of the MPO
    ///
    /// Returns the scalar sum of every matrix-product-operator entry. Use this
    /// for small validation checks or algorithms that need the total operator
    /// weight without materializing the full operator.
    ///
    /// # Errors
    ///
    /// Returns an error when the tenferro-backed contraction fails (a backend
    /// failure) or an internally constructed MPO has inconsistent bond
    /// dimensions (a shape mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::mpo::MPO;
    ///
    /// let mpo = MPO::<f64>::constant(&[(2, 2)], 3.0);
    /// let sum = mpo.sum().unwrap();
    /// assert!((sum - 12.0).abs() < 1e-10);
    /// ```
    #[allow(clippy::needless_range_loop)]
    /// # Errors
    ///
    /// Returns an error when the tenferro-backed contraction fails (a backend
    ///     /// failure) or an internally constructed MPO is inconsistent (an
    ///     /// invalid-state failure).
    ///
    pub fn sum(&self) -> Result<T>
    where
        T: EinsumScalar,
    {
        if self.is_empty() {
            return Ok(T::zero());
        }

        // Start with sum over first tensor
        let first = &self.tensors[0];
        let mut current =
            tensor_to_col_major_vec(&einsum_tensors("lstr->r", &[first.as_inner()]).map_err(
                |err| MPOError::InvalidOperation {
                    message: format!("Failed to sum first MPO site: {err}"),
                },
            )?);

        // Contract with sums of remaining tensors
        for site in 1..self.len() {
            let tensor = &self.tensors[site];
            let site_sum = tensor_to_col_major_vec(
                &einsum_tensors("lstr->lr", &[tensor.as_inner()]).map_err(|err| {
                    MPOError::InvalidOperation {
                        message: format!("Failed to sum MPO site {site}: {err}"),
                    }
                })?,
            );
            current =
                row_vector_times_matrix(&current, &site_sum, tensor.left_dim(), tensor.right_dim())
                    .map_err(|err| MPOError::InvalidOperation {
                        message: format!("Failed to accumulate MPO sum at site {site}: {err}"),
                    })?;
        }

        if current.len() != 1 {
            return Err(MPOError::InvalidOperation {
                message: format!(
                    "Final MPO sum contraction resulted in {} elements, expected 1",
                    current.len()
                ),
            });
        }

        Ok(current[0])
    }

    /// Multiply the MPO by a scalar
    pub fn scale(&mut self, factor: T) {
        if !self.tensors.is_empty() {
            let last = self.tensors.len() - 1;
            let tensor = &mut self.tensors[last];
            for l in 0..tensor.left_dim() {
                for s1 in 0..tensor.site_dim_1() {
                    for s2 in 0..tensor.site_dim_2() {
                        for r in 0..tensor.right_dim() {
                            let val = *tensor.get4(l, s1, s2, r);
                            tensor.set4(l, s1, s2, r, val * factor);
                        }
                    }
                }
            }
        }
    }

    /// Create a scaled copy of the MPO
    pub fn scaled(&self, factor: T) -> Self {
        let mut result = self.clone();
        result.scale(factor);
        result
    }

    /// Convert the MPO to a full tensor
    ///
    /// Returns a flat vector containing all tensor elements in column-major order,
    /// along with the shape (alternating site_dim_1, site_dim_2 dimensions).
    ///
    /// Warning: This can be very large for high-dimensional operators!
    pub fn fulltensor(&self) -> (Vec<T>, Vec<usize>)
    where
        T: EinsumScalar,
    {
        if self.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let site_dims = self.site_dims();
        let shape: Vec<usize> = site_dims.iter().flat_map(|&(d1, d2)| [d1, d2]).collect();
        let total_size: usize = shape.iter().product();

        if total_size == 0 {
            return (Vec::new(), shape);
        }

        // Build full tensor by iterating over all indices
        let mut result = Vec::with_capacity(total_size);
        let mut indices = vec![0usize; shape.len()];

        loop {
            // Evaluate at current indices
            if let Ok(val) = self.evaluate(&indices) {
                result.push(val);
            } else {
                result.push(T::zero());
            }

            // Increment indices in column-major order, leftmost index fastest.
            let mut carry = true;
            for i in 0..shape.len() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= shape[i] {
                        indices[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }

            if carry {
                break; // All indices wrapped around
            }
        }

        (result, shape)
    }
}

#[cfg(test)]
mod tests;
