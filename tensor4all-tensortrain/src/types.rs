//! Core types for tensor train operations

/// Local index type (index within a single tensor site)
pub type LocalIndex = usize;

/// Multi-index type (indices across all sites)
pub type MultiIndex = Vec<LocalIndex>;

/// A 3D tensor represented as a flat Vec with shape information
/// Shape is (left_dim, site_dim, right_dim)
#[derive(Debug, Clone)]
pub struct Tensor3<T> {
    data: Vec<T>,
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
}

impl<T: Clone + Default> Tensor3<T> {
    /// Create a new tensor filled with default values
    pub fn zeros(left_dim: usize, site_dim: usize, right_dim: usize) -> Self {
        Self {
            data: vec![T::default(); left_dim * site_dim * right_dim],
            left_dim,
            site_dim,
            right_dim,
        }
    }

    /// Create from flat data with shape
    pub fn from_data(data: Vec<T>, left_dim: usize, site_dim: usize, right_dim: usize) -> Self {
        assert_eq!(data.len(), left_dim * site_dim * right_dim);
        Self {
            data,
            left_dim,
            site_dim,
            right_dim,
        }
    }

    /// Get the left (bond) dimension
    pub fn left_dim(&self) -> usize {
        self.left_dim
    }

    /// Get the site (physical) dimension
    pub fn site_dim(&self) -> usize {
        self.site_dim
    }

    /// Get the right (bond) dimension
    pub fn right_dim(&self) -> usize {
        self.right_dim
    }

    /// Get element at (left, site, right)
    pub fn get(&self, l: usize, s: usize, r: usize) -> &T {
        let idx = (l * self.site_dim + s) * self.right_dim + r;
        &self.data[idx]
    }

    /// Get mutable element at (left, site, right)
    pub fn get_mut(&mut self, l: usize, s: usize, r: usize) -> &mut T {
        let idx = (l * self.site_dim + s) * self.right_dim + r;
        &mut self.data[idx]
    }

    /// Set element at (left, site, right)
    pub fn set(&mut self, l: usize, s: usize, r: usize, value: T) {
        let idx = (l * self.site_dim + s) * self.right_dim + r;
        self.data[idx] = value;
    }

    /// Get a slice for fixed site index: returns (left_dim, right_dim) matrix
    pub fn slice_site(&self, s: usize) -> Vec<T> {
        let mut result = Vec::with_capacity(self.left_dim * self.right_dim);
        for l in 0..self.left_dim {
            for r in 0..self.right_dim {
                result.push(self.get(l, s, r).clone());
            }
        }
        result
    }

    /// Reshape this tensor to a matrix (left_dim * site_dim, right_dim)
    pub fn as_left_matrix(&self) -> (Vec<T>, usize, usize) {
        let rows = self.left_dim * self.site_dim;
        let cols = self.right_dim;
        (self.data.clone(), rows, cols)
    }

    /// Reshape this tensor to a matrix (left_dim, site_dim * right_dim)
    pub fn as_right_matrix(&self) -> (Vec<T>, usize, usize) {
        let rows = self.left_dim;
        let cols = self.site_dim * self.right_dim;
        // Data needs to be transposed for row-major
        let mut result = Vec::with_capacity(rows * cols);
        for l in 0..self.left_dim {
            for s in 0..self.site_dim {
                for r in 0..self.right_dim {
                    result.push(self.get(l, s, r).clone());
                }
            }
        }
        (result, rows, cols)
    }
}

impl<T: Clone + Default + num_traits::Zero> Default for Tensor3<T> {
    fn default() -> Self {
        Self::zeros(1, 1, 1)
    }
}
