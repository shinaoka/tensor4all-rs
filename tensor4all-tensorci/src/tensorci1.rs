//! TensorCI1 - One-site Tensor Cross Interpolation algorithm

use crate::error::{Result, TCIError};
use crate::indexset::{IndexSet, MultiIndex};
use tensor4all_matrixci::util::{a_times_b_inv, zeros, Matrix, Scalar};
use tensor4all_matrixci::{AbstractMatrixCI, MatrixACA};
use tensor4all_tensortrain::{Tensor3, TensorTrain, TTScalar};

/// Sweep strategy for TCI optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SweepStrategy {
    /// Sweep forward only
    Forward,
    /// Sweep backward only
    Backward,
    /// Sweep back and forth
    #[default]
    BackAndForth,
}

/// Returns true if this iteration should be a forward sweep
fn forward_sweep(strategy: SweepStrategy, iter: usize) -> bool {
    match strategy {
        SweepStrategy::Forward => true,
        SweepStrategy::Backward => false,
        SweepStrategy::BackAndForth => iter % 2 == 1,
    }
}

/// Options for TCI1 algorithm
#[derive(Debug, Clone)]
pub struct TCI1Options {
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Sweep strategy
    pub sweep_strategy: SweepStrategy,
    /// Pivot tolerance (minimum error to add new pivot)
    pub pivot_tolerance: f64,
    /// Whether to normalize error by max sample value
    pub normalize_error: bool,
    /// Verbosity level
    pub verbosity: usize,
}

impl Default for TCI1Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 200,
            sweep_strategy: SweepStrategy::BackAndForth,
            pivot_tolerance: 1e-12,
            normalize_error: true,
            verbosity: 0,
        }
    }
}

/// TensorCI1 - One-site Tensor Cross Interpolation
///
/// Represents a tensor train constructed using the TCI1 algorithm.
#[derive(Debug, Clone)]
pub struct TensorCI1<T: Scalar + TTScalar> {
    /// Index sets I for each site
    i_set: Vec<IndexSet<MultiIndex>>,
    /// Index sets J for each site
    j_set: Vec<IndexSet<MultiIndex>>,
    /// Local dimensions
    local_dims: Vec<usize>,
    /// T tensors (3-leg tensors)
    t_tensors: Vec<Tensor3<T>>,
    /// P matrices (pivot matrices)
    p_matrices: Vec<Matrix<T>>,
    /// ACA decompositions at each bond
    aca: Vec<MatrixACA<T>>,
    /// Pi matrices at each bond
    pi: Vec<Matrix<T>>,
    /// Pi I sets
    pi_i_set: Vec<IndexSet<MultiIndex>>,
    /// Pi J sets
    pi_j_set: Vec<IndexSet<MultiIndex>>,
    /// Pivot errors at each bond
    pivot_errors: Vec<f64>,
    /// Maximum sample value found
    max_sample_value: f64,
}

impl<T: Scalar + TTScalar + Default> TensorCI1<T> {
    /// Create a new empty TensorCI1
    pub fn new(local_dims: Vec<usize>) -> Self {
        let n = local_dims.len();
        Self {
            i_set: (0..n).map(|_| IndexSet::new()).collect(),
            j_set: (0..n).map(|_| IndexSet::new()).collect(),
            local_dims: local_dims.clone(),
            t_tensors: local_dims.iter().map(|&d| Tensor3::zeros(0, d, 0)).collect(),
            p_matrices: (0..n).map(|_| zeros(0, 0)).collect(),
            aca: (0..n).map(|_| MatrixACA::new(0, 0)).collect(),
            pi: (0..n).map(|_| zeros(0, 0)).collect(),
            pi_i_set: (0..n).map(|_| IndexSet::new()).collect(),
            pi_j_set: (0..n).map(|_| IndexSet::new()).collect(),
            pivot_errors: vec![f64::INFINITY; n.saturating_sub(1)],
            max_sample_value: 0.0,
        }
    }

    /// Number of sites
    pub fn len(&self) -> usize {
        self.t_tensors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.t_tensors.is_empty()
    }

    /// Get local dimensions
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Get current rank (maximum bond dimension)
    pub fn rank(&self) -> usize {
        if self.t_tensors.len() <= 1 {
            return if self.i_set.is_empty() || self.i_set[0].is_empty() { 0 } else { 1 };
        }
        self.t_tensors
            .iter()
            .skip(1)
            .map(|t| t.left_dim())
            .max()
            .unwrap_or(0)
    }

    /// Get bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.t_tensors.len() <= 1 {
            return Vec::new();
        }
        self.t_tensors.iter().skip(1).map(|t| t.left_dim()).collect()
    }

    /// Get the maximum pivot error from the last sweep
    pub fn last_sweep_pivot_error(&self) -> f64 {
        self.pivot_errors.iter().cloned().fold(0.0, f64::max)
    }

    /// Get site tensor at position p (T * P^{-1})
    pub fn site_tensor(&self, p: usize) -> Tensor3<T> {
        if p >= self.len() {
            return Tensor3::zeros(1, 1, 1);
        }

        let t = &self.t_tensors[p];
        let shape = (t.left_dim(), t.site_dim(), t.right_dim());

        // If empty, return identity-like tensor
        if shape.0 == 0 || shape.2 == 0 {
            return t.clone();
        }

        // Compute T * P^{-1}
        let t_mat = tensor3_to_matrix(t);
        let p_mat = &self.p_matrices[p];

        if p_mat.nrows() == 0 || p_mat.ncols() == 0 {
            return self.t_tensors[p].clone();
        }

        let result = a_times_b_inv(&t_mat, p_mat);
        matrix_to_tensor3(&result, shape.0, shape.1, shape.2)
    }

    /// Get all site tensors
    pub fn site_tensors(&self) -> Vec<Tensor3<T>> {
        (0..self.len()).map(|p| self.site_tensor(p)).collect()
    }

    /// Convert to TensorTrain
    pub fn to_tensor_train(&self) -> Result<TensorTrain<T>> {
        let tensors = self.site_tensors();
        TensorTrain::new(tensors).map_err(TCIError::TensorTrainError)
    }

    /// Get maximum sample value
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Update maximum sample value from a slice of values
    fn update_max_sample(&mut self, values: &[T]) {
        for v in values {
            let abs_val = f64::sqrt(Scalar::abs_sq(*v));
            if abs_val > self.max_sample_value {
                self.max_sample_value = abs_val;
            }
        }
    }

    /// Update maximum sample value from a matrix
    fn update_max_sample_matrix(&mut self, mat: &Matrix<T>) {
        for i in 0..mat.nrows() {
            for j in 0..mat.ncols() {
                let abs_val = f64::sqrt(Scalar::abs_sq(mat[[i, j]]));
                if abs_val > self.max_sample_value {
                    self.max_sample_value = abs_val;
                }
            }
        }
    }

    /// Evaluate the TCI at a specific set of indices
    #[allow(clippy::needless_range_loop)]
    pub fn evaluate(&self, indices: &[usize]) -> Result<T> {
        if indices.len() != self.len() {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "Index length ({}) must match number of sites ({})",
                    indices.len(),
                    self.len()
                ),
            });
        }

        if self.is_empty() {
            return Err(TCIError::Empty);
        }

        // Check rank
        if self.rank() == 0 {
            return Err(TCIError::Empty);
        }

        // Evaluate by contracting site tensors
        // For each site p, compute T[p][:, indices[p], :] * P[p]^{-1}
        let mut result = {
            let t = &self.t_tensors[0];
            let idx = indices[0];
            if idx >= t.site_dim() {
                return Err(TCIError::IndexOutOfBounds {
                    message: format!("Index {} out of bounds at site 0 (max {})", idx, t.site_dim()),
                });
            }
            // Get slice at index
            let mut slice = vec![T::zero(); t.right_dim()];
            for r in 0..t.right_dim() {
                slice[r] = *t.get(0, idx, r);
            }

            // Apply P^{-1} if needed
            let p = &self.p_matrices[0];
            if p.nrows() > 0 && p.ncols() > 0 {
                let slice_mat = vec_to_row_matrix(&slice);
                let result_mat = a_times_b_inv(&slice_mat, p);
                row_matrix_to_vec(&result_mat)
            } else {
                slice
            }
        };

        // Contract with remaining sites
        for p in 1..self.len() {
            let t = &self.t_tensors[p];
            let idx = indices[p];
            if idx >= t.site_dim() {
                return Err(TCIError::IndexOutOfBounds {
                    message: format!("Index {} out of bounds at site {} (max {})", idx, p, t.site_dim()),
                });
            }

            let left_dim = t.left_dim();
            let right_dim = t.right_dim();

            // Contract: result (size left_dim) with T[:, idx, :] (left_dim x right_dim)
            let mut next = vec![T::zero(); right_dim];
            for r in 0..right_dim {
                let mut sum = T::zero();
                for l in 0..left_dim {
                    sum = sum + result[l] * *t.get(l, idx, r);
                }
                next[r] = sum;
            }

            // Apply P^{-1}
            let p_mat = &self.p_matrices[p];
            if p_mat.nrows() > 0 && p_mat.ncols() > 0 {
                let next_mat = vec_to_row_matrix(&next);
                let result_mat = a_times_b_inv(&next_mat, p_mat);
                result = row_matrix_to_vec(&result_mat);
            } else {
                result = next;
            }
        }

        if result.len() != 1 {
            return Err(TCIError::InvalidOperation {
                message: format!("Final result should have size 1, got {}", result.len()),
            });
        }

        Ok(result[0])
    }

    /// Get I set for a site
    pub fn i_set(&self, p: usize) -> &IndexSet<MultiIndex> {
        &self.i_set[p]
    }

    /// Get J set for a site
    pub fn j_set(&self, p: usize) -> &IndexSet<MultiIndex> {
        &self.j_set[p]
    }

    /// Build the Pi I set for site p
    /// PiIset[p] = { [i..., up] : i in Iset[p], up in 1..localdims[p] }
    fn get_pi_i_set(&self, p: usize) -> IndexSet<MultiIndex> {
        let mut result = Vec::new();
        for i_multi in self.i_set[p].iter() {
            for up in 0..self.local_dims[p] {
                let mut new_idx = i_multi.clone();
                new_idx.push(up);
                result.push(new_idx);
            }
        }
        IndexSet::from_vec(result)
    }

    /// Build the Pi J set for site p
    /// PiJset[p] = { [up+1, j...] : up+1 in 1..localdims[p], j in Jset[p] }
    fn get_pi_j_set(&self, p: usize) -> IndexSet<MultiIndex> {
        let mut result = Vec::new();
        for up1 in 0..self.local_dims[p] {
            for j_multi in self.j_set[p].iter() {
                let mut new_idx = vec![up1];
                new_idx.extend(j_multi.iter().cloned());
                result.push(new_idx);
            }
        }
        IndexSet::from_vec(result)
    }

    /// Build the Pi matrix at bond p
    /// Pi[p][i, j] = f([PiIset[p][i]..., PiJset[p+1][j]...])
    fn get_pi<F>(&mut self, p: usize, f: &F) -> Matrix<T>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let i_set = &self.pi_i_set[p];
        let j_set = &self.pi_j_set[p + 1];

        let mut pi = zeros(i_set.len(), j_set.len());
        for (i, i_multi) in i_set.iter().enumerate() {
            for (j, j_multi) in j_set.iter().enumerate() {
                let mut full_idx = i_multi.clone();
                full_idx.extend(j_multi.iter().cloned());
                pi[[i, j]] = f(&full_idx);
            }
        }

        self.update_max_sample_matrix(&pi);
        pi
    }

    /// Update Pi rows at site p (after I set changed at p+1)
    fn update_pi_rows<F>(&mut self, p: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        let new_i_set = self.get_pi_i_set(p);
        // Clone the old set to avoid borrow issues
        let old_i_set: Vec<MultiIndex> = self.pi_i_set[p].iter().cloned().collect();
        let old_i_set_ref = IndexSet::from_vec(old_i_set.clone());

        // Find new indices
        let new_indices: Vec<MultiIndex> = new_i_set
            .iter()
            .filter(|i| old_i_set_ref.pos(i).is_none())
            .cloned()
            .collect();

        // Create new Pi matrix
        let mut new_pi = zeros(new_i_set.len(), self.pi[p].ncols());

        // Copy old rows
        for (old_i, i_multi) in old_i_set.iter().enumerate() {
            if let Some(new_i) = new_i_set.pos(i_multi) {
                for j in 0..new_pi.ncols() {
                    new_pi[[new_i, j]] = self.pi[p][[old_i, j]];
                }
            }
        }

        // Compute new rows
        for i_multi in &new_indices {
            if let Some(new_i) = new_i_set.pos(i_multi) {
                for (j, j_multi) in self.pi_j_set[p + 1].iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    new_pi[[new_i, j]] = f(&full_idx);
                }
                // Update max sample
                let row: Vec<T> = (0..new_pi.ncols()).map(|j| new_pi[[new_i, j]]).collect();
                self.update_max_sample(&row);
            }
        }

        self.pi[p] = new_pi;
        self.pi_i_set[p] = new_i_set;

        // Update ACA rows
        let t_shape = (
            self.t_tensors[p].left_dim(),
            self.t_tensors[p].site_dim(),
            self.t_tensors[p].right_dim(),
        );
        let t_p = tensor3_to_matrix_cols(&self.t_tensors[p], t_shape.0 * t_shape.1, t_shape.2);
        let permutation: Vec<usize> = old_i_set
            .iter()
            .filter_map(|i| self.pi_i_set[p].pos(i))
            .collect();
        self.aca[p].set_rows(&t_p, &permutation);
    }

    /// Update Pi cols at site p (after J set changed at p)
    fn update_pi_cols<F>(&mut self, p: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        let new_j_set = self.get_pi_j_set(p + 1);
        // Clone the old set to avoid borrow issues
        let old_j_set: Vec<MultiIndex> = self.pi_j_set[p + 1].iter().cloned().collect();
        let old_j_set_ref = IndexSet::from_vec(old_j_set.clone());

        // Find new indices
        let new_indices: Vec<MultiIndex> = new_j_set
            .iter()
            .filter(|j| old_j_set_ref.pos(j).is_none())
            .cloned()
            .collect();

        // Create new Pi matrix
        let mut new_pi = zeros(self.pi[p].nrows(), new_j_set.len());

        // Copy old columns
        for (old_j, j_multi) in old_j_set.iter().enumerate() {
            if let Some(new_j) = new_j_set.pos(j_multi) {
                for i in 0..new_pi.nrows() {
                    new_pi[[i, new_j]] = self.pi[p][[i, old_j]];
                }
            }
        }

        // Compute new columns
        for j_multi in &new_indices {
            if let Some(new_j) = new_j_set.pos(j_multi) {
                for (i, i_multi) in self.pi_i_set[p].iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    new_pi[[i, new_j]] = f(&full_idx);
                }
                // Update max sample
                let col: Vec<T> = (0..new_pi.nrows()).map(|i| new_pi[[i, new_j]]).collect();
                self.update_max_sample(&col);
            }
        }

        self.pi[p] = new_pi;
        self.pi_j_set[p + 1] = new_j_set;

        // Update ACA cols
        let t_p1 = &self.t_tensors[p + 1];
        let t_shape = (t_p1.left_dim(), t_p1.site_dim(), t_p1.right_dim());
        let t_mat = tensor3_to_matrix_rows(t_p1, t_shape.0, t_shape.1 * t_shape.2);
        let permutation: Vec<usize> = old_j_set
            .iter()
            .filter_map(|j| self.pi_j_set[p + 1].pos(j))
            .collect();
        self.aca[p].set_cols(&t_mat, &permutation);
    }

    /// Add a pivot row at bond p
    fn add_pivot_row<F>(&mut self, p: usize, new_i: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        // Add to ACA
        let _ = self.aca[p].add_pivot_row(&self.pi[p], new_i);

        // Add to I set at p+1
        let new_i_multi = self.pi_i_set[p].get(new_i).unwrap().clone();
        self.i_set[p + 1].push(new_i_multi);

        // Update T[p+1] - get all pivot rows from Pi
        // Each row in the I set corresponds to a pivot row in Pi
        let i_set_len = self.i_set[p + 1].len();
        let local_dim = self.local_dims[p + 1];
        let j_set_len = self.j_set[p + 1].len();

        // Build pivot rows matrix: shape (i_set_len, local_dim * j_set_len)
        let mut pivot_rows = zeros(i_set_len, local_dim * j_set_len);
        for (row_idx, i_multi) in self.i_set[p + 1].iter().enumerate() {
            if let Some(pi_row) = self.pi_i_set[p].pos(i_multi) {
                for j in 0..self.pi[p].ncols() {
                    pivot_rows[[row_idx, j]] = self.pi[p][[pi_row, j]];
                }
            }
        }

        self.t_tensors[p + 1] = matrix_to_tensor3(&pivot_rows, i_set_len, local_dim, j_set_len);

        // Update P matrix using pivot values
        self.update_p_matrix(p);

        // Update adjacent Pi matrix if exists
        if p < self.len() - 2 {
            self.update_pi_rows(p + 1, f);
        }
    }

    /// Add a pivot col at bond p
    fn add_pivot_col<F>(&mut self, p: usize, new_j: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        // Add to ACA
        let _ = self.aca[p].add_pivot_col(&self.pi[p], new_j);

        // Add to J set at p
        let new_j_multi = self.pi_j_set[p + 1].get(new_j).unwrap().clone();
        self.j_set[p].push(new_j_multi);

        // Update T[p] - get all pivot columns from Pi
        // Each column in the J set corresponds to a pivot column in Pi
        let i_set_len = self.i_set[p].len();
        let local_dim = self.local_dims[p];
        let j_set_len = self.j_set[p].len();

        // Build pivot cols matrix: shape (i_set_len * local_dim, j_set_len)
        let mut pivot_cols = zeros(i_set_len * local_dim, j_set_len);
        for (col_idx, j_multi) in self.j_set[p].iter().enumerate() {
            if let Some(pi_col) = self.pi_j_set[p + 1].pos(j_multi) {
                for i in 0..self.pi[p].nrows() {
                    pivot_cols[[i, col_idx]] = self.pi[p][[i, pi_col]];
                }
            }
        }

        self.t_tensors[p] = matrix_to_tensor3(&pivot_cols, i_set_len, local_dim, j_set_len);

        // Update P matrix using pivot values
        self.update_p_matrix(p);

        // Update adjacent Pi matrix if exists
        if p > 0 {
            self.update_pi_cols(p - 1, f);
        }
    }

    /// Update P matrix at bond p from current I and J sets
    fn update_p_matrix(&mut self, p: usize) {
        let i_set_len = self.i_set[p + 1].len();
        let j_set_len = self.j_set[p].len();

        let mut p_mat = zeros(i_set_len, j_set_len);
        for (i, i_multi) in self.i_set[p + 1].iter().enumerate() {
            for (j, j_multi) in self.j_set[p].iter().enumerate() {
                if let (Some(pi_i), Some(pi_j)) = (
                    self.pi_i_set[p].pos(i_multi),
                    self.pi_j_set[p + 1].pos(j_multi),
                ) {
                    p_mat[[i, j]] = self.pi[p][[pi_i, pi_j]];
                }
            }
        }
        self.p_matrices[p] = p_mat;
    }

    /// Add a pivot at bond p
    fn add_pivot<F>(&mut self, p: usize, f: &F, tolerance: f64)
    where
        F: Fn(&MultiIndex) -> T,
    {
        if p >= self.len() - 1 {
            return;
        }

        // Check if we've reached full rank
        let pi_rows = self.pi[p].nrows();
        let pi_cols = self.pi[p].ncols();
        if self.aca[p].rank() >= pi_rows.min(pi_cols) {
            self.pivot_errors[p] = 0.0;
            return;
        }

        // Find new pivot using ACA
        let new_pivot = self.aca[p].find_new_pivot(&self.pi[p]);

        match new_pivot {
            Ok(((new_i, new_j), error)) => {
                let error_val = f64::sqrt(Scalar::abs_sq(error));
                self.pivot_errors[p] = error_val;

                if error_val < tolerance {
                    return;
                }

                // Add pivot column first, then row
                self.add_pivot_col(p, new_j, f);
                self.add_pivot_row(p, new_i, f);
            }
            Err(_) => {
                self.pivot_errors[p] = 0.0;
            }
        }
    }

    /// Initialize from function with first pivot
    fn initialize_from_pivot<F>(&mut self, f: &F, first_pivot: &MultiIndex) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let first_value = f(first_pivot);
        if Scalar::abs_sq(first_value) < 1e-30 {
            return Err(TCIError::InvalidPivot {
                message: "First pivot must have non-zero function value".to_string(),
            });
        }

        self.max_sample_value = f64::sqrt(Scalar::abs_sq(first_value));
        let n = self.len();

        // Initialize I and J sets from first pivot
        for p in 0..n {
            let i_indices: MultiIndex = first_pivot[0..p].to_vec();
            let j_indices: MultiIndex = first_pivot[p + 1..].to_vec();
            self.i_set[p] = IndexSet::from_vec(vec![i_indices]);
            self.j_set[p] = IndexSet::from_vec(vec![j_indices]);
        }

        // Build Pi I and J sets
        for p in 0..n {
            self.pi_i_set[p] = self.get_pi_i_set(p);
            self.pi_j_set[p] = self.get_pi_j_set(p);
        }

        // Build Pi matrices
        for p in 0..n - 1 {
            self.pi[p] = self.get_pi(p, f);
        }

        // Initialize ACA and T tensors for each bond
        for p in 0..n - 1 {
            // Find local pivot position in Pi
            let local_pivot = (
                self.pi_i_set[p].pos(&self.i_set[p + 1].get(0).cloned().unwrap_or_default()).unwrap_or(0),
                self.pi_j_set[p + 1].pos(&self.j_set[p].get(0).cloned().unwrap_or_default()).unwrap_or(0),
            );

            // Initialize ACA from Pi with the pivot
            self.aca[p] = MatrixACA::from_matrix_with_pivot(&self.pi[p], local_pivot);

            // Update T tensors
            if p == 0 {
                // T[0] from pivot column of Pi[0]
                let pivot_col: Vec<T> = (0..self.pi[p].nrows())
                    .map(|i| self.pi[p][[i, local_pivot.1]])
                    .collect();
                let col_mat = vec_to_col_matrix(&pivot_col);
                self.t_tensors[0] = matrix_to_tensor3(&col_mat, 1, self.local_dims[0], 1);
            }

            // T[p+1] from pivot row of Pi[p]
            let pivot_row: Vec<T> = (0..self.pi[p].ncols())
                .map(|j| self.pi[p][[local_pivot.0, j]])
                .collect();
            let row_mat = vec_to_row_matrix(&pivot_row);
            self.t_tensors[p + 1] = matrix_to_tensor3(&row_mat, 1, self.local_dims[p + 1], 1);

            // P[p] = Pi[p][local_pivot]
            let mut p_mat = zeros(1, 1);
            p_mat[[0, 0]] = self.pi[p][[local_pivot.0, local_pivot.1]];
            self.p_matrices[p] = p_mat;
        }

        // P[n-1] = identity (1x1 of ones)
        let mut p_last = zeros(1, 1);
        p_last[[0, 0]] = T::one();
        self.p_matrices[n - 1] = p_last;

        Ok(())
    }
}

// Helper functions

/// Convert a vector to a row matrix
fn vec_to_row_matrix<T: Scalar>(v: &[T]) -> Matrix<T> {
    let mut mat = zeros(1, v.len());
    for (j, &val) in v.iter().enumerate() {
        mat[[0, j]] = val;
    }
    mat
}

/// Convert a vector to a column matrix
fn vec_to_col_matrix<T: Scalar>(v: &[T]) -> Matrix<T> {
    let mut mat = zeros(v.len(), 1);
    for (i, &val) in v.iter().enumerate() {
        mat[[i, 0]] = val;
    }
    mat
}

/// Convert a row matrix to a vector
fn row_matrix_to_vec<T: Scalar>(mat: &Matrix<T>) -> Vec<T> {
    (0..mat.ncols()).map(|j| mat[[0, j]]).collect()
}

/// Convert Tensor3 to Matrix (reshape for columns: (left*site, right))
fn tensor3_to_matrix<T: Scalar + Default>(tensor: &Tensor3<T>) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();
    let rows = left_dim * site_dim;
    let cols = right_dim;

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                mat[[l * site_dim + s, r]] = *tensor.get(l, s, r);
            }
        }
    }
    mat
}

/// Convert Tensor3 to Matrix for columns (left*site, right)
fn tensor3_to_matrix_cols<T: Scalar + Default>(tensor: &Tensor3<T>, rows: usize, cols: usize) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                if l * site_dim + s < rows && r < cols {
                    mat[[l * site_dim + s, r]] = *tensor.get(l, s, r);
                }
            }
        }
    }
    mat
}

/// Convert Tensor3 to Matrix for rows (left, site*right)
fn tensor3_to_matrix_rows<T: Scalar + Default>(tensor: &Tensor3<T>, rows: usize, cols: usize) -> Matrix<T> {
    let left_dim = tensor.left_dim();
    let site_dim = tensor.site_dim();
    let right_dim = tensor.right_dim();

    let mut mat = zeros(rows, cols);
    for l in 0..left_dim {
        for s in 0..site_dim {
            for r in 0..right_dim {
                if l < rows && s * right_dim + r < cols {
                    mat[[l, s * right_dim + r]] = *tensor.get(l, s, r);
                }
            }
        }
    }
    mat
}

/// Convert Matrix to Tensor3
fn matrix_to_tensor3<T: Scalar + Default>(mat: &Matrix<T>, left_dim: usize, site_dim: usize, right_dim: usize) -> Tensor3<T> {
    let mut tensor = Tensor3::zeros(left_dim, site_dim, right_dim);

    // Determine the layout based on matrix dimensions
    if mat.nrows() == left_dim * site_dim && mat.ncols() == right_dim {
        // Column layout: (left*site, right)
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    tensor.set(l, s, r, mat[[l * site_dim + s, r]]);
                }
            }
        }
    } else if mat.nrows() == left_dim && mat.ncols() == site_dim * right_dim {
        // Row layout: (left, site*right)
        for l in 0..left_dim {
            for s in 0..site_dim {
                for r in 0..right_dim {
                    tensor.set(l, s, r, mat[[l, s * right_dim + r]]);
                }
            }
        }
    } else if mat.nrows() == 1 && mat.ncols() == site_dim {
        // Single row with site values
        for s in 0..site_dim {
            tensor.set(0, s, 0, mat[[0, s]]);
        }
    } else if mat.nrows() == site_dim && mat.ncols() == 1 {
        // Single column with site values
        for s in 0..site_dim {
            tensor.set(0, s, 0, mat[[s, 0]]);
        }
    }

    tensor
}

/// Cross interpolate a function using TCI1 algorithm
///
/// # Arguments
/// * `f` - Function to interpolate, takes a multi-index and returns a value
/// * `local_dims` - Local dimensions for each site
/// * `first_pivot` - Initial pivot point
/// * `options` - Algorithm options
///
/// # Returns
/// * `TensorCI1` - The constructed tensor cross interpolation
/// * `Vec<usize>` - Ranks at each iteration
/// * `Vec<f64>` - Errors at each iteration
pub fn crossinterpolate1<T, F>(
    f: F,
    local_dims: Vec<usize>,
    first_pivot: MultiIndex,
    options: TCI1Options,
) -> Result<(TensorCI1<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default,
    F: Fn(&MultiIndex) -> T,
{
    if local_dims.len() != first_pivot.len() {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "local_dims length ({}) must match first_pivot length ({})",
                local_dims.len(),
                first_pivot.len()
            ),
        });
    }

    let mut tci = TensorCI1::new(local_dims.clone());
    tci.initialize_from_pivot(&f, &first_pivot)?;

    let n = tci.len();
    let mut errors = Vec::new();
    let mut ranks = Vec::new();

    // Main iteration loop
    for iter in 1..=options.max_iter {
        // Sweep
        if forward_sweep(options.sweep_strategy, iter) {
            for bond_index in 0..n - 1 {
                tci.add_pivot(bond_index, &f, options.pivot_tolerance);
            }
        } else {
            for bond_index in (0..n - 1).rev() {
                tci.add_pivot(bond_index, &f, options.pivot_tolerance);
            }
        }

        // Record error and rank
        let error = tci.last_sweep_pivot_error();
        let error_normalized = if options.normalize_error && tci.max_sample_value > 0.0 {
            error / tci.max_sample_value
        } else {
            error
        };

        errors.push(error_normalized);
        ranks.push(tci.rank());

        if options.verbosity > 0 && iter % 10 == 0 {
            println!(
                "iteration = {}, rank = {}, error = {:.2e}",
                iter,
                tci.rank(),
                error_normalized
            );
        }

        // Check convergence
        if error_normalized < options.tolerance {
            break;
        }
    }

    Ok((tci, ranks, errors))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorci1_new() {
        let tci = TensorCI1::<f64>::new(vec![2, 3, 2]);
        assert_eq!(tci.len(), 3);
        assert_eq!(tci.local_dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_crossinterpolate1_constant() {
        let f = |_: &MultiIndex| 1.0f64;
        let local_dims = vec![2, 2];
        let first_pivot = vec![0, 0];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot, options).unwrap();
        assert_eq!(tci.len(), 2);
        assert!(tci.rank() >= 1);
    }

    #[test]
    fn test_crossinterpolate1_simple() {
        // f(i, j) = i + j + 1
        let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
        let local_dims = vec![3, 3];
        let first_pivot = vec![1, 1];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot, options).unwrap();
        assert_eq!(tci.len(), 2);
    }

    #[test]
    fn test_crossinterpolate1_evaluate_at_pivot() {
        // f(i, j) = (i + 1) * (j + 1)
        let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
        let local_dims = vec![3, 3];
        let first_pivot = vec![1, 1];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims.clone(), first_pivot.clone(), options).unwrap();

        // The TCI should exactly reproduce the function at the pivot
        let val = tci.evaluate(&first_pivot).unwrap();
        let expected = f(&first_pivot);
        assert!(
            (val - expected).abs() < 1e-10,
            "TCI evaluate at pivot: got {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_crossinterpolate1_evaluate_on_cross() {
        // f(i, j) = (i + 1) * (j + 1)
        let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
        let local_dims = vec![3, 3];
        let first_pivot = vec![1, 1];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims.clone(), first_pivot.clone(), options).unwrap();

        // Test evaluation at points on the cross through the pivot
        // Points (i, 1) for all i
        for i in 0..3 {
            let idx = vec![i, 1];
            let val = tci.evaluate(&idx).unwrap();
            let expected = f(&idx);
            assert!(
                (val - expected).abs() < 1e-10,
                "TCI evaluate at {:?}: got {}, expected {}",
                idx,
                val,
                expected
            );
        }

        // Points (1, j) for all j
        for j in 0..3 {
            let idx = vec![1, j];
            let val = tci.evaluate(&idx).unwrap();
            let expected = f(&idx);
            assert!(
                (val - expected).abs() < 1e-10,
                "TCI evaluate at {:?}: got {}, expected {}",
                idx,
                val,
                expected
            );
        }
    }

    #[test]
    fn test_crossinterpolate1_to_tensor_train() {
        let f = |idx: &MultiIndex| ((idx[0] + 1) * (idx[1] + 1)) as f64;
        let local_dims = vec![3, 3];
        let first_pivot = vec![1, 1];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot.clone(), options).unwrap();

        // Convert to TensorTrain
        let tt = tci.to_tensor_train().unwrap();
        assert_eq!(tt.len(), 2);

        // Evaluate at the pivot using the TensorTrain
        use tensor4all_tensortrain::AbstractTensorTrain;
        let val = tt.evaluate(&first_pivot).unwrap();
        let expected = f(&first_pivot);
        assert!(
            (val - expected).abs() < 1e-10,
            "TensorTrain evaluate at pivot: got {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_crossinterpolate1_3d() {
        // 3D function: f(i, j, k) = i + j + k + 1
        let f = |idx: &MultiIndex| (idx[0] + idx[1] + idx[2] + 1) as f64;
        let local_dims = vec![2, 2, 2];
        let first_pivot = vec![0, 0, 0];
        let options = TCI1Options::default();

        let (tci, _ranks, _errors) = crossinterpolate1(f, local_dims, first_pivot.clone(), options).unwrap();

        assert_eq!(tci.len(), 3);

        // Test evaluation at the pivot
        let val = tci.evaluate(&first_pivot).unwrap();
        let expected = f(&first_pivot);
        assert!(
            (val - expected).abs() < 1e-10,
            "3D TCI evaluate at pivot: got {}, expected {}",
            val,
            expected
        );
    }

    #[test]
    fn test_crossinterpolate1_rank2_function() {
        // A rank-2 function: f(i, j) = i + j (not separable)
        let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
        let local_dims = vec![4, 4];
        let first_pivot = vec![1, 1];
        let mut options = TCI1Options::default();
        options.tolerance = 1e-12;

        let (tci, _ranks, errors) = crossinterpolate1(f, local_dims.clone(), first_pivot, options).unwrap();

        // Should converge to rank 2
        assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());

        // Check error is small
        let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
        assert!(final_error < 1e-8, "Expected small error, got {}", final_error);

        // Test all values
        for i in 0..4 {
            for j in 0..4 {
                let idx = vec![i, j];
                let val = tci.evaluate(&idx).unwrap();
                let expected = f(&idx);
                assert!(
                    (val - expected).abs() < 1e-8,
                    "TCI evaluate at {:?}: got {}, expected {}",
                    idx,
                    val,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_crossinterpolate1_converges() {
        // A smooth function that should converge well
        let f = |idx: &MultiIndex| {
            let x = idx[0] as f64 / 4.0;
            let y = idx[1] as f64 / 4.0;
            (x * y + 0.1).sin()
        };
        let local_dims = vec![5, 5];
        let first_pivot = vec![2, 2];
        let mut options = TCI1Options::default();
        options.tolerance = 1e-6;

        let (tci, _ranks, errors) = crossinterpolate1(f, local_dims.clone(), first_pivot, options).unwrap();

        // Should achieve reasonable rank
        assert!(tci.rank() <= 5, "Expected rank <= 5, got {}", tci.rank());

        // Check errors decrease
        if errors.len() > 1 {
            let first_error = errors[0];
            let last_error = errors.last().copied().unwrap();
            assert!(
                last_error <= first_error || last_error < 1e-6,
                "Errors should decrease or converge: first={}, last={}",
                first_error,
                last_error
            );
        }
    }
}
