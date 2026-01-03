//! TensorCI2 - Two-site Tensor Cross Interpolation algorithm
//!
//! This implements the TCI2 algorithm which uses two-site updates for
//! more efficient convergence. Unlike TCI1, it supports batch evaluation
//! of function values through an explicit batch function parameter.

use crate::error::{Result, TCIError};
use crate::indexset::MultiIndex;
use tensor4all_matrixci::util::{zeros, Scalar};
use tensor4all_matrixci::{AbstractMatrixCI, MatrixLUCI, RrLUOptions};
use tensor4all_tensortrain::{Tensor3, TensorTrain, TTScalar};

/// Options for TCI2 algorithm
#[derive(Debug, Clone)]
pub struct TCI2Options {
    /// Tolerance for convergence (relative)
    pub tolerance: f64,
    /// Maximum number of iterations (half-sweeps)
    pub max_iter: usize,
    /// Maximum bond dimension
    pub max_bond_dim: usize,
    /// Pivot search strategy
    pub pivot_search: PivotSearchStrategy,
    /// Whether to normalize error by max sample value
    pub normalize_error: bool,
    /// Verbosity level
    pub verbosity: usize,
    /// Number of global pivots to search per iteration
    pub max_nglobal_pivot: usize,
    /// Number of random searches for global pivots
    pub nsearch: usize,
}

impl Default for TCI2Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: 50,
            pivot_search: PivotSearchStrategy::Full,
            normalize_error: true,
            verbosity: 0,
            max_nglobal_pivot: 5,
            nsearch: 100,
        }
    }
}

/// Pivot search strategy for TCI2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PivotSearchStrategy {
    /// Full search: evaluate entire Pi matrix
    #[default]
    Full,
    /// Rook search: use partial pivoting (more efficient for large matrices)
    Rook,
}

/// TensorCI2 - Two-site Tensor Cross Interpolation
///
/// This structure represents a tensor train constructed using the TCI2 algorithm.
/// TCI2 uses two-site updates which can be more efficient than TCI1 for some functions.
#[derive(Debug, Clone)]
pub struct TensorCI2<T: Scalar + TTScalar> {
    /// Index sets I for each site
    i_set: Vec<Vec<MultiIndex>>,
    /// Index sets J for each site
    j_set: Vec<Vec<MultiIndex>>,
    /// Local dimensions
    local_dims: Vec<usize>,
    /// Site tensors (3-leg tensors)
    site_tensors: Vec<Tensor3<T>>,
    /// Pivot errors during back-truncation
    pivot_errors: Vec<f64>,
    /// Bond errors from 2-site sweep
    bond_errors: Vec<f64>,
    /// Maximum sample value found
    max_sample_value: f64,
}

impl<T: Scalar + TTScalar + Default> TensorCI2<T> {
    /// Create a new empty TensorCI2
    pub fn new(local_dims: Vec<usize>) -> Result<Self> {
        if local_dims.len() < 2 {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims should have at least 2 elements".to_string(),
            });
        }

        let n = local_dims.len();
        Ok(Self {
            i_set: (0..n).map(|_| Vec::new()).collect(),
            j_set: (0..n).map(|_| Vec::new()).collect(),
            local_dims: local_dims.clone(),
            site_tensors: local_dims.iter().map(|&d| Tensor3::zeros(0, d, 0)).collect(),
            pivot_errors: Vec::new(),
            bond_errors: vec![0.0; n.saturating_sub(1)],
            max_sample_value: 0.0,
        })
    }

    /// Number of sites
    pub fn len(&self) -> usize {
        self.local_dims.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.local_dims.is_empty()
    }

    /// Get local dimensions
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Get current rank (maximum bond dimension)
    pub fn rank(&self) -> usize {
        if self.len() <= 1 {
            return if self.i_set.is_empty() || self.i_set[0].is_empty() { 0 } else { 1 };
        }
        self.i_set
            .iter()
            .skip(1)
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
    }

    /// Get bond dimensions
    pub fn link_dims(&self) -> Vec<usize> {
        if self.len() <= 1 {
            return Vec::new();
        }
        self.i_set.iter().skip(1).map(|s| s.len()).collect()
    }

    /// Get maximum sample value
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Get maximum bond error
    pub fn max_bond_error(&self) -> f64 {
        self.bond_errors.iter().cloned().fold(0.0, f64::max)
    }

    /// Get pivot errors from back-truncation
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }

    /// Check if site tensors are available
    pub fn is_site_tensors_available(&self) -> bool {
        self.site_tensors.iter().all(|t| {
            t.left_dim() > 0 || t.right_dim() > 0
        })
    }

    /// Get site tensor at position p
    pub fn site_tensor(&self, p: usize) -> &Tensor3<T> {
        &self.site_tensors[p]
    }

    /// Convert to TensorTrain
    pub fn to_tensor_train(&self) -> Result<TensorTrain<T>> {
        let tensors = self.site_tensors.clone();
        TensorTrain::new(tensors).map_err(TCIError::TensorTrainError)
    }

    /// Add global pivots to the TCI
    pub fn add_global_pivots(&mut self, pivots: &[MultiIndex]) -> Result<()> {
        for pivot in pivots {
            if pivot.len() != self.len() {
                return Err(TCIError::DimensionMismatch {
                    message: format!(
                        "Pivot length ({}) must match number of sites ({})",
                        pivot.len(),
                        self.len()
                    ),
                });
            }

            // Add to I and J sets
            for p in 0..self.len() {
                let i_indices: MultiIndex = pivot[0..p].to_vec();
                let j_indices: MultiIndex = pivot[p + 1..].to_vec();

                if !self.i_set[p].contains(&i_indices) {
                    self.i_set[p].push(i_indices);
                }
                if !self.j_set[p].contains(&j_indices) {
                    self.j_set[p].push(j_indices);
                }
            }
        }

        // Invalidate site tensors after adding pivots
        self.invalidate_site_tensors();

        Ok(())
    }

    /// Invalidate all site tensors
    fn invalidate_site_tensors(&mut self) {
        for p in 0..self.len() {
            self.site_tensors[p] = Tensor3::zeros(0, self.local_dims[p], 0);
        }
    }

    /// Expand indices by Kronecker product with local dimension
    fn kronecker_i(&self, p: usize) -> Vec<MultiIndex> {
        let mut result = Vec::new();
        for i_multi in &self.i_set[p] {
            for local_idx in 0..self.local_dims[p] {
                let mut new_idx = i_multi.clone();
                new_idx.push(local_idx);
                result.push(new_idx);
            }
        }
        result
    }

    fn kronecker_j(&self, p: usize) -> Vec<MultiIndex> {
        let mut result = Vec::new();
        for local_idx in 0..self.local_dims[p] {
            for j_multi in &self.j_set[p] {
                let mut new_idx = vec![local_idx];
                new_idx.extend(j_multi.iter().cloned());
                result.push(new_idx);
            }
        }
        result
    }
}

/// Cross interpolate a function using TCI2 algorithm
///
/// # Arguments
/// * `f` - Function to interpolate, takes a multi-index and returns a value
/// * `batched_f` - Optional batch evaluation function for efficiency
/// * `local_dims` - Local dimensions for each site
/// * `initial_pivots` - Initial pivot points
/// * `options` - Algorithm options
///
/// # Returns
/// * `TensorCI2` - The constructed tensor cross interpolation
/// * `Vec<usize>` - Ranks at each iteration
/// * `Vec<f64>` - Errors at each iteration
pub fn crossinterpolate2<T, F, B>(
    f: F,
    batched_f: Option<B>,
    local_dims: Vec<usize>,
    initial_pivots: Vec<MultiIndex>,
    options: TCI2Options,
) -> Result<(TensorCI2<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    if local_dims.len() < 2 {
        return Err(TCIError::DimensionMismatch {
            message: "local_dims should have at least 2 elements".to_string(),
        });
    }

    let pivots = if initial_pivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        initial_pivots
    };

    let mut tci = TensorCI2::new(local_dims)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value
    for pivot in &pivots {
        let value = f(pivot);
        let abs_val = f64::sqrt(Scalar::abs_sq(value));
        if abs_val > tci.max_sample_value {
            tci.max_sample_value = abs_val;
        }
    }

    if tci.max_sample_value < 1e-30 {
        return Err(TCIError::InvalidPivot {
            message: "Initial pivots have zero function values".to_string(),
        });
    }

    let n = tci.len();
    let mut errors = Vec::new();
    let mut ranks = Vec::new();

    // Main optimization loop
    for iter in 0..options.max_iter {
        let is_forward = iter % 2 == 0;

        // Sweep through bonds
        if is_forward {
            for b in 0..n - 1 {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    &batched_f,
                    true, // left orthogonal in forward sweep
                    &options,
                )?;
            }
        } else {
            for b in (0..n - 1).rev() {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    &batched_f,
                    false, // right orthogonal in backward sweep
                    &options,
                )?;
            }
        }

        // Record error and rank
        let error = tci.max_bond_error();
        let error_normalized = if options.normalize_error && tci.max_sample_value > 0.0 {
            error / tci.max_sample_value
        } else {
            error
        };

        errors.push(error_normalized);
        ranks.push(tci.rank());

        if options.verbosity > 0 {
            println!(
                "iteration = {}, rank = {}, error = {:.2e}",
                iter + 1,
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

/// Update pivots at bond b using LU-based cross interpolation
fn update_pivots<T, F, B>(
    tci: &mut TensorCI2<T>,
    b: usize,
    f: &F,
    batched_f: &Option<B>,
    left_orthogonal: bool,
    options: &TCI2Options,
) -> Result<()>
where
    T: Scalar + TTScalar + Default,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    // Invalidate site tensors
    tci.invalidate_site_tensors();

    // Build combined index sets
    let i_combined = tci.kronecker_i(b);
    let j_combined = tci.kronecker_j(b + 1);

    if i_combined.is_empty() || j_combined.is_empty() {
        return Ok(());
    }

    // Build Pi matrix
    let mut pi = zeros(i_combined.len(), j_combined.len());

    // Use batch evaluation if available, otherwise use single evaluation
    if let Some(ref batch_fn) = batched_f {
        // Build all index pairs
        let mut all_indices: Vec<MultiIndex> = Vec::with_capacity(i_combined.len() * j_combined.len());
        for i_multi in &i_combined {
            for j_multi in &j_combined {
                let mut full_idx = i_multi.clone();
                full_idx.extend(j_multi.iter().cloned());
                all_indices.push(full_idx);
            }
        }

        // Batch evaluate
        let values = batch_fn(&all_indices);

        // Fill Pi matrix
        let mut idx = 0;
        for i in 0..i_combined.len() {
            for j in 0..j_combined.len() {
                pi[[i, j]] = values[idx];
                let abs_val = f64::sqrt(Scalar::abs_sq(values[idx]));
                if abs_val > tci.max_sample_value {
                    tci.max_sample_value = abs_val;
                }
                idx += 1;
            }
        }
    } else {
        // Single evaluation
        for (i, i_multi) in i_combined.iter().enumerate() {
            for (j, j_multi) in j_combined.iter().enumerate() {
                let mut full_idx = i_multi.clone();
                full_idx.extend(j_multi.iter().cloned());
                let value = f(&full_idx);
                pi[[i, j]] = value;

                let abs_val = f64::sqrt(Scalar::abs_sq(value));
                if abs_val > tci.max_sample_value {
                    tci.max_sample_value = abs_val;
                }
            }
        }
    }

    // Apply LU-based cross interpolation
    let lu_options = RrLUOptions {
        max_rank: options.max_bond_dim,
        rel_tol: options.tolerance,
        abs_tol: 0.0,
        left_orthogonal,
    };

    let luci = MatrixLUCI::from_matrix(&pi, Some(lu_options));

    // Update I and J sets
    let row_indices = luci.row_indices();
    let col_indices = luci.col_indices();

    tci.i_set[b + 1] = row_indices.iter().map(|&i| i_combined[i].clone()).collect();
    tci.j_set[b] = col_indices.iter().map(|&j| j_combined[j].clone()).collect();

    // Update site tensors
    let left = luci.left();
    let right = luci.right();

    // Convert left matrix to tensor at site b
    let left_dim = if b == 0 { 1 } else { tci.i_set[b].len() };
    let site_dim_b = tci.local_dims[b];
    let new_bond_dim = luci.rank().max(1);

    let mut tensor_b = Tensor3::zeros(left_dim, site_dim_b, new_bond_dim);
    for l in 0..left_dim {
        for s in 0..site_dim_b {
            for r in 0..new_bond_dim {
                let row = l * site_dim_b + s;
                if row < left.nrows() && r < left.ncols() {
                    tensor_b.set(l, s, r, left[[row, r]]);
                }
            }
        }
    }
    tci.site_tensors[b] = tensor_b;

    // Convert right matrix to tensor at site b+1
    let site_dim_bp1 = tci.local_dims[b + 1];
    let right_dim = if b + 1 == tci.len() - 1 { 1 } else { tci.j_set[b + 1].len() };

    let mut tensor_bp1 = Tensor3::zeros(new_bond_dim, site_dim_bp1, right_dim);
    for l in 0..new_bond_dim {
        for s in 0..site_dim_bp1 {
            for r in 0..right_dim {
                let col = s * right_dim + r;
                if l < right.nrows() && col < right.ncols() {
                    tensor_bp1.set(l, s, r, right[[l, col]]);
                }
            }
        }
    }
    tci.site_tensors[b + 1] = tensor_bp1;

    // Update bond error
    let errors = luci.pivot_errors();
    if !errors.is_empty() {
        tci.bond_errors[b] = *errors.last().unwrap_or(&0.0);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensorci2_new() {
        let tci = TensorCI2::<f64>::new(vec![2, 3, 2]).unwrap();
        assert_eq!(tci.len(), 3);
        assert_eq!(tci.local_dims(), &[2, 3, 2]);
    }

    #[test]
    fn test_tensorci2_requires_two_sites() {
        let result = TensorCI2::<f64>::new(vec![2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_crossinterpolate2_constant() {
        let f = |_: &MultiIndex| 1.0f64;
        let local_dims = vec![2, 2];
        let first_pivot = vec![vec![0, 0]];
        let options = TCI2Options::default();

        let (tci, _ranks, _errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
            f,
            None,
            local_dims,
            first_pivot,
            options,
        ).unwrap();

        assert_eq!(tci.len(), 2);
        assert!(tci.rank() >= 1);
    }

    #[test]
    fn test_crossinterpolate2_with_batch_function() {
        // Use batch evaluation
        let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
        let batched_f = |indices: &[MultiIndex]| -> Vec<f64> {
            indices.iter().map(|idx| (idx[0] + idx[1] + 1) as f64).collect()
        };

        let local_dims = vec![3, 3];
        let first_pivot = vec![vec![1, 1]];
        let options = TCI2Options::default();

        let (tci, _ranks, _errors) = crossinterpolate2(
            f,
            Some(batched_f),
            local_dims,
            first_pivot,
            options,
        ).unwrap();

        assert_eq!(tci.len(), 2);
    }

    #[test]
    fn test_crossinterpolate2_rank2_function() {
        // f(i, j) = i + j
        let f = |idx: &MultiIndex| (idx[0] + idx[1]) as f64;
        let local_dims = vec![4, 4];
        let first_pivot = vec![vec![1, 1]];
        let mut options = TCI2Options::default();
        options.tolerance = 1e-12;
        options.max_iter = 10;

        let (tci, _ranks, errors) = crossinterpolate2::<f64, _, fn(&[MultiIndex]) -> Vec<f64>>(
            f,
            None,
            local_dims,
            first_pivot,
            options,
        ).unwrap();

        // Should converge to rank 2
        assert!(tci.rank() <= 3, "Expected rank <= 3, got {}", tci.rank());

        // Check error is small
        let final_error = errors.last().copied().unwrap_or(f64::INFINITY);
        assert!(final_error < 0.1, "Expected small error, got {}", final_error);
    }
}
