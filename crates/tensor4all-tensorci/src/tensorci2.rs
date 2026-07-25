//! TensorCI2 - Two-site Tensor Cross Interpolation algorithm
//!
//! This implements the TCI2 algorithm which uses two-site updates for
//! more efficient convergence. It supports batch evaluation of function
//! values through an explicit batch function parameter.

use crate::error::{Result, TCIError};
use crate::globalpivot::{DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput};
use rand::SeedableRng;
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use tensor4all_simplett::{tensor3_zeros, TTScalar, Tensor3, Tensor3Ops, TensorTrain};
use tensor4all_tcicore::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix, MatrixLuciScalar, MultiIndex,
    RrLUOptions, Scalar,
};
use tensor4all_tensorbackend::{solve_matrix, transpose, Matrix};

/// Configuration for the TCI2 algorithm ([`crossinterpolate2`]).
///
/// Controls convergence criteria, bond dimension limits, pivot search
/// strategy, and global pivot search parameters.
///
/// # Recommended starting point
///
/// For most problems the defaults work well. The fields you will most
/// commonly adjust are:
///
/// | Field | Typical range | Purpose |
/// |---|---|---|
/// | `tolerance` | `1e-6` -- `1e-12` | Relative convergence threshold |
/// | `max_bond_dim` | `50` -- `500` | Hard cap on bond dimension |
/// | `max_iter` | `20` -- `100` | Maximum number of half-sweeps |
/// | `seed` | `Some(42)` | Fix seed for reproducibility |
///
/// # Convergence criterion
///
/// The algorithm stops when **all** of the following hold for
/// `ncheck_history` consecutive iterations:
///
/// 1. The normalized bond error is below `tolerance`.
/// 2. No global pivots were added.
/// 3. The rank is stable.
///
/// Alternatively, it stops when the rank reaches `max_bond_dim`.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::TCI2Options;
///
/// // Default options
/// let opts = TCI2Options::default();
/// assert!((opts.tolerance - 1e-8).abs() < 1e-15);
/// assert_eq!(opts.max_iter, 20);
/// assert_eq!(opts.max_bond_dim, usize::MAX);
/// assert_eq!(opts.verbosity, 0);
///
/// // Custom options via struct update syntax
/// let custom = TCI2Options {
///     tolerance: 1e-12,
///     max_bond_dim: 100,
///     seed: Some(42),
///     ..TCI2Options::default()
/// };
/// assert!((custom.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(custom.max_bond_dim, 100);
/// assert_eq!(custom.seed, Some(42));
/// ```
#[derive(Debug, Clone)]
pub struct TCI2Options {
    /// Convergence tolerance (default: `1e-8`).
    ///
    /// When `normalize_error` is enabled (the default), this is the
    /// *relative* threshold: the bond error is divided by the maximum
    /// absolute function value seen so far. Typical choices are `1e-8` for
    /// moderate accuracy and `1e-12` for high accuracy.
    pub tolerance: f64,
    /// Maximum number of half-sweep iterations (default: `20`).
    ///
    /// Each iteration performs one forward or backward two-site sweep
    /// followed by a global pivot search. Increase if the function is
    /// difficult to converge.
    pub max_iter: usize,
    /// Hard upper bound on bond dimension (default: `usize::MAX`, i.e. no limit).
    ///
    /// The algorithm stops early once the rank reaches this value. For
    /// expensive functions, setting this to `50`--`500` avoids runaway
    /// computation.
    pub max_bond_dim: usize,
    /// Pivot search strategy (default: [`PivotSearchStrategy::Full`]).
    ///
    /// `Full` materializes the entire candidate matrix and finds the best
    /// pivot exactly. `Rook` uses lazy block-rook search and is faster for
    /// very large local dimensions but may miss some pivots.
    pub pivot_search: PivotSearchStrategy,
    /// Whether to normalize the bond error by the maximum observed sample
    /// value (default: `true`).
    ///
    /// When enabled, `tolerance` acts as a *relative* threshold. Disable
    /// to use `tolerance` as an *absolute* threshold.
    pub normalize_error: bool,
    /// Verbosity level (default: `0` = silent).
    ///
    /// `1` prints per-iteration summaries, `2` adds per-bond details.
    pub verbosity: usize,
    /// Maximum number of global pivots to add per iteration (default: `5`).
    pub max_nglobal_pivot: usize,
    /// Number of random starting points for global pivot search (default: `5`).
    ///
    /// Each starting point undergoes local optimization to find regions of
    /// high interpolation error.
    pub nsearch: usize,
    /// Sweep strategy for 2-site sweeps (default: [`Sweep2Strategy::BackAndForth`]).
    pub sweep_strategy: Sweep2Strategy,
    /// Number of recent iterations checked for convergence (default: `3`).
    ///
    /// The algorithm requires `ncheck_history` consecutive converged
    /// iterations before declaring convergence.
    pub ncheck_history: usize,
    /// Whether to use strictly nested index sets (default: `false`).
    ///
    /// When `false`, the algorithm keeps a history of previous index sets
    /// and merges them during sweeps, which generally improves convergence.
    pub strictly_nested: bool,
    /// Tolerance margin for global pivot search (default: `10.0`).
    ///
    /// Global pivots are accepted when their error exceeds
    /// `abs_tolerance * tol_margin_global_search`.
    pub tol_margin_global_search: f64,
    /// Random seed for reproducibility (default: `None` = OS entropy).
    ///
    /// Set to `Some(seed)` for deterministic results.
    pub seed: Option<u64>,
}

impl Default for TCI2Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 20,
            max_bond_dim: usize::MAX,
            pivot_search: PivotSearchStrategy::Full,
            normalize_error: true,
            verbosity: 0,
            max_nglobal_pivot: 5,
            nsearch: 5,
            sweep_strategy: Sweep2Strategy::BackAndForth,
            ncheck_history: 3,
            strictly_nested: false,
            tol_margin_global_search: 10.0,
            seed: None,
        }
    }
}

/// Reason why a TCI2 optimization stopped.
///
/// Only [`Self::Converged`] means that the complete history-based convergence
/// criterion was satisfied. Reaching a configured limit is reported separately
/// so adaptive callers can choose whether to accept or refine the result.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};
///
/// let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
///     |_| 1.0,
///     None,
///     vec![2, 2],
///     vec![vec![0, 0]],
///     TCI2Options {
///         ncheck_history: 1,
///         nsearch: 0,
///         max_nglobal_pivot: 0,
///         ..TCI2Options::default()
///     },
/// ).unwrap();
/// assert_eq!(result.termination, TCI2Termination::Converged);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TCI2Termination {
    /// Bond error, global-pivot history, and rank stability all converged.
    Converged,
    /// The recent rank history reached the configured maximum bond dimension.
    MaxBondDimension,
    /// The iteration limit was exhausted before another stopping condition held.
    MaxIterations,
}

/// Result of a complete TCI2 optimization.
///
/// This groups the optimized state with its sweep histories and termination
/// reason. Use [`TCI2Termination`] rather than the last error alone when deciding
/// whether the optimization converged.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::AbstractTensorTrain;
/// use tensor4all_tensorci::{crossinterpolate2, TCI2Options, TCI2Termination};
///
/// let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
///     |idx| ((idx[0] + 1) * (idx[1] + 1)) as f64,
///     None,
///     vec![2, 2],
///     vec![vec![1, 1]],
///     TCI2Options {
///         ncheck_history: 1,
///         nsearch: 0,
///         max_nglobal_pivot: 0,
///         ..TCI2Options::default()
///     },
/// ).unwrap();
/// assert_eq!(result.termination, TCI2Termination::Converged);
/// let tt = result.tci.to_tensor_train().unwrap();
/// assert!((tt.evaluate(&[1, 0]).unwrap() - 2.0).abs() < 1.0e-12);
/// ```
#[derive(Debug, Clone)]
pub struct TCI2OptimizationResult<T: Scalar + TTScalar> {
    /// Optimized interpolation state after the final one-site cleanup sweep.
    pub tci: TensorCI2<T>,
    /// Bond dimension recorded after each half-sweep.
    pub ranks: Vec<usize>,
    /// Normalized or absolute error estimate recorded after each half-sweep.
    pub errors: Vec<f64>,
    /// Reason why optimization stopped.
    pub termination: TCI2Termination,
}

#[derive(Clone, Copy)]
struct Sweep1SiteBondConfig {
    rel_tol: f64,
    abs_tol: f64,
    max_bond_dim: usize,
    update_tensors: bool,
}

struct PivotUpdateContext<'a, B> {
    batched_f: &'a Option<B>,
    left_orthogonal: bool,
    options: &'a TCI2Options,
    extra_i_set: &'a [MultiIndex],
    extra_j_set: &'a [MultiIndex],
}

/// Strategy for finding new pivots during a two-site update.
///
/// Controls how much of the candidate matrix is evaluated when searching
/// for the next pivot at each bond.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::PivotSearchStrategy;
///
/// // Full is the default
/// let strategy = PivotSearchStrategy::default();
/// assert_eq!(strategy, PivotSearchStrategy::Full);
///
/// // Both variants can be compared
/// assert_ne!(PivotSearchStrategy::Full, PivotSearchStrategy::Rook);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PivotSearchStrategy {
    /// Evaluate the entire candidate matrix and pick the best pivot.
    ///
    /// Exact but costs O(D_left * d * d * D_right) evaluations per bond,
    /// where d is the local dimension. This is the default and recommended
    /// for most problems.
    #[default]
    Full,
    /// Use lazy block-rook pivoting over partial matrix blocks.
    ///
    /// Avoids materializing the full candidate matrix, making it faster
    /// for very large local dimensions. Error normalization uses the
    /// maximum sample value observed through the lazy requests rather
    /// than a full-grid scan.
    Rook,
}

/// Direction of two-site sweeps.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::Sweep2Strategy;
///
/// // BackAndForth is the default
/// let strategy = Sweep2Strategy::default();
/// assert_eq!(strategy, Sweep2Strategy::BackAndForth);
///
/// // All three variants
/// let fwd = Sweep2Strategy::Forward;
/// let bwd = Sweep2Strategy::Backward;
/// assert_ne!(fwd, bwd);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Sweep2Strategy {
    /// Sweep left-to-right only.
    Forward,
    /// Sweep right-to-left only.
    Backward,
    /// Alternate between forward and backward sweeps (default).
    ///
    /// This is the most robust choice: forward sweeps improve
    /// right-canonical form while backward sweeps improve left-canonical
    /// form, and alternating achieves better overall convergence.
    #[default]
    BackAndForth,
}

/// State object for the two-site Tensor Cross Interpolation algorithm.
///
/// Holds the current index sets (I, J), site tensors, and error statistics
/// produced by [`crossinterpolate2`]. After interpolation, call
/// [`to_tensor_train`](Self::to_tensor_train) to extract the resulting
/// [`TensorTrain`].
///
/// # Key methods
///
/// | Method | Purpose |
/// |---|---|
/// | [`rank`](Self::rank) | Maximum bond dimension |
/// | [`link_dims`](Self::link_dims) | Bond dimensions at each bond |
/// | [`to_tensor_train`](Self::to_tensor_train) | Extract the tensor train |
/// | [`max_bond_error`](Self::max_bond_error) | Largest bond error from last sweep |
/// | [`pivot_errors`](Self::pivot_errors) | Per-bond pivot errors from back-truncation |
///
/// You normally do not construct `TensorCI2` directly; use
/// [`crossinterpolate2`] instead.
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
    /// Maximum observed sample value found during function evaluation
    max_sample_value: f64,
    /// History of I-sets for non-strictly-nested mode
    i_set_history: Vec<Vec<Vec<MultiIndex>>>,
    /// History of J-sets for non-strictly-nested mode
    j_set_history: Vec<Vec<Vec<MultiIndex>>>,
}

impl<T> TensorCI2<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
{
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
            site_tensors: local_dims.iter().map(|&d| tensor3_zeros(0, d, 0)).collect(),
            pivot_errors: Vec::new(),
            bond_errors: vec![0.0; n.saturating_sub(1)],
            max_sample_value: 0.0,
            i_set_history: Vec::new(),
            j_set_history: Vec::new(),
        })
    }

    pub(crate) fn from_parts_for_conversion(
        local_dims: Vec<usize>,
        i_set: Vec<Vec<MultiIndex>>,
        j_set: Vec<Vec<MultiIndex>>,
        site_tensors: Vec<Tensor3<T>>,
        pivot_errors: Vec<f64>,
        bond_errors: Vec<f64>,
        max_sample_value: f64,
    ) -> Result<Self> {
        if local_dims.len() < 2 {
            return Err(TCIError::DimensionMismatch {
                message: "local_dims should have at least 2 elements".to_string(),
            });
        }
        let n = local_dims.len();
        if i_set.len() != n || j_set.len() != n || site_tensors.len() != n {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "conversion parts should have length {n}; got I={}, J={}, tensors={}",
                    i_set.len(),
                    j_set.len(),
                    site_tensors.len()
                ),
            });
        }
        Ok(Self {
            i_set,
            j_set,
            local_dims,
            site_tensors,
            pivot_errors,
            bond_errors,
            max_sample_value,
            i_set_history: Vec::new(),
            j_set_history: Vec::new(),
        })
    }

    /// Construct a [`TensorCI2`] from an existing tensor train, consuming the train.
    ///
    /// # Arguments
    ///
    /// * `tt` -- Tensor train whose site tensors are consumed by the
    ///   resulting `TensorCI2`.
    /// * `options` -- One-site index extraction tolerance, rank cap, and
    ///   sweep count.
    ///
    /// # Returns
    ///
    /// A `TensorCI2` with site tensors and index sets extracted from `tt`.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError`] if the tensor train has fewer than two sites, if
    /// options are invalid, or if one-site LU index extraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::{AbstractTensorTrain, TensorTrain};
    /// use tensor4all_tensorci::{TensorCI2, TensorCI2FromTensorTrainOptions};
    ///
    /// let tt = TensorTrain::<f64>::constant(&[2, 3], 2.5);
    /// let tci = TensorCI2::from_tensor_train(
    ///     tt,
    ///     TensorCI2FromTensorTrainOptions::default(),
    /// )
    /// .unwrap();
    ///
    /// let roundtrip = tci.to_tensor_train().unwrap();
    /// assert!((roundtrip.evaluate(&[1, 2]).unwrap() - 2.5).abs() < 1e-12);
    /// assert_eq!(tci.link_dims(), vec![1]);
    /// ```
    pub fn from_tensor_train(
        tt: TensorTrain<T>,
        options: crate::conversion::TensorCI2FromTensorTrainOptions,
    ) -> Result<Self>
    where
        T: tensor4all_tensorbackend::BlasMul,
    {
        crate::conversion::tensorci2_from_tensor_train(tt, options)
    }

    /// Construct a [`TensorCI2`] from explicit pivot index sets.
    ///
    /// This constructor is the strict Rust boundary for index-set based
    /// conversion code. The index sets are represented as
    /// `Vec<Vec<MultiIndex>>`: one outer entry per site, and one multi-index
    /// per pivot key at that site.
    ///
    /// # Arguments
    ///
    /// * `local_dims` -- Number of values at each site. Every dimension must
    ///   be nonzero and at least two sites are required.
    /// * `i_set` -- Left index sets. `i_set[p]` contains prefixes of length
    ///   `p`, and `i_set[0]` must be `[[]]`.
    /// * `j_set` -- Right index sets. `j_set[p]` contains suffixes of length
    ///   `n - p - 1`, and `j_set[n - 1]` must be `[[]]`.
    /// * `f` -- Function used to populate site tensors from the explicit
    ///   pivot sets.
    ///
    /// # Returns
    ///
    /// A `TensorCI2` with validated index sets and freshly populated site
    /// tensors.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError`] if dimensions are invalid, index-set lengths do
    /// not match the number of sites, indices are out of range, boundary
    /// conventions are violated, adjacent bond ranks are inconsistent, all
    /// sampled values are zero, duplicate pivot keys are present, or the
    /// one-site interpolation solve fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::AbstractTensorTrain;
    /// use tensor4all_tensorci::TensorCI2;
    /// use tensor4all_tcicore::MultiIndex;
    ///
    /// let f = |idx: &MultiIndex| (idx[0] + idx[1] + 1) as f64;
    /// let tci = TensorCI2::from_index_sets(
    ///     vec![4, 4],
    ///     vec![vec![vec![]], vec![vec![0], vec![1]]],
    ///     vec![vec![vec![0], vec![1]], vec![vec![]]],
    ///     &f,
    /// )
    /// .unwrap();
    ///
    /// let tt = tci.to_tensor_train().unwrap();
    /// assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
    /// assert_eq!(tci.link_dims(), vec![2]);
    /// ```
    pub fn from_index_sets<F>(
        local_dims: Vec<usize>,
        i_set: Vec<Vec<MultiIndex>>,
        j_set: Vec<Vec<MultiIndex>>,
        f: &F,
    ) -> Result<Self>
    where
        F: Fn(&MultiIndex) -> T,
    {
        validate_explicit_index_sets(&local_dims, &i_set, &j_set)?;
        let max_sample_value = max_sample_over_index_sets(&local_dims, &i_set, &j_set, f);
        if max_sample_value < 1e-30 {
            return Err(TCIError::InvalidPivot {
                message: "explicit TensorCI2 index sets only sample zero values".to_string(),
            });
        }

        let n = local_dims.len();
        let mut tci = Self {
            i_set,
            j_set,
            local_dims: local_dims.clone(),
            site_tensors: local_dims.iter().map(|&d| tensor3_zeros(0, d, 0)).collect(),
            pivot_errors: Vec::new(),
            bond_errors: vec![0.0; n - 1],
            max_sample_value,
            i_set_history: Vec::new(),
            j_set_history: Vec::new(),
        };
        tci.fill_site_tensors(f)?;
        Ok(tci)
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
            return if self.i_set.is_empty() || self.i_set[0].is_empty() {
                0
            } else {
                1
            };
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

    /// Get the maximum observed sample value seen so far
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
        self.site_tensors
            .iter()
            .all(|t| t.left_dim() > 0 || t.right_dim() > 0)
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

    /// Get I-set at site p
    pub fn i_set(&self, p: usize) -> &[MultiIndex] {
        &self.i_set[p]
    }

    /// Get J-set at site p
    pub fn j_set(&self, p: usize) -> &[MultiIndex] {
        &self.j_set[p]
    }

    /// Invalidate all site tensors
    pub fn invalidate_site_tensors(&mut self) {
        for p in 0..self.len() {
            self.site_tensors[p] = tensor3_zeros(0, self.local_dims[p], 0);
        }
    }

    /// Flush pivot errors (reset to empty)
    pub fn flush_pivot_errors(&mut self) {
        self.pivot_errors.clear();
    }

    /// Perform one 2-site sweep.
    ///
    /// This is a public wrapper around the internal `update_pivots` logic,
    /// suitable for calling from C-API.
    pub fn sweep2site<F, B>(
        &mut self,
        f: &F,
        batched_f: &Option<B>,
        forward: bool,
        options: &TCI2Options,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
        B: Fn(&[MultiIndex]) -> Vec<T>,
    {
        let n = self.len();
        self.invalidate_site_tensors();
        self.flush_pivot_errors();

        let empty: Vec<MultiIndex> = Vec::new();
        if forward {
            for b in 0..n - 1 {
                update_pivots(
                    self,
                    b,
                    f,
                    PivotUpdateContext {
                        batched_f,
                        left_orthogonal: true,
                        options,
                        extra_i_set: &empty,
                        extra_j_set: &empty,
                    },
                )?;
            }
        } else {
            for b in (0..n - 1).rev() {
                update_pivots(
                    self,
                    b,
                    f,
                    PivotUpdateContext {
                        batched_f,
                        left_orthogonal: false,
                        options,
                        extra_i_set: &empty,
                        extra_j_set: &empty,
                    },
                )?;
            }
        }

        // Fill site tensors after sweep
        self.fill_site_tensors(f)?;
        Ok(())
    }

    /// Update pivot errors with element-wise max
    fn update_pivot_errors(&mut self, errors: &[f64]) {
        if self.pivot_errors.len() < errors.len() {
            self.pivot_errors.resize(errors.len(), 0.0);
        }
        for (i, &e) in errors.iter().enumerate() {
            self.pivot_errors[i] = self.pivot_errors[i].max(e);
        }
    }

    /// Evaluate function at all combinations of left indices × local index × right indices.
    ///
    /// Returns a 3D array of shape (len(i_indices), local_dim, len(j_indices)).
    fn fill_tensor<F>(
        &self,
        f: &F,
        i_indices: &[MultiIndex],
        j_indices: &[MultiIndex],
        local_dim: usize,
        site: usize,
    ) -> Tensor3<T>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let ni = i_indices.len();
        let nj = j_indices.len();
        let mut tensor = tensor3_zeros(ni, local_dim, nj);
        for (ii, i_multi) in i_indices.iter().enumerate() {
            for s in 0..local_dim {
                for (jj, j_multi) in j_indices.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.push(s);
                    full_idx.extend(j_multi.iter().cloned());
                    debug_assert_eq!(
                        full_idx.len(),
                        self.local_dims.len(),
                        "fill_tensor: full_idx length {} != n_sites {} at site {}",
                        full_idx.len(),
                        self.local_dims.len(),
                        site
                    );
                    let val = f(&full_idx);
                    tensor.set3(ii, s, jj, val);
                }
            }
        }
        tensor
    }

    /// Perform a 1-site sweep, updating I/J sets and optionally site tensors.
    ///
    /// This is used for cleanup after adding global pivots, and for computing
    /// canonical site tensors.
    ///
    /// Port of Julia's `sweep1site!` from `tensorci2.jl`.
    pub fn sweep1site<F>(
        &mut self,
        f: &F,
        forward: bool,
        rel_tol: f64,
        abs_tol: f64,
        max_bond_dim: usize,
        update_tensors: bool,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.flush_pivot_errors();
        self.invalidate_site_tensors();

        let n = self.len();
        let bond_config = Sweep1SiteBondConfig {
            rel_tol,
            abs_tol,
            max_bond_dim,
            update_tensors,
        };

        if forward {
            for b in 0..n - 1 {
                self.sweep1site_at_bond(f, b, true, bond_config)?;
            }
        } else {
            for b in (1..n).rev() {
                self.sweep1site_at_bond(f, b, false, bond_config)?;
            }
        }

        // Update last tensor according to last index set
        if update_tensors {
            let last_idx = if forward { n - 1 } else { 0 };
            let tensor = self.fill_tensor(
                f,
                &self.i_set[last_idx].clone(),
                &self.j_set[last_idx].clone(),
                self.local_dims[last_idx],
                last_idx,
            );
            self.site_tensors[last_idx] = tensor;
        }

        Ok(())
    }

    /// Process one bond during 1-site sweep.
    fn sweep1site_at_bond<F>(
        &mut self,
        f: &F,
        b: usize,
        forward: bool,
        config: Sweep1SiteBondConfig,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        // Build combined indices: for forward, Kronecker(I_b, local_b) × J_b
        //                         for backward, I_b × Kronecker(local_b, J_b)
        let (is, js) = if forward {
            (self.kronecker_i(b), self.j_set[b].clone())
        } else {
            (self.i_set[b].clone(), self.kronecker_j(b))
        };

        if is.is_empty() || js.is_empty() {
            return Ok(());
        }

        // Build Pi matrix by evaluating function at all (I, J) combinations
        let ni = is.len();
        let nj = js.len();
        let mut pi = Matrix::zeros(ni, nj);
        for (i, i_multi) in is.iter().enumerate() {
            for (j, j_multi) in js.iter().enumerate() {
                let mut full_idx = i_multi.clone();
                full_idx.extend(j_multi.iter().cloned());
                let val = f(&full_idx);
                pi[[i, j]] = val;
                let abs_val = f64::sqrt(Scalar::abs_sq(val));
                if abs_val > self.max_sample_value {
                    self.max_sample_value = abs_val;
                }
            }
        }

        // LU-based cross interpolation
        let lu_options = RrLUOptions {
            max_rank: config.max_bond_dim,
            rel_tol: config.rel_tol,
            abs_tol: config.abs_tol,
            left_orthogonal: forward,
        };
        let factors = matrix_luci_factors_from_matrix(&pi, Some(lu_options))?;

        let row_indices = &factors.row_indices;
        let col_indices = &factors.col_indices;

        // Update I/J sets
        if forward {
            self.i_set[b + 1] = row_indices.iter().map(|&i| is[i].clone()).collect();
            self.j_set[b] = col_indices.iter().map(|&j| js[j].clone()).collect();
        } else {
            self.i_set[b] = row_indices.iter().map(|&i| is[i].clone()).collect();
            self.j_set[b - 1] = col_indices.iter().map(|&j| js[j].clone()).collect();
        }

        // Update site tensor
        if config.update_tensors {
            let mat = if forward {
                factors.left.clone()
            } else {
                factors.right.clone()
            };
            let local_dim = self.local_dims[b];
            if forward {
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let right_dim = factors.rank.max(1);
                let mut tensor = tensor3_zeros(left_dim, local_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..local_dim {
                        for r in 0..right_dim {
                            let row = l * local_dim + s;
                            if row < mat.nrows() && r < mat.ncols() {
                                tensor.set3(l, s, r, mat[[row, r]]);
                            }
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            } else {
                let left_dim = factors.rank.max(1);
                let right_dim = if b == self.len() - 1 {
                    1
                } else {
                    self.j_set[b].len()
                };
                let mut tensor = tensor3_zeros(left_dim, local_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..local_dim {
                        for r in 0..right_dim {
                            let col = s * right_dim + r;
                            if l < mat.nrows() && col < mat.ncols() {
                                tensor.set3(l, s, r, mat[[l, col]]);
                            }
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            }
        }

        // Update errors
        let errors = &factors.pivot_errors;
        if !errors.is_empty() {
            let bond_idx = if forward { b } else { b - 1 };
            self.bond_errors[bond_idx] = *errors.last().unwrap_or(&0.0);
        }
        self.update_pivot_errors(errors);

        Ok(())
    }

    /// Fill all site tensors using 1-site LU decomposition at each bond.
    ///
    /// For each site b (except the last), computes the Pi matrix
    /// (Kronecker(I_b, d_b) × J_b) and the pivot matrix P (I_{b+1} × J_b),
    /// then solves P^T \ Pi^T to get the site tensor T_b = Pi * P^{-1}.
    /// The last site tensor is set by direct evaluation.
    ///
    /// Port of Julia's `fillsitetensors!` / `setsitetensor!`.
    pub fn fill_site_tensors<F>(&mut self, f: &F) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let n = self.len();
        for b in 0..n {
            let i_kron = self.kronecker_i(b);
            let j_set_b = self.j_set[b].clone();

            if i_kron.is_empty() || j_set_b.is_empty() {
                continue;
            }

            // Pi1: evaluate f at (Kronecker(I_b, d_b), J_b)
            let ni = i_kron.len();
            let nj = j_set_b.len();
            let mut pi1 = Matrix::zeros(ni, nj);
            for (i, i_multi) in i_kron.iter().enumerate() {
                for (j, j_multi) in j_set_b.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    pi1[[i, j]] = f(&full_idx);
                }
            }

            if b == n - 1 {
                // Last site: store Pi1 directly
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let site_dim = self.local_dims[b];
                let right_dim = 1; // last site
                let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        let row = l * site_dim + s;
                        if row < ni {
                            tensor.set3(l, s, 0, pi1[[row, 0]]);
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            } else {
                // Non-last site: solve P^T \ Pi1^T to get Tmat = Pi1 * P^{-1}
                // P = pivot matrix (I_{b+1} × J_b)
                let i_set_bp1 = self.i_set[b + 1].clone();
                let np = i_set_bp1.len();

                let mut p_mat = Matrix::zeros(np, nj);
                for (i, i_multi) in i_set_bp1.iter().enumerate() {
                    for (j, j_multi) in j_set_b.iter().enumerate() {
                        let mut full_idx = i_multi.clone();
                        full_idx.extend(j_multi.iter().cloned());
                        p_mat[[i, j]] = f(&full_idx);
                    }
                }

                // Solve P^T * X_t = Pi1^T through the configured tensor backend.
                let p_t = transpose(&p_mat);
                let pi1_t = transpose(&pi1);
                let x_t = solve_matrix(&p_t, &pi1_t).map_err(|err| TCIError::InvalidOperation {
                    message: format!("one-site interpolation solve failed: {err}"),
                })?;

                // X = X_t^T → shape (ni, np) = (|I_b|*d_b, |I_{b+1}|)
                let left_dim = if b == 0 { 1 } else { self.i_set[b].len() };
                let site_dim = self.local_dims[b];
                let right_dim = np; // = |I_{b+1}|
                let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);
                for l in 0..left_dim {
                    for s in 0..site_dim {
                        for r in 0..right_dim {
                            let row = l * site_dim + s;
                            tensor.set3(l, s, r, x_t[[r, row]]);
                        }
                    }
                }
                self.site_tensors[b] = tensor;
            }
        }
        Ok(())
    }

    /// Make the TCI canonical by performing 3 one-site sweeps.
    ///
    /// 1. Forward sweep (exact, no truncation)
    /// 2. Backward sweep (with truncation)
    /// 3. Forward sweep (with truncation + update tensors)
    ///
    /// Port of Julia's `makecanonical!`.
    pub fn make_canonical<F>(
        &mut self,
        f: &F,
        rel_tol: f64,
        abs_tol: f64,
        max_bond_dim: usize,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        // First half-sweep: exact, no truncation
        self.sweep1site(f, true, 0.0, 0.0, usize::MAX, false)?;
        // Second half-sweep: backward with truncation
        self.sweep1site(f, false, rel_tol, abs_tol, max_bond_dim, false)?;
        // Third half-sweep: forward with truncation and tensor updates
        self.sweep1site(f, true, rel_tol, abs_tol, max_bond_dim, true)?;
        Ok(())
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

fn validate_explicit_index_sets(
    local_dims: &[usize],
    i_set: &[Vec<MultiIndex>],
    j_set: &[Vec<MultiIndex>],
) -> Result<()> {
    if local_dims.len() < 2 {
        return Err(TCIError::DimensionMismatch {
            message: "local_dims should have at least 2 elements".to_string(),
        });
    }
    if let Some((site, _)) = local_dims.iter().enumerate().find(|&(_, &dim)| dim == 0) {
        return Err(TCIError::DimensionMismatch {
            message: format!("local_dims[{site}] must be nonzero"),
        });
    }
    let n = local_dims.len();
    if i_set.len() != n || j_set.len() != n {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "I/J set length must match number of sites {n}; got I={}, J={}",
                i_set.len(),
                j_set.len()
            ),
        });
    }
    if i_set[0] != [Vec::<usize>::new()] {
        return Err(TCIError::DimensionMismatch {
            message: "I set at site 0 must be exactly [[]]".to_string(),
        });
    }
    if j_set[n - 1] != [Vec::<usize>::new()] {
        return Err(TCIError::DimensionMismatch {
            message: "J set at the last site must be exactly [[]]".to_string(),
        });
    }

    for site in 0..n {
        if i_set[site].is_empty() || j_set[site].is_empty() {
            return Err(TCIError::DimensionMismatch {
                message: format!("I/J set at site {site} must be nonempty"),
            });
        }
        validate_left_index_set(local_dims, site, &i_set[site])?;
        validate_right_index_set(local_dims, site, &j_set[site])?;
        validate_unique_index_set("I", site, &i_set[site])?;
        validate_unique_index_set("J", site, &j_set[site])?;
    }

    for bond in 0..n - 1 {
        if i_set[bond + 1].len() != j_set[bond].len() {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "I/J set rank mismatch at bond {bond}: I[{}] has {}, J[{bond}] has {}",
                    bond + 1,
                    i_set[bond + 1].len(),
                    j_set[bond].len()
                ),
            });
        }
    }
    Ok(())
}

fn validate_unique_index_set(name: &str, site: usize, values: &[MultiIndex]) -> Result<()> {
    let mut seen = HashSet::with_capacity(values.len());
    for value in values {
        if !seen.insert(value.as_slice()) {
            return Err(TCIError::DimensionMismatch {
                message: format!("{name}[{site}] contains duplicate index {value:?}"),
            });
        }
    }
    Ok(())
}

fn validate_left_index_set(local_dims: &[usize], site: usize, values: &[MultiIndex]) -> Result<()> {
    for value in values {
        if value.len() != site {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "I[{site}] contains index {value:?} with length {}, expected {site}",
                    value.len()
                ),
            });
        }
        for (axis, &index) in value.iter().enumerate() {
            let dim = local_dims[axis];
            if index >= dim {
                return Err(TCIError::IndexOutOfBounds {
                    message: format!(
                        "I[{site}] contains coordinate {index} at axis {axis}, outside 0..{dim}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_right_index_set(
    local_dims: &[usize],
    site: usize,
    values: &[MultiIndex],
) -> Result<()> {
    let expected_len = local_dims.len() - site - 1;
    for value in values {
        if value.len() != expected_len {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "J[{site}] contains index {value:?} with length {}, expected {expected_len}",
                    value.len()
                ),
            });
        }
        for (offset, &index) in value.iter().enumerate() {
            let axis = site + 1 + offset;
            let dim = local_dims[axis];
            if index >= dim {
                return Err(TCIError::IndexOutOfBounds {
                    message: format!(
                        "J[{site}] contains coordinate {index} at axis {axis}, outside 0..{dim}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn max_sample_over_index_sets<T, F>(
    local_dims: &[usize],
    i_set: &[Vec<MultiIndex>],
    j_set: &[Vec<MultiIndex>],
    f: &F,
) -> f64
where
    T: Scalar + TTScalar,
    F: Fn(&MultiIndex) -> T,
{
    let mut max_sample = 0.0_f64;
    for site in 0..local_dims.len() {
        for i_multi in &i_set[site] {
            for local in 0..local_dims[site] {
                for j_multi in &j_set[site] {
                    let mut full_idx = i_multi.clone();
                    full_idx.push(local);
                    full_idx.extend(j_multi.iter().copied());
                    max_sample = max_sample.max(Scalar::abs_sq(f(&full_idx)).sqrt());
                }
            }
        }
    }
    max_sample
}

/// Check convergence based on history of ranks, errors, and global pivots.
///
/// Port of Julia's `convergencecriterion`.
fn convergence_criterion(
    ranks: &[usize],
    errors: &[f64],
    nglobal_pivots: &[usize],
    tolerance: f64,
    max_bond_dim: usize,
    ncheck_history: usize,
) -> Option<TCI2Termination> {
    if errors.len() < ncheck_history {
        return None;
    }

    let n = errors.len();
    let last_errors = &errors[n - ncheck_history..];
    let last_ranks = &ranks[n - ncheck_history..];
    let last_ngp = &nglobal_pivots[n - ncheck_history..];

    let errors_converged = last_errors.iter().all(|&e| e < tolerance);
    let no_global_pivots = last_ngp.iter().all(|&n| n == 0);
    let rank_stable =
        last_ranks.iter().min().copied().unwrap_or(0) == last_ranks.last().copied().unwrap_or(0);
    let at_max_bond = last_ranks.iter().all(|&r| r >= max_bond_dim);

    if at_max_bond {
        Some(TCI2Termination::MaxBondDimension)
    } else if errors_converged && no_global_pivots && rank_stable {
        Some(TCI2Termination::Converged)
    } else {
        None
    }
}

/// Approximate a function as a tensor train using the TCI2 algorithm.
///
/// This is the main entry point for tensor cross interpolation. It
/// performs alternating two-site sweeps with global pivot search until
/// convergence, then cleans up with a final one-site sweep.
///
/// # Arguments
///
/// * `f` -- Function to interpolate. Takes a multi-index `&Vec<usize>` where
///   each element is in `0..local_dims[i]` (0-indexed) and returns a scalar.
/// * `batched_f` -- Optional batch evaluation function for efficiency.
///   Takes `&[Vec<usize>]` and returns `Vec<T>`. Pass `None` to use
///   element-wise evaluation only.
/// * `local_dims` -- Number of values each index can take. Must have at
///   least 2 elements (TCI requires at least 2 sites).
/// * `initial_pivots` -- Starting multi-indices for the algorithm. At least
///   one pivot must have a non-zero function value. Choose pivots where
///   `|f|` is large for best convergence. If empty, defaults to the
///   all-zeros index.
/// * `options` -- Algorithm configuration; see [`TCI2Options`].
///
/// # Returns
///
/// A [`TCI2OptimizationResult`] containing the interpolation state, sweep
/// histories, and the reason optimization stopped. Check
/// [`TCI2OptimizationResult::termination`] rather than inferring convergence
/// from the final error alone.
///
/// # Errors
///
/// Returns [`TCIError::DimensionMismatch`] if `local_dims` has fewer than
/// 2 elements, or [`TCIError::InvalidPivot`] if all initial pivots
/// evaluate to zero.
///
/// # Examples
///
/// Basic usage: interpolate `f(i, j) = i + j + 1` on a 4x4 grid.
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
/// use tensor4all_simplett::AbstractTensorTrain;
///
/// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
/// let local_dims = vec![4, 4];
/// let initial_pivots = vec![vec![3, 3]]; // pick where |f| is large
///
/// let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
///     f,
///     None,
///     local_dims,
///     initial_pivots,
///     TCI2Options {
///         tolerance: 1e-10,
///         seed: Some(42),
///         ..TCI2Options::default()
///     },
/// )
/// .unwrap();
///
/// // Verify convergence
/// assert_eq!(result.termination, tensor4all_tensorci::TCI2Termination::Converged);
///
/// // Evaluate through the tensor train
/// let tt = result.tci.to_tensor_train().unwrap();
/// let val = tt.evaluate(&[2, 3]).unwrap();
/// assert!((val - 6.0).abs() < 1e-10); // f(2,3) = 2+3+1 = 6
///
/// // Bond dimensions are available
/// assert!(!tt.link_dims().is_empty());
/// ```
pub fn crossinterpolate2<T, F, B>(
    f: F,
    batched_f: Option<B>,
    local_dims: Vec<usize>,
    initial_pivots: Vec<MultiIndex>,
    options: TCI2Options,
) -> Result<TCI2OptimizationResult<T>>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
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

    let finder = DefaultGlobalPivotFinder::new(
        options.nsearch,
        options.max_nglobal_pivot,
        options.tol_margin_global_search,
    );

    optimize_with_finder(tci, f, batched_f, options, finder)
}

/// Optimize an existing [`TensorCI2`] state with an injected global pivot finder.
///
/// Use this when the default random global pivot search is not appropriate for
/// the function being interpolated. The input state must already contain at
/// least one pivot, usually by calling [`TensorCI2::add_global_pivots`].
///
/// # Arguments
///
/// * `tci` -- Existing TCI2 state to optimize. It is consumed and returned
///   after optimization.
/// * `f` -- Scalar function to interpolate on zero-based multi-indices.
/// * `batched_f` -- Optional Torch-style batch callback. It receives a slice of
///   multi-indices and must return one value per input index.
/// * `options` -- Sweep, convergence, and truncation settings.
/// * `finder` -- Global pivot finder used after each two-site sweep.
///
/// # Returns
///
/// Returns a [`TCI2OptimizationResult`], matching [`crossinterpolate2`]. Its
/// rank and error histories contain one entry per half-sweep.
///
/// # Errors
///
/// Returns [`TCIError::InvalidPivot`] when the input state has no pivots. It
/// also forwards errors from two-site sweeps, tensor-train conversion, callback
/// length validation, and final one-site cleanup.
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::AbstractTensorTrain;
/// use tensor4all_tensorci::{
///     optimize_with_finder, DefaultGlobalPivotFinder, TCI2Options, TensorCI2,
/// };
///
/// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
/// let mut tci = TensorCI2::<f64>::new(vec![4, 4]).unwrap();
/// tci.add_global_pivots(&[vec![3, 3]]).unwrap();
///
/// let finder = DefaultGlobalPivotFinder::new(0, 0, 10.0);
/// let result = optimize_with_finder::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>, _>(
///     tci,
///     f,
///     None,
///     TCI2Options {
///         tolerance: 1e-10,
///         max_iter: 5,
///         ..TCI2Options::default()
///     },
///     finder,
/// )
/// .unwrap();
///
/// assert!(!result.ranks.is_empty());
/// assert!(!result.errors.is_empty());
/// let tt = result.tci.to_tensor_train().unwrap();
/// assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
/// ```
pub fn optimize_with_finder<T, F, B, G>(
    mut tci: TensorCI2<T>,
    f: F,
    batched_f: Option<B>,
    options: TCI2Options,
    finder: G,
) -> Result<TCI2OptimizationResult<T>>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
    G: GlobalPivotFinder,
{
    if tci.rank() == 0 {
        return Err(TCIError::InvalidPivot {
            message: "TensorCI2 state must contain at least one pivot before optimization"
                .to_string(),
        });
    }

    let n = tci.len();
    let mut errors = Vec::new();
    let mut ranks = Vec::new();
    let mut nglobal_pivots_history: Vec<usize> = Vec::new();
    let mut termination = TCI2Termination::MaxIterations;

    let mut rng = if let Some(seed) = options.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_os_rng()
    };

    for iter in 0..options.max_iter {
        let error_normalization = if options.normalize_error && tci.max_sample_value > 0.0 {
            tci.max_sample_value
        } else {
            1.0
        };
        let abs_tol = options.tolerance * error_normalization;

        // Determine sweep direction
        let is_forward = match options.sweep_strategy {
            Sweep2Strategy::Forward => true,
            Sweep2Strategy::Backward => false,
            Sweep2Strategy::BackAndForth => iter % 2 == 0,
        };

        // Get extra index sets from history for non-strictly-nested mode
        let (extra_i_set, extra_j_set) =
            if !options.strictly_nested && !tci.i_set_history.is_empty() {
                let last = tci.i_set_history.len() - 1;
                (
                    tci.i_set_history[last].clone(),
                    tci.j_set_history[last].clone(),
                )
            } else {
                let empty: Vec<Vec<MultiIndex>> = (0..n).map(|_| Vec::new()).collect();
                (empty.clone(), empty)
            };

        // Save current sets to history
        tci.i_set_history.push(tci.i_set.clone());
        tci.j_set_history.push(tci.j_set.clone());

        // 2-site sweep
        tci.invalidate_site_tensors();
        tci.flush_pivot_errors();

        if is_forward {
            for b in 0..n - 1 {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    PivotUpdateContext {
                        batched_f: &batched_f,
                        left_orthogonal: true,
                        options: &options,
                        extra_i_set: &extra_i_set[b + 1],
                        extra_j_set: &extra_j_set[b],
                    },
                )?;
            }
        } else {
            for b in (0..n - 1).rev() {
                update_pivots(
                    &mut tci,
                    b,
                    &f,
                    PivotUpdateContext {
                        batched_f: &batched_f,
                        left_orthogonal: false,
                        options: &options,
                        extra_i_set: &extra_i_set[b + 1],
                        extra_j_set: &extra_j_set[b],
                    },
                )?;
            }
        }

        // Fill site tensors after sweep
        tci.fill_site_tensors(&f)?;

        // Record error
        let error = tci.max_bond_error();
        let error_normalized = error / error_normalization;
        errors.push(error_normalized);

        // Global pivot search
        let tt = tci.to_tensor_train()?;
        let input = GlobalPivotSearchInput {
            local_dims: tci.local_dims.clone(),
            current_tt: tt,
            max_sample_value: tci.max_sample_value,
            i_set: tci.i_set.clone(),
            j_set: tci.j_set.clone(),
        };

        let global_pivots = finder.find_global_pivots(&input, &f, abs_tol, &mut rng);
        let n_global = global_pivots.len();
        tci.add_global_pivots(&global_pivots)?;
        nglobal_pivots_history.push(n_global);

        ranks.push(tci.rank());

        if options.verbosity > 0 {
            println!(
                "iteration = {}, rank = {}, error = {:.2e}, maxsamplevalue = {:.2e}, nglobalpivot = {}",
                iter + 1,
                tci.rank(),
                error_normalized,
                tci.max_sample_value,
                n_global
            );
        }

        // `errors` is already divided by `error_normalization`, so compare it
        // with the configured relative tolerance rather than `abs_tol`.
        if let Some(reason) = convergence_criterion(
            &ranks,
            &errors,
            &nglobal_pivots_history,
            options.tolerance,
            options.max_bond_dim,
            options.ncheck_history,
        ) {
            termination = reason;
            break;
        }
    }

    // Final 1-site sweep to:
    // 1. Remove unnecessary pivots added by global pivots
    // 2. Compute site tensors
    let error_normalization = if options.normalize_error && tci.max_sample_value > 0.0 {
        tci.max_sample_value
    } else {
        1.0
    };
    let abs_tol = options.tolerance * error_normalization;
    tci.sweep1site(&f, true, 1e-14, abs_tol, options.max_bond_dim, true)?;

    Ok(TCI2OptimizationResult {
        tci,
        ranks,
        errors,
        termination,
    })
}

/// Update pivots at bond b using LU-based cross interpolation
fn update_pivots<T, F, B>(
    tci: &mut TensorCI2<T>,
    b: usize,
    f: &F,
    context: PivotUpdateContext<'_, B>,
) -> Result<()>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    // Build combined index sets, including extra sets from history
    let mut i_combined = tci.kronecker_i(b);
    let mut j_combined = tci.kronecker_j(b + 1);

    // Union with extra sets (for non-strictly-nested mode)
    for extra in context.extra_i_set {
        if !i_combined.contains(extra) {
            i_combined.push(extra.clone());
        }
    }
    for extra in context.extra_j_set {
        if !j_combined.contains(extra) {
            j_combined.push(extra.clone());
        }
    }

    if i_combined.is_empty() || j_combined.is_empty() {
        return Ok(());
    }

    let factors = if context.options.pivot_search == PivotSearchStrategy::Full {
        let mut pi = Matrix::zeros(i_combined.len(), j_combined.len());

        if let Some(ref batch_fn) = context.batched_f {
            let mut all_indices: Vec<MultiIndex> =
                Vec::with_capacity(i_combined.len() * j_combined.len());
            for i_multi in &i_combined {
                for j_multi in &j_combined {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    all_indices.push(full_idx);
                }
            }

            let values = batch_fn(&all_indices);
            if values.len() != all_indices.len() {
                return Err(callback_length_mismatch(values.len(), all_indices.len()));
            }
            let mut idx = 0;
            for i in 0..i_combined.len() {
                for j in 0..j_combined.len() {
                    pi[[i, j]] = values[idx];
                    update_max_sample_value(tci, values[idx]);
                    idx += 1;
                }
            }
        } else {
            for (i, i_multi) in i_combined.iter().enumerate() {
                for (j, j_multi) in j_combined.iter().enumerate() {
                    let mut full_idx = i_multi.clone();
                    full_idx.extend(j_multi.iter().cloned());
                    let value = f(&full_idx);
                    pi[[i, j]] = value;
                    update_max_sample_value(tci, value);
                }
            }
        }

        matrix_luci_factors_from_matrix(
            &pi,
            Some(RrLUOptions {
                max_rank: context.options.max_bond_dim,
                rel_tol: context.options.tolerance,
                abs_tol: 0.0,
                left_orthogonal: context.left_orthogonal,
            }),
        )?
    } else {
        let evaluator = LazyPiEvaluator::new(
            &i_combined,
            &j_combined,
            f,
            context.batched_f,
            tci.max_sample_value,
        );
        let factors_result = matrix_luci_factors_from_blocks(
            i_combined.len(),
            j_combined.len(),
            |rows, cols, out: &mut [T]| {
                evaluator.fill_block(rows, cols, out);
            },
            RrLUOptions {
                max_rank: context.options.max_bond_dim,
                rel_tol: context.options.tolerance,
                abs_tol: 0.0,
                left_orthogonal: context.left_orthogonal,
            },
        );
        if let Some(err) = evaluator.take_error() {
            return Err(err);
        }
        tci.max_sample_value = evaluator.sampled_max();
        factors_result?
    };

    // Update I and J sets
    let row_indices = &factors.row_indices;
    let col_indices = &factors.col_indices;

    tci.i_set[b + 1] = row_indices.iter().map(|&i| i_combined[i].clone()).collect();
    tci.j_set[b] = col_indices.iter().map(|&j| j_combined[j].clone()).collect();

    // Skip site tensor update if extra sets were used (tensors will be
    // filled separately by fill_site_tensors after the sweep).
    if !context.extra_i_set.is_empty() || !context.extra_j_set.is_empty() {
        // Update bond error only
        let errors = &factors.pivot_errors;
        if !errors.is_empty() {
            tci.bond_errors[b] = *errors.last().unwrap_or(&0.0);
        }
        return Ok(());
    }

    // Update site tensors
    // Convert left matrix to tensor at site b
    let left_dim = if b == 0 { 1 } else { tci.i_set[b].len() };
    let site_dim_b = tci.local_dims[b];
    let new_bond_dim = factors.rank.max(1);

    let mut tensor_b = tensor3_zeros(left_dim, site_dim_b, new_bond_dim);
    for l in 0..left_dim {
        for s in 0..site_dim_b {
            for r in 0..new_bond_dim {
                let row = l * site_dim_b + s;
                if row < factors.left.nrows() && r < factors.left.ncols() {
                    tensor_b.set3(l, s, r, factors.left[[row, r]]);
                }
            }
        }
    }
    tci.site_tensors[b] = tensor_b;

    // Convert right matrix to tensor at site b+1
    let site_dim_bp1 = tci.local_dims[b + 1];
    let right_dim = if b + 1 == tci.len() - 1 {
        1
    } else {
        tci.j_set[b + 1].len()
    };

    let mut tensor_bp1 = tensor3_zeros(new_bond_dim, site_dim_bp1, right_dim);
    for l in 0..new_bond_dim {
        for s in 0..site_dim_bp1 {
            for r in 0..right_dim {
                let col = s * right_dim + r;
                if l < factors.right.nrows() && col < factors.right.ncols() {
                    tensor_bp1.set3(l, s, r, factors.right[[l, col]]);
                }
            }
        }
    }
    tci.site_tensors[b + 1] = tensor_bp1;

    // Update bond error
    if !factors.pivot_errors.is_empty() {
        tci.bond_errors[b] = *factors.pivot_errors.last().unwrap_or(&0.0);
    }

    Ok(())
}

fn update_max_sample_value<T: Scalar + TTScalar>(tci: &mut TensorCI2<T>, value: T) {
    let abs_val = f64::sqrt(Scalar::abs_sq(value));
    if abs_val > tci.max_sample_value {
        tci.max_sample_value = abs_val;
    }
}

fn build_full_index(
    i_combined: &[MultiIndex],
    j_combined: &[MultiIndex],
    row: usize,
    col: usize,
) -> MultiIndex {
    let mut full_idx = i_combined[row].clone();
    full_idx.extend(j_combined[col].iter().cloned());
    full_idx
}

fn callback_length_mismatch(actual: usize, expected: usize) -> TCIError {
    TCIError::InvalidOperation {
        message: format!(
            "batch callback returned {actual} values for {expected} requested entries"
        ),
    }
}

struct LazyPiEvaluator<'a, T, F, B>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    i_combined: &'a [MultiIndex],
    j_combined: &'a [MultiIndex],
    f: &'a F,
    batched_f: &'a Option<B>,
    cache: RefCell<HashMap<(usize, usize), T>>,
    pending_error: RefCell<Option<TCIError>>,
    sampled_max: Cell<f64>,
}

impl<'a, T, F, B> LazyPiEvaluator<'a, T, F, B>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar,
    F: Fn(&MultiIndex) -> T,
    B: Fn(&[MultiIndex]) -> Vec<T>,
{
    fn new(
        i_combined: &'a [MultiIndex],
        j_combined: &'a [MultiIndex],
        f: &'a F,
        batched_f: &'a Option<B>,
        initial_max: f64,
    ) -> Self {
        Self {
            i_combined,
            j_combined,
            f,
            batched_f,
            cache: RefCell::new(HashMap::new()),
            pending_error: RefCell::new(None),
            sampled_max: Cell::new(initial_max),
        }
    }

    fn fill_block(&self, rows: &[usize], cols: &[usize], out: &mut [T]) {
        if self.pending_error.borrow().is_some() {
            out.fill(T::zero());
            return;
        }

        let mut missing_entries = Vec::new();
        let mut missing_indices = Vec::new();

        {
            let cache_ref = self.cache.borrow();
            for (j_pos, &col) in cols.iter().enumerate() {
                for (i_pos, &row) in rows.iter().enumerate() {
                    let out_idx = i_pos + rows.len() * j_pos;
                    if let Some(&value) = cache_ref.get(&(row, col)) {
                        out[out_idx] = value;
                    } else {
                        missing_entries.push((out_idx, row, col));
                        missing_indices.push(build_full_index(
                            self.i_combined,
                            self.j_combined,
                            row,
                            col,
                        ));
                    }
                }
            }
        }

        if missing_entries.is_empty() {
            return;
        }

        let values = if let Some(batch_fn) = self.batched_f {
            batch_fn(&missing_indices)
        } else {
            missing_indices.iter().map(self.f).collect()
        };
        if values.len() != missing_entries.len() {
            *self.pending_error.borrow_mut() = Some(callback_length_mismatch(
                values.len(),
                missing_entries.len(),
            ));
            for (out_idx, _, _) in missing_entries {
                out[out_idx] = T::zero();
            }
            return;
        }

        let mut cache_ref = self.cache.borrow_mut();
        for ((out_idx, row, col), value) in missing_entries.into_iter().zip(values) {
            out[out_idx] = value;
            cache_ref.insert((row, col), value);

            let abs_val = f64::sqrt(Scalar::abs_sq(value));
            if abs_val > self.sampled_max.get() {
                self.sampled_max.set(abs_val);
            }
        }
    }

    fn sampled_max(&self) -> f64 {
        self.sampled_max.get()
    }

    fn take_error(&self) -> Option<TCIError> {
        self.pending_error.borrow_mut().take()
    }
}

#[cfg(test)]
mod tests;
