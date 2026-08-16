//! TensorCI1 - legacy one-site Tensor Cross Interpolation algorithm.

use crate::error::{validate_nonnegative_finite, validate_positive, Result, TCIError};
use tensor4all_core::{
    AbstractMatrixCI, IndexSet, MatrixACA, MatrixLuciScalar, MultiIndex, Scalar,
};
use tensor4all_simplett::{
    tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, TTScalar, Tensor3Ops,
};
use tensor4all_tensorbackend::{solve_matrix, transpose, Matrix, MatrixSolveScalar};

mod matrix_ci;

/// Direction of one-site TCI1 sweeps.
///
/// Use this with [`TCI1Options::sweep_strategy`] to match legacy TCI1
/// workflows that require deterministic forward-only, backward-only, or
/// alternating sweeps.
///
/// Related types: [`Sweep2Strategy`](crate::Sweep2Strategy) controls the
/// primary two-site TCI2 sweeps; `TCI1SweepStrategy` is the one-site TCI1
/// counterpart.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::TCI1SweepStrategy;
///
/// let strategy = TCI1SweepStrategy::default();
/// assert_eq!(strategy, TCI1SweepStrategy::BackAndForth);
/// assert_ne!(TCI1SweepStrategy::Forward, TCI1SweepStrategy::Backward);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TCI1SweepStrategy {
    /// Sweep left-to-right only.
    Forward,
    /// Sweep right-to-left only.
    Backward,
    /// Alternate between forward and backward sweeps.
    ///
    /// This is the default and generally the most robust choice when no
    /// legacy workflow requires a fixed direction.
    #[default]
    BackAndForth,
}

/// Configuration for [`crossinterpolate1`].
///
/// Use this to control convergence tolerance, sweep count, local pivot
/// acceptance, error normalization, and additional pivots for the legacy
/// one-site TensorCI workflow.
///
/// Related types: [`TCI2Options`](crate::TCI2Options) configures the primary
/// two-site algorithm; `TCI1Options` is the legacy one-site counterpart.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::TCI1Options;
///
/// let options = TCI1Options {
///     tolerance: 1e-12,
///     max_iter: 32,
///     ..TCI1Options::default()
/// };
///
/// assert!((options.tolerance - 1e-12).abs() < 1e-20);
/// assert_eq!(options.max_iter, 32);
/// assert_eq!(options.sweep_strategy, tensor4all_tensorci::TCI1SweepStrategy::BackAndForth);
/// assert!(options.normalize_error);
/// ```
#[derive(Debug, Clone)]
pub struct TCI1Options {
    /// Relative convergence tolerance when [`normalize_error`](Self::normalize_error) is `true`.
    ///
    /// The default is `1e-8`. Use smaller values such as `1e-12` for
    /// high-accuracy tests; use larger values to stop earlier on expensive
    /// functions.
    pub tolerance: f64,
    /// Maximum rank target reached by local sweeps.
    ///
    /// The initial rank is one. Each iteration tries to add one pivot per
    /// bond and then records the new rank. The default is `200`.
    pub max_iter: usize,
    /// Absolute tolerance for accepting one new local pivot.
    ///
    /// The default is `1e-12`. Increase it to skip numerically tiny pivots;
    /// decrease it when reproducing legacy TCI1 rank growth exactly.
    pub pivot_tolerance: f64,
    /// Sweep direction policy.
    ///
    /// The default is [`TCI1SweepStrategy::BackAndForth`]. Use
    /// [`TCI1SweepStrategy::Forward`] or [`TCI1SweepStrategy::Backward`] when
    /// matching a fixed-direction legacy run.
    pub sweep_strategy: TCI1SweepStrategy,
    /// Whether to normalize pivot errors by the maximum sampled value.
    ///
    /// The default is `true`, so [`tolerance`](Self::tolerance) is relative
    /// to the largest absolute sample observed so far. Set to `false` to use
    /// `tolerance` as an absolute threshold.
    pub normalize_error: bool,
    /// Additional global pivots inserted before local sweeps.
    ///
    /// Each pivot must contain one in-range zero-based index per site. The
    /// default is empty. Duplicate pivots are ignored by the index sets.
    pub additional_pivots: Vec<MultiIndex>,
}

impl TCI1Options {
    fn validate(&self) -> Result<()> {
        validate_nonnegative_finite("tolerance", self.tolerance)?;
        validate_nonnegative_finite("pivot_tolerance", self.pivot_tolerance)?;
        validate_positive("max_iter", self.max_iter)
    }
}

impl Default for TCI1Options {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iter: 200,
            pivot_tolerance: 1e-12,
            sweep_strategy: TCI1SweepStrategy::BackAndForth,
            normalize_error: true,
            additional_pivots: Vec::new(),
        }
    }
}

/// Legacy one-site tensor cross interpolation state.
///
/// Use this state when interoperating with workflows that expect the TCI1
/// representation. For repeated evaluation, convert it to
/// [`SimpleTensorTrain`] with [`to_tensor_train`](Self::to_tensor_train).
///
/// Related types: [`TensorCI2`](crate::TensorCI2) is the primary two-site TCI
/// state; [`SimpleTensorTrain`] is the reusable tensor train representation produced
/// by both algorithms.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
///
/// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
/// let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
///     f,
///     vec![4, 4],
///     vec![3, 3],
///     TCI1Options {
///         tolerance: 1e-12,
///         ..TCI1Options::default()
///     },
/// )
/// .unwrap();
///
/// assert_eq!(tci.local_dims(), &[4, 4]);
/// assert_eq!(tci.rank(), 2);
/// assert_eq!(ranks.last().copied(), Some(2));
/// assert!(errors.last().copied().unwrap() < 1e-12);
/// assert!((tci.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone)]
pub struct TensorCI1<T: Scalar + TTScalar> {
    i_set: Vec<IndexSet<MultiIndex>>,
    j_set: Vec<IndexSet<MultiIndex>>,
    local_dims: Vec<usize>,
    t_tensors: Vec<tensor4all_simplett::Tensor3<T>>,
    pivot_matrices: Vec<Matrix<T>>,
    aca: Vec<MatrixACA<T>>,
    pi: Vec<Matrix<T>>,
    pi_i_set: Vec<IndexSet<MultiIndex>>,
    pi_j_set: Vec<IndexSet<MultiIndex>>,
    pivot_errors: Vec<f64>,
    max_sample_value: f64,
}

impl<T> TensorCI1<T>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + MatrixSolveScalar,
{
    /// Construct an empty TCI1 state.
    ///
    /// This creates the state container without running interpolation. Use
    /// [`crossinterpolate1`] to build a state with site tensors.
    ///
    /// # Arguments
    ///
    /// * `local_dims` -- Number of values at each site. Must contain at least
    ///
    ///   two nonzero dimensions.
    ///
    /// # Returns
    ///
    /// An empty `TensorCI1` with `local_dims` recorded and no site tensors.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::DimensionMismatch`] if fewer than two sites are
    /// provided or any local dimension is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let tci = TensorCI1::<f64>::new(vec![2, 3, 4]).unwrap();
    /// assert_eq!(tci.len(), 3);
    /// assert_eq!(tci.local_dims(), &[2, 3, 4]);
    /// assert_eq!(tci.link_dims(), vec![0, 0]);
    /// assert!(tci.to_tensor_train().is_err());
    /// ```
    pub fn new(local_dims: Vec<usize>) -> Result<Self> {
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
        Ok(Self {
            i_set: (0..n).map(|_| IndexSet::new()).collect(),
            j_set: (0..n).map(|_| IndexSet::new()).collect(),
            t_tensors: local_dims
                .iter()
                .map(|&dim| tensor3_zeros(0, dim, 0))
                .collect(),
            pivot_matrices: (0..n).map(|_| Matrix::zeros(0, 0)).collect(),
            aca: (0..n).map(|_| MatrixACA::new(0, 0)).collect(),
            pi: (0..n.saturating_sub(1))
                .map(|_| Matrix::zeros(0, 0))
                .collect(),
            pi_i_set: (0..n).map(|_| IndexSet::new()).collect(),
            pi_j_set: (0..n).map(|_| IndexSet::new()).collect(),
            pivot_errors: vec![f64::INFINITY; n - 1],
            local_dims,
            max_sample_value: 0.0,
        })
    }

    /// Construct the initial rank-1 TCI1 state from a nonzero first pivot.
    ///
    /// # Arguments
    ///
    /// * `f` -- Function to interpolate on zero-based multi-indices.
    /// * `local_dims` -- Number of values at each site. Every dimension must
    ///
    ///   be nonzero and at least two sites are required.
    /// * `first_pivot` -- Initial multi-index. It must be in range and
    ///
    ///   evaluate to a nonzero value.
    ///
    /// # Returns
    ///
    /// A rank-1 `TensorCI1` initialized with the pivot row and column data.
    ///
    /// # Errors
    ///
    /// Returns a [`TCIError`] if dimensions are invalid, the first pivot is
    /// malformed or out of range, the pivot value is zero, or the initial
    /// matrix cross interpolation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let f = |idx: &Vec<usize>| ((idx[0] + 1) * (idx[1] + 2) * (idx[2] + 3)) as f64;
    /// let tci = TensorCI1::<f64>::from_function(&f, vec![3, 3, 3], vec![0, 0, 0]).unwrap();
    ///
    /// assert_eq!(tci.link_dims(), vec![1, 1]);
    /// assert!((tci.evaluate(&[0, 0, 0]).unwrap() - f(&vec![0, 0, 0])).abs() < 1e-12);
    /// ```
    pub fn from_function<F>(f: &F, local_dims: Vec<usize>, first_pivot: MultiIndex) -> Result<Self>
    where
        F: Fn(&MultiIndex) -> T,
    {
        validate_first_pivot(&local_dims, &first_pivot)?;
        let first_value = f(&first_pivot);
        let max_sample_value = Scalar::abs_sq(first_value).sqrt();
        if max_sample_value < 1e-30 {
            return Err(TCIError::InvalidPivot {
                message: "Please provide a first pivot where f(pivot) != 0".to_string(),
            });
        }

        let mut tci = Self::new(local_dims)?;
        tci.max_sample_value = max_sample_value;
        let n = tci.len();

        for site in 0..n {
            tci.i_set[site] = IndexSet::from_vec(vec![first_pivot[..site].to_vec()]);
            tci.j_set[site] = IndexSet::from_vec(vec![first_pivot[site + 1..].to_vec()]);
        }
        for site in 0..n {
            tci.pi_i_set[site] = tci.build_pi_i_set(site);
            tci.pi_j_set[site] = tci.build_pi_j_set(site);
        }
        for bond in 0..n - 1 {
            let pi = tci.build_pi(bond, f);
            tci.pi[bond] = pi;
        }

        for bond in 0..n - 1 {
            let row = tci.position_in_pi_i(bond, &tci.i_set[bond + 1][0])?;
            let col = tci.position_in_pi_j(bond + 1, &tci.j_set[bond][0])?;
            tci.aca[bond] = MatrixACA::from_matrix_with_pivot(&tci.pi[bond], (row, col))?;

            if bond == 0 {
                let first_tensor = matrix_col_as_matrix(&tci.pi[bond], col);
                tci.update_site_tensor_from_matrix(0, &first_tensor)?;
            }
            let next_tensor = matrix_row_as_matrix(&tci.pi[bond], row);
            tci.update_site_tensor_from_matrix(bond + 1, &next_tensor)?;
            tci.pivot_matrices[bond] = single_entry_matrix(tci.pi[bond][[row, col]]);
        }
        tci.pivot_matrices[n - 1] = single_entry_matrix(T::one());

        Ok(tci)
    }

    /// Number of sites.
    pub fn len(&self) -> usize {
        self.local_dims.len()
    }

    /// Whether the state has no sites.
    pub fn is_empty(&self) -> bool {
        self.local_dims.is_empty()
    }

    /// Local dimensions.
    pub fn local_dims(&self) -> &[usize] {
        &self.local_dims
    }

    /// Maximum sampled absolute value.
    pub fn max_sample_value(&self) -> f64 {
        self.max_sample_value
    }

    /// Pivot errors from the latest sweep.
    pub fn pivot_errors(&self) -> &[f64] {
        &self.pivot_errors
    }

    /// Get current rank (maximum bond dimension).
    pub fn rank(&self) -> usize {
        self.link_dims().into_iter().max().unwrap_or(0)
    }

    /// Get bond dimensions between adjacent sites.
    pub fn link_dims(&self) -> Vec<usize> {
        if !self.is_site_tensors_available() {
            vec![0; self.local_dims.len().saturating_sub(1)]
        } else {
            self.t_tensors
                .iter()
                .skip(1)
                .map(|tensor| tensor.left_dim())
                .collect()
        }
    }

    /// Get the left pivot index set at site `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
    /// let tci = TensorCI1::<f64>::from_function(&f, vec![3, 3], vec![1, 1]).unwrap();
    ///
    /// assert_eq!(tci.i_set(0), &[Vec::<usize>::new()]);
    /// assert_eq!(tci.i_set(1), &[vec![1]]);
    /// ```
    pub fn i_set(&self, p: usize) -> &[MultiIndex] {
        self.i_set[p].values()
    }

    /// Get the right pivot index set at site `p`.
    ///
    /// # Panics
    ///
    /// Panics if `p >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
    /// let tci = TensorCI1::<f64>::from_function(&f, vec![3, 3], vec![1, 1]).unwrap();
    ///
    /// assert_eq!(tci.j_set(0), &[vec![1]]);
    /// assert_eq!(tci.j_set(1), &[Vec::<usize>::new()]);
    /// ```
    pub fn j_set(&self, p: usize) -> &[MultiIndex] {
        self.j_set[p].values()
    }

    fn is_site_tensors_available(&self) -> bool {
        self.t_tensors
            .iter()
            .all(|tensor| tensor.left_dim() > 0 && tensor.right_dim() > 0)
    }

    /// Convert the TCI1 state to a tensor train for repeated evaluation.
    ///
    /// # Returns
    ///
    /// A [`SimpleTensorTrain`] with the current TCI1 site tensors.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::InvalidOperation`] if the state was created with
    /// [`TensorCI1::new`] and has not been optimized.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_simplett::AbstractTensorTrain;
    /// use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
    ///
    /// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
    /// let (tci, _ranks, _errors) = crossinterpolate1::<f64, _>(
    ///     f,
    ///     vec![4, 4],
    ///     vec![3, 3],
    ///     TCI1Options::default(),
    /// )
    /// .unwrap();
    /// let tt = tci.to_tensor_train().unwrap();
    ///
    /// assert!((tt.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
    /// ```
    pub fn to_tensor_train(&self) -> Result<SimpleTensorTrain<T>> {
        if !self.is_site_tensors_available() {
            return Err(TCIError::InvalidOperation {
                message: "TensorCI1 has no site tensors; run crossinterpolate1 first".to_string(),
            });
        }

        let mut tensors = Vec::with_capacity(self.len());
        for site in 0..self.len() {
            tensors.push(self.normalized_site_tensor(site)?);
        }
        SimpleTensorTrain::new(tensors).map_err(Into::into)
    }

    /// Evaluate the TCI1 approximation at one multi-index.
    ///
    /// # Arguments
    ///
    /// * `index` -- Zero-based multi-index with one coordinate per site.
    ///
    /// # Returns
    ///
    /// The interpolated scalar value at `index`.
    ///
    /// # Errors
    ///
    /// Returns an error if site tensors are unavailable or if `index` is
    /// invalid for the tensor-train representation.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
    ///
    /// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
    /// let (tci, _ranks, _errors) = crossinterpolate1::<f64, _>(
    ///     f,
    ///     vec![4, 4],
    ///     vec![3, 3],
    ///     TCI1Options::default(),
    /// )
    /// .unwrap();
    ///
    /// assert!((tci.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
    /// ```
    pub fn evaluate(&self, index: &[usize]) -> Result<T> {
        self.to_tensor_train()?.evaluate(index).map_err(Into::into)
    }

    /// Add one local pivot to a bond unless the local error is below `tolerance`.
    ///
    /// # Arguments
    ///
    /// * `bond` -- Zero-based bond index in `0..self.len() - 1`.
    /// * `f` -- Function being interpolated. It must match the function used
    ///
    ///   to initialize the state.
    /// * `tolerance` -- Absolute local pivot error below which no pivot is
    ///
    ///   inserted. Typical values are `1e-8` to `1e-12`.
    ///
    /// # Returns
    ///
    /// `Ok(())` after either inserting one pivot or recording that the local
    /// error is already below `tolerance`.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::InvalidConfiguration`] when `tolerance` is negative
    /// or non-finite. It also returns a [`TCIError`] if `bond` is out of range,
    /// a matrix cross interpolation update fails, or an internal index set is
    /// inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let f = |idx: &Vec<usize>| 1.0 / ((idx[0] + idx[1] + 2) as f64);
    /// let mut tci = TensorCI1::<f64>::from_function(&f, vec![4, 4], vec![0, 0]).unwrap();
    /// let before = tci.rank();
    /// tci.add_pivot(0, &f, 1e-12).unwrap();
    ///
    /// assert!(tci.rank() >= before);
    /// assert!((tci.evaluate(&[2, 3]).unwrap() - f(&vec![2, 3])).abs() < 1e-10);
    /// ```
    pub fn add_pivot<F>(&mut self, bond: usize, f: &F, tolerance: f64) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        validate_nonnegative_finite("tolerance", tolerance)?;
        if bond >= self.len().saturating_sub(1) {
            return Err(TCIError::IndexOutOfBounds {
                message: format!("bond {bond} is outside 0..{}", self.len().saturating_sub(1)),
            });
        }

        if self.aca[bond].rank() >= self.pi[bond].nrows().min(self.pi[bond].ncols()) {
            self.pivot_errors[bond] = 0.0;
            return Ok(());
        }

        let ((new_row, new_col), error) = self.aca[bond].find_new_pivot(&self.pi[bond])?;
        let error = Scalar::abs_sq(error).sqrt();
        self.pivot_errors[bond] = error;
        if error < tolerance {
            return Ok(());
        }

        let mut cross = self.get_cross(bond)?;
        self.add_pivot_col_with_cross(&mut cross, bond, new_col, f)?;
        self.add_pivot_row_with_cross(&mut cross, bond, new_row, f)?;
        Ok(())
    }

    /// Add a full multi-index as pivots across all bonds.
    ///
    /// # Arguments
    ///
    /// * `f` -- Function being interpolated. It must match the function used
    ///
    ///   to initialize the state.
    /// * `pivot` -- Zero-based global multi-index with one coordinate per
    ///
    ///   site.
    /// * `abstol` -- Absolute error threshold. If the current approximation is
    ///
    ///   already within this threshold at `pivot`, no new pivots are inserted.
    ///
    /// # Returns
    ///
    /// `Ok(())` after inserting any missing prefix/suffix pivots needed to
    /// interpolate `pivot`.
    ///
    /// # Errors
    ///
    /// Returns [`TCIError::InvalidConfiguration`] when `abstol` is negative or
    /// non-finite. It also returns a [`TCIError`] if `pivot` is malformed or out
    /// of range, if a pivot update fails, or if index sets become inconsistent.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorci::TensorCI1;
    ///
    /// let f = |idx: &Vec<usize>| ((idx[0] + 1) * (idx[1] + 2) * (idx[2] + 3)) as f64;
    /// let mut tci = TensorCI1::<f64>::from_function(&f, vec![3, 3, 3], vec![0, 0, 0]).unwrap();
    /// let pivot = vec![2, 1, 2];
    /// tci.add_global_pivot(&f, pivot.clone(), 1e-12).unwrap();
    ///
    /// assert!((tci.evaluate(&pivot).unwrap() - f(&pivot)).abs() < 1e-12);
    /// assert!(tci.rank() >= 1);
    /// ```
    pub fn add_global_pivot<F>(&mut self, f: &F, pivot: MultiIndex, abstol: f64) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        validate_nonnegative_finite("abstol", abstol)?;
        validate_first_pivot(&self.local_dims, &pivot)?;
        let exact = f(&pivot);
        let current = self.evaluate(&pivot).unwrap_or_else(|_| T::zero());
        if Scalar::abs_sq(exact - current).sqrt() < abstol {
            return Ok(());
        }

        for bond in 0..self.len() - 1 {
            let target = pivot[..bond + 1].to_vec();
            if !self.i_set[bond + 1].contains(&target) {
                let row = self.position_in_pi_i(bond, &target)?;
                let mut cross = self.get_cross(bond)?;
                cross.add_pivot_row(&self.pi[bond], row)?;
                self.i_set[bond + 1].push(self.pi_i_set[bond][row].clone());
                self.update_site_tensor_from_matrix(bond + 1, cross.pivot_rows())?;
                self.pivot_matrices[bond] = cross.pivot_matrix();
                if bond + 1 < self.len() - 1 {
                    self.refresh_pi_rows(bond + 1, f);
                }
            }
        }

        for bond in (0..self.len() - 1).rev() {
            let target = pivot[bond + 1..].to_vec();
            if !self.j_set[bond].contains(&target) {
                let col = self.position_in_pi_j(bond + 1, &target)?;
                let mut cross = self.get_cross(bond)?;
                cross.add_pivot_col(&self.pi[bond], col)?;
                self.j_set[bond].push(self.pi_j_set[bond + 1][col].clone());
                self.update_site_tensor_from_matrix(bond, cross.pivot_cols())?;
                self.pivot_matrices[bond] = cross.pivot_matrix();
                if bond > 0 {
                    self.refresh_pi_cols(bond - 1, f);
                }
            }
        }

        for bond in 0..self.len() - 1 {
            self.rebuild_aca(bond)?;
        }

        Ok(())
    }

    fn build_pi_i_set(&self, site: usize) -> IndexSet<MultiIndex> {
        let mut values = Vec::with_capacity(self.i_set[site].len() * self.local_dims[site]);
        for i_multi in self.i_set[site].values() {
            for local in 0..self.local_dims[site] {
                let mut value = i_multi.clone();
                value.push(local);
                values.push(value);
            }
        }
        IndexSet::from_vec(values)
    }

    fn build_pi_j_set(&self, site: usize) -> IndexSet<MultiIndex> {
        let mut values = Vec::with_capacity(self.local_dims[site] * self.j_set[site].len());
        for local in 0..self.local_dims[site] {
            for j_multi in self.j_set[site].values() {
                let mut value = vec![local];
                value.extend(j_multi.iter().copied());
                values.push(value);
            }
        }
        IndexSet::from_vec(values)
    }

    fn build_pi<F>(&mut self, bond: usize, f: &F) -> Matrix<T>
    where
        F: Fn(&MultiIndex) -> T,
    {
        let rows = self.pi_i_set[bond].values().to_vec();
        let cols = self.pi_j_set[bond + 1].values().to_vec();
        let mut pi = Matrix::zeros(rows.len(), cols.len());
        for (i, i_multi) in rows.iter().enumerate() {
            for (j, j_multi) in cols.iter().enumerate() {
                let mut index = i_multi.clone();
                index.extend(j_multi.iter().copied());
                let value = f(&index);
                self.max_sample_value = self.max_sample_value.max(Scalar::abs_sq(value).sqrt());
                pi[[i, j]] = value;
            }
        }
        pi
    }

    fn position_in_pi_i(&self, site: usize, index: &MultiIndex) -> Result<usize> {
        self.pi_i_set[site]
            .pos(index)
            .ok_or_else(|| TCIError::IndexInconsistency {
                message: format!("missing I index {index:?} in PiIset[{site}]"),
            })
    }

    fn position_in_pi_j(&self, site: usize, index: &MultiIndex) -> Result<usize> {
        self.pi_j_set[site]
            .pos(index)
            .ok_or_else(|| TCIError::IndexInconsistency {
                message: format!("missing J index {index:?} in PiJset[{site}]"),
            })
    }

    fn update_site_tensor_from_matrix(&mut self, site: usize, matrix: &Matrix<T>) -> Result<()> {
        let left_dim = self.i_set[site].len();
        let site_dim = self.local_dims[site];
        let right_dim = self.j_set[site].len();
        let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);

        if matrix.nrows() == left_dim * site_dim && matrix.ncols() == right_dim {
            for left in 0..left_dim {
                for local in 0..site_dim {
                    for right in 0..right_dim {
                        tensor.set3(left, local, right, matrix[[left * site_dim + local, right]]);
                    }
                }
            }
        } else if matrix.nrows() == left_dim && matrix.ncols() == site_dim * right_dim {
            for left in 0..left_dim {
                for local in 0..site_dim {
                    for right in 0..right_dim {
                        tensor.set3(
                            left,
                            local,
                            right,
                            matrix[[left, local * right_dim + right]],
                        );
                    }
                }
            }
        } else {
            return Err(TCIError::DimensionMismatch {
                message: format!(
                    "cannot reshape {}x{} matrix into TensorCI1 site {} with shape ({left_dim}, {site_dim}, {right_dim})",
                    matrix.nrows(),
                    matrix.ncols(),
                    site
                ),
            });
        }

        self.t_tensors[site] = tensor;
        Ok(())
    }

    fn normalized_site_tensor(&self, site: usize) -> Result<tensor4all_simplett::Tensor3<T>> {
        let tensor = &self.t_tensors[site];
        let (data, rows, cols) = tensor.as_left_matrix();
        let matrix = Matrix::from_col_major_vec(rows, cols, data);
        let normalized = right_solve(&matrix, &self.pivot_matrices[site])?;
        tensor_from_left_matrix(
            tensor.left_dim(),
            tensor.site_dim(),
            tensor.right_dim(),
            &normalized,
        )
    }

    fn get_cross(&self, bond: usize) -> Result<matrix_ci::MatrixCI<T>> {
        let row_indices = positions_in_set(&self.pi_i_set[bond], self.i_set[bond + 1].values())?;
        let col_indices = positions_in_set(&self.pi_j_set[bond + 1], self.j_set[bond].values())?;
        let (left_data, left_rows, left_cols) = self.t_tensors[bond].as_left_matrix();
        let (right_data, right_rows, right_cols) = self.t_tensors[bond + 1].as_right_matrix();
        matrix_ci::MatrixCI::new(
            row_indices,
            col_indices,
            Matrix::from_col_major_vec(left_rows, left_cols, left_data),
            Matrix::from_col_major_vec(right_rows, right_cols, right_data),
        )
    }

    fn add_pivot_col_with_cross<F>(
        &mut self,
        cross: &mut matrix_ci::MatrixCI<T>,
        bond: usize,
        new_col: usize,
        f: &F,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.aca[bond].add_pivot_col(&self.pi[bond], new_col)?;
        cross.add_pivot_col(&self.pi[bond], new_col)?;
        self.j_set[bond].push(self.pi_j_set[bond + 1][new_col].clone());
        self.update_site_tensor_from_matrix(bond, cross.pivot_cols())?;
        self.pivot_matrices[bond] = cross.pivot_matrix();
        if bond > 0 {
            self.update_pi_cols(bond - 1, f)?;
        }
        Ok(())
    }

    fn add_pivot_row_with_cross<F>(
        &mut self,
        cross: &mut matrix_ci::MatrixCI<T>,
        bond: usize,
        new_row: usize,
        f: &F,
    ) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.aca[bond].add_pivot_row(&self.pi[bond], new_row)?;
        cross.add_pivot_row(&self.pi[bond], new_row)?;
        self.i_set[bond + 1].push(self.pi_i_set[bond][new_row].clone());
        self.update_site_tensor_from_matrix(bond + 1, cross.pivot_rows())?;
        self.pivot_matrices[bond] = cross.pivot_matrix();
        if bond + 1 < self.len() - 1 {
            self.update_pi_rows(bond + 1, f)?;
        }
        Ok(())
    }

    fn update_pi_rows<F>(&mut self, bond: usize, f: &F) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.refresh_pi_rows(bond, f);
        self.rebuild_aca(bond)
    }

    fn update_pi_cols<F>(&mut self, bond: usize, f: &F) -> Result<()>
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.refresh_pi_cols(bond, f);
        self.rebuild_aca(bond)
    }

    fn refresh_pi_rows<F>(&mut self, bond: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.pi_i_set[bond] = self.build_pi_i_set(bond);
        self.pi[bond] = self.build_pi(bond, f);
    }

    fn refresh_pi_cols<F>(&mut self, bond: usize, f: &F)
    where
        F: Fn(&MultiIndex) -> T,
    {
        self.pi_j_set[bond + 1] = self.build_pi_j_set(bond + 1);
        self.pi[bond] = self.build_pi(bond, f);
    }

    fn rebuild_aca(&mut self, bond: usize) -> Result<()> {
        let row_indices = positions_in_set(&self.pi_i_set[bond], self.i_set[bond + 1].values())?;
        let col_indices = positions_in_set(&self.pi_j_set[bond + 1], self.j_set[bond].values())?;
        let mut aca = MatrixACA::new(self.pi[bond].nrows(), self.pi[bond].ncols());
        for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
            aca.add_pivot_col(&self.pi[bond], col)?;
            aca.add_pivot_row(&self.pi[bond], row)?;
        }
        self.aca[bond] = aca;
        Ok(())
    }
}

/// Approximate a function with the legacy TCI1 algorithm.
///
/// # Arguments
///
/// * `f` -- Function to interpolate on zero-based multi-indices.
/// * `local_dims` -- Number of values at each site. Must have at least two
///
///   nonzero dimensions.
/// * `first_pivot` -- Initial pivot. It must have one in-range coordinate per
///
///   site and evaluate to a nonzero value.
/// * `options` -- Tolerance, sweep, normalization, and additional pivot
///
///   settings.
///
/// # Returns
///
/// `(tci, ranks, errors)`, where `tci` stores the final TCI1 approximation,
/// `ranks` records the rank history, and `errors` records normalized error
/// estimates.
///
/// # Errors
///
/// Returns [`TCIError::InvalidConfiguration`] for invalid tolerances or a
/// zero iteration limit. It also returns typed errors for invalid dimensions,
/// invalid pivots, interpolation failures, or tensor-train conversion failures.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
///
/// let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
/// let (tci, ranks, errors) = crossinterpolate1::<f64, _>(
///     f,
///     vec![4, 4],
///     vec![3, 3],
///     TCI1Options {
///         tolerance: 1e-12,
///         ..TCI1Options::default()
///     },
/// )
/// .unwrap();
///
/// assert_eq!(ranks.last().copied(), Some(2));
/// assert!(errors.last().copied().unwrap() < 1e-12);
/// assert!((tci.evaluate(&[2, 3]).unwrap() - 6.0).abs() < 1e-10);
/// ```
pub fn crossinterpolate1<T, F>(
    f: F,
    local_dims: Vec<usize>,
    first_pivot: MultiIndex,
    options: TCI1Options,
) -> Result<(TensorCI1<T>, Vec<usize>, Vec<f64>)>
where
    T: Scalar + TTScalar + Default + MatrixLuciScalar + MatrixSolveScalar,
    F: Fn(&MultiIndex) -> T,
{
    options.validate()?;
    let mut tci = TensorCI1::from_function(&f, local_dims, first_pivot)?;
    let mut ranks = Vec::new();
    let mut errors = Vec::new();

    for pivot in options.additional_pivots {
        tci.add_global_pivot(&f, pivot, options.tolerance)?;
    }

    for iter in tci.rank() + 1..=options.max_iter {
        let is_forward = match options.sweep_strategy {
            TCI1SweepStrategy::Forward => true,
            TCI1SweepStrategy::Backward => false,
            TCI1SweepStrategy::BackAndForth => iter % 2 == 1,
        };
        if is_forward {
            for bond in 0..tci.len() - 1 {
                tci.add_pivot(bond, &f, options.pivot_tolerance)?;
            }
        } else {
            for bond in (0..tci.len() - 1).rev() {
                tci.add_pivot(bond, &f, options.pivot_tolerance)?;
            }
        }

        let raw_error = tci.pivot_errors().iter().copied().fold(0.0_f64, f64::max);
        let normalization = if options.normalize_error && tci.max_sample_value() > 0.0 {
            tci.max_sample_value()
        } else {
            1.0
        };
        ranks.push(tci.rank());
        errors.push(raw_error / normalization);
        if raw_error < options.tolerance * normalization {
            break;
        }
    }

    Ok((tci, ranks, errors))
}

#[cfg(test)]
mod tests;

fn validate_first_pivot(local_dims: &[usize], first_pivot: &[usize]) -> Result<()> {
    if local_dims.len() < 2 {
        return Err(TCIError::DimensionMismatch {
            message: "local_dims should have at least 2 elements".to_string(),
        });
    }
    if first_pivot.len() != local_dims.len() {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "first_pivot length ({}) must match local_dims length ({})",
                first_pivot.len(),
                local_dims.len()
            ),
        });
    }
    for (site, (&index, &dim)) in first_pivot.iter().zip(local_dims.iter()).enumerate() {
        if dim == 0 {
            return Err(TCIError::DimensionMismatch {
                message: format!("local_dims[{site}] must be nonzero"),
            });
        }
        if index >= dim {
            return Err(TCIError::IndexOutOfBounds {
                message: format!("first_pivot[{site}] = {index} is outside 0..{dim}"),
            });
        }
    }
    Ok(())
}

fn matrix_col_as_matrix<T>(matrix: &Matrix<T>, col: usize) -> Matrix<T>
where
    T: Scalar,
{
    let mut result = Matrix::zeros(matrix.nrows(), 1);
    for row in 0..matrix.nrows() {
        result[[row, 0]] = matrix[[row, col]];
    }
    result
}

fn matrix_row_as_matrix<T>(matrix: &Matrix<T>, row: usize) -> Matrix<T>
where
    T: Scalar,
{
    let mut result = Matrix::zeros(1, matrix.ncols());
    for col in 0..matrix.ncols() {
        result[[0, col]] = matrix[[row, col]];
    }
    result
}

fn single_entry_matrix<T>(value: T) -> Matrix<T>
where
    T: Scalar,
{
    Matrix::from_col_major_vec(1, 1, vec![value])
}

fn positions_in_set(set: &IndexSet<MultiIndex>, values: &[MultiIndex]) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            set.pos(value).ok_or_else(|| TCIError::IndexInconsistency {
                message: format!("missing index {value:?} in TensorCI1 index set"),
            })
        })
        .collect()
}

fn right_solve<T>(a: &Matrix<T>, p: &Matrix<T>) -> Result<Matrix<T>>
where
    T: Scalar + MatrixSolveScalar,
{
    if p.nrows() != p.ncols() || a.ncols() != p.nrows() {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "right solve shape mismatch: A is {}x{}, P is {}x{}",
                a.nrows(),
                a.ncols(),
                p.nrows(),
                p.ncols()
            ),
        });
    }
    let solution_t =
        solve_matrix(&transpose(p), &transpose(a)).map_err(|err| TCIError::InvalidOperation {
            message: format!("TensorCI1 right solve failed: {err}"),
        })?;
    Ok(transpose(&solution_t))
}

fn tensor_from_left_matrix<T>(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
    matrix: &Matrix<T>,
) -> Result<tensor4all_simplett::Tensor3<T>>
where
    T: Scalar + TTScalar,
{
    if matrix.nrows() != left_dim * site_dim || matrix.ncols() != right_dim {
        return Err(TCIError::DimensionMismatch {
            message: format!(
                "cannot reshape {}x{} matrix into tensor shape ({left_dim}, {site_dim}, {right_dim})",
                matrix.nrows(),
                matrix.ncols()
            ),
        });
    }
    let mut tensor = tensor3_zeros(left_dim, site_dim, right_dim);
    for left in 0..left_dim {
        for local in 0..site_dim {
            for right in 0..right_dim {
                tensor.set3(left, local, right, matrix[[left * site_dim + local, right]]);
            }
        }
    }
    Ok(tensor)
}
