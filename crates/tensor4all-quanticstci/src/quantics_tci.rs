//! QuanticsTensorCI2 and interpolation functions.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::{QuanticsTCIError, Result as QtciResult};
use anyhow::{anyhow, Result};
use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid};
use rand::Rng;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, TTScalar};
use tensor4all_tensorbackend::FullPivLuScalar;
use tensor4all_treetci::materialize::to_treetn;
use tensor4all_treetci::{
    optimize_with_proposer, DefaultProposer, GlobalIndexBatch, TreeTCI2, TreeTciGraph,
};
use tensor4all_treetn::treetn_to_tensor_train as bridge_treetn_to_tensor_train;

use crate::options::QtciOptions;

fn point_from_batch(batch: GlobalIndexBatch<'_>, point: usize) -> Result<Vec<usize>> {
    (0..batch.n_sites())
        .map(|site| {
            batch.get(site, point).ok_or_else(|| {
                anyhow!(
                    "invalid batch index: site {site}, point {point}, batch shape {}x{}",
                    batch.n_sites(),
                    batch.n_points()
                )
            })
        })
        .collect()
}

fn evaluate_grid_point<V>(
    quantics: &[usize],
    to_coord: impl FnOnce(&[usize]) -> Result<Vec<f64>>,
    evaluate: impl FnOnce(&[f64]) -> V,
) -> Result<V> {
    let coords = to_coord(quantics)
        .map_err(|error| anyhow!("failed to convert quantics index {quantics:?}: {error}"))?;
    Ok(evaluate(&coords))
}

/// TCI result wrapped with grid information.
///
/// Combines a [`SimpleTensorTrain`] approximation with grid metadata so you
/// can [`evaluate`](Self::evaluate) at grid indices, compute
/// [`sum`](Self::sum) and [`integral`](Self::integral), and access the
/// underlying [`tensor_train`](Self::tensor_train) for further
/// manipulation.
///
/// Created by [`quanticscrossinterpolate`], [`quanticscrossinterpolate_discrete`],
/// or [`quanticscrossinterpolate_from_arrays`].
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
///
/// // Interpolate f(i) = i on a grid of size 8 (0-indexed)
/// let f = |idx: &[usize]| idx[0] as f64;
/// let (qtci, _ranks, _errors) =
///     quanticscrossinterpolate_discrete::<f64, _>(
///         &[8], f, None, QtciOptions::default(),
///     ).unwrap();
///
/// // Evaluate at grid point 4
/// let val = qtci.evaluate(&[4]).unwrap();
/// assert!((val - 4.0).abs() < 1e-8);
///
/// // Sum over all grid points: 0 + 1 + ... + 7 = 28
/// let sum = qtci.sum().unwrap();
/// assert!((sum - 28.0).abs() < 1e-6);
///
/// // rank() gives the maximum bond dimension
/// assert!(qtci.rank() >= 1);
///
/// // link_dims() gives bond dimensions between sites
/// assert!(!qtci.link_dims().is_empty());
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2<V: TTScalar> {
    /// Underlying tensor train
    tt: SimpleTensorTrain<V>,
    /// TreeTCI2 state (pivot sets, graph, etc.)
    tci_state: TreeTCI2<V>,
    /// Grid for coordinate conversion (DiscretizedGrid)
    discretized_grid: Option<DiscretizedGrid>,
    /// Grid for coordinate conversion (InherentDiscreteGrid)
    inherent_grid: Option<InherentDiscreteGrid>,
    /// Cached function values (quantics index -> value)
    cache: HashMap<Vec<usize>, V>,
}

impl<V> QuanticsTensorCI2<V>
where
    V: TTScalar + Default + Clone,
{
    /// Create a new QuanticsTensorCI2 from a SimpleTensorTrain, TreeTCI2 state, and discretized grid.
    pub fn from_discretized(
        tt: SimpleTensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: DiscretizedGrid,
        cache: HashMap<Vec<usize>, V>,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: Some(grid),
            inherent_grid: None,
            cache,
        }
    }

    /// Create a new QuanticsTensorCI2 from a SimpleTensorTrain, TreeTCI2 state, and inherent discrete grid.
    pub fn from_inherent(
        tt: SimpleTensorTrain<V>,
        tci_state: TreeTCI2<V>,
        grid: InherentDiscreteGrid,
        cache: HashMap<Vec<usize>, V>,
    ) -> Self {
        Self {
            tt,
            tci_state,
            discretized_grid: None,
            inherent_grid: Some(grid),
            cache,
        }
    }

    /// Get the discretized grid (if available).
    pub fn discretized_grid(&self) -> Option<&DiscretizedGrid> {
        self.discretized_grid.as_ref()
    }

    /// Get the inherent discrete grid (if available).
    pub fn inherent_grid(&self) -> Option<&InherentDiscreteGrid> {
        self.inherent_grid.as_ref()
    }

    /// Get the bond dimension (maximum rank).
    pub fn rank(&self) -> usize {
        self.tt.rank()
    }

    /// Get link dimensions.
    pub fn link_dims(&self) -> Vec<usize> {
        self.tt.link_dims()
    }

    /// Convert grid indices to quantics indices.
    fn grididx_to_quantics(&self, indices: &[usize]) -> Result<Vec<usize>> {
        if let Some(grid) = &self.discretized_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else if let Some(grid) = &self.inherent_grid {
            grid.grididx_to_quantics(indices)
                .map_err(|e| anyhow!("Grid index conversion error: {}", e))
        } else {
            Err(anyhow!("No grid available"))
        }
    }

    /// Evaluate at grid indices.
    ///
    /// # Arguments
    /// * `indices` - Grid indices (0-indexed). For a grid of size N,
    ///
    ///   valid indices are `0..N`.
    ///
    /// # Returns
    /// The interpolated value at the specified grid point.
    ///
    /// # Errors
    ///
    /// Returns an error when the grid coordinate conversion fails (an
    /// /// grid shape mismatch failure) or the evaluation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
    ///
    /// let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    ///     &[4, 4], f, None, QtciOptions::default(),
    /// ).unwrap();
    ///
    /// // Indices are 0-indexed: f(1, 2) = 1 + 2 = 3
    /// let val = qtci.evaluate(&[1, 2]).unwrap();
    /// assert!((val - 3.0).abs() < 1e-8);
    /// ```
    pub fn evaluate(&self, indices: &[usize]) -> QtciResult<V> {
        let quantics = self.grididx_to_quantics(indices)?;
        self.tt
            .evaluate(&quantics)
            .map_err(|e| anyhow!("Evaluation error: {e}"))
            .map_err(QuanticsTCIError::from)
    }

    /// Factorized sum over all grid points.
    ///
    /// Computes the sum efficiently using the tensor train structure,
    /// without visiting every grid point individually.
    ///
    /// # Errors
    ///
    /// This method is infallible and always returns `Ok`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
    ///
    /// // f(i) = 1 on a grid of size 8 => sum = 8
    /// let f = |_idx: &[usize]| 1.0_f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    ///     &[8], f, None, QtciOptions::default(),
    /// ).unwrap();
    ///
    /// let sum = qtci.sum().unwrap();
    /// assert!((sum - 8.0).abs() < 1e-8);
    /// ```
    pub fn sum(&self) -> QtciResult<V> {
        Ok(self.tt.sum())
    }

    /// Integral over the continuous domain (left Riemann sum).
    ///
    /// Computes `sum(f(x_i)) * product(step_sizes)`, a left Riemann sum
    /// with O(h) convergence where h is the grid spacing. The result
    /// depends on the `include_endpoint` setting of the [`DiscretizedGrid`].
    ///
    /// For inherent discrete grids (created via
    /// [`quanticscrossinterpolate_discrete`]), there is no continuous
    /// domain, so this returns the plain [`sum`](Self::sum).
    ///
    /// # Errors
    ///
    /// Returns an error when the underlying summation reports a failure (a
    /// [`QuanticsTCIError::Operation`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
    /// };
    ///
    /// // Integrate f(x) = 1 over [0, 1) with 16 points => integral = 1.0
    /// let grid = DiscretizedGrid::builder(&[4])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    /// let f = |_: &[f64]| 1.0_f64;
    /// let (qtci, _, _) = quanticscrossinterpolate::<f64, _>(
    ///     &grid, f, None, QtciOptions::default(),
    /// ).unwrap();
    ///
    /// let integral = qtci.integral().unwrap();
    /// assert!((integral - 1.0).abs() < 1e-8);
    /// ```
    pub fn integral(&self) -> QtciResult<V>
    where
        V: std::ops::Mul<f64, Output = V>,
    {
        let sum_val = self.sum()?;
        if let Some(grid) = &self.discretized_grid {
            let step_product: f64 = grid.grid_step().iter().product();
            Ok(sum_val * step_product)
        } else {
            // For inherent discrete grids, just return the sum
            Ok(sum_val)
        }
    }

    /// Get the underlying [`SimpleTensorTrain`].
    ///
    /// Returns a clone of the tensor train. Use this to pass the result
    /// to other tensor-train operations (contraction, SVD compression, etc.).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_discrete, AbstractTensorTrain, QtciOptions,
    /// };
    ///
    /// let f = |idx: &[usize]| idx[0] as f64;
    /// let (qtci, _, _) = quanticscrossinterpolate_discrete::<f64, _>(
    ///     &[4], f, None, QtciOptions::default(),
    /// ).unwrap();
    ///
    /// let tt = qtci.tensor_train();
    /// assert!(tt.rank() >= 1);
    /// assert!(tt.len() > 0);
    /// ```
    pub fn tensor_train(&self) -> SimpleTensorTrain<V> {
        self.tt.clone()
    }

    /// Access the TreeTCI2 state.
    pub fn tci(&self) -> &TreeTCI2<V> {
        &self.tci_state
    }

    /// Access cached evaluation points.
    ///
    /// Returns a map from quantics indices to function values.
    pub fn cachedata(&self) -> &HashMap<Vec<usize>, V> {
        &self.cache
    }

    /// Access cached evaluation points with original coordinates.
    ///
    /// Only available for discretized grids.
    /// Returns a vector of (coordinates, value) pairs since f64 is not hashable.
    /// # Errors
    ///
    /// Returns an error when the grid is not discretized (a
    /// [`QuanticsTCIError::DiscreteGridRequired`]) or the coordinate conversion
    /// fails (a [`QuanticsTCIError::Operation`]).
    ///
    pub fn cachedata_origcoord(&self) -> std::result::Result<Vec<(Vec<f64>, V)>, QuanticsTCIError>
    where
        V: Clone,
    {
        if let Some(grid) = &self.discretized_grid {
            let mut result = Vec::new();
            for (quantics, value) in &self.cache {
                let coord = grid
                    .quantics_to_origcoord(quantics)
                    .map_err(|e| anyhow!("Coordinate conversion error: {}", e))?;
                #[allow(clippy::clone_on_copy)]
                result.push((coord, value.clone()));
            }
            Ok(result)
        } else {
            Err(QuanticsTCIError::DiscreteGridRequired)
        }
    }
}

/// Interpolate a function with an explicit Grid.
///
/// # Arguments
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Function to interpolate, takes original coordinates
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of (QuanticsTensorCI2, ranks, errors)
///
/// # Errors
///
/// Returns an error when the grid or options are invalid (an
/// /// invalid-configuration failure), an initial pivot conversion fails, or the
/// /// interpolation fails to converge (a non-convergence failure).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate, QtciOptions};
/// use quanticsgrids::DiscretizedGrid;
///
/// // Interpolate f(x) = sin(x) on [0, pi) with 2^4 = 16 points
/// let grid = DiscretizedGrid::builder(&[4])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[std::f64::consts::PI])
///     .build()
///     .unwrap();
///
/// let f = |coords: &[f64]| coords[0].sin();
/// let opts = QtciOptions::default().with_tolerance(1e-8);
/// let (qtci, _ranks, errors) =
///     quanticscrossinterpolate::<f64, _>(&grid, f, None, opts).unwrap();
///
/// // Last error should be within tolerance
/// assert!(*errors.last().unwrap() < 1e-6);
///
/// // Sum (integral * step) approximates the Riemann sum of sin(x)
/// let sum = qtci.sum().unwrap();
/// assert!(sum > 0.0); // sin(x) > 0 on (0, pi)
/// ```
pub fn quanticscrossinterpolate<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[f64]) -> V + 'static,
{
    let local_dims = grid.local_dimensions();
    let n_sites = local_dims.len();

    // Use RefCell to allow mutation from within the closure
    let cache: Rc<RefCell<HashMap<Vec<usize>, V>>> = Rc::new(RefCell::new(HashMap::new()));
    let cache_clone = cache.clone();

    // Wrap function to accept quantics indices (usize 0-indexed for TCI)
    let grid_clone = grid.clone();
    let qf = move |q: &Vec<usize>| -> Result<V> {
        // Check cache first
        if let Some(v) = cache_clone.borrow().get(q) {
            #[allow(clippy::clone_on_copy)]
            return Ok(v.clone());
        }

        // Compute and cache
        let value = evaluate_grid_point(
            q,
            |quantics| {
                grid_clone
                    .quantics_to_origcoord(quantics)
                    .map_err(|error| anyhow!("{error}"))
            },
            |coords| f(coords),
        )?;
        #[allow(clippy::clone_on_copy)]
        cache_clone.borrow_mut().insert(q.clone(), value.clone());
        Ok(value)
    };

    // Batch adapter: treetci expects Fn(GlobalIndexBatch) -> Result<Vec<V>>
    let batch_eval = move |batch: GlobalIndexBatch<'_>| -> Result<Vec<V>> {
        let n_points = batch.n_points();
        let mut results = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let point = point_from_batch(batch, p)?;
            results.push(qf(&point)?);
        }
        Ok(results)
    };

    // Prepare initial pivots
    let mut qinitialpivots: Vec<Vec<usize>> = if let Some(pivots) = initial_pivots {
        pivots
            .iter()
            .map(|pivot| {
                grid.grididx_to_quantics(pivot)
                    .map_err(|error| anyhow!("initial pivot {pivot:?} conversion failed: {error}"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        // Default to first grid point (0-indexed for TCI)
        vec![vec![0; n_sites]]
    };

    // Add random initial pivots (0-indexed for TCI)
    let mut rng = rand::rng();
    for _ in 0..options.n_random_init_pivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        qinitialpivots.push(pivot);
    }

    // Run TreeTCI with linear chain (lower-level API)
    let graph = TreeTciGraph::linear_chain(n_sites)?;
    let tree_opts = options.to_treetci_options();
    let proposer = DefaultProposer;

    let pivots = if qinitialpivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        qinitialpivots
    };

    let mut tci = TreeTCI2::<V>::new(local_dims.clone(), graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let init_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = batch_eval(init_batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| <V as tensor4all_core::Scalar>::abs_val(*v))
        .fold(0.0f64, f64::max);
    if tci.max_sample_value <= 0.0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "initial pivots must not all evaluate to zero".to_string(),
        });
    }

    let (ranks, errors) = optimize_with_proposer(&mut tci, &batch_eval, &tree_opts, &proposer)?;
    let treetn = to_treetn(&tci, &batch_eval, Some(0))?;

    // Convert TreeTN → SimpleTensorTrain<V> via the sanctioned bridge
    let tt: SimpleTensorTrain<V> =
        bridge_treetn_to_tensor_train(treetn).map_err(|error| QuanticsTCIError::Operation {
            source: anyhow::Error::new(error)
                .context("TreeTN to SimpleTensorTrain conversion failed"),
        })?;

    // Drop batch_eval (and its captured Rc clone) before extracting the cache
    drop(batch_eval);

    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_discretized(tt, tci, grid.clone(), final_cache),
        ranks,
        errors,
    ))
}

/// Interpolate from explicit grid point arrays.
///
/// Convenience wrapper around [`quanticscrossinterpolate_discrete`] that
/// evaluates `f` at the exact coordinates supplied in `xvals`.
///
/// # Arguments
/// * `xvals` - Strictly increasing, finite coordinate arrays. All dimensions must have
///   the **same** number of points and each must be a power of 2.
/// * `f` - Function to interpolate, takes original coordinates as `&[f64]`
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of ([`QuanticsTensorCI2`], ranks per sweep, errors per sweep)
///
/// # Errors
///
/// Returns an error when the grid or options are invalid (an
/// /// invalid-configuration failure), an initial pivot conversion fails, or the
/// /// interpolation fails to converge (a non-convergence failure).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate_from_arrays, QtciOptions};
///
/// // 4 points in [0, 3]
/// let xvals = vec![vec![0.0, 1.0, 2.0, 3.0]];
/// let f = |coords: &[f64]| coords[0] * coords[0]; // f(x) = x^2
/// let (qtci, _, _) = quanticscrossinterpolate_from_arrays::<f64, _>(
///     &xvals, f, None, QtciOptions::default(),
/// ).unwrap();
///
/// // Grid index 2 maps to x = 2.0, so f = 4.0
/// let val = qtci.evaluate(&[2]).unwrap();
/// assert!((val - 4.0).abs() < 1e-8);
/// ```
pub fn quanticscrossinterpolate_from_arrays<V, F>(
    xvals: &[Vec<f64>],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[f64]) -> V + 'static,
{
    if xvals.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "xvals must not be empty".to_string(),
        });
    }
    if xvals.iter().any(|x| x.is_empty()) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "xvals must not contain empty dimensions".to_string(),
        });
    }

    for (dimension, values) in xvals.iter().enumerate() {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: format!("xvals[{dimension}] must contain only finite values"),
            });
        }
        if values.windows(2).any(|window| window[0] >= window[1]) {
            return Err(QuanticsTCIError::InvalidConfiguration {
                message: format!(
                    "xvals[{dimension}] must be strictly increasing without duplicates"
                ),
            });
        }
    }

    let sizes = xvals.iter().map(Vec::len).collect::<Vec<_>>();
    let dimensions: Vec<f64> = sizes.iter().map(|&size| (size as f64).log2()).collect();
    if !dimensions
        .windows(2)
        .all(|window| (window[0] - window[1]).abs() < 1e-10)
    {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message:
                "this method only supports grids with equal number of points in each direction"
                    .to_string(),
        });
    }
    if !dimensions
        .iter()
        .all(|&dimension| (dimension - dimension.round()).abs() < 1e-10)
    {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method only supports grid sizes that are powers of 2".to_string(),
        });
    }

    let is_uniform = xvals.iter().all(|values| {
        let Some(first_window) = values.windows(2).next() else {
            return true;
        };
        let step = first_window[1] - first_window[0];
        values
            .windows(2)
            .all(|window| (window[1] - window[0] - step).abs() <= 1e-12)
    });
    if is_uniform {
        let rs = dimensions
            .iter()
            .map(|&dimension| dimension as usize)
            .collect::<Vec<_>>();
        let lower = xvals
            .iter()
            .map(|values| {
                values
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("xvals must not be empty"))
            })
            .collect::<Result<Vec<_>>>()?;
        let upper = xvals
            .iter()
            .map(|values| {
                values
                    .last()
                    .copied()
                    .ok_or_else(|| anyhow!("xvals must not be empty"))
            })
            .collect::<Result<Vec<_>>>()?;
        let grid = DiscretizedGrid::builder(&rs)
            .with_lower_bound(&lower)
            .with_upper_bound(&upper)
            .with_unfolding_scheme(options.unfolding_scheme)
            .include_endpoint(true)
            .build()
            .map_err(|error| anyhow!("Failed to build grid: {error}"))?;
        return quanticscrossinterpolate(&grid, f, initial_pivots, options);
    }

    let coordinates = xvals.to_vec();
    let mapped_f = move |indices: &[usize]| -> V {
        let point = indices
            .iter()
            .enumerate()
            .map(|(dimension, &index)| coordinates[dimension][index])
            .collect::<Vec<_>>();
        f(&point)
    };

    quanticscrossinterpolate_discrete(&sizes, mapped_f, initial_pivots, options)
}

/// Interpolate a function defined on a discrete integer grid.
///
/// Use this when your function is naturally indexed by integers (e.g.,
/// lattice models, combinatorial functions). Grid indices are
/// **0-indexed**: the first grid point is `[0, 0, ...]`, and the last
/// is `[size[0] - 1, size[1] - 1, ...]`.
///
/// For functions on continuous domains, use [`quanticscrossinterpolate`]
/// with a [`DiscretizedGrid`] instead.
///
/// # Arguments
/// * `size` - Grid size in each dimension. All dimensions must have the **same** number of
///
///   points and each must be a power of 2 (e.g., `&[16, 16]`).
/// * `f` - Function to interpolate, taking **0-indexed** grid indices as `&[usize]`
/// * `initial_pivots` - Initial pivot grid indices (0-indexed, optional)
/// * `options` - TCI options
///
/// # Returns
/// Tuple of ([`QuanticsTensorCI2`], ranks per sweep, errors per sweep)
///
/// # Errors
///
/// Returns an error when the grid or options are invalid (an
/// /// invalid-configuration failure), an initial pivot conversion fails, or the
/// /// interpolation fails to converge (a non-convergence failure).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
///
/// // Interpolate f(i, j) = i * j on a 16x16 grid
/// let f = |idx: &[usize]| (idx[0] * idx[1]) as f64;
/// let (qtci, ranks, errors) = quanticscrossinterpolate_discrete::<f64, _>(
///     &[16, 16],
///     f,
///     None,
///     QtciOptions::default().with_tolerance(1e-10),
/// ).unwrap();
///
/// // Check convergence
/// assert!(*errors.last().unwrap() < 1e-8);
///
/// // Evaluate: f(2, 4) = 8
/// let val = qtci.evaluate(&[2, 4]).unwrap();
/// assert!((val - 8.0).abs() < 1e-8);
/// ```
pub fn quanticscrossinterpolate_discrete<V, F>(
    size: &[usize],
    f: F,
    initial_pivots: Option<Vec<Vec<usize>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2<V>, Vec<usize>, Vec<f64>)>
where
    V: TTScalar
        + Default
        + Clone
        + 'static
        + tensor4all_core::TensorElement
        + tensor4all_core::MatrixLuciScalar
        + FullPivLuScalar
        + tensor4all_treetci::globalpivot::ScalarParts,
    F: Fn(&[usize]) -> V + 'static,
{
    if size.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method requires at least one grid dimension, got an empty size"
                .to_string(),
        });
    }
    // Validate sizes are powers of 2
    let dimensions: Vec<f64> = size.iter().map(|&s| (s as f64).log2()).collect();

    if !dimensions.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message:
                "this method only supports grids with equal number of points in each direction"
                    .to_string(),
        });
    }

    if !dimensions.iter().all(|&d| (d - d.round()).abs() < 1e-10) {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "this method only supports grid sizes that are powers of 2".to_string(),
        });
    }

    let r = dimensions[0] as usize;
    let n = size.len();

    // Build inherent discrete grid - rs is the number of bits per variable
    let rs: Vec<usize> = vec![r; n];
    let grid = InherentDiscreteGrid::builder(&rs)
        .with_unfolding_scheme(options.unfolding_scheme)
        .build()
        .map_err(|e| anyhow!("Failed to build grid: {}", e))?;

    let local_dims = grid.local_dimensions();
    let n_sites = local_dims.len();

    // Use RefCell to allow mutation from within the closure
    let cache: Rc<RefCell<HashMap<Vec<usize>, V>>> = Rc::new(RefCell::new(HashMap::new()));
    let cache_clone = cache.clone();

    // Wrap function to accept quantics indices (usize 0-indexed for TCI)
    let grid_clone = grid.clone();
    let qf = move |q: &[usize]| -> Result<V> {
        // Check cache first
        if let Some(v) = cache_clone.borrow().get(q) {
            #[allow(clippy::clone_on_copy)]
            return Ok(v.clone());
        }

        // Compute and cache
        let grididx = grid_clone
            .quantics_to_grididx(q)
            .map_err(|err| anyhow!("failed to convert quantics index {q:?}: {err}"))?;
        let value = f(&grididx);
        #[allow(clippy::clone_on_copy)]
        cache_clone.borrow_mut().insert(q.to_vec(), value.clone());
        Ok(value)
    };

    // Batch adapter: treetci expects Fn(GlobalIndexBatch) -> Result<Vec<V>>
    let batch_eval = move |batch: GlobalIndexBatch<'_>| -> Result<Vec<V>> {
        let n_points = batch.n_points();
        let mut results = Vec::with_capacity(n_points);
        for p in 0..n_points {
            let point = point_from_batch(batch, p)?;
            results.push(qf(&point)?);
        }
        Ok(results)
    };

    // Prepare initial pivots
    let mut qinitialpivots: Vec<Vec<usize>> = if let Some(pivots) = initial_pivots {
        pivots
            .iter()
            .map(|pivot| {
                grid.grididx_to_quantics(pivot)
                    .map_err(|error| anyhow!("initial pivot {pivot:?} conversion failed: {error}"))
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        // Default to first grid point (0-indexed for TCI)
        vec![vec![0; local_dims.len()]]
    };

    // Add random initial pivots (0-indexed for TCI)
    let mut rng = rand::rng();
    for _ in 0..options.n_random_init_pivot {
        let pivot: Vec<usize> = local_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        qinitialpivots.push(pivot);
    }

    // Run TreeTCI with linear chain (lower-level API)
    let graph = TreeTciGraph::linear_chain(n_sites)?;
    let tree_opts = options.to_treetci_options();
    let proposer = DefaultProposer;

    let pivots = if qinitialpivots.is_empty() {
        vec![vec![0; local_dims.len()]]
    } else {
        qinitialpivots
    };

    let mut tci = TreeTCI2::<V>::new(local_dims.clone(), graph)?;
    tci.add_global_pivots(&pivots)?;

    // Initialize max_sample_value via batch evaluate
    let flat: Vec<usize> = pivots.iter().flat_map(|p| p.iter().copied()).collect();
    let init_batch = GlobalIndexBatch::new(&flat, n_sites, pivots.len())?;
    let init_vals = batch_eval(init_batch)?;
    tci.max_sample_value = init_vals
        .iter()
        .map(|v| <V as tensor4all_core::Scalar>::abs_val(*v))
        .fold(0.0f64, f64::max);
    if tci.max_sample_value <= 0.0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "initial pivots must not all evaluate to zero".to_string(),
        });
    }

    let (ranks, errors) = optimize_with_proposer(&mut tci, &batch_eval, &tree_opts, &proposer)?;
    let treetn = to_treetn(&tci, &batch_eval, Some(0))?;

    // Convert TreeTN → SimpleTensorTrain<V> via the sanctioned bridge
    let tt: SimpleTensorTrain<V> =
        bridge_treetn_to_tensor_train(treetn).map_err(|error| QuanticsTCIError::Operation {
            source: anyhow::Error::new(error)
                .context("TreeTN to SimpleTensorTrain conversion failed"),
        })?;

    // Drop batch_eval (and its captured Rc clone) before extracting the cache
    drop(batch_eval);

    let final_cache = Rc::try_unwrap(cache)
        .map_err(|_| anyhow!("Failed to extract cache"))?
        .into_inner();

    Ok((
        QuanticsTensorCI2::from_inherent(tt, tci, grid, final_cache),
        ranks,
        errors,
    ))
}

#[cfg(test)]
mod tests;
