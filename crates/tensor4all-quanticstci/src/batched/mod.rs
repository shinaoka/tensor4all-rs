//! Batched (vector/tensor-valued) Quantics TCI interpolation.
//!
//! This module provides [`quanticscrossinterpolate_batched`], which interpolates
//! vector- or tensor-valued functions. Each output component is interpolated
//! independently using scalar TCI, and the results are combined into a single
//! [`TensorTrain`] with an additional component site.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::error::{QuanticsTCIError, Result as QtciResult};
use anyhow::{anyhow, Result};
use quanticsgrids::DiscretizedGrid;
use tensor4all_core::TensorElement;
use tensor4all_simplett::{AbstractTensorTrain, TTScalar, TensorTrain};
use tensor4all_simplett::{Tensor3, Tensor3Ops};
use tensor4all_tensorbackend::FullPivLuScalar;

use crate::options::QtciOptions;
use crate::quantics_tci::quanticscrossinterpolate;

/// Result of batched (vector/tensor-valued) Quantics TCI interpolation.
///
/// Wraps a [`TensorTrain`] where the last site is a component index,
/// plus the output shape and grid information.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_batched, AbstractTensorTrain, DiscretizedGrid, QtciOptions,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
///     &grid,
///     |x: &[f64]| vec![x[0] + 1.0, 2.0 * x[0] + 1.0],
///     &[2],
///     None,
///     QtciOptions::default(),
/// ).unwrap();
///
/// assert_eq!(result.output_dims(), &[2]);
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// ```
#[derive(Clone)]
pub struct QuanticsTensorCI2Batched<V: TTScalar> {
    /// Combined tensor train with component index as the last site.
    tt: TensorTrain<V>,
    /// Shape of the output (e.g., [3] for 3-vector, [2, 2] for 2x2 matrix).
    output_dims: Vec<usize>,
    /// Grid for coordinate conversion.
    grid: DiscretizedGrid,
}

impl<V> QuanticsTensorCI2Batched<V>
where
    V: TTScalar + Default + Clone,
{
    /// Get the combined tensor train.
    ///
    /// The last site of the tensor train is the component index with
    /// dimension equal to the product of `output_dims`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_batched, AbstractTensorTrain, DiscretizedGrid, QtciOptions,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
    ///     &grid,
    ///     |x: &[f64]| vec![x[0] + 1.0, x[0] * x[0] + 1.0],
    ///     &[2],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// let tt = result.tensor_train();
    /// assert_eq!(tt.len(), 3); // 2 grid sites + 1 component site
    /// ```
    pub fn tensor_train(&self) -> &TensorTrain<V> {
        &self.tt
    }

    /// Get the output dimensions (shape of the function output).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_batched, QtciOptions, DiscretizedGrid,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
    ///     &grid,
    ///     |x: &[f64]| vec![x[0] + 1.0, x[0] * x[0] + 1.0],
    ///     &[2],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// assert_eq!(result.output_dims(), &[2]);
    /// ```
    pub fn output_dims(&self) -> &[usize] {
        &self.output_dims
    }

    /// Get the discretized grid.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_quanticstci::{
    ///     quanticscrossinterpolate_batched, QtciOptions, DiscretizedGrid,
    /// };
    ///
    /// let grid = DiscretizedGrid::builder(&[2])
    ///     .with_lower_bound(&[0.0])
    ///     .with_upper_bound(&[1.0])
    ///     .build()
    ///     .unwrap();
    ///
    /// let (result, _, _) = quanticscrossinterpolate_batched::<f64, _>(
    ///     &grid,
    ///     |x: &[f64]| vec![x[0] + 1.0],
    ///     &[1],
    ///     None,
    ///     QtciOptions::default(),
    /// ).unwrap();
    ///
    /// assert!(result.grid().grid_step().len() > 0);
    /// ```
    pub fn grid(&self) -> &DiscretizedGrid {
        &self.grid
    }
}

/// Interpolate a vector/tensor-valued function using batched Quantics TCI.
///
/// Each output component is interpolated independently using scalar TCI via
/// [`quanticscrossinterpolate`]. Function evaluations are cached so each grid
/// point is evaluated at most once across all components. The per-component
/// tensor trains are then combined into a single [`TensorTrain`] with an
/// additional component site at the end.
///
/// # Arguments
///
/// * `grid` - Discretized grid describing the function domain
/// * `f` - Function to interpolate, returns a `Vec<V>` of length `product(output_dims)`
/// * `output_dims` - Shape of the function output (e.g., `&[3]` for 3-vector,
///
///   `&[2, 2]` for 2x2 matrix)
/// * `initial_pivots` - Initial pivot grid indices (optional)
/// * `options` - TCI options
///
/// # Returns
///
/// Tuple of ([`QuanticsTensorCI2Batched`], max_ranks_across_components, max_errors_across_components)
///
/// # Errors
///
/// Returns an error when the grid, output dimensions, or options are invalid (a
/// [`QuanticsTCIError::InvalidConfiguration`]), an initial pivot conversion
/// fails, or the underlying TCI construction fails (a
/// [`QuanticsTCIError::Operation`]).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstci::{
///     quanticscrossinterpolate_batched, AbstractTensorTrain, DiscretizedGrid, QtciOptions,
/// };
///
/// let grid = DiscretizedGrid::builder(&[2])
///     .with_lower_bound(&[0.0])
///     .with_upper_bound(&[1.0])
///     .build()
///     .unwrap();
///
/// let (result, ranks, errors) = quanticscrossinterpolate_batched::<f64, _>(
///     &grid,
///     |x: &[f64]| vec![
///         x[0] + 1.0,
///         2.0 * x[0] + 1.0,
///     ],
///     &[2],
///     None,
///     QtciOptions::default().with_tolerance(1e-8),
/// ).unwrap();
///
/// assert_eq!(result.tensor_train().len(), 3); // 2 grid sites + 1 component site
/// assert_eq!(result.output_dims(), &[2]);
/// assert!(!ranks.is_empty());
/// assert!(!errors.is_empty());
/// ```
pub fn quanticscrossinterpolate_batched<V, F>(
    grid: &DiscretizedGrid,
    f: F,
    output_dims: &[usize],
    initial_pivots: Option<Vec<Vec<i64>>>,
    options: QtciOptions,
) -> QtciResult<(QuanticsTensorCI2Batched<V>, Vec<usize>, Vec<f64>)>
where
    F: Fn(&[f64]) -> Vec<V> + 'static,
    V: TTScalar
        + Default
        + Clone
        + 'static
        + TensorElement
        + tensor4all_tcicore::MatrixLuciScalar
        + FullPivLuScalar,
{
    // Validate output_dims
    if output_dims.is_empty() {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: "output_dims must not be empty".to_string(),
        });
    }
    let n_components: usize = output_dims.iter().product();
    if n_components == 0 {
        return Err(QuanticsTCIError::InvalidConfiguration {
            message: format!("product of output_dims must be positive, got 0 from {output_dims:?}"),
        });
    }

    // Shared cache: maps coordinate bits to function output vector.
    // We use f64::to_bits() for exact hashing of floating-point coordinates.
    let cache: Arc<Mutex<HashMap<Vec<u64>, Vec<V>>>> = Arc::new(Mutex::new(HashMap::new()));

    // Wrap f in Arc so it can be shared across component closures.
    type BatchFn<V> = dyn Fn(&[f64]) -> Vec<V>;
    let f_arc: Arc<BatchFn<V>> = Arc::from(f);

    // Run scalar TCI for each component independently.
    let mut component_tts: Vec<TensorTrain<V>> = Vec::with_capacity(n_components);
    let mut all_ranks: Vec<Vec<usize>> = Vec::with_capacity(n_components);
    let mut all_errors: Vec<Vec<f64>> = Vec::with_capacity(n_components);

    for comp in 0..n_components {
        let cache_clone = cache.clone();
        let f_clone = f_arc.clone();

        // Create a scalar wrapper for this component.
        let scalar_f = move |coords: &[f64]| -> V {
            let key: Vec<u64> = coords.iter().map(|c| c.to_bits()).collect();

            // Check cache first.
            {
                let guard = cache_clone.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(values) = guard.get(&key) {
                    return values[comp];
                }
            }

            // Evaluate and cache.
            let values = f_clone(coords);
            let result = values[comp];
            {
                let mut guard = cache_clone.lock().unwrap_or_else(|e| e.into_inner());
                guard.insert(key, values);
            }
            result
        };

        let (qtci, ranks, errors) =
            quanticscrossinterpolate(grid, scalar_f, initial_pivots.clone(), options.clone())?;

        component_tts.push(qtci.tensor_train());
        all_ranks.push(ranks);
        all_errors.push(errors);
    }

    // Combine component TTs into a single TT with a component selector site.
    let combined_tt = combine_component_tts(&component_tts)?;

    // Compute aggregate ranks and errors (element-wise max across components).
    let max_len = all_ranks.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut max_ranks = vec![0usize; max_len];
    for ranks in &all_ranks {
        for (i, &r) in ranks.iter().enumerate() {
            max_ranks[i] = max_ranks[i].max(r);
        }
    }

    let max_err_len = all_errors.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut max_errors = vec![0.0f64; max_err_len];
    for errors in &all_errors {
        for (i, &e) in errors.iter().enumerate() {
            max_errors[i] = max_errors[i].max(e);
        }
    }

    let result = QuanticsTensorCI2Batched {
        tt: combined_tt,
        output_dims: output_dims.to_vec(),
        grid: grid.clone(),
    };

    Ok((result, max_ranks, max_errors))
}

/// Combine per-component tensor trains into a single TT with a component
/// selector as the final site.
///
/// The combination strategy ensures TensorTrain validity:
/// - **First site** (`left_dim = 1` for all components): the component tensors
///
///   are concatenated along the right bond, giving shape `(1, site_dim, sum_right)`.
/// - **Middle sites**: block-diagonal in both bond dimensions.
/// - **Last grid site** (`right_dim = 1` for all components): block-diagonal in
///
///   left bond, concatenated right bond gives `(sum_left, site_dim, n_components)`.
///
/// A final selector site of shape `(total_right_bond, n_components, 1)` is
/// appended so that fixing the component index selects the corresponding
/// component's value.
fn combine_component_tts<V>(component_tts: &[TensorTrain<V>]) -> Result<TensorTrain<V>>
where
    V: TTScalar + Default + Clone,
{
    let n_components = component_tts.len();
    if n_components == 0 {
        return Err(anyhow!("no component tensor trains to combine"));
    }

    let n_sites = component_tts[0].len();
    if n_sites == 0 {
        return Err(anyhow!(
            "component tensor trains must have at least one site"
        ));
    }

    // Verify all components have the same number of sites and site dimensions.
    for (c, tt) in component_tts.iter().enumerate() {
        if tt.len() != n_sites {
            return Err(anyhow!(
                "component {} has {} sites, expected {}",
                c,
                tt.len(),
                n_sites
            ));
        }
        for s in 0..n_sites {
            if tt.site_tensor(s).site_dim() != component_tts[0].site_tensor(s).site_dim() {
                return Err(anyhow!(
                    "component {} site {} has site_dim {}, expected {}",
                    c,
                    s,
                    tt.site_tensor(s).site_dim(),
                    component_tts[0].site_tensor(s).site_dim()
                ));
            }
        }
    }

    let mut combined_tensors: Vec<Tensor3<V>> = Vec::with_capacity(n_sites + 1);

    for s in 0..n_sites {
        let site_dim = component_tts[0].site_tensor(s).site_dim();

        let total_right: usize = component_tts
            .iter()
            .map(|tt| tt.site_tensor(s).right_dim())
            .sum();

        if s == 0 {
            // First site: all components have left_dim = 1.
            // Concatenate along right bond: shape (1, site_dim, total_right).
            let mut combined = tensor3_zeros_generic::<V>(1, site_dim, total_right);

            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let rd = t.right_dim();

                for sd in 0..site_dim {
                    for r in 0..rd {
                        combined.set3(0, sd, right_offset + r, *t.get3(0, sd, r));
                    }
                }

                right_offset += rd;
            }

            combined_tensors.push(combined);
        } else {
            // Middle and last grid sites: block-diagonal in both bond dims.
            let total_left: usize = component_tts
                .iter()
                .map(|tt| tt.site_tensor(s).left_dim())
                .sum();

            let mut combined = tensor3_zeros_generic::<V>(total_left, site_dim, total_right);

            let mut left_offset = 0;
            let mut right_offset = 0;
            for tt in component_tts.iter() {
                let t = tt.site_tensor(s);
                let ld = t.left_dim();
                let rd = t.right_dim();

                for l in 0..ld {
                    for sd in 0..site_dim {
                        for r in 0..rd {
                            combined.set3(left_offset + l, sd, right_offset + r, *t.get3(l, sd, r));
                        }
                    }
                }

                left_offset += ld;
                right_offset += rd;
            }

            combined_tensors.push(combined);
        }
    }

    // Build the component selector site.
    // Shape: (total_right_of_last_grid_site, n_components, 1)
    let total_right: usize = component_tts
        .iter()
        .map(|tt| tt.site_tensor(n_sites - 1).right_dim())
        .sum();

    let mut selector = tensor3_zeros_generic::<V>(total_right, n_components, 1);

    let mut offset = 0;
    for (c, tt) in component_tts.iter().enumerate() {
        let rd = tt.site_tensor(n_sites - 1).right_dim();
        for i in 0..rd {
            selector.set3(offset + i, c, 0, V::one());
        }
        offset += rd;
    }

    combined_tensors.push(selector);

    TensorTrain::new(combined_tensors).map_err(|e| anyhow!("Failed to build combined TT: {}", e))
}

/// Create a zero-filled Tensor3 using TTScalar bounds (avoids importing tensor3_zeros
/// which may have different trait bounds).
fn tensor3_zeros_generic<V: TTScalar + Default + Clone>(
    left: usize,
    site: usize,
    right: usize,
) -> Tensor3<V> {
    use tensor4all_simplett::tensor3_zeros;
    tensor3_zeros(left, site, right)
}

#[cfg(test)]
mod tests;
