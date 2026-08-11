//! Public elementwise Alternating Cross Interpolation sweep APIs.

use crate::scalar::AciScalar;
use crate::validation::{validate_inputs, validate_options};
use crate::{AciOptions, AciResult, ElementwiseBatch, ElementwiseProblem, Result};
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, SimpleTensorTrain};

/// Runs batched elementwise ACI over tensor-train inputs.
///
/// This function approximates the pointwise application of `op` to `inputs`.
/// The callback receives batches in column-major input/point layout through
/// [`ElementwiseBatch`] and writes one output value per interpolation point.
/// Forward and backward sweeps alternate until the configured iteration limit
/// is reached or the conservative convergence rule accepts the current ranks
/// and maximum error metric. When [`AciOptions::scale_tolerance`] is enabled,
/// each bond's pivot error is divided by that bond's largest sampled
/// operator-output magnitude from the completed sweep, and the largest
/// normalized value is used.
///
/// Sweeping also stops once the solution rank has sat at
/// [`AciOptions::max_bond_dim`] for [`AciOptions::min_iters`] consecutive
/// sweeps. At the cap the sweep can no longer add pivots, so the tolerance can
/// never be met and further sweeps only repeat full-rank work. A run that stops
/// this way returns normally with a final entry of [`AciResult::errors`] above
/// [`AciOptions::tolerance`]: under a binding `max_bond_dim` that outcome is by
/// design, not a failure. Compare the last [`AciResult::ranks`] entry against
/// `max_bond_dim` to tell a rank-limited run from a converged one.
///
/// For single-site tensor trains there are no bonds to sweep. In that case the
/// operator is evaluated once over all site points, and the returned
/// [`AciResult`] contains empty `ranks` and `errors` histories.
///
/// # Arguments
///
/// * `op` - Batched callback. For each point, write the elementwise output to
///
///   the corresponding entry of the output slice. Return
///   [`AciError::Operator`](crate::AciError::Operator) or another ACI error to
///   stop the sweep.
/// * `inputs` - Non-empty tensor trains with the same length and site
///
///   dimensions. Core dimensions must be positive.
/// * `options` - Sweep limits, tolerance, maximum bond dimension, random seed,
///
///   and optional initial guess. When in doubt, start with
///   [`AciOptions::default`].
///
/// # Returns
///
/// Returns an [`AciResult`] containing the approximating tensor train plus one
/// rank and error sample after each completed sweep.
///
/// # Errors
///
/// Returns [`AciError`](crate::AciError) when options or inputs are invalid,
/// initial guess construction fails, the operator callback fails, or a local
/// matrix-CI update fails.
///
/// # Panics
///
/// This function does not intentionally panic. A panic from the caller-provided
/// `op` callback is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::{elementwise_batched, AciOptions, ElementwiseBatch};
/// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
///
/// let a = SimpleTensorTrain::<f64>::constant(&[2, 2], 2.0);
/// let b = SimpleTensorTrain::<f64>::constant(&[2, 2], 3.0);
/// let result = elementwise_batched(
///     |batch: ElementwiseBatch<'_, f64>, output: &mut [f64]| {
///         for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
///             *value = batch.get(0, point)? * batch.get(1, point)?;
///         }
///         Ok(())
///     },
///     &[a, b],
///     &AciOptions::default(),
/// )
/// .unwrap();
///
/// assert!((result.tensor_train.evaluate(&[0, 0]).unwrap() - 6.0).abs() < 1e-12);
/// assert!((result.tensor_train.evaluate(&[1, 1]).unwrap() - 6.0).abs() < 1e-12);
/// assert_eq!(result.ranks.len(), result.errors.len());
/// ```
pub fn elementwise_batched<T, F>(
    mut op: F,
    inputs: &[SimpleTensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: for<'batch> FnMut(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    validate_options(options)?;
    validate_inputs(inputs)?;

    if inputs[0].len() == 1 {
        return elementwise_batched_one_site(op, inputs);
    }

    let mut problem = ElementwiseProblem::new(inputs.to_vec(), options.clone())?;

    let mut ranks = Vec::new();
    let mut errors = Vec::new();

    for iteration in 0..options.max_iters {
        let forward = iteration % 2 == 0;
        if forward {
            for bond in 0..problem.len() - 1 {
                problem.local_update(bond, true, options, &mut op)?;
            }
        } else {
            for bond in (0..problem.len() - 1).rev() {
                problem.local_update(bond, false, options, &mut op)?;
            }
        }

        let max_error_metric = max_error_metric(
            &problem.pivot_errors,
            &problem.pivot_scales,
            options.scale_tolerance,
        );
        ranks.push(problem.solution.rank());
        errors.push(max_error_metric);

        if convergence_criterion_like_julia(
            iteration + 1,
            &ranks,
            &errors,
            options.min_iters,
            options.tolerance,
        ) {
            break;
        }

        if rank_is_saturated(&ranks, options.min_iters, options.max_bond_dim) {
            break;
        }
    }

    Ok(AciResult {
        tensor_train: problem.solution,
        ranks,
        errors,
    })
}

fn elementwise_batched_one_site<T, F>(
    mut op: F,
    inputs: &[SimpleTensorTrain<T>],
) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: for<'batch> FnMut(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let n_inputs = inputs.len();
    let n_points = inputs[0].site_dim(0);
    let mut input_values = vec![T::zero(); n_inputs * n_points];
    for point in 0..n_points {
        for input in 0..n_inputs {
            input_values[input + n_inputs * point] = inputs[input].evaluate(&[point])?;
        }
    }

    let batch = ElementwiseBatch::new(&input_values, n_inputs, n_points)?;
    let mut output = vec![T::zero(); n_points];
    op(batch, &mut output)?;

    let core = tensor3_from_data(output, 1, n_points, 1)?;
    Ok(AciResult {
        tensor_train: SimpleTensorTrain::new(vec![core])?,
        ranks: Vec::new(),
        errors: Vec::new(),
    })
}

/// Runs scalar elementwise ACI over tensor-train inputs.
///
/// This convenience wrapper evaluates `op` once per interpolation point. The
/// callback receives one value from each input tensor train in input order and
/// returns the corresponding output value. Use [`elementwise_batched`] when the
/// operator can evaluate many points more efficiently at once or can fail.
///
/// # Arguments
///
/// * `op` - Scalar callback applied to one interpolation point. The input slice
///
///   has length `inputs.len()`.
/// * `inputs` - Non-empty tensor trains with the same length and site
///
///   dimensions. Core dimensions must be positive.
/// * `options` - Sweep limits, tolerance, maximum bond dimension, random seed,
///
///   and optional initial guess.
///
/// # Returns
///
/// Returns an [`AciResult`] containing the approximating tensor train plus
/// sweep-by-sweep rank and error histories.
///
/// # Errors
///
/// Returns [`AciError`](crate::AciError) when options or inputs are invalid,
/// batch extraction fails, initial guess construction fails, or a local
/// matrix-CI update fails.
///
/// # Panics
///
/// This function does not intentionally panic. A panic from the caller-provided
/// `op` callback is not caught.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::{elementwise, AciOptions};
/// use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
///
/// let a = SimpleTensorTrain::<f64>::constant(&[2, 2], 2.0);
/// let b = SimpleTensorTrain::<f64>::constant(&[2, 2], 5.0);
/// let result = elementwise(
///     |values| values[0] + values[1],
///     &[a, b],
///     &AciOptions::default(),
/// )
/// .unwrap();
///
/// assert!((result.tensor_train.evaluate(&[0, 0]).unwrap() - 7.0).abs() < 1e-12);
/// assert!((result.tensor_train.evaluate(&[1, 1]).unwrap() - 7.0).abs() < 1e-12);
/// assert_eq!(result.ranks.len(), result.errors.len());
/// ```
pub fn elementwise<T, F>(
    mut op: F,
    inputs: &[SimpleTensorTrain<T>],
    options: &AciOptions<T>,
) -> Result<AciResult<T>>
where
    T: AciScalar,
    F: FnMut(&[T]) -> T,
{
    let mut scratch = Vec::new();
    elementwise_batched(
        |batch, output| {
            scratch.clear();
            scratch.reserve(batch.n_inputs());
            for (point, value) in output.iter_mut().enumerate().take(batch.n_points()) {
                scratch.clear();
                for input in 0..batch.n_inputs() {
                    scratch.push(batch.get(input, point)?);
                }
                *value = op(&scratch);
            }
            Ok(())
        },
        inputs,
        options,
    )
}

#[cfg(test)]
pub(crate) fn ranks_are_stable(ranks: &[usize], min_iters: usize) -> bool {
    if ranks.is_empty() || min_iters == 0 {
        return min_iters == 0;
    }

    let baseline_index = ranks.len().saturating_sub(min_iters);
    let baseline = ranks[baseline_index];
    ranks[baseline_index..].iter().all(|&rank| rank <= baseline)
}

pub(crate) fn convergence_criterion_like_julia(
    iteration: usize,
    ranks: &[usize],
    errors: &[f64],
    min_iters: usize,
    tolerance: f64,
) -> bool {
    debug_assert!(iteration > 0);
    debug_assert_eq!(ranks.len(), iteration);
    debug_assert_eq!(errors.len(), iteration);

    if iteration == 0 || min_iters == 0 {
        return false;
    }
    if iteration < min_iters {
        return false;
    }
    if errors[iteration - 1] > tolerance {
        return false;
    }

    let baseline = ranks[iteration - min_iters];
    !ranks[(iteration - min_iters)..iteration]
        .iter()
        .any(|&rank| rank > baseline)
}

/// Reports whether the solution rank has been pinned at `max_bond_dim` long
/// enough that further sweeps cannot reduce the pivot error below tolerance.
///
/// Once the rank reaches the cap the tolerance branch of
/// [`convergence_criterion_like_julia`] can never fire, because the sweep has no
/// remaining freedom to add pivots. Without this exit a capped run burns every
/// remaining sweep at full rank. The dwell of `min_iters` consecutive sweeps at
/// the cap guards against stopping on the sweep where the cap is first reached,
/// when pivot re-selection inside the fixed rank can still improve the error.
///
/// `min_iters == 0` disables the exit, matching
/// [`convergence_criterion_like_julia`], and the default `max_bond_dim` of
/// `None` is unreachable, so unconstrained runs are unaffected.
///
/// This is the `all(lastranks .>= max_bond_dim)` disjunct of the Julia
/// `convergencecriterion` that [`convergence_criterion_like_julia`] otherwise
/// ports, over the same trailing window. `tensor4all-treetci` gained the
/// equivalent exit in its own sweep loop in #575.
pub(crate) fn rank_is_saturated(
    ranks: &[usize],
    min_iters: usize,
    max_bond_dim: Option<usize>,
) -> bool {
    if min_iters == 0 || ranks.len() < min_iters {
        return false;
    }

    max_bond_dim.is_some_and(|cap| {
        ranks[(ranks.len() - min_iters)..]
            .iter()
            .all(|&rank| rank >= cap)
    })
}

pub(crate) fn error_metric(
    max_pivot_error: f64,
    max_sampled_scale: f64,
    scale_tolerance: bool,
) -> f64 {
    if scale_tolerance && max_sampled_scale > 0.0 {
        max_pivot_error / max_sampled_scale
    } else {
        max_pivot_error
    }
}

pub(crate) fn max_error_metric(
    pivot_errors: &[f64],
    pivot_scales: &[f64],
    scale_tolerance: bool,
) -> f64 {
    pivot_errors
        .iter()
        .enumerate()
        .map(|(bond, &error)| {
            let scale = pivot_scales.get(bond).copied().unwrap_or(0.0);
            error_metric(error, scale, scale_tolerance)
        })
        .fold(0.0_f64, f64::max)
}
