//! Greedy coordinate-descent (floating-zone) search for high-error points.
//!
//! The floating-zone walk is the strongest global pivot search used in this
//! codebase: from a starting point it repeatedly sweeps every site
//! coordinate, moving each coordinate to the value with the largest
//! interpolation error, until the error stops improving or exceeds a
//! tolerance. It is a strict generalization of the single-cross search used
//! elsewhere (one sweep with no repeats equals a cross scan).

use crate::MultiIndex;

/// Walk one floating-zone search trajectory.
///
/// Mirrors `TensorCrossInterpolation.jl`'s `_floatingzone`: starting from
/// `init_p`, each sweep visits every site in order and moves that site's
/// coordinate to the value with the largest error (as measured by
/// `eval_batch`), keeping the running maximum error monotonically
/// non-decreasing. The walk stops when a sweep does not increase the
/// maximum error (the trajectory is stuck on a local maximum) or when the
/// maximum error exceeds `early_stop_tol` (the point is already
/// significant), or after `max_sweeps` sweeps as a safety bound.
///
/// # Arguments
///
/// * `local_dims` - Local dimension of each site.
/// * `init_p` - Starting multi-index; must have length `local_dims.len()`.
/// * `max_sweeps` - Upper bound on the number of coordinate sweeps. The
///   no-improvement early stop almost always fires first.
/// * `early_stop_tol` - Stop walking once the maximum error exceeds this
///   value; the caller has found a significantly wrong point.
/// * `eval_batch` - Evaluates the error magnitude `|f - tt|` at a batch of
///   multi-indices. Called once per site per sweep with that site's
///   `local_dims[site]` candidate points (the current pivot with the site
///   coordinate varied), so callers can batch shared contractions.
///
/// # Returns
///
/// The final pivot and the maximum error encountered along the walk. The
/// returned error may exceed `early_stop_tol`; the caller decides whether
/// the point is significant.
///
/// # Errors
///
/// Propagates the error returned by `eval_batch` unchanged — typically an
/// operation failure or an index mismatch from the underlying evaluator.
pub fn floating_zone_walk<E, Err>(
    local_dims: &[usize],
    init_p: &MultiIndex,
    max_sweeps: usize,
    early_stop_tol: f64,
    mut eval_batch: E,
) -> std::result::Result<(MultiIndex, f64), Err>
where
    E: FnMut(&[MultiIndex]) -> std::result::Result<Vec<f64>, Err>,
{
    let n = local_dims.len();

    let mut pivot = init_p.clone();

    // Initial error at the starting point.
    let start_errors = eval_batch(&[pivot.clone()])?;
    let mut max_error = start_errors.first().copied().unwrap_or(0.0);

    for _ in 0..max_sweeps {
        let prev_max_error = max_error;
        for ipos in 0..n {
            // Candidate points: every value at this site, the rest fixed at
            // the current pivot (updated greedily within this sweep).
            let mut points = Vec::with_capacity(local_dims[ipos]);
            for value in 0..local_dims[ipos] {
                let mut point = pivot.clone();
                point[ipos] = value;
                points.push(point);
            }
            let errors = eval_batch(&points)?;

            let mut best_local_idx = pivot[ipos];
            let mut best_local_error = 0.0f64;
            for (value, &error) in errors.iter().enumerate() {
                if error > best_local_error {
                    best_local_error = error;
                    best_local_idx = value;
                }
            }
            pivot[ipos] = best_local_idx;
            max_error = max_error.max(best_local_error);
        }

        if max_error == prev_max_error || max_error > early_stop_tol {
            break;
        }
    }

    Ok((pivot, max_error))
}
