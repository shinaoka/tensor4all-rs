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
///   multi-indices. It is called once for the starting point with a scan site
///   of `None`, and then once per site per sweep with `Some(site)` and that
///   site's `local_dims[site] - 1` *other* candidate points: the candidate
///   equal to the current pivot is not re-evaluated, because the walk already
///   knows its error from the step that moved there. The scan site is passed
///   so that a caller whose evaluator exploits scan structure can declare it
///   rather than infer it from the batch, which a single-point batch cannot
///   support.
///
/// # Returns
///
/// The final pivot and the maximum error encountered along the walk. The
/// returned error may exceed `early_stop_tol`; the caller decides whether
/// the point is significant.
///
/// # Errors
///
/// Propagates the error returned by `eval_batch` unchanged - typically an
/// operation failure or an index mismatch from the underlying evaluator.
///
/// # Examples
///
/// ```
/// use tensor4all_core::floating_zone_walk;
///
/// // A separable error surface whose maximum is the all-last-coordinate
/// // point, so a greedy coordinate walk must find it exactly.
/// let local_dims = [3usize, 4, 2];
/// let error_at = |point: &Vec<usize>| point.iter().map(|&c| c as f64).sum::<f64>();
/// let mut evaluated = 0usize;
/// let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
///     &local_dims,
///     &vec![0usize, 0, 0],
///     16,
///     f64::INFINITY,
///     |_site, points| {
///         evaluated += points.len();
///         Ok(points.iter().map(error_at).collect())
///     },
/// )?;
///
/// assert_eq!(pivot, vec![2, 3, 1]);
/// assert_eq!(error, 6.0);
/// // One point for the start, then `local_dims[site] - 1` per site scan: the
/// // coordinate the pivot already holds is never re-evaluated.
/// assert_eq!(evaluated, 1 + 2 * ((3 - 1) + (4 - 1) + (2 - 1)));
/// # Ok::<(), std::convert::Infallible>(())
/// ```
pub fn floating_zone_walk<E, Err>(
    local_dims: &[usize],
    init_p: &MultiIndex,
    max_sweeps: usize,
    early_stop_tol: f64,
    mut eval_batch: E,
) -> std::result::Result<(MultiIndex, f64), Err>
where
    E: FnMut(Option<usize>, &[MultiIndex]) -> std::result::Result<Vec<f64>, Err>,
{
    let n = local_dims.len();

    let mut pivot = init_p.clone();

    // Initial error at the starting point. This also seeds `pivot_error`,
    // which is what lets every site scan below skip the candidate the pivot
    // already holds.
    let start_errors = eval_batch(None, &[pivot.clone()])?;
    let mut max_error = start_errors.first().copied().unwrap_or(0.0);
    let mut pivot_error = max_error;

    for _ in 0..max_sweeps {
        let prev_max_error = max_error;
        for ipos in 0..n {
            // Candidate points: every value at this site except the one the
            // pivot already holds, the rest fixed at the current pivot
            // (updated greedily within this sweep). The skipped candidate is
            // the current pivot itself, whose error is `pivot_error`.
            let held = pivot[ipos];
            let mut points = Vec::with_capacity(local_dims[ipos].saturating_sub(1));
            for value in 0..local_dims[ipos] {
                if value == held {
                    continue;
                }
                let mut point = pivot.clone();
                point[ipos] = value;
                points.push(point);
            }
            let errors = if points.is_empty() {
                Vec::new()
            } else {
                eval_batch(Some(ipos), &points)?
            };

            // Fold in value order, with the held candidate's known error in
            // its own place, so the greedy choice is the one the full batch
            // would have made.
            let mut best_local_idx = held;
            let mut best_local_error = 0.0f64;
            let mut evaluated = errors.iter();
            for value in 0..local_dims[ipos] {
                let error = if value == held {
                    pivot_error
                } else {
                    match evaluated.next() {
                        Some(&error) => error,
                        None => break,
                    }
                };
                if error > best_local_error {
                    best_local_error = error;
                    best_local_idx = value;
                }
            }
            pivot[ipos] = best_local_idx;
            // The pivot is now the winning candidate of this scan, so its
            // error is that candidate's error.
            pivot_error = best_local_error;
            max_error = max_error.max(best_local_error);
        }

        if max_error == prev_max_error || max_error > early_stop_tol {
            break;
        }
    }

    Ok((pivot, max_error))
}

#[cfg(test)]
mod tests {
    use super::floating_zone_walk;

    /// The reference implementation this walk replaces: every candidate at a
    /// site is evaluated, including the one the pivot already holds.
    fn full_batch_walk(
        local_dims: &[usize],
        init_p: &[usize],
        max_sweeps: usize,
        early_stop_tol: f64,
        error_at: &dyn Fn(&[usize]) -> f64,
    ) -> (Vec<usize>, f64, usize) {
        let mut pivot = init_p.to_vec();
        let mut evaluated = 1usize;
        let mut max_error = error_at(&pivot);
        for _ in 0..max_sweeps {
            let previous = max_error;
            for site in 0..local_dims.len() {
                let mut best_index = pivot[site];
                let mut best_error = 0.0f64;
                for value in 0..local_dims[site] {
                    let mut point = pivot.clone();
                    point[site] = value;
                    evaluated += 1;
                    let error = error_at(&point);
                    if error > best_error {
                        best_error = error;
                        best_index = value;
                    }
                }
                pivot[site] = best_index;
                max_error = max_error.max(best_error);
            }
            if max_error == previous || max_error > early_stop_tol {
                break;
            }
        }
        (pivot, max_error, evaluated)
    }

    fn rough_error(point: &[usize]) -> f64 {
        let mut value = 0.0;
        for (site, &coordinate) in point.iter().enumerate() {
            value += ((site as f64 + 1.7) * (coordinate as f64 + 0.3)).sin();
        }
        value.abs()
    }

    /// Skipping the held candidate must not change the trajectory, the pivot,
    /// or the reported error, only the number of points evaluated.
    #[test]
    fn skipping_the_held_candidate_matches_the_full_batch_walk() {
        for local_dims in [
            vec![2usize, 2, 2, 2, 2],
            vec![3usize, 2, 4],
            vec![2usize, 5, 3, 2],
        ] {
            for start in [
                vec![0usize; local_dims.len()],
                vec![1usize; local_dims.len()],
            ] {
                let start: Vec<usize> = start
                    .iter()
                    .zip(&local_dims)
                    .map(|(&coordinate, &dim)| coordinate % dim)
                    .collect();
                let (expected_pivot, expected_error, full_points) =
                    full_batch_walk(&local_dims, &start, 8, f64::INFINITY, &rough_error);

                let mut evaluated = 0usize;
                let mut scan_sites = Vec::new();
                let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
                    &local_dims,
                    &start,
                    8,
                    f64::INFINITY,
                    |site, points| {
                        scan_sites.push(site);
                        evaluated += points.len();
                        Ok(points.iter().map(|point| rough_error(point)).collect())
                    },
                )
                .unwrap();

                assert_eq!(pivot, expected_pivot);
                assert_eq!(error, expected_error);
                assert!(
                    evaluated < full_points,
                    "the skip must evaluate fewer points: {evaluated} against {full_points}"
                );
                // Every scan declares its site, and only the seed does not.
                assert_eq!(scan_sites.first().copied(), Some(None));
                assert!(scan_sites[1..].iter().all(Option::is_some));
                // Every batch a scan asks for excludes exactly one candidate.
                let sweeps = (scan_sites.len() - 1) / local_dims.len();
                let per_sweep: usize = local_dims.iter().map(|dim| dim - 1).sum();
                assert_eq!(evaluated, 1 + sweeps * per_sweep);
            }
        }
    }

    /// A site of local dimension one has no other candidate, so its scan asks
    /// for nothing at all and the pivot keeps its only value.
    #[test]
    fn a_singleton_site_is_never_evaluated() {
        let local_dims = [1usize, 2];
        let mut batches = Vec::new();
        let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
            &local_dims,
            &vec![0usize, 0],
            4,
            f64::INFINITY,
            |site, points| {
                batches.push((site, points.len()));
                Ok(points.iter().map(|point| point[1] as f64).collect())
            },
        )
        .unwrap();

        assert_eq!(pivot, vec![0, 1]);
        assert_eq!(error, 1.0);
        assert!(!batches.contains(&(Some(0), 1)));
        assert!(batches.contains(&(Some(1), 1)));
    }

    /// The error the seed call reports is the pivot's own error, so a start
    /// that is already the maximum is not lost by the first scan.
    #[test]
    fn a_maximal_start_is_kept() {
        let local_dims = [2usize, 2];
        let (pivot, error) = floating_zone_walk::<_, std::convert::Infallible>(
            &local_dims,
            &vec![1usize, 1],
            4,
            f64::INFINITY,
            |_site, points| {
                Ok(points
                    .iter()
                    .map(|point| if point == &vec![1usize, 1] { 5.0 } else { 1.0 })
                    .collect())
            },
        )
        .unwrap();

        assert_eq!(pivot, vec![1, 1]);
        assert_eq!(error, 5.0);
    }
}
