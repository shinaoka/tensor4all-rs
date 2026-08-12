//! Global pivot search guard for elementwise ACI convergence.
//!
//! ACI's sweeps estimate error from bond-local 2-site LU blocks only, so a
//! feature outside the sampled crosses (e.g. a near-degenerate second peak
//! far from the initial pivots) is invisible to the stopping rule: the run
//! self-reports convergence while the feature silently vanishes. This module
//! implements the same guard as `tensor4all-treetci`'s global pivot search:
//! before accepting convergence, the current solution is sampled against the
//! true operator at global points, and any significantly-wrong points are
//! returned for injection into the ACI frames.

use crate::scalar::AciScalar;
use crate::{AciOptions, ElementwiseBatch, ElementwiseProblem, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_simplett::{AbstractTensorTrain, EinsumScalar, TTCache};
use tensor4all_tcicore::MatrixLuciScalar;

/// Search for global multi-indices where the current solution is wrong.
///
/// Algorithm (mirrors `tensor4all-treetci`'s `find_global_pivots`):
///
/// 1. Draw `nsearch` random starting points.
/// 2. For each starting point, sweep every site coordinate over its full
///    local dimension, keeping the point with the largest interpolation
///    error `|op(inputs(idx)) - solution(idx)|`.
/// 3. Keep points whose error exceeds `abs_tol * tol_margin`, where
///    `abs_tol` is the configured tolerance (scaled by the maximum sampled
///    operator magnitude when `scale_tolerance` is enabled).
/// 4. Return at most `max_nglobal_pivot` distinct points.
///
/// Input values and the current solution are evaluated at all candidate
/// points in one batch per tensor train via [`TTCache::evaluate_many`]
/// (shared prefixes/suffixes are contracted once); the operator is called in
/// a single [`ElementwiseBatch`].
pub(crate) fn find_global_pivots<T, F>(
    problem: &ElementwiseProblem<T>,
    op: &mut F,
    options: &AciOptions<T>,
    seed: u64,
) -> Result<Vec<Vec<usize>>>
where
    T: AciScalar + EinsumScalar,
    F: for<'batch> FnMut(ElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let n_sites = problem.len();
    let n_inputs = problem.n_inputs();
    let nsearch = options.nsearch_global_pivots;
    let max_nglobal_pivot = options.max_nglobal_pivot;
    if nsearch == 0 || max_nglobal_pivot == 0 || n_sites < 2 {
        return Ok(Vec::new());
    }

    let site_dims: Vec<usize> = (0..n_sites)
        .map(|site| problem.solution.site_dim(site))
        .collect();

    // Candidate points: for each random start, sweep each site coordinate
    // over its full local dimension (same local search as the chain and tree
    // TCI2 finders; candidates share prefixes/suffixes heavily, which
    // `TTCache::evaluate_many` exploits).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points: Vec<Vec<usize>> = Vec::new();
    for _ in 0..nsearch {
        let start: Vec<usize> = site_dims.iter().map(|&d| rng.random_range(0..d)).collect();
        for site in 0..n_sites {
            for value in 0..site_dims[site] {
                let mut point = start.clone();
                point[site] = value;
                points.push(point);
            }
        }
    }
    let n_points = points.len();

    // Input values at all candidate points: one cached batch per input.
    let mut input_values = vec![T::zero(); n_inputs * n_points];
    for input in 0..n_inputs {
        let mut cache = TTCache::new(&problem.inputs[input]);
        let values = cache.evaluate_many(&points, None)?;
        for (point, value) in values.into_iter().enumerate() {
            input_values[input + n_inputs * point] = value;
        }
    }

    // Operator output at all candidate points: one batch call.
    let batch = ElementwiseBatch::new(&input_values, n_inputs, n_points)?;
    let mut op_values = vec![T::zero(); n_points];
    op(batch, &mut op_values)?;

    // Current solution at all candidate points: one cached batch.
    let mut solution_cache = TTCache::new(&problem.solution);
    let solution_values = solution_cache.evaluate_many(&points, None)?;

    let max_op_abs = op_values
        .iter()
        .map(|&value| MatrixLuciScalar::abs_val(value))
        .fold(0.0f64, f64::max);
    let abs_tol = if options.scale_tolerance && max_op_abs > 0.0 {
        options.tolerance * max_op_abs
    } else {
        options.tolerance
    };
    let threshold = abs_tol * options.tol_margin_global_search;

    // Local search per starting point.
    let mut best: Vec<(f64, Vec<usize>)> = Vec::new();
    let mut point_index = 0usize;
    for _ in 0..nsearch {
        let mut start_best: Option<(f64, Vec<usize>)> = None;
        for &site_dim in &site_dims {
            for _value in 0..site_dim {
                let error = MatrixLuciScalar::abs_val(
                    op_values[point_index] - solution_values[point_index],
                );
                if start_best
                    .as_ref()
                    .is_none_or(|(best_error, _)| error > *best_error)
                {
                    start_best = Some((error, points[point_index].clone()));
                }
                point_index += 1;
            }
        }
        if let Some((error, point)) = start_best {
            if error > threshold {
                best.push((error, point));
            }
        }
    }

    // Keep the strongest distinct points.
    best.sort_by(|(a, _), (b, _)| b.total_cmp(a));
    let mut pivots: Vec<Vec<usize>> = Vec::new();
    for (_, point) in best {
        if !pivots.contains(&point) {
            pivots.push(point);
            if pivots.len() >= max_nglobal_pivot {
                break;
            }
        }
    }
    Ok(pivots)
}

#[cfg(test)]
mod tests;
