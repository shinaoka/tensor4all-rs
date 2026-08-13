//! Global pivot search guard for elementwise ACI convergence.
//!
//! ACI's sweeps estimate error from bond-local 2-site LU blocks only, so a
//! feature outside the sampled crosses (e.g. a near-degenerate second peak
//! far from the initial pivots) is invisible to the stopping rule: the run
//! self-reports convergence while the feature silently vanishes. This module
//! implements the global pivot search guard: before convergence is accepted
//! the current solution is sampled against the true operator at global
//! points, and any significantly-wrong points are returned for injection
//! into the ACI frames.
//!
//! The search uses floating-zone walks ([`floating_zone_walk`]), the same
//! greedy coordinate-descent search as `TensorCrossInterpolation.jl`'s
//! `_floatingzone` and `tensor4all-tensorci`'s [`floating_zone`]
//! (tensor4all_tensorci::floating_zone): each random start moves one site
//! coordinate at a time toward the largest interpolation error. This is a
//! strict generalization of the single-cross scan (one sweep with no
//! repeats), so far features that a cross scan can never sample are reachable
//! when the error landscape has a detectable gradient.

use crate::scalar::AciScalar;
use crate::{AciOptions, ElementwiseBatch, ElementwiseProblem, Result};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_simplett::{AbstractTensorTrain, EinsumScalar, TTCache};
use tensor4all_tcicore::{floating_zone_walk, MatrixLuciScalar};

/// Search for global multi-indices where the current solution is wrong.
///
/// Algorithm:
///
/// 1. Draw `nsearch` random starting points.
/// 2. For each starting point, run a floating-zone walk: repeatedly sweep
///    every site coordinate, moving each coordinate to the value with the
///    largest interpolation error `|op(inputs(idx)) - solution(idx)|`, until
///    the error stops improving or exceeds `abs_tol * tol_margin`.
/// 3. Keep points whose error exceeds `abs_tol * tol_margin`, where
///    `abs_tol` is the configured tolerance (scaled by the maximum sampled
///    operator magnitude when `scale_tolerance` is enabled).
/// 4. Return at most `max_nglobal_pivot` distinct points.
///
/// Input values and the current solution are evaluated per walk step in one
/// cached batch per tensor train via [`TTCache::evaluate_many`] (shared
/// prefixes/suffixes are contracted once); the operator is called in a single
/// [`ElementwiseBatch`] per step.
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

    let mut rng = StdRng::seed_from_u64(seed);
    let starts: Vec<Vec<usize>> = (0..nsearch)
        .map(|_| site_dims.iter().map(|&d| rng.random_range(0..d)).collect())
        .collect();

    // Absolute threshold for "significantly wrong": abs_tol * tol_margin.
    // With scale_tolerance the operator scale is estimated from the random
    // starting points (the walk is not handed a precomputed candidate set).
    let mut start_input_values = vec![T::zero(); n_inputs * nsearch];
    for input in 0..n_inputs {
        let mut cache = TTCache::new(&problem.inputs[input]);
        let values = cache.evaluate_many(&starts, None)?;
        for (point, value) in values.into_iter().enumerate() {
            start_input_values[input + n_inputs * point] = value;
        }
    }
    let start_batch = ElementwiseBatch::new(&start_input_values, n_inputs, nsearch)?;
    let mut start_op_values = vec![T::zero(); nsearch];
    op(start_batch, &mut start_op_values)?;

    let max_op_abs = start_op_values
        .iter()
        .map(|&value| MatrixLuciScalar::abs_val(value))
        .fold(0.0f64, f64::max);
    let abs_tol = if options.scale_tolerance && max_op_abs > 0.0 {
        options.tolerance * max_op_abs
    } else {
        options.tolerance
    };
    let threshold = abs_tol * options.tol_margin_global_search;

    // Floating-zone walk per starting point.
    let mut best: Vec<(f64, Vec<usize>)> = Vec::new();
    for start in &starts {
        let (pivot, error) = floating_zone_walk(
            &site_dims,
            start,
            options.nsweeps_global_search,
            threshold,
            |points: &[Vec<usize>]| -> crate::Result<Vec<f64>> {
                let n_points = points.len();
                let mut input_values = vec![T::zero(); n_inputs * n_points];
                for input in 0..n_inputs {
                    let mut cache = TTCache::new(&problem.inputs[input]);
                    let values = cache.evaluate_many(points, None)?;
                    for (point, value) in values.into_iter().enumerate() {
                        input_values[input + n_inputs * point] = value;
                    }
                }
                let batch = ElementwiseBatch::new(&input_values, n_inputs, n_points)?;
                let mut op_values = vec![T::zero(); n_points];
                op(batch, &mut op_values)?;
                let mut solution_cache = TTCache::new(&problem.solution);
                let solution_values = solution_cache.evaluate_many(points, None)?;
                let errors = (0..n_points)
                    .map(|point| {
                        MatrixLuciScalar::abs_val(op_values[point] - solution_values[point])
                    })
                    .collect::<Vec<f64>>();
                Ok(errors)
            },
        )?;
        if error > threshold {
            best.push((error, pivot));
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
