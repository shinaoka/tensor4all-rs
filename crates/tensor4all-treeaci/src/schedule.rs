//! Deterministic execution of directional tree ACI passes.

#![allow(dead_code)]

use tensor4all_treetn::{CanonicalForm, CanonicalizationOptions};

use crate::{
    global_guard::{find_global_pivots, inject_global_pivots, InputEvaluators},
    path_cover::{OrientedEdgeStep, PathPhase},
    problem::DirectedEdgeId,
    state::TreeAciState,
    transaction::update_edge_transaction,
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeAciTermination,
    TreeElementwiseBatch,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PassDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PassReport {
    pub(crate) direction: PassDirection,
    pub(crate) updated_edges: Vec<DirectedEdgeId>,
    pub(crate) max_rank: usize,
    pub(crate) max_error: f64,
    pub(crate) rank_changed: bool,
    pub(crate) evaluated_points: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SweepHistory {
    pub(crate) max_ranks: Vec<usize>,
    pub(crate) max_errors: Vec<f64>,
    pub(crate) termination: TreeAciTermination,
    pub(crate) global_pivots_found: Vec<usize>,
    pub(crate) evaluated_points: u64,
}

impl PassReport {
    pub(crate) fn update_count(&self) -> usize {
        self.updated_edges.len()
    }
}

pub(crate) fn run_local_sweeps<'a, T, V, F>(
    state: &mut TreeAciState<'a, T, V>,
    input_evaluators: &mut InputEvaluators<'a, V>,
    options: &TreeAciOptions<V>,
    operator: &mut F,
) -> Result<SweepHistory>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let mut max_ranks = Vec::with_capacity(options.max_sweeps);
    let mut max_errors = Vec::with_capacity(options.max_sweeps);
    let mut global_pivots = Vec::with_capacity(options.max_sweeps);
    let mut rank_limited = Vec::with_capacity(options.max_sweeps);
    let mut evaluated_points = 0u64;
    let mut termination = TreeAciTermination::MaxSweeps;

    for pass in 0..options.max_sweeps {
        let direction = if pass % 2 == 0 {
            PassDirection::Forward
        } else {
            PassDirection::Reverse
        };
        let report = run_directional_pass(state, options, direction, operator)?;
        evaluated_points = evaluated_points
            .checked_add(report.evaluated_points)
            .ok_or(TreeAciError::SizeOverflow {
                context: "sweep evaluated point count",
            })?;
        max_ranks.push(report.max_rank);
        max_errors.push(report.max_error);
        rank_limited.push(current_state_is_rank_limited(state, options));
        let injection_edges = global_injection_edges(state, options);
        let found = if options.enable_global_guard
            && options.nsearch_global_pivots > 0
            && options.max_nglobal_pivots > 0
            && !rank_limited[pass]
            && injection_edges.iter().any(|activate| *activate)
        {
            let seed = options.rng_seed.wrapping_add((pass + 1) as u64);
            let search = find_global_pivots(state, input_evaluators, options, seed, operator)?;
            evaluated_points = evaluated_points
                .checked_add(search.evaluated_points)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "sweep evaluated point count",
                })?;
            let found = search.pivots.len();
            inject_global_pivots(state, input_evaluators, &search.pivots, &injection_edges)?;
            found
        } else {
            0
        };
        global_pivots.push(found);
        let completed = pass + 1;
        if convergence_criterion(
            completed,
            &max_ranks,
            &max_errors,
            &global_pivots,
            options.min_sweeps,
            options.tolerance,
        ) {
            termination = TreeAciTermination::Converged;
            break;
        }
        if trailing_all_true(&rank_limited, options.min_sweeps) {
            termination = TreeAciTermination::RankLimited;
            break;
        }
    }
    Ok(SweepHistory {
        max_ranks,
        max_errors,
        termination,
        global_pivots_found: global_pivots,
        evaluated_points,
    })
}

fn global_injection_edges<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
) -> Vec<bool> {
    state
        .edge_ranks
        .iter()
        .zip(&state.algebraic_edge_bounds)
        .map(|(&rank, &algebraic)| {
            let limit = options.max_bond_dim.unwrap_or(usize::MAX).min(algebraic);
            rank < limit
        })
        .collect()
}

pub(crate) fn run_directional_pass<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
    direction: PassDirection,
    operator: &mut F,
) -> Result<PassReport>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let phases = match direction {
        PassDirection::Forward => state.problem.schedule.forward.clone(),
        PassDirection::Reverse => state.problem.schedule.reverse.clone(),
    };
    let ranks_before = state.edge_ranks.clone();
    let mut updated_edges = Vec::with_capacity(state.problem.directed_edges.len() / 2);
    let mut evaluated_points = 0u64;

    for phase in &phases {
        if let Err(error) = run_phase_serial(
            state,
            options,
            phase,
            operator,
            &mut updated_edges,
            &mut evaluated_points,
        ) {
            if !updated_edges.is_empty() {
                finalize_deferred_canonicalization(state)?;
            }
            return Err(error);
        }
    }

    finalize_deferred_canonicalization(state)?;

    let max_rank = state.edge_ranks.iter().copied().max().unwrap_or(1);
    let max_error = state
        .edge_errors
        .iter()
        .zip(&state.edge_scales)
        .map(|(&error, &scale)| {
            if options.scale_tolerance && scale > 0.0 {
                error / scale
            } else {
                error
            }
        })
        .fold(0.0, f64::max);
    Ok(PassReport {
        direction,
        updated_edges,
        max_rank,
        max_error,
        rank_changed: state.edge_ranks != ranks_before,
        evaluated_points,
    })
}

fn finalize_deferred_canonicalization<T: TreeAciScalar, V: TreeAciNode>(
    state: &mut TreeAciState<'_, T, V>,
) -> Result<()> {
    if state.output.canonical_form().is_none() {
        let center = state
            .output
            .canonical_region()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        state.output.canonicalize_mut(
            center,
            CanonicalizationOptions::default().with_form(CanonicalForm::CI),
        )?;
    }
    Ok(())
}

fn run_phase_serial<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
    phase: &PathPhase,
    operator: &mut F,
    updated_edges: &mut Vec<DirectedEdgeId>,
    evaluated_points: &mut u64,
) -> Result<()>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    for path in &phase.paths {
        let first = path.steps.first().ok_or(TreeAciError::InternalInvariant {
            message: "a scheduled path has no edge steps",
        })?;
        let start = state
            .problem
            .node_order
            .get(first.from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a scheduled path starts at an unknown node",
            })?
            .clone();
        let region = state.output.canonical_region();
        if region.len() != 1 || !region.contains(&start) {
            return Err(TreeAciError::InternalInvariant {
                message: "continuous sweep does not start at the current canonical center",
            });
        }

        for step in &path.steps {
            let directed = directed_edge_for_step(state, step)?;
            let report = update_edge_transaction(state, directed, options, true, operator)?;
            *evaluated_points = evaluated_points
                .checked_add(u64::try_from(report.evaluated_points).map_err(|_| {
                    TreeAciError::SizeOverflow {
                        context: "pass evaluated point count",
                    }
                })?)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "pass evaluated point count",
                })?;
            updated_edges.push(directed);
        }
    }
    Ok(())
}

fn convergence_criterion(
    completed: usize,
    max_ranks: &[usize],
    errors: &[f64],
    global_pivots: &[usize],
    min_sweeps: usize,
    tolerance: f64,
) -> bool {
    if min_sweeps == 0 || completed < min_sweeps || errors[completed - 1] > tolerance {
        return false;
    }
    let baseline = max_ranks[completed - min_sweeps];
    if max_ranks[(completed - min_sweeps)..completed]
        .iter()
        .any(|&rank| rank > baseline)
    {
        return false;
    }
    global_pivots[(completed - min_sweeps)..completed]
        .iter()
        .all(|found| *found == 0)
}

fn current_state_is_rank_limited<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
) -> bool {
    let mut has_bad_edge = false;
    for (((&rank, &algebraic), &error), &scale) in state
        .edge_ranks
        .iter()
        .zip(&state.algebraic_edge_bounds)
        .zip(&state.edge_errors)
        .zip(&state.edge_scales)
    {
        let metric = edge_error_metric(error, scale, options.scale_tolerance);
        if metric > options.tolerance {
            has_bad_edge = true;
            let limit = options.max_bond_dim.unwrap_or(usize::MAX).min(algebraic);
            if rank < limit {
                return false;
            }
        }
    }
    has_bad_edge
}

fn edge_error_metric(error: f64, scale: f64, relative: bool) -> f64 {
    if relative && scale > 0.0 {
        error / scale
    } else {
        error
    }
}

fn trailing_all_true(values: &[bool], dwell: usize) -> bool {
    dwell > 0
        && values.len() >= dwell
        && values[(values.len() - dwell)..].iter().all(|value| *value)
}

fn directed_edge_for_step<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    step: &OrientedEdgeStep,
) -> Result<DirectedEdgeId> {
    let base = step.edge.checked_mul(2).ok_or(TreeAciError::SizeOverflow {
        context: "directed edge identifier",
    })?;
    let from = state
        .problem
        .node_order
        .get(step.from)
        .ok_or(TreeAciError::InternalInvariant {
            message: "a scheduled edge source is unknown",
        })?;
    let forward =
        state
            .problem
            .directed_edges
            .get(base)
            .ok_or(TreeAciError::InternalInvariant {
                message: "a scheduled edge is unknown",
            })?;
    if &forward.from == from {
        Ok(base)
    } else if &forward.to == from {
        Ok(forward.reverse)
    } else {
        Err(TreeAciError::InternalInvariant {
            message: "a scheduled edge orientation disagrees with the prepared topology",
        })
    }
}

#[cfg(test)]
mod tests;
