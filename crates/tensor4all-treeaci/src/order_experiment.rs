//! Test-only pivot state transitions for edge-order experiments.

use crate::{
    frames::InputFrameStore,
    local_update::materialize_and_factor_edge,
    problem::{DirectedEdgeId, PreparedTreeProblem},
    state::TreeAciState,
    Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

pub(crate) fn pivot_only_update<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    forward: DirectedEdgeId,
    options: &TreeAciOptions<V>,
    operator: &mut F,
) -> Result<usize>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let proposal = materialize_and_factor_edge(
        state.inputs,
        &state.problem,
        &state.candidates,
        &state.input_frames,
        forward,
        options,
        true,
        operator,
    )?;
    let directed =
        state
            .problem
            .directed_edges
            .get(forward)
            .ok_or(TreeAciError::InternalInvariant {
                message: "pivot-only update references an unknown directed edge",
            })?;
    let reverse = directed.reverse;
    let edge_number = forward / 2;
    let evaluated_points =
        proposal
            .row_count
            .checked_mul(proposal.col_count)
            .ok_or(TreeAciError::SizeOverflow {
                context: "pivot-only evaluated point count",
            })?;
    let next_generation = state
        .generation
        .checked_add(1)
        .ok_or(TreeAciError::SizeOverflow {
            context: "tree ACI state generation",
        })?;

    let mut arena = state.sample_arena.clone();
    let left_ids = proposal
        .row_samples
        .iter()
        .cloned()
        .map(|sample| arena.intern_component(&state.problem, forward, sample))
        .collect::<Result<Vec<_>>>()?;
    let right_ids = proposal
        .col_samples
        .iter()
        .cloned()
        .map(|sample| arena.intern_component(&state.problem, reverse, sample))
        .collect::<Result<Vec<_>>>()?;
    let mut candidates = state.candidates.clone();
    candidates.ids[forward] = left_ids.clone();
    candidates.ids[reverse] = right_ids.clone();
    candidates.generation = next_generation;

    let pairs = if forward.is_multiple_of(2) {
        left_ids.iter().copied().zip(right_ids.iter().copied())
    } else {
        right_ids.iter().copied().zip(left_ids.iter().copied())
    };
    let mut pivots = state.pivots.clone();
    pivots.set(edge_number, pairs.collect());
    let input_frames = InputFrameStore::from_samples(state.inputs, &state.problem, &arena)?;
    let pivot_error = proposal.pivot_errors.last().copied().unwrap_or(0.0);

    state.sample_arena = arena;
    state.candidates = candidates;
    state.pivots = pivots;
    state.input_frames = input_frames;
    state.edge_ranks[edge_number] = state.pivots.rank(edge_number);
    state.edge_errors[edge_number] = pivot_error;
    state.edge_scales[edge_number] = proposal.sampled_scale;
    state.generation = next_generation;
    Ok(evaluated_points)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EdgeOrder {
    ContinuousWalk,
    RandomPermutation,
    EdgeIndex,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ArmOutcome {
    pub(crate) edge_updates: usize,
    pub(crate) evaluated_points: usize,
    pub(crate) held_out_error: f64,
    pub(crate) pivot_error: f64,
    pub(crate) ranks: Vec<usize>,
    pub(crate) nested_fraction: f64,
}

pub(crate) fn run_arm<T, V, F>(
    state: &mut TreeAciState<'_, T, V>,
    options: &TreeAciOptions<V>,
    order: EdgeOrder,
    sweeps: usize,
    operator: &mut F,
) -> Result<ArmOutcome>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let edge_count = state.edge_ranks.len();
    let mut rng = DeterministicRng::new(options.rng_seed);
    let mut edge_updates = 0usize;
    let mut evaluated_points = 0usize;
    for _ in 0..sweeps {
        let mut plan = match order {
            EdgeOrder::ContinuousWalk => continuous_walk_edges(&state.problem)?,
            EdgeOrder::RandomPermutation | EdgeOrder::EdgeIndex => (0..edge_count)
                .map(|edge| edge.checked_mul(2))
                .collect::<Option<Vec<_>>>()
                .ok_or(TreeAciError::SizeOverflow {
                    context: "edge-order directed edge identifier",
                })?,
        };
        if matches!(order, EdgeOrder::RandomPermutation) {
            shuffle(&mut plan, &mut rng);
        }
        for forward in plan {
            let evaluated = pivot_only_update(state, forward, options, operator)?;
            evaluated_points =
                evaluated_points
                    .checked_add(evaluated)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "edge-order evaluated point count",
                    })?;
            edge_updates = edge_updates
                .checked_add(1)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "edge-order update count",
                })?;
        }
    }

    let nesting = crate::validate::check_nesting(
        &state.problem,
        &state.sample_arena,
        &state.candidates,
        &state.pivots,
    )?;
    let held_out_error = held_out_error(state, operator)?;
    let pivot_error = crate::validate::check_interpolation_for_state(state, operator)?;
    Ok(ArmOutcome {
        edge_updates,
        evaluated_points,
        held_out_error,
        pivot_error,
        ranks: state.edge_ranks.clone(),
        nested_fraction: nesting.fraction(),
    })
}

fn continuous_walk_edges<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
) -> Result<Vec<DirectedEdgeId>> {
    let mut edges = Vec::new();
    for phase in &problem.schedule.forward {
        for path in &phase.paths {
            for step in &path.steps {
                let base = step.edge.checked_mul(2).ok_or(TreeAciError::SizeOverflow {
                    context: "continuous edge-order identifier",
                })?;
                let directed =
                    problem
                        .directed_edges
                        .get(base)
                        .ok_or(TreeAciError::InternalInvariant {
                            message: "continuous edge-order step references an unknown edge",
                        })?;
                let from =
                    problem
                        .node_order
                        .get(step.from)
                        .ok_or(TreeAciError::InternalInvariant {
                            message: "continuous edge-order step references an unknown node",
                        })?;
                if &directed.from == from {
                    edges.push(base);
                } else if &directed.to == from {
                    edges.push(directed.reverse);
                } else {
                    return Err(TreeAciError::InternalInvariant {
                        message: "continuous edge-order step has the wrong orientation",
                    });
                }
            }
        }
    }
    Ok(edges)
}

fn held_out_error<T, V, F>(state: &TreeAciState<'_, T, V>, operator: &mut F) -> Result<f64>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let skeleton = {
        let mut oracle =
            |point: &[usize]| crate::validate::evaluate_operator_at_point(state, point, operator);
        crate::skeleton::skeleton_tensors(
            &state.problem,
            &state.sample_arena,
            &state.pivots,
            &mut oracle,
        )?
    };
    let mut rng = DeterministicRng::new(0x8f3c_5a17_2d91_04b7);
    let mut maximum = 0.0_f64;
    for _ in 0..128 {
        let point = random_point(&state.problem, &mut rng);
        let expected = crate::validate::evaluate_operator_at_point(state, &point, operator)?;
        let actual = crate::skeleton::skeleton_evaluate(&skeleton, &state.problem, &point)?;
        maximum = maximum.max(tensor4all_core::Scalar::abs_val(expected - actual));
    }
    Ok(maximum)
}

fn random_point<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    rng: &mut DeterministicRng,
) -> Vec<usize> {
    problem
        .physical
        .iter()
        .map(|physical| rng.next_usize(physical.local_dim))
        .collect()
}

#[derive(Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 7;
        value ^= value >> 9;
        value ^= value << 8;
        self.state = value;
        value
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0);
        (self.next_u64() as usize) % bound
    }
}

fn shuffle<T>(values: &mut [T], rng: &mut DeterministicRng) {
    for index in (1..values.len()).rev() {
        let other = rng.next_usize(index + 1);
        values.swap(index, other);
    }
}

#[cfg(test)]
mod tests;
