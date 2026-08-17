use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::TreeTN;

use crate::{
    problem::PreparedTreeProblem,
    samples::{CandidateSets, PivotPairs, SampleArena},
    skeleton::{skeleton_evaluate, skeleton_tensors},
    state::TreeAciState,
    Result, TreeAciError, TreeAciNode, TreeAciScalar, TreeElementwiseBatch,
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NestingReport {
    pub(crate) nested: Vec<bool>,
}

impl NestingReport {
    pub(crate) fn fraction(&self) -> f64 {
        if self.nested.is_empty() {
            return 1.0;
        }
        self.nested.iter().filter(|nested| **nested).count() as f64 / self.nested.len() as f64
    }
}

pub(crate) fn check_nesting<V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    arena: &SampleArena,
    candidates: &CandidateSets,
    pivots: &PivotPairs,
) -> Result<NestingReport> {
    if candidates.ids.len() != problem.directed_edges.len()
        || pivots.per_edge.len() * 2 != problem.directed_edges.len()
    {
        return Err(TreeAciError::InternalInvariant {
            message: "nesting state dimensions differ from prepared topology",
        });
    }
    let mut nested = vec![true; problem.directed_edges.len()];
    for (directed, is_nested) in nested.iter_mut().enumerate() {
        let edge_number = directed / 2;
        let ids = if directed == 2 * edge_number {
            pivots.forward_ids(edge_number)
        } else {
            pivots.reverse_ids(edge_number)
        };
        for sample in ids {
            let record = arena.record(directed, sample)?;
            if record
                .incoming
                .iter()
                .any(|&(incoming, child)| !candidates.ids[incoming].contains(&child))
            {
                *is_nested = false;
                break;
            }
        }
    }
    Ok(NestingReport { nested })
}

pub(crate) fn check_interpolation_for_state<T, V, F>(
    state: &TreeAciState<'_, T, V>,
    operator: &mut F,
) -> Result<f64>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let skeleton = {
        let mut oracle = |point: &[usize]| evaluate_operator_at_point(state, point, operator);
        skeleton_tensors(
            &state.problem,
            &state.sample_arena,
            &state.pivots,
            &mut oracle,
        )?
    };
    let mut maximum = 0.0_f64;
    for (edge_number, pairs) in state.pivots.per_edge.iter().enumerate() {
        let forward = 2 * edge_number;
        for &(left, right) in pairs {
            let point = state.sample_arena.materialize_global_point(
                &state.problem,
                forward,
                left,
                right,
            )?;
            let expected = evaluate_operator_at_point(state, &point, operator)?;
            let actual = skeleton_evaluate(&skeleton, &state.problem, &point)?;
            maximum = maximum.max(tensor4all_core::Scalar::abs_val(expected - actual));
        }
    }
    Ok(maximum)
}

pub(crate) fn check_gauge_equivalence<T, V, F>(
    state: &TreeAciState<'_, T, V>,
    operator: &mut F,
) -> Result<f64>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let skeleton = {
        let mut oracle = |point: &[usize]| evaluate_operator_at_point(state, point, operator);
        skeleton_tensors(
            &state.problem,
            &state.sample_arena,
            &state.pivots,
            &mut oracle,
        )?
    };
    let mut seed = 0x4d595df4d0f33173_u64;
    let mut maximum = 0.0_f64;
    for _ in 0..64 {
        let mut point = Vec::with_capacity(state.problem.node_order.len());
        for physical in &state.problem.physical {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            point.push((seed as usize) % physical.local_dim);
        }
        let stored: T = evaluate_tree_at_point(&state.output, &state.problem, &point)?;
        let expected = skeleton_evaluate(&skeleton, &state.problem, &point)?;
        maximum = maximum.max(tensor4all_core::Scalar::abs_val(stored - expected));
    }
    Ok(maximum)
}

pub(crate) fn evaluate_operator_at_point<T, V, F>(
    state: &TreeAciState<'_, T, V>,
    point: &[usize],
    operator: &mut F,
) -> Result<T>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let mut values = Vec::with_capacity(state.inputs.len());
    for input in state.inputs {
        values.push(evaluate_tree_at_point(input, &state.problem, point)?);
    }
    let batch = TreeElementwiseBatch::new(&values, state.inputs.len(), 1)?;
    let mut output = vec![T::default(); 1];
    operator(batch, &mut output)?;
    Ok(output[0])
}

fn evaluate_tree_at_point<T, V>(
    tree: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    point: &[usize],
) -> Result<T>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    if point.len() != problem.node_order.len() {
        return Err(TreeAciError::PointLengthMismatch {
            expected: problem.node_order.len(),
            actual: point.len(),
        });
    }
    let mut indices = Vec::<DynIndex>::new();
    let mut coordinates = Vec::new();
    for (node, (&local, physical)) in point.iter().zip(&problem.physical).enumerate() {
        if local >= physical.local_dim {
            return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
                node,
                coordinate: local,
                local_dim: physical.local_dim,
            });
        }
        indices.extend(physical.indices.iter().cloned());
        for (&stride, &dimension) in physical.strides.iter().zip(&physical.dims) {
            coordinates.push((local / stride) % dimension);
        }
    }
    let value = tree.evaluate_point(&indices, &coordinates)?;
    T::from_evaluated_scalar(value).map_err(|message| TreeAciError::ScalarKind {
        message: message.into(),
    })
}

#[cfg(test)]
mod tests;
