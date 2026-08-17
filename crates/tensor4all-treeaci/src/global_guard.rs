//! Global floating-zone validation for locally sampled tree ACI sweeps.

#![allow(dead_code)]

use std::mem::size_of;

use rand::{rngs::StdRng, Rng, SeedableRng};
use tensor4all_core::floating_zone_walk;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

use crate::{
    frames::InputFrameStore, state::TreeAciState, Result, TreeAciError, TreeAciNode,
    TreeAciOptions, TreeAciScalar, TreeElementwiseBatch,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlobalSearchReport {
    pub(crate) pivots: Vec<Vec<usize>>,
    pub(crate) evaluated_points: u64,
}

pub(crate) fn find_global_pivots<'a, T, V, F>(
    state: &TreeAciState<'a, T, V>,
    input_evaluators: &mut InputEvaluators<'a, V>,
    options: &TreeAciOptions<V>,
    seed: u64,
    operator: &mut F,
) -> Result<GlobalSearchReport>
where
    T: TreeAciScalar,
    V: TreeAciNode,
    F: for<'batch> FnMut(TreeElementwiseBatch<'batch, T>, &mut [T]) -> Result<()>,
{
    let nsearch = options.nsearch_global_pivots;
    let max_pivots = options.max_nglobal_pivots;
    if nsearch == 0 || max_pivots == 0 || state.problem.node_order.len() < 2 {
        return Ok(GlobalSearchReport {
            pivots: Vec::new(),
            evaluated_points: 0,
        });
    }
    let site_dims = state
        .problem
        .physical
        .iter()
        .map(|physical| physical.local_dim)
        .collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(seed);
    let starts = (0..nsearch)
        .map(|_| {
            site_dims
                .iter()
                .map(|dimension| rng.random_range(0..*dimension))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output_evaluator = GuardOutputEvaluator::new(&state.output, &state.problem)?;
    let mut evaluated_points = 0usize;
    let start_inputs = input_evaluators.evaluate::<T>(&starts, None)?;
    let start_batch = TreeElementwiseBatch::new(&start_inputs, state.inputs.len(), nsearch)?;
    let mut start_outputs = vec![T::default(); nsearch];
    operator(start_batch, &mut start_outputs)?;
    evaluated_points = checked_add_points(evaluated_points, nsearch)?;
    let max_output = start_outputs
        .iter()
        .copied()
        .map(tensor4all_core::Scalar::abs_val)
        .fold(0.0, f64::max);
    let absolute_tolerance = if options.scale_tolerance && max_output > 0.0 {
        options.tolerance * max_output
    } else {
        options.tolerance
    };
    let threshold = absolute_tolerance * options.global_tolerance_margin;

    let mut candidates = Vec::new();
    for start in &starts {
        let (pivot, error) = floating_zone_walk(
            &site_dims,
            start,
            options.nsweeps_global_search,
            threshold,
            |points: &[Vec<usize>]| -> Result<Vec<f64>> {
                evaluated_points = checked_add_points(evaluated_points, points.len())?;
                let input_values = input_evaluators.evaluate::<T>(points, None)?;
                let batch =
                    TreeElementwiseBatch::new(&input_values, state.inputs.len(), points.len())?;
                let mut target = vec![T::default(); points.len()];
                operator(batch, &mut target)?;
                let approximation = output_evaluator.evaluate(input_evaluators, points)?;
                Ok(target
                    .into_iter()
                    .zip(approximation)
                    .map(|(target, approximation)| {
                        tensor4all_core::Scalar::abs_val(target - approximation)
                    })
                    .collect())
            },
        )?;
        if error > threshold {
            candidates.push((error, pivot));
        }
    }
    candidates.sort_by(|(left, _), (right, _)| right.total_cmp(left));
    let mut pivots = Vec::new();
    for (_, point) in candidates {
        if !pivots.contains(&point) {
            pivots.push(point);
            if pivots.len() == max_pivots {
                break;
            }
        }
    }
    Ok(GlobalSearchReport {
        pivots,
        evaluated_points: u64::try_from(evaluated_points).map_err(|_| {
            TreeAciError::SizeOverflow {
                context: "global guard evaluated point count",
            }
        })?,
    })
}

pub(crate) fn inject_global_pivots<'a, T: TreeAciScalar, V: TreeAciNode>(
    state: &mut TreeAciState<'a, T, V>,
    _input_evaluators: &mut InputEvaluators<'a, V>,
    points: &[Vec<usize>],
    activate_edge: &[bool],
) -> Result<usize> {
    if activate_edge.len() != state.edge_ranks.len() {
        return Err(TreeAciError::InternalInvariant {
            message: "global-pivot activation mask differs from tree edge count",
        });
    }
    let activate_directed_cut = activate_edge
        .iter()
        .flat_map(|&activate| [activate, activate])
        .collect::<Vec<_>>();
    let mut proposed_arena = state.sample_arena.clone();
    let mut proposed_active = state.candidates.clone();
    let mut injected = 0usize;
    let mut growth = vec![0usize; state.edge_ranks.len()];
    for point in points {
        let report = proposed_arena.inject_global_point_masked(
            &mut proposed_active,
            &state.problem,
            point,
            &activate_directed_cut,
        )?;
        if report.total_added > 0 {
            injected = injected.checked_add(1).ok_or(TreeAciError::SizeOverflow {
                context: "injected global pivot count",
            })?;
            for (edge, added) in report.added_by_edge.chunks_exact(2).enumerate() {
                if added[0] != added[1] {
                    return Err(TreeAciError::InternalInvariant {
                        message: "global pivot changed only one direction of a tree cut",
                    });
                }
                growth[edge] =
                    growth[edge]
                        .checked_add(added[0])
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "global-pivot bond growth",
                        })?;
            }
        }
    }
    let proposed_output = pad_output_bonds(state, &growth)?;
    let proposed_frames =
        InputFrameStore::from_samples(state.inputs, &state.problem, &proposed_arena)?;
    state.output = proposed_output;
    state.sample_arena = proposed_arena;
    state.candidates = proposed_active;
    state.input_frames = proposed_frames;
    for (rank, added) in state.edge_ranks.iter_mut().zip(growth) {
        *rank = rank.checked_add(added).ok_or(TreeAciError::SizeOverflow {
            context: "global-pivot output rank",
        })?;
    }
    state.generation = state.candidates.generation;
    Ok(injected)
}

fn pad_output_bonds<T: TreeAciScalar, V: TreeAciNode>(
    state: &TreeAciState<'_, T, V>,
    growth: &[usize],
) -> Result<tensor4all_treetn::TreeTN<IdxTensor, V>> {
    if growth.iter().all(|&amount| amount == 0) {
        return Ok(state.output.clone());
    }
    let mut replacements = Vec::with_capacity(growth.len());
    for (edge_number, &amount) in growth.iter().enumerate() {
        let directed = &state.problem.directed_edges[2 * edge_number];
        let graph_edge = state
            .output
            .edge_between(&directed.from, &directed.to)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references a missing output edge",
            })?;
        let old = state
            .output
            .bond_index(graph_edge)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references an output edge without a bond",
            })?
            .clone();
        let dimension = old
            .dim()
            .checked_add(amount)
            .ok_or(TreeAciError::SizeOverflow {
                context: "global-pivot padded bond dimension",
            })?;
        replacements.push((graph_edge, old, DynIndex::new_dyn(dimension)));
    }

    // Plan every padded core before allocating any of it. Sizing and
    // allocation used to share one loop, so the aggregate `max_working_bytes`
    // check only ran after every core was already allocated and retained: a
    // caller could set a small ceiling and still pay the full peak before being
    // told the request was rejected.
    // Holds the node name rather than its index: `TreeTN::node_index` returns
    // petgraph's `NodeIndex`, which treetn does not re-export, so naming it
    // here would mean depending on petgraph directly. Re-resolving the name is
    // a hash lookup and keeps the dependency boundary intact.
    struct PaddedCorePlan<V> {
        node: V,
        new_indices: Vec<DynIndex>,
        old_dims: Vec<usize>,
        new_dims: Vec<usize>,
        new_len: usize,
    }

    let mut plans = Vec::with_capacity(state.problem.node_order.len());
    let mut working_elements = 0usize;
    for node in &state.problem.node_order {
        let node_index = state
            .output
            .node_index(node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references a missing output node",
            })?;
        let tensor = state
            .output
            .tensor(node_index)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references a missing output tensor",
            })?;
        let old_indices = tensor.indices();
        let new_indices = old_indices
            .iter()
            .map(|index| {
                replacements
                    .iter()
                    .find_map(|(_, old, new)| (old == index).then(|| new.clone()))
                    .unwrap_or_else(|| index.clone())
            })
            .collect::<Vec<_>>();
        let old_dims = old_indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        let new_dims = new_indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        let new_len = new_dims.iter().try_fold(1usize, |product, &dimension| {
            product
                .checked_mul(dimension)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "global-pivot padded core elements",
                })
        })?;
        if new_len > state.problem.max_core_elements {
            return Err(TreeAciError::ResourceLimit {
                resource: "core elements",
                requested: new_len,
                limit: state.problem.max_core_elements,
            });
        }
        working_elements =
            working_elements
                .checked_add(new_len)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "global-pivot padding working elements",
                })?;
        plans.push(PaddedCorePlan {
            node: node.clone(),
            new_indices,
            old_dims,
            new_dims,
            new_len,
        });
    }

    let planned_bytes =
        working_elements
            .checked_mul(size_of::<T>())
            .ok_or(TreeAciError::SizeOverflow {
                context: "global-pivot padding working bytes",
            })?;
    if planned_bytes > state.problem.max_working_bytes {
        return Err(TreeAciError::ResourceLimit {
            resource: "working bytes",
            requested: planned_bytes,
            limit: state.problem.max_working_bytes,
        });
    }

    let mut tensors = Vec::with_capacity(plans.len());
    for PaddedCorePlan {
        node,
        new_indices,
        old_dims,
        new_dims,
        new_len,
    } in plans
    {
        let node_index = state
            .output
            .node_index(&node)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references a missing output node",
            })?;
        let tensor = state
            .output
            .tensor(node_index)
            .ok_or(TreeAciError::InternalInvariant {
                message: "global-pivot padding references a missing output tensor",
            })?;
        let mut padded = vec![T::default(); new_len];
        let values = tensor
            .to_vec::<T>()
            .map_err(|error| TreeAciError::Numerical {
                message: error.to_string(),
            })?;
        for (old_linear, value) in values.into_iter().enumerate() {
            let mut quotient = old_linear;
            let mut new_linear = 0usize;
            let mut new_stride = 1usize;
            for (&old_dim, &new_dim) in old_dims.iter().zip(&new_dims) {
                let coordinate = quotient % old_dim;
                quotient /= old_dim;
                new_linear = new_linear.checked_add(coordinate * new_stride).ok_or(
                    TreeAciError::SizeOverflow {
                        context: "global-pivot padded core offset",
                    },
                )?;
                new_stride = new_stride
                    .checked_mul(new_dim)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "global-pivot padded core stride",
                    })?;
            }
            padded[new_linear] = value;
        }
        tensors.push((
            node_index,
            IdxTensor::from_dense(new_indices, padded).map_err(|error| {
                TreeAciError::Numerical {
                    message: error.to_string(),
                }
            })?,
        ));
    }
    let mut output = state.output.clone();
    for (edge, _, new) in &replacements {
        output.replace_edge_bond(*edge, new.clone())?;
    }
    for (node, tensor) in tensors {
        output.replace_tensor(node, tensor)?;
    }
    output.verify_internal_consistency()?;
    Ok(output)
}

pub(crate) struct InputEvaluators<'a, V: TreeAciNode> {
    inputs: Vec<TreeTNCachedEvaluator<'a, V>>,
    indices_per_node: Vec<usize>,
    strides: Vec<Vec<usize>>,
    dims: Vec<Vec<usize>>,
    evaluations: usize,
    max_working_bytes: usize,
    /// Prepared node order, so a varying coordinate maps back to a node name.
    node_order: Vec<V>,
}

/// The single node whose coordinate differs across `points`, if there is one.
///
/// A floating-zone scan varies one site and holds the rest fixed, so naming that
/// site lets the evaluator contract around it and compute each incoming message
/// once instead of once per scanned value. Zero or several varying nodes means
/// this is not a scan, and the caller falls back to the evaluator's own centre
/// selection.
fn sole_varying_node(points: &[Vec<usize>]) -> Option<usize> {
    let first = points.first()?;
    let mut varying = None;
    for point in points.iter().skip(1) {
        if point.len() != first.len() {
            return None;
        }
        for (site, (a, b)) in first.iter().zip(point).enumerate() {
            if a != b {
                match varying {
                    None => varying = Some(site),
                    Some(known) if known == site => {}
                    Some(_) => return None,
                }
            }
        }
    }
    varying
}

impl<'a, V: TreeAciNode> InputEvaluators<'a, V> {
    pub(crate) fn new(
        inputs: &'a [TreeTN<IdxTensor, V>],
        problem: &crate::problem::PreparedTreeProblem<V>,
    ) -> Result<Self> {
        let indices = problem
            .physical
            .iter()
            .flat_map(|physical| physical.indices.iter().cloned())
            .collect::<Vec<_>>();
        let options = CachedEvaluatorOptions::<V>::default();
        let inputs = inputs
            .iter()
            .map(|input| TreeTNCachedEvaluator::new(input, &indices, options.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            inputs,
            indices_per_node: problem
                .physical
                .iter()
                .map(|physical| physical.indices.len())
                .collect(),
            strides: problem
                .physical
                .iter()
                .map(|physical| physical.strides.clone())
                .collect(),
            dims: problem
                .physical
                .iter()
                .map(|physical| physical.dims.clone())
                .collect(),
            evaluations: 0,
            max_working_bytes: problem.max_working_bytes,
            node_order: problem.node_order.clone(),
        })
    }

    pub(crate) fn evaluations(&self) -> usize {
        self.evaluations
    }

    pub(crate) fn evaluate<T: TreeAciScalar>(
        &mut self,
        points: &[Vec<usize>],
        _split: Option<usize>,
    ) -> Result<Vec<T>> {
        // A floating-zone scan varies one site; contracting around it keeps
        // every incoming message constant across the batch. `_split` used to be
        // an unfilled seam here -- the parameter existed and every caller passed
        // `None` -- so the centre stayed wherever greedy search put it on the
        // first batch of the run.
        let hint = sole_varying_node(points)
            .and_then(|site| self.node_order.get(site).cloned())
            .map(EvaluationHint::around)
            .unwrap_or_default();
        let cold_evaluators = self
            .inputs
            .iter()
            .filter(|evaluator| evaluator.center().is_none())
            .count();
        let coordinates = self.expand_points(points)?;
        let n_indices = self.indices_per_node.iter().sum();
        let shape = [n_indices, points.len()];
        let values = ColMajorArrayRef::new(&coordinates, &shape).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?;
        let input_count = self.inputs.len();
        // Checked, and charged against the working budget: this buffer is
        // sized by the caller's batch, so it is the guard's largest transient
        // allocation and the one a small `max_working_bytes` is meant to stop.
        let result_elements =
            input_count
                .checked_mul(points.len())
                .ok_or(TreeAciError::SizeOverflow {
                    context: "guard input evaluation buffer",
                })?;
        let result_bytes =
            result_elements
                .checked_mul(size_of::<T>())
                .ok_or(TreeAciError::SizeOverflow {
                    context: "guard input evaluation bytes",
                })?;
        crate::problem::enforce_limit("working bytes", result_bytes, self.max_working_bytes)?;
        let mut result = vec![T::default(); result_elements];
        for (input_number, evaluator) in self.inputs.iter_mut().enumerate() {
            for (point, value) in evaluator
                .evaluate_batched_with_hint(values, hint.clone())?
                .into_iter()
                .enumerate()
            {
                result[input_number + input_count * point] = T::from_evaluated_scalar(value)
                    .map_err(|message| TreeAciError::ScalarKind {
                        message: message.into(),
                    })?;
            }
        }
        self.evaluations = self
            .evaluations
            .checked_add(cold_evaluators.checked_mul(points.len()).ok_or(
                TreeAciError::SizeOverflow {
                    context: "global guard evaluator count",
                },
            )?)
            .ok_or(TreeAciError::SizeOverflow {
                context: "global guard evaluator count",
            })?;
        Ok(result)
    }

    fn expand_points(&self, points: &[Vec<usize>]) -> Result<Vec<usize>> {
        let n_indices = self.indices_per_node.iter().sum::<usize>();
        let capacity = n_indices
            .checked_mul(points.len())
            .ok_or(TreeAciError::SizeOverflow {
                context: "global guard coordinate batch",
            })?;
        let mut expanded = Vec::with_capacity(capacity);
        for point in points {
            if point.len() != self.indices_per_node.len() {
                return Err(TreeAciError::PointLengthMismatch {
                    expected: self.indices_per_node.len(),
                    actual: point.len(),
                });
            }
            for (node, coordinate) in point.iter().copied().enumerate() {
                for (&stride, &dimension) in self.strides[node].iter().zip(&self.dims[node]) {
                    expanded.push((coordinate / stride) % dimension);
                }
            }
        }
        Ok(expanded)
    }
}

struct GuardOutputEvaluator<'a, V: TreeAciNode> {
    evaluator: TreeTNCachedEvaluator<'a, V>,
}

impl<'a, V: TreeAciNode> GuardOutputEvaluator<'a, V> {
    fn new(
        output: &'a TreeTN<IdxTensor, V>,
        problem: &crate::problem::PreparedTreeProblem<V>,
    ) -> Result<Self> {
        let indices = problem
            .physical
            .iter()
            .flat_map(|physical| physical.indices.iter().cloned())
            .collect::<Vec<_>>();
        let evaluator =
            TreeTNCachedEvaluator::new(output, &indices, CachedEvaluatorOptions::<V>::default())?;
        Ok(Self { evaluator })
    }

    fn evaluate<T: TreeAciScalar>(
        &mut self,
        input_evaluators: &InputEvaluators<'_, V>,
        points: &[Vec<usize>],
    ) -> Result<Vec<T>> {
        let coordinates = input_evaluators.expand_points(points)?;
        let shape = [input_evaluators.indices_per_node.iter().sum(), points.len()];
        let values = ColMajorArrayRef::new(&coordinates, &shape).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?;
        self.evaluator
            .evaluate_batched(values)?
            .into_iter()
            .map(|value| {
                T::from_evaluated_scalar(value).map_err(|message| TreeAciError::ScalarKind {
                    message: message.into(),
                })
            })
            .collect()
    }
}

fn checked_add_points(current: usize, additional: usize) -> Result<usize> {
    current
        .checked_add(additional)
        .ok_or(TreeAciError::SizeOverflow {
            context: "global guard evaluated point count",
        })
}

#[cfg(test)]
mod tests;
