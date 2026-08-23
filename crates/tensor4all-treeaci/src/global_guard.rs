//! Global floating-zone validation for locally sampled tree ACI sweeps.

use std::{collections::HashMap, mem::size_of};

use rand::{rngs::StdRng, Rng, SeedableRng};
use tensor4all_core::floating_zone_walk;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

use crate::{
    state::TreeAciState, Result, TreeAciError, TreeAciNode, TreeAciOptions, TreeAciScalar,
    TreeElementwiseBatch,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GlobalSearchReport {
    pub(crate) pivots: Vec<Vec<usize>>,
    pub(crate) evaluated_points: u64,
}

pub(crate) fn per_evaluator_message_cache_budget(
    total_budget: usize,
    input_count: usize,
) -> Result<usize> {
    let evaluator_count = input_count
        .checked_add(1)
        .ok_or(TreeAciError::SizeOverflow {
            context: "guard evaluator count",
        })?;
    Ok(total_budget / evaluator_count)
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
    let site_dims_bytes =
        site_dims
            .len()
            .checked_mul(size_of::<usize>())
            .ok_or(TreeAciError::SizeOverflow {
                context: "guard site-dimension bytes",
            })?;
    // Refuse the complete start/evaluation peak before allocating the nested
    // point vectors. Previously a caller could set a tiny working limit and
    // still allocate `nsearch * node_count` coordinates first.
    input_evaluators.enforce_guard_batch_budget_with_retained::<T>(nsearch, site_dims_bytes)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let starts = (0..nsearch)
        .map(|_| {
            site_dims
                .iter()
                .map(|dimension| rng.random_range(0..*dimension))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut output_evaluator = GuardOutputEvaluator::new(
        &state.output,
        &state.problem,
        per_evaluator_message_cache_budget(options.message_cache_max_bytes, state.inputs.len())?,
    )?;
    let mut evaluated_points = 0usize;
    let start_inputs = input_evaluators.evaluate::<T>(&starts)?;
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
    let start_storage_bytes = point_vector_storage_bytes(nsearch, site_dims.len())?;
    let mut candidate_storage_bytes = 0usize;
    for start in &starts {
        let (pivot, error) = floating_zone_walk(
            &site_dims,
            start,
            options.nsweeps_global_search,
            threshold,
            |points: &[Vec<usize>]| -> Result<Vec<f64>> {
                evaluated_points = checked_add_points(evaluated_points, points.len())?;
                let retained_bytes = site_dims_bytes
                    .checked_add(start_storage_bytes)
                    .and_then(|bytes| bytes.checked_add(candidate_storage_bytes))
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "guard retained search bytes",
                    })?;
                input_evaluators
                    .enforce_guard_batch_budget_with_retained::<T>(points.len(), retained_bytes)?;
                let coordinates = input_evaluators.expand_points(points)?;
                let input_values = input_evaluators.evaluate_expanded::<T>(points, &coordinates)?;
                let batch =
                    TreeElementwiseBatch::new(&input_values, state.inputs.len(), points.len())?;
                let mut target = vec![T::default(); points.len()];
                operator(batch, &mut target)?;
                let approximation =
                    output_evaluator.evaluate_expanded(input_evaluators, points, &coordinates)?;
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
            let entry_bytes = size_of::<(f64, Vec<usize>)>()
                .checked_add(pivot.len().checked_mul(size_of::<usize>()).ok_or(
                    TreeAciError::SizeOverflow {
                        context: "guard pivot coordinate bytes",
                    },
                )?)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "guard candidate bytes",
                })?;
            candidate_storage_bytes = candidate_storage_bytes.checked_add(entry_bytes).ok_or(
                TreeAciError::SizeOverflow {
                    context: "guard candidate bytes",
                },
            )?;
            let retained_bytes = site_dims_bytes
                .checked_add(start_storage_bytes)
                .and_then(|bytes| bytes.checked_add(candidate_storage_bytes))
                .ok_or(TreeAciError::SizeOverflow {
                    context: "guard retained search bytes",
                })?;
            crate::problem::enforce_limit(
                "working bytes",
                retained_bytes,
                input_evaluators.max_working_bytes,
            )?;
            candidates.push((error, pivot));
        }
    }
    drop(starts);
    drop(site_dims);
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
    points: &[Vec<usize>],
    growth_capacity: &[usize],
) -> Result<usize> {
    if growth_capacity.len() != state.edge_ranks.len() {
        return Err(TreeAciError::InternalInvariant {
            message: "global-pivot growth capacities differ from tree edge count",
        });
    }
    let checkpoint = state.sample_arena.checkpoint();
    let mut proposed_active = state.candidates.clone();
    let mut injected = 0usize;
    let mut growth = vec![0usize; state.edge_ranks.len()];
    let staged = (|| {
        for point in points {
            let activate_directed_cut = growth_capacity
                .iter()
                .zip(&growth)
                .flat_map(|(&capacity, &grown)| {
                    let active = grown < capacity;
                    [active, active]
                })
                .collect::<Vec<_>>();
            if !activate_directed_cut.iter().any(|active| *active) {
                break;
            }
            let report = state.sample_arena.inject_global_point_masked(
                &mut proposed_active,
                &state.problem,
                point,
                &activate_directed_cut,
            )?;
            if report.total_added > 0 {
                injected = injected.checked_add(1).ok_or(TreeAciError::SizeOverflow {
                    context: "injected global pivot count",
                })?;
                for (edge, added) in report.added_by_edge.as_chunks::<2>().0.iter().enumerate() {
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
        if growth
            .iter()
            .zip(growth_capacity)
            .any(|(&grown, &capacity)| grown > capacity)
        {
            return Err(TreeAciError::InternalInvariant {
                message: "global-pivot injection exceeded a cut's growth capacity",
            });
        }
        if injected == 0 {
            return Ok(None);
        }
        let proposed_output = pad_output_bonds(state, &growth)?;
        let proposed_frames =
            state
                .input_frames
                .extend(state.inputs, &state.problem, &state.sample_arena)?;
        Ok(Some((proposed_output, proposed_frames)))
    })();
    let Some((proposed_output, proposed_frames)) = (match staged {
        Ok(staged) => staged,
        Err(error) => {
            state.sample_arena.rollback(checkpoint)?;
            return Err(error);
        }
    }) else {
        state.sample_arena.rollback(checkpoint)?;
        return Ok(0);
    };
    state.output = proposed_output;
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
    let mut replacement_edges = Vec::with_capacity(growth.len());
    let mut replacement_indices = HashMap::with_capacity(growth.len());
    for (edge_number, &amount) in growth.iter().enumerate() {
        if amount == 0 {
            continue;
        }
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
        let new = DynIndex::new_dyn(dimension);
        replacement_indices.insert(old, new.clone());
        replacement_edges.push((graph_edge, new));
    }

    // Plan every affected core before allocating any of it. Sizing and
    // allocation used to share one loop, so the aggregate `max_working_bytes`
    // check only ran after every core was already allocated and retained: a
    // caller could set a small ceiling and still pay the full peak before being
    // told the request was rejected. Unaffected cores are deliberately omitted:
    // rebuilding them used to replace inactive bond identities and copy the
    // complete output even when the guard grew only one cut.
    // Holds the node name rather than its index: `TreeTN::node_index` returns
    // petgraph's `NodeIndex`, which treetn does not re-export, so naming it
    // here would mean depending on petgraph directly. Re-resolving the name is
    // a hash lookup and keeps the dependency boundary intact.
    struct PaddedCorePlan<V> {
        node: V,
        new_indices: Vec<DynIndex>,
        old_dims: Vec<usize>,
        new_strides: Vec<usize>,
        new_len: usize,
    }

    let mut plans = Vec::with_capacity(state.problem.node_order.len());
    let mut working_elements = 0usize;
    let mut largest_source_core = 0usize;
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
        if !old_indices
            .iter()
            .any(|index| replacement_indices.contains_key(index))
        {
            continue;
        }
        let new_indices = old_indices
            .iter()
            .map(|index| {
                replacement_indices
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| index.clone())
            })
            .collect::<Vec<_>>();
        let old_dims = old_indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        let new_dims = new_indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
        let old_len = old_dims.iter().try_fold(1usize, |product, &dimension| {
            product
                .checked_mul(dimension)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "global-pivot source core elements",
                })
        })?;
        let new_len = new_dims.iter().try_fold(1usize, |product, &dimension| {
            product
                .checked_mul(dimension)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "global-pivot padded core elements",
                })
        })?;
        let mut new_strides = Vec::with_capacity(new_dims.len());
        let mut new_stride = 1usize;
        for &dimension in &new_dims {
            new_strides.push(new_stride);
            new_stride = new_stride
                .checked_mul(dimension)
                .ok_or(TreeAciError::SizeOverflow {
                    context: "global-pivot padded core strides",
                })?;
        }
        if new_stride != new_len {
            return Err(TreeAciError::InternalInvariant {
                message: "global-pivot padded core length disagrees with its strides",
            });
        }
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
        largest_source_core = largest_source_core.max(old_len);
        plans.push(PaddedCorePlan {
            node: node.clone(),
            new_indices,
            old_dims,
            new_strides,
            new_len,
        });
    }

    // `to_vec` materializes one source core while all already-built padded
    // cores and the current destination allocation are live. Charge the
    // largest such source buffer in addition to the final padded payload.
    let peak_elements =
        working_elements
            .checked_add(largest_source_core)
            .ok_or(TreeAciError::SizeOverflow {
                context: "global-pivot padding peak elements",
            })?;
    let planned_bytes =
        peak_elements
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
        new_strides,
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
            for (&old_dim, &new_stride) in old_dims.iter().zip(&new_strides) {
                let coordinate = quotient % old_dim;
                quotient /= old_dim;
                let offset =
                    coordinate
                        .checked_mul(new_stride)
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "global-pivot padded core offset",
                        })?;
                new_linear = new_linear
                    .checked_add(offset)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "global-pivot padded core offset",
                    })?;
            }
            let slot = padded
                .get_mut(new_linear)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "global-pivot padded core offset is out of bounds",
                })?;
            *slot = value;
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
    for (edge, new) in &replacement_edges {
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
    index_count: usize,
    indices_per_node: Vec<usize>,
    local_dims: Vec<usize>,
    strides: Vec<Vec<usize>>,
    dims: Vec<Vec<usize>>,
    max_working_bytes: usize,
    node_order: Vec<V>,
}

#[cfg(test)]
pub(crate) mod input_evaluator_debug_stats {
    use std::cell::Cell;

    thread_local! {
        static CONSTRUCTIONS: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn record_construction() {
        CONSTRUCTIONS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn constructions() -> u64 {
        CONSTRUCTIONS.with(Cell::get)
    }

    pub(crate) fn reset() {
        CONSTRUCTIONS.with(|count| count.set(0));
    }
}

/// Returns the one logical site that varies across a floating-zone batch.
///
/// Single-point or multi-site batches do not provide enough scan structure and
/// deliberately fall back to the evaluator's ordinary center selection.
fn sole_varying_site(points: &[Vec<usize>]) -> Option<usize> {
    let first = points.first()?;
    let mut varying = None;
    for point in points.iter().skip(1) {
        if point.len() != first.len() {
            return None;
        }
        for (site, (left, right)) in first.iter().zip(point).enumerate() {
            if left == right {
                continue;
            }
            match varying {
                None => varying = Some(site),
                Some(known) if known == site => {}
                Some(_) => return None,
            }
        }
    }
    varying
}

impl<'a, V: TreeAciNode> InputEvaluators<'a, V> {
    #[cfg(test)]
    pub(crate) fn new(
        inputs: &'a [TreeTN<IdxTensor, V>],
        problem: &crate::problem::PreparedTreeProblem<V>,
    ) -> Result<Self> {
        Self::new_with_message_cache_max_bytes(inputs, problem, usize::MAX)
    }

    pub(crate) fn new_with_message_cache_max_bytes(
        inputs: &'a [TreeTN<IdxTensor, V>],
        problem: &crate::problem::PreparedTreeProblem<V>,
        message_cache_max_bytes: usize,
    ) -> Result<Self> {
        #[cfg(test)]
        input_evaluator_debug_stats::record_construction();
        let indices = problem
            .physical
            .iter()
            .flat_map(|physical| physical.indices.iter().cloned())
            .collect::<Vec<_>>();
        let options = CachedEvaluatorOptions {
            message_cache_max_bytes,
            ..CachedEvaluatorOptions::<V>::default()
        };
        let inputs = inputs
            .iter()
            .map(|input| TreeTNCachedEvaluator::new(input, &indices, options.clone()))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let index_count = indices.len();
        Ok(Self {
            inputs,
            index_count,
            indices_per_node: problem
                .physical
                .iter()
                .map(|physical| physical.indices.len())
                .collect(),
            local_dims: problem
                .physical
                .iter()
                .map(|physical| physical.local_dim)
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
            max_working_bytes: problem.max_working_bytes,
            node_order: problem.node_order.clone(),
        })
    }

    pub(crate) fn evaluate<T: TreeAciScalar>(&mut self, points: &[Vec<usize>]) -> Result<Vec<T>> {
        self.enforce_guard_batch_budget::<T>(points.len())?;
        let coordinates = self.expand_points(points)?;
        self.evaluate_expanded(points, &coordinates)
    }

    fn enforce_guard_batch_budget<T: TreeAciScalar>(&self, point_count: usize) -> Result<()> {
        self.enforce_guard_batch_budget_with_retained::<T>(point_count, 0)
    }

    fn enforce_guard_batch_budget_with_retained<T: TreeAciScalar>(
        &self,
        point_count: usize,
        retained_bytes: usize,
    ) -> Result<()> {
        let point_bytes = point_vector_storage_bytes(point_count, self.node_order.len())?;
        let coordinate_bytes = self
            .index_count
            .checked_mul(point_count)
            .and_then(|count| count.checked_mul(size_of::<usize>()))
            .ok_or(TreeAciError::SizeOverflow {
                context: "guard coordinate bytes",
            })?;
        let scalar_buffers = self
            .inputs
            .len()
            .checked_add(2)
            .and_then(|count| count.checked_mul(point_count))
            .and_then(|count| count.checked_mul(size_of::<T>()))
            .ok_or(TreeAciError::SizeOverflow {
                context: "guard scalar buffer bytes",
            })?;
        let evaluator_bytes = point_count
            .checked_mul(size_of::<tensor4all_core::AnyScalar>().max(size_of::<f64>()))
            .ok_or(TreeAciError::SizeOverflow {
                context: "guard evaluator buffer bytes",
            })?;
        let working_bytes = retained_bytes
            .checked_add(point_bytes)
            .and_then(|count| count.checked_add(coordinate_bytes))
            .and_then(|count| count.checked_add(scalar_buffers))
            .and_then(|count| count.checked_add(evaluator_bytes))
            .ok_or(TreeAciError::SizeOverflow {
                context: "guard working bytes",
            })?;
        crate::problem::enforce_limit("working bytes", working_bytes, self.max_working_bytes)
    }

    fn evaluate_expanded<T: TreeAciScalar>(
        &mut self,
        points: &[Vec<usize>],
        coordinates: &[usize],
    ) -> Result<Vec<T>> {
        let hint = self.evaluation_hint(points);
        let shape = [self.index_count, points.len()];
        let values = ColMajorArrayRef::new(coordinates, &shape).map_err(|error| {
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
        let mut result = vec![T::default(); result_elements];
        for (input_number, evaluator) in self.inputs.iter_mut().enumerate() {
            let evaluated = evaluator.evaluate_batched_with_hint(values, hint.clone())?;
            for (point, value) in evaluated.into_iter().enumerate() {
                result[input_number + input_count * point] = T::from_evaluated_scalar(value)
                    .map_err(|message| TreeAciError::ScalarKind {
                        message: message.into(),
                    })?;
            }
        }
        Ok(result)
    }

    fn evaluation_hint(&self, points: &[Vec<usize>]) -> EvaluationHint<V> {
        sole_varying_site(points)
            .and_then(|site| self.node_order.get(site).cloned())
            .map(EvaluationHint::around)
            .unwrap_or_default()
    }

    fn expand_points(&self, points: &[Vec<usize>]) -> Result<Vec<usize>> {
        let capacity =
            self.index_count
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
                let local_dim = self.local_dims[node];
                if coordinate >= local_dim {
                    return Err(TreeAciError::PhysicalCoordinateOutOfBounds {
                        node,
                        coordinate,
                        local_dim,
                    });
                }
                for (&stride, &dimension) in self.strides[node].iter().zip(&self.dims[node]) {
                    expanded.push((coordinate / stride) % dimension);
                }
            }
        }
        Ok(expanded)
    }
}

fn point_vector_storage_bytes(point_count: usize, node_count: usize) -> Result<usize> {
    let payload = point_count
        .checked_mul(node_count)
        .and_then(|count| count.checked_mul(size_of::<usize>()))
        .ok_or(TreeAciError::SizeOverflow {
            context: "guard point-coordinate bytes",
        })?;
    point_count
        .checked_mul(size_of::<Vec<usize>>())
        .and_then(|headers| headers.checked_add(payload))
        .ok_or(TreeAciError::SizeOverflow {
            context: "guard point-vector bytes",
        })
}

struct GuardOutputEvaluator<'a, V: TreeAciNode> {
    evaluator: TreeTNCachedEvaluator<'a, V>,
}

impl<'a, V: TreeAciNode> GuardOutputEvaluator<'a, V> {
    fn new(
        output: &'a TreeTN<IdxTensor, V>,
        problem: &crate::problem::PreparedTreeProblem<V>,
        message_cache_max_bytes: usize,
    ) -> Result<Self> {
        let indices = problem
            .physical
            .iter()
            .flat_map(|physical| physical.indices.iter().cloned())
            .collect::<Vec<_>>();
        let evaluator = TreeTNCachedEvaluator::new(
            output,
            &indices,
            CachedEvaluatorOptions {
                message_cache_max_bytes,
                ..CachedEvaluatorOptions::<V>::default()
            },
        )?;
        Ok(Self { evaluator })
    }

    #[cfg(test)]
    fn evaluate<T: TreeAciScalar>(
        &mut self,
        input_evaluators: &InputEvaluators<'_, V>,
        points: &[Vec<usize>],
    ) -> Result<Vec<T>> {
        input_evaluators.enforce_guard_batch_budget::<T>(points.len())?;
        let coordinates = input_evaluators.expand_points(points)?;
        self.evaluate_expanded(input_evaluators, points, &coordinates)
    }

    fn evaluate_expanded<T: TreeAciScalar>(
        &mut self,
        input_evaluators: &InputEvaluators<'_, V>,
        points: &[Vec<usize>],
        coordinates: &[usize],
    ) -> Result<Vec<T>> {
        let shape = [input_evaluators.index_count, points.len()];
        let values = ColMajorArrayRef::new(coordinates, &shape).map_err(|error| {
            TreeAciError::Numerical {
                message: error.to_string(),
            }
        })?;
        self.evaluator
            .evaluate_batched_with_hint(values, input_evaluators.evaluation_hint(points))?
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
