//! Exact per-input contractions for immutable directed component samples.

// INVARIANT: frame state is consumed by the upcoming local-update engine; it
// remains crate-private while that engine is staged.
#![allow(dead_code)]

use std::mem::size_of;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use crate::{
    problem::{DirectedEdgeId, PreparedTreeProblem},
    samples::{ComponentSample, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciScalar,
};

/// Test-only counter of `contract_prepared_core` invocations via the
/// memoized `FrameBuilder::compute` path, used to prove
/// `InputFrameStore::extend` recomputes only newly interned samples (see
/// `frames::tests::extend_recomputes_only_the_newly_interned_samples`).
#[cfg(test)]
pub(crate) mod debug_stats {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static COMPUTE_CALLS: AtomicU64 = AtomicU64::new(0);

    pub(crate) fn reset() {
        COMPUTE_CALLS.store(0, Ordering::Relaxed);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct DirectedFrame<T> {
    pub(crate) sample_count: usize,
    pub(crate) bond_dim: usize,
    pub(crate) sample_ids: Vec<SampleId>,
    pub(crate) values: Matrix<T>,
}

impl<T: TreeAciScalar> DirectedFrame<T> {
    fn row(&self, sample: SampleId) -> Vec<T> {
        (0..self.bond_dim)
            .map(|bond| self.values[[sample, bond]])
            .collect()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct InputFrameStore<T> {
    pub(crate) frames: Vec<Vec<DirectedFrame<T>>>,
    cores: Vec<Vec<PreparedCore<T>>>,
    /// Number of retained directed frames, across every input and edge.
    records: usize,
    /// Logical payload bytes retained by those frames.
    ///
    /// The cache's own accounting, not an allocator or process measurement:
    /// `sample_count * bond_dim * size_of::<T>()` summed over what is retained.
    retained_bytes: usize,
}

#[derive(Clone, Debug)]
struct PreparedCore<T> {
    indices: Vec<DynIndex>,
    dims: Vec<usize>,
    strides: Vec<usize>,
    values: Vec<T>,
}

impl<T: TreeAciScalar> InputFrameStore<T> {
    pub(crate) fn from_samples<V: TreeAciNode>(
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        arena: &SampleArena,
    ) -> Result<Self> {
        Self::build_or_extend(inputs, problem, arena, None)
    }

    /// Extends this store to cover every sample now retained by `arena`,
    /// reusing every already-computed frame instead of recomputing it.
    ///
    /// `SampleArena` is append-only and its `SampleId`s are immutable (see
    /// `samples.rs`): a sample already interned when this store was built
    /// names exactly the same component forever. Only samples interned since
    /// then need a fresh `contract_prepared_core` call. This is the fix for
    /// the root cause in
    /// `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s update
    /// on `commit_edge_proposal`: that call site previously discarded this
    /// store and rebuilt every sample on every directed edge from scratch
    /// after every single-edge commit, `O(edges)` work repeated `O(edges)`
    /// times per sweep.
    pub(crate) fn extend<V: TreeAciNode>(
        &self,
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        arena: &SampleArena,
    ) -> Result<Self> {
        Self::build_or_extend(inputs, problem, arena, Some(self))
    }

    fn build_or_extend<V: TreeAciNode>(
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        arena: &SampleArena,
        existing: Option<&Self>,
    ) -> Result<Self> {
        let mut all_inputs = Vec::with_capacity(inputs.len());
        let mut all_cores = Vec::with_capacity(inputs.len());
        // `max_frame_elements` bounds one frame; this cache keeps one per input
        // per directed edge, so without an aggregate the retained total grows as
        // inputs x directed_edges x that per-frame ceiling. Accumulated and
        // checked before each allocation, so an over-budget run is refused
        // rather than reaching the ceiling first.
        let mut retained_bytes = 0usize;
        let mut records = 0usize;
        for (input_index, input) in inputs.iter().enumerate() {
            let existing_input = existing.and_then(|store| store.frames.get(input_index));
            let cores = match existing.and_then(|store| store.cores.get(input_index)) {
                Some(cores) => cores.clone(),
                None => prepare_cores::<T, V>(input, problem)?,
            };
            let memo = problem
                .directed_edges
                .iter()
                .enumerate()
                .map(|(edge, _)| {
                    let count = arena.directed_record_count(edge)?;
                    let mut samples = vec![None; count];
                    if let Some(previous) = existing_input.and_then(|frames| frames.get(edge)) {
                        for (sample, slot) in
                            samples.iter_mut().enumerate().take(previous.sample_count)
                        {
                            *slot = Some(previous.row(sample));
                        }
                    }
                    Ok(samples)
                })
                .collect::<Result<Vec<_>>>()?;
            let mut builder = FrameBuilder {
                input,
                problem,
                arena,
                cores,
                memo,
            };
            for edge in 0..problem.directed_edges.len() {
                let known = existing_input
                    .and_then(|frames| frames.get(edge))
                    .map(|frame| frame.sample_count)
                    .unwrap_or(0);
                for sample in known..builder.memo[edge].len() {
                    builder.compute(edge, sample)?;
                }
            }
            let mut input_frames = Vec::with_capacity(problem.directed_edges.len());
            let bond_dims = (0..problem.directed_edges.len())
                .map(|edge| builder.outgoing_bond(edge).map(IndexLike::dim))
                .collect::<Result<Vec<_>>>()?;
            let memo = std::mem::take(&mut builder.memo);
            for (edge, samples) in memo.into_iter().enumerate() {
                let sample_count = samples.len();
                let bond_dim = bond_dims[edge];
                let elements =
                    sample_count
                        .checked_mul(bond_dim)
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "directed frame elements",
                        })?;
                if elements > problem.max_frame_elements {
                    return Err(TreeAciError::ResourceLimit {
                        resource: "frame elements",
                        requested: elements,
                        limit: problem.max_frame_elements,
                    });
                }
                let frame_bytes =
                    elements
                        .checked_mul(size_of::<T>())
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "directed frame bytes",
                        })?;
                retained_bytes =
                    retained_bytes
                        .checked_add(frame_bytes)
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "retained frame bytes",
                        })?;
                if retained_bytes > problem.max_frame_bytes {
                    return Err(TreeAciError::ResourceLimit {
                        resource: "frame bytes",
                        requested: retained_bytes,
                        limit: problem.max_frame_bytes,
                    });
                }
                records = records.checked_add(1).ok_or(TreeAciError::SizeOverflow {
                    context: "retained frame count",
                })?;
                let mut data = vec![T::default(); elements];
                for (sample, values) in samples.into_iter().enumerate() {
                    let values = values.ok_or(TreeAciError::InternalInvariant {
                        message: "directed frame memoization left a sample uncomputed",
                    })?;
                    if values.len() != bond_dim {
                        return Err(TreeAciError::InternalInvariant {
                            message: "computed frame length differs from cut bond dimension",
                        });
                    }
                    for (bond, value) in values.into_iter().enumerate() {
                        data[sample + sample_count * bond] = value;
                    }
                }
                input_frames.push(DirectedFrame {
                    sample_count,
                    bond_dim,
                    sample_ids: (0..sample_count).collect(),
                    values: Matrix::from_col_major_vec(sample_count, bond_dim, data),
                });
            }
            all_inputs.push(input_frames);
            all_cores.push(builder.cores);
        }
        Ok(Self {
            frames: all_inputs,
            cores: all_cores,
            records,
            retained_bytes,
        })
    }

    /// Number of retained directed frames.
    pub(crate) fn records(&self) -> usize {
        self.records
    }

    /// Logical payload bytes retained by this cache.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn frame_values(
        &self,
        input: usize,
        directed_edge: DirectedEdgeId,
        sample: SampleId,
    ) -> Result<Vec<T>> {
        let frame = self
            .frames
            .get(input)
            .and_then(|edges| edges.get(directed_edge))
            .ok_or(TreeAciError::InternalInvariant {
                message: "frame lookup references an unknown input or directed edge",
            })?;
        if sample >= frame.sample_count || frame.sample_ids.get(sample) != Some(&sample) {
            return Err(TreeAciError::InternalInvariant {
                message: "frame lookup references an unknown immutable sample ID",
            });
        }
        Ok((0..frame.bond_dim)
            .map(|bond| frame.values[[sample, bond]])
            .collect())
    }

    pub(crate) fn candidate_frame<V: TreeAciNode>(
        &self,
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        sample: &ComponentSample,
    ) -> Result<Vec<T>> {
        let tree = inputs.get(input).ok_or(TreeAciError::InternalInvariant {
            message: "candidate frame references an unknown input",
        })?;
        let cores = self
            .cores
            .get(input)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate frame has no prepared input cores",
            })?;
        let incoming = sample
            .incoming
            .iter()
            .map(|&(edge, id)| {
                self.frame_values(input, edge, id)
                    .map(|values| (edge, values))
            })
            .collect::<Result<Vec<_>>>()?;
        contract_prepared_core(
            tree,
            problem,
            cores,
            directed_edge,
            sample.local_coordinate,
            &incoming,
        )
    }
}

struct FrameBuilder<'a, T, V>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    input: &'a TreeTN<IdxTensor, V>,
    problem: &'a PreparedTreeProblem<V>,
    arena: &'a SampleArena,
    cores: Vec<PreparedCore<T>>,
    memo: Vec<Vec<Option<Vec<T>>>>,
}

impl<T: TreeAciScalar, V: TreeAciNode> FrameBuilder<'_, T, V> {
    fn compute(&mut self, edge: DirectedEdgeId, sample: SampleId) -> Result<Vec<T>> {
        if let Some(values) = self
            .memo
            .get(edge)
            .and_then(|samples| samples.get(sample))
            .and_then(Clone::clone)
        {
            return Ok(values);
        }
        #[cfg(test)]
        debug_stats::COMPUTE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let record = self.arena.record(edge, sample)?.clone();
        let mut incoming_frames = Vec::with_capacity(record.incoming.len());
        for &(incoming_edge, incoming_sample) in &record.incoming {
            incoming_frames.push((incoming_edge, self.compute(incoming_edge, incoming_sample)?));
        }
        let values = contract_prepared_core(
            self.input,
            self.problem,
            &self.cores,
            edge,
            record.local_coordinate,
            &incoming_frames,
        )?;
        let slot = self
            .memo
            .get_mut(edge)
            .and_then(|samples| samples.get_mut(sample))
            .ok_or(TreeAciError::InternalInvariant {
                message: "computed frame has no memoization slot",
            })?;
        *slot = Some(values.clone());
        Ok(values)
    }

    fn outgoing_bond(&self, edge: DirectedEdgeId) -> Result<&DynIndex> {
        let edge =
            self.problem
                .directed_edges
                .get(edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "frame references an unknown directed edge",
                })?;
        let graph_edge = self.input.edge_between(&edge.from, &edge.to).ok_or(
            TreeAciError::InternalInvariant {
                message: "prepared input is missing a directed cut bond",
            },
        )?;
        self.input
            .bond_index(graph_edge)
            .ok_or(TreeAciError::InternalInvariant {
                message: "prepared input edge is missing its bond index",
            })
    }
}

fn contract_prepared_core<T: TreeAciScalar, V: TreeAciNode>(
    input: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    cores: &[PreparedCore<T>],
    edge: DirectedEdgeId,
    local_coordinate: usize,
    incoming_frames: &[(DirectedEdgeId, Vec<T>)],
) -> Result<Vec<T>> {
    let directed = &problem.directed_edges[edge];
    let node =
        *problem
            .node_positions
            .get(&directed.from)
            .ok_or(TreeAciError::InternalInvariant {
                message: "frame source has no prepared node position",
            })?;
    let core = &cores[node];
    let outgoing = outgoing_bond(input, problem, edge)?;
    let outgoing_axis = axis_of(&core.indices, outgoing)?;
    let physical = &problem.physical[node];
    let physical_axes = physical
        .indices
        .iter()
        .map(|index| axis_of(&core.indices, index))
        .collect::<Result<Vec<_>>>()?;
    let mut incoming_axes = Vec::with_capacity(incoming_frames.len());
    for (incoming_edge, values) in incoming_frames {
        let incoming_bond = outgoing_bond(input, problem, *incoming_edge)?;
        if values.len() != incoming_bond.dim() {
            return Err(TreeAciError::InternalInvariant {
                message: "incoming frame length differs from its bond dimension",
            });
        }
        incoming_axes.push((axis_of(&core.indices, incoming_bond)?, values));
    }

    // Fix the physical axes once via direct offset arithmetic, instead of
    // scanning every element of the core (including every other physical
    // value) and discarding the ones that do not match. This is the fix for
    // the root cause in `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s
    // "Update" section: `contract_prepared_core` was measured to be 96.7% of
    // a full tree ACI run's wall time at chi=128, visiting 4.99 billion
    // elements via a per-element `axis_coordinate` divmod even though only
    // `outgoing.dim() * product(incoming dims)` elements are ever used.
    let mut base_offset = 0usize;
    for (physical_axis, &axis) in physical_axes.iter().enumerate() {
        let wanted =
            (local_coordinate / physical.strides[physical_axis]) % physical.dims[physical_axis];
        base_offset += wanted * core.strides[axis];
    }
    let outgoing_stride = core.strides[outgoing_axis];

    let mut result = vec![T::default(); outgoing.dim()];
    for (outgoing_value, slot) in result.iter_mut().enumerate() {
        let outgoing_offset = base_offset + outgoing_value * outgoing_stride;
        *slot = accumulate_incoming(core, &incoming_axes, 0, outgoing_offset);
    }
    Ok(result)
}

/// Sums `core.values[offset]` over the cartesian product of `incoming_axes`'
/// values, each axis contracted with its frame vector, without ever touching
/// an element the physical/outgoing fixing above did not select.
fn accumulate_incoming<T: TreeAciScalar>(
    core: &PreparedCore<T>,
    incoming_axes: &[(usize, &Vec<T>)],
    axis_index: usize,
    offset: usize,
) -> T {
    let Some(&(axis, values)) = incoming_axes.get(axis_index) else {
        return core.values[offset];
    };
    let stride = core.strides[axis];
    let mut sum = T::default();
    for (value_index, &value) in values.iter().enumerate() {
        sum = sum
            + value
                * accumulate_incoming(
                    core,
                    incoming_axes,
                    axis_index + 1,
                    offset + value_index * stride,
                );
    }
    sum
}

fn outgoing_bond<'a, V: TreeAciNode>(
    input: &'a TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    edge: DirectedEdgeId,
) -> Result<&'a DynIndex> {
    let edge = problem
        .directed_edges
        .get(edge)
        .ok_or(TreeAciError::InternalInvariant {
            message: "frame references an unknown directed edge",
        })?;
    let graph_edge =
        input
            .edge_between(&edge.from, &edge.to)
            .ok_or(TreeAciError::InternalInvariant {
                message: "prepared input is missing a directed cut bond",
            })?;
    input
        .bond_index(graph_edge)
        .ok_or(TreeAciError::InternalInvariant {
            message: "prepared input edge is missing its bond index",
        })
}

fn prepare_cores<T: TreeAciScalar, V: TreeAciNode>(
    input: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
) -> Result<Vec<PreparedCore<T>>> {
    problem
        .node_order
        .iter()
        .map(|node| {
            let node_index = input
                .node_index(node)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "frame input is missing a prepared node",
                })?;
            let tensor = input
                .tensor(node_index)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "frame input is missing a prepared core",
                })?;
            let indices = tensor.indices().to_vec();
            let dims = indices.iter().map(IndexLike::dim).collect::<Vec<_>>();
            let mut strides = Vec::with_capacity(dims.len());
            let mut stride = 1usize;
            for dim in &dims {
                strides.push(stride);
                stride = stride.checked_mul(*dim).ok_or(TreeAciError::SizeOverflow {
                    context: "prepared core strides",
                })?;
            }
            let values = tensor
                .to_vec::<T>()
                .map_err(|error| TreeAciError::ScalarKind {
                    message: error.to_string(),
                })?;
            Ok(PreparedCore {
                indices,
                dims,
                strides,
                values,
            })
        })
        .collect()
}

fn axis_of(indices: &[DynIndex], target: &DynIndex) -> Result<usize> {
    indices
        .iter()
        .position(|index| index == target)
        .ok_or(TreeAciError::InternalInvariant {
            message: "prepared core is missing a required full-equality index",
        })
}

#[cfg(test)]
mod tests;
