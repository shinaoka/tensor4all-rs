//! Exact per-input contractions for immutable directed component samples.

// INVARIANT: frame state is consumed by the upcoming local-update engine; it
// remains crate-private while that engine is staged.
#![allow(dead_code)]

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};
use std::mem::size_of;
use std::rc::Rc;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use crate::{
    problem::{DirectedEdge, DirectedEdgeId, PreparedTreeProblem},
    samples::{ComponentSample, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciScalar,
};

/// Test-only counter of `contract_prepared_core` invocations via the
/// memoized `FrameBuilder::compute` path, used to prove
/// `InputFrameStore::extend` recomputes only newly interned samples (see
/// `frames::tests::extend_recomputes_only_the_newly_interned_samples`).
///
/// `thread_local!`, not a process-global `static`: Rust's default test
/// harness runs each `#[test]` fn on its own thread, so a `static` counter
/// is shared -- and raced on -- by every test in the binary that happens to
/// execute concurrently and touch this code path, not just the one test
/// that means to read it.
#[cfg(test)]
pub(crate) mod debug_stats {
    use std::cell::Cell;

    thread_local! {
        static COMPUTE_CALLS: Cell<u64> = const { Cell::new(0) };
        static SCALAR_COMPUTE_CALLS: Cell<u64> = const { Cell::new(0) };
        static BATCHED_COMPUTE_CALLS: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn record_scalar_compute_call() {
        COMPUTE_CALLS.with(|count| count.set(count.get() + 1));
        SCALAR_COMPUTE_CALLS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn record_batched_compute_call() {
        COMPUTE_CALLS.with(|count| count.set(count.get() + 1));
        BATCHED_COMPUTE_CALLS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn compute_calls() -> u64 {
        COMPUTE_CALLS.with(Cell::get)
    }

    pub(crate) fn scalar_compute_calls() -> u64 {
        SCALAR_COMPUTE_CALLS.with(Cell::get)
    }

    pub(crate) fn batched_compute_calls() -> u64 {
        BATCHED_COMPUTE_CALLS.with(Cell::get)
    }

    pub(crate) fn reset() {
        COMPUTE_CALLS.with(|count| count.set(0));
        SCALAR_COMPUTE_CALLS.with(|count| count.set(0));
        BATCHED_COMPUTE_CALLS.with(|count| count.set(0));
    }
}

/// Test-only hit/miss counters for the candidate-frame cache, used to prove
/// repeated candidate lookups actually hit the cache (see
/// `frames::tests::candidate_frame_hits_the_cache_on_a_repeated_lookup`).
/// `thread_local!` for the same reason as `debug_stats` above.
#[cfg(test)]
pub(crate) mod candidate_debug_stats {
    use std::cell::Cell;

    thread_local! {
        static HITS: Cell<u64> = const { Cell::new(0) };
        static MISSES: Cell<u64> = const { Cell::new(0) };
    }

    pub(crate) fn record_hit() {
        HITS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn record_miss() {
        MISSES.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn hits() -> u64 {
        HITS.with(Cell::get)
    }

    pub(crate) fn misses() -> u64 {
        MISSES.with(Cell::get)
    }

    pub(crate) fn reset() {
        HITS.with(|count| count.set(0));
        MISSES.with(|count| count.set(0));
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

/// Identifies a pivot-search candidate's frame independent of whether it is
/// (yet) interned in a `SampleArena`: which input, which directed cut, the
/// node's own physical value, and the exact ordered set of incoming-edge
/// frame identities it was built from.
type CandidateCacheKey = (
    usize,
    DirectedEdgeId,
    usize,
    Vec<(DirectedEdgeId, SampleId)>,
);

#[derive(Clone, Debug)]
pub(crate) struct InputFrameStore<T> {
    pub(crate) frames: Vec<Vec<Rc<DirectedFrame<T>>>>,
    cores: Vec<Rc<Vec<PreparedCore<T>>>>,
    /// Number of retained directed frames, across every input and edge.
    records: usize,
    /// Logical payload bytes retained by those frames.
    ///
    /// The cache's own accounting, not an allocator or process measurement:
    /// `sample_count * bond_dim * size_of::<T>()` summed over what is retained.
    retained_bytes: usize,
    /// Memoized `candidate_frame` results, keyed by candidate identity.
    ///
    /// Unlike `frames`, these candidates are usually never interned into a
    /// `SampleArena` (most are proposed, not selected, by one pivot search),
    /// so they cannot ride the arena's own deduplication. Persisted across
    /// `extend` calls (i.e. across the whole run, not just one local update)
    /// because the same candidate identity recurs across sweeps and across
    /// neighbouring edges once ranks stabilize -- see
    /// `docs/worklogs/2026-08-18-treeaci-message-cache-prototype.md`'s
    /// second #646 continuation for the measured duplication rate (45-65%
    /// of calls). Shares `retained_bytes`'s budget against
    /// `PreparedTreeProblem::max_frame_bytes`: once the combined total would
    /// exceed it, new candidates are still computed but simply not cached,
    /// rather than evicting or erroring.
    ///
    /// `Rc`-shared rather than deep-cloned on `extend`: `extend` runs once
    /// per directed-edge commit, so a deep clone here would reintroduce the
    /// same `O(edges)`-work-repeated-`O(edges)`-times shape this file's
    /// `extend` was written to eliminate for `frames`, just relocated to the
    /// candidate cache instead. An initial deep-clone version was measured
    /// to be a net regression at chi=128 for exactly this reason; see the
    /// worklog for the before/after numbers.
    candidate_cache: Rc<RefCell<HashMap<CandidateCacheKey, Vec<T>>>>,
    candidate_cache_bytes: Rc<std::cell::Cell<usize>>,
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
        let edge_count = problem.directed_edges.len();
        let sample_counts = (0..edge_count)
            .map(|edge| arena.directed_record_count(edge))
            .collect::<Result<Vec<_>>>()?;
        let frame_order = dependency_order(&problem.directed_edges)?;
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
                Some(cores) => Rc::clone(cores),
                None => Rc::new(prepare_cores::<T, V>(input, problem)?),
            };
            // Every directed edge gets a memo spine, including the ones this
            // call will reuse wholesale: a grown edge's `compute_batch`
            // priming recursion walks its ancestor chain regardless of
            // whether those ancestor edges are themselves being rebuilt, and
            // `FrameBuilder::compute` needs a slot to memoize each pulled or
            // computed row into. A spine slot is one `Option<Vec<T>>`
            // (a pointer triple), negligible next to the `bond_dim`-wide row
            // it would hold; the reused edges' spines are allocated at full
            // length exactly like every other edge's -- they just stay
            // `None`-filled unless something reads through them.
            //
            // What is deliberately NOT done here any more is the eager seed
            // loop this function used to run: copying every already-known
            // sample's row out of `existing_input` into `memo` up front, for
            // every edge, on every call. That copy was measured at chi=256 to
            // be 17.5% of total ACI wall time, all of it pure data movement.
            // `existing_frames` below replaces it with a lazy pull that only
            // fires for a row something actually reads.
            let memo = sample_counts
                .iter()
                .map(|&count| vec![None; count])
                .collect::<Vec<_>>();
            let mut builder = FrameBuilder {
                input,
                problem,
                arena,
                cores,
                memo,
                existing_frames: existing_input.map(Vec::as_slice),
            };
            let bond_dims = (0..edge_count)
                .map(|edge| builder.outgoing_bond(edge).map(IndexLike::dim))
                .collect::<Result<Vec<_>>>()?;

            // Pass 1: account for every edge (in edge-index order, so the
            // running `retained_bytes` total and the point at which a
            // resource limit trips are exactly what they were before this
            // function was restructured), then either reuse the previous
            // store's frame for that edge or record that it needs
            // materialization.
            //
            // Results are written into a pre-sized, edge-indexed slot vector
            // rather than pushed: the reuse decision happens here but
            // reconstruction happens in pass 2 below, and two independent
            // `push` sequences over two differently-filtered loops would
            // interleave the two kinds of edge out of edge order.
            let mut input_frames: Vec<Option<Rc<DirectedFrame<T>>>> = vec![None; edge_count];
            let mut frame_elements = vec![0usize; edge_count];
            let mut known_samples = vec![0usize; edge_count];
            for edge in 0..edge_count {
                let sample_count = sample_counts[edge];
                let bond_dim = bond_dims[edge];
                let elements =
                    sample_count
                        .checked_mul(bond_dim)
                        .ok_or(TreeAciError::SizeOverflow {
                            context: "directed frame elements",
                        })?;
                frame_elements[edge] = elements;
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

                let previous = existing_input.and_then(|frames| frames.get(edge));
                let known = previous.map_or(0, |frame| frame.sample_count);
                known_samples[edge] = known;
                // `SampleArena` is append-only with immutable `SampleId`s (see
                // `samples.rs`), so an unchanged sample count means an
                // identical, identically-ordered sample set: the previous
                // store's frame for this edge is already exactly the frame
                // this store needs. Share it instead of recomputing or even
                // re-copying it -- no `compute_batch` call, no memo fill, no
                // fresh `Matrix`. The bytes/records accounted above still
                // count: this store's `frames` genuinely retains them.
                if let Some(previous) = previous.filter(|frame| {
                    frame.sample_count == sample_count && frame.bond_dim == bond_dim
                }) {
                    input_frames[edge] = Some(Rc::clone(previous));
                    continue;
                }
            }

            // Materialize missing edges only after their incoming frame
            // dependencies have been materialized. The old edge-index order
            // could call `compute_batch` on an edge before its single-
            // incoming ancestor; that edge then reached the ancestor through
            // scalar priming, defeating the batched path on the ancestor's
            // first materialization. `frame_order` is a topological order of
            // this directed-frame dependency graph, so a single-incoming
            // ancestor is fully batched before it is read by its dependent.
            for &edge in &frame_order {
                if input_frames[edge].is_some() {
                    continue;
                }
                builder.compute_batch(edge, known_samples[edge]..sample_counts[edge])?;
            }

            // Pass 2: rebuild only the edges pass 1 left empty (grown or
            // brand new). Reused edges keep the `Rc` pass 1 put in their slot
            // and are not touched.
            for edge in 0..edge_count {
                if input_frames[edge].is_some() {
                    continue;
                }
                let sample_count = sample_counts[edge];
                let bond_dim = bond_dims[edge];
                let previous = existing_input.and_then(|frames| frames.get(edge));
                let mut data = vec![T::default(); frame_elements[edge]];
                for sample in 0..sample_count {
                    // Samples at or above `known` were just materialized into
                    // `memo` by pass 1's `compute_batch`. Samples below it are
                    // in `memo` only if something read them -- an ancestor
                    // priming recursion, which lazily pulls through
                    // `existing_frames` -- so an untouched old sample is
                    // pulled from the previous store right here instead.
                    let values = match std::mem::take(&mut builder.memo[edge][sample]) {
                        Some(values) => values,
                        None => previous
                            .filter(|frame| sample < frame.sample_count)
                            .map(|frame| frame.row(sample))
                            .ok_or(TreeAciError::InternalInvariant {
                                message: "directed frame memoization left a sample uncomputed",
                            })?,
                    };
                    if values.len() != bond_dim {
                        return Err(TreeAciError::InternalInvariant {
                            message: "computed frame length differs from cut bond dimension",
                        });
                    }
                    for (bond, value) in values.into_iter().enumerate() {
                        data[sample + sample_count * bond] = value;
                    }
                }
                input_frames[edge] = Some(Rc::new(DirectedFrame {
                    sample_count,
                    bond_dim,
                    sample_ids: (0..sample_count).collect(),
                    values: Matrix::from_col_major_vec(sample_count, bond_dim, data),
                }));
            }

            let input_frames = input_frames
                .into_iter()
                .map(|frame| {
                    frame.ok_or(TreeAciError::InternalInvariant {
                        message: "directed frame reconstruction left an edge unfilled",
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            all_inputs.push(input_frames);
            all_cores.push(builder.cores);
        }
        let (candidate_cache, candidate_cache_bytes) = match existing {
            Some(store) => (
                Rc::clone(&store.candidate_cache),
                Rc::clone(&store.candidate_cache_bytes),
            ),
            None => (Rc::new(RefCell::new(HashMap::new())), Rc::new(Cell::new(0))),
        };
        Ok(Self {
            frames: all_inputs,
            cores: all_cores,
            records,
            retained_bytes,
            candidate_cache,
            candidate_cache_bytes,
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

    /// Computes every candidate's frame vector for one input and directed
    /// edge, using the batched BLAS path (one `mat_mul` call per distinct
    /// `local_coordinate`) when the edge's source node has exactly one
    /// incoming edge, and falling back to the scalar
    /// [`Self::candidate_frame`] path (called once per candidate, and still
    /// consulting/populating `candidate_cache`) otherwise.
    ///
    /// The batched path also consults `candidate_cache` per candidate before
    /// grouping it into a BLAS call. A one-off instrumented run of
    /// `tree_elementwise` on a 24-node `separated_two_peak_tree` chain (see
    /// Task 4's report) measured a 0% candidate-cache hit rate for that
    /// workload, unlike the 45-65% this file's `candidate_cache` doc cites
    /// from an older worklog measurement. The check is kept anyway: it costs
    /// one `HashMap` lookup per candidate against an `O(bond_dim)`-or-larger
    /// BLAS contraction, negligible even when it never hits, and it keeps
    /// this path's cache semantics identical to the scalar
    /// [`Self::candidate_frame`] path it replaces for other workloads or
    /// call patterns where reuse may still occur.
    pub(crate) fn candidate_frames_for_edge<V: TreeAciNode>(
        &self,
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        candidates: &[ComponentSample],
    ) -> Result<Vec<Vec<T>>> {
        let directed =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate frame references an unknown directed edge",
                })?;
        if directed.incoming_to_from.len() != 1 {
            return candidates
                .iter()
                .map(|candidate| {
                    self.candidate_frame(inputs, problem, input, directed_edge, candidate)
                })
                .collect();
        }

        let node =
            *problem
                .node_positions
                .get(&directed.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate source has no prepared node position",
                })?;
        let tree = inputs.get(input).ok_or(TreeAciError::InternalInvariant {
            message: "candidate frame references an unknown input",
        })?;
        let cores = self
            .cores
            .get(input)
            .ok_or(TreeAciError::InternalInvariant {
                message: "candidate frame has no prepared input cores",
            })?;
        let core = cores.get(node).ok_or(TreeAciError::InternalInvariant {
            message: "candidate frame source node has no prepared core",
        })?;
        let outgoing = outgoing_bond(tree, problem, directed_edge)?;
        let outgoing_axis = axis_of(&core.indices, outgoing)?;
        let physical = &problem.physical[node];
        let physical_axes = physical
            .indices
            .iter()
            .map(|index| axis_of(&core.indices, index))
            .collect::<Result<Vec<_>>>()?;
        let incoming_edge = directed.incoming_to_from[0];
        let incoming_bond = outgoing_bond(tree, problem, incoming_edge)?;
        let incoming_axis = axis_of(&core.indices, incoming_bond)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim = core.dims[incoming_axis];

        // Group candidate indices by local_coordinate. Candidates sharing a
        // local_coordinate are not contiguous in `candidates` -- see
        // `enumerate_candidates` in `local_update.rs`, whose mixed-radix
        // encoding cycles `local_coordinate` fastest -- so grouping must
        // bucket explicitly rather than slice.
        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut results: Vec<Option<Vec<T>>> = vec![None; candidates.len()];
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let key: CandidateCacheKey = (
                input,
                directed_edge,
                candidate.local_coordinate,
                candidate.incoming.clone(),
            );
            if let Some(cached) = self.candidate_cache.borrow().get(&key) {
                #[cfg(test)]
                candidate_debug_stats::record_hit();
                results[candidate_index] = Some(cached.clone());
                continue;
            }
            #[cfg(test)]
            candidate_debug_stats::record_miss();
            groups
                .entry(candidate.local_coordinate)
                .or_default()
                .push(candidate_index);
        }

        for (local_coordinate, indices) in groups {
            let mut base_offset = 0usize;
            for (physical_axis, &axis) in physical_axes.iter().enumerate() {
                let wanted = (local_coordinate / physical.strides[physical_axis])
                    % physical.dims[physical_axis];
                base_offset += wanted * core.strides[axis];
            }
            let core_matrix = single_incoming_core_matrix(
                core,
                outgoing_axis,
                incoming_axis,
                base_offset,
                outgoing_dim,
                incoming_dim,
            );
            let mut frame_data = Vec::with_capacity(incoming_dim * indices.len());
            for &candidate_index in &indices {
                let incoming = &candidates[candidate_index].incoming;
                if incoming.len() != 1 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "single-incoming-edge candidate does not have exactly one incoming sample",
                    });
                }
                let (edge, sample) = incoming[0];
                if edge != incoming_edge {
                    return Err(TreeAciError::InternalInvariant {
                        message: "single-incoming-edge candidate's incoming sample is on the wrong directed edge",
                    });
                }
                let values = self.frame_values(input, edge, sample)?;
                if values.len() != incoming_dim {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                frame_data.extend(values);
            }
            let frame_matrix = Matrix::from_col_major_vec(incoming_dim, indices.len(), frame_data);
            let batched = contract_prepared_core_batched(&core_matrix, &frame_matrix)?;
            for (column, &candidate_index) in indices.iter().enumerate() {
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|row| batched[[row, column]])
                    .collect();
                let entry_bytes = values.len().saturating_mul(size_of::<T>());
                let candidate_bytes = self.candidate_cache_bytes.get();
                let projected = self
                    .retained_bytes
                    .saturating_add(candidate_bytes)
                    .saturating_add(entry_bytes);
                if projected <= problem.max_frame_bytes {
                    let candidate = &candidates[candidate_index];
                    let key: CandidateCacheKey = (
                        input,
                        directed_edge,
                        candidate.local_coordinate,
                        candidate.incoming.clone(),
                    );
                    self.candidate_cache_bytes
                        .set(candidate_bytes.saturating_add(entry_bytes));
                    self.candidate_cache
                        .borrow_mut()
                        .insert(key, values.clone());
                }
                results[candidate_index] = Some(values);
            }
        }

        results
            .into_iter()
            .map(|value| {
                value.ok_or(TreeAciError::InternalInvariant {
                    message: "candidate frame batching left a candidate unfilled",
                })
            })
            .collect()
    }

    pub(crate) fn candidate_frame<V: TreeAciNode>(
        &self,
        inputs: &[TreeTN<IdxTensor, V>],
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        sample: &ComponentSample,
    ) -> Result<Vec<T>> {
        let key: CandidateCacheKey = (
            input,
            directed_edge,
            sample.local_coordinate,
            sample.incoming.clone(),
        );
        if let Some(cached) = self.candidate_cache.borrow().get(&key) {
            #[cfg(test)]
            candidate_debug_stats::record_hit();
            return Ok(cached.clone());
        }
        #[cfg(test)]
        candidate_debug_stats::record_miss();
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
        let values = contract_prepared_core(
            tree,
            problem,
            cores,
            directed_edge,
            sample.local_coordinate,
            &incoming,
        )?;
        let entry_bytes = values.len().saturating_mul(size_of::<T>());
        let candidate_bytes = self.candidate_cache_bytes.get();
        let projected = self
            .retained_bytes
            .saturating_add(candidate_bytes)
            .saturating_add(entry_bytes);
        if projected <= problem.max_frame_bytes {
            self.candidate_cache_bytes
                .set(candidate_bytes.saturating_add(entry_bytes));
            self.candidate_cache
                .borrow_mut()
                .insert(key, values.clone());
        }
        Ok(values)
    }
}

/// Returns an order in which every directed frame's incoming dependencies are
/// available before the frame itself is materialized.
///
/// The dependency graph is acyclic for a validated tree: a directed edge
/// depends on the other directed edges entering its source node, and following
/// those dependencies walks away from the edge's source. Kahn's algorithm is
/// used instead of relying on numeric directed-edge ids, whose construction
/// order is unrelated to this dependency direction on a chain.
fn dependency_order<V>(directed_edges: &[DirectedEdge<V>]) -> Result<Vec<DirectedEdgeId>> {
    let edge_count = directed_edges.len();
    let mut remaining = directed_edges
        .iter()
        .map(|edge| edge.incoming_to_from.len())
        .collect::<Vec<_>>();
    let mut dependents = vec![Vec::<DirectedEdgeId>::new(); edge_count];

    for (edge, directed) in directed_edges.iter().enumerate() {
        for &dependency in &directed.incoming_to_from {
            if dependency >= edge_count {
                return Err(TreeAciError::InternalInvariant {
                    message: "directed frame dependency references an unknown edge",
                });
            }
            dependents[dependency].push(edge);
        }
    }

    let mut ready = VecDeque::new();
    for (edge, &count) in remaining.iter().enumerate() {
        if count == 0 {
            ready.push_back(edge);
        }
    }

    let mut order = Vec::with_capacity(edge_count);
    while let Some(edge) = ready.pop_front() {
        order.push(edge);
        for &dependent in &dependents[edge] {
            let count = remaining
                .get_mut(dependent)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "directed frame dependency has no indegree entry",
                })?;
            *count = count
                .checked_sub(1)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "directed frame dependency indegree underflowed",
                })?;
            if *count == 0 {
                ready.push_back(dependent);
            }
        }
    }

    if order.len() != edge_count {
        return Err(TreeAciError::InternalInvariant {
            message: "directed frame dependency graph contains a cycle",
        });
    }
    Ok(order)
}

struct FrameBuilder<'a, T, V>
where
    T: TreeAciScalar,
    V: TreeAciNode,
{
    input: &'a TreeTN<IdxTensor, V>,
    problem: &'a PreparedTreeProblem<V>,
    arena: &'a SampleArena,
    cores: Rc<Vec<PreparedCore<T>>>,
    memo: Vec<Vec<Option<Vec<T>>>>,
    /// The previous `InputFrameStore`'s frames for this same input, indexed
    /// by directed edge, when this builder is extending an existing store.
    ///
    /// `SampleArena` is append-only (see `samples.rs`): a sample already
    /// interned when the previous store was built names exactly the same
    /// component forever, so its frame row can be pulled directly from the
    /// previous store's `Rc`-shared `DirectedFrame` (a single O(bond_dim)
    /// copy via [`DirectedFrame::row`]) instead of recomputed via
    /// `contract_prepared_core`. `None` for a from-scratch build, where there
    /// is no previous store to pull from.
    existing_frames: Option<&'a [Rc<DirectedFrame<T>>]>,
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
        // A sample already known to the previous store names exactly the
        // same component (see `existing_frames`'s doc comment) -- pull its
        // row directly instead of recomputing it, and memoize the pull so
        // repeat reads within this builder don't pull twice. This must not
        // record a `debug_stats` compute call: that counter tracks genuine
        // `contract_prepared_core` invocations only (see
        // `frames::tests::compute_pulls_already_known_samples_from_the_previous_store_without_recomputing`).
        if let Some(values) = self
            .existing_frames
            .and_then(|frames| frames.get(edge))
            .filter(|frame| sample < frame.sample_count)
            .map(|frame| frame.row(sample))
        {
            let slot = self
                .memo
                .get_mut(edge)
                .and_then(|samples| samples.get_mut(sample))
                .ok_or(TreeAciError::InternalInvariant {
                    message: "computed frame has no memoization slot",
                })?;
            *slot = Some(values.clone());
            return Ok(values);
        }
        #[cfg(test)]
        debug_stats::record_scalar_compute_call();
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

    /// Computes and memoizes every sample in `samples` for `edge`, using the
    /// batched BLAS path ([`contract_prepared_core_batched`]) when `edge`'s
    /// source node has exactly one incoming edge -- the same precondition and
    /// grouping strategy [`InputFrameStore::candidate_frames_for_edge`]
    /// already uses for pivot-search candidates -- and falling back to
    /// [`Self::compute`] per sample otherwise (0 or >=2 incoming edges).
    ///
    /// Unlike `compute`, this has no return value: every result lands in
    /// `self.memo[edge]`, which is where `build_or_extend`'s caller reads
    /// results back from regardless of which path computed them.
    fn compute_batch(
        &mut self,
        edge: DirectedEdgeId,
        samples: std::ops::Range<SampleId>,
    ) -> Result<()> {
        let directed = &self.problem.directed_edges[edge];
        if directed.incoming_to_from.len() != 1 {
            for sample in samples {
                self.compute(edge, sample)?;
            }
            return Ok(());
        }
        let incoming_edge = directed.incoming_to_from[0];

        // Skip samples already memoized, and fetch each remaining sample's
        // `ComponentSample` exactly once (reused below for both the priming
        // recursion and the local_coordinate grouping, rather than
        // re-fetched from `self.arena` in each of three separate loops).
        //
        // The skip matters for correctness-of-effort, not correctness of
        // result: dependency priming or a direct caller can already have
        // memoized a sample in this range before this batch is assembled.
        // Without this check those samples would be redundantly re-grouped
        // and re-contracted through a second, wasted `mat_mul`. Mirrors
        // `candidate_frames_for_edge`'s existing `candidate_cache` check at
        // the equivalent point in its own loop.
        let mut pending: Vec<(SampleId, ComponentSample)> = Vec::new();
        for sample in samples {
            if self.memo[edge][sample].is_some() {
                continue;
            }
            let record = self.arena.record(edge, sample)?.clone();
            if record.incoming.len() != 1 {
                return Err(TreeAciError::InternalInvariant {
                    message:
                        "single-incoming-edge sample does not have exactly one incoming sample",
                });
            }
            let (incoming_edge_of_sample, _) = record.incoming[0];
            if incoming_edge_of_sample != incoming_edge {
                return Err(TreeAciError::InternalInvariant {
                    message: "single-incoming-edge sample's incoming sample is on the wrong directed edge",
                });
            }
            pending.push((sample, record));
        }
        if pending.is_empty() {
            return Ok(());
        }

        // Ensure every pending sample's single incoming frame is memoized
        // first. This recursion is `compute`'s existing one -- it is already
        // memoized, so a sample whose incoming frame was computed by an
        // earlier call (this one or a sibling directed edge sharing an
        // ancestor) does no repeated work.
        for (_, record) in &pending {
            let (_, incoming_sample) = record.incoming[0];
            self.compute(incoming_edge, incoming_sample)?;
        }

        let node = *self.problem.node_positions.get(&directed.from).ok_or(
            TreeAciError::InternalInvariant {
                message: "frame source has no prepared node position",
            },
        )?;
        let core = &self.cores[node];
        let outgoing = self.outgoing_bond(edge)?;
        let outgoing_axis = axis_of(&core.indices, outgoing)?;
        let physical = &self.problem.physical[node];
        let physical_axes = physical
            .indices
            .iter()
            .map(|index| axis_of(&core.indices, index))
            .collect::<Result<Vec<_>>>()?;
        let incoming_bond = self.outgoing_bond(incoming_edge)?;
        let incoming_axis = axis_of(&core.indices, incoming_bond)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim = core.dims[incoming_axis];

        // Group by local_coordinate -- same rationale as
        // `candidate_frames_for_edge`: samples sharing a local_coordinate are
        // not guaranteed contiguous in `samples`. Each group entry carries
        // the sample's own id plus its (already-resolved) incoming sample
        // id, so the frame-gathering loop below never needs to re-fetch the
        // `ComponentSample` a third time.
        let mut groups: std::collections::BTreeMap<usize, Vec<(SampleId, SampleId)>> =
            std::collections::BTreeMap::new();
        for (sample, record) in &pending {
            let (_, incoming_sample) = record.incoming[0];
            groups
                .entry(record.local_coordinate)
                .or_default()
                .push((*sample, incoming_sample));
        }

        for (local_coordinate, group_samples) in groups {
            let mut base_offset = 0usize;
            for (physical_axis, &axis) in physical_axes.iter().enumerate() {
                let wanted = (local_coordinate / physical.strides[physical_axis])
                    % physical.dims[physical_axis];
                base_offset += wanted * core.strides[axis];
            }
            let core_matrix = single_incoming_core_matrix(
                core,
                outgoing_axis,
                incoming_axis,
                base_offset,
                outgoing_dim,
                incoming_dim,
            );
            let mut frame_data = Vec::with_capacity(incoming_dim * group_samples.len());
            for &(_, incoming_sample) in &group_samples {
                let values = self.memo[incoming_edge][incoming_sample].clone().ok_or(
                    TreeAciError::InternalInvariant {
                        message:
                            "incoming sample frame was not memoized before batched contraction",
                    },
                )?;
                if values.len() != incoming_dim {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                frame_data.extend(values);
            }
            let frame_matrix =
                Matrix::from_col_major_vec(incoming_dim, group_samples.len(), frame_data);
            let batched = contract_prepared_core_batched(&core_matrix, &frame_matrix)?;
            for (column, &(sample, _)) in group_samples.iter().enumerate() {
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|row| batched[[row, column]])
                    .collect();
                #[cfg(test)]
                debug_stats::record_batched_compute_call();
                let slot = self
                    .memo
                    .get_mut(edge)
                    .and_then(|s| s.get_mut(sample))
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "computed frame has no memoization slot",
                    })?;
                *slot = Some(values);
            }
        }
        Ok(())
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

/// Gathers a `PreparedCore`'s fixed-physical-value slice into a plain
/// `outgoing_dim x incoming_dim` column-major matrix, for nodes with exactly
/// one incoming directed edge.
///
/// This exists so the slice can be fed to a single BLAS `mat_mul` call
/// against every candidate's incoming frame vector at once, instead of the
/// scalar per-candidate loop in [`accumulate_incoming`]. It is only valid
/// for the single-incoming-edge case: with two or more incoming edges the
/// "matrix" this node's core induces is not two-dimensional, and
/// [`contract_prepared_core`] must be used instead.
fn single_incoming_core_matrix<T: TreeAciScalar>(
    core: &PreparedCore<T>,
    outgoing_axis: usize,
    incoming_axis: usize,
    physical_base_offset: usize,
    outgoing_dim: usize,
    incoming_dim: usize,
) -> Matrix<T> {
    let outgoing_stride = core.strides[outgoing_axis];
    let incoming_stride = core.strides[incoming_axis];
    let mut data = Vec::with_capacity(outgoing_dim * incoming_dim);
    for incoming_value in 0..incoming_dim {
        for outgoing_value in 0..outgoing_dim {
            let offset = physical_base_offset
                + incoming_value * incoming_stride
                + outgoing_value * outgoing_stride;
            data.push(core.values[offset]);
        }
    }
    Matrix::from_col_major_vec(outgoing_dim, incoming_dim, data)
}

/// Contracts a single-incoming-edge core matrix against a batch of candidate
/// incoming frame vectors (one per column) in one BLAS call.
///
/// `core_matrix` is `outgoing_dim x incoming_dim` (from
/// [`single_incoming_core_matrix`]); `incoming_frame_matrix` is
/// `incoming_dim x n_candidates`. Returns `outgoing_dim x n_candidates`,
/// column `c` being the same result [`contract_prepared_core`] would have
/// produced for candidate `c` alone.
fn contract_prepared_core_batched<T: TreeAciScalar>(
    core_matrix: &Matrix<T>,
    incoming_frame_matrix: &Matrix<T>,
) -> Result<Matrix<T>> {
    tensor4all_tensorbackend::mat_mul(core_matrix, incoming_frame_matrix).map_err(|error| {
        TreeAciError::Numerical {
            message: error.to_string(),
        }
    })
}

/// Contracts a core slice's two incoming axes against batches of candidate
/// frame vectors for both incoming edges, computing every combination in
/// the cartesian product of `v1`'s and `v2`'s columns via `incoming_dim_2 + 1`
/// BLAS `mat_mul` calls (`incoming_dim_2` calls fold in `v1` one slice of the
/// second axis at a time, then one final call folds in `v2`) instead of one
/// scalar [`accumulate_incoming`] walk per `(n1, n2)` combination.
///
/// `v1` is `incoming_dim_1 x n1`, `v2` is `incoming_dim_2 x n2`. Returns an
/// `(outgoing_dim * n1) x n2` matrix: column `n2`, rows
/// `[outgoing_dim * n1_index, outgoing_dim * (n1_index + 1))`, holds the
/// `outgoing_dim`-length frame vector [`contract_prepared_core`] would
/// produce for the `(n1_index, n2)` candidate alone.
fn two_incoming_core_matrix_batched<T: TreeAciScalar>(
    core: &PreparedCore<T>,
    outgoing_axis: usize,
    incoming_axis_1: usize,
    incoming_axis_2: usize,
    physical_base_offset: usize,
    outgoing_dim: usize,
    incoming_dim_1: usize,
    incoming_dim_2: usize,
    v1: &Matrix<T>,
    v2: &Matrix<T>,
) -> Result<Matrix<T>> {
    let n1 = v1.ncols();
    let stride_2 = core.strides[incoming_axis_2];
    let mut stage1_data = Vec::with_capacity(outgoing_dim * n1 * incoming_dim_2);
    for i2 in 0..incoming_dim_2 {
        let core_matrix = single_incoming_core_matrix(
            core,
            outgoing_axis,
            incoming_axis_1,
            physical_base_offset + i2 * stride_2,
            outgoing_dim,
            incoming_dim_1,
        );
        let stage1 = contract_prepared_core_batched(&core_matrix, v1)?;
        stage1_data.extend(stage1.into_col_major_vec());
    }
    let stage1_matrix = Matrix::from_col_major_vec(outgoing_dim * n1, incoming_dim_2, stage1_data);
    contract_prepared_core_batched(&stage1_matrix, v2)
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
