//! Exact per-input contractions for immutable directed component samples.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::mem::size_of;
use std::rc::Rc;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_tensorbackend::Matrix;
use tensor4all_treetn::TreeTN;

use crate::{
    problem::{enforce_limit, DirectedEdgeId, LocalPhysicalPlan, PreparedTreeProblem},
    samples::{CandidateSets, ComponentSample, SampleArena, SampleId},
    Result, TreeAciError, TreeAciNode, TreeAciScalar,
};

fn checked_product(factors: &[usize], context: &'static str) -> Result<usize> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(TreeAciError::SizeOverflow { context })
    })
}

fn checked_sum(terms: &[usize], context: &'static str) -> Result<usize> {
    terms.iter().try_fold(0usize, |sum, &term| {
        sum.checked_add(term)
            .ok_or(TreeAciError::SizeOverflow { context })
    })
}

fn two_incoming_scratch_elements(
    outgoing_dim: usize,
    incoming_dim_1: usize,
    incoming_dim_2: usize,
    n1: usize,
    n2: usize,
) -> Result<usize> {
    checked_sum(
        &[
            checked_product(&[incoming_dim_1, n1], "two-incoming first frame matrix")?,
            checked_product(&[incoming_dim_2, n2], "two-incoming second frame matrix")?,
            checked_product(&[outgoing_dim, incoming_dim_1], "two-incoming core matrix")?,
            checked_product(
                &[outgoing_dim, n1, incoming_dim_2],
                "two-incoming stage-one matrix",
            )?,
            checked_product(&[outgoing_dim, n1], "two-incoming stage-one slice")?,
            checked_product(&[outgoing_dim, n1, n2], "two-incoming output matrix")?,
        ],
        "two-incoming scratch elements",
    )
}

fn enforce_frame_working_elements<T: TreeAciScalar, V: TreeAciNode>(
    problem: &PreparedTreeProblem<V>,
    elements: usize,
) -> Result<()> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(TreeAciError::SizeOverflow {
            context: "candidate frame working bytes",
        })?;
    enforce_limit("working bytes", bytes, problem.max_working_bytes)
}

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
    /// Sample-major frame values, so one sample's bond vector is contiguous.
    pub(crate) values: Vec<T>,
}

impl<T: TreeAciScalar> DirectedFrame<T> {
    fn row_slice(&self, sample: SampleId) -> &[T] {
        let start = sample * self.bond_dim;
        &self.values[start..start + self.bond_dim]
    }

    fn row(&self, sample: SampleId) -> Vec<T> {
        self.row_slice(sample).to_vec()
    }
}

/// Compact identity for the common leaf, chain, and trivalent-tree cases.
///
/// The directed edge already fixes the ordered incoming-edge identities, so
/// only their immutable sample IDs belong in the key after that order has
/// been validated. Nodes with three or more incoming cuts deliberately skip
/// this optional cache: retaining an arbitrary `Vec<usize>` key would make
/// every hot lookup hash and allocate an unbounded multi-index.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum CandidateIncomingKey {
    None,
    One(SampleId),
    Two(SampleId, SampleId),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CandidateCacheKey {
    input: usize,
    directed_edge: DirectedEdgeId,
    local_coordinate: usize,
    incoming: CandidateIncomingKey,
}

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
        let frame_order = &problem.directed_dependency_order;
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
                // fresh buffer. The bytes/records accounted above still
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
            for &edge in frame_order {
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
                let mut data = Vec::with_capacity(frame_elements[edge]);
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
                    data.extend(values);
                }
                input_frames[edge] = Some(Rc::new(DirectedFrame {
                    sample_count,
                    bond_dim,
                    values: data,
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
        let combined_bytes = retained_bytes.checked_add(candidate_cache_bytes.get());
        if combined_bytes.is_none_or(|bytes| bytes > problem.max_frame_bytes) {
            // Base frames are mandatory; candidate frames are only a reusable
            // acceleration. Growth of the arena can therefore reclaim the
            // optional cache before publishing a store whose combined
            // retained payload exceeds `max_frame_bytes`.
            candidate_cache.borrow_mut().clear();
            candidate_cache_bytes.set(0);
        }
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

    /// Logical payload bytes retained by directed and candidate frames.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
            .saturating_add(self.candidate_cache_bytes.get())
    }

    fn cache_candidate_if_fits(
        &self,
        problem: &PreparedTreeProblem<impl TreeAciNode>,
        key: CandidateCacheKey,
        values: &[T],
    ) {
        let Some(entry_bytes) = values
            .len()
            .checked_mul(size_of::<T>())
            .and_then(|bytes| bytes.checked_add(size_of::<CandidateCacheKey>()))
        else {
            return;
        };
        let candidate_bytes = self.candidate_cache_bytes.get();
        let Some(projected) = self
            .retained_bytes
            .checked_add(candidate_bytes)
            .and_then(|bytes| bytes.checked_add(entry_bytes))
        else {
            return;
        };
        if projected <= problem.max_frame_bytes {
            if let std::collections::hash_map::Entry::Vacant(entry) =
                self.candidate_cache.borrow_mut().entry(key)
            {
                // Duplicate candidates in one batched call are all cache
                // misses during grouping. `entry` hashes once and ensures
                // their shared key is stored and charged only once.
                entry.insert(values.to_vec());
                self.candidate_cache_bytes
                    .set(candidate_bytes + entry_bytes);
            }
        }
    }

    fn candidate_cache_key<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        candidate: &ComponentSample,
    ) -> Result<Option<CandidateCacheKey>> {
        let directed =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate cache references an unknown directed edge",
                })?;
        if candidate
            .incoming
            .iter()
            .map(|(edge, _)| *edge)
            .ne(directed.incoming_to_from.iter().copied())
        {
            return Err(TreeAciError::InternalInvariant {
                message: "candidate cache key has the wrong ordered incoming branches",
            });
        }
        let incoming = match candidate.incoming.as_slice() {
            [] => CandidateIncomingKey::None,
            [(_, sample)] => CandidateIncomingKey::One(*sample),
            [(_, first), (_, second)] => CandidateIncomingKey::Two(*first, *second),
            // INVARIANT: the general-degree scalar contraction remains
            // correct without retention; candidate row/column limits bound
            // its work. Skipping this optional cache avoids an unbounded
            // vector-valued key in a persistent hot-path HashMap.
            _ => return Ok(None),
        };
        Ok(Some(CandidateCacheKey {
            input,
            directed_edge,
            local_coordinate: candidate.local_coordinate,
            incoming,
        }))
    }

    /// Peak scalar scratch for candidates produced by `enumerate_candidates`,
    /// excluding the returned vectors. That enumerator emits the complete
    /// local-coordinate/incoming-sample Cartesian product, so its group sizes
    /// are available directly from `candidate_sets`; no hot-path regrouping or
    /// hashing is needed merely to enforce the working-byte limit.
    pub(crate) fn enumerated_candidate_frame_scratch_elements<V: TreeAciNode>(
        &self,
        problem: &PreparedTreeProblem<V>,
        input: usize,
        directed_edge: DirectedEdgeId,
        candidate_sets: &CandidateSets,
    ) -> Result<usize> {
        let directed =
            problem
                .directed_edges
                .get(directed_edge)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate frame references an unknown directed edge",
                })?;
        let outgoing_dim = self.bond_dim(input, directed_edge)?;
        let node =
            *problem
                .node_positions
                .get(&directed.from)
                .ok_or(TreeAciError::InternalInvariant {
                    message: "candidate source has no prepared node position",
                })?;

        match directed.incoming_to_from.as_slice() {
            [] => Ok(0),
            [incoming_edge] => {
                let incoming_dim = self.bond_dim(input, *incoming_edge)?;
                let count = candidate_sets
                    .ids
                    .get(*incoming_edge)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "candidate sets are missing an incoming directed edge",
                    })?
                    .len();
                checked_sum(
                    &[
                        checked_product(
                            &[outgoing_dim, problem.physical[node].local_dim, incoming_dim],
                            "single-incoming candidate core matrix",
                        )?,
                        checked_product(
                            &[incoming_dim, count],
                            "single-incoming candidate frame matrix",
                        )?,
                        checked_product(
                            &[outgoing_dim, problem.physical[node].local_dim, count],
                            "single-incoming candidate output matrix",
                        )?,
                    ],
                    "single-incoming candidate scratch elements",
                )
            }
            [incoming_edge_1, incoming_edge_2] => {
                let incoming_dim_1 = self.bond_dim(input, *incoming_edge_1)?;
                let incoming_dim_2 = self.bond_dim(input, *incoming_edge_2)?;
                let n1 = candidate_sets
                    .ids
                    .get(*incoming_edge_1)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "candidate sets are missing the first incoming directed edge",
                    })?
                    .len();
                let n2 = candidate_sets
                    .ids
                    .get(*incoming_edge_2)
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "candidate sets are missing the second incoming directed edge",
                    })?
                    .len();
                two_incoming_scratch_elements(outgoing_dim, incoming_dim_1, incoming_dim_2, n1, n2)
            }
            incoming_edges => incoming_edges.iter().try_fold(0usize, |sum, &edge| {
                let dimension = self.bond_dim(input, edge)?;
                sum.checked_add(dimension)
                    .ok_or(TreeAciError::SizeOverflow {
                        context: "scalar candidate incoming frame elements",
                    })
            }),
        }
    }

    pub(crate) fn bond_dim(&self, input: usize, directed_edge: DirectedEdgeId) -> Result<usize> {
        self.frames
            .get(input)
            .and_then(|edges| edges.get(directed_edge))
            .map(|frame| frame.bond_dim)
            .ok_or(TreeAciError::InternalInvariant {
                message: "frame dimension lookup references an unknown input or directed edge",
            })
    }

    fn frame_slice(
        &self,
        input: usize,
        directed_edge: DirectedEdgeId,
        sample: SampleId,
    ) -> Result<&[T]> {
        let frame = self
            .frames
            .get(input)
            .and_then(|edges| edges.get(directed_edge))
            .ok_or(TreeAciError::InternalInvariant {
                message: "frame lookup references an unknown input or directed edge",
            })?;
        if sample >= frame.sample_count {
            return Err(TreeAciError::InternalInvariant {
                message: "frame lookup references an unknown immutable sample ID",
            });
        }
        Ok(frame.row_slice(sample))
    }

    #[cfg(test)]
    pub(crate) fn frame_values(
        &self,
        input: usize,
        directed_edge: DirectedEdgeId,
        sample: SampleId,
    ) -> Result<Vec<T>> {
        Ok(self.frame_slice(input, directed_edge, sample)?.to_vec())
    }

    /// Computes every candidate's frame vector for one input and directed
    /// edge. Dispatches to a batched BLAS path when the edge's source node
    /// has exactly one incoming edge (one `mat_mul` call per distinct
    /// `local_coordinate`) or exactly two incoming edges (see
    /// [`Self::candidate_frames_for_edge_two_incoming`] and
    /// [`two_incoming_core_matrix_batched`]), and falls back to the scalar
    /// [`Self::candidate_frame`] path for a leaf edge (zero incoming edges)
    /// or a node with three or more incoming edges (out of scope for the
    /// batched paths -- see issue #671 and
    /// `docs/worklogs/2026-08-22-treeaci-branch-batched-frames.md`).
    /// Leaves use the compact candidate cache; 3+-incoming candidates skip
    /// it because their exact identity would require an unbounded vector key.
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
        if directed.incoming_to_from.len() == 2 {
            return self.candidate_frames_for_edge_two_incoming(
                inputs,
                problem,
                input,
                directed_edge,
                candidates,
            );
        }
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

        // Contract all physical coordinates in one matrix multiplication.
        // `enumerate_candidates` emits their Cartesian product with incoming
        // sample IDs, so one `(outgoing * physical) x incoming` core matrix is
        // both simpler and substantially cheaper than one small BLAS dispatch
        // per physical coordinate.
        let mut pending = Vec::new();
        let mut results: Vec<Option<Vec<T>>> = vec![None; candidates.len()];
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let key = self
                .candidate_cache_key(problem, input, directed_edge, candidate)?
                .ok_or(TreeAciError::InternalInvariant {
                    message: "single-incoming candidate has no compact cache key",
                })?;
            if let Some(cached) = self.candidate_cache.borrow().get(&key) {
                #[cfg(test)]
                candidate_debug_stats::record_hit();
                results[candidate_index] = Some(cached.clone());
                continue;
            }
            #[cfg(test)]
            candidate_debug_stats::record_miss();
            if candidate.local_coordinate >= physical.local_dim
                || candidate.incoming.len() != 1
                || candidate.incoming[0].0 != incoming_edge
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "single-incoming-edge candidate does not match its prepared component",
                });
            }
            pending.push(candidate_index);
        }

        if !pending.is_empty() {
            let mut incoming_ids = Vec::new();
            let mut incoming_positions = HashMap::new();
            for &candidate_index in &pending {
                let sample = candidates[candidate_index].incoming[0].1;
                incoming_positions.entry(sample).or_insert_with(|| {
                    incoming_ids.push(sample);
                    incoming_ids.len() - 1
                });
            }
            let scratch = checked_sum(
                &[
                    checked_product(
                        &[outgoing_dim, physical.local_dim, incoming_dim],
                        "single-incoming candidate core matrix",
                    )?,
                    checked_product(
                        &[incoming_dim, incoming_ids.len()],
                        "single-incoming candidate frame matrix",
                    )?,
                    checked_product(
                        &[outgoing_dim, physical.local_dim, incoming_ids.len()],
                        "single-incoming candidate output matrix",
                    )?,
                ],
                "single-incoming candidate scratch elements",
            )?;
            enforce_frame_working_elements::<T, V>(problem, scratch)?;
            let core_matrix = single_incoming_all_physical_core_matrix(
                core,
                outgoing_axis,
                incoming_axis,
                physical,
                &physical_axes,
                outgoing_dim,
                incoming_dim,
            );
            let mut frame_data = Vec::with_capacity(incoming_dim * incoming_ids.len());
            for &sample in &incoming_ids {
                let values = self.frame_slice(input, incoming_edge, sample)?;
                if values.len() != incoming_dim {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                frame_data.extend_from_slice(values);
            }
            let frame_matrix =
                Matrix::from_col_major_vec(incoming_dim, incoming_ids.len(), frame_data);
            let batched = contract_prepared_core_batched(&core_matrix, &frame_matrix)?;
            for candidate_index in pending {
                let candidate = &candidates[candidate_index];
                let column = incoming_positions[&candidate.incoming[0].1];
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|row| batched[[row + outgoing_dim * candidate.local_coordinate, column]])
                    .collect();
                let key = self
                    .candidate_cache_key(problem, input, directed_edge, candidate)?
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "single-incoming candidate has no compact cache key",
                    })?;
                self.cache_candidate_if_fits(problem, key, &values);
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

    /// Batched counterpart to [`Self::candidate_frames_for_edge`]'s
    /// single-incoming-edge path, for directed edges whose source node has
    /// exactly two incoming edges (every hub of a 3-valent tree branch
    /// point). Groups candidates by `local_coordinate` exactly as the
    /// single-incoming path does, then for each group gathers the distinct
    /// sample ids referenced on each incoming edge, builds one frame-vector
    /// matrix per incoming edge, and contracts both via
    /// [`two_incoming_core_matrix_batched`] in one shot -- computing the
    /// full cartesian product of the group's distinct incoming ids (a
    /// superset of the group's actual candidates whenever the group is not
    /// already the full product) and reading back only the entries the
    /// group's candidates actually need.
    fn candidate_frames_for_edge_two_incoming<V: TreeAciNode>(
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
        let incoming_edge_1 = directed.incoming_to_from[0];
        let incoming_edge_2 = directed.incoming_to_from[1];
        let incoming_bond_1 = outgoing_bond(tree, problem, incoming_edge_1)?;
        let incoming_bond_2 = outgoing_bond(tree, problem, incoming_edge_2)?;
        let incoming_axis_1 = axis_of(&core.indices, incoming_bond_1)?;
        let incoming_axis_2 = axis_of(&core.indices, incoming_bond_2)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim_1 = core.dims[incoming_axis_1];
        let incoming_dim_2 = core.dims[incoming_axis_2];

        let mut groups: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        let mut results: Vec<Option<Vec<T>>> = vec![None; candidates.len()];
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let key = self
                .candidate_cache_key(problem, input, directed_edge, candidate)?
                .ok_or(TreeAciError::InternalInvariant {
                    message: "two-incoming candidate has no compact cache key",
                })?;
            if let Some(cached) = self.candidate_cache.borrow().get(&key) {
                #[cfg(test)]
                candidate_debug_stats::record_hit();
                results[candidate_index] = Some(cached.clone());
                continue;
            }
            #[cfg(test)]
            candidate_debug_stats::record_miss();
            if candidate.incoming.len() != 2
                || candidate.incoming[0].0 != incoming_edge_1
                || candidate.incoming[1].0 != incoming_edge_2
            {
                return Err(TreeAciError::InternalInvariant {
                    message: "two-incoming-edge candidate does not match the edge's incoming order",
                });
            }
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

            let mut ids_1: Vec<SampleId> = Vec::new();
            let mut position_1: HashMap<SampleId, usize> = HashMap::new();
            let mut ids_2: Vec<SampleId> = Vec::new();
            let mut position_2: HashMap<SampleId, usize> = HashMap::new();
            for &candidate_index in &indices {
                let (_, sample_1) = candidates[candidate_index].incoming[0];
                let (_, sample_2) = candidates[candidate_index].incoming[1];
                position_1.entry(sample_1).or_insert_with(|| {
                    ids_1.push(sample_1);
                    ids_1.len() - 1
                });
                position_2.entry(sample_2).or_insert_with(|| {
                    ids_2.push(sample_2);
                    ids_2.len() - 1
                });
            }

            let n1 = ids_1.len();
            let n2 = ids_2.len();
            let scratch = two_incoming_scratch_elements(
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                n1,
                n2,
            )?;
            enforce_frame_working_elements::<T, V>(problem, scratch)?;

            let mut v1_data = Vec::with_capacity(incoming_dim_1 * ids_1.len());
            for &sample in &ids_1 {
                let values = self.frame_slice(input, incoming_edge_1, sample)?;
                if values.len() != incoming_dim_1 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v1_data.extend_from_slice(values);
            }
            let v1 = Matrix::from_col_major_vec(incoming_dim_1, ids_1.len(), v1_data);

            let mut v2_data = Vec::with_capacity(incoming_dim_2 * ids_2.len());
            for &sample in &ids_2 {
                let values = self.frame_slice(input, incoming_edge_2, sample)?;
                if values.len() != incoming_dim_2 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v2_data.extend_from_slice(values);
            }
            let v2 = Matrix::from_col_major_vec(incoming_dim_2, ids_2.len(), v2_data);

            let batched = two_incoming_core_matrix_batched(
                core,
                outgoing_axis,
                incoming_axis_1,
                incoming_axis_2,
                base_offset,
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                &v1,
                &v2,
            )?;

            for &candidate_index in &indices {
                let (_, sample_1) = candidates[candidate_index].incoming[0];
                let (_, sample_2) = candidates[candidate_index].incoming[1];
                let n1 = position_1[&sample_1];
                let n2 = position_2[&sample_2];
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|out| batched[[out + outgoing_dim * n1, n2]])
                    .collect();
                let candidate = &candidates[candidate_index];
                let key = self
                    .candidate_cache_key(problem, input, directed_edge, candidate)?
                    .ok_or(TreeAciError::InternalInvariant {
                        message: "two-incoming candidate has no compact cache key",
                    })?;
                self.cache_candidate_if_fits(problem, key, &values);
                results[candidate_index] = Some(values);
            }
        }

        results
            .into_iter()
            .map(|value| {
                value.ok_or(TreeAciError::InternalInvariant {
                    message: "two-incoming candidate frame batching left a candidate unfilled",
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
        let cache_key = self.candidate_cache_key(problem, input, directed_edge, sample)?;
        if let Some(key) = cache_key {
            if let Some(cached) = self.candidate_cache.borrow().get(&key) {
                #[cfg(test)]
                candidate_debug_stats::record_hit();
                return Ok(cached.clone());
            }
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
                self.frame_slice(input, edge, id)
                    .map(|values| (edge, values))
            })
            .collect::<Result<Vec<_>>>()?;
        let values = contract_prepared_core_slices(
            tree,
            problem,
            cores,
            directed_edge,
            sample.local_coordinate,
            &incoming,
        )?;
        if let Some(key) = cache_key {
            self.cache_candidate_if_fits(problem, key, &values);
        }
        Ok(values)
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
    /// already uses for pivot-search candidates -- delegating to
    /// [`Self::compute_batch_two_incoming`] for exactly two incoming edges,
    /// and falling back to [`Self::compute`] per sample otherwise (0
    /// incoming edges, or 3+).
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
        if directed.incoming_to_from.len() == 2 {
            return self.compute_batch_two_incoming(edge, samples);
        }
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

        let mut incoming_ids = Vec::new();
        let mut incoming_positions = HashMap::new();
        for (_, record) in &pending {
            if record.local_coordinate >= physical.local_dim {
                return Err(TreeAciError::InternalInvariant {
                    message: "single-incoming-edge sample has an invalid local coordinate",
                });
            }
            let incoming_sample = record.incoming[0].1;
            incoming_positions
                .entry(incoming_sample)
                .or_insert_with(|| {
                    incoming_ids.push(incoming_sample);
                    incoming_ids.len() - 1
                });
        }
        let scratch = checked_sum(
            &[
                checked_product(
                    &[outgoing_dim, physical.local_dim, incoming_dim],
                    "single-incoming frame core matrix",
                )?,
                checked_product(
                    &[incoming_dim, incoming_ids.len()],
                    "single-incoming frame input matrix",
                )?,
                checked_product(
                    &[outgoing_dim, physical.local_dim, incoming_ids.len()],
                    "single-incoming frame output matrix",
                )?,
            ],
            "single-incoming frame scratch elements",
        )?;
        enforce_frame_working_elements::<T, V>(self.problem, scratch)?;
        let core_matrix = single_incoming_all_physical_core_matrix(
            core,
            outgoing_axis,
            incoming_axis,
            physical,
            &physical_axes,
            outgoing_dim,
            incoming_dim,
        );
        let mut frame_data = Vec::with_capacity(incoming_dim * incoming_ids.len());
        for &incoming_sample in &incoming_ids {
            let values = self.memo[incoming_edge][incoming_sample].as_ref().ok_or(
                TreeAciError::InternalInvariant {
                    message: "incoming sample frame was not memoized before batched contraction",
                },
            )?;
            if values.len() != incoming_dim {
                return Err(TreeAciError::InternalInvariant {
                    message: "incoming frame length differs from its bond dimension",
                });
            }
            frame_data.extend_from_slice(values);
        }
        let frame_matrix = Matrix::from_col_major_vec(incoming_dim, incoming_ids.len(), frame_data);
        let batched = contract_prepared_core_batched(&core_matrix, &frame_matrix)?;
        for (sample, record) in pending {
            let incoming_sample = record.incoming[0].1;
            let column = incoming_positions[&incoming_sample];
            let values: Vec<T> = (0..outgoing_dim)
                .map(|row| batched[[row + outgoing_dim * record.local_coordinate, column]])
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
        Ok(())
    }

    /// Batched counterpart to [`Self::compute_batch`]'s single-incoming-edge
    /// path, for directed edges whose source node has exactly two incoming
    /// edges. Primes both incoming edges' needed samples via [`Self::compute`]
    /// (as `compute_batch` already does for its one incoming edge), then
    /// groups the pending samples by `local_coordinate` and contracts each
    /// group via [`two_incoming_core_matrix_batched`], mirroring
    /// [`InputFrameStore::candidate_frames_for_edge_two_incoming`]'s
    /// structure but reading incoming frame vectors from `self.memo` instead
    /// of a committed `InputFrameStore`.
    fn compute_batch_two_incoming(
        &mut self,
        edge: DirectedEdgeId,
        samples: std::ops::Range<SampleId>,
    ) -> Result<()> {
        let directed = &self.problem.directed_edges[edge];
        let incoming_edge_1 = directed.incoming_to_from[0];
        let incoming_edge_2 = directed.incoming_to_from[1];

        let mut pending: Vec<(SampleId, ComponentSample)> = Vec::new();
        for sample in samples {
            if self.memo[edge][sample].is_some() {
                continue;
            }
            let record = self.arena.record(edge, sample)?.clone();
            if record.incoming.len() != 2
                || record.incoming[0].0 != incoming_edge_1
                || record.incoming[1].0 != incoming_edge_2
            {
                return Err(TreeAciError::InternalInvariant {
                    message:
                        "two-incoming-edge sample does not have exactly two incoming samples on the expected edges",
                });
            }
            pending.push((sample, record));
        }
        if pending.is_empty() {
            return Ok(());
        }

        for (_, record) in &pending {
            self.compute(incoming_edge_1, record.incoming[0].1)?;
            self.compute(incoming_edge_2, record.incoming[1].1)?;
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
        let incoming_bond_1 = self.outgoing_bond(incoming_edge_1)?;
        let incoming_bond_2 = self.outgoing_bond(incoming_edge_2)?;
        let incoming_axis_1 = axis_of(&core.indices, incoming_bond_1)?;
        let incoming_axis_2 = axis_of(&core.indices, incoming_bond_2)?;
        let outgoing_dim = core.dims[outgoing_axis];
        let incoming_dim_1 = core.dims[incoming_axis_1];
        let incoming_dim_2 = core.dims[incoming_axis_2];

        let mut groups: std::collections::BTreeMap<usize, Vec<(SampleId, SampleId, SampleId)>> =
            std::collections::BTreeMap::new();
        for (sample, record) in &pending {
            let (_, sample_1) = record.incoming[0];
            let (_, sample_2) = record.incoming[1];
            groups
                .entry(record.local_coordinate)
                .or_default()
                .push((*sample, sample_1, sample_2));
        }

        for (local_coordinate, group_samples) in groups {
            let mut base_offset = 0usize;
            for (physical_axis, &axis) in physical_axes.iter().enumerate() {
                let wanted = (local_coordinate / physical.strides[physical_axis])
                    % physical.dims[physical_axis];
                base_offset += wanted * core.strides[axis];
            }

            let mut ids_1: Vec<SampleId> = Vec::new();
            let mut position_1: HashMap<SampleId, usize> = HashMap::new();
            let mut ids_2: Vec<SampleId> = Vec::new();
            let mut position_2: HashMap<SampleId, usize> = HashMap::new();
            for &(_, sample_1, sample_2) in &group_samples {
                position_1.entry(sample_1).or_insert_with(|| {
                    ids_1.push(sample_1);
                    ids_1.len() - 1
                });
                position_2.entry(sample_2).or_insert_with(|| {
                    ids_2.push(sample_2);
                    ids_2.len() - 1
                });
            }

            let n1 = ids_1.len();
            let n2 = ids_2.len();
            let scratch = two_incoming_scratch_elements(
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                n1,
                n2,
            )?;
            enforce_frame_working_elements::<T, V>(self.problem, scratch)?;

            let mut v1_data = Vec::with_capacity(incoming_dim_1 * ids_1.len());
            for &sample_1 in &ids_1 {
                let values = self.memo[incoming_edge_1][sample_1].clone().ok_or(
                    TreeAciError::InternalInvariant {
                        message:
                            "incoming sample frame was not memoized before batched contraction",
                    },
                )?;
                if values.len() != incoming_dim_1 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v1_data.extend(values);
            }
            let v1 = Matrix::from_col_major_vec(incoming_dim_1, ids_1.len(), v1_data);

            let mut v2_data = Vec::with_capacity(incoming_dim_2 * ids_2.len());
            for &sample_2 in &ids_2 {
                let values = self.memo[incoming_edge_2][sample_2].clone().ok_or(
                    TreeAciError::InternalInvariant {
                        message:
                            "incoming sample frame was not memoized before batched contraction",
                    },
                )?;
                if values.len() != incoming_dim_2 {
                    return Err(TreeAciError::InternalInvariant {
                        message: "incoming frame length differs from its bond dimension",
                    });
                }
                v2_data.extend(values);
            }
            let v2 = Matrix::from_col_major_vec(incoming_dim_2, ids_2.len(), v2_data);

            let batched = two_incoming_core_matrix_batched(
                core,
                outgoing_axis,
                incoming_axis_1,
                incoming_axis_2,
                base_offset,
                outgoing_dim,
                incoming_dim_1,
                incoming_dim_2,
                &v1,
                &v2,
            )?;

            for (sample, sample_1, sample_2) in group_samples {
                let n1 = position_1[&sample_1];
                let n2 = position_2[&sample_2];
                let values: Vec<T> = (0..outgoing_dim)
                    .map(|out| batched[[out + outgoing_dim * n1, n2]])
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
    let incoming_views = incoming_frames
        .iter()
        .map(|(edge, values)| (*edge, values.as_slice()))
        .collect::<Vec<_>>();
    contract_prepared_core_slices(
        input,
        problem,
        cores,
        edge,
        local_coordinate,
        &incoming_views,
    )
}

fn contract_prepared_core_slices<T: TreeAciScalar, V: TreeAciNode>(
    input: &TreeTN<IdxTensor, V>,
    problem: &PreparedTreeProblem<V>,
    cores: &[PreparedCore<T>],
    edge: DirectedEdgeId,
    local_coordinate: usize,
    incoming_frames: &[(DirectedEdgeId, &[T])],
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
        incoming_axes.push((axis_of(&core.indices, incoming_bond)?, *values));
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
    incoming_axes: &[(usize, &[T])],
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

/// Gathers one fixed-physical, single-incoming core slice into a column-major
/// matrix for the one- and two-incoming batched contraction kernels.
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

/// Gathers every physical slice of a single-incoming core into one matrix.
///
/// Rows are `(outgoing, local_physical)` in column-major product order, so a
/// single multiplication by incoming frame columns produces all candidate
/// physical coordinates without repeated small BLAS dispatches.
#[allow(clippy::too_many_arguments)]
fn single_incoming_all_physical_core_matrix<T: TreeAciScalar>(
    core: &PreparedCore<T>,
    outgoing_axis: usize,
    incoming_axis: usize,
    physical: &LocalPhysicalPlan,
    physical_axes: &[usize],
    outgoing_dim: usize,
    incoming_dim: usize,
) -> Matrix<T> {
    let rows = outgoing_dim * physical.local_dim;
    let outgoing_stride = core.strides[outgoing_axis];
    let incoming_stride = core.strides[incoming_axis];
    let mut data = Vec::with_capacity(rows * incoming_dim);
    for incoming_value in 0..incoming_dim {
        for local_coordinate in 0..physical.local_dim {
            let physical_offset = physical_axes
                .iter()
                .enumerate()
                .map(|(physical_axis, &core_axis)| {
                    let coordinate = (local_coordinate / physical.strides[physical_axis])
                        % physical.dims[physical_axis];
                    coordinate * core.strides[core_axis]
                })
                .sum::<usize>();
            for outgoing_value in 0..outgoing_dim {
                let offset = physical_offset
                    + incoming_value * incoming_stride
                    + outgoing_value * outgoing_stride;
                data.push(core.values[offset]);
            }
        }
    }
    Matrix::from_col_major_vec(rows, incoming_dim, data)
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
#[allow(clippy::too_many_arguments)]
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
