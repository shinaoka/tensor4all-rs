//! Owned native state for tree ACI sweeps.

use tensor4all_core::{IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use crate::{
    frames::InputFrameStore,
    initialize::{
        algebraic_edge_bounds, bootstrap_samples, build_random_output, initial_edge_ranks,
        validate_initial_guess,
    },
    problem::{prepare_problem, PreparedTreeProblem},
    samples::{CandidateSets, PivotPairs, SampleArena},
    Result, TreeAciNode, TreeAciOptions, TreeAciScalar,
};

#[cfg(test)]
pub(crate) mod profile_debug_stats {
    use std::{cell::RefCell, collections::HashSet, time::Duration};

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Snapshot {
        pub(crate) preparation: Duration,
        pub(crate) output: Duration,
        pub(crate) bootstrap: Duration,
        pub(crate) frames: Duration,
        pub(crate) proposals: Duration,
        pub(crate) schedule_clone: Duration,
        pub(crate) deferred_canonicalization: Duration,
        pub(crate) local_preparation: Duration,
        pub(crate) local_input_frames: Duration,
        // [AI Supplied] Diagnostic-only decomposition of `local_input_frames`.
        pub(crate) local_row_frames: Duration,
        pub(crate) local_col_frames: Duration,
        pub(crate) local_frame_pack: Duration,
        pub(crate) local_frame_matmul: Duration,
        pub(crate) local_frame_scatter: Duration,
        // [AI Supplied] Paired #714 counters: the legacy diagnostic extracts
        // one owned vector per candidate, while the production path consumes
        // one packed batch per side.
        pub(crate) local_legacy_frame_vectors: usize,
        pub(crate) local_legacy_frame_values: usize,
        pub(crate) local_packed_frame_batches: usize,
        pub(crate) local_packed_frame_values: usize,
        pub(crate) operator: Duration,
        pub(crate) luci: Duration,
        pub(crate) output_staging: Duration,
        // [AI Supplied] Diagnostic-only split of whole-tree metadata cloning
        // from the local edge-core replacement performed during a commit.
        pub(crate) output_clone: Duration,
        pub(crate) output_replace: Duration,
        pub(crate) output_factor_indices: Duration,
        pub(crate) output_tensor_build: Duration,
        pub(crate) output_lookup: Duration,
        pub(crate) output_bond_replace: Duration,
        pub(crate) output_tensor_replace: Duration,
        pub(crate) output_metadata: Duration,
        pub(crate) sample_staging: Duration,
        pub(crate) frame_extension: Duration,
        // [AI Supplied] Diagnostic-only accounting for the default global
        // Guard and its dynamic-scalar TreeTN evaluator boundary.
        pub(crate) global_guard: Duration,
        pub(crate) global_injection: Duration,
        pub(crate) guard_candidate_clone: Duration,
        pub(crate) guard_projection: Duration,
        pub(crate) guard_output_padding: Duration,
        pub(crate) guard_frame_extension: Duration,
        pub(crate) guard_input_evaluation: Duration,
        pub(crate) guard_output_evaluation: Duration,
        pub(crate) guard_input_points: usize,
        pub(crate) guard_output_points: usize,
        pub(crate) guard_input_calls: usize,
        pub(crate) guard_output_calls: usize,
        // [AI Supplied] Diagnostic-only decomposition and work counts for
        // repeated `InputFrameStore::extend` calls.
        pub(crate) frame_extension_setup: Duration,
        pub(crate) frame_extension_scan: Duration,
        pub(crate) frame_extension_compute: Duration,
        pub(crate) frame_extension_rebuild: Duration,
        pub(crate) frame_extension_finalize: Duration,
        pub(crate) frame_extension_calls: usize,
        pub(crate) frame_extension_scanned_edges: usize,
        pub(crate) frame_extension_reused_edges: usize,
        pub(crate) frame_extension_grown_edges: usize,
        pub(crate) frame_extension_memo_slots: usize,
        pub(crate) frame_extension_old_values_copied: usize,
        pub(crate) frame_extension_new_values_copied: usize,
        // [AI Supplied] Core-repacking telemetry for the two one-incoming
        // call sites, separated into candidate and stored-frame work.
        pub(crate) candidate_core_pack: Duration,
        pub(crate) candidate_core_pack_calls: usize,
        pub(crate) candidate_core_pack_values: usize,
        pub(crate) candidate_cache_scan: Duration,
        pub(crate) candidate_group_setup: Duration,
        pub(crate) candidate_frame_pack: Duration,
        pub(crate) candidate_backend: Duration,
        pub(crate) candidate_result_cache: Duration,
        pub(crate) candidate_scan_items: usize,
        pub(crate) stored_core_pack: Duration,
        pub(crate) stored_core_pack_calls: usize,
        pub(crate) stored_core_pack_values: usize,
        pub(crate) unique_core_pack_calls: usize,
        pub(crate) unique_core_pack_values: usize,
        pub(crate) commits: usize,
    }

    thread_local! {
        static STATS: RefCell<Snapshot> = RefCell::new(Snapshot::default());
        static CORE_PACK_KEYS: RefCell<HashSet<(usize, usize)>> = RefCell::new(HashSet::new());
    }

    pub(crate) fn reset() {
        STATS.with(|stats| *stats.borrow_mut() = Snapshot::default());
        CORE_PACK_KEYS.with(|keys| keys.borrow_mut().clear());
    }

    pub(crate) fn record(update: impl FnOnce(&mut Snapshot)) {
        STATS.with(|stats| update(&mut stats.borrow_mut()));
    }

    pub(crate) fn snapshot() -> Snapshot {
        STATS.with(|stats| *stats.borrow())
    }

    pub(crate) fn record_core_pack_identity(input: usize, edge: usize, values: usize) {
        let inserted = CORE_PACK_KEYS.with(|keys| keys.borrow_mut().insert((input, edge)));
        if inserted {
            record(|stats| {
                stats.unique_core_pack_calls += 1;
                stats.unique_core_pack_values += values;
            });
        }
    }
}

#[derive(Clone)]
pub(crate) struct TreeAciState<'a, T: TreeAciScalar, V: TreeAciNode> {
    pub(crate) problem: PreparedTreeProblem<V>,
    pub(crate) inputs: &'a [TreeTN<IdxTensor, V>],
    pub(crate) output: TreeTN<IdxTensor, V>,
    pub(crate) sample_arena: SampleArena,
    pub(crate) candidates: CandidateSets,
    pub(crate) pivots: PivotPairs,
    pub(crate) input_frames: InputFrameStore<T>,
    pub(crate) edge_ranks: Vec<usize>,
    pub(crate) algebraic_edge_bounds: Vec<usize>,
    pub(crate) edge_errors: Vec<f64>,
    pub(crate) edge_scales: Vec<f64>,
    pub(crate) generation: u64,
}

impl<'a, T: TreeAciScalar, V: TreeAciNode> TreeAciState<'a, T, V> {
    pub(crate) fn initialize(
        inputs: &'a [TreeTN<IdxTensor, V>],
        options: &TreeAciOptions<V>,
    ) -> Result<Self> {
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let problem = prepare_problem::<T, V>(inputs, options)?;
        let algebraic_edge_bounds = algebraic_edge_bounds(&problem)?;
        let initial_edge_ranks =
            initial_edge_ranks(inputs, &problem, options, &algebraic_edge_bounds)?;
        #[cfg(test)]
        profile_debug_stats::record(|stats| stats.preparation = stage_started.elapsed());
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let mut output = if let Some(guess) = &options.initial_guess {
            validate_initial_guess::<T, V>(guess, &inputs[0], &problem)?;
            guess.clone()
        } else {
            build_random_output::<T, V>(&inputs[0], &problem, &initial_edge_ranks, options)?
        };
        // A complete first directional pass replaces every core with CI
        // factors before `finalize_deferred_canonicalization` establishes the
        // numerical form. Doing a full-rank CI canonicalization here is both
        // redundant and pathological for a high-rank explicit guess. The
        // generated-output path has always deferred this work; explicit
        // guesses follow the same state transition because bootstrap depends
        // only on their validated bond dimensions, not on their gauge.
        output.set_canonical_region([problem.root.clone()])?;
        #[cfg(test)]
        profile_debug_stats::record(|stats| stats.output = stage_started.elapsed());
        let edge_ranks = initial_edge_ranks;
        for (edge_number, expected) in edge_ranks.iter().copied().enumerate() {
            let edge = &problem.directed_edges[2 * edge_number];
            let graph_edge = output.edge_between(&edge.from, &edge.to).ok_or(
                crate::TreeAciError::InternalInvariant {
                    message: "initialized output is missing a prepared edge",
                },
            )?;
            let actual = output
                .bond_index(graph_edge)
                .ok_or(crate::TreeAciError::InternalInvariant {
                    message: "initialized output edge has no bond index",
                })?
                .dim();
            if actual != expected {
                return Err(crate::TreeAciError::InternalInvariant {
                    message: "initialized output rank differs from active rank target",
                });
            }
        }
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let (sample_arena, candidates, pivots) = bootstrap_samples(&problem, &edge_ranks)?;
        #[cfg(test)]
        profile_debug_stats::record(|stats| stats.bootstrap = stage_started.elapsed());
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let input_frames = InputFrameStore::from_samples(inputs, &problem, &sample_arena)?;
        #[cfg(test)]
        profile_debug_stats::record(|stats| stats.frames = stage_started.elapsed());
        let generation = candidates.generation;
        let edge_count = edge_ranks.len();
        Ok(Self {
            problem,
            inputs,
            output,
            sample_arena,
            candidates,
            pivots,
            input_frames,
            edge_ranks,
            algebraic_edge_bounds,
            edge_errors: vec![0.0; edge_count],
            edge_scales: vec![0.0; edge_count],
            generation,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
