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
    use std::{cell::RefCell, time::Duration};

    #[derive(Clone, Copy, Debug, Default)]
    pub(crate) struct Snapshot {
        pub(crate) preparation: Duration,
        pub(crate) output: Duration,
        pub(crate) bootstrap: Duration,
        pub(crate) frames: Duration,
        pub(crate) proposals: Duration,
        pub(crate) local_preparation: Duration,
        pub(crate) local_input_frames: Duration,
        pub(crate) operator: Duration,
        pub(crate) luci: Duration,
        pub(crate) output_staging: Duration,
        pub(crate) sample_staging: Duration,
        pub(crate) frame_extension: Duration,
        pub(crate) commits: usize,
    }

    thread_local! {
        static STATS: RefCell<Snapshot> = RefCell::new(Snapshot::default());
    }

    pub(crate) fn reset() {
        STATS.with(|stats| *stats.borrow_mut() = Snapshot::default());
    }

    pub(crate) fn record(update: impl FnOnce(&mut Snapshot)) {
        STATS.with(|stats| update(&mut stats.borrow_mut()));
    }

    pub(crate) fn snapshot() -> Snapshot {
        STATS.with(|stats| *stats.borrow())
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
        let problem = prepare_problem(inputs, options)?;
        let algebraic_edge_bounds = algebraic_edge_bounds(&problem)?;
        let initial_edge_ranks = initial_edge_ranks(inputs, &problem, options)?;
        #[cfg(test)]
        profile_debug_stats::record(|stats| stats.preparation = stage_started.elapsed());
        #[cfg(test)]
        let stage_started = std::time::Instant::now();
        let mut output = if let Some(guess) = &options.initial_guess {
            validate_initial_guess::<T, V>(guess, &inputs[0], &problem, options)?;
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
mod tests;
