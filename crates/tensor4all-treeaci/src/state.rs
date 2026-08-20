//! Owned native state for tree ACI sweeps.

// INVARIANT: the state is crate-private until the directional sweep executor
// and public elementwise entry point are complete.
#![allow(dead_code)]

use tensor4all_core::{IdxTensor, IndexLike};
use tensor4all_treetn::{CanonicalForm, CanonicalizationOptions, TreeTN};

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
        let problem = prepare_problem(inputs, options)?;
        let algebraic_edge_bounds = algebraic_edge_bounds(&problem)?;
        let edge_ranks = initial_edge_ranks(inputs, &problem, options)?;
        let output = if let Some(guess) = &options.initial_guess {
            validate_initial_guess::<T, V>(guess, &inputs[0], &problem, options)?;
            let mut output = guess.clone();
            output.canonicalize_mut(
                [problem.root.clone()],
                CanonicalizationOptions::default().with_form(CanonicalForm::CI),
            )?;
            output
        } else {
            let mut output =
                build_random_output::<T, V>(&inputs[0], &problem, &edge_ranks, options)?;
            output.set_canonical_region([problem.root.clone()])?;
            output
        };
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
        let (sample_arena, candidates, pivots) = bootstrap_samples(&problem, &edge_ranks)?;
        let input_frames = InputFrameStore::from_samples(inputs, &problem, &sample_arena)?;
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
