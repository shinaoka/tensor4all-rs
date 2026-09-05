use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

use super::TreeAciState;
use crate::initialize::{algebraic_edge_bounds, build_random_output, initial_edge_ranks};
use crate::problem::prepare_problem;
use crate::schedule::run_local_sweeps;
use crate::{TreeAciError, TreeAciOptions};

fn two_node_tree(sites: [DynIndex; 2], rank: usize, scale: f64) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(rank);
    let left_len = sites[0].dim() * rank;
    let right_len = sites[1].dim() * rank;
    let left = IdxTensor::from_dense(
        vec![sites[0].clone(), bond.clone()],
        (0..left_len)
            .map(|offset| scale * (offset + 1) as f64)
            .collect(),
    )
    .unwrap();
    let right = IdxTensor::from_dense(
        vec![bond, sites[1].clone()],
        (0..right_len)
            .map(|offset| scale * (offset + 3) as f64)
            .collect(),
    )
    .unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn rank_deficient_two_node_tree(sites: [DynIndex; 2]) -> TreeTN<IdxTensor, usize> {
    let bond = DynIndex::new_dyn(2);
    let left = IdxTensor::from_dense(
        vec![sites[0].clone(), bond.clone()],
        vec![1.0, 2.0, 1.0, 2.0],
    )
    .unwrap();
    let right =
        IdxTensor::from_dense(vec![bond, sites[1].clone()], vec![3.0, 4.0, 3.0, 4.0]).unwrap();
    TreeTN::from_tensors(vec![left, right], vec![0, 1]).unwrap()
}

fn full_rank_chain_guess(n_sites: usize, chi: usize) -> TreeTN<IdxTensor, usize> {
    let sites = (0..n_sites)
        .map(|_| DynIndex::new_dyn(2))
        .collect::<Vec<_>>();
    let link_dims = (0..n_sites - 1)
        .map(|bond| {
            let exact = 2usize.pow((bond + 1).min(n_sites - bond - 1) as u32);
            chi.min(exact)
        })
        .collect::<Vec<_>>();
    let bonds = link_dims
        .iter()
        .map(|&dim| DynIndex::new_dyn(dim))
        .collect::<Vec<_>>();
    let tensors = (0..n_sites)
        .map(|site| {
            let left_dim = if site == 0 { 1 } else { link_dims[site - 1] };
            let right_dim = link_dims.get(site).copied().unwrap_or(1);
            let mut indices = Vec::with_capacity(3);
            if site > 0 {
                indices.push(bonds[site - 1].clone());
            }
            indices.push(sites[site].clone());
            if site + 1 < n_sites {
                indices.push(bonds[site].clone());
            }
            let mut values = Vec::with_capacity(left_dim * 2 * right_dim);
            for right in 0..right_dim {
                for physical in 0..2 {
                    for left in 0..left_dim {
                        let input_f = 2.0;
                        let site_f = site as f64 + 1.0;
                        let physical_f = physical as f64 + 1.0;
                        let left_f = left as f64 + 1.0;
                        let right_f = right as f64 + 1.0;
                        let phase = 0.173 * input_f * site_f
                            + 0.193 * physical_f
                            + 0.071 * left_f * right_f
                            + 0.109 * input_f * left_f
                            + 0.131 * site_f * right_f;
                        let mixing = 0.29 * phase.sin()
                            + 0.23
                                * (0.157 * input_f * physical_f * right_f
                                    + 0.211 * site_f * left_f)
                                    .cos()
                            + 0.17
                                * (left_f / (left_dim as f64 + 1.0)
                                    - right_f / (right_dim as f64 + 1.0))
                                * physical_f;
                        values.push((0.31 + mixing) / ((left_dim * right_dim) as f64).powf(0.25));
                    }
                }
            }
            IdxTensor::from_dense(indices, values).unwrap()
        })
        .collect::<Vec<_>>();
    TreeTN::from_tensors(tensors, (0..n_sites).collect()).unwrap()
}

fn output_values(state: &TreeAciState<'_, f64, usize>) -> Vec<Vec<f64>> {
    tree_values(&state.output, &state.problem.node_order)
}

fn tree_values(tree: &TreeTN<IdxTensor, usize>, node_order: &[usize]) -> Vec<Vec<f64>> {
    node_order
        .iter()
        .map(|node| {
            let index = tree.node_index(node).unwrap();
            tree.tensor(index).unwrap().to_vec::<f64>().unwrap()
        })
        .collect()
}

#[test]
fn random_initialization_is_reproducible_and_seed_sensitive() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let input = two_node_tree(sites, 2, 1.0);
    let options = TreeAciOptions {
        rng_seed: 7,
        ..TreeAciOptions::default()
    };
    let first_inputs = vec![input.clone()];
    let second_inputs = vec![input.clone()];
    let different_inputs = vec![input];
    let first = TreeAciState::<f64, usize>::initialize(&first_inputs, &options).unwrap();
    let second = TreeAciState::<f64, usize>::initialize(&second_inputs, &options).unwrap();
    let different = TreeAciState::<f64, usize>::initialize(
        &different_inputs,
        &TreeAciOptions {
            rng_seed: 8,
            ..TreeAciOptions::default()
        },
    )
    .unwrap();

    assert_eq!(output_values(&first), output_values(&second));
    assert_ne!(output_values(&first), output_values(&different));
}

#[test]
fn ranks_active_samples_and_frames_start_consistent() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(3)];
    let inputs = vec![two_node_tree(sites, 3, 1.0)];
    let state =
        TreeAciState::<f64, usize>::initialize(&inputs, &TreeAciOptions::default()).unwrap();

    assert_eq!(state.edge_ranks, vec![2]);
    assert_eq!(state.algebraic_edge_bounds, vec![2]);
    assert_eq!(
        state
            .output
            .bond_index(state.output.edge_between(&0, &1).unwrap())
            .unwrap()
            .dim(),
        2
    );
    assert!(state.candidates.ids.iter().all(|ids| ids.len() == 2));
    for (edge, ids) in state.candidates.ids.iter().enumerate() {
        assert!(state.input_frames.frames[0][edge].sample_count >= ids.len());
    }
    assert!(state.output.same_topology(&state.inputs[0]));
    assert_eq!(state.output.site_space(&0), state.inputs[0].site_space(&0));
    assert_eq!(state.output.site_space(&1), state.inputs[0].site_space(&1));
}

#[test]
fn explicit_guess_is_preserved_before_the_first_sweep() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let input = two_node_tree(sites.clone(), 2, 1.0);
    let guess = two_node_tree(sites, 1, 11.0);
    let expected = guess.to_dense().unwrap();
    let inputs = vec![input];
    let state = TreeAciState::<f64, usize>::initialize(
        &inputs,
        &TreeAciOptions {
            initial_guess: Some(guess),
            ..TreeAciOptions::default()
        },
    )
    .unwrap();

    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&expected, 1.0e-12, 0.0)
        .unwrap());
    assert_eq!(state.edge_ranks, vec![1]);
    assert_eq!(state.output.canonical_form(), None);
    assert_eq!(state.output.canonical_region().len(), 1);
}

#[test]
fn rank_deficient_initial_guess_defers_rank_reduction_to_the_first_sweep() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(sites.clone(), 1, 1.0)];
    let guess = rank_deficient_two_node_tree(sites);

    let state = TreeAciState::<f64, usize>::initialize(
        &inputs,
        &TreeAciOptions {
            initial_guess: Some(guess),
            ..TreeAciOptions::default()
        },
    )
    .expect("rank-deficient initial guess should initialize");

    assert_eq!(state.edge_ranks, vec![2]);
    assert_eq!(
        state
            .output
            .bond_index(state.output.edge_between(&0, &1).unwrap())
            .unwrap()
            .dim(),
        2
    );
    assert_eq!(state.output.canonical_form(), None);
}

#[test]
fn full_rank_chain_initial_guess_is_accepted_and_preserved() {
    let guess = full_rank_chain_guess(16, 128);
    let expected = guess.to_dense().unwrap();
    let inputs = vec![guess.clone()];
    let state = TreeAciState::<f64, usize>::initialize(
        &inputs,
        &TreeAciOptions {
            initial_guess: Some(guess),
            ..TreeAciOptions::default()
        },
    )
    .expect("a valid full-rank chain guess should initialize");

    assert!(state
        .output
        .to_dense()
        .unwrap()
        .isapprox(&expected, 1.0e-10, 1.0e-12)
        .unwrap());
    assert_eq!(state.edge_ranks.iter().copied().max(), Some(128));
}

/// Opt-in phase timing and candidate-cache telemetry for the high-rank chain
/// parity workload. Run with `--ignored --nocapture`; wall time is diagnostic,
/// while the hit/miss counts expose whether persistent candidate retention is
/// doing useful work on this path.
#[test]
#[ignore]
fn profile_high_rank_chain_phases_and_candidate_cache() {
    // [AI Supplied] Use non-smoke bond dimensions and expose the phase scaling.
    for chi in [64, 128, 256] {
        let enable_global_guard =
            std::env::var("T4A_TREEACI_ENABLE_PROFILE_GUARD").as_deref() == Ok("1");
        let guess = full_rank_chain_guess(32, chi);
        let inputs = vec![guess.clone(), guess.clone()];
        let options = TreeAciOptions {
            tolerance: 1.0e-8,
            max_bond_dim: Some(4096),
            max_sweeps: 20,
            min_sweeps: 2,
            initial_guess: Some(guess),
            enable_global_guard,
            ..TreeAciOptions::default()
        };

        crate::frames::candidate_debug_stats::reset();
        crate::frames::debug_stats::reset();
        super::profile_debug_stats::reset();
        let initialize_started = std::time::Instant::now();
        let mut state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();
        let initialize_elapsed = initialize_started.elapsed();
        let sweep_started = std::time::Instant::now();
        let history = run_local_sweeps(&mut state, &options, &mut |batch, output| {
            for (point, value) in output.iter_mut().enumerate() {
                *value = batch.get(0, point)? * batch.get(1, point)?;
            }
            Ok(())
        })
        .unwrap();
        let sweep_elapsed = sweep_started.elapsed();
        let profile = super::profile_debug_stats::snapshot();
        let (base_frame_bytes, candidate_cache_entries, candidate_cache_bytes) =
            state.input_frames.cache_debug_totals();

        let owned_matmul =
            std::env::var("T4A_TREEACI_USE_OWNED_LOCAL_MATMUL").as_deref() == Ok("1");
        let candidate_cache =
            std::env::var("T4A_TREEACI_DISABLE_CANDIDATE_CACHE").as_deref() != Ok("1");
        eprintln!(
            "high-rank chain owned_matmul={owned_matmul} candidate_cache={candidate_cache} chi={chi}: initialize={initialize_elapsed:?} [prepare={:?}, output={:?}, bootstrap={:?}, frames={:?}], sweeps={sweep_elapsed:?} [proposals={:?}: prepare={:?}, input_frames={:?} {{row={:?}, col={:?}, pack={:?}, matmul={:?}, scatter={:?}}}, operator={:?}, luci={:?}; output_staging={:?} {{clone={:?}, replace={:?}}}, sample_staging={:?}, frame_extension={:?}, commits={}], completed={}, candidate_hits={}, candidate_misses={}",
            profile.preparation,
            profile.output,
            profile.bootstrap,
            profile.frames,
            profile.proposals,
            profile.local_preparation,
            profile.local_input_frames,
            profile.local_row_frames,
            profile.local_col_frames,
            profile.local_frame_pack,
            profile.local_frame_matmul,
            profile.local_frame_scatter,
            profile.operator,
            profile.luci,
            profile.output_staging,
            profile.output_clone,
            profile.output_replace,
            profile.sample_staging,
            profile.frame_extension,
            profile.commits,
            history.max_ranks.len(),
            crate::frames::candidate_debug_stats::hits(),
            crate::frames::candidate_debug_stats::misses(),
        );
        eprintln!(
            "high-rank chain chi={chi} cumulative small-work ledger: core_pack candidate={:?}/{} calls/{} values, stored={:?}/{} calls/{} values, unique={} calls/{} values; extend={:?} over {} calls {{setup={:?}, scan={:?} across {} edge-input visits (reused={}, grown={}), compute={:?}, rebuild={:?}, finalize={:?}, memo_slots={}, old_values_copied={}, new_values_copied={}}}",
            profile.candidate_core_pack,
            profile.candidate_core_pack_calls,
            profile.candidate_core_pack_values,
            profile.stored_core_pack,
            profile.stored_core_pack_calls,
            profile.stored_core_pack_values,
            profile.unique_core_pack_calls,
            profile.unique_core_pack_values,
            profile.frame_extension,
            profile.frame_extension_calls,
            profile.frame_extension_setup,
            profile.frame_extension_scan,
            profile.frame_extension_scanned_edges,
            profile.frame_extension_reused_edges,
            profile.frame_extension_grown_edges,
            profile.frame_extension_compute,
            profile.frame_extension_rebuild,
            profile.frame_extension_finalize,
            profile.frame_extension_memo_slots,
            profile.frame_extension_old_values_copied,
            profile.frame_extension_new_values_copied,
        );
        eprintln!(
            "high-rank chain chi={chi} recursive frame builder: compute_calls={} (scalar={}, batched={}), memo_hit_vec_clones={}",
            crate::frames::debug_stats::compute_calls(),
            crate::frames::debug_stats::scalar_compute_calls(),
            crate::frames::debug_stats::batched_compute_calls(),
            crate::frames::debug_stats::memo_hit_copies(),
        );
        eprintln!(
            "high-rank chain chi={chi} pass metadata: schedule_clone={:?}, deferred_canonicalization={:?}",
            profile.schedule_clone, profile.deferred_canonicalization,
        );
        eprintln!(
            "high-rank chain chi={chi} retained frame payload: base={base_frame_bytes} bytes, candidate_cache={candidate_cache_entries} entries/{candidate_cache_bytes} bytes"
        );
        eprintln!(
            "high-rank chain chi={chi} output replacement details: factor_indices={:?}, tensor_build={:?}, lookup={:?}, bond_replace={:?}, tensor_replace={:?}, metadata={:?}",
            profile.output_factor_indices,
            profile.output_tensor_build,
            profile.output_lookup,
            profile.output_bond_replace,
            profile.output_tensor_replace,
            profile.output_metadata,
        );
        eprintln!(
            "high-rank chain chi={chi} global guard enabled={enable_global_guard}: search={:?}, injection={:?} {{candidate_clone={:?}, projection={:?}, output_padding={:?}, frame_extension={:?}}}, input_eval={:?}/{} calls/{} scalar results, output_eval={:?}/{} calls/{} scalar results",
            profile.global_guard,
            profile.global_injection,
            profile.guard_candidate_clone,
            profile.guard_projection,
            profile.guard_output_padding,
            profile.guard_frame_extension,
            profile.guard_input_evaluation,
            profile.guard_input_calls,
            profile.guard_input_points,
            profile.guard_output_evaluation,
            profile.guard_output_calls,
            profile.guard_output_points,
        );
        eprintln!(
            "high-rank chain chi={chi} single-incoming candidate details: cache_scan={:?}/{} items, group_setup={:?}, core_pack={:?}, frame_pack={:?}, backend={:?}, result_extract_and_cache={:?}",
            profile.candidate_cache_scan,
            profile.candidate_scan_items,
            profile.candidate_group_setup,
            profile.candidate_core_pack,
            profile.candidate_frame_pack,
            profile.candidate_backend,
            profile.candidate_result_cache,
        );
        if enable_global_guard {
            assert!(history.max_ranks.len() >= 2);
        } else {
            assert_eq!(history.max_ranks.len(), 2);
        }
    }
}

#[test]
fn unseeded_initialization_defers_numeric_canonicalization() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(sites, 2, 1.0)];
    let options = TreeAciOptions {
        root: Some(1),
        rng_seed: 7,
        ..TreeAciOptions::default()
    };
    let problem = prepare_problem::<f64, _>(&inputs, &options).unwrap();
    let algebraic_bounds = algebraic_edge_bounds(&problem).unwrap();
    let edge_ranks = initial_edge_ranks(&inputs, &problem, &options, &algebraic_bounds).unwrap();
    let raw =
        build_random_output::<f64, usize>(&inputs[0], &problem, &edge_ranks, &options).unwrap();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).unwrap();

    assert_eq!(state.output.canonical_region().len(), 1);
    assert!(state.output.canonical_region().contains(&1));
    assert_eq!(state.output.canonical_form(), None);
    assert_eq!(
        output_values(&state),
        tree_values(&raw, &state.problem.node_order)
    );
}

#[test]
fn explicit_guess_rank_and_resource_limits_are_rejected() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(sites.clone(), 1, 1.0)];
    let guess = two_node_tree(sites.clone(), 2, 1.0);
    let limited_options = TreeAciOptions {
        max_bond_dim: Some(1),
        initial_guess: Some(guess.clone()),
        ..TreeAciOptions::default()
    };
    assert!(matches!(
        TreeAciState::<f64, usize>::initialize(&inputs, &limited_options),
        Err(TreeAciError::InvalidInitialGuess { .. })
    ));
    let inputs = vec![two_node_tree(sites, 1, 1.0)];
    assert!(matches!(
        TreeAciState::<f64, usize>::initialize(
            &inputs,
            &TreeAciOptions {
                initial_guess: Some(guess),
                max_core_elements: Some(3),
                ..TreeAciOptions::default()
            }
        ),
        Err(TreeAciError::ResourceLimit {
            resource: "core elements",
            requested: 4,
            limit: 3
        })
    ));
}

#[test]
fn explicit_guess_requires_full_physical_index_compatibility() {
    let sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let inputs = vec![two_node_tree(sites, 1, 1.0)];
    let incompatible = two_node_tree([DynIndex::new_dyn(2), DynIndex::new_dyn(2)], 1, 1.0);
    assert!(matches!(
        TreeAciState::<f64, usize>::initialize(
            &inputs,
            &TreeAciOptions {
                initial_guess: Some(incompatible),
                ..TreeAciOptions::default()
            }
        ),
        Err(TreeAciError::InvalidInitialGuess { .. })
    ));
}

#[test]
fn state_borrows_inputs_rather_than_owning_them() {
    let inputs = vec![two_node_tree(
        [DynIndex::new_dyn(2), DynIndex::new_dyn(2)],
        1,
        1.0,
    )];
    let options = TreeAciOptions::default();
    let state = TreeAciState::<f64, usize>::initialize(&inputs, &options).expect("initialize");

    assert_eq!(state.inputs.len(), inputs.len());
    assert!(std::ptr::eq(&state.inputs[0], &inputs[0]));
}
