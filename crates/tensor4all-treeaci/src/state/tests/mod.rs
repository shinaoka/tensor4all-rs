use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::CanonicalForm;
use tensor4all_treetn::TreeTN;

use super::TreeAciState;
use crate::initialize::{build_random_output, initial_edge_ranks};
use crate::problem::prepare_problem;
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
    assert_eq!(state.output.canonical_form(), Some(CanonicalForm::CI));
}

#[test]
fn rank_deficient_initial_guess_uses_canonicalized_active_rank() {
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

    assert_eq!(state.edge_ranks, vec![1]);
    assert_eq!(
        state
            .output
            .bond_index(state.output.edge_between(&0, &1).unwrap())
            .unwrap()
            .dim(),
        1
    );
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
    let problem = prepare_problem(&inputs, &options).unwrap();
    let edge_ranks = initial_edge_ranks(&inputs, &problem, &options).unwrap();
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
                max_core_elements: 3,
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
