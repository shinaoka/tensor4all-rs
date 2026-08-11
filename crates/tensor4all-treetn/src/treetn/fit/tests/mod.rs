use super::*;
use crate::treetn::localupdate::{LocalUpdateStep, LocalUpdater};
use tensor4all_core::{DynIndex, TensorDynLen};

/// Create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> TreeTN<TensorDynLen, String> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    make_two_node_treetn_with_sites(&s0, &s1)
}

fn make_two_node_treetn_with_sites(s0: &DynIndex, s1: &DynIndex) -> TreeTN<TensorDynLen, String> {
    let bond = DynIndex::new_dyn(3);

    let t0 = TensorDynLen::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();

    TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap()
}

fn make_contractible_two_node_pair() -> (TreeTN<TensorDynLen, String>, TreeTN<TensorDynLen, String>)
{
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    (
        make_two_node_treetn_with_sites(&s0, &s1),
        make_two_node_treetn_with_sites(&s0, &s1),
    )
}

fn make_contractible_two_node_pair_with_surviving_sites(
) -> (TreeTN<TensorDynLen, String>, TreeTN<TensorDynLen, String>) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let a0 = DynIndex::new_dyn(2);
    let a1 = DynIndex::new_dyn(2);
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![s0.clone(), a0, bond_a.clone()],
                (1..=8).map(|value| value as f64 / 8.0).collect(),
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![bond_a, s1.clone(), a1],
                (1..=8).map(|value| value as f64 / 10.0).collect(),
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![s0, b0, bond_b.clone()],
                (1..=8).map(|value| (value as f64 - 2.0) / 9.0).collect(),
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![bond_b, s1, b1],
                (1..=8).map(|value| (value as f64 + 1.0) / 11.0).collect(),
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    (tn_a, tn_b)
}

fn make_fit_initial_c(
    tn_a: &TreeTN<TensorDynLen, String>,
    tn_b: &TreeTN<TensorDynLen, String>,
    center: &str,
) -> TreeTN<TensorDynLen, String> {
    tn_a.contract_zipup_preserving_topology_with(
        tn_b,
        &center.to_string(),
        crate::CanonicalForm::Unitary,
        None,
        None,
    )
    .unwrap()
}

fn make_single_node_treetn() -> TreeTN<TensorDynLen, String> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let t = TensorDynLen::from_dense(vec![s0, s1], vec![1.0; 6]).unwrap();
    TreeTN::<TensorDynLen, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap()
}

fn make_three_node_treetn() -> TreeTN<TensorDynLen, String> {
    let s0 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0, bond01.clone()], vec![1.0; 6]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond01, s1, bond12.clone()], vec![1.0; 18]).unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond12, s2], vec![1.0; 6]).unwrap();

    TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0, t1, t2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap()
}

// ========================================================================
// FitEnvironment tests
// ========================================================================

#[test]
fn test_fit_environment_new() {
    let env = FitEnvironment::<TensorDynLen, String>::new();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
}

#[test]
fn test_fit_environment_default() {
    let env = FitEnvironment::<TensorDynLen, String>::default();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
}

#[test]
fn test_fit_environment_insert_and_get() {
    let mut env = FitEnvironment::<TensorDynLen, String>::new();

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

    env.insert("A".to_string(), "B".to_string(), t.clone());

    assert!(!env.is_empty());
    assert_eq!(env.len(), 1);
    assert!(env.contains(&"A".to_string(), &"B".to_string()));
    assert!(!env.contains(&"B".to_string(), &"A".to_string()));

    let retrieved = env.get(&"A".to_string(), &"B".to_string());
    assert!(retrieved.is_some());
}

#[test]
fn test_fit_environment_get_nonexistent() {
    let env = FitEnvironment::<TensorDynLen, String>::new();
    assert!(env.get(&"A".to_string(), &"B".to_string()).is_none());
}

#[test]
fn test_fit_environment_clear() {
    let mut env = FitEnvironment::<TensorDynLen, String>::new();

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

    env.insert("A".to_string(), "B".to_string(), t.clone());
    env.insert("B".to_string(), "A".to_string(), t.clone());
    assert_eq!(env.len(), 2);

    env.clear();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
    assert!(!env.contains(&"A".to_string(), &"B".to_string()));
}

#[test]
fn test_fit_environment_invalidate() {
    let mut env = FitEnvironment::<TensorDynLen, String>::new();
    let tn = make_two_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

    // Insert environments for both directions
    env.insert("A".to_string(), "B".to_string(), t.clone());
    env.insert("B".to_string(), "A".to_string(), t.clone());
    assert_eq!(env.len(), 2);

    // Invalidate node A - should remove env[(A, B)] and propagate
    env.invalidate(&["A".to_string()], &tn);

    // env[(A, B)] should be removed
    assert!(!env.contains(&"A".to_string(), &"B".to_string()));
    // env[(B, A)] should also be removed via propagation from A
    // (A's neighbor is B, so we remove env[(A, B)]; then propagate from A to B,
    //  but env[(B, A)] needs to check: from=A, to=B removes env[(A,B)],
    //  then propagates to env[(B, neighbor)] for neighbor != A - there are none for B except A)
    // Actually, invalidate_recursive removes env[(from, to)] = env[(A, B)],
    // then recursively goes to env[(B, x)] for x != A. B has no neighbors except A, so stops.
    // env[(B, A)] is NOT removed by invalidation of A.
    assert!(env.contains(&"B".to_string(), &"A".to_string()));
}

#[test]
fn test_fit_environment_verify_structural_consistency_empty() {
    let env = FitEnvironment::<TensorDynLen, String>::new();
    let tn = make_two_node_treetn();
    assert!(env.verify_structural_consistency(&tn).is_ok());
}

#[test]
fn test_fit_environment_verify_structural_consistency_valid() {
    let mut env = FitEnvironment::<TensorDynLen, String>::new();
    let tn = make_two_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

    // A is a leaf with only neighbor B. env[(A, B)] is valid alone.
    env.insert("A".to_string(), "B".to_string(), t.clone());
    assert!(env.verify_structural_consistency(&tn).is_ok());
}

#[test]
fn test_fit_environment_get_or_compute_caches_leaf_environment() {
    let (tn_a, tn_b) = make_contractible_two_node_pair_with_surviving_sites();
    let tn_c = make_fit_initial_c(&tn_a, &tn_b, "A");
    let mut env = FitEnvironment::<TensorDynLen, String>::new();

    let from = "A".to_string();
    let to = "B".to_string();
    let computed = env.get_or_compute(&from, &to, &tn_a, &tn_b, &tn_c).unwrap();
    assert!(env.contains(&from, &to));
    assert_eq!(env.len(), 1);

    let cached = env.get_or_compute(&from, &to, &tn_a, &tn_b, &tn_c).unwrap();
    assert_eq!(env.len(), 1);
    assert!(computed.distance(&cached).unwrap() < 1e-12);
}

#[test]
fn test_fit_environment_verify_structural_consistency_detects_missing_child_env() {
    let mut env = FitEnvironment::<TensorDynLen, String>::new();
    let tn = make_three_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s], vec![1.0, 2.0]).unwrap();

    // B is non-leaf toward A, so env[(C, B)] must also exist.
    env.insert("B".to_string(), "A".to_string(), t);
    let err = env
        .verify_structural_consistency(&tn)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Structural inconsistency"));
    assert!(err.contains("C"));
}

// ========================================================================
// FitContractionOptions tests
// ========================================================================

#[test]
fn test_fit_contraction_options_default() {
    let opts = FitContractionOptions::default();
    assert_eq!(opts.nfullsweeps, 1);
    assert!(opts.max_bond_dim.is_none());
    assert!(opts.rtol.is_none());
    assert_eq!(opts.factorize_alg, FactorizeAlg::SVD);
    assert!(opts.convergence_tol.is_none());
}

#[test]
fn test_fit_contraction_options_new() {
    let opts = FitContractionOptions::new(5);
    assert_eq!(opts.nfullsweeps, 5);
}

#[test]
fn test_fit_contraction_options_builders() {
    let opts = FitContractionOptions::new(2)
        .with_max_bond_dim(10)
        .with_rtol(1e-8)
        .with_factorize_alg(FactorizeAlg::LU)
        .with_convergence_tol(1e-6);

    assert_eq!(opts.nfullsweeps, 2);
    assert_eq!(opts.max_bond_dim, Some(10));
    assert_eq!(opts.rtol, Some(1e-8));
    assert_eq!(opts.factorize_alg, FactorizeAlg::LU);
    assert_eq!(opts.convergence_tol, Some(1e-6));
}

// ========================================================================
// FitUpdater tests
// ========================================================================

#[test]
fn test_fit_updater_new() {
    let tn_a = make_two_node_treetn();
    let tn_b = make_two_node_treetn();

    let updater = FitUpdater::new(tn_a, tn_b, Some(5), Some(1e-8));
    assert_eq!(updater.max_bond_dim, Some(5));
    assert_eq!(updater.rtol, Some(1e-8));
    assert_eq!(updater.factorize_alg, FactorizeAlg::SVD);
    assert!(updater.envs.is_empty());
}

#[test]
fn test_fit_updater_with_factorize_alg() {
    let tn_a = make_two_node_treetn();
    let tn_b = make_two_node_treetn();

    let updater = FitUpdater::new(tn_a, tn_b, None, None).with_factorize_alg(FactorizeAlg::LU);
    assert_eq!(updater.factorize_alg, FactorizeAlg::LU);
}

#[test]
fn test_fit_updater_update_requires_two_nodes() {
    let tn_a = make_two_node_treetn();
    let tn_b = make_two_node_treetn();
    let full_treetn = make_two_node_treetn();
    let mut updater = FitUpdater::new(tn_a, tn_b, None, None);

    let step = LocalUpdateStep {
        nodes: vec!["A".to_string()],
        new_center: "A".to_string(),
    };
    let err = updater
        .update(full_treetn.clone(), &step, &full_treetn)
        .unwrap_err()
        .to_string();
    assert!(err.contains("requires exactly 2 nodes"));
}

#[test]
fn test_fit_updater_after_step_invalidates_cached_region() {
    let tn_a = make_two_node_treetn();
    let tn_b = make_two_node_treetn();
    let full_treetn = make_two_node_treetn();
    let mut updater = FitUpdater::new(tn_a, tn_b, None, None);

    let s = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(vec![s], vec![1.0, 2.0]).unwrap();
    updater
        .envs
        .insert("A".to_string(), "B".to_string(), t.clone());
    updater.envs.insert("B".to_string(), "A".to_string(), t);

    let step = LocalUpdateStep {
        nodes: vec!["A".to_string(), "B".to_string()],
        new_center: "B".to_string(),
    };
    updater.after_step(&step, &full_treetn).unwrap();
    assert!(updater.envs.is_empty());
}

#[test]
fn test_fit_updater_update_pins_same_id_prime_pair_leg_order() {
    // Node A carries a same-ID primed pair (s, s.prime()) in its site space.
    // Both survive the 2-site contraction as external legs, so `left_inds`
    // contains two indices that compare equal by `.id()` alone. The left
    // factor's leg order must follow the canonical full-index sort (unprimed
    // before primed), not the contraction output order.
    let s = DynIndex::new_dyn(2);
    let s_prime = s.prime();
    let t_a = DynIndex::new_dyn(2);
    let t_b = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);

    // The primed index appears first in the first contracted tensor, so the
    // unsorted left_inds insertion order is [s', s] (the buggy stable sort
    // preserves it; the full-index sort must pin [s, s']).
    let a_u = TensorDynLen::from_dense(
        vec![s_prime.clone(), bond.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b_u = TensorDynLen::from_dense(
        vec![s.clone(), bond.clone()],
        vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap();
    let a_v = TensorDynLen::from_dense(
        vec![bond.clone(), t_a.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let b_v = TensorDynLen::from_dense(
        vec![bond.clone(), t_b.clone()],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![a_u, a_v],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![b_u, b_v],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Full network C: node A's site space holds the same-ID primed pair.
    let c_a = TensorDynLen::from_dense(
        vec![s_prime.clone(), s.clone(), bond.clone()],
        vec![1.0; 12],
    )
    .unwrap();
    let c_b = TensorDynLen::from_dense(vec![bond.clone(), t_a.clone(), t_b.clone()], vec![1.0; 12])
        .unwrap();
    let full_treetn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![c_a, c_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let site_c_u = full_treetn.site_space(&"A".to_string()).unwrap();
    assert!(site_c_u.contains(&s));
    assert!(site_c_u.contains(&s_prime));

    let mut updater = FitUpdater::new(tn_a, tn_b, None, None);
    let step = LocalUpdateStep {
        nodes: vec!["A".to_string(), "B".to_string()],
        new_center: "B".to_string(),
    };
    let updated = updater
        .update(full_treetn.clone(), &step, &full_treetn)
        .unwrap();

    let a_idx = updated.node_index(&"A".to_string()).unwrap();
    let new_a_inds = updated.tensor(a_idx).unwrap().indices();
    let pos_s = new_a_inds.iter().position(|i| i == &s).unwrap();
    let pos_s_prime = new_a_inds.iter().position(|i| i == &s_prime).unwrap();
    assert!(
        pos_s < pos_s_prime,
        "left factor must order unprimed before primed; got indices {new_a_inds:?}"
    );
}

#[test]
fn test_contract_fit_rejects_topology_mismatch() {
    let tn_a = make_two_node_treetn();
    let tn_b = make_single_node_treetn();
    let err = contract_fit(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        FitContractionOptions::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("same topology"));
}

#[test]
fn test_contract_fit_matches_naive_contraction_on_two_node_tree() {
    let (tn_a, tn_b) = make_contractible_two_node_pair();

    let fitted = contract_fit(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        FitContractionOptions::new(1).with_convergence_tol(1e-12),
    )
    .unwrap();

    let fitted_dense = fitted.to_dense().unwrap();
    let expected_dense = tn_a.contract_naive(&tn_b).unwrap();
    assert!(fitted_dense.sub(&expected_dense).unwrap().maxabs().unwrap() < 1e-10);
}

#[test]
fn test_contract_fit_positive_sweeps_do_not_skip_without_truncation_options() {
    set_fit_profile_enabled_for_tests(true);
    FIT_PROFILE_STATE.with(|state| {
        *state.borrow_mut() = None;
    });

    let (tn_a, tn_b) = make_contractible_two_node_pair();

    let fitted = contract_fit(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        FitContractionOptions::new(1),
    )
    .unwrap();

    set_fit_profile_enabled_for_tests(false);

    let dangling_profile = FIT_PROFILE_STATE.with(|state| state.borrow().is_some());
    assert!(
        !dangling_profile,
        "positive-sweep contract_fit should run the sweep path and consume fit profile state"
    );

    let fitted_dense = fitted.to_dense().unwrap();
    let expected_dense = tn_a.contract_naive(&tn_b).unwrap();
    assert!(fitted_dense.distance(&expected_dense).unwrap() < 1e-10);
}

#[test]
fn test_contract_fit_handles_leaf_site_space_that_contracts_away() {
    let left = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);
    let shared_left = DynIndex::new_dyn(2);
    let shared_mid = DynIndex::new_dyn(2);
    let shared_leaf = DynIndex::new_dyn(2);

    let a_ab = DynIndex::new_dyn(2);
    let a_bc = DynIndex::new_dyn(2);
    let b_ab = DynIndex::new_dyn(2);
    let b_bc = DynIndex::new_dyn(2);

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![left.clone(), shared_left.clone(), a_ab.clone()],
                (1..=8).map(|value| value as f64 / 8.0).collect(),
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![a_ab.clone(), shared_mid.clone(), a_bc.clone()],
                (1..=8).map(|value| value as f64 / 10.0).collect(),
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![a_bc.clone(), shared_leaf.clone()],
                vec![0.5, 1.5, -0.5, 2.0],
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![
            TensorDynLen::from_dense(
                vec![b_ab.clone(), shared_left.clone()],
                vec![1.0, -0.5, 0.25, 0.75],
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![
                    b_ab.clone(),
                    shared_mid.clone(),
                    right.clone(),
                    b_bc.clone(),
                ],
                (1..=16).map(|value| (value as f64 - 3.0) / 7.0).collect(),
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![b_bc.clone(), shared_leaf.clone()],
                vec![2.0, -1.0, 0.25, 0.75],
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let fitted = contract_fit(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        FitContractionOptions::new(1),
    )
    .unwrap();

    assert_eq!(fitted.node_count(), 3);
    let fitted_dense = fitted.to_dense().unwrap();
    let expected_dense = tn_a.contract_naive(&tn_b).unwrap();
    assert!(fitted_dense.distance(&expected_dense).unwrap() < 1e-10);
}
