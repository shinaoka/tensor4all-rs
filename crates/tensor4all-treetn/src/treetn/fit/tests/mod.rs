use super::*;
use crate::treetn::localupdate::{LocalUpdateStep, LocalUpdater};
use tensor4all_core::{DynIndex, IdxTensor};

/// Create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> TreeTN<IdxTensor, String> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    make_two_node_treetn_with_sites(&s0, &s1)
}

fn make_two_node_treetn_with_sites(s0: &DynIndex, s1: &DynIndex) -> TreeTN<IdxTensor, String> {
    let bond = DynIndex::new_dyn(3);

    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();

    TreeTN::<IdxTensor, String>::from_tensors(vec![t0, t1], vec!["A".to_string(), "B".to_string()])
        .unwrap()
}

fn make_contractible_two_node_pair() -> (TreeTN<IdxTensor, String>, TreeTN<IdxTensor, String>) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    (
        make_two_node_treetn_with_sites(&s0, &s1),
        make_two_node_treetn_with_sites(&s0, &s1),
    )
}

fn make_contractible_two_node_pair_with_surviving_sites(
) -> (TreeTN<IdxTensor, String>, TreeTN<IdxTensor, String>) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let a0 = DynIndex::new_dyn(2);
    let a1 = DynIndex::new_dyn(2);
    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);

    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0.clone(), a0, bond_a.clone()],
                (1..=8).map(|value| value as f64 / 8.0).collect(),
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![bond_a, s1.clone(), a1],
                (1..=8).map(|value| value as f64 / 10.0).collect(),
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![s0, b0, bond_b.clone()],
                (1..=8).map(|value| (value as f64 - 2.0) / 9.0).collect(),
            )
            .unwrap(),
            IdxTensor::from_dense(
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
    tn_a: &TreeTN<IdxTensor, String>,
    tn_b: &TreeTN<IdxTensor, String>,
    center: &str,
) -> TreeTN<IdxTensor, String> {
    tn_a.contract_zipup_preserving_topology_with(
        tn_b,
        &center.to_string(),
        crate::CanonicalForm::Unitary,
        None,
        None,
    )
    .unwrap()
}

fn make_single_node_treetn() -> TreeTN<IdxTensor, String> {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let t = IdxTensor::from_dense(vec![s0, s1], vec![1.0; 6]).unwrap();
    TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap()
}

fn make_three_node_treetn() -> TreeTN<IdxTensor, String> {
    let s0 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0, bond01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond01, s1, bond12.clone()], vec![1.0; 18]).unwrap();
    let t2 = IdxTensor::from_dense(vec![bond12, s2], vec![1.0; 6]).unwrap();

    TreeTN::<IdxTensor, String>::from_tensors(
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
    let env = FitEnvironment::<IdxTensor, String>::new();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
}

#[test]
fn test_fit_environment_default() {
    let env = FitEnvironment::<IdxTensor, String>::default();
    assert!(env.is_empty());
    assert_eq!(env.len(), 0);
}

#[test]
fn test_fit_environment_insert_and_get() {
    let mut env = FitEnvironment::<IdxTensor, String>::new();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

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
    let env = FitEnvironment::<IdxTensor, String>::new();
    assert!(env.get(&"A".to_string(), &"B".to_string()).is_none());
}

#[test]
fn test_fit_environment_clear() {
    let mut env = FitEnvironment::<IdxTensor, String>::new();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

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
    let mut env = FitEnvironment::<IdxTensor, String>::new();
    let tn = make_two_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

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
    let env = FitEnvironment::<IdxTensor, String>::new();
    let tn = make_two_node_treetn();
    assert!(env.verify_structural_consistency(&tn).is_ok());
}

#[test]
fn test_fit_environment_verify_structural_consistency_valid() {
    let mut env = FitEnvironment::<IdxTensor, String>::new();
    let tn = make_two_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0]).unwrap();

    // A is a leaf with only neighbor B. env[(A, B)] is valid alone.
    env.insert("A".to_string(), "B".to_string(), t.clone());
    assert!(env.verify_structural_consistency(&tn).is_ok());
}

#[test]
fn test_fit_environment_get_or_compute_caches_leaf_environment() {
    let (tn_a, tn_b) = make_contractible_two_node_pair_with_surviving_sites();
    let tn_c = make_fit_initial_c(&tn_a, &tn_b, "A");
    let mut env = FitEnvironment::<IdxTensor, String>::new();

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
    let mut env = FitEnvironment::<IdxTensor, String>::new();
    let tn = make_three_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap();

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
    let t = IdxTensor::from_dense(vec![s], vec![1.0, 2.0]).unwrap();
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
    let a_u = IdxTensor::from_dense(
        vec![s_prime.clone(), bond.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let b_u = IdxTensor::from_dense(
        vec![s.clone(), bond.clone()],
        vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    )
    .unwrap();
    let a_v = IdxTensor::from_dense(
        vec![bond.clone(), t_a.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let b_v = IdxTensor::from_dense(
        vec![bond.clone(), t_b.clone()],
        vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(
        vec![a_u, a_v],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(
        vec![b_u, b_v],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Full network C: node A's site space holds the same-ID primed pair.
    let c_a = IdxTensor::from_dense(
        vec![s_prime.clone(), s.clone(), bond.clone()],
        vec![1.0; 12],
    )
    .unwrap();
    let c_b =
        IdxTensor::from_dense(vec![bond.clone(), t_a.clone(), t_b.clone()], vec![1.0; 12]).unwrap();
    let full_treetn = TreeTN::<IdxTensor, String>::from_tensors(
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

    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![left.clone(), shared_left.clone(), a_ab.clone()],
                (1..=8).map(|value| value as f64 / 8.0).collect(),
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![a_ab.clone(), shared_mid.clone(), a_bc.clone()],
                (1..=8).map(|value| value as f64 / 10.0).collect(),
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![a_bc.clone(), shared_leaf.clone()],
                vec![0.5, 1.5, -0.5, 2.0],
            )
            .unwrap(),
        ],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            IdxTensor::from_dense(
                vec![b_ab.clone(), shared_left.clone()],
                vec![1.0, -0.5, 0.25, 0.75],
            )
            .unwrap(),
            IdxTensor::from_dense(
                vec![
                    b_ab.clone(),
                    shared_mid.clone(),
                    right.clone(),
                    b_bc.clone(),
                ],
                (1..=16).map(|value| (value as f64 - 3.0) / 7.0).collect(),
            )
            .unwrap(),
            IdxTensor::from_dense(
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

// ========================================================================
// Low-rank adaptive initializer tests
// ========================================================================

/// Bond dimensions of every edge of `tn`, sorted ascending.
fn sorted_bond_dims<V>(tn: &TreeTN<IdxTensor, V>) -> Vec<usize>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let mut dims: Vec<usize> = tn
        .site_index_network()
        .edges()
        .map(|(u, v)| {
            tn.edge_between(&u, &v)
                .map(|e| tn.bond_index(e).map(|b| b.dim()).unwrap())
                .unwrap()
        })
        .collect();
    dims.sort_unstable();
    dims
}

/// Build a chain of `n` nodes with a shared site `s_i` in both A and B,
/// a private site `a_i` in A and `b_i` in B, A-bonds of dim `chi_a` and
/// B-bonds of dim `chi_b`. Node names are `0..n` as `usize`.
fn make_chain_pair(
    n: usize,
    chi_a: usize,
    chi_b: usize,
    phys: usize,
    complex: bool,
) -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let mut rng = StdRng::seed_from_u64(0xABBAu64 + n as u64);
    let mut shared: Vec<DynIndex> = Vec::with_capacity(n);
    let mut a_site: Vec<DynIndex> = Vec::with_capacity(n);
    let mut b_site: Vec<DynIndex> = Vec::with_capacity(n);
    for i in 0..n {
        shared.push(DynIndex::new_dyn_with_tag(phys, &format!("s={}", i)).unwrap());
        a_site.push(DynIndex::new_dyn_with_tag(phys, &format!("a={}", i)).unwrap());
        b_site.push(DynIndex::new_dyn_with_tag(phys, &format!("b={}", i)).unwrap());
    }
    let la: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(chi_a))
        .collect();
    let lb: Vec<DynIndex> = (0..n.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(chi_b))
        .collect();
    let (_shared, _a_site, _b_site) = (shared.clone(), a_site.clone(), b_site.clone());

    let mut ta = Vec::with_capacity(n);
    let mut tb = Vec::with_capacity(n);
    for i in 0..n {
        let mut ia = Vec::new();
        let mut ib = Vec::new();
        if i > 0 {
            ia.push(la[i - 1].clone());
            ib.push(lb[i - 1].clone());
        }
        ia.push(shared[i].clone());
        ib.push(shared[i].clone());
        ia.push(a_site[i].clone());
        ib.push(b_site[i].clone());
        if i < n - 1 {
            ia.push(la[i].clone());
            ib.push(lb[i].clone());
        }
        let ta_i = if complex {
            IdxTensor::random::<num_complex::Complex64, _>(&mut rng, ia)
        } else {
            IdxTensor::random::<f64, _>(&mut rng, ia)
        }
        .unwrap();
        let tb_i = if complex {
            IdxTensor::random::<num_complex::Complex64, _>(&mut rng, ib)
        } else {
            IdxTensor::random::<f64, _>(&mut rng, ib)
        }
        .unwrap();
        ta.push(ta_i);
        tb.push(tb_i);
    }
    let names: Vec<usize> = (0..n).collect();
    (
        TreeTN::from_tensors(ta, names.clone()).unwrap(),
        TreeTN::from_tensors(tb, names).unwrap(),
    )
}

#[test]
fn test_low_rank_initializer_stays_small_no_product_bond() {
    // A and B bonds are 6 and 7; an exact product bond would be 42. The
    // low-rank initializer must never form it: every edge stays at bond_dim.
    let (tn_a, tn_b) = make_chain_pair(4, 6, 7, 2, false);
    let center = 3usize;

    let init1 = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 7).unwrap();
    assert_eq!(sorted_bond_dims(&init1), vec![1, 1, 1]);
    assert_eq!(init1.node_count(), 4);
    assert!(init1.same_topology(&tn_a));
    assert!(tn_a.same_topology(&init1));

    let init3 = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 3, 7).unwrap();
    assert_eq!(sorted_bond_dims(&init3), vec![3, 3, 3]);
}

#[test]
fn test_low_rank_initializer_is_deterministic() {
    // Bond index identities are freshly allocated per call, so two runs only
    // agree up to index renaming. The user-visible determinism property is that
    // fit sweeps from two same-seeded initializers produce identical results.
    let (tn_a, tn_b) = make_chain_pair(3, 5, 5, 2, false);
    let center = 2usize;

    let build_init = || {
        let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 1234).unwrap();
        contract_fit_from_initial(
            &tn_a,
            &tn_b,
            &center,
            FitContractionOptions::new(2).with_svd_policy(SvdTruncationPolicy::new(1e-6)),
            init,
        )
        .unwrap()
    };
    let r1 = build_init().to_dense().unwrap();
    let r2 = build_init().to_dense().unwrap();
    assert!(
        r1.distance(&r2).unwrap() < 1e-12,
        "same seed must reproduce the identical fit result"
    );
}

#[test]
fn test_low_rank_initializer_carries_surviving_sites() {
    let n = 3;
    let (tn_a, tn_b) = make_chain_pair(n, 4, 4, 2, false);
    let center = 2usize;
    let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 5).unwrap();

    // C's site space at node i = (A sites at i ∪ B sites at i) minus the
    // sites shared between A and B (those are contracted in A·B).
    for i in 0..n {
        let a_sites = tn_a.site_index_network().site_space(&i).cloned().unwrap();
        let b_sites = tn_b.site_index_network().site_space(&i).cloned().unwrap();
        let shared: HashSet<_> = a_sites.intersection(&b_sites).cloned().collect();
        let expect: HashSet<DynIndex> = a_sites
            .iter()
            .filter(|s| !shared.contains(*s))
            .chain(b_sites.iter().filter(|s| !shared.contains(*s)))
            .cloned()
            .collect();
        let c_sites = init.site_index_network().site_space(&i).cloned().unwrap();
        assert_eq!(c_sites, expect, "node {i} site space mismatch");
    }
}

#[test]
fn test_contract_fit_from_rank1_grows_adaptively() {
    // Two-node pair with surviving sites (from existing helpers). The exact
    // product needs bond rank > 1; a rank-1 C0 with a tolerance and no
    // max_bond_dim must grow and converge.
    let (tn_a, tn_b) = make_contractible_two_node_pair_with_surviving_sites();
    let center = "A".to_string();
    let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 9).unwrap();

    // Zero sweeps returns the initializer as-is (rank stays 1).
    let out0 = contract_fit_from_initial(
        &tn_a,
        &tn_b,
        &center,
        FitContractionOptions::new(0),
        init.clone(),
    )
    .unwrap();
    assert_eq!(sorted_bond_dims(&out0), vec![1]);
    // No option set: ranks are preserved, so the state stays rank-1.
    let out1 = contract_fit_from_initial(
        &tn_a,
        &tn_b,
        &center,
        FitContractionOptions::new(2),
        init.clone(),
    )
    .unwrap();
    assert_eq!(sorted_bond_dims(&out1), vec![1]);

    // With a tolerance and no max_bond_dim the bond must grow beyond rank 1.
    let opts = FitContractionOptions::new(2).with_svd_policy(SvdTruncationPolicy::new(1e-6));
    let out = contract_fit_from_initial(&tn_a, &tn_b, &center, opts, init).unwrap();
    let bd = out
        .edge_between(&"A".to_string(), &"B".to_string())
        .map(|e| out.bond_index(e).map(|b| b.dim()).unwrap())
        .unwrap();
    assert!(bd > 1, "fit must grow the bond from rank 1, got {bd}");

    let fitted_dense = out.to_dense().unwrap();
    let expected = tn_a.contract_naive(&tn_b).unwrap();
    assert!(fitted_dense.distance(&expected).unwrap() < 1e-6);
}

#[test]
fn test_contract_fit_from_initial_rejects_wrong_topology() {
    let (tn_a, tn_b) = make_contractible_two_node_pair_with_surviving_sites();
    // A three-node tree has a different node set than the two-node pair.
    let bad_init = make_three_node_treetn();
    let err = contract_fit_from_initial(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        FitContractionOptions::new(1),
        bad_init,
    )
    .unwrap_err()
    .to_string();
    assert!(err.contains("same topology"), "unexpected error: {err}");
}

#[test]
fn test_low_rank_initializer_requires_bond_dim_at_least_one() {
    let (tn_a, tn_b) = make_contractible_two_node_pair_with_surviving_sites();
    let err = low_rank_initializer_tree_tn(&tn_a, &tn_b, &"A".to_string(), 0, 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("bond_dim >= 1"), "unexpected error: {err}");
}

// ========================================================================
// Complex and branched-tree adaptive fit tests
// ========================================================================

/// Build a star tree: a center node 0 connected to leaves 1..=n_leaves.
/// Each A node and B node carries a shared site (contracted) plus a private
/// site; A and B have their own bonds with the given dimensions.
#[allow(clippy::too_many_arguments)]
fn make_star_pair(
    n_leaves: usize,
    chi_a: usize,
    chi_b: usize,
    phys: usize,
    complex: bool,
) -> (TreeTN<IdxTensor, usize>, TreeTN<IdxTensor, usize>) {
    let mut rng = StdRng::seed_from_u64(0x51a2u64 + n_leaves as u64);
    let n_nodes = 1 + n_leaves;
    let mut shared: Vec<DynIndex> = (0..n_nodes)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("s={}", i)).unwrap())
        .collect();
    let mut a_site: Vec<DynIndex> = (0..n_nodes)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("a={}", i)).unwrap())
        .collect();
    let mut b_site: Vec<DynIndex> = (0..n_nodes)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("b={}", i)).unwrap())
        .collect();
    let mut la: Vec<DynIndex> = (0..n_leaves).map(|_| DynIndex::new_dyn(chi_a)).collect();
    let mut lb: Vec<DynIndex> = (0..n_leaves).map(|_| DynIndex::new_dyn(chi_b)).collect();
    let (_s, _a, _b) = (shared.clone(), a_site.clone(), b_site.clone());
    let (_la, _lb) = (&la, &lb);
    let _ = (&mut shared, &mut a_site, &mut b_site, &mut la, &mut lb);

    let mut ta = Vec::with_capacity(n_nodes);
    let mut tb = Vec::with_capacity(n_nodes);
    // center (node 0): [bonds to each leaf..., shared, a_site] / [...] b_site
    {
        let mut ia: Vec<DynIndex> = (0..n_leaves).map(|k| la[k].clone()).collect();
        ia.push(shared[0].clone());
        ia.push(a_site[0].clone());
        let mut ib: Vec<DynIndex> = (0..n_leaves).map(|k| lb[k].clone()).collect();
        ib.push(shared[0].clone());
        ib.push(b_site[0].clone());
        ta.push(pick_random(&mut rng, ia, complex));
        tb.push(pick_random(&mut rng, ib, complex));
    }
    for leaf in 1..=n_leaves {
        let mut ia: Vec<DynIndex> = vec![
            la[leaf - 1].clone(),
            shared[leaf].clone(),
            a_site[leaf].clone(),
        ];
        ia.sort_by_key(|x| x.dim());
        let mut ib: Vec<DynIndex> = vec![
            lb[leaf - 1].clone(),
            shared[leaf].clone(),
            b_site[leaf].clone(),
        ];
        ib.sort_by_key(|x| x.dim());
        ta.push(pick_random(&mut rng, ia, complex));
        tb.push(pick_random(&mut rng, ib, complex));
    }
    let names: Vec<usize> = (0..n_nodes).collect();
    (
        TreeTN::from_tensors(ta, names.clone()).unwrap(),
        TreeTN::from_tensors(tb, names).unwrap(),
    )
}

fn pick_random(rng: &mut StdRng, indices: Vec<DynIndex>, complex: bool) -> IdxTensor {
    if complex {
        IdxTensor::random::<num_complex::Complex64, _>(rng, indices).unwrap()
    } else {
        IdxTensor::random::<f64, _>(rng, indices).unwrap()
    }
}

#[test]
fn test_low_rank_initializer_and_fit_on_branched_tree() {
    // Star: center 0 with 3 leaves. Exact product of the two random networks is
    // compared against the low-rank adaptive fit.
    let (tn_a, tn_b) = make_star_pair(3, 3, 3, 2, false);
    let center = 0usize;
    let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 21).unwrap();
    assert_eq!(sorted_bond_dims(&init), vec![1, 1, 1]);

    let fit = contract_fit_from_initial(
        &tn_a,
        &tn_b,
        &center,
        FitContractionOptions::new(3).with_svd_policy(SvdTruncationPolicy::new(1e-8)),
        init,
    )
    .unwrap();
    let fitted_dense = fit.to_dense().unwrap();
    let expected = tn_a.contract_naive(&tn_b).unwrap();
    let err = fitted_dense.distance(&expected).unwrap();
    eprintln!("branched fit rel err = {err:.6e}");
    assert!(
        err < 1e-6,
        "branched adaptive fit should match the exact product, rel_err={err:.6e}"
    );
}

#[test]
fn test_adaptive_fit_complex_matches_naive() {
    // Two-node chain with surviving sites, complex dtype.
    let (tn_a, tn_b) = make_chain_pair(2, 3, 3, 2, true);
    let center = 1usize;
    let init = low_rank_initializer_tree_tn(&tn_a, &tn_b, &center, 1, 5).unwrap();
    let fit = contract_fit_from_initial(
        &tn_a,
        &tn_b,
        &center,
        FitContractionOptions::new(4).with_svd_policy(SvdTruncationPolicy::new(1e-8)),
        init,
    )
    .unwrap();
    let fitted_dense = fit.to_dense().unwrap();
    let expected = tn_a.contract_naive(&tn_b).unwrap();
    let err = fitted_dense.distance(&expected).unwrap();
    eprintln!("complex fit rel err = {err:.6e}");
    assert!(
        err < 1e-6,
        "complex adaptive fit should match the exact product, rel_err={err:.6e}"
    );
}
