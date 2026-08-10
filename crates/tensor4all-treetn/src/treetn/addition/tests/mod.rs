use std::collections::HashSet;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};

use crate::treetn::TreeTN;

/// Create two TreeTNs with the same topology for testing addition.
/// Both are 2-node chains: A -- bond -- B
fn make_two_matching_treetns() -> (
    TreeTN<TensorDynLen, String>,
    TreeTN<TensorDynLen, String>,
    DynIndex, // s0
    DynIndex, // s1
) {
    let s0 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

    let t0_a = TensorDynLen::from_dense(
        vec![s0.clone(), bond_a.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1_a = TensorDynLen::from_dense(
        vec![bond_a.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_a, t1_a],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let bond_b = DynIndex::new_dyn(2);

    let t0_b = TensorDynLen::from_dense(vec![s0.clone(), bond_b.clone()], vec![0.0, 1.0, 1.0, 0.0])
        .unwrap();
    let t1_b = TensorDynLen::from_dense(vec![bond_b.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 1.0])
        .unwrap();

    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_b, t1_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn_a, tn_b, s0, s1)
}

fn make_two_matching_treetns_different_site_ids(
) -> (TreeTN<TensorDynLen, String>, TreeTN<TensorDynLen, String>) {
    let s0_a = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(3);
    let s1_a = DynIndex::new_dyn(2);

    let t0_a = TensorDynLen::from_dense(
        vec![s0_a.clone(), bond_a.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1_a = TensorDynLen::from_dense(
        vec![bond_a.clone(), s1_a.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_a, t1_a],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let s0_b = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s1_b = DynIndex::new_dyn(2);

    let t0_b =
        TensorDynLen::from_dense(vec![s0_b.clone(), bond_b.clone()], vec![0.0, 1.0, 1.0, 0.0])
            .unwrap();
    let t1_b =
        TensorDynLen::from_dense(vec![bond_b.clone(), s1_b.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0_b, t1_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn_a, tn_b)
}

fn make_same_id_prime_pair_mpo_like_indices() -> (Vec<DynIndex>, Vec<DynIndex>, Vec<DynIndex>) {
    let inputs: Vec<_> = (0..3).map(|_| DynIndex::new_dyn(3)).collect();
    let outputs: Vec<_> = inputs.iter().map(DynIndex::prime).collect();
    let links = vec![
        DynIndex::new_link(2).unwrap(),
        DynIndex::new_link(2).unwrap(),
    ];
    (inputs, outputs, links)
}

fn scaled_sequence(scale: f64, len: usize) -> Vec<f64> {
    (1..=len).map(|n| scale * n as f64).collect()
}

fn make_same_id_prime_pair_mpo_like_treetn(
    scale: f64,
    inputs: &[DynIndex],
    outputs: &[DynIndex],
    links: &[DynIndex],
) -> TreeTN<TensorDynLen, usize> {
    let tensors = vec![
        TensorDynLen::from_dense(
            vec![inputs[0].clone(), outputs[0].clone(), links[0].clone()],
            scaled_sequence(scale, 18),
        )
        .unwrap(),
        TensorDynLen::from_dense(
            vec![
                links[0].clone(),
                inputs[1].clone(),
                outputs[1].clone(),
                links[1].clone(),
            ],
            scaled_sequence(scale, 36),
        )
        .unwrap(),
        TensorDynLen::from_dense(
            vec![links[1].clone(), inputs[2].clone(), outputs[2].clone()],
            scaled_sequence(scale, 18),
        )
        .unwrap(),
    ];
    TreeTN::<TensorDynLen, usize>::from_tensors(tensors, vec![0, 1, 2]).unwrap()
}

#[test]
fn test_compute_merged_bond_indices() {
    let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

    let merged = tn_a.compute_merged_bond_indices(&tn_b).unwrap();
    assert_eq!(merged.len(), 1);

    // The key should be ("A", "B") in canonical order
    let key = ("A".to_string(), "B".to_string());
    let info = merged.get(&key).unwrap();
    assert_eq!(info.dim_a, 3); // tn_a bond dim
    assert_eq!(info.dim_b, 2); // tn_b bond dim
}

#[test]
fn test_add_basic() {
    let (tn_a, tn_b, s0, s1) = make_two_matching_treetns();

    let result = tn_a.add(&tn_b).unwrap();

    // Result should have same number of nodes
    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);

    // Result should have the same site indices
    let ext_ids: Vec<_> = result.external_indices().iter().map(|i| *i.id()).collect();
    assert_eq!(ext_ids.len(), 2);
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    // Bond dimension should be sum of original dimensions
    let edge = result.graph.graph().edge_indices().next().unwrap();
    let merged_bond = result.bond_index(edge).unwrap();
    assert_eq!(merged_bond.dim(), 3 + 2); // dim_a + dim_b
}

#[test]
fn test_add_verifies_with_contraction() {
    let (tn_a, tn_b, _s0, _s1) = make_two_matching_treetns();

    let result = tn_a.add(&tn_b).unwrap();

    // Contract all three and verify: contract(result) ≈ contract(tn_a) + contract(tn_b)
    let tensor_a = tn_a.contract_to_tensor().unwrap();
    let tensor_b = tn_b.contract_to_tensor().unwrap();
    let tensor_sum = result.contract_to_tensor().unwrap();

    // expected = tensor_a + tensor_b
    let expected = tensor_a
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tensor_b,
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(
        tensor_sum.isapprox(&expected, 1e-10, 0.0).unwrap(),
        "contract(result) != contract(tn_a) + contract(tn_b): maxabs diff = {}",
        tensor_sum.distance(&expected).unwrap()
    );
}

#[test]
fn test_add_topology_mismatch() {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 6]).unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Single-node TN (different topology)
    let t_single = TensorDynLen::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(vec![t_single], vec!["X".to_string()])
        .unwrap();

    let result = tn_a.add(&tn_b);
    assert!(result.is_err());
}

/// Regression test for #353: add() failed for single-site TTs (no bond indices).
#[test]
fn test_add_single_site() {
    let s = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s.clone()], vec![10.0, 20.0, 30.0]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let result = tn_a.add(&tn_b).unwrap();
    assert_eq!(result.node_count(), 1);

    let dense = result.contract_to_tensor().unwrap();
    let expected = TensorDynLen::from_dense(vec![s.clone()], vec![11.0, 22.0, 33.0]).unwrap();
    assert!(
        dense.isapprox(&expected, 1e-10, 0.0).unwrap(),
        "single-site add failed: maxabs diff = {}",
        dense.distance(&expected).unwrap()
    );
}

#[test]
fn test_reindex_site_space_like_matches_template_ids() {
    let (tn_a, tn_b) = make_two_matching_treetns_different_site_ids();

    let reindexed = tn_b.reindex_site_space_like(&tn_a).unwrap();
    assert!(reindexed.share_equivalent_site_index_network(&tn_a));
}

#[test]
fn test_sorted_site_space_orders_deterministically() {
    let a = DynIndex::new_dyn(3);
    let b = DynIndex::new_dyn(2);
    let c = b.prime();
    let mut site_space = HashSet::new();
    site_space.insert(c.clone());
    site_space.insert(a.clone());
    site_space.insert(b.clone());

    let sorted = TreeTN::<TensorDynLen, usize>::sorted_site_space(&site_space);
    assert_eq!(sorted, vec![b, c, a]);
}

#[test]
fn test_reindex_site_space_like_rejects_incompatible_inputs() {
    let (tn_a, _) = make_two_matching_treetns_different_site_ids();

    let lone_site = DynIndex::new_dyn(2);
    let lone_tensor = TensorDynLen::from_dense(vec![lone_site], vec![1.0, 2.0]).unwrap();
    let single =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![lone_tensor], vec!["A".to_string()])
            .unwrap();
    let err = tn_a.reindex_site_space_like(&single).unwrap_err();
    assert!(err.to_string().contains("incompatible topologies"));

    let self_s0 = DynIndex::new_dyn(2);
    let self_s1 = DynIndex::new_dyn(2);
    let self_bond = DynIndex::new_dyn(1);
    let self_t0 =
        TensorDynLen::from_dense(vec![self_s0, self_bond.clone()], vec![1.0, 2.0]).unwrap();
    let self_t1 = TensorDynLen::from_dense(vec![self_bond, self_s1], vec![1.0, 2.0]).unwrap();
    let self_two_node = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![self_t0, self_t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let tmpl_s0 = DynIndex::new_dyn(2);
    let tmpl_extra = DynIndex::new_dyn(2);
    let tmpl_s1 = DynIndex::new_dyn(2);
    let tmpl_bond = DynIndex::new_dyn(1);
    let tmpl_t0 = TensorDynLen::from_dense(
        vec![tmpl_s0, tmpl_extra, tmpl_bond.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let tmpl_t1 = TensorDynLen::from_dense(vec![tmpl_bond, tmpl_s1], vec![1.0, 2.0]).unwrap();
    let template_extra_site = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tmpl_t0, tmpl_t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let err = self_two_node
        .reindex_site_space_like(&template_extra_site)
        .unwrap_err();
    assert!(err
        .to_string()
        .contains("has 1 site indices in self but 2 in template"));

    let tmpl_s0 = DynIndex::new_dyn(3);
    let tmpl_s1 = DynIndex::new_dyn(2);
    let tmpl_bond = DynIndex::new_dyn(1);
    let tmpl_t0 =
        TensorDynLen::from_dense(vec![tmpl_s0, tmpl_bond.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let tmpl_t1 = TensorDynLen::from_dense(vec![tmpl_bond, tmpl_s1], vec![1.0, 2.0]).unwrap();
    let template_dim_mismatch = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tmpl_t0, tmpl_t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let err = self_two_node
        .reindex_site_space_like(&template_dim_mismatch)
        .unwrap_err();
    assert!(err.to_string().contains("site dimension mismatch"));
}

#[test]
fn test_add_reindexed_like_self_accepts_equivalent_site_space_with_different_ids() {
    let (tn_a, tn_b) = make_two_matching_treetns_different_site_ids();
    let tn_b_aligned = tn_b.reindex_site_space_like(&tn_a).unwrap();

    assert!(!tn_a.share_equivalent_site_index_network(&tn_b));

    let sum = tn_a.add_reindexed_like_self(&tn_b).unwrap();
    assert!(sum.share_equivalent_site_index_network(&tn_a));

    let dense_sum = sum.contract_to_tensor().unwrap();
    let dense_expected = tn_a
        .contract_to_tensor()
        .unwrap()
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tn_b_aligned.contract_to_tensor().unwrap(),
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(
        dense_sum.isapprox(&dense_expected, 1e-10, 0.0).unwrap(),
        "add_reindexed_like_self failed: maxabs diff = {}",
        dense_sum.distance(&dense_expected).unwrap()
    );
}

#[test]
fn test_axpby_is_strict_about_site_index_ids() {
    let (tn_a, tn_b) = make_two_matching_treetns_different_site_ids();
    let err = tn_a
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tn_b,
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("site") || err.to_string().contains("index"),
        "unexpected error: {err}"
    );
}

#[test]
fn test_add_preserves_same_id_prime_pair_site_indices() {
    let (inputs, outputs, links) = make_same_id_prime_pair_mpo_like_indices();
    let tn_a = make_same_id_prime_pair_mpo_like_treetn(1.0, &inputs, &outputs, &links);
    let tn_b = make_same_id_prime_pair_mpo_like_treetn(-0.5, &inputs, &outputs, &links);

    for site in 0..3 {
        let site_space = tn_a.site_space(&site).unwrap();
        assert_eq!(site_space.len(), 2);
        assert!(site_space.contains(&inputs[site]));
        assert!(site_space.contains(&outputs[site]));
    }

    let result = tn_a.add(&tn_b).unwrap();

    assert_eq!(result.node_count(), 3);
    assert_eq!(result.edge_count(), 2);
    for site in 0..3 {
        let site_space = result.site_space(&site).unwrap();
        assert_eq!(site_space.len(), 2);
        assert!(site_space.contains(&inputs[site]));
        assert!(site_space.contains(&outputs[site]));
    }

    let dense_result = result.contract_to_tensor().unwrap();
    let dense_expected = tn_a
        .contract_to_tensor()
        .unwrap()
        .axpby(
            tensor4all_core::AnyScalar::new_real(1.0),
            &tn_b.contract_to_tensor().unwrap(),
            tensor4all_core::AnyScalar::new_real(1.0),
        )
        .unwrap();
    assert!(
        dense_result.isapprox(&dense_expected, 1e-10, 0.0).unwrap(),
        "same-id prime-pair MPO-like add failed: maxabs diff = {}",
        dense_result.distance(&dense_expected).unwrap()
    );
}
