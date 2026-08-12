use super::*;
use tensor4all_core::{
    DynIndex, IdxTensor, IndexLike, SvdTruncationPolicy, TensorContractionLike, TensorIndex,
};

/// Helper to create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> (TreeTN<IdxTensor, String>, DynIndex, DynIndex, DynIndex) {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);

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

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn, s0, bond, s1)
}

#[test]
fn test_contraction_method_default() {
    let method = ContractionMethod::default();
    assert_eq!(method, ContractionMethod::Zipup);
}

#[test]
fn test_contraction_options_default() {
    let opts = ContractionOptions::default();
    assert_eq!(opts.method, ContractionMethod::Zipup);
    assert!(opts.max_bond_dim.is_none());
    assert!(opts.svd_policy.is_none());
    assert!(opts.qr_rtol.is_none());
    assert_eq!(opts.nfullsweeps, 1);
    assert!(opts.convergence_tol.is_none());
    assert!(opts.dense_reference_limit.is_none());
    assert!(opts.mismatched_topology_dense_limit.is_none());
}

#[test]
fn test_contraction_options_new() {
    let opts = ContractionOptions::new(ContractionMethod::Fit);
    assert_eq!(opts.method, ContractionMethod::Fit);
}

#[test]
fn test_contraction_options_zipup() {
    let opts = ContractionOptions::zipup();
    assert_eq!(opts.method, ContractionMethod::Zipup);
}

#[test]
fn test_contraction_options_fit() {
    let opts = ContractionOptions::fit();
    assert_eq!(opts.method, ContractionMethod::Fit);
}

#[test]
fn test_contraction_options_builders() {
    let policy = SvdTruncationPolicy::new(1e-8)
        .with_squared_values()
        .with_discarded_tail_sum();
    let opts = ContractionOptions::zipup()
        .with_max_bond_dim(10)
        .with_svd_policy(policy)
        .with_nfullsweeps(3)
        .with_convergence_tol(1e-6)
        .with_factorize_alg(FactorizeAlg::LU)
        .with_dense_reference_limit(128)
        .with_mismatched_topology_dense_limit(64);

    assert_eq!(opts.max_bond_dim, Some(10));
    assert_eq!(opts.svd_policy, Some(policy));
    assert_eq!(opts.qr_rtol, None);
    assert_eq!(opts.nfullsweeps, 3);
    assert_eq!(opts.convergence_tol, Some(1e-6));
    assert_eq!(opts.factorize_alg, FactorizeAlg::LU);
    assert_eq!(opts.dense_reference_limit, Some(128));
    assert_eq!(opts.mismatched_topology_dense_limit, Some(64));
}

#[test]
fn test_contraction_options_qr_builder() {
    let opts = ContractionOptions::fit()
        .with_factorize_alg(FactorizeAlg::QR)
        .with_qr_rtol(1e-7);

    assert_eq!(opts.factorize_alg, FactorizeAlg::QR);
    assert_eq!(opts.qr_rtol, Some(1e-7));
    assert!(opts.svd_policy.is_none());
}

#[test]
fn test_contract_to_tensor_empty_error() {
    let tn = TreeTN::<IdxTensor, String>::new();
    let result = tn.contract_to_tensor();
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("empty"));
}

#[test]
fn test_contract_to_tensor_single_node() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let t = IdxTensor::from_dense(
        vec![s0.clone(), s1.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();
    let tn = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();

    let result = tn.contract_to_tensor().unwrap();
    assert_eq!(result.external_indices().len(), 2);

    // Verify the contracted single-node TN returns the tensor data itself
    let result_data = result.to_vec::<f64>().unwrap();
    let expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    assert_eq!(result_data.len(), expected.len());
    for (i, (&got, &exp)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Element {} mismatch: got {} expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_contract_to_tensor_two_nodes() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();
    let result = tn.contract_to_tensor().unwrap();

    // Result should have the two site indices
    let ext_ids: Vec<_> = result.external_indices().iter().map(|i| *i.id()).collect();
    assert_eq!(ext_ids.len(), 2);
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    // Verify values against the equivalent high-level tensor contraction.
    let t0 = IdxTensor::from_dense(
        vec![s0.clone(), _bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![_bond.clone(), s1.clone()],
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
    .unwrap();
    let expected = t0.contract_pair(&t1).unwrap().to_vec::<f64>().unwrap();

    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), expected.len());
    for (i, (&got, &exp)) in result_data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "Element {} mismatch: got {} expected {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_sim_internal_inds() {
    let (tn, s0, bond, s1) = make_two_node_treetn();
    let sim_tn = tn.sim_internal_inds();

    // Site indices should remain the same
    assert_eq!(sim_tn.node_count(), 2);
    assert_eq!(sim_tn.edge_count(), 1);

    // The bond index should have a different ID
    let edge = sim_tn.graph.graph().edge_indices().next().unwrap();
    let new_bond = sim_tn.bond_index(edge).unwrap();
    assert_ne!(*new_bond.id(), *bond.id());

    // Site indices should still exist (same IDs)
    let site_a = sim_tn.site_space(&"A".to_string()).unwrap();
    let site_a_ids: Vec<_> = site_a.iter().map(|i| *i.id()).collect();
    assert!(site_a_ids.contains(s0.id()));

    let site_b = sim_tn.site_space(&"B".to_string()).unwrap();
    let site_b_ids: Vec<_> = site_b.iter().map(|i| *i.id()).collect();
    assert!(site_b_ids.contains(s1.id()));
}

#[test]
fn test_validate_ortho_consistency_uncanonicalized() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();
    // Not canonicalized, no ortho_towards set
    assert!(tn.validate_ortho_consistency().is_ok());
}

#[test]
fn test_validate_ortho_consistency_empty_region_with_ortho() {
    let (mut tn, _s0, _bond, _s1) = make_two_node_treetn();
    // Set ortho_towards without a canonical_region -> should fail
    let edge = tn.graph.graph().edge_indices().next().unwrap();
    let bond = tn.bond_index(edge).unwrap().clone();
    tn.ortho_towards.insert(bond, "A".to_string());

    let result = tn.validate_ortho_consistency();
    assert!(result.is_err());
}

#[test]
fn test_contract_naive_topology_mismatch() {
    let (tn1, _s0, _bond, _s1) = make_two_node_treetn();

    // Create a single-node TN (different topology)
    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let tn2 = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["X".to_string()]).unwrap();

    let result = tn1.contract_naive(&tn2);
    assert!(result.is_err());
}

#[test]
fn test_contract_zipup_topology_mismatch() {
    let (tn1, _s0, _bond, _s1) = make_two_node_treetn();

    let s = DynIndex::new_dyn(2);
    let t = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let tn2 = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["X".to_string()]).unwrap();

    let result = tn1.contract_zipup(&tn2, &"A".to_string(), None, None);
    assert!(result.is_err());
}

#[test]
fn zipup_parent_bond_filter_keeps_same_id_primed_nonbond_index() {
    let site = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let bond_a_prime = bond_a.prime();
    let indices = vec![
        site.clone(),
        bond_a.clone(),
        bond_a_prime.clone(),
        bond_b.clone(),
    ];

    let kept = indices_except_exact(&indices, &[bond_a, bond_b]);

    assert_eq!(kept, vec![site, bond_a_prime]);
}

#[test]
fn test_contract_naive_requires_dense_reference_limit() {
    let s = DynIndex::new_dyn(3);
    let t_a = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let t_b = IdxTensor::from_dense(vec![s], vec![1.0, 1.0, 1.0]).unwrap();
    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let err = contract(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive),
    )
    .unwrap_err();
    assert!(err.to_string().contains("explicit dense/reference limit"));
}

#[test]
fn test_contract_naive_dense_reference_limit_bounds_materialization() {
    let s = DynIndex::new_dyn(3);
    let t_a = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let t_b = IdxTensor::from_dense(vec![s], vec![1.0, 1.0, 1.0]).unwrap();
    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let err = contract(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(2),
    )
    .unwrap_err();
    assert!(err.to_string().contains("exceeding limit 2"));

    let result = contract(
        &tn_a,
        &tn_b,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(3),
    )
    .unwrap();
    assert_eq!(result.node_count(), 1);
}

#[test]
fn test_find_common_indices() {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(4);

    let t_a = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    let t_b = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 12]).unwrap();

    let common = find_common_indices(&t_a, &t_b);
    assert_eq!(common.len(), 1);
    assert_eq!(*common[0].id(), *bond.id());
}

#[test]
fn test_find_common_indices_no_common() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);

    let t_a = IdxTensor::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    let t_b = IdxTensor::from_dense(vec![s1.clone()], vec![1.0, 2.0, 3.0]).unwrap();

    let common = find_common_indices(&t_a, &t_b);
    assert_eq!(common.len(), 0);
}

/// Regression test for #352: naive contraction fails when result is rank 0 (scalar).
#[test]
fn test_naive_contraction_scalar_result() {
    // Two single-site TreeTNs that share an index → contraction produces a scalar
    let s = DynIndex::new_dyn(3);

    let t_a = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 2.0, 3.0]).unwrap();
    let t_b = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 1.0, 1.0]).unwrap();

    let tn_a = TreeTN::<IdxTensor, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b = TreeTN::<IdxTensor, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    // Naive contraction: inner product = 1+2+3 = 6
    let result =
        contract_naive_to_treetn(&tn_a, &tn_b, &"A".to_string(), None, None, None).unwrap();

    assert_eq!(result.node_count(), 1);
    let dense = result.contract_to_tensor().unwrap();
    let val = dense.sum().unwrap().real();
    assert!(
        (val - 6.0).abs() < 1e-10,
        "scalar contraction expected 6.0, got {}",
        val
    );
}
