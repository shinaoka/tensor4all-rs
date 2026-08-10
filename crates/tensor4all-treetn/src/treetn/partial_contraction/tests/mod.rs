use super::*;
use crate::treetn::contraction::{ContractionMethod, ContractionOptions};
use crate::{
    factorize_tensor_to_treetn, RestructureOptions, SelectedIndexContractionError,
    SiteIndexNetwork, TreeTopology,
};
use num_complex::Complex64;
use tensor4all_core::{
    DynIndex, FactorizeAlg, SvdTruncationPolicy, TensorContractionLike, TensorDynLen, TensorIndex,
};

struct PartialContractionInputs {
    tn_a: TreeTN<TensorDynLen, String>,
    tn_b: TreeTN<TensorDynLen, String>,
    s_contract_a: DynIndex,
    s_contract_b: DynIndex,
    s_multiply_a: DynIndex,
    s_multiply_b: DynIndex,
    s_a_only: DynIndex,
    s_b_only: DynIndex,
}

fn make_partial_contraction_inputs() -> PartialContractionInputs {
    let s_contract_a = DynIndex::new_dyn(2);
    let s_multiply_a = DynIndex::new_dyn(2);
    let s_a_only = DynIndex::new_dyn(3);
    let bond_a = DynIndex::new_dyn(2);

    let s_contract_b = DynIndex::new_dyn(2);
    let s_multiply_b = DynIndex::new_dyn(2);
    let s_b_only = DynIndex::new_dyn(4);
    let bond_b = DynIndex::new_dyn(2);

    let t_a0 = TensorDynLen::from_dense(
        vec![s_contract_a.clone(), s_multiply_a.clone(), bond_a.clone()],
        vec![1.0; 8],
    )
    .unwrap();
    let t_a1 =
        TensorDynLen::from_dense(vec![bond_a.clone(), s_a_only.clone()], vec![1.0; 6]).unwrap();

    let t_b0 = TensorDynLen::from_dense(
        vec![s_contract_b.clone(), s_multiply_b.clone(), bond_b.clone()],
        vec![2.0; 8],
    )
    .unwrap();
    let t_b1 =
        TensorDynLen::from_dense(vec![bond_b.clone(), s_b_only.clone()], vec![2.0; 8]).unwrap();

    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_a0, t_a1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        s_a_only,
        s_b_only,
    }
}

#[test]
fn test_partial_contraction_spec_creation() {
    let idx_a = DynIndex::new_dyn(4);
    let idx_b = DynIndex::new_dyn(4);
    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![(idx_a.clone(), idx_b.clone())],
        diagonal_pairs: vec![],
        output_order: None,
    };
    assert_eq!(spec.contract_pairs.len(), 1);
    assert!(spec.diagonal_pairs.is_empty());
}

#[test]
fn helper_topology_and_dense_limit_paths_are_exercised() {
    assert_eq!(canonical_edge(&2usize, &1usize), (1, 2));

    let empty = validate_union_topology::<usize>(&[], &[]).unwrap_err();
    assert!(empty.to_string().contains("at least one node"));

    let unknown = validate_union_topology(&[0usize], &[(0usize, 1usize)]).unwrap_err();
    assert!(
        unknown.to_string().contains("unknown node")
            || unknown.to_string().contains("incompatible topologies")
    );

    let disconnected = validate_union_topology(&[0usize, 1usize], &[]).unwrap_err();
    assert!(disconnected.to_string().contains("incompatible topologies"));

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    assert_eq!(dense_element_count(&[i.clone(), j.clone()]).unwrap(), 6);
    assert_eq!(
        dense_contract_output_indices(&[i.clone(), j.clone()], std::slice::from_ref(&j)),
        vec![i.clone()]
    );
    assert!(
        ensure_dense_reference_limit("lhs", &[i.clone(), j.clone()], 5)
            .unwrap_err()
            .to_string()
            .contains("exceeding limit")
    );
    ensure_dense_reference_limit("lhs", &[i.clone(), j.clone()], 6).unwrap();
}

#[test]
fn helper_factorize_options_and_union_topology_paths_are_exercised() {
    let policy = SvdTruncationPolicy::new(1.0e-8);
    let options = ContractionOptions::default()
        .with_factorize_alg(FactorizeAlg::SVD)
        .with_max_rank(3)
        .with_svd_policy(policy);
    let factorize_options = factorize_options_from_contraction_options(&options).unwrap();
    assert_eq!(factorize_options.alg, FactorizeAlg::SVD);
    assert_eq!(factorize_options.max_rank, Some(3));
    assert_eq!(factorize_options.svd_policy, Some(policy));

    let qr_options = factorize_options_from_contraction_options(
        &ContractionOptions::default()
            .with_factorize_alg(FactorizeAlg::QR)
            .with_qr_rtol(1.0e-7),
    )
    .unwrap();
    assert_eq!(qr_options.alg, FactorizeAlg::QR);
    assert_eq!(qr_options.qr_rtol, Some(1.0e-7));

    let PartialContractionInputs { tn_a, tn_b, .. } = make_partial_contraction_inputs();
    let dense_a = tn_a.contract_to_tensor().unwrap();
    let dense_b = tn_b.contract_to_tensor().unwrap();
    let contracted = dense_a.contract_pair(&dense_b).unwrap();
    let topology = union_result_topology(&tn_a, &tn_b, &contracted).unwrap();
    assert_eq!(topology.nodes.len(), 2);

    validate_mismatched_dense_reference_fallback(
        &tn_a,
        &tn_b,
        &ContractionOptions::default().with_mismatched_topology_dense_limit(4096),
    )
    .unwrap();
    let err =
        validate_mismatched_dense_reference_fallback(&tn_a, &tn_b, &ContractionOptions::default())
            .unwrap_err();
    assert!(err.to_string().contains("explicit dense/reference limit"));
}

#[test]
fn hadamard_multiplies_paired_external_indices() {
    let left_index = DynIndex::new_dyn(2);
    let right_index = DynIndex::new_dyn(2);
    let lhs = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(vec![left_index.clone()], vec![2.0, 3.0]).unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();
    let rhs = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(vec![right_index.clone()], vec![5.0, 7.0]).unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();

    let result = hadamard(
        &lhs,
        &rhs,
        &[(left_index.clone(), right_index)],
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let dense = result.to_dense().unwrap();
    assert_eq!(dense.external_indices(), vec![left_index]);
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![10.0, 21.0]);
}

#[test]
fn weighted_sum_over_index_pairs_contracts_only_selected_axes() {
    let x = DynIndex::new_dyn(2);
    let z = DynIndex::new_dyn(3);
    let wz = DynIndex::new_dyn(3);
    let state = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(
            vec![x.clone(), z.clone()],
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
        )
        .unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();
    let weights = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(vec![wz.clone()], vec![2.0, 3.0, 5.0]).unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();

    let result = weighted_sum_over_index_pairs(
        &state,
        &weights,
        &[(z, wz)],
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let dense = result.to_dense().unwrap();
    assert_eq!(dense.external_indices(), vec![x]);
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![130.0, 140.0]);
}

#[test]
fn sum_over_indices_uses_factorized_ones_weights() {
    let x = DynIndex::new_dyn(2);
    let z = DynIndex::new_dyn(3);
    let state = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(
            vec![x.clone(), z.clone()],
            vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
        )
        .unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();

    let result = sum_over_indices(
        &state,
        std::slice::from_ref(&z),
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let dense = result.to_dense().unwrap();
    assert_eq!(dense.external_indices(), vec![x]);
    assert_eq!(dense.to_vec::<f64>().unwrap(), vec![30.0, 33.0]);
}

#[test]
fn sum_over_indices_reports_typed_selection_errors() {
    let x = DynIndex::new_dyn(2);
    let state = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![TensorDynLen::from_dense(vec![x.clone()], vec![1.0, 2.0]).unwrap()],
        vec!["A".to_string()],
    )
    .unwrap();

    let duplicate = sum_over_indices(
        &state,
        &[x.clone(), x],
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        duplicate,
        SelectedIndexContractionError::DuplicateIndex { .. }
    ));

    let missing = sum_over_indices(
        &state,
        &[DynIndex::new_dyn(2)],
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(
        missing,
        SelectedIndexContractionError::IndexNotFound { .. }
    ));
}

#[test]
fn test_partial_contract_allows_same_node_contract_and_diagonal() {
    // contract_pairs and diagonal_pairs may target the same node.
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        ..
    } = make_partial_contraction_inputs();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a, s_contract_b)],
        diagonal_pairs: vec![(s_multiply_a, s_multiply_b)],
        output_order: None,
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"B".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(128),
    );
    assert!(result.is_ok());
}

#[test]
fn test_partial_contract_rejects_duplicate_pair_usage() {
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        s_multiply_b,
        ..
    } = make_partial_contraction_inputs();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a.clone(), s_contract_b.clone())],
        diagonal_pairs: vec![(s_contract_a, s_multiply_b), (s_multiply_a, s_contract_b)],
        output_order: None,
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"B".to_string(),
        ContractionOptions::default(),
    );

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("appears in multiple pairs"));
}

#[test]
fn test_partial_contract_rejects_dimension_mismatch() {
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_b_only,
        ..
    } = make_partial_contraction_inputs();

    // s_contract_a has dim 2, s_b_only has dim 4 → mismatch
    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a, s_b_only)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("shape mismatch"));
}

#[test]
fn test_partial_contract_rejects_index_not_in_network() {
    let PartialContractionInputs { tn_a, tn_b, .. } = make_partial_contraction_inputs();

    let unknown = DynIndex::new_dyn(2);
    let spec = PartialContractionSpec {
        contract_pairs: vec![(unknown, DynIndex::new_dyn(2))],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_partial_contract_rejects_index_not_in_second_network() {
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        ..
    } = make_partial_contraction_inputs();

    let unknown = DynIndex::new_dyn(2);
    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_contract_a, unknown)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("not found in second TreeTN"));
}

#[test]
fn test_partial_contract_rejects_duplicate_second_index_usage() {
    let PartialContractionInputs {
        tn_a,
        tn_b,
        s_contract_a,
        s_contract_b,
        s_multiply_a,
        ..
    } = make_partial_contraction_inputs();

    let spec = PartialContractionSpec {
        contract_pairs: vec![
            (s_contract_a, s_contract_b.clone()),
            (s_multiply_a, s_contract_b),
        ],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("second TreeTN index"));
}

#[test]
fn test_partial_contract_contract_only() {
    // Contract s_contract pair, no multiply. Sites on different nodes.
    let s_a = DynIndex::new_dyn(3);
    let s_b = DynIndex::new_dyn(3);
    let extra_a = DynIndex::new_dyn(2);
    let extra_b = DynIndex::new_dyn(2);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone(), extra_a.clone()], vec![1.0; 6]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b.clone(), extra_b.clone()], vec![2.0; 6]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_a, s_b)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(128),
    )
    .unwrap();

    // s_a/s_b contracted away; extra_a and extra_b remain
    assert_eq!(result.external_indices().len(), 2);
}

#[test]
fn partial_contract_fit_inserts_dummy_links_for_nodewise_outer_product_spectators() {
    let k_left = DynIndex::new_dyn(2);
    let k_right = DynIndex::new_dyn(2);
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let bond_left = DynIndex::new_dyn(2);
    let bond_right = DynIndex::new_dyn(2);

    let a0 = TensorDynLen::from_dense(
        vec![k_left.clone(), bond_left.clone()],
        vec![1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let a1 = TensorDynLen::from_dense(vec![bond_left.clone(), i.clone()], vec![5.0, 6.0, 7.0, 8.0])
        .unwrap();
    let b0 = TensorDynLen::from_dense(
        vec![k_right.clone(), bond_right.clone()],
        vec![0.5, 1.5, 2.5, 3.5],
    )
    .unwrap();
    let b1 = TensorDynLen::from_dense(
        vec![bond_right.clone(), j.clone()],
        vec![1.0, -1.0, 2.0, -2.0],
    )
    .unwrap();
    let lhs = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![a0, a1],
        vec!["left".to_string(), "right".to_string()],
    )
    .unwrap();
    let rhs = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![b0, b1],
        vec!["left".to_string(), "right".to_string()],
    )
    .unwrap();
    let spec = PartialContractionSpec {
        contract_pairs: vec![(k_left.clone(), k_right.clone())],
        diagonal_pairs: vec![],
        output_order: None,
    };

    let rhs_aligned = rhs.replaceind(&k_right, &k_left).unwrap();
    let reference = tensordot(
        &lhs.contract_to_tensor().unwrap(),
        &rhs_aligned.contract_to_tensor().unwrap(),
        &[(k_left.clone(), k_left)],
    )
    .unwrap();
    let actual = partial_contract(
        &lhs,
        &rhs,
        &spec,
        &"right".to_string(),
        ContractionOptions::fit(),
    )
    .unwrap()
    .to_dense()
    .unwrap();

    assert!(actual.distance(&reference).unwrap() < 1.0e-10);
}

#[test]
fn test_partial_contract_empty_spec() {
    // Empty spec: no contract, no diagonal pair → full outer product
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b.clone()], vec![2.0; 3]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(128),
    )
    .unwrap();

    // Both s_a and s_b remain as external
    assert_eq!(result.external_indices().len(), 2);
}

#[test]
fn test_partial_contract_rejects_bad_output_order_length() {
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b], vec![2.0; 3]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: Some(vec![s_a]),
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(128),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("output_order length"));
}

#[test]
fn test_partial_contract_rejects_unknown_output_order_index() {
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(3);
    let unknown = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let t_b = TensorDynLen::from_dense(vec![s_b], vec![2.0; 3]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: Some(vec![s_a, unknown]),
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::new(ContractionMethod::Naive).with_dense_reference_limit(128),
    );
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("output_order must contain exactly"));
}

#[test]
fn test_partial_contract_allows_same_node_in_second_network() {
    // Same-node combinations in the second network are allowed with diagonal semantics too.
    let s_a1 = DynIndex::new_dyn(2);
    let s_a2 = DynIndex::new_dyn(2);
    let bond_a = DynIndex::new_dyn(2);

    // tn_a: node "A" has s_a1 only, node "B" has s_a2 only
    let t_a0 = TensorDynLen::from_dense(vec![s_a1.clone(), bond_a.clone()], vec![1.0; 4]).unwrap();
    let t_a1 = TensorDynLen::from_dense(vec![bond_a.clone(), s_a2.clone()], vec![1.0; 4]).unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_a0, t_a1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // tn_b: node "A" has s_b_contract AND s_b_multiply on same node
    let s_b_contract = DynIndex::new_dyn(2);
    let s_b_multiply = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s_b_only = DynIndex::new_dyn(3);

    let t_b0 = TensorDynLen::from_dense(
        vec![s_b_contract.clone(), s_b_multiply.clone(), bond_b.clone()],
        vec![2.0; 8],
    )
    .unwrap();
    let t_b1 =
        TensorDynLen::from_dense(vec![bond_b.clone(), s_b_only.clone()], vec![2.0; 6]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // contract s_a1 ↔ s_b_contract (node "A" in both)
    // diagonal s_a2 ↔ s_b_multiply (node "B" in tn_a, node "A" in tn_b)
    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_a1, s_b_contract)],
        diagonal_pairs: vec![(s_a2, s_b_multiply)],
        output_order: None,
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_ok());
}

#[test]
fn test_partial_contract_moves_misaligned_same_topology_contract_pair() {
    let a_row0 = DynIndex::new_dyn(2);
    let a_contract = DynIndex::new_dyn(2);
    let a_row1 = DynIndex::new_dyn(2);
    let a_b01 = DynIndex::new_dyn(1);
    let a_b12 = DynIndex::new_dyn(1);
    let a_b23 = DynIndex::new_dyn(1);
    let a_b34 = DynIndex::new_dyn(1);

    let b_contract = DynIndex::new_dyn(2);
    let b_col0 = DynIndex::new_dyn(2);
    let b_col1 = DynIndex::new_dyn(2);
    let b_b01 = DynIndex::new_dyn(1);
    let b_b12 = DynIndex::new_dyn(1);
    let b_b23 = DynIndex::new_dyn(1);
    let b_b34 = DynIndex::new_dyn(1);

    let tn_a = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![a_row0.clone(), a_b01.clone()], vec![1.0; 2]).unwrap(),
            TensorDynLen::from_dense(
                vec![a_b01.clone(), a_contract.clone(), a_b12.clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![a_b12.clone(), a_row1.clone(), a_b23.clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            TensorDynLen::from_dense(vec![a_b23.clone(), a_b34.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(vec![a_b34.clone()], vec![1.0]).unwrap(),
        ],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap();

    let tn_b = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![b_b01.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(vec![b_b01.clone(), b_b12.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(vec![b_b12.clone(), b_b23.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(
                vec![
                    b_b23.clone(),
                    b_contract.clone(),
                    b_col0.clone(),
                    b_b34.clone(),
                ],
                vec![1.0; 4],
            )
            .unwrap(),
            TensorDynLen::from_dense(vec![b_b34.clone(), b_col1.clone()], vec![1.0; 2]).unwrap(),
        ],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap();

    let output_order = vec![
        a_row0.clone(),
        a_row1.clone(),
        b_col0.clone(),
        b_col1.clone(),
    ];
    let spec = PartialContractionSpec {
        contract_pairs: vec![(a_contract, b_contract)],
        diagonal_pairs: vec![],
        output_order: Some(output_order.clone()),
    };

    let result =
        partial_contract(&tn_a, &tn_b, &spec, &0usize, ContractionOptions::default()).unwrap();
    let dense = result.to_dense().unwrap();
    assert_eq!(dense.external_indices(), output_order);
    for value in dense.to_vec::<f64>().unwrap() {
        assert!((value - 2.0).abs() < 1e-12);
    }
}

#[test]
fn test_partial_contract_allows_compatible_topology_mismatch_with_gap_leaf() {
    // tn_a has 1 node, tn_b has 2 nodes. The union topology is still a tree,
    // so partial_contract should treat the missing node in tn_a as a scalar gap.
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(2);
    let s_b2 = DynIndex::new_dyn(3);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0; 2]).unwrap();
    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();

    let t_b0 = TensorDynLen::from_dense(vec![s_b.clone(), bond_b.clone()], vec![2.0; 4]).unwrap();
    let t_b1 = TensorDynLen::from_dense(vec![bond_b.clone(), s_b2.clone()], vec![2.0; 6]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_ok(), "{result:?}");

    let result = result.unwrap();
    let external = result.external_indices();
    assert_eq!(external.len(), 3);
    assert!(external.contains(&s_a));
    assert!(external.contains(&s_b));
    assert!(external.contains(&s_b2));
}

#[test]
fn test_partial_contract_aligns_long_mismatched_chain_without_dense_limit() {
    fn binary_chain(node_count: usize) -> TreeTN<TensorDynLen, usize> {
        let mut tensors = Vec::with_capacity(node_count);
        let mut names = Vec::with_capacity(node_count);
        let mut left_bond: Option<DynIndex> = None;

        for site in 0..node_count {
            let site_index = DynIndex::new_dyn(2);
            let right_bond = (site + 1 < node_count).then(|| DynIndex::new_dyn(1));
            let mut indices = Vec::new();
            if let Some(bond) = left_bond.take() {
                indices.push(bond);
            }
            indices.push(site_index);
            if let Some(bond) = &right_bond {
                indices.push(bond.clone());
            }

            let element_count = indices.iter().map(IndexLike::dim).product();
            tensors.push(TensorDynLen::from_dense(indices, vec![1.0; element_count]).unwrap());
            names.push(site);
            left_bond = right_bond;
        }

        TreeTN::<TensorDynLen, usize>::from_tensors(tensors, names).unwrap()
    }

    let tn_a = binary_chain(24);
    let tn_b = binary_chain(25);
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: None,
    };

    let result =
        partial_contract(&tn_a, &tn_b, &spec, &0usize, ContractionOptions::default()).unwrap();
    assert_eq!(result.node_count(), 25);
    assert_eq!(result.edge_count(), 24);
    assert_eq!(result.external_indices().len(), 49);
}

#[test]
fn test_partial_contract_rejects_incompatible_topology_union() {
    // tn_a: A-B-C chain
    // tn_b: A-B and A-C star
    // The union contains a cycle, so the topology mismatch is not admissible.
    let sa = DynIndex::new_dyn(2);
    let sb = DynIndex::new_dyn(2);
    let sc = DynIndex::new_dyn(2);
    let ab = DynIndex::new_dyn(2);
    let bc = DynIndex::new_dyn(2);
    let ab2 = DynIndex::new_dyn(2);
    let ac2 = DynIndex::new_dyn(2);

    let ta0 = TensorDynLen::from_dense(vec![sa.clone(), ab.clone()], vec![1.0; 4]).unwrap();
    let ta1 =
        TensorDynLen::from_dense(vec![ab.clone(), sb.clone(), bc.clone()], vec![1.0; 8]).unwrap();
    let ta2 = TensorDynLen::from_dense(vec![bc.clone(), sc.clone()], vec![1.0; 4]).unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![ta0, ta1, ta2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let tb0 =
        TensorDynLen::from_dense(vec![sa.sim(), ab2.clone(), ac2.clone()], vec![2.0; 8]).unwrap();
    let tb1 = TensorDynLen::from_dense(vec![ab2.clone(), sb.sim()], vec![2.0; 4]).unwrap();
    let tb2 = TensorDynLen::from_dense(vec![ac2.clone(), sc.sim()], vec![2.0; 4]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![tb0, tb1, tb2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let spec: PartialContractionSpec<DynIndex> = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: None,
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    );
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("topolog"));
}

#[test]
fn test_partial_contract_mismatched_topology_scalar_result() {
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(2);
    let bond_b = DynIndex::new_dyn(1);

    let t_a = TensorDynLen::from_dense(vec![s_a.clone()], vec![1.0, 2.0]).unwrap();
    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![t_a], vec!["A".to_string()]).unwrap();

    let t_b0 = TensorDynLen::from_dense(vec![s_b.clone(), bond_b.clone()], vec![3.0, 4.0]).unwrap();
    let t_b1 = TensorDynLen::from_dense(vec![bond_b], vec![1.0]).unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(s_a, s_b)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default().with_mismatched_topology_dense_limit(64),
    )
    .unwrap();

    assert!(result.external_indices().is_empty());
    let dense = result.contract_to_tensor().unwrap();
    assert_eq!(dense.as_slice_f64().unwrap(), vec![11.0]);
}

#[test]
fn test_partial_contract_honors_output_order() {
    let a0 = DynIndex::new_dyn(2);
    let a1 = DynIndex::new_dyn(2);
    let a2 = DynIndex::new_dyn(2);
    let bond_a0 = DynIndex::new_dyn(2);
    let bond_a1 = DynIndex::new_dyn(2);

    let t_a0 =
        TensorDynLen::from_dense(vec![a0.clone(), bond_a0.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let t_a1 = TensorDynLen::from_dense(
        vec![bond_a0.clone(), a1.clone(), bond_a1.clone()],
        vec![1.0; 8],
    )
    .unwrap();
    let t_a2 =
        TensorDynLen::from_dense(vec![bond_a1.clone(), a2.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let tn_a = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_a0, t_a1, t_a2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let b0 = DynIndex::new_dyn(2);
    let b1 = DynIndex::new_dyn(2);
    let b2 = DynIndex::new_dyn(2);
    let bond_b0 = DynIndex::new_dyn(2);
    let bond_b1 = DynIndex::new_dyn(2);

    let t_b0 =
        TensorDynLen::from_dense(vec![b0.clone(), bond_b0.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let t_b1 = TensorDynLen::from_dense(
        vec![bond_b0.clone(), b1.clone(), bond_b1.clone()],
        vec![1.0; 8],
    )
    .unwrap();
    let t_b2 =
        TensorDynLen::from_dense(vec![bond_b1.clone(), b2.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let tn_b = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![t_b0, t_b1, t_b2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(a1.clone(), b1.clone())],
        diagonal_pairs: vec![(a0.clone(), b0.clone()), (a2.clone(), b2.clone())],
        output_order: Some(vec![a2.clone(), a0.clone()]),
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"B".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let (indices, _) = result.all_site_indices().unwrap();
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0].id(), a2.id());
    assert_eq!(indices[1].id(), a0.id());
}

#[test]
fn test_partial_contract_output_order_rejects_multiple_survivors_on_one_node() {
    let a_row0 = DynIndex::new_dyn(2);
    let a_contract = DynIndex::new_dyn(2);
    let a_row1 = DynIndex::new_dyn(2);
    let a_b01 = DynIndex::new_dyn(1);
    let a_b12 = DynIndex::new_dyn(1);
    let a_b23 = DynIndex::new_dyn(1);
    let a_b34 = DynIndex::new_dyn(1);

    let b_contract = DynIndex::new_dyn(2);
    let b_col0 = DynIndex::new_dyn(2);
    let b_col1 = DynIndex::new_dyn(2);
    let b_b01 = DynIndex::new_dyn(1);
    let b_b12 = DynIndex::new_dyn(1);
    let b_b23 = DynIndex::new_dyn(1);
    let b_b34 = DynIndex::new_dyn(1);

    let tn_a = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![a_row0.clone(), a_b01.clone()], vec![1.0; 2]).unwrap(),
            TensorDynLen::from_dense(
                vec![a_b01.clone(), a_contract.clone(), a_b12.clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![a_b12.clone(), a_row1.clone(), a_b23.clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            TensorDynLen::from_dense(vec![a_b23.clone(), a_b34.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(vec![a_b34.clone()], vec![1.0]).unwrap(),
        ],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap();
    let tn_b = TreeTN::<TensorDynLen, usize>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![b_b01.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(vec![b_b01.clone(), b_b12.clone()], vec![1.0]).unwrap(),
            TensorDynLen::from_dense(
                vec![b_b12.clone(), b_col1.clone(), b_b23.clone()],
                vec![1.0; 2],
            )
            .unwrap(),
            TensorDynLen::from_dense(
                vec![
                    b_b23.clone(),
                    b_contract.clone(),
                    b_col0.clone(),
                    b_b34.clone(),
                ],
                vec![1.0; 4],
            )
            .unwrap(),
            TensorDynLen::from_dense(vec![b_b34.clone()], vec![1.0]).unwrap(),
        ],
        vec![0, 1, 2, 3, 4],
    )
    .unwrap();

    let output_order = vec![
        a_row0.clone(),
        a_row1.clone(),
        b_col0.clone(),
        b_col1.clone(),
    ];
    let spec = PartialContractionSpec {
        contract_pairs: vec![(a_contract, b_contract)],
        diagonal_pairs: vec![],
        output_order: Some(output_order.clone()),
    };

    let result = partial_contract(&tn_a, &tn_b, &spec, &0usize, ContractionOptions::default());

    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("partial_contract_to_site_network"));
}

#[test]
fn test_partial_contract_to_site_network_splits_onto_explicit_target() {
    let i = DynIndex::new_dyn(2);
    let k_left = DynIndex::new_dyn(2);
    let k_right = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);

    let a = TreeTN::<TensorDynLen, &str>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![i.clone(), k_left.clone()], vec![1.0, 2.0, 3.0, 4.0])
                .unwrap(),
        ],
        vec!["center"],
    )
    .unwrap();
    let b = TreeTN::<TensorDynLen, &str>::from_tensors(
        vec![
            TensorDynLen::from_dense(vec![k_right.clone(), j.clone()], vec![5.0, 6.0, 7.0, 8.0])
                .unwrap(),
        ],
        vec!["center"],
    )
    .unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(k_left, k_right)],
        diagonal_pairs: vec![],
        output_order: None,
    };
    let mut target = SiteIndexNetwork::new();
    target
        .add_node("0_row", std::collections::HashSet::from([i.clone()]))
        .unwrap();
    target
        .add_node("1_col", std::collections::HashSet::from([j.clone()]))
        .unwrap();
    target.add_edge(&"0_row", &"1_col").unwrap();

    let result = partial_contract_to_site_network(
        &a,
        &b,
        &spec,
        &"center",
        &target,
        ContractionOptions::default(),
        &RestructureOptions::default(),
    )
    .unwrap();
    let dense = result.to_dense().unwrap();

    assert_eq!(dense.external_indices(), vec![i.clone(), j.clone()]);
    assert_eq!(
        result.site_index_network().find_node_by_index(&i),
        Some(&"0_row")
    );
    assert_eq!(
        result.site_index_network().find_node_by_index(&j),
        Some(&"1_col")
    );
    for (actual, expected) in dense
        .to_vec::<f64>()
        .unwrap()
        .into_iter()
        .zip([23.0, 34.0, 31.0, 46.0])
    {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn test_partial_contract_to_site_network_rejects_output_order() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);
    let a = TreeTN::<TensorDynLen, &str>::from_tensors(
        vec![TensorDynLen::from_dense(vec![i.clone()], vec![1.0, 2.0]).unwrap()],
        vec!["center"],
    )
    .unwrap();
    let b = TreeTN::<TensorDynLen, &str>::from_tensors(
        vec![TensorDynLen::from_dense(vec![j], vec![3.0, 4.0]).unwrap()],
        vec!["center"],
    )
    .unwrap();
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![],
        output_order: Some(vec![i.clone()]),
    };
    let mut target = SiteIndexNetwork::new();
    target
        .add_node("out", std::collections::HashSet::from([i]))
        .unwrap();

    let result = partial_contract_to_site_network(
        &a,
        &b,
        &spec,
        &"center",
        &target,
        ContractionOptions::default(),
        &RestructureOptions::default(),
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("output_order"));
}

#[test]
fn test_partial_contract_complex_diagonal_pair_keeps_left_leg() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);

    let a = TensorDynLen::from_dense(
        vec![i.clone()],
        vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, -1.0)],
    )
    .unwrap();
    let b = TensorDynLen::from_dense(
        vec![j.clone()],
        vec![Complex64::new(3.0, 0.5), Complex64::new(-1.0, 4.0)],
    )
    .unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![(i.clone(), j.clone())],
        output_order: Some(vec![i.clone()]),
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let dense = result.contract_to_tensor().unwrap();
    let expected = TensorDynLen::from_dense(
        vec![i],
        vec![Complex64::new(2.5, 3.5), Complex64::new(2.0, 9.0)],
    )
    .unwrap();
    assert!(dense.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_partial_contract_diagonal_pair_keeps_left_leg() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(2);

    let a = TensorDynLen::from_dense(vec![i.clone()], vec![1.0_f64, 2.0]).unwrap();
    let b = TensorDynLen::from_dense(vec![j.clone()], vec![10.0_f64, 20.0]).unwrap();

    let tn_a =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![a], vec!["A".to_string()]).unwrap();
    let tn_b =
        TreeTN::<TensorDynLen, String>::from_tensors(vec![b], vec!["A".to_string()]).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs: vec![(i.clone(), j.clone())],
        output_order: Some(vec![i.clone()]),
    };

    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &"A".to_string(),
        ContractionOptions::default(),
    )
    .unwrap();

    let dense = result.contract_to_tensor().unwrap();
    let expected = TensorDynLen::from_dense(vec![i], vec![10.0_f64, 40.0]).unwrap();
    assert!(dense.isapprox(&expected, 1e-12, 0.0).unwrap());
}

#[test]
fn test_partial_contract_matches_dense_reference_for_cross_topology_chain() {
    let x_a = DynIndex::new_dyn(2);
    let w_a = DynIndex::new_dyn(2);
    let i_a = DynIndex::new_dyn(2);
    let x_b = DynIndex::new_dyn(2);
    let w_b = DynIndex::new_dyn(2);
    let z_b = DynIndex::new_dyn(2);
    let i_b = DynIndex::new_dyn(2);
    let j_b = DynIndex::new_dyn(3);

    let dense_a = TensorDynLen::from_dense(vec![x_a.clone(), w_a.clone(), i_a.clone()], {
        let mut values = Vec::new();
        for i in 0..i_a.dim() {
            for w in 0..w_a.dim() {
                for x in 0..x_a.dim() {
                    values.push(1.0 + 2.0 * x as f64 + 3.0 * w as f64 + 5.0 * i as f64);
                }
            }
        }
        values
    })
    .unwrap();
    let dense_b = TensorDynLen::from_dense(
        vec![
            x_b.clone(),
            w_b.clone(),
            z_b.clone(),
            i_b.clone(),
            j_b.clone(),
        ],
        {
            let mut values = Vec::new();
            for j in 0..j_b.dim() {
                for i in 0..i_b.dim() {
                    for z in 0..z_b.dim() {
                        for w in 0..w_b.dim() {
                            for x in 0..x_b.dim() {
                                values.push(
                                    7.0 + 11.0 * x as f64
                                        + 13.0 * w as f64
                                        + 17.0 * z as f64
                                        + 19.0 * i as f64
                                        + 23.0 * j as f64,
                                );
                            }
                        }
                    }
                }
            }
            values
        },
    )
    .unwrap();

    let topology_a = TreeTopology::new(
        [
            (0usize, vec![x_a.clone()]),
            (1usize, vec![w_a.clone()]),
            (2usize, vec![i_a.clone()]),
        ]
        .into_iter()
        .collect(),
        vec![(0usize, 1usize), (1usize, 2usize)],
    );
    let topology_b = TreeTopology::new(
        [
            (0usize, vec![x_b.clone()]),
            (1usize, vec![w_b.clone()]),
            (2usize, vec![z_b.clone()]),
            (3usize, vec![i_b.clone()]),
            (4usize, vec![j_b.clone()]),
        ]
        .into_iter()
        .collect(),
        vec![
            (0usize, 1usize),
            (1usize, 2usize),
            (2usize, 3usize),
            (3usize, 4usize),
        ],
    );

    let tn_a = factorize_tensor_to_treetn(&dense_a, &topology_a, &1usize).unwrap();
    let tn_b = factorize_tensor_to_treetn(&dense_b, &topology_b, &2usize).unwrap();

    let spec = PartialContractionSpec {
        contract_pairs: vec![(w_a.clone(), w_b.clone()), (i_a.clone(), i_b.clone())],
        diagonal_pairs: vec![(x_a.clone(), x_b.clone())],
        output_order: None,
    };
    let result = partial_contract(
        &tn_a,
        &tn_b,
        &spec,
        &1usize,
        ContractionOptions::new(ContractionMethod::Naive)
            .with_dense_reference_limit(128)
            .with_mismatched_topology_dense_limit(128),
    )
    .unwrap();
    let result_dense = result.to_dense().unwrap();

    let expected = TensorDynLen::from_dense(vec![x_a, z_b, j_b], {
        let mut values = Vec::new();
        for j in 0..3 {
            for z in 0..2 {
                for x in 0..2 {
                    let mut sum = 0.0;
                    for i in 0..2 {
                        for w in 0..2 {
                            let a_val = 1.0 + 2.0 * x as f64 + 3.0 * w as f64 + 5.0 * i as f64;
                            let b_val = 7.0
                                + 11.0 * x as f64
                                + 13.0 * w as f64
                                + 17.0 * z as f64
                                + 19.0 * i as f64
                                + 23.0 * j as f64;
                            sum += a_val * b_val;
                        }
                    }
                    values.push(sum);
                }
            }
        }
        values
    })
    .unwrap();

    assert!(
        result_dense.isapprox(&expected, 1.0e-10, 0.0).unwrap(),
        "cross-topology partial_contract mismatch:\nresult={:?}\nexpected={:?}",
        result_dense.as_slice_f64().unwrap(),
        expected.as_slice_f64().unwrap()
    );
}
