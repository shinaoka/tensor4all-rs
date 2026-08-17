use super::*;
use std::collections::{HashMap, HashSet};
use tensor4all_core::{DynIndex, IdxTensor, LinearizationOrder, TensorConstructionLike};

use crate::operator::index_mapping::IndexMapping;
use crate::operator::Operator;
use crate::treetn::TreeTN;
use crate::{RestructureOptions, SiteIndexNetwork};

/// Create a simple "MPO" TreeTN with two site indices per node (input + output).
/// Structure: single node "A" with indices (s_in_tmp, s_out_tmp)
/// Also returns a "state" TreeTN with a single site index s.
fn make_simple_mpo_and_state() -> (
    TreeTN<IdxTensor, String>,
    TreeTN<IdxTensor, String>,
    DynIndex, // s (true site index)
    DynIndex, // s_in_tmp
    DynIndex, // s_out_tmp
) {
    let s = DynIndex::new_dyn(2); // true site index
    let s_in_tmp = DynIndex::new_dyn(2); // MPO input index
    let s_out_tmp = DynIndex::new_dyn(2); // MPO output index

    // MPO tensor: identity operator (2x2 -> 2x2)
    // s_in_tmp x s_out_tmp with identity values
    let mpo_data = vec![1.0, 0.0, 0.0, 1.0]; // identity matrix
    let mpo_tensor =
        IdxTensor::from_dense(vec![s_in_tmp.clone(), s_out_tmp.clone()], mpo_data).unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()]).unwrap();

    // State tensor
    let state_data = vec![1.0, 0.0]; // |0> state
    let state_tensor = IdxTensor::from_dense(vec![s.clone()], state_data).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    (mpo, state, s, s_in_tmp, s_out_tmp)
}

/// Create a simple LinearOperator for testing.
/// Returns (operator, s, s_in_tmp, s_out_tmp)
fn make_linear_operator() -> (
    LinearOperator<IdxTensor, String>,
    DynIndex, // s (true site index)
    DynIndex, // s_in_tmp
    DynIndex, // s_out_tmp
) {
    let (mpo, _state, s, s_in_tmp, s_out_tmp) = make_simple_mpo_and_state();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s.clone(),
            internal_index: s_in_tmp.clone(),
        },
    );

    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s.clone(),
            internal_index: s_out_tmp.clone(),
        },
    );

    let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    (op, s, s_in_tmp, s_out_tmp)
}

fn make_three_node_linear_operator() -> LinearOperator<IdxTensor, usize> {
    let site0 = DynIndex::new_dyn(2);
    let site1 = DynIndex::new_dyn(2);
    let site2 = DynIndex::new_dyn(2);
    let in0 = DynIndex::new_dyn(2);
    let in1 = DynIndex::new_dyn(2);
    let in2 = DynIndex::new_dyn(2);
    let out0 = DynIndex::new_dyn(2);
    let out1 = DynIndex::new_dyn(2);
    let out2 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);

    let t0 = IdxTensor::from_dense(
        vec![in0.clone(), out0.clone(), bond01.clone()],
        vec![1.0_f64; 2 * 2 * 2],
    )
    .unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond01.clone(), in1.clone(), out1.clone(), bond12.clone()],
        vec![1.0_f64; 2 * 2 * 2 * 3],
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(
        vec![bond12.clone(), in2.clone(), out2.clone()],
        vec![1.0_f64; 3 * 2 * 2],
    )
    .unwrap();
    let mpo = TreeTN::from_tensors(vec![t0, t1, t2], vec![0usize, 1, 2]).unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        0,
        IndexMapping {
            true_index: site0.clone(),
            internal_index: in0,
        },
    );
    input_mapping.insert(
        1,
        IndexMapping {
            true_index: site1.clone(),
            internal_index: in1,
        },
    );
    input_mapping.insert(
        2,
        IndexMapping {
            true_index: site2.clone(),
            internal_index: in2,
        },
    );

    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        0,
        IndexMapping {
            true_index: site0,
            internal_index: out0,
        },
    );
    output_mapping.insert(
        1,
        IndexMapping {
            true_index: site1,
            internal_index: out1,
        },
    );
    output_mapping.insert(
        2,
        IndexMapping {
            true_index: site2,
            internal_index: out2,
        },
    );

    LinearOperator::new(mpo, input_mapping, output_mapping)
}

fn make_fused_identity_operator() -> (
    LinearOperator<IdxTensor, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let input_true = DynIndex::new_dyn(4);
    let output_true = DynIndex::new_dyn(4);
    let input_internal = DynIndex::new_dyn(4);
    let output_internal = DynIndex::new_dyn(4);

    let mut data = vec![0.0; 16];
    for value in 0..4 {
        data[value + 4 * value] = 1.0;
    }
    let tensor =
        IdxTensor::from_dense(vec![output_internal.clone(), input_internal.clone()], data).unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: input_true.clone(),
            internal_index: input_internal.clone(),
        },
    );
    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: output_true.clone(),
            internal_index: output_internal.clone(),
        },
    );

    (
        LinearOperator::new(mpo, input_mapping, output_mapping),
        input_true,
        output_true,
        input_internal,
        output_internal,
    )
}

#[test]
fn test_linear_operator_new() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    assert!(op.get_input_mapping(&"A".to_string()).is_some());
    assert!(op.get_output_mapping(&"A".to_string()).is_some());
    assert!(op.get_input_mapping(&"B".to_string()).is_none());
    assert!(op.get_output_mapping(&"B".to_string()).is_none());
}

#[test]
fn test_linear_operator_mpo_accessor() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Test mpo() accessor
    assert_eq!(op.mpo().node_count(), 1);
}

#[test]
fn test_linear_operator_into_mpo_consumes_operator() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let mpo = op.into_mpo();

    assert_eq!(mpo.node_count(), 1);
}

#[test]
fn test_linear_operator_input_output_site_indices() {
    let (op, s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let input_indices = op.input_site_indices();
    assert_eq!(input_indices.len(), 1);
    assert!(input_indices.contains(&s));

    let output_indices = op.output_site_indices();
    assert_eq!(output_indices.len(), 1);
    assert!(output_indices.contains(&s));
}

#[test]
fn rename_nodes_rebuilds_mpo_and_mappings_for_colliding_targets() {
    let op = make_three_node_linear_operator();

    let renamed = op.rename_nodes(&[(0, 0), (1, 2), (2, 4)]).unwrap();

    assert!(renamed.mpo().node_index(&0).is_some());
    assert!(renamed.mpo().node_index(&1).is_none());
    assert!(renamed.mpo().node_index(&2).is_some());
    assert!(renamed.mpo().node_index(&4).is_some());
    assert!(renamed.mpo().edge_between(&0, &2).is_some());
    assert!(renamed.mpo().edge_between(&2, &4).is_some());
    assert!(renamed.get_input_mapping(&0).is_some());
    assert!(renamed.get_input_mapping(&2).is_some());
    assert!(renamed.get_input_mapping(&4).is_some());
    assert!(renamed.get_output_mapping(&0).is_some());
    assert!(renamed.get_output_mapping(&2).is_some());
    assert!(renamed.get_output_mapping(&4).is_some());
}

#[test]
fn rename_nodes_rejects_duplicate_new_names() {
    let op = make_three_node_linear_operator();

    let err = op.rename_nodes(&[(0, 1)]).unwrap_err();

    assert!(err.to_string().contains("duplicate node name"));
}

#[test]
fn test_linear_operator_input_output_mappings() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    assert_eq!(op.input_mappings().len(), 1);
    assert_eq!(op.output_mappings().len(), 1);
}

#[test]
fn test_linear_operator_operator_trait_site_indices() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Operator trait: site_indices returns union of true input and output indices
    let site_indices = Operator::site_indices(&op);
    // Since input and output use the same true index s, should be just 1
    assert_eq!(site_indices.len(), 1);
}

#[test]
fn test_linear_operator_operator_trait_node_names() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let names = Operator::node_names(&op);
    assert_eq!(names.len(), 1);
    assert!(names.contains("A"));
}

#[test]
fn test_linear_operator_apply_local_identity() {
    let (op, s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Create a local tensor to apply operator to: |0> = [1, 0]
    let local_tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();

    let result = op.apply_local(&local_tensor, &["A".to_string()]).unwrap();

    // Identity operator should preserve the state
    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), 2);
    // Values should be approximately [1, 0]
    assert!((result_data[0] - 1.0).abs() < 1e-10);
    assert!((result_data[1]).abs() < 1e-10);
}

#[test]
fn test_linear_operator_operator_trait_site_index_network() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let sin = Operator::site_index_network(&op);
    assert_eq!(sin.node_names().len(), 1);
}

#[test]
fn test_from_mpo_and_state() {
    let (mpo, state, _s, _s_in_tmp, _s_out_tmp) = make_simple_mpo_and_state();
    let op = LinearOperator::from_mpo_and_state(mpo, &state).unwrap();

    // Should have input and output mappings for node "A"
    assert!(op.get_input_mapping(&"A".to_string()).is_some());
    assert!(op.get_output_mapping(&"A".to_string()).is_some());
    assert_eq!(op.input_mappings().len(), 1);
    assert_eq!(op.output_mappings().len(), 1);
}

#[test]
fn from_mpo_and_state_rejects_multi_site_auto_mapping() {
    let state_sites = [DynIndex::new_dyn(2), DynIndex::new_dyn(2)];
    let state_tensor = IdxTensor::from_dense(state_sites.to_vec(), vec![1.0_f64; 4]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    let mpo_sites = [
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
        DynIndex::new_dyn(2),
    ];
    let mpo_tensor = IdxTensor::from_dense(mpo_sites.to_vec(), vec![0.0_f64; 16]).unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()]).unwrap();

    let error = LinearOperator::from_mpo_and_state(mpo, &state).unwrap_err();
    assert!(error.to_string().contains("exactly one state site index"));
}

#[test]
fn test_from_mpo_and_state_mismatched_site_count() {
    // MPO with 3 site indices but state with 1 -> 2*1 != 3
    let s = DynIndex::new_dyn(2);
    let s_in_tmp = DynIndex::new_dyn(2);
    let s_out_tmp = DynIndex::new_dyn(2);
    let s_extra = DynIndex::new_dyn(2);

    let mpo_tensor = IdxTensor::from_dense(
        vec![s_in_tmp.clone(), s_out_tmp.clone(), s_extra.clone()],
        vec![0.0; 8],
    )
    .unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()]).unwrap();

    let state_tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_err());
}

#[test]
fn test_from_mpo_and_state_no_site_indices() {
    // Both MPO and state with no site indices at a node
    // This is the (None, None) branch
    let bond = DynIndex::new_dyn(2);
    let s = DynIndex::new_dyn(2);
    let s_in = DynIndex::new_dyn(2);
    let s_out = DynIndex::new_dyn(2);

    // State: A has site index, B has no site index (only bond)
    let t_a = IdxTensor::from_dense(vec![s.clone(), bond.clone()], vec![1.0; 4]).unwrap();
    let t_b = IdxTensor::from_dense(vec![bond.clone()], vec![1.0; 2]).unwrap();
    let mut state = TreeTN::<IdxTensor, String>::new();
    state.add_tensor("A".to_string(), t_a).unwrap();
    state.add_tensor("B".to_string(), t_b).unwrap();
    let a = state.node_index(&"A".to_string()).unwrap();
    let b = state.node_index(&"B".to_string()).unwrap();
    state.connect(a, &bond, b, &bond).unwrap();

    // MPO: A has two site indices, B has no site index (only bond)
    let bond2 = DynIndex::new_dyn(2);
    let t_a_mpo = IdxTensor::from_dense(
        vec![s_in.clone(), s_out.clone(), bond2.clone()],
        vec![0.0; 8],
    )
    .unwrap();
    let t_b_mpo = IdxTensor::from_dense(vec![bond2.clone()], vec![0.0; 2]).unwrap();
    let mut mpo = TreeTN::<IdxTensor, String>::new();
    mpo.add_tensor("A".to_string(), t_a_mpo).unwrap();
    mpo.add_tensor("B".to_string(), t_b_mpo).unwrap();
    let a2 = mpo.node_index(&"A".to_string()).unwrap();
    let b2 = mpo.node_index(&"B".to_string()).unwrap();
    mpo.connect(a2, &bond2, b2, &bond2).unwrap();

    // Should succeed - the (None, None) branch for node B
    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_ok());
}

#[test]
fn test_from_mpo_and_state_mismatched_presence() {
    // State has site indices at A, but MPO has none at A -> (Some, None) branch
    let s = DynIndex::new_dyn(2);

    let state_tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    // MPO with no site indices
    let mpo_idx = DynIndex::new_dyn(1);
    let mpo_tensor = IdxTensor::from_dense(vec![mpo_idx.clone()], vec![1.0]).unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()]).unwrap();

    // The state has site indices but the mpo's site indices are different,
    // so this should trigger a mismatch error
    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    // This hits the (Some, None) or (None, Some) branch
    assert!(result.is_err());
}

#[test]
fn test_from_mpo_and_state_not_enough_matching_dims() {
    // MPO has 2 site indices but with different dimension from state
    let s = DynIndex::new_dyn(2);
    let s_in_tmp = DynIndex::new_dyn(3); // dim 3 != dim 2
    let s_out_tmp = DynIndex::new_dyn(3);

    let mpo_tensor =
        IdxTensor::from_dense(vec![s_in_tmp.clone(), s_out_tmp.clone()], vec![0.0; 9]).unwrap();
    let mpo =
        TreeTN::<IdxTensor, String>::from_tensors(vec![mpo_tensor], vec!["A".to_string()]).unwrap();

    let state_tensor = IdxTensor::from_dense(vec![s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    let result = LinearOperator::from_mpo_and_state(mpo, &state);
    assert!(result.is_err());
}

/// Create a 2-node MPO and state for multi-node apply_local test.
fn make_two_node_mpo_and_operator() -> (
    LinearOperator<IdxTensor, String>,
    DynIndex, // s0 (true site index for A)
    DynIndex, // s1 (true site index for B)
) {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s0_in = DynIndex::new_dyn(2);
    let s0_out = DynIndex::new_dyn(2);
    let s1_in = DynIndex::new_dyn(2);
    let s1_out = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);

    // Identity MPO: two nodes with bond dim 1
    // A tensor: s0_in x s0_out x bond
    let t_a = IdxTensor::from_dense(
        vec![s0_in.clone(), s0_out.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0], // identity for each bond=0 slice
    )
    .unwrap();
    // B tensor: bond x s1_in x s1_out
    let t_b = IdxTensor::from_dense(
        vec![bond.clone(), s1_in.clone(), s1_out.clone()],
        vec![1.0, 0.0, 0.0, 1.0],
    )
    .unwrap();

    let mut mpo = TreeTN::<IdxTensor, String>::new();
    mpo.add_tensor("A".to_string(), t_a).unwrap();
    mpo.add_tensor("B".to_string(), t_b).unwrap();
    let a = mpo.node_index(&"A".to_string()).unwrap();
    let b = mpo.node_index(&"B".to_string()).unwrap();
    mpo.connect(a, &bond, b, &bond).unwrap();

    let mut input_mapping = HashMap::new();
    input_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_in.clone(),
        },
    );
    input_mapping.insert(
        "B".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_in.clone(),
        },
    );

    let mut output_mapping = HashMap::new();
    output_mapping.insert(
        "A".to_string(),
        IndexMapping {
            true_index: s0.clone(),
            internal_index: s0_out.clone(),
        },
    );
    output_mapping.insert(
        "B".to_string(),
        IndexMapping {
            true_index: s1.clone(),
            internal_index: s1_out.clone(),
        },
    );

    let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    (op, s0, s1)
}

#[test]
fn test_apply_local_multi_node_region() {
    let (op, s0, s1) = make_two_node_mpo_and_operator();

    // Create a local tensor spanning both nodes: |00> = [1, 0, 0, 0]
    let local_tensor =
        IdxTensor::from_dense(vec![s0.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 0.0]).unwrap();

    let result = op
        .apply_local(&local_tensor, &["A".to_string(), "B".to_string()])
        .unwrap();

    // Identity operator should preserve the state
    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), 4);
    assert!((result_data[0] - 1.0).abs() < 1e-10);
    for &v in &result_data[1..] {
        assert!(v.abs() < 1e-10);
    }
}

#[test]
fn test_align_to_state_single_node() {
    // Create an operator with one set of true indices
    let (mut op, original_s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Create a new state with a *different* site index (same dimension)
    let new_s = DynIndex::new_dyn(2);
    assert_ne!(original_s, new_s);

    let state_tensor = IdxTensor::from_dense(vec![new_s.clone()], vec![0.5, 0.5]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    // Align operator to the new state
    op.align_to_state(&state).unwrap();

    // Verify: true indices should now match the state's site index
    let input_mapping = op.get_input_mapping(&"A".to_string()).unwrap();
    assert_eq!(input_mapping.true_index, new_s);

    let output_mapping = op.get_output_mapping(&"A".to_string()).unwrap();
    assert_eq!(output_mapping.true_index, new_s);

    // Verify: operator should still work correctly after alignment
    let local_tensor = IdxTensor::from_dense(vec![new_s.clone()], vec![1.0, 0.0]).unwrap();
    let result = op.apply_local(&local_tensor, &["A".to_string()]).unwrap();
    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), 2);
    assert!((result_data[0] - 1.0).abs() < 1e-10);
    assert!(result_data[1].abs() < 1e-10);
}

#[test]
fn test_align_to_state_two_nodes() {
    // Create a two-node operator
    let (mut op, original_s0, original_s1) = make_two_node_mpo_and_operator();

    // Create a new state with different site indices (same dimensions)
    let new_s0 = DynIndex::new_dyn(2);
    let new_s1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(1);
    assert_ne!(original_s0, new_s0);
    assert_ne!(original_s1, new_s1);

    let t_a = IdxTensor::from_dense(vec![new_s0.clone(), bond.clone()], vec![1.0; 2]).unwrap();
    let t_b = IdxTensor::from_dense(vec![bond.clone(), new_s1.clone()], vec![1.0; 2]).unwrap();
    let mut state = TreeTN::<IdxTensor, String>::new();
    state.add_tensor("A".to_string(), t_a).unwrap();
    state.add_tensor("B".to_string(), t_b).unwrap();
    let a = state.node_index(&"A".to_string()).unwrap();
    let b = state.node_index(&"B".to_string()).unwrap();
    state.connect(a, &bond, b, &bond).unwrap();

    // Align operator to the new state
    op.align_to_state(&state).unwrap();

    // Verify: true indices should match the state's site indices
    let input_a = op.get_input_mapping(&"A".to_string()).unwrap();
    assert_eq!(input_a.true_index, new_s0);

    let input_b = op.get_input_mapping(&"B".to_string()).unwrap();
    assert_eq!(input_b.true_index, new_s1);

    let output_a = op.get_output_mapping(&"A".to_string()).unwrap();
    assert_eq!(output_a.true_index, new_s0);

    let output_b = op.get_output_mapping(&"B".to_string()).unwrap();
    assert_eq!(output_b.true_index, new_s1);

    // Verify: operator still works after alignment
    let local_tensor = IdxTensor::from_dense(
        vec![new_s0.clone(), new_s1.clone()],
        vec![1.0, 0.0, 0.0, 0.0],
    )
    .unwrap();
    let result = op
        .apply_local(&local_tensor, &["A".to_string(), "B".to_string()])
        .unwrap();
    let result_data = result.to_vec::<f64>().unwrap();
    assert_eq!(result_data.len(), 4);
    assert!((result_data[0] - 1.0).abs() < 1e-10);
    for &v in &result_data[1..] {
        assert!(v.abs() < 1e-10);
    }
}

#[test]
fn test_align_to_state_dimension_mismatch() {
    let (mut op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Create a state with a different dimension
    let new_s = DynIndex::new_dyn(3); // dim 3 != dim 2
    let state_tensor = IdxTensor::from_dense(vec![new_s.clone()], vec![1.0, 0.0, 0.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["A".to_string()])
            .unwrap();

    // Should fail due to shape mismatch
    let result = op.align_to_state(&state);
    assert!(result.is_err());
}

#[test]
fn test_align_to_state_missing_node() {
    let (mut op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Create a state that doesn't have node "A"
    let new_s = DynIndex::new_dyn(2);
    let state_tensor = IdxTensor::from_dense(vec![new_s.clone()], vec![1.0, 0.0]).unwrap();
    let state =
        TreeTN::<IdxTensor, String>::from_tensors(vec![state_tensor], vec!["B".to_string()])
            .unwrap();

    // Should fail because node "A" is not in the state
    let result = op.align_to_state(&state);
    assert!(result.is_err());
}

#[test]
fn test_linear_operator_transpose_swaps_mappings() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    // Snapshot the expected swapped mappings before consuming `op`.
    let original_input = op.input_mapping.clone();
    let original_output = op.output_mapping.clone();

    let transposed = op.transpose();

    // After transpose, the new input_mapping is the original output_mapping,
    // and vice versa.
    assert_eq!(transposed.input_mapping.len(), original_output.len());
    assert_eq!(transposed.output_mapping.len(), original_input.len());
    let tin = transposed.get_input_mapping(&"A".to_string()).unwrap();
    let tout = transposed.get_output_mapping(&"A".to_string()).unwrap();
    let expected_in = original_output
        .get("A")
        .and_then(|mappings| mappings.first())
        .unwrap();
    let expected_out = original_input
        .get("A")
        .and_then(|mappings| mappings.first())
        .unwrap();
    assert_eq!(tin.true_index, expected_in.true_index);
    assert_eq!(tin.internal_index, expected_in.internal_index);
    assert_eq!(tout.true_index, expected_out.true_index);
    assert_eq!(tout.internal_index, expected_out.internal_index);
}

#[test]
fn test_linear_operator_transpose_is_involutive() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();

    let original_input = op.input_mapping.clone();
    let original_output = op.output_mapping.clone();

    let round_trip = op.transpose().transpose();

    // Double transpose == identity on mappings.
    assert_eq!(round_trip.input_mapping.len(), original_input.len());
    assert_eq!(round_trip.output_mapping.len(), original_output.len());
    let rin = round_trip.get_input_mapping(&"A".to_string()).unwrap();
    let rout = round_trip.get_output_mapping(&"A".to_string()).unwrap();
    let ein = original_input
        .get("A")
        .and_then(|mappings| mappings.first())
        .unwrap();
    let eout = original_output
        .get("A")
        .and_then(|mappings| mappings.first())
        .unwrap();
    assert_eq!(rin.true_index, ein.true_index);
    assert_eq!(rin.internal_index, ein.internal_index);
    assert_eq!(rout.true_index, eout.true_index);
    assert_eq!(rout.internal_index, eout.internal_index);
}

#[test]
fn test_linear_operator_transpose_preserves_mpo() {
    let (op, _s, _s_in_tmp, _s_out_tmp) = make_linear_operator();
    let original_node_count = op.mpo().node_count();

    let transposed = op.transpose();

    assert_eq!(transposed.mpo().node_count(), original_node_count);
}

#[test]
fn test_unfuse_input_and_output_indices_splits_internal_mpo_axes() {
    let (op, input_true, output_true, _input_internal, _output_internal) =
        make_fused_identity_operator();
    let input0 = DynIndex::new_dyn(2);
    let input1 = DynIndex::new_dyn(2);
    let output0 = DynIndex::new_dyn(2);
    let output1 = DynIndex::new_dyn(2);

    let op = op
        .unfuse_output_index(
            &output_true,
            &[output0.clone(), output1.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap()
        .unfuse_input_index(
            &input_true,
            &[input0.clone(), input1.clone()],
            LinearizationOrder::ColumnMajor,
        )
        .unwrap();

    let input_mappings = op.get_input_mappings(&"A".to_string()).unwrap();
    assert_eq!(input_mappings.len(), 2);
    assert_eq!(input_mappings[0].true_index, input0);
    assert_eq!(input_mappings[1].true_index, input1);

    let output_mappings = op.get_output_mappings(&"A".to_string()).unwrap();
    assert_eq!(output_mappings.len(), 2);
    assert_eq!(output_mappings[0].true_index, output0);
    assert_eq!(output_mappings[1].true_index, output1);

    let output_internal = output_mappings
        .iter()
        .map(|mapping| mapping.internal_index.clone())
        .collect::<Vec<_>>();
    let input_internal = input_mappings
        .iter()
        .map(|mapping| mapping.internal_index.clone())
        .collect::<Vec<_>>();
    let expected =
        <IdxTensor as TensorConstructionLike>::delta(&output_internal, &input_internal).unwrap();
    let actual = op.mpo().contract_to_tensor().unwrap();

    assert!(actual.distance(&expected).unwrap() < 1.0e-12);
}

#[test]
fn test_linear_operator_restructure_to_moves_mapping_nodes() {
    let (op, s0, s1) = make_two_node_mpo_and_operator();
    let node_a = "A".to_string();
    let node_b = "B".to_string();
    let a_in = op
        .get_input_mapping(&node_a)
        .unwrap()
        .internal_index
        .clone();
    let a_out = op
        .get_output_mapping(&node_a)
        .unwrap()
        .internal_index
        .clone();
    let b_in = op
        .get_input_mapping(&node_b)
        .unwrap()
        .internal_index
        .clone();
    let b_out = op
        .get_output_mapping(&node_b)
        .unwrap()
        .internal_index
        .clone();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("left".to_string(), HashSet::from([b_in, b_out]))
        .unwrap();
    target
        .add_node("right".to_string(), HashSet::from([a_in, a_out]))
        .unwrap();
    target
        .add_edge(&"left".to_string(), &"right".to_string())
        .unwrap();

    let expected = op.mpo().contract_to_tensor().unwrap();
    let moved = op
        .restructure_to(&target, &RestructureOptions::default())
        .unwrap();
    let actual = moved.mpo().contract_to_tensor().unwrap();

    assert!(actual.distance(&expected).unwrap() < 1.0e-12);
    assert_eq!(
        moved
            .get_input_mapping(&"left".to_string())
            .unwrap()
            .true_index,
        s1
    );
    assert_eq!(
        moved
            .get_input_mapping(&"right".to_string())
            .unwrap()
            .true_index,
        s0
    );
}
