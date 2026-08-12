use std::collections::{HashMap, HashSet};

use tensor4all_core::index::DynId;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorIndex};

use crate::operator::IndexMapping;
use crate::treetn::TreeTN;

use super::*;

fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize) -> DynIndex {
    loop {
        let idx = DynIndex::new_dyn(dim);
        if used.insert(*idx.id()) {
            return idx;
        }
    }
}

fn tensor_indices(state: &TreeTN<IdxTensor, String>, node: &str) -> Vec<DynIndex> {
    state
        .tensor(state.node_index(&node.to_string()).unwrap())
        .unwrap()
        .external_indices()
}

#[test]
fn test_local_linop_new() {
    use crate::linsolve::common::ProjectedOperator;

    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    state.add_tensor("site0".to_string(), t0).unwrap();

    let reference_state = state.clone();
    let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

    let expected = tensor_indices(&state, "site0");
    let linop = LocalLinOp::with_expected_input_indices(
        projected_op,
        vec!["site0".to_string()],
        &state,
        &reference_state,
        expected,
    );

    assert_eq!(linop.region.len(), 1);
}

/// Projected apply requires output to stay in the input local vector space.
#[test]
fn test_local_linop_apply_projected_rejects_mismatch() {
    use crate::linsolve::common::ProjectedOperator;

    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    state.add_tensor("site0".to_string(), t0).unwrap();

    let reference_state = state.clone();
    let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

    let expected = tensor_indices(&state, "site0");
    let linop = LocalLinOp::with_expected_input_indices(
        projected_op,
        vec!["site0".to_string()],
        &state,
        &reference_state,
        expected,
    );

    let site0 = "site0".to_string();
    let x = state
        .tensor(state.node_index(&site0).unwrap())
        .unwrap()
        .clone();
    let err = linop.apply_projected(&x).unwrap_err();
    assert!(err.to_string().contains("index structure mismatch"));
}

/// Apply with x whose index structure differs from operator output triggers index mismatch error.
#[test]
fn test_local_linop_apply_index_mismatch() {
    use crate::linsolve::common::ProjectedOperator;

    let mut state = TreeTN::<IdxTensor, String>::new();
    let s0 = DynIndex::new_dyn(2);
    let t0 = IdxTensor::from_dense(vec![s0.clone()], vec![1.0, 2.0]).unwrap();
    state.add_tensor("site0".to_string(), t0).unwrap();

    let reference_state = state.clone();
    let projected_op = Arc::new(RwLock::new(ProjectedOperator::new(state.clone())));

    let expected = tensor_indices(&state, "site0");
    let linop = LocalLinOp::with_expected_input_indices(
        projected_op,
        vec!["site0".to_string()],
        &state,
        &reference_state,
        expected,
    );

    let other = DynIndex::new_dyn(2);
    let x = IdxTensor::from_dense(vec![other], vec![1.0, 0.0]).unwrap();
    let err = linop.apply_projected(&x).unwrap_err();
    assert!(err.to_string().contains("index structure mismatch"));
}

/// Apply success with 1-node MPO-like state and identity operator (index mappings).
#[test]
fn test_local_linop_apply_success_mappings() {
    use crate::linsolve::common::ProjectedOperator;

    let phys_dim = 2usize;
    let ext_dim = 2usize;
    let mut used = HashSet::<DynId>::new();
    let contracted = unique_dyn_index(&mut used, phys_dim);
    let external = unique_dyn_index(&mut used, ext_dim);

    let mut state = TreeTN::<IdxTensor, String>::new();
    let nelem = ext_dim * phys_dim;
    let t = IdxTensor::from_dense(vec![external.clone(), contracted.clone()], vec![1.0; nelem])
        .unwrap();
    state.add_tensor("site0".to_string(), t).unwrap();

    let s_in = unique_dyn_index(&mut used, phys_dim);
    let s_out = unique_dyn_index(&mut used, phys_dim);
    let mut id_data = vec![0.0_f64; phys_dim * phys_dim];
    for k in 0..phys_dim {
        id_data[k * phys_dim + k] = 1.0;
    }
    let op_t = IdxTensor::from_dense(vec![s_out.clone(), s_in.clone()], id_data).unwrap();
    let mut op_tn = TreeTN::<IdxTensor, String>::new();
    op_tn.add_tensor("site0".to_string(), op_t).unwrap();

    let mut im = HashMap::new();
    im.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: contracted.clone(),
            internal_index: s_in,
        },
    );
    let mut om = HashMap::new();
    om.insert(
        "site0".to_string(),
        IndexMapping {
            true_index: contracted,
            internal_index: s_out,
        },
    );

    let projected_op = Arc::new(RwLock::new(ProjectedOperator::with_index_mappings(
        op_tn, im, om,
    )));
    let reference_state = state.clone();

    let expected = tensor_indices(&state, "site0");
    let linop = LocalLinOp::with_expected_input_indices(
        projected_op,
        vec!["site0".to_string()],
        &state,
        &reference_state,
        expected,
    );

    let site0 = "site0".to_string();
    let x = state
        .tensor(state.node_index(&site0).unwrap())
        .unwrap()
        .clone();
    let y = linop.apply_projected(&x).unwrap();
    let x_ids: HashSet<_> = x
        .external_indices()
        .iter()
        .map(|i: &DynIndex| *i.id())
        .collect();
    let y_ids: HashSet<_> = y
        .external_indices()
        .iter()
        .map(|i: &DynIndex| *i.id())
        .collect();
    assert_eq!(x_ids, y_ids);
}
