use std::collections::HashSet;

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};

use crate::operator::Operator;
use crate::treetn::TreeTN;

/// Helper to create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> (
    TreeTN<IdxTensor, String>,
    DynIndex, // s0
    DynIndex, // bond
    DynIndex, // s1
) {
    let s0 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(4);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 12]).unwrap();

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    (tn, s0, bond, s1)
}

#[test]
fn test_site_indices() {
    let (tn, s0, bond, s1) = make_two_node_treetn();
    let site_indices = tn.site_indices();

    // Site indices should include s0 and s1
    let site_ids: HashSet<_> = site_indices.iter().map(|i| *i.id()).collect();
    assert!(site_ids.contains(s0.id()));
    assert!(site_ids.contains(s1.id()));
    // Bond should NOT be a site index
    assert!(!site_ids.contains(bond.id()));
    assert_eq!(site_indices.len(), 2);
}

#[test]
fn test_site_index_network() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();
    let sin = Operator::site_index_network(&tn);

    // Should have same node count as the TreeTN
    assert_eq!(sin.node_names().len(), 2);
}

#[test]
fn test_node_names() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();
    let names: HashSet<String> = Operator::node_names(&tn);

    assert_eq!(names.len(), 2);
    assert!(names.contains("A"));
    assert!(names.contains("B"));
}

#[test]
fn test_site_indices_single_node() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let t = IdxTensor::from_dense(vec![s0.clone(), s1.clone()], vec![0.0; 6]).unwrap();
    let tn = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();

    let site_indices = tn.site_indices();
    assert_eq!(site_indices.len(), 2);

    let names: HashSet<String> = Operator::node_names(&tn);
    assert_eq!(names.len(), 1);
    assert!(names.contains("A"));
}

#[test]
fn test_site_indices_three_node_chain() {
    let s0 = DynIndex::new_dyn(2);
    let bond01 = DynIndex::new_dyn(3);
    let s1 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);
    let s2 = DynIndex::new_dyn(2);

    let t0 = IdxTensor::from_dense(vec![s0.clone(), bond01.clone()], vec![1.0; 6]).unwrap();
    let t1 = IdxTensor::from_dense(
        vec![bond01.clone(), s1.clone(), bond12.clone()],
        vec![1.0; 18],
    )
    .unwrap();
    let t2 = IdxTensor::from_dense(vec![bond12.clone(), s2.clone()], vec![1.0; 6]).unwrap();

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1, t2],
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    )
    .unwrap();

    let site_indices = tn.site_indices();
    assert_eq!(site_indices.len(), 3);

    let names: HashSet<String> = Operator::node_names(&tn);
    assert_eq!(names.len(), 3);
    assert!(names.contains("A"));
    assert!(names.contains("B"));
    assert!(names.contains("C"));
}
