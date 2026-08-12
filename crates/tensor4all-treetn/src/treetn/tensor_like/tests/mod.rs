use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TagSet, TensorIndex};

use crate::treetn::TreeTN;
use crate::NumberedTagSelectionError;

/// Helper to create a simple 2-node TreeTN: A -- bond -- B
fn make_two_node_treetn() -> (
    TreeTN<IdxTensor, String>,
    DynIndex, // s0
    DynIndex, // bond
    DynIndex, // s1
) {
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
fn test_external_indices() {
    let (tn, s0, bond, s1) = make_two_node_treetn();
    let ext = tn.external_indices();
    assert_eq!(ext.len(), 2);

    let ext_ids: Vec<_> = ext.iter().map(|i| *i.id()).collect();
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));
    // Bond should NOT be in external indices
    assert!(!ext_ids.contains(bond.id()));
}

#[test]
fn test_num_external_indices() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();
    assert_eq!(tn.num_external_indices(), 2);
}

#[test]
fn test_num_external_indices_single_node() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);
    let t = IdxTensor::from_dense(vec![i.clone(), j.clone(), k.clone()], vec![0.0; 24]).unwrap();
    let tn = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();
    assert_eq!(tn.num_external_indices(), 3);
}

#[test]
fn test_external_indices_with_tag_filters_site_indices() {
    let x = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,x").unwrap());
    let y = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,y").unwrap());
    let bond = DynIndex::new_dyn(3);
    let t0 = IdxTensor::from_dense(
        vec![x.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )
    .unwrap();
    let t1 =
        IdxTensor::from_dense(vec![bond, y.clone()], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![t0, t1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    assert_eq!(tn.external_indices_with_tag("x"), vec![x]);
    assert_eq!(
        tn.external_indices_with_tag("missing"),
        Vec::<DynIndex>::new()
    );
}

#[test]
fn test_external_indices_with_numbered_tag_uses_explicit_range() {
    let k2 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    let k3 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=3").unwrap());
    let x1 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,x=1").unwrap());
    let tensor = IdxTensor::from_dense(vec![x1, k3.clone(), k2.clone()], vec![0.0; 8]).unwrap();
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let indices = tn.external_indices_with_numbered_tag("k", 2, 2).unwrap();

    assert_eq!(indices, vec![k2, k3]);
}

#[test]
fn test_external_indices_with_numbered_tag_keeps_same_id_tags_distinct() {
    let base = DynIndex::new_dyn(2);
    let k1 = DynIndex::new_with_tags(*base.id(), 2, TagSet::from_str("Qubit,k=1").unwrap());
    let other = DynIndex::new_with_tags(*base.id(), 2, TagSet::from_str("Qubit,other").unwrap());
    let tensor = IdxTensor::from_dense(vec![other, k1.clone()], vec![0.0; 4]).unwrap();
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    assert_eq!(tn.external_indices_with_tag("k=1"), vec![k1.clone()]);
    assert_eq!(
        tn.external_indices_with_numbered_tag("k", 1, 1).unwrap(),
        vec![k1]
    );
}

#[test]
fn test_external_indices_with_numbered_tag_rejects_missing_number() {
    let k2 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    let k4 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=4").unwrap());
    let tensor = IdxTensor::from_dense(vec![k2, k4], vec![0.0; 4]).unwrap();
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let err = tn
        .external_indices_with_numbered_tag("k", 2, 3)
        .unwrap_err();

    assert!(matches!(
        err,
        NumberedTagSelectionError::MissingTag { ref tag } if tag == "k=3"
    ));
}

#[test]
fn test_external_indices_with_numbered_tag_rejects_duplicate_number() {
    let k1_left = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let k1_right = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let tensor = IdxTensor::from_dense(vec![k1_left, k1_right], vec![0.0; 4]).unwrap();
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let err = tn
        .external_indices_with_numbered_tag("k", 1, 1)
        .unwrap_err();

    assert!(matches!(
        err,
        NumberedTagSelectionError::AmbiguousTag { ref tag } if tag == "k=1"
    ));
}

#[test]
fn test_external_indices_with_numbered_tag_rejects_prefix_with_equals() {
    let k1 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    let tensor = IdxTensor::from_dense(vec![k1], vec![0.0; 2]).unwrap();
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let err = tn
        .external_indices_with_numbered_tag("k=1", 1, 1)
        .unwrap_err();

    assert!(matches!(
        err,
        NumberedTagSelectionError::InvalidPrefix { ref tag_prefix } if tag_prefix == "k=1"
    ));
}

#[test]
fn test_replaceind_site_index() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let tn2 = tn.replaceind(&s0, &s0_new).unwrap();

    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(!ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s0_new.id()));
    assert!(ext_ids.contains(s1.id()));
}

#[test]
fn test_replaceind_matches_same_id_prime_pair_site_index_exactly() {
    let i = DynIndex::new_dyn(2);
    let i_prime = i.prime();
    let replacement = i_prime.sim();
    let t =
        IdxTensor::from_dense(vec![i.clone(), i_prime.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let tn = TreeTN::<IdxTensor, String>::from_tensors(vec![t], vec!["A".to_string()]).unwrap();

    let replaced = tn.replaceind(&i_prime, &replacement).unwrap();
    let site_space = replaced.site_space(&"A".to_string()).unwrap();
    assert!(site_space.contains(&i));
    assert!(!site_space.contains(&i_prime));
    assert!(site_space.contains(&replacement));

    let tensor_indices = replaced
        .tensor(replaced.node_index(&"A".to_string()).unwrap())
        .unwrap()
        .external_indices();
    assert_eq!(tensor_indices, vec![i, replacement]);
}

#[test]
fn test_replaceind_link_index_via_sim_link_indices() {
    let (tn, s0, bond, s1) = make_two_node_treetn();

    // sim_link_indices replaces all link indices with fresh IDs
    let tn2 = tn.sim_link_indices().unwrap();

    // Site indices should remain unchanged
    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    // The bond index should have a different ID from the original
    let edge = tn2.graph.graph().edge_indices().next().unwrap();
    let new_bond = tn2.bond_index(edge).unwrap();
    assert_ne!(*new_bond.id(), *bond.id());
    // But same dimension
    assert_eq!(new_bond.dim(), bond.dim());
}

#[test]
fn test_replaceind_dimension_mismatch() {
    let (tn, s0, _bond, _s1) = make_two_node_treetn();

    let wrong_dim = DynIndex::new_dyn(5); // s0 has dim 2
    let result = tn.replaceind(&s0, &wrong_dim);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Index space mismatch"));
}

#[test]
fn test_replaceind_not_found() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();

    let unknown = DynIndex::new_dyn(7);
    let new_idx = DynIndex::new_dyn(7);
    let result = tn.replaceind(&unknown, &new_idx);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_replace_indices_multiple() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let s1_new = DynIndex::new_dyn(2);

    let tn2 = tn
        .replace_indices(&[s0.clone(), s1.clone()], &[s0_new.clone(), s1_new.clone()])
        .unwrap();

    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(!ext_ids.contains(s0.id()));
    assert!(!ext_ids.contains(s1.id()));
    assert!(ext_ids.contains(s0_new.id()));
    assert!(ext_ids.contains(s1_new.id()));
}

#[test]
fn test_replace_indices_length_mismatch() {
    let (tn, s0, _bond, _s1) = make_two_node_treetn();

    let s0_new = DynIndex::new_dyn(2);
    let s1_new = DynIndex::new_dyn(2);

    let result = tn.replace_indices(std::slice::from_ref(&s0), &[s0_new.clone(), s1_new.clone()]);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("Length mismatch"));
}

#[test]
fn test_replaceind_link_index_direct() {
    let (tn, s0, _bond, s1) = make_two_node_treetn();

    let edge = tn.graph.graph().edge_indices().next().unwrap();
    let actual_bond = tn.bond_index(edge).unwrap().clone();

    let new_bond = actual_bond.sim();
    let tn2 = tn.replaceind(&actual_bond, &new_bond).unwrap();

    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s1.id()));

    let edge2 = tn2.graph.graph().edge_indices().next().unwrap();
    let updated_bond = tn2.bond_index(edge2).unwrap();
    assert_eq!(*updated_bond.id(), *new_bond.id());
    assert_ne!(*updated_bond.id(), *actual_bond.id());

    // The contracted result should be the same
    let orig_dense = tn.contract_to_tensor().unwrap();
    let new_dense = tn2.contract_to_tensor().unwrap();
    assert!(orig_dense.distance(&new_dense).unwrap() < 1e-12);
}

#[test]
fn test_replaceind_site_index_with_ortho_towards() {
    let (mut tn, s0, _bond, s1) = make_two_node_treetn();

    // Set ortho_towards for s0 to point towards "B"
    tn.set_ortho_towards(&s0, Some("B".to_string()));
    assert_eq!(tn.ortho_towards_for_index(&s0), Some(&"B".to_string()));

    // Replace s0 with a new index
    let s0_new = DynIndex::new_dyn(2);
    let tn2 = tn.replaceind(&s0, &s0_new).unwrap();

    // ortho_towards should now use the new index
    assert!(tn2.ortho_towards_for_index(&s0).is_none());
    assert_eq!(tn2.ortho_towards_for_index(&s0_new), Some(&"B".to_string()));

    // Site indices should be updated
    let ext_ids: Vec<_> = tn2.external_indices().iter().map(|i| *i.id()).collect();
    assert!(!ext_ids.contains(s0.id()));
    assert!(ext_ids.contains(s0_new.id()));
    assert!(ext_ids.contains(s1.id()));
}

#[test]
fn test_replaceind_link_index_dimension_mismatch() {
    let (tn, _s0, _bond, _s1) = make_two_node_treetn();

    // Get the actual bond index from the TreeTN
    let edge = tn.graph.graph().edge_indices().next().unwrap();
    let actual_bond = tn.bond_index(edge).unwrap().clone();

    let wrong_dim = DynIndex::new_dyn(7); // bond has dim 3
    let result = tn.replaceind(&actual_bond, &wrong_dim);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("Index space mismatch"));
}
