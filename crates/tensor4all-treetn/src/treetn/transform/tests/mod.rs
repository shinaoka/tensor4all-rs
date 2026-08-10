use super::*;
use crate::site_index_network::SiteIndexNetwork;
use std::collections::HashSet;
use tensor4all_core::{DynIndex, TensorDynLen};

fn make_two_node_chain() -> (TreeTN<TensorDynLen, String>, DynIndex, DynIndex) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    // Make a simple chain A --bond-- B
    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond.clone()], vec![1.0; 2 * 4]).unwrap();
    let t1 = TensorDynLen::from_dense(vec![bond.clone(), s1.clone()], vec![1.0; 4 * 3]).unwrap();

    tn.add_tensor("A".to_string(), t0).unwrap();
    tn.add_tensor("B".to_string(), t1).unwrap();

    let a = tn.node_index(&"A".to_string()).unwrap();
    let b = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(a, &bond, b, &bond).unwrap();

    (tn, s0, s1)
}

#[test]
fn test_fuse_to_two_nodes_into_one() {
    let (tn, s0, s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 1);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "fused tensor should match original contraction"
    );
}

#[test]
fn test_split_to_one_node_into_two() {
    let (tn, s0, s1) = make_two_node_chain();

    // First fuse into a single node AB.
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    // Now split AB into A -- B.
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    split_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    split_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let split = fused
        .split_to(&split_target, &SplitOptions::default())
        .unwrap();
    assert_eq!(split.node_count(), 2);

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(
        fused_full.distance(&split_full).unwrap() < 1e-12,
        "split tensor should match fused tensor"
    );
}

#[test]
fn test_fuse_to_errors_for_unknown_site_index() {
    let (tn, _s0, _s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("X".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();

    assert!(tn.fuse_to(&target).is_err());
}

/// Create a 3-node chain: A -- B -- C
fn make_three_node_chain() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex, // s0
    DynIndex, // s1
    DynIndex, // s2
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(3);
    let bond_ab = DynIndex::new_dyn(2);
    let bond_bc = DynIndex::new_dyn(2);

    let t0 = TensorDynLen::from_dense(vec![s0.clone(), bond_ab.clone()], vec![1.0; 2 * 2]).unwrap();
    let t1 = TensorDynLen::from_dense(
        vec![bond_ab.clone(), s1.clone(), bond_bc.clone()],
        vec![1.0; 2 * 2 * 2],
    )
    .unwrap();
    let t2 = TensorDynLen::from_dense(vec![bond_bc.clone(), s2.clone()], vec![1.0; 2 * 3]).unwrap();

    tn.add_tensor("A".to_string(), t0).unwrap();
    tn.add_tensor("B".to_string(), t1).unwrap();
    tn.add_tensor("C".to_string(), t2).unwrap();

    let a = tn.node_index(&"A".to_string()).unwrap();
    let b = tn.node_index(&"B".to_string()).unwrap();
    let c = tn.node_index(&"C".to_string()).unwrap();
    tn.connect(a, &bond_ab, b, &bond_ab).unwrap();
    tn.connect(b, &bond_bc, c, &bond_bc).unwrap();

    (tn, s0, s1, s2)
}

fn make_two_node_groups_of_two() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let x0 = DynIndex::new_dyn(2);
    let x1 = DynIndex::new_dyn(2);
    let y0 = DynIndex::new_dyn(2);
    let y1 = DynIndex::new_dyn(2);
    let old_bond = DynIndex::new_dyn(2);

    let left = TensorDynLen::from_dense(
        vec![x0.clone(), x1.clone(), old_bond.clone()],
        vec![1.0; 2 * 2 * 2],
    )
    .unwrap();
    let right =
        TensorDynLen::from_dense(vec![old_bond, y0.clone(), y1.clone()], vec![1.0; 2 * 2 * 2])
            .unwrap();
    let tn = TreeTN::<TensorDynLen, String>::from_tensors(
        vec![left, right],
        vec!["left".to_string(), "right".to_string()],
    )
    .unwrap();

    (tn, x0, x1, y0, y1)
}

fn error_chain_contains(error: &anyhow::Error, needle: &str) -> bool {
    error
        .chain()
        .any(|cause| cause.to_string().contains(needle))
}

#[test]
fn test_fuse_to_three_nodes_pairwise() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    // Fuse A+B into AB, keep C
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([s2.clone()]))
        .unwrap();
    target
        .add_edge(&"AB".to_string(), &"C".to_string())
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full).unwrap() < 1e-12);
}

#[test]
fn test_fuse_to_three_nodes_into_one() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node(
            "ABC".to_string(),
            HashSet::from([s0.clone(), s1.clone(), s2.clone()]),
        )
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 1);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full).unwrap() < 1e-12);
}

#[test]
fn test_fuse_to_missing_target_for_current_node() {
    let (tn, s0, _s1, _s2) = make_three_node_chain();

    // Target only accounts for node A, missing B and C
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("X".to_string(), HashSet::from([s0.clone()]))
        .unwrap();

    let result = tn.fuse_to(&target);
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("missing target") || msg.contains("no corresponding target"));
}

#[test]
fn test_split_to_with_actual_splitting() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    // Fuse all into one node
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node(
            "ABC".to_string(),
            HashSet::from([s0.clone(), s1.clone(), s2.clone()]),
        )
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();
    assert_eq!(fused.node_count(), 1);

    // Split back into 3 nodes
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    split_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    split_target
        .add_node("C".to_string(), HashSet::from([s2.clone()]))
        .unwrap();
    split_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    split_target
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();

    let split = fused
        .split_to(&split_target, &SplitOptions::default())
        .unwrap();
    assert_eq!(split.node_count(), 3);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&split_target),
        "split_to must preserve the requested chain topology, not silently produce a star/tree"
    );

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(fused_full.distance(&split_full).unwrap() < 1e-12);
}

#[test]
fn test_split_to_preserves_requested_chain_edges_across_original_bond() {
    let (tn, s11, s12, s21, s22) = make_two_node_groups_of_two();

    let mut target = SiteIndexNetwork::<usize, DynIndex>::new();
    target.add_node(0, HashSet::from([s11.clone()])).unwrap();
    target.add_node(1, HashSet::from([s12.clone()])).unwrap();
    target.add_node(2, HashSet::from([s21.clone()])).unwrap();
    target.add_node(3, HashSet::from([s22.clone()])).unwrap();
    target.add_edge(&0, &1).unwrap();
    target.add_edge(&1, &2).unwrap();
    target.add_edge(&2, &3).unwrap();

    let split = tn.split_to(&target, &SplitOptions::default()).unwrap();

    assert_eq!(split.node_count(), 4);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "split_to must preserve the requested target site grouping and chain topology"
    );
}

#[test]
fn test_split_to_allows_edgeless_intermediate_fragments() {
    let (tn, x0, x1, y0, y1) = make_two_node_groups_of_two();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("left_x".to_string(), HashSet::from([x0.clone()]))
        .unwrap();
    target
        .add_node("left_y".to_string(), HashSet::from([x1.clone()]))
        .unwrap();
    target
        .add_node("right_y".to_string(), HashSet::from([y0.clone()]))
        .unwrap();
    target
        .add_node("right_z".to_string(), HashSet::from([y1.clone()]))
        .unwrap();

    let split = tn.split_to(&target, &SplitOptions::default()).unwrap();

    assert_eq!(split.node_count(), 4);
    assert_eq!(
        split
            .site_index_network()
            .find_node_by_index(&x0)
            .map(|name| name.as_str()),
        Some("left_x")
    );
    assert_eq!(
        split
            .site_index_network()
            .find_node_by_index(&x1)
            .map(|name| name.as_str()),
        Some("left_y")
    );
    assert_eq!(
        split
            .site_index_network()
            .find_node_by_index(&y0)
            .map(|name| name.as_str()),
        Some("right_y")
    );
    assert_eq!(
        split
            .site_index_network()
            .find_node_by_index(&y1)
            .map(|name| name.as_str()),
        Some("right_z")
    );
    assert!(
        tn.contract_to_tensor()
            .unwrap()
            .distance(&split.contract_to_tensor().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn test_split_to_rejects_missing_explicit_target_boundary_edge() {
    let (tn, x0, x1, y0, y1) = make_two_node_groups_of_two();

    let mut target = SiteIndexNetwork::<usize, DynIndex>::new();
    target.add_node(0, HashSet::from([x0])).unwrap();
    target.add_node(1, HashSet::from([x1])).unwrap();
    target.add_node(2, HashSet::from([y0])).unwrap();
    target.add_node(3, HashSet::from([y1])).unwrap();
    target.add_edge(&0, &1).unwrap();
    target.add_edge(&2, &3).unwrap();

    let error = tn.split_to(&target, &SplitOptions::default()).unwrap_err();

    assert!(error_chain_contains(
        &anyhow::Error::new(error),
        "expected exactly one target edge crossing current edge"
    ));
}

#[test]
fn test_split_to_rejects_ambiguous_explicit_target_boundary_edges() {
    let (tn, x0, x1, y0, y1) = make_two_node_groups_of_two();

    let mut target = SiteIndexNetwork::<usize, DynIndex>::new();
    target.add_node(0, HashSet::from([x0])).unwrap();
    target.add_node(1, HashSet::from([x1])).unwrap();
    target.add_node(2, HashSet::from([y0])).unwrap();
    target.add_node(3, HashSet::from([y1])).unwrap();
    target.add_edge(&0, &2).unwrap();
    target.add_edge(&1, &3).unwrap();

    let error = tn.split_to(&target, &SplitOptions::default()).unwrap_err();

    assert!(error_chain_contains(
        &anyhow::Error::new(error),
        "expected exactly one target edge crossing current edge"
    ));
}

#[test]
fn test_fuse_to_rejects_ambiguous_current_node_mapping() {
    let (tn, s0, s1) = make_two_node_chain();

    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    let mut incompatible_target = SiteIndexNetwork::<String, DynIndex>::new();
    incompatible_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    incompatible_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    incompatible_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let error = fused.fuse_to(&incompatible_target).unwrap_err();

    assert!(error_chain_contains(
        &anyhow::Error::new(error),
        "ambiguous mapping"
    ));
}

#[test]
fn test_fuse_to_rejects_disconnected_current_group() {
    let (tn, s0, s1, s2) = make_three_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("AC".to_string(), HashSet::from([s0.clone(), s2.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    target
        .add_edge(&"AC".to_string(), &"B".to_string())
        .unwrap();

    let error = tn.fuse_to(&target).unwrap_err();

    let err = anyhow::Error::new(error);
    assert!(error_chain_contains(&err, "failed to contract nodes"));
    assert!(error_chain_contains(&err, "connected subtree"));
}

#[test]
fn test_split_to_identity_preserves_explicit_edge() {
    let (tn, s0, s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let split = tn.split_to(&target, &SplitOptions::default()).unwrap();

    assert_eq!(split.node_count(), 2);
    assert!(split
        .site_index_network()
        .share_equivalent_site_index_network(&target));
    assert!(
        tn.contract_to_tensor()
            .unwrap()
            .distance(&split.contract_to_tensor().unwrap())
            .unwrap()
            < 1e-12
    );
}

#[test]
fn test_split_to_rejects_target_node_spanning_current_nodes() {
    let (tn, x0, x1, y0, y1) = make_two_node_groups_of_two();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("mixed".to_string(), HashSet::from([x0.clone(), y0.clone()]))
        .unwrap();
    target
        .add_node("left_tail".to_string(), HashSet::from([x1.clone()]))
        .unwrap();
    target
        .add_node("right_tail".to_string(), HashSet::from([y1.clone()]))
        .unwrap();

    let error = tn.split_to(&target, &SplitOptions::default()).unwrap_err();

    let err = anyhow::Error::new(error);
    assert!(error_chain_contains(&err, "spans multiple current nodes"));
}

#[test]
fn test_split_to_rejects_current_site_missing_from_target() {
    let (tn, s0, _s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();

    let error = tn.split_to(&target, &SplitOptions::default()).unwrap_err();

    let err = anyhow::Error::new(error);
    assert!(error_chain_contains(
        &err,
        "has no corresponding target node"
    ));
    assert!(error_chain_contains(&err, "incompatible target structure"));
}

#[test]
fn test_split_to_with_final_sweep() {
    let (tn, s0, s1) = make_two_node_chain();

    // Fuse into one node
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node("AB".to_string(), HashSet::from([s0.clone(), s1.clone()]))
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    // Split back with final_sweep enabled
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    split_target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    split_target
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let options = SplitOptions {
        final_sweep: true,
        ..SplitOptions::default()
    };

    let split = fused.split_to(&split_target, &options).unwrap();
    assert_eq!(split.node_count(), 2);

    let fused_full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(fused_full.distance(&split_full).unwrap() < 1e-10);
}

#[test]
fn test_split_to_incompatible_target() {
    let (tn, _s0, _s1) = make_two_node_chain();

    // Target has a site index not in current TN
    let mut split_target = SiteIndexNetwork::<String, DynIndex>::new();
    split_target
        .add_node("X".to_string(), HashSet::from([DynIndex::new_dyn(7)]))
        .unwrap();
    split_target
        .add_node("Y".to_string(), HashSet::from([DynIndex::new_dyn(8)]))
        .unwrap();
    split_target
        .add_edge(&"X".to_string(), &"Y".to_string())
        .unwrap();

    let result = tn.split_to(&split_target, &SplitOptions::default());
    assert!(result.is_err());
}

#[test]
fn test_fuse_identity_mapping() {
    // Fuse where each current node maps to exactly one target (identity mapping)
    let (tn, s0, s1) = make_two_node_chain();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s0.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s1.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(orig_full.distance(&fused_full).unwrap() < 1e-12);
}

fn make_four_site_fused_node() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let s_a = DynIndex::new_dyn(2);
    let s_b = DynIndex::new_dyn(2);
    let s_c = DynIndex::new_dyn(2);
    let s_d = DynIndex::new_dyn(2);
    let t = TensorDynLen::from_dense(
        vec![s_a.clone(), s_b.clone(), s_c.clone(), s_d.clone()],
        vec![1.0; 16],
    )
    .unwrap();
    let mut tn = TreeTN::<TensorDynLen, String>::new();
    tn.add_tensor("fused".to_string(), t).unwrap();
    (tn, s_a, s_b, s_c, s_d)
}

#[test]
fn test_split_to_y_shape() {
    let (tn, s_a, s_b, s_c, s_d) = make_four_site_fused_node();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s_a.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s_b.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([s_c.clone()]))
        .unwrap();
    target
        .add_node("D".to_string(), HashSet::from([s_d.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

    let split = tn.split_to(&target, &SplitOptions::default()).unwrap();
    assert_eq!(split.node_count(), 4);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "Y-shape topology must be preserved",
    );

    let full = tn.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(full.distance(&split_full).unwrap() < 1e-12);
}

#[test]
fn test_split_to_nested_tree() {
    let (tn, s_a, s_b, s_c, s_d) = make_four_site_fused_node();

    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([s_a.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([s_b.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([s_c.clone()]))
        .unwrap();
    target
        .add_node("D".to_string(), HashSet::from([s_d.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    target.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let split = tn.split_to(&target, &SplitOptions::default()).unwrap();
    assert_eq!(split.node_count(), 4);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "nested tree topology must be preserved"
    );

    let full = tn.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(full.distance(&split_full).unwrap() < 1e-12);
}

#[test]
fn test_split_to_y_shape_across_original_bond() {
    let (tn, x0, x1, y0, y1) = make_two_node_groups_of_two();

    // Fuse both nodes into one first
    let mut fuse_target = SiteIndexNetwork::<String, DynIndex>::new();
    fuse_target
        .add_node(
            "F".to_string(),
            HashSet::from([x0.clone(), x1.clone(), y0.clone(), y1.clone()]),
        )
        .unwrap();
    let fused = tn.fuse_to(&fuse_target).unwrap();

    // Y-shape: A(x0)--B(x1)--C(y0)--D(y1) → chain is just chain, not Y
    // Let's do: A(x0)-B(x1), B(x1)-C(y0), B(x1)-D(y1) with all sites in one node first
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([x0.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([x1.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([y0.clone()]))
        .unwrap();
    target
        .add_node("D".to_string(), HashSet::from([y1.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

    let split = fused.split_to(&target, &SplitOptions::default()).unwrap();
    assert_eq!(split.node_count(), 4);
    assert!(
        split
            .site_index_network()
            .share_equivalent_site_index_network(&target),
        "Y-shape from fused node must be preserved"
    );

    let full = fused.contract_to_tensor().unwrap();
    let split_full = split.contract_to_tensor().unwrap();
    assert!(full.distance(&split_full).unwrap() < 1e-12);
}

// ============================================================================
// Y-shape branching topology tests for fuse_to
// ============================================================================

/// Create a Y-shape TreeTN with a site index on each node:
///     A (site_a)
///     |
///     B (site_b)
///    / \
///   C   D
/// (site_c) (site_d)
fn make_y_shape_treetn() -> (
    TreeTN<TensorDynLen, String>,
    DynIndex, // site_a
    DynIndex, // site_b
    DynIndex, // site_c
    DynIndex, // site_d
) {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    let tensor_a =
        TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    let tensor_b = TensorDynLen::from_dense(
        vec![
            bond_ab.clone(),
            bond_bc.clone(),
            bond_bd.clone(),
            site_b.clone(),
        ],
        vec![1.0; 54],
    )
    .unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    let tensor_c =
        TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    let tensor_d =
        TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("D".to_string(), tensor_d).unwrap();

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

    (tn, site_a, site_b, site_c, site_d)
}

#[test]
fn test_fuse_to_y_shape_fuse_subtree() {
    let (tn, site_a, site_b, site_c, site_d) = make_y_shape_treetn();

    // Fuse B+C+D into one node, keep A separate
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a.clone()]))
        .unwrap();
    target
        .add_node(
            "BCD".to_string(),
            HashSet::from([site_b.clone(), site_c.clone(), site_d.clone()]),
        )
        .unwrap();
    target
        .add_edge(&"A".to_string(), &"BCD".to_string())
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "fused Y-shape branch should match original contraction"
    );
}

#[test]
fn test_fuse_to_y_shape_all_into_one() {
    let (tn, site_a, site_b, site_c, site_d) = make_y_shape_treetn();

    // Fuse all 4 nodes into 1
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node(
            "ABCD".to_string(),
            HashSet::from([
                site_a.clone(),
                site_b.clone(),
                site_c.clone(),
                site_d.clone(),
            ]),
        )
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 1);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "fused entire Y-shape should match original contraction"
    );
}

#[test]
fn test_fuse_to_y_shape_identity() {
    let (tn, site_a, site_b, site_c, site_d) = make_y_shape_treetn();

    // Identity mapping: each current node → same target (no actual fusion)
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a.clone()]))
        .unwrap();
    target
        .add_node("B".to_string(), HashSet::from([site_b.clone()]))
        .unwrap();
    target
        .add_node("C".to_string(), HashSet::from([site_c.clone()]))
        .unwrap();
    target
        .add_node("D".to_string(), HashSet::from([site_d.clone()]))
        .unwrap();
    target.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    target.add_edge(&"B".to_string(), &"D".to_string()).unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 4);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "identity fuse on Y-shape should match original contraction"
    );
}

#[test]
fn test_fuse_to_y_shape_internal_center_node() {
    // Y-shape where center node B has NO site index:
    //     A(site_a)
    //       |
    //     B(no site)   <-- internal node, only bonds
    //      / \
    // C(site_c) D(site_d)
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);

    let tensor_a =
        TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // B has NO site index -- only bond indices
    let tensor_b = TensorDynLen::from_dense(
        vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
        vec![1.0; 27],
    )
    .unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    let tensor_c =
        TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    let tensor_d =
        TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("D".to_string(), tensor_d).unwrap();

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();
    let n_d = tn.node_index(&"D".to_string()).unwrap();
    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

    // Fuse C+D into one node (B is internal, pulled in automatically)
    let mut target = SiteIndexNetwork::<String, DynIndex>::new();
    target
        .add_node("A".to_string(), HashSet::from([site_a.clone()]))
        .unwrap();
    target
        .add_node(
            "CD".to_string(),
            HashSet::from([site_c.clone(), site_d.clone()]),
        )
        .unwrap();
    target
        .add_edge(&"A".to_string(), &"CD".to_string())
        .unwrap();

    let fused = tn.fuse_to(&target).unwrap();
    assert_eq!(fused.node_count(), 2);

    let orig_full = tn.contract_to_tensor().unwrap();
    let fused_full = fused.contract_to_tensor().unwrap();
    assert!(
        orig_full.distance(&fused_full).unwrap() < 1e-12,
        "fused Y-shape with internal center should match original contraction"
    );
}
