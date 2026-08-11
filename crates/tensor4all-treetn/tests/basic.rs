//! Basic tests for TreeTN
//!
//! In Einsum mode, tensors that share a bond must use the SAME Index (same ID).
//! When connecting node_a to node_b, both tensors must contain the same bond index.
//!
//! Migrated from tensor4all-rs main branch to index-like branch with type adjustments:
//! - Index<DynId> -> DynIndex
//! - idx.id -> idx.id()
//! - Tests using random_treetn_f64/LinkSpace are commented out (random module disabled)

use petgraph::graph::NodeIndex;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_treetn::CanonicalForm;
use tensor4all_treetn::{CanonicalizationOptions, TreeTN, TruncationOptions};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simple 2-node TreeTN with shared bond index.
/// Returns (tn, node1, node2, edge, physical1, bond, physical2)
fn create_two_node_treetn() -> (
    TreeTN<TensorDynLen, NodeIndex>,
    NodeIndex,
    NodeIndex,
    petgraph::graph::EdgeIndex,
    DynIndex,
    DynIndex,
    DynIndex,
) {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let phys1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let phys2 = DynIndex::new_dyn(4);

    let tensor1 =
        TensorDynLen::from_dense(vec![phys1.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let tensor2 =
        TensorDynLen::from_dense(vec![bond.clone(), phys2.clone()], vec![1.0; 12]).unwrap();
    let node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    let edge = tn.connect(node1, &bond, node2, &bond).unwrap();

    (tn, node1, node2, edge, phys1, bond, phys2)
}

/// Create a 3-node chain: n1 -- n2 -- n3
fn create_three_node_chain() -> (
    TreeTN<TensorDynLen, NodeIndex>,
    NodeIndex,
    NodeIndex,
    NodeIndex,
    petgraph::graph::EdgeIndex,
    petgraph::graph::EdgeIndex,
    DynIndex, // bond12
    DynIndex, // bond23
) {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let phys1 = DynIndex::new_dyn(2);
    let bond12 = DynIndex::new_dyn(3);
    let bond23 = DynIndex::new_dyn(4);
    let phys3 = DynIndex::new_dyn(5);

    let tensor1 =
        TensorDynLen::from_dense(vec![phys1.clone(), bond12.clone()], vec![1.0; 6]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let tensor2 =
        TensorDynLen::from_dense(vec![bond12.clone(), bond23.clone()], vec![1.0; 12]).unwrap();
    let node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    let tensor3 =
        TensorDynLen::from_dense(vec![bond23.clone(), phys3.clone()], vec![1.0; 20]).unwrap();
    let node3 = tn.add_tensor_auto_name(tensor3).unwrap();

    let edge12 = tn.connect(node1, &bond12, node2, &bond12).unwrap();
    let edge23 = tn.connect(node2, &bond23, node3, &bond23).unwrap();

    (tn, node1, node2, node3, edge12, edge23, bond12, bond23)
}

// ============================================================================
// Basic TreeTN Tests
// ============================================================================

#[test]
fn test_treetn_add_tensor() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = TensorDynLen::from_dense(vec![i, j], vec![0.0; 6]).unwrap();

    let node = tn.add_tensor_auto_name(tensor).unwrap();
    assert_eq!(tn.node_count(), 1);

    let retrieved = tn.tensor(node);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().dims(), vec![2, 3]);
}

#[test]
fn test_add_tensor_auto_name_returns_result() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = TensorDynLen::from_dense(vec![i, j], vec![0.0; 6]).unwrap();

    let node = tn.add_tensor_auto_name(tensor).unwrap();

    assert_eq!(node.index(), 0);
    assert_eq!(tn.node_count(), 1);
}

#[test]
fn test_treetn_replace_tensor() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![0.0; 6]).unwrap();
    let node = tn.add_tensor_auto_name(tensor).unwrap();

    // Replace with a tensor that has the same indices
    let new_tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![0.0; 6]).unwrap();
    let result = tn.replace_tensor(node, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_treetn_replace_tensor_nonexistent() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = TensorDynLen::from_dense(vec![i, j], vec![0.0; 6]).unwrap();

    let invalid_node = NodeIndex::new(999);
    let result = tn.replace_tensor(invalid_node, tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_treetn_replace_tensor_missing_bond_index() {
    let (mut tn, node1, _node2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Try to replace node1 with a tensor that doesn't have the bond index
    let new_i = DynIndex::new_dyn(5);
    let new_j = DynIndex::new_dyn(6);
    let new_tensor = TensorDynLen::from_dense(
        vec![new_i, new_j], // bond is missing!
        vec![0.0; 30],
    )
    .unwrap();

    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("missing") || err_msg.contains("index") || err_msg.contains("indices")
    );
}

#[test]
fn test_treetn_replace_tensor_with_bond() {
    let (mut tn, node1, _node2, _edge, _phys1, bond, _phys2) = create_two_node_treetn();

    // Replace node1 with a tensor that has the same bond index
    let new_i = DynIndex::new_dyn(5);
    let new_tensor = TensorDynLen::from_dense(vec![new_i, bond.clone()], vec![0.0; 15]).unwrap();

    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

// ============================================================================
// Connection Tests
// ============================================================================

#[test]
fn test_treetn_connect() {
    let (tn, _node1, _node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    assert_eq!(tn.edge_count(), 1);
    let bond_idx = tn.bond_index(edge);
    assert!(bond_idx.is_some());
    assert_eq!(bond_idx.unwrap().size(), 3);
}

#[test]
fn test_treetn_connect_id_mismatch() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i1 = DynIndex::new_dyn(2);
    let j1 = DynIndex::new_dyn(3);
    let tensor1 = TensorDynLen::from_dense(vec![i1.clone(), j1.clone()], vec![0.0; 6]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let i2 = DynIndex::new_dyn(3); // Different ID than j1
    let k2 = DynIndex::new_dyn(4);
    let tensor2 = TensorDynLen::from_dense(vec![i2.clone(), k2.clone()], vec![0.0; 12]).unwrap();
    let node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    // Try to connect with different index IDs - should fail in Einsum mode
    let result = tn.connect(node1, &j1, node2, &i2);
    assert!(result.is_err());
    // Use Debug format to get full error chain, or check root cause
    let err = result.unwrap_err();
    let err_chain = format!("{:?}", err);
    assert!(
        err_chain.contains("ID") || err_chain.contains("Einsum") || err_chain.contains("match"),
        "Error chain should mention ID mismatch: {}",
        err_chain
    );
}

#[test]
fn test_treetn_connect_invalid_node() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = TensorDynLen::from_dense(vec![i.clone(), j.clone()], vec![0.0; 6]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor).unwrap();

    let invalid_node = NodeIndex::new(999);
    let result = tn.connect(node1, &j, invalid_node, &j);
    assert!(result.is_err());
}

#[test]
fn test_treetn_connect_index_not_in_tensor() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i1 = DynIndex::new_dyn(2);
    let j1 = DynIndex::new_dyn(3);
    let tensor1 = TensorDynLen::from_dense(vec![i1.clone(), j1.clone()], vec![0.0; 6]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let bond = DynIndex::new_dyn(3);
    let k2 = DynIndex::new_dyn(4);
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k2.clone()], vec![0.0; 12]).unwrap();
    let node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    // Try to connect with an index that doesn't exist in tensor1
    let fake_index = DynIndex::new_dyn(3);
    let result = tn.connect(node1, &fake_index, node2, &bond);
    assert!(result.is_err());
}

// ============================================================================
// Bond Index Tests
// ============================================================================

#[test]
fn test_treetn_bond_index() {
    let (tn, _node1, _node2, edge, _phys1, bond, _phys2) = create_two_node_treetn();

    let bond_idx = tn.bond_index(edge).unwrap();
    assert_eq!(bond_idx.id(), bond.id());
    assert_eq!(bond_idx.size(), 3);
}

#[test]
fn test_treetn_replace_edge_bond() {
    let (mut tn, node1, _node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    assert_eq!(tn.bond_index(edge).unwrap().size(), 3);

    // Ensure ortho_towards is set for the old bond, and remains valid after replacement.
    tn.set_edge_ortho_towards(edge, Some(node1)).unwrap();
    assert_eq!(tn.ortho_towards_node(edge), Some(&node1));

    let old_bond_id = *tn.bond_index(edge).unwrap().id();

    // Replace with new bond index
    let new_idx = DynIndex::new_dyn(5);
    let result = tn.replace_edge_bond(edge, new_idx);
    assert!(result.is_ok());
    assert_eq!(tn.bond_index(edge).unwrap().size(), 5);

    // Bond ID should change, and ortho_towards should still point to the same node.
    assert_ne!(*tn.bond_index(edge).unwrap().id(), old_bond_id);
    assert_eq!(tn.ortho_towards_node(edge), Some(&node1));
}

// ============================================================================
// Ortho Towards Tests
// ============================================================================

#[test]
fn test_treetn_set_edge_ortho_towards() {
    let (mut tn, node1, node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Set ortho direction to node1
    let result = tn.set_edge_ortho_towards(edge, Some(node1));
    assert!(result.is_ok());
    assert_eq!(tn.ortho_towards_node(edge), Some(&node1));

    // Set ortho direction to node2
    let result = tn.set_edge_ortho_towards(edge, Some(node2));
    assert!(result.is_ok());
    assert_eq!(tn.ortho_towards_node(edge), Some(&node2));

    // Clear ortho direction
    let result = tn.set_edge_ortho_towards(edge, None);
    assert!(result.is_ok());
    assert!(tn.ortho_towards_node(edge).is_none());
}

#[test]
fn test_treetn_set_edge_ortho_towards_invalid() {
    let (mut tn, _node1, _node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Create a third node that is not connected to this edge
    let i3 = DynIndex::new_dyn(5);
    let k3 = DynIndex::new_dyn(6);
    let tensor3 = TensorDynLen::from_dense(vec![i3, k3], vec![0.0; 30]).unwrap();
    let node3 = tn.add_tensor_auto_name(tensor3).unwrap();

    // Try to set ortho direction to a node that's not an endpoint
    let result = tn.set_edge_ortho_towards(edge, Some(node3));
    assert!(result.is_err());
}

// ============================================================================
// Tree Validation Tests
// ============================================================================

#[test]
fn test_treetn_validate_tree_simple() {
    let (tn, _node1, _node2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_three_nodes() {
    let (tn, _n1, _n2, _n3, _e12, _e23, _b12, _b23) = create_three_node_chain();
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_cycle() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    // Create a cycle: n1 -- n2 -- n3 -- n1
    let bond12 = DynIndex::new_dyn(3);
    let bond23 = DynIndex::new_dyn(4);
    let bond31 = DynIndex::new_dyn(5);

    let tensor1 =
        TensorDynLen::from_dense(vec![bond12.clone(), bond31.clone()], vec![0.0; 15]).unwrap();
    let node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let tensor2 =
        TensorDynLen::from_dense(vec![bond12.clone(), bond23.clone()], vec![0.0; 12]).unwrap();
    let node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    let tensor3 =
        TensorDynLen::from_dense(vec![bond23.clone(), bond31.clone()], vec![0.0; 20]).unwrap();
    let node3 = tn.add_tensor_auto_name(tensor3).unwrap();

    tn.connect(node1, &bond12, node2, &bond12).unwrap();
    tn.connect(node2, &bond23, node3, &bond23).unwrap();
    tn.connect(node3, &bond31, node1, &bond31).unwrap();

    // Should fail: 3 nodes, 3 edges (cycle)
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_disconnected() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    // Create two disconnected nodes
    let i1 = DynIndex::new_dyn(2);
    let tensor1 = TensorDynLen::from_dense(vec![i1], vec![0.0; 2]).unwrap();
    let _node1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let i2 = DynIndex::new_dyn(3);
    let tensor2 = TensorDynLen::from_dense(vec![i2], vec![0.0; 3]).unwrap();
    let _node2 = tn.add_tensor_auto_name(tensor2).unwrap();

    // Should fail: not connected
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_empty() {
    let tn = TreeTN::<TensorDynLen, NodeIndex>::new();
    assert!(tn.validate_tree().is_ok());
}

// ============================================================================
// Ortho Region Tests
// ============================================================================

#[test]
fn test_canonical_region_empty() {
    let tn = TreeTN::<TensorDynLen, NodeIndex>::new();
    assert!(!tn.is_canonicalized());
    assert!(tn.canonical_region().is_empty());
}

#[test]
fn test_set_canonical_region() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i], vec![0.0; 2]).unwrap();
    let node = tn.add_tensor_auto_name(tensor).unwrap();

    assert!(!tn.is_canonicalized());

    let result = tn.set_canonical_region(vec![node]);
    assert!(result.is_ok());
    assert!(tn.is_canonicalized());
    assert!(tn.canonical_region().contains(&node));
}

#[test]
fn test_set_canonical_region_invalid_node() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let invalid_node = NodeIndex::new(999);
    let result = tn.set_canonical_region(vec![invalid_node]);
    assert!(result.is_err());
}

#[test]
fn test_clear_canonical_region() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i], vec![0.0; 2]).unwrap();
    let node = tn.add_tensor_auto_name(tensor).unwrap();

    tn.set_canonical_region(vec![node]).unwrap();
    assert!(tn.is_canonicalized());

    tn.clear_canonical_region();
    assert!(!tn.is_canonicalized());
    assert!(tn.canonical_region().is_empty());
}

// ============================================================================
// add_to/remove_from_canonical_region Tests
// ============================================================================

#[test]
fn test_add_to_canonical_region() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i], vec![0.0; 2]).unwrap();
    let node = tn.add_tensor_auto_name(tensor).unwrap();

    // Add valid node to canonical region
    tn.add_to_canonical_region(node).unwrap();
    assert!(tn.is_canonicalized());
    assert!(tn.canonical_region().contains(&node));
}

#[test]
fn test_add_to_canonical_region_invalid_node() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let invalid_node = NodeIndex::new(999);
    let result = tn.add_to_canonical_region(invalid_node);
    assert!(result.is_err());
}

#[test]
fn test_remove_from_canonical_region() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let tensor = TensorDynLen::from_dense(vec![i], vec![0.0; 2]).unwrap();
    let node = tn.add_tensor_auto_name(tensor).unwrap();

    tn.set_canonical_region(vec![node]).unwrap();
    assert!(tn.canonical_region().contains(&node));

    // Remove existing node
    let removed = tn.remove_from_canonical_region(&node);
    assert!(removed);
    assert!(!tn.canonical_region().contains(&node));

    // Remove non-existing node
    let removed = tn.remove_from_canonical_region(&node);
    assert!(!removed);
}

// ============================================================================
// Debug / Clone Tests
// ============================================================================

#[test]
fn test_treetn_debug_format() {
    let (tn, _n1, _n2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();
    let debug_str = format!("{:?}", tn);
    assert!(debug_str.contains("TreeTN"));
    assert!(debug_str.contains("node_count"));
    assert!(debug_str.contains("canonical_region"));
}

// ============================================================================
// Log Norm - Multi-site Canonical Region Path
// ============================================================================

#[test]
fn test_log_norm_already_canonicalized_single_site() {
    let (mut tn, n1, _n2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Canonicalize to single site with Unitary form
    tn = tn
        .canonicalize([n1], CanonicalizationOptions::default())
        .unwrap();
    assert!(tn.is_canonicalized());

    let log_norm = tn.log_norm().unwrap();
    assert!(log_norm.is_finite());
}

// ============================================================================
// Validate Ortho Consistency Tests
// ============================================================================

#[test]
fn test_validate_ortho_consistency_disconnected_centers() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    // Create two disconnected nodes
    let i1 = DynIndex::new_dyn(2);
    let tensor1 = TensorDynLen::from_dense(vec![i1], vec![0.0; 2]).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let i2 = DynIndex::new_dyn(3);
    let tensor2 = TensorDynLen::from_dense(vec![i2], vec![0.0; 3]).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    // Two centers that are not connected should fail
    tn.set_canonical_region(vec![n1, n2]).unwrap();
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_none_only_inside_centers() {
    let (mut tn, _node1, node2, edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Only node2 is center
    tn.set_canonical_region(vec![node2]).unwrap();

    // Clear ortho_towards on a boundary edge (should be forbidden)
    tn.set_edge_ortho_towards(edge, None).unwrap();
    assert!(tn.validate_ortho_consistency().is_err());
}

#[test]
fn test_validate_ortho_consistency_chain_pointing_towards_center() {
    let (mut tn, _n1, n2, _n3, e12, e23, _b12, _b23) = create_three_node_chain();

    // n2 is center
    tn.set_canonical_region(vec![n2]).unwrap();

    // Both boundary edges must point into center
    tn.set_edge_ortho_towards(e12, Some(n2)).unwrap();
    tn.set_edge_ortho_towards(e23, Some(n2)).unwrap();

    assert!(tn.validate_ortho_consistency().is_ok());
}

// ============================================================================
// Canonicalization Tests
// ============================================================================

#[test]
fn test_canonicalize_simple() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let phys1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let phys2 = DynIndex::new_dyn(4);

    let data1: Vec<f64> = vec![1.0; 6];
    let tensor1 = TensorDynLen::from_dense(vec![phys1.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let data2: Vec<f64> = vec![1.0; 12];
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), phys2.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    tn.connect(n1, &bond, n2, &bond).unwrap();

    // Canonicalize towards n2
    let tn_canon = tn
        .canonicalize(std::iter::once(n2), CanonicalizationOptions::default())
        .unwrap();

    assert!(tn_canon.is_canonicalized());
    assert!(tn_canon.canonical_region().contains(&n2));
    assert!(tn_canon.validate_ortho_consistency().is_ok());
}

#[test]
fn canonicalize_preserves_near_dependent_components() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let left = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(2);
    let right = DynIndex::new_dyn(2);

    let tensor1 = TensorDynLen::from_dense(
        vec![left.clone(), bond.clone()],
        vec![1.0, 0.0, 0.0, 1.0e-16],
    )
    .unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let tensor2 =
        TensorDynLen::from_dense(vec![bond.clone(), right.clone()], vec![1.0, 0.0, 0.0, 1.0])
            .unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    tn.connect(n1, &bond, n2, &bond).unwrap();
    let expected = tn.contract_to_tensor().unwrap();

    for form in [CanonicalForm::Unitary, CanonicalForm::LU] {
        let actual = tn
            .clone()
            .canonicalize(
                std::iter::once(n2),
                CanonicalizationOptions::default().with_form(form),
            )
            .unwrap()
            .contract_to_tensor()
            .unwrap();

        let error = actual.distance(&expected).unwrap();
        assert!(
            error < 1.0e-18,
            "{form:?} canonicalization changed the represented tensor: maxabs diff = {error}"
        );
    }
}

// ============================================================================
// Contraction Tests
// ============================================================================

#[test]
fn test_contract_two_nodes() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);

    let data1: Vec<f64> = (0..6).map(|x| x as f64).collect();
    let tensor1 = TensorDynLen::from_dense(vec![i.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let data2: Vec<f64> = (0..12).map(|x| x as f64).collect();
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    tn.connect(n1, &bond, n2, &bond).unwrap();

    let result = tn.contract_to_tensor().unwrap();

    // Result should have physical indices [i, k] with dims [2, 4]
    assert_eq!(result.dims(), vec![2, 4]);
    assert_eq!(result.indices.len(), 2);
}

#[test]
fn test_contract_chain() {
    let (tn, _n1, _n2, _n3, _e12, _e23, _b12, _b23) = create_three_node_chain();

    let result = tn.contract_to_tensor().unwrap();

    // Result should have physical indices from n1 and n3
    assert_eq!(result.dims(), vec![2, 5]);
}

/// Test that contract_to_tensor returns indices in canonical order.
/// The canonical order is: node names sorted, then site indices per node.
#[test]
fn test_contract_to_tensor_index_ordering() {
    // Create a TreeTN with string node names for predictable ordering
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    // Create site indices with known IDs
    // Site index for node "A" (first in alphabetical order)
    let site_a = DynIndex::new_dyn(2);
    // Site index for node "B" (second in alphabetical order)
    let site_b = DynIndex::new_dyn(3);
    // Bond index
    let bond = DynIndex::new_dyn(4);

    // Create tensors - note: add "B" first to test that result is reordered correctly
    let tensor_b =
        TensorDynLen::from_dense(vec![bond.clone(), site_b.clone()], vec![1.0; 12]).unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    let tensor_a =
        TensorDynLen::from_dense(vec![site_a.clone(), bond.clone()], vec![1.0; 8]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Connect A and B
    let node_a = tn.node_index(&"A".to_string()).unwrap();
    let node_b = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(node_a, &bond, node_b, &bond).unwrap();

    // Contract to tensor
    let result = tn.contract_to_tensor().unwrap();

    // Result should have indices in canonical order: site_a (from "A"), then site_b (from "B")
    let result_indices = result.external_indices();
    assert_eq!(result_indices.len(), 2);
    assert_eq!(result_indices[0].id(), site_a.id());
    assert_eq!(result_indices[1].id(), site_b.id());

    // Dimensions should match: [2, 3]
    assert_eq!(result.dims(), vec![2, 3]);
}

/// Test contract_to_tensor index ordering with reverse alphabetical node names.
#[test]
fn test_contract_to_tensor_index_ordering_reverse() {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    // Site indices
    let site_z = DynIndex::new_dyn(4); // Node "Z"
    let site_a = DynIndex::new_dyn(5); // Node "A"
    let bond = DynIndex::new_dyn(3);

    // Add "Z" first
    let tensor_z =
        TensorDynLen::from_dense(vec![site_z.clone(), bond.clone()], vec![1.0; 12]).unwrap();
    tn.add_tensor("Z".to_string(), tensor_z).unwrap();

    // Add "A" second
    let tensor_a =
        TensorDynLen::from_dense(vec![bond.clone(), site_a.clone()], vec![1.0; 15]).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Connect
    let node_z = tn.node_index(&"Z".to_string()).unwrap();
    let node_a = tn.node_index(&"A".to_string()).unwrap();
    tn.connect(node_z, &bond, node_a, &bond).unwrap();

    // Contract
    let result = tn.contract_to_tensor().unwrap();

    // Canonical order: "A" comes before "Z", so site_a should be first
    let result_indices = result.external_indices();
    assert_eq!(result_indices.len(), 2);
    assert_eq!(result_indices[0].id(), site_a.id());
    assert_eq!(result_indices[1].id(), site_z.id());

    // Dimensions should be [5, 4] (site_a dim=5, site_z dim=4)
    assert_eq!(result.dims(), vec![5, 4]);
}

// ============================================================================
// Log Norm Tests
// ============================================================================

#[test]
fn test_log_norm_simple() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);

    let data1: Vec<f64> = vec![1.0; 6];
    let tensor1 = TensorDynLen::from_dense(vec![i.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let data2: Vec<f64> = vec![1.0; 12];
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    tn.connect(n1, &bond, n2, &bond).unwrap();

    let log_norm = tn.log_norm().unwrap();
    assert!(log_norm.is_finite());
}

// ============================================================================
// Truncation Tests
// ============================================================================

#[test]
fn test_truncate_simple() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(10); // Large bond
    let k = DynIndex::new_dyn(4);

    // Create tensors with rank-deficient data (effectively rank 2)
    // tensor1[i, bond] = data[i * 10 + bond], only bond=0,1 are nonzero
    let mut data1 = vec![0.0f64; 20];
    for idx in 0..2 {
        data1[idx * 10] = 1.0;
        data1[idx * 10 + 1] = 0.5;
    }
    let tensor1 = TensorDynLen::from_dense(vec![i.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    // tensor2[bond, k] similar structure
    let mut data2 = vec![0.0f64; 40];
    for k_idx in 0..4 {
        data2[k_idx] = 1.0;
        data2[4 + k_idx] = 0.3;
    }
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    let edge = tn.connect(n1, &bond, n2, &bond).unwrap();

    // Original bond dimension is 10
    assert_eq!(tn.bond_index(edge).unwrap().size(), 10);

    // Truncate with max_bond_dim = 3
    let truncated = tn
        .truncate(
            std::iter::once(n2),
            TruncationOptions::default().with_max_bond_dim(3),
        )
        .unwrap();

    // Bond should now be at most 3
    let new_edge = truncated.edges_for_node(n1)[0].0;
    assert!(truncated.bond_index(new_edge).unwrap().size() <= 3);
}

#[test]
fn test_truncate_mut_simple() {
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(8);
    let k = DynIndex::new_dyn(4);

    let data1: Vec<f64> = (0..16).map(|x| x as f64).collect();
    let tensor1 = TensorDynLen::from_dense(vec![i.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let data2: Vec<f64> = (0..32).map(|x| x as f64).collect();
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    let edge = tn.connect(n1, &bond, n2, &bond).unwrap();

    // Original bond dimension
    assert_eq!(tn.bond_index(edge).unwrap().size(), 8);

    // Truncate in-place with max_bond_dim = 4
    tn.truncate_mut(
        std::iter::once(n2),
        TruncationOptions::default().with_max_bond_dim(4),
    )
    .unwrap();

    // Bond should now be at most 4
    let new_edge = tn.edges_for_node(n1)[0].0;
    assert!(tn.bond_index(new_edge).unwrap().size() <= 4);
}

#[test]
fn test_truncate_three_node_chain() {
    let (tn, n1, _n2, n3, e12, e23, _b12, _b23) = create_three_node_chain();

    // Original bond dimensions
    assert_eq!(tn.bond_index(e12).unwrap().size(), 3);
    assert_eq!(tn.bond_index(e23).unwrap().size(), 4);

    // Truncate towards center (n1) with max_bond_dim = 2
    let truncated = tn
        .truncate(
            std::iter::once(n1),
            TruncationOptions::default().with_max_bond_dim(2),
        )
        .unwrap();

    // All bonds should now be at most 2
    for (edge, _) in truncated.edges_for_node(n1) {
        assert!(truncated.bond_index(edge).unwrap().size() <= 2);
    }
    for (edge, _) in truncated.edges_for_node(n3) {
        assert!(truncated.bond_index(edge).unwrap().size() <= 2);
    }
}

#[test]
fn test_truncate_with_svd_policy() {
    use tensor4all_core::SvdTruncationPolicy;

    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let i = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(5);
    let k = DynIndex::new_dyn(3);

    // Create a low-rank tensor (rank ~1)
    let mut data1 = vec![0.0f64; 10];
    data1[0] = 1.0;
    data1[5] = 1.0;
    let tensor1 = TensorDynLen::from_dense(vec![i.clone(), bond.clone()], data1).unwrap();
    let n1 = tn.add_tensor_auto_name(tensor1).unwrap();

    let mut data2 = vec![0.0f64; 15];
    data2[0] = 1.0;
    data2[1] = 1.0;
    data2[2] = 1.0;
    let tensor2 = TensorDynLen::from_dense(vec![bond.clone(), k.clone()], data2).unwrap();
    let n2 = tn.add_tensor_auto_name(tensor2).unwrap();

    tn.connect(n1, &bond, n2, &bond).unwrap();

    // Truncate with an explicit SVD policy - should reduce rank for low-rank data
    let truncated = tn
        .truncate(
            std::iter::once(n2),
            TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(1e-10)),
        )
        .unwrap();

    // Bond dimension should be reduced
    let new_edge = truncated.edges_for_node(n1)[0].0;
    let new_bond_size = truncated.bond_index(new_edge).unwrap().size();
    // The effective rank should be 1 or 2
    assert!(
        new_bond_size <= 2,
        "Expected bond_size <= 2, got {}",
        new_bond_size
    );
}

#[test]
fn test_truncate_y_shape() {
    // Create a Y-shape TreeTN:
    //     A     (bond_ab)
    //     |
    //     B     (bond_bc, bond_bd)
    //    / \
    //   C   D
    let mut tn = TreeTN::<TensorDynLen, NodeIndex>::new();

    let site_a = DynIndex::new_dyn(2);
    let bond_ab = DynIndex::new_dyn(6);
    let bond_bc = DynIndex::new_dyn(6);
    let bond_bd = DynIndex::new_dyn(6);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);

    let tensor_a =
        TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], vec![1.0; 12]).unwrap();
    let n_a = tn.add_tensor_auto_name(tensor_a).unwrap();

    let tensor_b = TensorDynLen::from_dense(
        vec![bond_ab.clone(), bond_bc.clone(), bond_bd.clone()],
        vec![1.0; 216],
    )
    .unwrap();
    let n_b = tn.add_tensor_auto_name(tensor_b).unwrap();

    let tensor_c =
        TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], vec![1.0; 12]).unwrap();
    let n_c = tn.add_tensor_auto_name(tensor_c).unwrap();

    let tensor_d =
        TensorDynLen::from_dense(vec![bond_bd.clone(), site_d.clone()], vec![1.0; 12]).unwrap();
    let n_d = tn.add_tensor_auto_name(tensor_d).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();
    tn.connect(n_b, &bond_bd, n_d, &bond_bd).unwrap();

    // All bonds are dimension 6 before truncation
    let e_ab = tn.edge_between(&n_a, &n_b).unwrap();
    let e_bc = tn.edge_between(&n_b, &n_c).unwrap();
    let e_bd = tn.edge_between(&n_b, &n_d).unwrap();
    assert_eq!(tn.bond_index(e_ab).unwrap().size(), 6);
    assert_eq!(tn.bond_index(e_bc).unwrap().size(), 6);
    assert_eq!(tn.bond_index(e_bd).unwrap().size(), 6);

    // Truncate towards center (n_b) with max_bond_dim = 3
    let truncated = tn
        .truncate(
            std::iter::once(n_b),
            TruncationOptions::default().with_max_bond_dim(3),
        )
        .unwrap();

    // All bonds should now be at most 3
    let new_e_ab = truncated.edge_between(&n_a, &n_b).unwrap();
    let new_e_bc = truncated.edge_between(&n_b, &n_c).unwrap();
    let new_e_bd = truncated.edge_between(&n_b, &n_d).unwrap();
    assert!(
        truncated.bond_index(new_e_ab).unwrap().size() <= 3,
        "bond_ab size should be <= 3 after truncation"
    );
    assert!(
        truncated.bond_index(new_e_bc).unwrap().size() <= 3,
        "bond_bc size should be <= 3 after truncation"
    );
    assert!(
        truncated.bond_index(new_e_bd).unwrap().size() <= 3,
        "bond_bd size should be <= 3 after truncation"
    );

    // Structure preserved: still 4 nodes, 3 edges
    assert_eq!(truncated.node_count(), 4);
    assert_eq!(truncated.edge_count(), 3);
}

// ============================================================================
// sim_internal_inds Tests
// ============================================================================

#[test]
fn test_sim_internal_inds() {
    let (tn, node1, node2, _edge, phys1, bond, phys2) = create_two_node_treetn();

    // Get original bond ID
    let original_bond_id = *bond.id();
    let original_bond_size = bond.size();

    // Create a copy with simulated internal indices
    let tn_sim = tn.sim_internal_inds();

    // Check that structure is preserved
    assert_eq!(tn_sim.node_count(), 2);
    assert_eq!(tn_sim.edge_count(), 1);

    // Check that bond index has new ID but same size
    let new_edge = tn_sim.edges_for_node(node1)[0].0;
    let new_bond = tn_sim.bond_index(new_edge).unwrap();
    assert_eq!(new_bond.size(), original_bond_size);
    assert_ne!(
        *new_bond.id(),
        original_bond_id,
        "Bond ID should be different after sim_internal_inds"
    );

    // Check that physical indices are unchanged
    let tensor1 = tn_sim.tensor(node1).unwrap();
    let tensor2 = tn_sim.tensor(node2).unwrap();

    // Find physical indices (those with size 2 and 4)
    let has_phys1 = tensor1.indices.iter().any(|idx| idx.same_id(&phys1));
    let has_phys2 = tensor2.indices.iter().any(|idx| idx.same_id(&phys2));
    assert!(has_phys1, "Physical index 1 should be preserved");
    assert!(has_phys2, "Physical index 2 should be preserved");

    // Check that tensors contain the new bond index
    let has_new_bond_in_t1 = tensor1.indices.iter().any(|idx| idx.same_id(new_bond));
    let has_new_bond_in_t2 = tensor2.indices.iter().any(|idx| idx.same_id(new_bond));
    assert!(has_new_bond_in_t1, "Tensor 1 should have new bond index");
    assert!(has_new_bond_in_t2, "Tensor 2 should have new bond index");

    // Original tensors should still have old bond
    let orig_tensor1 = tn.tensor(node1).unwrap();
    let has_old_bond = orig_tensor1
        .indices
        .iter()
        .any(|idx| *idx.id() == original_bond_id);
    assert!(
        has_old_bond,
        "Original tensor should still have original bond"
    );
}

// ============================================================================
// sim_link_indices Tests
// ============================================================================

#[test]
fn test_sim_link_indices() {
    let (tn, node1, node2, _edge, phys1, bond, phys2) = create_two_node_treetn();

    let original_bond_id = *bond.id();
    let original_bond_size = bond.size();

    // Create a copy with simulated link indices (ITensorMPS-like)
    let tn_sim = tn.sim_link_indices().unwrap();

    // Should remain structurally consistent
    tn_sim.verify_internal_consistency().unwrap();

    // Check that bond index has new ID but same size
    let new_edge = tn_sim.edges_for_node(node1)[0].0;
    let new_bond = tn_sim.bond_index(new_edge).unwrap();
    assert_eq!(new_bond.size(), original_bond_size);
    assert_ne!(
        *new_bond.id(),
        original_bond_id,
        "Bond ID should be different after sim_link_indices"
    );

    // Physical indices are unchanged
    let tensor1 = tn_sim.tensor(node1).unwrap();
    let tensor2 = tn_sim.tensor(node2).unwrap();
    assert!(
        tensor1.indices.iter().any(|idx| idx.same_id(&phys1)),
        "Physical index 1 should be preserved"
    );
    assert!(
        tensor2.indices.iter().any(|idx| idx.same_id(&phys2)),
        "Physical index 2 should be preserved"
    );

    // Tensors contain the new bond index
    assert!(
        tensor1.indices.iter().any(|idx| idx.same_id(new_bond)),
        "Tensor 1 should have new bond index"
    );
    assert!(
        tensor2.indices.iter().any(|idx| idx.same_id(new_bond)),
        "Tensor 2 should have new bond index"
    );
}

#[test]
fn test_sim_link_indices_mut_in_place() {
    let (mut tn, node1, node2, _edge, phys1, bond, phys2) = create_two_node_treetn();

    let original_bond_id = *bond.id();
    let original_bond_size = bond.size();

    tn.sim_link_indices_mut().unwrap();
    tn.verify_internal_consistency().unwrap();

    let new_edge = tn.edges_for_node(node1)[0].0;
    let new_bond = tn.bond_index(new_edge).unwrap();
    assert_eq!(new_bond.size(), original_bond_size);
    assert_ne!(*new_bond.id(), original_bond_id);

    let tensor1 = tn.tensor(node1).unwrap();
    let tensor2 = tn.tensor(node2).unwrap();
    assert!(tensor1.indices.iter().any(|idx| idx.same_id(&phys1)));
    assert!(tensor2.indices.iter().any(|idx| idx.same_id(&phys2)));
    assert!(tensor1.indices.iter().any(|idx| idx.same_id(new_bond)));
    assert!(tensor2.indices.iter().any(|idx| idx.same_id(new_bond)));
}

// ============================================================================
// DynTreeTN Tests (in dyn_treetn module)
// ============================================================================

// DynTreeTN tests are in the dyn_treetn.rs file as unit tests.
// Note: dyn_treetn module has been removed in the index-like branch.

// ============================================================================
// contract_zipup tests
// ============================================================================

// Note: contract_zipup is a complex algorithm that needs more testing.
// The current implementation has known limitations with cases where
// all site indices are contracted (inner product case).
// These tests verify basic functionality.

// NOTE: Tests using random_treetn_f64 and LinkSpace are commented out
// because the random module is currently disabled in the index-like branch.
// To re-enable these tests, uncomment `pub mod random;` in lib.rs and
// uncomment the corresponding `pub use random::...` line.

/*
#[test]
fn test_contract_zipup_basic_api() {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::collections::HashSet;
    use tensor4all_treetn::{random_treetn_f64, LinkSpace, SiteIndexNetwork};

    // Create two networks with the same topology but different site indices
    let mut site_network1 = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a1 = DynIndex::new_dyn(2);
    let site_b1 = DynIndex::new_dyn(3);
    site_network1
        .add_node("A".to_string(), HashSet::from([site_a1.clone()]))
        .unwrap();
    site_network1
        .add_node("B".to_string(), HashSet::from([site_b1.clone()]))
        .unwrap();
    site_network1
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let mut site_network2 = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a2 = DynIndex::new_dyn(2);
    let site_b2 = DynIndex::new_dyn(3);
    site_network2
        .add_node("A".to_string(), HashSet::from([site_a2.clone()]))
        .unwrap();
    site_network2
        .add_node("B".to_string(), HashSet::from([site_b2.clone()]))
        .unwrap();
    site_network2
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    let mut rng1 = ChaCha8Rng::seed_from_u64(42);
    let mut rng2 = ChaCha8Rng::seed_from_u64(123);
    let tn1 = random_treetn::<f64, _, _>(&mut rng1, &site_network1, LinkSpace::uniform(4));
    let tn2 = random_treetn::<f64, _, _>(&mut rng2, &site_network2, LinkSpace::uniform(4));

    // Verify same_topology works
    assert!(tn1.same_topology(&tn2));

    // Note: contract_zipup with non-shared site indices currently has issues
    // because tensordot requires at least one contraction pair.
    // This test verifies the API exists and type checks correctly.
}
*/

/// Test contract_zipup with shared site indices (inner product case).
/// This is the typical use case: computing overlap <ψ|φ>.
#[test]
fn test_contract_zipup_shared_site_indices() {
    // Create shared site indices (same Index objects for both networks)
    let site_a = DynIndex::new_dyn(2); // Physical index at node A
    let site_b = DynIndex::new_dyn(3); // Physical index at node B

    // Create bond indices (different for each network)
    let bond1 = DynIndex::new_dyn(4); // Bond for network 1
    let bond2 = DynIndex::new_dyn(4); // Bond for network 2

    // Create tensor for node A in network 1: A1[site_a, bond1]
    let tensor_a1 =
        TensorDynLen::from_dense(vec![site_a.clone(), bond1.clone()], vec![1.0; 8]).unwrap();

    // Create tensor for node B in network 1: B1[bond1, site_b]
    let tensor_b1 =
        TensorDynLen::from_dense(vec![bond1.clone(), site_b.clone()], vec![1.0; 12]).unwrap();

    // Create TreeTN 1
    let tn1: TreeTN<TensorDynLen, String> = TreeTN::from_tensors(
        vec![tensor_a1, tensor_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Create tensor for node A in network 2: A2[site_a, bond2]
    // Uses the SAME site_a index
    let tensor_a2 =
        TensorDynLen::from_dense(vec![site_a.clone(), bond2.clone()], vec![2.0; 8]).unwrap();

    // Create tensor for node B in network 2: B2[bond2, site_b]
    // Uses the SAME site_b index
    let tensor_b2 =
        TensorDynLen::from_dense(vec![bond2.clone(), site_b.clone()], vec![2.0; 12]).unwrap();

    // Create TreeTN 2
    let tn2: TreeTN<TensorDynLen, String> = TreeTN::from_tensors(
        vec![tensor_a2, tensor_b2],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Verify same topology
    assert!(tn1.same_topology(&tn2));
    assert_eq!(tn1.node_count(), 2);
    assert_eq!(tn2.node_count(), 2);

    // Contract with center at B
    // Since all site indices are contracted (inner product case),
    // all tensors collapse into the center node
    let result = tn1
        .contract_zipup(&tn2, &"B".to_string(), None, None)
        .unwrap();

    // For inner product: all nodes get absorbed into center
    // Result should have 1 node (the center B) containing a scalar
    assert_eq!(result.node_count(), 1);

    // Result should have 0 edges (single node)
    assert_eq!(result.edge_count(), 0);

    // The center node should exist
    assert!(result.node_index(&"B".to_string()).is_some());
}

/// Test contract_zipup with partial site contraction (external indices remain).
/// This simulates computing <ψ|O|φ> where O is an MPO.
#[test]
fn test_contract_zipup_partial_contraction() {
    // Create indices
    // Shared site indices (will be contracted between tn1 and tn2)
    let site_a = DynIndex::new_dyn(2); // Physical index at node A
    let site_b = DynIndex::new_dyn(3); // Physical index at node B

    // External index (exists only in tn2, will remain in result)
    let site_external_a = DynIndex::new_dyn(2); // External index at A (like MPO output)
    let site_external_b = DynIndex::new_dyn(3); // External index at B (like MPO output)

    // Bond indices (different for each network, will be replaced by sim_internal_inds)
    let bond1 = DynIndex::new_dyn(4); // Bond for network 1
    let bond2 = DynIndex::new_dyn(4); // Bond for network 2

    // Network 1 (like bra <ψ|):
    // A1[site_a, bond1] -- B1[bond1, site_b]
    let tensor_a1 =
        TensorDynLen::from_dense(vec![site_a.clone(), bond1.clone()], vec![1.0; 8]).unwrap();

    let tensor_b1 =
        TensorDynLen::from_dense(vec![bond1.clone(), site_b.clone()], vec![1.0; 12]).unwrap();

    let tn1: TreeTN<TensorDynLen, String> = TreeTN::from_tensors(
        vec![tensor_a1, tensor_b1],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Network 2 (like O|φ> with MPO indices):
    // A2[site_a, site_external_a, bond2] -- B2[bond2, site_b, site_external_b]
    // Has both shared and external site indices at each node
    let tensor_a2 = TensorDynLen::from_dense(
        vec![site_a.clone(), site_external_a.clone(), bond2.clone()],
        vec![2.0; 16],
    )
    .unwrap();

    let tensor_b2 = TensorDynLen::from_dense(
        vec![bond2.clone(), site_b.clone(), site_external_b.clone()],
        vec![2.0; 36],
    )
    .unwrap();

    let tn2: TreeTN<TensorDynLen, String> = TreeTN::from_tensors(
        vec![tensor_a2, tensor_b2],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Verify same topology
    assert!(tn1.same_topology(&tn2));

    // Contract with center at B
    let result = tn1
        .contract_zipup(&tn2, &"B".to_string(), None, None)
        .unwrap();

    // Result should preserve structure because both nodes have external indices
    assert_eq!(result.node_count(), 2);
    assert_eq!(result.edge_count(), 1);

    // Check that node A exists and has the external index
    let node_a = result.node_index(&"A".to_string()).unwrap();
    let tensor_a = result.tensor(node_a).unwrap();
    assert!(tensor_a
        .indices
        .iter()
        .any(|idx| idx.same_id(&site_external_a)));

    // Check that node B exists and has the external index
    let node_b = result.node_index(&"B".to_string()).unwrap();
    let tensor_b = result.tensor(node_b).unwrap();
    assert!(tensor_b
        .indices
        .iter()
        .any(|idx| idx.same_id(&site_external_b)));
}

// ============================================================================
// Helper function for comparing contract_zipup vs contract_naive
// ============================================================================

// NOTE: The following tests use random_treetn_f64 and LinkSpace which are
// disabled in the index-like branch. Commenting them out for now.

/*
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use tensor4all_treetn::{random_treetn_f64, LinkSpace, SiteIndexNetwork};

/// Helper to test contract_zipup vs contract_naive for a given topology.
///
/// Creates two random TreeTNs with the given site network and link space,
/// contracts them using both methods, and verifies the results match.
///
// ============================================================================
// Contraction Method Testing Helpers
// ============================================================================
use tensor4all_treetn::{contract, ContractionMethod, ContractionOptions};

/// Generic comparison function for contraction methods vs naive
/// # Arguments
/// * `method` - The contraction method to test (Zipup or Fit)
/// * `site_network` - The site index network defining topology and site indices
/// * `link_space` - The bond dimensions
/// * `center` - The center node name
/// * `seed1`, `seed2` - Random seeds for the two networks
/// * `rtol` - Relative tolerance for comparison
fn compare_contract_vs_naive(
    method: ContractionMethod,
    site_network: &SiteIndexNetwork<String, DynIndex>,
    link_space: LinkSpace<String>,
    center: &str,
    seed1: u64,
    seed2: u64,
    rtol: f64,
) {
    // Create two random TreeTNs
    let mut rng1 = ChaCha8Rng::seed_from_u64(seed1);
    let mut rng2 = ChaCha8Rng::seed_from_u64(seed2);
    let tn1 = random_treetn::<f64, _, _>(&mut rng1, site_network, link_space.clone()).unwrap();
    let tn2 = random_treetn::<f64, _, _>(&mut rng2, site_network, link_space).unwrap();

    // Contract using naive method (reference)
    let naive_result = tn1
        .contract_naive(&tn2)
        .expect("contract_naive should succeed");

    // Contract using the specified method via dispatcher
    let options = ContractionOptions::new(method);
    let result =
        contract(&tn1, &tn2, &center.to_string(), options).expect("contract should succeed");

    // Convert result to tensor for comparison
    let result_tensor = result
        .contract_to_tensor()
        .expect("contract_to_tensor should succeed");

    // Compare using distance
    let dist = naive_result.distance(&result_tensor);
    assert!(
        dist < rtol,
        "{:?} vs contract_naive distance too large: {} >= {} (center={})",
        method,
        dist,
        rtol,
        center
    );
}

/// Wrapper for zipup vs naive
fn compare_zipup_vs_naive(
    site_network: &SiteIndexNetwork<String, DynIndex>,
    link_space: LinkSpace<String>,
    center: &str,
    seed1: u64,
    seed2: u64,
    rtol: f64,
) {
    compare_contract_vs_naive(
        ContractionMethod::Zipup,
        site_network,
        link_space,
        center,
        seed1,
        seed2,
        rtol,
    );
}

/// Wrapper for fit vs naive
fn compare_fit_vs_naive(
    site_network: &SiteIndexNetwork<String, DynIndex>,
    link_space: LinkSpace<String>,
    center: &str,
    seed1: u64,
    seed2: u64,
    rtol: f64,
) {
    compare_contract_vs_naive(
        ContractionMethod::Fit,
        site_network,
        link_space,
        center,
        seed1,
        seed2,
        rtol,
    );
}

// ============================================================================
// Topology tests using helper function
// ============================================================================

/// Test contract_zipup vs contract_naive for 2-node chain: A -- B
#[test]
fn test_zipup_vs_naive_2node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(3);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(4), "B", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for 3-node chain: A -- B -- C
#[test]
fn test_zipup_vs_naive_3node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();

    // Test with center at B (middle)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);

    // Test with center at C (other end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "C", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for star topology:
///       A
///       |
///   C - B - D
#[test]
fn test_zipup_vs_naive_star() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_node("D".to_string(), HashSet::from([site_d]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"D".to_string())
        .unwrap();

    // Test with center at B (hub)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (leaf)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for Y-shaped topology:
///   A - B - C
///       |
///       D
#[test]
fn test_zipup_vs_naive_y_shape() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_node("D".to_string(), HashSet::from([site_d]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"D".to_string())
        .unwrap();

    // Test with center at B
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at D
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(3), "D", 42, 123, 1e-10);
}

/// Test contract_zipup vs contract_naive for 5-node chain: A -- B -- C -- D -- E
#[test]
fn test_zipup_vs_naive_5node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    for name in ["A", "B", "C", "D", "E"] {
        let site = DynIndex::new_dyn(2);
        site_network
            .add_node(name.to_string(), HashSet::from([site]))
            .unwrap();
    }
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"C".to_string(), &"D".to_string())
        .unwrap();
    site_network
        .add_edge(&"D".to_string(), &"E".to_string())
        .unwrap();

    // Test with center at C (middle)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(2), "C", 42, 123, 1e-10);

    // Test with center at E (end)
    compare_zipup_vs_naive(&site_network, LinkSpace::uniform(2), "E", 42, 123, 1e-10);
}
*/

// ============================================================================
// Internal Consistency Tests
// ============================================================================

#[test]
fn test_verify_internal_consistency_basic() {
    // Create a simple 2-node TreeTN and verify its consistency
    let (tn, _n1, _n2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();

    // Should pass consistency check
    tn.verify_internal_consistency()
        .expect("Consistency check should pass for basic TreeTN");
}

#[test]
fn test_verify_internal_consistency_chain_3node() {
    // Create 3-node chain
    let (tn, _n1, _n2, _n3, _e12, _e23, _bond12, _bond23) = create_three_node_chain();

    // Should pass consistency check
    tn.verify_internal_consistency()
        .expect("Consistency check should pass for 3-node chain");
}

#[test]
fn test_verify_internal_consistency_after_canonicalization() {
    let (tn, _n1, n2, _n3, _e12, _e23, _bond12, _bond23) = create_three_node_chain();

    // Canonicalize to center n2
    let tn = tn
        .canonicalize(
            vec![n2],
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .expect("Canonicalization should succeed");

    // Should still pass consistency check after canonicalization
    tn.verify_internal_consistency()
        .expect("Consistency check should pass after canonicalization");
}

#[test]
fn test_verify_internal_consistency_with_string_node_names() {
    // Create TreeTN with string node names
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    let tensor_a =
        TensorDynLen::from_dense(vec![site_a.clone(), bond.clone()], vec![1.0; 8]).unwrap();

    let tensor_b =
        TensorDynLen::from_dense(vec![bond.clone(), site_b.clone()], vec![1.0; 12]).unwrap();

    tn.add_tensor("A".to_string(), tensor_a).unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();
    tn.connect(
        tn.node_index(&"A".to_string()).unwrap(),
        &bond,
        tn.node_index(&"B".to_string()).unwrap(),
        &bond,
    )
    .unwrap();

    // Should pass consistency check
    tn.verify_internal_consistency()
        .expect("Consistency check should pass for string-named TreeTN");
}

// ============================================================================
// Appearance Comparison Tests
// ============================================================================

#[test]
fn test_same_appearance_cloned_treetnns() {
    // Create a TreeTN and clone it - should have same appearance
    let (tn1, _n1, _n2, _edge, _phys1, _bond, _phys2) = create_two_node_treetn();
    let tn2 = tn1.clone();

    // Should have same appearance (both have no ortho_towards set)
    assert!(tn1.same_appearance(&tn2));
    assert!(tn2.same_appearance(&tn1)); // Symmetric
}

#[test]
fn test_same_appearance_after_same_canonicalization() {
    // Create a TreeTN, clone, and canonicalize both to the same center
    let (tn1, _n1, n2, _n3, _, _, _, _) = create_three_node_chain();
    let tn2 = tn1.clone();

    let tn1 = tn1
        .canonicalize(
            vec![n2],
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap();

    let tn2 = tn2
        .canonicalize(
            vec![n2],
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap();

    // Both canonicalized to same center - should have same appearance
    assert!(tn1.same_appearance(&tn2));
}

#[test]
fn test_same_appearance_different_ortho_towards() {
    // Create a TreeTN, clone, and canonicalize to different centers
    let (tn1, n1, _n2, n3, _, _, _, _) = create_three_node_chain();
    let tn2 = tn1.clone();

    let tn1 = tn1
        .canonicalize(
            vec![n1], // Canonicalize to left
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap();

    let tn2 = tn2
        .canonicalize(
            vec![n3], // Canonicalize to right
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap();

    // Different canonical centers - different ortho_towards directions
    // Note: They still share equivalent site index network (same topology and site space)
    assert!(tn1.share_equivalent_site_index_network(&tn2));
    // But different appearance due to ortho_towards
    assert!(!tn1.same_appearance(&tn2));
}

#[test]
fn test_same_appearance_one_canonicalized_one_not() {
    // Create a TreeTN, clone, and only canonicalize one
    let (tn1, _n1, n2, _n3, _, _, _, _) = create_three_node_chain();
    let tn2 = tn1.clone();

    let tn1 = tn1
        .canonicalize(
            vec![n2],
            tensor4all_treetn::CanonicalizationOptions::default(),
        )
        .unwrap();

    // tn2 is not canonicalized - different ortho_towards state
    assert!(tn1.share_equivalent_site_index_network(&tn2));
    assert!(!tn1.same_appearance(&tn2));
}

// ============================================================================
// Fit Algorithm Tests
// ============================================================================

// NOTE: FitContractionOptions may not be available in the index-like branch.
// Commenting out until confirmed.

/*
use tensor4all_treetn::FitContractionOptions;

/// Create a simple MPS-like chain for fit testing: A - B - C
/// Each node has physical index (site) and bond indices.
#[allow(dead_code)]
fn create_mps_chain_for_fit() -> TreeTN<TensorDynLen, String> {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    // Physical indices
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);

    // Bond indices
    let bond_ab = DynIndex::new_dyn(3);
    let bond_bc = DynIndex::new_dyn(3);

    // Tensor A: [site_a, bond_ab]
    let data_a: Vec<f64> = (0..6).map(|x| (x as f64 + 1.0) / 10.0).collect();
    let tensor_a = TensorDynLen::from_dense(vec![site_a.clone(), bond_ab.clone()], data_a).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Tensor B: [bond_ab, site_b, bond_bc]
    let data_b: Vec<f64> = (0..18).map(|x| (x as f64 + 1.0) / 20.0).collect();
    let tensor_b = TensorDynLen::from_dense(vec![bond_ab.clone(), site_b.clone(), bond_bc.clone()], data_b).unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Tensor C: [bond_bc, site_c]
    let data_c: Vec<f64> = (0..6).map(|x| (x as f64 + 1.0) / 10.0).collect();
    let tensor_c = TensorDynLen::from_dense(vec![bond_bc.clone(), site_c.clone()], data_c).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Connect
    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();

    // Set site spaces
    if let Some(ss) = tn.site_space_mut(&"A".to_string()) {
        ss.insert(site_a);
    }
    if let Some(ss) = tn.site_space_mut(&"B".to_string()) {
        ss.insert(site_b);
    }
    if let Some(ss) = tn.site_space_mut(&"C".to_string()) {
        ss.insert(site_c);
    }

    tn
}

/// Create an MPO-like chain (same topology as MPS but with two physical indices per site)
#[allow(dead_code)]
fn create_mpo_chain_for_fit() -> TreeTN<TensorDynLen, String> {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    // Physical indices (input and output for each site)
    let site_a_in = DynIndex::new_dyn(2);
    let site_a_out = DynIndex::new_dyn(2);
    let site_b_in = DynIndex::new_dyn(2);
    let site_b_out = DynIndex::new_dyn(2);
    let site_c_in = DynIndex::new_dyn(2);
    let site_c_out = DynIndex::new_dyn(2);

    // Bond indices
    let bond_ab = DynIndex::new_dyn(2);
    let bond_bc = DynIndex::new_dyn(2);

    // Tensor A: [site_a_in, site_a_out, bond_ab]
    let data_a: Vec<f64> = (0..8).map(|x| (x as f64 + 1.0) / 10.0).collect();
    let tensor_a = TensorDynLen::from_dense(vec![site_a_in.clone(), site_a_out.clone(), bond_ab.clone()], data_a).unwrap();
    tn.add_tensor("A".to_string(), tensor_a).unwrap();

    // Tensor B: [bond_ab, site_b_in, site_b_out, bond_bc]
    let data_b: Vec<f64> = (0..16).map(|x| (x as f64 + 1.0) / 20.0).collect();
    let tensor_b = TensorDynLen::from_dense(vec![
            bond_ab.clone(),
            site_b_in.clone(),
            site_b_out.clone(),
            bond_bc.clone(),
        ], data_b).unwrap();
    tn.add_tensor("B".to_string(), tensor_b).unwrap();

    // Tensor C: [bond_bc, site_c_in, site_c_out]
    let data_c: Vec<f64> = (0..8).map(|x| (x as f64 + 1.0) / 10.0).collect();
    let tensor_c = TensorDynLen::from_dense(vec![bond_bc.clone(), site_c_in.clone(), site_c_out.clone()], data_c).unwrap();
    tn.add_tensor("C".to_string(), tensor_c).unwrap();

    // Connect
    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    let n_c = tn.node_index(&"C".to_string()).unwrap();

    tn.connect(n_a, &bond_ab, n_b, &bond_ab).unwrap();
    tn.connect(n_b, &bond_bc, n_c, &bond_bc).unwrap();

    // Set site spaces (both in and out are site indices for MPO)
    if let Some(ss) = tn.site_space_mut(&"A".to_string()) {
        ss.insert(site_a_in);
        ss.insert(site_a_out);
    }
    if let Some(ss) = tn.site_space_mut(&"B".to_string()) {
        ss.insert(site_b_in);
        ss.insert(site_b_out);
    }
    if let Some(ss) = tn.site_space_mut(&"C".to_string()) {
        ss.insert(site_c_in);
        ss.insert(site_c_out);
    }

    tn
}

#[test]
fn test_fit_contraction_options() {
    use tensor4all_core::SvdTruncationPolicy;

    let policy = SvdTruncationPolicy::new(1e-8)
        .with_squared_values()
        .with_discarded_tail_sum();
    let options = FitContractionOptions::new(5)
        .with_max_bond_dim(10)
        .with_svd_policy(policy);

    assert_eq!(options.nfullsweeps, 5);
    assert_eq!(options.max_bond_dim, Some(10));
    assert_eq!(options.svd_policy, Some(policy));
    assert_eq!(options.qr_rtol, None);
}

// Note: test_fit_environment_basic was removed because FitEnvironment now uses
// lazy evaluation with private insert method. The functionality is tested
// through the fit vs naive topology tests below.

// ============================================================================
// Fit vs Naive Topology Tests (same pattern as zipup tests)
// ============================================================================

/// Test contract_fit vs contract_naive for 2-node chain: A -- B
#[test]
fn test_fit_vs_naive_2node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(3);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    compare_fit_vs_naive(&site_network, LinkSpace::uniform(4), "B", 42, 123, 1e-10);
}

/// Test contract_fit vs contract_naive for 3-node chain: A -- B -- C
#[test]
fn test_fit_vs_naive_3node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();

    // Test with center at B (middle)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (end)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);

    // Test with center at C (other end)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "C", 42, 123, 1e-10);
}

/// Test contract_fit vs contract_naive for star topology:
///       A
///       |
///   C - B - D
#[test]
fn test_fit_vs_naive_star() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_node("D".to_string(), HashSet::from([site_d]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"D".to_string())
        .unwrap();

    // Test with center at B (hub)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at A (leaf)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "A", 42, 123, 1e-10);
}

/// Test contract_fit vs contract_naive for Y-shaped topology:
///   A - B - C
///       |
///       D
#[test]
fn test_fit_vs_naive_y_shape() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    let site_a = DynIndex::new_dyn(2);
    let site_b = DynIndex::new_dyn(2);
    let site_c = DynIndex::new_dyn(2);
    let site_d = DynIndex::new_dyn(2);
    site_network
        .add_node("A".to_string(), HashSet::from([site_a]))
        .unwrap();
    site_network
        .add_node("B".to_string(), HashSet::from([site_b]))
        .unwrap();
    site_network
        .add_node("C".to_string(), HashSet::from([site_c]))
        .unwrap();
    site_network
        .add_node("D".to_string(), HashSet::from([site_d]))
        .unwrap();
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"D".to_string())
        .unwrap();

    // Test with center at B
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "B", 42, 123, 1e-10);

    // Test with center at D
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(3), "D", 42, 123, 1e-10);
}

/// Test contract_fit vs contract_naive for 5-node chain: A -- B -- C -- D -- E
#[test]
fn test_fit_vs_naive_5node_chain() {
    let mut site_network = SiteIndexNetwork::<String, DynIndex>::new();
    for name in ["A", "B", "C", "D", "E"] {
        let site = DynIndex::new_dyn(2);
        site_network
            .add_node(name.to_string(), HashSet::from([site]))
            .unwrap();
    }
    site_network
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();
    site_network
        .add_edge(&"B".to_string(), &"C".to_string())
        .unwrap();
    site_network
        .add_edge(&"C".to_string(), &"D".to_string())
        .unwrap();
    site_network
        .add_edge(&"D".to_string(), &"E".to_string())
        .unwrap();

    // Test with center at C (middle)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(2), "C", 42, 123, 1e-10);

    // Test with center at E (end)
    compare_fit_vs_naive(&site_network, LinkSpace::uniform(2), "E", 42, 123, 1e-10);
}
*/

// ============================================================================
// Tests for contract_zipup_with
// ============================================================================

/// Helper: Create a simple 2-node TreeTN with String node names for testing
fn create_two_node_treetn_string() -> TreeTN<TensorDynLen, String> {
    let mut tn = TreeTN::<TensorDynLen, String>::new();

    let phys1 = DynIndex::new_dyn(2);
    let bond = DynIndex::new_dyn(3);
    let phys2 = DynIndex::new_dyn(4);

    let tensor1 =
        TensorDynLen::from_dense(vec![phys1.clone(), bond.clone()], vec![1.0; 6]).unwrap();
    tn.add_tensor("A".to_string(), tensor1).unwrap();

    let tensor2 =
        TensorDynLen::from_dense(vec![bond.clone(), phys2.clone()], vec![1.0; 12]).unwrap();
    tn.add_tensor("B".to_string(), tensor2).unwrap();

    let n_a = tn.node_index(&"A".to_string()).unwrap();
    let n_b = tn.node_index(&"B".to_string()).unwrap();
    tn.connect(n_a, &bond, n_b, &bond).unwrap();

    tn
}

/// Test single node contraction
#[test]
fn test_zipup_with_single_node() {
    let mut tn_a = TreeTN::<TensorDynLen, String>::new();
    let mut tn_b = TreeTN::<TensorDynLen, String>::new();

    // Use the same site index so this is a real contraction, not an implicit
    // outer product of unrelated inputs.
    let phys_a = DynIndex::new_dyn(2);
    let phys_b = phys_a.clone();
    let tensor_a = TensorDynLen::from_dense(vec![phys_a.clone()], vec![1.0, 2.0]).unwrap();
    let tensor_b = TensorDynLen::from_dense(vec![phys_b.clone()], vec![3.0, 4.0]).unwrap();

    tn_a.add_tensor("X".to_string(), tensor_a).unwrap();
    tn_b.add_tensor("X".to_string(), tensor_b).unwrap();

    let result = tn_a
        .contract_zipup_with(&tn_b, &"X".to_string(), CanonicalForm::Unitary, None, None)
        .unwrap();

    assert_eq!(result.node_count(), 1);
    let result_tensor = result
        .tensor(result.node_index(&"X".to_string()).unwrap())
        .unwrap();
    assert!(result_tensor.external_indices().is_empty());
    assert!(result.canonical_region().contains(&"X".to_string()));
}

/// Test 2-node chain contraction
#[test]
fn test_zipup_with_two_node_chain() {
    let tn_a = create_two_node_treetn_string();
    let tn_b = create_two_node_treetn_string();

    let result = tn_a
        .contract_zipup_with(&tn_b, &"B".to_string(), CanonicalForm::Unitary, None, None)
        .unwrap();

    assert_eq!(result.node_count(), 2);
    assert!(result.node_index(&"A".to_string()).is_some());
    assert!(result.node_index(&"B".to_string()).is_some());
    // Check canonical center is set
    assert!(result.canonical_region().contains(&"B".to_string()));
}

/// Test 3-node chain contraction
#[test]
fn test_zipup_with_three_node_chain() {
    let mut tn_a = TreeTN::<TensorDynLen, String>::new();
    let mut tn_b = TreeTN::<TensorDynLen, String>::new();

    let phys1_a = DynIndex::new_dyn(2);
    let bond12_a = DynIndex::new_dyn(3);
    let bond23_a = DynIndex::new_dyn(4);
    let phys3_a = DynIndex::new_dyn(5);

    let phys1_b = DynIndex::new_dyn(2);
    let bond12_b = DynIndex::new_dyn(3);
    let bond23_b = DynIndex::new_dyn(4);
    let phys3_b = DynIndex::new_dyn(5);

    // Create chain: A -- B -- C (tn_a)
    let tensor_a =
        TensorDynLen::from_dense(vec![phys1_a.clone(), bond12_a.clone()], vec![1.0; 6]).unwrap();
    tn_a.add_tensor("A".to_string(), tensor_a).unwrap();

    let tensor_b =
        TensorDynLen::from_dense(vec![bond12_a.clone(), bond23_a.clone()], vec![1.0; 12]).unwrap();
    tn_a.add_tensor("B".to_string(), tensor_b).unwrap();

    let tensor_c =
        TensorDynLen::from_dense(vec![bond23_a.clone(), phys3_a.clone()], vec![1.0; 20]).unwrap();
    tn_a.add_tensor("C".to_string(), tensor_c).unwrap();

    // Create chain: A -- B -- C (tn_b) with distinct site indices
    let tensor_a_b =
        TensorDynLen::from_dense(vec![phys1_b.clone(), bond12_b.clone()], vec![1.0; 6]).unwrap();
    tn_b.add_tensor("A".to_string(), tensor_a_b).unwrap();

    let tensor_b_b =
        TensorDynLen::from_dense(vec![bond12_b.clone(), bond23_b.clone()], vec![1.0; 12]).unwrap();
    tn_b.add_tensor("B".to_string(), tensor_b_b).unwrap();

    let tensor_c_b =
        TensorDynLen::from_dense(vec![bond23_b.clone(), phys3_b.clone()], vec![1.0; 20]).unwrap();
    tn_b.add_tensor("C".to_string(), tensor_c_b).unwrap();

    // Connect nodes
    let n_a_a = tn_a.node_index(&"A".to_string()).unwrap();
    let n_b_a = tn_a.node_index(&"B".to_string()).unwrap();
    let n_c_a = tn_a.node_index(&"C".to_string()).unwrap();
    tn_a.connect(n_a_a, &bond12_a, n_b_a, &bond12_a).unwrap();
    tn_a.connect(n_b_a, &bond23_a, n_c_a, &bond23_a).unwrap();

    let n_a_b = tn_b.node_index(&"A".to_string()).unwrap();
    let n_b_b = tn_b.node_index(&"B".to_string()).unwrap();
    let n_c_b = tn_b.node_index(&"C".to_string()).unwrap();
    tn_b.connect(n_a_b, &bond12_b, n_b_b, &bond12_b).unwrap();
    tn_b.connect(n_b_b, &bond23_b, n_c_b, &bond23_b).unwrap();

    let result = tn_a
        .contract_zipup_with(&tn_b, &"C".to_string(), CanonicalForm::Unitary, None, None)
        .unwrap();

    assert_eq!(result.node_count(), 3);
    assert!(result.canonical_region().contains(&"C".to_string()));
}

/// Test star topology (multiple leaves connected to root)
#[test]
fn test_zipup_with_star_topology() {
    let mut tn_a = TreeTN::<TensorDynLen, String>::new();
    let mut tn_b = TreeTN::<TensorDynLen, String>::new();

    // Create star: A, B, C all connected to center D
    let phys_a = DynIndex::new_dyn(2);
    let phys_b = DynIndex::new_dyn(2);
    let phys_c = DynIndex::new_dyn(2);
    let phys_d = DynIndex::new_dyn(2);
    let bond_ad = DynIndex::new_dyn(3);
    let bond_bd = DynIndex::new_dyn(3);
    let bond_cd = DynIndex::new_dyn(3);

    // Distinct indices for tn_b to avoid contracting site indices
    let phys_a_b = DynIndex::new_dyn(2);
    let phys_b_b = DynIndex::new_dyn(2);
    let phys_c_b = DynIndex::new_dyn(2);
    let phys_d_b = DynIndex::new_dyn(2);
    let bond_ad_b = DynIndex::new_dyn(3);
    let bond_bd_b = DynIndex::new_dyn(3);
    let bond_cd_b = DynIndex::new_dyn(3);

    // Node A
    let tensor_a =
        TensorDynLen::from_dense(vec![phys_a.clone(), bond_ad.clone()], vec![1.0; 6]).unwrap();
    tn_a.add_tensor("A".to_string(), tensor_a.clone()).unwrap();
    let tensor_a_b =
        TensorDynLen::from_dense(vec![phys_a_b.clone(), bond_ad_b.clone()], vec![1.0; 6]).unwrap();
    tn_b.add_tensor("A".to_string(), tensor_a_b).unwrap();

    // Node B
    let tensor_b =
        TensorDynLen::from_dense(vec![phys_b.clone(), bond_bd.clone()], vec![1.0; 6]).unwrap();
    tn_a.add_tensor("B".to_string(), tensor_b.clone()).unwrap();
    let tensor_b_b =
        TensorDynLen::from_dense(vec![phys_b_b.clone(), bond_bd_b.clone()], vec![1.0; 6]).unwrap();
    tn_b.add_tensor("B".to_string(), tensor_b_b).unwrap();

    // Node C
    let tensor_c =
        TensorDynLen::from_dense(vec![phys_c.clone(), bond_cd.clone()], vec![1.0; 6]).unwrap();
    tn_a.add_tensor("C".to_string(), tensor_c.clone()).unwrap();
    let tensor_c_b =
        TensorDynLen::from_dense(vec![phys_c_b.clone(), bond_cd_b.clone()], vec![1.0; 6]).unwrap();
    tn_b.add_tensor("C".to_string(), tensor_c_b).unwrap();

    // Node D (center)
    let tensor_d = TensorDynLen::from_dense(
        vec![
            bond_ad.clone(),
            bond_bd.clone(),
            bond_cd.clone(),
            phys_d.clone(),
        ],
        vec![1.0; 54],
    )
    .unwrap();
    tn_a.add_tensor("D".to_string(), tensor_d.clone()).unwrap();
    let tensor_d_b = TensorDynLen::from_dense(
        vec![
            bond_ad_b.clone(),
            bond_bd_b.clone(),
            bond_cd_b.clone(),
            phys_d_b.clone(),
        ],
        vec![1.0; 54],
    )
    .unwrap();
    tn_b.add_tensor("D".to_string(), tensor_d_b).unwrap();

    // Connect
    let n_a_a = tn_a.node_index(&"A".to_string()).unwrap();
    let n_b_a = tn_a.node_index(&"B".to_string()).unwrap();
    let n_c_a = tn_a.node_index(&"C".to_string()).unwrap();
    let n_d_a = tn_a.node_index(&"D".to_string()).unwrap();
    tn_a.connect(n_a_a, &bond_ad, n_d_a, &bond_ad).unwrap();
    tn_a.connect(n_b_a, &bond_bd, n_d_a, &bond_bd).unwrap();
    tn_a.connect(n_c_a, &bond_cd, n_d_a, &bond_cd).unwrap();

    let n_a_b = tn_b.node_index(&"A".to_string()).unwrap();
    let n_b_b = tn_b.node_index(&"B".to_string()).unwrap();
    let n_c_b = tn_b.node_index(&"C".to_string()).unwrap();
    let n_d_b = tn_b.node_index(&"D".to_string()).unwrap();
    tn_b.connect(n_a_b, &bond_ad_b, n_d_b, &bond_ad_b).unwrap();
    tn_b.connect(n_b_b, &bond_bd_b, n_d_b, &bond_bd_b).unwrap();
    tn_b.connect(n_c_b, &bond_cd_b, n_d_b, &bond_cd_b).unwrap();

    let result = tn_a
        .contract_zipup_with(&tn_b, &"D".to_string(), CanonicalForm::Unitary, None, None)
        .unwrap();

    assert_eq!(result.node_count(), 4);
    assert!(result.canonical_region().contains(&"D".to_string()));
}
