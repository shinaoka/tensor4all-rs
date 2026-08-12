//! Tests for TreeTN operations.
//!
//! Note: TreeTN does NOT implement TensorLike (see TENSOR_LIKE_DESIGN.md).
//! This file tests TreeTN's own API which provides similar functionality:
//! - `site_index_network()` instead of `external_indices()`
//! - `contract_to_tensor()` instead of `to_tensor()`
//!
//! Migrated from tensor4all-rs main branch with type adjustments:
//! - TreeTN<DynId, NoSymmSpace, String> -> TreeTN<IdxTensor, String>
//! - Index<DynId> -> DynIndex
//! - IdxTensor<DynId, NoSymmSpace> -> IdxTensor
//! - idx.id -> idx.id() (IndexLike trait method)

use tensor4all_core::{DynIndex, IdxTensor, IndexLike, TensorIndex};
use tensor4all_treetn::TreeTN;

/// Helper to create a simple tensor with given indices
fn make_tensor(indices: Vec<DynIndex>) -> IdxTensor {
    let dims: Vec<usize> = indices.iter().map(|idx| idx.dim()).collect();
    let total_size: usize = dims.iter().product();
    let data: Vec<f64> = (0..total_size).map(|i| i as f64).collect();
    IdxTensor::from_dense(indices, data).unwrap()
}

/// Helper to collect all site (physical) indices from a TreeTN.
/// This corresponds to `external_indices()` in the original TensorLike trait.
fn collect_site_indices(tn: &TreeTN<IdxTensor, String>) -> Vec<DynIndex> {
    let mut indices = Vec::new();
    for node_name in tn.node_names() {
        if let Some(site_space) = tn.site_space(&node_name) {
            indices.extend(site_space.iter().cloned());
        }
    }
    indices
}

#[test]
fn test_treetn_site_indices_single_node() {
    // Create a TreeTN with a single node containing a 2x3 tensor
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let tensor = make_tensor(vec![i.clone(), j.clone()]);

    // Use from_tensors to create the network
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    // Site indices should be all indices (since no connections)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);

    // Check that the indices have the correct dimensions
    let sizes: Vec<usize> = site_indices.iter().map(|idx| idx.dim()).collect();
    assert!(sizes.contains(&2));
    assert!(sizes.contains(&3));
}

#[test]
fn test_treetn_site_indices_connected_nodes() {
    // Create a TreeTN with two connected nodes
    // Node A: indices (i, bond_ab)
    // Node B: indices (bond_ab, j)
    // After connecting, site indices should be (i, j) - not bond_ab

    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond_ab = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond_ab.clone()]);
    let tensor_b = make_tensor(vec![bond_ab.clone(), j.clone()]);

    // from_tensors automatically connects tensors that share common indices
    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Site indices should be i and j (not bond_ab)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);

    // Verify the site indices are i and j (not bond)
    let site_ids: Vec<_> = site_indices.iter().map(|idx| idx.id()).collect();
    assert!(site_ids.contains(&i.id()));
    assert!(site_ids.contains(&j.id()));
    assert!(!site_ids.contains(&bond_ab.id()));
}

#[test]
fn test_treetn_num_site_indices() {
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
    let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 2);
}

#[test]
fn test_treetn_contract_to_tensor() {
    // Create a simple MPS-like network: A -- B
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(2);

    // Tensor A: 2x2 with values
    let tensor_a =
        IdxTensor::from_dense(vec![i.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap(); // identity-like

    // Tensor B: 2x3 with values
    let tensor_b = IdxTensor::from_dense(
        vec![bond.clone(), j.clone()],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Contract to tensor using the explicit method (exponential cost!)
    let result = tn
        .contract_to_tensor()
        .expect("contract_to_tensor should succeed");

    // Result should be 2x3 (dimensions of i and j)
    assert_eq!(result.external_indices().len(), 2);
    let dims: Vec<usize> = result
        .external_indices()
        .iter()
        .map(|idx| idx.dim())
        .collect();
    assert!(dims.contains(&2));
    assert!(dims.contains(&3));
}

#[test]
fn test_treetn_site_indices_deterministic_ordering() {
    // Create a TreeTN with multiple connected nodes and verify ordering is deterministic
    // Structure: C -- A -- B (linear chain)
    let idx_a = DynIndex::new_dyn(2); // site index for A
    let idx_b = DynIndex::new_dyn(3); // site index for B
    let idx_c = DynIndex::new_dyn(4); // site index for C
    let bond_ca = DynIndex::new_dyn(2); // bond between C and A
    let bond_ab = DynIndex::new_dyn(2); // bond between A and B

    // Add nodes in non-alphabetical order (C, A, B)
    // C has site index idx_c and bond to A
    // A has site index idx_a and bonds to C and B
    // B has site index idx_b and bond to A
    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![
            make_tensor(vec![idx_c.clone(), bond_ca.clone()]),
            make_tensor(vec![bond_ca.clone(), idx_a.clone(), bond_ab.clone()]),
            make_tensor(vec![bond_ab.clone(), idx_b.clone()]),
        ],
        vec!["C".to_string(), "A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Site indices - node names are deterministic (from the same TreeTN)
    let site_indices = collect_site_indices(&tn);
    assert_eq!(site_indices.len(), 3);

    // First call
    let ids1: Vec<_> = site_indices.iter().map(|idx| idx.id()).collect();

    // Second call should give same order
    let site_indices2 = collect_site_indices(&tn);
    let ids2: Vec<_> = site_indices2.iter().map(|idx| idx.id()).collect();

    assert_eq!(ids1, ids2, "site_indices should be deterministic");
}

// Note: The following tests from the original file have been removed because
// TreeTN does NOT implement TensorLike in the index-like branch.
// See TENSOR_LIKE_DESIGN.md for the design rationale.
//
// Removed tests:
// - test_treetn_tensor_like_object_safety
// - test_treetn_clone_trait_object
// - test_treetn_as_any
// - test_treetn_downcast_via_trait_object
// - test_mixed_downcast
//
// These tests tested dyn TensorLike functionality which no longer exists.
// For heterogeneous tensor collections, use an enum wrapper instead.

// ============================================================================
// TensorIndex trait tests for TreeTN
// ============================================================================

#[test]
fn test_treetn_external_indices_via_trait() {
    // Test that external_indices() via TensorIndex trait works correctly
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
    let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Use TensorIndex trait method
    let ext_indices = tn.external_indices();
    assert_eq!(ext_indices.len(), 2);

    // Verify that site indices are i and j (not bond)
    let ext_ids: Vec<_> = ext_indices.iter().map(|idx| idx.id()).collect();
    assert!(ext_ids.contains(&i.id()));
    assert!(ext_ids.contains(&j.id()));
    assert!(!ext_ids.contains(&bond.id()));
}

#[test]
fn test_treetn_replaceind_site_index() {
    // Test replacing a site (physical) index
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let bond = DynIndex::new_dyn(4);

    let tensor_a = make_tensor(vec![i.clone(), bond.clone()]);
    let tensor_b = make_tensor(vec![bond.clone(), j.clone()]);

    let tn = TreeTN::<IdxTensor, String>::from_tensors(
        vec![tensor_a, tensor_b],
        vec!["A".to_string(), "B".to_string()],
    )
    .unwrap();

    // Debug: print site indices
    eprintln!(
        "TreeTN external_indices: {:?}",
        tn.external_indices()
            .iter()
            .map(|idx| idx.id())
            .collect::<Vec<_>>()
    );
    eprintln!("i.id() = {:?}", i.id());
    eprintln!("j.id() = {:?}", j.id());
    eprintln!("bond.id() = {:?}", bond.id());

    // Debug: print site_space for node A
    if let Some(space) = tn.site_space(&"A".to_string()) {
        eprintln!(
            "Node A site_space IDs: {:?}",
            space.iter().map(|idx| idx.id()).collect::<Vec<_>>()
        );
        eprintln!("Node A site_space contains(&i): {}", space.contains(&i));
        // Check by iterating
        for idx in space {
            eprintln!(
                "  Checking idx.id() == i.id(): {:?} == {:?} => {}",
                idx.id(),
                i.id(),
                idx.id() == i.id()
            );
            eprintln!("  idx == i: {}", *idx == i);
        }
    }
    if let Some(space) = tn.site_space(&"B".to_string()) {
        eprintln!(
            "Node B site_space IDs: {:?}",
            space.iter().map(|idx| idx.id()).collect::<Vec<_>>()
        );
    }

    // Create a new index with same dimension
    let i_new = DynIndex::new_dyn(2);

    // Debug: clone and check the cloned site_space
    let tn_cloned = tn.clone();
    if let Some(cloned_space) = tn_cloned.site_space(&"A".to_string()) {
        eprintln!(
            "Cloned Node A site_space IDs: {:?}",
            cloned_space.iter().map(|idx| idx.id()).collect::<Vec<_>>()
        );
        eprintln!(
            "Cloned Node A site_space contains(&i): {}",
            cloned_space.contains(&i)
        );
    }

    // Replace i with i_new
    let tn_replaced = tn
        .replaceind(&i, &i_new)
        .expect("replaceind should succeed");

    // Check that the new index is present and old is not
    let ext_indices = tn_replaced.external_indices();
    let ext_ids: Vec<_> = ext_indices.iter().map(|idx| idx.id()).collect();

    assert!(!ext_ids.contains(&i.id()), "Old index should be replaced");
    assert!(ext_ids.contains(&i_new.id()), "New index should be present");
    assert!(ext_ids.contains(&j.id()), "Other indices should remain");
}

#[test]
fn test_treetn_replaceind_not_found() {
    // Test that replaceind fails gracefully for unknown index
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let tensor = make_tensor(vec![i.clone(), j.clone()]);
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    // Try to replace an index that doesn't exist
    let unknown = DynIndex::new_dyn(5);
    let new_idx = DynIndex::new_dyn(5);

    let result = tn.replaceind(&unknown, &new_idx);
    assert!(result.is_err(), "replaceind should fail for unknown index");
}

#[test]
fn test_treetn_replaceind_dimension_mismatch() {
    // Test that replaceind fails when dimensions don't match
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);

    let tensor = make_tensor(vec![i.clone(), j.clone()]);
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    // Try to replace with index of different dimension
    let wrong_size = DynIndex::new_dyn(5);

    let result = tn.replaceind(&i, &wrong_size);
    assert!(
        result.is_err(),
        "replaceind should fail for dimension mismatch"
    );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Index space mismatch"),
        "Error should mention dimension mismatch"
    );
}

#[test]
fn test_treetn_replace_indices_multiple() {
    // Test replacing multiple indices at once
    let i = DynIndex::new_dyn(2);
    let j = DynIndex::new_dyn(3);
    let k = DynIndex::new_dyn(4);

    let tensor = make_tensor(vec![i.clone(), j.clone(), k.clone()]);
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    // Replace i and j with new indices
    let i_new = DynIndex::new_dyn(2);
    let j_new = DynIndex::new_dyn(3);

    let tn_replaced = tn
        .replace_indices(&[i.clone(), j.clone()], &[i_new.clone(), j_new.clone()])
        .expect("replace_indices should succeed");

    let ext_indices = tn_replaced.external_indices();
    let ext_ids: Vec<_> = ext_indices.iter().map(|idx| idx.id()).collect();

    assert!(!ext_ids.contains(&i.id()));
    assert!(!ext_ids.contains(&j.id()));
    assert!(ext_ids.contains(&i_new.id()));
    assert!(ext_ids.contains(&j_new.id()));
    assert!(ext_ids.contains(&k.id())); // k should remain
}

#[test]
fn test_treetn_external_indices_single_node_multiple_indices() {
    // Test external_indices with single node having multiple indices
    let indices: Vec<DynIndex> = (0..5).map(|_| DynIndex::new_dyn(2)).collect();

    let tensor = make_tensor(indices.clone());
    let tn =
        TreeTN::<IdxTensor, String>::from_tensors(vec![tensor], vec!["A".to_string()]).unwrap();

    let ext = tn.external_indices();
    assert_eq!(ext.len(), 5);

    // All original indices should be present
    for idx in &indices {
        assert!(
            ext.iter().any(|e| e.id() == idx.id()),
            "Index {:?} should be in external_indices",
            idx.id()
        );
    }
}
