use tensor4all_treetn::{Connection, TreeTN};
use tensor4all_core::index::{DefaultIndex as Index, DynId};
use tensor4all_tensor::{TensorDynLen, Storage};
use std::sync::Arc;
use petgraph::graph::NodeIndex;

#[test]
fn test_connection_creation() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let conn = Connection::new(idx_a, idx_b);
    assert!(conn.is_ok());
    let conn = conn.unwrap();
    assert_eq!(conn.bond_dim(), 3);
}

#[test]
fn test_connection_dimension_mismatch() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(5);
    
    let conn = Connection::new(idx_a, idx_b);
    assert!(conn.is_err());
    let err_msg = conn.unwrap_err().to_string();
    assert!(err_msg.contains("dimension") || err_msg.contains("mismatch") || err_msg.contains("matching"));
}

#[test]
fn test_connection_replace_bond_indices() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    assert_eq!(conn.bond_dim(), 3);
    
    let new_idx_a = Index::new_dyn(5);
    let new_idx_b = Index::new_dyn(5);
    
    let result = conn.replace_bond_indices(new_idx_a, new_idx_b);
    assert!(result.is_ok());
    assert_eq!(conn.bond_dim(), 5);
}

#[test]
fn test_connection_replace_bond_indices_mismatch() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    
    let new_idx_a = Index::new_dyn(5);
    let new_idx_b = Index::new_dyn(7);
    
    let result = conn.replace_bond_indices(new_idx_a, new_idx_b);
    assert!(result.is_err());
}

#[test]
fn test_connection_ortho_towards() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    
    let mut conn = Connection::new(idx_a.clone(), idx_b.clone()).unwrap();
    assert!(conn.ortho_towards.is_none());
    
    // Set ortho_towards to idx_a
    let result = conn.set_ortho_towards(Some(idx_a.clone()));
    assert!(result.is_ok());
    assert_eq!(conn.ortho_towards.as_ref().map(|idx| &idx.id), Some(&idx_a.id));
    
    // Set ortho_towards to idx_b
    let result = conn.set_ortho_towards(Some(idx_b.clone()));
    assert!(result.is_ok());
    assert_eq!(conn.ortho_towards.as_ref().map(|idx| &idx.id), Some(&idx_b.id));
    
    // Clear ortho_towards
    let result = conn.set_ortho_towards(None);
    assert!(result.is_ok());
    assert!(conn.ortho_towards.is_none());
}

#[test]
fn test_connection_ortho_towards_invalid() {
    let idx_a = Index::new_dyn(3);
    let idx_b = Index::new_dyn(3);
    let idx_c = Index::new_dyn(3); // Different ID
    
    let mut conn = Connection::new(idx_a, idx_b).unwrap();
    
    // Try to set ortho_towards to an index that's not in the connection
    let result = conn.set_ortho_towards(Some(idx_c));
    assert!(result.is_err());
}

#[test]
fn test_treetn_add_tensor() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    
    let node = tn.add_tensor(tensor);
    assert_eq!(tn.node_count(), 1);
    
    let retrieved = tn.tensor(node);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().dims, vec![2, 3]);
}

#[test]
fn test_treetn_replace_tensor() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    
    let node = tn.add_tensor(tensor);
    assert_eq!(tn.tensor(node).unwrap().dims, vec![2, 3]);
    
    // Replace with a new tensor
    let new_i = Index::new_dyn(4);
    let new_j = Index::new_dyn(5);
    let new_indices = vec![new_i, new_j];
    let new_dims = vec![4, 5];
    let new_storage = Arc::new(Storage::new_dense_f64(20));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let old_tensor_result = tn.replace_tensor(node, new_tensor);
    assert!(old_tensor_result.is_ok());
    let old_tensor = old_tensor_result.unwrap();
    assert!(old_tensor.is_some());
    assert_eq!(old_tensor.unwrap().dims, vec![2, 3]);
    assert_eq!(tn.tensor(node).unwrap().dims, vec![4, 5]);
}

#[test]
fn test_treetn_replace_tensor_invalid_node() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let new_tensor = TensorDynLen::new(indices, dims, storage);
    
    // Try to replace a non-existent node
    let invalid_node = NodeIndex::new(999);
    let result = tn.replace_tensor(invalid_node, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_none());
}

#[test]
fn test_treetn_replace_tensor_missing_connection_index() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create two connected tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Connect via j1 and i2
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to replace node1 with a tensor that doesn't have j1
    let new_i = Index::new_dyn(5);
    let new_j = Index::new_dyn(6);
    let new_indices = vec![new_i, new_j]; // j1 is missing!
    let new_dims = vec![5, 6];
    let new_storage = Arc::new(Storage::new_dense_f64(30));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_err());
    // Error should indicate missing index
    match result {
        Err(e) => {
            let err_msg = e.to_string();
            assert!(err_msg.contains("missing") || err_msg.contains("index") || err_msg.contains("connection") || err_msg.contains("indices"));
        }
        Ok(_) => panic!("Expected error but got Ok"),
    }
}

#[test]
fn test_treetn_replace_tensor_with_connection() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create two connected tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Connect via j1 and i2
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Replace node1 with a tensor that has j1 (same ID)
    let new_i = Index::new_dyn(5);
    let new_j1 = j1.clone(); // Keep the same j1 index
    let new_indices = vec![new_i, new_j1];
    let new_dims = vec![5, 3];
    let new_storage = Arc::new(Storage::new_dense_f64(15));
    let new_tensor = TensorDynLen::new(new_indices, new_dims, new_storage);
    
    let result = tn.replace_tensor(node1, new_tensor);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_treetn_connect() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create two tensors
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Connect via j1 and i2 (both dimension 3)
    let edge = tn.connect(node1, &j1, node2, &i2);
    assert!(edge.is_ok());
    assert_eq!(tn.edge_count(), 1);
    
    // Verify connection
    let conn = tn.connection(edge.unwrap());
    assert!(conn.is_some());
    assert_eq!(conn.unwrap().bond_dim(), 3);
}

#[test]
fn test_treetn_connect_dimension_mismatch() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(5); // Different dimension
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![5, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(20));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Try to connect j1 (dim 3) with i2 (dim 5) - should fail
    // Note: i2 has dimension 5, but j1 has dimension 3, so they can't be connected
    // However, the test setup has i2 in tensor2 with dimension 5, so the connection
    // will fail at the dimension mismatch check in Connection::new
    let edge = tn.connect(node1, &j1, node2, &i2);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    // Error should mention dimension mismatch or matching dimensions
    assert!(err_msg.contains("dimension") || err_msg.contains("mismatch") || err_msg.contains("matching") || err_msg.contains("Failed to create connection"));
}

#[test]
fn test_treetn_connect_invalid_node() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    // Create invalid node index
    let invalid_node = NodeIndex::new(999);
    
    let edge = tn.connect(node1, &j1, invalid_node, &i1);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    // Error should mention that nodes don't exist
    assert!(err_msg.contains("exist") || err_msg.contains("found") || err_msg.contains("node") || err_msg.contains("One or both") || err_msg.contains("Failed to connect"));
}

#[test]
fn test_treetn_connect_index_not_in_tensor() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Try to connect with an index that doesn't exist in tensor1
    let fake_index = Index::new_dyn(3);
    let edge = tn.connect(node1, &fake_index, node2, &i2);
    assert!(edge.is_err());
    let err_msg = edge.unwrap_err().to_string();
    assert!(err_msg.contains("found") || err_msg.contains("Index") || err_msg.contains("index"));
}

#[test]
fn test_treetn_edge_index_for_node() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Get index for node1
    let idx1 = tn.edge_index_for_node(edge, node1);
    assert!(idx1.is_ok());
    assert_eq!(idx1.unwrap().id, j1.id);
    
    // Get index for node2
    let idx2 = tn.edge_index_for_node(edge, node2);
    assert!(idx2.is_ok());
    assert_eq!(idx2.unwrap().id, i2.id);
    
    // Try with invalid node
    let invalid_node = NodeIndex::new(999);
    let result = tn.edge_index_for_node(edge, invalid_node);
    assert!(result.is_err());
}

#[test]
fn test_treetn_replace_edge_bond() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    assert_eq!(tn.connection(edge).unwrap().bond_dim(), 3);
    
    // Replace with new indices (dimension 5)
    let new_idx1 = Index::new_dyn(5);
    let new_idx2 = Index::new_dyn(5);
    let result = tn.replace_edge_bond(edge, new_idx1, new_idx2);
    assert!(result.is_ok());
    assert_eq!(tn.connection(edge).unwrap().bond_dim(), 5);
}

#[test]
fn test_treetn_replace_edge_bond_mismatch() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to replace with mismatched dimensions
    let new_idx1 = Index::new_dyn(5);
    let new_idx2 = Index::new_dyn(7);
    let result = tn.replace_edge_bond(edge, new_idx1, new_idx2);
    assert!(result.is_err());
}

#[test]
fn test_treetn_set_edge_ortho_towards() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Set ortho direction to j1 (node1 side)
    let result = tn.set_edge_ortho_towards(edge, Some(j1.clone()));
    assert!(result.is_ok());
    assert_eq!(
        tn.connection(edge).unwrap().ortho_towards.as_ref().map(|idx| &idx.id),
        Some(&j1.id)
    );
    
    // Set ortho direction to i2 (node2 side)
    let result = tn.set_edge_ortho_towards(edge, Some(i2.clone()));
    assert!(result.is_ok());
    assert_eq!(
        tn.connection(edge).unwrap().ortho_towards.as_ref().map(|idx| &idx.id),
        Some(&i2.id)
    );
    
    // Clear ortho direction
    let result = tn.set_edge_ortho_towards(edge, None);
    assert!(result.is_ok());
    assert!(tn.connection(edge).unwrap().ortho_towards.is_none());
}

#[test]
fn test_treetn_set_edge_ortho_towards_invalid() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let edge = tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Try to set ortho direction to an index that's not in the connection
    let invalid_index = Index::new_dyn(5); // Different ID
    let result = tn.set_edge_ortho_towards(edge, Some(invalid_index));
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("index") || err_msg.contains("index_source") || err_msg.contains("index_target"));
}

#[test]
fn test_treetn_validate_tree_simple() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create a simple tree: node1 -- node2
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Should be valid: 2 nodes, 1 edge
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_three_nodes() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create a tree: node1 -- node2 -- node3
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let j2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), j2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let i3 = Index::new_dyn(4);
    let k3 = Index::new_dyn(5);
    let indices3 = vec![i3.clone(), k3.clone()];
    let dims3 = vec![4, 5];
    let storage3 = Arc::new(Storage::new_dense_f64(20));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let node3 = tn.add_tensor(tensor3);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    tn.connect(node2, &j2, node3, &i3).unwrap();
    
    // Should be valid: 3 nodes, 2 edges
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_treetn_validate_tree_cycle() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create a cycle: node1 -- node2 -- node3 -- node1
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let k1 = Index::new_dyn(4);
    let indices1 = vec![i1.clone(), j1.clone(), k1.clone()];
    let dims1 = vec![2, 3, 4];
    let storage1 = Arc::new(Storage::new_dense_f64(24));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let j2 = Index::new_dyn(5);
    let indices2 = vec![i2.clone(), j2.clone()];
    let dims2 = vec![3, 5];
    let storage2 = Arc::new(Storage::new_dense_f64(15));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let i3 = Index::new_dyn(4);
    let j3 = Index::new_dyn(5);
    let indices3 = vec![i3.clone(), j3.clone()];
    let dims3 = vec![4, 5];
    let storage3 = Arc::new(Storage::new_dense_f64(20));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let node3 = tn.add_tensor(tensor3);
    
    tn.connect(node1, &j1, node2, &i2).unwrap();
    tn.connect(node2, &j2, node3, &j3).unwrap();
    tn.connect(node3, &i3, node1, &k1).unwrap();
    
    // Should fail: 3 nodes, 3 edges (cycle, not a tree)
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_disconnected() {
    let mut tn = TreeTN::<DynId>::new();
    
    // Create two disconnected components
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1.clone(), j1.clone()];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2.clone(), k2.clone()];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    let i3 = Index::new_dyn(5);
    let j3 = Index::new_dyn(6);
    let indices3 = vec![i3.clone(), j3.clone()];
    let dims3 = vec![5, 6];
    let storage3 = Arc::new(Storage::new_dense_f64(30));
    let tensor3 = TensorDynLen::new(indices3, dims3, storage3);
    let _node3 = tn.add_tensor(tensor3);
    
    // Connect node1 and node2, but leave node3 disconnected
    tn.connect(node1, &j1, node2, &i2).unwrap();
    
    // Should fail: graph is not connected
    assert!(tn.validate_tree().is_err());
}

#[test]
fn test_treetn_validate_tree_empty() {
    let tn = TreeTN::<DynId>::new();
    // Empty graph should be valid
    assert!(tn.validate_tree().is_ok());
}

#[test]
fn test_auto_centers_empty() {
    let tn = TreeTN::<DynId>::new();
    assert!(!tn.is_orthogonalized());
    assert!(tn.auto_centers().is_empty());
}

#[test]
fn test_set_auto_centers() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    let node = tn.add_tensor(tensor);
    
    // Initially not orthogonalized
    assert!(!tn.is_orthogonalized());
    
    // Set auto_centers
    let result = tn.set_auto_centers(vec![node]);
    assert!(result.is_ok());
    assert!(tn.is_orthogonalized());
    assert_eq!(tn.auto_centers().len(), 1);
    assert!(tn.auto_centers().contains(&node));
}

#[test]
fn test_set_auto_centers_invalid_node() {
    let mut tn = TreeTN::<DynId>::new();
    
    let invalid_node = NodeIndex::new(999);
    let result = tn.set_auto_centers(vec![invalid_node]);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result);
    assert!(err_msg.contains("exist") || err_msg.contains("valid"));
}

#[test]
fn test_add_remove_auto_center() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i1 = Index::new_dyn(2);
    let j1 = Index::new_dyn(3);
    let indices1 = vec![i1, j1];
    let dims1 = vec![2, 3];
    let storage1 = Arc::new(Storage::new_dense_f64(6));
    let tensor1 = TensorDynLen::new(indices1, dims1, storage1);
    let node1 = tn.add_tensor(tensor1);
    
    let i2 = Index::new_dyn(3);
    let k2 = Index::new_dyn(4);
    let indices2 = vec![i2, k2];
    let dims2 = vec![3, 4];
    let storage2 = Arc::new(Storage::new_dense_f64(12));
    let tensor2 = TensorDynLen::new(indices2, dims2, storage2);
    let node2 = tn.add_tensor(tensor2);
    
    // Add first center
    let result = tn.add_auto_center(node1);
    assert!(result.is_ok());
    assert!(tn.is_orthogonalized());
    assert_eq!(tn.auto_centers().len(), 1);
    
    // Add second center
    let result = tn.add_auto_center(node2);
    assert!(result.is_ok());
    assert_eq!(tn.auto_centers().len(), 2);
    assert!(tn.auto_centers().contains(&node1));
    assert!(tn.auto_centers().contains(&node2));
    
    // Remove first center
    tn.remove_auto_center(node1);
    assert_eq!(tn.auto_centers().len(), 1);
    assert!(!tn.auto_centers().contains(&node1));
    assert!(tn.auto_centers().contains(&node2));
    
    // Remove second center
    tn.remove_auto_center(node2);
    assert!(!tn.is_orthogonalized());
    assert!(tn.auto_centers().is_empty());
}

#[test]
fn test_add_auto_center_invalid() {
    let mut tn = TreeTN::<DynId>::new();
    
    let invalid_node = NodeIndex::new(999);
    let result = tn.add_auto_center(invalid_node);
    assert!(result.is_err());
}

#[test]
fn test_clear_auto_centers() {
    let mut tn = TreeTN::<DynId>::new();
    
    let i = Index::new_dyn(2);
    let j = Index::new_dyn(3);
    let indices = vec![i, j];
    let dims = vec![2, 3];
    let storage = Arc::new(Storage::new_dense_f64(6));
    let tensor = TensorDynLen::new(indices, dims, storage);
    let node = tn.add_tensor(tensor);
    
    tn.set_auto_centers(vec![node]).unwrap();
    assert!(tn.is_orthogonalized());
    
    tn.clear_auto_centers();
    assert!(!tn.is_orthogonalized());
    assert!(tn.auto_centers().is_empty());
}

