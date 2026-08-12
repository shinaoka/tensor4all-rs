use super::*;
use std::collections::HashMap;
use tensor4all_core::{DynIndex, IdxTensor};

#[test]
fn test_tree_topology_validate() {
    // Test empty topology (use usize as a dummy ID type)
    let empty: TreeTopology<String, usize> = TreeTopology::new(HashMap::new(), Vec::new());
    assert!(empty.validate().is_err());

    // Test single node (valid)
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    let single = TreeTopology::new(nodes, Vec::new());
    assert!(single.validate().is_ok());

    // Test valid tree (2 nodes, 1 edge)
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    nodes.insert("node1".to_string(), vec![1usize]);
    let edges = vec![("node0".to_string(), "node1".to_string())];
    let tree = TreeTopology::new(nodes, edges);
    assert!(tree.validate().is_ok());

    // Test invalid: wrong number of edges
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    nodes.insert("node1".to_string(), vec![1usize]);
    let edges = vec![
        ("node0".to_string(), "node1".to_string()),
        ("node0".to_string(), "node1".to_string()),
    ];
    let invalid = TreeTopology::new(nodes, edges);
    assert!(invalid.validate().is_err());

    // Test invalid: edge references unknown node
    let mut nodes = HashMap::new();
    nodes.insert("node0".to_string(), vec![0usize]);
    let edges = vec![("node0".to_string(), "node2".to_string())];
    let invalid = TreeTopology::new(nodes, edges);
    assert!(invalid.validate().is_err());
}

#[test]
fn test_factorize_tensor_to_treetn_rejects_duplicate_index_assignment() {
    let i0 = DynIndex::new_dyn(2);
    let i1 = DynIndex::new_dyn(2);
    let tensor =
        IdxTensor::from_dense(vec![i0.clone(), i1.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut nodes: HashMap<String, Vec<DynIndex>> = HashMap::new();
    nodes.insert("node0".to_string(), vec![i0.clone()]);
    nodes.insert("node1".to_string(), vec![i0.clone()]); // duplicate on purpose

    let topo = TreeTopology::new(nodes, vec![("node0".to_string(), "node1".to_string())]);

    let result = factorize_tensor_to_treetn_with(
        &tensor,
        &topo,
        FactorizeOptions::qr(),
        &"node0".to_string(),
    );
    assert!(result.is_err());
}

#[test]
fn test_factorize_tensor_to_treetn_preserves_requested_chain_topology_for_nonroot_leaf() {
    let s0 = DynIndex::new_dyn(2);
    let s1 = DynIndex::new_dyn(2);
    let s2 = DynIndex::new_dyn(2);
    let tensor = IdxTensor::from_dense(
        vec![s0.clone(), s1.clone(), s2.clone()],
        vec![10.0, 30.0, 20.0, 40.0, 11.0, 31.0, 21.0, 41.0],
    )
    .unwrap();

    let mut nodes = HashMap::new();
    nodes.insert(0usize, vec![s0.clone()]);
    nodes.insert(1usize, vec![s1.clone()]);
    nodes.insert(2usize, vec![s2.clone()]);
    let topology = TreeTopology::new(nodes, vec![(0usize, 1usize), (1usize, 2usize)]);

    let tn = factorize_tensor_to_treetn(&tensor, &topology, &2usize).unwrap();
    let mut edges = tn.site_index_network().edges().collect::<Vec<_>>();
    edges.sort();

    assert_eq!(edges, vec![(0usize, 1usize), (1usize, 2usize)]);
}

#[test]
fn test_factorize_tensor_to_treetn_accepts_full_index_topology_for_same_id_prime_pair() {
    let i = DynIndex::new_dyn(2);
    let i_prime = i.prime();
    let tensor =
        IdxTensor::from_dense(vec![i.clone(), i_prime.clone()], vec![1.0, 0.0, 0.0, 1.0]).unwrap();

    let mut nodes: HashMap<String, Vec<DynIndex>> = HashMap::new();
    nodes.insert("input".to_string(), vec![i.clone()]);
    nodes.insert("output".to_string(), vec![i_prime.clone()]);
    let topology = TreeTopology::new(nodes, vec![("input".to_string(), "output".to_string())]);

    let tn = factorize_tensor_to_treetn(&tensor, &topology, &"output".to_string()).unwrap();

    assert_eq!(
        tn.site_index_network()
            .find_node_by_index(&i)
            .map(|name| name.as_str()),
        Some("input")
    );
    assert_eq!(
        tn.site_index_network()
            .find_node_by_index(&i_prime)
            .map(|name| name.as_str()),
        Some("output")
    );
}
