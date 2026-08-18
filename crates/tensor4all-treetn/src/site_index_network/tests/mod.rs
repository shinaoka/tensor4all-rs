use super::*;
use tensor4all_core::DynIndex;

#[test]
fn test_site_index_network_basic() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    net.add_node("A".to_string(), site1).unwrap();
    net.add_node("B".to_string(), site2).unwrap();

    assert_eq!(net.node_count(), 2);
    assert!(net.has_node(&"A".to_string()));

    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert_eq!(net.edge_count(), 1);
}

#[test]
fn test_post_order_dfs_chain() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let result = net.post_order_dfs(&"B".to_string()).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.last().unwrap(), "B");
    assert!(result.contains(&"A".to_string()));
    assert!(result.contains(&"C".to_string()));
}

#[test]
fn test_path_between_chain() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let path = net.path_between(a, d).unwrap();
    assert_eq!(path, vec![a, b, c, d]);
}

#[test]
fn test_edges_to_canonicalize_full() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let edges = net.edges_to_canonicalize(None, d);
    assert_eq!(edges.len(), 3);
    let edge_vec: Vec<_> = edges.iter().cloned().collect();
    assert_eq!(edge_vec, vec![(a, b), (b, c), (c, d)]);
}

#[test]
fn test_is_connected_subset() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    let _d = net.add_node("D".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    assert!(net.is_connected_subset(&HashSet::new()));
    assert!(net.is_connected_subset(&[a].into()));
    assert!(net.is_connected_subset(&[b, c].into()));
    assert!(!net.is_connected_subset(&[a, c].into()));
}

#[test]
fn test_steiner_tree_nodes_forwarding() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();

    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty.clone()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    assert_eq!(net.steiner_tree_nodes(&[a, c].into()), [a, b, c].into());
}

#[test]
fn test_share_equivalent_site_index_network() {
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net1.add_node("A".to_string(), site1.clone()).unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node("A".to_string(), site1.clone()).unwrap();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.share_equivalent_site_index_network(&net2));

    // Different site space
    let mut net3: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site3: HashSet<_> = [DynIndex::new_dyn(5)].into();
    net3.add_node("A".to_string(), site3).unwrap();
    net3.add_node("B".to_string(), HashSet::new()).unwrap();
    net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(!net1.share_equivalent_site_index_network(&net3));
}

#[test]
fn test_apply_operator_topology() {
    // Create state network
    let mut state: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    state.add_node("A".to_string(), site1.clone()).unwrap();
    state.add_node("B".to_string(), site2.clone()).unwrap();
    state.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // Create operator with same topology
    let mut operator: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    // Operator has different index IDs but same dimensions
    let op_site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    let op_site2: HashSet<_> = [DynIndex::new_dyn(3)].into();
    operator.add_node("A".to_string(), op_site1).unwrap();
    operator.add_node("B".to_string(), op_site2).unwrap();
    operator
        .add_edge(&"A".to_string(), &"B".to_string())
        .unwrap();

    // Should succeed
    let result = state.apply_operator_topology(&operator);
    assert!(result.is_ok());

    // Create operator with different topology
    let mut bad_operator: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    bad_operator
        .add_node("A".to_string(), HashSet::new())
        .unwrap();
    bad_operator
        .add_node("C".to_string(), HashSet::new())
        .unwrap(); // Different node name
    bad_operator
        .add_edge(&"A".to_string(), &"C".to_string())
        .unwrap();

    // Should fail
    let result = state.apply_operator_topology(&bad_operator);
    assert!(result.is_err());
}

#[test]
fn test_compatible_site_dimensions() {
    // Create two networks with same dimensions
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site1: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net1.add_node("A".to_string(), site1).unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    // Different ID but same dimension
    let site2: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net2.add_node("A".to_string(), site2).unwrap();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.compatible_site_dimensions(&net2));

    // Create network with different dimension
    let mut net3: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site3: HashSet<_> = [DynIndex::new_dyn(5)].into(); // Different dim
    net3.add_node("A".to_string(), site3).unwrap();
    net3.add_node("B".to_string(), HashSet::new()).unwrap();
    net3.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(!net1.compatible_site_dimensions(&net3));
}

#[test]
fn test_with_capacity() {
    let net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::with_capacity(4, 3);
    assert_eq!(net.node_count(), 0);
    assert_eq!(net.edge_count(), 0);
}

#[test]
fn test_default_site_index_network() {
    let net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::default();
    assert_eq!(net.node_count(), 0);
}

#[test]
fn test_rename_node() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx = DynIndex::new_dyn(2);
    let site: HashSet<_> = [idx.clone()].into();
    net.add_node("A".to_string(), site).unwrap();

    // Rename to same name is a no-op
    net.rename_node(&"A".to_string(), "A".to_string()).unwrap();
    assert!(net.has_node(&"A".to_string()));

    // Rename to different name
    net.rename_node(&"A".to_string(), "B".to_string()).unwrap();
    assert!(!net.has_node(&"A".to_string()));
    assert!(net.has_node(&"B".to_string()));

    // Reverse lookup should be updated
    assert_eq!(net.find_node_by_index(&idx).unwrap(), "B");
}

#[test]
fn test_contains_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx = DynIndex::new_dyn(2);
    let other = DynIndex::new_dyn(3);
    let site: HashSet<_> = [idx.clone()].into();
    net.add_node("A".to_string(), site).unwrap();

    assert!(net.contains_index(&idx));
    assert!(!net.contains_index(&other));
}

#[test]
fn test_add_site_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    let site: HashSet<_> = [idx1.clone()].into();
    net.add_node("A".to_string(), site).unwrap();

    // Add another site index
    net.add_site_index(&"A".to_string(), idx2.clone()).unwrap();
    assert!(net.contains_index(&idx2));
    assert_eq!(net.site_space(&"A".to_string()).unwrap().len(), 2);

    // Error for non-existent node
    assert!(net.add_site_index(&"Z".to_string(), idx2.clone()).is_err());
}

#[test]
fn test_remove_site_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    let site: HashSet<_> = [idx1.clone(), idx2.clone()].into();
    net.add_node("A".to_string(), site).unwrap();

    // Remove existing index
    let removed = net.remove_site_index(&"A".to_string(), &idx1).unwrap();
    assert!(removed);
    assert!(!net.contains_index(&idx1));

    // Remove non-existing index returns false
    let removed = net.remove_site_index(&"A".to_string(), &idx1).unwrap();
    assert!(!removed);

    // Error for non-existent node
    assert!(net.remove_site_index(&"Z".to_string(), &idx2).is_err());
}

#[test]
fn test_replace_site_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    let site: HashSet<_> = [idx1.clone()].into();
    net.add_node("A".to_string(), site).unwrap();

    // Replace idx1 with idx2
    net.replace_site_index(&"A".to_string(), &idx1, idx2.clone())
        .unwrap();
    assert!(!net.contains_index(&idx1));
    assert!(net.contains_index(&idx2));
    assert_eq!(net.find_node_by_index(&idx2).unwrap(), "A");

    // Error: replacing non-existent index
    assert!(net
        .replace_site_index(&"A".to_string(), &idx1, DynIndex::new_dyn(4))
        .is_err());

    // Error: non-existent node
    assert!(net
        .replace_site_index(&"Z".to_string(), &idx2, DynIndex::new_dyn(4))
        .is_err());
}

#[test]
fn test_site_space_by_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx = DynIndex::new_dyn(2);
    let site: HashSet<_> = [idx.clone()].into();
    let node_idx = net.add_node("A".to_string(), site).unwrap();

    let space = net.site_space_by_index(node_idx).unwrap();
    assert_eq!(space.len(), 1);
    assert!(space.contains(&idx));
}

#[test]
fn test_graph_access() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // graph() read access
    assert_eq!(net.graph().node_count(), 2);
    assert_eq!(net.graph().edge_count(), 1);
}

#[test]
fn test_post_order_dfs_by_index() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    let a = net.add_node("A".to_string(), empty.clone()).unwrap();
    let b = net.add_node("B".to_string(), empty.clone()).unwrap();
    let c = net.add_node("C".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let order = net.post_order_dfs_by_index(b);
    assert_eq!(order.len(), 3);
    // Root (b) should be last
    assert_eq!(*order.last().unwrap(), b);
    assert!(order.contains(&a));
    assert!(order.contains(&c));
}

#[test]
fn test_edges_to_canonicalize_to_region_by_names() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let region: HashSet<String> = ["B".to_string()].into();
    let edges = net
        .edges_to_canonicalize_to_region_by_names(&region)
        .unwrap();
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_compatible_site_dimensions_different_count() {
    // Network where one node has different number of site indices
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    net1.add_node("A".to_string(), HashSet::from([idx1.clone()]))
        .unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node(
        "A".to_string(),
        HashSet::from([idx2.clone(), DynIndex::new_dyn(4)]),
    )
    .unwrap();

    assert!(!net1.compatible_site_dimensions(&net2));
}

#[test]
fn test_compatible_site_dimensions_one_has_none() {
    // One node has site space, the other doesn't
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net1.add_node("A".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // net2 has same topology but B has site indices while A has none
    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node("A".to_string(), HashSet::new()).unwrap();
    net2.add_node("B".to_string(), HashSet::from([DynIndex::new_dyn(2)]))
        .unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // A has Some vs None, should fail
    assert!(!net1.compatible_site_dimensions(&net2));
}

#[test]
fn test_set_site_space() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx1 = DynIndex::new_dyn(2);
    let idx2 = DynIndex::new_dyn(3);
    net.add_node("A".to_string(), HashSet::from([idx1.clone()]))
        .unwrap();

    // Replace site space entirely
    let new_set = HashSet::from([idx2.clone()]);
    net.set_site_space(&"A".to_string(), new_set).unwrap();
    assert!(!net.contains_index(&idx1));
    assert!(net.contains_index(&idx2));
    assert_eq!(net.find_node_by_index(&idx2).unwrap(), "A");
    assert!(net.find_node_by_index(&idx1).is_none());

    // Error for non-existent node
    assert!(net
        .set_site_space(&"Z".to_string(), HashSet::new())
        .is_err());
}

#[test]
fn test_edges_to_canonicalize_by_names() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let edges = net
        .edges_to_canonicalize_by_names(&"B".to_string())
        .unwrap();
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_site_space_mut() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let idx = DynIndex::new_dyn(2);
    net.add_node("A".to_string(), HashSet::from([idx.clone()]))
        .unwrap();

    // Mutable access to modify site space
    let space = net.site_space_mut(&"A".to_string()).unwrap();
    assert_eq!(space.len(), 1);

    // Non-existent node
    assert!(net.site_space_mut(&"Z".to_string()).is_none());
}

#[test]
fn test_edges_iterator() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let edges: Vec<_> = net.edges().collect();
    assert_eq!(edges.len(), 2);
}

#[test]
fn test_neighbors() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let empty: HashSet<DynIndex> = HashSet::new();
    net.add_node("A".to_string(), empty.clone()).unwrap();
    net.add_node("B".to_string(), empty.clone()).unwrap();
    net.add_node("C".to_string(), empty).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let neighbors_b: Vec<_> = net.neighbors(&"B".to_string()).collect();
    assert_eq!(neighbors_b.len(), 2);
    assert!(neighbors_b.contains(&"A".to_string()));
    assert!(neighbors_b.contains(&"C".to_string()));
}

#[test]
fn test_share_equivalent_topology_mismatch() {
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net1.add_node("A".to_string(), HashSet::new()).unwrap();
    net1.add_node("B".to_string(), HashSet::new()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    // net2 has different topology (no edge)
    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node("A".to_string(), HashSet::new()).unwrap();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();

    assert!(!net1.share_equivalent_site_index_network(&net2));
}

#[test]
fn test_compatible_different_topology() {
    let mut net1: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net1.add_node("A".to_string(), HashSet::new()).unwrap();

    let mut net2: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    net2.add_node("B".to_string(), HashSet::new()).unwrap();

    assert!(!net1.compatible_site_dimensions(&net2));
}

#[test]
fn test_site_index_network_add_node_returns_anyhow_error() {
    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    let site: HashSet<_> = [DynIndex::new_dyn(2)].into();
    net.add_node("A".to_string(), site).unwrap();

    let result = net.add_node("A".to_string(), HashSet::new());
    let err = result.unwrap_err();

    assert!(err.to_string().contains("Node already exists"));
}
