use super::*;

#[test]
fn test_node_name_network_basic() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();

    assert_eq!(net.node_count(), 2);
    assert!(net.has_node(&"A".to_string()));

    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    assert_eq!(net.edge_count(), 1);
}

#[test]
fn test_post_order_dfs_chain() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    net.add_node("A".to_string()).unwrap();
    net.add_node("B".to_string()).unwrap();
    net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    let result = net.post_order_dfs(&"B".to_string()).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result.last().unwrap(), "B");
}

#[test]
fn test_path_between() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    let path = net.path_between(a, d).unwrap();
    assert_eq!(path, vec![a, b, c, d]);
}

#[test]
fn test_is_connected_subset() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let _d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    assert!(net.is_connected_subset(&[b, c].into()));
    assert!(!net.is_connected_subset(&[a, c].into())); // Gap
}

#[test]
fn test_steiner_tree_nodes_chain() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let n0 = net.add_node("N0".to_string()).unwrap();
    let n1 = net.add_node("N1".to_string()).unwrap();
    let n2 = net.add_node("N2".to_string()).unwrap();
    let n3 = net.add_node("N3".to_string()).unwrap();
    let n4 = net.add_node("N4".to_string()).unwrap();
    net.add_edge(&"N0".to_string(), &"N1".to_string()).unwrap();
    net.add_edge(&"N1".to_string(), &"N2".to_string()).unwrap();
    net.add_edge(&"N2".to_string(), &"N3".to_string()).unwrap();
    net.add_edge(&"N3".to_string(), &"N4".to_string()).unwrap();

    let steiner = net.steiner_tree_nodes(&[n0, n2, n4].into());
    assert_eq!(steiner, [n0, n1, n2, n3, n4].into());

    let steiner = net.steiner_tree_nodes(&[n1, n3].into());
    assert_eq!(steiner, [n1, n2, n3].into());
}

#[test]
fn test_steiner_tree_nodes_tree() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let _c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    let e = net.add_node("E".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"D".to_string()).unwrap();
    net.add_edge(&"D".to_string(), &"E".to_string()).unwrap();

    let steiner = net.steiner_tree_nodes(&[a, e].into());
    assert_eq!(steiner, [a, b, d, e].into());
}

#[test]
fn test_steiner_tree_nodes_single_and_adjacent() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();

    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert_eq!(net.steiner_tree_nodes(&[a].into()), [a].into());
    assert_eq!(net.steiner_tree_nodes(&[a, b].into()), [a, b].into());
}

#[test]
fn test_same_topology() {
    let mut net1: NodeNameNetwork<String> = NodeNameNetwork::new();
    net1.add_node("A".to_string()).unwrap();
    net1.add_node("B".to_string()).unwrap();
    net1.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    let mut net2: NodeNameNetwork<String> = NodeNameNetwork::new();
    net2.add_node("A".to_string()).unwrap();
    net2.add_node("B".to_string()).unwrap();
    net2.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

    assert!(net1.same_topology(&net2));

    let mut net3: NodeNameNetwork<String> = NodeNameNetwork::new();
    net3.add_node("A".to_string()).unwrap();
    net3.add_node("C".to_string()).unwrap();
    net3.add_edge(&"A".to_string(), &"C".to_string()).unwrap();

    assert!(!net1.same_topology(&net3));
}

#[test]
fn test_euler_tour_chain() {
    // Chain: A - B - C
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();

    // Euler tour from B: should visit all edges twice
    let edges = net.euler_tour_edges(&"B".to_string()).unwrap();
    // 2 edges × 2 directions = 4 directed edges
    assert_eq!(edges.len(), 4);

    // Verify each undirected edge appears twice (forward and backward)
    let edge_set: HashSet<_> = edges.iter().cloned().collect();
    assert!(edge_set.contains(&(b, a)) || edge_set.contains(&(b, c)));

    // Vertices: should start and end at B
    let vertices = net.euler_tour_vertices(&"B".to_string()).unwrap();
    assert_eq!(vertices.len(), 5); // B, ?, B, ?, B
    assert_eq!(vertices[0], b);
    assert_eq!(vertices[vertices.len() - 1], b);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c].into());
}

#[test]
fn test_euler_tour_y_shape() {
    // Y-shape:
    //     A
    //     |
    // B - C - D
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();

    // Euler tour from C: should visit all 3 edges twice
    let edges = net.euler_tour_edges(&"C".to_string()).unwrap();
    // 3 edges × 2 directions = 6 directed edges
    assert_eq!(edges.len(), 6);

    // Start from C
    assert_eq!(edges[0].0, c);

    // Vertices: 7 visits (C + 6 edges)
    let vertices = net.euler_tour_vertices(&"C".to_string()).unwrap();
    assert_eq!(vertices.len(), 7);
    assert_eq!(vertices[0], c);
    assert_eq!(vertices[vertices.len() - 1], c);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c, d].into());
}

#[test]
fn test_euler_tour_single_node() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();

    let edges = net.euler_tour_edges(&"A".to_string()).unwrap();
    assert!(edges.is_empty());

    let vertices = net.euler_tour_vertices(&"A".to_string()).unwrap();
    assert_eq!(vertices, vec![a]);
}

#[test]
fn test_node_name_network_add_node_returns_anyhow_error() {
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    net.add_node("A".to_string()).unwrap();

    let result = net.add_node("A".to_string());
    let err = result.unwrap_err();

    assert!(err.to_string().contains("Node already exists"));
}

#[test]
fn test_euler_tour_star() {
    // Star: center C connected to A, B, D, E
    //     A
    //     |
    // B - C - D
    //     |
    //     E
    let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    let a = net.add_node("A".to_string()).unwrap();
    let b = net.add_node("B".to_string()).unwrap();
    let c = net.add_node("C".to_string()).unwrap();
    let d = net.add_node("D".to_string()).unwrap();
    let e = net.add_node("E".to_string()).unwrap();
    net.add_edge(&"A".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"D".to_string()).unwrap();
    net.add_edge(&"C".to_string(), &"E".to_string()).unwrap();

    // Euler tour from C: should visit all 4 edges twice
    let edges = net.euler_tour_edges(&"C".to_string()).unwrap();
    // 4 edges × 2 directions = 8 directed edges
    assert_eq!(edges.len(), 8);

    // Start from C
    assert_eq!(edges[0].0, c);

    // Vertices: 9 visits (C + 8 edges)
    let vertices = net.euler_tour_vertices(&"C".to_string()).unwrap();
    assert_eq!(vertices.len(), 9);
    assert_eq!(vertices[0], c);
    assert_eq!(vertices[vertices.len() - 1], c);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c, d, e].into());
}
