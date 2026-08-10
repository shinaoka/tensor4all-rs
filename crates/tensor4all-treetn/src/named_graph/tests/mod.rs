use super::*;

#[test]
fn test_named_graph_basic() {
    let mut g: NamedGraph<String, i32, ()> = NamedGraph::new();

    // Add nodes
    g.add_node("A".to_string(), 1).unwrap();
    g.add_node("B".to_string(), 2).unwrap();
    g.add_node("C".to_string(), 3).unwrap();

    assert_eq!(g.node_count(), 3);
    assert!(g.has_node(&"A".to_string()));
    assert!(!g.has_node(&"D".to_string()));

    // Add edges
    g.add_edge(&"A".to_string(), &"B".to_string(), ()).unwrap();
    g.add_edge(&"B".to_string(), &"C".to_string(), ()).unwrap();

    assert_eq!(g.edge_count(), 2);

    // Get neighbors
    let neighbors = g.neighbors(&"B".to_string());
    assert_eq!(neighbors.len(), 2);
    assert!(neighbors.contains(&&"A".to_string()));
    assert!(neighbors.contains(&&"C".to_string()));

    // Get data
    assert_eq!(g.node_data(&"A".to_string()), Some(&1));
    assert_eq!(g.node_data(&"B".to_string()), Some(&2));
}

#[test]
fn test_named_graph_tuple_nodes() {
    let mut g: NamedGraph<(i32, i32), String, f64> = NamedGraph::new();

    g.add_node((1, 1), "site1".to_string()).unwrap();
    g.add_node((1, 2), "site2".to_string()).unwrap();
    g.add_node((2, 1), "site3".to_string()).unwrap();

    g.add_edge(&(1, 1), &(1, 2), 1.5).unwrap();
    g.add_edge(&(1, 1), &(2, 1), 2.0).unwrap();

    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 2);

    let neighbors = g.neighbors(&(1, 1));
    assert_eq!(neighbors.len(), 2);
}

#[test]
fn test_named_graph_remove() {
    let mut g: NamedGraph<String, i32, ()> = NamedGraph::new();

    g.add_node("A".to_string(), 1).unwrap();
    g.add_node("B".to_string(), 2).unwrap();
    g.add_edge(&"A".to_string(), &"B".to_string(), ()).unwrap();

    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 1);

    // Remove edge
    g.remove_edge(&"A".to_string(), &"B".to_string());
    assert_eq!(g.edge_count(), 0);

    // Remove node
    let data = g.remove_node(&"A".to_string());
    assert_eq!(data, Some(1));
    assert_eq!(g.node_count(), 1);
    assert!(!g.has_node(&"A".to_string()));
}

#[test]
fn test_named_graph_rename_node() {
    let mut g: NamedGraph<String, i32, ()> = NamedGraph::new();

    g.add_node("A".to_string(), 1).unwrap();
    g.add_node("B".to_string(), 2).unwrap();
    g.add_edge(&"A".to_string(), &"B".to_string(), ()).unwrap();

    g.rename_node(&"B".to_string(), "C".to_string()).unwrap();

    assert!(g.has_node(&"A".to_string()));
    assert!(g.has_node(&"C".to_string()));
    assert!(!g.has_node(&"B".to_string()));
    assert_eq!(g.node_data(&"C".to_string()), Some(&2));
    let neighbors = g.neighbors(&"A".to_string());
    assert_eq!(neighbors, vec![&"C".to_string()]);
}

#[test]
fn test_euler_tour_chain() {
    // Chain: A - B - C
    let mut g: NamedGraph<String, (), ()> = NamedGraph::new();
    let a = g.add_node("A".to_string(), ()).unwrap();
    let b = g.add_node("B".to_string(), ()).unwrap();
    let c = g.add_node("C".to_string(), ()).unwrap();
    g.add_edge(&"A".to_string(), &"B".to_string(), ()).unwrap();
    g.add_edge(&"B".to_string(), &"C".to_string(), ()).unwrap();

    // Euler tour from B: should visit all edges twice
    let edges = g.euler_tour_edges(&"B".to_string()).unwrap();
    // 2 edges × 2 directions = 4 directed edges
    assert_eq!(edges.len(), 4);

    // Vertices: should start and end at B
    let vertices = g.euler_tour_vertices(&"B".to_string()).unwrap();
    assert_eq!(vertices.len(), 5); // B, ?, B, ?, B
    assert_eq!(vertices[0], b);
    assert_eq!(vertices[vertices.len() - 1], b);

    // All vertices should be visited
    let unique_vertices: HashSet<_> = vertices.iter().cloned().collect();
    assert_eq!(unique_vertices, [a, b, c].into());
}

#[test]
fn test_euler_tour_single_node() {
    let mut g: NamedGraph<String, (), ()> = NamedGraph::new();
    let a = g.add_node("A".to_string(), ()).unwrap();

    let edges = g.euler_tour_edges(&"A".to_string()).unwrap();
    assert!(edges.is_empty());

    let vertices = g.euler_tour_vertices(&"A".to_string()).unwrap();
    assert_eq!(vertices, vec![a]);
}

#[test]
fn test_named_graph_add_node_returns_anyhow_error() {
    let mut g: NamedGraph<String, i32, ()> = NamedGraph::new();
    g.add_node("A".to_string(), 1).unwrap();

    let result = g.add_node("A".to_string(), 2);
    let err = result.unwrap_err();

    assert!(err.to_string().contains("Node already exists"));
}
