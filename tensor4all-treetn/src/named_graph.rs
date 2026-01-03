//! Named graph wrapper inspired by NamedGraphs.jl
//! (https://github.com/mtfishman/NamedGraphs.jl)
//!
//! Provides a mapping between arbitrary node name types (NodeName) and internal NodeIndex.
//! This allows using meaningful identifiers (coordinates, strings, etc.) instead of raw indices.

use petgraph::stable_graph::{StableGraph, NodeIndex, EdgeIndex};
use petgraph::Undirected;
use petgraph::EdgeType;
use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;

/// Generic named graph wrapper (inspired by NamedGraphs.jl)
///
/// Provides a mapping between arbitrary node name types (NodeName) and internal NodeIndex.
/// This allows using meaningful identifiers (coordinates, strings, etc.) instead of raw indices.
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
/// - `NodeData`: Node weight/data type
/// - `EdgeData`: Edge weight/data type
/// - `Ty`: Edge type (Directed or Undirected, default: Undirected)
pub struct NamedGraph<NodeName, NodeData, EdgeData, Ty = Undirected>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Ty: EdgeType,
{
    /// Internal graph structure (uses NodeIndex)
    graph: StableGraph<NodeData, EdgeData, Ty>,
    
    /// Mapping: node name (NodeName) -> NodeIndex
    node_name_to_index: HashMap<NodeName, NodeIndex>,
    
    /// Reverse mapping: NodeIndex -> node name (NodeName)
    index_to_node_name: HashMap<NodeIndex, NodeName>,
}

impl<NodeName, NodeData, EdgeData, Ty> NamedGraph<NodeName, NodeData, EdgeData, Ty>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Ty: EdgeType,
{
    /// Create a new empty NamedGraph.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::with_capacity(0, 0),
            node_name_to_index: HashMap::new(),
            index_to_node_name: HashMap::new(),
        }
    }
    
    /// Create a new NamedGraph with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: StableGraph::with_capacity(nodes, edges),
            node_name_to_index: HashMap::with_capacity(nodes),
            index_to_node_name: HashMap::with_capacity(nodes),
        }
    }
    
    /// Add a node with the given name and data.
    ///
    /// Returns an error if the node already exists.
    pub fn add_node(&mut self, node_name: NodeName, data: NodeData) -> Result<NodeIndex, String> {
        if self.node_name_to_index.contains_key(&node_name) {
            return Err(format!("Node already exists: {:?}", node_name));
        }
        let node = self.graph.add_node(data);
        self.node_name_to_index.insert(node_name.clone(), node);
        self.index_to_node_name.insert(node, node_name);
        Ok(node)
    }
    
    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.node_name_to_index.contains_key(node_name)
    }
    
    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, node_name: &NodeName) -> Option<NodeIndex> {
        self.node_name_to_index.get(node_name).copied()
    }
    
    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&NodeName> {
        self.index_to_node_name.get(&node)
    }
    
    /// Get a reference to the data of a node (by node name).
    pub fn node_data(&self, node_name: &NodeName) -> Option<&NodeData> {
        self.node_name_to_index.get(node_name)
            .and_then(|node| self.graph.node_weight(*node))
    }
    
    /// Get a mutable reference to the data of a node (by node name).
    pub fn node_data_mut(&mut self, node_name: &NodeName) -> Option<&mut NodeData> {
        self.node_name_to_index.get(node_name)
            .and_then(|node| self.graph.node_weight_mut(*node))
    }
    
    /// Get a reference to the data of a node (by NodeIndex).
    pub fn node_weight(&self, node: NodeIndex) -> Option<&NodeData> {
        self.graph.node_weight(node)
    }
    
    /// Get a mutable reference to the data of a node (by NodeIndex).
    pub fn node_weight_mut(&mut self, node: NodeIndex) -> Option<&mut NodeData> {
        self.graph.node_weight_mut(node)
    }
    
    /// Add an edge between two nodes.
    ///
    /// Returns an error if either node doesn't exist.
    pub fn add_edge(&mut self, n1: &NodeName, n2: &NodeName, weight: EdgeData) -> Result<EdgeIndex, String>
    where
        EdgeData: Clone,
    {
        let node1 = self.node_name_to_index.get(n1)
            .ok_or_else(|| format!("Node not found: {:?}", n1))?;
        let node2 = self.node_name_to_index.get(n2)
            .ok_or_else(|| format!("Node not found: {:?}", n2))?;
        Ok(self.graph.add_edge(*node1, *node2, weight))
    }
    
    /// Get the weight of an edge between two nodes.
    pub fn edge_weight(&self, n1: &NodeName, n2: &NodeName) -> Option<&EdgeData> {
        let node1 = self.node_name_to_index.get(n1)?;
        let node2 = self.node_name_to_index.get(n2)?;
        self.graph.find_edge(*node1, *node2)
            .and_then(|edge| self.graph.edge_weight(edge))
    }
    
    /// Get a mutable reference to the weight of an edge between two nodes.
    pub fn edge_weight_mut(&mut self, n1: &NodeName, n2: &NodeName) -> Option<&mut EdgeData> {
        let node1 = self.node_name_to_index.get(n1)?;
        let node2 = self.node_name_to_index.get(n2)?;
        self.graph.find_edge(*node1, *node2)
            .and_then(|edge| self.graph.edge_weight_mut(edge))
    }
    
    /// Get all neighbors of a node.
    pub fn neighbors(&self, node_name: &NodeName) -> Vec<&NodeName> {
        self.node_name_to_index.get(node_name)
            .map(|node| {
                self.graph.neighbors(*node)
                    .filter_map(|n| self.index_to_node_name.get(&n))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get all node names.
    pub fn node_names(&self) -> Vec<&NodeName> {
        self.node_name_to_index.keys().collect()
    }
    
    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_name_to_index.len()
    }
    
    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    
    /// Remove a node and all its edges.
    ///
    /// Returns the node data if the node existed.
    pub fn remove_node(&mut self, node_name: &NodeName) -> Option<NodeData> {
        let node = self.node_name_to_index.remove(node_name)?;
        self.index_to_node_name.remove(&node);
        self.graph.remove_node(node)
    }
    
    /// Remove an edge between two nodes.
    ///
    /// Returns the edge weight if the edge existed.
    pub fn remove_edge(&mut self, n1: &NodeName, n2: &NodeName) -> Option<EdgeData> {
        let node1 = self.node_name_to_index.get(n1)?;
        let node2 = self.node_name_to_index.get(n2)?;
        self.graph.find_edge(*node1, *node2)
            .and_then(|edge| self.graph.remove_edge(edge))
    }
    
    /// Check if a node exists in the internal graph.
    pub fn contains_node(&self, node: NodeIndex) -> bool {
        self.graph.contains_node(node)
    }
    
    /// Get a reference to the internal graph.
    ///
    /// This allows direct access to petgraph algorithms that work with NodeIndex.
    pub fn graph(&self) -> &StableGraph<NodeData, EdgeData, Ty> {
        &self.graph
    }
    
    /// Get a mutable reference to the internal graph.
    ///
    /// **Warning**: Directly modifying the internal graph can break the node-name-to-index mapping.
    /// Use the provided methods instead.
    pub fn graph_mut(&mut self) -> &mut StableGraph<NodeData, EdgeData, Ty> {
        &mut self.graph
    }
}

impl<NodeName, NodeData, EdgeData, Ty> Default for NamedGraph<NodeName, NodeData, EdgeData, Ty>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Ty: EdgeType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<NodeName, NodeData, EdgeData, Ty> Clone for NamedGraph<NodeName, NodeData, EdgeData, Ty>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    NodeData: Clone,
    EdgeData: Clone,
    Ty: EdgeType,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            node_name_to_index: self.node_name_to_index.clone(),
            index_to_node_name: self.index_to_node_name.clone(),
        }
    }
}

impl<NodeName, NodeData, EdgeData, Ty> Debug for NamedGraph<NodeName, NodeData, EdgeData, Ty>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    NodeData: Debug,
    EdgeData: Debug,
    Ty: EdgeType,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NamedGraph")
            .field("graph", &self.graph)
            .field("node_name_to_index", &self.node_name_to_index)
            .field("index_to_node_name", &self.index_to_node_name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
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
}

