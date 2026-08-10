//! Named graph wrapper inspired by NamedGraphs.jl
//! (<https://github.com/mtfishman/NamedGraphs.jl>)
//!
//! Provides a mapping between arbitrary node name types (NodeName) and internal NodeIndex.
//! This allows using meaningful identifiers (coordinates, strings, etc.) instead of raw indices.

use crate::error::TreeTNOperationError;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use petgraph::EdgeType;
use petgraph::Undirected;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

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
    /// # Errors
    ///
    /// Returns an error when the graph is invalid (an invalid-topology
    ///  failure).
    ///
    pub fn add_node(
        &mut self,
        node_name: NodeName,
        data: NodeData,
    ) -> std::result::Result<NodeIndex, TreeTNOperationError> {
        if self.node_name_to_index.contains_key(&node_name) {
            return Err(TreeTNOperationError::from(anyhow::anyhow!(
                "Node already exists: {:?}",
                node_name
            )));
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

    /// Rename an existing node without changing its data or incident edges.
    ///
    /// Returns an error if the old node doesn't exist or the new name is already in use.
    /// # Errors
    ///
    /// Returns an error when the node is not found (a missing-index failure)
    ///  or the new name is already in use (a duplicate operation failure).
    ///
    pub fn rename_node(
        &mut self,
        old_name: &NodeName,
        new_name: NodeName,
    ) -> std::result::Result<(), TreeTNOperationError> {
        if old_name == &new_name {
            return Ok(());
        }
        if self.node_name_to_index.contains_key(&new_name) {
            return Err(TreeTNOperationError::from(anyhow::anyhow!(
                "Node already exists: {:?}",
                new_name
            )));
        }

        let node_idx = self
            .node_name_to_index
            .remove(old_name)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {:?}", old_name))?;

        self.node_name_to_index.insert(new_name.clone(), node_idx);
        self.index_to_node_name.insert(node_idx, new_name);
        Ok(())
    }

    /// Get a reference to the data of a node (by node name).
    pub fn node_data(&self, node_name: &NodeName) -> Option<&NodeData> {
        self.node_name_to_index
            .get(node_name)
            .and_then(|node| self.graph.node_weight(*node))
    }

    /// Get a mutable reference to the data of a node (by node name).
    pub fn node_data_mut(&mut self, node_name: &NodeName) -> Option<&mut NodeData> {
        self.node_name_to_index
            .get(node_name)
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
    /// # Errors
    ///
    /// Returns an error when the graph is invalid (an invalid-topology
    ///  failure).
    ///
    pub fn add_edge(
        &mut self,
        n1: &NodeName,
        n2: &NodeName,
        weight: EdgeData,
    ) -> std::result::Result<EdgeIndex, TreeTNOperationError>
    where
        EdgeData: Clone,
    {
        let node1 = self
            .node_name_to_index
            .get(n1)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {:?}", n1))?;
        let node2 = self
            .node_name_to_index
            .get(n2)
            .ok_or_else(|| anyhow::anyhow!("Node not found: {:?}", n2))?;
        Ok(self.graph.add_edge(*node1, *node2, weight))
    }

    /// Get the weight of an edge between two nodes.
    pub fn edge_weight(&self, n1: &NodeName, n2: &NodeName) -> Option<&EdgeData> {
        let node1 = self.node_name_to_index.get(n1)?;
        let node2 = self.node_name_to_index.get(n2)?;
        self.graph
            .find_edge(*node1, *node2)
            .and_then(|edge| self.graph.edge_weight(edge))
    }

    /// Get a mutable reference to the weight of an edge between two nodes.
    pub fn edge_weight_mut(&mut self, n1: &NodeName, n2: &NodeName) -> Option<&mut EdgeData> {
        let node1 = self.node_name_to_index.get(n1)?;
        let node2 = self.node_name_to_index.get(n2)?;
        self.graph
            .find_edge(*node1, *node2)
            .and_then(|edge| self.graph.edge_weight_mut(edge))
    }

    /// Get all neighbors of a node.
    pub fn neighbors(&self, node_name: &NodeName) -> Vec<&NodeName> {
        self.node_name_to_index
            .get(node_name)
            .map(|node| {
                self.graph
                    .neighbors(*node)
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
        self.graph
            .find_edge(*node1, *node2)
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

    /// Perform an Euler tour traversal starting from the given root node.
    ///
    /// The Euler tour visits each edge exactly twice (once in each direction),
    /// forming a closed walk that covers all edges. This is useful for sweep
    /// algorithms where we need to visit all nodes while maintaining the
    /// canonical center at a single point.
    ///
    /// # Algorithm
    /// Uses DFS with backtracking. When visiting a node:
    /// 1. For each unvisited neighbor, record the edge and recurse
    /// 2. When backtracking, record the edge back to the parent
    ///
    /// # Returns
    /// A vector of directed edges `(from, to)` in traversal order.
    /// Each undirected edge appears twice: once as (u, v) and once as (v, u).
    ///
    /// # Example
    /// For a Y-shaped tree with root at center C:
    /// ```text
    ///     A
    ///     |
    /// B - C - D
    /// ```
    /// Starting from C, the tour might be:
    /// `[(C,A), (A,C), (C,B), (B,C), (C,D), (D,C)]`
    pub fn euler_tour_edges(&self, root: &NodeName) -> Option<Vec<(NodeIndex, NodeIndex)>> {
        let root_idx = self.node_index(root)?;
        Some(self.euler_tour_edges_by_index(root_idx))
    }

    /// Perform an Euler tour traversal starting from the given root NodeIndex.
    ///
    /// See [`Self::euler_tour_edges`] for details.
    pub fn euler_tour_edges_by_index(&self, root: NodeIndex) -> Vec<(NodeIndex, NodeIndex)> {
        // Iterative DFS-based Euler tour: each undirected edge is emitted once
        // in each direction (forward on descent, backward on backtrack). This is
        // a single O(V + E) pass with a visited-edge set; there is no
        // per-vertex rescan, so the total cost stays linear in the graph size.
        let mut visited_edges: HashSet<(NodeIndex, NodeIndex)> = HashSet::new();
        let mut tour = Vec::new();
        let mut stack = vec![root];

        while let Some(u) = stack.last().copied() {
            let mut pushed = false;

            // Try to find an unvisited neighbor
            for v in self.graph.neighbors(u) {
                if !visited_edges.contains(&(u, v)) {
                    // Mark both directions as visited (undirected edge)
                    visited_edges.insert((u, v));
                    visited_edges.insert((v, u));
                    // Record the forward edge
                    tour.push((u, v));
                    // Push neighbor onto stack
                    stack.push(v);
                    pushed = true;
                    break; // Handle one neighbor at a time
                }
            }

            if !pushed {
                // No unvisited neighbors, backtrack
                stack.pop();
                if let Some(&parent) = stack.last() {
                    // Record the backtracking edge
                    tour.push((u, parent));
                }
            }
        }

        tour
    }

    /// Perform an Euler tour traversal and return the vertex sequence.
    ///
    /// This returns the sequence of vertices visited, including repeated visits.
    /// Each vertex appears multiple times based on its degree in the tree.
    ///
    /// # Returns
    /// A vector of NodeIndex in visitation order, or None if root doesn't exist.
    ///
    /// # Example
    /// For a chain A - B - C starting from B:
    /// - Edges: `[(B,A), (A,B), (B,C), (C,B)]`
    /// - Vertices: `[B, A, B, C, B]`
    pub fn euler_tour_vertices(&self, root: &NodeName) -> Option<Vec<NodeIndex>> {
        let root_idx = self.node_index(root)?;
        Some(self.euler_tour_vertices_by_index(root_idx))
    }

    /// Perform an Euler tour traversal and return the vertex sequence by NodeIndex.
    ///
    /// See [`Self::euler_tour_vertices`] for details.
    pub fn euler_tour_vertices_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        let edges = self.euler_tour_edges_by_index(root);
        if edges.is_empty() {
            // Single node case
            return vec![root];
        }
        // Start with the first vertex, then append destinations
        let mut vertices = vec![edges[0].0];
        for (_, to) in &edges {
            vertices.push(*to);
        }
        vertices
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
mod tests;
