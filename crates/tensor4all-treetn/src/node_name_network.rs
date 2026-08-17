//! Node Name Network - Graph structure for node name connections.
//!
//! Provides a pure graph structure where:
//! - Nodes are identified by names (generic type `NodeName`)
//! - Edges represent connections between nodes (no data stored)
//!
//! This is a foundation for `SiteIndexNetwork` and can be used independently
//! when only the graph structure (without index information) is needed.

use crate::error::TreeTNOperationError;
use crate::named_graph::NamedGraph;
use petgraph::algo::astar;
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use petgraph::visit::{Bfs, DfsPostOrder};
use petgraph::Undirected;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

/// Ordered sequence of directed edges for canonicalization.
///
/// Each edge is `(from, to)` where:
/// - `from` is the node being orthogonalized away from
/// - `to` is the direction towards the orthogonality center
///
/// # Note on ordering
/// - For path-based canonicalization (moving ortho center), edges are connected:
///
///   each edge's `to` equals the next edge's `from`.
/// - For full canonicalization (from scratch), edges represent parent edges in
///
///   post-order DFS traversal, which may not be connected as a path but
///   guarantees correct processing order (children before parents).
///
/// # Example
/// For a chain A - B - C - D (canonizing towards D):
/// ```text
/// edges = [(A, B), (B, C), (C, D)]
/// ```
///
/// For a star with center C (canonizing towards C):
/// ```text
///     A
///     |
/// B - C - D
///     |
///     E
///
/// edges = [(A, C), (B, C), (D, C), (E, C)]  (order depends on DFS)
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalizeEdges {
    edges: Vec<(NodeIndex, NodeIndex)>,
}

impl CanonicalizeEdges {
    /// Create an empty edge sequence (no-op canonicalization).
    pub fn empty() -> Self {
        Self { edges: Vec::new() }
    }

    /// Create from a list of edges.
    ///
    /// Note: For path-based canonicalization, edges should be connected (each edge's `to`
    /// equals next edge's `from`). For full canonicalization, edges may not be connected
    /// but must be in correct processing order.
    pub fn from_edges(edges: Vec<(NodeIndex, NodeIndex)>) -> Self {
        Self { edges }
    }

    /// Check if empty (already at target, no work needed).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Number of edges to process.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Iterate over edges in order.
    pub fn iter(&self) -> impl Iterator<Item = &(NodeIndex, NodeIndex)> {
        self.edges.iter()
    }

    /// Get the final target node (orthogonality center).
    ///
    /// Returns `None` if empty.
    pub fn target(&self) -> Option<NodeIndex> {
        self.edges.last().map(|(_, to)| *to)
    }

    /// Get the starting node (first node to be factorized).
    ///
    /// Returns `None` if empty.
    pub fn start(&self) -> Option<NodeIndex> {
        self.edges.first().map(|(from, _)| *from)
    }
}

impl IntoIterator for CanonicalizeEdges {
    type Item = (NodeIndex, NodeIndex);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.into_iter()
    }
}

impl<'a> IntoIterator for &'a CanonicalizeEdges {
    type Item = &'a (NodeIndex, NodeIndex);
    type IntoIter = std::slice::Iter<'a, (NodeIndex, NodeIndex)>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}

/// Node Name Network - Pure graph structure for node connections.
///
/// Represents the topology of a network without any data attached to nodes or edges.
/// This is useful for graph algorithms that only need connectivity information.
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
#[derive(Debug, Clone)]
pub struct NodeNameNetwork<NodeName>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Named graph with unit node and edge data.
    graph: NamedGraph<NodeName, (), ()>,
}

impl<NodeName> NodeNameNetwork<NodeName>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Create a new empty NodeNameNetwork.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
        }
    }

    /// Create a new NodeNameNetwork with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: NamedGraph::with_capacity(nodes, edges),
        }
    }

    /// Add a node to the network.
    ///
    /// Returns an error if the node already exists.
    /// # Errors
    ///
    /// Returns an error when the node is a duplicate (a duplicate operation
    ///  failure) or the network is invalid (an invalid-topology failure).
    ///
    pub fn add_node(
        &mut self,
        node_name: NodeName,
    ) -> std::result::Result<NodeIndex, TreeTNOperationError> {
        self.graph.add_node(node_name, ())
    }

    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.graph.has_node(node_name)
    }

    /// Add an edge between two nodes.
    ///
    /// Returns an error if either node doesn't exist.
    /// # Errors
    ///
    /// Returns an error when an endpoint node is not found (a missing-index
    ///  failure) or the edge is a duplicate (a duplicate operation failure).
    ///
    pub fn add_edge(
        &mut self,
        n1: &NodeName,
        n2: &NodeName,
    ) -> std::result::Result<EdgeIndex, TreeTNOperationError> {
        self.graph.add_edge(n1, n2, ())
    }

    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, node_name: &NodeName) -> Option<NodeIndex> {
        self.graph.node_index(node_name)
    }

    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&NodeName> {
        self.graph.node_name(node)
    }

    /// Rename an existing node.
    /// # Errors
    ///
    /// Returns an error when the node is not found (a missing-index failure) or
    ///  the new name is a duplicate (a duplicate operation failure).
    ///
    pub fn rename_node(
        &mut self,
        old_name: &NodeName,
        new_name: NodeName,
    ) -> std::result::Result<(), TreeTNOperationError> {
        self.graph.rename_node(old_name, new_name)
    }

    /// Get all node names.
    pub fn node_names(&self) -> Vec<&NodeName> {
        self.graph.node_names()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a reference to the internal graph.
    pub fn graph(&self) -> &StableGraph<(), (), Undirected> {
        self.graph.graph()
    }

    /// Perform a post-order DFS traversal starting from the given root node.
    ///
    /// Returns node names in post-order (children before parents, leaves first).
    ///
    /// # Arguments
    /// * `root` - The node name to start traversal from
    ///
    /// # Returns
    /// `Some(Vec<NodeName>)` with nodes in post-order, or `None` if root doesn't exist.
    pub fn post_order_dfs(&self, root: &NodeName) -> Option<Vec<NodeName>> {
        let root_idx = self.graph.node_index(root)?;
        let g = self.graph.graph();

        let mut dfs = DfsPostOrder::new(g, root_idx);
        let mut result = Vec::new();

        while let Some(node_idx) = dfs.next(g) {
            if let Some(name) = self.graph.node_name(node_idx) {
                result.push(name.clone());
            }
        }

        Some(result)
    }

    /// Perform a post-order DFS traversal starting from the given root NodeIndex.
    ///
    /// Returns NodeIndex in post-order (children before parents, leaves first).
    pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        let g = self.graph.graph();
        let mut dfs = DfsPostOrder::new(g, root);
        let mut result = Vec::new();

        while let Some(node_idx) = dfs.next(g) {
            result.push(node_idx);
        }

        result
    }

    /// Perform an Euler tour traversal starting from the given root node.
    ///
    /// Delegates to [`NamedGraph::euler_tour_edges`].
    pub fn euler_tour_edges(&self, root: &NodeName) -> Option<Vec<(NodeIndex, NodeIndex)>> {
        self.graph.euler_tour_edges(root)
    }

    /// Perform an Euler tour traversal starting from the given root NodeIndex.
    ///
    /// Delegates to [`NamedGraph::euler_tour_edges_by_index`].
    pub fn euler_tour_edges_by_index(&self, root: NodeIndex) -> Vec<(NodeIndex, NodeIndex)> {
        self.graph.euler_tour_edges_by_index(root)
    }

    /// Perform an Euler tour traversal and return the vertex sequence.
    ///
    /// Delegates to [`NamedGraph::euler_tour_vertices`].
    pub fn euler_tour_vertices(&self, root: &NodeName) -> Option<Vec<NodeIndex>> {
        self.graph.euler_tour_vertices(root)
    }

    /// Perform an Euler tour traversal and return the vertex sequence by NodeIndex.
    ///
    /// Delegates to [`NamedGraph::euler_tour_vertices_by_index`].
    pub fn euler_tour_vertices_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        self.graph.euler_tour_vertices_by_index(root)
    }

    /// Find the shortest path between two nodes using A* algorithm.
    ///
    /// Since this is an unweighted graph, we use unit edge weights.
    ///
    /// # Returns
    /// `Some(Vec<NodeIndex>)` containing the path from `from` to `to` (inclusive),
    /// or `None` if no path exists.
    pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<NodeIndex>> {
        let g = self.graph.graph();

        // Check if both nodes exist
        if g.node_weight(from).is_none() || g.node_weight(to).is_none() {
            return None;
        }

        // Same node case
        if from == to {
            return Some(vec![from]);
        }

        // Use A* with trivial heuristic (unit edge cost, zero estimate)
        astar(
            g,
            from,
            |n| n == to,
            |_| 1usize, // Unit edge cost
            |_| 0usize, // No heuristic (behaves like Dijkstra/BFS)
        )
        .map(|(_, path)| path)
    }

    /// Check if a subset of nodes forms a connected subgraph.
    ///
    /// Uses DFS to verify that all nodes in the subset are reachable from each other
    /// within the induced subgraph.
    ///
    /// # Returns
    /// `true` if the subset is connected (or empty), `false` otherwise.
    pub fn is_connected_subset(&self, nodes: &HashSet<NodeIndex>) -> bool {
        if nodes.is_empty() || nodes.len() == 1 {
            return true;
        }

        let g = self.graph.graph();

        // DFS within the subset only: petgraph's Dfs visits all reachable
        // nodes, but we must restrict traversal to edges whose both endpoints
        // are in the subset.
        let Some(&start) = nodes.iter().next() else {
            return true;
        };
        let mut seen = HashSet::new();
        let mut stack = vec![start];
        seen.insert(start);

        while let Some(v) = stack.pop() {
            for nb in g.neighbors(v) {
                if nodes.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }

        seen.len() == nodes.len()
    }

    /// Compute the Steiner tree nodes spanning a set of terminal nodes.
    ///
    /// For tree graphs, the Steiner tree is the union of the unique paths from
    /// one terminal to every other terminal.
    ///
    /// # Arguments
    /// * `terminals` - Terminal node indices to span
    ///
    /// # Returns
    /// The set of nodes in the minimal connected subtree spanning `terminals`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use tensor4all_treetn::NodeNameNetwork;
    ///
    /// let mut net: NodeNameNetwork<String> = NodeNameNetwork::new();
    /// let a = net.add_node("A".to_string()).unwrap();
    /// let b = net.add_node("B".to_string()).unwrap();
    /// let c = net.add_node("C".to_string()).unwrap();
    /// net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    /// net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    ///
    /// let steiner = net.steiner_tree_nodes(&[a, c].into_iter().collect::<HashSet<_>>());
    /// assert_eq!(steiner, [a, b, c].into_iter().collect());
    /// ```
    pub fn steiner_tree_nodes(&self, terminals: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
        if terminals.len() <= 1 {
            return terminals.clone();
        }

        let terminals_vec: Vec<NodeIndex> = terminals.iter().copied().collect();
        let root = terminals_vec[0];
        let mut result = HashSet::new();
        result.insert(root);

        for &terminal in &terminals_vec[1..] {
            if let Some(path) = self.path_between(root, terminal) {
                result.extend(path);
            }
        }

        result
    }

    /// Convert a node sequence to an edge sequence.
    fn nodes_to_edges(nodes: &[NodeIndex]) -> CanonicalizeEdges {
        if nodes.len() < 2 {
            return CanonicalizeEdges::empty();
        }
        let edges: Vec<_> = nodes.windows(2).map(|w| (w[0], w[1])).collect();
        CanonicalizeEdges::from_edges(edges)
    }

    /// Compute edges to canonicalize from current state to target.
    ///
    /// # Arguments
    /// * `current_region` - Current ortho region (`None` = not canonicalized)
    /// * `target` - Target node for the orthogonality center
    ///
    /// # Returns
    /// Ordered `CanonicalizeEdges` to process for canonicalization.
    pub fn edges_to_canonicalize(
        &self,
        current_region: Option<&HashSet<NodeIndex>>,
        target: NodeIndex,
    ) -> CanonicalizeEdges {
        match current_region {
            None => {
                // Not canonicalized: compute parent edges for each node in post-order.
                let post_order = self.post_order_dfs_by_index(target);
                self.compute_parent_edges(&post_order, target)
            }
            Some(current) if current.contains(&target) => {
                // Already at target: no-op
                CanonicalizeEdges::empty()
            }
            Some(current) => {
                // Move from current to target: find path
                if let Some(&start) = current.iter().next() {
                    if let Some(path) = self.path_between(start, target) {
                        Self::nodes_to_edges(&path)
                    } else {
                        CanonicalizeEdges::empty()
                    }
                } else {
                    CanonicalizeEdges::empty()
                }
            }
        }
    }

    /// Compute edges to canonicalize from leaves to target, returning node names.
    ///
    /// This is similar to `edges_to_canonicalize(None, target)` but returns
    /// `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.
    ///
    /// Useful for operations that work with two networks that have the same
    /// topology but different NodeIndex values (e.g., contract_zipup).
    ///
    /// # Arguments
    /// * `target` - Target node name for the orthogonality center
    ///
    /// # Returns
    /// `None` if target node doesn't exist, otherwise a vector of `(from, to)` pairs
    /// where `from` is the node being processed and `to` is its parent (towards target).
    pub fn edges_to_canonicalize_by_names(
        &self,
        target: &NodeName,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        let target_idx = self.node_index(target)?;
        let edges = self.edges_to_canonicalize(None, target_idx);

        let result: Vec<_> = edges
            .into_iter()
            .filter_map(|(from_idx, to_idx)| {
                let from_name = self.node_name(from_idx)?.clone();
                let to_name = self.node_name(to_idx)?.clone();
                Some((from_name, to_name))
            })
            .collect();

        Some(result)
    }

    /// Compute parent edges for each node in the given order.
    fn compute_parent_edges(&self, nodes: &[NodeIndex], root: NodeIndex) -> CanonicalizeEdges {
        let g = self.graph.graph();
        let mut edges = Vec::with_capacity(nodes.len().saturating_sub(1));

        // Build parent map using BFS from root: a single O(V + E) pass where
        // each vertex's first discoverer becomes its parent, yielding a rooted
        // tree orientation over the connected component.
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut bfs = Bfs::new(g, root);
        while let Some(node) = bfs.next(g) {
            for neighbor in g.neighbors(node) {
                if neighbor != root {
                    parent.entry(neighbor).or_insert(node);
                }
            }
        }

        // For each node in order, add edge to its parent
        for &node in nodes {
            if node != root {
                if let Some(&p) = parent.get(&node) {
                    edges.push((node, p));
                }
            }
        }

        CanonicalizeEdges::from_edges(edges)
    }

    /// Compute edges to canonicalize from leaves towards a connected region (multiple centers).
    ///
    /// Given a set of target nodes forming a connected region, this function returns
    /// all edges (src, dst) where:
    /// - `src` is a node outside the target region
    /// - `dst` is the next node towards the target region
    ///
    /// The edges are ordered so that nodes farther from the target region are processed first
    /// (children before parents), which is the correct order for canonicalization.
    ///
    /// # Arguments
    /// * `target_region` - Set of NodeIndex that forms the canonical center region
    ///
    ///   (must be non-empty and connected)
    ///
    /// # Returns
    /// `CanonicalizeEdges` with all edges pointing towards the target region.
    /// Returns empty edges if target_region is empty.
    ///
    /// # Panics
    /// Does not panic, but if target_region is disconnected, behavior is undefined
    /// (may return partial results).
    pub fn edges_to_canonicalize_to_region(
        &self,
        target_region: &HashSet<NodeIndex>,
    ) -> CanonicalizeEdges {
        if target_region.is_empty() {
            return CanonicalizeEdges::empty();
        }

        let g = self.graph.graph();

        // Multi-source BFS from target_region to compute distances and parent pointers
        let mut dist: HashMap<NodeIndex, usize> = HashMap::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue = VecDeque::new();

        // Initialize all target region nodes at distance 0
        for &node in target_region {
            dist.insert(node, 0);
            queue.push_back(node);
        }

        // BFS to find distances and parents
        while let Some(node) = queue.pop_front() {
            let d = dist[&node];
            for neighbor in g.neighbors(node) {
                let was_new = !dist.contains_key(&neighbor);
                dist.entry(neighbor).or_insert_with(|| {
                    parent.insert(neighbor, node);
                    d + 1
                });
                if was_new {
                    queue.push_back(neighbor);
                }
            }
        }

        // Collect edges from nodes outside target region towards their parent
        // Sort by distance (descending) so farther nodes are processed first
        let mut node_dist_pairs: Vec<(NodeIndex, usize)> = dist
            .iter()
            .filter(|(node, _)| !target_region.contains(node))
            .map(|(&node, &d)| (node, d))
            .collect();

        node_dist_pairs.sort_by_key(|pair| std::cmp::Reverse(pair.1)); // Descending by distance

        let edges: Vec<(NodeIndex, NodeIndex)> = node_dist_pairs
            .iter()
            .filter_map(|(node, _)| {
                let p = parent.get(node)?;
                Some((*node, *p))
            })
            .collect();

        CanonicalizeEdges::from_edges(edges)
    }

    /// Compute edges to canonicalize towards a region, returning node names.
    ///
    /// This is similar to `edges_to_canonicalize_to_region` but takes and returns
    /// node names instead of NodeIndex.
    ///
    /// # Arguments
    /// * `target_region` - Set of node names that forms the canonical center region
    ///
    /// # Returns
    /// `None` if any target node doesn't exist, otherwise `Some(Vec<(from, to)>)`
    /// where edges point towards the target region.
    pub fn edges_to_canonicalize_to_region_by_names(
        &self,
        target_region: &HashSet<NodeName>,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        // Convert node names to NodeIndex
        let target_indices: HashSet<NodeIndex> = target_region
            .iter()
            .map(|name| self.node_index(name))
            .collect::<Option<HashSet<_>>>()?;

        let edges = self.edges_to_canonicalize_to_region(&target_indices);

        let result: Vec<_> = edges
            .into_iter()
            .filter_map(|(from_idx, to_idx)| {
                let from_name = self.node_name(from_idx)?.clone();
                let to_name = self.node_name(to_idx)?.clone();
                Some((from_name, to_name))
            })
            .collect();

        Some(result)
    }

    /// Check if two networks have the same topology (same nodes and edges).
    pub fn same_topology(&self, other: &Self) -> bool {
        if self.node_count() != other.node_count() {
            return false;
        }
        if self.edge_count() != other.edge_count() {
            return false;
        }

        // Check all nodes exist in both
        for name in self.node_names() {
            if !other.has_node(name) {
                return false;
            }
        }

        // Check edges match (by checking neighbors for each node)
        let self_graph = self.graph.graph();
        for name in self.node_names() {
            let Some(self_idx) = self.node_index(name) else {
                return false;
            };
            let other_idx = match other.node_index(name) {
                Some(idx) => idx,
                None => return false,
            };

            let self_neighbors: HashSet<_> = self_graph
                .neighbors(self_idx)
                .filter_map(|n| self.node_name(n))
                .collect();

            let other_graph = other.graph.graph();
            let other_neighbors: HashSet<_> = other_graph
                .neighbors(other_idx)
                .filter_map(|n| other.node_name(n))
                .collect();

            if self_neighbors != other_neighbors {
                return false;
            }
        }

        true
    }
}

impl<NodeName> Default for NodeNameNetwork<NodeName>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
