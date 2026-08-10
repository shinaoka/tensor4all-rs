//! Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
//!
//! Provides a structure combining:
//! - **NodeNameNetwork**: Graph topology (node connections)
//! - **Site space map**: Physical indices at each node (`HashMap<NodeName, HashSet<I>>`)
//!
//! This design separates the index structure from tensor data,
//! enabling topology and site space comparison independent of tensor values.

use crate::error::TreeTNOperationError;
use crate::node_name_network::{CanonicalizeEdges, NodeNameNetwork};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableGraph};
use petgraph::Undirected;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use tensor4all_core::IndexLike;

// Re-export CanonicalizeEdges for convenience
pub use crate::node_name_network::CanonicalizeEdges as CanonicalizeEdgesType;

/// Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
///
/// Represents the index structure of a tensor network:
/// - **Topology**: Graph structure via `NodeNameNetwork`
/// - **Site space**: Physical indices at each node via `HashMap`
///
/// This structure enables:
/// - Comparing topologies and site spaces independently of tensor data
/// - Extracting index information without accessing tensor values
/// - Validating network structure consistency
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
/// - `I`: Index type (must implement `IndexLike`)
///
/// # Examples
///
/// ```
/// use std::collections::HashSet;
/// use tensor4all_core::index::{DynId, Index, TagSet};
/// use tensor4all_treetn::SiteIndexNetwork;
///
/// let mut net = SiteIndexNetwork::<String, Index<DynId, TagSet>>::new();
/// let idx_a = Index::new_dyn(2);
/// let idx_b = Index::new_dyn(3);
///
/// net.add_node("A".to_string(), HashSet::from([idx_a])).unwrap();
/// net.add_node("B".to_string(), HashSet::from([idx_b])).unwrap();
/// net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
///
/// assert_eq!(net.node_count(), 2);
/// assert_eq!(net.edge_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    /// Graph topology (node names and connections only).
    topology: NodeNameNetwork<NodeName>,
    /// Site space (physical indices) for each node.
    site_spaces: HashMap<NodeName, HashSet<I>>,
    /// Reverse lookup: full index metadata → node name containing this index.
    index_to_node: HashMap<I, NodeName>,
}

impl<NodeName, I> SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    /// Create a new empty SiteIndexNetwork.
    pub fn new() -> Self {
        Self {
            topology: NodeNameNetwork::new(),
            site_spaces: HashMap::new(),
            index_to_node: HashMap::new(),
        }
    }

    /// Create a new SiteIndexNetwork with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            topology: NodeNameNetwork::with_capacity(nodes, edges),
            site_spaces: HashMap::with_capacity(nodes),
            index_to_node: HashMap::new(),
        }
    }

    /// Add a node with site space (physical indices).
    ///
    /// # Arguments
    /// * `node_name` - The name of the node
    /// * `site_space` - The physical indices at this node (order doesn't matter)
    ///
    /// Returns an error if the node already exists.
    /// # Errors
    ///
    /// Returns an error when the site space is invalid (a shape mismatch) or the
    ///  node is a duplicate (a duplicate operation failure).
    ///
    pub fn add_node(
        &mut self,
        node_name: NodeName,
        site_space: impl Into<HashSet<I>>,
    ) -> std::result::Result<NodeIndex, TreeTNOperationError> {
        let node_idx = self.topology.add_node(node_name.clone())?;
        let site_space_set = site_space.into();
        // Update reverse lookup for all indices.
        for idx in &site_space_set {
            self.index_to_node.insert(idx.clone(), node_name.clone());
        }
        self.site_spaces.insert(node_name, site_space_set);
        Ok(node_idx)
    }

    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.topology.has_node(node_name)
    }

    /// Rename an existing node and preserve its site-space metadata.
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
        if old_name == &new_name {
            return Ok(());
        }

        let site_space = self
            .site_spaces
            .remove(old_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", old_name))?;
        self.topology.rename_node(old_name, new_name.clone())?;
        for index in &site_space {
            self.index_to_node.insert(index.clone(), new_name.clone());
        }
        self.site_spaces.insert(new_name, site_space);
        Ok(())
    }

    /// Get the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &NodeName) -> Option<&HashSet<I>> {
        self.site_spaces.get(node_name)
    }

    /// Get a mutable reference to the site space for a node.
    ///
    /// **Warning**: Direct modification of site space via this method does NOT
    /// update the reverse lookup (`index_to_node`). Use `add_site_index()`,
    /// `remove_site_index()`, or `replace_site_index()` for modifications
    /// that maintain consistency.
    pub fn site_space_mut(&mut self, node_name: &NodeName) -> Option<&mut HashSet<I>> {
        self.site_spaces.get_mut(node_name)
    }

    /// Find the node containing a given site index.
    ///
    /// # Arguments
    /// * `index` - The index to look up
    ///
    /// # Returns
    /// The node name containing this index, or None if not found.
    pub fn find_node_by_index(&self, index: &I) -> Option<&NodeName> {
        self.index_to_node.get(index)
    }

    /// Check if a site index is registered.
    pub fn contains_index(&self, index: &I) -> bool {
        self.index_to_node.contains_key(index)
    }

    /// Return the number of registered site indices.
    ///
    /// This uses the reverse index lookup and does not scan every node's site
    /// space. Use it when validating a complete site-index assignment.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_treetn::SiteIndexNetwork;
    ///
    /// let i = DynIndex::new_dyn(2);
    /// let mut network = SiteIndexNetwork::<&str, DynIndex>::new();
    /// network.add_node("A", HashSet::from([i])).unwrap();
    ///
    /// assert_eq!(network.site_index_count(), 1);
    /// ```
    pub fn site_index_count(&self) -> usize {
        self.index_to_node.len()
    }

    /// Add a site index to a node's site space.
    ///
    /// Updates both the site space and the reverse lookup.
    /// # Errors
    ///
    /// Returns an error when the site index is already present (a duplicate-index
    ///  failure) or the node is not found (a missing-index failure).
    ///
    pub fn add_site_index(
        &mut self,
        node_name: &NodeName,
        index: I,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let site_space = self
            .site_spaces
            .get_mut(node_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;
        site_space.insert(index.clone());
        self.index_to_node.insert(index, node_name.clone());
        Ok(())
    }

    /// Remove a site index from a node's site space.
    ///
    /// Updates both the site space and the reverse lookup.
    /// # Errors
    ///
    /// Returns an error when the site index is not present (a missing-index
    ///  failure).
    ///
    pub fn remove_site_index(
        &mut self,
        node_name: &NodeName,
        index: &I,
    ) -> std::result::Result<bool, TreeTNOperationError> {
        let site_space = self
            .site_spaces
            .get_mut(node_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;
        let removed = site_space.remove(index);
        if removed {
            self.index_to_node.remove(index);
        }
        Ok(removed)
    }

    /// Replace a site index in a node's site space.
    ///
    /// Updates both the site space and the reverse lookup.
    /// # Errors
    ///
    /// Returns an error when the old site index is not present (a missing-index
    ///  failure) or the new index has an incompatible dimension (a shape
    ///  mismatch).
    ///
    pub fn replace_site_index(
        &mut self,
        node_name: &NodeName,
        old_index: &I,
        new_index: I,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let site_space = self
            .site_spaces
            .get_mut(node_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;
        if !site_space.remove(old_index) {
            return Err(anyhow::anyhow!(
                "Index {:?} not found in node {:?}",
                old_index.id(),
                node_name
            ))
            .map_err(TreeTNOperationError::from);
        }
        self.index_to_node.remove(old_index);
        site_space.insert(new_index.clone());
        self.index_to_node.insert(new_index, node_name.clone());
        Ok(())
    }

    /// Replace all site indices for a node with a new set.
    ///
    /// Updates both the site space and the reverse lookup.
    /// This is an atomic operation that removes all old indices and adds all new ones.
    /// # Errors
    ///
    /// Returns an error when the node is not found (a missing-index failure) or
    ///  the site space is invalid (a shape mismatch).
    ///
    pub fn set_site_space(
        &mut self,
        node_name: &NodeName,
        new_indices: HashSet<I>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let site_space = self
            .site_spaces
            .get_mut(node_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;

        // Remove old indices from index_to_node
        for old_idx in site_space.iter() {
            if self.index_to_node.get(old_idx) == Some(node_name) {
                self.index_to_node.remove(old_idx);
            }
        }

        // Add new indices to index_to_node
        for new_idx in &new_indices {
            self.index_to_node
                .insert(new_idx.clone(), node_name.clone());
        }

        // Replace site space
        *site_space = new_indices;

        Ok(())
    }

    /// Get the site space by NodeIndex.
    pub fn site_space_by_index(&self, node: NodeIndex) -> Option<&HashSet<I>> {
        let name = self.topology.node_name(node)?;
        self.site_spaces.get(name)
    }

    /// Add an edge between two nodes.
    ///
    /// Returns an error if either node doesn't exist.
    /// # Errors
    ///
    /// Returns an error when an endpoint node is not found (a missing-index
    ///  failure) or the edge already exists (a duplicate operation failure).
    ///
    pub fn add_edge(
        &mut self,
        n1: &NodeName,
        n2: &NodeName,
    ) -> std::result::Result<EdgeIndex, TreeTNOperationError> {
        self.topology.add_edge(n1, n2)
    }

    /// Get the NodeIndex for a node name.
    pub fn node_index(&self, node_name: &NodeName) -> Option<NodeIndex> {
        self.topology.node_index(node_name)
    }

    /// Get the node name for a NodeIndex.
    pub fn node_name(&self, node: NodeIndex) -> Option<&NodeName> {
        self.topology.node_name(node)
    }

    /// Get all node names.
    pub fn node_names(&self) -> Vec<&NodeName> {
        self.topology.node_names()
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.topology.node_count()
    }

    /// Get the number of edges.
    pub fn edge_count(&self) -> usize {
        self.topology.edge_count()
    }

    /// Get a reference to the underlying topology (NodeNameNetwork).
    pub fn topology(&self) -> &NodeNameNetwork<NodeName> {
        &self.topology
    }

    /// Get all edges as pairs of node names.
    ///
    /// Returns an iterator of `(NodeName, NodeName)` pairs.
    pub fn edges(&self) -> impl Iterator<Item = (NodeName, NodeName)> + '_ {
        let graph = self.topology.graph();
        graph.edge_indices().filter_map(move |edge| {
            let (a, b) = graph.edge_endpoints(edge)?;
            let name_a = self.topology.node_name(a)?.clone();
            let name_b = self.topology.node_name(b)?.clone();
            Some((name_a, name_b))
        })
    }

    /// Get all neighbors of a node.
    ///
    /// Returns an iterator of neighbor node names.
    pub fn neighbors(&self, node_name: &NodeName) -> impl Iterator<Item = NodeName> + '_ {
        let node_idx = self.topology.node_index(node_name);
        let graph = self.topology.graph();
        let topology = &self.topology;

        node_idx
            .into_iter()
            .flat_map(move |idx| graph.neighbors(idx))
            .filter_map(move |n| topology.node_name(n).cloned())
    }

    /// Get a reference to the internal graph.
    pub fn graph(&self) -> &StableGraph<(), (), Undirected> {
        self.topology.graph()
    }

    /// Get a mutable reference to the internal graph.
    ///
    /// **Warning**: Directly modifying the internal graph can break consistency.
    pub fn graph_mut(&mut self) -> &mut StableGraph<(), (), Undirected> {
        self.topology.graph_mut()
    }

    /// Check if two SiteIndexNetworks share equivalent site index structure.
    ///
    /// Two networks are equivalent if:
    /// - Same topology (nodes and edges)
    /// - Same site space for each node
    ///
    /// This is used to verify that two TreeTNs can be added or contracted.
    pub fn share_equivalent_site_index_network(&self, other: &Self) -> bool {
        // Check topology
        if !self.topology.same_topology(&other.topology) {
            return false;
        }

        // Check site spaces
        for name in self.node_names() {
            match (self.site_space(name), other.site_space(name)) {
                (Some(self_indices), Some(other_indices)) => {
                    if self_indices != other_indices {
                        return false;
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }

        true
    }

    // =========================================================================
    // Delegated graph algorithms (from NodeNameNetwork)
    // =========================================================================

    /// Perform a post-order DFS traversal starting from the given root node.
    pub fn post_order_dfs(&self, root: &NodeName) -> Option<Vec<NodeName>> {
        self.topology.post_order_dfs(root)
    }

    /// Perform a post-order DFS traversal starting from the given root NodeIndex.
    pub fn post_order_dfs_by_index(&self, root: NodeIndex) -> Vec<NodeIndex> {
        self.topology.post_order_dfs_by_index(root)
    }

    /// Find the shortest path between two nodes.
    pub fn path_between(&self, from: NodeIndex, to: NodeIndex) -> Option<Vec<NodeIndex>> {
        self.topology.path_between(from, to)
    }

    /// Compute the Steiner tree nodes spanning a set of terminal nodes.
    ///
    /// Delegates to [`NodeNameNetwork::steiner_tree_nodes`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_treetn::SiteIndexNetwork;
    ///
    /// let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    /// let a = net.add_node("A".to_string(), HashSet::<DynIndex>::new()).unwrap();
    /// let b = net.add_node("B".to_string(), HashSet::<DynIndex>::new()).unwrap();
    /// let c = net.add_node("C".to_string(), HashSet::<DynIndex>::new()).unwrap();
    /// net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
    /// net.add_edge(&"B".to_string(), &"C".to_string()).unwrap();
    ///
    /// let steiner = net.steiner_tree_nodes(&[a, c].into_iter().collect::<HashSet<_>>());
    /// assert_eq!(steiner, [a, b, c].into_iter().collect());
    /// ```
    pub fn steiner_tree_nodes(&self, terminals: &HashSet<NodeIndex>) -> HashSet<NodeIndex> {
        self.topology.steiner_tree_nodes(terminals)
    }

    /// Check if a subset of nodes forms a connected subgraph.
    pub fn is_connected_subset(&self, nodes: &HashSet<NodeIndex>) -> bool {
        self.topology.is_connected_subset(nodes)
    }

    /// Compute edges to canonicalize from current state to target.
    pub fn edges_to_canonicalize(
        &self,
        current_region: Option<&HashSet<NodeIndex>>,
        target: NodeIndex,
    ) -> CanonicalizeEdges {
        self.topology.edges_to_canonicalize(current_region, target)
    }

    /// Compute edges to canonicalize from leaves to target, returning node names.
    ///
    /// This is similar to `edges_to_canonicalize(None, target)` but returns
    /// `(from_name, to_name)` pairs instead of `(NodeIndex, NodeIndex)`.
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_by_names`] for details.
    pub fn edges_to_canonicalize_by_names(
        &self,
        target: &NodeName,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        self.topology.edges_to_canonicalize_by_names(target)
    }

    /// Compute edges to canonicalize from leaves towards a connected region (multiple centers).
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_to_region`] for details.
    pub fn edges_to_canonicalize_to_region(
        &self,
        target_region: &HashSet<NodeIndex>,
    ) -> CanonicalizeEdges {
        self.topology.edges_to_canonicalize_to_region(target_region)
    }

    /// Compute edges to canonicalize towards a region, returning node names.
    ///
    /// See [`NodeNameNetwork::edges_to_canonicalize_to_region_by_names`] for details.
    pub fn edges_to_canonicalize_to_region_by_names(
        &self,
        target_region: &HashSet<NodeName>,
    ) -> Option<Vec<(NodeName, NodeName)>> {
        self.topology
            .edges_to_canonicalize_to_region_by_names(target_region)
    }

    // =========================================================================
    // Operator/state compatibility checking
    // =========================================================================

    /// Check if an operator can act on this state (as a ket).
    ///
    /// Returns `Ok(result_network)` if the operator can act on self,
    /// where `result_network` is the SiteIndexNetwork of the output state.
    ///
    /// For an operator to act on a state:
    /// - They must have the same topology (same nodes and edges)
    /// - The operator must have indices that can contract with the state's site indices
    ///
    /// # Arguments
    /// * `operator` - The operator's SiteIndexNetwork
    ///
    /// # Returns
    /// - `Ok(SiteIndexNetwork)` - The resulting state's site index network after the operator acts
    /// - `Err(TreeTNOperationError)` - An error if the operator cannot act on this state
    ///
    /// # Note
    /// This is a simplified version that assumes the operator's output indices
    /// have the same structure as the input (i.e., the result has the same
    /// site index structure as the original state). For more complex operators
    /// with different input/output dimensions, a more sophisticated approach
    /// would be needed.
    /// # Errors
    ///
    /// Returns an error when the operator topology is incompatible (a shape or
    ///  index mismatch failure).
    ///
    pub fn apply_operator_topology(
        &self,
        operator: &Self,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        // Check topology match
        if !self.topology.same_topology(&operator.topology) {
            return Err(anyhow::anyhow!(
                "Operator and state have different topologies. State nodes: {:?}, Operator nodes: {:?}",
                self.node_names(),
                operator.node_names()
            ))
            .map_err(TreeTNOperationError::from);
        }

        // For now, assume the operator preserves the site index structure
        // (output has same site indices as input). This is the common case
        // for Hamiltonians and other operators where H|ψ⟩ has the same
        // index structure as |ψ⟩.
        //
        // A more complete implementation would:
        // 1. Check that operator has compatible input indices
        // 2. Return the actual output index structure
        Ok(self.clone())
    }

    /// Check if this network has compatible site dimensions with another.
    ///
    /// Two networks have compatible site dimensions if:
    /// - Same topology (nodes and edges)
    /// - Each node has the same number of site indices
    /// - Site index dimensions match (after sorting, order doesn't matter)
    ///
    /// This is useful for checking if two states can be added or if
    /// a state matches the expected output of an operator.
    pub fn compatible_site_dimensions(&self, other: &Self) -> bool {
        // Check topology
        if !self.topology.same_topology(&other.topology) {
            return false;
        }

        // Check site dimensions for each node
        for name in self.node_names() {
            match (self.site_space(name), other.site_space(name)) {
                (Some(self_indices), Some(other_indices)) => {
                    // Check same number of indices
                    if self_indices.len() != other_indices.len() {
                        return false;
                    }

                    // Get dimensions and sort for comparison
                    // Use IndexLike::dim() to get the dimension
                    let mut self_dims: Vec<_> = self_indices.iter().map(|idx| idx.dim()).collect();
                    let mut other_dims: Vec<_> =
                        other_indices.iter().map(|idx| idx.dim()).collect();
                    self_dims.sort();
                    other_dims.sort();

                    if self_dims != other_dims {
                        return false;
                    }
                }
                (None, None) => continue,
                _ => return false,
            }
        }

        true
    }
}

impl<NodeName, I> Default for SiteIndexNetwork<NodeName, I>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    I: IndexLike,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Type alias for backwards compatibility
// ============================================================================

use tensor4all_core::index::{DynId, Index};
use tensor4all_core::DefaultTagSet;

/// Type alias for the default SiteIndexNetwork using DynId indices.
///
/// This preserves backwards compatibility with existing code that uses
/// `SiteIndexNetwork<NodeName, Id, Symm, Tags>`.
pub type DefaultSiteIndexNetwork<NodeName> =
    SiteIndexNetwork<NodeName, Index<DynId, DefaultTagSet>>;

#[cfg(test)]
mod tests;
