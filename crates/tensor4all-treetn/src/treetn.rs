use petgraph::stable_graph::{NodeIndex, EdgeIndex};
use petgraph::visit::{Dfs, EdgeRef};
use std::collections::VecDeque;
use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;
use tensor4all_tensor::TensorDynLen;
use tensor4all_tensor::Storage;
use tensor4all_core::index::{Index, NoSymmSpace, Symmetry, DynId};
use tensor4all_core::index_ops::common_inds;
use crate::connection::Connection;
use crate::named_graph::NamedGraph;
use crate::site_index_network::SiteIndexNetwork;
use anyhow::{Result, Context};
use num_complex::Complex64;

/// Tree Tensor Network structure (inspired by ITensorNetworks.jl's TreeTensorNetwork).
///
/// Maintains a graph of tensors connected by bonds (edges).
/// Each node stores a tensor, and edges store `Connection` objects
/// that hold the Index objects on both sides of the bond.
///
/// The structure uses SiteIndexNetwork to manage:
/// - **Topology**: Graph structure (which nodes connect to which)
/// - **Site Space**: Physical indices organized by node
///
/// # Type Parameters
/// - `Id`: Index ID type
/// - `Symm`: Symmetry type (default: NoSymmSpace)
/// - `V`: Node name type for named nodes (default: NodeIndex for backward compatibility)
pub struct TreeTN<Id, Symm = NoSymmSpace, V = NodeIndex>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Named graph wrapper: provides mapping between node names (V) and NodeIndex
    graph: NamedGraph<V, TensorDynLen<Id, Symm>, Connection<Id, Symm>>,
    /// Orthogonalization region (ortho_region).
    /// When empty, the network is not orthogonalized.
    /// When non-empty, contains the node names (V) of the orthogonalization region.
    /// The region must form a connected subtree in the network.
    ortho_region: HashSet<V>,
    /// Site index network: manages topology and site space (physical indices).
    /// This structure enables topology and site space comparison independent of tensor data.
    site_index_network: SiteIndexNetwork<V, Id, Symm>,
}

impl<Id, Symm, V> TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a new empty TreeTN.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
            ortho_region: HashSet::new(),
            site_index_network: SiteIndexNetwork::new(),
        }
    }

    /// Add a tensor to the network with a node name.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    ///
    /// Also updates the site_index_network with the physical indices (all indices initially,
    /// as no connections exist yet).
    pub fn add_tensor_with_vertex(&mut self, node_name: V, tensor: TensorDynLen<Id, Symm>) -> Result<NodeIndex> {
        // Extract physical indices: initially all indices are physical (no connections yet)
        let physical_indices: HashSet<Index<Id, Symm>> = tensor.indices.iter().cloned().collect();
        
        // Add to graph
        let node_idx = self.graph.add_node(node_name.clone(), tensor)
            .map_err(|e| anyhow::anyhow!(e))?;
        
        // Add to site_index_network
        self.site_index_network.add_node(node_name, physical_indices)
            .map_err(|e| anyhow::anyhow!("Failed to add node to site_index_network: {}", e))?;
        
        Ok(node_idx)
    }

    /// Add a tensor to the network (backward compatibility for V = NodeIndex).
    ///
    /// This method only works when `V = NodeIndex`. For other node name types, use `add_tensor_with_vertex` instead.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    ///
    /// Also updates the site_index_network with the physical indices.
    pub fn add_tensor(&mut self, tensor: TensorDynLen<Id, Symm>) -> NodeIndex
    where
        V: From<NodeIndex> + Into<NodeIndex>,
    {
        // For V = NodeIndex, we need to add the node first to get a NodeIndex,
        // then use that NodeIndex as the node name
        // This is a bit awkward, but necessary for backward compatibility
        let node = self.graph.graph_mut().add_node(tensor);
        // Register the node as a named node (for V = NodeIndex, the node name is the same as the node)
        // We need to get the tensor back to register it, but we can't clone it
        // So we'll use a workaround: remove and re-add
        let tensor = self.graph.graph_mut().remove_node(node).unwrap();
        
        // Extract physical indices: initially all indices are physical (no connections yet)
        let physical_indices: HashSet<Index<Id, Symm>> = tensor.indices.iter().cloned().collect();
        
        // Add to graph with node name
        let _ = self.graph.add_node(V::from(node), tensor);
        
        // Add to site_index_network
        let _ = self.site_index_network.add_node(V::from(node), physical_indices);
        
        node
    }

    /// Get a reference to a tensor by NodeIndex.
    pub fn tensor(&self, node: NodeIndex) -> Option<&TensorDynLen<Id, Symm>> {
        self.graph.graph().node_weight(node)
    }

    /// Get a mutable reference to a tensor by NodeIndex.
    pub fn tensor_mut(&mut self, node: NodeIndex) -> Option<&mut TensorDynLen<Id, Symm>> {
        self.graph.graph_mut().node_weight_mut(node)
    }

    /// Replace a tensor at the given node with a new tensor.
    ///
    /// Validates that the new tensor contains all indices used in connections
    /// to this node. Returns an error if any connection index is missing.
    ///
    /// Returns the old tensor if the node exists and validation passes.
    pub fn replace_tensor(
        &mut self,
        node: NodeIndex,
        new_tensor: TensorDynLen<Id, Symm>,
    ) -> Result<Option<TensorDynLen<Id, Symm>>> {
        // Check if node exists
        if !self.graph.contains_node(node) {
            return Ok(None);
        }

        // Validate that all connection indices exist in the new tensor
        let edges = self.edges_for_node(node);
        let connection_indices: Vec<Index<Id, Symm>> = edges
            .iter()
            .map(|(edge_idx, _neighbor)| {
                self.edge_index_for_node(*edge_idx, node)
                    .context("Failed to get connection index for validation")
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .cloned()
            .collect();
        
        // Check if all connection indices are present in the new tensor
        let common = common_inds(&connection_indices, &new_tensor.indices);
        if common.len() != connection_indices.len() {
            return Err(anyhow::anyhow!(
                "New tensor is missing {} connection index(es): found {} out of {} required indices",
                connection_indices.len() - common.len(),
                common.len(),
                connection_indices.len()
            ))
            .context("replace_tensor: new tensor must contain all indices used in connections");
        }

        // Get node name for site_index_network update
        let node_name = self.graph.node_name(node)
            .ok_or_else(|| anyhow::anyhow!("Node name not found"))?
            .clone();

        // Calculate new physical indices: all indices minus connection indices
        let connection_indices_set: HashSet<Index<Id, Symm>> = connection_indices.iter().cloned().collect();
        let new_physical_indices: HashSet<Index<Id, Symm>> = new_tensor.indices
            .iter()
            .filter(|idx| !connection_indices_set.contains(idx))
            .cloned()
            .collect();

        // All validations passed, replace the tensor
        let old_tensor = self.graph.graph_mut().node_weight_mut(node).map(|old| {
            std::mem::replace(old, new_tensor)
        });

        // Update site_index_network with new physical indices
        if let Some(site_space) = self.site_index_network.site_space_mut(&node_name) {
            *site_space = new_physical_indices;
        }

        Ok(old_tensor)
    }

    /// Connect two tensors with a bond.
    ///
    /// The indices must exist in the respective tensors and have matching dimensions.
    /// The connection's `index_source` will correspond to `node_a` and `index_target` to `node_b`.
    pub fn connect(
        &mut self,
        node_a: NodeIndex,
        index_a: &Index<Id, Symm>,
        node_b: NodeIndex,
        index_b: &Index<Id, Symm>,
    ) -> Result<EdgeIndex> {
        // Validate that nodes exist
        if !self.graph.contains_node(node_a) || !self.graph.contains_node(node_b) {
            return Err(anyhow::anyhow!("One or both nodes do not exist"))
                .context("Failed to connect tensors");
        }

        // Validate that indices exist in respective tensors
        let tensor_a = self.tensor(node_a)
            .ok_or_else(|| anyhow::anyhow!("Tensor for node_a not found"))?;
        let tensor_b = self.tensor(node_b)
            .ok_or_else(|| anyhow::anyhow!("Tensor for node_b not found"))?;

        // Check that indices exist in tensors (by ID)
        let has_index_a = tensor_a.indices.iter().any(|idx| idx.id == index_a.id);
        let has_index_b = tensor_b.indices.iter().any(|idx| idx.id == index_b.id);

        if !has_index_a {
            return Err(anyhow::anyhow!("Index not found in tensor_a"))
                .context("Failed to connect: index_a must exist in tensor_a");
        }
        if !has_index_b {
            return Err(anyhow::anyhow!("Index not found in tensor_b"))
                .context("Failed to connect: index_b must exist in tensor_b");
        }

        // Clone indices for the connection
        // Find the actual Index objects from the tensors to preserve all metadata
        let index_a_clone = tensor_a.indices.iter()
            .find(|idx| idx.id == index_a.id)
            .unwrap()
            .clone();
        let index_b_clone = tensor_b.indices.iter()
            .find(|idx| idx.id == index_b.id)
            .unwrap()
            .clone();

        // Get node names for site_index_network (before mutable borrow)
        let node_name_a = self.graph.node_name(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_a not found"))?
            .clone();
        let node_name_b = self.graph.node_name(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node name for node_b not found"))?
            .clone();

        // Create connection
        // index_source corresponds to node_a, index_target to node_b
        let connection = Connection::new(index_a_clone.clone(), index_b_clone.clone())
            .context("Failed to create connection")?;

        // Add edge to graph
        // petgraph will assign source/target based on the order (a, b)
        let edge_idx = self.graph.graph_mut().add_edge(node_a, node_b, connection);
        
        // Add edge to site_index_network
        self.site_index_network.add_edge(&node_name_a, &node_name_b)
            .map_err(|e| anyhow::anyhow!("Failed to add edge to site_index_network: {}", e))?;
        
        // Update physical indices: remove connection indices from physical indices
        // Connection indices are no longer physical (they are bond indices)
        if let Some(site_space_a) = self.site_index_network.site_space_mut(&node_name_a) {
            site_space_a.remove(&index_a_clone);
        }
        if let Some(site_space_b) = self.site_index_network.site_space_mut(&node_name_b) {
            site_space_b.remove(&index_b_clone);
        }
        
        Ok(edge_idx)
    }

    /// Get a reference to a connection by EdgeIndex.
    pub fn connection(&self, edge: EdgeIndex) -> Option<&Connection<Id, Symm>> {
        self.graph.graph().edge_weight(edge)
    }

    /// Get a mutable reference to a connection by EdgeIndex.
    pub fn connection_mut(&mut self, edge: EdgeIndex) -> Option<&mut Connection<Id, Symm>> {
        self.graph.graph_mut().edge_weight_mut(edge)
    }

    /// Get the index on the side of the given node for an edge.
    ///
    /// Returns the Index that belongs to the specified node in the given edge.
    pub fn edge_index_for_node(
        &self,
        edge: EdgeIndex,
        node: NodeIndex,
    ) -> Result<&Index<Id, Symm>> {
        let (source, target) = self.graph.graph()
            .edge_endpoints(edge)
            .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;

        if node != source && node != target {
            return Err(anyhow::anyhow!("Node is not an endpoint of the edge"))
                .context("edge_index_for_node: node must be one of the edge endpoints");
        }

        let conn = self.connection(edge)
            .ok_or_else(|| anyhow::anyhow!("Connection not found"))?;

        // index_source corresponds to source, index_target to target
        if node == source {
            Ok(&conn.index_source)
        } else {
            Ok(&conn.index_target)
        }
    }

    /// Get all edges connected to a node.
    pub fn edges_for_node(&self, node: NodeIndex) -> Vec<(EdgeIndex, NodeIndex)> {
        self.graph.graph()
            .edges(node)
            .map(|edge| {
                let target = edge.target();
                (edge.id(), target)
            })
            .collect()
    }

    /// Replace the bond indices for an edge (e.g., after SVD creates new bond indices).
    ///
    /// The new indices must have matching dimensions and correspond to the edge endpoints.
    ///
    /// Also updates site_index_network: removes old bond indices from physical indices
    /// and adds new bond indices to physical indices (they become bond indices, not physical).
    pub fn replace_edge_bond(
        &mut self,
        edge: EdgeIndex,
        new_index_on_a: Index<Id, Symm>,
        new_index_on_b: Index<Id, Symm>,
    ) -> Result<()> {
        // Validate edge exists and get endpoints
        let (source, target) = self.graph.graph()
            .edge_endpoints(edge)
            .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;

        // Get old connection indices before updating
        let conn = self.connection(edge)
            .ok_or_else(|| anyhow::anyhow!("Connection not found"))?;
        let old_index_on_a = conn.index_source.clone();
        let old_index_on_b = conn.index_target.clone();

        // Get node names for site_index_network update
        let node_name_a = self.graph.node_name(source)
            .ok_or_else(|| anyhow::anyhow!("Node name for source not found"))?
            .clone();
        let node_name_b = self.graph.node_name(target)
            .ok_or_else(|| anyhow::anyhow!("Node name for target not found"))?
            .clone();

        // Update the connection indices
        // new_index_on_a corresponds to source, new_index_on_b to target
        let conn_mut = self.connection_mut(edge)
            .ok_or_else(|| anyhow::anyhow!("Connection not found"))?;
        conn_mut.replace_bond_indices(new_index_on_a.clone(), new_index_on_b.clone())
            .context("Failed to replace bond indices")?;

        // Update site_index_network: 
        // - Remove old bond indices from physical indices (they are no longer bond indices)
        // - Remove new bond indices from physical indices (they become bond indices)
        if let Some(site_space_a) = self.site_index_network.site_space_mut(&node_name_a) {
            // Old index is no longer a bond, so it becomes physical again
            site_space_a.insert(old_index_on_a);
            // New index becomes a bond, so remove it from physical
            site_space_a.remove(&new_index_on_a);
        }
        if let Some(site_space_b) = self.site_index_network.site_space_mut(&node_name_b) {
            // Old index is no longer a bond, so it becomes physical again
            site_space_b.insert(old_index_on_b);
            // New index becomes a bond, so remove it from physical
            site_space_b.remove(&new_index_on_b);
        }

        Ok(())
    }

    /// Set the orthogonalization direction for an edge.
    ///
    /// The direction must be one of the edge's indices (or None).
    /// The specified index must match either index_source or index_target of the connection.
    pub fn set_edge_ortho_towards(
        &mut self,
        edge: EdgeIndex,
        dir: Option<Index<Id, Symm>>,
    ) -> Result<()> {
        let conn = self.connection_mut(edge)
            .ok_or_else(|| anyhow::anyhow!("Connection not found"))?;

        // Validate that the specified index (if any) is one of the connection indices
        if let Some(ref ortho_idx) = dir {
            if ortho_idx.id != conn.index_source.id && ortho_idx.id != conn.index_target.id {
                return Err(anyhow::anyhow!(
                    "ortho_towards index must be either index_source or index_target of the connection"
                ))
                .context("set_edge_ortho_towards: invalid index");
            }
        }

        conn.set_ortho_towards(dir)
            .context("Failed to set ortho_towards")?;
        Ok(())
    }

    /// Get the node that corresponds to the orthogonalization direction.
    ///
    /// Returns None if ortho_towards is None, or the NodeIndex of the node
    /// that has the ortho_towards index.
    pub fn ortho_towards_node(
        &self,
        edge: EdgeIndex,
    ) -> Option<NodeIndex> {
        let conn = self.connection(edge)?;
        let ortho_idx = conn.ortho_towards.as_ref()?;
        
        let (source, target) = self.graph.graph().edge_endpoints(edge)?;
        
        // Check which node has the ortho_towards index
        if ortho_idx.id == conn.index_source.id {
            Some(source)
        } else if ortho_idx.id == conn.index_target.id {
            Some(target)
        } else {
            None
        }
    }

    /// Validate that the graph is a tree (or forest).
    ///
    /// Checks:
    /// - The graph is connected (all nodes reachable from the first node)
    /// - For each connected component: edges = nodes - 1 (tree condition)
    pub fn validate_tree(&self) -> Result<()> {
        let g = self.graph.graph();
        if g.node_count() == 0 {
            return Ok(()); // Empty graph is trivially valid
        }

        // Check if graph is connected
        let mut visited = std::collections::HashSet::new();
        let start_node = g.node_indices().next()
            .ok_or_else(|| anyhow::anyhow!("Graph has no nodes"))?;

        // DFS to count reachable nodes
        let mut dfs = Dfs::new(g, start_node);
        while let Some(node) = dfs.next(g) {
            visited.insert(node);
        }

        if visited.len() != g.node_count() {
            return Err(anyhow::anyhow!(
                "Graph is not connected: {} nodes reachable out of {}",
                visited.len(),
                g.node_count()
            ))
            .context("validate_tree: graph must be connected");
        }

        // Check tree condition: edges = nodes - 1
        let node_count = g.node_count();
        let edge_count = g.edge_count();

        if edge_count != node_count - 1 {
            return Err(anyhow::anyhow!(
                "Graph does not satisfy tree condition: {} edges != {} nodes - 1",
                edge_count,
                node_count
            ))
            .context("validate_tree: tree must have edges = nodes - 1");
        }

        Ok(())
    }

    /// Get the number of nodes in the network.
    pub fn node_count(&self) -> usize {
        self.graph.graph().node_count()
    }

    /// Get the number of edges in the network.
    pub fn edge_count(&self) -> usize {
        self.graph.graph().edge_count()
    }

    /// Get a reference to the orthogonalization region (using node names).
    ///
    /// When empty, the network is not orthogonalized.
    pub fn ortho_region(&self) -> &HashSet<V> {
        &self.ortho_region
    }

    /// Get a reference to the orthogonalization region (deprecated, NodeIndex-based).
    #[deprecated(note = "Use ortho_region() instead. For NodeIndex vertices, convert manually.")]
    pub fn auto_centers(&self) -> HashSet<NodeIndex>
    where
        V: Into<NodeIndex> + Clone,
    {
        // This is a compatibility method - it doesn't work for arbitrary V
        // For V = NodeIndex, we can provide a conversion
        // But for now, we'll just return an empty set as a placeholder
        // This method should be removed or properly implemented
        HashSet::new()
    }

    /// Check if the network is orthogonalized.
    ///
    /// Returns `true` if `ortho_region` is non-empty, `false` otherwise.
    pub fn is_orthogonalized(&self) -> bool {
        !self.ortho_region.is_empty()
    }

    /// Set the orthogonalization region (using node names).
    ///
    /// Validates that all specified nodes exist in the graph.
    pub fn set_ortho_region(&mut self, region: impl IntoIterator<Item = V>) -> Result<()> {
        let region: HashSet<V> = region.into_iter().collect();
        
        // Validate that all nodes exist in the graph
        for node_name in &region {
            if !self.graph.has_node(node_name) {
                return Err(anyhow::anyhow!("Node {:?} does not exist in the graph", node_name))
                    .context("set_ortho_region: all nodes must be valid");
            }
        }
        
        self.ortho_region = region;
        Ok(())
    }

    /// Set the orthogonalization region (deprecated, NodeIndex-based).
    #[deprecated(note = "Use set_ortho_region() with node names instead")]
    pub fn set_auto_centers(&mut self, centers: impl IntoIterator<Item = NodeIndex>) -> Result<()>
    where
        V: From<NodeIndex>,
    {
        let region: HashSet<V> = centers.into_iter().map(V::from).collect();
        self.set_ortho_region(region)
    }

    /// Clear the orthogonalization region (mark network as not orthogonalized).
    pub fn clear_ortho_region(&mut self) {
        self.ortho_region.clear();
    }

    /// Clear the orthogonalization region (deprecated).
    #[deprecated(note = "Use clear_ortho_region() instead")]
    pub fn clear_auto_centers(&mut self) {
        self.clear_ortho_region();
    }

    /// Add a node to the orthogonalization region.
    ///
    /// Validates that the node exists in the graph.
    pub fn add_to_ortho_region(&mut self, node_name: V) -> Result<()> {
        if !self.graph.has_node(&node_name) {
            return Err(anyhow::anyhow!("Node {:?} does not exist in the graph", node_name))
                .context("add_to_ortho_region: node must be valid");
        }
        self.ortho_region.insert(node_name);
        Ok(())
    }

    /// Add a node to the orthogonalization region (deprecated, NodeIndex-based).
    #[deprecated(note = "Use add_to_ortho_region() with node name instead")]
    pub fn add_auto_center(&mut self, center: NodeIndex) -> Result<()>
    where
        V: From<NodeIndex>,
    {
        self.add_to_ortho_region(V::from(center))
    }

    /// Remove a node from the orthogonalization region.
    ///
    /// Returns `true` if the node was in the region, `false` otherwise.
    pub fn remove_from_ortho_region(&mut self, node_name: &V) -> bool {
        self.ortho_region.remove(node_name)
    }

    /// Remove a node from the orthogonalization region (deprecated, NodeIndex-based).
    #[deprecated(note = "Use remove_from_ortho_region() with node name instead")]
    pub fn remove_auto_center(&mut self, center: NodeIndex)
    where
        V: From<NodeIndex> + PartialEq,
    {
        let node_name = V::from(center);
        self.remove_from_ortho_region(&node_name);
    }

    /// Get a reference to the physical indices manager (site space).
    #[deprecated(note = "Use site_index_network() instead")]
    pub fn physical_indices(&self) -> &SiteIndexNetwork<V, Id, Symm> {
        &self.site_index_network
    }

    /// Get a mutable reference to the physical indices manager (site space).
    #[deprecated(note = "Use site_index_network_mut() instead")]
    pub fn physical_indices_mut(&mut self) -> &mut SiteIndexNetwork<V, Id, Symm> {
        &mut self.site_index_network
    }

    /// Get a reference to the site index network.
    ///
    /// The site index network contains both topology (graph structure) and site space (physical indices).
    pub fn site_index_network(&self) -> &SiteIndexNetwork<V, Id, Symm> {
        &self.site_index_network
    }

    /// Get a mutable reference to the site index network.
    pub fn site_index_network_mut(&mut self) -> &mut SiteIndexNetwork<V, Id, Symm> {
        &mut self.site_index_network
    }

    /// Get a reference to the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &V) -> Option<&std::collections::HashSet<Index<Id, Symm>>> {
        self.site_index_network.site_space(node_name)
    }

    /// Get a mutable reference to the site space (physical indices) for a node.
    pub fn site_space_mut(&mut self, node_name: &V) -> Option<&mut std::collections::HashSet<Index<Id, Symm>>> {
        self.site_index_network.site_space_mut(node_name)
    }

    /// Check if two TreeTN can be added together.
    ///
    /// Two TreeTN can be added if:
    /// - Site index networks are compatible (same topology and site space)
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to check compatibility with
    ///
    /// # Returns
    /// `true` if the networks can be added, `false` otherwise.
    pub fn can_add(&self, other: &Self) -> bool
    where
        Id: Ord,
    {
        // Check site index network compatibility (includes both topology and site space)
        self.site_index_network.is_compatible(&other.site_index_network)
    }

    /// Add two TreeTN together (element-wise addition of tensors).
    ///
    /// This method adds corresponding tensors element-wise. The networks must have
    /// compatible topologies and identical site spaces.
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to add
    ///
    /// # Returns
    /// A new TreeTN with added tensors, or an error if the networks are incompatible.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Topologies are not compatible
    /// - Site spaces are not equal
    /// - Tensor addition fails (e.g., dimension mismatch)
    pub fn add(self, other: Self) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        // Validate compatibility
        if !self.can_add(&other) {
            return Err(anyhow::anyhow!(
                "Cannot add TreeTN: topologies or site spaces are incompatible"
            ));
        }

        // TODO: Implement tensor addition
        // This requires:
        // 1. Matching tensors by tensor_id across both networks
        // 2. Element-wise addition of tensor data
        // 3. Preserving topology and site_space
        // 
        // For now, return an error indicating this is not yet implemented.
        // The structure is in place (can_add() works), but actual tensor addition
        // requires TensorDynLen to support addition operations.
        Err(anyhow::anyhow!(
            "Tensor addition not yet implemented. This requires TensorDynLen::add() method or similar."
        ))
    }

    /// Validate that `ortho_region` and edge `ortho_towards` are consistent.
    ///
    /// Rules:
    /// - If `ortho_region` is empty (not orthogonalized), all edges must have `ortho_towards == None`.
    /// - If `ortho_region` is non-empty, it must be connected in the tree (forms a subtree).
    /// - `ortho_towards == None` is allowed **only** when both edge endpoints are in `ortho_region`.
    /// - For any edge with at least one endpoint outside `ortho_region`, `ortho_towards` must be `Some(...)`
    ///   and must point towards the endpoint with smaller distance to `ortho_region`.
    pub fn validate_ortho_consistency(&self) -> Result<()> {
        // If not orthogonalized, require no edge directions.
        if self.ortho_region.is_empty() {
            for e in self.graph.graph().edge_indices() {
                let conn = self.connection(e).ok_or_else(|| anyhow::anyhow!("Connection not found"))?;
                if conn.ortho_towards.is_some() {
                    return Err(anyhow::anyhow!(
                        "Found ortho_towards on edge {:?} but ortho_region is empty",
                        e
                    ))
                    .context("validate_ortho_consistency: ortho_region empty implies no directions");
                }
            }
            return Ok(());
        }

        // Validate all ortho_region exist and convert to NodeIndex for graph operations
        let g = self.graph.graph();
        let mut ortho_nodes = HashSet::new();
        for c in &self.ortho_region {
            if !self.graph.has_node(c) {
                return Err(anyhow::anyhow!("ortho_center {:?} does not exist in graph", c))
                    .context("validate_ortho_consistency: all ortho_region must be valid nodes");
            }
            if let Some(node) = self.graph.node_index(c) {
                ortho_nodes.insert(node);
            }
        }

        // Check ortho_region connectivity in the induced subgraph.
        let start_node = ortho_nodes.iter().next()
            .ok_or_else(|| anyhow::anyhow!("ortho_region unexpectedly empty"))?;
        let mut stack = vec![*start_node];
        let mut seen = HashSet::new();
        seen.insert(*start_node);
        while let Some(v) = stack.pop() {
            for nb in g.neighbors(v) {
                if ortho_nodes.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        if seen.len() != ortho_nodes.len() {
            return Err(anyhow::anyhow!(
                "ortho_region is not connected: reached {} out of {} centers",
                seen.len(),
                ortho_nodes.len()
            ))
            .context("validate_ortho_consistency: ortho_region must form a connected subtree");
        }

        // Multi-source BFS distances to ortho_region.
        let mut dist: std::collections::HashMap<NodeIndex, usize> = std::collections::HashMap::new();
        let mut q = VecDeque::new();
        for c_node in &ortho_nodes {
            dist.insert(*c_node, 0);
            q.push_back(*c_node);
        }
        while let Some(v) = q.pop_front() {
            let dv = dist[&v];
            for nb in g.neighbors(v) {
                if !dist.contains_key(&nb) {
                    dist.insert(nb, dv + 1);
                    q.push_back(nb);
                }
            }
        }

        // Check each edge.
        for e in g.edge_indices() {
            let (source, target) = g
                .edge_endpoints(e)
                .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;
            let conn = self.connection(e).ok_or_else(|| anyhow::anyhow!("Connection not found"))?;

            // Check if source and target nodes are in ortho_region
            let s_node_name = self.graph.node_name(source);
            let t_node_name = self.graph.node_name(target);
            let s_in = s_node_name.map(|v| self.ortho_region.contains(v)).unwrap_or(false);
            let t_in = t_node_name.map(|v| self.ortho_region.contains(v)).unwrap_or(false);

            // None allowed only when both endpoints are centers.
            if conn.ortho_towards.is_none() {
                if !(s_in && t_in) {
                    return Err(anyhow::anyhow!(
                        "ortho_towards is None on edge {:?} but not both endpoints are in ortho_region",
                        e
                    ))
                    .context("validate_ortho_consistency: None allowed only inside ortho_region");
                }
                continue;
            }

            // From here on, ortho_towards must be Some.
            let ortho_idx = conn.ortho_towards.as_ref().unwrap();

            // Distances must exist.
            let ds = *dist
                .get(&source)
                .ok_or_else(|| anyhow::anyhow!("No distance for source node {:?}", source))?;
            let dt = *dist
                .get(&target)
                .ok_or_else(|| anyhow::anyhow!("No distance for target node {:?}", target))?;

            // Choose expected direction: towards smaller distance (closer to ortho_region).
            // In a tree with connected ortho_region, adjacent nodes should differ by exactly 1
            // unless both are in ortho_region (handled above).
            if ds == dt {
                return Err(anyhow::anyhow!(
                    "Ambiguous direction on edge {:?}: distances are equal ({} == {})",
                    e,
                    ds,
                    dt
                ))
                .context("validate_ortho_consistency: outside ortho_region, distances should differ");
            }

            let expect_source_side = ds < dt;
            let expected_id = if expect_source_side {
                &conn.index_source.id
            } else {
                &conn.index_target.id
            };

            if &ortho_idx.id != expected_id {
                return Err(anyhow::anyhow!(
                    "Invalid ortho_towards on edge {:?}: expected to point to {} side",
                    e,
                    if expect_source_side { "source" } else { "target" }
                ))
                .context("validate_ortho_consistency: ortho_towards must point towards ortho_region");
            }

            // Additionally enforce that boundary edges (one in centers, one out) always point into centers.
            if s_in ^ t_in {
                let expected_into_centers_id = if s_in { &conn.index_source.id } else { &conn.index_target.id };
                if &ortho_idx.id != expected_into_centers_id {
                    return Err(anyhow::anyhow!(
                        "Boundary edge {:?} must point into ortho_region",
                        e
                    ))
                    .context("validate_ortho_consistency: boundary must point into ortho_region");
                }
            }
        }

        Ok(())
    }

    /// Orthogonalize the network using QR decomposition towards the specified ortho_region.
    ///
    /// This method consumes the TreeTN and returns a new orthogonalized TreeTN.
    /// The algorithm:
    /// 1. Validates that the graph is a tree
    /// 2. Sets the ortho_region and validates connectivity
    /// 3. Computes distances from ortho_region using BFS
    /// 4. Processes nodes in order of decreasing distance (farthest first)
    /// 5. For each node, performs QR decomposition on edges pointing towards ortho_region
    /// 6. Absorbs R factors into parent nodes using contract_pairs
    ///
    /// # Arguments
    /// * `ortho_region` - The nodes that will serve as orthogonalization centers
    ///
    /// # Returns
    /// A new orthogonalized TreeTN, or an error if validation fails or QR decomposition fails.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The graph is not a tree
    /// - ortho_region are not connected
    /// - QR decomposition fails
    /// - Tensor storage types are not DenseF64 or DenseC64
    pub fn orthogonalize_with_qr(
        mut self,
        ortho_region: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: From<NodeIndex>,
    {
        use tensor4all_linalg::qr;

        // 1. Validate tree structure
        self.validate_tree()
            .context("orthogonalize_with_qr: graph must be a tree")?;

        // 2. Set ortho_region and validate connectivity
        // Convert NodeIndex to V
        let ortho_region_v: Vec<V> = ortho_region.into_iter()
            .map(V::from)
            .collect();
        self.set_ortho_region(ortho_region_v)
            .context("orthogonalize_with_qr: failed to set ortho_region")?;

        if self.ortho_region.is_empty() {
            return Ok(self); // Nothing to do if no centers
        }

        // Validate ortho_region connectivity (similar to validate_ortho_consistency)
        let g = self.graph.graph();
        let start_vertex = self.ortho_region.iter().next()
            .ok_or_else(|| anyhow::anyhow!("ortho_region unexpectedly empty"))?;
        let start_node = self.graph.node_index(start_vertex)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in graph", start_vertex))?;
        let mut stack = vec![start_node];
        let mut seen_nodes = HashSet::new();
        let mut seen_vertices = HashSet::new();
        seen_nodes.insert(start_node);
        seen_vertices.insert(start_vertex);
        while let Some(v_node) = stack.pop() {
            for nb_node in g.neighbors(v_node) {
                if let Some(nb_node_name) = self.graph.node_name(nb_node) {
                    if self.ortho_region.contains(nb_node_name) && seen_nodes.insert(nb_node) {
                        seen_vertices.insert(nb_node_name);
                        stack.push(nb_node);
                    }
                }
            }
        }
        if seen_vertices.len() != self.ortho_region.len() {
            return Err(anyhow::anyhow!(
                "ortho_region is not connected: reached {} out of {} centers",
                seen_vertices.len(),
                self.ortho_region.len()
            ))
            .context("orthogonalize_with_qr: ortho_region must form a connected subtree");
        }

        // 3. Multi-source BFS to compute distances from ortho_region
        let mut dist: HashMap<NodeIndex, usize> = HashMap::new();
        let mut q = VecDeque::new();
        for c in &self.ortho_region {
            if let Some(c_node) = self.graph.node_index(c) {
                dist.insert(c_node, 0);
                q.push_back(c_node);
            }
        }
        {
            let g = self.graph.graph();
            while let Some(v) = q.pop_front() {
                let dv = dist[&v];
                for nb in g.neighbors(v) {
                    if !dist.contains_key(&nb) {
                        dist.insert(nb, dv + 1);
                        q.push_back(nb);
                    }
                }
            }
        }

        // 4. Process nodes in order of decreasing distance (farthest first)
        let mut nodes_by_distance: Vec<(NodeIndex, usize)> = dist
            .iter()
            .map(|(&node, &d)| (node, d))
            .collect();
        nodes_by_distance.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by distance

        for (v, _dv) in nodes_by_distance {
            // Skip nodes in ortho_region (they are already at distance 0)
            let v_node_name = self.graph.node_name(v);
            if v_node_name.map(|v| self.ortho_region.contains(v)).unwrap_or(false) {
                continue;
            }

            // Find parent: neighbor with distance one less
            let v_dist = dist[&v];
            let parent_dist = v_dist.checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} at distance 0 but not in ortho_region", v))
                .context("orthogonalize_with_qr: distance calculation error")?;

            let (parent, edge) = {
                let g = self.graph.graph();
                let parent = g
                    .neighbors(v)
                    .find(|&nb| {
                        dist.get(&nb).map(|&d| d == parent_dist).unwrap_or(false)
                    })
                    .ok_or_else(|| anyhow::anyhow!(
                        "Node {:?} at distance {} has no parent (neighbor with distance {})",
                        v,
                        v_dist,
                        parent_dist
                    ))
                    .context("orthogonalize_with_qr: tree structure violation")?;

                // Find the edge between v and parent
                let edge = g
                    .edges_connecting(v, parent)
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("No edge found between node {:?} and parent {:?}", v, parent))
                    .context("orthogonalize_with_qr: edge not found")?
                    .id();
                (parent, edge)
            };

            // Get the bond index on v-side corresponding to the parent edge.
            // We will place this index on the RIGHT side of the QR unfolding so that
            // R carries this bond and can be absorbed into the parent tensor.
            let parent_bond_v = self
                .edge_index_for_node(edge, v)
                .context("orthogonalize_with_qr: failed to get parent bond index on v")?
                .clone();

            // Get the tensor at node v (reference)
            let tensor_v = self
                .tensor(v)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", v))
                .context("orthogonalize_with_qr: tensor not found")?;

            // Build left_inds = all indices of tensor_v EXCEPT the parent bond.
            let left_inds: Vec<Index<Id, Symm>> = tensor_v
                .indices
                .iter()
                .filter(|idx| idx.id != parent_bond_v.id)
                .cloned()
                .collect();
            if left_inds.is_empty() || left_inds.len() == tensor_v.indices.len() {
                return Err(anyhow::anyhow!(
                    "Cannot QR-orthogonalize node {:?}: need at least one left index and at least one right index",
                    v
                ))
                .context("orthogonalize_with_qr: invalid tensor rank for QR");
            }

            // Determine storage type and perform QR decomposition
            let (q_tensor, r_tensor) = match tensor_v.storage.as_ref() {
                Storage::DenseF64(_) => qr::<Id, Symm, f64>(tensor_v, &left_inds)
                    .map_err(|e| anyhow::anyhow!("QR decomposition failed: {}", e))
                    .context("orthogonalize_with_qr: QR decomposition failed for f64")?,
                Storage::DenseC64(_) => qr::<Id, Symm, Complex64>(tensor_v, &left_inds)
                    .map_err(|e| anyhow::anyhow!("QR decomposition failed: {}", e))
                    .context("orthogonalize_with_qr: QR decomposition failed for Complex64")?,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported storage type for QR decomposition (only DenseF64 and DenseC64 are supported)"
                    ))
                    .context("orthogonalize_with_qr: unsupported storage type");
                }
            };

            // In this split, R must contain the parent bond (as part of its right indices).
            // We will absorb R into the parent along (edge_index_parent, parent_bond_v).
            let edge_index_parent = self
                .edge_index_for_node(edge, parent)
                .context("orthogonalize_with_qr: failed to get edge index for parent")?
                .clone();

            let parent_tensor = self
                .tensor(parent)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for parent node {:?}", parent))
                .context("orthogonalize_with_qr: parent tensor not found")?;

            let updated_parent_tensor = parent_tensor
                .contract_pairs(&r_tensor, &[(edge_index_parent.clone(), parent_bond_v.clone())])
                .context("orthogonalize_with_qr: failed to absorb R into parent tensor")?;

            // The new bond index is the QR-created bond shared between Q and R.
            // It is always the last index of Q and the first index of R.
            let new_bond_index = q_tensor
                .indices
                .last()
                .ok_or_else(|| anyhow::anyhow!("Q tensor has no indices"))?
                .clone();

            // Update the connection bond indices FIRST, so replace_tensor validation matches.
            self.replace_edge_bond(edge, new_bond_index.clone(), new_bond_index.clone())
                .context("orthogonalize_with_qr: failed to update edge bond indices")?;

            // Now update tensors. These validations should pass because the edge expects new_bond_index.
            self.replace_tensor(v, q_tensor)
                .context("orthogonalize_with_qr: failed to replace tensor at node v")?;
            self.replace_tensor(parent, updated_parent_tensor)
                .context("orthogonalize_with_qr: failed to replace tensor at parent node")?;

            // Set ortho_towards to point towards parent (ortho_region direction)
            let ortho_towards_index = self
                .edge_index_for_node(edge, parent)
                .context("orthogonalize_with_qr: failed to get ortho_towards index for parent")?
                .clone();
            self.set_edge_ortho_towards(edge, Some(ortho_towards_index))
                .context("orthogonalize_with_qr: failed to set ortho_towards")?;
        }

        Ok(self)
    }
}

impl<Id, Symm, V> Default for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}


