use petgraph::stable_graph::{NodeIndex, EdgeIndex};
use petgraph::visit::{Dfs, EdgeRef};
use std::collections::VecDeque;
use std::collections::HashSet;
use std::collections::HashMap;
use std::hash::Hash;
use tensor4all::TensorDynLen;
use tensor4all::Storage;
use tensor4all::index::{Index, NoSymmSpace, Symmetry, DynId};
use tensor4all::index_ops::common_inds;
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

    /// Create a TreeTN from a list of tensors and node names using einsum rule.
    ///
    /// This function connects tensors that share common indices (by ID).
    /// The algorithm is O(n) where n is the number of tensors:
    /// 1. Add all tensors as nodes
    /// 2. Build a map from index ID to (node, index) pairs in a single pass
    /// 3. Connect nodes that share the same index ID
    ///
    /// # Arguments
    /// * `tensors` - Vector of tensors to add to the network
    /// * `node_names` - Vector of node names corresponding to each tensor
    ///
    /// # Returns
    /// A new TreeTN with tensors connected by common indices, or an error if:
    /// - The lengths of `tensors` and `node_names` don't match
    /// - An index ID appears in more than 2 tensors (TreeTN is a tree, so each bond connects exactly 2 nodes)
    /// - Connection fails (e.g., dimension mismatch)
    ///
    /// # Errors
    /// Returns an error if validation fails or connection fails.
    pub fn from_tensors_with_names(
        tensors: Vec<TensorDynLen<Id, Symm>>,
        node_names: Vec<V>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug,
    {
        // Validate input lengths
        if tensors.len() != node_names.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} tensors but {} node names",
                tensors.len(),
                node_names.len()
            ))
            .context("from_tensors_with_names: tensors and node_names must have the same length");
        }

        // Create empty TreeTN
        let mut treetn = Self::new();

        // Step 1: Add all tensors as nodes and collect NodeIndex mappings
        let mut node_indices = Vec::with_capacity(tensors.len());
        for (tensor, node_name) in tensors.into_iter().zip(node_names.into_iter()) {
            let node_idx = treetn.add_tensor_with_vertex(node_name, tensor)?;
            node_indices.push(node_idx);
        }

        // Step 2: Build a map from index ID to (node_index, index) pairs in O(n) time
        // Key: index ID, Value: vector of (NodeIndex, Index) pairs
        let mut index_map: HashMap<Id, Vec<(NodeIndex, Index<Id, Symm>)>> = HashMap::new();
        
        for node_idx in &node_indices {
            let tensor = treetn.tensor(*node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_idx))?;
            
            for index in &tensor.indices {
                index_map
                    .entry(index.id.clone())
                    .or_insert_with(Vec::new)
                    .push((*node_idx, index.clone()));
            }
        }

        // Step 3: Connect nodes that share the same index ID
        // For TreeTN (tree structure), each index ID should appear in exactly 2 tensors
        for (index_id, nodes_with_index) in index_map {
            match nodes_with_index.len() {
                0 => unreachable!(),
                1 => {
                    // Index appears in only one tensor - this is a physical index, no connection needed
                    continue;
                }
                2 => {
                    // Index appears in exactly 2 tensors - connect them
                    let (node_a, index_a) = &nodes_with_index[0];
                    let (node_b, index_b) = &nodes_with_index[1];
                    
                    treetn.connect(*node_a, index_a, *node_b, index_b)
                        .with_context(|| format!(
                            "Failed to connect nodes {:?} and {:?} via index ID {:?}",
                            node_a, node_b, index_id
                        ))?;
                }
                n => {
                    // Index appears in more than 2 tensors - this violates tree structure
                    return Err(anyhow::anyhow!(
                        "Index ID {:?} appears in {} tensors, but TreeTN requires exactly 2 (tree structure)",
                        index_id, n
                    ))
                    .context("from_tensors_with_names: each bond index must connect exactly 2 nodes");
                }
            }
        }

        Ok(treetn)
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

    /// Get all node indices in the tree tensor network.
    pub fn node_indices(&self) -> Vec<NodeIndex> {
        self.graph.graph().node_indices().collect()
    }

    /// Get all vertex names in the tree tensor network.
    pub fn vertex_names(&self) -> Vec<V> {
        self.graph.graph().node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .collect()
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

    /// Add two TreeTN together using direct-sum (block) construction.
    ///
    /// This method constructs a new TTN whose bond indices are the **direct sums**
    /// of the original bond indices, so that the resulting network represents the
    /// exact sum of the two input networks.
    ///
    /// # Algorithm
    ///
    /// For each edge `e`, create a new bond index with dimension `dA(e) + dB(e)`.
    /// For each vertex `v`, create a new tensor that embeds:
    /// - `T_A(v)` in the "A block" (first part of each bond dimension)
    /// - `T_B(v)` in the "B block" (second part of each bond dimension)
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to add
    ///
    /// # Returns
    /// A new TreeTN representing `self + other`, or an error if the networks are incompatible.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Topologies are not compatible
    /// - Site spaces are not equal
    /// - Storage types are not dense (only DenseF64 and DenseC64 are supported)
    ///
    /// # Notes
    /// - The result is not orthogonalized; `ortho_region` is cleared.
    /// - Bond dimensions increase: new_dim = dim_A + dim_B.
    /// - Only dense storage (DenseF64/DenseC64) is currently supported.
    pub fn add(self, other: Self) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + Ord + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
        V: Ord,
    {
        use tensor4all::storage::{DenseStorageF64, DenseStorageC64};

        // Validate compatibility
        if !self.can_add(&other) {
            return Err(anyhow::anyhow!(
                "Cannot add TreeTN: topologies or site spaces are incompatible"
            ));
        }

        if self.node_count() == 0 {
            return Ok(other);
        }
        if other.node_count() == 0 {
            return Ok(self);
        }

        // Build a mapping from V -> NodeIndex for both networks
        let node_names: Vec<V> = {
            let mut names: Vec<V> = self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .collect();
            names.sort();
            names
        };

        // Build edge info: for each edge in self, store (src_name, tgt_name, bond_dim_A, bond_dim_B, shared_index)
        // Use a SINGLE shared index for both endpoints of each edge (fix for issue #2)
        let mut edge_info: HashMap<(V, V), (usize, usize, Index<Id, Symm>)> = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (src, tgt) = self.graph.graph().edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge has no endpoints"))?;
            let conn_a = self.connection(edge)
                .ok_or_else(|| anyhow::anyhow!("Connection not found in self"))?;
            let bond_dim_a = conn_a.bond_dim();

            let src_name = self.graph.node_name(src)
                .ok_or_else(|| anyhow::anyhow!("Source node name not found"))?
                .clone();
            let tgt_name = self.graph.node_name(tgt)
                .ok_or_else(|| anyhow::anyhow!("Target node name not found"))?
                .clone();

            // Find corresponding edge in other
            let src_idx_other = other.graph.node_index(&src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in other"))?;
            let tgt_idx_other = other.graph.node_index(&tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in other"))?;

            // Find edge between these nodes in other
            let edge_other = other.graph.graph().edges_connecting(src_idx_other, tgt_idx_other)
                .next()
                .or_else(|| other.graph.graph().edges_connecting(tgt_idx_other, src_idx_other).next())
                .ok_or_else(|| anyhow::anyhow!("Edge not found in other"))?;
            let conn_b = other.connection(edge_other.id())
                .ok_or_else(|| anyhow::anyhow!("Connection not found in other"))?;
            let bond_dim_b = conn_b.bond_dim();

            // Create ONE shared bond index for both endpoints (same ID)
            let new_dim = bond_dim_a + bond_dim_b;
            let shared_index = Index::<Id, Symm>::new_link(new_dim)
                .map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))?;

            // Store in canonical order (smaller name first)
            let key = if src_name < tgt_name {
                (src_name.clone(), tgt_name.clone())
            } else {
                (tgt_name.clone(), src_name.clone())
            };
            edge_info.insert(key, (bond_dim_a, bond_dim_b, shared_index));
        }

        // Create new TreeTN
        let mut result = Self::new();

        // Process each node
        for node_name in &node_names {
            let node_idx_a = self.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in self"))?;
            let node_idx_b = other.graph.node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node not found in other"))?;

            let tensor_a = self.tensor(node_idx_a)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in self"))?;
            let tensor_b = other.tensor(node_idx_b)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in other"))?;

            // Get physical indices for this node (from site_index_network)
            let physical_indices_a: HashSet<&Index<Id, Symm>> = self.site_space(node_name)
                .map(|s| s.iter().collect())
                .unwrap_or_default();

            // Separate physical and bond indices for tensor A, preserving order
            let mut physical_inds_a: Vec<Index<Id, Symm>> = Vec::new();
            let mut bond_inds_a: Vec<(Index<Id, Symm>, V, usize)> = Vec::new(); // (index, neighbor_name, original_position)

            for (pos, idx) in tensor_a.indices.iter().enumerate() {
                if physical_indices_a.contains(idx) {
                    physical_inds_a.push(idx.clone());
                } else {
                    // This is a bond index - find which neighbor it connects to
                    for edge in self.edges_for_node(node_idx_a) {
                        if let Ok(edge_idx) = self.edge_index_for_node(edge.0, node_idx_a) {
                            if edge_idx.id == idx.id {
                                let neighbor_name = self.graph.node_name(edge.1)
                                    .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                                    .clone();
                                bond_inds_a.push((idx.clone(), neighbor_name, pos));
                                break;
                            }
                        }
                    }
                }
            }

            // Build the canonical index order: physical indices first (in original order), then bonds (sorted by neighbor)
            let mut canonical_indices: Vec<Index<Id, Symm>> = physical_inds_a.clone();
            let mut canonical_dims: Vec<usize> = physical_inds_a.iter()
                .map(|idx| idx.size())
                .collect();

            // Sort bonds by neighbor name for deterministic ordering
            let mut sorted_bonds = bond_inds_a.clone();
            sorted_bonds.sort_by(|a, b| a.1.cmp(&b.1));

            // Build axis mapping: for each axis in canonical order, what's the corresponding new dim?
            // And track bond positions for embedding
            let mut bond_axis_info: Vec<(usize, usize, usize)> = Vec::new(); // (canonical_axis, dim_a, dim_b)

            for (_old_idx, neighbor_name, _orig_pos) in &sorted_bonds {
                let key = if *node_name < *neighbor_name {
                    (node_name.clone(), neighbor_name.clone())
                } else {
                    (neighbor_name.clone(), node_name.clone())
                };
                let (dim_a, dim_b, shared_idx) = edge_info.get(&key)
                    .ok_or_else(|| anyhow::anyhow!("Edge info not found"))?;

                bond_axis_info.push((canonical_indices.len(), *dim_a, *dim_b));
                canonical_indices.push(shared_idx.clone());
                canonical_dims.push(dim_a + dim_b);
            }

            // Now we need to permute tensor_a and tensor_b to match canonical order
            // Build permutation for tensor_a
            let mut perm_a: Vec<usize> = Vec::new();
            // First, add physical indices in the order they appear in physical_inds_a
            for phys_idx in &physical_inds_a {
                let pos = tensor_a.indices.iter().position(|i| i.id == phys_idx.id)
                    .ok_or_else(|| anyhow::anyhow!("Physical index not found in tensor_a"))?;
                perm_a.push(pos);
            }
            // Then add bond indices in sorted order
            for (_old_idx, neighbor_name, _) in &sorted_bonds {
                // Find the original position of this bond in tensor_a
                for (_orig_bond_idx, orig_neighbor, orig_pos) in &bond_inds_a {
                    if orig_neighbor == neighbor_name {
                        perm_a.push(*orig_pos);
                        break;
                    }
                }
            }

            // Create permuted version of tensor_a storage
            let permuted_dims_a: Vec<usize> = perm_a.iter().map(|&i| tensor_a.dims[i]).collect();

            // Do the same for tensor_b
            let physical_indices_b: HashSet<&Index<Id, Symm>> = other.site_space(node_name)
                .map(|s| s.iter().collect())
                .unwrap_or_default();

            let mut bond_inds_b: Vec<(Index<Id, Symm>, V, usize)> = Vec::new();
            for (pos, idx) in tensor_b.indices.iter().enumerate() {
                if !physical_indices_b.contains(idx) {
                    for edge in other.edges_for_node(node_idx_b) {
                        if let Ok(edge_idx) = other.edge_index_for_node(edge.0, node_idx_b) {
                            if edge_idx.id == idx.id {
                                let neighbor_name = other.graph.node_name(edge.1)
                                    .ok_or_else(|| anyhow::anyhow!("Neighbor name not found"))?
                                    .clone();
                                bond_inds_b.push((idx.clone(), neighbor_name, pos));
                                break;
                            }
                        }
                    }
                }
            }

            let mut perm_b: Vec<usize> = Vec::new();
            // Physical indices in same order as physical_inds_a (matched by ID in site_space)
            for phys_idx in &physical_inds_a {
                // Find matching physical index in tensor_b by ID
                let pos = tensor_b.indices.iter().position(|i| {
                    physical_indices_b.contains(i) &&
                    self.site_space(node_name).map_or(false, |s| s.iter().any(|si| si.id == i.id && si.id == phys_idx.id))
                }).or_else(|| {
                    // Fallback: match by position in physical index set
                    tensor_b.indices.iter().position(|i| physical_indices_b.contains(i))
                }).ok_or_else(|| anyhow::anyhow!("Physical index not found in tensor_b"))?;
                if !perm_b.contains(&pos) {
                    perm_b.push(pos);
                }
            }
            // Bond indices in sorted order (by neighbor name)
            for (_old_idx, neighbor_name, _) in &sorted_bonds {
                for (_, orig_neighbor, orig_pos) in &bond_inds_b {
                    if orig_neighbor == neighbor_name {
                        perm_b.push(*orig_pos);
                        break;
                    }
                }
            }

            let permuted_dims_b: Vec<usize> = perm_b.iter().map(|&i| tensor_b.dims[i]).collect();

            // Create the new tensor storage
            let total_size: usize = canonical_dims.iter().product();
            let is_complex = matches!(tensor_a.storage.as_ref(), Storage::DenseC64(_))
                || matches!(tensor_b.storage.as_ref(), Storage::DenseC64(_));

            // If there are no bonds for this node, add tensors element-wise.
            // Block embedding only makes sense when there are bond dimensions to separate the blocks.
            let has_bonds = !bond_axis_info.is_empty();

            let new_storage = if is_complex {
                let data_a = match tensor_a.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>(),
                    Storage::DenseC64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };
                let data_b = match tensor_b.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>(),
                    Storage::DenseC64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };

                // Permute data before embedding/adding
                let permuted_data_a = permute_data_c64(&data_a, &tensor_a.dims, &perm_a);
                let permuted_data_b = permute_data_c64(&data_b, &tensor_b.dims, &perm_b);

                if has_bonds {
                    // Use block embedding for nodes with bonds
                    let mut result_data = vec![Complex64::new(0.0, 0.0); total_size];
                    embed_block_c64(&mut result_data, &canonical_dims, &permuted_data_a, &permuted_dims_a, &bond_axis_info, true)?;
                    embed_block_c64(&mut result_data, &canonical_dims, &permuted_data_b, &permuted_dims_b, &bond_axis_info, false)?;
                    Storage::DenseC64(DenseStorageC64::from_vec(result_data))
                } else {
                    // No bonds: add tensors element-wise
                    let result_data: Vec<Complex64> = permuted_data_a.iter()
                        .zip(permuted_data_b.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    Storage::DenseC64(DenseStorageC64::from_vec(result_data))
                }
            } else {
                let data_a = match tensor_a.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };
                let data_b = match tensor_b.storage.as_ref() {
                    Storage::DenseF64(d) => d.as_slice().to_vec(),
                    _ => return Err(anyhow::anyhow!("Only dense storage is supported for TTN addition")),
                };

                // Permute data before embedding/adding
                let permuted_data_a = permute_data_f64(&data_a, &tensor_a.dims, &perm_a);
                let permuted_data_b = permute_data_f64(&data_b, &tensor_b.dims, &perm_b);

                if has_bonds {
                    // Use block embedding for nodes with bonds
                    let mut result_data = vec![0.0_f64; total_size];
                    embed_block_f64(&mut result_data, &canonical_dims, &permuted_data_a, &permuted_dims_a, &bond_axis_info, true)?;
                    embed_block_f64(&mut result_data, &canonical_dims, &permuted_data_b, &permuted_dims_b, &bond_axis_info, false)?;
                    Storage::DenseF64(DenseStorageF64::from_vec(result_data))
                } else {
                    // No bonds: add tensors element-wise
                    let result_data: Vec<f64> = permuted_data_a.iter()
                        .zip(permuted_data_b.iter())
                        .map(|(&a, &b)| a + b)
                        .collect();
                    Storage::DenseF64(DenseStorageF64::from_vec(result_data))
                }
            };

            let new_tensor = TensorDynLen::new(canonical_indices, canonical_dims, Arc::new(new_storage));
            result.add_tensor_with_vertex(node_name.clone(), new_tensor)?;
        }

        // Add connections using the SAME shared index on both endpoints
        for ((src_name, tgt_name), (_, _, shared_idx)) in &edge_info {
            let src_node = result.graph.node_index(src_name)
                .ok_or_else(|| anyhow::anyhow!("Source node not found in result"))?;
            let tgt_node = result.graph.node_index(tgt_name)
                .ok_or_else(|| anyhow::anyhow!("Target node not found in result"))?;
            // Use the same shared_idx for both endpoints - this ensures contraction works
            result.connect(src_node, shared_idx, tgt_node, shared_idx)?;
        }

        // Clear ortho_region (sum is not orthogonalized)
        result.clear_ortho_region();

        Ok(result)
    }

    /// Contract the TreeTN to a single dense tensor.
    ///
    /// This method contracts all tensors in the network into a single tensor
    /// containing all physical indices. The contraction is performed using
    /// a tree-aware order (post-order from leaves to root) with edge-based
    /// contraction that uses Connection information to identify which indices
    /// to contract, rather than relying on index ID matching.
    ///
    /// # Returns
    /// A single tensor representing the full contraction of the network.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - The graph is not a valid tree
    /// - Tensor contraction fails
    pub fn contract_to_tensor(&self) -> Result<TensorDynLen<Id, Symm>>
    where
        V: Ord,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot contract empty TreeTN"));
        }

        if self.node_count() == 1 {
            // Single node - just return a clone of its tensor
            let node = self.graph.graph().node_indices().next()
                .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
            return self.tensor(node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found"));
        }

        // Validate tree structure
        self.validate_tree()
            .context("contract_to_tensor: graph must be a tree")?;

        // Choose a deterministic root (minimum vertex name)
        let root_name = self.graph.graph().node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .min()
            .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
        let root = self.graph.node_index(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node not found"))?;

        // Build post-order traversal (leaves first, root last)
        let post_order = self.post_order_dfs(root);

        // Track which nodes have been contracted and their resulting tensors
        // For each node in post-order, we contract its tensor with all
        // already-contracted children, using the edge's Connection info.
        let mut contracted: HashMap<NodeIndex, TensorDynLen<Id, Symm>> = HashMap::new();

        for node in post_order {
            let mut current = self.tensor(node)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node))?
                .clone();

            // Find all edges to already-contracted neighbors (children in the rooted tree)
            for (edge, neighbor) in self.edges_for_node(node) {
                if let Some(child_tensor) = contracted.remove(&neighbor) {
                    // Determine which index belongs to which node
                    // edge_index_for_node returns the index for a given node on this edge
                    let idx_current = self.edge_index_for_node(edge, node)?.clone();
                    let idx_child = self.edge_index_for_node(edge, neighbor)?.clone();

                    // Contract current tensor with child tensor along the bond indices
                    // We use contract_pairs which takes explicit index pairs
                    current = current.contract_pairs(&child_tensor, &[(idx_current, idx_child)])
                        .context("Failed to contract along edge")?;
                }
            }

            // Store the contracted result for this node
            contracted.insert(node, current);
        }

        // The root's tensor (last in post-order) is the final result
        contracted.remove(&root)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result"))
    }

    /// Perform a post-order DFS traversal starting from the given root.
    ///
    /// Returns nodes in post-order (children before parents, leaves first).
    fn post_order_dfs(&self, root: NodeIndex) -> Vec<NodeIndex> {
        let g = self.graph.graph();
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut stack = vec![(root, false)]; // (node, processed)

        while let Some((node, processed)) = stack.pop() {
            if processed {
                result.push(node);
            } else if !visited.contains(&node) {
                visited.insert(node);
                stack.push((node, true)); // Push back for post-processing

                // Push unvisited neighbors
                for neighbor in g.neighbors(node) {
                    if !visited.contains(&neighbor) {
                        stack.push((neighbor, false));
                    }
                }
            }
        }

        result
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
        use tensor4all::qr;

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

// ============================================================================
// Scalar multiplication for TreeTN
// ============================================================================

use std::ops::Mul;
use std::sync::Arc;

impl<Id, Symm, V> Mul<f64> for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = Self;

    /// Multiply the TreeTN by a scalar with distributed scaling.
    ///
    /// # Distribution Rules
    ///
    /// - If `ortho_region` is **non-empty**: multiply each center vertex tensor by `a^{1/|C|}`.
    /// - If `ortho_region` is **empty**: multiply each tensor by `a^{1/N}`.
    /// - The product over all applied factors must equal `a`.
    ///
    /// # Negative Scalar Handling
    ///
    /// - Distribute the magnitude `|a|` using the rules above.
    /// - Apply the sign `-1` to exactly one representative node.
    /// - Representative node: `min(ortho_region)` if non-empty, else `min(all_vertices)`.
    ///
    /// # Zero Scalar Handling
    ///
    /// - Multiply only the representative node by `0.0`, leaving others unchanged.
    /// - The whole network then evaluates to `0`.
    fn mul(mut self, a: f64) -> Self::Output {
        let n = self.node_count();
        if n == 0 {
            return self;
        }

        // Determine representative node
        let representative_vertex: V = if !self.ortho_region.is_empty() {
            self.ortho_region.iter().min().cloned().unwrap()
        } else {
            // Get minimum vertex from all nodes
            self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .min()
                .unwrap()
        };
        let representative_node = self.graph.node_index(&representative_vertex).unwrap();

        // Handle zero scalar
        if a == 0.0 {
            // Multiply only the representative node by 0
            if let Some(tensor) = self.tensor_mut(representative_node) {
                let new_storage = tensor.storage.as_ref() * 0.0;
                tensor.storage = Arc::new(new_storage);
            }
            return self;
        }

        // Handle non-zero scalar
        let (magnitude, sign) = if a < 0.0 {
            (-a, -1.0_f64)
        } else {
            (a, 1.0_f64)
        };

        // Determine which nodes to scale and the per-node factor
        let (nodes_to_scale, per_node_factor): (Vec<NodeIndex>, f64) = if !self.ortho_region.is_empty() {
            // Scale only ortho_region nodes
            let center_count = self.ortho_region.len();
            let factor = magnitude.powf(1.0 / center_count as f64);
            let nodes: Vec<NodeIndex> = self.ortho_region.iter()
                .filter_map(|v| self.graph.node_index(v))
                .collect();
            (nodes, factor)
        } else {
            // Scale all nodes
            let factor = magnitude.powf(1.0 / n as f64);
            let nodes: Vec<NodeIndex> = self.graph.graph().node_indices().collect();
            (nodes, factor)
        };

        // Apply scaling to the appropriate nodes
        for node_idx in nodes_to_scale {
            if let Some(tensor) = self.tensor_mut(node_idx) {
                let scale = if node_idx == representative_node {
                    per_node_factor * sign
                } else {
                    per_node_factor
                };
                let new_storage = tensor.storage.as_ref() * scale;
                tensor.storage = Arc::new(new_storage);
            }
        }

        self
    }
}

impl<Id, Symm, V> Mul<Complex64> for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = Self;

    /// Multiply the TreeTN by a complex scalar with distributed scaling.
    ///
    /// # Distribution Rules
    ///
    /// For complex scalars, we distribute the magnitude and apply the phase to one node.
    ///
    /// - If `ortho_region` is **non-empty**: multiply each center vertex tensor by `|a|^{1/|C|}`.
    /// - If `ortho_region` is **empty**: multiply each tensor by `|a|^{1/N}`.
    /// - The phase `a/|a|` is applied to the representative node.
    ///
    /// # Zero Scalar Handling
    ///
    /// - Multiply only the representative node by `0`, leaving others unchanged.
    fn mul(mut self, a: Complex64) -> Self::Output {
        let n = self.node_count();
        if n == 0 {
            return self;
        }

        // Determine representative node
        let representative_vertex: V = if !self.ortho_region.is_empty() {
            self.ortho_region.iter().min().cloned().unwrap()
        } else {
            self.graph.graph().node_indices()
                .filter_map(|idx| self.graph.node_name(idx).cloned())
                .min()
                .unwrap()
        };
        let representative_node = self.graph.node_index(&representative_vertex).unwrap();

        // Handle zero scalar
        let magnitude = a.norm();
        if magnitude == 0.0 {
            // Multiply only the representative node by 0
            if let Some(tensor) = self.tensor_mut(representative_node) {
                let new_storage = tensor.storage.as_ref() * 0.0;
                tensor.storage = Arc::new(new_storage);
            }
            return self;
        }

        // Compute phase (unit complex number)
        let phase = a / magnitude;

        // Determine which nodes to scale and the per-node factor (real)
        let (nodes_to_scale, per_node_factor): (Vec<NodeIndex>, f64) = if !self.ortho_region.is_empty() {
            let center_count = self.ortho_region.len();
            let factor = magnitude.powf(1.0 / center_count as f64);
            let nodes: Vec<NodeIndex> = self.ortho_region.iter()
                .filter_map(|v| self.graph.node_index(v))
                .collect();
            (nodes, factor)
        } else {
            let factor = magnitude.powf(1.0 / n as f64);
            let nodes: Vec<NodeIndex> = self.graph.graph().node_indices().collect();
            (nodes, factor)
        };

        // Apply scaling to the appropriate nodes
        for node_idx in nodes_to_scale {
            if let Some(tensor) = self.tensor_mut(node_idx) {
                if node_idx == representative_node {
                    // Apply both magnitude factor and phase to representative
                    let complex_scale = Complex64::new(per_node_factor, 0.0) * phase;
                    let new_storage = tensor.storage.as_ref() * complex_scale;
                    tensor.storage = Arc::new(new_storage);
                } else {
                    // Apply only magnitude factor to other nodes
                    let new_storage = tensor.storage.as_ref() * per_node_factor;
                    tensor.storage = Arc::new(new_storage);
                }
            }
        }

        self
    }
}

// Implement Mul with reference to avoid consuming TreeTN
impl<Id, Symm, V> Mul<f64> for &TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = TreeTN<Id, Symm, V>;

    fn mul(self, a: f64) -> Self::Output {
        self.clone() * a
    }
}

impl<Id, Symm, V> Mul<Complex64> for &TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    type Output = TreeTN<Id, Symm, V>;

    fn mul(self, a: Complex64) -> Self::Output {
        self.clone() * a
    }
}

// Clone implementation for TreeTN
impl<Id, Symm, V> Clone for TreeTN<Id, Symm, V>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            ortho_region: self.ortho_region.clone(),
            site_index_network: self.site_index_network.clone(),
        }
    }
}

// ============================================================================
// Helper functions for TreeTN addition (direct-sum block embedding)
// ============================================================================

/// Embed a source tensor block into a larger destination tensor for f64 data.
///
/// This function places the source data into the appropriate block of the destination
/// tensor, where bond indices are expanded according to `bond_positions`.
///
/// # Arguments
/// * `dest` - Destination tensor data (mutable, will be modified)
/// * `dest_dims` - Dimensions of the destination tensor
/// * `src` - Source tensor data
/// * `src_dims` - Dimensions of the source tensor
/// * `bond_positions` - For each bond axis: (position_in_dest, dim_a, dim_b)
/// * `is_a_block` - If true, embed in the "A" block (0..dim_a); else in "B" block (dim_a..dim_a+dim_b)
fn embed_block_f64(
    dest: &mut [f64],
    dest_dims: &[usize],
    src: &[f64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    // For each element in src, compute its position in dest
    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    // Build stride arrays for both tensors
    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    // Create a map from dest axis position to bond info
    let bond_map: HashMap<usize, (usize, usize)> = bond_positions.iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    // Iterate over all elements in source
    for src_linear in 0..src_total {
        // Convert to multi-index
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        // Compute destination multi-index
        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                // This is a bond axis
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx // A block: 0..dim_a
                } else {
                    src_bond_idx + dim_a // B block: dim_a..dim_a+dim_b
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                // Physical axis - direct copy
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        // Convert dest multi-index to linear
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Embed a source tensor block into a larger destination tensor for Complex64 data.
fn embed_block_c64(
    dest: &mut [Complex64],
    dest_dims: &[usize],
    src: &[Complex64],
    src_dims: &[usize],
    bond_positions: &[(usize, usize, usize)],
    is_a_block: bool,
) -> Result<()> {
    if src.is_empty() {
        return Ok(());
    }

    let src_total: usize = src_dims.iter().product();
    if src.len() != src_total {
        return Err(anyhow::anyhow!(
            "Source data length {} doesn't match dims product {}",
            src.len(),
            src_total
        ));
    }

    let src_strides = compute_strides(src_dims);
    let dest_strides = compute_strides(dest_dims);

    let bond_map: HashMap<usize, (usize, usize)> = bond_positions.iter()
        .map(|&(pos, dim_a, dim_b)| (pos, (dim_a, dim_b)))
        .collect();

    for src_linear in 0..src_total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, src_dims.len());

        let mut dest_multi = Vec::with_capacity(dest_dims.len());
        let mut src_idx = 0;

        for (dest_axis, _dest_dim) in dest_dims.iter().enumerate() {
            if let Some(&(dim_a, _dim_b)) = bond_map.get(&dest_axis) {
                let src_bond_idx = src_multi[src_idx];
                let dest_bond_idx = if is_a_block {
                    src_bond_idx
                } else {
                    src_bond_idx + dim_a
                };
                dest_multi.push(dest_bond_idx);
                src_idx += 1;
            } else {
                dest_multi.push(src_multi[src_idx]);
                src_idx += 1;
            }
        }

        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);
        dest[dest_linear] = src[src_linear];
    }

    Ok(())
}

/// Compute strides for row-major (C-order) indexing.
fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Convert linear index to multi-index.
fn linear_to_multi_index(mut linear: usize, strides: &[usize], rank: usize) -> Vec<usize> {
    let mut multi = vec![0; rank];
    for i in 0..rank {
        multi[i] = linear / strides[i];
        linear %= strides[i];
    }
    multi
}

/// Convert multi-index to linear index.
fn multi_to_linear_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(&m, &s)| m * s).sum()
}

/// Permute tensor data according to axis permutation.
///
/// Given tensor data in row-major order with shape `dims`, rearrange the data
/// so that axis `i` in the output corresponds to axis `perm[i]` in the input.
///
/// # Arguments
/// * `data` - Input tensor data in row-major order
/// * `dims` - Dimensions of the input tensor
/// * `perm` - Permutation: perm[i] = j means output axis i comes from input axis j
///
/// # Returns
/// Permuted data with shape `[dims[perm[0]], dims[perm[1]], ...]`
fn permute_data_f64(data: &[f64], dims: &[usize], perm: &[usize]) -> Vec<f64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    // Check if permutation is identity
    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![0.0_f64; total];

    // Inverse permutation: inv_perm[j] = i means input axis j maps to output axis i
    let mut inv_perm = vec![0; rank];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);

        // Map source multi-index to destination multi-index
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);

        result[dest_linear] = data[src_linear];
    }

    result
}

/// Permute tensor data according to axis permutation (Complex64 version).
fn permute_data_c64(data: &[Complex64], dims: &[usize], perm: &[usize]) -> Vec<Complex64> {
    if perm.is_empty() || data.is_empty() {
        return data.to_vec();
    }

    // Check if permutation is identity
    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
    if is_identity {
        return data.to_vec();
    }

    let rank = dims.len();
    let new_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
    let total: usize = dims.iter().product();

    let src_strides = compute_strides(dims);
    let dest_strides = compute_strides(&new_dims);

    let mut result = vec![Complex64::new(0.0, 0.0); total];

    for src_linear in 0..total {
        let src_multi = linear_to_multi_index(src_linear, &src_strides, rank);

        // Map source multi-index to destination multi-index
        let dest_multi: Vec<usize> = (0..rank).map(|i| src_multi[perm[i]]).collect();
        let dest_linear = multi_to_linear_index(&dest_multi, &dest_strides);

        result[dest_linear] = data[src_linear];
    }

    result
}

// ============================================================================
// TreeTN decomposition from dense tensor
// ============================================================================

/// Specification for tree topology: defines nodes and edges.
#[derive(Debug, Clone)]
pub struct TreeTopology<V> {
    /// Nodes in the tree (node name -> set of physical index positions in the input tensor)
    pub nodes: HashMap<V, Vec<usize>>,
    /// Edges in the tree: (node_a, node_b)
    pub edges: Vec<(V, V)>,
}

impl<V: Clone + Hash + Eq> TreeTopology<V> {
    /// Create a new tree topology with the given nodes and edges.
    ///
    /// # Arguments
    /// * `nodes` - Map from node name to the positions of physical indices in the input tensor
    /// * `edges` - List of edges as (node_a, node_b) pairs
    pub fn new(nodes: HashMap<V, Vec<usize>>, edges: Vec<(V, V)>) -> Self {
        Self { nodes, edges }
    }

    /// Validate that this topology describes a tree.
    pub fn validate(&self) -> Result<()> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(anyhow::anyhow!("Tree topology must have at least one node"));
        }
        if n > 1 && self.edges.len() != n - 1 {
            return Err(anyhow::anyhow!(
                "Tree must have exactly n-1 edges: got {} nodes and {} edges",
                n,
                self.edges.len()
            ));
        }
        // Check all edge endpoints are valid nodes
        for (a, b) in &self.edges {
            if !self.nodes.contains_key(a) {
                return Err(anyhow::anyhow!("Edge refers to unknown node"));
            }
            if !self.nodes.contains_key(b) {
                return Err(anyhow::anyhow!("Edge refers to unknown node"));
            }
        }
        Ok(())
    }
}

/// Decompose a dense tensor into a TreeTN using QR-based factorization.
///
/// This function takes a dense tensor and a tree topology specification, then
/// recursively decomposes the tensor using QR factorization to create a TreeTN.
///
/// # Algorithm
///
/// 1. Start from a leaf node, QR decompose to separate that node's physical indices
/// 2. Contract R with remaining tensor, repeat for next edge
/// 3. Continue until all edges are processed
///
/// # Arguments
/// * `tensor` - The dense tensor to decompose
/// * `topology` - Tree topology specifying nodes, edges, and physical index assignments
///
/// # Returns
/// A TreeTN representing the decomposed tensor.
///
/// # Errors
/// Returns an error if:
/// - The topology is invalid
/// - Physical index positions don't match the tensor
/// - QR decomposition fails
pub fn decompose_tensor_to_treetn<Id, Symm, V>(
    tensor: &TensorDynLen<Id, Symm>,
    topology: &TreeTopology<V>,
) -> Result<TreeTN<Id, Symm, V>>
where
    Id: Clone + std::hash::Hash + Eq + From<DynId> + Ord,
    Symm: Clone + Symmetry + From<NoSymmSpace>,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    use tensor4all::qr::qr;

    topology.validate()?;

    if topology.nodes.len() == 1 {
        // Single node - just wrap the tensor
        let mut tn = TreeTN::<Id, Symm, V>::new();
        let node_name = topology.nodes.keys().next().unwrap().clone();
        tn.add_tensor_with_vertex(node_name, tensor.clone())?;
        return Ok(tn);
    }

    // Validate that all index positions are valid
    for (_, positions) in &topology.nodes {
        for &pos in positions {
            if pos >= tensor.indices.len() {
                return Err(anyhow::anyhow!(
                    "Index position {} out of bounds (tensor has {} indices)",
                    pos,
                    tensor.indices.len()
                ));
            }
        }
    }

    // Build adjacency list for the tree
    let mut adj: HashMap<V, Vec<V>> = HashMap::new();
    for node in topology.nodes.keys() {
        adj.insert(node.clone(), Vec::new());
    }
    for (a, b) in &topology.edges {
        adj.get_mut(a).unwrap().push(b.clone());
        adj.get_mut(b).unwrap().push(a.clone());
    }

    // Find leaves (nodes with degree 1) - not currently used but kept for reference
    let _leaves: Vec<V> = adj.iter()
        .filter(|(_, neighbors)| neighbors.len() == 1)
        .map(|(node, _)| node.clone())
        .collect();

    // Choose root as the node with highest degree, or first non-leaf
    let root = adj.iter()
        .max_by_key(|(_, neighbors)| neighbors.len())
        .map(|(node, _)| node.clone())
        .ok_or_else(|| anyhow::anyhow!("Cannot find root node"))?;

    // Build traversal order using BFS from root
    let mut traversal_order: Vec<(V, Option<V>)> = Vec::new(); // (node, parent)
    let mut visited: HashSet<V> = HashSet::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((root.clone(), None::<V>));

    while let Some((node, parent)) = queue.pop_front() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node.clone());
        traversal_order.push((node.clone(), parent));

        for neighbor in adj.get(&node).unwrap() {
            if !visited.contains(neighbor) {
                queue.push_back((neighbor.clone(), Some(node.clone())));
            }
        }
    }

    // Reverse traversal order to process leaves first (post-order)
    traversal_order.reverse();

    // Store intermediate tensors as we decompose
    let mut current_tensor = tensor.clone();

    // Store the resulting node tensors
    let mut node_tensors: HashMap<V, TensorDynLen<Id, Symm>> = HashMap::new();

    // Store bond indices between nodes: (node_a, node_b) -> (index_on_a, index_on_b)
    let mut bond_indices: HashMap<(V, V), (Index<Id, Symm>, Index<Id, Symm>)> = HashMap::new();

    // Process nodes in post-order (leaves first)
    for i in 0..traversal_order.len() - 1 {
        let (node, parent) = &traversal_order[i];
        let parent_node = parent.as_ref().unwrap();

        // Get the physical index positions for this node
        let node_positions = topology.nodes.get(node).unwrap();

        // Find physical indices for this node in current_tensor
        let left_inds: Vec<Index<Id, Symm>> = node_positions.iter()
            .filter_map(|&pos| current_tensor.indices.get(pos).cloned())
            .collect();

        if left_inds.is_empty() && current_tensor.indices.len() > 1 {
            // No physical indices to separate - use first index
            // This happens when indices have already been separated
            continue;
        }

        // Perform QR decomposition
        // Q will have the node's physical indices + bond index
        // R will have bond index + remaining indices
        let (q, r) = qr::<Id, Symm, f64>(&current_tensor, &left_inds)
            .map_err(|e| anyhow::anyhow!("QR decomposition failed: {:?}", e))?;

        // Store Q as the node's tensor (with physical indices + bond to parent)
        node_tensors.insert(node.clone(), q.clone());

        // Store the bond index connecting this node to its parent
        let bond_idx_node = q.indices.last().cloned()
            .ok_or_else(|| anyhow::anyhow!("Q has no indices"))?;
        let bond_idx_parent = r.indices.first().cloned()
            .ok_or_else(|| anyhow::anyhow!("R has no indices"))?;

        // Store in canonical order
        let key = if *node < *parent_node {
            (node.clone(), parent_node.clone())
        } else {
            (parent_node.clone(), node.clone())
        };
        if *node < *parent_node {
            bond_indices.insert(key, (bond_idx_node, bond_idx_parent));
        } else {
            bond_indices.insert(key, (bond_idx_parent, bond_idx_node));
        }

        // R becomes the current tensor for the next iteration
        current_tensor = r;
    }

    // The last node (root) gets the remaining tensor
    let (root_node, _) = &traversal_order.last().unwrap();
    node_tensors.insert(root_node.clone(), current_tensor);

    // Build the TreeTN
    let mut tn = TreeTN::<Id, Symm, V>::new();

    // Add all tensors
    for (node_name, tensor) in &node_tensors {
        tn.add_tensor_with_vertex(node_name.clone(), tensor.clone())?;
    }

    // Add connections
    for (a, b) in &topology.edges {
        let key = if *a < *b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };

        if let Some((idx_a, idx_b)) = bond_indices.get(&key) {
            let node_a = tn.graph.node_index(a)
                .ok_or_else(|| anyhow::anyhow!("Node not found: {:?}", a))?;
            let node_b = tn.graph.node_index(b)
                .ok_or_else(|| anyhow::anyhow!("Node not found: {:?}", b))?;

            if *a < *b {
                tn.connect(node_a, idx_a, node_b, idx_b)?;
            } else {
                tn.connect(node_b, idx_a, node_a, idx_b)?;
            }
        }
    }

    Ok(tn)
}
