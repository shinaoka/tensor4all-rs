use petgraph::stable_graph::{StableGraph, NodeIndex, EdgeIndex};
use petgraph::Undirected;
use petgraph::visit::{Dfs, EdgeRef};
use std::collections::VecDeque;
use std::collections::HashSet;
use std::collections::HashMap;
use tensor4all_tensor::TensorDynLen;
use tensor4all_tensor::Storage;
use tensor4all_core::index::{Index, NoSymmSpace, Symmetry, DynId};
use tensor4all_core::index_ops::common_inds;
use crate::connection::Connection;
use anyhow::{Result, Context};
use num_complex::Complex64;

/// Tree Tensor Network structure.
///
/// Maintains a graph of tensors connected by bonds (edges).
/// Each node stores a `TensorDynLen`, and edges store `Connection` objects
/// that hold the Index objects on both sides of the bond.
pub struct TreeTN<Id, Symm = NoSymmSpace> {
    /// Graph structure: nodes are tensors, edges are connections
    graph: StableGraph<TensorDynLen<Id, Symm>, Connection<Id, Symm>, Undirected>,
    /// Orthogonalization centers (auto_centers).
    /// When empty, the network is not orthogonalized.
    /// When non-empty, contains the NodeIndex values of the orthogonalization centers.
    auto_centers: HashSet<NodeIndex>,
}

impl<Id, Symm> TreeTN<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    /// Create a new empty TreeTN.
    pub fn new() -> Self {
        Self {
            graph: StableGraph::with_capacity(0, 0),
            auto_centers: HashSet::new(),
        }
    }

    /// Add a tensor to the network.
    ///
    /// Returns the NodeIndex for the newly added tensor.
    pub fn add_tensor(&mut self, tensor: TensorDynLen<Id, Symm>) -> NodeIndex {
        self.graph.add_node(tensor)
    }

    /// Get a reference to a tensor by NodeIndex.
    pub fn tensor(&self, node: NodeIndex) -> Option<&TensorDynLen<Id, Symm>> {
        self.graph.node_weight(node)
    }

    /// Get a mutable reference to a tensor by NodeIndex.
    pub fn tensor_mut(&mut self, node: NodeIndex) -> Option<&mut TensorDynLen<Id, Symm>> {
        self.graph.node_weight_mut(node)
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

        // All validations passed, replace the tensor
        Ok(self.graph.node_weight_mut(node).map(|old| {
            std::mem::replace(old, new_tensor)
        }))
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

        // Create connection
        // index_source corresponds to node_a, index_target to node_b
        let connection = Connection::new(index_a_clone, index_b_clone)
            .context("Failed to create connection")?;

        // Add edge to graph
        // petgraph will assign source/target based on the order (a, b)
        let edge_idx = self.graph.add_edge(node_a, node_b, connection);
        Ok(edge_idx)
    }

    /// Get a reference to a connection by EdgeIndex.
    pub fn connection(&self, edge: EdgeIndex) -> Option<&Connection<Id, Symm>> {
        self.graph.edge_weight(edge)
    }

    /// Get a mutable reference to a connection by EdgeIndex.
    pub fn connection_mut(&mut self, edge: EdgeIndex) -> Option<&mut Connection<Id, Symm>> {
        self.graph.edge_weight_mut(edge)
    }

    /// Get the index on the side of the given node for an edge.
    ///
    /// Returns the Index that belongs to the specified node in the given edge.
    pub fn edge_index_for_node(
        &self,
        edge: EdgeIndex,
        node: NodeIndex,
    ) -> Result<&Index<Id, Symm>> {
        let (source, target) = self.graph
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
        self.graph
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
    pub fn replace_edge_bond(
        &mut self,
        edge: EdgeIndex,
        new_index_on_a: Index<Id, Symm>,
        new_index_on_b: Index<Id, Symm>,
    ) -> Result<()> {
        // Validate edge exists
        let (_source, _target) = self.graph
            .edge_endpoints(edge)
            .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;

        let conn = self.connection_mut(edge)
            .ok_or_else(|| anyhow::anyhow!("Connection not found"))?;

        // Update the connection indices
        // new_index_on_a corresponds to source, new_index_on_b to target
        conn.replace_bond_indices(new_index_on_a, new_index_on_b)
            .context("Failed to replace bond indices")?;

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
        
        let (source, target) = self.graph.edge_endpoints(edge)?;
        
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
        if self.graph.node_count() == 0 {
            return Ok(()); // Empty graph is trivially valid
        }

        // Check if graph is connected
        let mut visited = std::collections::HashSet::new();
        let start_node = self.graph.node_indices().next()
            .ok_or_else(|| anyhow::anyhow!("Graph has no nodes"))?;

        // DFS to count reachable nodes
        let mut dfs = Dfs::new(&self.graph, start_node);
        while let Some(node) = dfs.next(&self.graph) {
            visited.insert(node);
        }

        if visited.len() != self.graph.node_count() {
            return Err(anyhow::anyhow!(
                "Graph is not connected: {} nodes reachable out of {}",
                visited.len(),
                self.graph.node_count()
            ))
            .context("validate_tree: graph must be connected");
        }

        // Check tree condition: edges = nodes - 1
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();

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
        self.graph.node_count()
    }

    /// Get the number of edges in the network.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a reference to the auto_centers (orthogonalization centers).
    ///
    /// When empty, the network is not orthogonalized.
    pub fn auto_centers(&self) -> &HashSet<NodeIndex> {
        &self.auto_centers
    }

    /// Check if the network is orthogonalized.
    ///
    /// Returns `true` if `auto_centers` is non-empty, `false` otherwise.
    pub fn is_orthogonalized(&self) -> bool {
        !self.auto_centers.is_empty()
    }

    /// Set the auto_centers (orthogonalization centers).
    ///
    /// Validates that all specified nodes exist in the graph.
    pub fn set_auto_centers(&mut self, centers: impl IntoIterator<Item = NodeIndex>) -> Result<()> {
        let centers: HashSet<NodeIndex> = centers.into_iter().collect();
        
        // Validate that all centers exist in the graph
        for center in &centers {
            if !self.graph.contains_node(*center) {
                return Err(anyhow::anyhow!("Node {:?} does not exist in the graph", center))
                    .context("set_auto_centers: all centers must be valid nodes");
            }
        }
        
        self.auto_centers = centers;
        Ok(())
    }

    /// Clear the auto_centers (mark network as not orthogonalized).
    pub fn clear_auto_centers(&mut self) {
        self.auto_centers.clear();
    }

    /// Add a node to auto_centers.
    ///
    /// Validates that the node exists in the graph.
    pub fn add_auto_center(&mut self, center: NodeIndex) -> Result<()> {
        if !self.graph.contains_node(center) {
            return Err(anyhow::anyhow!("Node {:?} does not exist in the graph", center))
                .context("add_auto_center: center must be a valid node");
        }
        self.auto_centers.insert(center);
        Ok(())
    }

    /// Remove a node from auto_centers.
    pub fn remove_auto_center(&mut self, center: NodeIndex) {
        self.auto_centers.remove(&center);
    }

    /// Validate that `auto_centers` and edge `ortho_towards` are consistent.
    ///
    /// Rules:
    /// - If `auto_centers` is empty (not orthogonalized), all edges must have `ortho_towards == None`.
    /// - If `auto_centers` is non-empty, it must be connected in the tree (forms a subtree).
    /// - `ortho_towards == None` is allowed **only** when both edge endpoints are in `auto_centers`.
    /// - For any edge with at least one endpoint outside `auto_centers`, `ortho_towards` must be `Some(...)`
    ///   and must point towards the endpoint with smaller distance to `auto_centers`.
    pub fn validate_ortho_consistency(&self) -> Result<()> {
        // If not orthogonalized, require no edge directions.
        if self.auto_centers.is_empty() {
            for e in self.graph.edge_indices() {
                let conn = self.connection(e).ok_or_else(|| anyhow::anyhow!("Connection not found"))?;
                if conn.ortho_towards.is_some() {
                    return Err(anyhow::anyhow!(
                        "Found ortho_towards on edge {:?} but auto_centers is empty",
                        e
                    ))
                    .context("validate_ortho_consistency: auto_centers empty implies no directions");
                }
            }
            return Ok(());
        }

        // Validate all auto_centers exist.
        for c in &self.auto_centers {
            if !self.graph.contains_node(*c) {
                return Err(anyhow::anyhow!("auto_center {:?} does not exist in graph", c))
                    .context("validate_ortho_consistency: all auto_centers must be valid nodes");
            }
        }

        // Check auto_centers connectivity in the induced subgraph.
        let start = *self
            .auto_centers
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("auto_centers unexpectedly empty"))?;
        let mut stack = vec![start];
        let mut seen = HashSet::new();
        seen.insert(start);
        while let Some(v) = stack.pop() {
            for nb in self.graph.neighbors(v) {
                if self.auto_centers.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        if seen.len() != self.auto_centers.len() {
            return Err(anyhow::anyhow!(
                "auto_centers is not connected: reached {} out of {} centers",
                seen.len(),
                self.auto_centers.len()
            ))
            .context("validate_ortho_consistency: auto_centers must form a connected subtree");
        }

        // Multi-source BFS distances to auto_centers.
        let mut dist: std::collections::HashMap<NodeIndex, usize> = std::collections::HashMap::new();
        let mut q = VecDeque::new();
        for &c in &self.auto_centers {
            dist.insert(c, 0);
            q.push_back(c);
        }
        while let Some(v) = q.pop_front() {
            let dv = dist[&v];
            for nb in self.graph.neighbors(v) {
                if !dist.contains_key(&nb) {
                    dist.insert(nb, dv + 1);
                    q.push_back(nb);
                }
            }
        }

        // Check each edge.
        for e in self.graph.edge_indices() {
            let (source, target) = self
                .graph
                .edge_endpoints(e)
                .ok_or_else(|| anyhow::anyhow!("Edge does not exist"))?;
            let conn = self.connection(e).ok_or_else(|| anyhow::anyhow!("Connection not found"))?;

            let s_in = self.auto_centers.contains(&source);
            let t_in = self.auto_centers.contains(&target);

            // None allowed only when both endpoints are centers.
            if conn.ortho_towards.is_none() {
                if !(s_in && t_in) {
                    return Err(anyhow::anyhow!(
                        "ortho_towards is None on edge {:?} but not both endpoints are in auto_centers",
                        e
                    ))
                    .context("validate_ortho_consistency: None allowed only inside auto_centers");
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

            // Choose expected direction: towards smaller distance (closer to auto_centers).
            // In a tree with connected auto_centers, adjacent nodes should differ by exactly 1
            // unless both are in auto_centers (handled above).
            if ds == dt {
                return Err(anyhow::anyhow!(
                    "Ambiguous direction on edge {:?}: distances are equal ({} == {})",
                    e,
                    ds,
                    dt
                ))
                .context("validate_ortho_consistency: outside auto_centers, distances should differ");
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
                .context("validate_ortho_consistency: ortho_towards must point towards auto_centers");
            }

            // Additionally enforce that boundary edges (one in centers, one out) always point into centers.
            if s_in ^ t_in {
                let expected_into_centers_id = if s_in { &conn.index_source.id } else { &conn.index_target.id };
                if &ortho_idx.id != expected_into_centers_id {
                    return Err(anyhow::anyhow!(
                        "Boundary edge {:?} must point into auto_centers",
                        e
                    ))
                    .context("validate_ortho_consistency: boundary must point into auto_centers");
                }
            }
        }

        Ok(())
    }

    /// Orthogonalize the network using QR decomposition towards the specified auto_centers.
    ///
    /// This method consumes the TreeTN and returns a new orthogonalized TreeTN.
    /// The algorithm:
    /// 1. Validates that the graph is a tree
    /// 2. Sets the auto_centers and validates connectivity
    /// 3. Computes distances from auto_centers using BFS
    /// 4. Processes nodes in order of decreasing distance (farthest first)
    /// 5. For each node, performs QR decomposition on edges pointing towards auto_centers
    /// 6. Absorbs R factors into parent nodes using contract_pairs
    ///
    /// # Arguments
    /// * `auto_centers` - The nodes that will serve as orthogonalization centers
    ///
    /// # Returns
    /// A new orthogonalized TreeTN, or an error if validation fails or QR decomposition fails.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The graph is not a tree
    /// - auto_centers are not connected
    /// - QR decomposition fails
    /// - Tensor storage types are not DenseF64 or DenseC64
    pub fn orthogonalize_with_qr(
        mut self,
        auto_centers: impl IntoIterator<Item = NodeIndex>,
    ) -> Result<Self>
    where
        Id: Clone + std::hash::Hash + Eq + From<DynId>,
        Symm: Clone + Symmetry + From<NoSymmSpace>,
    {
        use tensor4all_linalg::qr;

        // 1. Validate tree structure
        self.validate_tree()
            .context("orthogonalize_with_qr: graph must be a tree")?;

        // 2. Set auto_centers and validate connectivity
        self.set_auto_centers(auto_centers)
            .context("orthogonalize_with_qr: failed to set auto_centers")?;

        if self.auto_centers.is_empty() {
            return Ok(self); // Nothing to do if no centers
        }

        // Validate auto_centers connectivity (similar to validate_ortho_consistency)
        let start = *self
            .auto_centers
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("auto_centers unexpectedly empty"))?;
        let mut stack = vec![start];
        let mut seen = HashSet::new();
        seen.insert(start);
        while let Some(v) = stack.pop() {
            for nb in self.graph.neighbors(v) {
                if self.auto_centers.contains(&nb) && seen.insert(nb) {
                    stack.push(nb);
                }
            }
        }
        if seen.len() != self.auto_centers.len() {
            return Err(anyhow::anyhow!(
                "auto_centers is not connected: reached {} out of {} centers",
                seen.len(),
                self.auto_centers.len()
            ))
            .context("orthogonalize_with_qr: auto_centers must form a connected subtree");
        }

        // 3. Multi-source BFS to compute distances from auto_centers
        let mut dist: HashMap<NodeIndex, usize> = HashMap::new();
        let mut q = VecDeque::new();
        for &c in &self.auto_centers {
            dist.insert(c, 0);
            q.push_back(c);
        }
        while let Some(v) = q.pop_front() {
            let dv = dist[&v];
            for nb in self.graph.neighbors(v) {
                if !dist.contains_key(&nb) {
                    dist.insert(nb, dv + 1);
                    q.push_back(nb);
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
            // Skip nodes in auto_centers (they are already at distance 0)
            if self.auto_centers.contains(&v) {
                continue;
            }

            // Find parent: neighbor with distance one less
            let v_dist = dist[&v];
            let parent_dist = v_dist.checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} at distance 0 but not in auto_centers", v))
                .context("orthogonalize_with_qr: distance calculation error")?;

            let parent = self.graph
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
            let edge = self.graph
                .edges_connecting(v, parent)
                .next()
                .ok_or_else(|| anyhow::anyhow!("No edge found between node {:?} and parent {:?}", v, parent))
                .context("orthogonalize_with_qr: edge not found")?
                .id();

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

            // Set ortho_towards to point towards parent (auto_centers direction)
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

impl<Id, Symm> Default for TreeTN<Id, Symm>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
{
    fn default() -> Self {
        Self::new()
    }
}


