//! Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
//!
//! Provides a graph structure where:
//! - Nodes store site space (physical indices): `HashSet<Index>`
//! - Edges represent connections (no data stored on edges)
//!
//! This design separates the index structure from tensor data,
//! enabling topology and site space comparison independent of tensor values.
//!
//! Note: Link space (bond indices) is not stored here, as it can be derived
//! from tensor data or is already stored in Connection objects in TreeTN.

use crate::named_graph::NamedGraph;
use petgraph::stable_graph::{StableGraph, NodeIndex, EdgeIndex};
use petgraph::Undirected;
use tensor4all::index::{Index, NoSymmSpace, Symmetry};
use tensor4all::DefaultTagSet;
use std::collections::HashSet;
use std::hash::Hash;
use std::fmt::Debug;

/// Site Index Network (inspired by ITensorNetworks.jl's IndsNetwork)
///
/// Represents the index structure of a tensor network:
/// - **Site space** (node data): Physical indices at each node
/// - **Edges**: Graph structure only (no edge data)
///
/// This structure enables:
/// - Comparing topologies and site spaces independently of tensor data
/// - Extracting index information without accessing tensor values
/// - Validating network structure consistency
///
/// # Type Parameters
/// - `NodeName`: Node name type (must be Clone, Hash, Eq, Send, Sync, Debug)
/// - `Id`: Index ID type
/// - `Symm`: Symmetry type
/// - `Tags`: Tag type (default: DefaultTagSet)
#[derive(Debug, Clone)]
pub struct SiteIndexNetwork<NodeName, Id, Symm = NoSymmSpace, Tags = DefaultTagSet>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Named graph: maps node names to site space.
    ///
    /// Type parameters:
    /// - 1st (NodeName): Node name type
    /// - 2nd (NodeData): Site space (physical indices) - `HashSet<Index>` stored at each node
    /// - 3rd (EdgeData): `()` - No edge data (edges represent connections only)
    graph: NamedGraph<NodeName, HashSet<Index<Id, Symm, Tags>>, ()>,
}

impl<NodeName, Id, Symm, Tags> SiteIndexNetwork<NodeName, Id, Symm, Tags>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Create a new empty SiteIndexNetwork.
    pub fn new() -> Self {
        Self {
            graph: NamedGraph::new(),
        }
    }

    /// Create a new SiteIndexNetwork with initial capacity.
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: NamedGraph::with_capacity(nodes, edges),
        }
    }

    /// Add a node with site space (physical indices).
    ///
    /// # Arguments
    /// * `node_name` - The name of the node
    /// * `site_space` - The physical indices at this node (order doesn't matter)
    ///
    /// Returns an error if the node already exists.
    pub fn add_node(
        &mut self,
        node_name: NodeName,
        site_space: impl Into<HashSet<Index<Id, Symm, Tags>>>,
    ) -> Result<NodeIndex, String> {
        self.graph.add_node(node_name, site_space.into())
    }

    /// Check if a node exists.
    pub fn has_node(&self, node_name: &NodeName) -> bool {
        self.graph.has_node(node_name)
    }

    /// Get the site space (physical indices) for a node.
    pub fn site_space(&self, node_name: &NodeName) -> Option<&HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_data(node_name)
    }

    /// Get a mutable reference to the site space for a node.
    pub fn site_space_mut(&mut self, node_name: &NodeName) -> Option<&mut HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_data_mut(node_name)
    }

    /// Get the site space by NodeIndex.
    pub fn site_space_by_index(&self, node: NodeIndex) -> Option<&HashSet<Index<Id, Symm, Tags>>> {
        self.graph.node_weight(node)
    }

    /// Add an edge between two nodes (no edge data).
    ///
    /// # Arguments
    /// * `n1` - First node name
    /// * `n2` - Second node name
    ///
    /// Returns an error if either node doesn't exist.
    pub fn add_edge(&mut self, n1: &NodeName, n2: &NodeName) -> Result<EdgeIndex, String> {
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
    ///
    /// This allows direct access to petgraph algorithms.
    pub fn graph(&self) -> &StableGraph<HashSet<Index<Id, Symm, Tags>>, (), Undirected> {
        self.graph.graph()
    }

    /// Get a mutable reference to the internal graph.
    ///
    /// **Warning**: Directly modifying the internal graph can break the node-name-to-index mapping.
    pub fn graph_mut(&mut self) -> &mut StableGraph<HashSet<Index<Id, Symm, Tags>>, (), Undirected> {
        self.graph.graph_mut()
    }

    /// Check if two SiteIndexNetworks have compatible topology and site space.
    ///
    /// Two networks are compatible if:
    /// - Same number of nodes and edges
    /// - Same node names
    /// - Same site space for each node (HashSet equality, order doesn't matter)
    pub fn is_compatible(&self, other: &Self) -> bool {
        // Check node count
        if self.node_count() != other.node_count() {
            return false;
        }

        // Check edge count
        if self.edge_count() != other.edge_count() {
            return false;
        }

        // Check all nodes have same site space (HashSet comparison, order-independent)
        for node_name in self.node_names() {
            let self_site = self.site_space(node_name);
            let other_site = other.site_space(node_name);

            match (self_site, other_site) {
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
}

impl<NodeName, Id, Symm, Tags> Default for SiteIndexNetwork<NodeName, Id, Symm, Tags>
where
    NodeName: Clone + Hash + Eq + Send + Sync + Debug,
    Id: Clone + Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tensor4all::index::Index;

    #[test]
    fn test_site_index_network_basic() {
        let mut net: SiteIndexNetwork<String, u128, NoSymmSpace, DefaultTagSet> = SiteIndexNetwork::new();

        // Add nodes with site space (order doesn't matter with HashSet)
        let site1: HashSet<_> = [Index::new(1u128, NoSymmSpace::new(2))].into();
        let site2: HashSet<_> = [Index::new(2u128, NoSymmSpace::new(3))].into();
        net.add_node("A".to_string(), site1).unwrap();
        net.add_node("B".to_string(), site2).unwrap();

        assert_eq!(net.node_count(), 2);
        assert!(net.has_node(&"A".to_string()));

        // Add edge (no edge data)
        net.add_edge(&"A".to_string(), &"B".to_string()).unwrap();

        assert_eq!(net.edge_count(), 1);
    }
}

