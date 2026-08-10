//! Random tensor network generation.
//!
//! Provides utilities for creating random tensor networks, useful for testing.
//!
//! Note: Currently only supports `DynId` indices (the default dynamic index type).

use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;
use anyhow::{anyhow, Result};
use rand::Rng;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::tensor::RandomScalar;
use tensor4all_core::TensorDynLen;

/// Specification for link (bond) dimensions.
///
/// Used when creating random tensor networks to specify the dimension of each bond.
#[derive(Debug, Clone)]
pub enum LinkSpace<V> {
    /// All links have the same dimension.
    Uniform(usize),
    /// Each edge has its own dimension.
    /// The map uses ordered pairs `(min(a, b), max(a, b))` as keys for consistency.
    PerEdge(HashMap<(V, V), usize>),
}

impl<V> LinkSpace<V> {
    /// Create a uniform link space where all bonds have the same dimension.
    pub fn uniform(dim: usize) -> Self {
        Self::Uniform(dim)
    }

    /// Create a per-edge link space from a map of edge dimensions.
    pub fn per_edge(dims: HashMap<(V, V), usize>) -> Self {
        Self::PerEdge(dims)
    }
}

impl<V: Clone + Ord + Hash> LinkSpace<V> {
    /// Get the dimension for an edge between two nodes.
    ///
    /// For `PerEdge`, the key is normalized to `(min(a, b), max(a, b))`.
    pub fn get(&self, a: &V, b: &V) -> Option<usize> {
        match self {
            LinkSpace::Uniform(dim) => Some(*dim),
            LinkSpace::PerEdge(map) => {
                let key = if a < b {
                    (a.clone(), b.clone())
                } else {
                    (b.clone(), a.clone())
                };
                map.get(&key).copied()
            }
        }
    }
}

/// Type alias for the default index type used in random generation.
pub type DefaultIndex = Index<DynId, TagSet>;

/// Create a random TreeTN from a site index network (generic over scalar type).
///
/// Generates random tensors at each node with:
/// - Site indices from the `site_network`
/// - Link indices created according to `link_space`
///
/// # Type Parameters
/// * `T` - Scalar type (e.g. `f64` or `Complex64`)
/// * `R` - RNG type
/// * `V` - Node name type
///
/// # Arguments
/// * `rng` - Random number generator for tensor data
/// * `site_network` - Network topology and site (physical) indices
/// * `link_space` - Specification for bond dimensions
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Example
/// ```
/// use tensor4all_treetn::{SiteIndexNetwork, random_treetn, LinkSpace};
/// use tensor4all_core::index::{Index, DynId, TagSet};
/// use rand::SeedableRng;
/// use rand_chacha::ChaCha8Rng;
/// use std::collections::HashSet;
///
/// // Create a simple 2-node network
/// let mut site_network = SiteIndexNetwork::<String, Index<DynId, TagSet>>::new();
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// site_network.add_node("A".to_string(), HashSet::from([i.clone()])).unwrap();
/// site_network.add_node("B".to_string(), HashSet::from([j.clone()])).unwrap();
/// site_network.add_edge(&"A".to_string(), &"B".to_string()).unwrap();
///
/// let mut rng = ChaCha8Rng::seed_from_u64(42);
/// let treetn = random_treetn::<f64, _, _>(&mut rng, &site_network, LinkSpace::uniform(4)).unwrap();
///
/// assert_eq!(treetn.node_count(), 2);
/// ```
pub fn random_treetn<T, R, V>(
    rng: &mut R,
    site_network: &SiteIndexNetwork<V, DefaultIndex>,
    link_space: LinkSpace<V>,
) -> Result<TreeTN<TensorDynLen, V>>
where
    T: RandomScalar,
    R: Rng,
    V: Clone + Hash + Eq + Ord + Send + Sync + Debug,
{
    // Step 1: Create link indices for each edge
    // Key: (smaller_name, larger_name), Value: link index
    let mut link_indices: HashMap<(V, V), DefaultIndex> = HashMap::new();

    // Get all edges from the site network topology
    for (a, b) in site_network.edges() {
        let key = if a < b {
            (a.clone(), b.clone())
        } else {
            (b.clone(), a.clone())
        };
        let key_clone = (key.0.clone(), key.1.clone());

        if let std::collections::hash_map::Entry::Vacant(entry) = link_indices.entry(key) {
            let dim = link_space
                .get(&key_clone.0, &key_clone.1)
                .ok_or_else(|| anyhow!("LinkSpace has no dimension for edge {:?}", key_clone))?;
            entry.insert(Index::new_dyn(dim));
        }
    }

    // Step 2: For each node, collect all indices and create random tensor
    let mut tensors = Vec::new();
    let mut node_names = Vec::new();

    for node_name in site_network.node_names() {
        let node_name = node_name.clone();

        // Collect site indices
        let site_inds = site_network
            .site_space(&node_name)
            .cloned()
            .unwrap_or_default();

        // Collect link indices from edges connected to this node
        let mut all_indices: Vec<DefaultIndex> = site_inds.into_iter().collect();

        for neighbor in site_network.neighbors(&node_name) {
            let key = if node_name < neighbor {
                (node_name.clone(), neighbor.clone())
            } else {
                (neighbor.clone(), node_name.clone())
            };

            if let Some(link_idx) = link_indices.get(&key) {
                all_indices.push(link_idx.clone());
            }
        }

        // Create random tensor
        let tensor = TensorDynLen::random::<T, R>(rng, all_indices)?;

        tensors.push(tensor);
        node_names.push(node_name);
    }

    // Step 3: Create TreeTN from tensors
    TreeTN::from_tensors(tensors, node_names).map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests;
