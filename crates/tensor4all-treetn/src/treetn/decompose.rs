//! TreeTN decomposition from dense tensor.
//!
//! This module provides functions to decompose a dense tensor into a TreeTN
//! using factorization algorithms.

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::{Canonical, FactorizeOptions, IndexLike, TensorLike};

use super::TreeTN;

// ============================================================================
// TreeTopology specification
// ============================================================================

/// Specification for tree topology: defines nodes and index assignments.
///
/// `I` is the full index type. Each node maps to the physical indices in the
/// input tensor. This ensures correct index lookup regardless of tensor index
/// ordering, which can change during factorization, while preserving distinct
/// same-ID indices with different prime levels, tags, or directions.
#[derive(Debug, Clone)]
pub struct TreeTopology<V, I> {
    /// Nodes in the tree (node name -> list of indices belonging to this node)
    pub nodes: HashMap<V, Vec<I>>,
    /// Edges in the tree: (node_a, node_b)
    pub edges: Vec<(V, V)>,
}

impl<V: Clone + Hash + Eq, I: Clone + Eq> TreeTopology<V, I> {
    /// Create a new tree topology with the given nodes and edges.
    ///
    /// # Arguments
    /// * `nodes` - Map from node name to the indices belonging to that node
    /// * `edges` - List of edges as (node_a, node_b) pairs
    pub fn new(nodes: HashMap<V, Vec<I>>, edges: Vec<(V, V)>) -> Self {
        Self { nodes, edges }
    }

    /// Validate that this topology describes a tree.
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn validate(&self) -> std::result::Result<(), TreeTNOperationError> {
        let n = self.nodes.len();
        if n == 0 {
            return Err(anyhow::anyhow!("Tree topology must have at least one node").into());
        }
        if n > 1 && self.edges.len() != n - 1 {
            return Err(anyhow::anyhow!(
                "Tree must have exactly n-1 edges: got {} nodes and {} edges",
                n,
                self.edges.len()
            )
            .into());
        }
        // Check all edge endpoints are valid nodes
        for (a, b) in &self.edges {
            if !self.nodes.contains_key(a) {
                return Err(anyhow::anyhow!("Edge refers to unknown node").into());
            }
            if !self.nodes.contains_key(b) {
                return Err(anyhow::anyhow!("Edge refers to unknown node").into());
            }
        }
        Ok(())
    }
}

// ============================================================================
// Decomposition functions
// ============================================================================

/// Decompose a dense tensor into a TreeTN using QR-based factorization.
///
/// This function takes a dense tensor and a tree topology specification, then
/// recursively decomposes the tensor using QR factorization to create a TreeTN.
///
/// # Algorithm
///
/// 1. Start from a leaf node, factorize to separate that node's physical indices
/// 2. Contract the right factor with remaining tensor, repeat for next edge
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
/// - Factorization fails
pub fn factorize_tensor_to_treetn<T, V>(
    tensor: &T,
    topology: &TreeTopology<V, T::Index>,
    root: &V,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    factorize_tensor_to_treetn_with(tensor, topology, FactorizeOptions::qr(), root)
}

/// Factorize a dense tensor into a TreeTN using specified factorization options.
///
/// This function takes a dense tensor and a tree topology specification, then
/// recursively decomposes the tensor using the specified algorithm to create a TreeTN.
///
/// # Algorithm
///
/// 1. Start from a leaf node, factorize to separate that node's physical indices
/// 2. Contract the right factor with remaining tensor, repeat for next edge
/// 3. Continue until all edges are processed
///
/// # Arguments
/// * `tensor` - The dense tensor to decompose
/// * `topology` - Tree topology specifying nodes, edges, and physical index assignments
/// * `options` - Factorization options (algorithm, max_bond_dim, rtol, etc.)
///
/// # Returns
/// A TreeTN representing the decomposed tensor.
///
/// # Errors
/// Returns an error if:
/// - The topology is invalid
/// - Physical index positions don't match the tensor
/// - Factorization fails
pub fn factorize_tensor_to_treetn_with<T, V>(
    tensor: &T,
    topology: &TreeTopology<V, T::Index>,
    options: FactorizeOptions,
    root: &V,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    factorize_tensor_to_treetn_with_root_impl(tensor, topology, options, root)
        .map_err(TreeTNOperationError::from)
}

fn factorize_tensor_to_treetn_with_root_impl<T, V>(
    tensor: &T,
    topology: &TreeTopology<V, T::Index>,
    options: FactorizeOptions,
    root: &V,
) -> Result<TreeTN<T, V>>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug + Ord,
{
    topology.validate()?;

    let tensor_indices = tensor.external_indices();

    if topology.nodes.len() == 1 {
        // Single node - just wrap the tensor
        let node_name = topology
            .nodes
            .keys()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Tree topology must have at least one node"))?;
        if &node_name != root {
            return Err(anyhow::anyhow!("Requested root node not found in topology"));
        }
        let mut tn = TreeTN::<T, V>::new();
        tn.add_tensor(node_name.clone(), tensor.clone())?;
        tn.set_canonical_region([node_name])?;
        return Ok(tn);
    }

    // Validate that all requested indices exist in the tensor.
    let tensor_index_set: HashSet<_> = tensor_indices.iter().cloned().collect();
    for (node, indices) in &topology.nodes {
        for index in indices {
            if !tensor_index_set.contains(index) {
                return Err(anyhow::anyhow!(
                    "Index {:?} for node {:?} not found in tensor (tensor has {} indices)",
                    index,
                    node,
                    tensor_indices.len()
                ));
            }
        }
    }

    // Validate that each physical index is assigned to at most one node.
    // Duplicate assignment is almost always a topology specification bug and will
    // lead to ambiguous/missing node tensors during decomposition.
    let mut assigned_indices: HashSet<T::Index> = HashSet::new();
    for (node, indices) in &topology.nodes {
        for index in indices {
            if !assigned_indices.insert(index.clone()) {
                return Err(anyhow::anyhow!(
                    "Index {:?} is assigned to multiple nodes (at least {:?})",
                    index,
                    node
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
        adj.get_mut(a)
            .ok_or_else(|| anyhow::anyhow!("Edge refers to unknown node {:?}", a))?
            .push(b.clone());
        adj.get_mut(b)
            .ok_or_else(|| anyhow::anyhow!("Edge refers to unknown node {:?}", b))?
            .push(a.clone());
    }
    // Sort each adjacency list to ensure deterministic traversal order
    for neighbors in adj.values_mut() {
        neighbors.sort();
    }

    // Root is required. This ensures the norm-carrying tensor (the final `S*Vh` in
    // a left-canonical decomposition) ends up on a caller-chosen node.
    if !adj.contains_key(root) {
        return Err(anyhow::anyhow!("Requested root node not found in topology"));
    }

    // Build traversal order using BFS from root
    let mut traversal_order: Vec<(V, Option<V>)> = Vec::new(); // (node, parent)
    let mut visited: HashSet<V> = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back((root.clone(), None::<V>));

    while let Some((node, parent)) = queue.pop_front() {
        if visited.contains(&node) {
            continue;
        }
        visited.insert(node.clone());
        traversal_order.push((node.clone(), parent));

        let neighbors = adj
            .get(&node)
            .ok_or_else(|| anyhow::anyhow!("Traversal reached unknown node {:?}", node))?;
        for neighbor in neighbors {
            if !visited.contains(neighbor) {
                queue.push_back((neighbor.clone(), Some(node.clone())));
            }
        }
    }
    if visited.len() != topology.nodes.len() {
        return Err(anyhow::anyhow!(
            "Tree topology must be connected: reached {} of {} nodes from root {:?}",
            visited.len(),
            topology.nodes.len(),
            root
        ));
    }

    let mut children_by_parent: HashMap<V, Vec<V>> = HashMap::new();
    for (node, parent) in &traversal_order {
        if let Some(parent) = parent {
            children_by_parent
                .entry(parent.clone())
                .or_default()
                .push(node.clone());
        }
    }
    for children in children_by_parent.values_mut() {
        children.sort();
    }

    // Reverse traversal order to process leaves first (post-order)
    traversal_order.reverse();

    // Store intermediate tensors as we decompose
    let mut current_tensor = tensor.clone();

    // Store the resulting node tensors
    let mut node_tensors: HashMap<V, T> = HashMap::new();
    // Store the bond each processed child uses to connect to its parent.
    let mut child_bonds: HashMap<V, T::Index> = HashMap::new();

    // Use provided factorization options with Left canonical direction
    let factorize_options = FactorizeOptions {
        canonical: Canonical::Left,
        ..options
    };

    // Process nodes in post-order (leaves first)
    for (node, _parent) in traversal_order
        .iter()
        .take(traversal_order.len().saturating_sub(1))
    {
        // Get the indices for this node.
        let node_indices = topology
            .nodes
            .get(node)
            .ok_or_else(|| anyhow::anyhow!("Topology is missing node {:?}", node))?;

        // Keep this node's physical indices and the bonds to already-factorized
        // children on the left side. This preserves the requested tree topology
        // instead of collapsing all processed children directly into the root.
        let current_indices = current_tensor.external_indices();
        let mut desired_indices: HashSet<T::Index> = node_indices.iter().cloned().collect();
        if let Some(children) = children_by_parent.get(node) {
            for child in children {
                let bond = child_bonds.get(child).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing child bond for node {:?} while processing parent {:?}",
                        child,
                        node
                    )
                })?;
                desired_indices.insert(bond.clone());
            }
        }
        let left_inds: Vec<_> = current_indices
            .iter()
            .filter(|idx| desired_indices.contains(*idx))
            .cloned()
            .collect();

        if left_inds.is_empty() && current_indices.len() > 1 {
            // This indicates an inconsistent topology specification for the current tensor.
            // Previously we "skipped" such nodes, but that can lead to missing tensors and
            // panics later when building the TreeTN.
            return Err(anyhow::anyhow!(
                "No physical indices found for node {:?} (requested indices={:?}) in current tensor indices={:?}",
                node,
                node_indices,
                current_indices
            ));
        }

        // Perform factorization using TensorFactorizationLike::factorize
        // left will have the node's physical indices + bond index
        // right will have bond index + remaining indices
        let factorize_result = current_tensor
            .factorize(&left_inds, &factorize_options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

        let left_indices = factorize_result.left.external_indices();
        let right_indices = factorize_result.right.external_indices();
        let shared_bonds =
            tensor4all_core::index_ops::common_inds::<T::Index>(&left_indices, &right_indices);
        if shared_bonds.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected exactly one parent bond for node {:?}, found {}",
                node,
                shared_bonds.len()
            ));
        }
        child_bonds.insert(node.clone(), shared_bonds[0].clone());

        // Store left as the node's tensor (with physical indices + bond to parent)
        node_tensors.insert(node.clone(), factorize_result.left);

        // right becomes the current tensor for the next iteration
        current_tensor = factorize_result.right;
    }

    // The last node (root) gets the remaining tensor
    let (root_node, _) = traversal_order
        .last()
        .ok_or_else(|| anyhow::anyhow!("Tree traversal produced no nodes"))?;
    node_tensors.insert(root_node.clone(), current_tensor);

    // Build the TreeTN using from_tensors (auto-connection by contractable indices).
    // Since factorize() returns shared bond_index, tensors already have matching bonds.
    // IMPORTANT: Sort node_names to ensure deterministic ordering (HashMap iteration is non-deterministic)
    let mut node_names: Vec<V> = topology.nodes.keys().cloned().collect();
    node_names.sort();
    let tensors: Vec<T> = node_names
        .iter()
        .map(|name| {
            node_tensors
                .get(name)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Missing tensor for topology node {:?}", name))
        })
        .collect::<Result<_>>()?;

    let mut tn = TreeTN::from_tensors(tensors, node_names)?;
    tn.set_canonical_region([root.clone()])?;
    Ok(tn)
}

#[cfg(test)]
mod tests;
