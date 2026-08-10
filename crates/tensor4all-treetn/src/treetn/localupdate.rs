//! Local update operations for TreeTN.
//!
//! This module provides APIs for:
//! - Extracting a sub-tree from a TreeTN (creating a new TreeTN object)
//! - Replacing a sub-tree with another TreeTN of the same appearance
//! - Generating sweep plans for local update algorithms (truncation, fitting)
//!
//! These operations are fundamental for local update algorithms in tensor networks.

use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

use anyhow::{Context, Result};

use tensor4all_core::{IndexLike, TensorLike};

use super::TreeTN;
use crate::node_name_network::NodeNameNetwork;

// ============================================================================
// Local Update Sweep Plan
// ============================================================================

/// A single step in a local update sweep.
///
/// Each step specifies:
/// - Which nodes to extract for local update
/// - Where the canonical center should be after the update
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalUpdateStep<V> {
    /// Nodes to extract for local update.
    /// For nsite=1: single node
    /// For nsite=2: two adjacent nodes (an edge)
    pub nodes: Vec<V>,

    /// The canonical center after this update step.
    /// For nsite=1: same as `nodes[0]`.
    /// For nsite=2: the node in the direction of the next step.
    pub new_center: V,
}

/// A complete sweep plan for local updates.
///
/// Generated from an Euler tour, this plan specifies the sequence of
/// local update operations for algorithms like truncation and fitting.
///
/// # Sweep Direction
/// The sweep follows an Euler tour starting from the root, visiting each
/// edge twice (forward and backward). This ensures all bonds are updated
/// in both directions, which is essential for algorithms like DMRG/TEBD.
///
/// # nsite Parameter
/// - `nsite=1`: Single-site updates. Each step extracts one node.
/// - `nsite=2`: Two-site updates. Each step extracts two adjacent nodes (an edge).
///
/// Two-site updates are more expensive but can change bond dimensions and
/// are necessary for algorithms like TDVP-2 or two-site DMRG.
#[derive(Debug, Clone)]
pub struct LocalUpdateSweepPlan<V> {
    /// The sequence of update steps.
    pub steps: Vec<LocalUpdateStep<V>>,

    /// Number of sites per update (1 or 2).
    pub nsite: usize,
}

impl<V> LocalUpdateSweepPlan<V>
where
    V: Clone + Hash + Eq + Send + Sync + Debug,
{
    /// Generate a sweep plan from a TreeTN's topology.
    ///
    /// Convenience method that extracts the NodeNameNetwork topology from a TreeTN.
    pub fn from_treetn<T>(treetn: &TreeTN<T, V>, root: &V, nsite: usize) -> Option<Self>
    where
        T: TensorLike,
    {
        Self::new(treetn.site_index_network().topology(), root, nsite)
    }

    /// Generate a sweep plan from a NodeNameNetwork.
    ///
    /// Uses Euler tour traversal to visit all edges in both directions.
    ///
    /// # Arguments
    /// * `network` - The network topology
    /// * `root` - The starting node for the sweep
    /// * `nsite` - Number of sites per update (1 or 2)
    ///
    /// # Returns
    /// A sweep plan, or `None` if the root doesn't exist or nsite is invalid.
    ///
    /// # Example
    /// For nsite=1 on chain A-B-C with root B:
    /// - Euler tour vertices: [B, A, B, C, B]
    /// - Steps: [(B, B), (A, A), (B, B), (C, C)] (each vertex except last)
    ///
    /// For nsite=2 on chain A-B-C with root B:
    /// - Euler tour edges: [(B,A), (A,B), (B,C), (C,B)]
    /// - Steps: [({B,A}, A), ({A,B}, B), ({B,C}, C), ({C,B}, B)]
    pub fn new(network: &NodeNameNetwork<V>, root: &V, nsite: usize) -> Option<Self> {
        if nsite != 1 && nsite != 2 {
            return None;
        }

        let root_idx = network.node_index(root)?;

        match nsite {
            1 => {
                // nsite=1: Use vertex sequence from Euler tour
                let vertices = network.euler_tour_vertices_by_index(root_idx);
                if vertices.is_empty() {
                    return Some(Self::empty(nsite));
                }

                // Each vertex (except the last return to root) is a step
                // The new_center is the current vertex itself
                let steps: Vec<_> = vertices
                    .iter()
                    .take(vertices.len().saturating_sub(1))
                    .filter_map(|&v| {
                        let name = network.node_name(v)?.clone();
                        Some(LocalUpdateStep {
                            nodes: vec![name.clone()],
                            new_center: name,
                        })
                    })
                    .collect();

                Some(Self { steps, nsite })
            }
            2 => {
                // nsite=2: Use edge sequence from Euler tour
                let edges = network.euler_tour_edges_by_index(root_idx);
                if edges.is_empty() {
                    // Single node: no edges to update
                    return Some(Self::empty(nsite));
                }

                // Each edge (u, v) becomes a step with nodes [u, v]
                // The new_center is v (the direction we're moving)
                let steps: Vec<_> = edges
                    .iter()
                    .filter_map(|&(u, v)| {
                        let u_name = network.node_name(u)?.clone();
                        let v_name = network.node_name(v)?.clone();
                        Some(LocalUpdateStep {
                            nodes: vec![u_name, v_name.clone()],
                            new_center: v_name,
                        })
                    })
                    .collect();

                Some(Self { steps, nsite })
            }
            _ => None,
        }
    }

    /// Create an empty sweep plan.
    pub fn empty(nsite: usize) -> Self {
        Self {
            steps: Vec::new(),
            nsite,
        }
    }

    /// Check if the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of update steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Iterate over the steps.
    pub fn iter(&self) -> impl Iterator<Item = &LocalUpdateStep<V>> {
        self.steps.iter()
    }
}

// ============================================================================
// Boundary edge/bond utilities
// ============================================================================

/// Boundary edge information: (node_in_region, neighbor_outside, bond_index).
///
/// Represents an edge connecting a node inside the region to a neighbor outside the region.
#[derive(Debug, Clone)]
pub struct BoundaryEdge<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq,
{
    /// Node inside the region
    pub node_in_region: V,
    /// Neighbor outside the region
    pub neighbor_outside: V,
    /// Bond index connecting node_in_region to neighbor_outside
    pub bond_index: T::Index,
}

/// Get all boundary edges for a given region in a TreeTN.
///
/// Returns edges connecting nodes inside the region to neighbors outside the region.
/// This is useful for maintaining stable bond IDs across updates (e.g., for environment cache consistency).
///
/// # Arguments
/// * `treetn` - The TreeTN to analyze
/// * `region` - Nodes that are inside the region
///
/// # Returns
/// Vector of boundary edges, each containing the node in region, neighbor outside, and bond index.
/// # Errors
///
/// Returns an error when the region or graph is invalid (an invalid-topology
/// /// failure).
///
pub fn get_boundary_edges<T, V>(
    treetn: &TreeTN<T, V>,
    region: &[V],
) -> Result<Vec<BoundaryEdge<T, V>>>
where
    T: TensorLike,
    T::Index: IndexLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    let mut boundary_edges = Vec::new();
    let region_set: HashSet<&V> = region.iter().collect();

    for node in region {
        for neighbor in treetn.site_index_network().neighbors(node) {
            if !region_set.contains(&neighbor) {
                // This is a boundary edge: node is in region, neighbor is outside
                if let Some(edge) = treetn.edge_between(node, &neighbor) {
                    if let Some(bond) = treetn.bond_index(edge) {
                        boundary_edges.push(BoundaryEdge {
                            node_in_region: node.clone(),
                            neighbor_outside: neighbor.clone(),
                            bond_index: bond.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(boundary_edges)
}

// ============================================================================
// LocalUpdater trait
// ============================================================================

/// Trait for local update operations during a sweep.
///
/// Implementors of this trait provide the actual update logic that transforms
/// a local subtree into an updated version. This allows different algorithms
/// (truncation, fitting, DMRG, TDVP) to share the same sweep infrastructure.
///
/// # Type Parameters
/// - `T`: Tensor type implementing TensorLike
/// - `V`: Node name type
///
/// # Workflow
/// During `apply_local_update_sweep`:
/// 1. For each step in the sweep plan:
///    a. Extract the subtree containing `step.nodes`
///    b. Call `update()` with the extracted subtree and step info
///    c. Replace the subtree in the original TreeTN with the updated one
///    d. Update the canonical center to `step.new_center`
pub trait LocalUpdater<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Optional hook called before performing an update step.
    ///
    /// This is called with the full TreeTN state *before* the update is applied.
    /// Implementors can use it to validate assumptions or prefetch/update caches.
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn before_step(
        &mut self,
        _step: &LocalUpdateStep<V>,
        _full_treetn_before: &TreeTN<T, V>,
    ) -> Result<()> {
        Ok(())
    }

    /// Update a local subtree.
    ///
    /// # Arguments
    /// * `subtree` - The extracted subtree to update
    /// * `step` - The current step information (nodes and new_center)
    /// * `full_treetn` - Reference to the full (global) TreeTN. This provides global context
    ///
    ///   (e.g., topology, neighbor relations, and index/bond metadata) that some update
    ///   algorithms may need. It may be unused by simple updaters.
    ///
    /// # Returns
    /// The updated subtree, which must have the same "appearance" as the input
    /// (same nodes, same external indices, same ortho_towards structure).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn update(
        &mut self,
        subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>>;

    /// Optional hook called after an update step has been applied to the full TreeTN.
    ///
    /// This is called after:
    /// - The updated subtree has been inserted back into the full TreeTN
    /// - The canonical center has been moved to `step.new_center`
    ///
    /// Implementors can use this to update caches that must see the post-update state.
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn after_step(
        &mut self,
        _step: &LocalUpdateStep<V>,
        _full_treetn_after: &TreeTN<T, V>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Apply a local update sweep to a TreeTN.
///
/// This function orchestrates the sweep by:
/// 1. Iterating through the sweep plan
/// 2. For each step:
///    a. Validate that the canonical center is a single node within the extracted subtree
///    b. Extract the local subtree
///    c. Call the updater to transform it
///    d. Replace the subtree back into the TreeTN
///
/// # Arguments
/// * `treetn` - The TreeTN to update (modified in place)
/// * `plan` - The sweep plan specifying the update order
/// * `updater` - The local updater implementation
///
/// # Preconditions
/// - The TreeTN must be canonicalized with a single-node canonical center
/// - The canonical center must be within the first step's nodes
///
/// # Returns
/// `Ok(())` if the sweep completes successfully.
///
/// # Errors
///
/// Returns an error when a local update in the sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
/// # Errors
///
/// Returns an error when a local update in the sweep fails (a shape or index
/// /// mismatch, a non-convergence failure, or a backend failure).
///
pub fn apply_local_update_sweep<T, V, U>(
    treetn: &mut TreeTN<T, V>,
    plan: &LocalUpdateSweepPlan<V>,
    updater: &mut U,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    U: LocalUpdater<T, V>,
{
    for step in plan.iter() {
        // Validate: canonical_region must be a single node within the step's nodes
        let canonical_region = treetn.canonical_region();
        if canonical_region.is_empty() {
            return Err(anyhow::anyhow!(
                "TreeTN is not canonicalized: canonical_region is empty"
            ))
            .context("apply_local_update_sweep: TreeTN must be canonicalized before sweep");
        }
        if canonical_region.len() != 1 {
            return Err(anyhow::anyhow!(
                "canonical_region must be a single node, got {} nodes",
                canonical_region.len()
            ))
            .context("apply_local_update_sweep: canonical_region must be a single node");
        }
        let center_node = canonical_region.iter().next().ok_or_else(|| {
            anyhow::anyhow!("canonical_region reported one node but yielded none")
        })?;
        let step_nodes_set: HashSet<V> = step.nodes.iter().cloned().collect();
        if !step_nodes_set.contains(center_node) {
            return Err(anyhow::anyhow!(
                "canonical_region {:?} is not within the extracted subtree {:?}",
                center_node,
                step.nodes
            ))
            .context(
                "apply_local_update_sweep: canonical_region must be within extracted subtree",
            );
        }

        updater
            .before_step(step, treetn)
            .context("apply_local_update_sweep: LocalUpdater::before_step failed")?;

        // Extract subtree for the nodes in this step
        let subtree = treetn.extract_subtree(&step.nodes)?;

        // Apply the update
        let updated_subtree = updater.update(subtree, step, treetn)?;

        // Replace the subtree back
        treetn.replace_subtree(&step.nodes, &updated_subtree)?;

        // Update canonical center
        treetn.set_canonical_region([step.new_center.clone()])?;

        updater
            .after_step(step, treetn)
            .context("apply_local_update_sweep: LocalUpdater::after_step failed")?;
    }

    Ok(())
}

// ============================================================================
// TruncateUpdater - LocalUpdater implementation for truncation
// ============================================================================

use tensor4all_core::{Canonical, FactorizeOptions, SvdTruncationPolicy};

/// Truncation updater for nsite=2 sweeps.
///
/// This updater performs SVD-based truncation on two-site subtrees,
/// compressing bond dimensions while preserving the tensor train structure.
///
/// # Algorithm
/// For each step with nodes [A, B] where B is the new center:
/// 1. Contract tensors A and B into a single tensor AB
/// 2. Factorize AB using SVD with truncation (left indices = A's external + bond to A's other neighbors)
/// 3. The left tensor becomes the new A, the right tensor becomes the new B
/// 4. B is the orthogonality center (isometry pointing towards B)
///
/// # Usage
/// ```
/// use tensor4all_core::{DynIndex, TensorDynLen};
/// use tensor4all_treetn::{apply_local_update_sweep, LocalUpdateSweepPlan, TreeTN, TruncateUpdater};
///
/// # fn main() -> anyhow::Result<()> {
/// let s0 = DynIndex::new_dyn(2);
/// let bond = DynIndex::new_dyn(1);
/// let s1 = DynIndex::new_dyn(2);
/// let t0 = TensorDynLen::from_dense(vec![s0, bond.clone()], vec![1.0, 0.0])?;
/// let t1 = TensorDynLen::from_dense(vec![bond, s1], vec![1.0, 0.0])?;
/// let mut treetn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1])?;
/// treetn.canonicalize_mut(std::iter::once(0usize), Default::default())?;
///
/// let plan = LocalUpdateSweepPlan::from_treetn(&treetn, &0usize, 2).unwrap();
/// let mut updater = TruncateUpdater::new(
///     Some(4),
///     Some(tensor4all_core::SvdTruncationPolicy::new(1e-10)),
/// );
/// apply_local_update_sweep(&mut treetn, &plan, &mut updater)?;
///
/// assert_eq!(treetn.node_count(), 2);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TruncateUpdater {
    /// Maximum bond dimension after truncation.
    pub max_rank: Option<usize>,
    /// Explicit SVD truncation policy.
    pub svd_policy: Option<SvdTruncationPolicy>,
}

impl TruncateUpdater {
    /// Create a new truncation updater.
    ///
    /// # Arguments
    /// * `max_rank` - Maximum bond dimension (None for no limit)
    /// * `svd_policy` - SVD truncation policy override (None uses the global default)
    pub fn new(max_rank: Option<usize>, svd_policy: Option<SvdTruncationPolicy>) -> Self {
        Self {
            max_rank,
            svd_policy,
        }
    }
}

impl<T, V> LocalUpdater<T, V> for TruncateUpdater
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    fn update(
        &mut self,
        mut subtree: TreeTN<T, V>,
        step: &LocalUpdateStep<V>,
        _full_treetn: &TreeTN<T, V>,
    ) -> Result<TreeTN<T, V>> {
        // TruncateUpdater is designed for nsite=2
        if step.nodes.len() != 2 {
            return Err(anyhow::anyhow!(
                "TruncateUpdater requires exactly 2 nodes, got {}",
                step.nodes.len()
            ));
        }

        let node_a = &step.nodes[0];
        let node_b = &step.nodes[1];

        // Get node indices
        let idx_a = subtree
            .node_index(node_a)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_a))?;
        let idx_b = subtree
            .node_index(node_b)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", node_b))?;

        // Get the bond between A and B
        let edge_ab = subtree
            .edge_between(node_a, node_b)
            .ok_or_else(|| anyhow::anyhow!("No edge between {:?} and {:?}", node_a, node_b))?;
        let bond_ab = subtree
            .bond_index(edge_ab)
            .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
            .clone();

        // Contract A and B
        let tensor_a = subtree
            .tensor(idx_a)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in subtree", node_a))?;
        let tensor_b = subtree
            .tensor(idx_b)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?} in subtree", node_b))?;
        let tensor_ab = T::contract(&[tensor_a, tensor_b]).context("Failed to contract A and B")?;

        // Determine left indices (indices that will remain on A after factorization)
        // These are: all indices of A except the bond to B
        let left_inds: Vec<_> = tensor_a
            .external_indices()
            .iter()
            .filter(|idx| *idx != &bond_ab)
            .cloned()
            .collect();

        // Set up factorization options
        let mut options = FactorizeOptions::svd().with_canonical(Canonical::Left); // Left canonical: A is isometry, B has the norm

        if let Some(max_rank) = self.max_rank {
            options = options.with_max_rank(max_rank);
        }
        if let Some(policy) = self.svd_policy {
            options = options.with_svd_policy(policy);
        }

        // Factorize
        let factorize_result = tensor_ab
            .factorize(&left_inds, &options)
            .map_err(|e| anyhow::anyhow!("Factorization failed: {}", e))?;

        let new_tensor_a = factorize_result.left;
        let new_tensor_b = factorize_result.right;
        let new_bond = factorize_result.bond_index;

        // Update the subtree - first update the edge bond, then the tensors
        // The factorize result creates a new bond index, so we update the edge to use it
        subtree.replace_edge_bond(edge_ab, new_bond.clone())?;
        subtree.replace_tensor(idx_a, new_tensor_a)?;
        subtree.replace_tensor(idx_b, new_tensor_b)?;

        // Set ortho_towards: bond points towards new_center (B)
        subtree.set_ortho_towards(&new_bond, Some(step.new_center.clone()));

        // Set canonical center to the new center
        subtree.set_canonical_region([step.new_center.clone()])?;

        Ok(subtree)
    }
}

// ============================================================================
// Sub-tree extraction
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Extract a sub-tree from this TreeTN.
    ///
    /// Creates a new TreeTN containing only the specified nodes and their
    /// connecting edges. Tensors are cloned into the new TreeTN.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to include in the sub-tree
    ///
    /// # Returns
    /// A new TreeTN containing the specified sub-tree, or an error if:
    /// - Any specified node doesn't exist
    /// - The specified nodes don't form a connected subtree
    ///
    /// # Notes
    /// - Bond indices between included nodes are preserved
    /// - Bond indices to excluded nodes become external (site) indices in the sub-tree
    /// - ortho_towards directions are copied for edges within the sub-tree
    /// - canonical_region is intersected with the extracted nodes
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn extract_subtree(&self, node_names: &[V]) -> Result<Self>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Ord,
    {
        if node_names.is_empty() {
            return Err(anyhow::anyhow!("Cannot extract empty subtree"));
        }

        // Validate all nodes exist
        for name in node_names {
            if self.graph.node_index(name).is_none() {
                return Err(anyhow::anyhow!("Node {:?} does not exist", name))
                    .context("extract_subtree: invalid node name");
            }
        }

        // Check connectivity: the specified nodes must form a connected subtree
        let node_indices: HashSet<_> = node_names
            .iter()
            .filter_map(|n| self.graph.node_index(n))
            .collect();

        if !self.site_index_network.is_connected_subset(&node_indices) {
            return Err(anyhow::anyhow!(
                "Specified nodes do not form a connected subtree"
            ))
            .context("extract_subtree: nodes must be connected");
        }

        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();

        // Create new TreeTN with extracted tensors
        let mut subtree = TreeTN::<T, V>::new();

        // Step 1: Add all nodes with their tensors
        for name in node_names {
            let node_idx = self
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} does not exist", name))?;
            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", name))?
                .clone();

            subtree
                .add_tensor(name.clone(), tensor)
                .context("extract_subtree: failed to add tensor")?;
        }

        // Step 2: Add edges between nodes in the subtree
        // Track which edges we've already added to avoid duplicates
        let mut added_edges: HashSet<(V, V)> = HashSet::new();

        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                // Only add edge if neighbor is also in the subtree
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                // Avoid adding the same edge twice (undirected)
                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if added_edges.contains(&edge_key) {
                    continue;
                }
                added_edges.insert(edge_key);

                // Get bond index from original TreeTN
                let orig_edge = self.edge_between(name, &neighbor).ok_or_else(|| {
                    anyhow::anyhow!("Edge not found between {:?} and {:?}", name, neighbor)
                })?;

                let bond_index = self
                    .bond_index(orig_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found"))?
                    .clone();

                // Get node indices in new subtree
                let subtree_node_a = subtree
                    .graph
                    .node_index(name)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", name))?;
                let subtree_node_b = subtree
                    .graph
                    .node_index(&neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in subtree", neighbor))?;

                // Connect in subtree
                subtree
                    .connect(subtree_node_a, &bond_index, subtree_node_b, &bond_index)
                    .context("extract_subtree: failed to connect nodes")?;

                // Copy ortho_towards if it exists (keyed by full bond index)
                if let Some(ortho_dir) = self.ortho_towards.get(&bond_index) {
                    // Only copy if the direction node is in the subtree
                    if node_name_set.contains(ortho_dir) {
                        subtree
                            .ortho_towards
                            .insert(bond_index.clone(), ortho_dir.clone());
                    }
                }
            }
        }

        // Step 3: Set canonical_region to intersection with extracted nodes
        let new_center: HashSet<V> = self
            .canonical_region
            .intersection(&node_name_set)
            .cloned()
            .collect();
        subtree.canonical_region = new_center;

        // Copy canonical_form if any center nodes were included
        if !subtree.canonical_region.is_empty() {
            subtree.canonical_form = self.canonical_form;
        }

        Ok(subtree)
    }
}

// ============================================================================
// Sub-tree replacement
// ============================================================================

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Replace a sub-tree with another TreeTN of the same topology.
    ///
    /// This method replaces the tensors and ortho_towards directions for a subset
    /// of nodes with those from another TreeTN. The replacement TreeTN must have
    /// the same topology (nodes and edges) as the sub-tree being replaced.
    ///
    /// # Arguments
    /// * `node_names` - The names of nodes to replace
    /// * `replacement` - The TreeTN to use as replacement
    ///
    /// # Returns
    /// `Ok(())` if the replacement succeeds, or an error if:
    /// - Any specified node doesn't exist
    /// - The replacement doesn't have the same topology as the extracted sub-tree
    /// - Tensor replacement fails
    ///
    /// # Notes
    /// - The replacement TreeTN must have the same nodes, edges, and site indices
    /// - Bond dimensions may differ (this is the typical use case for truncation)
    /// - ortho_towards may differ (will be copied from replacement)
    /// - The original TreeTN is modified in-place
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn replace_subtree(&mut self, node_names: &[V], replacement: &Self) -> Result<()>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
        V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        if node_names.is_empty() {
            return Ok(()); // Nothing to replace
        }

        // Extract current subtree for comparison
        let current_subtree = self.extract_subtree(node_names)?;

        // Verify that replacement has the same topology (nodes and edges)
        // Note: site index network may differ due to bond dimension changes in truncation
        if !current_subtree.same_topology(replacement) {
            return Err(anyhow::anyhow!(
                "Replacement TreeTN does not have the same topology as the current subtree"
            ))
            .context("replace_subtree: topology mismatch");
        }

        let node_name_set: HashSet<V> = node_names.iter().cloned().collect();
        let mut processed_edges: HashSet<(V, V)> = HashSet::new();

        // Step 1: Update edge bond indices FIRST (before replacing tensors)
        // This is crucial because replace_tensor validates that tensors contain connection indices
        for name in node_names {
            let neighbors: Vec<V> = self.site_index_network.neighbors(name).collect();

            for neighbor in neighbors {
                // Only process edges within the subtree
                if !node_name_set.contains(&neighbor) {
                    continue;
                }

                let edge_key = if *name < neighbor {
                    (name.clone(), neighbor.clone())
                } else {
                    (neighbor.clone(), name.clone())
                };

                if processed_edges.contains(&edge_key) {
                    continue;
                }
                processed_edges.insert(edge_key.clone());

                // Get edges in both self and replacement
                let self_edge = self
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found in self"))?;
                let replacement_edge = replacement
                    .edge_between(name, &neighbor)
                    .ok_or_else(|| anyhow::anyhow!("Edge not found in replacement"))?;

                // Get new bond index from replacement
                let new_bond = replacement
                    .bond_index(replacement_edge)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in replacement"))?
                    .clone();

                // Update bond index in self
                self.replace_edge_bond(self_edge, new_bond.clone())
                    .with_context(|| {
                        format!(
                            "replace_subtree: failed to update bond between {:?} and {:?}",
                            name, neighbor
                        )
                    })?;

                // Copy ortho_towards from replacement (using the new bond)
                match replacement.ortho_towards.get(&new_bond) {
                    Some(dir) => {
                        self.ortho_towards.insert(new_bond, dir.clone());
                    }
                    None => {
                        self.ortho_towards.remove(&new_bond);
                    }
                }
            }
        }

        // Step 2: Replace tensors (now bond indices match)
        for name in node_names {
            let self_node_idx = self
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", name))?;
            let replacement_node_idx = replacement
                .graph
                .node_index(name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in replacement", name))?;

            let new_tensor = replacement
                .tensor(replacement_node_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in replacement", name)
                })?
                .clone();

            self.replace_tensor(self_node_idx, new_tensor)
                .with_context(|| {
                    format!(
                        "replace_subtree: failed to replace tensor at node {:?}",
                        name
                    )
                })?;
        }

        // Update canonical_region: remove old nodes, add from replacement
        for name in node_names {
            self.canonical_region.remove(name);
        }
        for name in &replacement.canonical_region {
            if node_name_set.contains(name) {
                self.canonical_region.insert(name.clone());
            }
        }

        // Update canonical_form if replacement has one
        if replacement.canonical_form.is_some() {
            self.canonical_form = replacement.canonical_form;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests;
