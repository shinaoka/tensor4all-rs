//! Structural transformation operations for TreeTN.
//!
//! This module provides methods to transform a TreeTN's structure:
//! - [`fuse_to`](TreeTN::fuse_to): Merge adjacent nodes to match a target structure
//! - [`split_to`](TreeTN::split_to): Split nodes to match a target structure

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};
use petgraph::stable_graph::{NodeIndex, StableGraph};

use tensor4all_core::{
    index_ops, Canonical, FactorizeAlg, FactorizeOptions, IndexLike, TensorLike,
};

use super::TreeTN;
use crate::options::SplitOptions;
use crate::site_index_network::SiteIndexNetwork;

type SplitBoundaryIndices<V, TargetV, I> = HashMap<V, HashMap<TargetV, HashSet<I>>>;

/// Compute the Steiner tree in the full graph spanning a set of terminal nodes.
///
/// For tree graphs, the Steiner tree is the union of unique shortest paths
/// from one terminal node to every other terminal node.
fn steiner_tree_indices<T: TensorLike>(
    graph: &StableGraph<T, T::Index, petgraph::Undirected>,
    terminals: &HashSet<NodeIndex>,
) -> HashSet<NodeIndex> {
    if terminals.len() <= 1 {
        return terminals.clone();
    }
    let terms: Vec<NodeIndex> = terminals.iter().copied().collect();
    let root = terms[0];
    let mut result = HashSet::new();
    result.insert(root);
    for &term in &terms[1..] {
        if let Some((_, path)) =
            petgraph::algo::astar(graph, root, |n| n == term, |_| 1usize, |_| 0usize)
        {
            result.extend(path);
        }
    }
    result
}

fn canonical_node_pair<NodeName>(left: &NodeName, right: &NodeName) -> (NodeName, NodeName)
where
    NodeName: Clone + Ord,
{
    if left <= right {
        (left.clone(), right.clone())
    } else {
        (right.clone(), left.clone())
    }
}

fn choose_site_free_absorption_target<NodeName>(
    neighbor_targets: &HashSet<NodeName>,
    target_edges: &HashSet<(NodeName, NodeName)>,
) -> Option<NodeName>
where
    NodeName: Clone + Hash + Eq + Ord,
{
    if neighbor_targets.is_empty() {
        return None;
    }

    let mut candidates = neighbor_targets.iter().cloned().collect::<Vec<_>>();
    candidates.sort();
    candidates.into_iter().find(|candidate| {
        neighbor_targets.iter().all(|other| {
            candidate == other || target_edges.contains(&canonical_node_pair(candidate, other))
        })
    })
}

/// Check if nodes form a connected induced subgraph in the given graph.
/// DFS restricted to edges where both endpoints are in `nodes`.
fn is_connected_subset_on_graph<T: TensorLike>(
    graph: &StableGraph<T, T::Index, petgraph::Undirected>,
    nodes: &HashSet<NodeIndex>,
) -> bool {
    if nodes.is_empty() || nodes.len() == 1 {
        return true;
    }
    let Some(&start) = nodes.iter().next() else {
        return true;
    };
    let mut seen = HashSet::new();
    let mut stack = vec![start];
    seen.insert(start);
    while let Some(v) = stack.pop() {
        for nb in graph.neighbors(v) {
            if nodes.contains(&nb) && seen.insert(nb) {
                stack.push(nb);
            }
        }
    }
    seen.len() == nodes.len()
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Fuse (merge) adjacent nodes to match the target structure.
    ///
    /// This operation contracts adjacent nodes that should be merged according to
    /// the target `SiteIndexNetwork`. The target structure must be a "coarsening"
    /// of the current structure: each target node should contain the site indices
    /// of one or more adjacent current nodes.
    ///
    /// # Algorithm
    ///
    /// 1. Compare current structure with target structure
    /// 2. Map each current node to its target node (by matching site indices)
    /// 3. For each group of current nodes mapping to the same target node:
    ///    - Contract all nodes in the group into a single node
    /// 4. Build the new TreeTN with the fused structure
    ///
    /// # Arguments
    /// * `target` - The target `SiteIndexNetwork` defining the desired structure
    ///
    /// # Returns
    /// A new TreeTN with the fused structure, or an error if:
    /// - The target structure is incompatible with the current structure
    /// - Nodes to be fused are not connected
    ///
    /// # Properties
    /// - **Bond dimension**: Unchanged (pure contraction, no truncation)
    /// - **Commutative**: Non-overlapping groups can be merged in any order
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    /// # Example
    /// ```text
    /// Before: x1_1---x2_1---x1_2---x2_2---x1_3---x2_3  (6 nodes)
    /// After:  {x1_1,x2_1}---{x1_2,x2_2}---{x1_3,x2_3}  (3 nodes)
    /// ```
    pub fn fuse_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
    ) -> std::result::Result<TreeTN<T, TargetV>, TreeTNOperationError>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build a mapping from site index to current node name
        let mut site_to_current_node: HashMap<T::Index, V> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                for site_idx in site_space {
                    site_to_current_node.insert(site_idx.clone(), current_node_name.clone());
                }
            }
        }

        // Step 2: For each target node, find which current nodes should be merged
        // Map: target node name -> set of current node names
        let mut target_to_current: HashMap<TargetV, HashSet<V>> = HashMap::new();

        for target_node_name in target.node_names() {
            let target_site_space = target.site_space(target_node_name).ok_or_else(|| {
                anyhow::anyhow!("Target node {:?} has no site space", target_node_name)
            })?;

            let mut current_nodes_for_target: HashSet<V> = HashSet::new();
            for target_site_idx in target_site_space {
                if let Some(current_node) = site_to_current_node.get(target_site_idx) {
                    current_nodes_for_target.insert(current_node.clone());
                }
            }

            if current_nodes_for_target.is_empty() {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Target node {:?} has site indices not found in current TreeTN",
                        target_node_name
                    )
                    .context("fuse_to: incompatible target structure"),
                ));
            }

            target_to_current.insert(target_node_name.clone(), current_nodes_for_target);
        }

        // Step 3: Validate that every current node maps to exactly one target node
        let mut current_to_target: HashMap<V, TargetV> = HashMap::new();
        for (target_name, current_nodes) in &target_to_current {
            for current_node in current_nodes {
                if let Some(existing_target) = current_to_target.get(current_node) {
                    return Err(TreeTNOperationError::from(
                        anyhow::anyhow!(
                            "Current node {:?} maps to multiple target nodes: {:?} and {:?}",
                            current_node,
                            existing_target,
                            target_name
                        )
                        .context("fuse_to: ambiguous mapping"),
                    ));
                }
                current_to_target.insert(current_node.clone(), target_name.clone());
            }
        }

        // Check all current nodes are accounted for.
        // Nodes without site indices (internal bonding nodes) are skipped —
        // contract_node_group auto-expands via Steiner tree to include them.
        for current_name in self.node_names() {
            if !current_to_target.contains_key(&current_name)
                && self
                    .site_space(&current_name)
                    .is_some_and(|s| !s.is_empty())
            {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Current node {:?} has site indices but no corresponding target node",
                        current_name
                    )
                    .context("fuse_to: missing target for current node"),
                ));
            }
        }

        // Step 4: Expand target groups to include internal connector nodes
        // (nodes without site indices that serve as connectors between site-bearing members).
        let g = self.graph.graph();
        for current_nodes in target_to_current.values_mut() {
            let seed_indices: HashSet<NodeIndex> = current_nodes
                .iter()
                .filter_map(|name| self.graph.node_index(name))
                .collect();
            if seed_indices.len() <= 1 {
                continue;
            }
            let steiner = steiner_tree_indices(g, &seed_indices);
            for idx in &steiner {
                if let Some(name) = self.graph.node_name(*idx) {
                    // Only include internal nodes (no private site indices) —
                    // site-bearing nodes belong to their own target group.
                    if self.site_space(name).is_none_or(|s| s.is_empty()) {
                        current_nodes.insert(name.clone());
                    }
                }
            }
        }

        // Step 4b: Absorb site-free dangling subtrees into the unique adjacent
        // target group, or into an adjacent target group that preserves all
        // requested quotient edges. These nodes carry only gauge/internal
        // factors; dropping them would change values.
        let target_edges: HashSet<_> = target
            .edges()
            .map(|(left, right)| canonical_node_pair(&left, &right))
            .collect();
        let mut current_to_target = HashMap::<V, TargetV>::new();
        for (target_name, current_nodes) in &target_to_current {
            for current_node in current_nodes {
                current_to_target.insert(current_node.clone(), target_name.clone());
            }
        }
        loop {
            let mut additions = Vec::<(TargetV, V)>::new();
            for current_name in self.node_names() {
                if current_to_target.contains_key(&current_name)
                    || self
                        .site_space(&current_name)
                        .is_some_and(|site_space| !site_space.is_empty())
                {
                    continue;
                }

                let neighbor_targets: HashSet<TargetV> = self
                    .site_index_network
                    .neighbors(&current_name)
                    .filter_map(|neighbor| current_to_target.get(&neighbor).cloned())
                    .collect();
                if let Some(target_name) =
                    choose_site_free_absorption_target(&neighbor_targets, &target_edges)
                {
                    additions.push((target_name, current_name));
                }
            }

            if additions.is_empty() {
                break;
            }

            for (target_name, current_name) in additions {
                target_to_current
                    .entry(target_name.clone())
                    .or_default()
                    .insert(current_name.clone());
                current_to_target.insert(current_name, target_name);
            }
        }

        // Step 5: For each target node, contract all its current nodes into one tensor
        let mut result_tensors: HashMap<TargetV, T> = HashMap::new();

        for (target_name, current_nodes) in &target_to_current {
            let contracted = self.contract_node_group(current_nodes).with_context(|| {
                format!(
                    "fuse_to: failed to contract nodes for target {:?}",
                    target_name
                )
            })?;
            result_tensors.insert(target_name.clone(), contracted);
        }

        // Step 5: Build the new TreeTN
        // Sort target node names for deterministic ordering
        let mut target_names: Vec<TargetV> = target.node_names().into_iter().cloned().collect();
        target_names.sort();

        let tensors: Vec<T> = target_names
            .iter()
            .map(|name| {
                result_tensors.remove(name).ok_or_else(|| {
                    anyhow::anyhow!("fuse_to: missing contracted tensor for target {:?}", name)
                })
            })
            .collect::<Result<_>>()?;

        let result = TreeTN::<T, TargetV>::from_tensors(tensors, target_names)
            .context("fuse_to: failed to build result TreeTN")?;

        Ok(result)
    }

    /// Contract a group of nodes into a single tensor.
    ///
    /// The nodes must form a connected subtree in the current TreeTN.
    /// Contracts all internal bonds (bonds between nodes in the group),
    /// keeping external bonds and site indices.
    fn contract_node_group(&self, nodes: &HashSet<V>) -> Result<T>
    where
        V: Ord,
    {
        if nodes.is_empty() {
            return Err(anyhow::anyhow!("Cannot contract empty node group"));
        }

        // Convert node names to NodeIndex
        let node_indices: HashSet<NodeIndex> = nodes
            .iter()
            .filter_map(|name| self.graph.node_index(name))
            .collect();

        if node_indices.len() != nodes.len() {
            return Err(anyhow::anyhow!(
                "Some nodes not found in graph: expected {} nodes, found {}",
                nodes.len(),
                node_indices.len()
            ));
        }

        // Validate connectivity on the full graph (includes internal nodes)
        let g = self.graph.graph();
        if !is_connected_subset_on_graph(g, &node_indices) {
            return Err(anyhow::anyhow!(
                "Nodes to contract do not form a connected subtree"
            ));
        }

        // Pick a root (smallest node name for determinism)
        let root_name = nodes
            .iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("Cannot contract empty node group"))?;
        let root_idx = self
            .graph
            .node_index(root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found in graph", root_name))?;

        // Get edges within the group, ordered from leaves to root. Reuse the
        // network's post-order DFS (single O(V+E) pass over the named graph)
        // instead of duplicating a local traversal.
        let post_order = self.site_index_network().post_order_dfs_by_index(root_idx);

        // BFS from the root builds the parent map needed to orient internal
        // edges toward the contraction root (single O(V+E) pass). This is not
        // equivalent to the post-order list alone: the post-order gives the
        // contraction sequence, while parents orient each edge for the
        // einsum pairing.
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut bfs = petgraph::visit::Bfs::new(g, root_idx);
        while let Some(node) = bfs.next(g) {
            for neighbor in g.neighbors(node) {
                if neighbor != root_idx {
                    parent.entry(neighbor).or_insert(node);
                }
            }
        }
        let internal_edges: Vec<(NodeIndex, NodeIndex)> = post_order
            .iter()
            .filter(|&&node| node != root_idx && node_indices.contains(&node))
            .filter_map(|&node| {
                let p = parent.get(&node)?;
                if node_indices.contains(p) {
                    Some((node, *p))
                } else {
                    None
                }
            })
            .collect();

        // Initialize with cloned tensors
        let mut tensors: HashMap<NodeIndex, T> = node_indices
            .iter()
            .filter_map(|&idx| self.tensor(idx).cloned().map(|t| (idx, t)))
            .collect();

        // Contract along each internal edge (from leaves to root)
        for (from, to) in internal_edges {
            let from_tensor = tensors
                .remove(&from)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", from))?;
            let to_tensor = tensors
                .remove(&to)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", to))?;

            // Contract using TensorContractionLike::contract
            // (bond indices are auto-detected via is_contractable)
            let contracted = T::contract(&[&to_tensor, &from_tensor])
                .map_err(|e| anyhow::anyhow!("Failed to contract tensors: {}", e))?;

            tensors.insert(to, contracted);
        }

        // The root tensor is the result
        tensors
            .remove(&root_idx)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result at root"))
    }

    /// Split nodes to match the target structure.
    ///
    /// This operation splits nodes that contain site indices belonging to multiple
    /// target nodes. The target structure must be a "refinement" of the current
    /// structure: each current node's site indices should map to one or more
    /// target nodes.
    ///
    /// # Algorithm (Two-Phase Approach)
    ///
    /// **Phase 1: Exact factorization (no truncation)**
    /// 1. Build mapping: site index -> target node name
    /// 2. For each current node, check if its site indices map to multiple target nodes
    /// 3. If so, split the node using QR factorization
    /// 4. Repeat until all nodes match the target structure
    ///
    /// **Phase 2: Truncation sweep (optional)**
    /// If `options.final_sweep` is true, perform a truncation sweep to optimize
    /// bond dimensions globally.
    ///
    /// # Arguments
    /// * `target` - The target `SiteIndexNetwork` defining the desired structure
    /// * `options` - Options controlling truncation and final sweep
    ///
    /// # Returns
    /// A new TreeTN with the split structure, or an error if:
    /// - The target structure is incompatible with the current structure
    /// - Factorization fails
    ///
    /// # Properties
    /// - **Bond dimension**: May increase during split, controlled by truncation
    /// - **Exact (Phase 1)**: Without truncation, represents the same tensor
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, an
    /// /// SVD or non-convergence failure, or a backend failure).
    ///
    /// # Example
    /// ```text
    /// Before: {x1_1,x2_1}---{x1_2,x2_2}---{x1_3,x2_3}  (3 nodes, fused)
    /// After:  x1_1---x2_1---x1_2---x2_2---x1_3---x2_3  (6 nodes, interleaved)
    /// ```
    pub fn split_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        options: &SplitOptions,
    ) -> std::result::Result<TreeTN<T, TargetV>, TreeTNOperationError>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Step 1: Build mapping from site index to target node name
        let mut site_to_target: HashMap<T::Index, TargetV> = HashMap::new();
        for target_node_name in target.node_names() {
            if let Some(site_space) = target.site_space(target_node_name) {
                for site_idx in site_space {
                    site_to_target.insert(site_idx.clone(), target_node_name.clone());
                }
            }
        }

        let mut site_to_current: HashMap<T::Index, V> = HashMap::new();
        for current_node_name in self.node_names() {
            let site_space = self.site_space(&current_node_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: current node {:?} has no registered site space",
                    current_node_name
                )
            })?;
            for site_idx in site_space {
                site_to_current.insert(site_idx.clone(), current_node_name.clone());
            }
        }

        let mut target_to_current: HashMap<TargetV, V> = HashMap::new();
        for target_node_name in target.node_names() {
            let site_space = target.site_space(target_node_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: target node {:?} has no registered site space",
                    target_node_name
                )
            })?;
            let current_names: HashSet<_> = site_space
                .iter()
                .map(|site_idx| {
                    site_to_current.get(site_idx).cloned().ok_or_else(|| {
                        anyhow::anyhow!(
                            "split_to: target site index {:?} is missing from the current network",
                            site_idx
                        )
                    })
                })
                .collect::<Result<_>>()?;
            if current_names.len() != 1 {
                return Err(anyhow::anyhow!(
                    "split_to: target node {:?} spans multiple current nodes; use restructure_to for mixed regrouping",
                    target_node_name
                ).into());
            }
            let current_name = current_names.into_iter().next().ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: target node {:?} has no current owner",
                    target_node_name
                )
            })?;
            target_to_current.insert(target_node_name.clone(), current_name);
        }

        let boundary_indices = if target.edge_count() == 0 {
            HashMap::new()
        } else {
            self.split_boundary_indices_by_target(target, &target_to_current)
                .context("split_to: assign current bonds to target boundary fragments")?
        };

        // Step 2: For each current node, determine which target nodes it maps to
        // and validate the mapping
        let mut current_to_targets: HashMap<V, HashSet<TargetV>> = HashMap::new();
        for current_node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&current_node_name) {
                let mut targets_for_node: HashSet<TargetV> = HashSet::new();
                for site_idx in site_space {
                    if let Some(target_name) = site_to_target.get(site_idx) {
                        targets_for_node.insert(target_name.clone());
                    } else {
                        return Err(TreeTNOperationError::from(
                            anyhow::anyhow!(
                                "Site index {:?} in current node {:?} has no corresponding target node",
                                site_idx,
                                current_node_name
                            )
                            .context("split_to: incompatible target structure"),
                        ));
                    }
                }
                current_to_targets.insert(current_node_name.clone(), targets_for_node);
            }
        }

        // Step 3: Phase 1 - Split all nodes that need splitting
        // Collect all resulting tensors with their target node names
        let mut result_tensors: Vec<(TargetV, T)> = Vec::new();

        for current_node_name in self.node_names() {
            let node_idx = self
                .node_index(&current_node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", current_node_name))?;
            let tensor = self
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for {:?}", current_node_name))?;

            let targets_for_node = current_to_targets.get(&current_node_name).ok_or_else(|| {
                anyhow::anyhow!("No target mapping for node {:?}", current_node_name)
            })?;

            if targets_for_node.len() == 1 {
                // No split needed - just relabel
                let target_name = targets_for_node.iter().next().cloned().ok_or_else(|| {
                    anyhow::anyhow!("No target mapping for node {:?}", current_node_name)
                })?;
                result_tensors.push((target_name, tensor.clone()));
            } else {
                // Need to split this node
                let fragment_target = build_fragment_sub_target(target, targets_for_node)?;
                let split_tensors = self
                    .split_tensor_for_targets(
                        tensor,
                        &site_to_target,
                        boundary_indices.get(&current_node_name),
                        &fragment_target,
                    )
                    .with_context(|| {
                        format!("split_to: failed to split node {:?}", current_node_name)
                    })?;
                result_tensors.extend(split_tensors);
            }
        }

        // Step 4: Build the result TreeTN with target node names
        // Sort by target name for deterministic ordering
        result_tensors.sort_by(|(a, _), (b, _)| a.cmp(b));

        let names: Vec<TargetV> = result_tensors
            .iter()
            .map(|(name, _)| name.clone())
            .collect();
        let tensors: Vec<T> = result_tensors.into_iter().map(|(_, t)| t).collect();

        let result = TreeTN::<T, TargetV>::from_tensors(tensors, names)
            .context("split_to: failed to build result TreeTN")?;

        // Step 5: Validate result topology matches target
        // Skip validation if target has no edges (caller doesn't care about topology)
        if target.edge_count() > 0
            && !result
                .site_index_network()
                .share_equivalent_site_index_network(target)
        {
            return Err(anyhow::anyhow!(
                "split_to: result topology does not match target: \
                 expected edges {:?}, got {:?}",
                target
                    .edges()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect::<Vec<_>>(),
                result
                    .site_index_network()
                    .edges()
                    .map(|(a, b)| (a.clone(), b.clone()))
                    .collect::<Vec<_>>(),
            )
            .into());
        }

        // Step 6: Phase 2 - Optional truncation sweep
        if options.final_sweep {
            // Find a center node for truncation
            let center = result.node_names().into_iter().min().ok_or_else(|| {
                anyhow::anyhow!("split_to: no nodes in result for truncation sweep")
            })?;

            let truncation_options = crate::TruncationOptions {
                form: options.form,
                truncation: options.truncation,
            };

            return result
                .truncate([center], truncation_options)
                .context("split_to: truncation sweep failed")
                .map_err(TreeTNOperationError::from);
        }

        Ok(result)
    }

    fn split_boundary_indices_by_target<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        target_to_current: &HashMap<TargetV, V>,
    ) -> Result<SplitBoundaryIndices<V, TargetV, T::Index>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        let mut boundary_indices: SplitBoundaryIndices<V, TargetV, T::Index> = HashMap::new();

        for edge in self.graph.graph().edge_indices() {
            let (node_a, node_b) = self.graph.graph().edge_endpoints(edge).ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing endpoints", edge)
            })?;
            let current_a = self.graph.node_name(node_a).cloned().ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing source node", edge)
            })?;
            let current_b = self.graph.node_name(node_b).cloned().ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has missing target node", edge)
            })?;
            let bond_index = self.graph.graph().edge_weight(edge).ok_or_else(|| {
                anyhow::anyhow!("split_to: current edge {:?} has no bond index", edge)
            })?;

            let mut crossing_edges = Vec::new();
            for (target_a, target_b) in target.edges() {
                let owner_a = target_to_current.get(&target_a).ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: target edge references node {:?} without a current owner",
                        target_a
                    )
                })?;
                let owner_b = target_to_current.get(&target_b).ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: target edge references node {:?} without a current owner",
                        target_b
                    )
                })?;

                if owner_a == &current_a && owner_b == &current_b {
                    crossing_edges.push((target_a, target_b));
                } else if owner_a == &current_b && owner_b == &current_a {
                    crossing_edges.push((target_b, target_a));
                }
            }

            let [(target_for_a, target_for_b)] = crossing_edges.as_slice() else {
                return Err(anyhow::anyhow!(
                    "split_to: expected exactly one target edge crossing current edge {:?}-{:?}, found {}",
                    current_a,
                    current_b,
                    crossing_edges.len()
                ));
            };

            boundary_indices
                .entry(current_a)
                .or_default()
                .entry(target_for_a.clone())
                .or_default()
                .insert(bond_index.clone());
            boundary_indices
                .entry(current_b)
                .or_default()
                .entry(target_for_b.clone())
                .or_default()
                .insert(bond_index.clone());
        }

        Ok(boundary_indices)
    }

    /// Split a tensor into multiple tensors using post-order tree decomposition.
    ///
    /// This uses QR factorization following the target sub-topology (edges between
    /// fragment nodes within the current node). The algorithm mirrors
    /// `factorize_tensor_to_treetn_with_root_impl`: process leaves first, include
    /// child bonds in `left_inds`, root gets the remaining tensor.
    fn split_tensor_for_targets<TargetV>(
        &self,
        tensor: &T,
        site_to_target: &HashMap<T::Index, TargetV>,
        boundary_indices: Option<&HashMap<TargetV, HashSet<T::Index>>>,
        fragment_target: &SiteIndexNetwork<TargetV, T::Index>,
    ) -> Result<Vec<(TargetV, T)>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        // Group tensor's site indices by their target node
        let mut partition: HashMap<TargetV, HashSet<T::Index>> = HashMap::new();
        for idx in tensor.external_indices() {
            if let Some(target_name) = site_to_target.get(&idx) {
                partition
                    .entry(target_name.clone())
                    .or_default()
                    .insert(idx);
            }
        }
        if let Some(boundary_indices) = boundary_indices {
            let tensor_index_set: HashSet<_> = tensor.external_indices().into_iter().collect();
            for (target_name, indices) in boundary_indices {
                let target_partition = partition.entry(target_name.clone()).or_default();
                for index in indices {
                    if !tensor_index_set.contains(index) {
                        return Err(anyhow::anyhow!(
                            "split_to: boundary index {:?} is not present in the tensor for target {:?}",
                            index,
                            target_name
                        ));
                    }
                    target_partition.insert(index.clone());
                }
            }
        }

        let mut target_names: Vec<TargetV> =
            fragment_target.node_names().into_iter().cloned().collect();
        target_names.sort();

        if target_names.len() <= 1 {
            let name = target_names
                .first()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No target nodes found for this tensor"))?;
            return Ok(vec![(name, tensor.clone())]);
        }

        // For edgeless targets, fall back to sequential QR (caller doesn't
        // care about topology, any decomposition is acceptable).
        if fragment_target.edge_count() == 0 {
            return self.split_tensor_for_targets_sequential(tensor, &partition, &target_names);
        }

        // Choose root: prefer the fragment with boundary indices (those need
        // to be on the root so the original bond connects correctly).
        let root = if let Some(bi) = boundary_indices {
            if let Some((name, _)) = bi.iter().max_by_key(|(_, ids)| ids.len()) {
                if fragment_target.node_names().contains(&name) {
                    name.clone()
                } else {
                    target_names[0].clone()
                }
            } else {
                target_names[0].clone()
            }
        } else {
            target_names[0].clone()
        };

        // Post-order traversal (leaves first, root last) via petgraph DfsPostOrder
        let traversal = fragment_target
            .post_order_dfs(&root)
            .ok_or_else(|| anyhow::anyhow!("Root {:?} not found in fragment target", root))?;

        // Build children_by_parent from target edges. The endpoint closer to
        // the root (higher position in post-order) is the parent.
        let mut children_by_parent: HashMap<TargetV, Vec<TargetV>> = HashMap::new();
        for (ref parent_name, ref child_name) in fragment_target.edges() {
            let pos_a = traversal
                .iter()
                .position(|n| *n == *parent_name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: fragment edge references node {:?} outside traversal",
                        parent_name
                    )
                })?;
            let pos_b = traversal
                .iter()
                .position(|n| *n == *child_name)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "split_to: fragment edge references node {:?} outside traversal",
                        child_name
                    )
                })?;
            if pos_a > pos_b {
                children_by_parent
                    .entry(parent_name.clone())
                    .or_default()
                    .push(child_name.clone());
            } else {
                children_by_parent
                    .entry(child_name.clone())
                    .or_default()
                    .push(parent_name.clone());
            }
        }

        // Process nodes in post-order (skip root — it gets the remaining tensor)
        let mut remaining_tensor = tensor.clone();
        let mut result: Vec<(TargetV, T)> = Vec::new();
        let mut child_bonds: HashMap<TargetV, T::Index> = HashMap::new();

        for node in traversal.iter().take(traversal.len() - 1) {
            let node_indices = partition.get(node).cloned().unwrap_or_default();
            let current_indices = remaining_tensor.external_indices();

            // Build set of desired indices: this node's site+boundary indices
            // plus bonds from already-processed children.
            let mut desired_indices = node_indices;
            if let Some(children) = children_by_parent.get(node) {
                for child in children {
                    if let Some(bond) = child_bonds.get(child) {
                        desired_indices.insert(bond.clone());
                    }
                }
            }

            let left_inds: Vec<_> = current_indices
                .iter()
                .filter(|idx| desired_indices.contains(*idx))
                .cloned()
                .collect();

            if left_inds.is_empty() {
                continue;
            }

            let factorize_options = FactorizeOptions {
                alg: FactorizeAlg::QR,
                canonical: Canonical::Left,
                max_bond_dim: None,
                svd_policy: None,
                qr_rtol: Some(0.0),
            };

            let factorize_result = remaining_tensor
                .factorize(&left_inds, &factorize_options)
                .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

            // Detect the parent bond (shared between left and right tensors)
            let left_indices = factorize_result.left.external_indices();
            let right_indices = factorize_result.right.external_indices();
            let shared = index_ops::common_inds(&left_indices, &right_indices);
            if shared.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Expected exactly one bond for node {:?}, found {}",
                    node,
                    shared.len()
                ));
            }
            child_bonds.insert(node.clone(), shared[0].clone());

            result.push((node.clone(), factorize_result.left));
            remaining_tensor = factorize_result.right;
        }

        // Root gets the remaining tensor with all child bonds
        let root_name = traversal
            .last()
            .ok_or_else(|| anyhow::anyhow!("split_to: fragment traversal produced no nodes"))?;
        result.push((root_name.clone(), remaining_tensor));

        Ok(result)
    }

    /// Sequential QR fallback for edgeless targets.
    ///
    /// Processes targets in sorted order, peeling each via QR. The last target
    /// accumulates all bonds (star topology). This is acceptable when the caller
    /// does not specify target edges.
    fn split_tensor_for_targets_sequential<TargetV>(
        &self,
        tensor: &T,
        partition: &HashMap<TargetV, HashSet<T::Index>>,
        target_names: &[TargetV],
    ) -> Result<Vec<(TargetV, T)>>
    where
        TargetV: Clone + Hash + Eq + Ord + std::fmt::Debug,
    {
        let mut remaining_tensor = tensor.clone();
        let mut result: Vec<(TargetV, T)> = Vec::new();

        for target_name in target_names.iter().take(target_names.len() - 1) {
            let site_indices_for_target = partition.get(target_name).ok_or_else(|| {
                anyhow::anyhow!(
                    "split_to: no partitioned indices found for target {:?}",
                    target_name
                )
            })?;

            let left_inds: Vec<_> = remaining_tensor
                .external_indices()
                .iter()
                .filter(|idx| site_indices_for_target.contains(*idx))
                .cloned()
                .collect();

            if left_inds.is_empty() {
                continue;
            }

            let factorize_options = FactorizeOptions {
                alg: FactorizeAlg::QR,
                canonical: Canonical::Left,
                max_bond_dim: None,
                svd_policy: None,
                qr_rtol: Some(0.0),
            };

            let factorize_result = remaining_tensor
                .factorize(&left_inds, &factorize_options)
                .map_err(|e| anyhow::anyhow!("Factorization failed: {:?}", e))?;

            result.push((target_name.clone(), factorize_result.left));
            remaining_tensor = factorize_result.right;
        }

        let last_target = target_names
            .last()
            .ok_or_else(|| anyhow::anyhow!("split_to: sequential split has no target nodes"))?;
        result.push((last_target.clone(), remaining_tensor));

        Ok(result)
    }
}

/// Build a sub-`SiteIndexNetwork` containing only the fragment nodes that
/// belong to a given current node, preserving edges between them.
fn build_fragment_sub_target<TargetV, I>(
    target: &SiteIndexNetwork<TargetV, I>,
    fragment_names: &HashSet<TargetV>,
) -> Result<SiteIndexNetwork<TargetV, I>>
where
    TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    I: IndexLike,
{
    let mut sub = SiteIndexNetwork::with_capacity(fragment_names.len(), 0);
    for name in fragment_names {
        let site_space = target
            .site_space(name)
            .ok_or_else(|| anyhow::anyhow!("Fragment node {:?} not found in target", name))?;
        sub.add_node(name.clone(), site_space.clone())?;
    }
    for (a, b) in target.edges() {
        if fragment_names.contains(&a) && fragment_names.contains(&b) {
            sub.add_edge(&a, &b)?;
        }
    }
    Ok(sub)
}

// Tests are disabled until random module is refactored
#[cfg(test)]
mod tests;
