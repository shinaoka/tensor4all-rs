//! Contraction and operations for TreeTN.
//!
//! This module provides methods for:
//! - Replacing internal indices with fresh IDs (`sim_internal_inds`)
//! - Contracting TreeTN to tensor (`contract_to_tensor`)
//! - Zip-up contraction (`contract_zipup`)
//! - Naive contraction (`contract_naive`)
//! - Validation (`validate_ortho_consistency`)

use crate::error::TreeTNOperationError;
use petgraph::stable_graph::{EdgeIndex, NodeIndex};
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use tensor4all_core::{
    Canonical, FactorizeAlg, FactorizeOptions, IndexLike, SvdTruncationPolicy, TensorIndex,
    TensorLike,
};

use super::TreeTN;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ZipupTopologyMode {
    PruneScalarSubtrees,
    PreserveInputTopology,
}

fn indices_except_exact<I: IndexLike>(indices: &[I], excluded: &[I]) -> Vec<I> {
    tensor4all_core::index_ops::unique_inds(indices, excluded)
}

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Create a copy with all internal (link/bond) indices replaced by fresh IDs.
    ///
    /// External (site/physical) indices remain unchanged. This is useful when
    /// contracting two TreeTNs that might have overlapping internal index IDs.
    ///
    /// # Returns
    /// A new TreeTN with all bond indices replaced by `sim` indices (same dimension,
    /// new unique ID).
    pub fn sim_internal_inds(&self) -> Self {
        // Clone the structure
        let mut result = self.clone();

        // For each edge, create a sim index and update both the edge and tensors
        let edges: Vec<EdgeIndex> = result.graph.graph().edge_indices().collect();

        for edge in edges {
            // Get the current bond index
            let old_bond_idx = match result.bond_index(edge) {
                Some(idx) => idx.clone(),
                None => continue,
            };

            // Create a new sim index (same dimension, new ID)
            let new_bond_idx = old_bond_idx.sim();

            // Get the endpoint nodes
            let (node_a, node_b) = match result.graph.graph().edge_endpoints(edge) {
                Some(endpoints) => endpoints,
                None => continue,
            };

            // Update the edge weight
            if let Some(edge_weight) = result.graph.graph_mut().edge_weight_mut(edge) {
                *edge_weight = new_bond_idx.clone();
            }

            // Update tensor at node_a
            if let Some(tensor_a) = result.graph.graph_mut().node_weight_mut(node_a) {
                if let Ok(new_tensor) = tensor_a.replaceind(&old_bond_idx, &new_bond_idx) {
                    *tensor_a = new_tensor;
                }
            }

            // Update tensor at node_b
            if let Some(tensor_b) = result.graph.graph_mut().node_weight_mut(node_b) {
                if let Ok(new_tensor) = tensor_b.replaceind(&old_bond_idx, &new_bond_idx) {
                    *tensor_b = new_tensor;
                }
            }
        }

        result
    }

    /// Contract the TreeTN to a single tensor.
    ///
    /// This method contracts all tensors in the network into a single tensor
    /// containing all physical indices. The contraction is performed using
    /// an edge-based order (post-order DFS edges towards root), processing
    /// each edge in sequence and using Connection information to identify
    /// which indices to contract.
    ///
    /// The result has only site (physical) indices; all bond indices are summed out.
    /// See also [`to_dense`](Self::to_dense), which is an alias for this method.
    ///
    /// # Returns
    /// A single tensor representing the full contraction of the network.
    ///
    /// # Errors
    ///
    /// Returns an error when the dense contraction fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex, TensorLike};
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond, s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    /// let dense = tn.contract_to_tensor().unwrap();
    ///
    /// // Result has only site indices
    /// assert_eq!(dense.num_external_indices(), 2);
    /// ```
    pub fn contract_to_tensor(&self) -> std::result::Result<T, TreeTNOperationError>
    where
        V: Ord,
    {
        if self.node_count() == 0 {
            return Err(anyhow::anyhow!("Cannot contract empty TreeTN").into());
        }

        if self.node_count() == 1 {
            // Single node - just return a clone of its tensor
            let node = self
                .graph
                .graph()
                .node_indices()
                .next()
                .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
            return self
                .tensor(node)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Tensor not found"))
                .map_err(TreeTNOperationError::from);
        }

        // Validate tree structure
        self.validate_tree()
            .context("contract_to_tensor: graph must be a tree")?;

        // Choose a deterministic root (minimum node name)
        let root_name = self
            .graph
            .graph()
            .node_indices()
            .filter_map(|idx| self.graph.node_name(idx).cloned())
            .min()
            .ok_or_else(|| anyhow::anyhow!("No nodes found"))?;
        let root = self
            .graph
            .node_index(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node not found"))?;

        // Get edges to process (post-order DFS edges towards root)
        let edges = self.site_index_network.edges_to_canonicalize(None, root);

        // Initialize with original tensors
        let mut tensors: HashMap<NodeIndex, T> = self
            .graph
            .graph()
            .node_indices()
            .filter_map(|n| self.tensor(n).cloned().map(|t| (n, t)))
            .collect();

        // Process each edge: contract tensor at `from` into tensor at `to`
        for (from, to) in edges {
            let from_tensor = tensors
                .remove(&from)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", from))?;
            let to_tensor = tensors
                .remove(&to)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", to))?;

            // Contract and store result at `to`
            // (bond indices are auto-detected via is_contractable)
            let contracted = T::contract(&[&to_tensor, &from_tensor])
                .context("Failed to contract along edge")?;
            tensors.insert(to, contracted);
        }

        // The root's tensor is the final result
        let result = tensors
            .remove(&root)
            .ok_or_else(|| anyhow::anyhow!("Contraction produced no result"))?;

        // Permute result indices to match canonical site index order:
        // node names sorted, then site indices per node in consistent order
        let mut expected_indices: Vec<T::Index> = Vec::new();
        let mut node_names: Vec<V> = self.node_names();
        node_names.sort();
        for node_name in &node_names {
            if let Some(site_space) = self.site_space(node_name) {
                // Use site indices in insertion order (deterministic from site_space)
                expected_indices.extend(site_space.iter().cloned());
            }
        }

        // Get current index order from result tensor
        let current_indices = result.external_indices();

        // Check if permutation is needed
        if current_indices.len() != expected_indices.len() {
            // This shouldn't happen, but return as-is if sizes don't match
            return Ok(result);
        }

        // Check if already in correct order
        let already_ordered = current_indices
            .iter()
            .zip(expected_indices.iter())
            .all(|(c, e)| c == e);

        if already_ordered {
            return Ok(result);
        }

        // Build permutation: for each expected index, find its position in current indices
        // Then use replaceind to reorder (permuting indices)
        result
            .permuteinds(&expected_indices)
            .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))
    }

    /// Contract two TreeTNs with the same topology using the zip-up algorithm.
    ///
    /// The zip-up algorithm traverses from leaves towards the center, contracting
    /// corresponding nodes from both networks and optionally truncating at each step.
    ///
    /// # Algorithm
    /// 1. Replace internal (bond) indices of both networks with fresh IDs to avoid collision
    /// 2. Traverse from leaves towards center
    /// 3. At each edge (child → parent):
    ///    - Contract the child tensors from both networks (along their shared site indices)
    ///    - Factorize, keeping site indices + parent bond on left (canonical form)
    ///    - Store left factor as child tensor in result
    ///    - Contract right factor into parent tensor
    /// 4. Contract the final center tensors
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    /// * `center` - The center node name towards which to contract
    /// * `svd_policy` - Optional SVD truncation policy
    /// * `max_bond_dim` - Optional maximum bond dimension
    ///
    /// # Returns
    /// The contracted TreeTN result, or an error if topologies don't match or contraction fails.
    /// # Errors
    ///
    /// Returns an error when the zip-up contraction fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn contract_zipup(
        &self,
        other: &Self,
        center: &V,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        self.contract_zipup_with(
            other,
            center,
            CanonicalForm::Unitary,
            svd_policy,
            max_bond_dim,
        )
    }

    /// Contract two TreeTNs with the same topology using zip-up and a specified canonical form.
    ///
    /// # Algorithm
    /// 1. Process leaves: contract `A[leaf] * B[leaf]`, factorize, store R at parent
    /// 2. Process internal nodes: contract incoming factors with `A[node]` and `B[node]`, factorize, store R\_new at parent
    /// 3. Process root: contract `[R_list..., A[root], B[root]]`, store as final result
    /// 4. Set canonical center
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    /// * `center` - The center node name towards which to contract
    /// * `form` - Canonical form (Unitary/LU/CI)
    /// * `svd_policy` - Optional SVD truncation policy
    /// * `max_bond_dim` - Optional maximum bond dimension
    ///
    /// # Returns
    /// The contracted TreeTN result, or an error if topologies don't match or contraction fails.
    /// # Errors
    ///
    /// Returns an error when the zip-up contraction fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn contract_zipup_with(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
    ) -> std::result::Result<Self, TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        self.contract_zipup_impl(
            other,
            center,
            form,
            svd_policy,
            max_bond_dim,
            ZipupTopologyMode::PruneScalarSubtrees,
        )
        .map_err(TreeTNOperationError::from)
    }

    pub(crate) fn contract_zipup_preserving_topology_with(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        self.contract_zipup_impl(
            other,
            center,
            form,
            svd_policy,
            max_bond_dim,
            ZipupTopologyMode::PreserveInputTopology,
        )
    }

    fn contract_zipup_impl(
        &self,
        other: &Self,
        center: &V,
        form: CanonicalForm,
        svd_policy: Option<SvdTruncationPolicy>,
        max_bond_dim: Option<usize>,
        topology_mode: ZipupTopologyMode,
    ) -> Result<Self>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // 1. Verify topologies are compatible
        if !self.same_topology(other) {
            return Err(anyhow::anyhow!(
                "contract_zipup_with: networks have incompatible topologies"
            ));
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn_a = self.sim_internal_inds();
        let tn_b = other.sim_internal_inds();

        // 3. Get traversal edges from leaves to center (post-order DFS)
        let edges = tn_a.edges_to_canonicalize_by_names(center).ok_or_else(|| {
            anyhow::anyhow!("contract_zipup_with: center node {:?} not found", center)
        })?;

        // 4. Handle single node case
        if edges.is_empty() && self.node_count() == 1 {
            let node_idx = tn_a
                .graph
                .graph()
                .node_indices()
                .next()
                .ok_or_else(|| anyhow::anyhow!("contract_zipup_with: no nodes found"))?;
            let t_a = tn_a
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup_with: tensor not found in tn_a"))?;
            let t_b = tn_b
                .tensor(tn_b.graph.graph().node_indices().next().ok_or_else(|| {
                    anyhow::anyhow!("contract_zipup_with: tensor not found in tn_b")
                })?)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup_with: tensor not found in tn_b"))?;

            let contracted = t_a.contract_pair(t_b)?;
            let node_name = tn_a
                .graph
                .node_name(node_idx)
                .ok_or_else(|| anyhow::anyhow!("contract_zipup_with: node name not found"))?;

            let mut result = TreeTN::new();
            result.add_tensor(node_name.clone(), contracted)?;
            result.set_canonical_region(std::iter::once(center.clone()))?;
            return Ok(result);
        }

        // 5. Initialize intermediate tensors storage: HashMap<node_name, Vec<intermediate_tensor>>
        let mut intermediate_tensors: HashMap<V, Vec<T>> = HashMap::new();

        // 6. Initialize result tensors: HashMap<node_name, tensor>
        let mut result_tensors: HashMap<V, T> = HashMap::new();

        // 7. Determine which nodes are leaves (for processing logic)
        let root_name = center.clone();

        // Helper: Get bond index between two nodes
        let get_bond_index = |tn: &TreeTN<T, V>, node_a: &V, node_b: &V| -> Result<T::Index> {
            let edge = tn.edge_between(node_a, node_b).ok_or_else(|| {
                anyhow::anyhow!("Edge not found between {:?} and {:?}", node_a, node_b)
            })?;
            tn.bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))
                .cloned()
        };

        // 8. Set up factorization options based on form
        let alg = match form {
            CanonicalForm::Unitary => FactorizeAlg::SVD,
            CanonicalForm::LU => FactorizeAlg::LU,
            CanonicalForm::CI => FactorizeAlg::CI,
        };

        let mut factorize_options = match alg {
            FactorizeAlg::SVD => FactorizeOptions::svd(),
            FactorizeAlg::QR => FactorizeOptions::qr(),
            FactorizeAlg::LU => FactorizeOptions::lu(),
            FactorizeAlg::CI => FactorizeOptions::ci(),
        }
        .with_canonical(Canonical::Left);

        if let Some(max_bond_dim) = max_bond_dim {
            factorize_options = factorize_options.with_max_bond_dim(max_bond_dim);
        }
        if let Some(policy) = svd_policy {
            factorize_options = factorize_options.with_svd_policy(policy);
        }
        factorize_options
            .validate()
            .map_err(|err| anyhow::anyhow!("invalid zipup factorization options: {err}"))?;

        // 9. Process edges from leaves towards root
        for (source_name, destination_name) in &edges {
            // Get tensors from both networks
            let node_a_idx = tn_a
                .node_index(source_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_a", source_name))?;
            let node_b_idx = tn_b
                .node_index(source_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in tn_b", source_name))?;

            let tensor_a = tn_a
                .tensor(node_a_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in tn_a", source_name)
                })?
                .clone();
            let tensor_b = tn_b
                .tensor(node_b_idx)
                .ok_or_else(|| {
                    anyhow::anyhow!("Tensor not found for node {:?} in tn_b", source_name)
                })?
                .clone();

            // Check if this is a leaf node (no intermediate tensors accumulated yet)
            let is_leaf = !intermediate_tensors.contains_key(source_name)
                || intermediate_tensors
                    .get(source_name)
                    .map(|v| v.is_empty())
                    .unwrap_or(true);

            let c_temp = if is_leaf {
                // Leaf node: contract A[source] * B[source]
                tensor_a
                    .contract_pair(&tensor_b)
                    .context("Failed to contract leaf tensors")?
            } else {
                // Internal node: contract [R_accumulated..., A[source], B[source]]
                let mut tensor_list = Vec::new();
                if let Some(r_list) = intermediate_tensors.remove(source_name) {
                    tensor_list.extend(r_list);
                }
                tensor_list.push(tensor_a);
                tensor_list.push(tensor_b);
                let tensor_refs: Vec<&T> = tensor_list.iter().collect();
                T::contract(&tensor_refs).context("Failed to contract internal node tensors")?
            };

            // Factorize child tensor and pass the right factor to destination (even if destination is root)
            let bond_to_dest_a = get_bond_index(&tn_a, source_name, destination_name)
                .context("Failed to get bond index to destination in tn_a")?;
            let bond_to_dest_b = get_bond_index(&tn_b, source_name, destination_name)
                .context("Failed to get bond index to destination in tn_b")?;

            // left_inds = all indices except the two parent bonds (keep site + child bonds)
            let c_temp_indices = c_temp.external_indices();
            let excluded_parent_bonds = [bond_to_dest_a, bond_to_dest_b];
            let left_inds = indices_except_exact(&c_temp_indices, &excluded_parent_bonds);

            if left_inds.is_empty() {
                match topology_mode {
                    ZipupTopologyMode::PruneScalarSubtrees => {
                        // If no left indices remain, pass the tensor directly to destination.
                        intermediate_tensors
                            .entry(destination_name.clone())
                            .or_default()
                            .push(c_temp);
                    }
                    ZipupTopologyMode::PreserveInputTopology => {
                        // Fit sweeps require C to retain A/B's node set. When a
                        // subtree has no surviving site indices, keep it connected
                        // by a dimension-1 dummy link instead of pruning the node.
                        let (dummy_left, dummy_right) = T::Index::create_dummy_link_pair();
                        let left_tensor = T::ones(std::slice::from_ref(&dummy_left))
                            .context("Failed to create topology-preserving dummy left tensor")?;
                        let dummy_right_tensor = T::ones(std::slice::from_ref(&dummy_right))
                            .context("Failed to create topology-preserving dummy right tensor")?;
                        let right_tensor = c_temp
                            .outer_product(&dummy_right_tensor)
                            .context("Failed to attach topology-preserving dummy bond")?;

                        result_tensors.insert(source_name.clone(), left_tensor);
                        intermediate_tensors
                            .entry(destination_name.clone())
                            .or_default()
                            .push(right_tensor);
                    }
                }
                continue;
            }

            let factorize_result = c_temp
                .factorize(&left_inds, &factorize_options)
                .context("Failed to factorize")?;

            // Store left factor as result tensor for source node
            result_tensors.insert(source_name.clone(), factorize_result.left);

            // Store right factor (intermediate tensor R) at destination
            intermediate_tensors
                .entry(destination_name.clone())
                .or_default()
                .push(factorize_result.right);

            // Note: bond index update will be handled when building the result TreeTN
        }

        // 9.5. Process root node (if it has intermediate tensors accumulated)
        if let Some(r_list) = intermediate_tensors.remove(&root_name) {
            // Get root tensors from both networks
            let root_a_idx = tn_a
                .node_index(&root_name)
                .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found in tn_a", root_name))?;
            let root_b_idx = tn_b
                .node_index(&root_name)
                .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found in tn_b", root_name))?;

            let root_tensor_a = tn_a
                .tensor(root_a_idx)
                .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_a"))?
                .clone();
            let root_tensor_b = tn_b
                .tensor(root_b_idx)
                .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_b"))?
                .clone();

            // Contract [R_list..., A[root], B[root]]
            let mut tensor_list = r_list;
            tensor_list.push(root_tensor_a);
            tensor_list.push(root_tensor_b);
            let tensor_refs: Vec<&T> = tensor_list.iter().collect();
            let root_result =
                T::contract(&tensor_refs).context("Failed to contract root node tensors")?;

            // Store root result (no factorization needed)
            result_tensors.insert(root_name.clone(), root_result);
        } else {
            // No intermediate tensors: root is a single node or already processed
            // Check if root tensors need to be contracted
            if !result_tensors.contains_key(&root_name) {
                let root_a_idx = tn_a.node_index(&root_name).ok_or_else(|| {
                    anyhow::anyhow!("Root node {:?} not found in tn_a", root_name)
                })?;
                let root_b_idx = tn_b.node_index(&root_name).ok_or_else(|| {
                    anyhow::anyhow!("Root node {:?} not found in tn_b", root_name)
                })?;

                let root_tensor_a = tn_a
                    .tensor(root_a_idx)
                    .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_a"))?;
                let root_tensor_b = tn_b
                    .tensor(root_b_idx)
                    .ok_or_else(|| anyhow::anyhow!("Root tensor not found in tn_b"))?;

                let root_result = T::contract(&[root_tensor_a, root_tensor_b])
                    .context("Failed to contract root node tensors")?;

                result_tensors.insert(root_name.clone(), root_result);
            }
        }

        // 10. Build result TreeTN
        let mut result = TreeTN::new();

        // Add all result tensors
        for (node_name, tensor) in result_tensors {
            result.add_tensor(node_name, tensor)?;
        }

        // Connect nodes based on original topology
        for (source_name, destination_name) in &edges {
            if let (Some(node_a_idx), Some(node_b_idx)) = (
                result.node_index(source_name),
                result.node_index(destination_name),
            ) {
                let tensor_a = result.tensor(node_a_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_zipup_with: result tensor not found for node {:?}",
                        source_name
                    )
                })?;
                let tensor_b = result.tensor(node_b_idx).ok_or_else(|| {
                    anyhow::anyhow!(
                        "contract_zipup_with: result tensor not found for node {:?}",
                        destination_name
                    )
                })?;

                // Find the common index (should be the bond index)
                use tensor4all_core::index_ops::common_inds;
                let indices_a = tensor_a.external_indices();
                let indices_b = tensor_b.external_indices();
                let common = common_inds::<T::Index>(&indices_a, &indices_b);
                if let Some(bond_idx) = common.first() {
                    result.connect_internal(node_a_idx, bond_idx, node_b_idx, bond_idx)?;
                }
            }
        }

        // 11. Set canonical center
        if result.node_index(center).is_some() {
            result.set_canonical_region(std::iter::once(center.clone()))?;
        }

        Ok(result)
    }

    /// Contract two TreeTNs using naive full contraction.
    ///
    /// This is a reference implementation that:
    /// 1. Replaces internal indices with fresh IDs (sim_internal_inds)
    /// 2. Converts both TreeTNs to full tensors
    /// 3. Contracts along common site indices
    ///
    /// The result is a single tensor, not a TreeTN. This is useful for:
    /// - Testing correctness of more sophisticated algorithms like `contract_zipup`
    /// - Computing exact results for small networks
    ///
    /// # Arguments
    /// * `other` - The other TreeTN to contract with (must have same topology)
    ///
    /// # Returns
    /// A tensor representing the contracted result.
    ///
    /// # Note
    /// This method is O(exp(n)) in both time and memory where n is the number of nodes.
    /// Use `contract_zipup` for efficient contraction of large networks.
    /// # Errors
    ///
    /// Returns an error when the naive contraction fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn contract_naive(&self, other: &Self) -> std::result::Result<T, TreeTNOperationError>
    where
        V: Ord,
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        // 1. Verify topologies are compatible
        if !self.same_topology(other) {
            return Err(
                anyhow::anyhow!("contract_naive: networks have incompatible topologies").into(),
            );
        }

        // 2. Replace internal indices with fresh IDs to avoid collision
        let tn1 = self.sim_internal_inds();
        let tn2 = other.sim_internal_inds();

        // 3. Convert both networks to full tensors
        let tensor1 = tn1
            .contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn1: {}", e))?;
        let tensor2 = tn2
            .contract_to_tensor()
            .map_err(|e| anyhow::anyhow!("contract_naive: failed to contract tn2: {}", e))?;

        // 4. Exact pairwise product over common site indices. If the caller
        // explicitly asks for a reference product with no common site indices
        // (for example partial_contract with an empty spec), this is the
        // corresponding outer product reference.
        tensor1
            .contract_pair(&tensor2)
            .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))
    }

    /// Validate that `canonical_region` and edge `ortho_towards` are consistent.
    ///
    /// Rules:
    /// - If `canonical_region` is empty (not canonicalized), all indices must have `ortho_towards == None`.
    /// - If `canonical_region` is non-empty:
    ///
    ///   - It must form a connected subtree
    ///   - All edges from outside the center region must have `ortho_towards` pointing towards the center
    ///   - Edges entirely inside the center region may have `ortho_towards == None`
    /// # Errors
    ///
    /// Returns an error when the orthogonality structure is inconsistent (an
    /// /// graph consistency failure).
    ///
    pub fn validate_ortho_consistency(&self) -> std::result::Result<(), TreeTNOperationError> {
        // If not canonicalized, require no ortho_towards at all
        if self.canonical_region.is_empty() {
            if !self.ortho_towards.is_empty() {
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Found {} ortho_towards entries but canonical_region is empty",
                        self.ortho_towards.len()
                    )
                    .context(
                        "validate_ortho_consistency: canonical_region empty implies no ortho_towards",
                    ),
                ));
            }
            return Ok(());
        }

        // Validate all canonical_region nodes exist and convert to NodeIndex
        let mut center_indices = HashSet::new();
        for c in &self.canonical_region {
            let idx = self
                .graph
                .node_index(c)
                .ok_or_else(|| anyhow::anyhow!("canonical_region node {:?} does not exist", c))?;
            center_indices.insert(idx);
        }

        // Check canonical_region connectivity
        if !self.site_index_network.is_connected_subset(&center_indices) {
            return Err(TreeTNOperationError::from(
                anyhow::anyhow!("canonical_region is not connected").context(
                    "validate_ortho_consistency: canonical_region must form a connected subtree",
                ),
            ));
        }

        // Get expected edges from edges_to_canonicalize_to_region
        let expected_edges = self
            .site_index_network
            .edges_to_canonicalize_to_region(&center_indices);

        // Build a set of expected (bond, expected_direction) pairs
        let mut expected_directions: HashMap<T::Index, V> = HashMap::new();
        for (src, dst) in expected_edges.iter() {
            // Find the edge between src and dst
            let edge = self
                .graph
                .graph()
                .find_edge(*src, *dst)
                .or_else(|| self.graph.graph().find_edge(*dst, *src))
                .ok_or_else(|| anyhow::anyhow!("Edge not found between {:?} and {:?}", src, dst))?;

            let bond = self
                .bond_index(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found for edge"))?
                .clone();

            // The expected ortho_towards direction is dst (towards center)
            let dst_name = self
                .graph
                .node_name(*dst)
                .ok_or_else(|| anyhow::anyhow!("Node name not found for {:?}", dst))?
                .clone();

            expected_directions.insert(bond, dst_name);
        }

        // Verify all expected directions are present in ortho_towards
        for (bond, expected_dir) in &expected_directions {
            match self.ortho_towards.get(bond) {
                Some(actual_dir) => {
                    if actual_dir != expected_dir {
                        return Err(TreeTNOperationError::from(
                            anyhow::anyhow!(
                                "ortho_towards for bond {:?} points to {:?} but expected {:?}",
                                bond,
                                actual_dir,
                                expected_dir
                            )
                            .context("validate_ortho_consistency: wrong direction"),
                        ));
                    }
                }
                None => {
                    return Err(TreeTNOperationError::from(
                        anyhow::anyhow!(
                            "ortho_towards for bond {:?} is missing, expected to point to {:?}",
                            bond,
                            expected_dir
                        )
                        .context("validate_ortho_consistency: missing ortho_towards"),
                    ));
                }
            }
        }

        // Verify no unexpected bond ortho_towards entries
        // (site index ortho_towards are allowed even if not in expected_directions)
        let bond_indices: HashSet<T::Index> = self
            .graph
            .graph()
            .edge_indices()
            .filter_map(|e| self.bond_index(e))
            .cloned()
            .collect();

        for idx in self.ortho_towards.keys() {
            if bond_indices.contains(idx) && !expected_directions.contains_key(idx) {
                // This is a bond inside the canonical_region - should not have ortho_towards
                return Err(TreeTNOperationError::from(
                    anyhow::anyhow!(
                        "Unexpected ortho_towards for bond {:?} (inside canonical_region)",
                        idx
                    )
                    .context(
                        "validate_ortho_consistency: bonds inside center should not have ortho_towards",
                    ),
                ));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Find common indices between two tensors by exact index identity.
fn find_common_indices<T: TensorLike>(a: &T, b: &T) -> Vec<T::Index>
where
    T::Index: Eq + std::hash::Hash,
{
    let a_indices: HashSet<_> = a.external_indices().into_iter().collect();
    b.external_indices()
        .into_iter()
        .filter(|i| a_indices.contains(i))
        .collect()
}

// ============================================================================
// Contraction Method Dispatcher
// ============================================================================

use super::fit::FitContractionOptions;

/// Contraction method for TreeTN operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContractionMethod {
    /// Zip-up contraction (faster, one-pass).
    #[default]
    Zipup,
    /// Fit/variational contraction (iterative optimization).
    Fit,
    /// Generic dense/reference contraction: contract to a full tensor, then
    /// decompose back to TreeTN.
    ///
    /// Useful for small debugging cases, but O(exp(n)) in memory. This variant
    /// is not the algorithm used by `apply_linear_operator(...,
    /// ApplyOptions::naive())`, which has a dedicated local exact apply path.
    Naive,
}

/// Options for the generic contract function.
#[derive(Debug, Clone)]
pub struct ContractionOptions {
    /// Contraction method to use.
    pub method: ContractionMethod,
    /// Maximum bond dimension (optional).
    pub max_bond_dim: Option<usize>,
    /// Explicit SVD truncation policy (optional).
    pub svd_policy: Option<SvdTruncationPolicy>,
    /// QR-specific relative tolerance (optional).
    pub qr_rtol: Option<f64>,
    /// Number of full sweeps for Fit method.
    ///
    /// A full sweep visits each edge twice (forward and backward) using an Euler tour.
    pub nfullsweeps: usize,
    /// Convergence tolerance for Fit method (None = fixed sweeps).
    pub convergence_tol: Option<f64>,
    /// Factorization algorithm for Fit method.
    pub factorize_alg: FactorizeAlg,
    /// Maximum dense elements allowed for generic Naive dense/reference
    /// contraction.
    ///
    /// `None` rejects the generic dense/reference path. Set this only for
    /// small debugging or reference cases where full dense materialization of
    /// both inputs and the contracted result is expected and bounded.
    pub dense_reference_limit: Option<usize>,
    /// Maximum dense elements allowed for explicit mismatched-topology
    /// reference fallback in `partial_contract`.
    ///
    /// `None` rejects the fallback; compatible tree-union mismatches are first
    /// handled by structural dimension-1 topology alignment without dense
    /// materialization. Set this only for small reference/debug cases where
    /// full dense materialization is acceptable if structural alignment fails.
    pub mismatched_topology_dense_limit: Option<usize>,
}

impl Default for ContractionOptions {
    fn default() -> Self {
        Self {
            method: ContractionMethod::default(),
            max_bond_dim: None,
            svd_policy: None,
            qr_rtol: None,
            nfullsweeps: 1,
            convergence_tol: None,
            factorize_alg: FactorizeAlg::default(),
            dense_reference_limit: None,
            mismatched_topology_dense_limit: None,
        }
    }
}

impl ContractionOptions {
    /// Create options with specified method.
    pub fn new(method: ContractionMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Create options for zipup contraction.
    pub fn zipup() -> Self {
        Self::new(ContractionMethod::Zipup)
    }

    /// Create options for fit contraction.
    pub fn fit() -> Self {
        Self::new(ContractionMethod::Fit)
    }

    /// Set maximum bond dimension.
    pub fn with_max_bond_dim(mut self, max_bond_dim: usize) -> Self {
        self.max_bond_dim = Some(max_bond_dim);
        self
    }

    /// Set the SVD truncation policy.
    pub fn with_svd_policy(mut self, policy: SvdTruncationPolicy) -> Self {
        self.svd_policy = Some(policy);
        self
    }

    /// Set the QR-specific relative tolerance.
    pub fn with_qr_rtol(mut self, rtol: f64) -> Self {
        self.qr_rtol = Some(rtol);
        self
    }

    /// Set number of full sweeps for Fit method.
    pub fn with_nfullsweeps(mut self, nfullsweeps: usize) -> Self {
        self.nfullsweeps = nfullsweeps;
        self
    }

    /// Set convergence tolerance for Fit method.
    pub fn with_convergence_tol(mut self, tol: f64) -> Self {
        self.convergence_tol = Some(tol);
        self
    }

    /// Set factorization algorithm for Fit method.
    pub fn with_factorize_alg(mut self, alg: FactorizeAlg) -> Self {
        self.factorize_alg = alg;
        self
    }

    /// Allow generic Naive dense/reference contraction up to `max_elements`.
    ///
    /// # Arguments
    /// * `max_elements` - Maximum number of elements allowed in each full dense
    ///
    ///   input tensor and in the dense contracted result. Typical values should
    ///   remain small and test-sized.
    ///
    /// # Returns
    /// Updated options with the dense/reference contraction limit enabled.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::contraction::ContractionOptions;
    ///
    /// let options = ContractionOptions::default().with_dense_reference_limit(1024);
    /// assert_eq!(options.dense_reference_limit, Some(1024));
    /// ```
    pub fn with_dense_reference_limit(mut self, max_elements: usize) -> Self {
        self.dense_reference_limit = Some(max_elements);
        self
    }

    /// Allow `partial_contract` to use its mismatched-topology dense reference
    /// fallback up to `max_elements` elements if structural alignment fails.
    ///
    /// # Arguments
    /// * `max_elements` - Maximum number of elements allowed in each dense
    ///
    ///   intermediate and in the dense contracted result. Typical values should
    ///   remain small and test-sized.
    ///
    /// # Returns
    /// Updated options with the dense/reference fallback limit enabled. This
    /// does not force dense materialization; compatible tree-union topology
    /// mismatches are handled by structural dimension-1 alignment first.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::contraction::ContractionOptions;
    ///
    /// let options = ContractionOptions::default()
    ///     .with_mismatched_topology_dense_limit(1024);
    /// assert_eq!(options.mismatched_topology_dense_limit, Some(1024));
    /// ```
    pub fn with_mismatched_topology_dense_limit(mut self, max_elements: usize) -> Self {
        self.mismatched_topology_dense_limit = Some(max_elements);
        self
    }
}

fn dense_element_count<I: IndexLike>(indices: &[I], context: &str) -> Result<usize> {
    indices.iter().try_fold(1usize, |acc, index| {
        acc.checked_mul(index.dim()).ok_or_else(|| {
            anyhow::anyhow!(
                "{context}: dense/reference element count overflowed while checking limit"
            )
        })
    })
}

fn dense_contract_output_indices<I: IndexLike>(a_indices: &[I], b_indices: &[I]) -> Vec<I> {
    let mut b_remaining = b_indices.to_vec();
    let mut output = Vec::new();

    for idx_a in a_indices {
        if let Some(position) = b_remaining
            .iter()
            .position(|idx_b| idx_a.is_contractable(idx_b))
        {
            b_remaining.remove(position);
        } else {
            output.push(idx_a.clone());
        }
    }

    output.extend(b_remaining);
    output
}

fn ensure_dense_reference_limit<I: IndexLike>(
    context: &str,
    label: &str,
    indices: &[I],
    max_elements: usize,
) -> Result<()> {
    let elements = dense_element_count(indices, context)?;
    if elements > max_elements {
        anyhow::bail!(
            "{context}: Naive dense/reference contraction would materialize {label} with {elements} elements, exceeding limit {max_elements}"
        );
    }
    Ok(())
}

fn validate_naive_dense_reference_limit<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    options: &ContractionOptions,
) -> Result<()>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    let Some(max_elements) = options.dense_reference_limit else {
        anyhow::bail!(
            "contract: Naive dense/reference contraction requires an explicit dense/reference limit; call ContractionOptions::with_dense_reference_limit(max_elements) for small reference cases"
        );
    };

    let a_external = tn_a.external_indices();
    let b_external = tn_b.external_indices();
    let output = dense_contract_output_indices(&a_external, &b_external);

    ensure_dense_reference_limit("contract", "first TreeTN", &a_external, max_elements)?;
    ensure_dense_reference_limit("contract", "second TreeTN", &b_external, max_elements)?;
    ensure_dense_reference_limit("contract", "contracted result", &output, max_elements)
}

/// Contract two TreeTNs using the specified method.
///
/// This is the main entry point for TreeTN contraction. It dispatches to the
/// appropriate algorithm based on the options.
/// # Errors
///
/// Returns an error when the contraction fails (a shape or index mismatch,
/// /// or a backend failure).
///
pub fn contract<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    options: ContractionOptions,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    match options.method {
        ContractionMethod::Zipup => {
            tn_a.contract_zipup(tn_b, center, options.svd_policy, options.max_bond_dim)
        }
        ContractionMethod::Fit => {
            let fit_options = FitContractionOptions::new(options.nfullsweeps)
                .with_factorize_alg(options.factorize_alg);
            let fit_options = if let Some(max_bond_dim) = options.max_bond_dim {
                fit_options.with_max_bond_dim(max_bond_dim)
            } else {
                fit_options
            };
            let fit_options = if let Some(policy) = options.svd_policy {
                fit_options.with_svd_policy(policy)
            } else {
                fit_options
            };
            let fit_options = if let Some(qr_rtol) = options.qr_rtol {
                fit_options.with_qr_rtol(qr_rtol)
            } else {
                fit_options
            };
            let fit_options = if let Some(tol) = options.convergence_tol {
                fit_options.with_convergence_tol(tol)
            } else {
                fit_options
            };
            super::fit::contract_fit(tn_a, tn_b, center, fit_options)
        }
        ContractionMethod::Naive => {
            validate_naive_dense_reference_limit(tn_a, tn_b, &options)?;
            contract_naive_to_treetn(
                tn_a,
                tn_b,
                center,
                options.max_bond_dim,
                options.svd_policy,
                options.qr_rtol,
            )
        }
    }
}

/// Contract two TreeTNs using naive contraction, then decompose back to TreeTN.
///
/// This method:
/// 1. Contracts both networks to full tensors
/// 2. Contracts the tensors along common (site) indices
/// 3. Decomposes the result back to a TreeTN using the original topology
///
/// This is O(exp(n)) in memory and is primarily useful for debugging and testing.
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns an error when the contraction fails (a shape or index mismatch,
/// /// or a backend failure).
///
pub fn contract_naive_to_treetn<T, V>(
    tn_a: &TreeTN<T, V>,
    tn_b: &TreeTN<T, V>,
    center: &V,
    _max_bond_dim: Option<usize>,
    _svd_policy: Option<SvdTruncationPolicy>,
    _qr_rtol: Option<f64>,
) -> std::result::Result<TreeTN<T, V>, TreeTNOperationError>
where
    T: TensorLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    // 1. Contract to full tensor using existing contract_naive
    let contracted_tensor = tn_a.contract_naive(tn_b)?;

    // Handle rank-0 (scalar) result: wrap directly in a single-node TreeTN
    if contracted_tensor.external_indices().is_empty() {
        let mut tn = TreeTN::<T, V>::new();
        tn.add_tensor(center.clone(), contracted_tensor)?;
        tn.set_canonical_region([center.clone()])?;
        return Ok(tn);
    }

    // 2. Build topology from tn_a's structure and decompose
    use super::decompose::factorize_tensor_to_treetn_with;

    // Build topology using full indices (not positions).
    // Consider site indices from BOTH tn_a and tn_b, since the contracted result
    // may contain indices from either network (non-contracted ones remain).
    let mut nodes: HashMap<V, Vec<T::Index>> = HashMap::new();
    let contracted_indices = contracted_tensor.external_indices();
    let contracted_index_set: HashSet<_> = contracted_indices.iter().cloned().collect();

    // Collect node names in sorted order for deterministic assignment
    let mut node_names: Vec<_> = tn_a.node_names();
    node_names.sort();

    for node_name in &node_names {
        let mut indices: Vec<T::Index> = Vec::new();

        // Collect remaining site indices from tn_a at this node
        if let Some(site_space_a) = tn_a.site_index_network.site_space(node_name) {
            for site_idx in site_space_a {
                if contracted_index_set.contains(site_idx) {
                    indices.push(site_idx.clone());
                }
            }
        }

        // Also collect remaining site indices from tn_b at the same node
        if let Some(site_space_b) = tn_b.site_index_network.site_space(node_name) {
            for site_idx in site_space_b {
                if contracted_index_set.contains(site_idx) && !indices.contains(site_idx) {
                    indices.push(site_idx.clone());
                }
            }
        }

        nodes.insert(node_name.clone(), indices);
    }

    // Get edges from the graph
    let edges: Vec<(V, V)> = tn_a
        .graph
        .graph()
        .edge_indices()
        .filter_map(|e| {
            let (src, dst) = tn_a.graph.graph().edge_endpoints(e)?;
            let src_name = tn_a.graph.node_name(src)?;
            let dst_name = tn_a.graph.node_name(dst)?;
            Some((src_name.clone(), dst_name.clone()))
        })
        .collect();

    let topology = super::decompose::TreeTopology::new(nodes, edges);

    // 3. Decompose back to TreeTN
    let result = factorize_tensor_to_treetn_with(
        &contracted_tensor,
        &topology,
        FactorizeOptions::svd(),
        center,
    )?;

    Ok(result)
}

#[cfg(test)]
mod tests;
