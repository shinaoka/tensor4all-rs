//! Trait implementations and operations for TreeTN.
//!
//! This module provides:
//! - `Default` implementation
//! - `Clone` implementation
//! - `Debug` implementation
//! - `log_norm` for computing the logarithm of the Frobenius norm
//! - `norm`, `norm_squared` for computing the Frobenius norm
//! - `inner` for computing inner products of two TreeTNs
//! - `to_dense` for contracting to a single tensor
//! - `evaluate` for evaluating at specific index values using full indices
//! - `all_site_indices` for retrieving all site indices and their owning vertices

use std::collections::HashMap;
use std::hash::Hash;

use tensor4all_core::{AnyScalar, ColMajorArrayRef, IndexLike, TensorLike};

use super::{TreeTN, TreeTNEvaluator};

// ============================================================================
// Default implementation
// ============================================================================

impl<T, V> Default for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Clone implementation
// ============================================================================

impl<T, V> Clone for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph.clone(),
            canonical_region: self.canonical_region.clone(),
            canonical_form: self.canonical_form,
            site_index_network: self.site_index_network.clone(),
            link_index_network: self.link_index_network.clone(),
            ortho_towards: self.ortho_towards.clone(),
        }
    }
}

// ============================================================================
// Debug implementation
// ============================================================================

impl<T, V> std::fmt::Debug for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TreeTN")
            .field("node_count", &self.node_count())
            .field("edge_count", &self.edge_count())
            .field("canonical_region", &self.canonical_region)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Norm Computation
// ============================================================================

use anyhow::{Context, Result};

use crate::algorithm::CanonicalForm;
use crate::CanonicalizationOptions;

impl<T, V> TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Compute log(||TreeTN||_F), the log of the Frobenius norm.
    ///
    /// Uses canonicalization to avoid numerical overflow:
    /// when canonicalized to a single site with Unitary form,
    /// the Frobenius norm of the whole network equals the norm of the center tensor.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network
    /// to a single Unitary center. Use `log_norm` (without canonicalization) if you
    /// already have a properly canonicalized network.
    ///
    /// # Returns
    /// The natural logarithm of the Frobenius norm.
    ///
    /// # Errors
    /// Returns an error when the tensor train contains a non-finite value or its
    /// scaling is invalid (an invalid-state or dtype mismatch failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s], vec![3.0_f64, 4.0]).unwrap();
    /// let mut tn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    ///
    /// // log(||[3, 4]||) = log(5)
    /// let ln = tn.log_norm().unwrap();
    /// assert!((ln - 5.0_f64.ln()).abs() < 1e-10);
    /// ```
    pub fn log_norm(&mut self) -> Result<f64> {
        let n = self.node_count();
        if n == 0 {
            return Err(anyhow::anyhow!("Cannot compute log_norm of empty TreeTN"))
                .context("log_norm: network must have at least one node");
        }

        // Determine the single center site (by name)
        let center_name: V =
            if self.is_canonicalized() && self.canonical_form() == Some(CanonicalForm::Unitary) {
                if self.canonical_region.len() == 1 {
                    // Already Unitary canonicalized to single site - use it
                    self.canonical_region
                        .iter()
                        .next()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("Canonical region is empty"))
                        .context("log_norm: canonical region must contain one center")?
                } else {
                    // Unitary canonicalized to multiple sites - canonicalize to min site
                    let min_center = self
                        .canonical_region
                        .iter()
                        .min()
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("Canonical region is empty"))
                        .context("log_norm: canonical region must contain centers")?;
                    self.canonicalize_mut(
                        std::iter::once(min_center.clone()),
                        CanonicalizationOptions::default(),
                    )
                    .context("log_norm: failed to canonicalize to single site")?;
                    min_center
                }
            } else {
                // Not canonicalized or not Unitary - canonicalize to min node name
                let min_node_name = self
                    .node_names()
                    .into_iter()
                    .min()
                    .ok_or_else(|| anyhow::anyhow!("No nodes in TreeTN"))
                    .context("log_norm: network must have nodes")?;
                self.canonicalize_mut(
                    std::iter::once(min_node_name.clone()),
                    CanonicalizationOptions::default(),
                )
                .context("log_norm: failed to canonicalize")?;
                min_node_name
            };

        // Get center node index and tensor
        let center_node = self
            .node_index(&center_name)
            .ok_or_else(|| anyhow::anyhow!("Center node not found"))
            .context("log_norm: center node must exist")?;

        let center_tensor = self
            .tensor(center_node)
            .ok_or_else(|| anyhow::anyhow!("Center tensor not found"))
            .context("log_norm: center tensor must exist")?;

        let norm_sq = center_tensor.norm_squared()?;
        let norm = norm_sq.sqrt();

        Ok(norm.ln())
    }

    /// Compute the Frobenius norm of the TreeTN.
    ///
    /// Uses `log_norm` internally: `norm = exp(log_norm)`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error when the norm cannot be evaluated (a materialization or
    /// backend failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// // Single-node TreeTN with tensor [1, 0, 0, 1] (identity 2x2)
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(
    ///     vec![s0.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    ///
    /// let mut tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t],
    ///     vec!["A".to_string()],
    /// ).unwrap();
    ///
    /// // Frobenius norm of [[1,0],[0,1]] = sqrt(2)
    /// let n = tn.norm().unwrap();
    /// assert!((n - 2.0_f64.sqrt()).abs() < 1e-10);
    /// ```
    pub fn norm(&mut self) -> Result<f64> {
        let log_n = self
            .log_norm()
            .context("norm: failed to compute log_norm")?;
        Ok(log_n.exp())
    }

    /// Compute the squared Frobenius norm of the TreeTN.
    ///
    /// Returns `||self||^2 = norm()^2`.
    ///
    /// # Note
    /// This method is mutable because it may need to canonicalize the network.
    ///
    /// # Errors
    /// Returns an error when the norm cannot be evaluated (a materialization or
    /// backend failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s], vec![3.0_f64, 4.0]).unwrap();
    /// let mut tn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    ///
    /// // ||[3, 4]||^2 = 9 + 16 = 25
    /// let nsq = tn.norm_squared().unwrap();
    /// assert!((nsq - 25.0).abs() < 1e-10);
    /// ```
    pub fn norm_squared(&mut self) -> Result<f64> {
        let n = self
            .norm()
            .context("norm_squared: failed to compute norm")?;
        Ok(n * n)
    }

    /// Scale the tensor network by a complex scalar.
    ///
    /// This multiplies a single node tensor, chosen deterministically as the
    /// minimum-named node, so the represented state is scaled once rather than
    /// applying `scalar^n` across all nodes.
    ///
    /// Scaling a non-center tensor generally invalidates any existing
    /// canonicalization metadata, so this method clears the cached canonical
    /// region and orthogonality directions after updating the tensor.
    ///
    /// # Arguments
    /// * `scalar` - Scalar multiplier applied to the represented tensor network
    ///
    /// # Returns
    /// `Ok(())` after the selected node tensor has been updated in place
    ///
    /// # Errors
    /// Returns an error when the scaling fails (a dtype mismatch or backend
    /// failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen, TensorIndex, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s], vec![1.0_f64, -2.0]).unwrap();
    /// let mut tn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    ///
    /// tn.scale(AnyScalar::new_real(2.0)).unwrap();
    ///
    /// let dense = tn.to_dense().unwrap();
    /// let expected = TensorDynLen::from_dense(
    ///     dense.external_indices(),
    ///     vec![2.0_f64, -4.0],
    /// ).unwrap();
    /// assert!(dense.distance(&expected).unwrap() < 1e-12);
    /// ```
    pub fn scale(&mut self, scalar: AnyScalar) -> Result<()> {
        let min_node = self
            .node_names()
            .into_iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("Cannot scale empty TreeTN"))
            .context("scale: network must have at least one node")?;
        let node_idx = self
            .node_index(&min_node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", min_node))
            .context("scale: selected node must exist")?;
        let tensor = self
            .tensor(node_idx)
            .ok_or_else(|| anyhow::anyhow!("Node tensor not found for {:?}", min_node))
            .context("scale: selected node tensor must exist")?
            .clone();
        let scaled = tensor
            .scale(scalar)
            .context("scale: tensor scaling failed")?;
        self.replace_tensor(node_idx, scaled)?
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", min_node))
            .context("scale: failed to replace scaled tensor")?;

        self.clear_canonical_region();
        self.ortho_towards.clear();

        Ok(())
    }

    /// Compute the inner product of two TreeTNs.
    ///
    /// Computes `<self | other>` = sum over all indices of `conj(self) * other`.
    ///
    /// Both TreeTNs must have the same site indices (same IDs).
    /// Link indices may differ between the two TreeTNs.
    ///
    /// # Algorithm
    /// 1. Replace link indices in `other` with fresh IDs to avoid collision.
    /// 2. At each node, contract `conj(self_tensor) * other_tensor` pairwise.
    /// 3. Sweep from leaves to root, contracting the environment.
    ///
    /// This is equivalent to contracting the entire network
    /// `conj(self) * other` into a scalar.
    ///
    /// # Errors
    /// Returns an error when the two tensor trains have incompatible site spaces
    /// (a shape mismatch) or the contraction fails (a backend failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let t = TensorDynLen::from_dense(vec![s], vec![3.0_f64, 4.0]).unwrap();
    /// let tn = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
    ///
    /// // <v|v> = 3^2 + 4^2 = 25
    /// let ip = tn.inner(&tn).unwrap();
    /// assert!((ip.real() - 25.0).abs() < 1e-10);
    /// ```
    pub fn inner(&self, other: &Self) -> Result<AnyScalar>
    where
        <T::Index as IndexLike>::Id:
            Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    {
        if self.node_count() == 0 && other.node_count() == 0 {
            return Ok(AnyScalar::new_real(0.0));
        }
        if !self.share_equivalent_site_index_network(other) {
            return Err(anyhow::anyhow!(
                "inner: TreeTNs must have the same topology and site indices"
            ));
        }

        let root_name = self
            .node_names()
            .into_iter()
            .min()
            .ok_or_else(|| anyhow::anyhow!("Cannot compute inner product of empty TreeTN"))
            .context("inner: network must have at least one node")?;
        let other_sim = other.sim_internal_inds();

        let post_order = self
            .site_index_network()
            .post_order_dfs(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root node {:?} not found", root_name))
            .context("inner: failed to build post-order traversal")?;

        let mut parent_of: HashMap<V, Option<V>> = HashMap::new();
        parent_of.insert(root_name.clone(), None);
        let mut stack = vec![root_name.clone()];
        while let Some(node_name) = stack.pop() {
            let mut neighbors: Vec<V> = self.site_index_network().neighbors(&node_name).collect();
            neighbors.sort();
            for neighbor in neighbors {
                if parent_of.contains_key(&neighbor) {
                    continue;
                }
                parent_of.insert(neighbor.clone(), Some(node_name.clone()));
                stack.push(neighbor);
            }
        }

        let mut envs: HashMap<V, T> = HashMap::new();

        for node_name in post_order {
            let node_idx_self = self
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in self", node_name))
                .context("inner: self node must exist")?;
            let node_idx_other = other_sim
                .node_index(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in other", node_name))
                .context("inner: other node must exist")?;

            let mut env = self
                .tensor(node_idx_self)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("inner: self tensor must exist")?
                .conj();

            let mut children: Vec<V> = parent_of
                .iter()
                .filter_map(|(child, parent)| {
                    if parent.as_ref() == Some(&node_name) {
                        Some(child.clone())
                    } else {
                        None
                    }
                })
                .collect();
            children.sort();

            for child_name in children {
                let child_env = envs.remove(&child_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Missing child environment for child {:?} of node {:?}",
                        child_name,
                        node_name
                    )
                })?;
                env = T::contract(&[&env, &child_env])
                    .context("inner: failed to absorb child environment")?;
            }

            let other_tensor = other_sim
                .tensor(node_idx_other)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))
                .context("inner: other tensor must exist")?;
            env = T::contract(&[&env, other_tensor])
                .context("inner: failed to contract node bra-ket tensors")?;

            envs.insert(node_name, env);
        }

        let result_tensor = envs
            .remove(&root_name)
            .ok_or_else(|| anyhow::anyhow!("Root environment was not produced"))
            .context("inner: root contraction failed")?;
        if !envs.is_empty() {
            return Err(anyhow::anyhow!(
                "inner: contraction left {} dangling environments",
                envs.len()
            ));
        }

        let scalar_one = T::scalar_one().context("inner: failed to create scalar_one")?;
        scalar_one
            .inner_product(&result_tensor)
            .context("inner: failed to extract scalar value")
    }

    /// Convert the TreeTN to a single dense tensor.
    ///
    /// This contracts all tensors in the network along their link/bond indices,
    /// producing a single tensor with only site (physical) indices.
    ///
    /// This is an alias for `contract_to_tensor()`.
    ///
    /// # Warning
    /// This operation can be very expensive for large networks,
    /// as the result size grows exponentially with the number of sites.
    ///
    /// # Errors
    /// Returns an error when the dense materialization fails (a materialization or
    /// backend failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treetn::TreeTN;
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorIndex, TensorLike};
    ///
    /// // Build a 2-node chain
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    ///
    /// // Identity matrices
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond.clone(), s1.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    ///
    /// let tn = TreeTN::<_, String>::from_tensors(
    ///     vec![t0, t1],
    ///     vec!["A".to_string(), "B".to_string()],
    /// ).unwrap();
    ///
    /// // Contract to a single dense tensor over site indices s0 and s1
    /// let dense = tn.to_dense().unwrap();
    /// // Result is rank-2 (two site indices s0 and s1)
    /// assert_eq!(dense.num_external_indices(), 2);
    /// ```
    pub fn to_dense(&self) -> Result<T> {
        self.contract_to_tensor()
            .context("to_dense: failed to contract network to tensor")
    }

    /// Create a reusable evaluator for repeated batched point evaluation.
    ///
    /// This is a convenience wrapper around [`TreeTNEvaluator::new`]. Use the
    /// returned evaluator when repeatedly evaluating the same TreeTN with the
    /// same site-index order.
    ///
    /// # Arguments
    ///
    /// * `indices` - Complete site-index order used as the row order of future
    ///
    ///   batch value arrays.
    ///
    /// # Returns
    ///
    /// A reusable evaluator bound to this TreeTN.
    ///
    /// # Errors
    /// Returns an error when the evaluator cannot be constructed (an
    /// invalid-configuration failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s = DynIndex::new_dyn(2);
    /// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![1.0, 2.0])?;
    /// let tree = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0])?;
    ///
    /// let evaluator = tree.evaluator(&[s])?;
    /// assert_eq!(evaluator.input_count(), 1);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluator(&self, indices: &[T::Index]) -> Result<TreeTNEvaluator<T, V>>
    where
        T::Index: Clone + Hash + Eq,
        <T::Index as IndexLike>::Id: Ord,
    {
        TreeTNEvaluator::new(self, indices)
    }

    /// Evaluate the TreeTN at multiple multi-indices (batch).
    ///
    /// This is a convenience method that constructs a temporary
    /// [`TreeTNEvaluator`]. For repeated calls with the same site-index order,
    /// create an evaluator once with [`Self::evaluator`] and call
    /// [`TreeTNEvaluator::evaluate_batch`].
    ///
    /// # Arguments
    /// * `indices` - Identifies each site index by full `Index` value (from
    ///
    ///   [`all_site_indices()`](Self::all_site_indices)).
    ///   Must enumerate every site index exactly once.
    /// * `values` - Column-major array of shape `[n_indices, n_points]`.
    ///
    ///   `values.get(&[i, p])` is the value of `indices[i]` at point `p`.
    ///
    /// # Returns
    /// A `Vec<AnyScalar>` of length `n_points`.
    ///
    /// # Errors
    /// Returns an error if:
    /// - The network is empty
    /// - `values` shape is inconsistent with `indices`
    /// - An index is unknown
    /// - Index values are out of bounds
    /// - Contraction fails
    pub fn evaluate(
        &self,
        indices: &[T::Index],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>>
    where
        T::Index: Clone + Hash + Eq,
        <T::Index as IndexLike>::Id: Ord,
    {
        TreeTNEvaluator::new(self, indices)?.evaluate_batch(values)
    }

    /// Evaluate the TreeTN at one multi-index.
    ///
    /// This is a convenience wrapper around [`Self::evaluate`] for the common
    /// single-point case.
    ///
    /// # Arguments
    ///
    /// * `indices` - Identifies each site index by full `Index` value. It must
    ///
    ///   enumerate every site index exactly once.
    /// * `values` - One site value for each entry in `indices`.
    ///
    /// # Returns
    ///
    /// The scalar value of this TreeTN at the requested multi-index.
    ///
    /// # Errors
    ///
    /// Returns an error if `values.len() != indices.len()`, if the index list is
    /// invalid, if any value is out of bounds, or if contraction fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s = DynIndex::new_dyn(3);
    /// let tensor = TensorDynLen::from_dense(vec![s.clone()], vec![10.0, 20.0, 30.0])?;
    /// let tree = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0])?;
    ///
    /// let value = tree.evaluate_point(&[s], &[2])?;
    /// assert!((value.real() - 30.0).abs() < 1e-12);
    /// assert!(value.imag().abs() < 1e-12);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn evaluate_point(&self, indices: &[T::Index], values: &[usize]) -> Result<AnyScalar>
    where
        T::Index: Clone + Hash + Eq,
        <T::Index as IndexLike>::Id: Ord,
    {
        anyhow::ensure!(
            values.len() == indices.len(),
            "TreeTN::evaluate_point: values.len() ({}) != indices.len() ({})",
            values.len(),
            indices.len()
        );

        let shape = [indices.len(), 1usize];
        let values = ColMajorArrayRef::new(values, &shape)
            .context("TreeTN::evaluate_point: values must contain one point")?;
        let result = self.evaluate(indices, values)?;

        result
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("TreeTN::evaluate_point: evaluation returned no values"))
    }

    /// Return the dimensions of all internal bond links.
    ///
    /// The dimensions are ordered by sorted node-name edge pairs, giving a stable
    /// profile for reports and examples. Site dimensions are not included.
    ///
    /// # Returns
    ///
    /// A vector containing one dimension for each TreeTN edge.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let s1 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let left = TensorDynLen::from_dense(
    ///     vec![s0, bond.clone()],
    ///     vec![1.0_f64; 2 * 3],
    /// )?;
    /// let right = TensorDynLen::from_dense(
    ///     vec![bond, s1],
    ///     vec![1.0_f64; 3 * 2],
    /// )?;
    /// let tree = TreeTN::<TensorDynLen, usize>::from_tensors(vec![left, right], vec![0, 1])?;
    ///
    /// assert_eq!(tree.link_dims(), vec![3]);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn link_dims(&self) -> Vec<usize> {
        let mut edges = self.site_index_network.edges().collect::<Vec<_>>();
        for (left, right) in &mut edges {
            if right < left {
                std::mem::swap(left, right);
            }
        }
        edges.sort();
        edges.dedup();

        edges
            .into_iter()
            .filter_map(|(left, right)| {
                let edge = self.edge_between(&left, &right)?;
                self.bond_index(edge).map(IndexLike::dim)
            })
            .collect()
    }

    /// Returns all site indices and their owning vertex names.
    ///
    /// Returns `(indices, vertex_names)` where `indices[i]` belongs to
    /// vertex `vertex_names[i]`. Order is unspecified but consistent
    /// between the two vectors.
    ///
    /// # Errors
    /// Returns an error if a node's site space cannot be found (a missing-index
    /// failure).
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(3);
    /// let s1 = DynIndex::new_dyn(2);
    /// let t0 = TensorDynLen::from_dense(
    ///     vec![s0.clone(), bond.clone()], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    /// ).unwrap();
    /// let t1 = TensorDynLen::from_dense(
    ///     vec![bond.clone(), s1.clone()], vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    /// ).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0, t1], vec![0, 1]).unwrap();
    ///
    /// let (indices, vertices) = tn.all_site_indices().unwrap();
    /// assert_eq!(indices.len(), 2);
    /// assert_eq!(vertices.len(), 2);
    ///
    /// // The returned indices contain both s0 and s1
    /// assert!(indices.contains(&s0));
    /// assert!(indices.contains(&s1));
    /// ```
    #[allow(clippy::type_complexity)]
    /// # Errors
    /// Returns an error when a node's site space cannot be found (a
    /// missing-index failure).
    ///
    pub fn all_site_indices(&self) -> Result<(Vec<T::Index>, Vec<V>)>
    where
        V: Clone,
        T::Index: Clone,
    {
        let mut indices = Vec::new();
        let mut node_names = Vec::new();
        for node_name in self.node_names() {
            let site_space = self
                .site_space(&node_name)
                .ok_or_else(|| anyhow::anyhow!("Site space not found for node {:?}", node_name))
                .context("all_site_indices: site space must exist")?;
            for index in site_space {
                indices.push(index.clone());
                node_names.push(node_name.clone());
            }
        }
        Ok((indices, node_names))
    }

    /// Evaluate the TreeTN at multiple multi-indices (batch).
    ///
    /// Alias for [`evaluate()`](Self::evaluate).
    ///
    /// # Arguments
    /// * `indices` - Identifies each site index by its `Index` object
    ///
    ///   (e.g. from [`all_site_indices()`](Self::all_site_indices)).
    ///   Must enumerate every site index exactly once.
    /// * `values` - Column-major array of shape `[n_indices, n_points]`.
    ///
    ///   `values.get(&[i, p])` is the value of `indices[i]` at point `p`.
    ///
    /// # Returns
    /// A `Vec<AnyScalar>` of length `n_points`.
    ///
    /// # Errors
    /// Returns an error when the coordinate is out of range (an out of bounds
    /// failure) or evaluation fails (a backend failure).
    /// # Examples
    /// ```
    /// use tensor4all_core::{ColMajorArrayRef, DynIndex, IndexLike, TensorDynLen, TensorLike};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let s0 = DynIndex::new_dyn(3);
    /// let t0 = TensorDynLen::from_dense(vec![s0.clone()], vec![10.0, 20.0, 30.0]).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![t0], vec![0]).unwrap();
    ///
    /// let (indices, _vertices) = tn.all_site_indices().unwrap();
    ///
    /// // Evaluate at index value 2
    /// let data = [2usize];
    /// let shape = [indices.len(), 1];
    /// let values = ColMajorArrayRef::new(&data, &shape).unwrap();
    /// let result = tn.evaluate_at(&indices, values).unwrap();
    /// assert!((result[0].real() - 30.0).abs() < 1e-10);
    /// ```
    pub fn evaluate_at(
        &self,
        indices: &[T::Index],
        values: ColMajorArrayRef<'_, usize>,
    ) -> Result<Vec<AnyScalar>>
    where
        T::Index: Clone + Hash + Eq,
        <T::Index as IndexLike>::Id: Ord,
    {
        self.evaluate(indices, values)
    }
}

#[cfg(test)]
mod tests;
