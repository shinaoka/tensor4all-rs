//! LinearOperator: Wrapper for MPO with index mapping.
//!
//! This module provides a `LinearOperator` that wraps an MPO (Matrix Product Operator)
//! and handles the index ID mapping between true site indices and internal MPO indices.
//!
//! # Problem
//!
//! In the equation `A * x = b`:
//! - `A.s_in` should match `x`'s site indices
//! - `A.s_out` should match `b`'s site indices
//!
//! However, a tensor cannot have two indices with the same ID. So the MPO internally
//! uses `s_in_tmp` and `s_out_tmp` with independent IDs.
//!
//! # Solution
//!
//! `LinearOperator` stores:
//! - The MPO with internal index IDs (`s_in_tmp`, `s_out_tmp`)
//! - Mapping from true `s_in`/`s_out` to internal `s_in_tmp`/`s_out_tmp`
//!
//! When applying to `x`, it automatically handles the index transformations.

use crate::error::TreeTNOperationError;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use anyhow::Result;

use tensor4all_core::DynIndex;
use tensor4all_core::IdxTensor;
use tensor4all_core::IndexLike;
use tensor4all_core::LinearizationOrder;
use tensor4all_core::TensorLike;

use super::index_mapping::IndexMapping;
use crate::options::RestructureOptions;
use crate::site_index_network::SiteIndexNetwork;
use crate::treetn::TreeTN;

/// LinearOperator: Wraps an MPO with index mapping for automatic transformations.
///
/// # Type Parameters
///
/// * `T` - Tensor type implementing `TensorLike`
/// * `V` - Node name type
///
/// # Examples
///
/// A `LinearOperator` is typically obtained from a constructor function rather
/// than built directly. The [`mpo()`](Self::mpo) method exposes the underlying
/// TreeTN:
///
/// ```
/// use tensor4all_treetn::LinearOperator;
/// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
/// use tensor4all_treetn::TreeTN;
/// use std::collections::HashMap;
///
/// // Build a trivial single-node LinearOperator wrapping a 2x2 identity
/// let s_in = DynIndex::new_dyn(2);
/// let s_out = DynIndex::new_dyn(2);
/// let t = IdxTensor::from_dense(
///     vec![s_in.clone(), s_out.clone()],
///     vec![1.0_f64, 0.0, 0.0, 1.0],
/// ).unwrap();
///
/// let mpo = TreeTN::<_, usize>::from_tensors(vec![t], vec![0]).unwrap();
/// assert_eq!(mpo.node_count(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct LinearOperator<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// The MPO with internal index IDs
    pub(crate) mpo: TreeTN<T, V>,
    /// Input index mappings: node -> [(true s_in, internal s_in_tmp)].
    pub(crate) input_mapping: HashMap<V, Vec<IndexMapping<T::Index>>>,
    /// Output index mappings: node -> [(true s_out, internal s_out_tmp)].
    pub(crate) output_mapping: HashMap<V, Vec<IndexMapping<T::Index>>>,
}

impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike,
    <T::Index as IndexLike>::Id: Clone + std::hash::Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Create a new LinearOperator from an MPO and index mappings.
    ///
    /// # Arguments
    ///
    /// * `mpo` - The MPO with internal index IDs
    /// * `input_mapping` - Mapping from true input indices to internal indices
    /// * `output_mapping` - Mapping from true output indices to internal indices
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// // Build a 2x2 identity operator (single node)
    /// let site = DynIndex::new_dyn(2);
    /// let s_in = DynIndex::new_dyn(2);
    /// let s_out = DynIndex::new_dyn(2);
    /// let mpo_tensor = IdxTensor::from_dense(
    ///     vec![s_in.clone(), s_out.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<_, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_in });
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(0usize, IndexMapping { true_index: site.clone(), internal_index: s_out });
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// assert_eq!(op.mpo().node_count(), 1);
    /// ```
    pub fn new(
        mpo: TreeTN<T, V>,
        input_mapping: HashMap<V, IndexMapping<T::Index>>,
        output_mapping: HashMap<V, IndexMapping<T::Index>>,
    ) -> Self {
        let input_mapping = input_mapping
            .into_iter()
            .map(|(node, mapping)| (node, vec![mapping]))
            .collect();
        let output_mapping = output_mapping
            .into_iter()
            .map(|(node, mapping)| (node, vec![mapping]))
            .collect();
        Self {
            mpo,
            input_mapping,
            output_mapping,
        }
    }

    /// Create a new LinearOperator from an MPO and possibly multiple index mappings per node.
    ///
    /// Use this when one tree node owns more than one external input/output
    /// index, such as tensorized or multi-axis quantics nodes.
    ///
    /// # Arguments
    ///
    /// * `mpo` - The MPO with internal index IDs.
    /// * `input_mapping` - Mapping from node names to all true/internal input
    ///
    ///   index pairs for that node.
    /// * `output_mapping` - Mapping from node names to all true/internal output
    ///
    ///   index pairs for that node.
    ///
    /// # Examples
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor, TensorLike};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let x = DynIndex::new_dyn(2);
    /// let y = DynIndex::new_dyn(3);
    /// let x_in = DynIndex::new_dyn(2);
    /// let y_in = DynIndex::new_dyn(3);
    /// let x_out = DynIndex::new_dyn(2);
    /// let y_out = DynIndex::new_dyn(3);
    /// let mpo_tensor = IdxTensor::delta(
    ///     &[x_in.clone(), y_in.clone()],
    ///     &[x_out.clone(), y_out.clone()],
    /// ).unwrap();
    /// let mpo = TreeTN::<_, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(0usize, vec![
    ///     IndexMapping { true_index: x.clone(), internal_index: x_in },
    ///     IndexMapping { true_index: y.clone(), internal_index: y_in },
    /// ]);
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(0usize, vec![
    ///     IndexMapping { true_index: x, internal_index: x_out },
    ///     IndexMapping { true_index: y, internal_index: y_out },
    /// ]);
    ///
    /// let op = LinearOperator::new_multi(mpo, input_mapping, output_mapping);
    /// assert_eq!(op.get_input_mappings(&0).unwrap().len(), 2);
    /// ```
    pub fn new_multi(
        mpo: TreeTN<T, V>,
        input_mapping: HashMap<V, Vec<IndexMapping<T::Index>>>,
        output_mapping: HashMap<V, Vec<IndexMapping<T::Index>>>,
    ) -> Self {
        Self {
            mpo,
            input_mapping,
            output_mapping,
        }
    }

    /// Create a LinearOperator from an MPO and a reference state.
    ///
    /// This assumes:
    /// - The MPO has site indices that we need to map
    /// - The state's site indices define the true input space
    /// - For `A * x = b` with `space(x) = space(b)`, the output space equals input space
    ///
    /// # Arguments
    ///
    /// * `mpo` - The MPO (operator A)
    /// * `state` - Reference state (defines the true site index space)
    ///
    /// # Returns
    ///
    /// A LinearOperator with proper index mappings, or an error if structure is incompatible.
    /// # Errors
    ///
    /// Returns an error when the MPO and state are incompatible (a shape or index
    /// /// mismatch) or the binding fails.
    ///
    pub fn from_mpo_and_state(
        mpo: TreeTN<T, V>,
        state: &TreeTN<T, V>,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        let mut input_mapping = HashMap::new();
        let mut output_mapping = HashMap::new();

        for node in mpo.site_index_network().node_names() {
            // Get state's site indices for this node
            let state_site = state.site_space(node);

            // Get MPO's site indices for this node
            let mpo_site = mpo.site_space(node);

            match (state_site, mpo_site) {
                (Some(state_indices), Some(mpo_indices)) => {
                    // MPO should have exactly 2 site indices per state site index:
                    // one for input (s_in_tmp) and one for output (s_out_tmp)
                    // Both should have the same dimension as the state's site index.

                    if state_indices.len() * 2 != mpo_indices.len() {
                        return Err(anyhow::anyhow!(
                            "Node {:?}: MPO should have 2x site indices. State has {}, MPO has {}",
                            node,
                            state_indices.len(),
                            mpo_indices.len()
                        )
                        .into());
                    }

                    // For each state site index, find matching MPO indices by dimension
                    for state_idx in state_indices {
                        let dim = state_idx.dim();

                        // Find MPO indices with matching dimension
                        let matching_mpo: Vec<_> =
                            mpo_indices.iter().filter(|idx| idx.dim() == dim).collect();

                        if matching_mpo.len() < 2 {
                            return Err(anyhow::anyhow!(
                                "Node {:?}: Not enough MPO indices with dimension {}. Found {}",
                                node,
                                dim,
                                matching_mpo.len()
                            )
                            .into());
                        }

                        // Convention: first matching is s_in_tmp, second is s_out_tmp
                        // (This depends on how the MPO was constructed)
                        input_mapping
                            .entry(node.clone())
                            .or_insert_with(Vec::new)
                            .push(IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: matching_mpo[0].clone(),
                            });

                        // For output, use the same true index (space(x) = space(b))
                        output_mapping
                            .entry(node.clone())
                            .or_insert_with(Vec::new)
                            .push(IndexMapping {
                                true_index: state_idx.clone(),
                                internal_index: matching_mpo[1].clone(),
                            });
                    }
                }
                (None, None) => {
                    // No site indices for this node, OK
                }
                _ => {
                    return Err(anyhow::anyhow!(
                        "Node {:?}: Mismatched site space presence between state and MPO",
                        node
                    )
                    .into());
                }
            }
        }

        Ok(Self {
            mpo,
            input_mapping,
            output_mapping,
        })
    }

    /// Apply the operator to a local tensor at a specific region.
    ///
    /// This is used during the sweep for local updates.
    ///
    /// # Arguments
    ///
    /// * `local_tensor` - The local tensor (merged tensors from the region)
    /// * `region` - The nodes in the current region
    ///
    /// # Returns
    ///
    /// The result of applying the operator to the local tensor.
    /// # Errors
    ///
    /// Returns an error when the local application fails (a shape or index
    /// /// mismatch, or a backend failure).
    ///
    pub fn apply_local(
        &self,
        local_tensor: &T,
        region: &[V],
    ) -> std::result::Result<T, TreeTNOperationError> {
        // Step 1: Replace input indices in local_tensor with internal indices
        let mut transformed = local_tensor.clone();
        for node in region {
            if let Some(mappings) = self.input_mapping.get(node) {
                for mapping in mappings {
                    transformed = transformed
                        .replaceind(&mapping.true_index, &mapping.internal_index)
                        .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
                }
            }
        }

        // Step 2: Contract with local operator tensors
        let mut op_tensor: Option<T> = None;
        for node in region {
            let node_idx = self
                .mpo
                .node_index(node)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in MPO", node))?;
            let tensor = self
                .mpo
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found in MPO for node {:?}", node))?
                .clone();

            op_tensor = Some(match op_tensor {
                None => tensor,
                Some(t) => T::contract(&[&t, &tensor])
                    .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?,
            });
        }

        let op_tensor = op_tensor.ok_or_else(|| anyhow::anyhow!("Empty region"))?;

        // Contract transformed tensor with operator
        let contracted = T::contract(&[&transformed, &op_tensor])
            .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;

        // Step 3: Replace output indices back to true indices
        let mut result = contracted;
        for node in region {
            if let Some(mappings) = self.output_mapping.get(node) {
                for mapping in mappings {
                    result = result
                        .replaceind(&mapping.internal_index, &mapping.true_index)
                        .map_err(|e| TreeTNOperationError::from(anyhow::Error::new(e)))?;
                }
            }
        }

        Ok(result)
    }

    /// Get the internal MPO.
    pub fn mpo(&self) -> &TreeTN<T, V> {
        &self.mpo
    }

    /// Consume the operator and return its internal MPO.
    ///
    /// Use this when the caller needs to take ownership of the TreeTN rather
    /// than inspect it through [`Self::mpo`]. The input/output index mappings
    /// are discarded.
    ///
    /// # Returns
    ///
    /// The underlying MPO TreeTN.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let true_site = DynIndex::new_dyn(2);
    /// let input_internal = DynIndex::new_dyn(2);
    /// let output_internal = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![input_internal.clone(), output_internal.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// )?;
    /// let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(0, IndexMapping {
    ///     true_index: true_site.clone(),
    ///     internal_index: input_internal,
    /// });
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(0, IndexMapping {
    ///     true_index: true_site,
    ///     internal_index: output_internal,
    /// });
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// let mpo = op.into_mpo();
    /// assert_eq!(mpo.node_count(), 1);
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn into_mpo(self) -> TreeTN<T, V> {
        self.mpo
    }

    /// Rename operator nodes and update all mapping keys consistently.
    ///
    /// The method consumes the operator and returns a renamed operator. It
    /// rebuilds the internal MPO from the same tensors, so mappings such as
    /// `0 -> 1, 1 -> 2` work even though they would collide during a sequential
    /// in-place rename.
    ///
    /// # Arguments
    ///
    /// * `mapping` - Pairs of `(old_node, new_node)`. Nodes not listed in the
    ///
    ///   mapping keep their existing names.
    ///
    /// # Returns
    ///
    /// A `LinearOperator` with the same tensor data and index mappings attached
    /// to the renamed nodes.
    ///
    /// # Errors
    ///
    /// Returns an error when a node is not found (a missing-index failure) or the
    /// /// new name is a duplicate (a duplicate operation failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let true_site = DynIndex::new_dyn(2);
    /// let input_internal = DynIndex::new_dyn(2);
    /// let output_internal = DynIndex::new_dyn(2);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![input_internal.clone(), output_internal.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 1.0],
    /// )?;
    /// let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(0, IndexMapping {
    ///     true_index: true_site.clone(),
    ///     internal_index: input_internal,
    /// });
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(0, IndexMapping {
    ///     true_index: true_site,
    ///     internal_index: output_internal,
    /// });
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// let renamed = op.rename_nodes(&[(0, 2)])?;
    ///
    /// assert!(renamed.mpo().node_index(&0).is_none());
    /// assert!(renamed.mpo().node_index(&2).is_some());
    /// assert!(renamed.get_input_mapping(&2).is_some());
    /// assert!(renamed.get_output_mapping(&2).is_some());
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn rename_nodes(
        mut self,
        mapping: &[(V, V)],
    ) -> std::result::Result<Self, TreeTNOperationError> {
        let mut rename_map = HashMap::with_capacity(mapping.len());
        for (old_node, new_node) in mapping {
            let previous = rename_map.insert(old_node.clone(), new_node.clone());
            if !(previous.is_none()) {
                return Err(anyhow::anyhow!(
                    "LinearOperator::rename_nodes: duplicate old node {:?}",
                    old_node
                )
                .into());
            }
        }

        for old_node in rename_map.keys() {
            if !(self.mpo.node_index(old_node).is_some()) {
                return Err(anyhow::anyhow!(
                    "LinearOperator::rename_nodes: unknown old node {:?}",
                    old_node
                )
                .into());
            }
        }

        let old_names = self.mpo.node_names();
        let mut new_names = Vec::with_capacity(old_names.len());
        let mut seen_names = HashSet::with_capacity(old_names.len());
        for old_name in &old_names {
            let new_name = rename_map
                .get(old_name)
                .cloned()
                .unwrap_or_else(|| old_name.clone());
            if !(seen_names.insert(new_name.clone())) {
                return Err(anyhow::anyhow!(
                    "LinearOperator::rename_nodes: duplicate node name {:?} after renaming",
                    new_name
                )
                .into());
            }
            new_names.push(new_name);
        }

        let tensors = old_names
            .iter()
            .map(|old_name| {
                let node = self.mpo.node_index(old_name).ok_or_else(|| {
                    anyhow::anyhow!(
                        "LinearOperator::rename_nodes: node {:?} disappeared during rename",
                        old_name
                    )
                })?;
                self.mpo.tensor(node).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "LinearOperator::rename_nodes: tensor for node {:?} not found",
                        old_name
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        self.mpo = TreeTN::from_tensors(tensors, new_names)?;
        self.input_mapping = Self::rename_mapping_keys(self.input_mapping, &rename_map, "input")?;
        self.output_mapping =
            Self::rename_mapping_keys(self.output_mapping, &rename_map, "output")?;

        Ok(self)
    }

    fn rename_mapping_keys(
        mappings_by_node: HashMap<V, Vec<IndexMapping<T::Index>>>,
        rename_map: &HashMap<V, V>,
        kind: &str,
    ) -> Result<HashMap<V, Vec<IndexMapping<T::Index>>>> {
        let mut renamed = HashMap::with_capacity(mappings_by_node.len());
        for (old_node, mappings) in mappings_by_node {
            let new_node = rename_map
                .get(&old_node)
                .cloned()
                .unwrap_or_else(|| old_node.clone());
            anyhow::ensure!(
                renamed.insert(new_node.clone(), mappings).is_none(),
                "LinearOperator::rename_nodes: duplicate {kind} mapping node {:?}",
                new_node
            );
        }
        Ok(renamed)
    }

    /// Get input mapping for a node.
    pub fn get_input_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.input_mapping
            .get(node)
            .and_then(|mappings| mappings.first())
    }

    /// Get output mapping for a node.
    pub fn get_output_mapping(&self, node: &V) -> Option<&IndexMapping<T::Index>> {
        self.output_mapping
            .get(node)
            .and_then(|mappings| mappings.first())
    }

    /// Get all input mappings for a node.
    pub fn get_input_mappings(&self, node: &V) -> Option<&[IndexMapping<T::Index>]> {
        self.input_mapping.get(node).map(Vec::as_slice)
    }

    /// Get all output mappings for a node.
    pub fn get_output_mappings(&self, node: &V) -> Option<&[IndexMapping<T::Index>]> {
        self.output_mapping.get(node).map(Vec::as_slice)
    }

    fn single_site_index_from_state(state: &TreeTN<T, V>, node: &V) -> Result<T::Index> {
        let site_space = state
            .site_space(node)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found in state site space", node))?;

        if site_space.len() != 1 {
            return Err(anyhow::anyhow!(
                "Node {:?}: expected exactly 1 site index in state, found {}",
                node,
                site_space.len()
            ));
        }

        site_space
            .iter()
            .next()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Node {:?}: missing site index", node))
    }

    /// Reset true input indices to match the given state's site space.
    ///
    /// This only rewrites the external mapping. The internal MPO indices are unchanged.
    /// # Errors
    ///
    /// Returns an error when the state site space is incompatible (a shape
    /// /// mismatch) or the binding fails.
    ///
    pub fn set_input_space_from_state(
        &mut self,
        state: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let nodes: Vec<V> = self.input_mapping.keys().cloned().collect();
        for node in nodes {
            let new_true_index = Self::single_site_index_from_state(state, &node)?;
            let mappings = self
                .input_mapping
                .get_mut(&node)
                .ok_or_else(|| anyhow::anyhow!("Input mapping missing for node {:?}", node))?;
            if mappings.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Node {:?}: set_input_space_from_state only supports one input mapping per node; use explicit index binding for multi-index nodes",
                    node
                ).into());
            }
            let mapping = mappings
                .first_mut()
                .ok_or_else(|| anyhow::anyhow!("Input mapping missing for node {:?}", node))?;
            if mapping.internal_index.dim() != new_true_index.dim() {
                return Err(anyhow::anyhow!(
                    "Node {:?}: input mapping dimension {} does not match state site dimension {}",
                    node,
                    mapping.internal_index.dim(),
                    new_true_index.dim()
                )
                .into());
            }
            mapping.true_index = new_true_index;
        }
        Ok(())
    }

    /// Reset true output indices to match the given state's site space.
    ///
    /// This only rewrites the external mapping. The internal MPO indices are unchanged.
    /// # Errors
    ///
    /// Returns an error when the state site space is incompatible (a shape
    /// /// mismatch) or the binding fails.
    ///
    pub fn set_output_space_from_state(
        &mut self,
        state: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        let nodes: Vec<V> = self.output_mapping.keys().cloned().collect();
        for node in nodes {
            let new_true_index = Self::single_site_index_from_state(state, &node)?;
            let mappings = self
                .output_mapping
                .get_mut(&node)
                .ok_or_else(|| anyhow::anyhow!("Output mapping missing for node {:?}", node))?;
            if mappings.len() != 1 {
                return Err(anyhow::anyhow!(
                    "Node {:?}: set_output_space_from_state only supports one output mapping per node; use explicit index binding for multi-index nodes",
                    node
                ).into());
            }
            let mapping = mappings
                .first_mut()
                .ok_or_else(|| anyhow::anyhow!("Output mapping missing for node {:?}", node))?;
            if mapping.internal_index.dim() != new_true_index.dim() {
                return Err(anyhow::anyhow!(
                    "Node {:?}: output mapping dimension {} does not match state site dimension {}",
                    node,
                    mapping.internal_index.dim(),
                    new_true_index.dim()
                )
                .into());
            }
            mapping.true_index = new_true_index;
        }
        Ok(())
    }

    /// Align this operator's input and output site index mappings to match a target state.
    ///
    /// For each node in the operator's input and output mappings, the `true_index` is
    /// updated to the corresponding site index from the target state. The internal MPO
    /// indices remain unchanged.
    ///
    /// This is useful when an operator (e.g., from `shift_operator` or `affine_operator`)
    /// was constructed with its own site indices, but needs to be applied to a state that
    /// has different site index IDs. After calling `align_to_state`, the operator's
    /// `true_index` fields will reference the state's site indices, enabling correct
    /// index contraction during `apply_local`.
    ///
    /// # Arguments
    ///
    /// * `state` - The target state whose site indices define the true index space.
    ///
    ///   Each node in the operator's mappings must exist in the state with exactly one
    ///   site index of matching dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - A node in the operator's mapping is not found in the state
    /// - A node in the state has more than one site index
    /// - The dimension of the state's site index does not match the operator's internal index
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use tensor4all_core::{DynIndex, IndexLike, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let state_index = DynIndex::new_dyn(2);
    /// let state_tensor = IdxTensor::from_dense(vec![state_index.clone()], vec![1.0, 2.0]).unwrap();
    /// let state = TreeTN::<IdxTensor, usize>::from_tensors(vec![state_tensor], vec![0]).unwrap();
    ///
    /// let input_internal = DynIndex::new_dyn(2);
    /// let output_internal = DynIndex::new_dyn(2);
    /// let mpo_tensor = IdxTensor::from_dense(
    ///     vec![input_internal.clone(), output_internal.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(
    ///     0usize,
    ///     IndexMapping {
    ///         true_index: DynIndex::new_dyn(2),
    ///         internal_index: input_internal,
    ///     },
    /// );
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(
    ///     0usize,
    ///     IndexMapping {
    ///         true_index: DynIndex::new_dyn(2),
    ///         internal_index: output_internal,
    ///     },
    /// );
    ///
    /// let mut op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// op.align_to_state(&state).unwrap();
    ///
    /// assert_eq!(op.get_input_mapping(&0).unwrap().true_index, state_index);
    /// assert_eq!(op.get_output_mapping(&0).unwrap().true_index, state_index);
    /// ```
    pub fn align_to_state(
        &mut self,
        state: &TreeTN<T, V>,
    ) -> std::result::Result<(), TreeTNOperationError> {
        self.set_input_space_from_state(state)?;
        self.set_output_space_from_state(state)?;
        Ok(())
    }

    /// Returns the transposed operator by swapping input and output mappings.
    ///
    /// The pullback of a forward operator is its transpose: if the forward
    /// operator realizes the matrix `M_{y,x}`, the transposed operator
    /// realizes `M_{x,y}`. This method swaps `input_mapping` and
    /// `output_mapping` without copying the underlying MPO tensors — it is
    /// an O(1) operation.
    ///
    /// `.transpose().transpose()` yields an operator equivalent to the
    /// original (mappings restored, MPO unchanged).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use tensor4all_core::{DynIndex, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// let site_in = DynIndex::new_dyn(2);
    /// let site_out = DynIndex::new_dyn(3);
    /// let s_in_tmp = DynIndex::new_dyn(2);
    /// let s_out_tmp = DynIndex::new_dyn(3);
    ///
    /// let mpo_tensor = IdxTensor::from_dense(
    ///     vec![s_in_tmp.clone(), s_out_tmp.clone()],
    ///     vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0],
    /// ).unwrap();
    /// let mpo = TreeTN::<_, usize>::from_tensors(vec![mpo_tensor], vec![0]).unwrap();
    ///
    /// let mut input_mapping = HashMap::new();
    /// input_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: site_in.clone(), internal_index: s_in_tmp.clone() },
    /// );
    /// let mut output_mapping = HashMap::new();
    /// output_mapping.insert(
    ///     0usize,
    ///     IndexMapping { true_index: site_out.clone(), internal_index: s_out_tmp.clone() },
    /// );
    ///
    /// let op = LinearOperator::new(mpo, input_mapping, output_mapping);
    /// let t = op.transpose();
    ///
    /// // Input/output mappings are swapped.
    /// assert_eq!(t.get_input_mapping(&0).unwrap().true_index, site_out);
    /// assert_eq!(t.get_output_mapping(&0).unwrap().true_index, site_in);
    /// ```
    pub fn transpose(self) -> Self {
        Self {
            mpo: self.mpo,
            input_mapping: self.output_mapping,
            output_mapping: self.input_mapping,
        }
    }

    /// Restructure the internal MPO while preserving input and output mappings.
    ///
    /// This is the operator-level counterpart of [`TreeTN::restructure_to`].
    /// The `target` network is expressed in the operator's internal MPO site
    /// indices, not in the true input/output indices stored in
    /// [`IndexMapping`]. After the MPO is restructured, each mapping is moved
    /// to the target node that owns its internal index, preserving the mapping
    /// order from the original operator node order.
    ///
    /// Use this when an operator has the right local indices but its node
    /// grouping or topology needs to match another tensor network before
    /// selected-index application.
    ///
    /// # Arguments
    ///
    /// * `target` - Desired internal MPO site-index network.
    /// * `options` - Restructure options passed to [`TreeTN::restructure_to`].
    ///
    /// # Returns
    ///
    /// A new operator whose internal MPO and mapping node keys follow `target`.
    ///
    /// # Errors
    ///
    /// Returns an error when the restructure is incompatible (a shape or index
    /// /// mismatch) or the rebuild fails (a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::{HashMap, HashSet};
    ///
    /// use tensor4all_core::{DynIndex, IndexLike, IdxTensor};
    /// use tensor4all_treetn::{
    ///     IndexMapping, LinearOperator, RestructureOptions, SiteIndexNetwork, TreeTN,
    /// };
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let x_true = DynIndex::new_dyn(2);
    /// let y_true = DynIndex::new_dyn(2);
    /// let x_in = DynIndex::new_dyn(2);
    /// let x_out = DynIndex::new_dyn(2);
    /// let y_in = DynIndex::new_dyn(2);
    /// let y_out = DynIndex::new_dyn(2);
    /// let bond = DynIndex::new_dyn(1);
    ///
    /// let left = IdxTensor::from_dense(
    ///     vec![x_out.clone(), x_in.clone(), bond.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let right = IdxTensor::from_dense(
    ///     vec![bond, y_out.clone(), y_in.clone()],
    ///     vec![1.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let mut mpo = TreeTN::<IdxTensor, String>::new();
    /// mpo.add_tensor("x".to_string(), left)?;
    /// mpo.add_tensor("y".to_string(), right)?;
    /// let x_node = mpo.node_index(&"x".to_string()).unwrap();
    /// let y_node = mpo.node_index(&"y".to_string()).unwrap();
    /// let link = mpo.tensor(x_node).unwrap().indices()[2].clone();
    /// mpo.connect(x_node, &link, y_node, &link)?;
    ///
    /// let mut input = HashMap::new();
    /// input.insert("x".to_string(), IndexMapping { true_index: x_true, internal_index: x_in });
    /// input.insert("y".to_string(), IndexMapping { true_index: y_true, internal_index: y_in });
    /// let mut output = HashMap::new();
    /// output.insert("x".to_string(), IndexMapping { true_index: DynIndex::new_dyn(2), internal_index: x_out.clone() });
    /// output.insert("y".to_string(), IndexMapping { true_index: DynIndex::new_dyn(2), internal_index: y_out.clone() });
    /// let op = LinearOperator::new(mpo, input, output);
    ///
    /// let mut target = SiteIndexNetwork::new();
    /// target.add_node("left".to_string(), HashSet::from([y_out, op.get_input_mapping(&"y".to_string()).unwrap().internal_index.clone()]))?;
    /// target.add_node("right".to_string(), HashSet::from([x_out, op.get_input_mapping(&"x".to_string()).unwrap().internal_index.clone()]))?;
    /// target.add_edge(&"left".to_string(), &"right".to_string())?;
    ///
    /// let moved = op.restructure_to(&target, &RestructureOptions::default())?;
    /// assert!(moved.get_input_mapping(&"left".to_string()).is_some());
    /// assert!(moved.get_input_mapping(&"right".to_string()).is_some());
    /// # Ok(())
    /// # }
    /// ```
    pub fn restructure_to<TargetV>(
        &self,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        options: &RestructureOptions,
    ) -> std::result::Result<LinearOperator<T, TargetV>, TreeTNOperationError>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        let mpo = self.mpo.restructure_to(target, options)?;
        let input_mapping = self.restructure_mapping_nodes(&self.input_mapping, target, "input")?;
        let output_mapping =
            self.restructure_mapping_nodes(&self.output_mapping, target, "output")?;
        Ok(LinearOperator::new_multi(
            mpo,
            input_mapping,
            output_mapping,
        ))
    }

    fn restructure_mapping_nodes<TargetV>(
        &self,
        mappings_by_node: &HashMap<V, Vec<IndexMapping<T::Index>>>,
        target: &SiteIndexNetwork<TargetV, T::Index>,
        kind: &str,
    ) -> Result<HashMap<TargetV, Vec<IndexMapping<T::Index>>>>
    where
        TargetV: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    {
        let mut result: HashMap<TargetV, Vec<IndexMapping<T::Index>>> = HashMap::new();
        let mut nodes: Vec<V> = mappings_by_node.keys().cloned().collect();
        nodes.sort();

        for node in nodes {
            let Some(mappings) = mappings_by_node.get(&node) else {
                continue;
            };
            for mapping in mappings {
                let target_node = target
                    .find_node_by_index(&mapping.internal_index)
                    .cloned()
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "LinearOperator::restructure_to: {kind} internal index {:?} from node {:?} is missing from target",
                            mapping.internal_index.id(),
                            node
                        )
                    })?;
                result.entry(target_node).or_default().push(mapping.clone());
            }
        }

        Ok(result)
    }
}

impl<V> LinearOperator<IdxTensor, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
{
    /// Replace one fused input mapping with several ordered input mappings.
    ///
    /// The operator's internal input index is exactly unfused inside the MPO
    /// tensor using [`IdxTensor::unfuse_index`], then the corresponding
    /// [`IndexMapping`] entry is replaced by one entry per `new_true_indices`.
    /// New internal indices are generated automatically with matching
    /// dimensions and the same order as `new_true_indices`.
    ///
    /// Use this for tensorized operators whose local input dimension is a
    /// product, such as turning one dimension-4 input leg into two binary input
    /// legs before applying the operator to interleaved QTT groups.
    ///
    /// # Arguments
    ///
    /// * `old_true_index` - Existing true input index to replace.
    /// * `new_true_indices` - Ordered true input indices whose dimensions
    ///
    ///   multiply to `old_true_index.dim()`.
    /// * `order` - Linearization convention used to decode the old fused
    ///
    ///   coordinate into the new coordinates.
    ///
    /// # Returns
    ///
    /// A new operator with the input mapping unfused.
    ///
    /// # Errors
    ///
    /// Returns an error if the input mapping is missing or ambiguous, if the
    /// dimensions do not multiply correctly, or if the internal MPO reshape
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use tensor4all_core::{DynIndex, LinearizationOrder, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let x = DynIndex::new_dyn(4);
    /// let y = DynIndex::new_dyn(4);
    /// let x_internal = DynIndex::new_dyn(4);
    /// let y_internal = DynIndex::new_dyn(4);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![y_internal.clone(), x_internal.clone()],
    ///     vec![1.0, 0.0, 0.0, 0.0,
    ///          0.0, 1.0, 0.0, 0.0,
    ///          0.0, 0.0, 1.0, 0.0,
    ///          0.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let mut input = HashMap::new();
    /// input.insert(0, IndexMapping { true_index: x.clone(), internal_index: x_internal });
    /// let mut output = HashMap::new();
    /// output.insert(0, IndexMapping { true_index: y, internal_index: y_internal });
    /// let op = LinearOperator::new(mpo, input, output);
    ///
    /// let x0 = DynIndex::new_dyn(2);
    /// let x1 = DynIndex::new_dyn(2);
    /// let unfused = op.unfuse_input_index(&x, &[x0.clone(), x1.clone()], LinearizationOrder::ColumnMajor)?;
    ///
    /// let mappings = unfused.get_input_mappings(&0).unwrap();
    /// assert_eq!(mappings.len(), 2);
    /// assert_eq!(mappings[0].true_index, x0);
    /// assert_eq!(mappings[1].true_index, x1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unfuse_input_index(
        &self,
        old_true_index: &DynIndex,
        new_true_indices: &[DynIndex],
        order: LinearizationOrder,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        self.unfuse_mapping_index(old_true_index, new_true_indices, order, MappingKind::Input)
            .map_err(TreeTNOperationError::from)
    }

    /// Replace one fused output mapping with several ordered output mappings.
    ///
    /// This is the output-space counterpart of
    /// [`Self::unfuse_input_index`]. The internal MPO output index is exactly
    /// reshaped, and the output mapping vector is expanded in the order given
    /// by `new_true_indices`.
    ///
    /// # Arguments
    ///
    /// * `old_true_index` - Existing true output index to replace.
    /// * `new_true_indices` - Ordered true output indices whose dimensions
    ///
    ///   multiply to `old_true_index.dim()`.
    /// * `order` - Linearization convention used to decode the old fused
    ///
    ///   coordinate into the new coordinates.
    ///
    /// # Returns
    ///
    /// A new operator with the output mapping unfused.
    ///
    /// # Errors
    ///
    /// Returns an error if the output mapping is missing or ambiguous, if the
    /// dimensions do not multiply correctly, or if the internal MPO reshape
    /// fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::collections::HashMap;
    ///
    /// use tensor4all_core::{DynIndex, LinearizationOrder, IdxTensor};
    /// use tensor4all_treetn::{IndexMapping, LinearOperator, TreeTN};
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let x = DynIndex::new_dyn(4);
    /// let y = DynIndex::new_dyn(4);
    /// let x_internal = DynIndex::new_dyn(4);
    /// let y_internal = DynIndex::new_dyn(4);
    /// let tensor = IdxTensor::from_dense(
    ///     vec![y_internal.clone(), x_internal.clone()],
    ///     vec![1.0, 0.0, 0.0, 0.0,
    ///          0.0, 1.0, 0.0, 0.0,
    ///          0.0, 0.0, 1.0, 0.0,
    ///          0.0, 0.0, 0.0, 1.0],
    /// )?;
    /// let mpo = TreeTN::<IdxTensor, usize>::from_tensors(vec![tensor], vec![0])?;
    /// let mut input = HashMap::new();
    /// input.insert(0, IndexMapping { true_index: x, internal_index: x_internal });
    /// let mut output = HashMap::new();
    /// output.insert(0, IndexMapping { true_index: y.clone(), internal_index: y_internal });
    /// let op = LinearOperator::new(mpo, input, output);
    ///
    /// let y0 = DynIndex::new_dyn(2);
    /// let y1 = DynIndex::new_dyn(2);
    /// let unfused = op.unfuse_output_index(&y, &[y0.clone(), y1.clone()], LinearizationOrder::ColumnMajor)?;
    ///
    /// let mappings = unfused.get_output_mappings(&0).unwrap();
    /// assert_eq!(mappings.len(), 2);
    /// assert_eq!(mappings[0].true_index, y0);
    /// assert_eq!(mappings[1].true_index, y1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn unfuse_output_index(
        &self,
        old_true_index: &DynIndex,
        new_true_indices: &[DynIndex],
        order: LinearizationOrder,
    ) -> std::result::Result<Self, TreeTNOperationError> {
        self.unfuse_mapping_index(old_true_index, new_true_indices, order, MappingKind::Output)
            .map_err(TreeTNOperationError::from)
    }

    fn unfuse_mapping_index(
        &self,
        old_true_index: &DynIndex,
        new_true_indices: &[DynIndex],
        order: LinearizationOrder,
        kind: MappingKind,
    ) -> Result<Self> {
        if !(!new_true_indices.is_empty()) {
            return Err(anyhow::anyhow!(
                "LinearOperator::{kind}: replacement indices must not be empty"
            ));
        }
        let product = new_true_indices
            .iter()
            .try_fold(1usize, |acc, index| acc.checked_mul(index.dim()))
            .ok_or_else(|| anyhow::anyhow!("LinearOperator::{kind}: dimension product overflow"))?;
        anyhow::ensure!(
            product == old_true_index.dim(),
            "LinearOperator::{kind}: replacement dimension product {} does not match old dimension {}",
            product,
            old_true_index.dim()
        );

        let (node, position, old_internal_index) = self.find_mapping_entry(old_true_index, kind)?;
        let new_internal_indices = new_true_indices
            .iter()
            .map(|index| DynIndex::new_dyn(index.dim()))
            .collect::<Vec<_>>();

        let mpo = self.mpo.replace_site_index_with_indices(
            &old_internal_index,
            &new_internal_indices,
            order,
        )?;

        let mut result = self.clone();
        result.mpo = mpo;

        let mappings = match kind {
            MappingKind::Input => result.input_mapping.get_mut(&node),
            MappingKind::Output => result.output_mapping.get_mut(&node),
        }
        .ok_or_else(|| {
            anyhow::anyhow!(
                "LinearOperator::{kind}: mapping node {:?} disappeared during unfuse",
                node
            )
        })?;

        let replacements = new_true_indices
            .iter()
            .cloned()
            .zip(new_internal_indices)
            .map(|(true_index, internal_index)| IndexMapping {
                true_index,
                internal_index,
            })
            .collect::<Vec<_>>();
        mappings.splice(position..=position, replacements);

        Ok(result)
    }

    fn find_mapping_entry(
        &self,
        old_true_index: &DynIndex,
        kind: MappingKind,
    ) -> Result<(V, usize, DynIndex)> {
        let mappings_by_node = match kind {
            MappingKind::Input => &self.input_mapping,
            MappingKind::Output => &self.output_mapping,
        };
        let mut nodes: Vec<V> = mappings_by_node.keys().cloned().collect();
        nodes.sort();

        let mut found = None;
        for node in nodes {
            let Some(mappings) = mappings_by_node.get(&node) else {
                continue;
            };
            for (position, mapping) in mappings.iter().enumerate() {
                if mapping.true_index == *old_true_index {
                    if found.is_some() {
                        return Err(anyhow::anyhow!(
                            "LinearOperator::{kind}: true index {:?} appears in more than one mapping",
                            old_true_index.id()
                        ));
                    }
                    found = Some((node.clone(), position, mapping.internal_index.clone()));
                }
            }
        }

        found.ok_or_else(|| {
            anyhow::anyhow!(
                "LinearOperator::{kind}: true index {:?} not found",
                old_true_index.id()
            )
        })
    }
}

#[derive(Clone, Copy)]
enum MappingKind {
    Input,
    Output,
}

impl std::fmt::Display for MappingKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => f.write_str("unfuse_input_index"),
            Self::Output => f.write_str("unfuse_output_index"),
        }
    }
}

// ============================================================================
// Helper methods
// ============================================================================

use crate::operator::Operator;

// Implement Operator trait for LinearOperator
impl<T, V> Operator<T, V> for LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike + Clone + Hash + Eq,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    fn site_indices(&self) -> HashSet<T::Index> {
        // Return union of input and output true indices
        let mut result: HashSet<T::Index> = self
            .input_mapping
            .values()
            .flat_map(|mappings| mappings.iter().map(|m| m.true_index.clone()))
            .collect();
        result.extend(
            self.output_mapping
                .values()
                .flat_map(|mappings| mappings.iter().map(|m| m.true_index.clone())),
        );
        result
    }

    fn site_index_network(&self) -> &SiteIndexNetwork<V, T::Index> {
        self.mpo.site_index_network()
    }

    fn node_names(&self) -> HashSet<V> {
        self.mpo
            .site_index_network()
            .node_names()
            .into_iter()
            .cloned()
            .collect()
    }
}

impl<T, V> LinearOperator<T, V>
where
    T: TensorLike,
    T::Index: IndexLike,
    V: Clone + Hash + Eq + Send + Sync + std::fmt::Debug,
{
    /// Get all input site indices (true indices from state space).
    pub fn input_site_indices(&self) -> HashSet<T::Index> {
        self.input_mapping
            .values()
            .flat_map(|mappings| mappings.iter().map(|m| m.true_index.clone()))
            .collect()
    }

    /// Get all output site indices (true indices from result space).
    pub fn output_site_indices(&self) -> HashSet<T::Index> {
        self.output_mapping
            .values()
            .flat_map(|mappings| mappings.iter().map(|m| m.true_index.clone()))
            .collect()
    }

    /// Get all input mappings.
    pub fn input_mappings(&self) -> &HashMap<V, Vec<IndexMapping<T::Index>>> {
        &self.input_mapping
    }

    /// Get all output mappings.
    pub fn output_mappings(&self) -> &HashMap<V, Vec<IndexMapping<T::Index>>> {
        &self.output_mapping
    }
}

#[cfg(test)]
mod tests;
