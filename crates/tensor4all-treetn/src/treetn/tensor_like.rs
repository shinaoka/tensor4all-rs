//! TensorIndex and TensorLike implementations for TreeTN.
//!
//! **Design Decision**: TreeTN implements TensorIndex but NOT TensorLike.
//!
//! ## TensorIndex (Implemented)
//!
//! TreeTN implements TensorIndex because index operations are well-defined:
//! - `external_indices()`: Returns all site (physical) indices
//! - `replaceind()` / `replaceinds()`: Replace indices in tensors and metadata
//!
//! ## TensorLike (NOT Implemented)
//!
//! TreeTN does NOT implement TensorLike because:
//! 1. **Unclear semantics**: What would `tensordot` between two TreeTNs mean?
//! 2. **Hidden costs**: Full contraction has exponential cost
//! 3. **Separation of concerns**: Dense tensors and TNs are fundamentally different
//!
//! ## Alternative API
//!
//! TreeTN provides its own methods instead:
//! - `site_indices()`: Returns physical indices (not bonds)
//! - `contract_to_tensor()`: Explicit method for full contraction (exponential cost)
//! - `contract_nodes()`: Graph operations for node contraction

use std::hash::Hash;

use anyhow::Result;
use tensor4all_core::{
    DynIndex, IndexLike, LinearizationOrder, TensorDynLen, TensorIndex, TensorLike,
};

use crate::error::NumberedTagSelectionError;

use super::TreeTN;

// ============================================================================
// TensorIndex implementation for TreeTN
// ============================================================================

impl<T, V> TensorIndex for TreeTN<T, V>
where
    T: TensorLike,
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <T::Index as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    type Index = T::Index;
    type Error = crate::TreeTNOperationError;

    /// Return all external (site/physical) indices from all nodes.
    ///
    /// This collects all site indices from `site_index_network`.
    /// Bond indices are NOT included (they are internal to the network).
    fn external_indices(&self) -> Vec<Self::Index> {
        let mut result = Vec::new();
        for node_name in self.node_names() {
            if let Some(site_space) = self.site_space(&node_name) {
                result.extend(site_space.iter().cloned());
            }
        }
        result
    }

    fn num_external_indices(&self) -> usize {
        self.node_names()
            .iter()
            .filter_map(|name| self.site_space(name))
            .map(|space| space.len())
            .sum()
    }

    /// Replace an index in this TreeTN.
    ///
    /// Looks up the index location (site or link) and replaces it in:
    /// - The tensor containing it
    /// - The appropriate index network (site_index_network or link_index_network)
    ///
    /// Note: `replace_tensor` automatically updates the `site_index_network` based on
    /// the new tensor's indices, so we don't need to manually call `replace_site_index`.
    fn replaceind(
        &self,
        old_index: &Self::Index,
        new_index: &Self::Index,
    ) -> std::result::Result<Self, Self::Error> {
        // Validate dimension match
        if old_index.dim() != new_index.dim() {
            return Err(anyhow::anyhow!(
                "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                old_index.dim(),
                new_index.dim()
            )
            .into());
        }

        let mut result = self.clone();

        // Check if it's a site index
        if let Some(node_name) = self.site_index_network.find_node_by_index(old_index) {
            let node_idx = result
                .node_index(node_name)
                .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;

            // Replace in tensor - this also updates site_index_network via replace_tensor
            let tensor = result
                .tensor(node_idx)
                .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))?;
            let old_in_tensor = tensor
                .external_indices()
                .iter()
                .find(|idx| *idx == old_index)
                .ok_or_else(|| {
                    anyhow::anyhow!("Index not found in tensor at node {:?}", node_name)
                })?
                .clone();
            let new_tensor = tensor
                .replaceind(&old_in_tensor, new_index)
                .map_err(anyhow::Error::new)?;
            result.replace_tensor(node_idx, new_tensor)?;

            // Keep ortho_towards consistent (if present)
            if let Some(dir) = result.ortho_towards.remove(old_index) {
                result.ortho_towards.insert(new_index.clone(), dir);
            }

            return Ok(result);
        }

        // Check if it's a link index
        if let Some(edge) = self.link_index_network.find_edge(old_index) {
            let (node_a, node_b) = result
                .graph
                .graph()
                .edge_endpoints(edge)
                .ok_or_else(|| anyhow::anyhow!("Edge {:?} not found", edge))?;

            // IMPORTANT: Update edge weight FIRST so replace_tensor validation matches.
            *result
                .bond_index_mut(edge)
                .ok_or_else(|| anyhow::anyhow!("Bond index not found"))? = new_index.clone();

            // Replace in both endpoint tensors - this also updates site_index_network
            for node in [node_a, node_b] {
                let tensor = result
                    .tensor(node)
                    .ok_or_else(|| anyhow::anyhow!("Tensor not found"))?;
                let old_in_tensor = tensor
                    .external_indices()
                    .iter()
                    .find(|idx| *idx == old_index)
                    .ok_or_else(|| anyhow::anyhow!("Bond index not found in endpoint tensor"))?
                    .clone();
                let new_tensor = tensor
                    .replaceind(&old_in_tensor, new_index)
                    .map_err(anyhow::Error::new)?;
                result.replace_tensor(node, new_tensor)?;
            }

            // Keep ortho_towards consistent (if present)
            if let Some(dir) = result.ortho_towards.remove(old_index) {
                result.ortho_towards.insert(new_index.clone(), dir);
            }

            // Replace in link_index_network
            result
                .link_index_network
                .replace_index(old_index, new_index, edge)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            return Ok(result);
        }

        Err(anyhow::anyhow!("Index {:?} not found in TreeTN", old_index.id()).into())
    }

    /// Replace multiple indices in this TreeTN.
    fn replaceinds(
        &self,
        old_indices: &[Self::Index],
        new_indices: &[Self::Index],
    ) -> std::result::Result<Self, Self::Error> {
        if old_indices.len() != new_indices.len() {
            return Err(anyhow::anyhow!(
                "Length mismatch: {} old indices, {} new indices",
                old_indices.len(),
                new_indices.len()
            )
            .into());
        }

        let mut result = self.clone();
        for (old, new) in old_indices.iter().zip(new_indices.iter()) {
            result = result.replaceind(old, new).map_err(anyhow::Error::new)?;
        }
        Ok(result)
    }
}

impl<V> TreeTN<TensorDynLen, V>
where
    V: Clone + Hash + Eq + Ord + Send + Sync + std::fmt::Debug,
    <DynIndex as IndexLike>::Id: Clone + Hash + Eq + Ord + std::fmt::Debug + Send + Sync,
{
    /// Return external indices whose tag set contains `tag`.
    ///
    /// This filters only site/external indices; internal link indices are not
    /// considered. The tag is matched exactly against one tag entry, so
    /// `"k"` and `"k=1"` are different tags.
    ///
    /// # Arguments
    /// * `tag` - Exact tag to match, such as `"x"` or `"k=1"`.
    ///
    /// # Returns
    /// A vector of matching external indices. Multiple matches are allowed and
    /// an empty vector means no external index has the requested tag.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, TagSet, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let x = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,x").unwrap());
    /// let y = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,y").unwrap());
    /// let tensor = TensorDynLen::from_dense(vec![x.clone(), y], vec![0.0; 4]).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    ///
    /// assert_eq!(tn.external_indices_with_tag("x"), vec![x]);
    /// assert!(tn.external_indices_with_tag("missing").is_empty());
    /// ```
    pub fn external_indices_with_tag(&self, tag: &str) -> Vec<DynIndex> {
        self.external_indices()
            .into_iter()
            .filter(|index| index.tags().has_tag(tag))
            .collect()
    }

    /// Return external indices matching numbered tags in an explicit range.
    ///
    /// For `tag_prefix = "k"`, `start_index = 1`, and `count = 3`, this
    /// looks for exactly one external index with each of the tags `"k=1"`,
    /// `"k=2"`, and `"k=3"`, and returns them in that numeric order.
    ///
    /// # Arguments
    /// * `tag_prefix` - Prefix before the equals sign, such as `"k"` or `"x"`.
    ///   It must not contain `=`.
    /// * `start_index` - First numeric suffix to request. Use `1` for the
    ///   usual quantics convention, or `0` for zero-based tags.
    /// * `count` - Number of consecutive tags to request. `0` returns an
    ///   empty vector.
    ///
    /// # Returns
    /// The matching external indices ordered as `tag_prefix=start_index`,
    /// `tag_prefix=start_index+1`, and so on.
    ///
    /// # Errors
    /// Returns an error if `tag_prefix` contains `=`, if any requested numbered
    /// tag is absent, if a requested numbered tag matches more than one
    /// external index, or if `start_index + count` overflows `usize`.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, TagSet, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let k1 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=1").unwrap());
    /// let k2 = DynIndex::new_dyn_with_tags(2, TagSet::from_str("Qubit,k=2").unwrap());
    /// let tensor = TensorDynLen::from_dense(vec![k2.clone(), k1.clone()], vec![0.0; 4]).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    ///
    /// let indices = tn.external_indices_with_numbered_tag("k", 1, 2).unwrap();
    /// assert_eq!(indices, vec![k1, k2]);
    /// ```
    pub fn external_indices_with_numbered_tag(
        &self,
        tag_prefix: &str,
        start_index: usize,
        count: usize,
    ) -> std::result::Result<Vec<DynIndex>, NumberedTagSelectionError> {
        if tag_prefix.contains('=') {
            return Err(NumberedTagSelectionError::InvalidPrefix {
                tag_prefix: tag_prefix.to_string(),
            });
        }

        let external_indices = self.external_indices();
        let mut result = Vec::with_capacity(count);
        for offset in 0..count {
            let tag_number = start_index.checked_add(offset).ok_or(
                NumberedTagSelectionError::RangeOverflow {
                    start_index,
                    offset,
                },
            )?;
            let numbered_tag = format!("{tag_prefix}={tag_number}");
            let mut matches = external_indices
                .iter()
                .filter(|index| index.tags().has_tag(&numbered_tag))
                .cloned();
            let first = matches.next();
            let second = matches.next();

            match (first, second) {
                (None, _) => {
                    return Err(NumberedTagSelectionError::MissingTag { tag: numbered_tag });
                }
                (Some(index), None) => result.push(index),
                (Some(_), Some(_)) => {
                    return Err(NumberedTagSelectionError::AmbiguousTag { tag: numbered_tag });
                }
            }
        }
        Ok(result)
    }

    /// Replace one site index with multiple site indices using an exact reshape.
    ///
    /// This is a TreeTN-level wrapper around [`TensorDynLen::unfuse_index`].
    /// It updates the owning node tensor and the site-index metadata, without
    /// introducing any approximation.
    ///
    /// # Examples
    /// ```
    /// use tensor4all_core::{DynIndex, LinearizationOrder, TensorDynLen};
    /// use tensor4all_treetn::TreeTN;
    ///
    /// let fused = DynIndex::new_dyn(4);
    /// let tensor = TensorDynLen::from_dense(vec![fused.clone()], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// let tn = TreeTN::<TensorDynLen, usize>::from_tensors(vec![tensor], vec![0]).unwrap();
    /// let left = DynIndex::new_dyn(2);
    /// let right = DynIndex::new_dyn(2);
    ///
    /// let unfused = tn
    ///     .replace_site_index_with_indices(
    ///         &fused,
    ///         &[left.clone(), right.clone()],
    ///         LinearizationOrder::ColumnMajor,
    ///     )
    ///     .unwrap();
    ///
    /// let dense = unfused.contract_to_tensor().unwrap();
    /// let expected = TensorDynLen::from_dense(vec![left, right], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert!(dense.distance(&expected).unwrap() < 1.0e-12);
    /// ```
    pub fn replace_site_index_with_indices(
        &self,
        old_index: &DynIndex,
        new_indices: &[DynIndex],
        order: LinearizationOrder,
    ) -> Result<Self> {
        let node_name = self
            .site_index_network
            .find_node_by_index(old_index)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "site index {:?} not found in TreeTN site index network",
                    old_index.id()
                )
            })?;
        let node_idx = self
            .node_index(&node_name)
            .ok_or_else(|| anyhow::anyhow!("Node {:?} not found", node_name))?;
        let tensor = self
            .tensor(node_idx)
            .ok_or_else(|| anyhow::anyhow!("Tensor not found for node {:?}", node_name))?;
        let new_tensor = tensor.unfuse_index(old_index, new_indices, order)?;

        let mut result = self.clone();
        result.replace_tensor(node_idx, new_tensor)?;

        if let Some(dir) = result.ortho_towards.remove(old_index) {
            for new_index in new_indices {
                result.ortho_towards.insert(new_index.clone(), dir.clone());
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests;
