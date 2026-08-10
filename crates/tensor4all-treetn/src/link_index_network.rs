//! Link Index Network for bond/link index management.
//!
//! Provides efficient reverse lookup from full index metadata to edge.
//! This complements `SiteIndexNetwork` (which handles site/physical indices)
//! by handling link/bond indices between nodes.

use crate::error::TreeTNOperationError;
use petgraph::stable_graph::EdgeIndex;
use std::collections::HashMap;
use std::fmt::Debug;
use tensor4all_core::IndexLike;

/// Link Index Network: manages bond/link indices with reverse lookup.
///
/// Provides O(1) lookup from an index to the edge containing that index.
/// This is essential for efficient `replaceind` operations on TreeTN.
///
/// # Type Parameters
/// - `I`: Index type (must implement `IndexLike`)
#[derive(Debug, Clone)]
pub struct LinkIndexNetwork<I>
where
    I: IndexLike,
{
    /// Reverse lookup: full index metadata → edge containing this index.
    index_to_edge: HashMap<I, EdgeIndex>,
}

impl<I> Default for LinkIndexNetwork<I>
where
    I: IndexLike,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I> LinkIndexNetwork<I>
where
    I: IndexLike,
{
    /// Create a new empty LinkIndexNetwork.
    pub fn new() -> Self {
        Self {
            index_to_edge: HashMap::new(),
        }
    }

    /// Create with initial capacity.
    pub fn with_capacity(edges: usize) -> Self {
        Self {
            index_to_edge: HashMap::with_capacity(edges),
        }
    }

    /// Register a link index for an edge.
    ///
    /// # Arguments
    /// * `edge` - The edge index
    /// * `index` - The link index on this edge
    pub fn insert(&mut self, edge: EdgeIndex, index: &I) {
        self.index_to_edge.insert(index.clone(), edge);
    }

    /// Remove a link index registration.
    ///
    /// # Arguments
    /// * `index` - The link index to remove
    ///
    /// # Returns
    /// The edge that was associated with this index, if any.
    pub fn remove(&mut self, index: &I) -> Option<EdgeIndex> {
        self.index_to_edge.remove(index)
    }

    /// Find the edge containing a given index.
    ///
    /// # Arguments
    /// * `index` - The index to look up
    ///
    /// # Returns
    /// The edge containing this index, or None if not found.
    pub fn find_edge(&self, index: &I) -> Option<EdgeIndex> {
        self.index_to_edge.get(index).copied()
    }

    /// Check if an index is registered.
    pub fn contains(&self, index: &I) -> bool {
        self.index_to_edge.contains_key(index)
    }

    /// Update the index for an edge (e.g., after SVD creates new bond index).
    ///
    /// # Arguments
    /// * `old_index` - The old index to replace
    /// * `new_index` - The new index
    /// * `edge` - The edge (for validation)
    ///
    /// # Returns
    /// Ok if successful, Err if old_index was not registered or edge mismatch.
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    ///  a backend failure).
    ///
    pub fn replace_index(
        &mut self,
        old_index: &I,
        new_index: &I,
        edge: EdgeIndex,
    ) -> std::result::Result<(), TreeTNOperationError> {
        match self.index_to_edge.remove(old_index) {
            Some(old_edge) => {
                if old_edge != edge {
                    // Restore and return error
                    self.index_to_edge.insert(old_index.clone(), old_edge);
                    return Err(TreeTNOperationError::from(anyhow::anyhow!(
                        "Edge mismatch: old_index was on edge {:?}, not {:?}",
                        old_edge,
                        edge
                    )));
                }
                self.index_to_edge.insert(new_index.clone(), edge);
                Ok(())
            }
            None => Err(TreeTNOperationError::from(anyhow::anyhow!(
                "Index {:?} not found in link_index_network",
                old_index.id()
            ))),
        }
    }

    /// Number of registered link indices.
    pub fn len(&self) -> usize {
        self.index_to_edge.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.index_to_edge.is_empty()
    }

    /// Clear all registrations.
    pub fn clear(&mut self) {
        self.index_to_edge.clear();
    }

    /// Iterate over all (index, edge) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&I, &EdgeIndex)> {
        self.index_to_edge.iter()
    }
}

#[cfg(test)]
mod tests;
