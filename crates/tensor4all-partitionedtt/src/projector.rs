//! Projector type for defining subdomains in tensor index space
//!
//! A projector maps tensor indices (DynIndex) to fixed values, defining a subdomain
//! where specific indices are fixed to particular values.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::{PartitionedTTError, Result};
use tensor4all_core::{DynIndex, TagSetLike};

/// A projector maps tensor indices to fixed integer values.
///
/// Used to define subdomains in the tensor index space. Each index
/// can be projected to a specific value (0-indexed).
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtt::Projector;
/// use tensor4all_core::index::{Index, DynId};
///
/// let idx0 = Index::new_dyn(2);
/// let idx1 = Index::new_dyn(3);
///
/// // Create a projector that fixes idx0 to value 1
/// let p = Projector::from_pairs([(idx0.clone(), 1)]).unwrap();
///
/// assert!(p.is_projected_at(&idx0));
/// assert!(!p.is_projected_at(&idx1));
/// assert_eq!(p.get(&idx0), Some(1));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Projector {
    /// Maps index -> projected value (0-indexed)
    data: HashMap<DynIndex, usize>,
}

impl Projector {
    /// Create an empty projector (no indices projected)
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a projector from `(index, projected_value)` pairs.
    ///
    /// Coordinates are zero-based and must be smaller than their index
    /// dimension. If an identity appears more than once, the last validated
    /// pair replaces the earlier key and value together.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] when a
    /// coordinate is not in `0..index.dim()`.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Result<Self> {
        let mut projector = Self::new();
        for (index, value) in pairs {
            projector.insert(index, value)?;
        }
        Ok(projector)
    }

    /// Check if an index is projected
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.data.contains_key(index)
    }

    /// Get the projected value at an index (None if not projected)
    pub fn get(&self, index: &DynIndex) -> Option<usize> {
        self.data.get(index).copied()
    }

    /// Get all projected indices
    pub fn projected_indices(&self) -> impl Iterator<Item = &DynIndex> {
        self.data.keys()
    }

    /// Number of projected indices
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if no indices are projected
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over (index, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&DynIndex, &usize)> {
        self.data.iter()
    }

    /// Insert or update a projection.
    ///
    /// `value` is a zero-based coordinate and must be smaller than
    /// `index.dim()`. An update whose index is equal by full projector
    /// identity replaces the stored key and coordinate together.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTTError::ProjectorCoordinateOutOfBounds`] and
    /// leaves this projector unchanged when `value >= index.dim()`.
    pub fn insert(&mut self, index: DynIndex, value: usize) -> Result<()> {
        if value >= index.dim {
            let dim = index.dim;
            return Err(PartitionedTTError::ProjectorCoordinateOutOfBounds { index, value, dim });
        }

        // HashMap::insert keeps the old key when an equal key already exists.
        // Remove first so the stored identity and coordinate are replaced as a
        // pair, including when only the dimension differs.
        self.data.remove(&index);
        self.data.insert(index, value);
        Ok(())
    }

    /// Remove a projection
    pub fn remove(&mut self, index: &DynIndex) -> Option<usize> {
        self.data.remove(index)
    }

    /// Intersection: returns Some(merged) if projectors are compatible, None if conflicting.
    ///
    /// Two projectors are compatible if they don't have conflicting values at the same index.
    /// The result contains all projections from both projectors.
    ///
    /// This corresponds to Julia's `a & b` operator.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let mut result = self.data.clone();

        for (index, &value) in &other.data {
            if let Some(&existing) = result.get(index) {
                if existing != value {
                    // Conflict: same index has different values
                    return None;
                }
            } else {
                result.insert(index.clone(), value);
            }
        }

        Some(Self { data: result })
    }

    /// Common restriction: keeps only indices where both projectors agree.
    ///
    /// The result contains only indices that are projected in both projectors
    /// with the same value. This is the "agreement" or "common restriction"
    /// of two projectors.
    ///
    /// Note: This is NOT a set-theoretic union. It returns the common
    /// projections where both projectors project the same index to the same value.
    ///
    /// This corresponds to Julia's `a | b` operator.
    pub fn common_restriction(&self, other: &Self) -> Self {
        let mut result = HashMap::new();

        for (index, &value) in &self.data {
            if other.get(index) == Some(value) {
                result.insert(index.clone(), value);
            }
        }

        Self { data: result }
    }

    /// Check if two projectors have overlapping (compatible) regions.
    ///
    /// Returns true if the projectors can be merged without conflict.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.intersection(other).is_some()
    }

    /// Check if self is a subset of other.
    ///
    /// `a.is_subset_of(b)` means that `a` is more restrictive than `b`:
    /// `a` projects at least as many indices as `b`, and all indices projected
    /// in `b` have the same values in `a`.
    ///
    /// This corresponds to Julia's `a < b` (where a < b means a is more restrictive).
    pub fn is_subset_of(&self, other: &Self) -> bool {
        // All indices in other must be in self with the same value
        for (index, &value) in &other.data {
            match self.data.get(index) {
                Some(&v) if v == value => continue,
                _ => return false,
            }
        }

        // self must project at least as many indices as other
        self.data.len() >= other.data.len()
    }

    /// Check if a vector of projectors are mutually disjoint (non-overlapping).
    ///
    /// Returns true if no two projectors have overlapping regions.
    pub fn are_disjoint(projectors: &[Self]) -> bool {
        for (i, a) in projectors.iter().enumerate() {
            for b in projectors.iter().skip(i + 1) {
                if a.is_compatible_with(b) {
                    return false;
                }
            }
        }
        true
    }

    /// Filter projector to only include indices from the given set.
    ///
    /// Returns a new projector containing only the indices that are
    /// present in both this projector and the given index set.
    pub fn filter_indices(&self, indices: &[DynIndex]) -> Self {
        let index_set: std::collections::HashSet<_> = indices.iter().collect();
        Self {
            data: self
                .data
                .iter()
                .filter(|(k, _)| index_set.contains(k))
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }

    /// Return entries in the canonical identity and coordinate order.
    pub(crate) fn canonical_entries(&self) -> Vec<(&DynIndex, usize)> {
        let mut entries: Vec<_> = self
            .data
            .iter()
            .map(|(index, &value)| (index, value))
            .collect();
        entries.sort_by(canonical_entry_cmp);
        entries
    }

    /// Compare projectors by a deterministic total order compatible with Eq.
    pub(crate) fn canonical_cmp(&self, other: &Self) -> Ordering {
        let left = self.canonical_entries();
        let right = other.canonical_entries();
        match left.len().cmp(&right.len()) {
            Ordering::Equal => {}
            order => return order,
        }

        for (left_entry, right_entry) in left.iter().zip(&right) {
            match canonical_entry_cmp(left_entry, right_entry) {
                Ordering::Equal => {}
                order => return order,
            }
        }
        Ordering::Equal
    }
}

fn canonical_entry_cmp(
    (left_index, left_value): &(&DynIndex, usize),
    (right_index, right_value): &(&DynIndex, usize),
) -> Ordering {
    canonical_index_cmp(left_index, right_index).then_with(|| left_value.cmp(right_value))
}

fn canonical_index_cmp(left: &DynIndex, right: &DynIndex) -> Ordering {
    left.id
        .cmp(&right.id)
        .then_with(|| left.tags.iter().cmp(right.tags.iter()))
        .then_with(|| left.plev.cmp(&right.plev))
}

impl PartialEq for Projector {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Projector {}

impl std::hash::Hash for Projector {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.len().hash(state);
        for (index, value) in self.canonical_entries() {
            index.hash(state);
            value.hash(state);
        }
    }
}

impl PartialOrd for Projector {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self == other {
            Some(std::cmp::Ordering::Equal)
        } else if self.is_subset_of(other) {
            Some(std::cmp::Ordering::Less)
        } else if other.is_subset_of(self) {
            Some(std::cmp::Ordering::Greater)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for &'a Projector {
    type Item = (&'a DynIndex, &'a usize);
    type IntoIter = std::collections::hash_map::Iter<'a, DynIndex, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl IntoIterator for Projector {
    type Item = (DynIndex, usize);
    type IntoIter = std::collections::hash_map::IntoIter<DynIndex, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests;
