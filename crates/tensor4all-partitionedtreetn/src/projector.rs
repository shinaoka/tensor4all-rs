//! Projectors that fix selected TreeTN site indices to coordinates.
//!
//! The implementation follows the corrected `Projector` in
//! `tensor4all-partitionedtt/src/projector.rs` after issue #634. In particular,
//! equality, hashing, and deterministic ordering use the same full index
//! identity: ID, canonical tags, and prime level.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::error::{PartitionedTreeTNError, Result};
use tensor4all_core::{DynIndex, TagSetLike};

/// A map from full dynamic site indices to fixed zero-based coordinates.
///
/// Index equality is the core index contract: IDs, tags, and prime levels are
/// significant, while the dimension is checked separately when a projector is
/// applied to a TreeTN.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_partitionedtreetn::Projector;
///
/// let index = DynIndex::new_dyn(2);
/// let projector = Projector::from_pairs([(index.clone(), 1)]).unwrap();
/// assert_eq!(projector.get(&index), Some(1));
/// assert_eq!(projector.len(), 1);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Projector {
    data: HashMap<DynIndex, usize>,
}

impl Projector {
    /// Create an empty projector.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let projector = Projector::new();
    /// assert!(projector.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Create a projector from `(index, coordinate)` pairs.
    ///
    /// Coordinates are zero-based. Repeated equal full identities replace the
    /// stored key and coordinate together after validating the new coordinate.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] when
    /// a coordinate is not smaller than the supplied index dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let index = DynIndex::new_dyn(3);
    /// let projector = Projector::from_pairs([(index.clone(), 2)]).unwrap();
    /// assert_eq!(projector.get(&index), Some(2));
    /// ```
    pub fn from_pairs(pairs: impl IntoIterator<Item = (DynIndex, usize)>) -> Result<Self> {
        let mut projector = Self::new();
        for (index, value) in pairs {
            projector.insert(index, value)?;
        }
        Ok(projector)
    }

    /// Return whether `index` has a fixed coordinate.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let projector = Projector::from_pairs([(index.clone(), 0)]).unwrap();
    /// assert!(projector.is_projected_at(&index));
    /// ```
    pub fn is_projected_at(&self, index: &DynIndex) -> bool {
        self.data.contains_key(index)
    }

    /// Return the fixed coordinate for `index`, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let projector = Projector::from_pairs([(index.clone(), 1)]).unwrap();
    /// assert_eq!(projector.get(&index), Some(1));
    /// ```
    pub fn get(&self, index: &DynIndex) -> Option<usize> {
        self.data.get(index).copied()
    }

    /// Iterate over projected full indices.
    ///
    /// The iteration order is unspecified. Use the projector's deterministic
    /// ordering implementation when an order-independent result is required.
    pub fn projected_indices(&self) -> impl Iterator<Item = &DynIndex> {
        self.data.keys()
    }

    /// Return the number of projected indices.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return whether no indices are projected.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over `(index, coordinate)` pairs in unspecified hash-map order.
    pub fn iter(&self) -> impl Iterator<Item = (&DynIndex, &usize)> {
        self.data.iter()
    }

    /// Insert or replace a projection transactionally.
    ///
    /// # Errors
    ///
    /// Returns [`PartitionedTreeTNError::ProjectorCoordinateOutOfBounds`] and
    /// leaves the projector unchanged when `value >= index.dim`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let mut projector = Projector::new();
    /// projector.insert(index.clone(), 1).unwrap();
    /// assert_eq!(projector.get(&index), Some(1));
    /// ```
    pub fn insert(&mut self, index: DynIndex, value: usize) -> Result<()> {
        if value >= index.dim {
            let dim = index.dim;
            return Err(PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
                index,
                value,
                dim,
            });
        }

        // HashMap::insert keeps an equal old key. Remove first so the stored
        // full identity and coordinate are replaced as one transaction.
        self.data.remove(&index);
        self.data.insert(index, value);
        Ok(())
    }

    /// Remove a projection and return its coordinate, if present.
    pub fn remove(&mut self, index: &DynIndex) -> Option<usize> {
        self.data.remove(index)
    }

    /// Merge compatible projections, or return `None` for a coordinate conflict.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_partitionedtreetn::Projector;
    ///
    /// let index = DynIndex::new_dyn(2);
    /// let left = Projector::from_pairs([(index.clone(), 0)]).unwrap();
    /// let right = Projector::from_pairs([(index.clone(), 0)]).unwrap();
    /// assert!(left.intersection(&right).is_some());
    /// ```
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let mut result = self.data.clone();

        for (index, &value) in &other.data {
            if let Some(&existing) = result.get(index) {
                if existing != value {
                    return None;
                }
            } else {
                result.insert(index.clone(), value);
            }
        }

        Some(Self { data: result })
    }

    /// Keep only projections shared by both projectors with equal coordinates.
    pub fn common_restriction(&self, other: &Self) -> Self {
        let mut result = HashMap::new();

        for (index, &value) in &self.data {
            if other.get(index) == Some(value) {
                result.insert(index.clone(), value);
            }
        }

        Self { data: result }
    }

    /// Return whether two projectors describe an overlapping region.
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.intersection(other).is_some()
    }

    /// Return whether `self` fixes at least all coordinates fixed by `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        for (index, &value) in &other.data {
            match self.data.get(index) {
                Some(&stored) if stored == value => continue,
                _ => return false,
            }
        }
        self.data.len() >= other.data.len()
    }

    /// Return whether all projectors in `projectors` are pairwise disjoint.
    pub fn are_disjoint(projectors: &[Self]) -> bool {
        for (i, left) in projectors.iter().enumerate() {
            for right in projectors.iter().skip(i + 1) {
                if left.is_compatible_with(right) {
                    return false;
                }
            }
        }
        true
    }

    /// Return a projector restricted to the supplied full index set.
    pub fn filter_indices(&self, indices: &[DynIndex]) -> Self {
        let index_set: std::collections::HashSet<_> = indices.iter().collect();
        Self {
            data: self
                .data
                .iter()
                .filter(|(index, _)| index_set.contains(index))
                .map(|(index, value)| (index.clone(), *value))
                .collect(),
        }
    }

    /// Return entries in canonical full-identity and coordinate order.
    pub(crate) fn canonical_entries(&self) -> Vec<(&DynIndex, usize)> {
        let mut entries: Vec<_> = self
            .data
            .iter()
            .map(|(index, &value)| (index, value))
            .collect();
        entries.sort_by(canonical_entry_cmp);
        entries
    }

    /// Compare projectors by a deterministic total order compatible with `Eq`.
    #[allow(dead_code)]
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

pub(crate) fn canonical_index_cmp(left: &DynIndex, right: &DynIndex) -> Ordering {
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            Some(Ordering::Equal)
        } else if self.is_subset_of(other) {
            Some(Ordering::Less)
        } else if other.is_subset_of(self) {
            Some(Ordering::Greater)
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
