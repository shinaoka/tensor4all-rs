//! Index set for managing ordered collections with bidirectional lookup.
//!
//! [`IndexSet`] maintains insertion order while providing O(1) lookup in both
//! directions: from positional index to value and from value to positional
//! index. Duplicate insertions are silently ignored.
//!
//! This is used throughout the TCI infrastructure for managing pivot indices
//! in matrix cross interpolation algorithms.

use std::collections::HashMap;
use std::hash::Hash;

/// A bidirectional index set for efficient lookup
///
/// Provides O(1) lookup from integer index to value and from value to integer index.
///
/// # Examples
///
/// ```
/// use tensor4all_core::IndexSet;
///
/// let mut set: IndexSet<String> = IndexSet::new();
/// set.push("alpha".to_string());
/// set.push("beta".to_string());
/// set.push("alpha".to_string()); // duplicate, ignored
///
/// assert_eq!(set.len(), 2);
/// assert_eq!(set.get(0), Some(&"alpha".to_string()));
/// assert_eq!(set.pos(&"beta".to_string()), Some(1));
/// assert!(set.contains(&"alpha".to_string()));
/// assert!(!set.contains(&"gamma".to_string()));
/// ```
#[derive(Debug, Clone)]
pub struct IndexSet<T: Clone + Eq + Hash> {
    /// Map from value to integer index
    to_int: HashMap<T, usize>,
    /// Map from integer index to value
    from_int: Vec<T>,
}

impl<T: Clone + Eq + Hash> Default for IndexSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Eq + Hash> IndexSet<T> {
    /// Create an empty index set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set: IndexSet<usize> = IndexSet::new();
    /// assert!(set.is_empty());
    /// assert_eq!(set.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            to_int: HashMap::new(),
            from_int: Vec::new(),
        }
    }

    /// Create an index set from a vector
    ///
    /// Duplicate values are removed, keeping the first occurrence.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10usize, 20, 10, 30]);
    /// assert_eq!(set.len(), 3);
    /// assert_eq!(set[0], 10);
    /// assert_eq!(set[1], 20);
    /// assert_eq!(set[2], 30);
    /// ```
    pub fn from_vec(values: Vec<T>) -> Self {
        let mut to_int = HashMap::new();
        let mut from_int = Vec::new();
        for value in values {
            if !to_int.contains_key(&value) {
                let idx = from_int.len();
                to_int.insert(value.clone(), idx);
                from_int.push(value);
            }
        }
        Self { to_int, from_int }
    }

    /// Get the value at a positional index, or `None` if out of range.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20, 30]);
    /// assert_eq!(set.get(0), Some(&10));
    /// assert_eq!(set.get(2), Some(&30));
    /// assert_eq!(set.get(3), None);
    /// ```
    pub fn get(&self, i: usize) -> Option<&T> {
        self.from_int.get(i)
    }

    /// Get the positional index of a value, or `None` if not present.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20, 30]);
    /// assert_eq!(set.pos(&20), Some(1));
    /// assert_eq!(set.pos(&99), None);
    /// ```
    pub fn pos(&self, value: &T) -> Option<usize> {
        self.to_int.get(value).copied()
    }

    /// Get positional indices for a slice of values.
    ///
    /// Returns `None` if any value is not present in the set.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20, 30]);
    /// assert_eq!(set.positions(&[30, 10]), Some(vec![2, 0]));
    /// assert_eq!(set.positions(&[10, 99]), None);
    /// ```
    pub fn positions(&self, values: &[T]) -> Option<Vec<usize>> {
        values.iter().map(|v| self.pos(v)).collect()
    }

    /// Push a new value to the set.
    ///
    /// If the value already exists in the set, this is a no-op.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let mut set = IndexSet::new();
    /// set.push(10);
    /// set.push(20);
    /// set.push(10); // duplicate, ignored
    /// assert_eq!(set.len(), 2);
    /// assert_eq!(set[0], 10);
    /// assert_eq!(set[1], 20);
    /// ```
    pub fn push(&mut self, value: T) {
        if self.to_int.contains_key(&value) {
            return;
        }
        let idx = self.from_int.len();
        self.from_int.push(value.clone());
        self.to_int.insert(value, idx);
    }

    /// Check if the set contains a value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20]);
    /// assert!(set.contains(&10));
    /// assert!(!set.contains(&30));
    /// ```
    pub fn contains(&self, value: &T) -> bool {
        self.to_int.contains_key(value)
    }

    /// Number of elements in the set
    pub fn len(&self) -> usize {
        self.from_int.len()
    }

    /// Check if the set is empty
    pub fn is_empty(&self) -> bool {
        self.from_int.is_empty()
    }

    /// Iterate over values in insertion order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20, 30]);
    /// let collected: Vec<_> = set.iter().copied().collect();
    /// assert_eq!(collected, vec![10, 20, 30]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.from_int.iter()
    }

    /// Get all values as a slice in insertion order.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::IndexSet;
    ///
    /// let set = IndexSet::from_vec(vec![10, 20, 30]);
    /// assert_eq!(set.values(), &[10, 20, 30]);
    /// ```
    pub fn values(&self) -> &[T] {
        &self.from_int
    }
}

impl<T: Clone + Eq + Hash> std::ops::Index<usize> for IndexSet<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.from_int[i]
    }
}

impl<T: Clone + Eq + Hash> IntoIterator for IndexSet<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.from_int.into_iter()
    }
}

impl<'a, T: Clone + Eq + Hash> IntoIterator for &'a IndexSet<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.from_int.iter()
    }
}

/// A multi-index: a vector of site-local indices.
pub type MultiIndex = Vec<usize>;

/// A single site-local index.
pub type LocalIndex = usize;

#[cfg(test)]
mod tests;
