//! Index set for managing multi-indices with bidirectional lookup

use std::collections::HashMap;
use std::hash::Hash;

/// A bidirectional index set for efficient lookup
///
/// Provides O(1) lookup from integer index to value and from value to integer index.
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
    /// Create an empty index set
    pub fn new() -> Self {
        Self {
            to_int: HashMap::new(),
            from_int: Vec::new(),
        }
    }

    /// Create an index set from a vector
    pub fn from_vec(values: Vec<T>) -> Self {
        let to_int: HashMap<T, usize> = values
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();
        Self {
            to_int,
            from_int: values,
        }
    }

    /// Get the value at integer index
    pub fn get(&self, i: usize) -> Option<&T> {
        self.from_int.get(i)
    }

    /// Get the integer position of a value
    pub fn pos(&self, value: &T) -> Option<usize> {
        self.to_int.get(value).copied()
    }

    /// Get positions for a slice of values
    pub fn positions(&self, values: &[T]) -> Option<Vec<usize>> {
        values.iter().map(|v| self.pos(v)).collect()
    }

    /// Push a new value to the set
    pub fn push(&mut self, value: T) {
        let idx = self.from_int.len();
        self.from_int.push(value.clone());
        self.to_int.insert(value, idx);
    }

    /// Check if the set contains a value
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

    /// Iterate over values
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.from_int.iter()
    }

    /// Get all values as a slice
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

/// MultiIndex type alias
pub type MultiIndex = Vec<usize>;

/// LocalIndex type alias
pub type LocalIndex = usize;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexset_basic() {
        let mut set: IndexSet<Vec<usize>> = IndexSet::new();
        set.push(vec![1, 2, 3]);
        set.push(vec![4, 5, 6]);

        assert_eq!(set.len(), 2);
        assert_eq!(set.get(0), Some(&vec![1, 2, 3]));
        assert_eq!(set.pos(&vec![1, 2, 3]), Some(0));
        assert_eq!(set.pos(&vec![4, 5, 6]), Some(1));
        assert_eq!(set.pos(&vec![7, 8, 9]), None);
    }

    #[test]
    fn test_indexset_from_vec() {
        let set = IndexSet::from_vec(vec![vec![1], vec![2], vec![3]]);
        assert_eq!(set.len(), 3);
        assert_eq!(set[0], vec![1]);
        assert_eq!(set[2], vec![3]);
    }

    #[test]
    fn test_indexset_contains() {
        let set = IndexSet::from_vec(vec![1, 2, 3]);
        assert!(set.contains(&1));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));
    }

    #[test]
    fn test_indexset_iter() {
        let set = IndexSet::from_vec(vec![10, 20, 30]);
        let collected: Vec<_> = set.iter().copied().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }
}
