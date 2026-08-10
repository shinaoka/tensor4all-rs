use crate::error::Result as TreeTciResult;
use crate::{assemble::MultiIndex, column_2d, ncols_2d, SubtreeKey, TreeTciEdge, TreeTciGraph};
use anyhow::Result;

use std::collections::hash_map::Entry;
use std::collections::{BTreeMap, HashMap};
use std::marker::PhantomData;
use tensor4all_core::ColMajorArray;

/// TreeTCI state mirroring the upstream `SimpleTCI` layout.
///
/// Stores the current pivot sets, bond errors, and tree graph metadata
/// for tree tensor cross interpolation. The type parameter `T` is the
/// scalar type (e.g., `f64`, `Complex64`).
///
/// Use [`TreeTCI2::new`] to create a fresh state, then seed it with
/// [`TreeTCI2::add_global_pivots`] before passing to
/// [`optimize_default`](crate::optimize_default) or
/// [`crossinterpolate2`](crate::crossinterpolate2).
///
/// # Examples
///
/// ```
/// use tensor4all_treetci::{TreeTCI2, TreeTciEdge, TreeTciGraph};
///
/// let graph = TreeTciGraph::new(3, &[
///     TreeTciEdge::new(0, 1),
///     TreeTciEdge::new(1, 2),
/// ]).unwrap();
///
/// let mut state = TreeTCI2::<f64>::new(vec![2, 3, 2], graph).unwrap();
/// assert_eq!(state.max_rank(), 0);
/// assert!((state.max_bond_error() - 0.0).abs() < f64::EPSILON);
///
/// // Seed with a global pivot
/// state.add_global_pivots(&[vec![0, 0, 0]]).unwrap();
/// assert!(state.max_rank() >= 1);
/// ```
#[derive(Clone, Debug)]
pub struct TreeTCI2<T> {
    /// Pivot sets keyed by canonical subtree keys.
    /// Shape of each entry: [n_subtree_sites, n_pivots].
    pub ijset: HashMap<SubtreeKey, ColMajorArray<usize>>,
    /// Local dimensions for each site.
    pub local_dims: Vec<usize>,
    /// Tree graph metadata.
    pub graph: TreeTciGraph,
    /// Error estimate per edge bipartition.
    pub bond_errors: BTreeMap<TreeTciEdge, f64>,
    /// Back-truncation style pivot errors.
    pub pivot_errors: Vec<f64>,
    /// Maximum observed sample magnitude for normalization.
    pub max_sample_value: f64,
    /// Previous pivot sets for candidate-generation history.
    pub ijset_history: Vec<HashMap<SubtreeKey, ColMajorArray<usize>>>,
    marker: PhantomData<T>,
}

/// Deprecated alias for `TreeTCI2`.
#[deprecated(note = "use TreeTCI2")]
pub type SimpleTreeTci<T> = TreeTCI2<T>;

impl<T> TreeTCI2<T> {
    /// Create a new TreeTCI state from local dimensions and a tree graph.
    /// # Errors
    ///
    /// Returns an error when the construction or conversion fails (a shape or
    /// /// index mismatch, or a backend failure).
    ///
    pub fn new(local_dims: Vec<usize>, graph: TreeTciGraph) -> TreeTciResult<Self> {
        if !(local_dims.len() > 1) {
            return Err(anyhow::anyhow!("local_dims should have at least 2 elements").into());
        };
        if !(local_dims.len() == graph.n_sites()) {
            return Err(anyhow::anyhow!(
                "local_dims length {} must match graph site count {}",
                local_dims.len(),
                graph.n_sites()
            )
            .into());
        };

        let bond_errors = graph
            .edges()
            .into_iter()
            .map(|edge| (edge, 0.0))
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            ijset: HashMap::new(),
            local_dims,
            graph,
            bond_errors,
            pivot_errors: Vec::new(),
            max_sample_value: 0.0,
            ijset_history: Vec::new(),
            marker: PhantomData,
        })
    }

    /// Add global pivots and project them to every edge bipartition.
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    pub fn add_global_pivots(&mut self, pivots: &[MultiIndex]) -> TreeTciResult<()> {
        let n_sites = self.local_dims.len();
        if !(pivots.iter().all(|pivot| pivot.len() == n_sites)) {
            return Err(
                anyhow::anyhow!("each global pivot must contain one index per site").into(),
            );
        };

        for pivot in pivots {
            for edge in self.graph.edges() {
                let (left_key, right_key) = self.graph.subregion_vertices(edge)?;
                let left_projection = project_pivot(pivot, &left_key);
                let right_projection = project_pivot(pivot, &right_key);
                let n_left = left_key.as_slice().len();
                let n_right = right_key.as_slice().len();
                match self.ijset.entry(left_key) {
                    Entry::Occupied(mut entry) => {
                        push_unique_column(entry.get_mut(), &left_projection)?;
                    }
                    Entry::Vacant(entry) => {
                        let mut array = empty_2d(n_left)?;
                        push_unique_column(&mut array, &left_projection)?;
                        entry.insert(array);
                    }
                }
                match self.ijset.entry(right_key) {
                    Entry::Occupied(mut entry) => {
                        push_unique_column(entry.get_mut(), &right_projection)?;
                    }
                    Entry::Vacant(entry) => {
                        let mut array = empty_2d(n_right)?;
                        push_unique_column(&mut array, &right_projection)?;
                        entry.insert(array);
                    }
                }
            }
        }

        let full_key = SubtreeKey::new((0..n_sites).collect());
        if let Entry::Vacant(entry) = self.ijset.entry(full_key) {
            entry.insert(empty_2d(n_sites)?);
        }
        Ok(())
    }

    /// Reset the sweep-local pivot error accumulator.
    pub fn flush_pivot_errors(&mut self) {
        self.pivot_errors.clear();
    }

    /// Update one bond error.
    pub fn update_bond_error(&mut self, edge: TreeTciEdge, error: f64) {
        self.bond_errors.insert(edge, error);
    }

    /// Merge a new pivot-error vector into the sweep-local maximum.
    pub fn update_pivot_errors(&mut self, errors: &[f64]) {
        let len = self.pivot_errors.len().max(errors.len());
        self.pivot_errors.resize(len, 0.0);
        for (idx, &error) in errors.iter().enumerate() {
            self.pivot_errors[idx] = self.pivot_errors[idx].max(error);
        }
    }

    /// Maximum bond error across all edges.
    pub fn max_bond_error(&self) -> f64 {
        self.bond_errors.values().copied().fold(0.0, f64::max)
    }

    /// Maximum current bond dimension across stored subtree pivot sets.
    pub fn max_rank(&self) -> usize {
        self.ijset
            .values()
            .filter_map(|arr| arr.ncols())
            .max()
            .unwrap_or(0)
    }
}

fn project_pivot(pivot: &MultiIndex, key: &SubtreeKey) -> MultiIndex {
    key.as_slice().iter().map(|&site| pivot[site]).collect()
}

/// Create an empty 2D ColMajorArray with shape [nrows, 0].
fn empty_2d(nrows: usize) -> Result<ColMajorArray<usize>> {
    Ok(ColMajorArray::new(vec![], vec![nrows, 0])?)
}

/// Push a column to a ColMajorArray if it is not already present.
pub(crate) fn push_unique_column(array: &mut ColMajorArray<usize>, column: &[usize]) -> Result<()> {
    for j in 0..ncols_2d(array)? {
        if column_2d(array, j)? == column {
            return Ok(()); // duplicate
        }
    }
    array.push_column(column)?;
    Ok(())
}

#[cfg(test)]
mod tests;
