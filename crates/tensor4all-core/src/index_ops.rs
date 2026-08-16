use crate::IndexLike;
use smallvec::SmallVec;

const SMALL_CONTRACTION_INLINE: usize = 8;
const LINEAR_CONTRACTION_SCAN_LIMIT: usize = 64;

/// Small axis list used by contraction preparation.
pub(crate) type AxisVec = SmallVec<[usize; SMALL_CONTRACTION_INLINE]>;
/// Small index list used by contraction preparation.
pub(crate) type IndexVec<I> = SmallVec<[I; SMALL_CONTRACTION_INLINE]>;

type AxisPairVec = SmallVec<[(usize, usize); SMALL_CONTRACTION_INLINE]>;
type BoolVec = SmallVec<[bool; SMALL_CONTRACTION_INLINE]>;

/// Error type for index replacement operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplaceIndsError {
    /// The symmetry space of the replacement index does not match the original.
    SpaceMismatch {
        /// The dimension/size of the original index
        from_dim: usize,
        /// The dimension/size of the replacement index
        to_dim: usize,
    },
    /// Duplicate indices found in the collection.
    DuplicateIndices {
        /// The position of the first duplicate index
        first_pos: usize,
        /// The position of the duplicate index
        duplicate_pos: usize,
    },
}

impl std::fmt::Display for ReplaceIndsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplaceIndsError::SpaceMismatch { from_dim, to_dim } => {
                write!(
                    f,
                    "Index space mismatch: cannot replace index with dimension {} with index of dimension {}",
                    from_dim, to_dim
                )
            }
            ReplaceIndsError::DuplicateIndices {
                first_pos,
                duplicate_pos,
            } => {
                write!(
                    f,
                    "Duplicate indices found: index at position {} equals index at position {}",
                    duplicate_pos, first_pos
                )
            }
        }
    }
}

impl std::error::Error for ReplaceIndsError {}

/// Check if a collection of indices contains any duplicate full indices.
///
/// # Arguments
/// * `indices` - Collection of indices to check
///
/// # Returns
/// `Ok(())` if all indices are unique, or `Err(ReplaceIndsError::DuplicateIndices)` if duplicates are found.
///
/// # Errors
///
/// Returns an error when the operation fails (a shape or index mismatch, or
/// /// a backend failure).
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::check_unique_indices;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let indices = vec![i.clone(), j.clone()];
/// assert!(check_unique_indices(&indices).is_ok());
///
/// let duplicate = vec![i.clone(), i.clone()];
/// assert!(check_unique_indices(&duplicate).is_err());
/// ```
pub fn check_unique_indices<I: IndexLike>(indices: &[I]) -> Result<(), ReplaceIndsError> {
    use std::collections::HashMap;
    let mut seen: HashMap<&I, usize> = HashMap::with_capacity(indices.len());
    for (pos, idx) in indices.iter().enumerate() {
        if let Some(&first_pos) = seen.get(idx) {
            return Err(ReplaceIndsError::DuplicateIndices {
                first_pos,
                duplicate_pos: pos,
            });
        }
        seen.insert(idx, pos);
    }
    Ok(())
}

/// Replace indices in a collection based on full-index matching.
///
/// This corresponds to ITensors.jl's `replace_indices` function. It replaces indices
/// in `indices` that equal any of the `(old, new)` pairs in `replacements`.
/// The replacement index must have the same dimension as the original.
///
/// # Arguments
/// * `indices` - Collection of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices equal to `old_index` are replaced with `new_index`
///
/// # Returns
/// A new vector with replacements applied, or an error if any replacement has a shape mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different dimension than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replace_indices;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);  // Same size as j
///
/// let indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// let replaced = replace_indices(indices, &replacements).unwrap();
/// assert_eq!(replaced.len(), 3);
/// assert_eq!(replaced[1].id, new_j.id);
/// ```
pub fn replace_indices<I: IndexLike>(
    indices: Vec<I>,
    replacements: &[(I, I)],
) -> Result<Vec<I>, ReplaceIndsError> {
    // Check for duplicates in input indices
    check_unique_indices(&indices)?;

    // Build a map from old index to new index for fast lookup.
    let mut replacement_map = std::collections::HashMap::with_capacity(replacements.len());
    for (old, new) in replacements {
        // Validate dimension match
        if old.dim() != new.dim() {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.dim(),
                to_dim: new.dim(),
            });
        }
        replacement_map.insert(old.clone(), new);
    }

    // Apply replacements
    let mut result = Vec::with_capacity(indices.len());
    for idx in indices {
        if let Some(new_idx) = replacement_map.get(&idx) {
            result.push((*new_idx).clone());
        } else {
            result.push(idx);
        }
    }

    // Check for duplicates in result indices
    check_unique_indices(&result)?;
    Ok(result)
}

/// Replace indices in-place based on full-index matching.
///
/// This is an in-place variant of `replace_indices` that modifies the input slice directly.
/// Useful for performance-critical code where you want to avoid allocations.
///
/// # Arguments
/// * `indices` - Mutable slice of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices equal to `old_index` are replaced with `new_index`
///
/// # Returns
/// `Ok(())` on success, or an error if any replacement has a shape mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different dimension than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replace_indices_mut;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);
///
/// let mut indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// replace_indices_mut(&mut indices, &replacements).unwrap();
/// assert_eq!(indices[1].id, new_j.id);
/// ```
pub fn replace_indices_mut<I: IndexLike>(
    indices: &mut [I],
    replacements: &[(I, I)],
) -> Result<(), ReplaceIndsError> {
    // Check for duplicates in input indices
    check_unique_indices(indices)?;

    // Build a map from old index to new index for fast lookup.
    let mut replacement_map = std::collections::HashMap::with_capacity(replacements.len());
    for (old, new) in replacements {
        // Validate dimension match
        if old.dim() != new.dim() {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.dim(),
                to_dim: new.dim(),
            });
        }
        replacement_map.insert(old.clone(), new);
    }

    // Apply replacements in-place
    for idx in indices.iter_mut() {
        if let Some(new_idx) = replacement_map.get(idx) {
            *idx = (*new_idx).clone();
        }
    }

    // Check for duplicates in result indices
    check_unique_indices(indices)?;
    Ok(())
}

/// Find indices that are unique to the first collection (set difference A \ B).
///
/// Returns indices that appear in `indices_a` but not in `indices_b` (matched by full index).
/// This corresponds to ITensors.jl's `uniqueinds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices from `indices_a` that are not in `indices_b`.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::unique_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let unique = unique_inds(&indices_a, &indices_b);
/// assert_eq!(unique.len(), 1);
/// assert_eq!(unique[0].id, i.id);
/// ```
pub fn unique_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    indices_a
        .iter()
        .filter(|idx| !indices_b.iter().any(|other| other == *idx))
        .cloned()
        .collect()
}

/// Find indices that are not common between two collections (symmetric difference).
///
/// Returns indices that appear in either `indices_a` or `indices_b` but not in both
/// (matched by full index). This corresponds to ITensors.jl's `noncommoninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices from both collections that are not common to both.
/// Order: indices from A first (in original order), then indices from B (in original order).
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::noncommon_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let noncommon = noncommon_inds(&indices_a, &indices_b);
/// assert_eq!(noncommon.len(), 2);  // i and k
/// ```
pub fn noncommon_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    // Pre-allocate with estimated capacity (worst case: no common indices)
    let mut result = Vec::with_capacity(indices_a.len() + indices_b.len());

    // Add indices from A that are not in B
    result.extend(
        indices_a
            .iter()
            .filter(|idx| !indices_b.iter().any(|other| other == *idx))
            .cloned(),
    );
    // Add indices from B that are not in A
    result.extend(
        indices_b
            .iter()
            .filter(|idx| !indices_a.iter().any(|other| other == *idx))
            .cloned(),
    );
    result
}

/// Find the union of two index collections.
///
/// Returns all unique indices from both collections (matched by full index).
/// This corresponds to ITensors.jl's `unioninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing all unique indices from both collections.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::union_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let union = union_inds(&indices_a, &indices_b);
/// assert_eq!(union.len(), 3);  // i, j, k
/// ```
pub fn union_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    let mut seen: std::collections::HashSet<&I> =
        std::collections::HashSet::with_capacity(indices_a.len() + indices_b.len());
    let mut result = Vec::with_capacity(indices_a.len() + indices_b.len());

    for idx in indices_a {
        if seen.insert(idx) {
            result.push(idx.clone());
        }
    }
    for idx in indices_b {
        if seen.insert(idx) {
            result.push(idx.clone());
        }
    }
    result
}

/// Check if a collection contains a specific full index.
///
/// This corresponds to ITensors.jl's `hasind` function.
///
/// # Arguments
/// * `indices` - Collection of indices to search
/// * `index` - The index to look for
///
/// # Returns
/// `true` if an index with matching ID is found, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hasind;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let indices = vec![i.clone(), j.clone()];
///
/// assert!(hasind(&indices, &i));
/// assert!(!hasind(&indices, &Index::new_dyn(4)));
/// ```
pub fn hasind<I: IndexLike>(indices: &[I], index: &I) -> bool {
    indices.iter().any(|idx| idx == index)
}

/// Check if a collection contains all of the specified full indices.
///
/// This corresponds to ITensors.jl's `hasinds` function.
///
/// # Arguments
/// * `indices` - Collection of indices to search
/// * `targets` - The indices to look for
///
/// # Returns
/// `true` if all target indices are found, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::has_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let indices = vec![i.clone(), j.clone(), k.clone()];
///
/// assert!(has_inds(&indices, &[i.clone(), j.clone()]));
/// assert!(!has_inds(&indices, &[i.clone(), Index::new_dyn(5)]));
/// ```
#[doc(alias = "hasinds")]
pub fn has_inds<I: IndexLike>(indices: &[I], targets: &[I]) -> bool {
    targets
        .iter()
        .all(|target| indices.iter().any(|idx| idx == target))
}

/// Check if two collections have any common full indices.
///
/// This corresponds to ITensors.jl's `hascommoninds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// `true` if there is at least one common index, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::has_common_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// assert!(has_common_inds(&indices_a, &indices_b));
/// assert!(!has_common_inds(&[i.clone()], &[k.clone()]));
/// ```
#[doc(alias = "hascommoninds")]
pub fn has_common_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> bool {
    indices_a
        .iter()
        .any(|idx| indices_b.iter().any(|other| other == idx))
}

/// Find common indices between two index collections.
///
/// Returns a vector of indices that appear in both `indices_a` and `indices_b`
/// (set intersection). This is similar to ITensors.jl's `commoninds` function.
///
/// Time complexity: O(n + m) where n = len(indices_a), m = len(indices_b).
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices that are common to both collections (matched by full index).
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::common_inds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let common = common_inds(&indices_a, &indices_b);
/// assert_eq!(common.len(), 1);
/// assert_eq!(common[0].id, j.id);
/// ```
pub fn common_inds<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<I> {
    indices_a
        .iter()
        .filter(|idx| indices_b.iter().any(|other| other == *idx))
        .cloned()
        .collect()
}

/// Find contractable indices between two slices and return their positions.
///
/// Returns a vector of `(pos_a, pos_b)` tuples where each tuple indicates
/// that `indices_a[pos_a]` and `indices_b[pos_b]` are contractable
/// (same ID, same dimension, and compatible ConjState).
///
/// # Example
/// ```
/// use tensor4all_core::index::DefaultIndex as Index;
/// use tensor4all_core::index_ops::common_ind_positions;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// let positions = common_ind_positions(&indices_a, &indices_b);
/// assert_eq!(positions, vec![(1, 0)]); // j is at position 1 in a, position 0 in b
/// ```
pub fn common_ind_positions<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> Vec<(usize, usize)> {
    common_ind_positions_small(indices_a, indices_b).into_vec()
}

fn common_ind_positions_small<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> AxisPairVec {
    let scan_work = indices_a.len().saturating_mul(indices_b.len());
    if scan_work <= LINEAR_CONTRACTION_SCAN_LIMIT {
        return common_ind_positions_linear(indices_a, indices_b);
    }
    common_ind_positions_hashed(indices_a, indices_b)
}

fn common_ind_positions_linear<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> AxisPairVec {
    let mut positions = AxisPairVec::new();
    for (pos_a, idx_a) in indices_a.iter().enumerate() {
        for (pos_b, idx_b) in indices_b.iter().enumerate() {
            if idx_a.is_contractable(idx_b) {
                positions.push((pos_a, pos_b));
                break; // Each index in a can match at most one in b
            }
        }
    }
    positions
}

fn common_ind_positions_hashed<I: IndexLike>(indices_a: &[I], indices_b: &[I]) -> AxisPairVec {
    use std::collections::HashMap;

    // Bucket indices_b by their FULL value and probe each `idx_a` under the
    // key `conj(idx_a)`: the linear scan accepts exactly `b == conj(a)`
    // (undirected: conj is the identity; directed: is_contractable requires
    // `conj(a) == b`, even when conj changes the id). Keying by Id, or by
    // bare value with a same-value probe, would silently drop directed pairs
    // whose conj partner carries a different id (#615).
    let mut positions_by_value: HashMap<I, SmallVec<[usize; 2]>> =
        HashMap::with_capacity(indices_b.len());
    for (pos_b, idx_b) in indices_b.iter().enumerate() {
        positions_by_value
            .entry(idx_b.clone())
            .or_default()
            .push(pos_b);
    }

    let mut positions = AxisPairVec::new();
    for (pos_a, idx_a) in indices_a.iter().enumerate() {
        let Some(candidate_positions) = positions_by_value.get(&idx_a.conj()) else {
            continue;
        };
        for &pos_b in candidate_positions {
            if idx_a.is_contractable(&indices_b[pos_b]) {
                positions.push((pos_a, pos_b));
                break; // Each index in a can match at most one in b
            }
        }
    }
    positions
}

/// Result of preparing a tensor contraction.
///
/// Contains all the information needed to perform the contraction:
/// - Which axes to contract from each tensor
/// - The resulting indices after contraction
#[derive(Debug, Clone)]
pub(crate) struct ContractionSpec<I: IndexLike> {
    /// Axes to contract from the first tensor (positions in `indices_a`).
    pub axes_a: AxisVec,
    /// Axes to contract from the second tensor (positions in `indices_b`).
    pub axes_b: AxisVec,
    /// Indices of the result tensor (non-contracted indices from both tensors).
    pub result_indices: IndexVec<I>,
}

/// Error type for contraction preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ContractionError {
    /// No common indices found for contraction.
    NoCommonIndices,
    /// Dimension mismatch for a common index.
    DimensionMismatch {
        /// Position in the first tensor.
        pos_a: usize,
        /// Position in the second tensor.
        pos_b: usize,
        /// Dimension in the first tensor.
        dim_a: usize,
        /// Dimension in the second tensor.
        dim_b: usize,
    },
    /// Duplicate axis specified in contraction.
    DuplicateAxis {
        /// Which tensor has the duplicate ("self" or "other").
        tensor: &'static str,
        /// Position of the duplicate axis.
        pos: usize,
    },
    /// Index not found in tensor.
    IndexNotFound {
        /// Which tensor the index was not found in.
        tensor: &'static str,
    },
    /// Batch contraction not yet implemented.
    BatchContractionNotImplemented,
}

impl std::fmt::Display for ContractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContractionError::NoCommonIndices => {
                write!(f, "No common indices found for contraction")
            }
            ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a,
                dim_b,
            } => {
                write!(
                    f,
                    "Dimension mismatch: tensor_a[{}]={} != tensor_b[{}]={}",
                    pos_a, dim_a, pos_b, dim_b
                )
            }
            ContractionError::DuplicateAxis { tensor, pos } => {
                write!(f, "Duplicate axis {} in {} tensor", pos, tensor)
            }
            ContractionError::IndexNotFound { tensor } => {
                write!(f, "Index not found in {} tensor", tensor)
            }
            ContractionError::BatchContractionNotImplemented => {
                write!(f, "Batch contraction not yet implemented")
            }
        }
    }
}

impl std::error::Error for ContractionError {}

/// Prepare contraction data for two tensors that share common indices.
///
/// This internal helper finds common indices and computes the axes to contract
/// together with the resulting non-contracted indices.
pub(crate) fn prepare_contraction<I: IndexLike>(
    indices_a: &[I],
    dims_a: &[usize],
    indices_b: &[I],
    dims_b: &[usize],
) -> Result<ContractionSpec<I>, ContractionError> {
    // Find common indices and their positions.
    // If no common indices exist, this becomes an outer product (empty axes).
    let positions = common_ind_positions_small(indices_a, indices_b);

    let mut axes_a = AxisVec::with_capacity(positions.len());
    let mut axes_b = AxisVec::with_capacity(positions.len());
    for &(pos_a, pos_b) in &positions {
        axes_a.push(pos_a);
        axes_b.push(pos_b);
    }

    // Verify dimensions match
    for &(pos_a, pos_b) in &positions {
        if dims_a[pos_a] != dims_b[pos_b] {
            return Err(ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a: dims_a[pos_a],
                dim_b: dims_b[pos_b],
            });
        }
    }

    let result_indices = build_contraction_result_indices(indices_a, &axes_a, indices_b, &axes_b);

    Ok(ContractionSpec {
        axes_a,
        axes_b,
        result_indices,
    })
}

/// Prepare contraction data for explicit index pairs (like tensordot).
///
/// Unlike `prepare_contraction`, this internal helper takes explicit pairs of
/// indices to contract, allowing contraction of indices with different IDs.
pub(crate) fn prepare_contraction_pairs<I: IndexLike>(
    indices_a: &[I],
    dims_a: &[usize],
    indices_b: &[I],
    dims_b: &[usize],
    pairs: &[(I, I)],
) -> Result<ContractionSpec<I>, ContractionError> {
    if pairs.is_empty() {
        return Err(ContractionError::NoCommonIndices);
    }

    // Check for batch contraction (common indices not in pairs). The explicit
    // pair list identifies axes by full index metadata, not by ID alone.
    let common_positions = common_ind_positions_small(indices_a, indices_b);
    for (pos_a, pos_b) in &common_positions {
        let idx_a = &indices_a[*pos_a];
        let idx_b = &indices_b[*pos_b];
        if !pairs
            .iter()
            .any(|(contracted_idx, _)| contracted_idx == idx_a)
            || !pairs
                .iter()
                .any(|(_, contracted_idx)| contracted_idx == idx_b)
        {
            return Err(ContractionError::BatchContractionNotImplemented);
        }
    }

    // Find positions and validate
    let mut axes_a = AxisVec::with_capacity(pairs.len());
    let mut axes_b = AxisVec::with_capacity(pairs.len());
    let mut contracted_a = bool_flags(indices_a.len());
    let mut contracted_b = bool_flags(indices_b.len());

    for (idx_a, idx_b) in pairs {
        let pos_a = indices_a
            .iter()
            .position(|idx| idx == idx_a)
            .ok_or(ContractionError::IndexNotFound { tensor: "self" })?;

        let pos_b = indices_b
            .iter()
            .position(|idx| idx == idx_b)
            .ok_or(ContractionError::IndexNotFound { tensor: "other" })?;

        // Verify dimensions match
        if dims_a[pos_a] != dims_b[pos_b] {
            return Err(ContractionError::DimensionMismatch {
                pos_a,
                pos_b,
                dim_a: dims_a[pos_a],
                dim_b: dims_b[pos_b],
            });
        }

        // Check for duplicate axes
        if contracted_a[pos_a] {
            return Err(ContractionError::DuplicateAxis {
                tensor: "self",
                pos: pos_a,
            });
        }
        if contracted_b[pos_b] {
            return Err(ContractionError::DuplicateAxis {
                tensor: "other",
                pos: pos_b,
            });
        }

        contracted_a[pos_a] = true;
        contracted_b[pos_b] = true;
        axes_a.push(pos_a);
        axes_b.push(pos_b);
    }

    let result_indices = build_contraction_result_indices(indices_a, &axes_a, indices_b, &axes_b);

    Ok(ContractionSpec {
        axes_a,
        axes_b,
        result_indices,
    })
}

fn bool_flags(len: usize) -> BoolVec {
    let mut flags = BoolVec::with_capacity(len);
    flags.resize(len, false);
    flags
}

fn build_contraction_result_indices<I: IndexLike>(
    indices_a: &[I],
    axes_a: &[usize],
    indices_b: &[I],
    axes_b: &[usize],
) -> IndexVec<I> {
    let mut contracted_a = bool_flags(indices_a.len());
    let mut contracted_b = bool_flags(indices_b.len());
    for &axis in axes_a {
        contracted_a[axis] = true;
    }
    for &axis in axes_b {
        contracted_b[axis] = true;
    }

    let result_len = indices_a.len() + indices_b.len() - axes_a.len() - axes_b.len();
    let mut result_indices = IndexVec::with_capacity(result_len);

    for (i, idx) in indices_a.iter().enumerate() {
        if !contracted_a[i] {
            result_indices.push(idx.clone());
        }
    }

    for (i, idx) in indices_b.iter().enumerate() {
        if !contracted_b[i] {
            result_indices.push(idx.clone());
        }
    }

    result_indices
}

#[cfg(test)]
mod tests {
    use super::{common_ind_positions, common_ind_positions_hashed, common_ind_positions_linear};
    use super::{prepare_contraction, prepare_contraction_pairs};
    use crate::index::DefaultIndex as Index;
    use crate::{ConjState, IndexLike};

    /// Directed test index whose `conj()` toggles Ket <-> Bra AND remaps the
    /// id (Ket(id) <-> Bra(id + 1000)). This is the case where id-keyed
    /// contraction pairing silently drops contractable pairs (#615).
    #[derive(Clone, PartialEq, Eq, Hash, Debug)]
    struct DirectedIdIndex {
        id: u64,
        dim: usize,
        state: ConjState,
    }

    impl DirectedIdIndex {
        fn new(id: u64, dim: usize, state: ConjState) -> Self {
            Self { id, dim, state }
        }
    }

    impl IndexLike for DirectedIdIndex {
        type Id = u64;

        fn id(&self) -> &u64 {
            &self.id
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn conj_state(&self) -> ConjState {
            self.state
        }

        fn conj(&self) -> Self {
            let state = match self.state {
                ConjState::Undirected => ConjState::Undirected,
                ConjState::Ket => ConjState::Bra,
                ConjState::Bra => ConjState::Ket,
            };
            let id = match (self.state, state) {
                (ConjState::Ket, ConjState::Bra) => self.id + 1000,
                (ConjState::Bra, ConjState::Ket) => self.id - 1000,
                _ => self.id,
            };
            Self {
                id,
                dim: self.dim,
                state,
            }
        }

        fn sim(&self) -> Self {
            Self {
                id: self.id + 10000,
                ..self.clone()
            }
        }

        fn create_dummy_link_pair() -> (Self, Self) {
            (
                Self::new(0, 1, ConjState::Undirected),
                Self::new(0, 1, ConjState::Undirected),
            )
        }

        fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
            anyhow::ensure!(
                !indices.is_empty(),
                "product_link requires at least one index"
            );
            let dim = indices.iter().try_fold(1usize, |acc, idx| {
                acc.checked_mul(idx.dim)
                    .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
            })?;
            Ok(Self::new(9999, dim, ConjState::Undirected))
        }
    }

    #[test]
    fn hashed_and_linear_agree_on_directed_conj_id_change() {
        // a: Ket ids 0,1,2; b: Bra partners 1000,1002 plus an unrelated Ket.
        // Only (0,0) and (2,2) are contractable; the id-keyed hashed path
        // would drop both because no b shares a's id.
        let a = vec![
            DirectedIdIndex::new(0, 2, ConjState::Ket),
            DirectedIdIndex::new(1, 2, ConjState::Ket),
            DirectedIdIndex::new(2, 2, ConjState::Ket),
        ];
        let b = vec![
            DirectedIdIndex::new(1000, 2, ConjState::Bra),
            DirectedIdIndex::new(5, 2, ConjState::Ket),
            DirectedIdIndex::new(1002, 2, ConjState::Bra),
        ];

        let linear = common_ind_positions_linear(&a, &b);
        assert_eq!(linear.as_slice(), &[(0, 0), (2, 2)]);
        let hashed = common_ind_positions_hashed(&a, &b);
        assert_eq!(hashed, linear);
    }

    #[test]
    fn hashed_and_linear_agree_on_same_id_different_metadata() {
        // Same id, different prime level: not contractable; both paths must
        // reject the pair rather than pairing on id alone.
        let i = Index::new_dyn(2);
        let i_prime = i.prime();
        let j = Index::new_dyn(2);

        let a = vec![i.clone(), j.clone()];
        let b = vec![i_prime.clone(), j.clone()];
        let linear = common_ind_positions_linear(&a, &b);
        let hashed = common_ind_positions_hashed(&a, &b);
        assert_eq!(linear.as_slice(), &[(1, 1)]);
        assert_eq!(hashed, linear);
    }

    #[test]
    fn common_ind_positions_dispatch_hash_path_matches_linear_reference() {
        // scan_work = 9*9 = 81 > LINEAR_CONTRACTION_SCAN_LIMIT (64) so the
        // public entry point dispatches to the hashed path.
        let lhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
        let shared = lhs[7].clone();
        let mut rhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
        rhs[5] = shared.clone();

        let dispatched = common_ind_positions(&lhs, &rhs);
        let reference = common_ind_positions_linear(&lhs, &rhs).into_vec();
        assert_eq!(dispatched, vec![(7, 5)]);
        assert_eq!(dispatched, reference);
    }

    #[test]
    fn prepare_contraction_pairs_selects_exact_same_id_prime_index() {
        let i = Index::new_dyn(2);
        let i_prime = i.prime();
        let spec = prepare_contraction_pairs(
            &[i.clone(), i_prime.clone()],
            &[2, 2],
            std::slice::from_ref(&i_prime),
            &[2],
            &[(i_prime.clone(), i_prime.clone())],
        )
        .unwrap();

        assert_eq!(spec.axes_a.as_slice(), &[1]);
        assert_eq!(spec.axes_b.as_slice(), &[0]);
        assert_eq!(spec.result_indices.as_slice(), &[i]);
    }

    #[test]
    fn prepare_contraction_large_rank_uses_hash_fallback_semantics() {
        let mut lhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
        let shared = lhs[7].clone();
        let mut rhs: Vec<_> = (0..9).map(|_| Index::new_dyn(2)).collect();
        rhs[5] = shared;

        let lhs_dims = vec![2; lhs.len()];
        let rhs_dims = vec![2; rhs.len()];
        let spec = prepare_contraction(&lhs, &lhs_dims, &rhs, &rhs_dims).unwrap();

        assert_eq!(spec.axes_a.as_slice(), &[7]);
        assert_eq!(spec.axes_b.as_slice(), &[5]);
        assert_eq!(spec.result_indices.len(), lhs.len() + rhs.len() - 2);

        lhs.remove(7);
        rhs.remove(5);
        let mut expected = lhs;
        expected.extend(rhs);
        assert_eq!(spec.result_indices.as_slice(), expected.as_slice());
    }
}
