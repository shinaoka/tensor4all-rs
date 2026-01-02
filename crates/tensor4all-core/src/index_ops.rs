use crate::index::{Index, DynId, Symmetry, generate_id};

/// Create a similar index with the same space and tags but a new ID.
///
/// This corresponds to ITensors.jl's `sim(i::Index)` function. It creates a new
/// index with the same symmetry space (dimension/QN structure) and tags, but with
/// a freshly generated ID. This is commonly used in tensor decompositions like SVD
/// where you need a new bond index with the same structure.
///
/// # Arguments
/// * `i` - The index to create a similar copy of
///
/// # Returns
/// A new index with the same `symm` and `tags` but a new `id`.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::sim;
///
/// let idx = Index::new_dyn(8);
/// let similar = sim(&idx);
///
/// // Same size and tags, but different ID
/// assert_eq!(similar.size(), idx.size());
/// assert_ne!(similar.id, idx.id);
/// ```
pub fn sim<Id, Symm, Tags>(i: &Index<Id, Symm, Tags>) -> Index<Id, Symm, Tags>
where
    Id: From<DynId>,
    Symm: Clone,
    Tags: Clone,
{
    Index {
        id: DynId(generate_id()).into(),
        symm: i.symm.clone(),
        tags: i.tags.clone(),
    }
}

/// Create a similar index with the same space and tags but a new ID (consumes input).
///
/// This is an owned variant of `sim` that consumes the input index, avoiding
/// unnecessary clones when you already own the index.
///
/// # Arguments
/// * `i` - The index to create a similar copy of (consumed)
///
/// # Returns
/// A new index with the same `symm` and `tags` but a new `id`.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::sim_owned;
///
/// let idx = Index::new_dyn(8);
/// let similar = sim_owned(idx);  // idx is consumed
///
/// // Same size and tags, but different ID
/// assert_eq!(similar.size(), 8);
/// ```
pub fn sim_owned<Id, Symm, Tags>(i: Index<Id, Symm, Tags>) -> Index<Id, Symm, Tags>
where
    Id: From<DynId>,
    Symm: Clone,
    Tags: Clone,
{
    Index {
        id: DynId(generate_id()).into(),
        symm: i.symm,
        tags: i.tags,
    }
}

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
        }
    }
}

impl std::error::Error for ReplaceIndsError {}

/// Replace indices in a collection based on ID matching.
///
/// This corresponds to ITensors.jl's `replaceinds` function. It replaces indices
/// in `indices` that match (by ID) any of the `(old, new)` pairs in `replacements`.
/// The replacement index must have the same symmetry space as the original.
///
/// # Arguments
/// * `indices` - Collection of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices matching `old_index.id` are replaced with `new_index`
///
/// # Returns
/// A new vector with replacements applied, or an error if any replacement has a space mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different symmetry space than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replaceinds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);  // Same size as j
///
/// let indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// let replaced = replaceinds(indices, &replacements).unwrap();
/// assert_eq!(replaced.len(), 3);
/// assert_eq!(replaced[1].id, new_j.id);
/// ```
pub fn replaceinds<Id, Symm, Tags>(
    indices: Vec<Index<Id, Symm, Tags>>,
    replacements: &[(Index<Id, Symm, Tags>, Index<Id, Symm, Tags>)],
) -> Result<Vec<Index<Id, Symm, Tags>>, ReplaceIndsError>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Symmetry + Clone,
    Tags: Clone,
{
    // Build a map from old ID to new index for fast lookup
    let mut replacement_map = std::collections::HashMap::new();
    for (old, new) in replacements {
        // Validate space match
        if old.symm != new.symm {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.symm.total_dim(),
                to_dim: new.symm.total_dim(),
            });
        }
        replacement_map.insert(&old.id, new);
    }

    // Apply replacements
    let mut result = Vec::with_capacity(indices.len());
    for idx in indices {
        if let Some(new_idx) = replacement_map.get(&idx.id) {
            result.push((*new_idx).clone());
        } else {
            result.push(idx);
        }
    }
    Ok(result)
}

/// Replace indices in-place based on ID matching.
///
/// This is an in-place variant of `replaceinds` that modifies the input slice directly.
/// Useful for performance-critical code where you want to avoid allocations.
///
/// # Arguments
/// * `indices` - Mutable slice of indices to modify
/// * `replacements` - Pairs of `(old_index, new_index)` where indices matching `old_index.id` are replaced with `new_index`
///
/// # Returns
/// `Ok(())` on success, or an error if any replacement has a space mismatch.
///
/// # Errors
/// Returns `ReplaceIndsError::SpaceMismatch` if any replacement index has a different symmetry space than the original.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::replaceinds_in_place;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let new_j = Index::new_dyn(3);
///
/// let mut indices = vec![i.clone(), j.clone(), k.clone()];
/// let replacements = vec![(j.clone(), new_j.clone())];
///
/// replaceinds_in_place(&mut indices, &replacements).unwrap();
/// assert_eq!(indices[1].id, new_j.id);
/// ```
pub fn replaceinds_in_place<Id, Symm, Tags>(
    indices: &mut [Index<Id, Symm, Tags>],
    replacements: &[(Index<Id, Symm, Tags>, Index<Id, Symm, Tags>)],
) -> Result<(), ReplaceIndsError>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Symmetry + Clone,
    Tags: Clone,
{
    // Build a map from old ID to new index for fast lookup
    let mut replacement_map = std::collections::HashMap::new();
    for (old, new) in replacements {
        // Validate space match
        if old.symm != new.symm {
            return Err(ReplaceIndsError::SpaceMismatch {
                from_dim: old.symm.total_dim(),
                to_dim: new.symm.total_dim(),
            });
        }
        replacement_map.insert(&old.id, new);
    }

    // Apply replacements in-place
    for idx in indices.iter_mut() {
        if let Some(new_idx) = replacement_map.get(&idx.id) {
            *idx = (*new_idx).clone();
        }
    }
    Ok(())
}

/// Find indices that are unique to the first collection (set difference A \ B).
///
/// Returns indices that appear in `indices_a` but not in `indices_b` (matched by ID).
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
pub fn unique_inds<Id, Symm, Tags>(
    indices_a: &[Index<Id, Symm, Tags>],
    indices_b: &[Index<Id, Symm, Tags>],
) -> Vec<Index<Id, Symm, Tags>>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Clone,
    Tags: Clone,
{
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| &idx.id).collect();
    indices_a
        .iter()
        .filter(|idx| !b_ids.contains(&idx.id))
        .cloned()
        .collect()
}

/// Find indices that are not common between two collections (symmetric difference).
///
/// Returns indices that appear in either `indices_a` or `indices_b` but not in both
/// (matched by ID). This corresponds to ITensors.jl's `noncommoninds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices from both collections that are not common to both.
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
pub fn noncommon_inds<Id, Symm, Tags>(
    indices_a: &[Index<Id, Symm, Tags>],
    indices_b: &[Index<Id, Symm, Tags>],
) -> Vec<Index<Id, Symm, Tags>>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Clone,
    Tags: Clone,
{
    let a_ids: std::collections::HashSet<_> = indices_a.iter().map(|idx| &idx.id).collect();
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| &idx.id).collect();

    let mut result = Vec::new();
    // Add indices from A that are not in B
    result.extend(
        indices_a
            .iter()
            .filter(|idx| !b_ids.contains(&idx.id))
            .cloned(),
    );
    // Add indices from B that are not in A
    result.extend(
        indices_b
            .iter()
            .filter(|idx| !a_ids.contains(&idx.id))
            .cloned(),
    );
    result
}

/// Find the union of two index collections.
///
/// Returns all unique indices from both collections (matched by ID).
/// This corresponds to ITensors.jl's `unioninds` function.
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
pub fn union_inds<Id, Symm, Tags>(
    indices_a: &[Index<Id, Symm, Tags>],
    indices_b: &[Index<Id, Symm, Tags>],
) -> Vec<Index<Id, Symm, Tags>>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Clone,
    Tags: Clone,
{
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();

    for idx in indices_a {
        if seen.insert(&idx.id) {
            result.push(idx.clone());
        }
    }
    for idx in indices_b {
        if seen.insert(&idx.id) {
            result.push(idx.clone());
        }
    }
    result
}

/// Check if a collection contains a specific index (by ID).
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
pub fn hasind<Id, Symm, Tags>(
    indices: &[Index<Id, Symm, Tags>],
    index: &Index<Id, Symm, Tags>,
) -> bool
where
    Id: std::hash::Hash + Eq,
{
    indices.iter().any(|idx| idx.id == index.id)
}

/// Check if a collection contains all of the specified indices (by ID).
///
/// This corresponds to ITensors.jl's `hasinds` function.
///
/// # Arguments
/// * `indices` - Collection of indices to search
/// * `targets` - The indices to look for
///
/// # Returns
/// `true` if all target indices (by ID) are found, `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hasinds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
/// let indices = vec![i.clone(), j.clone(), k.clone()];
///
/// assert!(hasinds(&indices, &[i.clone(), j.clone()]));
/// assert!(!hasinds(&indices, &[i.clone(), Index::new_dyn(5)]));
/// ```
pub fn hasinds<Id, Symm, Tags>(
    indices: &[Index<Id, Symm, Tags>],
    targets: &[Index<Id, Symm, Tags>],
) -> bool
where
    Id: std::hash::Hash + Eq,
{
    let index_ids: std::collections::HashSet<_> = indices.iter().map(|idx| &idx.id).collect();
    targets.iter().all(|target| index_ids.contains(&target.id))
}

/// Check if two collections have any common indices (by ID).
///
/// This corresponds to ITensors.jl's `hascommoninds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// `true` if there is at least one common index (by ID), `false` otherwise.
///
/// # Example
/// ```
/// use tensor4all_core::index::{DefaultIndex as Index, DynId};
/// use tensor4all_core::index_ops::hascommoninds;
///
/// let i = Index::new_dyn(2);
/// let j = Index::new_dyn(3);
/// let k = Index::new_dyn(4);
///
/// let indices_a = vec![i.clone(), j.clone()];
/// let indices_b = vec![j.clone(), k.clone()];
///
/// assert!(hascommoninds(&indices_a, &indices_b));
/// assert!(!hascommoninds(&[i.clone()], &[k.clone()]));
/// ```
pub fn hascommoninds<Id, Symm, Tags>(
    indices_a: &[Index<Id, Symm, Tags>],
    indices_b: &[Index<Id, Symm, Tags>],
) -> bool
where
    Id: std::hash::Hash + Eq,
{
    let b_ids: std::collections::HashSet<_> = indices_b.iter().map(|idx| &idx.id).collect();
    indices_a.iter().any(|idx| b_ids.contains(&idx.id))
}

/// Find common indices between two index collections.
///
/// Returns a vector of indices that appear in both `indices_a` and `indices_b`
/// (set intersection). This is similar to ITensors.jl's `commoninds` function.
///
/// # Arguments
/// * `indices_a` - First collection of indices
/// * `indices_b` - Second collection of indices
///
/// # Returns
/// A vector containing indices that are common to both collections (matched by ID).
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
pub fn common_inds<Id, Symm, Tags>(
    indices_a: &[Index<Id, Symm, Tags>],
    indices_b: &[Index<Id, Symm, Tags>],
) -> Vec<Index<Id, Symm, Tags>>
where
    Id: std::hash::Hash + Eq + Clone,
    Symm: Clone,
    Tags: Clone,
{
    let mut result = Vec::new();
    for idx_a in indices_a {
        if indices_b.iter().any(|idx_b| idx_b.id == idx_a.id) {
            result.push(idx_a.clone());
        }
    }
    result
}

