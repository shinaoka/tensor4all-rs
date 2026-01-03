use tensor4all_core::index::{Index, Symmetry};
use tensor4all_core::DefaultTagSet;

/// Physical Indices manager for Tree Tensor Networks.
///
/// Manages physical indices organized by site (node).
/// Each site can have multiple physical indices, and the order of indices
/// within a site is preserved (index order takes priority).
///
/// Also maintains a tensor integer ID for each site.
/// A single site maps to at most one tensor ID, but multiple sites can map to the same tensor ID
/// (i.e., one tensor can own multiple site indices).
///
/// Two `PhysicalIndices` are considered equal if and only if their sorted flattened index lists match.
#[derive(Debug, Clone)]
pub struct PhysicalIndices<Id, Symm = tensor4all_core::index::NoSymmSpace, Tags = DefaultTagSet> {
    /// Physical indices organized by site.
    /// `physical_indices[i]` contains the list of physical indices for site `i`.
    /// The order of indices within each site is preserved.
    physical_indices: Vec<Vec<Index<Id, Symm, Tags>>>,
    
    /// Tensor integer ID per site.
    /// `tensor_id_by_site[i]` is the tensor ID associated with site `i` (or `None` if not set yet).
    tensor_id_by_site: Vec<Option<usize>>,
    
    /// Flattened list of all physical indices in the order they appear in tensors (unsorted).
    /// This preserves the original order of indices as they appear in the tensor network.
    unsorted_indices: Vec<Index<Id, Symm, Tags>>,
    
    /// Flattened list of all physical indices sorted by ID.
    /// Used for comparison: two `PhysicalIndices` are equal if and only if their `sorted_indices` match.
    sorted_indices: Vec<Index<Id, Symm, Tags>>,
}

impl<Id, Symm, Tags> PhysicalIndices<Id, Symm, Tags>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Create a new empty PhysicalIndices manager.
    pub fn new() -> Self {
        Self {
            physical_indices: Vec::new(),
            tensor_id_by_site: Vec::new(),
            unsorted_indices: Vec::new(),
            sorted_indices: Vec::new(),
        }
    }

    /// Create a new PhysicalIndices manager with the given capacity for sites.
    pub fn with_capacity(sites: usize) -> Self {
        Self {
            physical_indices: Vec::with_capacity(sites),
            tensor_id_by_site: Vec::with_capacity(sites),
            unsorted_indices: Vec::new(),
            sorted_indices: Vec::new(),
        }
    }

    /// Update the flattened index lists from the current physical_indices.
    ///
    /// This should be called whenever physical_indices are modified to keep
    /// the flattened lists in sync.
    fn update_flattened_indices(&mut self)
    where
        Id: Ord,
    {
        // Flatten all indices preserving the order (unsorted)
        self.unsorted_indices.clear();
        for site_indices in &self.physical_indices {
            self.unsorted_indices.extend(site_indices.iter().cloned());
        }

        // Create sorted version
        self.sorted_indices = self.unsorted_indices.clone();
        self.sorted_indices.sort_by(|a, b| a.id.cmp(&b.id));
    }

    /// Add physical indices to a site and bind the site to a tensor ID.
    ///
    /// If the site already has a tensor ID set, it must match `tensor_id`.
    /// The new indices are appended in order.
    ///
    /// # Arguments
    /// * `site_index` - The site index (0-based). If the site doesn't exist, it will be created.
    /// * `indices` - The physical indices for this site
    /// * `tensor_id` - The tensor integer ID this site belongs to
    ///
    pub fn add_site_indices(
        &mut self,
        site_index: usize,
        indices: Vec<Index<Id, Symm, Tags>>,
        tensor_id: usize,
    )
    where
        Id: Ord,
    {
        // Ensure we have enough sites
        while self.physical_indices.len() <= site_index {
            self.physical_indices.push(Vec::new());
            self.tensor_id_by_site.push(None);
        }

        // Ensure site -> tensor_id is consistent.
        if let Some(existing) = self.tensor_id_by_site[site_index] {
            assert_eq!(
                existing, tensor_id,
                "site {} is already bound to tensor_id {}, cannot bind to {}",
                site_index, existing, tensor_id
            );
        } else {
            self.tensor_id_by_site[site_index] = Some(tensor_id);
        }

        // Append indices to the site (order is preserved)
        self.physical_indices[site_index].extend(indices);
        
        // Update flattened indices
        self.update_flattened_indices();
    }

    /// Set physical indices for a site, replacing any existing indices.
    ///
    /// # Arguments
    /// * `site_index` - The site index (0-based)
    /// * `indices` - The physical indices for this site
    /// * `tensor_id` - The tensor integer ID this site belongs to
    pub fn set_site_indices(
        &mut self,
        site_index: usize,
        indices: Vec<Index<Id, Symm, Tags>>,
        tensor_id: usize,
    )
    where
        Id: Ord,
    {
        // Ensure we have enough sites
        while self.physical_indices.len() <= site_index {
            self.physical_indices.push(Vec::new());
            self.tensor_id_by_site.push(None);
        }

        // Replace indices and bind site to tensor_id.
        self.physical_indices[site_index] = indices;
        self.tensor_id_by_site[site_index] = Some(tensor_id);
        
        // Update flattened indices
        self.update_flattened_indices();
    }

    /// Get the physical indices for a site.
    ///
    /// Returns `None` if the site doesn't exist.
    pub fn get_site_indices(&self, site_index: usize) -> Option<&[Index<Id, Symm, Tags>]> {
        self.physical_indices.get(site_index).map(|v| v.as_slice())
    }

    /// Get the tensor integer ID for a site.
    ///
    /// Returns `None` if the site doesn't exist.
    pub fn get_site_tensor_id(&self, site_index: usize) -> Option<usize> {
        self.tensor_id_by_site.get(site_index).copied().flatten()
    }

    /// Get the number of sites.
    pub fn num_sites(&self) -> usize {
        self.physical_indices.len()
    }

    /// Get the total number of physical indices across all sites.
    pub fn total_indices(&self) -> usize {
        self.physical_indices.iter().map(|v| v.len()).sum()
    }

    /// Get a reference to all physical indices (organized by site).
    pub fn all_indices(&self) -> &[Vec<Index<Id, Symm, Tags>>] {
        &self.physical_indices
    }

    /// Get a reference to all tensor IDs (organized by site).
    pub fn all_tensor_ids_by_site(&self) -> &[Option<usize>] {
        &self.tensor_id_by_site
    }

    /// Get a reference to the unsorted flattened indices (in tensor order).
    pub fn unsorted_indices(&self) -> &[Index<Id, Symm, Tags>] {
        &self.unsorted_indices
    }

    /// Get a reference to the sorted flattened indices (sorted by ID).
    pub fn sorted_indices(&self) -> &[Index<Id, Symm, Tags>] {
        &self.sorted_indices
    }

    /// Clear all physical indices and tensor IDs.
    pub fn clear(&mut self) {
        self.physical_indices.clear();
        self.tensor_id_by_site.clear();
        self.unsorted_indices.clear();
        self.sorted_indices.clear();
    }

    /// Remove a site and all its physical indices.
    ///
    /// Returns `true` if the site existed and was removed, `false` otherwise.
    pub fn remove_site(&mut self, site_index: usize) -> bool
    where
        Id: Ord,
    {
        if site_index < self.physical_indices.len() {
            self.physical_indices.remove(site_index);
            self.tensor_id_by_site.remove(site_index);
            self.update_flattened_indices();
            true
        } else {
            false
        }
    }
}

impl<Id, Symm, Tags> Default for PhysicalIndices<Id, Symm, Tags>
where
    Id: Clone + std::hash::Hash + Eq,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<Id, Symm, Tags> PartialEq for PhysicalIndices<Id, Symm, Tags>
where
    Id: Clone + std::hash::Hash + Eq + Ord,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
    /// Two `PhysicalIndices` are equal if and only if their sorted flattened index lists match.
    fn eq(&self, other: &Self) -> bool {
        self.sorted_indices == other.sorted_indices
    }
}

impl<Id, Symm, Tags> Eq for PhysicalIndices<Id, Symm, Tags>
where
    Id: Clone + std::hash::Hash + Eq + Ord,
    Symm: Clone + Symmetry,
    Tags: Clone,
{
}

