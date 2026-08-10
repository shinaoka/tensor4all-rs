//! Index types for tensor network operations.
//!
//! This module provides the default index types:
//!
//! - [`DynId`]: Runtime identity (UUID-based unique identifier)
//! - [`TagSet`]: Tag set for metadata (Arc-wrapped for cheap cloning)
//! - [`Index`]: Generic index type parameterized by Id and Tags
//! - [`DynIndex`]: Default index type (`Index<DynId, TagSet>`)
//!
//! The `DynIndex` type implements the [`IndexLike`] trait.
//!
//! **Note**: Symmetry (quantum numbers) is not included in the default implementation.
//! For QSpace-compatible indices with non-Abelian symmetries, use a separate concrete type
//! that implements `IndexLike` directly.

use crate::index_like::IndexLike;
use crate::tagset::{DefaultTagSet as InlineTagSet, TagSetError, TagSetIterator, TagSetLike};
use anyhow::Result;
use rand::Rng;
use std::cell::RefCell;
use std::sync::Arc;

/// Runtime ID for ITensors-like dynamic identity.
///
/// Uses UInt64 for compatibility with ITensors.jl's `IDType = UInt64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DynId(pub u64);

impl DynId {
    /// Return the numeric ID value used for ITensors-compatible serialization.
    ///
    /// Prefer full [`Index`] equality for tensor index identity.
    /// This accessor exists for stable serialization and FFI query paths that
    /// must expose the raw numeric identifier.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynId;
    ///
    /// let id = DynId(42);
    /// assert_eq!(id.value(), 42);
    /// ```
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// Tag set wrapper using `Arc` for efficient cloning.
///
/// This wraps the underlying default tag storage in an `Arc` for cheap cloning
/// (reference count increment only). Tags are immutable and shared across
/// indices with the same tag set. The default storage is string-backed so
/// descriptive tags from language bindings are preserved losslessly.
///
/// # Example
/// ```
/// use tensor4all_core::index::TagSet;
///
/// let tags = TagSet::from_str("Site,Link").unwrap();
/// assert!(tags.has_tag("Site"));
/// assert!(tags.has_tag("Link"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TagSet(Arc<InlineTagSet>);

impl TagSet {
    /// Create an empty tag set.
    pub fn new() -> Self {
        Self(Arc::new(InlineTagSet::new()))
    }

    /// Create a tag set from a comma-separated string.
    #[allow(clippy::should_implement_trait)]
    /// # Errors
    ///
    /// Returns an error when the string cannot be parsed as an index (an
    /// /// invalid-input failure).
    ///
    pub fn from_str(s: &str) -> Result<Self, TagSetError> {
        Ok(Self(Arc::new(InlineTagSet::from_str(s)?)))
    }

    /// Create a tag set from a slice of tag strings.
    ///
    /// Returns an error if any tag contains a comma (reserved as separator in `from_str`).
    ///
    /// # Errors
    ///
    /// Returns an error when the tag string is invalid (an invalid-input
    /// /// failure).
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::index::TagSet;
    ///
    /// let tags = TagSet::from_tags(&["Site", "Link"]).unwrap();
    /// assert!(tags.has_tag("Site"));
    /// assert!(tags.has_tag("Link"));
    /// assert_eq!(tags.len(), 2);
    ///
    /// // Comma in tag is an error
    /// assert!(TagSet::from_tags(&["Site,Link"]).is_err());
    /// ```
    pub fn from_tags(tags: &[&str]) -> Result<Self, TagSetError> {
        let mut inner = InlineTagSet::new();
        for tag in tags {
            if tag.contains(',') {
                return Err(TagSetError::TagContainsComma {
                    tag: (*tag).to_string(),
                });
            }
            inner.add_tag(tag)?;
        }
        Ok(Self(Arc::new(inner)))
    }

    /// Check if a tag is present.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.0.has_tag(tag)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the tag set is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the inner Arc for advanced use.
    pub fn inner(&self) -> &Arc<InlineTagSet> {
        &self.0
    }
}

impl std::ops::Deref for TagSet {
    type Target = InlineTagSet;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl TagSetLike for TagSet {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn capacity(&self) -> usize {
        self.0.capacity()
    }

    fn get(&self, index: usize) -> Option<String> {
        TagSetLike::get(&*self.0, index)
    }

    fn iter(&self) -> TagSetIterator<'_> {
        TagSetLike::iter(&*self.0)
    }

    fn has_tag(&self, tag: &str) -> bool {
        self.0.has_tag(tag)
    }

    fn add_tag(&mut self, tag: &str) -> Result<(), TagSetError> {
        // Arc is immutable, so we need to clone and replace
        let mut inner = self.0.as_ref().clone();
        inner.add_tag(tag)?;
        self.0 = Arc::new(inner);
        Ok(())
    }

    fn remove_tag(&mut self, tag: &str) -> bool {
        // Arc is immutable, so we need to clone and replace
        let mut inner = self.0.as_ref().clone();
        let removed = inner.remove_tag(tag);
        if removed {
            self.0 = Arc::new(inner);
        }
        removed
    }
}

/// Index with generic identity type `Id` and tag type `Tags`.
///
/// - `Id = DynId` for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Tags = TagSet` for tags (default, Arc-wrapped for cheap cloning)
///
/// **Note**: This default implementation does not include symmetry (quantum numbers).
/// For QSpace-compatible indices with non-Abelian symmetries, use a separate concrete type.
///
/// # Memory Layout
/// With default types (`DynId`, `TagSet`):
/// - Size: 32 bytes (8 + 8 + 8 + 8)
/// - Tags are shared via `Arc`, so cloning is cheap (reference count increment only)
///
/// **Equality**: Two `Index` values are considered equal if and only if their `id` and `tags`
/// fields and `plev` match (matching ITensors.jl semantics where equality = id + plev + tags).
///
/// # Example
/// ```
/// use tensor4all_core::index::{Index, DynId, TagSet};
///
/// // Create shared tags once
/// let site_tags = TagSet::from_str("Site").unwrap();
///
/// // Share the same tags across many indices (cheap clone)
/// let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
/// let i2 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
/// // i1.tags and i2.tags point to the same Arc
/// ```
#[derive(Debug, Clone)]
pub struct Index<Id, Tags = TagSet> {
    /// The unique identifier for this index.
    pub id: Id,
    /// The dimension (size) of this index.
    pub dim: usize,
    /// The prime level of this index.
    pub plev: i64,
    /// The tag set associated with this index.
    pub tags: Tags,
}

impl<Id, Tags> Index<Id, Tags>
where
    Tags: Default,
{
    /// Create a new index with the given identity and dimension.
    pub fn new(id: Id, dim: usize) -> Self {
        Self {
            id,
            dim,
            plev: 0,
            tags: Tags::default(),
        }
    }

    /// Create a new index with the given identity, dimension, and tags.
    pub fn new_with_tags(id: Id, dim: usize, tags: Tags) -> Self {
        Self {
            id,
            dim,
            plev: 0,
            tags,
        }
    }

    /// Get the dimension (size) of the index.
    pub fn size(&self) -> usize {
        self.dim
    }

    /// Get a reference to the tags.
    pub fn tags(&self) -> &Tags {
        &self.tags
    }
}

impl<Id, Tags> Index<Id, Tags>
where
    Tags: Default,
{
    /// Create a new index from dimension (convenience constructor).
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            dim: size,
            plev: 0,
            tags: Tags::default(),
        }
    }

    /// Create a new index from dimension and tags.
    pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self {
        Self {
            id,
            dim: size,
            plev: 0,
            tags,
        }
    }
}

// Constructors for Index with TagSet (default)
impl Index<DynId, TagSet> {
    /// Create a new index with a generated dynamic ID and no tags.
    ///
    /// This is the most common way to create indices. Each call generates
    /// a unique random ID, so two calls to `new_dyn` with the same dimension
    /// produce non-equal indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let j = DynIndex::new_dyn(4);
    ///
    /// assert_eq!(i.dim(), 4);
    /// assert_ne!(i, j);  // different IDs
    /// assert!(i.is_contractable(&i));  // same index is contractable with itself
    /// assert!(!i.is_contractable(&j)); // different IDs
    /// ```
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            dim: size,
            plev: 0,
            tags: TagSet::new(),
        }
    }

    /// Create a new index with a generated dynamic ID and shared tags.
    ///
    /// This is the most efficient way to create many indices with the same tags.
    /// The `Arc` is cloned (reference count increment only), not the underlying data.
    ///
    /// # Example
    /// ```
    /// use tensor4all_core::index::{Index, DynId, TagSet};
    ///
    /// let site_tags = TagSet::from_str("Site").unwrap();
    /// let i1 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    /// let i2 = Index::<DynId>::new_dyn_with_tags(2, site_tags.clone());
    /// ```
    pub fn new_dyn_with_tags(size: usize, tags: TagSet) -> Self {
        Self {
            id: DynId(generate_id()),
            dim: size,
            plev: 0,
            tags,
        }
    }

    /// Create a new index with a generated dynamic ID and a single tag.
    ///
    /// This creates a new `TagSet` with the given tag.
    /// For sharing the same tag across many indices, create the `TagSet`
    /// once and use `new_dyn_with_tags` instead.
    ///
    /// # Errors
    ///
    /// Returns an error when the tag is invalid (an invalid-input failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_core::TagSetLike;
    ///
    /// let site = DynIndex::new_dyn_with_tag(2, "Site").unwrap();
    /// assert!(site.tags().has_tag("Site"));
    /// ```
    pub fn new_dyn_with_tag(size: usize, tag: &str) -> Result<Self, TagSetError> {
        Ok(Self {
            id: DynId(generate_id()),
            dim: size,
            plev: 0,
            tags: TagSet::from_str(tag)?,
        })
    }

    /// Create a new bond index with "Link" tag (for SVD, QR, etc.).
    ///
    /// This is a convenience method for creating bond indices commonly used in tensor
    /// decompositions like SVD and QR factorization.
    ///
    /// # Errors
    ///
    /// Returns an error when the link dimension is invalid (a shape mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::DynIndex;
    /// use tensor4all_core::TagSetLike;
    ///
    /// let link = DynIndex::new_link(4).unwrap();
    /// assert!(link.tags().has_tag("Link"));
    /// ```
    pub fn new_link(size: usize) -> Result<Self, TagSetError> {
        Self::new_dyn_with_tag(size, "Link")
    }
}

// Equality and Hash implementations: compare by `id`, `tags`, and `plev`
// (matching ITensors.jl semantics where equality = id + plev + tags)
impl<Id: PartialEq, Tags: PartialEq> PartialEq for Index<Id, Tags> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.tags == other.tags && self.plev == other.plev
    }
}

impl<Id: Eq, Tags: Eq> Eq for Index<Id, Tags> {}

impl<Id: std::hash::Hash, Tags: std::hash::Hash> std::hash::Hash for Index<Id, Tags> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.tags.hash(state);
        self.plev.hash(state);
    }
}

// Copy implementation: Index is Copy when Id and Tags are both Copy
impl<Id: Copy, Tags: Copy> Copy for Index<Id, Tags> {}

thread_local! {
    /// Thread-local random number generator for ID generation.
    ///
    /// Each thread has its own RNG, similar to ITensors.jl's task-local RNG.
    /// This provides thread-safe ID generation without global synchronization.
    static ID_RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::rng());
}

/// Generate a unique random ID for dynamic indices (thread-safe).
///
/// Uses thread-local random number generator to generate UInt64 IDs,
/// compatible with ITensors.jl's `IDType = UInt64`.
pub(crate) fn generate_id() -> u64 {
    ID_RNG.with(|rng| rng.borrow_mut().random())
}

/// Default Index type alias (same as `Index<Id>` with default tags).
///
/// This is provided for convenience and compatibility.
pub type DefaultIndex<Id> = Index<Id, TagSet>;

/// Type alias for backwards compatibility.
pub type DefaultTagSet = TagSet;

// ============================================================================
// DynIndex: Default index type with IndexLike implementation
// ============================================================================

/// Type alias for the default index type with IndexLike bound.
///
/// `DynIndex` uses:
/// - `DynId`: Dynamic identity (UUID-based unique identifier)
/// - `TagSet`: Default tag set for metadata
///
/// This is the recommended index type for most tensor network applications.
/// It does not include symmetry (quantum numbers); for QSpace-compatible indices,
/// use a separate concrete type that implements `IndexLike` directly.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_core::index_like::IndexLike;
///
/// // Create a dynamic index with dimension 4
/// let idx = DynIndex::new_dyn(4);
/// assert_eq!(idx.dim(), 4);
/// assert_eq!(idx.plev(), 0);
///
/// // Prime level manipulation
/// let primed = idx.prime();
/// assert_eq!(primed.plev(), 1);
///
/// let noprime = primed.noprime();
/// assert_eq!(noprime.plev(), 0);
///
/// // Bond index creation (for SVD/QR)
/// let bond = DynIndex::new_bond(8).unwrap();
/// assert_eq!(bond.dim(), 8);
/// ```
pub type DynIndex = Index<DynId, TagSet>;

impl IndexLike for DynIndex {
    type Id = DynId;

    fn id(&self) -> &Self::Id {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn plev(&self) -> i64 {
        self.plev
    }

    fn conj_state(&self) -> crate::ConjState {
        // Default indices are undirected (ITensors.jl-like behavior)
        crate::ConjState::Undirected
    }

    fn conj(&self) -> Self {
        // For undirected indices, conj() is a no-op
        self.clone()
    }

    fn sim(&self) -> Self {
        Index {
            id: DynId(generate_id()),
            dim: self.dim,
            plev: self.plev,
            tags: self.tags.clone(),
        }
    }

    fn create_dummy_link_pair() -> (Self, Self) {
        let id = DynId(generate_id());
        let idx1 = Index {
            id,
            dim: 1,
            plev: 0,
            tags: TagSet::default(),
        };
        let idx2 = Index {
            id,
            dim: 1,
            plev: 0,
            tags: TagSet::default(),
        };
        (idx1, idx2)
    }

    fn product_link(indices: &[Self]) -> Result<Self> {
        anyhow::ensure!(
            !indices.is_empty(),
            "product_link requires at least one index"
        );
        let dim = indices.iter().try_fold(1usize, |acc, idx| {
            acc.checked_mul(idx.dim())
                .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
        })?;
        DynIndex::new_bond(dim)
    }
}

impl DynIndex {
    /// Create a new bond index with a fresh identity and the specified dimension.
    ///
    /// This is used by factorization operations (SVD, QR) to create new internal
    /// bond indices connecting the factors.
    ///
    /// # Arguments
    /// * `dim` - The dimension of the new index
    ///
    /// # Returns
    /// A new index with a unique identity and the specified dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when the bond dimension is invalid (a shape mismatch).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike};
    ///
    /// let bond = DynIndex::new_bond(8).unwrap();
    /// assert_eq!(bond.dim(), 8);
    /// ```
    pub fn new_bond(dim: usize) -> Result<Self> {
        Index::new_link(dim).map_err(|e| anyhow::anyhow!("Failed to create bond index: {:?}", e))
    }

    /// Return a copy of this index with its prime level incremented by one.
    ///
    /// Primed indices are commonly used to distinguish bra and ket indices
    /// in MPO operations (e.g., `i` for input, `i'` for output).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// assert_eq!(i.plev(), 0);
    ///
    /// let i_prime = i.prime();
    /// assert_eq!(i_prime.plev(), 1);
    ///
    /// // Primed and unprimed indices have the same ID but are not equal
    /// assert!(i.same_id(&i_prime));
    /// assert_ne!(i, i_prime);
    ///
    /// // Primed indices are not contractable with unprimed ones
    /// assert!(!i.is_contractable(&i_prime));
    /// ```
    pub fn prime(&self) -> Self {
        let mut idx = self.clone();
        idx.plev += 1;
        idx
    }

    /// Return a copy of this index with its prime level reset to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let i2 = i.prime().prime();
    /// assert_eq!(i2.plev(), 2);
    ///
    /// let i0 = i2.noprime();
    /// assert_eq!(i0.plev(), 0);
    /// assert_eq!(i0, i);
    /// ```
    pub fn noprime(&self) -> Self {
        let mut idx = self.clone();
        idx.plev = 0;
        idx
    }

    /// Return a copy of this index with its prime level set explicitly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike};
    ///
    /// let i = DynIndex::new_dyn(4);
    /// let i3 = i.set_plev(3);
    /// assert_eq!(i3.plev(), 3);
    /// ```
    pub fn set_plev(&self, plev: i64) -> Self {
        let mut idx = self.clone();
        idx.plev = plev;
        idx
    }
}

#[cfg(test)]
mod tests;
