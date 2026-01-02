use std::sync::atomic::{AtomicU64, Ordering};
use crate::tagset::TagSet;

/// Runtime ID for ITensors-like dynamic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u64);

/// Trait for symmetry information (quantum number space).
///
/// This corresponds to the `T` parameter in ITensors.jl's `Index{T}`,
/// where `T = Int` for no symmetry and `T = QNBlocks` for quantum numbers.
pub trait Symmetry: Clone + PartialEq + Eq + std::hash::Hash {
    /// Return the total dimension of the space.
    ///
    /// For no symmetry, this is just the dimension.
    /// For quantum number spaces, this is the sum of all block dimensions.
    fn total_dim(&self) -> usize;
}

/// No symmetry space (corresponds to ITensors.jl's `Index{Int}`).
///
/// This represents a simple index with no quantum number structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NoSymmSpace {
    dim: usize,
}

impl NoSymmSpace {
    /// Create a new no-symmetry space with the given dimension.
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// Get the dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl Symmetry for NoSymmSpace {
    fn total_dim(&self) -> usize {
        self.dim
    }
}

/// Index with generic identity type `Id` and symmetry type `Symm`.
///
/// - `Id = DynId` (or `u64`) for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Symm = NoSymmSpace` for no symmetry (default, corresponds to `Index{Int}` in ITensors.jl)
/// - `Symm = QNSpace` (future) for quantum number spaces (corresponds to `Index{QNBlocks}` in ITensors.jl)
/// - `Tags = DefaultTagSet` for tags (default, max 4 tags, each max 16 characters)
///
/// **Equality**: Two `Index` values are considered equal if and only if their `id` fields match.
/// Tags are not used for equality comparison.
#[derive(Debug, Clone)]
pub struct Index<Id, Symm = NoSymmSpace, const MAX_TAGS: usize = 4, const MAX_TAG_LEN: usize = 16> {
    pub id: Id,
    pub symm: Symm,
    pub tags: TagSet<MAX_TAGS, MAX_TAG_LEN>,
}

impl<Id, Symm: Symmetry, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN> {
    /// Create a new index with the given identity and symmetry.
    pub fn new(id: Id, symm: Symm) -> Self {
        Self {
            id,
            symm,
            tags: TagSet::new(),
        }
    }

    /// Create a new index with the given identity, symmetry, and tags.
    pub fn new_with_tags(id: Id, symm: Symm, tags: TagSet<MAX_TAGS, MAX_TAG_LEN>) -> Self {
        Self { id, symm, tags }
    }

    /// Get the total dimension (size) of the index.
    ///
    /// This is computed from the symmetry information.
    pub fn size(&self) -> usize {
        self.symm.total_dim()
    }

    /// Get a reference to the tags.
    pub fn tags(&self) -> &TagSet<MAX_TAGS, MAX_TAG_LEN> {
        &self.tags
    }

    /// Get a mutable reference to the tags.
    pub fn tags_mut(&mut self) -> &mut TagSet<MAX_TAGS, MAX_TAG_LEN> {
        &mut self.tags
    }
}

impl<Id, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Index<Id, NoSymmSpace, MAX_TAGS, MAX_TAG_LEN> {
    /// Create a new index with no symmetry from dimension.
    ///
    /// This is a convenience constructor for the common case of no symmetry.
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags: TagSet::new(),
        }
    }

    /// Create a new index with no symmetry from dimension and tags.
    pub fn new_with_size_and_tags(id: Id, size: usize, tags: TagSet<MAX_TAGS, MAX_TAG_LEN>) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags,
        }
    }
}

impl<const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Index<DynId, NoSymmSpace, MAX_TAGS, MAX_TAG_LEN> {
    /// Create a new index with a generated dynamic ID and no symmetry.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
            tags: TagSet::new(),
        }
    }

    /// Create a new index with a generated dynamic ID, no symmetry, and tags.
    pub fn new_dyn_with_tags(size: usize, tags: TagSet<MAX_TAGS, MAX_TAG_LEN>) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
            tags,
        }
    }
}

// Equality and Hash implementations: only compare by `id`
impl<Id: PartialEq, Symm, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> PartialEq for Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Id: Eq, Symm, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Eq for Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN> {}

impl<Id: std::hash::Hash, Symm, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> std::hash::Hash for Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// Copy implementation: Index is Copy when Id, Symm, and TagSet are all Copy
impl<Id: Copy, Symm: Copy, const MAX_TAGS: usize, const MAX_TAG_LEN: usize> Copy 
    for Index<Id, Symm, MAX_TAGS, MAX_TAG_LEN> 
where
    TagSet<MAX_TAGS, MAX_TAG_LEN>: Copy,
{}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique ID for dynamic indices (thread-safe).
pub fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// Default Index type with default tag capacity (max 4 tags, each max 16 characters).
pub type DefaultIndex<Id, Symm = NoSymmSpace> = Index<Id, Symm, 4, 16>;
