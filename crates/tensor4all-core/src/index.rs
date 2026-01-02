use std::cell::RefCell;
use crate::tagset::DefaultTagSet;
use rand::Rng;

/// Runtime ID for ITensors-like dynamic identity.
///
/// Uses UInt128 for extremely low collision probability (see design.md for analysis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u128);

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

/// Index with generic identity type `Id`, symmetry type `Symm`, and tag type `Tags`.
///
/// - `Id = DynId` for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
/// - `Symm = NoSymmSpace` for no symmetry (default, corresponds to `Index{Int}` in ITensors.jl)
/// - `Symm = QNSpace` (future) for quantum number spaces (corresponds to `Index{QNBlocks}` in ITensors.jl)
/// - `Tags = DefaultTagSet` for tags (default, max 4 tags, each max 16 characters)
///
/// **Equality**: Two `Index` values are considered equal if and only if their `id` fields match.
/// Tags are not used for equality comparison.
#[derive(Debug, Clone)]
pub struct Index<Id, Symm = NoSymmSpace, Tags = DefaultTagSet> {
    pub id: Id,
    pub symm: Symm,
    pub tags: Tags,
}

impl<Id, Symm: Symmetry, Tags> Index<Id, Symm, Tags> 
where
    Tags: Default,
{
    /// Create a new index with the given identity and symmetry.
    pub fn new(id: Id, symm: Symm) -> Self {
        Self {
            id,
            symm,
            tags: Tags::default(),
        }
    }

    /// Create a new index with the given identity, symmetry, and tags.
    pub fn new_with_tags(id: Id, symm: Symm, tags: Tags) -> Self {
        Self { id, symm, tags }
    }

    /// Get the total dimension (size) of the index.
    ///
    /// This is computed from the symmetry information.
    pub fn size(&self) -> usize {
        self.symm.total_dim()
    }

    /// Get a reference to the tags.
    pub fn tags(&self) -> &Tags {
        &self.tags
    }

    /// Get a mutable reference to the tags.
    pub fn tags_mut(&mut self) -> &mut Tags {
        &mut self.tags
    }
}

impl<Id, Tags> Index<Id, NoSymmSpace, Tags>
where
    Tags: Default,
{
    /// Create a new index with no symmetry from dimension.
    ///
    /// This is a convenience constructor for the common case of no symmetry.
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags: Tags::default(),
        }
    }

    /// Create a new index with no symmetry from dimension and tags.
    pub fn new_with_size_and_tags(id: Id, size: usize, tags: Tags) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
            tags,
        }
    }
}

impl<Tags> Index<DynId, NoSymmSpace, Tags>
where
    Tags: Default,
{
    /// Create a new index with a generated dynamic ID and no symmetry.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
            tags: Tags::default(),
        }
    }

    /// Create a new index with a generated dynamic ID, no symmetry, and tags.
    pub fn new_dyn_with_tags(size: usize, tags: Tags) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
            tags,
        }
    }
}

// Equality and Hash implementations: only compare by `id`
impl<Id: PartialEq, Symm, Tags> PartialEq for Index<Id, Symm, Tags> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<Id: Eq, Symm, Tags> Eq for Index<Id, Symm, Tags> {}

impl<Id: std::hash::Hash, Symm, Tags> std::hash::Hash for Index<Id, Symm, Tags> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

// Copy implementation: Index is Copy when Id, Symm, and Tags are all Copy
impl<Id: Copy, Symm: Copy, Tags: Copy> Copy for Index<Id, Symm, Tags> {}

thread_local! {
    /// Thread-local random number generator for ID generation.
    ///
    /// Each thread has its own RNG, similar to ITensors.jl's task-local RNG.
    /// This provides thread-safe ID generation without global synchronization.
    ///
    /// **Seed initialization**: Each thread's RNG is automatically seeded with
    /// cryptographically secure random data from the OS (via `getrandom` crate)
    /// when first accessed. This ensures different threads use different seeds
    /// and produce independent random sequences.
    ///
    /// **Dynamic thread creation**: When new threads are created (e.g., thread pool
    /// resizing, new tasks), each new thread gets its own RNG with a fresh seed.
    /// Existing threads continue using their existing RNG instances. This ensures
    /// that IDs generated in different threads remain unique even when the thread
    /// count changes dynamically.
    ///
    /// **Thread reuse**: If a thread is reused (e.g., in a thread pool), it continues
    /// using the same RNG instance, maintaining the random sequence from where it
    /// left off. This is safe because the RNG state is thread-local and independent.
    static ID_RNG: RefCell<rand::rngs::ThreadRng> = RefCell::new(rand::thread_rng());
}

/// Generate a unique random ID for dynamic indices (thread-safe).
///
/// Uses thread-local random number generator to generate UInt128 IDs,
/// providing extremely low collision probability (see design.md for analysis).
pub fn generate_id() -> u128 {
    ID_RNG.with(|rng| rng.borrow_mut().gen())
}

/// Default Index type with default tag capacity (max 4 tags, each max 16 characters).
pub type DefaultIndex<Id, Symm = NoSymmSpace> = Index<Id, Symm, DefaultTagSet>;
