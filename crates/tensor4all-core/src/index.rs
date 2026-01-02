use std::sync::atomic::{AtomicU64, Ordering};

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
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index<Id, Symm = NoSymmSpace> {
    pub id: Id,
    pub symm: Symm,
}

impl<Id, Symm: Symmetry> Index<Id, Symm> {
    /// Create a new index with the given identity and symmetry.
    pub fn new(id: Id, symm: Symm) -> Self {
        Self { id, symm }
    }

    /// Get the total dimension (size) of the index.
    ///
    /// This is computed from the symmetry information.
    pub fn size(&self) -> usize {
        self.symm.total_dim()
    }
}

impl<Id> Index<Id, NoSymmSpace> {
    /// Create a new index with no symmetry from dimension.
    ///
    /// This is a convenience constructor for the common case of no symmetry.
    pub fn new_with_size(id: Id, size: usize) -> Self {
        Self {
            id,
            symm: NoSymmSpace::new(size),
        }
    }
}

impl Index<DynId, NoSymmSpace> {
    /// Create a new index with a generated dynamic ID and no symmetry.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
        }
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique ID for dynamic indices (thread-safe).
pub fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}
