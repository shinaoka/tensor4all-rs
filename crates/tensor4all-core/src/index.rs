use std::sync::atomic::{AtomicU64, Ordering};

/// Runtime ID for ITensors-like dynamic identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynId(pub u64);

/// Index with generic identity type `Id`.
///
/// - `Id = DynId` (or `u64`) for ITensors-like runtime identity
/// - `Id = ZST marker type` for compile-time-known identity
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index<Id> {
    pub id: Id,
    pub size: usize,
}

impl<Id> Index<Id> {
    pub fn new(id: Id, size: usize) -> Self {
        Self { id, size }
    }
}

impl Index<DynId> {
    /// Create a new index with a generated dynamic ID.
    pub fn new_dyn(size: usize) -> Self {
        Self {
            id: DynId(generate_id()),
            size,
        }
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Generate a unique ID for dynamic indices (thread-safe).
pub fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

