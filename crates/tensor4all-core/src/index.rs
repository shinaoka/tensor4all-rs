use std::cell::RefCell;
use crate::tagset::{DefaultTagSet, TagSetLike, TagSetError};
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

impl<Id, Symm, Tags> Index<Id, Symm, Tags>
where
    Id: From<DynId>,
    Symm: From<NoSymmSpace>,
    Tags: Default + TagSetLike,
{
    /// Create a new bond index with "Link" tag (for SVD, QR, etc.).
    ///
    /// This is a convenience method for creating bond indices commonly used in tensor
    /// decompositions like SVD and QR factorization. The ID is automatically generated,
    /// and the symmetry type is converted from `NoSymmSpace`.
    ///
    /// # Arguments
    /// * `size` - Dimension of the bond index
    ///
    /// # Returns
    /// A new index with "Link" tag and the specified size, or an error if the tag cannot be added.
    pub fn new_link(size: usize) -> Result<Self, TagSetError> {
        let mut tags = Tags::default();
        tags.add_tag("Link")?;
        Ok(Self {
            id: DynId(generate_id()).into(),
            symm: NoSymmSpace::new(size).into(),
            tags,
        })
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

impl<Tags> Index<DynId, NoSymmSpace, Tags>
where
    Tags: Default + TagSetLike,
{
    /// Create a new index with a generated dynamic ID, no symmetry, and a single tag.
    ///
    /// # Arguments
    /// * `size` - Dimension of the index
    /// * `tag` - Tag to add to the index
    ///
    /// # Returns
    /// A new index with the specified size and tag, or an error if the tag cannot be added.
    pub fn new_dyn_with_tag(size: usize, tag: &str) -> Result<Self, TagSetError> {
        let mut tags = Tags::default();
        tags.add_tag(tag)?;
        Ok(Self {
            id: DynId(generate_id()),
            symm: NoSymmSpace::new(size),
            tags,
        })
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
pub(crate) fn generate_id() -> u128 {
    ID_RNG.with(|rng| rng.borrow_mut().gen())
}

/// Default Index type with default tag capacity (max 4 tags, each max 16 characters).
pub type DefaultIndex<Id, Symm = NoSymmSpace> = Index<Id, Symm, DefaultTagSet>;

#[cfg(test)]
mod tests {
    use super::generate_id;
    use std::thread;
    use std::collections::HashSet;

    #[test]
    fn test_id_generation() {
        let id1 = generate_id();
        let id2 = generate_id();
        let id3 = generate_id();
        
        // IDs should be unique (random generation, not sequential)
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
        
        // IDs should be non-zero (very high probability with u128)
        assert_ne!(id1, 0);
        assert_ne!(id2, 0);
        assert_ne!(id3, 0);
    }

    #[test]
    fn test_thread_local_rng_different_seeds() {
        // Test that different threads produce different ID sequences
        // This verifies that each thread gets a different seed
        const NUM_THREADS: usize = 4;
        const IDS_PER_THREAD: usize = 100;
        
        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|_| {
                thread::spawn(|| {
                    let mut thread_ids = Vec::new();
                    for _ in 0..IDS_PER_THREAD {
                        thread_ids.push(generate_id());
                    }
                    thread_ids
                })
            })
            .collect();
        
        let mut all_ids = HashSet::new();
        let mut thread_sets = Vec::new();
        
        for handle in handles {
            let thread_ids = handle.join().unwrap();
            let thread_set: HashSet<_> = thread_ids.iter().cloned().collect();
            thread_sets.push(thread_set.clone());
            all_ids.extend(thread_ids);
        }
        
        // All IDs should be unique across all threads
        assert_eq!(all_ids.len(), NUM_THREADS * IDS_PER_THREAD, 
                   "All IDs should be unique across threads");
        
        // Check that different threads produce different sequences
        // (with extremely high probability if seeds are different)
        for i in 0..thread_sets.len() {
            for j in (i+1)..thread_sets.len() {
                let intersection: Vec<_> = thread_sets[i].intersection(&thread_sets[j]).collect();
                assert_eq!(intersection.len(), 0,
                           "Thread {} and {} should produce different ID sequences (different seeds)",
                           i, j);
            }
        }
    }

    #[test]
    fn test_thread_count_changes() {
        // Test behavior when thread count changes dynamically
        // This simulates scenarios like thread pool resizing or new threads being spawned
        
        // Phase 1: Generate IDs in initial threads
        const INITIAL_THREADS: usize = 2;
        const IDS_PER_PHASE: usize = 50;
        
        let mut all_ids = HashSet::new();
        let mut phase1_ids = HashSet::new();
        
        // Phase 1: Initial threads
        let handles1: Vec<_> = (0..INITIAL_THREADS)
            .map(|_| {
                thread::spawn(|| {
                    let mut thread_ids = Vec::new();
                    for _ in 0..IDS_PER_PHASE {
                        thread_ids.push(generate_id());
                    }
                    thread_ids
                })
            })
            .collect();
        
        for handle in handles1 {
            let thread_ids = handle.join().unwrap();
            for id in &thread_ids {
                all_ids.insert(*id);
                phase1_ids.insert(*id);
            }
        }
        
        assert_eq!(phase1_ids.len(), INITIAL_THREADS * IDS_PER_PHASE);
        
        // Phase 2: Spawn new threads (simulating thread count increase)
        const NEW_THREADS: usize = 3;
        let handles2: Vec<_> = (0..NEW_THREADS)
            .map(|_| {
                thread::spawn(|| {
                    let mut thread_ids = Vec::new();
                    for _ in 0..IDS_PER_PHASE {
                        thread_ids.push(generate_id());
                    }
                    thread_ids
                })
            })
            .collect();
        
        let mut phase2_ids = HashSet::new();
        for handle in handles2 {
            let thread_ids = handle.join().unwrap();
            for id in &thread_ids {
                all_ids.insert(*id);
                phase2_ids.insert(*id);
            }
        }
        
        assert_eq!(phase2_ids.len(), NEW_THREADS * IDS_PER_PHASE);
        
        // All IDs should still be unique across both phases
        assert_eq!(all_ids.len(), (INITIAL_THREADS + NEW_THREADS) * IDS_PER_PHASE,
                   "All IDs should be unique even when thread count changes");
        
        // Phase 1 and Phase 2 IDs should not overlap
        let intersection: Vec<_> = phase1_ids.intersection(&phase2_ids).collect();
        assert_eq!(intersection.len(), 0,
                   "IDs from different thread phases should not overlap");
        
        // Phase 3: Reuse a thread (simulating thread pool reuse)
        // When a thread is reused, it continues using its existing RNG
        let handle3 = thread::spawn(|| {
            let mut thread_ids = Vec::new();
            for _ in 0..IDS_PER_PHASE {
                thread_ids.push(generate_id());
            }
            thread_ids
        });
        
        let phase3_ids: HashSet<_> = handle3.join().unwrap().into_iter().collect();
        
        // Phase 3 IDs should be unique (new thread = new seed)
        assert_eq!(phase3_ids.len(), IDS_PER_PHASE);
        
        // Phase 3 should not overlap with previous phases
        let intersection_13: Vec<_> = phase1_ids.intersection(&phase3_ids).collect();
        let intersection_23: Vec<_> = phase2_ids.intersection(&phase3_ids).collect();
        assert_eq!(intersection_13.len(), 0, "Phase 1 and Phase 3 should not overlap");
        assert_eq!(intersection_23.len(), 0, "Phase 2 and Phase 3 should not overlap");
    }
}
