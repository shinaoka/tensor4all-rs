//! IndexLike trait for abstracting index types.
//!
//! This trait allows algorithms to be generic over different index types
//! without needing to know about the internal ID representation.

use std::fmt::Debug;
use std::hash::Hash;

/// Conjugate state (direction) of an index.
///
/// This enum represents whether an index has a direction (bra/ket) or is directionless.
/// The direction is used to determine contractability between indices.
///
/// # QSpace Compatibility
///
/// In QSpace (extern/qspace-v4-pub), index direction is encoded via trailing `*` in `itags`:
/// - **Ket** = ingoing index (QSpace: itag **without** trailing `*`)
/// - **Bra** = outgoing index (QSpace: itag **with** trailing `*`)
///
/// # ITensors.jl Compatibility
///
/// ITensors.jl uses directionless indices by default (convenient for general tensor operations).
/// The `Undirected` variant provides this behavior.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{DynIndex, IndexLike, ConjState};
///
/// let i = DynIndex::new_dyn(4);
/// // Default DynIndex is undirected
/// assert_eq!(i.conj_state(), ConjState::Undirected);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConjState {
    /// Directionless index (ITensors.jl-like default).
    ///
    /// Undirected indices can contract with other undirected indices
    /// only when their full index identity matches and their dimensions match.
    Undirected,
    /// Ket (ingoing) index.
    ///
    /// In QSpace terminology, this corresponds to an index without a trailing `*` in its itag.
    /// Ket indices can only contract with Bra indices (and vice versa).
    Ket,
    /// Bra (outgoing) index.
    ///
    /// In QSpace terminology, this corresponds to an index with a trailing `*` in its itag.
    /// Bra indices can only contract with Ket indices (and vice versa).
    Bra,
}

/// Trait for index-like types that can be used in tensor operations.
///
/// This trait abstracts away the identity mechanism of indices, allowing algorithms
/// to work with any index type that provides equality, hashing, and dimension access.
///
/// # Design Principles
///
/// - **`Id` as associated type**: Lightweight identifier (conjugate-independent)
/// - **`Eq` by object equality**: Two indices are equal iff they represent the same object
///
///   (including ID, dimension, and conjugate state if applicable)
/// - **`dim()`**: Returns the dimension of the index
/// - **`conj_state()`**: Returns the conjugate state (direction) of the index
///
/// # Key Properties
///
/// - **`Eq`**: Defines object equality (includes ID, dimension, and conjugate state)
/// - **`Hash`**: Enables efficient lookup in `HashMap<I, ...>` / `HashSet<I>`
/// - **`Clone`**: Indices are small value types, freely copyable
/// - **`is_contractable()`**: Determines if two indices can be contracted
///
/// # Conjugate State and Contractability
///
/// The `conj_state()` method returns the direction of an index:
/// - `Undirected`: Directionless index (ITensors.jl-like default)
/// - `Ket`: Ingoing index (QSpace: no trailing `*` in itag)
/// - `Bra`: Outgoing index (QSpace: trailing `*` in itag)
///
/// Two indices are contractable if:
/// - Their dimensions match
/// - Their full index identity is compatible:
///
///   - `(Undirected, Undirected)` → equal indices
///   - `(Ket, Bra)` or `(Bra, Ket)` → one index is the other's conjugate
///   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)
///
/// # Example
///
/// ```
/// use tensor4all_core::{DynIndex, IndexLike};
///
/// let i = DynIndex::new_dyn(2);
/// let j = DynIndex::new_dyn(3);
/// let k = DynIndex::new_dyn(4);
///
/// let a = vec![i.clone(), j.clone()];
/// let b = vec![j.clone(), k.clone()];
/// let common: Vec<_> = a.iter().filter(|idx| b.contains(idx)).cloned().collect();
///
/// assert_eq!(common, vec![j]);
/// ```
pub trait IndexLike: Clone + Eq + Hash + Debug + Send + Sync + 'static {
    /// Lightweight identifier type (conjugate-independent).
    ///
    /// **Rule**: Contractable indices must have compatible full index identity.
    ///
    /// The ID serves as a lightweight component of full index identity.
    /// In large tensor networks, IDs enable efficient graph-based lookups (O(1) with HashSet/HashMap)
    /// to find matching legs across many tensors.
    ///
    /// This is separate from the full `Eq`/`Hash` identity:
    /// - **ID**: raw logical identifier, useful for serialization and diagnostics
    /// - **Eq/Hash**: concrete tensor leg identity used by maps, sets, and contraction
    /// - **dim/ConjState**: mathematical compatibility checks
    type Id: Clone + Eq + Hash + Debug + Send + Sync;

    /// Get the identifier of this index.
    ///
    /// The ID is a lightweight raw identifier.
    /// **Do not use ID-only equality as concrete tensor leg identity**; use full index
    /// equality, or `is_contractable()` when deciding contraction compatibility.
    ///
    /// Two indices with the same ID may still be distinct concrete legs, for example when
    /// they differ by prime level, tags, or direction.
    fn id(&self) -> &Self::Id;

    /// Get the total dimension (state-space dimension) of the index.
    fn dim(&self) -> usize;

    /// Get the prime level of this index.
    /// Default: 0 (unprimed).
    fn plev(&self) -> i64 {
        0
    }

    /// Get the conjugate state (direction) of this index.
    ///
    /// Returns `ConjState::Undirected` for directionless indices (ITensors.jl-like default),
    /// or `ConjState::Ket`/`ConjState::Bra` for directed indices (QSpace-compatible).
    fn conj_state(&self) -> ConjState;

    /// Create the conjugate of this index.
    ///
    /// For directed indices, this toggles between `Ket` and `Bra`.
    /// For `Undirected` indices, this returns `self` unchanged (no-op).
    ///
    /// # Returns
    /// A new index with the conjugate state toggled (if directed) or unchanged (if undirected).
    fn conj(&self) -> Self;

    /// Check if this index can be contracted with another index.
    ///
    /// Two indices are contractable if:
    /// - They have the same dimension
    /// - Their full identity is compatible:
    ///
    ///   - `(Undirected, Undirected)` → equal indices
    ///   - `(Ket, Bra)` or `(Bra, Ket)` → one index is the other's conjugate
    ///   - Mixed `(Undirected, Ket/Bra)` → **not contractable** (mixing forbidden)
    ///
    /// # Default Implementation
    ///
    /// The default implementation checks:
    /// 1. Same dimension: `self.dim() == other.dim()`
    /// 2. Compatible conjugate states (see rules above)
    /// 3. Full index equality for undirected indices, or conjugate full equality for
    ///    directed pairs.
    fn is_contractable(&self, other: &Self) -> bool {
        if self.dim() != other.dim() {
            return false;
        }
        match (self.conj_state(), other.conj_state()) {
            (ConjState::Ket, ConjState::Bra) | (ConjState::Bra, ConjState::Ket) => {
                self.conj() == other.clone()
            }
            (ConjState::Undirected, ConjState::Undirected) => self == other,
            _ => false, // Mixed directed/undirected is forbidden
        }
    }

    /// Check if this index has the same ID as another.
    ///
    /// Default implementation compares IDs directly.
    /// This is a convenience method for pure ID comparison (does not check contractability).
    fn same_id(&self, other: &Self) -> bool {
        self.id() == other.id()
    }

    /// Check if this index has the given ID.
    ///
    /// Default implementation compares with the given ID.
    fn has_id(&self, id: &Self::Id) -> bool {
        self.id() == id
    }

    /// Create a similar index with a new identity but the same structure (dimension, tags, etc.).
    ///
    /// This is used to create "equivalent" indices that have the same properties
    /// but different identities, commonly needed in index replacement operations.
    ///
    /// # Returns
    /// A new index with a fresh identity and the same structure as `self`.
    fn sim(&self) -> Self
    where
        Self: Sized;

    /// Create a pair of contractable dummy indices with dimension 1.
    ///
    /// These are used for structural connections that don't carry quantum numbers,
    /// such as connecting components in a tree tensor network.
    ///
    /// Both indices will be `Undirected` and have the same ID, making them contractable.
    ///
    /// # Returns
    /// A pair `(idx1, idx2)` where `idx1.is_contractable(&idx2)` is true.
    fn create_dummy_link_pair() -> (Self, Self)
    where
        Self: Sized;

    /// Create a fresh link index representing the tensor-product space of input indices.
    ///
    /// Generic algorithms may use this to replace multiple local bond legs by one fused
    /// leg without depending on a concrete index implementation. The returned link must
    /// have a fresh identity and represent the exact tensor-product basis of `indices`.
    ///
    /// Implementations with symmetry or sector metadata should preserve the tensor-product
    /// basis and charge structure when possible. They should return `Err` if the fused
    /// product link cannot be represented exactly.
    ///
    /// # Arguments
    /// * `indices` - Non-empty input indices whose tensor-product space is represented by
    ///
    ///   the output. Typical inputs are link or bond indices being fused into one link.
    ///
    /// # Returns
    /// A new index with fresh identity and dimension equal to the checked product of all input
    /// dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation fails (a shape or index mismatch, or
    /// /// a backend failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::{DynIndex, IndexLike, TagSetLike};
    ///
    /// let a = DynIndex::new_link(2).unwrap();
    /// let b = DynIndex::new_link(3).unwrap();
    /// let product = DynIndex::product_link(&[a.clone(), b.clone()]).unwrap();
    ///
    /// assert_eq!(product.dim(), 6);
    /// assert!(product.tags().has_tag("Link"));
    /// assert_ne!(product.id(), a.id());
    /// assert_ne!(product.id(), b.id());
    /// ```
    fn product_link(indices: &[Self]) -> anyhow::Result<Self>
    where
        Self: Sized;
}

/// Sort indices by a deterministic comparator over `dim`, `plev`, `id`, then
/// `Debug` representation.
///
/// The `plev` and `Debug` terms keep distinct indices that share the same ID —
/// for example a same-ID pair differing by prime level or tags — in a pinned,
/// insertion-order-independent relative order instead of leaving them to the
/// stable sort's input order.
///
/// The comparator is a pure function of the index values, so the result never
/// depends on insertion order as long as `Debug` is deterministic and
/// distinguishes all identity-relevant fields. The concrete
/// [`DynIndex`](crate::defaults::DynIndex) satisfies the determinism part: its
/// derived `Debug` covers `id`, `dim`, `plev`, and `tags`, so two `DynIndex`
/// values compare `Equal` only when every field matches. Note that `dim`
/// participates in the sort even though `DynIndex`'s `Eq` deliberately ignores
/// it (ITensors semantics: equality = id + plev + tags). Implementors whose
/// `Debug` omits identity-relevant fields may still see `Equal` comparisons
/// between distinct indices, in which case the stable sort preserves the input
/// order for those ties.
///
/// # Arguments
/// * `indices` - The slice of indices to reorder in place.
///
/// # Returns
/// Nothing; `indices` is reordered in place.
///
/// # Panics
/// Never panics; the comparator relies only on `IndexLike` accessors and
/// `Debug` implementations.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{sort_indices_deterministic, DynIndex, IndexLike};
///
/// let s = DynIndex::new_dyn(2);
/// let s_prime = s.prime(); // same ID, plev 1
/// assert!(s.same_id(&s_prime));
///
/// // Insertion order (primed first) must not survive the sort.
/// let mut indices = vec![s_prime.clone(), s.clone()];
/// sort_indices_deterministic(&mut indices);
/// assert_eq!(indices, vec![s, s_prime]);
/// ```
pub fn sort_indices_deterministic<I: IndexLike>(indices: &mut [I])
where
    I::Id: Ord,
{
    indices.sort_by(|left, right| {
        left.dim()
            .cmp(&right.dim())
            .then_with(|| left.plev().cmp(&right.plev()))
            .then_with(|| left.id().cmp(right.id()))
            .then_with(|| format!("{left:?}").cmp(&format!("{right:?}")))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal IndexLike implementation that uses the default plev() method.
    /// Used to test coverage of the default trait implementations.
    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct TestIndex {
        id: u64,
        dim: usize,
        state: ConjState,
    }

    fn test_index(id: u64, dim: usize) -> TestIndex {
        TestIndex {
            id,
            dim,
            state: ConjState::Undirected,
        }
    }

    fn directed_test_index(id: u64, dim: usize, state: ConjState) -> TestIndex {
        TestIndex { id, dim, state }
    }

    impl IndexLike for TestIndex {
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
            TestIndex {
                state,
                ..self.clone()
            }
        }

        fn sim(&self) -> Self {
            TestIndex {
                id: self.id + 1000,
                ..self.clone()
            }
        }

        fn create_dummy_link_pair() -> (Self, Self) {
            (test_index(0, 1), test_index(0, 1))
        }

        /// # Errors
        ///
        /// Returns an error when the operation fails (a shape or index mismatch, or
        /// /// a backend failure).
        ///
        fn product_link(indices: &[Self]) -> anyhow::Result<Self> {
            anyhow::ensure!(
                !indices.is_empty(),
                "product_link requires at least one index"
            );
            let dim = indices.iter().try_fold(1usize, |acc, idx| {
                acc.checked_mul(idx.dim)
                    .ok_or_else(|| anyhow::anyhow!("product link dimension overflow"))
            })?;
            Ok(test_index(9999, dim))
        }
    }

    #[test]
    fn test_default_plev_is_zero() {
        let idx = test_index(1, 3);
        assert_eq!(idx.plev(), 0);
    }

    #[test]
    fn test_default_is_contractable_with_plev() {
        let a = test_index(1, 3);
        let b = test_index(1, 3);
        // Same id, dim, and default plev=0: contractable
        assert!(a.is_contractable(&b));
    }

    #[test]
    fn test_default_is_contractable_rejects_id_and_dim_mismatch() {
        let a = test_index(1, 3);
        let different_id = test_index(2, 3);
        let different_dim = test_index(1, 4);

        assert!(!a.is_contractable(&different_id));
        assert!(!a.is_contractable(&different_dim));
    }

    #[test]
    fn test_default_is_contractable_supports_directed_pairs_only() {
        let ket = directed_test_index(1, 3, ConjState::Ket);
        let bra = directed_test_index(1, 3, ConjState::Bra);
        let undirected = test_index(1, 3);

        assert!(ket.is_contractable(&bra));
        assert!(bra.is_contractable(&ket));
        assert!(!ket.is_contractable(&undirected));
        assert!(!undirected.is_contractable(&bra));
    }

    #[test]
    fn test_default_link_helpers_preserve_expected_structure() {
        let idx = directed_test_index(7, 5, ConjState::Ket);
        let conjugated = idx.conj();
        assert_eq!(conjugated.conj_state(), ConjState::Bra);
        assert_eq!(conjugated.id(), idx.id());
        assert_eq!(conjugated.dim(), idx.dim());

        let similar = idx.sim();
        assert_eq!(similar.dim(), idx.dim());
        assert_eq!(similar.conj_state(), idx.conj_state());
        assert_ne!(similar.id(), idx.id());

        let (left, right) = TestIndex::create_dummy_link_pair();
        assert_eq!(left.dim(), 1);
        assert!(left.is_contractable(&right));
    }

    #[test]
    fn test_product_link_helper_checks_empty_and_overflow_inputs() {
        let a = test_index(1, 2);
        let b = test_index(2, 3);
        let product = TestIndex::product_link(&[a, b]).unwrap();
        assert_eq!(product.dim(), 6);
        assert_eq!(product.conj_state(), ConjState::Undirected);

        assert!(TestIndex::product_link(&[]).is_err());
        assert!(TestIndex::product_link(&[test_index(1, usize::MAX), test_index(2, 2)]).is_err());
    }

    #[test]
    fn test_default_same_id() {
        let a = test_index(1, 3);
        let b = test_index(1, 5);
        let c = test_index(2, 3);
        assert!(a.same_id(&b));
        assert!(!a.same_id(&c));
    }

    #[test]
    fn test_default_has_id() {
        let a = test_index(42, 3);
        assert!(a.has_id(&42));
        assert!(!a.has_id(&99));
    }

    #[test]
    fn test_sort_indices_deterministic_pins_same_id_prime_pair() {
        use crate::defaults::DynIndex;

        let s = DynIndex::new_dyn(2);
        let s_prime = s.prime();
        assert!(s.same_id(&s_prime));
        assert_ne!(s, s_prime);

        // Primed index first (worst-case insertion order): the sort must pin
        // unprimed before primed by full-index order (plev 0 < plev 1), not
        // preserve insertion order.
        let mut indices = vec![s_prime.clone(), s.clone()];
        sort_indices_deterministic(&mut indices);
        assert_eq!(indices, vec![s.clone(), s_prime.clone()]);

        // Same canonical order regardless of input permutation.
        let mut indices = vec![s.clone(), s_prime.clone()];
        sort_indices_deterministic(&mut indices);
        assert_eq!(indices, vec![s, s_prime]);
    }

    #[test]
    fn test_sort_indices_deterministic_distinguishes_same_id_tag_pair() {
        use crate::defaults::{DynId, DynIndex, TagSet};

        let site = DynIndex::new_with_tags(DynId(42), 2, TagSet::from_tags(&["Site"]).unwrap());
        let link = DynIndex::new_with_tags(DynId(42), 2, TagSet::from_tags(&["Link"]).unwrap());
        assert!(site.same_id(&link));
        assert_ne!(site, link);

        // The Debug tiebreak must give one deterministic order, not leave the
        // pair to insertion order.
        let mut a = vec![link.clone(), site.clone()];
        let mut b = vec![site.clone(), link.clone()];
        sort_indices_deterministic(&mut a);
        sort_indices_deterministic(&mut b);
        assert_eq!(a, b);
        assert!(a.contains(&site));
        assert!(a.contains(&link));
    }
}
