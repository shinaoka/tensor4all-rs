//! Configuration for tree ACI preparation and execution.

use std::mem::size_of;

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::{TreeAciNode, TreeAciScalar, TreeAciTraversalStrategy};

/// Share of [`TreeAciOptions::max_working_bytes`] one derived object may claim.
///
/// A local candidate matrix, a node core, and a cached directed frame each
/// coexist with the other transient buffers of a single local update, so no
/// one of them may claim the whole budget. A quarter is the ratio the crate's
/// original hard-coded ceilings had against the original budget (`2^24` f64
/// elements is 128 MiB, against 512 MiB), so deriving at this share leaves the
/// `f64` defaults exactly where they were while making them follow the budget.
const WORKING_BUDGET_OBJECT_SHARE: usize = 4;

/// Controls tree ACI sweeps, global validation, and allocation limits.
///
/// The defaults match train ACI's convergence policy and add conservative
/// hard allocation ceilings. Increase a ceiling explicitly only after the
/// topology preflight identifies the resource that needs it.
///
/// Related types: [`TreeAciTraversalStrategy`] controls only edge scheduling;
/// numerical convergence and resource budgets remain strategy-independent.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::{TreeAciOptions, TreeAciTraversalStrategy};
///
/// let options = TreeAciOptions::<usize>::default();
/// assert_eq!(options.max_sweeps, 20);
/// assert!(options.scale_tolerance);
/// assert_eq!(options.traversal_strategy, TreeAciTraversalStrategy::MinimumRetracingWalk);
/// ```
#[derive(Clone, Debug)]
pub struct TreeAciOptions<V: TreeAciNode> {
    /// Maximum number of complete sweeps. Default: `20`.
    pub max_sweeps: usize,
    /// Minimum stable sweeps required before convergence. Default: `2`.
    pub min_sweeps: usize,
    /// Optional rank cap on every output edge. Default: no explicit cap.
    pub max_bond_dim: Option<usize>,
    /// Relative or absolute local error target. Default: `1e-12`.
    pub tolerance: f64,
    /// Scale `tolerance` by sampled output magnitude. Default: `true`.
    pub scale_tolerance: bool,
    /// Optional topology-compatible initial output. Default: `None`.
    pub initial_guess: Option<TreeTN<IdxTensor, V>>,
    /// Seed for reproducible randomized choices. Default: `0`.
    pub rng_seed: u64,
    /// Optional initial traversal root. Default: a deterministic diameter endpoint.
    pub root: Option<V>,
    /// Run independent global-pivot searches before convergence. Default: `true`.
    pub enable_global_guard: bool,
    /// Maximum logical bytes retained by all guard evaluators' persistent
    /// message caches combined. Default: 256 MiB.
    ///
    /// The budget is divided evenly among all input evaluators and the output
    /// evaluator used by global-pivot searches. A finite nonzero value retains
    /// useful reuse while preventing repeated floating-zone scans from
    /// retaining an unbounded set of message payloads. Set it to zero to
    /// disable message retention without disabling the guard itself.
    pub message_cache_max_bytes: usize,
    /// Random starts per global search. Default: `5`.
    pub nsearch_global_pivots: usize,
    /// Maximum pivots injected by one global search. Default: `5`.
    pub max_nglobal_pivots: usize,
    /// Coordinate sweeps allowed per global-search walk. Default: `100`.
    pub nsweeps_global_search: usize,
    /// Multiplier applied to the global-search acceptance threshold. Default: `10`.
    pub global_tolerance_margin: f64,
    /// Maximum rows in a materialized local candidate matrix. Default: `2^20`.
    ///
    /// This and [`Self::max_candidate_cols`] are shape guards rather than
    /// memory guards: they bound how far candidate enumeration may run away on
    /// one side of an edge, counted in candidates and not in bytes, and they
    /// therefore do not follow [`Self::max_working_bytes`]. What the resulting
    /// matrix may occupy is [`Self::max_local_matrix_elements`], which does.
    pub max_candidate_rows: usize,
    /// Maximum columns in a materialized local candidate matrix. Default: `2^20`.
    ///
    /// The column-side counterpart of [`Self::max_candidate_rows`], with the
    /// same units and the same independence from the working budget.
    pub max_candidate_cols: usize,
    /// Maximum elements in a local candidate matrix, or `None` (the default)
    /// to derive the ceiling from [`Self::max_working_bytes`].
    ///
    /// A derived ceiling is a quarter of the working budget expressed in
    /// elements of the run's scalar type, so raising the budget raises this
    /// ceiling with it; at the default 512 MiB budget it is `2^24` `f64`
    /// elements, the value this field was fixed at before it followed the
    /// budget. Set `Some(n)` only to pin a ceiling that must *not* follow the
    /// budget. [`Self::resolved_max_local_matrix_elements`] returns the value
    /// a run actually enforces.
    pub max_local_matrix_elements: Option<usize>,
    /// Maximum elements in any prepared or output node core, or `None` (the
    /// default) to derive the ceiling from [`Self::max_working_bytes`] exactly
    /// as [`Self::max_local_matrix_elements`] does.
    ///
    /// [`Self::resolved_max_core_elements`] returns the enforced value.
    pub max_core_elements: Option<usize>,
    /// Maximum elements in one cached directed frame, or `None` (the default)
    /// to derive the ceiling from [`Self::max_working_bytes`] exactly as
    /// [`Self::max_local_matrix_elements`] does.
    ///
    /// This bounds a single frame. The frame cache retains one frame per input
    /// per directed edge, so [`Self::max_frame_bytes`] is what bounds the
    /// cache as a whole. [`Self::resolved_max_frame_elements`] returns the
    /// enforced value.
    pub max_frame_elements: Option<usize>,
    /// Maximum logical bytes retained by the directed-frame cache, across every
    /// input and every directed edge, plus the pivot-search candidate-frame
    /// cache that shares this budget. Default: 256 MiB.
    ///
    /// Checked before each frame allocation, so an over-budget run is refused
    /// rather than reaching the ceiling and then reporting it. The candidate
    /// cache degrades instead of refusing: once it would push the combined
    /// total over this ceiling, new candidates are still computed but simply
    /// not cached. Counts the caches' owned payload, not allocator or process
    /// overhead.
    pub max_frame_bytes: usize,
    /// Maximum logical bytes retained by immutable component samples. Default: 256 MiB.
    pub max_sample_arena_bytes: usize,
    /// Maximum estimated temporary working storage. Default: 512 MiB.
    ///
    /// This is the governing budget for one local update's live buffers, and
    /// every element ceiling left unset follows it: raising this field raises
    /// [`Self::max_local_matrix_elements`], [`Self::max_core_elements`], and
    /// [`Self::max_frame_elements`] in step, each to a quarter of the budget
    /// in elements of the run's scalar type. A ceiling set explicitly keeps
    /// overriding the budget in either direction.
    ///
    /// The retention budgets are separate and do not follow this one, because
    /// they bound what is kept *between* updates rather than what one update
    /// may allocate: [`Self::max_frame_bytes`],
    /// [`Self::message_cache_max_bytes`], and
    /// [`Self::max_sample_arena_bytes`].
    pub max_working_bytes: usize,
    /// Edge traversal strategy. Default: continuous minimum-retracing walk.
    pub traversal_strategy: TreeAciTraversalStrategy,
}

impl<V: TreeAciNode> Default for TreeAciOptions<V> {
    fn default() -> Self {
        Self {
            max_sweeps: 20,
            min_sweeps: 2,
            max_bond_dim: None,
            tolerance: 1.0e-12,
            scale_tolerance: true,
            initial_guess: None,
            rng_seed: 0,
            root: None,
            enable_global_guard: true,
            message_cache_max_bytes: 256 << 20,
            nsearch_global_pivots: 5,
            max_nglobal_pivots: 5,
            nsweeps_global_search: 100,
            global_tolerance_margin: 10.0,
            max_candidate_rows: 1 << 20,
            max_candidate_cols: 1 << 20,
            max_local_matrix_elements: None,
            max_core_elements: None,
            max_frame_elements: None,
            max_frame_bytes: 256 << 20,
            max_sample_arena_bytes: 256 << 20,
            max_working_bytes: 512 << 20,
            traversal_strategy: TreeAciTraversalStrategy::default(),
        }
    }
}

impl<V: TreeAciNode> TreeAciOptions<V> {
    /// Element ceiling a run with scalar `T` enforces for a local candidate
    /// matrix.
    ///
    /// Returns [`Self::max_local_matrix_elements`] when it is set, and the
    /// budget-derived ceiling otherwise.
    ///
    /// # Returns
    ///
    /// The ceiling in elements of `T`, which is what
    /// [`TreeAciError::ResourceLimit`](crate::TreeAciError::ResourceLimit)
    /// reports as `limit` for the `local matrix elements` resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeAciOptions;
    ///
    /// let mut options = TreeAciOptions::<usize>::default();
    /// assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1 << 24);
    ///
    /// // The ceiling follows the budget instead of silently overriding it.
    /// options.max_working_bytes *= 4;
    /// assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1 << 26);
    ///
    /// // An explicit ceiling still wins, in either direction.
    /// options.max_local_matrix_elements = Some(1024);
    /// assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1024);
    /// ```
    pub fn resolved_max_local_matrix_elements<T: TreeAciScalar>(&self) -> usize {
        self.max_local_matrix_elements
            .unwrap_or_else(|| self.derived_element_ceiling::<T>())
    }

    /// Element ceiling a run with scalar `T` enforces for one node core.
    ///
    /// Returns [`Self::max_core_elements`] when it is set, and the
    /// budget-derived ceiling otherwise.
    ///
    /// # Returns
    ///
    /// The ceiling in elements of `T`, reported as `limit` for the
    /// `core elements` resource.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeAciOptions;
    ///
    /// let mut options = TreeAciOptions::<usize>::default();
    /// assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 24);
    ///
    /// // Complex runs get the same number of *bytes*, not of elements.
    /// assert_eq!(
    ///     options.resolved_max_core_elements::<num_complex::Complex64>(),
    ///     1 << 23
    /// );
    ///
    /// options.max_core_elements = Some(64);
    /// assert_eq!(options.resolved_max_core_elements::<f64>(), 64);
    /// ```
    pub fn resolved_max_core_elements<T: TreeAciScalar>(&self) -> usize {
        self.max_core_elements
            .unwrap_or_else(|| self.derived_element_ceiling::<T>())
    }

    /// Element ceiling a run with scalar `T` enforces for one cached directed
    /// frame.
    ///
    /// Returns [`Self::max_frame_elements`] when it is set, and the
    /// budget-derived ceiling otherwise.
    ///
    /// # Returns
    ///
    /// The ceiling in elements of `T`, reported as `limit` for the
    /// `frame elements` resource. It bounds one frame; the cache as a whole is
    /// bounded by [`Self::max_frame_bytes`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_treeaci::TreeAciOptions;
    ///
    /// let mut options = TreeAciOptions::<usize>::default();
    /// assert_eq!(options.resolved_max_frame_elements::<f64>(), 1 << 24);
    ///
    /// options.max_working_bytes /= 2;
    /// assert_eq!(options.resolved_max_frame_elements::<f64>(), 1 << 23);
    /// ```
    pub fn resolved_max_frame_elements<T: TreeAciScalar>(&self) -> usize {
        self.max_frame_elements
            .unwrap_or_else(|| self.derived_element_ceiling::<T>())
    }

    /// Elements of `T` one object may claim from the working budget.
    ///
    /// The floor of one element keeps a budget smaller than a single scalar
    /// from deriving a zero ceiling, which would refuse every allocation
    /// including the ones the prepared minimum needs;
    /// [`crate::TreeAciError::InvalidOption`] already refuses an explicit zero.
    fn derived_element_ceiling<T: TreeAciScalar>(&self) -> usize {
        (self.max_working_bytes / (WORKING_BUDGET_OBJECT_SHARE * size_of::<T>())).max(1)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::TreeAciOptions;

    #[test]
    fn default_message_cache_budget_is_bounded() {
        let options = TreeAciOptions::<usize>::default();

        assert_eq!(options.message_cache_max_bytes, 256 << 20);
    }

    /// The derived ceilings must reproduce the constants they replaced, so a
    /// caller who never touches `max_working_bytes` sees no change.
    #[test]
    fn default_element_ceilings_match_the_historical_constants() {
        let options = TreeAciOptions::<usize>::default();

        assert_eq!(options.max_local_matrix_elements, None);
        assert_eq!(options.max_core_elements, None);
        assert_eq!(options.max_frame_elements, None);
        assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1 << 24);
        assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 24);
        assert_eq!(options.resolved_max_frame_elements::<f64>(), 1 << 24);
    }

    /// Issue #729: a caller who raises the byte budget must actually get it.
    #[test]
    fn derived_ceilings_follow_the_working_budget() {
        let options = TreeAciOptions::<usize> {
            max_working_bytes: 10 << 30,
            ..TreeAciOptions::default()
        };

        let expected = (10usize << 30) / 4 / size_of::<f64>();
        assert_eq!(
            options.resolved_max_local_matrix_elements::<f64>(),
            expected
        );
        assert_eq!(options.resolved_max_core_elements::<f64>(), expected);
        assert_eq!(options.resolved_max_frame_elements::<f64>(), expected);
        // The 17,958,192-element local matrix of the reported failure fits a
        // 10 GiB budget, which is what the caller asked for.
        assert!(expected > 17_958_192);
    }

    /// The ceilings measure bytes, so a complex run gets half the elements and
    /// the same memory.
    #[test]
    fn derived_ceilings_charge_the_run_scalar() {
        let options = TreeAciOptions::<usize>::default();

        assert_eq!(
            options.resolved_max_core_elements::<f64>() * size_of::<f64>(),
            options.resolved_max_core_elements::<Complex64>() * size_of::<Complex64>()
        );
    }

    /// An explicit ceiling is a pin: it overrides the budget in both
    /// directions and does not move when the budget does.
    #[test]
    fn explicit_ceilings_override_the_derived_value() {
        let mut options = TreeAciOptions::<usize> {
            max_local_matrix_elements: Some(7),
            max_core_elements: Some(1 << 30),
            ..TreeAciOptions::default()
        };

        assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 7);
        assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 30);
        options.max_working_bytes = 1 << 40;
        assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 7);
        assert_eq!(options.resolved_max_core_elements::<f64>(), 1 << 30);
        assert_eq!(
            options.resolved_max_frame_elements::<f64>(),
            (1usize << 40) / 4 / size_of::<f64>()
        );
    }

    /// A budget below one scalar still derives a usable ceiling; the working
    /// budget itself is what refuses the allocation.
    #[test]
    fn derived_ceiling_never_reaches_zero() {
        let options = TreeAciOptions::<usize> {
            max_working_bytes: 1,
            ..TreeAciOptions::default()
        };

        assert_eq!(options.resolved_max_local_matrix_elements::<f64>(), 1);
    }
}
