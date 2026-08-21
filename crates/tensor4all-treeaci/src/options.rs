//! Configuration for tree ACI preparation and execution.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::{TreeAciNode, TreeAciTraversalStrategy};

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
    /// Maximum logical bytes retained by each evaluator's persistent message
    /// cache. Default: unlimited for backwards compatibility.
    ///
    /// This budget applies to both the input evaluators used by global-pivot
    /// searches and the output evaluator used to validate those searches.
    /// A finite nonzero value retains useful reuse while preventing repeated
    /// floating-zone scans from retaining an unbounded set of message payloads.
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
    pub max_candidate_rows: usize,
    /// Maximum columns in a materialized local candidate matrix. Default: `2^20`.
    pub max_candidate_cols: usize,
    /// Maximum elements in a local candidate matrix. Default: `2^24`.
    pub max_local_matrix_elements: usize,
    /// Maximum elements in any prepared or output node core. Default: `2^24`.
    pub max_core_elements: usize,
    /// Maximum elements in one cached directed frame. Default: `2^24`.
    ///
    /// This bounds a single frame. The frame cache retains one frame per input
    /// per directed edge, so [`Self::max_frame_bytes`] is what bounds the
    /// cache as a whole.
    pub max_frame_elements: usize,
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
            message_cache_max_bytes: usize::MAX,
            nsearch_global_pivots: 5,
            max_nglobal_pivots: 5,
            nsweeps_global_search: 100,
            global_tolerance_margin: 10.0,
            max_candidate_rows: 1 << 20,
            max_candidate_cols: 1 << 20,
            max_local_matrix_elements: 1 << 24,
            max_core_elements: 1 << 24,
            max_frame_elements: 1 << 24,
            max_frame_bytes: 256 << 20,
            max_sample_arena_bytes: 256 << 20,
            max_working_bytes: 512 << 20,
            traversal_strategy: TreeAciTraversalStrategy::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TreeAciOptions;

    #[test]
    fn default_message_cache_budget_is_unbounded_for_compatibility() {
        let options = TreeAciOptions::<usize>::default();

        assert_eq!(options.message_cache_max_bytes, usize::MAX);
    }
}
