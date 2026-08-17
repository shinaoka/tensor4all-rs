//! Result values and diagnostics returned by tree ACI.

use tensor4all_core::IdxTensor;
use tensor4all_treetn::TreeTN;

use crate::TreeAciNode;

/// Explains why a tree ACI run stopped.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciTermination;
/// assert_eq!(TreeAciTermination::default(), TreeAciTermination::MaxSweeps);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum TreeAciTermination {
    /// Error and rank stability criteria were satisfied.
    Converged,
    /// At least one inaccurate edge reached its algebraic or configured rank cap.
    RankLimited,
    /// The configured sweep limit was reached.
    #[default]
    MaxSweeps,
}

/// Structural and cache statistics from a tree ACI run.
///
/// Edge tuples use deterministic node-name order, making diagnostics stable
/// across equivalent insertion orders.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciDiagnostics;
/// let diagnostics = TreeAciDiagnostics::<usize>::default();
/// assert_eq!(diagnostics.evaluated_points, 0);
/// assert!(diagnostics.edge_ranks.is_empty());
/// assert!(diagnostics.candidate_set_sizes.is_empty());
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TreeAciDiagnostics<V> {
    /// Final `(lower endpoint, upper endpoint, rank)` values.
    pub edge_ranks: Vec<(V, V, usize)>,
    /// Edges whose algebraic or configured rank limit was reached.
    pub saturated_edges: Vec<(V, V)>,
    /// Total number of full input points evaluated.
    pub evaluated_points: u64,
    /// Number of immutable component-sample records retained at termination.
    pub sample_arena_records: usize,
    /// Logical bytes retained by component records and their deduplication keys.
    pub sample_arena_retained_bytes: usize,
    /// Candidate component samples retained per directed cut, as
    /// `(from, to, len)`.
    ///
    /// Candidate sets are replaced when their own edge is updated and appended
    /// to by global pivot injection, so they stay bounded without an eviction
    /// policy. They are reported so that a future phase can design one against
    /// measurements should that stop being true.
    pub candidate_set_sizes: Vec<(V, V, usize)>,
}

/// The interpolated tree and convergence history produced by tree ACI.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::{TreeAciDiagnostics, TreeAciResult, TreeAciTermination};
/// use tensor4all_treetn::TreeTN;
///
/// let result = TreeAciResult::<usize> {
///     tree: TreeTN::new(),
///     max_ranks: vec![],
///     max_errors: vec![],
///     global_pivots_found: vec![],
///     termination: TreeAciTermination::MaxSweeps,
///     diagnostics: TreeAciDiagnostics::default(),
/// };
/// assert_eq!(result.tree.node_count(), 0);
/// ```
#[derive(Clone, Debug)]
pub struct TreeAciResult<V: TreeAciNode> {
    /// Interpolated output with the prepared input topology and site indices.
    pub tree: TreeTN<IdxTensor, V>,
    /// Maximum output edge rank after each sweep.
    pub max_ranks: Vec<usize>,
    /// Maximum normalized local error after each sweep.
    pub max_errors: Vec<f64>,
    /// Number of distinct significant pivots found by each global guard run.
    pub global_pivots_found: Vec<usize>,
    /// Reason execution stopped.
    pub termination: TreeAciTermination,
    /// Final structural and cache diagnostics.
    pub diagnostics: TreeAciDiagnostics<V>,
}
