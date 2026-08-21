//! Selection of tree traversal strategies for ACI sweeps.

/// Selects how tree edges are grouped and ordered for ACI sweeps.
///
/// The initial implementation provides the shortest continuous edge-covering
/// walk. Additional strategies may later use sample-aware path restarts.
///
/// Related types: [`TreeElementwiseBatch`](crate::TreeElementwiseBatch) is the
/// operator input view used by the executor that consumes such a plan.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciTraversalStrategy;
///
/// let strategy = TreeAciTraversalStrategy::default();
/// assert_eq!(strategy, TreeAciTraversalStrategy::MinimumRetracingWalk);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub enum TreeAciTraversalStrategy {
    /// Use a deterministic shortest continuous edge-covering walk.
    ///
    /// With an automatic root its length is `2|E| - diameter`. A path therefore
    /// visits every edge once, while branches retrace only edges outside a
    /// diameter. An explicit root fixes the start and uses the farthest end.
    #[default]
    MinimumRetracingWalk,
}
