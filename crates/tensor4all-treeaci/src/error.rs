//! Typed errors for tree Alternating Cross Interpolation.

use tensor4all_treetn::TreeTNOperationError;
use thiserror::Error;

/// A result returned by tree ACI operations.
///
/// Related types: [`TreeAciError`] classifies validation, resource, numerical,
/// callback, and tensor-network failures.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::{Result, TreeElementwiseBatch};
///
/// fn make_batch(values: &[f64]) -> Result<TreeElementwiseBatch<'_, f64>> {
///     TreeElementwiseBatch::new(values, 2, 2)
/// }
///
/// let values = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(make_batch(&values)?.get(1, 1)?, 4.0);
/// # Ok::<(), tensor4all_treeaci::TreeAciError>(())
/// ```
pub type Result<T> = std::result::Result<T, TreeAciError>;

/// An error reported while preparing or running tree ACI.
///
/// Use this type to distinguish invalid public inputs from resource limits and
/// lower-layer failures. More variants will be added as the interpolation
/// state and execution engine land.
///
/// Related types: [`TreeElementwiseBatch`](crate::TreeElementwiseBatch) uses
/// these variants for shape and indexing failures.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::{TreeAciError, TreeElementwiseBatch};
///
/// let error = TreeElementwiseBatch::<f64>::new(&[1.0], 1, 2).unwrap_err();
/// assert!(matches!(
///     error,
///     TreeAciError::LengthMismatch {
///         expected: 2,
///         actual: 1
///     }
/// ));
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TreeAciError {
    /// No input tensor network was supplied.
    #[error("tree ACI requires at least one input tensor network")]
    NoInputs,

    /// An input tensor network contained no nodes.
    #[error("tree ACI input {input} has no nodes")]
    EmptyTree {
        /// Zero-based input position.
        input: usize,
    },

    /// An option value violated its documented numerical contract.
    #[error("invalid tree ACI option {option}: {message}")]
    InvalidOption {
        /// Public option field name.
        option: &'static str,
        /// Reason the value is invalid.
        message: &'static str,
    },

    /// An input tree did not share the reference input's labeled topology.
    #[error("tree ACI input {input} has a different labeled topology")]
    TopologyMismatch {
        /// Zero-based input position.
        input: usize,
    },

    /// An input tree assigned different full physical indices to a node.
    #[error("tree ACI input {input} has incompatible physical indices at node {node}")]
    PhysicalIndexMismatch {
        /// Zero-based input position.
        input: usize,
        /// Debug rendering of the affected node name.
        node: String,
    },

    /// A global physical point did not have one coordinate per prepared node.
    #[error("tree ACI point length mismatch: expected {expected}, got {actual}")]
    PointLengthMismatch {
        /// Prepared node count.
        expected: usize,
        /// Supplied coordinate count.
        actual: usize,
    },

    /// A local coordinate exceeded the flattened physical dimension of a node.
    #[error(
        "tree ACI physical coordinate out of bounds at node position {node}: coordinate {coordinate}, local dimension {local_dim}"
    )]
    PhysicalCoordinateOutOfBounds {
        /// Stable zero-based node position.
        node: usize,
        /// Supplied flattened local coordinate.
        coordinate: usize,
        /// Valid flattened local dimension.
        local_dim: usize,
    },

    /// A prepared allocation estimate exceeded its configured hard limit.
    #[error(
        "tree ACI resource limit exceeded for {resource}: requested {requested}, limit {limit}"
    )]
    ResourceLimit {
        /// Name of the rejected allocation category.
        resource: &'static str,
        /// Checked estimated size in the category's documented units.
        requested: usize,
        /// Configured maximum in the same units.
        limit: usize,
    },

    /// A required batch axis had length zero.
    #[error("tree ACI batch {axis} axis must be nonempty")]
    EmptyBatchAxis {
        /// Name of the empty axis.
        axis: &'static str,
    },

    /// Checked arithmetic overflowed while deriving a size.
    #[error("tree ACI size overflow while computing {context}")]
    SizeOverflow {
        /// Operation whose result did not fit in `usize`.
        context: &'static str,
    },

    /// A flat buffer length differed from its declared shape.
    #[error("tree ACI length mismatch: expected {expected} values, got {actual}")]
    LengthMismatch {
        /// Length implied by the validated shape.
        expected: usize,
        /// Length supplied by the caller.
        actual: usize,
    },

    /// A requested batch coordinate was outside its axis.
    #[error("tree ACI batch {axis} index out of bounds: index {index}, len {len}")]
    BatchIndexOutOfBounds {
        /// Name of the indexed axis.
        axis: &'static str,
        /// Requested zero-based coordinate.
        index: usize,
        /// Valid length of the axis.
        len: usize,
    },

    /// A caller-supplied elementwise callback returned an error.
    #[error("tree ACI callback failed: {message}")]
    Callback {
        /// Callback-provided context.
        message: String,
    },

    /// A rank-revealing interpolation operation failed.
    #[error("tree ACI numerical operation failed: {message}")]
    Numerical {
        /// Lower-layer numerical context.
        message: String,
    },

    /// A tensor payload could not be decoded as the requested scalar type.
    #[error("tree ACI scalar payload mismatch: {message}")]
    ScalarKind {
        /// Lower-layer scalar conversion diagnostic.
        message: String,
    },

    /// An explicitly supplied output guess is incompatible with the problem.
    #[error("invalid tree ACI initial guess: {message}")]
    InvalidInitialGuess {
        /// Topology, physical-index, rank, scalar, or consistency diagnostic.
        message: String,
    },

    /// A TreeTN operation rejected the prepared network.
    #[error(transparent)]
    TreeTN(#[from] TreeTNOperationError),

    /// An internal invariant failed after public inputs had been validated.
    #[error("tree ACI internal invariant failed: {message}")]
    InternalInvariant {
        /// Description intended for bug reports.
        message: &'static str,
    },
}
