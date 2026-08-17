//! Errors returned by partitioned TreeTN operations.

use tensor4all_core::{DynIndex, IdxTensorError, TensorStorageError};
use tensor4all_treetn::TreeTNOperationError;
use thiserror::Error;

/// The result type used by this crate.
///
/// # Examples
///
/// ```
/// use tensor4all_partitionedtreetn::Result;
///
/// let result: Result<usize> = Ok(3);
/// assert_eq!(result.unwrap(), 3);
/// ```
pub type Result<T> = std::result::Result<T, PartitionedTreeTNError>;

/// Errors raised while validating or operating on partitioned TreeTNs.
///
/// `Projector` validation errors are explicit so callers can repair input
/// coordinates and full index identities. Backend and TreeTN failures retain
/// their typed source errors for diagnostics.
///
/// # Examples
///
/// ```
/// use tensor4all_core::DynIndex;
/// use tensor4all_partitionedtreetn::{PartitionedTreeTNError, Projector};
///
/// let index = DynIndex::new_dyn(2);
/// let error = Projector::from_pairs([(index, 2)]).unwrap_err();
/// assert!(matches!(
///     error,
///     PartitionedTreeTNError::ProjectorCoordinateOutOfBounds { value: 2, dim: 2, .. }
/// ));
/// ```
#[derive(Debug, Error)]
pub enum PartitionedTreeTNError {
    /// A different projector overlaps an existing partition patch.
    #[error("projectors overlap; use disjoint projector keys or replace the exact key")]
    OverlappingProjectors,

    /// Two projector entries assign different coordinates to one full index.
    #[error("projector entries conflict; use compatible coordinates")]
    ProjectorConflict,

    /// Two TreeTNs do not have the same named node-and-edge topology.
    #[error("TreeTNs have different named topologies; use the same node names and edges")]
    TopologyMismatch,

    /// Two TreeTNs assign site indices differently or use different full site spaces.
    #[error(
        "TreeTNs have different site-index assignments; reindex explicitly before the operation"
    )]
    SiteIndexMismatch,

    /// Two subdomains do not use the same projector key for strict addition.
    #[error("strict subdomain addition requires identical projector keys")]
    ProjectorMismatch,

    /// An explicit TreeTN center is absent from the network.
    #[error("the requested TreeTN center is absent; choose an existing node name")]
    InvalidCenter,

    /// The supplied operation options are invalid or select a forbidden path.
    #[error("invalid {operation} options: {reason}")]
    InvalidOptions {
        /// Operation whose options were rejected.
        operation: &'static str,
        /// Repair guidance for the rejected option combination.
        reason: &'static str,
    },

    /// A checked patch-volume product or sum overflowed `usize`.
    #[error("partition volume overflowed usize; use smaller site dimensions or fewer patches")]
    VolumeOverflow,

    /// A checked logical local-tensor element count overflowed `usize`.
    #[error(
        "logical tensor parameter count overflowed usize; use smaller tensors or bond dimensions"
    )]
    LogicalParameterCountOverflow,

    /// A norm or budget required by adaptive patching was not finite.
    #[error("adaptive patching encountered a non-finite norm or truncation budget")]
    NonFiniteAdaptiveValue,

    /// The partition has no patches.
    #[error("partitioned TreeTN is empty; add at least one patch")]
    Empty,

    /// A projector refers to a site index absent by full identity.
    #[error("projector index {index:?} is absent from the TreeTN site space")]
    ProjectorIndexNotFound {
        /// The full index supplied by the caller.
        index: DynIndex,
    },

    /// A projector coordinate is outside the matched TreeTN site dimension.
    #[error(
        "projector coordinate {value} is out of range for index {index:?} with dimension {dim}"
    )]
    ProjectorCoordinateOutOfBounds {
        /// The full index supplied by the caller.
        index: DynIndex,
        /// The invalid zero-based coordinate.
        value: usize,
        /// The dimension of the matching TreeTN site index.
        dim: usize,
    },

    /// The input TreeTN is not one connected acyclic tree.
    #[error("invalid TreeTN topology: {source}")]
    InvalidTopology {
        /// The underlying topology diagnostic.
        #[source]
        source: TreeTNOperationError,
    },

    /// The TreeTN tensors have incompatible scalar dtypes.
    #[error("TreeTN tensors have mixed scalar dtypes: expected {expected}, found {actual}")]
    DTypeMismatch {
        /// The dtype established by the first node.
        expected: String,
        /// The incompatible dtype found at a later node.
        actual: String,
    },

    /// An IdxTensor dtype is not supported by this crate's masking path.
    #[error("unsupported IdxTensor scalar dtype: {dtype}")]
    UnsupportedDType {
        /// A diagnostic name for the unsupported dtype.
        dtype: String,
    },

    /// An operation on the underlying TreeTN failed.
    #[error("TreeTN operation failed: {source}")]
    TreeTN {
        /// The original TreeTN diagnostic.
        #[source]
        source: TreeTNOperationError,
    },

    /// Tensor storage materialization or access failed.
    #[error("tensor storage operation failed: {source}")]
    TensorStorage {
        /// The original storage diagnostic.
        #[source]
        source: TensorStorageError,
    },

    /// Tensor construction or backend execution failed.
    #[error("tensor construction or backend operation failed: {source}")]
    TensorConstruction {
        /// The original tensor diagnostic and source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<TreeTNOperationError> for PartitionedTreeTNError {
    fn from(source: TreeTNOperationError) -> Self {
        Self::TreeTN { source }
    }
}

impl From<IdxTensorError> for PartitionedTreeTNError {
    fn from(source: IdxTensorError) -> Self {
        match source {
            IdxTensorError::Storage { source } => Self::TensorStorage { source },
            source => Self::TensorConstruction {
                source: anyhow::Error::new(source),
            },
        }
    }
}

impl PartitionedTreeTNError {
    pub(crate) fn tree(message: impl Into<String>) -> Self {
        Self::TreeTN {
            source: anyhow::anyhow!(message.into()).into(),
        }
    }

    pub(crate) fn invalid_topology(source: TreeTNOperationError) -> Self {
        Self::InvalidTopology { source }
    }
}
