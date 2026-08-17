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

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;

    fn assert_display(error: PartitionedTreeTNError, expected: &str) {
        assert_eq!(error.to_string(), expected);
    }

    #[test]
    fn every_variant_has_the_documented_display() {
        assert_display(
            PartitionedTreeTNError::OverlappingProjectors,
            "projectors overlap; use disjoint projector keys or replace the exact key",
        );
        assert_display(
            PartitionedTreeTNError::ProjectorConflict,
            "projector entries conflict; use compatible coordinates",
        );
        assert_display(
            PartitionedTreeTNError::TopologyMismatch,
            "TreeTNs have different named topologies; use the same node names and edges",
        );
        assert_display(
            PartitionedTreeTNError::SiteIndexMismatch,
            "TreeTNs have different site-index assignments; reindex explicitly before the operation",
        );
        assert_display(
            PartitionedTreeTNError::ProjectorMismatch,
            "strict subdomain addition requires identical projector keys",
        );
        assert_display(
            PartitionedTreeTNError::InvalidCenter,
            "the requested TreeTN center is absent; choose an existing node name",
        );
        assert_display(
            PartitionedTreeTNError::InvalidOptions {
                operation: "merge",
                reason: "rtol must be nonnegative",
            },
            "invalid merge options: rtol must be nonnegative",
        );
        assert_display(
            PartitionedTreeTNError::VolumeOverflow,
            "partition volume overflowed usize; use smaller site dimensions or fewer patches",
        );
        assert_display(
            PartitionedTreeTNError::LogicalParameterCountOverflow,
            "logical tensor parameter count overflowed usize; use smaller tensors or bond dimensions",
        );
        assert_display(
            PartitionedTreeTNError::NonFiniteAdaptiveValue,
            "adaptive patching encountered a non-finite norm or truncation budget",
        );
        assert_display(
            PartitionedTreeTNError::Empty,
            "partitioned TreeTN is empty; add at least one patch",
        );

        let index = DynIndex::new_dyn(3);
        let index_debug = format!("{index:?}");
        let error = PartitionedTreeTNError::ProjectorIndexNotFound { index };
        assert!(error.to_string().contains(&index_debug));

        let index = DynIndex::new_dyn(3);
        let index_debug = format!("{index:?}");
        let error = PartitionedTreeTNError::ProjectorCoordinateOutOfBounds {
            index,
            value: 4,
            dim: 3,
        };
        let message = error.to_string();
        assert!(message.contains(&index_debug));
        assert!(message.contains("4") && message.contains("dimension 3"));

        let tree_error = TreeTNOperationError::from(anyhow::anyhow!("disconnected"));
        assert_display(
            PartitionedTreeTNError::InvalidTopology { source: tree_error },
            "invalid TreeTN topology: TreeTN operation failed: disconnected",
        );
        assert_display(
            PartitionedTreeTNError::DTypeMismatch {
                expected: "f64".to_string(),
                actual: "c64".to_string(),
            },
            "TreeTN tensors have mixed scalar dtypes: expected f64, found c64",
        );
        assert_display(
            PartitionedTreeTNError::UnsupportedDType {
                dtype: "f32".to_string(),
            },
            "unsupported IdxTensor scalar dtype: f32",
        );

        let tree_error = TreeTNOperationError::from(anyhow::anyhow!("contraction failed"));
        assert_display(
            PartitionedTreeTNError::TreeTN { source: tree_error },
            "TreeTN operation failed: TreeTN operation failed: contraction failed",
        );
        assert_display(
            PartitionedTreeTNError::TensorStorage {
                source: TensorStorageError::UnsupportedDtype { dtype: "f32" },
            },
            "tensor storage operation failed: compact IdxTensor storage does not support dtype f32; the eager payload remains authoritative",
        );
        assert_display(
            PartitionedTreeTNError::TensorConstruction {
                source: anyhow::anyhow!("backend unavailable"),
            },
            "tensor construction or backend operation failed: backend unavailable",
        );
    }

    #[test]
    fn conversions_preserve_typed_sources_and_messages() {
        let tree_source = TreeTNOperationError::from(anyhow::anyhow!("tree failure"));
        let error = PartitionedTreeTNError::from(tree_source);
        assert!(matches!(error, PartitionedTreeTNError::TreeTN { .. }));
        assert_eq!(
            error.to_string(),
            "TreeTN operation failed: TreeTN operation failed: tree failure"
        );
        assert!(Error::source(&error).is_some());

        let storage_source = TensorStorageError::UnsupportedDtype { dtype: "f32" };
        let error = PartitionedTreeTNError::from(IdxTensorError::Storage {
            source: storage_source,
        });
        assert!(matches!(
            error,
            PartitionedTreeTNError::TensorStorage { .. }
        ));
        assert!(error.to_string().contains("compact IdxTensor storage"));

        let error = PartitionedTreeTNError::from(IdxTensorError::NaNInput { operation: "norm" });
        assert!(matches!(
            error,
            PartitionedTreeTNError::TensorConstruction { .. }
        ));
        assert!(error
            .to_string()
            .contains("IdxTensor norm received NaN input"));
        assert!(Error::source(&error).is_some());
    }

    #[test]
    fn private_helpers_wrap_the_requested_diagnostics() {
        let tree = PartitionedTreeTNError::tree("tensor lookup failed");
        assert!(matches!(tree, PartitionedTreeTNError::TreeTN { .. }));
        assert_eq!(
            tree.to_string(),
            "TreeTN operation failed: TreeTN operation failed: tensor lookup failed"
        );
        assert!(Error::source(&tree).is_some());

        let topology_source = TreeTNOperationError::from(anyhow::anyhow!("cycle"));
        let topology = PartitionedTreeTNError::invalid_topology(topology_source);
        assert!(matches!(
            topology,
            PartitionedTreeTNError::InvalidTopology { .. }
        ));
        assert_eq!(
            topology.to_string(),
            "invalid TreeTN topology: TreeTN operation failed: cycle"
        );
        assert!(Error::source(&topology).is_some());
    }
}
