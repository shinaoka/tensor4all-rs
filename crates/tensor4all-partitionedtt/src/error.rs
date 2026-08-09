//! Error types for partitioned tensor train operations

use tensor4all_core::{DynIndex, TensorStorageError};
use tensor4all_itensorlike::TensorTrainError;
use thiserror::Error;

/// Result type for partitioned tensor train operations
pub type Result<T> = std::result::Result<T, PartitionedTTError>;

/// Errors that can occur during partitioned tensor train operations
#[derive(Error, Debug)]
pub enum PartitionedTTError {
    /// Projectors overlap
    #[error("Projectors overlap")]
    OverlappingProjectors,

    /// Projector conflict: same index has different values
    #[error("Projector conflict")]
    ProjectorConflict,

    /// No overlap between projectors (contraction would be zero)
    #[error("No overlap between projectors")]
    NoOverlap,

    /// Empty partitioned tensor train
    #[error("Partitioned tensor train is empty")]
    Empty,

    /// No matching subdomain for the given indices
    #[error("No matching subdomain found for indices")]
    NoMatchingSubdomain,

    /// Error from authoritative tensor storage materialization.
    #[error("Tensor storage error: {source}")]
    TensorStorage {
        /// Original typed storage diagnostic.
        #[source]
        source: TensorStorageError,
    },

    /// Error from a differentiable tensor construction or backend operation.
    #[error("Tensor construction error: {source}")]
    TensorConstruction {
        /// Original construction/backend diagnostic.
        #[source]
        source: anyhow::Error,
    },

    /// Error from tensor-train structure or contraction validation.
    #[error("Tensor train error: {source}")]
    TensorTrain {
        /// Original tensor-train diagnostic.
        #[source]
        source: TensorTrainError,
    },

    /// Projector refers to an index absent from the tensor train.
    #[error("projector index {index:?} is absent from the tensor train")]
    ProjectorIndexNotFound {
        /// Index supplied by the caller.
        index: DynIndex,
    },

    /// Projector coordinate is outside the selected index dimension.
    #[error(
        "projector coordinate {value} is out of range for index {index:?} with dimension {dim}"
    )]
    ProjectorCoordinateOutOfBounds {
        /// Index whose coordinate was requested.
        index: DynIndex,
        /// Invalid zero-based coordinate.
        value: usize,
        /// Dimension of the index.
        dim: usize,
    },

    /// Error from tensor cross interpolation
    #[error("Tensor cross interpolation error: {0}")]
    TensorCrossInterpolation(#[from] tensor4all_tensorci::TCIError),

    /// Invalid site-index or pivot input for adaptive interpolation
    #[error("Invalid adaptive interpolation input: {0}")]
    InvalidAdaptiveInterpolationInput(String),

    /// Feature not yet implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    /// Incompatible projector structure
    #[error("Incompatible projectors: {0}")]
    IncompatibleProjectors(String),

    /// Invalid options for a partitioned tensor train operation
    #[error("Invalid options: {0}")]
    InvalidOptions(String),
}

impl PartitionedTTError {
    pub(crate) fn tensor_train_operation(message: impl Into<String>) -> Self {
        Self::TensorTrain {
            source: TensorTrainError::OperationError {
                message: message.into(),
            },
        }
    }
}
