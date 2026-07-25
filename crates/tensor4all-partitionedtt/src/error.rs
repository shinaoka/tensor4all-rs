//! Error types for partitioned tensor train operations

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

    /// Error from underlying tensor train operations
    #[error("Tensor train error: {0}")]
    TensorTrainError(String),

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
