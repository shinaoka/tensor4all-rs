//! Error types for TensorTrain operations.

use thiserror::Error;

/// Result type for TensorTrain operations.
pub type Result<T> = std::result::Result<T, TensorTrainError>;

/// Errors that can occur in TensorTrain operations.
#[derive(Debug, Error)]
pub enum TensorTrainError {
    /// Tensor train is empty (has no tensors).
    #[error("Tensor train is empty")]
    Empty,

    /// Site index is out of bounds.
    #[error("Site index {site} is out of bounds (tensor train has {length} sites)")]
    SiteOutOfBounds {
        /// The requested site index.
        site: usize,
        /// The total number of sites in the tensor train.
        length: usize,
    },

    /// Bond dimension mismatch between adjacent tensors.
    #[error("Bond dimension mismatch at site {site}: left tensor has right dim {left_dim}, right tensor has left dim {right_dim}")]
    BondDimensionMismatch {
        /// The site index where the mismatch occurred.
        site: usize,
        /// The right bond dimension of the left tensor.
        left_dim: usize,
        /// The left bond dimension of the right tensor.
        right_dim: usize,
    },

    /// Tensor train does not have a well-defined orthogonality center.
    #[error("Tensor train does not have a well-defined orthogonality center (ortho_lims = {start}..{end})")]
    NoOrthogonalityCenter {
        /// The start of the orthogonality limits range.
        start: usize,
        /// The end of the orthogonality limits range.
        end: usize,
    },

    /// Invalid tensor structure for tensor train.
    #[error("Invalid tensor structure: {message}")]
    InvalidStructure {
        /// A description of the structural issue.
        message: String,
    },

    /// Factorization error.
    #[error("Factorization error: {0}")]
    Factorize(#[from] tensor4all_core::FactorizeError),

    /// General operation error.
    #[error("Operation error: {message}")]
    OperationError {
        /// A description of the operation error.
        message: String,
    },
}

impl From<anyhow::Error> for TensorTrainError {
    fn from(source: anyhow::Error) -> Self {
        Self::OperationError {
            message: source.to_string(),
        }
    }
}
