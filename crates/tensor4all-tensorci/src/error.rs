//! Error types for tensor cross interpolation operations

use thiserror::Error;

/// Result type for TCI operations
pub type Result<T> = std::result::Result<T, TCIError>;

/// Errors that can occur during tensor cross interpolation operations
#[derive(Error, Debug)]
pub enum TCIError {
    /// Dimension mismatch
    #[error("Dimension mismatch: {message}")]
    DimensionMismatch {
        /// Description of the shape mismatch
        message: String,
    },

    /// Invalid index
    #[error("Index out of bounds: {message}")]
    IndexOutOfBounds {
        /// Description of the index error
        message: String,
    },

    /// Invalid pivot
    #[error("Invalid pivot: {message}")]
    InvalidPivot {
        /// Description of the invalid pivot
        message: String,
    },

    /// Convergence failure
    #[error("Failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations before failure
        iterations: usize,
    },

    /// Empty tensor train
    #[error("Empty tensor structure")]
    Empty,

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of the invalid operation
        message: String,
    },
    /// Internal index inconsistency
    #[error("Index inconsistency: {message}")]
    IndexInconsistency {
        /// Description of the inconsistency
        message: String,
    },

    /// Matrix CI error
    #[error("Matrix CI error: {0}")]
    MatrixCIError(#[from] tensor4all_tcicore::MatrixCIError),

    /// Tensor train error
    #[error("Tensor train error: {0}")]
    TensorTrainError(#[from] tensor4all_simplett::TensorTrainError),
}
