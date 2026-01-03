//! Error types for tensor train operations

use thiserror::Error;

/// Result type for tensor train operations
pub type Result<T> = std::result::Result<T, TensorTrainError>;

/// Errors that can occur during tensor train operations
#[derive(Error, Debug)]
pub enum TensorTrainError {
    /// Dimension mismatch between tensors
    #[error("Dimension mismatch: tensor at site {site} has incompatible dimensions")]
    DimensionMismatch { site: usize },

    /// Invalid index provided
    #[error("Index out of bounds: index {index} at site {site} (max: {max})")]
    IndexOutOfBounds { site: usize, index: usize, max: usize },

    /// Length mismatch in index set
    #[error("Index set length mismatch: expected {expected}, got {got}")]
    IndexLengthMismatch { expected: usize, got: usize },

    /// Empty tensor train
    #[error("Tensor train is empty")]
    Empty,

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation { message: String },
}
