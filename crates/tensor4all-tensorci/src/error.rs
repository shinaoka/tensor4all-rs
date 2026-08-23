//! Error types for tensor cross interpolation operations

use thiserror::Error;

/// Result type for TCI operations
pub type Result<T> = std::result::Result<T, TCIError>;

pub(crate) fn validate_nonnegative_finite(name: &str, value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(TCIError::InvalidConfiguration {
            message: format!("{name} must be finite and nonnegative"),
        });
    }
    Ok(())
}

pub(crate) fn validate_positive(name: &str, value: usize) -> Result<()> {
    if value == 0 {
        return Err(TCIError::InvalidConfiguration {
            message: format!("{name} must be positive"),
        });
    }
    Ok(())
}

/// Errors that can occur during tensor cross interpolation operations
#[derive(Error, Debug)]
pub enum TCIError {
    /// Invalid algorithm configuration.
    #[error("Invalid configuration: {message}")]
    InvalidConfiguration {
        /// Description of the invalid option value.
        message: String,
    },

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
    MatrixCIError(#[from] tensor4all_core::MatrixCIError),

    /// Positional tensor train error.
    #[error("simple tensor train error: {0}")]
    SimpleTensorTrain(#[from] tensor4all_simplett::SimpleTensorTrainError),
}
