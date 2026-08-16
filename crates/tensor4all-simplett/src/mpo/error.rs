//! Error types for MPO contraction operations

use thiserror::Error;

/// Result type for MPO operations
pub type Result<T> = std::result::Result<T, MPOError>;

/// Errors that can occur during MPO operations
#[derive(Error, Debug)]
pub enum MPOError {
    /// Dimension mismatch between tensors
    #[error("Dimension mismatch: tensor at site {site} has incompatible dimensions")]
    DimensionMismatch {
        /// The site index where the mismatch occurred
        site: usize,
    },

    /// Bond shape mismatch between adjacent tensors
    #[error("Bond shape mismatch at site {site}: left tensor has right_dim={left_right}, right tensor has left_dim={right_left}")]
    BondDimensionMismatch {
        /// The site index where the mismatch occurred
        site: usize,
        /// Right dimension of the left tensor
        left_right: usize,
        /// Left dimension of the right tensor
        right_left: usize,
    },

    /// Shared shape mismatch between two MPOs
    #[error("Shared shape mismatch at site {site}: MPO A has site_dim_2={dim_a}, MPO B has site_dim_1={dim_b}")]
    SharedDimensionMismatch {
        /// The site index where the mismatch occurred
        site: usize,
        /// Second site dimension of MPO A
        dim_a: usize,
        /// First site dimension of MPO B
        dim_b: usize,
    },

    /// Length mismatch between two MPOs
    #[error("MPO length mismatch: expected {expected}, got {got}")]
    LengthMismatch {
        /// The expected length
        expected: usize,
        /// The actual length provided
        got: usize,
    },

    /// Invalid index provided
    #[error("Index out of bounds: index {index} at site {site} (max: {max})")]
    IndexOutOfBounds {
        /// The site index where the error occurred
        site: usize,
        /// The invalid index value
        index: usize,
        /// The maximum allowed index value
        max: usize,
    },

    /// Empty MPO
    #[error("MPO is empty")]
    Empty,

    /// Flat tensor data did not match the requested shape.
    #[error("Tensor data length mismatch: expected {expected} elements, got {got}")]
    DataLengthMismatch {
        /// The element count implied by the requested shape.
        expected: usize,
        /// The number of elements supplied by the caller.
        got: usize,
    },

    /// Invalid boundary conditions
    #[error("Invalid boundary conditions: first tensor must have left_dim=1, last tensor must have right_dim=1")]
    InvalidBoundary,

    /// Invalid orthogonality center
    #[error("Invalid orthogonality center: {center} is out of range [0, {max})")]
    InvalidCenter {
        /// The invalid center value
        center: usize,
        /// The maximum allowed center value
        max: usize,
    },

    /// Factorization error
    #[error("Factorization failed: {message}")]
    FactorizationError {
        /// Description of the factorization failure
        message: String,
    },

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of the invalid operation
        message: String,
    },

    /// Matrix CI error
    #[error("Matrix CI error: {0}")]
    MatrixCI(#[from] tensor4all_core::MatrixCIError),

    /// Convergence failure
    #[error("Failed to converge after {sweeps} sweeps (final error: {error})")]
    ConvergenceFailure {
        /// The number of sweeps performed before failure
        sweeps: usize,
        /// The final error value achieved
        error: f64,
    },
}
