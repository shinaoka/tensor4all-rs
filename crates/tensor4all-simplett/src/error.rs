//! Error types for tensor train operations

use thiserror::Error;

/// Result type for tensor train operations
pub type Result<T> = std::result::Result<T, TensorTrainError>;

/// Errors that can occur during tensor train operations
///
/// # Examples
///
/// ```
/// use tensor4all_simplett::{TensorTrainError, SimpleTensorTrain, AbstractTensorTrain};
///
/// // Empty index set triggers IndexLengthMismatch
/// let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 1.0);
/// let err = tt.evaluate(&[0]).unwrap_err();
/// assert!(matches!(err, TensorTrainError::IndexLengthMismatch { expected: 2, got: 1 }));
///
/// // DimensionMismatch can be constructed directly
/// let err = TensorTrainError::DimensionMismatch { site: 3 };
/// assert!(err.to_string().contains("site 3"));
///
/// // InvalidOperation carries an arbitrary message
/// let err = TensorTrainError::InvalidOperation {
///     message: "test error".to_string(),
/// };
/// assert!(err.to_string().contains("test error"));
/// ```
/// Error type for `SimpleTensorTrain` operations.
///
/// Note: `tensor4all-itensorlike` also defines a public type named
/// `TensorTrainError` (see its
/// [rustdoc](https://tensor4all.org/tensor4all-rs/rustdoc/tensor4all_itensorlike/error/enum.TensorTrainError.html))
/// with different variants (its tree-based `TensorTrain`). When both crates
/// are in scope, qualify the path (e.g. `tensor4all_simplett::TensorTrainError`
/// vs `tensor4all_itensorlike::TensorTrainError`).
#[derive(Error, Debug)]
pub enum TensorTrainError {
    /// Dimension mismatch between tensors
    #[error("Dimension mismatch: tensor at site {site} has incompatible dimensions")]
    DimensionMismatch {
        /// The site index where the mismatch occurred
        site: usize,
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

    /// Length mismatch in index set
    #[error("Index set length mismatch: expected {expected}, got {got}")]
    IndexLengthMismatch {
        /// The expected length
        expected: usize,
        /// The actual length provided
        got: usize,
    },

    /// Empty tensor train
    #[error("Tensor train is empty")]
    Empty,

    /// Flat tensor data did not match the requested shape.
    #[error("Tensor data length mismatch: expected {expected} elements, got {got}")]
    DataLengthMismatch {
        /// The element count implied by the requested shape.
        expected: usize,
        /// The number of elements supplied by the caller.
        got: usize,
    },

    /// Invalid operation
    #[error("Invalid operation: {message}")]
    InvalidOperation {
        /// Description of the invalid operation
        message: String,
    },

    /// Matrix CI error
    #[error("Matrix CI error: {0}")]
    MatrixCI(#[from] tensor4all_tcicore::MatrixCIError),
}
