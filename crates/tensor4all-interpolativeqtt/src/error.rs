//! Error types for interpolative QTT construction.

use thiserror::Error;

/// Result type used by interpolative QTT APIs.
pub type Result<T> = std::result::Result<T, InterpolativeQttError>;

/// Error type for interpolative QTT construction.
///
/// # Examples
///
/// ```
/// use tensor4all_interpolativeqtt::{get_chebyshev_grid, InterpolativeQttError};
///
/// let err = get_chebyshev_grid(0).unwrap_err();
/// assert!(matches!(err, InterpolativeQttError::InvalidArgument { .. }));
/// ```
#[derive(Debug, Error)]
pub enum InterpolativeQttError {
    /// The caller supplied an invalid argument.
    #[error("invalid argument: {message}")]
    InvalidArgument {
        /// Human-readable explanation of the invalid argument.
        message: String,
    },

    /// Underlying tensor train operation failed.
    #[error(transparent)]
    SimpleTensorTrain(#[from] tensor4all_simplett::TensorTrainError),
}

pub(crate) fn invalid_argument(message: impl Into<String>) -> InterpolativeQttError {
    InterpolativeQttError::InvalidArgument {
        message: message.into(),
    }
}
