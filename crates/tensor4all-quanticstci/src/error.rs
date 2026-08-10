//! Error types for quantics tensor cross interpolation operations.

use thiserror::Error;

/// Result type for quantics TCI operations.
pub type Result<T> = std::result::Result<T, QuanticsTCIError>;

/// Error returned by quantics TCI construction, evaluation, and grid operations.
#[derive(Debug, Error)]
pub enum QuanticsTCIError {
    /// Invalid quantics grid or interpolation configuration.
    #[error("invalid quantics configuration: {message}")]
    InvalidConfiguration {
        /// Description of the invalid configuration.
        message: String,
    },
    /// The requested operation requires a discretized (continuous) grid.
    #[error("original coordinates are only available for discretized grids")]
    DiscreteGridRequired,
    /// An underlying evaluation, coordinate conversion, or backend operation
    /// failed.
    #[error("quantics TCI operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for QuanticsTCIError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}
