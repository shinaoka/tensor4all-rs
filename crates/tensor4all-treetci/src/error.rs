//! Error types for tree tensor cross interpolation operations.

use thiserror::Error;

/// Result type for tree-TCI operations.
pub type Result<T> = std::result::Result<T, TreeTciError>;

/// Error returned by tree tensor cross interpolation construction and
/// traversal operations.
#[derive(Debug, Error)]
pub enum TreeTciError {
    /// The tree graph is invalid (disconnected, missing vertices, or
    /// inconsistent edge structure).
    #[error("invalid tree-TCI graph: {message}")]
    InvalidGraph {
        /// Description of the invalid graph structure.
        message: String,
    },
    /// An index or batch coordinate is out of bounds.
    #[error("index out of bounds: {message}")]
    IndexOutOfBounds {
        /// Description of the invalid index.
        message: String,
    },
    /// The interpolation failed to converge.
    #[error("tree-TCI interpolation failed to converge after {iterations} iterations")]
    ConvergenceFailure {
        /// Number of iterations attempted before failure.
        iterations: usize,
    },
    /// An underlying construction, materialization, or backend operation
    /// failed.
    #[error("tree-TCI operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for TreeTciError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}

impl From<tensor4all_core::TensorDynLenError> for TreeTciError {
    fn from(source: tensor4all_core::TensorDynLenError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}
