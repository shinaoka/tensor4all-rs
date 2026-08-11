//! Error types for TensorTrain operations.

use std::sync::Arc;

use thiserror::Error;

/// Result type for TensorTrain operations.
pub type Result<T> = std::result::Result<T, TensorTrainError>;

use tensor4all_core::TensorDynLenError;

/// Errors that can occur in `TensorTrain` operations.
///
/// Note: `tensor4all-simplett` also defines a public type named
/// [`TensorTrainError`](tensor4all_simplett::TensorTrainError) with different
/// variants (its positional `SimpleTensorTrain`). When both crates are in
/// scope, qualify the path (e.g. `tensor4all_itensorlike::TensorTrainError`
/// vs `tensor4all_simplett::TensorTrainError`).
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

    /// Bond shape mismatch between adjacent tensors.
    #[error("Bond shape mismatch at site {site}: left tensor has right dim {left_dim}, right tensor has left dim {right_dim}")]
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

    /// A typed TensorDynLen metric or materialization error.
    #[error("TensorDynLen operation error: {source}")]
    TensorDynLen {
        /// The typed tensor diagnostic.
        #[source]
        source: tensor4all_core::TensorDynLenError,
    },

    /// General operation error.
    #[error("Operation error: {message}")]
    OperationError {
        /// A description of the operation error.
        message: String,
    },

    /// An operation error retaining an owned backend source diagnostic.
    #[error("Operation error: {message}: {source}")]
    OperationErrorSource {
        /// Context describing the failed operation.
        message: String,
        /// Original operation diagnostic.
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

impl From<anyhow::Error> for TensorTrainError {
    fn from(source: anyhow::Error) -> Self {
        Self::operation_source("TensorTrain operation failed", source)
    }
}

impl From<TensorDynLenError> for TensorTrainError {
    fn from(source: TensorDynLenError) -> Self {
        Self::TensorDynLen { source }
    }
}

impl TensorTrainError {
    pub(crate) fn operation_source(message: impl Into<String>, source: anyhow::Error) -> Self {
        Self::OperationErrorSource {
            message: message.into(),
            source: Arc::from(source.into_boxed_dyn_error()),
        }
    }
}
