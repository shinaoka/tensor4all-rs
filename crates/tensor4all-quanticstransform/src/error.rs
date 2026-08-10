//! Error types for quantics transform operators.

use thiserror::Error;

/// Error returned by quantics transform operator construction.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::QuanticsTransformError;
///
/// let err = QuanticsTransformError::from(anyhow::anyhow!("backend failed"));
/// assert!(err.to_string().contains("backend failed"));
/// ```
#[derive(Debug, Error)]
pub enum QuanticsTransformError {
    /// Invalid transform configuration or grid parameters.
    #[error("invalid quantics transform configuration: {message}")]
    InvalidConfiguration {
        /// Description of the invalid configuration.
        message: String,
    },
    /// An underlying tensor, operator, or backend operation failed.
    #[error("quantics transform operation failed: {source}")]
    Operation {
        /// Original diagnostic, preserving the full source chain.
        #[source]
        source: anyhow::Error,
    },
}

impl From<anyhow::Error> for QuanticsTransformError {
    fn from(source: anyhow::Error) -> Self {
        Self::Operation { source }
    }
}

impl From<tensor4all_core::TensorDynLenError> for QuanticsTransformError {
    fn from(source: tensor4all_core::TensorDynLenError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}

impl From<tensor4all_treetn::TreeTNOperationError> for QuanticsTransformError {
    fn from(source: tensor4all_treetn::TreeTNOperationError) -> Self {
        Self::Operation {
            source: anyhow::Error::new(source),
        }
    }
}
