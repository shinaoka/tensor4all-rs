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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_displays_are_stable() {
        assert_eq!(
            QuanticsTransformError::InvalidConfiguration {
                message: "theta must be finite".to_string(),
            }
            .to_string(),
            "invalid quantics transform configuration: theta must be finite"
        );
        assert_eq!(
            QuanticsTransformError::from(anyhow::anyhow!("backend failed")).to_string(),
            "quantics transform operation failed: backend failed"
        );
    }

    #[test]
    fn from_conversions_preserve_source() {
        let e = QuanticsTransformError::from(anyhow::anyhow!("root cause"));
        assert_eq!(
            <dyn std::error::Error>::source(&e).unwrap().to_string(),
            "root cause"
        );

        let dyn_err = QuanticsTransformError::from(tensor4all_core::TensorDynLenError::NaNInput {
            operation: "test",
        });
        assert!(dyn_err
            .to_string()
            .contains("quantics transform operation failed"));

        let tree_err = QuanticsTransformError::from(tensor4all_treetn::TreeTNOperationError::from(
            anyhow::anyhow!("tree root"),
        ));
        assert!(tree_err
            .to_string()
            .contains("quantics transform operation failed"));
    }
}
