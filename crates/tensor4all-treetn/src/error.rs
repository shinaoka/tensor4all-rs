//! Public error types for tensor4all-treetn APIs.

use thiserror::Error;

/// Error returned when selecting external indices by numbered tags fails.
///
/// Use this for APIs that resolve tags such as `"k=1"`, `"k=2"`, ... into a
/// concrete ordered index list.
///
/// # Examples
/// ```
/// use tensor4all_treetn::NumberedTagSelectionError;
///
/// let err = NumberedTagSelectionError::MissingTag {
///     tag: "k=3".to_string(),
/// };
/// assert_eq!(err.to_string(), "no external index with tag k=3");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NumberedTagSelectionError {
    /// The tag prefix is malformed for numbered-tag lookup.
    #[error("invalid numbered tag prefix {tag_prefix}: prefix must not contain '='")]
    InvalidPrefix {
        /// Prefix provided by the caller.
        tag_prefix: String,
    },
    /// The requested numeric range overflows `usize`.
    #[error("numbered tag range overflows usize at offset {offset} from start {start_index}")]
    RangeOverflow {
        /// First requested numeric suffix.
        start_index: usize,
        /// Offset from `start_index` that overflowed.
        offset: usize,
    },
    /// A required numbered tag was not present on any external index.
    #[error("no external index with tag {tag}")]
    MissingTag {
        /// Missing tag, including the numeric suffix.
        tag: String,
    },
    /// A required numbered tag matched more than one external index.
    #[error("found more than one external index with tag {tag}")]
    AmbiguousTag {
        /// Ambiguous tag, including the numeric suffix.
        tag: String,
    },
}

/// Error returned when rebinding a [`LinearOperator`](crate::LinearOperator)'s
/// true input or output indices fails.
///
/// The error reports whether the failed binding belonged to the input or output
/// map through the `role` field.
///
/// # Examples
/// ```
/// use tensor4all_treetn::LinearOperatorIndexBindingError;
///
/// let err = LinearOperatorIndexBindingError::DimensionMismatch {
///     role: "input",
///     old_dim: 2,
///     new_dim: 3,
/// };
/// assert!(err.to_string().contains("input index shape mismatch"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorIndexBindingError {
    /// A replacement would change an index dimension.
    #[error("{role} index shape mismatch: old dimension {old_dim} vs new dimension {new_dim}")]
    DimensionMismatch {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Dimension of the source true index.
        old_dim: usize,
        /// Dimension of the requested replacement index.
        new_dim: usize,
    },
    /// The same source index was named more than once.
    #[error("duplicate {role} index binding for {index}")]
    DuplicateSourceIndex {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Debug representation of the duplicate source index.
        index: String,
    },
    /// The source index does not appear in the corresponding mapping.
    #[error("{role} index {index} not found in operator mappings")]
    MissingSourceIndex {
        /// Binding role, currently `"input"` or `"output"`.
        role: &'static str,
        /// Debug representation of the missing source index.
        index: String,
    },
}

/// Error returned when applying a linear operator after explicit index binding
/// fails.
///
/// This separates binding failures from failures in the underlying apply
/// algorithm.
///
/// # Examples
/// ```
/// use tensor4all_treetn::{
///     LinearOperatorIndexApplyError,
///     LinearOperatorIndexBindingError,
/// };
///
/// let err = LinearOperatorIndexApplyError::BindIndices(
///     LinearOperatorIndexBindingError::MissingSourceIndex {
///         role: "input",
///         index: "i".to_string(),
///     },
/// );
/// assert!(err.to_string().contains("failed to bind operator indices"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorIndexApplyError {
    /// Explicit true-index rebinding failed before application.
    #[error("failed to bind operator indices: {0}")]
    BindIndices(#[from] LinearOperatorIndexBindingError),
    /// The underlying operator application failed.
    #[error("failed to apply rebound linear operator: {message}")]
    ApplyFailed {
        /// Error message from the underlying apply path, including context.
        message: String,
    },
}

/// Error returned when applying a linear operator to state indices selected by
/// numbered tags.
///
/// This is used by [`apply_linear_operator_to_numbered_tags`](crate::apply_linear_operator_to_numbered_tags),
/// which resolves state indices tagged as `"k=1"`, `"k=2"`, ... and then uses
/// explicit index binding before applying the operator.
///
/// # Examples
/// ```
/// use tensor4all_treetn::LinearOperatorTaggedApplyError;
///
/// let err = LinearOperatorTaggedApplyError::MappingCountMismatch {
///     input_count: 2,
///     output_count: 3,
/// };
/// assert!(err.to_string().contains("input mappings"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LinearOperatorTaggedApplyError {
    /// State index selection from numbered tags failed.
    #[error("failed to select state indices by numbered tag: {0}")]
    SelectTaggedIndices(#[from] NumberedTagSelectionError),
    /// The operator does not have matching input and output mapping counts.
    #[error("operator has {input_count} input mappings but {output_count} output mappings")]
    MappingCountMismatch {
        /// Number of true input mappings found in operator node order.
        input_count: usize,
        /// Number of true output mappings found in operator node order.
        output_count: usize,
    },
    /// Explicit binding or application failed after tag selection.
    #[error("failed to apply operator to tagged indices: {0}")]
    ApplyToIndices(#[from] LinearOperatorIndexApplyError),
}

/// Error returned by selected-index TreeTN contraction helper APIs.
///
/// This covers helpers such as [`hadamard`](crate::hadamard),
/// [`weighted_sum_over_index_pairs`](crate::weighted_sum_over_index_pairs), and
/// [`sum_over_indices`](crate::sum_over_indices).
///
/// # Examples
/// ```
/// use tensor4all_treetn::SelectedIndexContractionError;
///
/// let err = SelectedIndexContractionError::DuplicateIndex {
///     index: "i".to_string(),
/// };
/// assert!(err.to_string().contains("duplicate index"));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SelectedIndexContractionError {
    /// The same selected index was provided more than once.
    #[error("sum_over_indices: duplicate index {index}")]
    DuplicateIndex {
        /// Debug representation of the duplicated index.
        index: String,
    },
    /// A selected index appeared in more than one site space.
    #[error("sum_over_indices: index {index} appears in more than one site space")]
    IndexInMultipleSiteSpaces {
        /// Debug representation of the repeated index.
        index: String,
    },
    /// A selected index was not external to the input TreeTN.
    #[error("sum_over_indices: index {index} not found in external indices")]
    IndexNotFound {
        /// Debug representation of the missing index.
        index: String,
    },
    /// Construction of a local unit-weight tensor failed.
    #[error("sum_over_indices: failed to build ones tensor at node {node}: {message}")]
    BuildOnesTensor {
        /// Debug representation of the node.
        node: String,
        /// Error message from the tensor constructor.
        message: String,
    },
    /// Construction of the weight TreeTN failed.
    #[error("sum_over_indices: failed to build ones TreeTN: {message}")]
    BuildWeightsTree {
        /// Error message from the TreeTN constructor.
        message: String,
    },
    /// The underlying partial contraction failed.
    #[error("selected-index contraction failed: {message}")]
    PartialContractFailed {
        /// Error message from the partial contraction path, including context.
        message: String,
    },
}

pub(crate) fn format_anyhow_error(error: anyhow::Error) -> String {
    format!("{error:#}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbered_tag_selection_error_messages_are_stable() {
        assert_eq!(
            NumberedTagSelectionError::InvalidPrefix {
                tag_prefix: "k=1".to_string(),
            }
            .to_string(),
            "invalid numbered tag prefix k=1: prefix must not contain '='"
        );
        assert_eq!(
            NumberedTagSelectionError::RangeOverflow {
                start_index: usize::MAX,
                offset: 1,
            }
            .to_string(),
            format!(
                "numbered tag range overflows usize at offset 1 from start {}",
                usize::MAX
            )
        );
        assert_eq!(
            NumberedTagSelectionError::MissingTag {
                tag: "k=3".to_string(),
            }
            .to_string(),
            "no external index with tag k=3"
        );
        assert_eq!(
            NumberedTagSelectionError::AmbiguousTag {
                tag: "k=1".to_string(),
            }
            .to_string(),
            "found more than one external index with tag k=1"
        );
    }

    #[test]
    fn linear_operator_binding_error_messages_are_stable() {
        assert_eq!(
            LinearOperatorIndexBindingError::DimensionMismatch {
                role: "input",
                old_dim: 2,
                new_dim: 3,
            }
            .to_string(),
            "input index shape mismatch: old dimension 2 vs new dimension 3"
        );
        assert_eq!(
            LinearOperatorIndexBindingError::DuplicateSourceIndex {
                role: "output",
                index: "i".to_string(),
            }
            .to_string(),
            "duplicate output index binding for i"
        );
        assert_eq!(
            LinearOperatorIndexBindingError::MissingSourceIndex {
                role: "input",
                index: "j".to_string(),
            }
            .to_string(),
            "input index j not found in operator mappings"
        );
    }

    #[test]
    fn linear_operator_apply_error_messages_are_stable() {
        assert_eq!(
            LinearOperatorIndexApplyError::BindIndices(
                LinearOperatorIndexBindingError::MissingSourceIndex {
                    role: "input",
                    index: "i".to_string(),
                }
            )
            .to_string(),
            "failed to bind operator indices: input index i not found in operator mappings"
        );
        assert_eq!(
            LinearOperatorIndexApplyError::ApplyFailed {
                message: "contraction failed".to_string(),
            }
            .to_string(),
            "failed to apply rebound linear operator: contraction failed"
        );
    }

    #[test]
    fn tagged_apply_error_messages_are_stable() {
        assert_eq!(
            LinearOperatorTaggedApplyError::SelectTaggedIndices(
                NumberedTagSelectionError::MissingTag {
                    tag: "x=2".to_string(),
                }
            )
            .to_string(),
            "failed to select state indices by numbered tag: no external index with tag x=2"
        );
        assert_eq!(
            LinearOperatorTaggedApplyError::MappingCountMismatch {
                input_count: 2,
                output_count: 3,
            }
            .to_string(),
            "operator has 2 input mappings but 3 output mappings"
        );
        assert_eq!(
            LinearOperatorTaggedApplyError::ApplyToIndices(
                LinearOperatorIndexApplyError::ApplyFailed {
                    message: "apply failed".to_string(),
                }
            )
            .to_string(),
            "failed to apply operator to tagged indices: failed to apply rebound linear operator: apply failed"
        );
    }

    #[test]
    fn selected_index_contraction_error_messages_are_stable() {
        assert_eq!(
            SelectedIndexContractionError::DuplicateIndex {
                index: "i".to_string(),
            }
            .to_string(),
            "sum_over_indices: duplicate index i"
        );
        assert_eq!(
            SelectedIndexContractionError::IndexInMultipleSiteSpaces {
                index: "i".to_string(),
            }
            .to_string(),
            "sum_over_indices: index i appears in more than one site space"
        );
        assert_eq!(
            SelectedIndexContractionError::IndexNotFound {
                index: "j".to_string(),
            }
            .to_string(),
            "sum_over_indices: index j not found in external indices"
        );
        assert_eq!(
            SelectedIndexContractionError::BuildOnesTensor {
                node: "v0".to_string(),
                message: "shape mismatch".to_string(),
            }
            .to_string(),
            "sum_over_indices: failed to build ones tensor at node v0: shape mismatch"
        );
        assert_eq!(
            SelectedIndexContractionError::BuildWeightsTree {
                message: "invalid tree".to_string(),
            }
            .to_string(),
            "sum_over_indices: failed to build ones TreeTN: invalid tree"
        );
        assert_eq!(
            SelectedIndexContractionError::PartialContractFailed {
                message: "zipup failed".to_string(),
            }
            .to_string(),
            "selected-index contraction failed: zipup failed"
        );
    }

    #[test]
    fn format_anyhow_error_preserves_context_chain() {
        let error = anyhow::anyhow!("root").context("outer context");

        assert_eq!(format_anyhow_error(error), "outer context: root");
    }
}

/// Error returned by general TreeTN and operator index operations.
///
/// Wraps the source diagnostic from the underlying tensor operation, so
/// callers propagating into `anyhow::Result` keep the full source chain.
#[derive(Debug, thiserror::Error)]
#[error("TreeTN operation failed: {source}")]
pub struct TreeTNOperationError {
    /// Original tensor/network diagnostic.
    #[source]
    pub source: anyhow::Error,
}

impl From<anyhow::Error> for TreeTNOperationError {
    fn from(source: anyhow::Error) -> Self {
        Self { source }
    }
}

impl From<tensor4all_core::TensorDynLenError> for TreeTNOperationError {
    fn from(source: tensor4all_core::TensorDynLenError) -> Self {
        Self {
            source: anyhow::Error::new(source),
        }
    }
}
