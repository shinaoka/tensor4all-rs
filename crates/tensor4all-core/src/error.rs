//! Error types for tensor4all-core.
//!
//! [`MatrixCIError`] enumerates errors that can occur during matrix cross
//! interpolation operations. All public API functions return
//! [`Result<T>`](Result) which uses this error type.

use thiserror::Error;

/// Errors that can occur during matrix cross interpolation operations
///
/// # Examples
///
/// ```
/// use tensor4all_core::MatrixCIError;
///
/// let err = MatrixCIError::DimensionMismatch {
///     expected_rows: 3,
///     expected_cols: 3,
///     actual_rows: 2,
///     actual_cols: 2,
/// };
/// assert!(format!("{err}").contains("Dimension mismatch"));
///
/// let err = MatrixCIError::FullRank;
/// assert!(format!("{err}").contains("full rank"));
///
/// // Matching on error variants
/// match err {
///     MatrixCIError::FullRank => assert!(true),
///     _ => panic!("unexpected variant"),
/// }
/// ```
#[derive(Debug, Error)]
pub enum MatrixCIError {
    /// Dimension mismatch between matrix and MatrixCI object
    #[error("Dimension mismatch: expected ({expected_rows}, {expected_cols}), got ({actual_rows}, {actual_cols})")]
    DimensionMismatch {
        /// Expected number of rows
        expected_rows: usize,
        /// Expected number of columns
        expected_cols: usize,
        /// Actual number of rows
        actual_rows: usize,
        /// Actual number of columns
        actual_cols: usize,
    },

    /// Index out of bounds
    #[error(
        "Index out of bounds: ({row}, {col}) is out of bounds for a ({nrows}, {ncols}) matrix"
    )]
    IndexOutOfBounds {
        /// Row index that was out of bounds
        row: usize,
        /// Column index that was out of bounds
        col: usize,
        /// Number of rows in the matrix
        nrows: usize,
        /// Number of columns in the matrix
        ncols: usize,
    },

    /// Duplicate pivot row
    #[error("Cannot add pivot: row {row} already has a pivot")]
    DuplicatePivotRow {
        /// Row index that already has a pivot
        row: usize,
    },

    /// Duplicate pivot column
    #[error("Cannot add pivot: column {col} already has a pivot")]
    DuplicatePivotCol {
        /// Column index that already has a pivot
        col: usize,
    },

    /// Matrix is already full rank
    #[error("Cannot find a new pivot: matrix is already full rank")]
    FullRank,

    /// Empty row or column set
    #[error("Cannot find a new pivot in an empty set of {dimension}")]
    EmptyIndexSet {
        /// Description of the empty dimension ("rows" or "columns")
        dimension: String,
    },

    /// Invalid argument
    #[error("Invalid argument: {message}")]
    InvalidArgument {
        /// Description of the invalid argument
        message: String,
    },

    /// Singular matrix encountered
    #[error("Singular matrix encountered during decomposition")]
    SingularMatrix,

    /// Rank deficient matrix
    #[error("Rank-deficient matrix is not supported for this operation")]
    RankDeficient,

    /// NaN values encountered
    #[error("NaN values encountered in {matrix}")]
    NaNEncountered {
        /// Name of the matrix where NaN was encountered
        matrix: String,
    },
}

/// Result type for matrix CI operations
pub type Result<T> = std::result::Result<T, MatrixCIError>;
