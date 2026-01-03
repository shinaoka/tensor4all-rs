//! Error types for tensor4all-matrixci

use thiserror::Error;

/// Errors that can occur during matrix cross interpolation operations
#[derive(Debug, Error)]
pub enum MatrixCIError {
    /// Dimension mismatch between matrix and MatrixCI object
    #[error("Dimension mismatch: expected ({expected_rows}, {expected_cols}), got ({actual_rows}, {actual_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// Index out of bounds
    #[error("Index out of bounds: ({row}, {col}) is out of bounds for a ({nrows}, {ncols}) matrix")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        nrows: usize,
        ncols: usize,
    },

    /// Duplicate pivot row
    #[error("Cannot add pivot: row {row} already has a pivot")]
    DuplicatePivotRow { row: usize },

    /// Duplicate pivot column
    #[error("Cannot add pivot: column {col} already has a pivot")]
    DuplicatePivotCol { col: usize },

    /// Matrix is already full rank
    #[error("Cannot find a new pivot: matrix is already full rank")]
    FullRank,

    /// Empty row or column set
    #[error("Cannot find a new pivot in an empty set of {dimension}")]
    EmptyIndexSet { dimension: String },

    /// Invalid argument
    #[error("Invalid argument: {message}")]
    InvalidArgument { message: String },

    /// Singular matrix encountered
    #[error("Singular matrix encountered during decomposition")]
    SingularMatrix,

    /// Rank deficient matrix
    #[error("Rank-deficient matrix is not supported for this operation")]
    RankDeficient,

    /// NaN values encountered
    #[error("NaN values encountered in {matrix}")]
    NaNEncountered { matrix: String },
}

/// Result type for matrix CI operations
pub type Result<T> = std::result::Result<T, MatrixCIError>;
