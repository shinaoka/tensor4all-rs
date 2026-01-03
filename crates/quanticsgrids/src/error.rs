//! Error types for quanticsgrids

use thiserror::Error;

/// Result type for quanticsgrids operations
pub type Result<T> = std::result::Result<T, QuanticsGridError>;

/// Errors that can occur during quantics grid operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum QuanticsGridError {
    /// Base must be at least 2
    #[error("Base must be at least 2, got {0}")]
    InvalidBase(usize),

    /// Step must be at least 1
    #[error("Step for dimension {dim} must be at least 1, got {value}")]
    InvalidStep { dim: usize, value: i64 },

    /// Resolution too large (base^R would overflow)
    #[error("Resolution {r} with base {base} is too large (base^R would overflow i64)")]
    ResolutionTooLarge { r: usize, base: usize },

    /// Variable names must be unique
    #[error("Variable names must be unique, found duplicate: {0}")]
    DuplicateVariableName(String),

    /// Index table contains unknown variable
    #[error("Index table contains unknown variable '{variable}'. Valid variables: {valid:?}")]
    UnknownVariable { variable: String, valid: Vec<String> },

    /// Index table contains invalid bit number
    #[error("Bit number {bit} for variable '{variable}' exceeds resolution {resolution}")]
    InvalidBitNumber {
        variable: String,
        bit: usize,
        resolution: usize,
    },

    /// Index table contains duplicate entry
    #[error("Index table contains duplicate entry for variable '{variable}' bit {bit}")]
    DuplicateIndexEntry { variable: String, bit: usize },

    /// Index table missing entry
    #[error("Index table missing entry for variable '{variable}' bit {bit}")]
    MissingIndexEntry { variable: String, bit: usize },

    /// Quantics vector has wrong length
    #[error("Quantics vector must have length {expected}, got {actual}")]
    WrongQuanticsLength { expected: usize, actual: usize },

    /// Quantics value out of range
    #[error("Quantics value {value} for site {site} out of range [1, {max}]")]
    QuanticsOutOfRange { site: usize, value: i64, max: i64 },

    /// Grid index out of bounds
    #[error("Grid index {value} for dimension {dim} out of bounds [1, {max}]")]
    GridIndexOutOfBounds { dim: usize, value: i64, max: i64 },

    /// Coordinate out of bounds
    #[error("Coordinate {value} for dimension {dim} out of bounds [{lower}, {upper}]")]
    CoordinateOutOfBounds {
        dim: usize,
        value: f64,
        lower: f64,
        upper: f64,
    },

    /// Site index out of bounds
    #[error("Site index {site} out of bounds [0, {max})")]
    SiteIndexOutOfBounds { site: usize, max: usize },

    /// Lower bound must be less than upper bound
    #[error("Lower bound {lower} must be less than upper bound {upper} for dimension {dim}")]
    InvalidBounds { dim: usize, lower: f64, upper: f64 },

    /// Cannot include endpoint with zero resolution
    #[error("Cannot include endpoint for dimension {dim} with zero resolution")]
    EndpointWithZeroResolution { dim: usize },

    /// Dimension mismatch
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// No resolutions specified
    #[error("At least one resolution must be specified")]
    NoResolutions,
}
