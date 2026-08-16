//! Core data types for the matrixluci substrate.
//!
//! These types are the building blocks for pivot-selection kernels.

/// Options for pivot-kernel factorization.
///
/// Controls rank truncation, tolerance thresholds, and the normalization
/// convention (left-orthogonal vs. right-orthogonal).
#[derive(Debug, Clone)]
pub(crate) struct PivotKernelOptions {
    /// Relative tolerance.
    pub rel_tol: f64,
    /// Absolute tolerance.
    pub abs_tol: f64,
    /// Maximum rank.
    pub max_bond_dim: usize,
    /// Whether the left factor is unit-diagonal.
    #[allow(dead_code)]
    pub left_orthogonal: bool,
}

/// Result of pivot selection: chosen row/column indices, rank, and error history.
#[derive(Debug, Clone)]
pub(crate) struct PivotSelectionCore {
    /// Selected row indices.
    pub row_indices: Vec<usize>,
    /// Selected column indices.
    pub col_indices: Vec<usize>,
    /// Pivot error history.
    pub pivot_errors: Vec<f64>,
    /// Selected rank.
    pub rank: usize,
}

impl PivotKernelOptions {
    /// Canonical options for dense no-truncation behavior.
    #[cfg(test)]
    pub fn no_truncation() -> Self {
        Self {
            rel_tol: 0.0,
            abs_tol: 0.0,
            max_bond_dim: usize::MAX,
            left_orthogonal: true,
        }
    }
}

impl Default for PivotKernelOptions {
    fn default() -> Self {
        Self {
            rel_tol: 1e-14,
            abs_tol: 0.0,
            max_bond_dim: usize::MAX,
            left_orthogonal: true,
        }
    }
}
