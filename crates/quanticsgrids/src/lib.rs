//! Quantics grid structures for efficient conversion between quantics indices,
//! grid indices, and original coordinates.
//!
//! This crate is a Rust port of [QuanticsGrids.jl](https://github.com/tensor4all/QuanticsGrids.jl)
//! and provides infrastructure for quantics tensor train (QTT) methods.
//!
//! # Overview
//!
//! This crate provides two main grid types:
//! - [`InherentDiscreteGrid`]: Low-level grid working with integer coordinates
//! - [`DiscretizedGrid`]: High-level grid with continuous domain and floating-point coordinates
//!
//! ## Coordinate Systems
//!
//! The grids support conversion between three coordinate representations:
//!
//! 1. **Quantics indices**: Integer values for each tensor core (1-indexed)
//! 2. **Grid indices**: Integer positions in each dimension (1-indexed)
//! 3. **Original coordinates**: Continuous values in the specified domain
//!
//! ## Unfolding Schemes
//!
//! Two tensor train layouts are supported via [`UnfoldingScheme`]:
//!
//! - **Fused** (default): Indices at the same bit level are grouped together
//! - **Interleaved**: Indices alternate between dimensions
//!
//! # Quick Start
//!
//! ```
//! use quanticsgrids::{DiscretizedGrid, UnfoldingScheme};
//!
//! // Create a 2D grid with resolution (3, 2) bits
//! // This discretizes [0, 1) x [0, 1) into 8 x 4 = 32 grid points
//! let grid = DiscretizedGrid::builder(&[3, 2])
//!     .with_variable_names(&["x", "y"])
//!     .build()
//!     .unwrap();
//!
//! // Convert coordinates to quantics indices
//! let quantics = grid.origcoord_to_quantics(&[0.5, 0.25]).unwrap();
//!
//! // Convert back
//! let grididx = grid.quantics_to_grididx(&quantics).unwrap();
//! let coords = grid.grididx_to_origcoord(&grididx).unwrap();
//! assert!((coords[0] - 0.5).abs() < 1e-10);
//! ```
//!
//! # Custom Domains
//!
//! ```
//! use quanticsgrids::DiscretizedGrid;
//!
//! // Grid over [-1, 1) x [0, 2*pi)
//! let grid = DiscretizedGrid::builder(&[8, 8])
//!     .with_lower_bound(&[-1.0, 0.0])
//!     .with_upper_bound(&[1.0, std::f64::consts::TAU])
//!     .with_variable_names(&["x", "theta"])
//!     .build()
//!     .unwrap();
//!
//! let step = grid.grid_step();
//! assert!((step[0] - 2.0 / 256.0).abs() < 1e-10);
//! ```
//!
//! # Custom Index Tables
//!
//! For advanced use cases, you can specify a custom tensor train structure:
//!
//! ```
//! use quanticsgrids::DiscretizedGrid;
//!
//! // Custom index table: [[(:a, 1), (:b, 2)], [(:a, 2)], [(:b, 1), (:a, 3)]]
//! let index_table = vec![
//!     vec![("a".to_string(), 1), ("b".to_string(), 2)],
//!     vec![("a".to_string(), 2)],
//!     vec![("b".to_string(), 1), ("a".to_string(), 3)],
//! ];
//!
//! let grid = DiscretizedGrid::from_index_table(&["a", "b"], index_table)
//!     .build()
//!     .unwrap();
//!
//! assert_eq!(grid.rs(), &[3, 2]); // a has 3 bits, b has 2 bits
//! assert_eq!(grid.len(), 3);      // 3 tensor cores
//! ```
//!
//! # Using with Functions
//!
//! ```
//! use quanticsgrids::{DiscretizedGrid, quantics_function};
//!
//! let grid = DiscretizedGrid::builder(&[8]).build().unwrap();
//!
//! // Wrap a function to accept quantics indices
//! let f = |x: &[f64]| x[0] * x[0];
//! let qf = quantics_function(&grid, f);
//!
//! let quantics = grid.origcoord_to_quantics(&[0.5]).unwrap();
//! let result = qf(&quantics).unwrap();
//! assert!((result - 0.25).abs() < 1e-10);
//! ```
//!
//! # Performance
//!
//! All conversion functions have O(R * D) time complexity where:
//! - R is the maximum resolution (bits) across dimensions
//! - D is the number of dimensions
//!
//! Typical performance (single conversion):
//! - 1D grids: ~30-80 ns
//! - 2D grids: ~40-100 ns
//! - With floating-point conversion: ~100-120 ns
//!
//! # Error Handling
//!
//! All conversion functions return [`Result`] with [`QuanticsGridError`]:
//!
//! ```
//! use quanticsgrids::{DiscretizedGrid, QuanticsGridError};
//!
//! let grid = DiscretizedGrid::builder(&[2]).build().unwrap();
//!
//! // Coordinate out of bounds
//! let result = grid.origcoord_to_grididx(&[1.5]);
//! assert!(matches!(result, Err(QuanticsGridError::CoordinateOutOfBounds { .. })));
//!
//! // Grid index out of bounds
//! let result = grid.grididx_to_quantics(&[10]);
//! assert!(matches!(result, Err(QuanticsGridError::GridIndexOutOfBounds { .. })));
//! ```

mod error;
mod inherent_discrete_grid;
mod discretized_grid;

pub use error::{QuanticsGridError, Result};
pub use inherent_discrete_grid::{InherentDiscreteGrid, InherentDiscreteGridBuilder};
pub use discretized_grid::{quantics_function, DiscretizedGrid, DiscretizedGridBuilder};

/// Unfolding scheme for tensor train structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnfoldingScheme {
    /// Each quantics index gets its own tensor core, alternating between variables.
    /// For variables (a, b) with Rs=(2, 2), produces: [a1], [b1], [a2], [b2]
    Interleaved,
    /// Quantics indices at the same bit level are fused into one tensor core.
    /// For variables (a, b) with Rs=(2, 2), produces: [b1, a1], [b2, a2]
    #[default]
    Fused,
}

/// A quantics index entry: (variable_name, bit_number)
/// bit_number is 1-indexed (1 = most significant bit)
pub type QuanticsIndex = (String, usize);

/// Index table: structure defining tensor train layout
/// Each inner Vec represents one tensor core, containing the quantics indices it holds
pub type IndexTable = Vec<Vec<QuanticsIndex>>;

/// Lookup entry: (site_index, position_in_site)
/// site_index is 0-indexed, position_in_site is 0-indexed
pub type LookupEntry = (usize, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unfolding_scheme_default() {
        assert_eq!(UnfoldingScheme::default(), UnfoldingScheme::Fused);
    }
}
