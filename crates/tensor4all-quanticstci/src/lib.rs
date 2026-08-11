#![warn(missing_docs)]
//! High-level Quantics TCI interface for function interpolation.
//!
//! This crate provides a user-friendly interface for interpolating functions
//! in Quantics Tensor Train (QTT) format. It wraps tensor4all-treetci and
//! quanticsgrids to provide seamless conversion between function domains
//! and quantics representations.
//!
//! This is a Rust port of [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl).
//!
//! # Important Conventions
//!
//! - **1-indexed grid indices**: All grid indices are **1-indexed** (following the Julia
//!   QuanticsTCI.jl convention). The first grid point is `[1, 1]`, not `[0, 0]`.
//! - **Equal dimensions**: [`quanticscrossinterpolate_discrete`] and
//!   [`quanticscrossinterpolate_from_arrays`] require all dimensions to have the **same**
//!   number of points (same power of 2). Use [`quanticscrossinterpolate`] with an explicit
//!   [`DiscretizedGrid`] for non-uniform grids.
//! - **Power-of-2 grid sizes**: All grid dimensions must be powers of 2.
//!
//! # Overview
//!
//! The main workflow is:
//! 1. Create a grid describing your function's domain
//! 2. Call [`quanticscrossinterpolate`] (or the `_discrete` / `_from_arrays` variants)
//! 3. Use the resulting [`QuanticsTensorCI2`] for evaluation, integration, etc.
//!
//! # Example: Discrete Grid
//!
//! ```rust
//! use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions};
//!
//! // Interpolate f(i, j) = i + j on a 16x16 grid
//! // Discrete indices are 1-indexed and passed as `&[i64]`.
//! let f = |idx: &[i64]| (idx[0] + idx[1]) as f64;
//! let sizes = vec![16, 16];
//!
//! let (qtci, _ranks, _errors) = quanticscrossinterpolate_discrete(
//!     &sizes,
//!     f,
//!     None,
//!     QtciOptions::default(),
//! ).unwrap();
//!
//! let value = qtci.evaluate(&[5, 10]).unwrap();  // 1-indexed
//! assert!((value - 15.0).abs() < 1e-10);
//! ```
//!
//! # Example: Continuous Grid with `DiscretizedGrid`
//!
//! ```rust
//! use tensor4all_quanticstci::{
//!     quanticscrossinterpolate, DiscretizedGrid, QtciOptions,
//! };
//!
//! let grid = DiscretizedGrid::builder(&[4])  // 2^4 = 16 points
//!     .with_lower_bound(&[0.0])
//!     .with_upper_bound(&[1.0])
//!     .build()
//!     .unwrap();
//!
//! let f = |x: &[f64]| x[0] * x[0];  // f(x) = x^2
//!
//! let (qtci, _ranks, _errors) = quanticscrossinterpolate(
//!     &grid,
//!     f,
//!     None,
//!     QtciOptions::default(),
//! ).unwrap();
//!
//! // integral() = sum * step_size (left Riemann sum of x^2 over [0, 1))
//! let integral = qtci.integral().unwrap();
//! assert!((integral - 1.0 / 3.0).abs() < 0.1); // rough Riemann sum with 16 points
//! ```
//!
//! # Choosing the Right API
//!
//! | Scenario | Function to use |
//! |---|---|
//! | Function on integer grid (e.g., lattice) | [`quanticscrossinterpolate_discrete`] |
//! | Function on a continuous interval `[a, b)` | [`quanticscrossinterpolate`] with [`DiscretizedGrid`] |
//! | Grid points given as explicit arrays | [`quanticscrossinterpolate_from_arrays`] |
//! | Vector/tensor-valued function | [`quanticscrossinterpolate_batched`] |

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

mod batched;
mod error;
mod options;
mod quantics_tci;

pub use batched::{quanticscrossinterpolate_batched, QuanticsTensorCI2Batched};
pub use error::QuanticsTCIError;
pub use options::QtciOptions;
pub use quantics_tci::{
    quanticscrossinterpolate, quanticscrossinterpolate_discrete,
    quanticscrossinterpolate_from_arrays, QuanticsTensorCI2,
};

// Re-export commonly used types from dependencies
pub use quanticsgrids::{DiscretizedGrid, InherentDiscreteGrid, UnfoldingScheme};
pub use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
pub use tensor4all_treetci::{DefaultProposer, TreeTciGraph, TreeTciOptions};
