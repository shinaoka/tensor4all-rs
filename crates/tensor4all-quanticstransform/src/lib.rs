#![warn(missing_docs)]
//! Quantics transformation operators for tensor train methods.
//!
//! This crate provides LinearOperator constructors for various transformations
//! in Quantics representation. It is a Rust port of transformation functionality
//! from [Quantics.jl](https://github.com/tensor4all/Quantics.jl).
//!
//! # Overview
//!
//! All transformations return `LinearOperator` from tensor4all-treetn for
//! consistent operator application to tensor train states.
//! Operators use their own input and output indices. Before applying an
//! operator to an existing state, replace the state's site indices with the
//! operator input indices from `get_input_mapping()`. The result then carries
//! the operator output indices.
//!
//! ## Available Transformations
//!
//! - **Flip**: f(x) = g(2^R - x)
//! - **Shift**: maps basis states as `|x> -> |x + offset>`; as a matrix action,
//!   `(M g)[x] = g[x - offset]` with the selected boundary condition
//! - **Phase Rotation**: f(x) = exp(i*θ*x) * g(x)
//! - **Cumulative Sum**: y_i = Σ_{j<i} x_j
//! - **Fourier Transform**: Quantics Fourier Transform (QFT)
//! - **Affine Transform**: y = A*x + b with rational coefficients
//! - **Constraint Row Normalization**: primitive integer rows for equality or
//!   halfspace constraints before deriving transform operators
//!
//! # Conventions
//!
//! - Quantics bits are big-endian by site: site 0 stores the most significant
//!   bit unless a function documents a different output ordering.
//! - The QFT output uses bit-reversed frequency order. A site configuration
//!   `(b_0, ..., b_{R-1})` corresponds to frequency
//!   `b_{R-1} * 2^{R-1} + ... + b_0`.
//! - Multi-variable operators use interleaved quantics encoding. At each bit
//!   site, the local index is `bit_var0 + 2 * bit_var1 + ...`.
//! - `AffineParams::a()` returns column-major coefficients:
//!   `[[1, 2], [3, 4]]` is stored as `[1, 3, 2, 4]`.
//! - Dense materialization is appropriate only for small reference checks and
//!   debugging. Production-size tensor-network workflows should use local
//!   operator application and scalable residual checks.
//!
//! # Errors
//!
//! Constructors return an error when dimensions are invalid, a bit count would
//! overflow an integer representation, an affine matrix/vector shape is
//! inconsistent, boundary-condition restrictions cannot be satisfied, or an
//! internal tensor-train/operator conversion fails.
//!
//! # Example
//!
//! ```
//! use tensor4all_quanticstransform::{flip_operator, BoundaryCondition};
//!
//! // Create a flip operator for 8-bit quantics representation
//! let op = flip_operator(8, BoundaryCondition::Periodic).unwrap();
//! assert_eq!(op.mpo().node_count(), 8);
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

mod affine;
mod common;
mod cumsum;
mod difference_kernel;
mod flip;
mod fourier;
mod phase_rotation;
mod shift;

mod error;

pub use affine::{
    affine_operator, affine_operator_interleaved, affine_transform_matrix,
    affine_transform_tensors_unfused, AffineParams, LinearConstraintRow, UnfusedTensorInfo,
};
pub use common::{
    identity_mpo, tensortrain_to_linear_operator, tensortrain_to_linear_operator_asymmetric,
    BoundaryCondition, CarryDirection, QuanticsOperator,
};
pub use cumsum::{cumsum_operator, triangle_operator, TriangleType};
pub use difference_kernel::{difference_kernel_mpo, difference_kernel_operator};
pub use error::QuanticsTransformError;
pub use flip::{flip_operator, flip_operator_multivar};
pub use fourier::{quantics_fourier_operator, FTCore, FourierOptions};
pub use phase_rotation::{phase_rotation_operator, phase_rotation_operator_multivar};
pub use shift::{shift_operator, shift_operator_multivar};
