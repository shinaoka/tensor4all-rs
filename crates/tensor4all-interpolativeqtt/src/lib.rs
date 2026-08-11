#![warn(missing_docs)]
//! Interpolative Quantics Tensor Train construction.
//!
//! This crate ports the tested construction surface of
//! `InterpolativeQTT.jl` to Rust. Constructors return
//! [`SimpleTensorTrain`] values from `tensor4all-simplett`, so callers can use the
//! same evaluation, compression, and norm APIs as the other tensor4all crates.
//!
//! # Examples
//!
//! ```
//! use tensor4all_interpolativeqtt::{
//!     interpolate_single_scale, AbstractTensorTrain, InterpolativeQttOptions,
//! };
//!
//! let tt = interpolate_single_scale(
//!     |x| (-x * x).exp(),
//!     -1.0,
//!     1.0,
//!     4,
//!     10,
//!     &InterpolativeQttOptions::default(),
//! ).unwrap();
//!
//! assert_eq!(tt.len(), 4);
//! let value = tt.evaluate(&[0, 0, 0, 0]).unwrap();
//! assert!((value - (-1.0_f64).exp()).abs() < 1e-10);
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

mod basis;
mod error;
mod interpolation;
mod interval;
mod options;

pub use basis::{
    direct_product_core_tensors, get_chebyshev_grid, interpolation_tensor, LagrangePolynomials,
};
pub use error::{InterpolativeQttError, Result};
pub use interpolation::{
    estimate_interpolation_error, estimate_interpolation_error_nd, interpolate_adaptive,
    interpolate_adaptive_nd, interpolate_multi_scale, interpolate_multi_scale_nd,
    interpolate_single_scale, interpolate_single_scale_nd, interpolate_single_scale_sparse,
    interpolate_single_scale_sparse_nd, invert_qtt,
};
pub use options::InterpolativeQttOptions;
pub use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};

#[cfg(test)]
mod tests;
