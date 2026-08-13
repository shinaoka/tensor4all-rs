#![warn(missing_docs)]
//! Alternating Cross Interpolation for elementwise tensor-train operations.
//!
//! This crate ports the Rust public API for
//! `AlternatingCrossInterpolation.jl`, originally authored by
//! Marc Ritter <mritter@flatironinstitute.org> and contributors.
//!
//! ```
//! use tensor4all_aci::{elementwise, AciOptions};
//! use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
//!
//! let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
//! let result = elementwise(
//!     |xs: &[f64]| xs[0] * xs[1],
//!     &[a, b],
//!     &AciOptions::default(),
//! )
//! .unwrap();
//!
//! assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 8.0).abs() < 1e-10);
//! assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
//! ```

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod prelude;

mod batch;
mod elementwise;
mod error;
mod global_guard;
#[allow(dead_code)]
mod local;
mod options;
#[allow(dead_code)]
mod random_tt;
mod result;
#[allow(dead_code)]
mod scalar;
#[allow(dead_code)]
mod state;
#[allow(dead_code)]
pub(crate) mod validation;

pub use batch::ElementwiseBatch;
pub use elementwise::{elementwise, elementwise_batched};
pub use error::{AciError, Result};
#[allow(unused_imports)]
pub(crate) use local::LocalBlockEvaluator;
pub use options::AciOptions;
#[allow(unused_imports)]
pub(crate) use random_tt::initial_guess;
pub use result::{AciResult, AciTermination};
pub use scalar::AciScalar;
#[allow(unused_imports)]
pub(crate) use state::ElementwiseProblem;

#[cfg(test)]
mod tests;
