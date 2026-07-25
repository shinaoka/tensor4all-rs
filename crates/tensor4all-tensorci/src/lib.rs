#![warn(missing_docs)]
//! Tensor Cross Interpolation (TCI) library
//!
//! This crate provides tensor cross interpolation algorithms for efficiently
//! approximating high-dimensional tensors as tensor trains. Given a function
//! `f(i_1, ..., i_N)` defined on a discrete multi-index grid, TCI finds a
//! low-rank [`TensorTrain`](tensor4all_simplett::TensorTrain) approximation by
//! evaluating only a small subset of the total entries.
//!
//! # Algorithms
//!
//! | Entry point | Algorithm | State type | Notes |
//! |---|---|---|---|
//! | [`crossinterpolate2`] | TCI2 (two-site) | [`TensorCI2`] | **Primary, actively maintained** |
//! | [`crossinterpolate1`] | TCI1 (one-site) | [`TensorCI1`] | Legacy parity with TensorCrossInterpolation.jl |
//!
//! `TCI2` uses [`MatrixLUCI`](tensor4all_tcicore::MatrixLUCI) for pivot
//! updates and supports batch evaluation, global pivot search, and two pivot
//! search strategies ([`PivotSearchStrategy::Full`] and
//! [`PivotSearchStrategy::Rook`]).
//! `TCI1` exposes the legacy one-site state, local/global pivot updates, and
//! sweep-direction control through [`TCI1SweepStrategy`].
//!
//! # Quick start
//!
//! ```
//! use tensor4all_tensorci::{crossinterpolate2, TCI2Options};
//! use tensor4all_simplett::AbstractTensorTrain;
//!
//! // Function to interpolate: f(i, j) = i + j + 1
//! let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
//! let local_dims = vec![4, 4];
//! let first_pivot = vec![vec![1, 1]];
//!
//! let result = crossinterpolate2::<f64, _, fn(&[Vec<usize>]) -> Vec<f64>>(
//!     f,
//!     None,
//!     local_dims,
//!     first_pivot,
//!     TCI2Options::default(),
//! )
//! .unwrap();
//!
//! // Check convergence
//! assert_eq!(result.termination, tensor4all_tensorci::TCI2Termination::Converged);
//!
//! // Evaluate through the tensor train
//! let tt = result.tci.to_tensor_train().unwrap();
//! let val = tt.evaluate(&[2, 3]).unwrap();
//! assert!((val - 6.0).abs() < 1e-10); // f(2,3) = 2+3+1 = 6
//! ```
//!
//! # Related crates
//!
//! - [`tensor4all_tcicore`] -- low-level matrix CI primitives and
//!   [`CachedFunction`](tensor4all_tcicore::CachedFunction)
//! - [`tensor4all_simplett`] -- the [`TensorTrain`](tensor4all_simplett::TensorTrain)
//!   data structure produced by TCI
//! - `tensor4all-quanticstci` -- higher-level quantics TCI on discrete
//!   or continuous grids (wraps this crate)

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod conversion;
pub mod error;
pub mod globalpivot;
pub mod globalsearch;
pub mod integration;
pub mod optfirstpivot;
pub mod tensorci1;
pub mod tensorci2;

// Re-export main types
pub use conversion::TensorCI2FromTensorTrainOptions;
pub use error::{Result, TCIError};
pub use globalpivot::{DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput};
pub use globalsearch::{estimate_true_error, floating_zone};
pub use optfirstpivot::opt_first_pivot;
pub use tensorci1::{crossinterpolate1, TCI1Options, TCI1SweepStrategy, TensorCI1};
pub use tensorci2::{
    crossinterpolate2, optimize_with_finder, PivotSearchStrategy, Sweep2Strategy,
    TCI2OptimizationResult, TCI2Options, TCI2Termination, TensorCI2,
};
