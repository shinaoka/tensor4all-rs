//! Tensor Cross Interpolation (TCI) library
//!
//! This crate provides tensor cross interpolation algorithms for efficiently
//! approximating high-dimensional tensors as tensor trains.
//!
//! # Main algorithms
//!
//! - `TensorCI1`: One-site TCI algorithm
//! - `crossinterpolate1`: Function to perform TCI1 interpolation
//!
//! # Example
//!
//! ```
//! use tensor4all_tensorci::{crossinterpolate1, TCI1Options};
//!
//! // Function to interpolate: f(i, j) = i + j + 1
//! let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
//! let local_dims = vec![4, 4];
//! let first_pivot = vec![1, 1];
//!
//! let (tci, ranks, errors) = crossinterpolate1(f, local_dims, first_pivot, TCI1Options::default()).unwrap();
//! println!("TCI rank: {}", tci.rank());
//! ```

pub mod cached_function;
pub mod error;
pub mod indexset;
pub mod tensorci1;
pub mod tensorci2;

// Re-export main types
pub use cached_function::CachedFunction;
pub use error::{Result, TCIError};
pub use indexset::{IndexSet, LocalIndex, MultiIndex};
pub use tensorci1::{crossinterpolate1, SweepStrategy, TCI1Options, TensorCI1};
pub use tensorci2::{crossinterpolate2, PivotSearchStrategy, TCI2Options, TensorCI2};
