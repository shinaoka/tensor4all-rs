#![warn(missing_docs)]
//! TCI Core infrastructure
//!
//! Shared foundation for tensor cross interpolation algorithms.
//!
//! This crate provides three categories of functionality:
//!
//! - **Matrix cross interpolation**: [`MatrixLUCI`] (LU-based), [`MatrixACA`]
//!   (adaptive cross approximation), and [`RrLU`] (rank-revealing LU
//!   decomposition). All CI types implement the [`AbstractMatrixCI`] trait.
//! - **Cached function evaluation**: [`CachedFunction`] wraps expensive
//!   multi-index functions with thread-safe memoization and automatic key
//!   type selection (up to 1024 bits by default, extensible via [`CacheKey`]).
//! - **Utility types**: [`IndexSet`] for bidirectional pivot management and
//!   [`Scalar`] for numeric scalar requirements.
//!
//! Higher-level crates (`tensor4all-tensorci`, `tensor4all-quanticstci`) build
//! on these primitives.
//!
//! # Examples
//!
//! Cross-interpolate a rank-2 matrix:
//!
//! ```
//! use tensor4all_tcicore::{AbstractMatrixCI, MatrixLUCI};
//! use tensor4all_tensorbackend::from_vec2d;
//!
//! let m = from_vec2d(vec![
//!     vec![1.0_f64, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ]);
//!
//! let ci = MatrixLUCI::from_matrix(&m, None).unwrap();
//! assert!(ci.rank() <= 3);
//! assert!(ci.rank() >= 1);
//!
//! // Reconstruction matches original at pivot positions
//! for &i in ci.row_indices() {
//!     for &j in ci.col_indices() {
//!         assert!((ci.evaluate(i, j) - m[[i, j]]).abs() < 1e-10);
//!     }
//! }
//! ```
//!
//! Cache an expensive function:
//!
//! ```
//! use tensor4all_tcicore::CachedFunction;
//!
//! let cf = CachedFunction::new(
//!     |idx: &[usize]| (idx[0] as f64) * (idx[1] as f64),
//!     &[3, 4],
//! ).unwrap();
//!
//! assert_eq!(cf.eval(&[2, 3]), 6.0);
//! assert_eq!(cf.num_evals(), 1);
//!
//! // Second call hits cache
//! assert_eq!(cf.eval(&[2, 3]), 6.0);
//! assert_eq!(cf.num_cache_hits(), 1);
//! ```

pub mod cached_function;
pub mod error;
pub mod indexset;
mod matrix_luci;
pub mod matrixaca;
pub mod matrixlu;
pub mod matrixluci;
pub mod scalar;
pub mod traits;

pub use self::matrixluci::Scalar as MatrixLuciScalar;
pub use cached_function::cache_key::CacheKey;
pub use cached_function::error::CacheKeyError;
pub use cached_function::index_int::IndexInt;
pub use cached_function::CachedFunction;
pub use error::{MatrixCIError, Result};
pub use indexset::{IndexSet, LocalIndex, MultiIndex};
pub use matrix_luci::{
    matrix_luci_factors_from_blocks, matrix_luci_factors_from_matrix,
    matrix_luci_factors_from_matrix_owned, MatrixLUCI, MatrixLuciFactors,
};
pub use matrixaca::MatrixACA;
pub use matrixlu::{rrlu, rrlu_mut, RrLU, RrLUOptions};
pub use scalar::Scalar;
pub use traits::AbstractMatrixCI;
