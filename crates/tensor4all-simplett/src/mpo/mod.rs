//! MPO (Matrix Product Operator) contraction algorithms
//!
//! This module provides algorithms for contracting two Matrix Product Operators (MPOs),
//! which are tensor trains with 4D site tensors of shape (left_bond, site_dim_1, site_dim_2, right_bond).
//!
//! # Main Types
//!
//! - [`MPO`]: Basic Matrix Product Operator representation
//! - [`SiteMPO`]: Center-canonical form with orthogonality center
//! - [`VidalMPO`]: Vidal canonical form with explicit singular values
//! - [`InverseMPO`]: Inverse form for efficient local updates
//!
//! # Contraction Algorithms
//!
//! - [`fn@contract_naive`]: Naive contraction (exact but memory-intensive)
//! - [`fn@contract_zipup`]: Zip-up contraction with on-the-fly compression
//! - [`fn@contract_fit`]: Reserved variational fitting API (currently
//!   unsupported)
//!
//! # Example
//!
//! ```
//! use tensor4all_simplett::mpo::{MPO, contract_naive, ContractionOptions};
//!
//! let mpo_a = MPO::constant(&[(2, 2), (2, 2)], 1.0_f64);
//! let mpo_b = MPO::constant(&[(2, 2), (2, 2)], 2.0_f64);
//! let result = contract_naive(&mpo_a, &mpo_b, Some(ContractionOptions::default())).unwrap();
//!
//! assert_eq!(result.len(), 2);
//! assert_eq!(result.site_dims(), vec![(2, 2), (2, 2)]);
//! ```

/// Type alias for 2D matrix.
pub type Matrix2<T> = crate::tensor::Tensor2<T>;

/// Helper function to create a zero-filled 2D tensor.
///
/// This is a shared utility used across multiple MPO modules.
#[inline]
pub(crate) fn matrix2_zeros<T: crate::traits::TTScalar + Default>(
    rows: usize,
    cols: usize,
) -> Matrix2<T> {
    crate::tensor::Tensor::from_elem([rows, cols], T::default())
}

pub(crate) mod canonical;
pub mod contract_fit;
pub mod contract_naive;
pub mod contract_zipup;
pub mod contraction;
pub mod dispatch;
pub mod environment;
pub mod error;
pub mod factorize;
pub mod inverse_mpo;
#[allow(clippy::module_inception)]
pub mod mpo;
pub mod site_mpo;
#[cfg(test)]
pub(crate) mod test_support;
pub mod tt_contraction;
pub mod types;
pub mod vidal_mpo;

// Re-export main types and functions
pub use contract_fit::{contract_fit, FitOptions};
pub use contract_naive::contract_naive;
pub use contract_zipup::contract_zipup;
pub use contraction::{Contraction, ContractionOptions};
pub use dispatch::{contract, ContractionAlgorithm};
pub use error::{MPOError, Result};
pub use factorize::{factorize, FactorizeMethod, FactorizeOptions, FactorizeResult};
pub use inverse_mpo::InverseMPO;
pub use mpo::MPO;
pub use site_mpo::SiteMPO;
pub use types::{tensor4_from_data, tensor4_zeros, Tensor4, Tensor4Ops};
pub use vidal_mpo::VidalMPO;
