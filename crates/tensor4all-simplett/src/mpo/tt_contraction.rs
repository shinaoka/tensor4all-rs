//! Contraction operations for tensor trains (TT-TT)
//!
//! This module re-exports TT contraction operations from `tensor4all-simplett`.
//! These operations are conceptually MPO-MPO contractions where the MPO
//! has trivial (dimension 1) "operator" indices.
//!
//! # Available operations
//!
//! - [`inner_product`]: Inner product (returns scalar)
//!
//! # Example
//!
//! ```
//! use tensor4all_simplett::mpo::tt_contraction::{inner_product, SimpleTensorTrain};
//!
//! let tt1 = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let tt2 = SimpleTensorTrain::<f64>::constant(&[2, 3], 3.0);
//!
//! // Inner product
//! let inner = inner_product(&tt1, &tt2).unwrap();
//! assert_eq!(inner, 36.0);
//! ```

// Re-export TT contraction types and functions from tensor4all-simplett
pub use crate::contraction::{inner_product, ContractionOptions};

// Also re-export SimpleTensorTrain for convenience
pub use crate::SimpleTensorTrain;
