//! Commonly used traits and types for tensor train (MPS) construction,
//! evaluation, and compression.
//!
//! ```rust
//! use tensor4all_simplett::prelude::*;
//!
//! let tt = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
//! assert!((tt.evaluate(&[1, 1]).unwrap() - 2.0).abs() < 1e-15);
//! let options = CompressionOptions {
//!     tolerance: 1e-10,
//!     max_bond_dim: Some(20),
//!     ..Default::default()
//! };
//! let compressed = tt.compressed(&options).unwrap();
//! assert!(compressed.rank() <= tt.rank());
//! ```

pub use crate::{
    center_canonicalize, inner_product, tensor3_from_data, tensor3_zeros, AbstractTensorTrain,
    CompressionMethod, CompressionOptions, ContractionOptions, DiagMatrix, InverseTensorTrain,
    LocalIndex, MultiIndex, SimpleTensorTrain, SiteTensorTrain, TTCache, TTScalar, Tensor3,
    Tensor3Ops, VidalTensorTrain,
};
