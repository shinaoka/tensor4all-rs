#![warn(missing_docs)]
//! Tensor Train (MPS) library for numerical tensor networks.
//!
//! A **tensor train** (TT), also known as a **Matrix Product State** (MPS),
//! decomposes a high-dimensional tensor into a chain of rank-3 cores:
//!
//! ```text
//! T[i0, i1, ..., iL-1] = A0[i0] * A1[i1] * ... * A_{L-1}[i_{L-1}]
//! ```
//!
//! where each `Ak[ik]` is a matrix of shape `(r_{k-1}, r_k)` and the
//! bond dimensions `r_k` control the approximation accuracy.
//!
//! # Main types
//!
//! | Type | Purpose |
//! |------|---------|
//! | [`SimpleTensorTrain`] | Primary tensor train container |
//! | [`AbstractTensorTrain`] | Common interface (evaluate, sum, norm) |
//! | [`CompressionOptions`] | Controls compression accuracy/cost trade-off |
//! | [`TTCache`] | Caches partial contractions for repeated evaluation |
//! | [`SiteTensorTrain`] | Center-canonical form for sweeping algorithms |
//! | [`VidalTensorTrain`] | Vidal canonical form with explicit singular values |
//! | [`Tensor3`] / [`Tensor3Ops`] | Rank-3 core tensors and their operations |
//!
//! # Typical workflow
//!
//! ```
//! use tensor4all_simplett::{SimpleTensorTrain, AbstractTensorTrain, CompressionOptions};
//!
//! // 1. Create a tensor train (e.g. constant function)
//! let tt = SimpleTensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
//! assert_eq!(tt.len(), 3);
//!
//! // 2. Evaluate at a specific multi-index
//! let value = tt.evaluate(&[0, 1, 1]).unwrap();
//! assert!((value - 1.0).abs() < 1e-12);
//!
//! // 3. Sum over all indices: 1.0 * 2 * 3 * 2 = 12.0
//! let sum = tt.sum();
//! assert!((sum - 12.0).abs() < 1e-10);
//!
//! // 4. Compress to reduce bond dimensions
//! let compressed = tt.compressed(&CompressionOptions::default()).unwrap();
//! let val2 = compressed.evaluate(&[0, 1, 1]).unwrap();
//! assert!((val2 - 1.0).abs() < 1e-10);
//! ```

pub mod arithmetic;
pub mod cache;
pub mod canonical;
pub mod compression;
pub mod contraction;
pub(crate) mod einsum_helper;
pub mod error;
pub mod mpo;
pub mod tensor;
pub mod tensortrain;
pub mod traits;
pub mod types;
pub mod vidal;

// Re-export main types
pub use cache::TTCache;
pub use canonical::{center_canonicalize, SiteTensorTrain};
pub use compression::{CompressionMethod, CompressionOptions};
pub use contraction::{dot, ContractionOptions};
pub use error::{Result, TensorTrainError};
pub use tensortrain::SimpleTensorTrain;
pub use traits::{AbstractTensorTrain, TTScalar};
pub use types::{tensor3_from_data, tensor3_zeros, LocalIndex, MultiIndex, Tensor3, Tensor3Ops};
pub use vidal::{DiagMatrix, InverseTensorTrain, VidalTensorTrain};
