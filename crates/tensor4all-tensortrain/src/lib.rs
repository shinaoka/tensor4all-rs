//! Tensor Train (MPS) library
//!
//! This crate provides tensor train (also known as Matrix Product State) algorithms,
//! including:
//! - `TensorTrain`: The main tensor train structure
//! - Compression algorithms (LU, CI, SVD)
//! - Arithmetic operations (add, subtract, scale)
//!
//! # Example
//!
//! ```
//! use tensor4all_tensortrain::{TensorTrain, AbstractTensorTrain};
//!
//! // Create a constant tensor train
//! let tt = TensorTrain::<f64>::constant(&[2, 3, 2], 1.0);
//!
//! // Evaluate at a specific index
//! let value = tt.evaluate(&[0, 1, 1]).unwrap();
//! println!("Value: {}", value);
//!
//! // Sum over all indices
//! let sum = tt.sum();
//! println!("Sum: {}", sum);
//! ```

pub mod arithmetic;
pub mod compression;
pub mod contraction;
pub mod error;
pub mod tensortrain;
pub mod traits;
pub mod types;

// Re-export main types
pub use compression::{CompressionMethod, CompressionOptions};
pub use contraction::{dot, hadamard, hadamard_zipup, ContractionOptions};
pub use error::{Result, TensorTrainError};
pub use tensortrain::TensorTrain;
pub use traits::{AbstractTensorTrain, TTScalar};
pub use types::{LocalIndex, MultiIndex, Tensor3};
