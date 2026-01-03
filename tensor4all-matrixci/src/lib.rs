//! Matrix Cross Interpolation library
//!
//! This crate provides matrix cross interpolation algorithms, including:
//! - `MatrixCI`: Standard matrix cross interpolation
//! - `MatrixACA`: Adaptive Cross Approximation
//! - `RrLU`: Rank-Revealing LU decomposition
//! - `MatrixLUCI`: LU-based Cross Interpolation
//!
//! # Example
//!
//! ```
//! use tensor4all_matrixci::{MatrixCI, crossinterpolate, AbstractMatrixCI, from_vec2d};
//!
//! // Create a matrix
//! let m = from_vec2d(vec![
//!     vec![1.0, 2.0, 3.0],
//!     vec![4.0, 5.0, 6.0],
//!     vec![7.0, 8.0, 9.0],
//! ]);
//!
//! // Perform cross interpolation
//! let ci = crossinterpolate(&m, None);
//!
//! // Get the rank of the approximation
//! println!("Rank: {}", ci.rank());
//! ```

pub mod error;
pub mod matrixaca;
pub mod matrixci;
pub mod matrixlu;
pub mod matrixluci;
pub mod traits;
pub mod util;

// Re-export main types
pub use error::{MatrixCIError, Result};
pub use matrixaca::MatrixACA;
pub use matrixci::{crossinterpolate, CrossInterpolateOptions, MatrixCI};
pub use matrixlu::{rrlu, rrlu_inplace, RrLU, RrLUOptions};
pub use matrixluci::MatrixLUCI;
pub use traits::AbstractMatrixCI;
pub use util::{from_vec2d, Matrix, Scalar};
