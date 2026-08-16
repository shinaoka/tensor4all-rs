#![warn(missing_docs)]
//! Low-level LUCI / rrLU substrate.
//!
//! The dense LU path materializes candidate matrices for exact pivot selection.
//! It uses the configured tensor backend for square complete-pivoting LU and
//! the internal rrLU implementation for rectangular fallback coverage.

pub(crate) mod block_rook;
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod dense;
pub(crate) mod error;
pub(crate) mod factors;
pub(crate) mod kernel;
pub(crate) mod scalar;
pub(crate) mod source;
pub(crate) mod types;

#[allow(unused_imports)]
pub(crate) use block_rook::LazyBlockRookKernel;
#[allow(unused_imports)]
pub(crate) use dense::DenseLuKernel;
#[allow(unused_imports)]
pub(crate) use error::{MatrixLuciError, Result};
#[allow(unused_imports)]
pub(crate) use factors::CrossFactors;
#[allow(unused_imports)]
pub(crate) use kernel::PivotKernel;
pub use scalar::Scalar;
#[allow(unused_imports)]
pub(crate) use source::{CandidateMatrixSource, DenseMatrixSource, LazyMatrixSource};
#[allow(unused_imports)]
pub(crate) use types::{PivotKernelOptions, PivotSelectionCore};
