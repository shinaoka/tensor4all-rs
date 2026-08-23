#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub mod contract;
pub mod error;
pub mod linsolve;
pub mod options;
pub mod prelude;
pub mod tensortrain;

pub use contract::contract;
pub use error::{Result, TensorTrainError};
pub use linsolve::linsolve;
pub use options::{
    CanonicalForm, ContractMethod, ContractOptions, LinsolveOptions, TruncateOptions,
};
pub use tensor4all_core::SvdOptions;
pub use tensortrain::TensorTrain;

/// Type alias for Matrix Product State (MPS).
///
/// In ITensors.jl, MPS and MPO share the same underlying type.
/// Here, both are aliases for [`TensorTrain`].
pub type MPS = TensorTrain;

/// Type alias for Matrix Product Operator (MPO).
///
/// In ITensors.jl, MPS and MPO share the same underlying type.
/// Here, both are aliases for [`TensorTrain`].
pub type MPO = TensorTrain;
