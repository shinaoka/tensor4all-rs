//! Linear equation solvers for Tree Tensor Networks.
//!
//! This module provides solvers for linear systems of the form
//! `(a₀ + a₁ * A) * x = b` where A is a TTN operator and x, b are TTN states.
//!
//! # Modules
//!
//! - [`common`]: Shared infrastructure (environment cache, options, projected operator)
//! - [`square`]: Solver for V_in = V_out case (input and output spaces are the same)
//!
//! # Usage
//!
//! For most use cases, use the `square` solver:
//!
//! ```
//! use tensor4all_treetn::{square_linsolve, LinsolveOptions};
//!
//! let options = LinsolveOptions::default().with_nfullsweeps(2);
//! let _solver = square_linsolve::<tensor4all_core::TensorDynLen, usize>;
//!
//! assert_eq!(options.nfullsweeps, 2);
//! ```

pub mod common;
pub mod square;

// Re-export commonly used types
pub use common::{EnvironmentCache, LinsolveOptions, NetworkTopology, ProjectedOperator};
pub use square::{
    relative_linear_system_residual, square_linsolve, LinsolveVerifyReport, NodeVerifyDetail,
    ProjectedState, SquareLinsolveResult, SquareLinsolveUpdater,
};
