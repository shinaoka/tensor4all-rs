//! Commonly used types for cross interpolation (TCI) tensor train
//! construction.
//!
//! ```rust
//! use tensor4all_tensorci::prelude::*;
//!
//! let f = |idx: &Vec<usize>| (idx[0] + idx[1] + 1) as f64;
//! let (tci, _ranks, _errors) =
//!     crossinterpolate1(f, vec![4, 4], vec![3, 3], TCI1Options::default()).unwrap();
//! let val = tci.evaluate(&[2, 3]).unwrap();
//! assert!((val - 6.0).abs() < 1e-10);
//! ```

pub use crate::{
    crossinterpolate1, crossinterpolate2, estimate_true_error, floating_zone, opt_first_pivot,
    optimize_with_finder, DefaultGlobalPivotFinder, GlobalPivotFinder, GlobalPivotSearchInput,
    PivotSearchStrategy, Sweep2Strategy, TCI1Options, TCI1SweepStrategy, TCI2OptimizationResult,
    TCI2Options, TCI2Termination, TensorCI1, TensorCI2, TensorCI2FromTensorTrainOptions,
};
