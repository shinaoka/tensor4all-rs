//! Commonly used types and functions for quantics tensor cross interpolation
//! (QTT) on discrete or continuous grids.
//!
//! ```rust
//! use tensor4all_quanticstci::prelude::*;
//!
//! let f = |idx: &[usize]| (idx[0] + idx[1]) as f64;
//! let (qtci, _ranks, errors) = quanticscrossinterpolate_discrete(
//!     &[16, 16],
//!     f,
//!     None,
//!     QtciOptions::default().with_tolerance(1e-10),
//! ).unwrap();
//! let value = qtci.evaluate(&[4, 9]).unwrap();
//! assert!((value - 13.0).abs() < 1e-10);
//! assert!(errors.last().copied().unwrap() < 1e-10);
//! ```

pub use crate::{
    quanticscrossinterpolate, quanticscrossinterpolate_batched, quanticscrossinterpolate_discrete,
    quanticscrossinterpolate_from_arrays, DefaultProposer, DiscretizedGrid, InherentDiscreteGrid,
    QtciOptions, QuanticsTensorCI2, QuanticsTensorCI2Batched, SimpleTensorTrain, TreeTciGraph,
    TreeTciOptions, UnfoldingScheme,
};
