//! Commonly used types and functions for alternating cross interpolation of
//! elementwise tensor train operations.
//!
//! ```rust
//! use tensor4all_aci::prelude::*;
//!
//! let a = SimpleTensorTrain::<f64>::constant(&[2, 3], 2.0);
//! let b = SimpleTensorTrain::<f64>::constant(&[2, 3], 4.0);
//! let result = elementwise(
//!     |xs: &[f64]| xs[0] * xs[1],
//!     &[a, b],
//!     &AciOptions::default(),
//! ).unwrap();
//! assert_eq!(result.tensor_train.site_dims(), vec![2, 3]);
//! assert!((result.tensor_train.evaluate(&[1, 2]).unwrap() - 8.0).abs() < 1e-10);
//! ```

pub use crate::{
    elementwise, elementwise_batched, AciOptions, AciResult, AciScalar, ElementwiseBatch,
};
pub use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
