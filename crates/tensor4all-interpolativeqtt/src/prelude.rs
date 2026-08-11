//! Commonly used types and functions for interpolative quantics tensor train
//! construction.
//!
//! ```rust
//! use tensor4all_interpolativeqtt::prelude::*;
//!
//! let tt = interpolate_single_scale(
//!     |x| (-x * x).exp(),
//!     -2.0,
//!     2.0,
//!     5,
//!     12,
//!     &InterpolativeQttOptions::default(),
//! ).unwrap();
//! let value = tt.evaluate(&[0, 0, 0, 0, 0]).unwrap();
//! let expected = (-4.0_f64).exp();
//! assert!((value - expected).abs() < 1e-10);
//! ```

pub use crate::{
    direct_product_core_tensors, estimate_interpolation_error, estimate_interpolation_error_nd,
    get_chebyshev_grid, interpolate_adaptive, interpolate_adaptive_nd, interpolate_multi_scale,
    interpolate_multi_scale_nd, interpolate_single_scale, interpolate_single_scale_nd,
    interpolate_single_scale_sparse, interpolate_single_scale_sparse_nd, interpolation_tensor,
    invert_qtt, AbstractTensorTrain, InterpolativeQttOptions, LagrangePolynomials,
    SimpleTensorTrain,
};
