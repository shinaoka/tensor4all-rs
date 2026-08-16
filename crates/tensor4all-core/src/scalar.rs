//! Common scalar trait for matrix and tensor operations.
//!
//! The [`Scalar`] trait provides a unified interface for numeric types used
//! in matrix cross interpolation and tensor train operations. It is
//! implemented for [`f64`], [`f32`], [`Complex64`], and [`Complex32`].
//!
//! The `scalar_tests!` macro generates dual `f64`/`Complex64` test
//! variants from a single generic test function.

use num_complex::{Complex32, Complex64};
use num_traits::{Float, One, Zero};
use tensor4all_tensorbackend::MatrixScalar;

/// Common scalar trait for matrix and tensor operations.
///
/// Defines the minimal requirements for scalar types used in matrix cross
/// interpolation and tensor train operations. Implemented for `f64`, `f32`,
/// `Complex64`, and `Complex32`.
///
/// # Examples
///
/// ```
/// use tensor4all_core::Scalar;
///
/// // f64
/// let x = 3.0_f64;
/// assert_eq!(x.abs_sq(), 9.0);
/// assert_eq!(x.conj(), 3.0);
///
/// // Complex64
/// use num_complex::Complex64;
/// let z = Complex64::new(3.0, 4.0);
/// assert!((z.abs_sq() - 25.0).abs() < 1e-10);
/// assert_eq!(z.conj(), Complex64::new(3.0, -4.0));
///
/// // Construction from f64
/// let val = f64::from_f64(2.5);
/// assert_eq!(val, 2.5);
/// ```
pub trait Scalar:
    Clone
    + Copy
    + Zero
    + One
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + Default
    + Send
    + Sync
    + MatrixScalar
    + 'static
{
    /// Complex conjugate of the value.
    fn conj(self) -> Self;

    /// Square of the absolute value (for complex numbers, |z|^2).
    fn abs_sq(self) -> f64;

    /// Absolute value as Self type.
    ///
    /// For real types, this returns the absolute value.
    /// For complex types, this returns a real-valued complex (re=|z|, im=0).
    fn abs(self) -> Self;

    /// Absolute value as f64.
    fn abs_val(self) -> f64 {
        self.abs_sq().sqrt()
    }

    /// Create from f64 value.
    fn from_f64(val: f64) -> Self;

    /// Check if value is NaN.
    fn is_nan(self) -> bool;

    /// Machine epsilon for numerical comparisons.
    ///
    /// Returns `f64::EPSILON` (~2.2e-16) by default. This is the smallest value
    /// such that `1.0 + epsilon != 1.0`.
    fn epsilon() -> f64 {
        f64::EPSILON
    }
}

impl Scalar for f64 {
    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self * self
    }

    #[inline]
    fn abs(self) -> Self {
        Float::abs(self)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        Float::abs(self)
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val
    }

    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for f32 {
    #[inline]
    fn conj(self) -> Self {
        self
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        (self * self) as f64
    }

    #[inline]
    fn abs(self) -> Self {
        Float::abs(self)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        Float::abs(self) as f64
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        val as f32
    }

    #[inline]
    fn is_nan(self) -> bool {
        Float::is_nan(self)
    }
}

impl Scalar for Complex64 {
    #[inline]
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self.norm_sqr()
    }

    #[inline]
    fn abs(self) -> Self {
        Complex64::new(self.norm(), 0.0)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        self.norm()
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        Complex64::new(val, 0.0)
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl Scalar for Complex32 {
    #[inline]
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }

    #[inline]
    fn abs_sq(self) -> f64 {
        self.norm_sqr() as f64
    }

    #[inline]
    fn abs(self) -> Self {
        Complex32::new(self.norm(), 0.0)
    }

    #[inline]
    fn abs_val(self) -> f64 {
        self.norm() as f64
    }

    #[inline]
    fn from_f64(val: f64) -> Self {
        Complex32::new(val as f32, 0.0)
    }

    #[inline]
    fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

/// Macro to generate f64 and Complex64 test variants from a generic test function.
///
/// # Example
///
/// ```
/// fn test_operation_generic<T: tensor4all_core::Scalar>() {
///     let value = T::from_f64(2.0);
///     assert_eq!(value.abs_sq(), 4.0);
/// }
///
/// # fn main() {}
/// tensor4all_core::scalar_tests!(test_operation, test_operation_generic);
/// ```
#[macro_export]
macro_rules! scalar_tests {
    ($name:ident, $test_fn:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _f64>]() {
                $test_fn::<f64>();
            }

            #[test]
            fn [<$name _c64>]() {
                $test_fn::<num_complex::Complex64>();
            }
        }
    };
}

#[cfg(test)]
mod tests;
