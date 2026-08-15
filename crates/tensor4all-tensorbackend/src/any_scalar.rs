use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use anyhow::{anyhow, ensure, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro::{DType, Tensor as NativeTensor, TensorScalar};

use crate::storage::{Storage, SumFromStorage};
use crate::tensor_element::TensorElement;

#[derive(Clone, Copy, Debug, PartialEq)]
enum ScalarValue {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    Bool(bool),
    C32(Complex32),
    C64(Complex64),
}

impl ScalarValue {
    fn real(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
            Self::I32(value) => value as f64,
            Self::I64(value) => value as f64,
            Self::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
            Self::C32(value) => value.re as f64,
            Self::C64(value) => value.re,
        }
    }

    fn imag(self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) | Self::I32(_) | Self::I64(_) | Self::Bool(_) => 0.0,
            Self::C32(value) => value.im as f64,
            Self::C64(value) => value.im,
        }
    }

    fn abs(self) -> f64 {
        match self {
            Self::F32(value) => value.abs() as f64,
            Self::F64(value) => value.abs(),
            Self::I32(value) => value.abs() as f64,
            Self::I64(value) => value.abs() as f64,
            Self::Bool(value) => {
                if value {
                    1.0
                } else {
                    0.0
                }
            }
            Self::C32(value) => {
                if value.re.is_nan() || value.im.is_nan() {
                    f64::NAN
                } else {
                    value.norm() as f64
                }
            }
            Self::C64(value) => {
                if value.re.is_nan() || value.im.is_nan() {
                    f64::NAN
                } else {
                    value.norm()
                }
            }
        }
    }

    fn is_complex(self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    fn is_zero(self) -> bool {
        match self {
            Self::F32(value) => value == 0.0,
            Self::F64(value) => value == 0.0,
            Self::I32(value) => value == 0,
            Self::I64(value) => value == 0,
            Self::Bool(value) => !value,
            Self::C32(value) => value == Complex32::new(0.0, 0.0),
            Self::C64(value) => value == Complex64::new(0.0, 0.0),
        }
    }

    fn into_complex(self) -> Complex64 {
        match self {
            Self::F32(value) => Complex64::new(value as f64, 0.0),
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::I32(value) => Complex64::new(value as f64, 0.0),
            Self::I64(value) => Complex64::new(value as f64, 0.0),
            Self::Bool(value) => Complex64::new(if value { 1.0 } else { 0.0 }, 0.0),
            Self::C32(value) => Complex64::new(value.re as f64, value.im as f64),
            Self::C64(value) => value,
        }
    }
}

fn scalar_value_from_storage(storage: &Storage) -> ScalarValue {
    if storage.is_f64() {
        ScalarValue::F64(f64::sum_from_storage(storage))
    } else {
        ScalarValue::C64(Complex64::sum_from_storage(storage))
    }
}

fn scalar_value_from_native(native: &NativeTensor) -> Result<ScalarValue> {
    ensure!(
        native.shape().is_empty(),
        "expected rank-0 scalar tensor, got shape {:?}",
        native.shape()
    );

    match native.dtype() {
        DType::F32 => native
            .as_slice::<f32>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::F32)
            .ok_or_else(|| anyhow!("failed to read f32 scalar tensor value")),
        DType::F64 => native
            .as_slice::<f64>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::F64)
            .ok_or_else(|| anyhow!("failed to read f64 scalar tensor value")),
        DType::I32 => native
            .as_slice::<i32>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::I32)
            .ok_or_else(|| anyhow!("failed to read i32 scalar tensor value")),
        DType::I64 => native
            .as_slice::<i64>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::I64)
            .ok_or_else(|| anyhow!("failed to read i64 scalar tensor value")),
        DType::Bool => native
            .as_slice::<bool>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::Bool)
            .ok_or_else(|| anyhow!("failed to read bool scalar tensor value")),
        DType::C32 => native
            .as_slice::<Complex32>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::C32)
            .ok_or_else(|| anyhow!("failed to read c32 scalar tensor value")),
        DType::C64 => native
            .as_slice::<Complex64>()
            .map_err(anyhow::Error::new)?
            .first()
            .copied()
            .map(ScalarValue::C64)
            .ok_or_else(|| anyhow!("failed to read c64 scalar tensor value")),
    }
}

trait ScalarTensorElement: TensorElement {
    fn scalar_value(value: Self) -> ScalarValue;
}

fn scalar_native<T: TensorScalar>(value: T) -> NativeTensor {
    crate::require_invariant(
        NativeTensor::from_vec_col_major(vec![], vec![value]),
        "rank-0 scalar construction failed",
    )
}

impl ScalarTensorElement for f32 {
    fn scalar_value(value: Self) -> ScalarValue {
        ScalarValue::F32(value)
    }
}

impl ScalarTensorElement for f64 {
    fn scalar_value(value: Self) -> ScalarValue {
        ScalarValue::F64(value)
    }
}

impl ScalarTensorElement for Complex32 {
    fn scalar_value(value: Self) -> ScalarValue {
        ScalarValue::C32(value)
    }
}

impl ScalarTensorElement for Complex64 {
    fn scalar_value(value: Self) -> ScalarValue {
        ScalarValue::C64(value)
    }
}

pub(crate) fn promote_scalar_native(native: &NativeTensor, target: DType) -> Result<NativeTensor> {
    let promoted = match (scalar_value_from_native(native)?, target) {
        (ScalarValue::F32(value), DType::F32) => Scalar::from_value(value),
        (ScalarValue::F32(value), DType::F64) => Scalar::from_value(value as f64),
        (ScalarValue::F32(value), DType::C32) => Scalar::from_value(Complex32::new(value, 0.0)),
        (ScalarValue::F32(value), DType::C64) => {
            Scalar::from_value(Complex64::new(value as f64, 0.0))
        }
        (ScalarValue::F32(_), DType::I64) => {
            return Err(anyhow!(
                "cannot promote f32 scalar to i64 without truncation"
            ));
        }
        (ScalarValue::F32(_), DType::I32 | DType::Bool) => {
            return Err(anyhow!(
                "cannot promote f32 scalar to integer/bool without truncation"
            ));
        }
        (ScalarValue::F64(value), DType::F32) => Scalar::from_value(value as f32),
        (ScalarValue::F64(value), DType::F64) => Scalar::from_value(value),
        (ScalarValue::F64(_), DType::I64) => {
            return Err(anyhow!(
                "cannot promote f64 scalar to i64 without truncation"
            ));
        }
        (ScalarValue::F64(_), DType::I32 | DType::Bool) => {
            return Err(anyhow!(
                "cannot promote f64 scalar to integer/bool without truncation"
            ));
        }
        (ScalarValue::F64(value), DType::C32) => {
            Scalar::from_value(Complex32::new(value as f32, 0.0))
        }
        (ScalarValue::F64(value), DType::C64) => Scalar::from_value(Complex64::new(value, 0.0)),
        (ScalarValue::I32(value), DType::F32) => Scalar::from_value(value as f32),
        (ScalarValue::I32(value), DType::F64) => Scalar::from_value(value as f64),
        (ScalarValue::I32(value), DType::I32) => return Ok(scalar_native(value)),
        (ScalarValue::I32(value), DType::I64) => Scalar::from_i64(value as i64),
        (ScalarValue::I32(value), DType::C32) => {
            Scalar::from_value(Complex32::new(value as f32, 0.0))
        }
        (ScalarValue::I32(value), DType::C64) => {
            Scalar::from_value(Complex64::new(value as f64, 0.0))
        }
        (ScalarValue::I32(_), DType::Bool) => {
            return Err(anyhow!("cannot promote i32 scalar to bool"));
        }
        (ScalarValue::I64(value), DType::F32) => Scalar::from_value(value as f32),
        (ScalarValue::I64(value), DType::F64) => Scalar::from_value(value as f64),
        (ScalarValue::I64(_), DType::I32 | DType::Bool) => {
            return Err(anyhow!("cannot promote i64 scalar to i32/bool"));
        }
        (ScalarValue::I64(value), DType::I64) => Scalar::from_i64(value),
        (ScalarValue::I64(value), DType::C32) => {
            Scalar::from_value(Complex32::new(value as f32, 0.0))
        }
        (ScalarValue::I64(value), DType::C64) => {
            Scalar::from_value(Complex64::new(value as f64, 0.0))
        }
        (ScalarValue::Bool(value), DType::F32) => {
            Scalar::from_value(if value { 1.0_f32 } else { 0.0_f32 })
        }
        (ScalarValue::Bool(value), DType::F64) => Scalar::from_value(if value { 1.0 } else { 0.0 }),
        (ScalarValue::Bool(value), DType::I32) => {
            return Ok(scalar_native(if value { 1 } else { 0 }));
        }
        (ScalarValue::Bool(value), DType::I64) => Scalar::from_i64(if value { 1 } else { 0 }),
        (ScalarValue::Bool(value), DType::Bool) => return Ok(scalar_native(value)),
        (ScalarValue::Bool(value), DType::C32) => {
            Scalar::from_value(Complex32::new(if value { 1.0 } else { 0.0 }, 0.0))
        }
        (ScalarValue::Bool(value), DType::C64) => {
            Scalar::from_value(Complex64::new(if value { 1.0 } else { 0.0 }, 0.0))
        }
        (ScalarValue::C32(value), DType::F32) => Scalar::from_value(value.re),
        (ScalarValue::C32(value), DType::F64) => Scalar::from_value(value.re as f64),
        (ScalarValue::C32(_), DType::I32 | DType::I64 | DType::Bool) => {
            return Err(anyhow!("cannot promote c32 scalar to integer/bool"));
        }
        (ScalarValue::C32(value), DType::C32) => Scalar::from_value(value),
        (ScalarValue::C32(value), DType::C64) => {
            Scalar::from_value(Complex64::new(value.re as f64, value.im as f64))
        }
        (ScalarValue::C64(value), DType::F32) => Scalar::from_value(value.re as f32),
        (ScalarValue::C64(value), DType::F64) => Scalar::from_value(value.re),
        (ScalarValue::C64(_), DType::I32 | DType::I64 | DType::Bool) => {
            return Err(anyhow!("cannot promote c64 scalar to integer/bool"));
        }
        (ScalarValue::C64(value), DType::C32) => {
            Scalar::from_value(Complex32::new(value.re as f32, value.im as f32))
        }
        (ScalarValue::C64(value), DType::C64) => Scalar::from_value(value),
    };
    Ok(promoted.native)
}

/// Dynamic scalar used across tensor4all backends.
///
/// This is a tensor4all-owned rank-0 wrapper over tenferro's dynamic tensor.
///
/// # Examples
///
/// ```
/// use tensor4all_tensorbackend::AnyScalar;
///
/// // Real scalar
/// let a = AnyScalar::new_real(3.14);
/// assert!((a.real() - 3.14).abs() < 1e-10);
/// assert_eq!(a.imag(), 0.0);
/// assert!(a.is_real());
/// assert!(!a.is_complex());
///
/// // Complex scalar
/// let b = AnyScalar::new_complex(1.0, 2.0);
/// assert!((b.real() - 1.0).abs() < 1e-10);
/// assert!((b.imag() - 2.0).abs() < 1e-10);
/// assert!(b.is_complex());
///
/// // Arithmetic
/// let c = AnyScalar::new_real(2.0);
/// let d = a + c;
/// assert!((d.real() - 5.14).abs() < 1e-10);
/// ```
pub struct Scalar {
    native: NativeTensor,
    value: ScalarValue,
}

/// Backward-compatible scalar type name used across tensor4all APIs.
pub type AnyScalar = Scalar;

impl Scalar {
    fn wrap_native(native: NativeTensor) -> Result<Self> {
        if native.shape().is_empty() {
            let value = scalar_value_from_native(&native)?;
            Ok(Self { native, value })
        } else {
            Err(anyhow!(
                "Scalar requires a rank-0 tensor, got shape {:?}",
                native.shape()
            ))
        }
    }

    fn value(&self) -> ScalarValue {
        self.value
    }

    fn from_i64(value: i64) -> Self {
        Self {
            native: scalar_native(value),
            value: ScalarValue::I64(value),
        }
    }

    fn from_i32(value: i32) -> Self {
        Self {
            native: scalar_native(value),
            value: ScalarValue::I32(value),
        }
    }

    fn from_bool(value: bool) -> Self {
        Self {
            native: scalar_native(value),
            value: ScalarValue::Bool(value),
        }
    }

    pub(crate) fn from_native(value: NativeTensor) -> Result<Self> {
        Self::wrap_native(value)
    }

    pub(crate) fn as_native(&self) -> &NativeTensor {
        &self.native
    }

    /// Creates a scalar from any supported public tensor element type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::from_value(3.14_f64);
    /// assert!((s.real() - 3.14).abs() < 1e-10);
    ///
    /// use num_complex::Complex64;
    /// let z = AnyScalar::from_value(Complex64::new(1.0, 2.0));
    /// assert!(z.is_complex());
    /// assert!((z.real() - 1.0).abs() < 1e-10);
    /// assert!((z.imag() - 2.0).abs() < 1e-10);
    /// ```
    #[allow(private_bounds)]
    pub fn from_value<T: ScalarTensorElement>(value: T) -> Self {
        Self {
            native: scalar_native(value),
            value: T::scalar_value(value),
        }
    }

    /// Creates a real scalar from an `f64` value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::from_real(2.5);
    /// assert!((s.real() - 2.5).abs() < 1e-10);
    /// assert!(s.is_real());
    /// ```
    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    /// Creates a complex scalar from real and imaginary parts.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::from_complex(1.0, -1.0);
    /// assert!((s.real() - 1.0).abs() < 1e-10);
    /// assert!((s.imag() - (-1.0)).abs() < 1e-10);
    /// assert!(s.is_complex());
    /// ```
    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    /// Backward-compatible constructor for a real scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(42.0);
    /// assert!((s.real() - 42.0).abs() < 1e-10);
    /// ```
    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    /// Backward-compatible constructor for a complex scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_complex(3.0, 4.0);
    /// assert!((s.abs() - 5.0).abs() < 1e-10); // |3 + 4i| = 5
    /// ```
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    /// Returns the scalar's plain rank-0 tensor value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(5.0);
    /// let p = s.primal();
    /// assert!((p.real() - 5.0).abs() < 1e-10);
    /// ```
    pub fn primal(&self) -> Self {
        self.clone()
    }

    /// Returns the real part while intentionally dropping AD metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_complex(3.0, 4.0);
    /// assert!((s.real() - 3.0).abs() < 1e-10);
    /// ```
    pub fn real(&self) -> f64 {
        self.value().real()
    }

    /// Returns the imaginary part while intentionally dropping AD metadata.
    ///
    /// Returns `0.0` for real scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_complex(3.0, 4.0);
    /// assert!((s.imag() - 4.0).abs() < 1e-10);
    ///
    /// let r = AnyScalar::new_real(5.0);
    /// assert_eq!(r.imag(), 0.0);
    /// ```
    pub fn imag(&self) -> f64 {
        self.value().imag()
    }

    /// Returns the absolute value while intentionally dropping AD metadata.
    ///
    /// For complex scalars, returns the complex modulus (`sqrt(re^2 + im^2)`).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_complex(3.0, 4.0);
    /// assert!((s.abs() - 5.0).abs() < 1e-10);
    ///
    /// let r = AnyScalar::new_real(-7.0);
    /// assert!((r.abs() - 7.0).abs() < 1e-10);
    /// ```
    pub fn abs(&self) -> f64 {
        self.value().abs()
    }

    /// Returns true when the scalar is complex.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// assert!(AnyScalar::new_complex(1.0, 0.0).is_complex());
    /// assert!(!AnyScalar::new_real(1.0).is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.value().is_complex()
    }

    /// Returns true when the scalar is real.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(1.0).is_real());
    /// assert!(!AnyScalar::new_complex(1.0, 2.0).is_real());
    /// ```
    pub fn is_real(&self) -> bool {
        !self.is_complex()
    }

    /// Returns true when the scalar is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(0.0).is_zero());
    /// assert!(!AnyScalar::new_real(1.0).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.value().is_zero()
    }

    /// Returns the underlying value as `f64` when real.
    ///
    /// Returns `None` for complex scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let r = AnyScalar::new_real(2.5);
    /// assert_eq!(r.as_f64(), Some(2.5));
    ///
    /// let c = AnyScalar::new_complex(1.0, 1.0);
    /// assert_eq!(c.as_f64(), None);
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self.value() {
            ScalarValue::F32(value) => Some(value as f64),
            ScalarValue::F64(value) => Some(value),
            ScalarValue::I32(value) => Some(value as f64),
            ScalarValue::I64(value) => Some(value as f64),
            ScalarValue::Bool(value) => Some(if value { 1.0 } else { 0.0 }),
            ScalarValue::C32(_) | ScalarValue::C64(_) => None,
        }
    }

    /// Returns the underlying value as `Complex64` when complex.
    ///
    /// Returns `None` for real scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    /// use num_complex::Complex64;
    ///
    /// let c = AnyScalar::new_complex(1.0, 2.0);
    /// assert_eq!(c.as_c64(), Some(Complex64::new(1.0, 2.0)));
    ///
    /// let r = AnyScalar::new_real(1.0);
    /// assert_eq!(r.as_c64(), None);
    /// ```
    pub fn as_c64(&self) -> Option<Complex64> {
        match self.value() {
            ScalarValue::F32(_)
            | ScalarValue::F64(_)
            | ScalarValue::I32(_)
            | ScalarValue::I64(_)
            | ScalarValue::Bool(_) => None,
            ScalarValue::C32(value) => Some(Complex64::new(value.re as f64, value.im as f64)),
            ScalarValue::C64(value) => Some(value),
        }
    }

    /// Returns the complex conjugate.
    ///
    /// For real scalars, returns a copy (conjugate of a real number is itself).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let c = AnyScalar::new_complex(3.0, 4.0);
    /// let cc = c.conj();
    /// assert!((cc.real() - 3.0).abs() < 1e-10);
    /// assert!((cc.imag() - (-4.0)).abs() < 1e-10);
    ///
    /// let r = AnyScalar::new_real(5.0);
    /// assert!((r.conj().real() - 5.0).abs() < 1e-10);
    /// ```
    pub fn conj(&self) -> Self {
        match self.value() {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::I32(value) => Self::from_i32(value),
            ScalarValue::I64(value) => Self::from_i64(value),
            ScalarValue::Bool(value) => Self::from_bool(value),
            ScalarValue::C32(value) => Self::from_value(value.conj()),
            ScalarValue::C64(value) => Self::from_value(value.conj()),
        }
    }

    /// Returns the real part as a scalar, preserving scalar semantics.
    ///
    /// Unlike [`real`](Self::real), this returns an `AnyScalar` rather than raw `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let c = AnyScalar::new_complex(3.0, 4.0);
    /// let re = c.real_part();
    /// assert!(re.is_real());
    /// assert!((re.real() - 3.0).abs() < 1e-10);
    /// ```
    pub fn real_part(&self) -> Self {
        Self::from_real(self.real())
    }

    /// Returns the imaginary part as a scalar, preserving scalar semantics.
    ///
    /// Unlike [`imag`](Self::imag), this returns an `AnyScalar` rather than raw `f64`.
    /// The result is always a real scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let c = AnyScalar::new_complex(3.0, 4.0);
    /// let im = c.imag_part();
    /// assert!(im.is_real());
    /// assert!((im.real() - 4.0).abs() < 1e-10);
    /// ```
    pub fn imag_part(&self) -> Self {
        Self::from_real(self.imag())
    }

    /// Compose a complex scalar from real-valued parts.
    ///
    /// # Errors
    ///
    /// Returns an error when either input is not a real scalar (a dtype
    /// mismatch failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let re = AnyScalar::new_real(3.0);
    /// let im = AnyScalar::new_real(4.0);
    /// let c = AnyScalar::compose_complex(re, im).unwrap();
    /// assert!(c.is_complex());
    /// assert!((c.real() - 3.0).abs() < 1e-10);
    /// assert!((c.imag() - 4.0).abs() < 1e-10);
    /// ```
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        if !real.is_real() || !imag.is_real() {
            return Err(anyhow!(
                "compose_complex requires real-valued inputs, got real={:?}, imag={:?}",
                real.native.dtype(),
                imag.native.dtype()
            ));
        }
        Ok(Self::from_complex(real.real(), imag.real()))
    }

    /// Square root, preserving AD metadata.
    ///
    /// Automatically promotes to complex if the value is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(9.0);
    /// assert!((s.sqrt().real() - 3.0).abs() < 1e-10);
    /// ```
    pub fn sqrt(&self) -> Self {
        if self.is_complex() || self.real() < 0.0 {
            let value = self.value().into_complex().sqrt();
            if value.im == 0.0 {
                Self::from_real(value.re)
            } else {
                Self::from_value(value)
            }
        } else {
            Self::from_real(self.real().sqrt())
        }
    }

    /// Real exponent power, preserving AD metadata.
    ///
    /// Automatically promotes to complex when the base is negative and the
    /// exponent is non-integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(2.0);
    /// assert!((s.powf(3.0).real() - 8.0).abs() < 1e-10);
    /// ```
    pub fn powf(&self, exponent: f64) -> Self {
        let needs_complex_promotion =
            self.is_complex() || (self.real() < 0.0 && exponent.fract() != 0.0);
        if needs_complex_promotion {
            let value = self.value().into_complex().powf(exponent);
            if value.im == 0.0 {
                Self::from_real(value.re)
            } else {
                Self::from_value(value)
            }
        } else {
            Self::from_real(self.real().powf(exponent))
        }
    }

    /// Integer exponent power, preserving AD metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_tensorbackend::AnyScalar;
    ///
    /// let s = AnyScalar::new_real(3.0);
    /// assert!((s.powi(2).real() - 9.0).abs() < 1e-10);
    /// ```
    pub fn powi(&self, exponent: i32) -> Self {
        self.powf(exponent as f64)
    }
}

impl SumFromStorage for Scalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match scalar_value_from_storage(storage) {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::I32(value) => Self::from_i32(value),
            ScalarValue::I64(value) => Self::from_i64(value),
            ScalarValue::Bool(value) => Self::from_bool(value),
            ScalarValue::C32(value) => Self::from_value(value),
            ScalarValue::C64(value) => Self::from_value(value),
        }
    }
}

impl From<f32> for Scalar {
    fn from(value: f32) -> Self {
        Self::from_value(value)
    }
}

impl From<f64> for Scalar {
    fn from(value: f64) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex32> for Scalar {
    fn from(value: Complex32) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex64> for Scalar {
    fn from(value: Complex64) -> Self {
        Self::from_value(value)
    }
}

impl TryFrom<Scalar> for f64 {
    type Error = &'static str;

    fn try_from(value: Scalar) -> std::result::Result<Self, Self::Error> {
        match value.value() {
            ScalarValue::F32(real) => Ok(real as f64),
            ScalarValue::F64(real) => Ok(real),
            ScalarValue::I32(real) => Ok(real as f64),
            ScalarValue::I64(real) => Ok(real as f64),
            ScalarValue::Bool(real) => Ok(if real { 1.0 } else { 0.0 }),
            ScalarValue::C32(_) | ScalarValue::C64(_) => {
                Err("cannot convert complex scalar to f64")
            }
        }
    }
}

impl From<Scalar> for Complex64 {
    fn from(value: Scalar) -> Self {
        value.value().into_complex()
    }
}

impl Add for Scalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs + rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() + rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() + rhs.real()),
        }
    }
}

impl Sub for Scalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl Mul for Scalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs * rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() * rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() * rhs.real()),
        }
    }
}

impl Div for Scalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self.value(), rhs.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => Self::from_value(lhs / rhs),
            (lhs, rhs) if lhs.is_complex() || rhs.is_complex() => {
                Self::from_value(lhs.into_complex() / rhs.into_complex())
            }
            (lhs, rhs) => Self::from_real(lhs.real() / rhs.real()),
        }
    }
}

impl Neg for Scalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self.value() {
            ScalarValue::F32(value) => Self::from_value(-value),
            ScalarValue::F64(value) => Self::from_value(-value),
            ScalarValue::I32(value) => Self::from_i32(-value),
            ScalarValue::I64(value) => Self::from_i64(-value),
            ScalarValue::Bool(value) => Self::from_real(if value { -1.0 } else { 0.0 }),
            ScalarValue::C32(value) => Self::from_value(-value),
            ScalarValue::C64(value) => Self::from_value(-value),
        }
    }
}

impl Mul<Scalar> for f64 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Scalar::from_real(self) * rhs
    }
}

impl Mul<Scalar> for Complex64 {
    type Output = Scalar;

    fn mul(self, rhs: Scalar) -> Self::Output {
        Scalar::from(self) * rhs
    }
}

impl Div<Scalar> for Complex64 {
    type Output = Scalar;

    fn div(self, rhs: Scalar) -> Self::Output {
        Scalar::from(self) / rhs
    }
}

impl Default for Scalar {
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for Scalar {
    fn zero() -> Self {
        Self::from_real(0.0)
    }

    fn is_zero(&self) -> bool {
        Scalar::is_zero(self)
    }
}

impl One for Scalar {
    fn one() -> Self {
        Self::from_real(1.0)
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        self.native.dtype() == other.native.dtype() && self.value() == other.value()
    }
}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.value(), other.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::I32(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::F32(lhs), ScalarValue::I64(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::F32(lhs), ScalarValue::Bool(rhs)) => {
                (lhs as f64).partial_cmp(&(if rhs { 1.0 } else { 0.0 }))
            }
            (ScalarValue::F64(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::F64(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F64(lhs), ScalarValue::I32(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::I64(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::Bool(rhs)) => {
                lhs.partial_cmp(&(if rhs { 1.0 } else { 0.0 }))
            }
            (ScalarValue::I32(lhs), ScalarValue::F32(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::I32(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::I32(lhs), ScalarValue::I32(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::I32(lhs), ScalarValue::I64(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::I32(lhs), ScalarValue::Bool(rhs)) => {
                (lhs as f64).partial_cmp(&(if rhs { 1.0 } else { 0.0 }))
            }
            (ScalarValue::I64(lhs), ScalarValue::F32(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::I64(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::I64(lhs), ScalarValue::I32(rhs)) => {
                (lhs as f64).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::I64(lhs), ScalarValue::I64(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::I64(lhs), ScalarValue::Bool(rhs)) => {
                (lhs as f64).partial_cmp(&(if rhs { 1.0 } else { 0.0 }))
            }
            (ScalarValue::Bool(lhs), ScalarValue::F32(rhs)) => {
                (if lhs { 1.0 } else { 0.0 }).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::Bool(lhs), ScalarValue::F64(rhs)) => {
                (if lhs { 1.0 } else { 0.0 }).partial_cmp(&rhs)
            }
            (ScalarValue::Bool(lhs), ScalarValue::I32(rhs)) => {
                (if lhs { 1.0 } else { 0.0 }).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::Bool(lhs), ScalarValue::I64(rhs)) => {
                (if lhs { 1.0 } else { 0.0 }).partial_cmp(&(rhs as f64))
            }
            (ScalarValue::Bool(lhs), ScalarValue::Bool(rhs)) => lhs.partial_cmp(&rhs),
            _ => None,
        }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            ScalarValue::F32(value) => value.fmt(f),
            ScalarValue::F64(value) => value.fmt(f),
            ScalarValue::I32(value) => value.fmt(f),
            ScalarValue::I64(value) => value.fmt(f),
            ScalarValue::Bool(value) => value.fmt(f),
            ScalarValue::C32(value) => value.fmt(f),
            ScalarValue::C64(value) => value.fmt(f),
        }
    }
}

impl fmt::Debug for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Scalar")
            .field("dtype", &self.native.dtype())
            .field("value", &self.value())
            .finish()
    }
}

impl Clone for Scalar {
    fn clone(&self) -> Self {
        match self.value {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::I32(value) => Self::from_i32(value),
            ScalarValue::I64(value) => Self::from_i64(value),
            ScalarValue::Bool(value) => Self::from_bool(value),
            ScalarValue::C32(value) => Self::from_value(value),
            ScalarValue::C64(value) => Self::from_value(value),
        }
    }
}

#[cfg(test)]
mod tests;
