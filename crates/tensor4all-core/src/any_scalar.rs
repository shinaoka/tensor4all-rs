#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tensor4all_tensorbackend::AnyScalar as BackendScalar;

use crate::defaults::tensordynlen::TensorDynLen;
use crate::TensorElement;
use tensor4all_tensorbackend::{Storage, SumFromStorage};

#[derive(Clone, Copy, Debug, PartialEq)]
enum ScalarValue {
    F32(f32),
    F64(f64),
    C32(Complex32),
    C64(Complex64),
}

impl ScalarValue {
    fn real(self) -> f64 {
        match self {
            Self::F32(value) => value as f64,
            Self::F64(value) => value,
            Self::C32(value) => value.re as f64,
            Self::C64(value) => value.re,
        }
    }

    fn imag(self) -> f64 {
        match self {
            Self::F32(_) | Self::F64(_) => 0.0,
            Self::C32(value) => value.im as f64,
            Self::C64(value) => value.im,
        }
    }

    fn abs(self) -> f64 {
        match self {
            Self::F32(value) => value.abs() as f64,
            Self::F64(value) => value.abs(),
            Self::C32(value) => value.norm() as f64,
            Self::C64(value) => value.norm(),
        }
    }

    fn is_complex(self) -> bool {
        matches!(self, Self::C32(_) | Self::C64(_))
    }

    fn is_zero(self) -> bool {
        match self {
            Self::F32(value) => value == 0.0,
            Self::F64(value) => value == 0.0,
            Self::C32(value) => value == Complex32::new(0.0, 0.0),
            Self::C64(value) => value == Complex64::new(0.0, 0.0),
        }
    }

    fn into_complex(self) -> Complex64 {
        match self {
            Self::F32(value) => Complex64::new(value as f64, 0.0),
            Self::F64(value) => Complex64::new(value, 0.0),
            Self::C32(value) => Complex64::new(value.re as f64, value.im as f64),
            Self::C64(value) => value,
        }
    }
}

trait ScalarTensorElement: TensorElement {
    fn scalar_value(value: Self) -> ScalarValue;
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

#[cfg(test)]
thread_local! {
    static FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE: Cell<bool> = const { Cell::new(false) };
}

#[derive(Debug, Clone, thiserror::Error)]
enum AnyScalarTensorError {
    #[error("AnyScalar tensor initialization failed: {source}")]
    Initialization {
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

fn initialize_tensor<T: ScalarTensorElement>(
    value: T,
) -> std::result::Result<TensorDynLen, AnyScalarTensorError> {
    #[cfg(test)]
    if FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(Cell::get) {
        return Err(AnyScalarTensorError::Initialization {
            source: Arc::new(std::io::Error::other(
                "forced AnyScalar eager initialization failure",
            )),
        });
    }

    TensorDynLen::scalar(value).map_err(|source| AnyScalarTensorError::Initialization {
        source: Arc::from(source.into_boxed_dyn_error()),
    })
}

/// Dynamic scalar compatibility wrapper for tensor4all-core.
///
/// This owns a rank-0 [`TensorDynLen`] so that scalar values can participate in
/// the same eager autodiff graph as tensors while preserving the existing
/// dynamic scalar API shape. The infallible scalar constructors retain a
/// tensor-initialization failure for later fallible tensor or AD operations.
#[derive(Clone)]
pub struct AnyScalar {
    tensor: std::result::Result<TensorDynLen, AnyScalarTensorError>,
    value: ScalarValue,
}

impl AnyScalar {
    fn wrap_tensor(tensor: TensorDynLen) -> Result<Self> {
        let dims = tensor.dims();
        anyhow::ensure!(
            dims.is_empty(),
            "AnyScalar requires a rank-0 tensor, got dims {:?}",
            dims
        );
        let value = Self::scalar_value_from_tensor(&tensor)?;
        Ok(Self {
            tensor: Ok(tensor),
            value,
        })
    }

    fn from_tensor_result(tensor: Result<TensorDynLen>, op: &'static str) -> Result<Self> {
        Self::wrap_tensor(
            tensor.map_err(|e| anyhow!("AnyScalar::{op} returned invalid scalar tensor: {e}"))?,
        )
        .map_err(|e| anyhow!("AnyScalar::{op} returned non-scalar tensor: {e}"))
    }

    fn from_eager_binary<E>(
        lhs: &Self,
        rhs: &Self,
        op: &'static str,
        f: impl FnOnce(
            &tenferro_ad::EagerTensor,
            &tenferro_ad::EagerTensor,
        ) -> std::result::Result<tenferro_ad::EagerTensor, E>,
    ) -> Result<Self>
    where
        E: fmt::Display,
    {
        let result = f(lhs.as_tensor()?.as_inner()?, rhs.as_tensor()?.as_inner()?)
            .map_err(|e| anyhow!("AnyScalar::{op} failed: {e}"))?;
        Self::from_tensor_result(TensorDynLen::from_inner(vec![], result), op)
    }

    fn from_eager_unary<E>(
        input: &Self,
        op: &'static str,
        f: impl FnOnce(&tenferro_ad::EagerTensor) -> std::result::Result<tenferro_ad::EagerTensor, E>,
    ) -> Result<Self>
    where
        E: fmt::Display,
    {
        let result = f(input.as_tensor()?.as_inner()?)
            .map_err(|e| anyhow!("AnyScalar::{op} failed: {e}"))?;
        Self::from_tensor_result(TensorDynLen::from_inner(vec![], result), op)
    }

    fn scalar_value_from_tensor(tensor: &TensorDynLen) -> Result<ScalarValue> {
        let storage = tensor.storage()?;
        if storage.is_c64() {
            let values = storage
                .payload_c64_col_major_vec()
                .map_err(|e| anyhow!("failed to read c64 scalar storage: {e}"))?;
            values
                .first()
                .copied()
                .map(ScalarValue::C64)
                .ok_or_else(|| anyhow!("rank-0 c64 scalar storage is empty"))
        } else {
            let values = storage
                .payload_f64_col_major_vec()
                .map_err(|e| anyhow!("failed to read f64 scalar storage: {e}"))?;
            values
                .first()
                .copied()
                .map(ScalarValue::F64)
                .ok_or_else(|| anyhow!("rank-0 f64 scalar storage is empty"))
        }
    }

    fn value(&self) -> ScalarValue {
        self.value
    }

    fn from_backend_scalar(value: BackendScalar) -> Self {
        if let Some(value) = value.as_c64() {
            Self::from_value(value)
        } else {
            Self::from_value(value.real())
        }
    }

    pub(crate) fn from_tensor(tensor: TensorDynLen) -> Result<Self> {
        Self::wrap_tensor(tensor)
    }

    pub(crate) fn as_tensor(&self) -> Result<&TensorDynLen> {
        self.tensor
            .as_ref()
            .map_err(|error| anyhow::Error::new(error.clone()))
    }

    /// Creates an `AnyScalar` from a tensor element.
    ///
    /// Use this when you already have a scalar value that implements
    /// [`TensorElement`] and want to lift it into the dynamic scalar wrapper.
    ///
    /// # Arguments
    ///
    /// * `value` - The scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing `value`.
    ///
    /// Tensor initialization is attempted eagerly. Because this constructor is
    /// infallible, an initialization failure is retained and returned by later
    /// tensor- or AD-dependent operations such as [`AnyScalar::enable_grad`].
    /// Value-only accessors and non-AD arithmetic remain available.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_value(3.0f64);
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    #[allow(private_bounds)]
    pub fn from_value<T: ScalarTensorElement>(value: T) -> Self {
        Self {
            tensor: initialize_tensor(value),
            value: T::scalar_value(value),
        }
    }

    /// Creates a real-valued `AnyScalar`.
    ///
    /// This is a convenience wrapper around [`AnyScalar::from_value`].
    ///
    /// # Arguments
    ///
    /// * `x` - The real scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` with real dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_real(1.25);
    /// assert_eq!(scalar.as_f64(), Some(1.25));
    /// assert!(scalar.is_real());
    /// ```
    pub fn from_real(x: f64) -> Self {
        Self::from_value(x)
    }

    /// Creates a complex-valued `AnyScalar`.
    ///
    /// This is a convenience wrapper around [`AnyScalar::from_value`].
    ///
    /// # Arguments
    ///
    /// * `re` - The real part of the complex value.
    /// * `im` - The imaginary part of the complex value.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing the requested complex number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::from_complex(1.0, -2.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((1.0, -2.0)));
    /// assert!(scalar.is_complex());
    /// ```
    pub fn from_complex(re: f64, im: f64) -> Self {
        Self::from_value(Complex64::new(re, im))
    }

    /// Creates a real-valued `AnyScalar`.
    ///
    /// This is an alias for [`AnyScalar::from_real`].
    ///
    /// # Arguments
    ///
    /// * `x` - The real scalar value to wrap.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` with real dtype.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.5);
    /// assert_eq!(scalar.real(), 2.5);
    /// assert!(scalar.is_real());
    /// ```
    pub fn new_real(x: f64) -> Self {
        Self::from_real(x)
    }

    /// Creates a complex-valued `AnyScalar`.
    ///
    /// This is an alias for [`AnyScalar::from_complex`].
    ///
    /// # Arguments
    ///
    /// * `re` - The real part of the complex value.
    /// * `im` - The imaginary part of the complex value.
    ///
    /// # Returns
    ///
    /// A rank-0 `AnyScalar` containing the requested complex number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(2.0, 3.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((2.0, 3.0)));
    /// assert!(scalar.is_complex());
    /// ```
    pub fn new_complex(re: f64, im: f64) -> Self {
        Self::from_complex(re, im)
    }

    /// Returns the detached primal value of this scalar.
    ///
    /// This is an alias for [`AnyScalar::detach`].
    ///
    /// # Returns
    ///
    /// A scalar with the same value and no gradient tracking.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let primal = AnyScalar::new_real(5.0).enable_grad().unwrap().primal().unwrap();
    /// assert_eq!(primal.real(), 5.0);
    /// assert!(!primal.tracks_grad());
    /// ```
    pub fn primal(&self) -> Result<Self> {
        self.detach()
    }

    /// Enables gradient tracking for this scalar.
    ///
    /// # Returns
    ///
    /// A new scalar that shares the same value but participates in autodiff.
    ///
    /// # Errors
    ///
    /// Returns the original tensor-initialization diagnostic if this scalar's
    /// eager backend tensor could not be created, or propagates an AD runtime
    /// failure while enabling gradients.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.0).enable_grad().unwrap();
    /// assert!(scalar.tracks_grad());
    /// ```
    pub fn enable_grad(self) -> Result<Self> {
        let tensor = self.tensor.map_err(anyhow::Error::new)?;
        Self::from_tensor(tensor.enable_grad()?)
    }

    /// Returns whether this scalar tracks gradients.
    ///
    /// # Returns
    ///
    /// `true` when the scalar participates in autodiff and can accumulate a
    /// gradient, otherwise `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(1.0);
    /// assert!(!scalar.tracks_grad());
    /// ```
    pub fn tracks_grad(&self) -> bool {
        match self.tensor.as_ref() {
            Ok(tensor) => tensor.tracks_grad(),
            Err(_) => false,
        }
    }

    /// Returns the stored gradient, if any.
    ///
    /// # Returns
    ///
    /// `Ok(Some(_))` when a gradient is available, `Ok(None)` when no gradient
    /// has been recorded, or an error if the backend cannot read it.
    ///
    /// # Errors
    ///
    /// Propagates autodiff or tensor access failures from the underlying
    /// tensor runtime.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap().unwrap();
    /// assert_eq!(grad.real(), 4.0);
    /// ```
    pub fn grad(&self) -> Result<Option<Self>> {
        self.as_tensor()?
            .grad()
            .and_then(|maybe_grad| maybe_grad.map(Self::from_tensor).transpose())
    }

    /// Clears the stored gradient for this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when the gradient buffer was cleared successfully.
    ///
    /// # Errors
    ///
    /// Propagates tensor runtime failures from the underlying autodiff state.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    /// assert!(x.grad().unwrap().is_some());
    ///
    /// x.clear_grad().unwrap();
    /// assert!(x.grad().unwrap().is_none());
    /// ```
    pub fn clear_grad(&self) -> Result<()> {
        self.as_tensor()?.clear_grad()
    }

    /// Runs reverse-mode autodiff starting from this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when gradients were accumulated successfully.
    ///
    /// # Errors
    ///
    /// Propagates failures from the underlying tensor autodiff engine.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
    /// let y = &x * &x;
    /// y.backward().unwrap();
    ///
    /// let grad = x.grad().unwrap().unwrap();
    /// assert_eq!(grad.real(), 4.0);
    /// ```
    pub fn backward(&self) -> Result<()> {
        self.as_tensor()?.backward()
    }

    /// Returns a detached copy of this scalar.
    ///
    /// # Returns
    ///
    /// A scalar with the same value but without gradient tracking.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let detached = AnyScalar::new_real(7.0)
    ///     .enable_grad()
    ///     .unwrap()
    ///     .detach()
    ///     .unwrap();
    /// assert_eq!(detached.real(), 7.0);
    /// assert!(!detached.tracks_grad());
    /// ```
    pub fn detach(&self) -> Result<Self> {
        Self::from_tensor(self.as_tensor()?.detach()?)
    }

    /// Returns the real part of this scalar.
    ///
    /// # Returns
    ///
    /// The real component as an `f64`, regardless of the underlying storage
    /// type.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.real(), 3.0);
    /// ```
    pub fn real(&self) -> f64 {
        self.value().real()
    }

    /// Returns the imaginary part of this scalar.
    ///
    /// # Returns
    ///
    /// The imaginary component as an `f64`. Real-valued scalars return `0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.imag(), -4.0);
    /// ```
    pub fn imag(&self) -> f64 {
        self.value().imag()
    }

    /// Returns the magnitude of this scalar.
    ///
    /// # Returns
    ///
    /// The absolute value for real scalars or the complex norm for complex
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0);
    /// assert_eq!(scalar.abs(), 5.0);
    /// ```
    pub fn abs(&self) -> f64 {
        self.value().abs()
    }

    /// Returns whether this scalar is complex-valued.
    ///
    /// # Returns
    ///
    /// `true` for complex dtypes and `false` for real or integer dtypes.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_complex(1.0, 2.0).is_complex());
    /// assert!(!AnyScalar::new_real(1.0).is_complex());
    /// ```
    pub fn is_complex(&self) -> bool {
        self.value().is_complex()
    }

    /// Returns whether this scalar is real-valued.
    ///
    /// # Returns
    ///
    /// `true` when the scalar is not complex-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(1.0).is_real());
    /// assert!(!AnyScalar::new_complex(1.0, 2.0).is_real());
    /// ```
    pub fn is_real(&self) -> bool {
        !self.is_complex()
    }

    /// Returns whether this scalar is exactly zero.
    ///
    /// # Returns
    ///
    /// `true` for exact zeros and `false` for any nonzero value.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert!(AnyScalar::new_real(0.0).is_zero());
    /// assert!(!AnyScalar::new_complex(0.0, 1.0).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        self.value().is_zero()
    }

    /// Returns this scalar as an `f64` when it is real-valued.
    ///
    /// # Returns
    ///
    /// `Some(value)` for real and integer scalars, or `None` for complex
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert_eq!(AnyScalar::new_real(2.5).as_f64(), Some(2.5));
    /// assert_eq!(AnyScalar::new_complex(2.5, 1.0).as_f64(), None);
    /// ```
    pub fn as_f64(&self) -> Option<f64> {
        match self.value() {
            ScalarValue::F32(value) => Some(value as f64),
            ScalarValue::F64(value) => Some(value),
            ScalarValue::C32(_) | ScalarValue::C64(_) => None,
        }
    }

    /// Returns this scalar as a `Complex64` when it is complex-valued.
    ///
    /// # Returns
    ///
    /// `Some(value)` for complex scalars or `None` for real and integer
    /// scalars.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(2.5, 1.0);
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((2.5, 1.0)));
    /// assert_eq!(AnyScalar::new_real(2.5).as_c64(), None);
    /// ```
    pub fn as_c64(&self) -> Option<Complex64> {
        match self.value() {
            ScalarValue::F32(_) | ScalarValue::F64(_) => None,
            ScalarValue::C32(value) => Some(Complex64::new(value.re as f64, value.im as f64)),
            ScalarValue::C64(value) => Some(value),
        }
    }

    /// Returns the complex conjugate of this scalar.
    ///
    /// # Returns
    ///
    /// The conjugated scalar. Real-valued inputs are returned unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).conj();
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((3.0, 4.0)));
    /// ```
    pub fn try_conj(&self) -> Result<Self> {
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(self.to_backend_scalar().conj()));
        }
        Self::from_eager_unary(self, "conj", |tensor| tensor.conj())
    }

    /// Returns the complex conjugate of this scalar.
    pub fn conj(&self) -> Self {
        self.try_conj()
            .unwrap_or_else(|_| Self::from_backend_scalar(self.to_backend_scalar().conj()))
    }

    /// Returns the real part as a real-valued scalar.
    ///
    /// # Returns
    ///
    /// A real-valued scalar containing the real component of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).real_part();
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn real_part(&self) -> Self {
        Self::from_real(self.real())
    }

    /// Returns the imaginary part as a real-valued scalar.
    ///
    /// # Returns
    ///
    /// A real-valued scalar containing the imaginary component of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).imag_part();
    /// assert_eq!(scalar.real(), -4.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn imag_part(&self) -> Self {
        Self::from_real(self.imag())
    }

    /// Combines two real-valued scalars into a complex scalar.
    ///
    /// # Arguments
    ///
    /// * `real` - The real component.
    /// * `imag` - The imaginary component.
    ///
    /// # Returns
    ///
    /// A complex `AnyScalar` whose real and imaginary parts come from the
    /// inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if either input is not real-valued.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::compose_complex(
    ///     AnyScalar::new_real(3.0),
    ///     AnyScalar::new_real(-4.0),
    /// )
    /// .unwrap();
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((3.0, -4.0)));
    /// ```
    pub fn compose_complex(real: Self, imag: Self) -> Result<Self> {
        if !real.is_real() || !imag.is_real() {
            return Err(anyhow!("compose_complex requires real-valued inputs"));
        }
        let imag_term = imag.try_mul(&Self::new_complex(0.0, 1.0))?;
        real.try_add(&imag_term)
    }

    /// Returns the square root of this scalar.
    ///
    /// # Returns
    ///
    /// The principal square root. Negative real inputs and complex inputs use
    /// complex arithmetic.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(9.0).sqrt();
    /// assert_eq!(scalar.real(), 3.0);
    /// assert!(scalar.is_real());
    /// ```
    pub fn sqrt(&self) -> Self {
        if !self.tracks_grad() || self.is_complex() || self.real() < 0.0 {
            Self::from_backend_scalar(self.to_backend_scalar().sqrt())
        } else {
            Self::from_eager_unary(self, "sqrt", |tensor| tensor.sqrt())
                .unwrap_or_else(|_| Self::from_backend_scalar(self.to_backend_scalar().sqrt()))
        }
    }

    /// Raises this scalar to a floating-point power.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The exponent to apply.
    ///
    /// # Returns
    ///
    /// The value of `self^exponent`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_real(2.0).powf(3.0);
    /// assert_eq!(scalar.real(), 8.0);
    /// ```
    pub fn powf(&self, exponent: f64) -> Self {
        Self::from_backend_scalar(self.to_backend_scalar().powf(exponent))
    }

    /// Raises this scalar to an integer power.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The integer exponent to apply. Negative exponents return
    ///   the reciprocal power.
    ///
    /// # Returns
    ///
    /// The value of `self^exponent`. Zero exponents return `1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// assert_eq!(AnyScalar::new_real(2.0).powi(3).real(), 8.0);
    /// assert_eq!(AnyScalar::new_real(2.0).powi(-1).real(), 0.5);
    /// ```
    pub fn powi(&self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::one();
        }

        let mut base = self.clone();
        let mut power = exponent.unsigned_abs();
        let mut acc = Self::one();

        while power > 0 {
            if power % 2 == 1 {
                acc = acc.try_mul(&base).unwrap_or_else(|_| {
                    Self::from_backend_scalar(acc.to_backend_scalar() * base.to_backend_scalar())
                });
            }
            power /= 2;
            if power > 0 {
                base = base.try_mul(&base).unwrap_or_else(|_| {
                    Self::from_backend_scalar(base.to_backend_scalar() * base.to_backend_scalar())
                });
            }
        }

        if exponent < 0 {
            Self::one().try_div(&acc).unwrap_or_else(|_| {
                Self::from_backend_scalar(Self::one().to_backend_scalar() / acc.to_backend_scalar())
            })
        } else {
            acc
        }
    }

    pub(crate) fn to_backend_scalar(&self) -> BackendScalar {
        match self.value() {
            ScalarValue::F32(value) => BackendScalar::from_value(value),
            ScalarValue::F64(value) => BackendScalar::from_value(value),
            ScalarValue::C32(value) => BackendScalar::from_value(value),
            ScalarValue::C64(value) => BackendScalar::from_value(value),
        }
    }

    pub(crate) fn try_add(&self, rhs: &Self) -> Result<Self> {
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() + rhs.to_backend_scalar(),
            ));
        }
        Self::from_eager_binary(self, rhs, "add", |lhs, rhs| lhs.add(rhs))
    }

    pub(crate) fn try_mul(&self, rhs: &Self) -> Result<Self> {
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() * rhs.to_backend_scalar(),
            ));
        }
        Self::from_eager_binary(self, rhs, "mul", |lhs, rhs| lhs.mul(rhs))
    }

    pub(crate) fn try_div(&self, rhs: &Self) -> Result<Self> {
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() / rhs.to_backend_scalar(),
            ));
        }
        if self.as_tensor()?.as_native()?.dtype() == rhs.as_tensor()?.as_native()?.dtype() {
            Self::from_eager_binary(self, rhs, "div", |lhs, rhs| lhs.div(rhs))
        } else {
            Ok(Self::from_backend_scalar(
                self.to_backend_scalar() / rhs.to_backend_scalar(),
            ))
        }
    }

    pub(crate) fn try_neg(&self) -> Result<Self> {
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(-self.to_backend_scalar()));
        }
        Self::from_eager_unary(self, "neg", |tensor| tensor.neg())
    }
}

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        Self::from_backend_scalar(BackendScalar::sum_from_storage(storage))
    }
}

impl From<f32> for AnyScalar {
    fn from(value: f32) -> Self {
        Self::from_value(value)
    }
}

impl From<f64> for AnyScalar {
    fn from(value: f64) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex32> for AnyScalar {
    fn from(value: Complex32) -> Self {
        Self::from_value(value)
    }
}

impl From<Complex64> for AnyScalar {
    fn from(value: Complex64) -> Self {
        Self::from_value(value)
    }
}

impl TryFrom<AnyScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: AnyScalar) -> std::result::Result<Self, Self::Error> {
        value.as_f64().ok_or("cannot convert complex scalar to f64")
    }
}

impl From<AnyScalar> for Complex64 {
    fn from(value: AnyScalar) -> Self {
        value.value().into_complex()
    }
}

impl Add<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: &AnyScalar) -> Self::Output {
        self.try_add(rhs).unwrap_or_else(|_| {
            AnyScalar::from_backend_scalar(self.to_backend_scalar() + rhs.to_backend_scalar())
        })
    }
}

impl Add<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: AnyScalar) -> Self::Output {
        Add::add(&self, &rhs)
    }
}

impl Add<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: AnyScalar) -> Self::Output {
        Add::add(self, &rhs)
    }
}

impl Add<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn add(self, rhs: &AnyScalar) -> Self::Output {
        Add::add(&self, rhs)
    }
}

impl Sub<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: &AnyScalar) -> Self::Output {
        Add::add(self, &Neg::neg(rhs))
    }
}

impl Sub<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: AnyScalar) -> Self::Output {
        Sub::sub(&self, &rhs)
    }
}

impl Sub<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: AnyScalar) -> Self::Output {
        Sub::sub(self, &rhs)
    }
}

impl Sub<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn sub(self, rhs: &AnyScalar) -> Self::Output {
        Sub::sub(&self, rhs)
    }
}

impl Mul<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: &AnyScalar) -> Self::Output {
        self.try_mul(rhs).unwrap_or_else(|_| {
            AnyScalar::from_backend_scalar(self.to_backend_scalar() * rhs.to_backend_scalar())
        })
    }
}

impl Mul<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        Mul::mul(&self, &rhs)
    }
}

impl Mul<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        Mul::mul(self, &rhs)
    }
}

impl Mul<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn mul(self, rhs: &AnyScalar) -> Self::Output {
        Mul::mul(&self, rhs)
    }
}

impl Div<&AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: &AnyScalar) -> Self::Output {
        self.try_div(rhs).unwrap_or_else(|_| {
            AnyScalar::from_backend_scalar(self.to_backend_scalar() / rhs.to_backend_scalar())
        })
    }
}

impl Div<AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        Div::div(&self, &rhs)
    }
}

impl Div<AnyScalar> for &AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        Div::div(self, &rhs)
    }
}

impl Div<&AnyScalar> for AnyScalar {
    type Output = AnyScalar;

    fn div(self, rhs: &AnyScalar) -> Self::Output {
        Div::div(&self, rhs)
    }
}

impl Neg for &AnyScalar {
    type Output = AnyScalar;

    fn neg(self) -> Self::Output {
        self.try_neg()
            .unwrap_or_else(|_| AnyScalar::from_backend_scalar(-self.to_backend_scalar()))
    }
}

impl Neg for AnyScalar {
    type Output = AnyScalar;

    fn neg(self) -> Self::Output {
        Neg::neg(&self)
    }
}

impl Mul<AnyScalar> for f64 {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from_real(self) * rhs
    }
}

impl Mul<AnyScalar> for Complex64 {
    type Output = AnyScalar;

    fn mul(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from(self) * rhs
    }
}

impl Div<AnyScalar> for Complex64 {
    type Output = AnyScalar;

    fn div(self, rhs: AnyScalar) -> Self::Output {
        AnyScalar::from(self) / rhs
    }
}

impl Default for AnyScalar {
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for AnyScalar {
    fn zero() -> Self {
        Self::from_real(0.0)
    }

    fn is_zero(&self) -> bool {
        AnyScalar::is_zero(self)
    }
}

impl One for AnyScalar {
    fn one() -> Self {
        Self::from_real(1.0)
    }
}

impl PartialEq for AnyScalar {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl PartialOrd for AnyScalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self.value(), other.value()) {
            (ScalarValue::F32(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&rhs),
            (ScalarValue::F32(lhs), ScalarValue::F64(rhs)) => (lhs as f64).partial_cmp(&rhs),
            (ScalarValue::F64(lhs), ScalarValue::F32(rhs)) => lhs.partial_cmp(&(rhs as f64)),
            (ScalarValue::F64(lhs), ScalarValue::F64(rhs)) => lhs.partial_cmp(&rhs),
            _ => None,
        }
    }
}

impl fmt::Display for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value() {
            ScalarValue::F32(value) => value.fmt(f),
            ScalarValue::F64(value) => value.fmt(f),
            ScalarValue::C32(value) => value.fmt(f),
            ScalarValue::C64(value) => value.fmt(f),
        }
    }
}

impl fmt::Debug for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dtype = match self.value {
            ScalarValue::F32(_) => "f32",
            ScalarValue::F64(_) => "f64",
            ScalarValue::C32(_) => "c32",
            ScalarValue::C64(_) => "c64",
        };
        f.debug_struct("AnyScalar")
            .field("dtype", &dtype)
            .field("value", &self.value())
            .field("tracks_grad", &self.tracks_grad())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_forced_tensor_initialization_failure<T>(f: impl FnOnce() -> T) -> T {
        let previous =
            FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(|failure| failure.replace(true));
        let result = f();
        FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(|failure| failure.set(previous));
        result
    }

    #[test]
    fn non_grad_scalar_arithmetic_uses_plain_values() {
        let a = AnyScalar::new_real(3.0);
        let b = AnyScalar::new_real(4.0);

        let value = ((a.clone() + b.clone()) * b.clone() - AnyScalar::new_real(8.0))
            / AnyScalar::new_real(2.0);

        assert_eq!(value.as_f64(), Some(10.0));
        assert!(!value.tracks_grad());
        assert!(value.as_tensor().is_ok());
    }

    #[test]
    fn tracked_scalar_arithmetic_preserves_autodiff() {
        let x = AnyScalar::new_real(2.0).enable_grad().unwrap();
        let y = &x * &x;

        assert!(y.tracks_grad());
        y.backward().unwrap();

        let grad = x.grad().unwrap().unwrap();
        assert_eq!(grad.as_f64(), Some(4.0));
    }

    #[test]
    fn scalar_tensor_initialization_failure_is_retained_for_tensor_operations() {
        let scalar = with_forced_tensor_initialization_failure(|| AnyScalar::new_real(2.0));
        assert_eq!(scalar.real(), 2.0);
        assert!(!scalar.tracks_grad());

        let error = scalar.as_tensor().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        assert!(error
            .to_string()
            .contains("AnyScalar tensor initialization failed"));
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));

        let error = scalar.clone().enable_grad().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
    }
}
