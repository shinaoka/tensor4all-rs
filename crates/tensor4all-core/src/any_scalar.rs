#[cfg(test)]
use std::cell::Cell;
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use num_complex::{Complex32, Complex64};
use num_traits::{One, Zero};
use tenferro::DType;
use tensor4all_tensorbackend::BackendScalar;

use crate::defaults::idx_tensor::IdxTensor;
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
            Self::C32(value) => {
                if value.re.is_nan() || value.im.is_nan() {
                    f64::NAN
                } else {
                    f64::from(value.re).hypot(f64::from(value.im))
                }
            }
            Self::C64(value) => {
                if value.re.is_nan() || value.im.is_nan() {
                    f64::NAN
                } else {
                    value.re.hypot(value.im)
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
    #[error("AnyScalar::{op} failed: {source}")]
    Operation {
        op: &'static str,
        #[source]
        source: Arc<dyn std::error::Error + Send + Sync + 'static>,
    },
}

fn initialize_tensor<T: ScalarTensorElement>(
    value: T,
) -> std::result::Result<IdxTensor, AnyScalarTensorError> {
    #[cfg(test)]
    if FORCE_ANY_SCALAR_TENSOR_INITIALIZATION_FAILURE.with(Cell::get) {
        return Err(AnyScalarTensorError::Initialization {
            source: Arc::new(std::io::Error::other(
                "forced AnyScalar eager initialization failure",
            )),
        });
    }

    IdxTensor::scalar(value).map_err(|source| AnyScalarTensorError::Initialization {
        source: Arc::from(anyhow::Error::new(source).into_boxed_dyn_error()),
    })
}

/// Error returned by eager-tensor `AnyScalar` operations (autodiff, conjugation,
/// and complex composition).
///
/// The full original diagnostic is preserved in [`AnyScalarError::source`], so
/// callers can inspect the underlying tensor, AD-runtime, or configuration
/// failure without losing context.
///
/// # Examples
///
/// ```
/// use tensor4all_core::{AnyScalar, AnyScalarError};
///
/// let result: Result<AnyScalar, AnyScalarError> =
///     AnyScalar::compose_complex(AnyScalar::new_real(1.0), AnyScalar::new_complex(0.0, 1.0));
/// let err = result.unwrap_err();
/// assert!(err.source.to_string().contains("real-valued"));
/// ```
#[derive(Debug, thiserror::Error)]
#[error("AnyScalar eager-tensor operation failed: {source}")]
pub struct AnyScalarError {
    /// Original tensor, AD-runtime, or configuration diagnostic, including any
    /// operation-specific context added by the failing call.
    #[source]
    pub source: anyhow::Error,
}

impl From<anyhow::Error> for AnyScalarError {
    fn from(source: anyhow::Error) -> Self {
        Self { source }
    }
}

fn operation_error<E>(op: &'static str, source: E) -> anyhow::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    anyhow::Error::new(AnyScalarTensorError::Operation {
        op,
        source: Arc::new(source),
    })
}

fn operation_error_from_anyhow(op: &'static str, source: anyhow::Error) -> anyhow::Error {
    match source.downcast::<AnyScalarTensorError>() {
        Ok(source) => anyhow::Error::new(source),
        Err(source) => anyhow::Error::new(AnyScalarTensorError::Operation {
            op,
            source: Arc::from(source.into_boxed_dyn_error()),
        }),
    }
}

/// Dynamic scalar compatibility wrapper for tensor4all-core.
/// This owns a rank-0 [`IdxTensor`] so that scalar values can participate in
/// the same eager autodiff graph as tensors while preserving the existing
/// dynamic scalar API shape. The infallible scalar constructors retain a
/// tensor-initialization failure for later fallible tensor or AD operations.
/// Infallible arithmetic also retains typed backend diagnostics and the
/// tracked-state marker when an eager operation fails.
#[derive(Clone)]
pub struct AnyScalar {
    tensor: std::result::Result<IdxTensor, AnyScalarTensorError>,
    value: ScalarValue,
    tracks_grad: bool,
}

impl AnyScalar {
    fn wrap_tensor(tensor: IdxTensor) -> Result<Self> {
        let dims = tensor.dims();
        anyhow::ensure!(
            dims.is_empty(),
            "AnyScalar requires a rank-0 tensor, got dims {:?}",
            dims
        );
        let value = Self::scalar_value_from_tensor(&tensor)?;
        let tracks_grad = tensor.tracks_grad();
        Ok(Self {
            tensor: Ok(tensor),
            value,
            tracks_grad,
        })
    }

    fn from_tensor_result(tensor: Result<IdxTensor>, op: &'static str) -> Result<Self> {
        let tensor = tensor.map_err(|error| operation_error_from_anyhow(op, error))?;
        Self::wrap_tensor(tensor).map_err(|error| operation_error_from_anyhow(op, error))
    }

    fn fallback_result(
        result: Result<Self>,
        op: &'static str,
        fallback: impl FnOnce() -> ScalarValue,
        tracks_grad: bool,
    ) -> Self {
        match result {
            Ok(result) => result,
            Err(error) => {
                let error = match error.downcast::<AnyScalarTensorError>() {
                    Ok(error) => error,
                    Err(error) => AnyScalarTensorError::Operation {
                        op,
                        source: Arc::from(error.into_boxed_dyn_error()),
                    },
                };
                Self {
                    tensor: Err(error),
                    value: fallback(),
                    tracks_grad,
                }
            }
        }
    }

    fn scalar_value_from_backend(value: BackendScalar) -> ScalarValue {
        value
            .as_c64()
            .map(ScalarValue::C64)
            .unwrap_or_else(|| ScalarValue::F64(value.real()))
    }

    fn zero_like(&self) -> Self {
        match self.value() {
            ScalarValue::F32(_) => Self::from_value(0.0_f32),
            ScalarValue::F64(_) => Self::from_value(0.0_f64),
            ScalarValue::C32(_) => Self::from_value(Complex32::new(0.0, 0.0)),
            ScalarValue::C64(_) => Self::from_value(Complex64::new(0.0, 0.0)),
        }
    }

    fn one_like(&self) -> Self {
        match self.value() {
            ScalarValue::F32(_) => Self::from_value(1.0_f32),
            ScalarValue::F64(_) => Self::from_value(1.0_f64),
            ScalarValue::C32(_) => Self::from_value(Complex32::new(1.0, 0.0)),
            ScalarValue::C64(_) => Self::from_value(Complex64::new(1.0, 0.0)),
        }
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
        E: std::error::Error + Send + Sync + 'static,
    {
        let result = f(lhs.as_tensor()?.as_inner()?, rhs.as_tensor()?.as_inner()?)
            .map_err(|error| operation_error(op, error))?;
        Self::from_tensor_result(IdxTensor::from_inner(vec![], result), op)
    }

    fn from_eager_unary<E>(
        input: &Self,
        op: &'static str,
        f: impl FnOnce(&tenferro_ad::EagerTensor) -> std::result::Result<tenferro_ad::EagerTensor, E>,
    ) -> Result<Self>
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        let result =
            f(input.as_tensor()?.as_inner()?).map_err(|error| operation_error(op, error))?;
        Self::from_tensor_result(IdxTensor::from_inner(vec![], result), op)
    }

    fn scalar_value_from_tensor(tensor: &IdxTensor) -> Result<ScalarValue> {
        let inner = tensor.as_inner()?;
        match inner.dtype() {
            DType::F32 => inner
                .value()?
                .as_slice::<f32>()?
                .first()
                .copied()
                .map(ScalarValue::F32)
                .ok_or_else(|| anyhow!("rank-0 f32 scalar tensor is empty")),
            DType::F64 => inner
                .value()?
                .as_slice::<f64>()?
                .first()
                .copied()
                .map(ScalarValue::F64)
                .ok_or_else(|| anyhow!("rank-0 f64 scalar tensor is empty")),
            DType::C32 => inner
                .value()?
                .as_slice::<Complex32>()?
                .first()
                .copied()
                .map(ScalarValue::C32)
                .ok_or_else(|| anyhow!("rank-0 c32 scalar tensor is empty")),
            DType::C64 => inner
                .value()?
                .as_slice::<Complex64>()?
                .first()
                .copied()
                .map(ScalarValue::C64)
                .ok_or_else(|| anyhow!("rank-0 c64 scalar tensor is empty")),
            dtype => Err(anyhow!("unsupported scalar tensor dtype {dtype:?}")),
        }
    }

    fn value(&self) -> ScalarValue {
        self.value
    }

    fn from_backend_scalar(value: BackendScalar) -> Self {
        match Self::scalar_value_from_backend(value) {
            ScalarValue::F32(value) => Self::from_value(value),
            ScalarValue::F64(value) => Self::from_value(value),
            ScalarValue::C32(value) => Self::from_value(value),
            ScalarValue::C64(value) => Self::from_value(value),
        }
    }

    pub(crate) fn from_tensor(tensor: IdxTensor) -> Result<Self> {
        Self::wrap_tensor(tensor)
    }

    pub(crate) fn as_tensor(&self) -> Result<&IdxTensor> {
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
    /// A rank-0 `AnyScalar` containing `value`. The supported scalar types are
    /// `f32`, `f64`, `Complex32`, and `Complex64`; the dtype is retained without
    /// promotion.
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
            tracks_grad: false,
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
    /// # Errors
    ///
    /// Returns an error when the scalar is not a tracked leaf (a missing-graph
    /// /// failure).
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
    pub fn primal(&self) -> std::result::Result<Self, AnyScalarError> {
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
    pub fn enable_grad(self) -> std::result::Result<Self, AnyScalarError> {
        let tensor = self.tensor.map_err(anyhow::Error::new)?;
        Self::from_tensor(tensor.enable_grad().map_err(anyhow::Error::from)?)
            .map_err(AnyScalarError::from)
    }

    /// Returns whether this scalar tracks gradients.
    ///
    /// # Returns
    ///
    /// `true` when the scalar participates in autodiff or retains a failed
    /// tracked operation, otherwise `false`.
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
        self.tracks_grad || self.tensor.as_ref().is_ok_and(IdxTensor::tracks_grad)
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
    /// Returns an error when the scalar is not a tracked leaf or the gradient is
    /// /// unavailable (a missing-graph or dtype mismatch failure).
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
    pub fn grad(&self) -> std::result::Result<Option<Self>, AnyScalarError> {
        self.as_tensor()?
            .grad()
            .map_err(anyhow::Error::from)
            .and_then(|maybe_grad| maybe_grad.map(Self::from_tensor).transpose())
            .map_err(AnyScalarError::from)
    }

    /// Clears the stored gradient for this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when the gradient buffer was cleared successfully.
    ///
    /// # Errors
    ///
    /// Returns an error when the scalar is not a tracked leaf (a missing-graph
    /// /// failure).
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
    pub fn clear_grad(&self) -> std::result::Result<(), AnyScalarError> {
        self.as_tensor()?
            .clear_grad()
            .map_err(anyhow::Error::from)
            .map_err(AnyScalarError::from)
    }

    /// Runs reverse-mode autodiff starting from this scalar.
    ///
    /// # Returns
    ///
    /// `Ok(())` when gradients were accumulated successfully.
    ///
    /// # Errors
    ///
    /// Returns an error when the scalar is not a scalar-valued leaf or the reverse
    /// /// pass fails (a graph failure).
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
    pub fn backward(&self) -> std::result::Result<(), AnyScalarError> {
        self.as_tensor()?
            .backward()
            .map_err(anyhow::Error::from)
            .map_err(AnyScalarError::from)
    }

    /// Returns a detached copy of this scalar.
    ///
    /// # Returns
    ///
    /// A scalar with the same value but without gradient tracking.
    ///
    /// # Errors
    ///
    /// Returns an error when the scalar is not a tracked leaf (a missing-graph
    /// /// failure).
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
    pub fn detach(&self) -> std::result::Result<Self, AnyScalarError> {
        Self::from_tensor(self.as_tensor()?.detach().map_err(anyhow::Error::from)?)
            .map_err(AnyScalarError::from)
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
    /// # Errors
    ///
    /// Returns an error when the conjugation fails (a dtype mismatch or backend
    /// /// failure).
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_core::AnyScalar;
    ///
    /// let scalar = AnyScalar::new_complex(3.0, -4.0).conj();
    /// assert_eq!(scalar.as_c64().map(|z| (z.re, z.im)), Some((3.0, 4.0)));
    /// ```
    pub fn try_conj(&self) -> std::result::Result<Self, AnyScalarError> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(self.to_backend_scalar().conj()));
        }
        Self::from_eager_unary(self, "conj", |tensor| tensor.conj()).map_err(AnyScalarError::from)
    }

    /// Returns the complex conjugate of this scalar.
    pub fn conj(&self) -> Self {
        Self::fallback_result(
            self.try_conj().map_err(|error| error.source),
            "conj",
            || Self::scalar_value_from_backend(self.to_backend_scalar().conj()),
            self.tracks_grad(),
        )
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
        Self::fallback_result(
            self.try_real_part(),
            "real_part",
            || Self::from_real(self.real()).value(),
            self.tracks_grad(),
        )
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
        Self::fallback_result(
            self.try_imag_part(),
            "imag_part",
            || Self::from_real(self.imag()).value(),
            self.tracks_grad(),
        )
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
    /// Returns an error when the components cannot be composed (a dtype mismatch
    /// or a backend failure).
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
    pub fn compose_complex(real: Self, imag: Self) -> std::result::Result<Self, AnyScalarError> {
        if !real.is_real() || !imag.is_real() {
            return Err(anyhow!("compose_complex requires real-valued inputs").into());
        }
        let imag_term = imag.try_mul(&Self::new_complex(0.0, 1.0))?;
        real.try_add(&imag_term).map_err(AnyScalarError::from)
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
        Self::fallback_result(
            self.try_sqrt(),
            "sqrt",
            || Self::scalar_value_from_backend(self.to_backend_scalar().sqrt()),
            self.tracks_grad(),
        )
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
        Self::fallback_result(
            self.try_powf(exponent),
            "powf",
            || Self::scalar_value_from_backend(self.to_backend_scalar().powf(exponent)),
            self.tracks_grad(),
        )
    }

    /// Raises this scalar to an integer power.
    ///
    /// # Arguments
    ///
    /// * `exponent` - The integer exponent to apply. Negative exponents return
    ///
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
        Self::fallback_result(
            self.try_powi(exponent),
            "powi",
            || Self::scalar_value_from_backend(self.to_backend_scalar().powi(exponent)),
            self.tracks_grad(),
        )
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
        self.as_tensor()?;
        rhs.as_tensor()?;
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() + rhs.to_backend_scalar(),
            ));
        }
        Self::from_eager_binary(self, rhs, "add", |lhs, rhs| lhs.add(rhs))
    }

    pub(crate) fn try_mul(&self, rhs: &Self) -> Result<Self> {
        self.as_tensor()?;
        rhs.as_tensor()?;
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() * rhs.to_backend_scalar(),
            ));
        }
        Self::from_eager_binary(self, rhs, "mul", |lhs, rhs| lhs.mul(rhs))
    }

    pub(crate) fn try_div(&self, rhs: &Self) -> Result<Self> {
        self.as_tensor()?;
        rhs.as_tensor()?;
        if !self.tracks_grad() && !rhs.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar() / rhs.to_backend_scalar(),
            ));
        }
        Self::from_eager_binary(self, rhs, "div", |lhs, rhs| lhs.div(rhs))
    }

    pub(crate) fn try_neg(&self) -> Result<Self> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(-self.to_backend_scalar()));
        }
        Self::from_eager_unary(self, "neg", |tensor| tensor.neg())
    }

    fn try_real_part(&self) -> Result<Self> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_real(self.real()));
        }
        if self.is_complex() {
            Self::from_eager_unary(self, "real_part", |tensor| tensor.cast(DType::F64))
        } else {
            self.try_mul(&Self::new_real(1.0))
        }
    }

    fn try_imag_part(&self) -> Result<Self> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_real(self.imag()));
        }
        if self.is_complex() {
            let factor = Self::new_complex(0.0, -1.0);
            let imaginary =
                Self::from_eager_binary(self, &factor, "imag_part", |value, factor| {
                    value.mul(factor)
                })?;
            Self::from_eager_unary(&imaginary, "imag_part", |tensor| tensor.cast(DType::F64))
        } else {
            self.try_mul(&Self::new_real(0.0))
        }
    }

    fn try_sqrt(&self) -> Result<Self> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(self.to_backend_scalar().sqrt()));
        }
        if self.is_real() && self.real() < 0.0 {
            let magnitude_input = Self::from_eager_unary(self, "sqrt", |tensor| tensor.neg())?;
            let magnitude = magnitude_input.try_sqrt()?;
            let factor = Self::new_complex(0.0, 1.0);
            return Self::from_eager_binary(&magnitude, &factor, "sqrt", |value, factor| {
                value.mul(factor)
            });
        }
        Self::from_eager_unary(self, "sqrt", |tensor| tensor.sqrt())
    }

    fn try_powf(&self, exponent: f64) -> Result<Self> {
        self.as_tensor()?;
        if !self.tracks_grad() {
            return Ok(Self::from_backend_scalar(
                self.to_backend_scalar().powf(exponent),
            ));
        }
        if self.is_real() && self.real() < 0.0 && exponent.fract() != 0.0 {
            let magnitude_input = Self::from_eager_unary(self, "powf", |tensor| tensor.neg())?;
            let magnitude = magnitude_input.try_powf(exponent)?;
            let phase = std::f64::consts::PI * exponent;
            let factor = Self::new_complex(phase.cos(), phase.sin());
            return Self::from_eager_binary(&magnitude, &factor, "powf", |value, factor| {
                value.mul(factor)
            });
        }
        let exponent = if self.is_complex() {
            Self::new_complex(exponent, 0.0)
        } else {
            Self::new_real(exponent)
        };
        Self::from_eager_binary(self, &exponent, "powf", |base, exponent| base.pow(exponent))
    }

    fn try_powi(&self, exponent: i32) -> Result<Self> {
        self.as_tensor()?;
        if exponent == 0 {
            if self.tracks_grad() {
                // Build 1 as `self * 0 + 1`, rather than evaluating x^0.
                // This keeps the result in the graph and has an exact zero
                // derivative even when the input is zero.
                let zeroed = self.try_mul(&self.zero_like())?;
                return zeroed.try_add(&self.one_like());
            }
            return Ok(Self::one());
        }
        if self.tracks_grad() {
            return self.try_powf(exponent as f64);
        }
        Ok(Self::from_backend_scalar(
            self.to_backend_scalar().powi(exponent),
        ))
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
        AnyScalar::fallback_result(
            self.try_add(rhs),
            "add",
            || {
                AnyScalar::scalar_value_from_backend(
                    self.to_backend_scalar() + rhs.to_backend_scalar(),
                )
            },
            self.tracks_grad() || rhs.tracks_grad(),
        )
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
        AnyScalar::fallback_result(
            self.try_mul(rhs),
            "mul",
            || {
                AnyScalar::scalar_value_from_backend(
                    self.to_backend_scalar() * rhs.to_backend_scalar(),
                )
            },
            self.tracks_grad() || rhs.tracks_grad(),
        )
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
        AnyScalar::fallback_result(
            self.try_div(rhs),
            "div",
            || {
                AnyScalar::scalar_value_from_backend(
                    self.to_backend_scalar() / rhs.to_backend_scalar(),
                )
            },
            self.tracks_grad() || rhs.tracks_grad(),
        )
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
        AnyScalar::fallback_result(
            self.try_neg(),
            "neg",
            || AnyScalar::scalar_value_from_backend(-self.to_backend_scalar()),
            self.tracks_grad(),
        )
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
    fn compact_sum_preserves_f32_and_c32_dtype_with_and_without_ad() {
        let indices = || vec![crate::DynIndex::new_dyn(2), crate::DynIndex::new_dyn(2)];
        for tensor in [
            IdxTensor::from_diag(indices(), vec![1.0_f32, 2.0_f32])
                .unwrap()
                .sum()
                .unwrap(),
            IdxTensor::from_diag(indices(), vec![1.0_f32, 2.0_f32])
                .unwrap()
                .enable_grad()
                .unwrap()
                .sum()
                .unwrap(),
        ] {
            assert!(matches!(tensor.value(), ScalarValue::F32(3.0)));
        }
        for tensor in [
            IdxTensor::from_diag(
                indices(),
                vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
            )
            .unwrap()
            .sum()
            .unwrap(),
            IdxTensor::from_diag(
                indices(),
                vec![Complex32::new(1.0, 2.0), Complex32::new(3.0, 4.0)],
            )
            .unwrap()
            .enable_grad()
            .unwrap()
            .sum()
            .unwrap(),
        ] {
            assert!(
                matches!(tensor.value(), ScalarValue::C32(value) if value == Complex32::new(4.0, 6.0))
            );
        }
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
        assert!(error
            .source
            .downcast_ref::<AnyScalarTensorError>()
            .is_some());
        assert!(error
            .source
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
    }

    #[test]
    fn tracked_scalar_operation_failure_retains_error_and_graph_state() {
        let scalar = AnyScalar::new_real(2.0).enable_grad().unwrap();
        let result = with_forced_tensor_initialization_failure(|| scalar.powf(2.0));

        assert!(result.tracks_grad());
        assert_eq!(result.real(), 4.0);

        let error = result.as_tensor().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        assert!(error
            .to_string()
            .contains("AnyScalar tensor initialization failed"));
        assert!(error
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));

        let error = result.clone().enable_grad().unwrap_err();
        assert!(error
            .source
            .downcast_ref::<AnyScalarTensorError>()
            .is_some());
        assert!(error
            .source
            .chain()
            .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
    }

    #[test]
    fn tracked_backend_failure_preserves_typed_diagnostic_through_fallback() {
        let lhs = AnyScalar::new_real(2.0).enable_grad().unwrap();
        let rhs = AnyScalar::new_real(3.0).enable_grad().unwrap();
        let operation = AnyScalar::from_eager_binary(&lhs, &rhs, "add", |_lhs, _rhs| {
            Err(tenferro_tensor::Error::backend_failure(
                "forced_add",
                "forced tracked backend failure",
            ))
        });
        let result = AnyScalar::fallback_result(operation, "add", || ScalarValue::F64(5.0), true);

        assert!(result.tracks_grad());
        assert_eq!(result.real(), 5.0);
        let error = result.as_tensor().unwrap_err();
        assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
        let stored = error.downcast_ref::<AnyScalarTensorError>().unwrap();
        match stored {
            AnyScalarTensorError::Operation { source, .. } => {
                assert!(source
                    .downcast_ref::<tenferro_tensor::Error>()
                    .is_some_and(|error| error
                        .to_string()
                        .contains("forced tracked backend failure")));
            }
            AnyScalarTensorError::Initialization { .. } => {
                panic!("operation failure was converted to initialization failure")
            }
        }
        let error = result.enable_grad().unwrap_err();
        assert!(error.to_string().contains("forced tracked backend failure"));
    }

    #[test]
    fn every_infallible_scalar_fallback_retains_a_tracked_error() {
        let failed = AnyScalar {
            tensor: Err(AnyScalarTensorError::Operation {
                op: "seed",
                source: Arc::new(std::io::Error::other("forced tracked scalar failure")),
            }),
            value: ScalarValue::F64(2.0),
            tracks_grad: true,
        };
        let one = AnyScalar::new_real(1.0);

        let results = [
            &failed + &one,
            &failed * &one,
            &failed / &one,
            -&failed,
            failed.conj(),
            failed.real_part(),
            failed.imag_part(),
            failed.sqrt(),
            failed.powf(2.0),
            failed.powi(2),
        ];
        for result in results {
            assert!(result.tracks_grad());
            let error = result.as_tensor().unwrap_err();
            assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
            assert!(error
                .chain()
                .any(|cause| cause.to_string() == "forced tracked scalar failure"));
        }
    }

    #[test]
    fn every_infallible_scalar_operation_retains_an_initialization_error() {
        let failed = with_forced_tensor_initialization_failure(|| AnyScalar::new_real(2.0));
        let one = AnyScalar::new_real(1.0);

        let results = [
            &failed + &one,
            &failed * &one,
            &failed / &one,
            -&failed,
            failed.conj(),
            failed.real_part(),
            failed.imag_part(),
            failed.sqrt(),
            failed.powf(2.0),
            failed.powi(0),
        ];
        for result in results {
            let error = result.as_tensor().unwrap_err();
            assert!(error.downcast_ref::<AnyScalarTensorError>().is_some());
            assert!(error
                .chain()
                .any(|cause| cause.to_string() == "forced AnyScalar eager initialization failure"));
        }
    }
}
