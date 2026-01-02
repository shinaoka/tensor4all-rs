use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;
use num_complex::{Complex64, ComplexFloat};
use num_traits::{Zero, One};

use crate::storage::{Storage, SumFromStorage};

/// Dynamic scalar value (for dynamic element type tensors).
///
/// Supports both real (`f64`) and complex (`Complex64`) scalar values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyScalar {
    /// Real number (f64)
    F64(f64),
    /// Complex number (Complex64)
    C64(Complex64),
}

impl SumFromStorage for AnyScalar {
    fn sum_from_storage(storage: &Storage) -> Self {
        match storage {
            Storage::DenseF64(_) | Storage::DiagF64(_) => AnyScalar::F64(f64::sum_from_storage(storage)),
            Storage::DenseC64(_) | Storage::DiagC64(_) => AnyScalar::C64(Complex64::sum_from_storage(storage)),
        }
    }
}

impl AnyScalar {
    /// Check if this scalar is complex.
    pub fn is_complex(&self) -> bool {
        matches!(self, AnyScalar::C64(_))
    }

    /// Get the real part of the scalar.
    pub fn real(&self) -> f64 {
        match self {
            AnyScalar::F64(x) => *x,
            AnyScalar::C64(z) => z.re,
        }
    }

    /// Get the absolute value (magnitude).
    pub fn abs(&self) -> f64 {
        match self {
            AnyScalar::F64(x) => x.abs(),
            AnyScalar::C64(z) => z.abs(),
        }
    }

    /// Compute square root.
    /// 
    /// For negative real numbers, returns a complex number with the principal value.
    pub fn sqrt(&self) -> Self {
        match self {
            AnyScalar::F64(x) => {
                if *x >= 0.0 {
                    AnyScalar::F64(x.sqrt())
                } else {
                    AnyScalar::C64(Complex64::new(*x, 0.0).sqrt())
                }
            }
            AnyScalar::C64(z) => AnyScalar::C64(z.sqrt()),
        }
    }

    /// Raise to a floating-point power.
    /// 
    /// For negative real numbers, returns a complex number with the principal value.
    pub fn powf(&self, exp: f64) -> Self {
        match self {
            AnyScalar::F64(x) => {
                if *x >= 0.0 {
                    AnyScalar::F64(x.powf(exp))
                } else {
                    AnyScalar::C64(Complex64::new(*x, 0.0).powf(exp))
                }
            }
            AnyScalar::C64(z) => AnyScalar::C64(z.powf(exp)),
        }
    }

    /// Raise to an integer power.
    pub fn powi(&self, exp: i32) -> Self {
        match self {
            AnyScalar::F64(x) => AnyScalar::F64(x.powi(exp)),
            AnyScalar::C64(z) => AnyScalar::C64(z.powi(exp)),
        }
    }
}

// 四則演算の実装
impl Add for AnyScalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a + b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) + b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a + Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a + b),
        }
    }
}

impl Sub for AnyScalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a - b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) - b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a - Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a - b),
        }
    }
}

impl Mul for AnyScalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a * b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) * b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a * Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a * b),
        }
    }
}

impl Div for AnyScalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => AnyScalar::F64(a / b),
            (AnyScalar::F64(a), AnyScalar::C64(b)) => AnyScalar::C64(Complex64::new(a, 0.0) / b),
            (AnyScalar::C64(a), AnyScalar::F64(b)) => AnyScalar::C64(a / Complex64::new(b, 0.0)),
            (AnyScalar::C64(a), AnyScalar::C64(b)) => AnyScalar::C64(a / b),
        }
    }
}

impl Neg for AnyScalar {
    type Output = Self;

    fn neg(self) -> Self::Output {
        match self {
            AnyScalar::F64(x) => AnyScalar::F64(-x),
            AnyScalar::C64(z) => AnyScalar::C64(-z),
        }
    }
}

// 型変換
impl From<f64> for AnyScalar {
    fn from(x: f64) -> Self {
        AnyScalar::F64(x)
    }
}

impl From<Complex64> for AnyScalar {
    fn from(z: Complex64) -> Self {
        AnyScalar::C64(z)
    }
}

impl std::convert::TryFrom<AnyScalar> for f64 {
    type Error = &'static str;

    fn try_from(value: AnyScalar) -> Result<Self, Self::Error> {
        match value {
            AnyScalar::F64(x) => Ok(x),
            AnyScalar::C64(_) => Err("Cannot convert complex number to f64"),
        }
    }
}

impl From<AnyScalar> for Complex64 {
    fn from(value: AnyScalar) -> Self {
        match value {
            AnyScalar::F64(x) => Complex64::new(x, 0.0),
            AnyScalar::C64(z) => z,
        }
    }
}

// 標準トレイト
impl Default for AnyScalar {
    fn default() -> Self {
        AnyScalar::F64(0.0)
    }
}

impl Zero for AnyScalar {
    fn zero() -> Self {
        AnyScalar::F64(0.0)
    }

    fn is_zero(&self) -> bool {
        match self {
            AnyScalar::F64(x) => x.is_zero(),
            AnyScalar::C64(z) => z.is_zero(),
        }
    }
}

impl One for AnyScalar {
    fn one() -> Self {
        AnyScalar::F64(1.0)
    }
}

impl PartialOrd for AnyScalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (AnyScalar::F64(a), AnyScalar::F64(b)) => a.partial_cmp(b),
            _ => None, // Complex numbers are not ordered
        }
    }
}

impl fmt::Display for AnyScalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnyScalar::F64(x) => write!(f, "{}", x),
            AnyScalar::C64(z) => write!(f, "{}", z),
        }
    }
}

