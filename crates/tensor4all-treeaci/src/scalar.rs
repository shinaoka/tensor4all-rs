//! Scalar and node-name bounds shared by the tree ACI API.

use std::{fmt::Debug, hash::Hash};

use num_complex::{Complex32, Complex64};
use tensor4all_core::AnyScalar;
use tensor4all_tensorbackend::TensorElement;

/// A scalar supported by TreeACI evaluation and rank-revealing LU.
///
/// This combines the tensor-backend, generic TCI, and MatrixLUCI contracts so
/// all numerical requirements are checked at the public API boundary.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciScalar;
///
/// fn accepts<T: TreeAciScalar>() {}
/// accepts::<f64>();
/// accepts::<num_complex::Complex64>();
/// ```
pub trait TreeAciScalar:
    TensorElement + tensor4all_core::Scalar + tensor4all_core::MatrixLuciScalar
{
    /// Converts a scalar returned by a high-level TreeTN evaluator.
    ///
    /// Real and complex kinds must agree with `Self`; precision conversion
    /// within the same kind is permitted.
    ///
    /// # Errors
    ///
    /// Returns `Err("cannot decode ...")` when the evaluator's input dtype is
    /// incompatible with `Self`, notably when a real TreeACI run receives a
    /// complex value.
    fn from_evaluated_scalar(value: AnyScalar) -> std::result::Result<Self, &'static str>;
}

impl TreeAciScalar for f32 {
    fn from_evaluated_scalar(value: AnyScalar) -> std::result::Result<Self, &'static str> {
        value
            .as_f64()
            .map(|real| real as f32)
            .ok_or("cannot decode a complex TreeTN value as f32")
    }
}

impl TreeAciScalar for f64 {
    fn from_evaluated_scalar(value: AnyScalar) -> std::result::Result<Self, &'static str> {
        value
            .as_f64()
            .ok_or("cannot decode a complex TreeTN value as f64")
    }
}

impl TreeAciScalar for Complex32 {
    fn from_evaluated_scalar(value: AnyScalar) -> std::result::Result<Self, &'static str> {
        if let Some(complex) = value.as_c64() {
            return Ok(Complex32::new(complex.re as f32, complex.im as f32));
        }
        value
            .as_f64()
            .map(|real| Complex32::new(real as f32, 0.0))
            .ok_or("cannot decode TreeTN value as Complex32")
    }
}

impl TreeAciScalar for Complex64 {
    fn from_evaluated_scalar(value: AnyScalar) -> std::result::Result<Self, &'static str> {
        if let Some(complex) = value.as_c64() {
            return Ok(complex);
        }
        value
            .as_f64()
            .map(|real| Complex64::new(real, 0.0))
            .ok_or("cannot decode TreeTN value as Complex64")
    }
}

/// A node-name type that can be used in deterministic tree ACI plans.
///
/// Ordering makes traversal independent of `petgraph` insertion order, while
/// hashing supports constant-time prepared-problem lookup.
///
/// # Examples
///
/// ```
/// use tensor4all_treeaci::TreeAciNode;
///
/// fn accepts<V: TreeAciNode>() {}
/// accepts::<usize>();
/// accepts::<String>();
/// ```
pub trait TreeAciNode: Clone + Debug + Eq + Hash + Ord + Send + Sync + 'static {}

impl<V> TreeAciNode for V where V: Clone + Debug + Eq + Hash + Ord + Send + Sync + 'static {}

#[cfg(test)]
mod tests {
    use num_complex::{Complex32, Complex64};
    use tensor4all_core::AnyScalar;

    use super::TreeAciScalar;

    #[test]
    fn evaluated_scalars_preserve_real_and_complex_kinds() {
        assert_eq!(
            f64::from_evaluated_scalar(AnyScalar::new_real(2.5)).unwrap(),
            2.5
        );
        assert!(f64::from_evaluated_scalar(AnyScalar::new_complex(2.5, 1.0)).is_err());
        assert_eq!(
            Complex32::from_evaluated_scalar(AnyScalar::new_complex(2.5, -1.0)).unwrap(),
            Complex32::new(2.5, -1.0)
        );
        assert_eq!(
            Complex64::from_evaluated_scalar(AnyScalar::new_real(2.5)).unwrap(),
            Complex64::new(2.5, 0.0)
        );
    }
}
