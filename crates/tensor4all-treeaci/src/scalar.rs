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
    TensorElement
    + tensor4all_core::Scalar
    + tensor4all_core::MatrixLuciScalar
    + tensor4all_tensorbackend::MatrixScalar
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

    /// A real TreeACI run must reject a complex evaluator value rather than
    /// silently dropping the imaginary part, at both real precisions.
    #[test]
    fn real_scalars_reject_complex_evaluator_values() {
        assert_eq!(
            f32::from_evaluated_scalar(AnyScalar::new_real(-0.5)).unwrap(),
            -0.5_f32
        );
        let f32_error = f32::from_evaluated_scalar(AnyScalar::new_complex(1.0, 2.0)).unwrap_err();
        assert!(
            f32_error.contains("f32"),
            "f32 rejection should name the target type: {f32_error}"
        );
        let f64_error = f64::from_evaluated_scalar(AnyScalar::new_complex(1.0, 2.0)).unwrap_err();
        assert!(
            f64_error.contains("f64"),
            "f64 rejection should name the target type: {f64_error}"
        );
    }

    /// A complex TreeACI run accepts a real evaluator value by widening it, so
    /// an input that happens to be real does not abort the run. Both complex
    /// precisions take both paths.
    #[test]
    fn complex_scalars_widen_real_evaluator_values() {
        assert_eq!(
            Complex32::from_evaluated_scalar(AnyScalar::new_real(1.25)).unwrap(),
            Complex32::new(1.25, 0.0)
        );
        assert_eq!(
            Complex64::from_evaluated_scalar(AnyScalar::new_complex(1.25, -3.5)).unwrap(),
            Complex64::new(1.25, -3.5)
        );
    }

    /// `TreeAciScalar` alone (no explicit `MatrixScalar` bound at the call
    /// site) must be enough to satisfy a function that requires
    /// `MatrixScalar`. This only compiles because `TreeAciScalar` has
    /// `MatrixScalar` as a supertrait; if that supertrait bound were removed,
    /// `requires_matrix_scalar::<T>()` inside `accepts` would fail to
    /// typecheck even though `accepts`'s own bound list is unchanged.
    #[test]
    fn tree_aci_scalar_implies_matrix_scalar() {
        fn requires_matrix_scalar<T: tensor4all_tensorbackend::MatrixScalar>() {}
        fn accepts<T: super::TreeAciScalar>() {
            requires_matrix_scalar::<T>();
        }
        accepts::<f32>();
        accepts::<f64>();
        accepts::<Complex32>();
        accepts::<Complex64>();
    }

    /// Precision conversion within one kind is lossy but permitted; the kind
    /// itself is what must be preserved.
    #[test]
    fn precision_narrows_within_a_kind() {
        let wide = f64::from(f32::MAX) * 2.0;
        assert!(
            f32::from_evaluated_scalar(AnyScalar::new_real(wide))
                .unwrap()
                .is_infinite(),
            "narrowing past f32::MAX saturates to infinity rather than erroring"
        );
        assert_eq!(
            Complex32::from_evaluated_scalar(AnyScalar::new_complex(1.0 / 3.0, 0.0)).unwrap(),
            Complex32::new((1.0_f64 / 3.0) as f32, 0.0)
        );
    }
}
