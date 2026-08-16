use num_complex::Complex64;
use rand_distr::{Distribution, StandardNormal};

mod private {
    use super::*;

    pub(super) trait Sealed {
        fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self;
    }

    impl Sealed for f64 {
        fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
            StandardNormal.sample(rng)
        }
    }

    impl Sealed for Complex64 {
        fn sample_standard_normal(rng: &mut rand_chacha::ChaCha8Rng) -> Self {
            let re = StandardNormal.sample(rng);
            let im = StandardNormal.sample(rng);
            Complex64::new(re, im)
        }
    }
}

/// Scalar types supported by Alternating Cross Interpolation.
///
/// Use this trait bound when writing generic helpers around ACI elementwise
/// operations. It combines the tensor-train and matrix-CI scalar requirements
/// needed by the sweep algorithm. The trait is sealed; use `f64` or
/// [`num_complex::Complex64`].
///
/// Related types: [`AciOptions`](crate::AciOptions) configures ACI runs for an
/// `AciScalar`, and [`AciResult`](crate::AciResult) returns a tensor train with
/// the same scalar type.
///
/// # Examples
///
/// ```
/// use tensor4all_aci::AciScalar;
///
/// fn accepts_aci_scalar<T: AciScalar>(value: T) -> T {
///     value
/// }
///
/// assert_eq!(accepts_aci_scalar(3.0_f64), 3.0);
/// ```
#[allow(private_bounds)]
pub trait AciScalar:
    tensor4all_simplett::TTScalar + tensor4all_core::MatrixLuciScalar + Copy + private::Sealed
{
}

impl<T> AciScalar for T where
    T: tensor4all_simplett::TTScalar + tensor4all_core::MatrixLuciScalar + Copy + private::Sealed
{
}

pub(crate) fn sample_standard_normal<T: AciScalar>(rng: &mut rand_chacha::ChaCha8Rng) -> T {
    <T as private::Sealed>::sample_standard_normal(rng)
}
