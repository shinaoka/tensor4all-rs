//! Difference-kernel MPO construction.

use crate::error::QuanticsTransformError;
use anyhow::Result;
use num_complex::Complex64;
use num_traits::Zero;
use tensor4all_simplett::{tensor3_from_data, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::affine::{affine_transform_tensors_unfused, AffineParams};
use crate::common::{
    checked_allocation_len, tensortrain_to_linear_operator, try_vec_with_capacity,
    BoundaryCondition, QuanticsOperator,
};

/// Build an MPO for the one-dimensional difference kernel `A[x, x'] = f(x - x')`.
///
/// The input `f` is a binary QTT over the difference coordinate `z`. The output
/// MPO has one binary output leg `x` and one binary input leg `x'` per site,
/// encoded as a fused local index `x_bit * 2 + xprime_bit`.
///
/// `BoundaryCondition::Periodic` uses `z = (x - x') mod 2^R`.
/// `BoundaryCondition::AntiPeriodic` multiplies by `-1` when `x < x'`.
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
pub fn difference_kernel_mpo(
    f: &TensorTrain<Complex64>,
    boundary: BoundaryCondition,
) -> std::result::Result<TensorTrain<Complex64>, QuanticsTransformError> {
    if f.len() == 0 {
        return Err(anyhow::anyhow!("difference kernel requires a non-empty QTT").into());
    }
    if boundary == BoundaryCondition::Open {
        return Err(
            anyhow::anyhow!("Open boundary is not supported for difference kernels").into(),
        );
    }
    for site in 0..f.len() {
        let tensor = f.site_tensor(site);
        if tensor.site_dim() != 2 {
            return Err(anyhow::anyhow!(
                "difference kernel requires binary QTT cores; site {site} has site_dim={}",
                tensor.site_dim()
            )
            .into());
        }
    }

    let params = AffineParams::from_integers(vec![1, -1], vec![0], 1, 2)?;
    let delta = affine_transform_tensors_unfused(f.len(), &params, &[boundary])?;
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "difference-kernel MPO site list",
        f.len(),
    )?;

    for (site, delta_core) in delta.iter().enumerate() {
        let f_core = f.site_tensor(site);

        let delta_left = delta_core.left_dim();
        let delta_right = delta_core.right_dim();
        let f_left = f_core.left_dim();
        let f_right = f_core.right_dim();

        let (left_dim, right_dim, total_size) =
            checked_difference_tensor_dims(delta_left, f_left, delta_right, f_right)?;
        let mut data = try_vec_with_capacity::<Complex64>("difference-kernel tensor", total_size)?;
        data.resize(total_size, Complex64::zero());
        let mut out = tensor3_from_data(data, left_dim, 4, right_dim)
            .map_err(|err| anyhow::anyhow!("Failed to allocate difference-kernel tensor: {err}"))?;

        for dl in 0..delta_left {
            for fl in 0..f_left {
                let left = dl * f_left + fl;
                for x_bit in 0..2 {
                    for xp_bit in 0..2 {
                        let mpo_site = x_bit * 2 + xp_bit;
                        for dr in 0..delta_right {
                            for fr in 0..f_right {
                                let right = dr * f_right + fr;
                                let mut value = Complex64::zero();
                                for z_bit in 0..2 {
                                    let delta_site = z_bit + 2 * x_bit + 4 * xp_bit;
                                    value += *delta_core.get3(dl, delta_site, dr)
                                        * *f_core.get3(fl, z_bit, fr);
                                }
                                if value != Complex64::zero() {
                                    let old = *out.get3(left, mpo_site, right);
                                    out.set3(left, mpo_site, right, old + value);
                                }
                            }
                        }
                    }
                }
            }
        }

        tensors.push(out);
    }

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create difference-kernel MPO: {e}"))
        .map_err(QuanticsTransformError::from)
}

fn checked_difference_tensor_dims(
    delta_left: usize,
    f_left: usize,
    delta_right: usize,
    f_right: usize,
) -> Result<(usize, usize, usize)> {
    let left_dim = delta_left
        .checked_mul(f_left)
        .ok_or_else(|| anyhow::anyhow!("difference-kernel left bond product overflows usize"))?;
    let right_dim = delta_right
        .checked_mul(f_right)
        .ok_or_else(|| anyhow::anyhow!("difference-kernel right bond product overflows usize"))?;
    let total_size =
        checked_allocation_len::<Complex64>(&[left_dim, 4, right_dim], "difference-kernel tensor")?;
    Ok((left_dim, right_dim, total_size))
}

/// Build a linear operator for the one-dimensional difference kernel.
///
/// See [`difference_kernel_mpo`] for the exact boundary convention and tensor
/// layout.
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
pub fn difference_kernel_operator(
    f: &TensorTrain<Complex64>,
    boundary: BoundaryCondition,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    let mpo = difference_kernel_mpo(f, boundary)?;
    let mut site_dims =
        try_vec_with_capacity::<usize>("difference-kernel site dimensions", f.len())?;
    site_dims.resize(f.len(), 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difference_tensor_dimensions_reject_product_overflow() {
        let error = checked_difference_tensor_dims(usize::MAX, 2, 1, 1).unwrap_err();
        assert!(error.to_string().contains("left bond product"));

        let error = checked_difference_tensor_dims(1, 1, usize::MAX, 2).unwrap_err();
        assert!(error.to_string().contains("right bond product"));
    }

    #[test]
    fn difference_tensor_backing_reservation_reports_failure() {
        let error =
            try_vec_with_capacity::<Complex64>("difference-kernel tensor", usize::MAX).unwrap_err();
        assert!(
            error.to_string().contains("allocation failed"),
            "unexpected error: {error}"
        );
    }
}
