//! Difference-kernel MPO construction.

use anyhow::{bail, Result};
use num_complex::Complex64;
use num_traits::Zero;
use tensor4all_simplett::{types::tensor3_zeros, AbstractTensorTrain, Tensor3Ops, TensorTrain};

use crate::affine::{affine_transform_tensors_unfused, AffineParams};
use crate::common::{
    tensortrain_to_linear_operator, try_vec_with_capacity, BoundaryCondition, QuanticsOperator,
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
/// Returns an error when `f` is empty, when any site dimension is not binary,
/// or when `boundary` is [`BoundaryCondition::Open`].
pub fn difference_kernel_mpo(
    f: &TensorTrain<Complex64>,
    boundary: BoundaryCondition,
) -> Result<TensorTrain<Complex64>> {
    if f.len() == 0 {
        bail!("difference kernel requires a non-empty QTT");
    }
    if boundary == BoundaryCondition::Open {
        bail!("Open boundary is not supported for difference kernels");
    }
    for site in 0..f.len() {
        let tensor = f.site_tensor(site);
        if tensor.site_dim() != 2 {
            bail!(
                "difference kernel requires binary QTT cores; site {site} has site_dim={}",
                tensor.site_dim()
            );
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

        let mut out = tensor3_zeros(delta_left * f_left, 4, delta_right * f_right);

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
}

/// Build a linear operator for the one-dimensional difference kernel.
///
/// See [`difference_kernel_mpo`] for the exact boundary convention and tensor
/// layout.
///
/// # Errors
///
/// Returns an error when [`difference_kernel_mpo`] fails or when the MPO cannot
/// be wrapped as a [`QuanticsOperator`].
pub fn difference_kernel_operator(
    f: &TensorTrain<Complex64>,
    boundary: BoundaryCondition,
) -> Result<QuanticsOperator> {
    let mpo = difference_kernel_mpo(f, boundary)?;
    let mut site_dims =
        try_vec_with_capacity::<usize>("difference-kernel site dimensions", f.len())?;
    site_dims.resize(f.len(), 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}
