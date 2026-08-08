//! Phase rotation operator: f(x) = exp(i*θ*x) * g(x)
//!
//! This transformation multiplies the function by a phase factor.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::One;
use std::f64::consts::PI;
use tensor4all_simplett::{types::tensor3_zeros, Tensor3Ops, TensorTrain};

use crate::common::{
    checked_multivar_dims, embed_single_var_mpo, tensortrain_to_linear_operator,
    tensortrain_to_linear_operator_asymmetric, QuanticsOperator,
};

/// Create a phase rotation operator: f(x) = exp(i*θ*x) * g(x)
///
/// This MPO multiplies a function g(x) by the phase factor exp(i*θ*x).
///
/// In quantics representation, x = Σ_n x_n * 2^(R-n), so:
/// exp(i*θ*x) = Π_n exp(i*θ*2^(R-n)*x_n)
///
/// Each site contributes an independent phase factor, making this a diagonal
/// operator with bond dimension 1.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `theta` - Phase angle in radians
///
/// # Returns
/// LinearOperator representing the phase rotation
///
/// # Errors
/// Returns an error when `r` is zero, when `theta` is not finite, or when
/// internal MPO/operator construction fails.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::phase_rotation_operator;
/// use std::f64::consts::PI;
///
/// // Create phase rotation by π/4 for 4-bit quantics
/// let op = phase_rotation_operator(4, PI / 4.0).unwrap();
///
/// // The operator has one MPO tensor per bit
/// assert_eq!(op.mpo().node_count(), 4);
///
/// // Phase rotation is a diagonal operator (bond dimension 1)
/// // Error on invalid input
/// assert!(phase_rotation_operator(0, 1.0).is_err());
/// assert!(phase_rotation_operator(4, f64::NAN).is_err());
/// assert!(phase_rotation_operator(4, f64::INFINITY).is_err());
/// ```
pub fn phase_rotation_operator(r: usize, theta: f64) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }
    if !theta.is_finite() {
        anyhow::bail!("theta must be finite, got {theta}");
    }

    let mpo = phase_rotation_mpo(r, theta)?;
    let site_dims = vec![2; r];
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create a phase rotation operator for one variable in a multi-variable system.
///
/// Acts as phase rotation on `target_var` and identity on all other variables.
/// The resulting operator works on interleaved quantics encoding where each
/// site has local dimension `2^nvariables`.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `theta` - Phase angle in radians
/// * `nvariables` - Total number of variables
/// * `target_var` - Which variable to apply phase rotation to (0-indexed)
///
/// # Errors
/// Returns an error when `r` is zero, when `theta` is not finite, when
/// `nvariables` or `target_var` is invalid, or when embedding/operator
/// construction fails.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::phase_rotation_operator_multivar;
/// use std::f64::consts::PI;
///
/// // Phase rotate only the x-variable of a 2-variable function f(x, y)
/// let op = phase_rotation_operator_multivar(4, PI / 4.0, 2, 0).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn phase_rotation_operator_multivar(
    r: usize,
    theta: f64,
    nvariables: usize,
    target_var: usize,
) -> Result<QuanticsOperator> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }
    if !theta.is_finite() {
        anyhow::bail!("theta must be finite, got {theta}");
    }

    let (dim_multi, _) = checked_multivar_dims(nvariables)?;
    let mpo = phase_rotation_mpo(r, theta)?;
    let embedded = embed_single_var_mpo(&mpo, nvariables, target_var)?;
    let dims = vec![dim_multi; r];
    tensortrain_to_linear_operator_asymmetric(&embedded, &dims, &dims)
}

/// Create the phase rotation MPO as a TensorTrain.
///
/// Each site tensor is diagonal with entries:
/// - For x_n = 0: 1
/// - For x_n = 1: exp(i*θ*2^(R-1-n))
///
/// Uses big-endian convention: site n corresponds to bit 2^(R-1-n) (MSB at site 0).
/// This matches Julia Quantics.jl's convention.
fn phase_rotation_mpo(r: usize, theta: f64) -> Result<TensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }
    if !theta.is_finite() {
        anyhow::bail!("theta must be finite, got {theta}");
    }

    // Normalize theta to [0, 2π)
    let theta_mod = theta.rem_euclid(2.0 * PI);

    let mut tensors = Vec::with_capacity(r);

    for n in 0..r {
        // Phase for this site: θ * 2^(R-1-n) (big-endian: site 0 = MSB)
        // This matches Julia's convention: site n=1 (Julia) has power R-n
        let power = (r - 1 - n) as f64;
        let site_phase = (theta_mod * 2.0_f64.powf(power)).rem_euclid(2.0 * PI);

        // Diagonal MPO tensor: out_bit == in_bit
        // Shape: (1, 4, 1) where 4 = 2*2 for (out, in)
        let mut t = tensor3_zeros(1, 4, 1);

        // s = out_bit * 2 + in_bit
        // Diagonal entries: out_bit == in_bit
        // s=0: (0,0), s=3: (1,1)
        t.set3(0, 0, 0, Complex64::one()); // x_n = 0: phase = 1

        // x_n = 1: phase = exp(i * site_phase)
        let phase_factor = Complex64::new(site_phase.cos(), site_phase.sin());
        t.set3(0, 3, 0, phase_factor);

        tensors.push(t);
    }

    TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create phase rotation MPO: {}", e))
}

#[cfg(test)]
mod tests;
