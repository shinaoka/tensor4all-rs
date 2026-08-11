//! Shift operator for basis-state translations.
//!
//! The operator maps basis states as `|x> -> |x + offset>`. As a matrix action
//! on sampled values, `(M g)[x] = g[x - offset]` with the selected boundary
//! condition.

use crate::error::QuanticsTransformError;
use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{
    types::tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops,
};

use crate::common::{
    checked_multivar_dims, embed_single_var_mpo, tensortrain_to_linear_operator,
    tensortrain_to_linear_operator_asymmetric, try_vec_with_capacity, BoundaryCondition,
    QuanticsOperator,
};

/// Create a shift operator for `|x> -> |x + offset>`.
///
/// As a matrix acting on a value vector `g`, the output satisfies
/// `(M g)[x] = g[x - offset]` for `x = 0, 1, ..., 2^R - 1`, subject to `bc`.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `offset` - Shift amount (can be negative)
/// * `bc` - Boundary condition
///
/// # Returns
/// LinearOperator representing the shift transformation
///
/// # Errors
/// Returns an error when `r` is zero, when `r > 63` would overflow the integer
/// index representation, or when internal MPO/operator construction fails.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{shift_operator, BoundaryCondition};
///
/// // Create a shift operator for 4-bit (2^4 = 16 points) quantics representation
/// let op = shift_operator(4, 3, BoundaryCondition::Periodic).unwrap();
///
/// // The operator has one MPO tensor per bit
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn shift_operator(
    r: usize,
    offset: i64,
    bc: BoundaryCondition,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }

    let mpo = shift_mpo(r, offset, bc)?;
    let mut site_dims = try_vec_with_capacity::<usize>("shift operator site dimensions", r)?;
    site_dims.resize(r, 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create a shift operator for one variable in a multi-variable system.
///
/// Acts as shift on `target_var` and identity on all other variables.
/// The resulting operator works on interleaved quantics encoding where each
/// site has local dimension `2^nvariables`.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `offset` - Shift amount (can be negative)
/// * `bc` - Boundary condition
/// * `nvariables` - Total number of variables (must be at least 2)
/// * `target_var` - Which variable to shift (0-indexed, must be < nvariables)
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{shift_operator_multivar, BoundaryCondition};
///
/// // Shift only the x-variable of a 2-variable function f(x, y) by 3
/// let op = shift_operator_multivar(4, 3, BoundaryCondition::Periodic, 2, 0).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn shift_operator_multivar(
    r: usize,
    offset: i64,
    bc: BoundaryCondition,
    nvariables: usize,
    target_var: usize,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }

    let (dim_multi, _) = checked_multivar_dims(nvariables)?;
    let mpo = shift_mpo(r, offset, bc)?;
    let embedded = embed_single_var_mpo(&mpo, nvariables, target_var)?;
    let mut dims = try_vec_with_capacity::<usize>("shift multivariable site dimensions", r)?;
    dims.resize(r, dim_multi);
    tensortrain_to_linear_operator_asymmetric(&embedded, &dims, &dims)
}

/// Create the shift MPO as a SimpleTensorTrain.
///
/// The shift operation computes x + offset using binary addition with carry propagation.
/// Uses big-endian convention: site n contains bit 2^(R-1-n) (MSB at site 0).
/// This matches Julia Quantics.jl's convention.
///
/// At each site n, we compute: out_n = x_n + offset_n + carry_in (mod 2)
/// with carry_out = (x_n + offset_n + carry_in) / 2
///
/// Carry propagates from LSB to MSB, so in big-endian:
/// - Site 0 (MSB): applies BC on left, receives carry from right
/// - Site R-1 (LSB): initial carry = 0, sends carry to left
pub(crate) fn shift_mpo(
    r: usize,
    offset: i64,
    bc: BoundaryCondition,
) -> Result<SimpleTensorTrain<Complex64>> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive"));
    }
    if r > 63 {
        return Err(anyhow::anyhow!(
            "Number of sites must be at most 63 to avoid integer overflow"
        ));
    }

    if bc == BoundaryCondition::Open && offset < 0 {
        let positive = offset
            .checked_neg()
            .ok_or_else(|| anyhow::anyhow!("open-boundary shift offset overflow"))?;
        let mpo = shift_mpo(r, positive, bc)?;
        return transpose_binary_operator_mpo(&mpo);
    }

    let shift = u32::try_from(r)
        .map_err(|_| anyhow::anyhow!("number of sites {r} exceeds signed shift width"))?;
    let n_max = 1i64
        .checked_shl(shift)
        .ok_or_else(|| anyhow::anyhow!("number of sites {r} exceeds signed shift width"))?;

    // Normalize offset to [0, 2^R)
    let (nbc, offset_mod) = {
        let offset_mod = offset.rem_euclid(n_max);
        let nbc = (offset - offset_mod) / n_max;
        (nbc, offset_mod as usize)
    };

    // Convert offset to binary (big-endian: MSB first)
    // Site n contains bit 2^(R-1-n)
    // offset_bits[n] = bit at position (R-1-n)
    let mut offset_bits = try_vec_with_capacity::<usize>("shift offset bits", r)?;
    for n in 0..r {
        offset_bits.push((offset_mod >> (r - 1 - n)) & 1);
    }

    let mut tensors =
        try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>("shift MPO site list", r)?;

    // Carry states: index 0 = carry 0, index 1 = carry 1
    // For addition, carry can be 0 or 1.
    //
    // In big-endian with SimpleTensorTrain (left-to-right contraction):
    // - Carry flows right-to-left (LSB at R-1 to MSB at 0)
    // - t[left, s, right] where left = carry_out (going left), right = carry_in (from right)

    #[allow(clippy::needless_range_loop)]
    for n in 0..r {
        let y_bit = offset_bits[n]; // The constant bit at position (R-1-n)

        if r == 1 {
            // Single site case: no carry propagation needed
            let mut t = tensor3_zeros(1, 4, 1);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit;
                let out_bit = sum % 2;
                let bc_factor = match bc {
                    BoundaryCondition::Periodic => Complex64::one(),
                    BoundaryCondition::AntiPeriodic => {
                        if sum >= 2 {
                            -Complex64::one()
                        } else {
                            Complex64::one()
                        }
                    }
                    BoundaryCondition::Open => {
                        if sum >= 2 {
                            Complex64::zero()
                        } else {
                            Complex64::one()
                        }
                    }
                };
                let s = out_bit * 2 + x_bit;
                t.set3(0, s, 0, bc_factor);
            }
            tensors.push(t);
        } else if n == 0 {
            // First tensor (MSB): apply BC on left, receive carry from right
            // Shape (1, 4, 2): left=1 (BC applied), site=4, right=2 (carry_in from site 1)
            let bc_val = match bc {
                BoundaryCondition::Periodic => Complex64::one(),
                BoundaryCondition::AntiPeriodic => -Complex64::one(),
                BoundaryCondition::Open => Complex64::zero(),
            };

            let mut t = tensor3_zeros(1, 4, 2);
            for carry_in in 0..2 {
                for x_bit in 0..2 {
                    let sum = x_bit + y_bit + carry_in;
                    let out_bit = sum % 2;
                    let carry_out = sum / 2;

                    // Weight by boundary condition based on carry_out
                    let weight = if carry_out == 0 {
                        Complex64::one()
                    } else {
                        bc_val
                    };

                    let s = out_bit * 2 + x_bit;
                    t.set3(0, s, carry_in, weight);
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor (LSB): initial carry = 0, send carry_out to left
            // Shape (2, 4, 1): left=2 (carry_out to site R-2), site=4, right=1 (no input)
            let mut t = tensor3_zeros(2, 4, 1);
            for x_bit in 0..2 {
                let sum = x_bit + y_bit; // carry_in = 0 at start
                let out_bit = sum % 2;
                let carry_out = sum / 2;
                let s = out_bit * 2 + x_bit;
                t.set3(carry_out, s, 0, Complex64::one());
            }
            tensors.push(t);
        } else {
            // Middle tensors: receive carry from right, send carry to left
            // Shape (2, 4, 2): left=2 (carry_out), site=4, right=2 (carry_in)
            let mut t = tensor3_zeros(2, 4, 2);
            for carry_in in 0..2 {
                for x_bit in 0..2 {
                    let sum = x_bit + y_bit + carry_in;
                    let out_bit = sum % 2;
                    let carry_out = sum / 2;
                    let s = out_bit * 2 + x_bit;
                    t.set3(carry_out, s, carry_in, Complex64::one());
                }
            }
            tensors.push(t);
        }
    }

    let mut mpo = SimpleTensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create shift MPO: {}", e))?;

    // Apply overall boundary condition factor for number of full cycles
    if nbc != 0 {
        let bc_factor = match bc {
            BoundaryCondition::Periodic => Complex64::one(),
            BoundaryCondition::AntiPeriodic => {
                if nbc.rem_euclid(2) == 0 {
                    Complex64::one()
                } else {
                    -Complex64::one()
                }
            }
            BoundaryCondition::Open => {
                // `nbc` is an Euclidean quotient, so negative offsets in (-n_max, 0)
                // still produce `nbc = -1`. Only true full-cycle offsets should zero.
                if offset >= n_max || offset <= -n_max {
                    Complex64::zero()
                } else {
                    Complex64::one()
                }
            }
        };
        mpo.scale_mut(bc_factor);
    }

    Ok(mpo)
}

fn transpose_binary_operator_mpo(
    mpo: &SimpleTensorTrain<Complex64>,
) -> Result<SimpleTensorTrain<Complex64>> {
    let mut transposed = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "transposed shift MPO site list",
        mpo.len(),
    )?;
    for site in 0..mpo.len() {
        let tensor = mpo.site_tensor(site);
        let mut new_tensor =
            tensor3_zeros(tensor.left_dim(), tensor.site_dim(), tensor.right_dim());
        for left in 0..tensor.left_dim() {
            for right in 0..tensor.right_dim() {
                for out_bit in 0..2 {
                    for in_bit in 0..2 {
                        let old_site = out_bit * 2 + in_bit;
                        let new_site = in_bit * 2 + out_bit;
                        new_tensor.set3(left, new_site, right, *tensor.get3(left, old_site, right));
                    }
                }
            }
        }
        transposed.push(new_tensor);
    }

    SimpleTensorTrain::new(transposed)
        .map_err(|e| anyhow::anyhow!("Failed to transpose binary shift MPO: {}", e))
}

#[cfg(test)]
mod tests;
