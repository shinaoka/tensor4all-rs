//! Flip operator: f(x) = g(2^R - x)
//!
//! This transformation maps x -> 2^R - x in quantics representation.

use crate::error::QuanticsTransformError;
use anyhow::Result;
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, SimpleTensorTrain, Tensor3Ops};

use crate::common::{
    checked_multivar_dims, embed_single_var_mpo, tensortrain_to_linear_operator,
    tensortrain_to_linear_operator_asymmetric, try_vec_with_capacity, BoundaryCondition,
    QuanticsOperator,
};

/// Create a flip operator: f(x) = g(2^R - x)
///
/// This MPO transforms a function g(x) to f(x) = g(2^R - x) for x = 0, 1, ..., 2^R - 1.
///
/// # Arguments
/// * `r` - Number of bits (sites)
/// * `bc` - Boundary condition (affects behavior at x=0)
///
/// # Returns
/// LinearOperator representing the flip transformation
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{flip_operator, BoundaryCondition};
///
/// // Create a flip operator for 4-bit (2^4 = 16 points) quantics representation
/// let op = flip_operator(4, BoundaryCondition::Periodic).unwrap();
///
/// // The operator has one MPO tensor per bit
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn flip_operator(
    r: usize,
    bc: BoundaryCondition,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }
    if r == 1 {
        return Err(
            anyhow::anyhow!("MPO with one tensor is not supported for flip operator").into(),
        );
    }

    let mpo = flip_mpo(r, bc)?;
    let mut site_dims = try_vec_with_capacity::<usize>("flip operator site dimensions", r)?;
    site_dims.resize(r, 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create a flip operator for one variable in a multi-variable system.
///
/// Acts as flip on `target_var` and identity on all other variables.
/// The resulting operator works on interleaved quantics encoding where each
/// site has local dimension `2^nvariables`.
///
/// # Arguments
/// * `r` - Number of bits (sites). Must be at least 2.
/// * `bc` - Boundary condition
/// * `nvariables` - Total number of variables (must be at least 2)
/// * `target_var` - Which variable to flip (0-indexed, must be < nvariables)
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{flip_operator_multivar, BoundaryCondition};
///
/// // Flip only the x-variable of a 2-variable function f(x, y)
/// let op = flip_operator_multivar(4, BoundaryCondition::Periodic, 2, 0).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn flip_operator_multivar(
    r: usize,
    bc: BoundaryCondition,
    nvariables: usize,
    target_var: usize,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r == 0 {
        return Err(anyhow::anyhow!("Number of sites must be positive").into());
    }
    if r == 1 {
        return Err(
            anyhow::anyhow!("MPO with one tensor is not supported for flip operator").into(),
        );
    }

    let (dim_multi, _) = checked_multivar_dims(nvariables)?;
    let mpo = flip_mpo(r, bc)?;
    let embedded = embed_single_var_mpo(&mpo, nvariables, target_var)?;
    let mut dims = try_vec_with_capacity::<usize>("flip multivariable site dimensions", r)?;
    dims.resize(r, dim_multi);
    tensortrain_to_linear_operator_asymmetric(&embedded, &dims, &dims)
}

/// Create the flip MPO as a SimpleTensorTrain.
///
/// The flip operation computes 2^R - x using two's complement-like arithmetic.
///
/// Uses big-endian convention: site 0 = MSB, site R-1 = LSB.
/// This matches Julia Quantics.jl's convention.
///
/// This is implemented using carry propagation where:
/// - carry_in values: [-1, 0] (we start with +1 to compute 2^R - x = ~x + 1)
/// - For each bit: out = -(a - 1) + carry_in
/// - carry_out = out < 0 ? -1 : 0
/// - result_bit = out mod 2
///
/// Carry propagates from LSB to MSB, so in big-endian convention:
/// - Site 0 (MSB): has carry input from site 1, applies boundary condition
/// - Site R-1 (LSB): initial carry = 0, has carry output to site R-2
#[allow(clippy::needless_range_loop)]
fn flip_mpo(r: usize, bc: BoundaryCondition) -> Result<SimpleTensorTrain<Complex64>> {
    let single_tensor = single_tensor_flip();
    let mut tensors =
        try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>("flip MPO site list", r)?;

    // Create link indices with dimension 2 (for carry states)
    // Carry states: index 0 = carry -1, index 1 = carry 0
    //
    // In big-endian convention with SimpleTensorTrain (left-to-right contraction):
    // - Site 0 (MSB): applies BC, has cout going right (to site 1)
    // - Site R-1 (LSB): initial carry cin=1 (carry=0), receives cin from left
    //
    // Carry propagates LSB → MSB, but in TT we contract left → right.
    // So we store: left_bond = cout (to previous site), right_bond = cin (from next site)
    // But this is reversed from standard. Instead, we flip the tensor storage:
    // t[left, s, right] where left = cout going left, right = cin from right.

    for n in 0..r {
        if n == 0 {
            // First tensor (MSB): apply boundary condition on left, receive cin from right
            //
            // Note: Julia's flipop only supports bc=1 (periodic) or bc=-1 (antisymmetric).
            // For Open BC, we zero out the flip(0) case by setting bc_val = 0.
            let bc_val = match bc {
                BoundaryCondition::Periodic => Complex64::one(),
                BoundaryCondition::AntiPeriodic => -Complex64::one(),
                BoundaryCondition::Open => Complex64::zero(),
            };

            // Shape (1, 4, 2): left=1 (BC applied), site=4, right=2 (cin from site 1)
            let mut t = tensor3_zeros(1, 4, 2);
            for cin in 0..2 {
                for a in 0..2 {
                    for b in 0..2 {
                        let mut sum = Complex64::zero();
                        for cout in 0..2 {
                            // cout=0 (carry=-1) gets weight 1.0, cout=1 (carry=0) gets weight bc
                            let bc_weight = if cout == 0 { Complex64::one() } else { bc_val };
                            sum += single_tensor[cin][cout][a][b] * bc_weight;
                        }
                        let s = a * 2 + b;
                        t.set3(0, s, cin, sum);
                    }
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor (LSB): select initial carry state cin=1 (carry=0), send cout to left
            // Shape (2, 4, 1): left=2 (cout to site R-2), site=4, right=1 (initial cin)
            let mut t = tensor3_zeros(2, 4, 1);
            for cout in 0..2 {
                for a in 0..2 {
                    for b in 0..2 {
                        let val = single_tensor[1][cout][a][b]; // cin=1 (carry=0) is fixed
                        let s = a * 2 + b;
                        t.set3(cout, s, 0, val);
                    }
                }
            }
            tensors.push(t);
        } else {
            // Middle tensors: receive cin from right, send cout to left
            // Shape (2, 4, 2): left=2 (cout to left), site=4, right=2 (cin from right)
            let mut t = tensor3_zeros(2, 4, 2);
            for cin in 0..2 {
                for cout in 0..2 {
                    for a in 0..2 {
                        for b in 0..2 {
                            let val = single_tensor[cin][cout][a][b];
                            let s = a * 2 + b;
                            t.set3(cout, s, cin, val);
                        }
                    }
                }
            }
            tensors.push(t);
        }
    }

    SimpleTensorTrain::new(tensors).map_err(|e| anyhow::anyhow!("Failed to create flip MPO: {}", e))
}

/// Create the single-site tensor for flip operation.
///
/// Returns a 4D tensor [cin][cout][a][b] where:
/// - cin: input carry state (0 = carry -1, 1 = carry 0)
/// - cout: output carry state
/// - a: corresponds to s' (output site index in ITensor convention)
/// - b: corresponds to s (input site index in ITensor convention)
///
/// The flip computes: out = -a + cval[cin]
/// where a is the input bit value and cval = [-1, 0] for cin = [0, 1]
///
/// Note: In the Julia Quantics.jl code:
/// - The loop variable `a` represents the input bit (0 or 1)
/// - The computed `b` represents the output bit
/// - The tensor is stored as tensor[cin, cout, a, b]
/// - The ITensor is created as ITensor(t, (link_l, link_r, s', s))
/// - This means a -> s' (output index) and b -> s (input index)
///
/// In SimpleTensorTrain MPO format, the combined site index is s = s' * 2 + s
/// where s' is the output bit and s is the input bit.
#[allow(clippy::needless_range_loop)]
fn single_tensor_flip() -> [[[[Complex64; 2]; 2]; 2]; 2] {
    let cval = [-1i32, 0i32];
    let mut tensor = [[[[Complex64::zero(); 2]; 2]; 2]; 2];

    for icin in 0..2 {
        for a in 0..2 {
            // a is the input bit value (0 or 1)
            // Formula: out = -a + cval[icin]
            let out = -(a as i32) + cval[icin];
            let icout = if out < 0 { 0 } else { 1 };
            let b = out.rem_euclid(2) as usize; // b is the output bit (0 or 1)
                                                // Store as tensor[cin][cout][a][b] matching Julia exactly
            tensor[icin][icout][a][b] = Complex64::one();
        }
    }

    tensor
}

#[cfg(test)]
mod tests;
