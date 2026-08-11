//! Cumulative sum operator
//!
//! This transformation computes cumulative sums: y_i = Σ_{j < i} x_j

use crate::error::QuanticsTransformError;
use anyhow::{Context, Result};
use num_complex::Complex64;
use num_traits::{One, Zero};
use tensor4all_simplett::{types::tensor3_zeros, SimpleTensorTrain, Tensor3Ops};

use crate::common::{tensortrain_to_linear_operator, try_vec_with_capacity, QuanticsOperator};

/// Type of triangular matrix for cumsum-like operators.
///
/// # Variants
///
/// - **`Lower`**: Strict lower triangle, M\[i,j\] = 1 when i > j.
///
///   Computes the prefix sum: y\_i = sum of x\_j for j < i.
/// - **`Upper`**: Strict upper triangle, M\[i,j\] = 1 when i < j.
///
///   Computes the suffix sum: y\_i = sum of x\_j for j > i.
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{triangle_operator, TriangleType};
///
/// // Prefix sum (lower triangle)
/// let lower = triangle_operator(4, TriangleType::Lower).unwrap();
/// assert_eq!(lower.mpo().node_count(), 4);
///
/// // Suffix sum (upper triangle)
/// let upper = triangle_operator(4, TriangleType::Upper).unwrap();
/// assert_eq!(upper.mpo().node_count(), 4);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriangleType {
    /// Strict lower triangle: M\[i,j\] = 1 when i > j.
    /// Computes y\_i = sum of x\_j for j < i (prefix sum).
    Lower,
    /// Strict upper triangle: M\[i,j\] = 1 when i < j.
    /// Computes y\_i = sum of x\_j for j > i (suffix sum).
    Upper,
}

/// Create a cumulative sum operator: y_i = Σ_{j < i} x_j
///
/// This MPO implements a strict lower triangular matrix filled with ones.
/// For a function g defined on {0, 1, ..., 2^R - 1}, it computes:
/// f(i) = Σ_{j < i} g(j)
///
/// # Arguments
/// * `r` - Number of bits (sites). Must be at least 2.
///
/// # Returns
/// LinearOperator representing the cumulative sum
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::cumsum_operator;
///
/// let op = cumsum_operator(4).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
///
/// // Requires at least 2 sites
/// assert!(cumsum_operator(1).is_err());
/// assert!(cumsum_operator(0).is_err());
/// ```
pub fn cumsum_operator(r: usize) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r < 2 {
        return Err(QuanticsTransformError::InvalidConfiguration {
            message: format!("Number of sites must be at least 2, got {r}"),
        });
    }

    let mpo = cumsum_mpo(r)?;
    let mut site_dims = try_vec_with_capacity::<usize>("cumsum operator site dimensions", r)?;
    site_dims.resize(r, 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create a triangular matrix operator with the specified triangle type.
///
/// - `Lower`: M\[i,j\] = 1 when i > j (prefix sum: y\_i = sum of x\_j for j < i)
/// - `Upper`: M\[i,j\] = 1 when i < j (suffix sum: y\_i = sum of x\_j for j > i)
///
/// # Arguments
/// * `r` - Number of bits (sites). Must be at least 2.
/// * `triangle` - Which triangle to use
///
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure, or a shape mismatch).
///
/// # Examples
///
/// ```
/// use tensor4all_quanticstransform::{triangle_operator, TriangleType};
///
/// let op = triangle_operator(4, TriangleType::Lower).unwrap();
/// assert_eq!(op.mpo().node_count(), 4);
///
/// // Requires at least 2 sites
/// assert!(triangle_operator(1, TriangleType::Lower).is_err());
/// ```
pub fn triangle_operator(
    r: usize,
    triangle: TriangleType,
) -> std::result::Result<QuanticsOperator, QuanticsTransformError> {
    if r < 2 {
        return Err(QuanticsTransformError::InvalidConfiguration {
            message: format!("Number of sites must be at least 2, got {r}"),
        });
    }

    let mpo = triangle_mpo(r, triangle)?;
    let mut site_dims = try_vec_with_capacity::<usize>("triangle operator site dimensions", r)?;
    site_dims.resize(r, 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the cumulative sum MPO as a SimpleTensorTrain.
///
/// The cumulative sum is implemented as an upper triangular matrix.
/// The MPO tracks whether a comparison has been made:
/// - State 0: No comparison yet (y and x equal so far)
/// - State 1: Comparison made (y > x, so this entry is 1)
///
/// Tensor entries t[left, right, y, x]:
/// - t[0, 0, y, x] = 1 if y == x (both 0 or both 1)
/// - t[0, 1, 1, 0] = 1 (y > x at this position)
/// - t[1, 1, *, *] = 1 (comparison already made)
#[allow(clippy::needless_range_loop)]
fn cumsum_mpo(r: usize) -> Result<SimpleTensorTrain<Complex64>> {
    if r < 2 {
        return Err(anyhow::anyhow!(
            "Number of sites must be at least 2, got {r}"
        ));
    }

    let single_tensor = upper_triangle_tensor();
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "cumulative-sum MPO site list",
        r,
    )?;

    for n in 0..r {
        if n == 0 {
            // First tensor: start in state 0 (no comparison yet)
            // Contract with [1, 0] on left link to select state 0
            let mut t = tensor3_zeros(1, 4, 2);
            for cout in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        let val = single_tensor[0][cout][y_bit][x_bit];
                        let s = y_bit * 2 + x_bit;
                        t.set3(0, s, cout, val);
                    }
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            // Last tensor: select entries where state is 1 (y > x)
            // The output is 1 only if comparison was made (state 1)
            // For upper triangle (strict), diagonal is excluded
            let mut t = tensor3_zeros(2, 4, 1);
            for cin in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        // Sum over cout states, weighted by whether comparison was made
                        let mut sum = Complex64::zero();
                        for cout in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            // Only count if we end in state 1 (comparison made)
                            if cout == 1 {
                                sum += val;
                            }
                        }
                        let s = y_bit * 2 + x_bit;
                        t.set3(cin, s, 0, sum);
                    }
                }
            }
            tensors.push(t);
        } else {
            // Middle tensors: full tensor
            let mut t = tensor3_zeros(2, 4, 2);
            for cin in 0..2 {
                for cout in 0..2 {
                    for y_bit in 0..2 {
                        for x_bit in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            let s = y_bit * 2 + x_bit;
                            t.set3(cin, s, cout, val);
                        }
                    }
                }
            }
            tensors.push(t);
        }
    }

    SimpleTensorTrain::new(tensors)
        .map_err(anyhow::Error::new)
        .context("Failed to create cumsum MPO")
}

/// Create a triangular matrix MPO as a SimpleTensorTrain.
///
/// This is a generalization of `cumsum_mpo` that supports both upper and lower triangles.
#[allow(clippy::needless_range_loop)]
fn triangle_mpo(r: usize, triangle: TriangleType) -> Result<SimpleTensorTrain<Complex64>> {
    if r < 2 {
        return Err(anyhow::anyhow!(
            "Number of sites must be at least 2, got {r}"
        ));
    }

    // upper_triangle_tensor() has y>x transition → M[i,j]=1 when i>j = Lower triangle
    // lower_triangle_tensor() has y<x transition → M[i,j]=1 when i<j = Upper triangle
    let single_tensor = match triangle {
        TriangleType::Lower => upper_triangle_tensor(),
        TriangleType::Upper => lower_triangle_tensor(),
    };
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "triangle MPO site list",
        r,
    )?;

    for n in 0..r {
        if n == 0 {
            let mut t = tensor3_zeros(1, 4, 2);
            for cout in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        let val = single_tensor[0][cout][y_bit][x_bit];
                        let s = y_bit * 2 + x_bit;
                        t.set3(0, s, cout, val);
                    }
                }
            }
            tensors.push(t);
        } else if n == r - 1 {
            let mut t = tensor3_zeros(2, 4, 1);
            for cin in 0..2 {
                for y_bit in 0..2 {
                    for x_bit in 0..2 {
                        let mut sum = Complex64::zero();
                        for cout in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            if cout == 1 {
                                sum += val;
                            }
                        }
                        let s = y_bit * 2 + x_bit;
                        t.set3(cin, s, 0, sum);
                    }
                }
            }
            tensors.push(t);
        } else {
            let mut t = tensor3_zeros(2, 4, 2);
            for cin in 0..2 {
                for cout in 0..2 {
                    for y_bit in 0..2 {
                        for x_bit in 0..2 {
                            let val = single_tensor[cin][cout][y_bit][x_bit];
                            let s = y_bit * 2 + x_bit;
                            t.set3(cin, s, cout, val);
                        }
                    }
                }
            }
            tensors.push(t);
        }
    }

    SimpleTensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create triangle MPO: {}", e))
}

/// Create the single-site tensor for upper triangular matrix.
///
/// Returns a 4D tensor [cin][cout][y_bit][x_bit] where:
/// - cin: input state (0 = no comparison yet, 1 = comparison made)
/// - cout: output state
/// - y_bit: output (row) bit
/// - x_bit: input (column) bit
///
/// The tensor implements strict upper triangle comparison:
/// - State 0: Comparing bits. If y > x, transition to state 1.
/// - State 1: Comparison made. All remaining entries are 1.
fn upper_triangle_tensor() -> [[[[Complex64; 2]; 2]; 2]; 2] {
    let mut tensor = [[[[Complex64::zero(); 2]; 2]; 2]; 2];

    // State 0 -> State 0: y == x (continue comparing)
    tensor[0][0][0][0] = Complex64::one(); // y=0, x=0
    tensor[0][0][1][1] = Complex64::one(); // y=1, x=1

    // State 0 -> State 1: y > x (comparison made, y is greater)
    tensor[0][1][1][0] = Complex64::one(); // y=1, x=0 (y > x at this bit)

    // State 0 -> nowhere: y < x (this entry is 0, not in upper triangle)
    // tensor[0][*][0][1] = 0 (implicit)

    // State 1 -> State 1: Comparison already made, all entries are 1
    tensor[1][1][0][0] = Complex64::one();
    tensor[1][1][0][1] = Complex64::one();
    tensor[1][1][1][0] = Complex64::one();
    tensor[1][1][1][1] = Complex64::one();

    tensor
}

/// Create the single-site tensor for strict upper triangle (i < j).
///
/// M[i,j] = 1 when i < j (suffix sum: y_i = Σ_{j > i} x_j).
///
/// The tensor is identical to lower_triangle but with the transition
/// on y < x (y=0, x=1) instead of y > x (y=1, x=0).
fn lower_triangle_tensor() -> [[[[Complex64; 2]; 2]; 2]; 2] {
    let mut tensor = [[[[Complex64::zero(); 2]; 2]; 2]; 2];

    // State 0 -> State 0: y == x (continue comparing)
    tensor[0][0][0][0] = Complex64::one(); // y=0, x=0
    tensor[0][0][1][1] = Complex64::one(); // y=1, x=1

    // State 0 -> State 1: y < x (comparison made, x is greater)
    tensor[0][1][0][1] = Complex64::one(); // y=0, x=1 (y < x at this bit)

    // State 1 -> State 1: Comparison already made, all entries are 1
    tensor[1][1][0][0] = Complex64::one();
    tensor[1][1][0][1] = Complex64::one();
    tensor[1][1][1][0] = Complex64::one();
    tensor[1][1][1][1] = Complex64::one();

    tensor
}

#[cfg(test)]
mod tests;
