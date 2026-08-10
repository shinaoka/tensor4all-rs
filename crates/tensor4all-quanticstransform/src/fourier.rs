//! Quantics Fourier Transform (QFT) operator
//!
//! This implements the Chen & Lindsey method for efficient QFT in tensor train form.
//! Reference: J. Chen and M. Lindsey, "Direct Interpolative Construction of the
//! Discrete Fourier Transform as a Matrix Product Operator", arXiv:2404.03182.

use anyhow::Result;
use num_complex::Complex64;
use num_traits::Zero;
use std::f64::consts::PI;
use tensor4all_simplett::mpo::{tensor4_from_data, Tensor4};
use tensor4all_simplett::{
    compression::{CompressionMethod, CompressionOptions},
    tensor3_from_data, Tensor3Ops, TensorTrain,
};

use crate::common::{
    checked_allocation_len, tensortrain_to_linear_operator, try_vec_with_capacity, QuanticsOperator,
};

fn checked_fourier_order(k: usize) -> Result<usize> {
    if k == 0 {
        anyhow::bail!("Fourier Chebyshev order k must be positive");
    }
    k.checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("Fourier Chebyshev grid size overflows usize for k={k}"))
}

fn checked_tensor4_size(dims: [usize; 4], name: &str) -> Result<usize> {
    checked_allocation_len::<Complex64>(&dims, name)
}

fn checked_fourier_core_size(k: usize) -> Result<(usize, usize)> {
    let count = checked_fourier_order(k)?;
    let total = checked_tensor4_size([count, 2, 2, count], "Fourier core tensor")?;
    Ok((count, total))
}

fn checked_tensor3_size(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
    name: &str,
) -> Result<usize> {
    checked_allocation_len::<Complex64>(&[left_dim, site_dim, right_dim], name)
}

fn zero_tensor3(
    left_dim: usize,
    site_dim: usize,
    right_dim: usize,
    name: &str,
) -> Result<tensor4all_simplett::Tensor3<Complex64>> {
    let total_size = checked_tensor3_size(left_dim, site_dim, right_dim, name)?;
    let mut data = try_vec_with_capacity::<Complex64>(name, total_size)?;
    data.resize(total_size, Complex64::zero());
    tensor3_from_data(data, left_dim, site_dim, right_dim)
        .map_err(|err| anyhow::anyhow!("Failed to allocate {name}: {err}"))
}

/// Options for Fourier transform construction.
/// Controls the sign convention, compression parameters, and normalization
/// of the quantics Fourier transform MPO.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::FourierOptions;
/// // Forward transform (default): sign = -1
/// let fwd = FourierOptions::forward();
/// assert_eq!(fwd.sign, -1.0);
/// assert!(fwd.normalize);
/// // Inverse transform: sign = +1
/// let inv = FourierOptions::inverse();
/// assert_eq!(inv.sign, 1.0);
/// // Custom options
/// let opts = FourierOptions {
///     maxbonddim: 20,
///     tolerance: 1e-12,
///     ..FourierOptions::forward()
/// };
/// assert_eq!(opts.maxbonddim, 20);
/// ```
#[derive(Clone, Debug)]
pub struct FourierOptions {
    /// Sign in the exponent: -1.0 (forward) or 1.0 (inverse)
    pub sign: f64,
    /// Maximum bond dimension after compression
    pub maxbonddim: usize,
    /// Tolerance for compression
    pub tolerance: f64,
    /// Number of Chebyshev basis functions (K+1 points); must be positive.
    pub k: usize,
    /// Whether to normalize as an isometry
    pub normalize: bool,
}

impl Default for FourierOptions {
    fn default() -> Self {
        Self {
            sign: -1.0,
            maxbonddim: 12,
            tolerance: 1e-14,
            k: 25,
            normalize: true,
        }
    }
}

impl FourierOptions {
    /// Create options for forward Fourier transform.
    /// # Errors
    ///
    /// Returns an error when the forward transform fails (a shape mismatch or
    /// /// backend failure).
    ///
    pub fn forward() -> Self {
        Self::default()
    }

    /// Create options for inverse Fourier transform.
    pub fn inverse() -> Self {
        Self {
            sign: 1.0,
            ..Self::default()
        }
    }
}

/// Convenience wrapper for forward/backward Fourier transform.
/// Caches the forward MPO so that repeated calls to [`FTCore::forward()`]
/// and [`FTCore::backward()`] avoid redundant MPO construction.
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{FTCore, FourierOptions};
/// let ft = FTCore::new(4, FourierOptions::default()).unwrap();
/// assert_eq!(ft.r(), 4);
/// let fwd_op = ft.forward().unwrap();
/// assert_eq!(fwd_op.mpo().node_count(), 4);
/// let bwd_op = ft.backward().unwrap();
/// assert_eq!(bwd_op.mpo().node_count(), 4);
/// ```
#[derive(Clone)]
pub struct FTCore {
    forward_mpo: TensorTrain<Complex64>,
    r: usize,
    options: FourierOptions,
}

impl FTCore {
    /// Create a new FTCore for r bits.
    ///
    /// # Errors
    ///
    /// Returns an error when the variable count or dimension overflows (an
    /// /// overflow or invalid-configuration failure).
    ///
    pub fn new(r: usize, options: FourierOptions) -> Result<Self> {
        if r < 2 {
            anyhow::bail!("Number of sites must be at least 2, got {r}");
        }
        let forward_options = FourierOptions {
            sign: -1.0,
            ..options.clone()
        };
        let forward_mpo = quantics_fourier_mpo(r, &forward_options)?;
        Ok(Self {
            forward_mpo,
            r,
            options,
        })
    }

    /// Get the forward Fourier transform operator.
    ///
    /// # Errors
    /// Returns an error when converting the cached Fourier MPO to a linear
    /// operator fails (a shape mismatch or backend failure) or a required
    /// site-dimension allocation fails (an overflow failure).
    pub fn forward(&self) -> Result<QuanticsOperator> {
        let mut site_dims =
            try_vec_with_capacity::<usize>("Fourier forward site dimensions", self.r)?;
        site_dims.resize(self.r, 2);
        tensortrain_to_linear_operator(&self.forward_mpo, &site_dims)
    }

    /// Get the backward (inverse) Fourier transform operator.
    ///
    /// # Errors
    ///
    /// Returns an error when the inverse transform fails (a shape mismatch or
    /// /// backend failure).
    ///
    pub fn backward(&self) -> Result<QuanticsOperator> {
        let inverse_options = FourierOptions {
            sign: 1.0,
            normalize: self.options.normalize,
            ..self.options.clone()
        };
        let inverse_mpo = quantics_fourier_mpo(self.r, &inverse_options)?;
        let mut site_dims =
            try_vec_with_capacity::<usize>("Fourier inverse site dimensions", self.r)?;
        site_dims.resize(self.r, 2);
        tensortrain_to_linear_operator(&inverse_mpo, &site_dims)
    }

    /// Get the number of bits.
    pub fn r(&self) -> usize {
        self.r
    }
}

/// Create a Quantics Fourier Transform operator.
/// This implements the Chen & Lindsey construction of the DFT as a matrix product operator.
/// The resulting operator transforms a quantics tensor train representing a function
/// to its Fourier transform in quantics tensor train form.
/// # Index ordering
/// Before the Fourier transform, the leftmost index corresponds to the most significant
/// bit (largest length scale). After transformation, the leftmost index corresponds to
/// the least significant bit (smallest length scale) - this allows for a small bond
/// dimension construction.
/// # Arguments
/// * `r` - Number of bits
/// * `options` - Fourier transform options
/// # Returns
/// LinearOperator representing the QFT
/// # Errors
///
/// Returns an error when the operator construction fails (an overflow or
/// /// invalid-configuration failure).
///
/// # Examples
/// ```
/// use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
/// // Create a forward QFT operator for 4-bit quantics representation
/// let op = quantics_fourier_operator(4, FourierOptions::forward()).unwrap();
/// // The operator has one MPO tensor per bit
/// assert_eq!(op.mpo().node_count(), 4);
/// ```
pub fn quantics_fourier_operator(r: usize, options: FourierOptions) -> Result<QuanticsOperator> {
    if r < 2 {
        anyhow::bail!("Number of sites must be at least 2, got {r}");
    }

    let mpo = quantics_fourier_mpo(r, &options)?;
    let mut site_dims = try_vec_with_capacity::<usize>("Fourier operator site dimensions", r)?;
    site_dims.resize(r, 2);
    tensortrain_to_linear_operator(&mpo, &site_dims)
}

/// Create the QFT MPO as a TensorTrain using Chen & Lindsey construction.
fn quantics_fourier_mpo(r: usize, options: &FourierOptions) -> Result<TensorTrain<Complex64>> {
    if r < 2 {
        anyhow::bail!("Number of sites must be at least 2, got {r}");
    }

    let k = options.k;
    let sign = options.sign;
    let (count, _) = checked_fourier_core_size(k)?;

    // Get Chebyshev grid and barycentric weights
    let (grid, bary_weights) = chebyshev_grid(k)?;

    // Build core tensor A[alpha, tau, sigma, beta]
    // alpha, beta in 0..=K (K+1 values each)
    // tau, sigma in 0..1 (2 values each)
    let core_tensor = build_dft_core_tensor(&grid, &bary_weights, sign)?;

    // Construct tensor train
    let mut tensors = try_vec_with_capacity::<tensor4all_simplett::Tensor3<Complex64>>(
        "Fourier MPO site list",
        r,
    )?;

    // First tensor: sum over alpha (contract with ones vector)
    // Shape: (1, 4, K+1) where 4 = 2*2 for (tau, sigma)
    {
        let mut t = zero_tensor3(1, 4, count, "Fourier first tensor")?;
        for tau in 0..2 {
            for sigma in 0..2 {
                for beta in 0..count {
                    let mut sum = Complex64::zero();
                    for alpha in 0..count {
                        sum += core_tensor[[alpha, tau, sigma, beta]];
                    }
                    let s = tau * 2 + sigma;
                    t.set3(0, s, beta, sum);
                }
            }
        }
        tensors.push(t);
    }

    // Middle tensors: full core tensor
    // Shape: (K+1, 4, K+1)
    for _ in 1..r - 1 {
        let mut t = zero_tensor3(count, 4, count, "Fourier middle tensor")?;
        for alpha in 0..count {
            for tau in 0..2 {
                for sigma in 0..2 {
                    for beta in 0..count {
                        let s = tau * 2 + sigma;
                        t.set3(alpha, s, beta, core_tensor[[alpha, tau, sigma, beta]]);
                    }
                }
            }
        }
        tensors.push(t);
    }

    // Last tensor: select beta = 0
    // Shape: (K+1, 4, 1)
    if r > 1 {
        let mut t = zero_tensor3(count, 4, 1, "Fourier last tensor")?;
        for alpha in 0..count {
            for tau in 0..2 {
                for sigma in 0..2 {
                    let s = tau * 2 + sigma;
                    t.set3(alpha, s, 0, core_tensor[[alpha, tau, sigma, 0]]);
                }
            }
        }
        tensors.push(t);
    }

    let mut tt = TensorTrain::new(tensors)
        .map_err(|e| anyhow::anyhow!("Failed to create Fourier MPO: {}", e))?;

    // Compress the tensor train
    let compress_options = CompressionOptions {
        method: CompressionMethod::LU,
        tolerance: options.tolerance,
        max_bond_dim: options.maxbonddim,
        normalize_error: true,
    };
    tt.compress(&compress_options)
        .map_err(|err| anyhow::anyhow!("Fourier MPO compression failed: {err}"))?;

    // Normalize if requested
    if options.normalize {
        let norm_factor = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        for tensor in tt.site_tensors_mut() {
            let (left_dim, site_dim, right_dim) =
                (tensor.left_dim(), tensor.site_dim(), tensor.right_dim());
            for l in 0..left_dim {
                for s in 0..site_dim {
                    for r in 0..right_dim {
                        let val = *tensor.get3(l, s, r);
                        tensor.set3(l, s, r, val * norm_factor);
                    }
                }
            }
        }
    }

    Ok(tt)
}

/// Get Chebyshev grid points and barycentric weights.
/// Returns (grid, bary_weights) where:
/// - grid[j] = 0.5 * (1 - cos(π*j/K)) for j = 0, ..., K
/// - bary_weights are the barycentric interpolation weights
fn chebyshev_grid(k: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let count = checked_fourier_order(k)?;
    let mut grid = try_vec_with_capacity::<f64>("Fourier Chebyshev grid", count)?;
    let mut bary_weights = try_vec_with_capacity::<f64>("Fourier barycentric weights", count)?;

    // Compute Chebyshev grid points
    for j in 0..count {
        let x = 0.5 * (1.0 - (PI * j as f64 / k as f64).cos());
        grid.push(x);
    }

    // Compute barycentric weights
    for j in 0..count {
        let mut weight = 1.0;
        for m in 0..count {
            if j != m {
                weight /= grid[j] - grid[m];
            }
        }
        bary_weights.push(weight);
    }

    Ok((grid, bary_weights))
}

/// Evaluate Lagrange polynomial P_alpha(x).
fn lagrange_polynomial(grid: &[f64], bary_weights: &[f64], alpha: usize, x: f64) -> f64 {
    // Check if x is very close to grid[alpha]
    if (x - grid[alpha]).abs() < 1e-14 {
        return 1.0;
    }

    // Compute product term
    let mut prod = 1.0;
    for &g in grid {
        prod *= x - g;
    }

    prod * bary_weights[alpha] / (x - grid[alpha])
}

/// Build the DFT core tensor A[alpha, tau, sigma, beta].
/// A[alpha, tau, sigma, beta] = P_alpha(x) * exp(2πi * sign * x * tau)
/// where x = (sigma + grid[beta]) / 2
/// Returns tensor of shape (k+1, 2, 2, k+1)
fn build_dft_core_tensor(
    grid: &[f64],
    bary_weights: &[f64],
    sign: f64,
) -> Result<Tensor4<Complex64>> {
    if grid.is_empty() || grid.len() != bary_weights.len() {
        anyhow::bail!("Fourier grid and barycentric weights have incompatible lengths");
    }
    let count = grid.len();
    let k = count - 1;
    let total_size = checked_tensor4_size([count, 2, 2, count], "Fourier core tensor")?;
    let mut data = try_vec_with_capacity::<Complex64>("Fourier core tensor", total_size)?;
    data.resize(total_size, Complex64::zero());
    let mut tensor = tensor4_from_data(data, count, 2, 2, count)
        .map_err(|err| anyhow::anyhow!("Failed to allocate Fourier core tensor: {err}"))?;

    for alpha in 0..=k {
        for tau in 0..2 {
            for sigma in 0..2 {
                for beta in 0..=k {
                    let x = (sigma as f64 + grid[beta]) / 2.0;
                    let p_alpha = lagrange_polynomial(grid, bary_weights, alpha, x);
                    let phase = 2.0 * PI * sign * x * tau as f64;
                    let exp_phase = Complex64::new(phase.cos(), phase.sin());
                    tensor[[alpha, tau, sigma, beta]] = Complex64::new(p_alpha, 0.0) * exp_phase;
                }
            }
        }
    }

    Ok(tensor)
}

#[cfg(test)]
mod tests;
