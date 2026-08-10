//! Lagrange basis and rank-3 core helpers.

use std::f64::consts::PI;

use tensor4all_simplett::{Tensor3, Tensor3Ops};

use crate::error::{invalid_argument, Result};

/// Barycentric Lagrange basis on a one-dimensional interpolation grid.
/// The basis stores interpolation nodes in `[0, 1]` and scaled barycentric
/// weights. It is used by the interpolative QTT constructors to transfer
/// local polynomial information across binary refinement levels.
/// # Related Types
/// `LagrangePolynomials` provides the local basis. The generated core tensors
/// are stored as `tensor4all-simplett` rank-3 tensors and assembled into a
/// `TensorTrain`.
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::get_chebyshev_grid;
/// let basis = get_chebyshev_grid(4).unwrap();
/// assert_eq!(basis.len(), 5);
/// assert!((basis.evaluate(0, basis.grid()[0]).unwrap() - 1.0).abs() < 1e-12);
/// assert!(basis.evaluate(1, basis.grid()[0]).unwrap().abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct LagrangePolynomials {
    grid: Vec<f64>,
    barycentric_weights: Vec<f64>,
}

impl LagrangePolynomials {
    /// Create a barycentric Lagrange basis from nodes in `[0, 1]`.
    ///
    /// `grid` must contain at least two distinct finite points. The returned
    /// basis can be evaluated with [`LagrangePolynomials::evaluate`].
    ///
    /// # Errors
    /// Returns an error when the basis data is invalid (an invalid-configuration
    /// failure).
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::LagrangePolynomials;
    ///
    /// let basis = LagrangePolynomials::new(vec![0.0, 1.0]).unwrap();
    /// assert!((basis.evaluate(0, 0.25).unwrap() - 0.75).abs() < 1e-12);
    /// assert!((basis.evaluate(1, 0.25).unwrap() - 0.25).abs() < 1e-12);
    /// ```
    pub fn new(grid: Vec<f64>) -> Result<Self> {
        if grid.len() < 2 {
            return Err(invalid_argument(
                "Lagrange grid must contain at least two points",
            ));
        }
        if grid.iter().any(|x| !x.is_finite()) {
            return Err(invalid_argument("Lagrange grid values must be finite"));
        }

        let barycentric_weights = scaled_barycentric_weights(&grid)?;

        Ok(Self {
            grid,
            barycentric_weights,
        })
    }

    fn from_grid_and_weights(grid: Vec<f64>, barycentric_weights: Vec<f64>) -> Result<Self> {
        if grid.len() < 2 {
            return Err(invalid_argument(
                "Lagrange grid must contain at least two points",
            ));
        }
        if grid.len() != barycentric_weights.len() {
            return Err(invalid_argument(format!(
                "Lagrange grid/weight length mismatch: grid has {}, weights have {}",
                grid.len(),
                barycentric_weights.len()
            )));
        }
        if grid.iter().any(|x| !x.is_finite()) {
            return Err(invalid_argument("Lagrange grid values must be finite"));
        }
        if barycentric_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight == 0.0)
        {
            return Err(invalid_argument(
                "barycentric weights must be finite and non-zero",
            ));
        }
        for j in 0..grid.len() {
            for m in 0..grid.len() {
                if j == m {
                    continue;
                }
                let diff = grid[j] - grid[m];
                if diff.abs() < 1.0e-15 {
                    return Err(invalid_argument("Lagrange grid values must be distinct"));
                }
            }
        }

        Ok(Self {
            grid,
            barycentric_weights,
        })
    }

    /// Number of basis functions.
    ///
    /// This equals the number of interpolation nodes, and is one more than the
    /// polynomial degree used by [`get_chebyshev_grid`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::get_chebyshev_grid;
    ///
    /// let basis = get_chebyshev_grid(3).unwrap();
    /// assert_eq!(basis.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Whether the basis has no grid points.
    ///
    /// A valid `LagrangePolynomials` value is never empty; this method is
    /// provided for consistency with other collection-like APIs.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::get_chebyshev_grid;
    ///
    /// let basis = get_chebyshev_grid(2).unwrap();
    /// assert!(!basis.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }

    /// Borrow the interpolation grid nodes.
    ///
    /// Nodes are ordered from `0.0` to `1.0` for grids created by
    /// [`get_chebyshev_grid`].
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::get_chebyshev_grid;
    ///
    /// let basis = get_chebyshev_grid(2).unwrap();
    /// assert!((basis.grid()[0] - 0.0).abs() < 1e-12);
    /// assert!((basis.grid()[2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    /// Borrow the scaled barycentric weights for the interpolation grid.
    ///
    /// Only relative weights matter for barycentric interpolation; common
    /// factors are intentionally omitted to avoid overflow at high degree.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::get_chebyshev_grid;
    ///
    /// let basis = get_chebyshev_grid(3).unwrap();
    /// assert_eq!(basis.barycentric_weights().len(), basis.len());
    /// ```
    pub fn barycentric_weights(&self) -> &[f64] {
        &self.barycentric_weights
    }

    /// Evaluate one Lagrange basis polynomial.
    ///
    /// `alpha` selects the basis function and must be smaller than
    /// [`LagrangePolynomials::len`]. `x` is usually in `[0, 1]`, but the
    /// formula is valid for any finite real value.
    ///
    /// # Errors
    ///
    /// Returns an error if `alpha` is out of bounds or `x` is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor4all_interpolativeqtt::get_chebyshev_grid;
    ///
    /// let basis = get_chebyshev_grid(4).unwrap();
    /// let x = basis.grid()[2];
    /// assert!((basis.evaluate(2, x).unwrap() - 1.0).abs() < 1e-12);
    /// assert!(basis.evaluate(1, x).unwrap().abs() < 1e-12);
    /// ```
    pub fn evaluate(&self, alpha: usize, x: f64) -> Result<f64> {
        if alpha >= self.len() {
            return Err(invalid_argument(format!(
                "basis index {alpha} is out of bounds for {} basis functions",
                self.len()
            )));
        }
        if !x.is_finite() {
            return Err(invalid_argument("basis evaluation point must be finite"));
        }

        if (x - self.grid[alpha]).abs() < 1.0e-14 {
            return Ok(1.0);
        }
        if self.grid.iter().any(|node| (x - node).abs() < 1.0e-14) {
            return Ok(0.0);
        }

        let mut denominator = 0.0;
        for (node, weight) in self.grid.iter().zip(&self.barycentric_weights) {
            denominator += weight / (x - node);
        }
        if denominator == 0.0 || !denominator.is_finite() {
            return Err(invalid_argument(
                "barycentric denominator must be finite and non-zero",
            ));
        }

        Ok((self.barycentric_weights[alpha] / (x - self.grid[alpha])) / denominator)
    }
}

/// Build a Chebyshev-Lobatto Lagrange basis on `[0, 1]`.
/// `degree` is the local polynomial degree and must be at least one. The
/// returned basis has `degree + 1` nodes.
/// # Errors
/// Returns an error when `degree` is zero (an invalid-configuration failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::get_chebyshev_grid;
/// let basis = get_chebyshev_grid(4).unwrap();
/// assert_eq!(basis.len(), 5);
/// assert!((basis.grid()[0] - 0.0).abs() < 1e-12);
/// assert!((basis.grid()[4] - 1.0).abs() < 1e-12);
/// ```
pub fn get_chebyshev_grid(degree: usize) -> Result<LagrangePolynomials> {
    if degree == 0 {
        return Err(invalid_argument("polynomial degree must be at least 1"));
    }

    let grid = (0..=degree)
        .map(|j| 0.5 * (1.0 - ((j as f64) * PI / (degree as f64)).cos()))
        .collect();
    let weights = chebyshev_lobatto_barycentric_weights(degree);
    LagrangePolynomials::from_grid_and_weights(grid, weights)
}

fn scaled_barycentric_weights(grid: &[f64]) -> Result<Vec<f64>> {
    let mut signs = Vec::with_capacity(grid.len());
    let mut log_abs_weights = Vec::with_capacity(grid.len());

    for j in 0..grid.len() {
        let mut sign = 1.0;
        let mut log_abs_weight = 0.0;
        for m in 0..grid.len() {
            if j == m {
                continue;
            }
            let diff = grid[j] - grid[m];
            if diff.abs() < 1.0e-15 {
                return Err(invalid_argument("Lagrange grid values must be distinct"));
            }
            if diff.is_sign_negative() {
                sign = -sign;
            }
            log_abs_weight -= diff.abs().ln();
        }
        signs.push(sign);
        log_abs_weights.push(log_abs_weight);
    }

    let max_log_abs_weight = log_abs_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = signs
        .into_iter()
        .zip(log_abs_weights)
        .map(|(sign, log_abs_weight)| sign * (log_abs_weight - max_log_abs_weight).exp())
        .collect();
    if weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight == 0.0)
    {
        return Err(invalid_argument(
            "failed to compute finite non-zero barycentric weights",
        ));
    }
    Ok(weights)
}

fn chebyshev_lobatto_barycentric_weights(degree: usize) -> Vec<f64> {
    (0..=degree)
        .map(|j| {
            let magnitude = if j == 0 || j == degree { 0.5 } else { 1.0 };
            if j % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

/// Build the dense local interpolation core for a Lagrange basis.
/// The returned tensor has shape `(degree + 1, 2, degree + 1)`. The middle
/// index is the binary refinement digit `sigma`, and the value is
/// `P_alpha((sigma + grid_beta) / 2)`.
/// # Errors
/// Returns an error when a basis evaluation fails (a non-convergence or
/// backend failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{get_chebyshev_grid, interpolation_tensor};
/// use tensor4all_simplett::Tensor3Ops;
/// let basis = get_chebyshev_grid(3).unwrap();
/// let core = interpolation_tensor(&basis).unwrap();
/// assert_eq!(core.left_dim(), 4);
/// assert_eq!(core.site_dim(), 2);
/// assert_eq!(core.right_dim(), 4);
/// ```
pub fn interpolation_tensor(basis: &LagrangePolynomials) -> Result<Tensor3<f64>> {
    let n = basis.len();
    let mut data = Vec::with_capacity(n * 2 * n);
    for alpha in 0..n {
        for sigma in 0..2 {
            for beta in 0..n {
                let x = (sigma as f64 + basis.grid[beta]) / 2.0;
                data.push(basis.evaluate(alpha, x)?);
            }
        }
    }

    Ok(Tensor3::from_fn([n, 2, n], |[alpha, sigma, beta]| {
        data[(alpha * 2 + sigma) * n + beta]
    }))
}

/// Build the fused direct product of rank-3 core tensors.
/// Given cores with shapes `(a_i, s_i, b_i)`, the result has shape
/// `(prod_i a_i, prod_i s_i, prod_i b_i)`. Index `0` is the fastest-moving
/// component in each fused axis, matching the fused quantics layout used by
/// `quanticsgrids`.
/// This is a dense local-core construction. For `D` fused dimensions and a
/// polynomial degree `p`, single-scale interpolation can produce bonds of
/// size `(p + 1)^D` and a site dimension of `2^D`, so callers should keep
/// `D` and `p` modest or use the sparse constructor.
/// # Errors
/// Returns an error when `cores` is empty or fused dimensions overflow
/// `usize`.
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::direct_product_core_tensors;
/// use tensor4all_simplett::{Tensor3, Tensor3Ops};
/// let a = Tensor3::from_fn([2, 2, 1], |[l, s, _]| (l + s) as f64);
/// let b = Tensor3::from_fn([3, 2, 1], |[l, s, _]| (10 + l + s) as f64);
/// let fused = direct_product_core_tensors(&[a, b]).unwrap();
/// assert_eq!(fused.left_dim(), 6);
/// assert_eq!(fused.site_dim(), 4);
/// assert_eq!(fused.right_dim(), 1);
/// assert!((*fused.get3(1, 0, 0) - 10.0).abs() < 1e-12);
/// ```
pub fn direct_product_core_tensors(cores: &[Tensor3<f64>]) -> Result<Tensor3<f64>> {
    let first = cores
        .first()
        .ok_or_else(|| invalid_argument("at least one core tensor is required"))?;
    let mut result = first.clone();
    for core in &cores[1..] {
        result = direct_product_two(&result, core)?;
    }
    Ok(result)
}

fn checked_fused_dim(left: usize, right: usize, axis: &str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid_argument(format!("fused {axis} dimension overflows usize")))
}

fn direct_product_two(a: &Tensor3<f64>, b: &Tensor3<f64>) -> Result<Tensor3<f64>> {
    let left_a = a.left_dim();
    let site_a = a.site_dim();
    let right_a = a.right_dim();
    let left_b = b.left_dim();
    let site_b = b.site_dim();
    let right_b = b.right_dim();
    let left_dim = checked_fused_dim(left_a, left_b, "left")?;
    let site_dim = checked_fused_dim(site_a, site_b, "site")?;
    let right_dim = checked_fused_dim(right_a, right_b, "right")?;

    Ok(Tensor3::from_fn(
        [left_dim, site_dim, right_dim],
        |[left, site, right]| {
            let left_0 = left % left_a;
            let left_1 = left / left_a;
            let site_0 = site % site_a;
            let site_1 = site / site_a;
            let right_0 = right % right_a;
            let right_1 = right / right_a;
            *a.get3(left_0, site_0, right_0) * *b.get3(left_1, site_1, right_1)
        },
    ))
}

pub(crate) fn angular_local_lagrange(
    basis: &LagrangePolynomials,
    window_radius: usize,
) -> Result<Tensor3<f64>> {
    let degree = basis.len() - 1;
    if degree < 2 * window_radius {
        return Err(invalid_argument(format!(
            "need degree >= 2 * window_radius, got degree {degree} and window_radius {window_radius}"
        )));
    }

    Ok(Tensor3::from_fn(
        [degree + 1, 2, degree + 1],
        |[alpha, sigma, beta]| {
            let x = (sigma as f64 + basis.grid[beta]) / 2.0;
            let theta = (1.0 - 2.0 * x).clamp(-1.0, 1.0).acos();
            let nearest = (theta * degree as f64 / PI)
                .round()
                .clamp(0.0, degree as f64) as usize;
            let lo = nearest
                .saturating_sub(window_radius)
                .min(degree - 2 * window_radius);
            let hi = lo + 2 * window_radius;
            if alpha < lo || alpha > hi {
                return 0.0;
            }

            let theta_alpha = PI * alpha as f64 / degree as f64;
            let mut value = 1.0;
            for gamma in lo..=hi {
                if gamma == alpha {
                    continue;
                }
                let theta_gamma = PI * gamma as f64 / degree as f64;
                value *= (theta - theta_gamma) / (theta_alpha - theta_gamma);
            }
            value
        },
    ))
}
