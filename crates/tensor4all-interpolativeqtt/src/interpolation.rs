//! Interpolation constructors and analysis routines.

use tensor4all_simplett::{
    AbstractTensorTrain, CompressionMethod, CompressionOptions, Tensor3, Tensor3Ops, TensorTrain,
};
use tensor4all_tensorbackend::{mat_mul, Matrix};

use crate::basis::{
    angular_local_lagrange, direct_product_core_tensors, get_chebyshev_grid, interpolation_tensor,
    LagrangePolynomials,
};
use crate::error::{invalid_argument, Result};
use crate::interval::{Interval, NInterval};
use crate::options::InterpolativeQttOptions;

/// Construct a one-dimensional single-scale interpolative QTT.
/// `f` is sampled through a Chebyshev-Lobatto local basis on `[a, b)`.
/// `num_bits` is the number of quantics sites, and `polynomial_degree`
/// controls the local interpolation order. The result is a binary
/// `TensorTrain<f64>` whose site indices are zero-based quantics digits.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_single_scale, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_single_scale(
///     |x| x * x,
///     0.0,
///     1.0,
///     4,
///     8,
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// let value = tt.evaluate(&[0, 0, 0, 0]).unwrap();
/// assert!(value.abs() < 1e-10);
/// ```
pub fn interpolate_single_scale<F>(
    f: F,
    a: f64,
    b: f64,
    num_bits: usize,
    polynomial_degree: usize,
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(f64) -> f64,
{
    interpolate_single_scale_nd(
        |coords| f(coords[0]),
        &[a],
        &[b],
        num_bits,
        polynomial_degree,
        options,
    )
}

/// Construct a fused multidimensional single-scale interpolative QTT.
/// `lower` and `upper` define the box. Each quantics site fuses all dimensions
/// at the same bit level, so a `D`-dimensional interpolant has site dimension
/// `2^D` and `num_bits` sites.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_single_scale_nd, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_single_scale_nd(
///     |x| x[0] + x[1],
///     &[0.0, 0.0],
///     &[1.0, 1.0],
///     3,
///     5,
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.len(), 3);
/// assert_eq!(tt.site_dims(), vec![4, 4, 4]);
/// ```
pub fn interpolate_single_scale_nd<F>(
    f: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_common(lower, upper, num_bits, polynomial_degree, options)?;
    let ndims = lower.len();
    let basis = get_chebyshev_grid(polynomial_degree)?;
    let site_dim = site_dim(ndims)?;
    let basis_dim = pow_usize(polynomial_degree + 1, ndims)?;
    let mut cores = Vec::with_capacity(num_bits);

    cores.push(left_core_nd(&f, lower, upper, &basis, site_dim, basis_dim));

    let center_1d = interpolation_tensor(&basis)?;
    let center = direct_product_core_tensors(&vec![center_1d; ndims])?;
    for _ in 0..num_bits.saturating_sub(2) {
        cores.push(center.clone());
    }

    let right_1d = right_core_1d(&basis)?;
    cores.push(direct_product_core_tensors(&vec![right_1d; ndims])?);

    compress_train(cores, options)
}

/// Construct a one-dimensional multiscale interpolative QTT.
/// `cusp_locations` identifies points that should remain on a refinement path
/// instead of being locally interpolated too early. This is useful for known
/// nonsmooth points such as absolute-value cusps or removable singularities.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_multi_scale, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_multi_scale(
///     |x| x.abs(),
///     -1.0,
///     1.0,
///     4,
///     8,
///     &[0.0],
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.len(), 4);
/// assert!((tt.evaluate(&[0, 0, 0, 0]).unwrap() - 1.0).abs() < 1e-10);
/// ```
pub fn interpolate_multi_scale<F>(
    f: F,
    a: f64,
    b: f64,
    num_bits: usize,
    polynomial_degree: usize,
    cusp_locations: &[f64],
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(f64) -> f64,
{
    let cusps: Vec<_> = cusp_locations.iter().map(|&x| vec![x]).collect();
    interpolate_multi_scale_nd(
        |coords| f(coords[0]),
        &[a],
        &[b],
        num_bits,
        polynomial_degree,
        &cusps,
        options,
    )
}

/// Construct a fused multidimensional multiscale interpolative QTT.
/// `cusp_locations` contains points in the original coordinate box. Any
/// interval containing one of these points is refined until the final level.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_multi_scale_nd, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_multi_scale_nd(
///     |x| x[0] * x[1],
///     &[0.0, 0.0],
///     &[1.0, 1.0],
///     3,
///     4,
///     &[vec![0.0, 0.0]],
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.site_dims(), vec![4, 4, 4]);
/// ```
pub fn interpolate_multi_scale_nd<F>(
    f: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    cusp_locations: &[Vec<f64>],
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_common(lower, upper, num_bits, polynomial_degree, options)?;
    validate_points(cusp_locations, lower.len(), "cusp location")?;

    let basis = get_chebyshev_grid(polynomial_degree)?;
    let domain = NInterval::new(lower, upper)?;
    let cusps = cusp_locations.to_vec();
    build_refined_qtt(
        &f,
        &domain,
        num_bits,
        &basis,
        options,
        |interval, _level| !cusps.iter().any(|cusp| interval.contains(cusp)),
    )
}

/// Construct a one-dimensional adaptive interpolative QTT.
/// The interval is recursively refined where the local interpolation error
/// exceeds `adaptive_tolerance`. Known singularities can be supplied to seed a
/// refinement path and avoid evaluating the function at those points.
/// # Errors
/// Returns an error when the inputs are invalid (an invalid-configuration failure) or the tensor train compression fails (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_adaptive, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_adaptive(
///     |x| x.sin(),
///     0.0,
///     1.0,
///     4,
///     6,
///     1e-8,
///     &[],
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.len(), 4);
/// ```
#[allow(clippy::too_many_arguments)]
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
///
pub fn interpolate_adaptive<F>(
    f: F,
    a: f64,
    b: f64,
    num_bits: usize,
    polynomial_degree: usize,
    adaptive_tolerance: f64,
    singularities: &[f64],
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(f64) -> f64,
{
    let singularities_nd: Vec<_> = singularities.iter().map(|&x| vec![x]).collect();
    interpolate_adaptive_nd(
        |coords| f(coords[0]),
        &[a],
        &[b],
        num_bits,
        polynomial_degree,
        adaptive_tolerance,
        &singularities_nd,
        options,
    )
}

/// Construct a fused multidimensional adaptive interpolative QTT.
/// The adaptive pass marks boxes whose local interpolation error exceeds
/// `adaptive_tolerance`, then builds a multiscale QTT using those marked
/// boxes.
/// # Errors
/// Returns an error if the inputs are invalid (an invalid-configuration
/// failure), the singularity dimensions do not match (a shape mismatch), or
/// the compression fails (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_adaptive_nd, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_adaptive_nd(
///     |x| x[0] + x[1],
///     &[0.0, 0.0],
///     &[1.0, 1.0],
///     3,
///     4,
///     1e-8,
///     &[],
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.len(), 3);
/// ```
#[allow(clippy::too_many_arguments)]
/// # Errors
/// Returns an error if the inputs are invalid (an invalid-configuration
/// failure), the singularity dimensions do not match (a shape mismatch), or
/// the compression fails (a non-convergence failure).
///
pub fn interpolate_adaptive_nd<F>(
    f: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    adaptive_tolerance: f64,
    singularities: &[Vec<f64>],
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_common(lower, upper, num_bits, polynomial_degree, options)?;
    if !adaptive_tolerance.is_finite() || adaptive_tolerance < 0.0 {
        return Err(invalid_argument(
            "adaptive tolerance must be finite and non-negative",
        ));
    }
    validate_points(singularities, lower.len(), "singularity")?;

    let basis = get_chebyshev_grid(polynomial_degree)?;
    let domain = NInterval::new(lower, upper)?;
    let mut dangerous_at_level = vec![Vec::<NInterval>::new(); num_bits];

    for singularity in singularities {
        if domain.contains(singularity) {
            add_singularity_path(&mut dangerous_at_level, &domain, singularity)?;
        }
    }

    for subinterval in domain.split()? {
        detect_dangerous_intervals(
            &mut dangerous_at_level,
            &f,
            &subinterval,
            0,
            num_bits - 2,
            &basis,
            adaptive_tolerance,
        )?;
    }

    if dangerous_at_level.iter().all(Vec::is_empty) {
        return interpolate_single_scale_nd(f, lower, upper, num_bits, polynomial_degree, options);
    }

    build_adaptive_qtt(&f, &domain, num_bits, &basis, &dangerous_at_level, options)
}

/// Construct a one-dimensional sparse single-scale interpolative QTT.
/// The sparse variant replaces the dense interpolation core by an angular
/// local Lagrange core with radius `window_radius`.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_single_scale_sparse, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_single_scale_sparse(
///     |x| x.cos(),
///     0.0,
///     1.0,
///     4,
///     8,
///     2,
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.len(), 4);
/// ```
pub fn interpolate_single_scale_sparse<F>(
    f: F,
    a: f64,
    b: f64,
    num_bits: usize,
    polynomial_degree: usize,
    window_radius: usize,
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(f64) -> f64,
{
    interpolate_single_scale_sparse_nd(
        |coords| f(coords[0]),
        &[a],
        &[b],
        num_bits,
        polynomial_degree,
        window_radius,
        options,
    )
}

/// Construct a fused multidimensional sparse single-scale interpolative QTT.
/// This is the multidimensional version of [`interpolate_single_scale_sparse`]
/// with fused local site dimensions.
/// # Errors
/// Returns an error when the grid or ranks are invalid (an invalid-configuration
/// failure) or interpolation fails to converge (a non-convergence failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     interpolate_single_scale_sparse_nd, AbstractTensorTrain, InterpolativeQttOptions,
/// };
/// let tt = interpolate_single_scale_sparse_nd(
///     |x| x[0] + x[1],
///     &[0.0, 0.0],
///     &[1.0, 1.0],
///     3,
///     6,
///     2,
///     &InterpolativeQttOptions::default(),
/// ).unwrap();
/// assert_eq!(tt.site_dims(), vec![4, 4, 4]);
/// ```
pub fn interpolate_single_scale_sparse_nd<F>(
    f: F,
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    window_radius: usize,
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    validate_common(lower, upper, num_bits, polynomial_degree, options)?;
    let ndims = lower.len();
    let basis = get_chebyshev_grid(polynomial_degree)?;
    let site_dim = site_dim(ndims)?;
    let basis_dim = pow_usize(polynomial_degree + 1, ndims)?;
    let mut cores = Vec::with_capacity(num_bits);

    cores.push(left_core_nd(&f, lower, upper, &basis, site_dim, basis_dim));

    let center_1d = angular_local_lagrange(&basis, window_radius)?;
    let center = direct_product_core_tensors(&vec![center_1d; ndims])?;
    for _ in 0..num_bits.saturating_sub(2) {
        cores.push(center.clone());
    }

    let right_1d = right_core_1d(&basis)?;
    cores.push(direct_product_core_tensors(&vec![right_1d; ndims])?);

    compress_train(cores, options)
}

/// Recover multiresolution Chebyshev-Lobatto values from a binary QTT.
/// The returned vector has one matrix per coarse level. Matrix `k - 1` has
/// `2^k` rows and `basis.len()` columns, where each row corresponds to one
/// subinterval and each column to one local Chebyshev-Lobatto node.
/// # Errors
/// Returns an error when the QTT is not invertible (a singular failure) or the
/// inversion fails (a backend failure).
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     get_chebyshev_grid, interpolate_single_scale, invert_qtt, InterpolativeQttOptions,
/// };
/// let basis = get_chebyshev_grid(6).unwrap();
/// let tt = interpolate_single_scale(
///     |x| (-x * x).exp(),
///     0.0,
///     1.0,
///     5,
///     6,
///     &InterpolativeQttOptions::default().with_tolerance(0.0),
/// ).unwrap();
/// let values = invert_qtt(&tt, &basis, 1).unwrap();
/// assert_eq!(values.len(), 4);
/// assert_eq!(values[3].ncols(), basis.len());
/// ```
pub fn invert_qtt(
    tt: &TensorTrain<f64>,
    basis: &LagrangePolynomials,
    q: usize,
) -> Result<Vec<Matrix<f64>>> {
    let num_sites = tt.len();
    if !(1..num_sites).contains(&q) {
        return Err(invalid_argument(format!(
            "q must satisfy 1 <= q < K={num_sites}, got {q}"
        )));
    }
    if q != 1 {
        return Err(invalid_argument("only q = 1 is currently supported"));
    }
    if tt.site_dims().iter().any(|&dim| dim != 2) {
        return Err(invalid_argument(
            "invert_qtt currently requires binary site dimensions",
        ));
    }

    let k_out = num_sites - q;
    let fine = invert_stage1(tt, basis)?;
    let (r_left, r_right) = build_restriction(basis)?;

    let mut results = vec![Matrix::zeros(0, 0); k_out];
    results[k_out - 1] = fine;
    for level in (0..k_out - 1).rev() {
        results[level] = apply_stage2(&results[level + 1], &r_left, &r_right)?;
    }
    Ok(results)
}

/// Estimate the local interpolation error on a one-dimensional interval.
/// The function is sampled at the basis nodes, then the interpolant is checked
/// on a denser Chebyshev-Lobatto grid. The return value is a maximum absolute
/// error estimate.
/// # Errors
/// Returns an error when the reference and interpolated functions have
/// incompatible grids (a shape mismatch) or the estimation fails.
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     estimate_interpolation_error, get_chebyshev_grid,
/// };
/// let basis = get_chebyshev_grid(6).unwrap();
/// let err = estimate_interpolation_error(|x| x.sin(), 0.0, 1.0, &basis).unwrap();
/// assert!(err >= 0.0);
/// ```
pub fn estimate_interpolation_error<F>(
    f: F,
    a: f64,
    b: f64,
    basis: &LagrangePolynomials,
) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let interval = Interval::new(a, b)?;
    estimate_interpolation_error_interval(&f, &interval, basis)
}

fn estimate_interpolation_error_interval<F>(
    f: &F,
    interval: &Interval,
    basis: &LagrangePolynomials,
) -> Result<f64>
where
    F: Fn(f64) -> f64,
{
    let values = eval_interval_1d(f, interval, basis);
    let test_points = dense_test_points(basis.len());
    let mut max_error = 0.0_f64;

    for t in test_points {
        let x = interval.start() + interval.length() * t;
        let mut interp_value = 0.0;
        for (alpha, value) in values.iter().enumerate() {
            interp_value += value * basis.evaluate(alpha, t)?;
        }
        max_error = max_error.max((interp_value - f(x)).abs());
    }
    Ok(max_error)
}

/// Estimate the local interpolation error on a multidimensional box.
/// The function is sampled at tensor-product basis nodes, then checked on a
/// denser tensor-product Chebyshev-Lobatto grid.
/// # Errors
/// Returns an error if basis evaluation fails or dimensions overflow.
/// # Examples
/// ```
/// use tensor4all_interpolativeqtt::{
///     estimate_interpolation_error_nd, get_chebyshev_grid,
/// };
/// let basis = get_chebyshev_grid(4).unwrap();
/// let err = estimate_interpolation_error_nd(
///     |x| x[0] + x[1],
///     &[0.0, 0.0],
///     &[1.0, 1.0],
///     &basis,
/// ).unwrap();
/// assert!(err >= 0.0);
/// assert!(err < 1e-10);
/// ```
pub fn estimate_interpolation_error_nd<F>(
    f: F,
    lower: &[f64],
    upper: &[f64],
    basis: &LagrangePolynomials,
) -> Result<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let interval = NInterval::new(lower, upper)?;
    estimate_interpolation_error_ninterval(&f, &interval, basis)
}

fn estimate_interpolation_error_ninterval<F>(
    f: &F,
    interval: &NInterval,
    basis: &LagrangePolynomials,
) -> Result<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let ndims = interval.ndims();
    let basis_len = basis.len();
    let basis_dim = pow_usize(basis_len, ndims)?;
    let values = eval_interval_nd(f, interval, basis, basis_dim);
    let test_points = dense_test_points(basis_len);
    let test_dims = vec![test_points.len(); ndims];
    let lengths = interval.lengths();
    let mut max_error = 0.0_f64;

    for test_index in MultiIndexIter::new(&test_dims) {
        let ts: Vec<_> = test_index.iter().map(|&i| test_points[i]).collect();
        let x: Vec<_> = (0..ndims)
            .map(|d| interval.start()[d] + lengths[d] * ts[d])
            .collect();

        let mut interp_value = 0.0;
        for alpha_index in MultiIndexIter::new(&vec![basis_len; ndims]) {
            let alpha_flat = flatten_index(&alpha_index, basis_len);
            let mut basis_value = 1.0;
            for d in 0..ndims {
                basis_value *= basis.evaluate(alpha_index[d], ts[d])?;
            }
            interp_value += values[alpha_flat] * basis_value;
        }

        max_error = max_error.max((interp_value - f(&x)).abs());
    }

    Ok(max_error)
}

fn validate_common(
    lower: &[f64],
    upper: &[f64],
    num_bits: usize,
    polynomial_degree: usize,
    options: &InterpolativeQttOptions,
) -> Result<()> {
    if lower.is_empty() {
        return Err(invalid_argument("domain must have at least one dimension"));
    }
    if lower.len() != upper.len() {
        return Err(invalid_argument(format!(
            "domain shape mismatch: lower has {}, upper has {}",
            lower.len(),
            upper.len()
        )));
    }
    if lower.iter().chain(upper.iter()).any(|x| !x.is_finite()) {
        return Err(invalid_argument("domain bounds must be finite"));
    }
    if num_bits < 2 {
        return Err(invalid_argument("num_bits must be at least 2"));
    }
    if polynomial_degree == 0 {
        return Err(invalid_argument("polynomial_degree must be at least 1"));
    }
    validate_options(options)
}

fn validate_options(options: &InterpolativeQttOptions) -> Result<()> {
    if !options.tolerance.is_finite() || options.tolerance < 0.0 {
        return Err(invalid_argument(
            "compression tolerance must be finite and non-negative",
        ));
    }
    if options.max_bond_dim == 0 {
        return Err(invalid_argument("max_bond_dim must be at least 1"));
    }
    Ok(())
}

fn validate_points(points: &[Vec<f64>], ndims: usize, name: &str) -> Result<()> {
    for point in points {
        if point.len() != ndims {
            return Err(invalid_argument(format!(
                "{name} shape mismatch: expected {ndims}, got {}",
                point.len()
            )));
        }
        if point.iter().any(|x| !x.is_finite()) {
            return Err(invalid_argument(format!(
                "{name} coordinates must be finite"
            )));
        }
    }
    Ok(())
}

fn site_dim(ndims: usize) -> Result<usize> {
    1usize
        .checked_shl(ndims as u32)
        .ok_or_else(|| invalid_argument("fused site dimension overflows usize"))
}

fn pow_usize(base: usize, exp: usize) -> Result<usize> {
    let mut result = 1usize;
    for _ in 0..exp {
        result = result
            .checked_mul(base)
            .ok_or_else(|| invalid_argument("dimension product overflows usize"))?;
    }
    Ok(result)
}

fn flatten_index(index: &[usize], base: usize) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for &value in index {
        flat += value * stride;
        stride *= base;
    }
    flat
}

fn decode_index(mut flat: usize, base: usize, ndims: usize) -> Vec<usize> {
    let mut index = Vec::with_capacity(ndims);
    for _ in 0..ndims {
        index.push(flat % base);
        flat /= base;
    }
    index
}

fn left_core_nd<F>(
    f: &F,
    lower: &[f64],
    upper: &[f64],
    basis: &LagrangePolynomials,
    site_dim: usize,
    basis_dim: usize,
) -> Tensor3<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let ndims = lower.len();
    let basis_len = basis.len();
    Tensor3::from_fn([1, site_dim, basis_dim], |[_, site, beta_flat]| {
        let sigmas = decode_index(site, 2, ndims);
        let betas = decode_index(beta_flat, basis_len, ndims);
        let coords: Vec<_> = (0..ndims)
            .map(|d| {
                let local = (sigmas[d] as f64 + basis.grid()[betas[d]]) / 2.0;
                lower[d] + (upper[d] - lower[d]) * local
            })
            .collect();
        f(&coords)
    })
}

fn right_core_1d(basis: &LagrangePolynomials) -> Result<Tensor3<f64>> {
    let basis_len = basis.len();
    let mut data = Vec::with_capacity(basis_len * 2);
    for alpha in 0..basis_len {
        for sigma in 0..2 {
            data.push(basis.evaluate(alpha, sigma as f64 / 2.0)?);
        }
    }
    Ok(Tensor3::from_fn([basis_len, 2, 1], |[alpha, sigma, _]| {
        data[alpha * 2 + sigma]
    }))
}

fn compress_train(
    cores: Vec<Tensor3<f64>>,
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>> {
    let tt = TensorTrain::new(cores)?;
    if options.tolerance == 0.0 && options.max_bond_dim == usize::MAX {
        return Ok(tt);
    }

    Ok(tt.compressed(&CompressionOptions {
        method: CompressionMethod::SVD,
        tolerance: options.tolerance,
        max_bond_dim: options.max_bond_dim,
        normalize_error: true,
    })?)
}

fn eval_interval_1d<F>(f: &F, interval: &Interval, basis: &LagrangePolynomials) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    basis
        .grid()
        .iter()
        .map(|&t| f(interval.start() + interval.length() * t))
        .collect()
}

fn eval_interval_nd<F>(
    f: &F,
    interval: &NInterval,
    basis: &LagrangePolynomials,
    basis_dim: usize,
) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let ndims = interval.ndims();
    let lengths = interval.lengths();
    let mut values = vec![0.0; basis_dim];
    for alpha_index in MultiIndexIter::new(&vec![basis.len(); ndims]) {
        let alpha_flat = flatten_index(&alpha_index, basis.len());
        let coords: Vec<_> = (0..ndims)
            .map(|d| interval.start()[d] + lengths[d] * basis.grid()[alpha_index[d]])
            .collect();
        values[alpha_flat] = f(&coords);
    }
    values
}

fn build_refined_qtt<F, S>(
    f: &F,
    domain: &NInterval,
    num_bits: usize,
    basis: &LagrangePolynomials,
    options: &InterpolativeQttOptions,
    is_safe: S,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
    S: Fn(&NInterval, usize) -> bool,
{
    let ndims = domain.ndims();
    let site_dim = site_dim(ndims)?;
    let basis_len = basis.len();
    let basis_dim = pow_usize(basis_len, ndims)?;
    let center_1d = interpolation_tensor(basis)?;
    let center = direct_product_core_tensors(&vec![center_1d; ndims])?;
    let mut cores = Vec::with_capacity(num_bits);

    let mut intervals = Vec::new();
    let mut first_unsafe = Vec::new();
    let mut first_safe_values = Vec::new();
    for (site, interval) in domain.split()?.into_iter().enumerate() {
        if is_safe(&interval, 0) {
            let values = eval_interval_nd(f, &interval, basis, basis_dim);
            first_safe_values.push((site, values));
        } else {
            let next = first_unsafe.len();
            first_unsafe.push((site, next));
            intervals.push(interval);
        }
    }

    let mut first = Tensor3::from_elem([1, site_dim, basis_dim + first_unsafe.len()], 0.0);
    for (site, values) in first_safe_values {
        for (beta, value) in values.into_iter().enumerate() {
            first[[0, site, beta]] = value;
        }
    }
    for (site, next) in first_unsafe {
        first[[0, site, basis_dim + next]] = 1.0;
    }
    cores.push(first);

    for level in 1..num_bits - 1 {
        let mut next_intervals = Vec::new();
        let qell = intervals.len();
        let mut safe_values = Vec::new();
        let mut transitions = Vec::new();

        for (interval_index, interval) in intervals.iter().enumerate() {
            for (site, subinterval) in interval.split()?.into_iter().enumerate() {
                if is_safe(&subinterval, level) {
                    let values = eval_interval_nd(f, &subinterval, basis, basis_dim);
                    safe_values.push((interval_index, site, values));
                } else {
                    let next = next_intervals.len();
                    transitions.push((interval_index, site, next));
                    next_intervals.push(subinterval);
                }
            }
        }

        let mut core = Tensor3::from_elem(
            [basis_dim + qell, site_dim, basis_dim + next_intervals.len()],
            0.0,
        );
        copy_core(&center, &mut core, 0, 0);

        for (interval_index, site, values) in safe_values {
            for (beta, value) in values.into_iter().enumerate() {
                core[[basis_dim + interval_index, site, beta]] = value;
            }
        }
        for (interval_index, site, next) in transitions {
            core[[basis_dim + interval_index, site, basis_dim + next]] = 1.0;
        }

        cores.push(core);
        intervals = next_intervals;
    }

    cores.push(last_core(f, basis, ndims, site_dim, basis_dim, &intervals)?);
    compress_train(cores, options)
}

fn build_adaptive_qtt<F>(
    f: &F,
    domain: &NInterval,
    num_bits: usize,
    basis: &LagrangePolynomials,
    dangerous_at_level: &[Vec<NInterval>],
    options: &InterpolativeQttOptions,
) -> Result<TensorTrain<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    build_refined_qtt(f, domain, num_bits, basis, options, |interval, level| {
        !is_dangerous(interval, &dangerous_at_level[level])
    })
}

fn copy_core(
    source: &Tensor3<f64>,
    target: &mut Tensor3<f64>,
    left_offset: usize,
    right_offset: usize,
) {
    for left in 0..source.left_dim() {
        for site in 0..source.site_dim() {
            for right in 0..source.right_dim() {
                target[[left_offset + left, site, right_offset + right]] =
                    *source.get3(left, site, right);
            }
        }
    }
}

fn last_core<F>(
    f: &F,
    basis: &LagrangePolynomials,
    ndims: usize,
    site_dim: usize,
    basis_dim: usize,
    intervals: &[NInterval],
) -> Result<Tensor3<f64>>
where
    F: Fn(&[f64]) -> f64,
{
    let right_1d = right_core_1d(basis)?;
    let right = direct_product_core_tensors(&vec![right_1d; ndims])?;
    let mut last = Tensor3::from_elem([basis_dim + intervals.len(), site_dim, 1], 0.0);
    copy_core(&right, &mut last, 0, 0);

    for (interval_index, interval) in intervals.iter().enumerate() {
        for (site, subinterval) in interval.split()?.into_iter().enumerate() {
            last[[basis_dim + interval_index, site, 0]] = f(subinterval.start());
        }
    }

    Ok(last)
}

fn add_singularity_path(
    dangerous_at_level: &mut [Vec<NInterval>],
    domain: &NInterval,
    singularity: &[f64],
) -> Result<()> {
    let mut interval = domain.clone();
    let levels_to_mark = dangerous_at_level.len().saturating_sub(1);
    for dangerous in dangerous_at_level.iter_mut().take(levels_to_mark) {
        if !is_dangerous(&interval, dangerous) {
            dangerous.push(interval.clone());
        }

        let mut next_interval = None;
        for subinterval in interval.split()? {
            if subinterval.contains(singularity) {
                next_interval = Some(subinterval);
                break;
            }
        }
        if let Some(next) = next_interval {
            interval = next;
        }
    }
    Ok(())
}

fn detect_dangerous_intervals<F>(
    dangerous_at_level: &mut [Vec<NInterval>],
    f: &F,
    interval: &NInterval,
    level: usize,
    max_level: usize,
    basis: &LagrangePolynomials,
    tolerance: f64,
) -> Result<()>
where
    F: Fn(&[f64]) -> f64,
{
    if level > max_level {
        return Ok(());
    }

    let error = estimate_interpolation_error_ninterval(f, interval, basis)?;
    if error > tolerance {
        dangerous_at_level[level].push(interval.clone());
        for subinterval in interval.split()? {
            detect_dangerous_intervals(
                dangerous_at_level,
                f,
                &subinterval,
                level + 1,
                max_level,
                basis,
                tolerance,
            )?;
        }
    }
    Ok(())
}

fn is_dangerous(interval: &NInterval, dangerous_list: &[NInterval]) -> bool {
    dangerous_list.iter().any(|candidate| {
        same_point(candidate.start(), interval.start())
            && same_point(candidate.stop(), interval.stop())
    })
}

fn same_point(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(&left, &right)| (left - right).abs() < 1.0e-14)
}

fn matrix_from_fn(
    nrows: usize,
    ncols: usize,
    mut f: impl FnMut(usize, usize) -> f64,
) -> Result<Matrix<f64>> {
    let len = nrows
        .checked_mul(ncols)
        .ok_or_else(|| invalid_argument("matrix shape overflows usize"))?;
    let mut data = Vec::with_capacity(len);
    for col in 0..ncols {
        for row in 0..nrows {
            data.push(f(row, col));
        }
    }
    Ok(Matrix::from_col_major_vec(nrows, ncols, data))
}

fn invert_stage1(tt: &TensorTrain<f64>, basis: &LagrangePolynomials) -> Result<Matrix<f64>> {
    let cores = tt.site_tensors();
    let num_sites = cores.len();
    let k_out = num_sites - 1;
    let basis_len = basis.len();
    let last = &cores[num_sites - 1];
    let r_last = last.left_dim();

    let mut decode = Matrix::zeros(r_last, basis_len);
    for left in 0..r_last {
        for beta in 0..basis_len {
            let c = basis.grid()[beta];
            let value =
                *last.get3(left, 0, 0) * (1.0 - 2.0 * c) + *last.get3(left, 1, 0) * (2.0 * c);
            decode[[left, beta]] = value;
        }
    }

    let first = &cores[0];
    let mut current = matrix_from_fn(first.site_dim(), first.right_dim(), |site, right| {
        *first.get3(0, site, right)
    })?;

    for core in cores.iter().take(k_out).skip(1) {
        let mut next = Matrix::zeros(current.nrows() * core.site_dim(), core.right_dim());
        for row in 0..current.nrows() {
            for site in 0..core.site_dim() {
                for right in 0..core.right_dim() {
                    let mut sum = 0.0;
                    for left in 0..core.left_dim() {
                        sum += current[[row, left]] * *core.get3(left, site, right);
                    }
                    next[[row * core.site_dim() + site, right]] = sum;
                }
            }
        }
        current = next;
    }

    if current.ncols() != decode.nrows() {
        return Err(invalid_argument(format!(
            "matrix shape mismatch: {}x{} times {}x{}",
            current.nrows(),
            current.ncols(),
            decode.nrows(),
            decode.ncols()
        )));
    }
    mat_mul(&current, &decode)
        .map_err(|err| invalid_argument(format!("decode multiply failed: {err}")))
}

fn build_restriction(basis: &LagrangePolynomials) -> Result<(Matrix<f64>, Matrix<f64>)> {
    let n = basis.len();
    let mut left = Matrix::zeros(n, n);
    let mut right = Matrix::zeros(n, n);

    for (gamma, &c) in basis.grid().iter().enumerate() {
        if c <= 0.5 {
            for beta in 0..n {
                left[[gamma, beta]] = basis.evaluate(beta, 2.0 * c)?;
            }
        } else {
            for beta in 0..n {
                right[[gamma, beta]] = basis.evaluate(beta, 2.0 * c - 1.0)?;
            }
        }
    }
    Ok((left, right))
}

fn apply_stage2(
    fine: &Matrix<f64>,
    r_left: &Matrix<f64>,
    r_right: &Matrix<f64>,
) -> Result<Matrix<f64>> {
    if !fine.nrows().is_multiple_of(2) {
        return Err(invalid_argument("fine matrix row count must be even"));
    }
    let mut coarse = Matrix::zeros(fine.nrows() / 2, fine.ncols());
    for row in 0..coarse.nrows() {
        for gamma in 0..coarse.ncols() {
            let mut value = 0.0;
            for beta in 0..fine.ncols() {
                value += r_left[[gamma, beta]] * fine[[2 * row, beta]];
                value += r_right[[gamma, beta]] * fine[[2 * row + 1, beta]];
            }
            coarse[[row, gamma]] = value;
        }
    }
    Ok(coarse)
}

fn dense_test_points(basis_len: usize) -> Vec<f64> {
    let denominator = (2 * basis_len - 1) as f64;
    (0..2 * basis_len)
        .map(|i| 0.5 * (1.0 - ((i as f64) * std::f64::consts::PI / denominator).cos()))
        .collect()
}

struct MultiIndexIter {
    dims: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl MultiIndexIter {
    fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
            current: vec![0; dims.len()],
            done: dims.contains(&0),
        }
    }
}

impl Iterator for MultiIndexIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();
        for axis in 0..self.dims.len() {
            self.current[axis] += 1;
            if self.current[axis] < self.dims[axis] {
                return Some(result);
            }
            self.current[axis] = 0;
        }
        self.done = true;
        Some(result)
    }
}
