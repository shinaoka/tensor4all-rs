//! Shared setup for the Fourier tutorial.
//!
//! These helpers keep the Gaussian Fourier example focused on the operator
//! flow instead of file output and coordinate plumbing.

use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use num_complex::Complex64;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::TensorDynLen;
use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, QuanticsTensorCI2, UnfoldingScheme,
};
use tensor4all_quanticstransform::{quantics_fourier_operator, FourierOptions};
use tensor4all_treetn::{
    apply_linear_operator, tensor_train_to_treetn, ApplyOptions, LinearOperator, TreeTN,
};

/// Quantics site index type used by the Fourier helper.
pub type SiteIndex = Index<DynId, TagSet>;

/// Configuration for the Fourier tutorial family.
#[derive(Debug, Clone, Copy)]
pub struct FourierTutorialConfig {
    pub bits: usize,
    pub x_lower_bound: f64,
    pub x_upper_bound: f64,
    pub include_endpoint: bool,
    pub tolerance: f64,
    pub max_bond_dim: usize,
}

/// Default configuration used throughout the Fourier tutorial.
pub const DEFAULT_FOURIER_CONFIG: FourierTutorialConfig = FourierTutorialConfig {
    bits: 10,
    x_lower_bound: -10.0,
    x_upper_bound: 10.0,
    include_endpoint: true,
    tolerance: 1e-12,
    max_bond_dim: 32,
};

/// QTT construction result returned by the Fourier helper.
pub type FourierQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);
/// Transformed Fourier state and the site indices needed for evaluation.
pub type FourierTransformOutput = (TreeTN<TensorDynLen, usize>, Vec<SiteIndex>);

/// One row in the Fourier sample table.
#[derive(Debug, Clone)]
pub struct SamplePoint {
    pub index: usize,
    pub x: f64,
    pub k: f64,
    pub analytic_re: f64,
    pub analytic_im: f64,
    pub qtt_re: f64,
    pub qtt_im: f64,
    pub abs_error: f64,
}

/// Standard Gaussian used as the input function.
pub fn gaussian_target(x: f64) -> f64 {
    (-0.5 * x * x).exp()
}

/// Analytic Fourier transform of the standard Gaussian.
///
/// This uses the convention
///
/// ```text
/// \hat{f}(k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} dx
/// ```
///
/// so `exp(-x^2 / 2)` transforms to `sqrt(2π) * exp(-2π² k²)`.
pub fn gaussian_fourier_reference(k: f64) -> Complex64 {
    Complex64::new((2.0 * PI).sqrt() * (-2.0 * PI * PI * k * k).exp(), 0.0)
}

/// Return the input-grid spacing used for the continuous Fourier approximation.
pub fn input_spacing(config: &FourierTutorialConfig) -> f64 {
    let npoints = 1usize << config.bits;
    (config.x_upper_bound - config.x_lower_bound) / (npoints as f64 - 1.0)
}

/// Return the physical frequency step implied by the sampled input grid.
pub fn physical_frequency_step(config: &FourierTutorialConfig) -> f64 {
    let npoints = 1usize << config.bits;
    1.0 / (npoints as f64 * input_spacing(config))
}

/// Return the centered physical frequency bounds for the tutorial output grid.
pub fn physical_frequency_bounds(config: &FourierTutorialConfig) -> (f64, f64) {
    let npoints = 1usize << config.bits;
    let frequency_step = physical_frequency_step(config);
    let lower_bound = -(npoints as f64 / 2.0) * frequency_step;
    let upper_bound = lower_bound + (npoints as f64 - 1.0) * frequency_step;
    (lower_bound, upper_bound)
}

/// Build the QTT approximation of the input Gaussian.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_gaussian_qtt(
    grid: &DiscretizedGrid,
    config: &FourierTutorialConfig,
) -> Result<FourierQttOutput, Box<dyn Error>> {
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    let f = |coords: &[f64]| -> f64 { gaussian_target(coords[0]) };
    let npoints = 1_i64 << config.bits;
    let initial_pivots = vec![vec![1], vec![npoints / 2], vec![npoints]];

    Ok(quanticscrossinterpolate(
        grid,
        f,
        Some(initial_pivots),
        options,
    )?)
}

/// Build the quantics Fourier operator.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_fourier_operator(
    config: &FourierTutorialConfig,
) -> Result<LinearOperator<TensorDynLen, usize>, Box<dyn Error>> {
    let options = FourierOptions {
        max_bond_dim: config.max_bond_dim,
        tolerance: config.tolerance,
        ..FourierOptions::forward()
    };
    Ok(quantics_fourier_operator(config.bits, options)?)
}

/// Convert a 1-based discrete index into the binary site values used by the quantics grid.
pub fn global_index_to_quantics_sites(index_1based: usize, bits: usize) -> Vec<usize> {
    let mut sites = Vec::with_capacity(bits);

    for bit in (0..bits).rev() {
        sites.push(((index_1based - 1) >> bit) & 1);
    }

    sites
}

/// Apply the Fourier operator to the Gaussian QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn transform_gaussian(
    qtci: &QuanticsTensorCI2<f64>,
    operator: &LinearOperator<TensorDynLen, usize>,
    _config: &FourierTutorialConfig,
) -> Result<FourierTransformOutput, Box<dyn Error>> {
    let tt = qtci.tensor_train();
    let (state, site_indices) = tensor_train_to_treetn(&tt)?;
    let mut aligned_operator = operator.clone();
    aligned_operator.align_to_state(&state)?;

    let transformed = apply_linear_operator(&aligned_operator, &state, ApplyOptions::naive())?;
    Ok((transformed, site_indices))
}

/// Evaluate a TreeTN at one set of site values and return the complex scalar.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn evaluate_tree_point(
    tn: &TreeTN<TensorDynLen, usize>,
    site_indices: &[SiteIndex],
    site_values: &[usize],
) -> Result<Complex64, Box<dyn Error>> {
    let value = tn.evaluate_point(site_indices, site_values)?;
    Ok(Complex64::new(value.real(), value.imag()))
}

/// Evaluate the transformed Gaussian on the frequency grid and collect a sample table.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples(
    transformed: &TreeTN<TensorDynLen, usize>,
    site_indices: &[SiteIndex],
    input_grid: &DiscretizedGrid,
    frequency_grid: &DiscretizedGrid,
    config: &FourierTutorialConfig,
) -> Result<Vec<SamplePoint>, Box<dyn Error>> {
    let x_coords = input_grid.grid_origcoords(0)?;
    let k_coords = frequency_grid.grid_origcoords(0)?;
    let delta_x = input_spacing(config);
    let npoints = x_coords.len();
    let mut rows = Vec::with_capacity(k_coords.len());

    for (row_index, &k) in k_coords.iter().enumerate() {
        let centered_bin = row_index as isize - (npoints as isize / 2);
        let coefficient_index = centered_bin.rem_euclid(npoints as isize) as usize;
        let mut site_values = global_index_to_quantics_sites(coefficient_index + 1, config.bits);
        site_values.reverse();
        // this is the actual extraction, the rest is just scaling and phase factors to match the continuous Fourier convention
        let raw_qtt = evaluate_tree_point(transformed, site_indices, &site_values)?;
        let qtt = raw_qtt
            * Complex64::new(delta_x * (npoints as f64).sqrt(), 0.0)
            * Complex64::from_polar(1.0, -2.0 * PI * k * config.x_lower_bound);
        let exact = gaussian_fourier_reference(k);

        rows.push(SamplePoint {
            index: row_index + 1,
            x: x_coords[coefficient_index],
            k,
            analytic_re: exact.re,
            analytic_im: exact.im,
            qtt_re: qtt.re,
            qtt_im: qtt.im,
            abs_error: (qtt - exact).norm(),
        });
    }

    Ok(rows)
}

/// Pair input and transformed bond dimensions for the tutorial CSV output.
pub fn collect_bond_dims_from_profiles(
    input_dims: &[usize],
    transformed_dims: &[usize],
) -> Vec<(usize, usize, usize)> {
    input_dims
        .iter()
        .zip(transformed_dims.iter())
        .enumerate()
        .map(|(i, (&input_dim, &transformed_dim))| (i + 1, input_dim, transformed_dim))
        .collect()
}

/// Print a compact summary for the Gaussian Fourier tutorial.
pub fn print_summary(
    input_qtci: &QuanticsTensorCI2<f64>,
    transformed: &TreeTN<TensorDynLen, usize>,
    ranks: &[usize],
    errors: &[f64],
    samples: &[SamplePoint],
    config: &FourierTutorialConfig,
) {
    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);

    println!("QTT Fourier tutorial");
    println!("bits = {}", config.bits);
    println!(
        "x interval = [{:.3}, {:.3}]",
        config.x_lower_bound, config.x_upper_bound
    );
    let (k_lower, k_upper) = physical_frequency_bounds(config);
    println!("k interval = [{:.3}, {:.3}]", k_lower, k_upper);
    println!("input rank = {}", input_qtci.rank());
    println!("rank history length = {}", ranks.len());
    println!("error history length = {}", errors.len());
    println!("output node count = {}", transformed.node_count());
    println!("max abs error = {:.3e}", max_abs_error);
    println!();
}

/// Write the Fourier sample table used by Julia plotting.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(path: &Path, samples: &[SamplePoint]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "index,x,k,analytic_re,analytic_im,qtt_re,qtt_im,abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.index,
            sample.x,
            sample.k,
            sample.analytic_re,
            sample.analytic_im,
            sample.qtt_re,
            sample.qtt_im,
            sample.abs_error
        )?;
    }

    Ok(())
}

/// Write the Fourier bond-dimension table used by Julia plotting.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(
    path: &Path,
    bond_dims: &[(usize, usize, usize)],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,input_bond_dim,transformed_bond_dim")?;
    for (index, input_dim, transformed_dim) in bond_dims {
        writeln!(w, "{},{},{}", index, input_dim, transformed_dim)?;
    }

    Ok(())
}

/// Write the Fourier MPO bond-dimension table used by Julia plotting.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_fourier_operator_bond_dims_csv(
    path: &Path,
    bond_dims: &[usize],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,bond_dim")?;
    for (index, bond_dim) in bond_dims.iter().enumerate() {
        writeln!(w, "{},{}", index + 1, bond_dim)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_target_peaks_at_one() {
        assert!((gaussian_target(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn gaussian_fourier_reference_is_real_and_even() {
        let left = gaussian_fourier_reference(-1.25);
        let right = gaussian_fourier_reference(1.25);

        assert!((left.re - right.re).abs() < 1e-12);
        assert!(left.im.abs() < 1e-12);
        assert!(right.im.abs() < 1e-12);
        assert!((gaussian_fourier_reference(0.0).re - (2.0 * PI).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn global_index_to_quantics_sites_uses_msb_first_order() {
        assert_eq!(global_index_to_quantics_sites(1, 3), vec![0, 0, 0]);
        assert_eq!(global_index_to_quantics_sites(5, 3), vec![1, 0, 0]);
        assert_eq!(global_index_to_quantics_sites(8, 3), vec![1, 1, 1]);
    }
}
