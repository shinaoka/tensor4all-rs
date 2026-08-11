//! Shared setup for the 2D partial Fourier tutorial.
//!
//! This module builds a 2D interleaved QTT for `f(x,t)`, applies a 1D Fourier
//! operator only to the x-sites, and compares the result with an analytic
//! partial Fourier transform.

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

pub type SiteIndex = Index<DynId, TagSet>;
pub type PartialFourier2dQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);
pub type PartialFourier2dTransformOutput = (TreeTN<TensorDynLen, usize>, Vec<SiteIndex>);

#[derive(Debug, Clone, Copy)]
pub struct PartialFourier2dConfig {
    pub bits: usize,
    pub x_lower_bound: f64,
    pub x_upper_bound: f64,
    pub t_lower_bound: f64,
    pub t_upper_bound: f64,
    pub include_endpoint: bool,
    pub t_frequency: usize,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub maxiter: usize,
}

pub const DEFAULT_PARTIAL_FOURIER2D_CONFIG: PartialFourier2dConfig = PartialFourier2dConfig {
    bits: 6,
    x_lower_bound: -10.0,
    x_upper_bound: 10.0,
    t_lower_bound: 0.0,
    t_upper_bound: 1.0,
    include_endpoint: true,
    t_frequency: 3,
    tolerance: 1e-12,
    max_bond_dim: 64,
    maxiter: 20,
};

#[derive(Debug, Clone)]
pub struct PartialFourier2dSamplePoint {
    pub k_index: usize,
    pub t_index: usize,
    pub source_x_index: usize,
    pub k: f64,
    pub t: f64,
    pub analytic_re: f64,
    pub analytic_im: f64,
    pub qtt_re: f64,
    pub qtt_im: f64,
    pub abs_error: f64,
}

pub type PartialFourier2dBondDimRow = (usize, Option<usize>, Option<usize>);

pub fn point_count(config: &PartialFourier2dConfig) -> usize {
    1usize << config.bits
}

pub fn source_function(x: f64, t: f64, config: &PartialFourier2dConfig) -> f64 {
    (-0.5 * x * x).exp() * (2.0 * PI * config.t_frequency as f64 * t).cos()
}

pub fn partial_fourier_reference(k: f64, t: f64, config: &PartialFourier2dConfig) -> Complex64 {
    let gaussian = (2.0 * PI).sqrt() * (-2.0 * PI * PI * k * k).exp();
    let temporal = (2.0 * PI * config.t_frequency as f64 * t).cos();
    Complex64::new(gaussian * temporal, 0.0)
}

pub fn input_spacing(config: &PartialFourier2dConfig) -> f64 {
    let npoints = point_count(config);
    (config.x_upper_bound - config.x_lower_bound) / (npoints as f64 - 1.0)
}

pub fn physical_frequency_step(config: &PartialFourier2dConfig) -> f64 {
    let npoints = point_count(config);
    1.0 / (npoints as f64 * input_spacing(config))
}

pub fn physical_frequency_bounds(config: &PartialFourier2dConfig) -> (f64, f64) {
    let npoints = point_count(config);
    let frequency_step = physical_frequency_step(config);
    let lower_bound = -(npoints as f64 / 2.0) * frequency_step;
    let upper_bound = lower_bound + (npoints as f64 - 1.0) * frequency_step;
    (lower_bound, upper_bound)
}

pub fn x_site_node_mapping(bits: usize) -> Vec<(usize, usize)> {
    (0..bits).map(|site| (site, 2 * site)).collect()
}

pub fn global_index_to_quantics_sites(index_1based: usize, bits: usize) -> Vec<usize> {
    let mut sites = Vec::with_capacity(bits);
    for bit in (0..bits).rev() {
        sites.push(((index_1based - 1) >> bit) & 1);
    }
    sites
}

pub fn interleaved_site_values(
    x_zero_based: usize,
    t_zero_based: usize,
    bits: usize,
) -> Vec<usize> {
    let x_sites = global_index_to_quantics_sites(x_zero_based + 1, bits);
    let t_sites = global_index_to_quantics_sites(t_zero_based + 1, bits);
    x_sites
        .into_iter()
        .zip(t_sites)
        .flat_map(|(x_site, t_site)| [x_site, t_site])
        .collect()
}

/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_input_grid(
    config: &PartialFourier2dConfig,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&x_t_names())
        .with_lower_bound(&[config.x_lower_bound, config.t_lower_bound])
        .with_upper_bound(&[config.x_upper_bound, config.t_upper_bound])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_frequency_grid(
    config: &PartialFourier2dConfig,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    let (k_lower_bound, k_upper_bound) = physical_frequency_bounds(config);
    Ok(DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&k_t_names())
        .with_lower_bound(&[k_lower_bound, config.t_lower_bound])
        .with_upper_bound(&[k_upper_bound, config.t_upper_bound])
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

fn x_t_names() -> Vec<&'static str> {
    vec!["x", "t"]
}

fn k_t_names() -> Vec<&'static str> {
    vec!["k", "t"]
}

/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_source_qtt(
    grid: &DiscretizedGrid,
    config: &PartialFourier2dConfig,
) -> Result<PartialFourier2dQttOutput, Box<dyn Error>> {
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    let npoints = point_count(config) as i64;
    let x_indices = [
        npoints / 8,
        npoints / 4,
        3 * npoints / 8,
        npoints / 2 - 1,
        npoints / 2,
        npoints / 2 + 1,
        5 * npoints / 8,
        3 * npoints / 4,
        7 * npoints / 8,
    ];
    let t_indices: Vec<i64> = (1..=npoints).collect();
    let initial_pivots = x_indices
        .iter()
        .flat_map(|&x_index| t_indices.iter().map(move |&t_index| vec![x_index, t_index]))
        .collect();

    let config_copy = *config;
    let f = move |coords: &[f64]| -> f64 { source_function(coords[0], coords[1], &config_copy) };
    Ok(quanticscrossinterpolate(
        grid,
        f,
        Some(initial_pivots),
        options,
    )?)
}
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_partial_fourier_operator(
    config: &PartialFourier2dConfig,
) -> Result<LinearOperator<TensorDynLen, usize>, Box<dyn Error>> {
    let options = FourierOptions {
        max_bond_dim: Some(config.max_bond_dim),
        tolerance: config.tolerance,
        ..FourierOptions::forward()
    };
    let operator = quantics_fourier_operator(config.bits, options)?;
    Ok(operator.rename_nodes(&x_site_node_mapping(config.bits))?)
}

fn single_site_index_from_state(
    state: &TreeTN<TensorDynLen, usize>,
    node: usize,
) -> Result<SiteIndex, Box<dyn Error>> {
    let site_space = state
        .site_space(&node)
        .ok_or_else(|| format!("state node {node} has no site space"))?;
    if site_space.len() != 1 {
        return Err(format!(
            "state node {node} should have exactly one site index, got {}",
            site_space.len()
        )
        .into());
    }
    site_space
        .iter()
        .next()
        .cloned()
        .ok_or_else(|| format!("state node {node} has an empty site space").into())
}

/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn transform_x_dimension(
    qtci: &QuanticsTensorCI2<f64>,
    operator: &LinearOperator<TensorDynLen, usize>,
) -> Result<PartialFourier2dTransformOutput, Box<dyn Error>> {
    let tt = qtci.tensor_train();
    let (state, _state_site_indices) = tensor_train_to_treetn(&tt)?;
    let mut aligned_operator = operator.clone();
    aligned_operator.align_to_state(&state)?;

    let transformed = apply_linear_operator(&aligned_operator, &state, ApplyOptions::naive())?;
    let output_site_indices = (0..(2 * operator.input_mappings().len()))
        .map(|node| single_site_index_from_state(&transformed, node))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((transformed, output_site_indices))
}

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
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples(
    transformed: &TreeTN<TensorDynLen, usize>,
    site_indices: &[SiteIndex],
    frequency_grid: &DiscretizedGrid,
    config: &PartialFourier2dConfig,
) -> Result<Vec<PartialFourier2dSamplePoint>, Box<dyn Error>> {
    let expected_sites = 2 * config.bits;
    if site_indices.len() != expected_sites {
        return Err(format!(
            "partial Fourier output should expose {expected_sites} interleaved (k,t) sites, got {}",
            site_indices.len()
        )
        .into());
    }

    let k_coords = frequency_grid.grid_origcoords(0)?;
    let t_coords = frequency_grid.grid_origcoords(1)?;
    let npoints = point_count(config);
    let delta_x = input_spacing(config);
    let mut rows = Vec::with_capacity(k_coords.len() * t_coords.len());

    for (k_offset, &k) in k_coords.iter().enumerate() {
        let centered_bin = k_offset as isize - (npoints as isize / 2);
        let coefficient_index = centered_bin.rem_euclid(npoints as isize) as usize;
        let mut k_sites = global_index_to_quantics_sites(coefficient_index + 1, config.bits);
        k_sites.reverse();

        for (t_offset, &t) in t_coords.iter().enumerate() {
            let t_sites = global_index_to_quantics_sites(t_offset + 1, config.bits);
            let site_values: Vec<usize> = k_sites
                .iter()
                .copied()
                .zip(t_sites)
                .flat_map(|(k_site, t_site)| [k_site, t_site])
                .collect();

            let raw_qtt = evaluate_tree_point(transformed, site_indices, &site_values)?;
            let qtt = raw_qtt
                * Complex64::new(delta_x * (npoints as f64).sqrt(), 0.0)
                * Complex64::from_polar(1.0, -2.0 * PI * k * config.x_lower_bound);
            let exact = partial_fourier_reference(k, t, config);

            rows.push(PartialFourier2dSamplePoint {
                k_index: k_offset + 1,
                t_index: t_offset + 1,
                source_x_index: coefficient_index + 1,
                k,
                t,
                analytic_re: exact.re,
                analytic_im: exact.im,
                qtt_re: qtt.re,
                qtt_im: qtt.im,
                abs_error: (qtt - exact).norm(),
            });
        }
    }

    Ok(rows)
}

pub fn collect_bond_dims(
    input: &[usize],
    transformed: &[usize],
) -> Vec<PartialFourier2dBondDimRow> {
    let row_count = input.len().max(transformed.len());
    (0..row_count)
        .map(|i| (i + 1, input.get(i).copied(), transformed.get(i).copied()))
        .collect()
}

fn write_optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(
    path: &Path,
    samples: &[PartialFourier2dSamplePoint],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "k_index,t_index,source_x_index,k,t,analytic_re,analytic_im,qtt_re,qtt_im,abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.k_index,
            sample.t_index,
            sample.source_x_index,
            sample.k,
            sample.t,
            sample.analytic_re,
            sample.analytic_im,
            sample.qtt_re,
            sample.qtt_im,
            sample.abs_error
        )?;
    }

    Ok(())
}

/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(
    path: &Path,
    bond_dims: &[PartialFourier2dBondDimRow],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,input_bond_dim,transformed_bond_dim")?;
    for (index, input_dim, transformed_dim) in bond_dims {
        writeln!(
            w,
            "{},{},{}",
            index,
            write_optional_usize(*input_dim),
            write_optional_usize(*transformed_dim)
        )?;
    }

    Ok(())
}

/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_operator_bond_dims_csv(
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

pub fn print_summary(
    input_qtci: &QuanticsTensorCI2<f64>,
    transformed: &TreeTN<TensorDynLen, usize>,
    ranks: &[usize],
    errors: &[f64],
    samples: &[PartialFourier2dSamplePoint],
    config: &PartialFourier2dConfig,
) {
    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    let (k_lower, k_upper) = physical_frequency_bounds(config);

    println!("QTT 2D partial Fourier tutorial");
    println!("bits = {}", config.bits);
    println!(
        "x interval = [{:.3}, {:.3}]",
        config.x_lower_bound, config.x_upper_bound
    );
    println!(
        "t interval = [{:.3}, {:.3}]",
        config.t_lower_bound, config.t_upper_bound
    );
    println!("k interval = [{:.3}, {:.3}]", k_lower, k_upper);
    println!("t frequency = {}", config.t_frequency);
    println!("input rank = {}", input_qtci.rank());
    println!("rank history length = {}", ranks.len());
    println!("error history length = {}", errors.len());
    println!("output node count = {}", transformed.node_count());
    println!("max abs error = {:.3e}", max_abs_error);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_function_peaks_at_one() {
        let config = DEFAULT_PARTIAL_FOURIER2D_CONFIG;
        assert!((source_function(0.0, 0.0, &config) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn partial_fourier_reference_is_real_and_even() {
        let config = DEFAULT_PARTIAL_FOURIER2D_CONFIG;
        let left = partial_fourier_reference(-1.25, 0.25, &config);
        let right = partial_fourier_reference(1.25, 0.25, &config);
        assert!((left.re - right.re).abs() < 1e-12);
        assert!(left.im.abs() < 1e-12);
        assert!(right.im.abs() < 1e-12);
        assert!(
            (partial_fourier_reference(0.0, 0.0, &config).re - (2.0 * PI).sqrt()).abs() < 1e-12
        );
    }
}
