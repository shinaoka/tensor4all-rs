//! Shared setup for the multivariate QTT tutorial.
//!
//! This helper implements a focused two-dimensional tutorial: grid creation,
//! QTT construction for two unfolding schemes, dense sampling, bond-dimension
//! pairing, CSV writers and a compact terminal summary.

use std::error::Error;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, QuanticsTensorCI2, UnfoldingScheme,
};
use tensor4all_simplett::{MultiIndex, TTCache};

/// Configuration for the multivariate tutorial.
#[derive(Debug, Clone, Copy)]
pub struct MultivariateTutorialConfig {
    pub bits: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub include_endpoint: bool,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub maxiter: usize,
}

/// Default configuration used throughout the tutorial.
pub const DEFAULT_MULTIVARIATE_CONFIG: MultivariateTutorialConfig = MultivariateTutorialConfig {
    bits: 7,
    lower_bound: -2.0,
    upper_bound: 2.0,
    include_endpoint: false,
    tolerance: 1e-12,
    max_bond_dim: 64,
    maxiter: 20,
};

/// QTT construction result returned by the multivariate helper.
pub type MultivariateQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);

pub fn point_count(config: &MultivariateTutorialConfig) -> usize {
    1usize << config.bits
}

/// Build the two-dimensional discretized grid used by the tutorial.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multivariate_grid(
    config: &MultivariateTutorialConfig,
    scheme: UnfoldingScheme,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&["x", "y"])
        .with_bounds(config.lower_bound, config.upper_bound)
        .with_unfolding_scheme(scheme)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

/// Build a QTT approximation for the provided 2D `target_fn`.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multivariate_qtt<F>(
    grid: &DiscretizedGrid,
    target_fn: F,
    config: &MultivariateTutorialConfig,
) -> Result<MultivariateQttOutput, Box<dyn Error>>
where
    F: Fn(f64, f64) -> f64 + Copy + 'static,
{
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_verbosity(0);

    let f = move |coords: &[f64]| -> f64 { target_fn(coords[0], coords[1]) };
    let npoints = point_count(config) as i64;
    let initial_pivots = vec![
        vec![1, 1],
        vec![npoints / 2, npoints / 2],
        vec![npoints, npoints],
    ];

    Ok(quanticscrossinterpolate(
        grid,
        f,
        Some(initial_pivots),
        options,
    )?)
}

/// One sample row in the exported dense table.
#[derive(Debug, Clone)]
pub struct MultivariateSamplePoint {
    pub x_index: usize,
    pub y_index: usize,
    pub x: f64,
    pub y: f64,
    pub exact: f64,
    pub interleaved_qtt: f64,
    pub grouped_qtt: f64,
    pub interleaved_abs_error: f64,
    pub grouped_abs_error: f64,
}

pub type BondDimRow = (usize, Option<usize>, Option<usize>);

/// Summary data for one QTT layout.
pub struct LayoutSummary<'a> {
    pub name: &'static str,
    pub qtt: &'a QuanticsTensorCI2<f64>,
    pub ranks: &'a [usize],
    pub errors: &'a [f64],
}

const DENSE_EVALUATION_BATCH_POINTS: usize = 8192;

fn invalid_multivariate_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

fn discretized_grid(qtt: &QuanticsTensorCI2<f64>) -> Result<&DiscretizedGrid, Box<dyn Error>> {
    qtt.discretized_grid()
        .ok_or_else(|| invalid_multivariate_input("QTT result does not contain a discretized grid"))
}

fn validate_dense_grid(grid: &DiscretizedGrid) -> Result<(), Box<dyn Error>> {
    if grid.ndims() != 2 {
        return Err(invalid_multivariate_input(format!(
            "multivariate tutorial expects a 2D grid, got {} dimensions",
            grid.ndims()
        )));
    }

    Ok(())
}

fn dense_quantics_batch(
    grid: &DiscretizedGrid,
    y_count: usize,
    start: usize,
    len: usize,
) -> Result<Vec<MultiIndex>, Box<dyn Error>> {
    let mut indices = Vec::with_capacity(len);

    for flat_index in start..start + len {
        let x_index = (flat_index / y_count) + 1;
        let y_index = (flat_index % y_count) + 1;
        let quantics = grid.grididx_to_quantics(&[x_index as i64, y_index as i64])?;
        indices.push(quantics.iter().map(|&q| (q - 1) as usize).collect());
    }

    Ok(indices)
}

fn evaluate_dense_qtt(qtt: &QuanticsTensorCI2<f64>) -> Result<Vec<f64>, Box<dyn Error>> {
    let grid = discretized_grid(qtt)?;
    validate_dense_grid(grid)?;

    let x_count = grid.grid_origcoords(0)?.len();
    let y_count = grid.grid_origcoords(1)?.len();
    let total_points = x_count * y_count;

    let tensor_train = qtt.tensor_train();
    let mut cache = TTCache::new(&tensor_train);
    let split = Some(cache.len().max(1) / 2).filter(|&split| split > 0);
    let mut values = Vec::with_capacity(total_points);

    let mut start = 0;
    while start < total_points {
        let batch_len = DENSE_EVALUATION_BATCH_POINTS.min(total_points - start);
        let quantics = dense_quantics_batch(grid, y_count, start, batch_len)?;
        let batch_values = cache.evaluate_many(&quantics, split)?;
        values.extend(batch_values);
        start += batch_len;
    }

    Ok(values)
}

/// Collect dense samples for both layouts on the full cartesian grid.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples<F>(
    interleaved: &QuanticsTensorCI2<f64>,
    grouped: &QuanticsTensorCI2<f64>,
    exact_fn: F,
) -> Result<Vec<MultivariateSamplePoint>, Box<dyn Error>>
where
    F: Fn(f64, f64) -> f64,
{
    let grid = discretized_grid(interleaved)?;
    validate_dense_grid(grid)?;

    let x_coords = grid.grid_origcoords(0)?;
    let y_coords = grid.grid_origcoords(1)?;
    let interleaved_values = evaluate_dense_qtt(interleaved)?;
    let grouped_values = evaluate_dense_qtt(grouped)?;
    let expected_count = x_coords.len() * y_coords.len();

    if interleaved_values.len() != expected_count || grouped_values.len() != expected_count {
        return Err(invalid_multivariate_input(
            "dense QTT evaluation did not produce the expected sample count",
        ));
    }

    let mut samples = Vec::with_capacity(x_coords.len() * y_coords.len());

    for (x_offset, &x) in x_coords.iter().enumerate() {
        for (y_offset, &y) in y_coords.iter().enumerate() {
            let x_index = x_offset + 1;
            let y_index = y_offset + 1;
            let flat_index = x_offset * y_coords.len() + y_offset;
            let exact = exact_fn(x, y);
            let interleaved_qtt = interleaved_values[flat_index];
            let grouped_qtt = grouped_values[flat_index];

            samples.push(MultivariateSamplePoint {
                x_index,
                y_index,
                x,
                y,
                exact,
                interleaved_qtt,
                grouped_qtt,
                interleaved_abs_error: (exact - interleaved_qtt).abs(),
                grouped_abs_error: (exact - grouped_qtt).abs(),
            });
        }
    }

    Ok(samples)
}

/// Pair two bond-dimension profiles into rows for CSV output.
pub fn collect_bond_dims(interleaved: &[usize], grouped: &[usize]) -> Vec<BondDimRow> {
    let row_count = interleaved.len().max(grouped.len());
    (0..row_count)
        .map(|i| (i + 1, interleaved.get(i).copied(), grouped.get(i).copied()))
        .collect()
}

/// Write dense samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(
    path: &Path,
    samples: &[MultivariateSamplePoint],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "x_index,y_index,x,y,exact,interleaved_qtt,grouped_qtt,interleaved_abs_error,grouped_abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.x_index,
            sample.y_index,
            sample.x,
            sample.y,
            sample.exact,
            sample.interleaved_qtt,
            sample.grouped_qtt,
            sample.interleaved_abs_error,
            sample.grouped_abs_error
        )?;
    }

    Ok(())
}

/// Write bond-dimension rows to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(path: &Path, rows: &[BondDimRow]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,interleaved_bond_dim,grouped_bond_dim")?;
    for (bond_index, interleaved, grouped) in rows {
        let interleaved_text = interleaved.map(|v| v.to_string()).unwrap_or_default();
        let grouped_text = grouped.map(|v| v.to_string()).unwrap_or_default();
        writeln!(w, "{},{},{}", bond_index, interleaved_text, grouped_text)?;
    }

    Ok(())
}

/// Print a compact summary to the terminal used by the binary.
pub fn print_summary(
    interleaved: LayoutSummary<'_>,
    grouped: LayoutSummary<'_>,
    samples: &[MultivariateSamplePoint],
    config: &MultivariateTutorialConfig,
) {
    let max_interleaved_error = samples
        .iter()
        .map(|sample| sample.interleaved_abs_error)
        .fold(0.0_f64, f64::max);
    let max_grouped_error = samples
        .iter()
        .map(|sample| sample.grouped_abs_error)
        .fold(0.0_f64, f64::max);

    println!("QTT multivariate tutorial");
    println!("bits per dimension = {}", config.bits);
    println!("max QTCI sweeps = {}", config.maxiter);
    println!(
        "domain = [{:.3}, {:.3}) x [{:.3}, {:.3})",
        config.lower_bound, config.upper_bound, config.lower_bound, config.upper_bound
    );
    println!("sample count = {}", samples.len());
    println!("{} rank = {}", interleaved.name, interleaved.qtt.rank());
    println!("{} rank = {}", grouped.name, grouped.qtt.rank());
    println!(
        "{} rank history length = {}",
        interleaved.name,
        interleaved.ranks.len()
    );
    println!(
        "{} rank history length = {}",
        grouped.name,
        grouped.ranks.len()
    );
    println!(
        "{} error history length = {}",
        interleaved.name,
        interleaved.errors.len()
    );
    println!(
        "{} error history length = {}",
        grouped.name,
        grouped.errors.len()
    );
    println!("max interleaved abs error = {:.3e}", max_interleaved_error);
    println!("max grouped abs error = {:.3e}", max_grouped_error);
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn test_multivariate_target(x: f64, y: f64) -> f64 {
        (20.0 * PI * x * y).cos() / 1000.0
    }

    #[test]
    fn multivariate_target_matches_known_values() {
        assert!((test_multivariate_target(0.0, 0.75) - 0.001).abs() < 1e-15);
        assert!((test_multivariate_target(0.25, 0.2) + 0.001).abs() < 1e-15);
    }

    #[test]
    fn default_multivariate_config_uses_smoke_resolution() {
        assert_eq!(DEFAULT_MULTIVARIATE_CONFIG.bits, 7);
        assert_eq!(point_count(&DEFAULT_MULTIVARIATE_CONFIG), 128);
    }

    #[test]
    fn point_count_uses_power_of_two_resolution() {
        let config = MultivariateTutorialConfig {
            bits: 4,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };
        assert_eq!(point_count(&config), 16);
    }

    #[test]
    fn multivariate_grid_is_two_dimensional_and_half_open() -> Result<(), Box<dyn Error>> {
        let config = MultivariateTutorialConfig {
            bits: 3,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };
        let grid = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
        let x = grid.grid_origcoords(0)?;
        let y = grid.grid_origcoords(1)?;

        assert_eq!(grid.ndims(), 2);
        assert_eq!(grid.rs(), &[3, 3]);
        assert_eq!(grid.variable_names(), &["x".to_string(), "y".to_string()]);
        assert_eq!(x.len(), 8);
        assert_eq!(y.len(), 8);
        assert!((x[0] - config.lower_bound).abs() < 1e-15);
        assert!((y[0] - config.lower_bound).abs() < 1e-15);
        assert!(x.last().copied().expect("x has points") < config.upper_bound);
        assert!(y.last().copied().expect("y has points") < config.upper_bound);

        Ok(())
    }

    #[test]
    fn multivariate_grid_uses_requested_unfolding_scheme() -> Result<(), Box<dyn Error>> {
        let config = MultivariateTutorialConfig {
            bits: 3,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };
        let interleaved = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
        let grouped = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;

        assert_eq!(interleaved.local_dimensions(), vec![2; 6]);
        assert_eq!(grouped.local_dimensions(), vec![2; 6]);
        assert_ne!(
            interleaved.grididx_to_quantics(&[2, 5])?,
            grouped.grididx_to_quantics(&[2, 5])?
        );

        Ok(())
    }

    #[test]
    fn both_unfolding_schemes_build_accurate_small_qtts() -> Result<(), Box<dyn Error>> {
        let config = MultivariateTutorialConfig {
            bits: 4,
            max_bond_dim: 32,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };

        for scheme in [UnfoldingScheme::Interleaved, UnfoldingScheme::Grouped] {
            let grid = build_multivariate_grid(&config, scheme)?;
            let (qtci, ranks, errors) =
                build_multivariate_qtt(&grid, test_multivariate_target, &config)?;

            assert!(qtci.rank() > 0);
            assert!(!ranks.is_empty());
            assert!(!errors.is_empty());

            let value = qtci.evaluate(&[1, 1])?;
            let exact = test_multivariate_target(0.0, 0.0);
            assert!((value - exact).abs() < 1e-9);
        }

        Ok(())
    }

    #[test]
    fn multivariate_samples_cover_the_cartesian_grid() -> Result<(), Box<dyn Error>> {
        let config = MultivariateTutorialConfig {
            bits: 3,
            max_bond_dim: 32,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };
        let interleaved_grid = build_multivariate_grid(&config, UnfoldingScheme::Interleaved)?;
        let grouped_grid = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;
        let (interleaved, _, _) =
            build_multivariate_qtt(&interleaved_grid, test_multivariate_target, &config)?;
        let (grouped, _, _) =
            build_multivariate_qtt(&grouped_grid, test_multivariate_target, &config)?;

        let samples = collect_samples(&interleaved, &grouped, test_multivariate_target)?;

        assert_eq!(samples.len(), 64);
        let first = samples.first().expect("samples are not empty");
        assert_eq!(first.x_index, 1);
        assert_eq!(first.y_index, 1);
        assert!((first.exact - 0.001).abs() < 1e-15);
        assert!(first.interleaved_abs_error < 1e-9);
        assert!(first.grouped_abs_error < 1e-9);

        Ok(())
    }

    #[test]
    fn dense_qtt_batch_evaluation_matches_direct_evaluation() -> Result<(), Box<dyn Error>> {
        let config = MultivariateTutorialConfig {
            bits: 3,
            max_bond_dim: 32,
            ..DEFAULT_MULTIVARIATE_CONFIG
        };
        let grid = build_multivariate_grid(&config, UnfoldingScheme::Grouped)?;
        let (qtt, _, _) = build_multivariate_qtt(&grid, test_multivariate_target, &config)?;

        let dense_values = evaluate_dense_qtt(&qtt)?;

        for (x_index, y_index) in [(1, 1), (2, 5), (8, 8)] {
            let flat_index = (x_index - 1) * point_count(&config) + (y_index - 1);
            let direct = qtt.evaluate(&[x_index as i64, y_index as i64])?;
            assert!((dense_values[flat_index] - direct).abs() < 1e-12);
        }

        Ok(())
    }

    #[test]
    fn multivariate_bond_rows_pair_layout_profiles() {
        let rows = collect_bond_dims(&[2, 3, 2], &[2, 4]);
        assert_eq!(
            rows,
            vec![
                (1, Some(2), Some(2)),
                (2, Some(3), Some(4)),
                (3, Some(2), None)
            ]
        );
    }
}
