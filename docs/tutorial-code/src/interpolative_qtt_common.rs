//! Shared setup for the Interpolative QTT tutorial.
//!
//! The helper keeps tutorial math, QTT construction, dense sampling, and CSV
//! export in one place so the mdBook page and tests exercise the same workflow.

use std::error::Error;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_interpolativeqtt::{
    interpolate_multi_scale, interpolate_multi_scale_nd, interpolate_single_scale,
    AbstractTensorTrain, InterpolativeQttOptions, SimpleTensorTrain,
};
use tensor4all_quanticstci::{DiscretizedGrid, UnfoldingScheme};

/// Configuration for the Interpolative QTT tutorial.
#[derive(Debug, Clone, Copy)]
pub struct InterpolativeQttTutorialConfig {
    pub bits_1d: usize,
    pub bits_2d: usize,
    pub single_scale_degree: usize,
    pub multi_scale_degree_1d: usize,
    pub multi_scale_degree_2d: usize,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub epsilon: f64,
}

/// Default configuration used by the tutorial binary and verification tests.
pub const DEFAULT_INTERPOLATIVE_QTT_CONFIG: InterpolativeQttTutorialConfig =
    InterpolativeQttTutorialConfig {
        bits_1d: 7,
        bits_2d: 6,
        single_scale_degree: 12,
        multi_scale_degree_1d: 20,
        multi_scale_degree_2d: 12,
        tolerance: 1.0e-12,
        max_bond_dim: 256,
        lower_bound: -1.0,
        upper_bound: 1.0,
        epsilon: 0.2,
    };

/// One row in the one-dimensional sample table.
#[derive(Debug, Clone)]
pub struct InterpolativeQtt1dSample {
    pub case_name: &'static str,
    pub index: usize,
    pub x: f64,
    pub exact: f64,
    pub qtt: f64,
    pub abs_error: f64,
}

/// One row in the two-dimensional sample table.
#[derive(Debug, Clone)]
pub struct InterpolativeQtt2dSample {
    pub x_index: usize,
    pub y_index: usize,
    pub x: f64,
    pub y: f64,
    pub exact: f64,
    pub qtt: f64,
    pub abs_error: f64,
}

/// One row in the bond-dimension table.
pub type InterpolativeBondDimRow = (usize, Option<usize>, Option<usize>, Option<usize>);

fn invalid_interpolative_input(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

/// Smooth single-scale target used in the first section.
pub fn smooth_target(x: f64) -> f64 {
    (-x * x).exp()
}

/// Softened one-dimensional inverse-square target used for multiscale QTT.
pub fn softened_inverse_square_1d(x: f64, epsilon: f64) -> f64 {
    1.0 / (x.abs() + epsilon).powi(2)
}

/// Softened radial inverse-square target used for multidimensional multiscale QTT.
pub fn softened_inverse_square_2d(coords: &[f64], epsilon: f64) -> f64 {
    let r2 = coords[0] * coords[0] + coords[1] * coords[1];
    1.0 / (r2 + epsilon * epsilon)
}

/// Options shared by the tutorial constructors.
pub fn interpolation_options(config: &InterpolativeQttTutorialConfig) -> InterpolativeQttOptions {
    InterpolativeQttOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
}

/// Build the smooth one-dimensional single-scale QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_single_scale_smooth_qtt(
    config: &InterpolativeQttTutorialConfig,
) -> Result<SimpleTensorTrain<f64>, Box<dyn Error>> {
    let options = interpolation_options(config);
    Ok(interpolate_single_scale(
        smooth_target,
        config.lower_bound,
        config.upper_bound,
        config.bits_1d,
        config.single_scale_degree,
        &options,
    )?)
}

/// Build the one-dimensional multiscale inverse-square QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multi_scale_inverse_square_1d_qtt(
    config: &InterpolativeQttTutorialConfig,
) -> Result<SimpleTensorTrain<f64>, Box<dyn Error>> {
    let options = interpolation_options(config);
    let epsilon = config.epsilon;
    Ok(interpolate_multi_scale(
        move |x| softened_inverse_square_1d(x, epsilon),
        config.lower_bound,
        config.upper_bound,
        config.bits_1d,
        config.multi_scale_degree_1d,
        &[0.0],
        &options,
    )?)
}

/// Build the two-dimensional multiscale radial inverse-square QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_multi_scale_inverse_square_2d_qtt(
    config: &InterpolativeQttTutorialConfig,
) -> Result<SimpleTensorTrain<f64>, Box<dyn Error>> {
    let options = interpolation_options(config);
    let epsilon = config.epsilon;
    let lower = [config.lower_bound, config.lower_bound];
    let upper = [config.upper_bound, config.upper_bound];
    let cusp_locations = vec![vec![0.0, 0.0]];

    Ok(interpolate_multi_scale_nd(
        move |coords| softened_inverse_square_2d(coords, epsilon),
        &lower,
        &upper,
        config.bits_2d,
        config.multi_scale_degree_2d,
        &cusp_locations,
        &options,
    )?)
}

fn grid_1d(bits: usize, lower: f64, upper: f64) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[bits])
        .with_lower_bound(&[lower])
        .with_upper_bound(&[upper])
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?)
}

fn grid_nd(bits: usize, lower: &[f64], upper: &[f64]) -> Result<DiscretizedGrid, Box<dyn Error>> {
    let bit_depths = vec![bits; lower.len()];
    Ok(DiscretizedGrid::builder(&bit_depths)
        .with_lower_bound(lower)
        .with_upper_bound(upper)
        .with_unfolding_scheme(UnfoldingScheme::Fused)
        .build()?)
}

fn quantics_to_tt_indices(quantics: &[usize]) -> Result<Vec<usize>, Box<dyn Error>> {
    Ok(quantics.to_vec())
}

/// Collect full one-dimensional grid samples for plotting and verification.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_1d_samples<F>(
    case_name: &'static str,
    tt: &SimpleTensorTrain<f64>,
    bits: usize,
    lower: f64,
    upper: f64,
    exact_fn: F,
) -> Result<Vec<InterpolativeQtt1dSample>, Box<dyn Error>>
where
    F: Fn(f64) -> f64,
{
    let grid = grid_1d(bits, lower, upper)?;
    let npoints = 1usize << bits;
    let mut samples = Vec::with_capacity(npoints);

    for index in 0..npoints {
        let grid_index = [index];
        let quantics = grid.grididx_to_quantics(&grid_index)?;
        let tt_indices = quantics_to_tt_indices(&quantics)?;
        let coords = grid.grididx_to_origcoord(&grid_index)?;
        let x = coords[0];
        let exact = exact_fn(x);
        let qtt = tt.evaluate(&tt_indices)?;
        samples.push(InterpolativeQtt1dSample {
            case_name,
            // 1-based display index; grid indices themselves are 0-based.
            index: index + 1,
            x,
            exact,
            qtt,
            abs_error: (exact - qtt).abs(),
        });
    }

    Ok(samples)
}

/// Collect full two-dimensional grid samples for plotting and verification.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_2d_samples<F>(
    tt: &SimpleTensorTrain<f64>,
    bits: usize,
    lower: &[f64],
    upper: &[f64],
    exact_fn: F,
) -> Result<Vec<InterpolativeQtt2dSample>, Box<dyn Error>>
where
    F: Fn(&[f64]) -> f64,
{
    if lower.len() != 2 || upper.len() != 2 {
        return Err(invalid_interpolative_input(
            "the interpolative QTT tutorial samples a two-dimensional grid",
        ));
    }

    let grid = grid_nd(bits, lower, upper)?;
    let npoints = 1usize << bits;
    let mut samples = Vec::with_capacity(npoints * npoints);

    for x_index in 0..npoints {
        for y_index in 0..npoints {
            let grid_index = [x_index, y_index];
            let quantics = grid.grididx_to_quantics(&grid_index)?;
            let tt_indices = quantics_to_tt_indices(&quantics)?;
            let coords = grid.grididx_to_origcoord(&grid_index)?;
            let exact = exact_fn(&coords);
            let qtt = tt.evaluate(&tt_indices)?;
            samples.push(InterpolativeQtt2dSample {
                // 1-based display indices; grid indices themselves are 0-based.
                x_index: x_index + 1,
                y_index: y_index + 1,
                x: coords[0],
                y: coords[1],
                exact,
                qtt,
                abs_error: (exact - qtt).abs(),
            });
        }
    }

    Ok(samples)
}

/// Pair the three tutorial bond-dimension profiles into CSV rows.
pub fn collect_bond_dims(
    single_scale_1d: &[usize],
    multi_scale_1d: &[usize],
    multi_scale_2d: &[usize],
) -> Vec<InterpolativeBondDimRow> {
    let row_count = single_scale_1d
        .len()
        .max(multi_scale_1d.len())
        .max(multi_scale_2d.len());
    (0..row_count)
        .map(|i| {
            (
                i + 1,
                single_scale_1d.get(i).copied(),
                multi_scale_1d.get(i).copied(),
                multi_scale_2d.get(i).copied(),
            )
        })
        .collect()
}

/// Return the largest absolute error in a one-dimensional sample set.
pub fn max_abs_error_1d(samples: &[InterpolativeQtt1dSample]) -> f64 {
    samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max)
}

/// Return the largest absolute error in a two-dimensional sample set.
pub fn max_abs_error_2d(samples: &[InterpolativeQtt2dSample]) -> f64 {
    samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max)
}

/// Write one-dimensional samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_1d_samples_csv(
    path: &Path,
    samples: &[InterpolativeQtt1dSample],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "case,index,x,exact,qtt,abs_error")?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{:.16}",
            sample.case_name, sample.index, sample.x, sample.exact, sample.qtt, sample.abs_error
        )?;
    }

    Ok(())
}

/// Write two-dimensional samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_2d_samples_csv(
    path: &Path,
    samples: &[InterpolativeQtt2dSample],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "x_index,y_index,x,y,exact,qtt,abs_error")?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.x_index,
            sample.y_index,
            sample.x,
            sample.y,
            sample.exact,
            sample.qtt,
            sample.abs_error
        )?;
    }

    Ok(())
}

/// Write bond dimensions to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(
    path: &Path,
    bond_dims: &[InterpolativeBondDimRow],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "bond_index,single_scale_1d,multi_scale_1d,multi_scale_2d"
    )?;
    for (bond_index, single, multi_1d, multi_2d) in bond_dims {
        writeln!(
            w,
            "{},{},{},{}",
            bond_index,
            single.map_or_else(String::new, |value| value.to_string()),
            multi_1d.map_or_else(String::new, |value| value.to_string()),
            multi_2d.map_or_else(String::new, |value| value.to_string())
        )?;
    }

    Ok(())
}

/// Print a compact terminal summary for local tutorial runs.
pub fn print_summary(
    config: &InterpolativeQttTutorialConfig,
    smooth: &SimpleTensorTrain<f64>,
    multi_1d: &SimpleTensorTrain<f64>,
    multi_2d: &SimpleTensorTrain<f64>,
    smooth_max_error: f64,
    multi_1d_max_error: f64,
    multi_2d_max_error: f64,
) {
    println!("Interpolative QTT tutorial");
    println!(
        "  1D grid: 2^{} points, 2D grid: 2^{} x 2^{} points",
        config.bits_1d, config.bits_2d, config.bits_2d
    );
    println!(
        "  degrees: single-scale {}, multiscale 1D {}, multiscale 2D {}",
        config.single_scale_degree, config.multi_scale_degree_1d, config.multi_scale_degree_2d
    );
    println!(
        "  max abs errors: smooth {:.3e}, inverse-square 1D {:.3e}, inverse-square 2D {:.3e}",
        smooth_max_error, multi_1d_max_error, multi_2d_max_error
    );
    println!(
        "  ranks: smooth {}, inverse-square 1D {}, inverse-square 2D {}",
        smooth.rank(),
        multi_1d.rank(),
        multi_2d.rank()
    );
}
