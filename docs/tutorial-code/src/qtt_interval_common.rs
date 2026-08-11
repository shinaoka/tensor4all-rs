//! Shared setup for the interval and integral tutorials.
//!
//! These helpers deliberately stay specific to the `x^2` tutorial on
//! `[-1, 2]`. They keep the binaries small without turning the examples into a
//! generic framework.
use std::error::Error;

use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, QuanticsTensorCI2, UnfoldingScheme,
};

/// Configuration for the fixed interval tutorial family.
#[derive(Debug, Clone, Copy)]
pub struct IntervalTutorialConfig {
    pub bits: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub include_endpoint: bool,
}

/// Default interval used throughout the interval and integral tutorials.
pub const DEFAULT_INTERVAL_CONFIG: IntervalTutorialConfig = IntervalTutorialConfig {
    bits: 7,
    lower_bound: -1.0,
    upper_bound: 2.0,
    include_endpoint: true,
};

const TOLERANCE: f64 = 1e-12;
const MAX_BOND_DIM: usize = 32;

/// QTT construction result returned by the interval helper.
pub type IntervalQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);

/// Target function for the fixed interval tutorial.
pub fn interval_target(x: f64) -> f64 {
    x * x //+ (3.0*x).sin() + (10.0*x).cos()
}

/// Antiderivative of the tutorial target function `x^2`.
pub fn interval_target_antiderivative(x: f64) -> f64 {
    x.powi(3) / 3.0 //- (3.0*x).cos()/3.0 + (10.0 *x).sin()/10.0
}

/// Exact integral of `x^2` over the configured interval.
pub fn exact_integral(config: &IntervalTutorialConfig) -> f64 {
    interval_target_antiderivative(config.upper_bound)
        - interval_target_antiderivative(config.lower_bound)
}

/// Build the one-dimensional physical grid used by the interval tutorials.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_interval_grid(
    config: &IntervalTutorialConfig,
) -> Result<DiscretizedGrid, Box<dyn Error>> {
    Ok(DiscretizedGrid::builder(&[config.bits])
        .with_variable_names(&["x"])
        .with_bounds(config.lower_bound, config.upper_bound)
        .include_endpoint(config.include_endpoint)
        .build()?)
}

/// Build the QTT for a scalar function on the fixed interval grid.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_interval_qtt<F>(
    grid: &DiscretizedGrid,
    target_fn: F,
    config: &IntervalTutorialConfig,
) -> Result<IntervalQttOutput, Box<dyn Error>>
where
    F: Fn(f64) -> f64 + 'static,
{
    let options = QtciOptions::default()
        .with_tolerance(TOLERANCE)
        .with_max_bond_dim(MAX_BOND_DIM)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    let f = move |coords: &[f64]| -> f64 { target_fn(coords[0]) };
    let npoints = 1_i64 << config.bits;
    let initial_pivots = vec![vec![1], vec![npoints / 2], vec![npoints]];

    Ok(quanticscrossinterpolate(
        grid,
        f,
        Some(initial_pivots),
        options,
    )?)
}
