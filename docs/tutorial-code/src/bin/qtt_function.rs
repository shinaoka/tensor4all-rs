//! Build a QTT for a scalar `f64 -> f64` function and export sample data for
//! Julia plotting.
//!
//! This binary keeps the mathematical construction close to the top of the
//! file.  Everything related to printing, CSV export, and sample formatting
//! lives in `qtt_function_utils.rs`.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme};
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_function_utils::{
    collect_samples, print_summary, write_bond_dims_csv, write_samples_csv,
};

/// The discrete grid has `2^BITS` sample points.
const BITS: usize = 7;
/// Total number of grid points used by the QTT construction.
const NPOINTS: usize = 1 << BITS;
/// Target accuracy for the quantics cross interpolation.
const TOLERANCE: f64 = 1e-12;
/// Upper bound on the internal bond dimension.
const MAX_BOND_DIM: usize = 32;
/// How many sample rows we print to the terminal for a quick sanity check.
const SAMPLE_PRINT_COUNT: usize = 8;

type QttDemoOutput = (
    tensor4all_quanticstci::QuanticsTensorCI2<f64>,
    Vec<usize>,
    Vec<f64>,
);

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();

    // The Julia script writes plots to docs/plots. Rust only needs the data
    // directory, and it can be overridden with TENSOR4ALL_DATA_DIR for smoke
    // tests that should not touch docs/data.
    fs::create_dir_all(&data_dir)?;

    let demo_label = "QTT function demo";
    let (qtci, ranks, errors) = build_qtt_demo(target_function)?;
    let tt = qtci.tensor_train();
    let samples = collect_samples(&qtci, NPOINTS, target_function)?;
    let bond_dims = qtci.link_dims();
    let max_abs_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);

    print_summary(
        &qtci,
        &tt,
        &ranks,
        &errors,
        &samples,
        demo_label,
        BITS,
        NPOINTS,
        SAMPLE_PRINT_COUNT,
        max_abs_error,
    );
    write_samples_csv(&data_dir.join("qtt_function_samples.csv"), &samples)?;
    write_bond_dims_csv(&data_dir.join("qtt_function_bond_dims.csv"), &bond_dims)?;

    println!(
        "wrote {}",
        data_dir.join("qtt_function_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_function_bond_dims.csv").display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_function_plot.jl")
            .display()
    );

    Ok(())
}

fn build_qtt_demo<F>(target_fn: F) -> Result<QttDemoOutput, Box<dyn Error>>
where
    F: Fn(f64) -> f64 + 'static,
{
    // Quantics QTTs work on discrete grids.  Here we ask for one dimension with
    // 2^BITS points; the library internally unfolds that into binary sites.
    let sizes = [NPOINTS];

    // The options below mirror the notebook-style workflow: a tight tolerance,
    // a reasonable maximum bond dimension, and deterministic initial pivots so
    // tutorial smoke tests are reproducible.
    let options = QtciOptions::default()
        .with_tolerance(TOLERANCE)
        .with_max_bond_dim(MAX_BOND_DIM)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    // The callback is the actual function we want to compress. The discrete API
    // uses 0-based integer grid indices, so we convert them to x in [0, 1).
    let f = move |idx: &[usize]| -> f64 {
        let x = idx[0] as f64 / NPOINTS as f64;
        target_fn(x)
    };

    let initial_pivots = vec![vec![1], vec![NPOINTS / 2 - 1], vec![NPOINTS - 1]];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        f,
        Some(initial_pivots),
        options,
    )?)
}

/// Default target function used by the demo.
///
/// Change this body to approximate a different function, or replace the whole
/// function with another `f64 -> f64` definition.
fn target_function(x: f64) -> f64 {
    x.cosh()
}
