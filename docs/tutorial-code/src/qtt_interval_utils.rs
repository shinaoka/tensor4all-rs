//! Helper utilities for the `qtt_interval` demo.
//!
//! This file holds the support code so the main QTT file stays focused on the
//! actual QTT construction.  The helpers below:
//! - build rows for CSV export,
//! - print a readable summary to the terminal,
//! - and keep the interval/grid bookkeeping in one place.
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_quanticstci::{DiscretizedGrid, QuanticsTensorCI2};
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};

/// One line in the exported CSV table.
#[derive(Debug, Clone)]
pub struct SamplePoint {
    pub index: usize,
    pub x: f64,
    pub exact: f64,
    pub qtt: f64,
    pub abs_error: f64,
}

/// Evaluate the QTT at every grid point and compare it with the analytic target.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples<F>(
    qtci: &QuanticsTensorCI2<f64>,
    grid: &DiscretizedGrid,
    exact_fn: F,
) -> Result<Vec<SamplePoint>, Box<dyn Error>>
where
    F: Fn(f64) -> f64,
{
    // The grid gives us the physical coordinates.  We keep the 1-based grid
    // index alongside each sample so the user can compare Rust and Julia output.
    //
    // `grid_origcoords(0)` is a `DiscretizedGrid` API call.  It returns the
    // physical coordinates for the first dimension, which is exactly what we
    // want for this 1D tutorial.
    let coords = grid.grid_origcoords(0)?;
    let mut samples = Vec::with_capacity(coords.len());

    for (i, x) in coords.into_iter().enumerate() {
        let index = i + 1;
        let exact = exact_fn(x);

        // `evaluate(...)` is the QTT read-back step.  The input index is
        // 1-based, matching the convention used throughout tensor4all-rs.
        let qtt = qtci.evaluate(&[index as i64])?;

        samples.push(SamplePoint {
            index,
            x,
            exact,
            qtt,
            abs_error: (exact - qtt).abs(),
        });
    }

    Ok(samples)
}

/// Extract the bond-dimension profile from the QTT.
pub fn collect_bond_dims(qtci: &QuanticsTensorCI2<f64>) -> Vec<usize> {
    // `link_dims()` is the library call that exposes the bond dimensions
    // between adjacent TT cores.
    qtci.link_dims()
}

/// Print a compact but informative summary of the QTT to the terminal.
#[allow(clippy::too_many_arguments)]
pub fn print_summary(
    qtci: &QuanticsTensorCI2<f64>,
    tt: &SimpleTensorTrain<f64>,
    grid: &DiscretizedGrid,
    ranks: &[usize],
    errors: &[f64],
    samples: &[SamplePoint],
    demo_label: &str,
    bits: usize,
    sample_print_count: usize,
    max_abs_error: f64,
) {
    println!("{demo_label}");
    println!("bits = {}", bits);
    println!(
        "interval = [{:.3}, {:.3}]",
        grid.lower_bound()[0],
        grid.upper_bound()[0]
    );
    // `grid_step()` comes from `DiscretizedGrid` and shows the spacing between
    // adjacent physical sample points.
    println!("grid step = {:?}", grid.grid_step());
    println!("number of samples = {}", samples.len());
    println!("qtt length = {}", tt.len());
    // `site_dims()` and `link_dims()` come from the TT/QTT API and show the
    // core sizes and bond sizes of the internal representation.
    println!("site_dims = {:?}", tt.site_dims());
    println!("link_dims = {:?}", qtci.link_dims());
    println!("rank = {}", qtci.rank());
    println!(
        "rank history = len {}, preview {}",
        ranks.len(),
        preview_usize(ranks, sample_print_count)
    );
    println!(
        "error history = len {}, preview {}",
        errors.len(),
        preview_f64(errors, sample_print_count)
    );
    println!("max abs error = {:.3e}", max_abs_error);
    println!();

    // Show the shape of each TT core so the structure is visible without
    // having to inspect the raw tensor data.
    for (i, core) in tt.site_tensors().iter().enumerate() {
        println!(
            "core {}: left={}, site={}, right={}",
            i,
            core.left_dim(),
            core.site_dim(),
            core.right_dim()
        );
    }
    println!();

    // Print a few representative samples so the user can compare exact and QTT
    // values directly in the terminal.
    println!("first {} samples", sample_print_count.min(samples.len()));
    for sample in samples.iter().take(sample_print_count) {
        println!(
            "i = {:>3}, x = {:.6}, exact = {:.12}, qtt = {:.12}, abs_err = {:.3e}",
            sample.index, sample.x, sample.exact, sample.qtt, sample.abs_error
        );
    }
    println!();
}

/// Write the point-wise comparison table used by Julia for plotting.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(path: &Path, samples: &[SamplePoint]) -> Result<(), Box<dyn Error>> {
    // Plain CSV output: one row per grid point so Julia can recreate the figure.
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "index,x,exact,qtt,abs_error")?;
    for sample in samples {
        writeln!(
            w,
            "{},{:.16},{:.16},{:.16},{:.16}",
            sample.index, sample.x, sample.exact, sample.qtt, sample.abs_error
        )?;
    }

    Ok(())
}

/// Write the bond-dimension profile used by the Julia line plot.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(path: &Path, bond_dims: &[usize]) -> Result<(), Box<dyn Error>> {
    // Plain CSV output: one row per bond in the QTT.
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,bond_dim")?;
    for (i, &bond_dim) in bond_dims.iter().enumerate() {
        writeln!(w, "{},{}", i + 1, bond_dim)?;
    }

    Ok(())
}

fn preview_usize(values: &[usize], n: usize) -> String {
    let shown = values
        .iter()
        .take(n)
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    if values.len() > n {
        format!("[{} ...]", shown.join(", "))
    } else {
        format!("[{}]", shown.join(", "))
    }
}

fn preview_f64(values: &[f64], n: usize) -> String {
    let shown = values
        .iter()
        .take(n)
        .map(|value| format!("{:.3e}", value))
        .collect::<Vec<_>>();
    if values.len() > n {
        format!("[{} ...]", shown.join(", "))
    } else {
        format!("[{}]", shown.join(", "))
    }
}
