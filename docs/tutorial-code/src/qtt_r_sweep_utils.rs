//! Utility helpers for the QTT `R` sweep demo.
//!
//! This file keeps the output handling away from the numerical code so the
//! main binary stays focused on the library workflow.
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_quanticstci::QuanticsTensorCI2;

/// One row in the per-grid-point sample table.
#[derive(Debug, Clone)]
pub struct SweepSample {
    pub r: usize,
    pub npoints: usize,
    pub index: usize,
    pub x: f64,
    pub exact: f64,
    pub qtt: f64,
    pub abs_error: f64,
}

/// One row in the per-R summary table.
#[derive(Debug, Clone)]
pub struct SweepStats {
    pub r: usize,
    pub npoints: usize,
    pub build_time_sec: f64,
    pub mean_abs_error: f64,
    pub max_abs_error: f64,
    pub rank: usize,
}

/// Convert a grid index into `x` in [0, 1).
pub fn discrete_index_to_unit_interval(index: i64, npoints: usize) -> f64 {
    index as f64 / npoints as f64
}

/// Re-evaluate one QTT on all grid points and collect the error against the
/// analytic target function.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples<F>(
    r: usize,
    npoints: usize,
    qtci: &QuanticsTensorCI2<f64>,
    target_fn: F,
) -> Result<Vec<SweepSample>, Box<dyn Error>>
where
    F: Fn(f64) -> f64,
{
    let mut rows = Vec::with_capacity(npoints);

    for index in 0..npoints {
        let x = discrete_index_to_unit_interval(index as i64, npoints);
        let exact = target_fn(x);
        // Library call: evaluate one grid point from the QTT approximation
        // (0-based index).
        let qtt = qtci.evaluate(&[index])?;
        rows.push(SweepSample {
            r,
            npoints,
            index: index + 1,
            x,
            exact,
            qtt,
            abs_error: (exact - qtt).abs(),
        });
    }

    Ok(rows)
}

/// Mean absolute error over all rows.
pub fn mean_abs_error(samples: &[SweepSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    samples.iter().map(|row| row.abs_error).sum::<f64>() / samples.len() as f64
}

/// Maximum absolute error over all rows.
pub fn max_abs_error(samples: &[SweepSample]) -> f64 {
    samples
        .iter()
        .map(|row| row.abs_error)
        .fold(0.0_f64, f64::max)
}

/// Print a short per-R summary in the terminal.
#[allow(clippy::too_many_arguments)]
pub fn print_sweep_summary(
    r: usize,
    npoints: usize,
    rank: usize,
    build_time_sec: f64,
    mean_abs_error: f64,
    max_abs_error: f64,
    ranks: &[usize],
    errors: &[f64],
    preview_count: usize,
) {
    println!(
        "R = {r}, N = {npoints}, rank = {rank}, build_time = {:.6} s, mean_abs_error = {:.3e}, max_abs_error = {:.3e}",
        build_time_sec, mean_abs_error, max_abs_error
    );
    println!(
        "  rank history = len {}, preview {}",
        ranks.len(),
        preview_usize(ranks, preview_count)
    );
    println!(
        "  error history = len {}, preview {}",
        errors.len(),
        preview_f64(errors, preview_count)
    );
    println!();
}

fn preview_usize(values: &[usize], n: usize) -> String {
    let shown = values
        .iter()
        .take(n)
        .map(|v| v.to_string())
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
        .map(|v| format!("{:.3e}", v))
        .collect::<Vec<_>>();
    if values.len() > n {
        format!("[{} ...]", shown.join(", "))
    } else {
        format!("[{}]", shown.join(", "))
    }
}

/// Write the sample table to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(path: &Path, samples: &[SweepSample]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "r,npoints,index,x,exact,qtt,abs_error")?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{},{:.16},{:.16},{:.16},{:.16}",
            sample.r,
            sample.npoints,
            sample.index,
            sample.x,
            sample.exact,
            sample.qtt,
            sample.abs_error
        )?;
    }

    println!("wrote {}", path.display());
    Ok(())
}

/// Write the per-R summary table to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_stats_csv(path: &Path, stats: &[SweepStats]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "r,npoints,build_time_sec,mean_abs_error,max_abs_error,rank"
    )?;
    for row in stats {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{}",
            row.r, row.npoints, row.build_time_sec, row.mean_abs_error, row.max_abs_error, row.rank
        )?;
    }

    println!("wrote {}", path.display());
    Ok(())
}
