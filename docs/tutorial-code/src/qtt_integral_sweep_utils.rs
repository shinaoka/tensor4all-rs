//! Output helpers for the integral convergence sweep.
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// One row in the integral convergence table.
#[derive(Debug, Clone)]
pub struct IntegralSweepRow {
    pub r: usize,
    pub npoints: usize,
    pub integral: f64,
    pub exact_integral: f64,
    pub abs_error: f64,
    pub rank: usize,
}

/// Print one compact terminal row for the sweep.
pub fn print_sweep_row(row: &IntegralSweepRow) {
    println!(
        "R = {}, N = {}, rank = {}, integral = {:.12}, abs_error = {:.3e}",
        row.r, row.npoints, row.rank, row.integral, row.abs_error
    );
}

/// Write the integral convergence table used by Julia for plotting.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_sweep_csv(path: &Path, rows: &[IntegralSweepRow]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "r,npoints,integral,exact_integral,abs_error,rank")?;
    for row in rows {
        writeln!(
            w,
            "{},{},{:.16},{:.16},{:.16},{}",
            row.r, row.npoints, row.integral, row.exact_integral, row.abs_error, row.rank
        )?;
    }

    Ok(())
}
