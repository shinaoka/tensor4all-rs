//! Sweep QTT accuracy and build time over different quantics resolutions.
//!
//! For each `R` in a small range we:
//! - build a QTT for `sin(30x)` on `2^R` grid points
//! - measure only the QTT construction time
//! - sample the approximation again on the same grid
//! - export the results for Julia plots
//!
//! The Rust side stays focused on the Tensor4all workflow. Julia handles all
//! plotting so the example remains beginner-friendly.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme};
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_r_sweep_utils::{
    collect_samples, max_abs_error, mean_abs_error, print_sweep_summary, write_samples_csv,
    write_stats_csv, SweepStats,
};

/// Sweep lower bound for the quantics depth.
const R_MIN: usize = 2;
/// Sweep upper bound for the quantics depth.
const R_MAX: usize = 15;
/// Target interpolation tolerance.
const TOLERANCE: f64 = 1e-12;
/// Upper bound on the internal bond dimensions during construction.
const MAX_BOND_DIM: usize = 32;
/// How many rank/error entries to preview in the terminal.
const PREVIEW_COUNT: usize = 6;
/// Frequency in the target function `sin(30x)`.
const OMEGA: f64 = 30.0;

type QttDemoOutput = (
    tensor4all_quanticstci::QuanticsTensorCI2<f64>,
    Vec<usize>,
    Vec<f64>,
);

/// Bundle of file names used by this experiment.
struct OutputSpec {
    samples_csv_name: &'static str,
    stats_csv_name: &'static str,
    plot_script_name: &'static str,
}

impl OutputSpec {
    fn new() -> Self {
        Self {
            samples_csv_name: "qtt_r_sweep_samples.csv",
            stats_csv_name: "qtt_r_sweep_stats.csv",
            plot_script_name: "qtt_r_sweep_plot.jl",
        }
    }

    fn samples_csv_path(&self, data_dir: &Path) -> PathBuf {
        data_dir.join(self.samples_csv_name)
    }

    fn stats_csv_path(&self, data_dir: &Path) -> PathBuf {
        data_dir.join(self.stats_csv_name)
    }

    fn plot_script_path(&self, project_root: &Path) -> PathBuf {
        project_root
            .join("docs")
            .join("plotting")
            .join(self.plot_script_name)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    fs::create_dir_all(&data_dir)?;
    let output = OutputSpec::new();

    let mut all_samples = Vec::new();
    let mut stats_rows = Vec::new();

    for r in R_MIN..=R_MAX {
        let npoints = 1usize << r;

        // Time only the QTT construction itself.
        let start = Instant::now();
        let (qtci, ranks, errors) = build_qtt_from_function(npoints, target_function)?;
        let build_time_sec = start.elapsed().as_secs_f64();

        let samples = collect_samples(r, npoints, &qtci, target_function)?;
        let mean_error = mean_abs_error(&samples);
        let max_error = max_abs_error(&samples);
        let rank = qtci.rank();

        print_sweep_summary(
            r,
            npoints,
            rank,
            build_time_sec,
            mean_error,
            max_error,
            &ranks,
            &errors,
            PREVIEW_COUNT,
        );

        all_samples.extend(samples);
        stats_rows.push(SweepStats {
            r,
            npoints,
            build_time_sec,
            mean_abs_error: mean_error,
            max_abs_error: max_error,
            rank,
        });
    }

    write_samples_csv(&output.samples_csv_path(&data_dir), &all_samples)?;
    write_stats_csv(&output.stats_csv_path(&data_dir), &stats_rows)?;
    println!(
        "next: run the Julia plotting script at {}",
        output.plot_script_path(project_root).display()
    );

    Ok(())
}

/// Build a QTT approximation for a scalar function on the unit interval.
///
/// The library function `quanticscrossinterpolate_discrete(...)` expects a
/// callback on discrete grid indices, so this helper performs the index-to-x
/// conversion first.
fn build_qtt_from_function<F>(npoints: usize, target_fn: F) -> Result<QttDemoOutput, Box<dyn Error>>
where
    F: Fn(f64) -> f64 + 'static,
{
    let sizes = [npoints];
    let options = QtciOptions::default()
        .with_tolerance(TOLERANCE)
        .with_max_bond_dim(MAX_BOND_DIM)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    // Library callback: the interpolator asks for discrete grid values
    // (0-based indices).
    let callback = move |idx: &[usize]| -> f64 {
        let x = idx[0] as f64 / npoints as f64;
        target_fn(x)
    };

    let initial_pivots = vec![vec![1], vec![npoints / 2 - 1], vec![npoints - 1]];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        callback,
        Some(initial_pivots),
        options,
    )?)
}

/// Target function for the sweep.
///
/// This is plain scalar math; the tensor library only sees the callback value
/// on each discrete grid point.
fn target_function(x: f64) -> f64 {
    (OMEGA * x).sin()
}
