//! Build QTTs for a two-dimensional function and compare quantics layouts.
//!
//! This tutorial builds the same 2D target function twice, once with
//! interleaved quantics bits and once with grouped quantics bits, then exports
//! dense samples and bond dimensions for Julia plotting.

use std::error::Error;
use std::fs;
use std::io;
use std::path::Path;

use tensor4all_quanticstci::{
    quanticscrossinterpolate, DiscretizedGrid, QtciOptions, UnfoldingScheme,
};
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_multivariate_common::{
    collect_bond_dims, collect_samples, print_summary, write_bond_dims_csv, write_samples_csv,
    LayoutSummary, MultivariateTutorialConfig, DEFAULT_MULTIVARIATE_CONFIG,
};

const BITS_ENV: &str = "QTT_MULTIVARIATE_BITS";
const MAXBONDDIM_ENV: &str = "QTT_MULTIVARIATE_MAXBONDDIM";
const MAXITER_ENV: &str = "QTT_MULTIVARIATE_MAXITER";

fn multivariate_target(x: f64, y: f64) -> f64 {
    x.cos() * y.cos() * x
}

fn parse_env_usize(name: &str) -> Result<Option<usize>, Box<dyn Error>> {
    let Ok(value) = std::env::var(name) else {
        return Ok(None);
    };

    let parsed = value.parse::<usize>().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be a non-negative integer, got {value:?}: {err}"),
        )
    })?;

    Ok(Some(parsed))
}

fn config_from_env() -> Result<MultivariateTutorialConfig, Box<dyn Error>> {
    let mut config = DEFAULT_MULTIVARIATE_CONFIG;

    // Keep the checked-in tutorial small, but make larger local experiments easy.
    if let Some(bits) = parse_env_usize(BITS_ENV)? {
        config.bits = bits;
    }

    // Higher resolutions or harder functions may need a larger rank cap.
    if let Some(max_bond_dim) = parse_env_usize(MAXBONDDIM_ENV)? {
        config.max_bond_dim = max_bond_dim;
    }

    // Tensor4all-rs currently runs the requested number of QTCI sweeps.
    if let Some(maxiter) = parse_env_usize(MAXITER_ENV)? {
        config.maxiter = maxiter;
    }

    Ok(config)
}

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = config_from_env()?;

    fs::create_dir_all(&data_dir)?;

    // The unfolding scheme belongs to the grid: it defines the quantics index order.
    let interleaved_grid = DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&["x", "y"])
        .with_bounds(config.lower_bound, config.upper_bound)
        .with_unfolding_scheme(UnfoldingScheme::Interleaved)
        .include_endpoint(config.include_endpoint)
        .build()?;

    let grouped_grid = DiscretizedGrid::builder(&[config.bits, config.bits])
        .with_variable_names(&["x", "y"])
        .with_bounds(config.lower_bound, config.upper_bound)
        .with_unfolding_scheme(UnfoldingScheme::Grouped)
        .include_endpoint(config.include_endpoint)
        .build()?;

    // QTCI options control accuracy, rank cap, initialization, and sweep count.
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_verbosity(0);

    // `quanticscrossinterpolate` receives physical coordinates from the grid.
    let target = |coords: &[f64]| -> f64 { multivariate_target(coords[0], coords[1]) };
    let npoints = 1usize << config.bits;
    // Grid indices are 0-based.
    let initial_pivots = vec![
        vec![0, 0],
        vec![npoints / 2 - 1, npoints / 2 - 1],
        vec![npoints - 1, npoints - 1],
    ];

    // Build two QTT approximations of the same physical function using different layouts.
    let (interleaved, interleaved_ranks, interleaved_errors) = quanticscrossinterpolate(
        &interleaved_grid,
        target,
        Some(initial_pivots.clone()),
        options.clone(),
    )?;

    let (grouped, grouped_ranks, grouped_errors) =
        quanticscrossinterpolate(&grouped_grid, target, Some(initial_pivots), options)?;

    // Reconstruct both QTTs on the full Cartesian grid for heatmaps and error plots.
    let samples = collect_samples(&interleaved, &grouped, multivariate_target)?;

    // `link_dims` gives the bond dimensions of each QTT layout.
    let bond_dims = collect_bond_dims(&interleaved.link_dims(), &grouped.link_dims());

    print_summary(
        LayoutSummary {
            name: "interleaved",
            qtt: &interleaved,
            ranks: &interleaved_ranks,
            errors: &interleaved_errors,
        },
        LayoutSummary {
            name: "grouped",
            qtt: &grouped,
            ranks: &grouped_ranks,
            errors: &grouped_errors,
        },
        &samples,
        &config,
    );

    write_samples_csv(&data_dir.join("qtt_multivariate_samples.csv"), &samples)?;
    write_bond_dims_csv(&data_dir.join("qtt_multivariate_bond_dims.csv"), &bond_dims)?;

    println!(
        "wrote {}",
        data_dir.join("qtt_multivariate_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_multivariate_bond_dims.csv").display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_multivariate_plot.jl")
            .display()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_function_lives_with_the_tutorial_binary() {
        let target_path = std::any::type_name_of_val(&multivariate_target);
        assert!(
            target_path.contains("qtt_multivariate::multivariate_target"),
            "target path was {target_path}"
        );
    }
}
