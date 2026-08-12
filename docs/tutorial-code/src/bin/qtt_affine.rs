//! Build and apply a 2D affine pullback operator on a fused quantics grid.

use std::error::Error;
use std::fs;
use std::path::Path;

use tensor4all_core::TensorIndex;
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme};
use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
use tensor4all_treetn::{apply_linear_operator, tensor_train_to_treetn, ApplyOptions};

use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_affine_common::{
    collect_bond_dims, collect_operator_bond_dims, collect_samples, point_count, print_summary,
    source_function, write_bond_dims_csv, write_operator_bond_dims_csv, write_samples_csv,
    DEFAULT_AFFINE_CONFIG,
};

fn main() -> Result<(), Box<dyn Error>> {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = output_paths::data_dir();
    let config = DEFAULT_AFFINE_CONFIG;

    fs::create_dir_all(&data_dir)?;

    // Build the source QTT directly so the tutorial shows the Tensor4all API.
    let n = point_count(config.bits);
    let source_grid = vec![n, n];
    // QTCI options control accuracy, rank cap, initialization, and sweep count.
    let source_options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Fused)
        .with_verbosity(0);

    // `quanticscrossinterpolate_discrete(...)` builds the fused quantics source QTT.
    let source_callback =
        move |grid_idx: &[usize]| -> f64 { source_function(grid_idx[0], grid_idx[1], n) };
    // Grid indices are 0-based.
    let initial_pivots = vec![vec![0, 0], vec![n / 2 - 1, n / 2 - 1], vec![n - 1, n - 1]];
    let (source, _ranks, _errors) = quanticscrossinterpolate_discrete(
        &source_grid,
        source_callback,
        Some(initial_pivots),
        source_options,
    )?;

    // `AffineParams::from_integers(...)` encodes the affine map used in the tutorial.
    let affine_params = AffineParams::from_integers(vec![1, 0, 1, 1], vec![0, 0], 2, 2)?;

    // `affine_operator(...)` builds the MPO, and `transpose()` switches to the passive pullback.
    let periodic_operator = affine_operator(
        config.bits,
        &affine_params,
        &[BoundaryCondition::Periodic; 2],
    )?
    .transpose();

    let antiperiodic_operator = affine_operator(
        config.bits,
        &affine_params,
        &[BoundaryCondition::AntiPeriodic, BoundaryCondition::Periodic],
    )?
    .transpose();

    let open_operator =
        affine_operator(config.bits, &affine_params, &[BoundaryCondition::Open; 2])?.transpose();

    // The source QTT stores the inherent discrete grid that its quantics sites live on.
    // `inherent_grid()` exposes the grid needed to translate from physical points to quantics sites.
    let evaluation_grid = source
        .inherent_grid()
        .expect("source QTT uses an inherent discrete grid");

    // Convert the tensor train to a TreeTN, then align each MPO before applying it.
    let source_tt = source.tensor_train();
    // `tensor_train_to_treetn(...)` turns the QTT into the TreeTN state used by the operator API.
    let (state, _source_site_indices) = tensor_train_to_treetn(&source_tt)?;

    let mut periodic_aligned = periodic_operator.clone();
    // `align_to_state(...)` matches the MPO site ordering to the state before contraction.
    periodic_aligned.align_to_state(&state)?;

    // `apply_linear_operator(...)` performs the actual MPO-MPS contraction.
    let periodic = apply_linear_operator(&periodic_aligned, &state, ApplyOptions::naive())?;
    // `TensorIndex::external_indices(...)` gives the output site labels for evaluation.
    let periodic_sites = TensorIndex::external_indices(&periodic);

    let mut antiperiodic_aligned = antiperiodic_operator.clone();
    // The anti-periodic case flips sign only when the transformed u-coordinate wraps.
    antiperiodic_aligned.align_to_state(&state)?;

    let antiperiodic = apply_linear_operator(&antiperiodic_aligned, &state, ApplyOptions::naive())?;
    let antiperiodic_sites = TensorIndex::external_indices(&antiperiodic);

    let mut open_aligned = open_operator.clone();
    // Align the second boundary-condition variant the same way.
    open_aligned.align_to_state(&state)?;

    // Reuse the same linear-operator application path for the open-boundary result.
    let open = apply_linear_operator(&open_aligned, &state, ApplyOptions::naive())?;
    // The output site labels are needed to evaluate the transformed TreeTN on the grid.
    let open_sites = TensorIndex::external_indices(&open);

    // Sample both transformed states against the analytic reference on the full grid.
    let samples = collect_samples(
        &periodic,
        &periodic_sites,
        &antiperiodic,
        &antiperiodic_sites,
        &open,
        &open_sites,
        evaluation_grid,
        &config,
    )?;

    let bond_dims = collect_bond_dims(
        &state.link_dims(),
        &periodic.link_dims(),
        &antiperiodic.link_dims(),
        &open.link_dims(),
    );
    let operator_bond_dims = collect_operator_bond_dims(
        &periodic_operator.mpo().link_dims(),
        &antiperiodic_operator.mpo().link_dims(),
        &open_operator.mpo().link_dims(),
    );

    print_summary(&source, &periodic, &antiperiodic, &open, &samples, &config);

    write_samples_csv(&data_dir.join("qtt_affine_samples.csv"), &samples)?;
    write_bond_dims_csv(&data_dir.join("qtt_affine_bond_dims.csv"), &bond_dims)?;
    write_operator_bond_dims_csv(
        &data_dir.join("qtt_affine_operator_bond_dims.csv"),
        &operator_bond_dims,
    )?;

    println!(
        "wrote {}",
        data_dir.join("qtt_affine_samples.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_affine_bond_dims.csv").display()
    );
    println!(
        "wrote {}",
        data_dir.join("qtt_affine_operator_bond_dims.csv").display()
    );
    println!(
        "next: run the Julia plotting script at {}",
        project_root
            .join("docs")
            .join("plotting")
            .join("qtt_affine_plot.jl")
            .display()
    );

    Ok(())
}
