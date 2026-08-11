//! Build two QTTs, multiply them pointwise via TreeTN partial contraction,
//! and export all data for Julia-based plotting.
//!
//! The example is intentionally pedagogical:
//! - one factor is `x^2`
//! - the other factor is `factor_b_function` (default: `sin(10x)`)
//! - the pointwise product is formed through the new public
//!   `tensor4all_treetn::partial_contract` API using `diagonal_pairs`
//! - the result is then truncated again to reduce bond growth
//! - the output plumbing stays separate from the mathematics so the code reads
//!   like a small tutorial

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use tensor4all_core::SvdTruncationPolicy;
use tensor4all_quanticstci::{quanticscrossinterpolate_discrete, QtciOptions, UnfoldingScheme};
use tensor4all_treetn::{
    contraction::{ContractionMethod, ContractionOptions},
    partial_contract, tensor_train_to_treetn, PartialContractionSpec, TruncationOptions,
};
use tensor4all_tutorial_code::output_paths;
use tensor4all_tutorial_code::qtt_elementwise_product_utils::{
    collect_bond_profile, collect_samples, print_qtt_summary, print_treetn_summary,
    write_bond_dims_csv, write_samples_csv,
};

/// Number of quantics bits. The grid therefore has `2^BITS` points.
const BITS: usize = 7;
/// Number of discrete sample points used for the QTT construction.
const NPOINTS: usize = 1 << BITS;
/// Target tolerance for the quantics interpolator.
const TOLERANCE: f64 = 1e-12;
/// Upper bound on the internal bond dimensions during construction.
const MAX_BOND_DIM: usize = 32;
/// How many sample rows to print in the terminal preview.
const SAMPLE_PRINT_COUNT: usize = 8;
/// Frequency in the second factor `sin(10x)`.
const SINE_FREQUENCY: f64 = 10.0;

type QttDemoOutput = (
    tensor4all_quanticstci::QuanticsTensorCI2<f64>,
    Vec<usize>,
    Vec<f64>,
);

/// Small container for all labels and output file names.
///
/// Keeping these values together makes `main()` read like the actual
/// numerical workflow instead of a mixture of computation and file plumbing.
struct OutputSpec {
    square_label: &'static str,
    factor_b_label: &'static str,
    product_label: &'static str,
    samples_csv_name: &'static str,
    bond_dims_csv_name: &'static str,
    plot_script_name: &'static str,
}

impl OutputSpec {
    fn new() -> Self {
        Self {
            square_label: "QTT factor: x^2",
            factor_b_label: "QTT factor B (default: sin(10x))",
            product_label: "TreeTN product via partial_contract: x^2 .* factor B",
            samples_csv_name: "qtt_elementwise_product_samples.csv",
            bond_dims_csv_name: "qtt_elementwise_product_bond_dims.csv",
            plot_script_name: "qtt_elementwise_product_plot.jl",
        }
    }

    fn samples_csv_path(&self, data_dir: &Path) -> PathBuf {
        data_dir.join(self.samples_csv_name)
    }

    fn bond_dims_csv_path(&self, data_dir: &Path) -> PathBuf {
        data_dir.join(self.bond_dims_csv_name)
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

    // Library call: build a QTT approximation from a scalar callback.
    // The callback itself is defined below; `quanticscrossinterpolate_discrete`
    // does the actual interpolation work.
    let (square_qtci, square_ranks, square_errors) = build_qtt_from_function(square_function)?;
    let (factor_b_qtci, factor_b_ranks, factor_b_errors) =
        build_qtt_from_function(factor_b_function)?;

    // Library call: extract the underlying simple TT representation so we can
    // inspect its structure and convert it to TreeTN next.
    let square_tt = square_qtci.tensor_train();
    let factor_b_tt = factor_b_qtci.tensor_train();

    // Library call: bridge the linear-chain TT representation into TreeTN.
    // This is what makes the new partial contraction API usable here.
    let (square_treetn, square_site_indices) = tensor_train_to_treetn(&square_tt)?;
    let (factor_b_treetn, factor_b_site_indices) = tensor_train_to_treetn(&factor_b_tt)?;

    // Library data structure: tell `partial_contract` which site indices should
    // be paired diagonally so that the two functions are multiplied pointwise.
    let diagonal_pairs = square_site_indices
        .iter()
        .cloned()
        .zip(factor_b_site_indices.iter().cloned())
        .collect();
    let spec = PartialContractionSpec {
        contract_pairs: vec![],
        diagonal_pairs,
        output_order: Some(square_site_indices.clone()),
    };

    // TreeTN is a graph structure, so we choose a center node for the
    // contraction/truncation routines used by the library.
    let mut center_nodes = square_treetn.node_names();
    center_nodes.sort();
    let center = center_nodes[center_nodes.len() / 2];

    // Library call: exact TreeTN-level partial contraction.
    // `partial_contract` performs the pointwise product on the paired sites.
    let raw_product_options = ContractionOptions::new(ContractionMethod::Naive)
        .with_dense_reference_limit(NPOINTS * NPOINTS);
    let product_raw_tn = partial_contract(
        &square_treetn,
        &factor_b_treetn,
        &spec,
        &center,
        raw_product_options,
    )?;

    // Library call: compress the exact product TreeTN again so the bond
    // dimensions stay manageable for inspection and plotting.
    let compression_options = TruncationOptions::default()
        .with_svd_policy(SvdTruncationPolicy::new(1e-12))
        .with_max_bond_dim(64);
    let product_compressed_tn = product_raw_tn
        .clone()
        .truncate([center], compression_options)?;

    // Helper code: re-evaluate the factor QTTs and the product TreeTNs on every
    // grid point so we can compare exact vs. reconstructed values.
    let samples = collect_samples(
        &square_qtci,
        &factor_b_qtci,
        &product_raw_tn,
        &product_compressed_tn,
        &square_site_indices,
        BITS,
        NPOINTS,
        square_function,
        factor_b_function,
    )?;
    // Helper code: collect the bond dimensions from the factor QTTs and the
    // product TreeTNs for the Julia plot.
    let bond_profile = collect_bond_profile(
        &square_tt,
        &factor_b_tt,
        &product_raw_tn,
        &product_compressed_tn,
    )?;

    let max_abs_error_raw = samples
        .iter()
        .map(|sample| sample.abs_error_raw)
        .fold(0.0_f64, f64::max);
    let max_abs_error_compressed = samples
        .iter()
        .map(|sample| sample.abs_error_compressed)
        .fold(0.0_f64, f64::max);

    print_qtt_summary(
        output.square_label,
        &square_qtci,
        &square_tt,
        &square_ranks,
        &square_errors,
        SAMPLE_PRINT_COUNT,
    );
    print_qtt_summary(
        output.factor_b_label,
        &factor_b_qtci,
        &factor_b_tt,
        &factor_b_ranks,
        &factor_b_errors,
        SAMPLE_PRINT_COUNT,
    );
    print_treetn_summary(output.product_label, &product_raw_tn)?;
    print_treetn_summary("Compressed TreeTN product", &product_compressed_tn)?;

    println!(
        "max abs error (raw product TreeTN) = {:.3e}",
        max_abs_error_raw
    );
    println!(
        "max abs error (compressed product TreeTN) = {:.3e}",
        max_abs_error_compressed
    );
    println!();

    // Output-only helpers: write the CSV files that Julia will read later.
    write_samples_csv(&output.samples_csv_path(&data_dir), &samples)?;
    write_bond_dims_csv(&output.bond_dims_csv_path(&data_dir), &bond_profile)?;
    println!(
        "next: run the Julia plotting script at {}",
        output.plot_script_path(project_root).display()
    );

    Ok(())
}

/// Build a QTT approximation for a scalar function on the unit interval.
///
/// The public library function `quanticscrossinterpolate_discrete(...)`
/// expects a callback that receives discrete grid indices, so this helper
/// converts those indices into `x in [0, 1)` first.
fn build_qtt_from_function<F>(target_fn: F) -> Result<QttDemoOutput, Box<dyn Error>>
where
    F: Fn(f64) -> f64 + 'static,
{
    let sizes = [NPOINTS];
    let options = QtciOptions::default()
        .with_tolerance(TOLERANCE)
        .with_max_bond_dim(MAX_BOND_DIM)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    // Library callback: the quantics interpolator asks for values on discrete
    // grid points, so we convert the 1-based index into a floating-point x.
    let callback = move |idx: &[i64]| -> f64 {
        let x = (idx[0] as f64 - 1.0) / NPOINTS as f64;
        target_fn(x)
    };

    let initial_pivots = vec![vec![2], vec![(NPOINTS / 2) as i64], vec![NPOINTS as i64]];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        callback,
        Some(initial_pivots),
        options,
    )?)
}

/// The first target function: `x^2`.
///
/// This is plain scalar math, not a tensor-library API call.
fn square_function(x: f64) -> f64 {
    x.powi(2)
}

/// The second target function: `sin(10x)`.
///
/// This is also plain scalar math; the tensor library only sees the callback
/// that returns a value for a given grid point.
fn factor_b_function(x: f64) -> f64 {
    (SINE_FREQUENCY * x).sin()
}
