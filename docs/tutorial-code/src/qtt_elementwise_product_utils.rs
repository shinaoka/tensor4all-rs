//! Helper utilities for the QTT elementwise-product demo.
//!
//! The binary stays focused on the library workflow:
//! - build two QTTs from functions,
//! - convert them to TreeTN,
//! - multiply them with `partial_contract`,
//! - compress the result with `TreeTN::truncate`,
//! - export the data for Julia plotting.
//!
//! All output formatting, CSV writing, and reusable bookkeeping lives here so
//! the binary remains readable for beginners.
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use tensor4all_core::{DynIndex, IndexLike, TensorDynLen};
use tensor4all_quanticstci::QuanticsTensorCI2;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops};
use tensor4all_treetn::TreeTN;

/// One row in the exported sample table.
///
/// The table records the two factor functions, their pointwise product,
/// and the QTT approximations before and after compressing the product tree.
#[derive(Debug, Clone)]
pub struct SamplePoint {
    pub index: usize,
    pub x: f64,
    pub cosh_exact: f64,
    pub cosh_qtt: f64,
    pub factor_b_exact: f64,
    pub factor_b_qtt: f64,
    pub product_exact: f64,
    pub product_raw: f64,
    pub product_compressed: f64,
    pub abs_error_raw: f64,
    pub abs_error_compressed: f64,
}

/// One row in the bond-dimension profile table.
#[derive(Debug, Clone)]
pub struct BondProfileRow {
    pub bond_index: usize,
    pub cosh: usize,
    pub factor_b: usize,
    pub product_raw: usize,
    pub product_compressed: usize,
}

/// Convert the 1-based indexing used by the quantics callback into x in [0, 1).
pub fn discrete_index_to_unit_interval(index_1based: i64, npoints: usize) -> f64 {
    (index_1based as f64 - 1.0) / npoints as f64
}

/// Convert a 1-based grid index into binary TT site indices.
///
/// The returned indices follow the Quantics grid convention used here:
/// the first TT site corresponds to the most significant bit.
pub fn global_index_to_quantics_sites(index_1based: usize, bits: usize) -> Vec<usize> {
    let mut sites = Vec::with_capacity(bits);

    for bit in (0..bits).rev() {
        sites.push(((index_1based - 1) >> bit) & 1);
    }

    sites
}

/// Print a compact summary for a QTT that came from
/// `quanticscrossinterpolate_discrete(...)`.
///
/// The `qtci` value is the object returned by the library interpolator, while
/// `tt` is the extracted `SimpleTensorTrain` that exposes site/bond structure.
pub fn print_qtt_summary(
    label: &str,
    qtci: &QuanticsTensorCI2<f64>,
    tt: &SimpleTensorTrain<f64>,
    ranks: &[usize],
    errors: &[f64],
    sample_print_count: usize,
) {
    println!("{label}");
    println!("qtt length = {}", tt.len());
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
    println!();

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

/// Print a summary for a TreeTN that represents the product and its compressed
/// version.
///
/// This is a pure reporting helper around the TreeTN library object.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn print_treetn_summary(
    label: &str,
    tn: &TreeTN<TensorDynLen, usize>,
) -> Result<(), Box<dyn Error>> {
    println!("{label}");
    println!("node_count = {}", tn.node_count());
    println!("edge_count = {}", tn.edge_count());
    println!("link_dims = {:?}", tn.link_dims());
    let (site_indices, owners) = tn.all_site_indices()?;
    println!("site indices = {}", site_indices.len());
    for (index, owner) in site_indices.iter().zip(owners.iter()) {
        println!("  node {}: site_dim = {}", owner, index.dim());
    }
    println!();
    Ok(())
}

fn evaluate_tree_point(
    tn: &TreeTN<TensorDynLen, usize>,
    site_indices: &[DynIndex],
    site_values: &[usize],
) -> Result<f64, Box<dyn Error>> {
    let value = tn.evaluate_point(site_indices, site_values)?;
    Ok(value.real())
}

/// Evaluate the QTTs on the shared grid and compare them with the analytic
/// product `cosh(x) * factor B`.
///
/// This function is mostly bookkeeping:
/// - the QTT library gives us `evaluate(...)` for the factor QTTs
/// - the TreeTN library gives us `evaluate_point(...)` for the product networks
/// - we combine all values into one row per grid point
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples<F, G>(
    cosh_qtci: &QuanticsTensorCI2<f64>,
    factor_b_qtci: &QuanticsTensorCI2<f64>,
    product_raw_tn: &TreeTN<TensorDynLen, usize>,
    product_compressed_tn: &TreeTN<TensorDynLen, usize>,
    product_site_indices: &[DynIndex],
    bits: usize,
    npoints: usize,
    cosh_fn: F,
    factor_b_fn: G,
) -> Result<Vec<SamplePoint>, Box<dyn Error>>
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut samples = Vec::with_capacity(npoints);

    for i in 1..=npoints {
        let x = discrete_index_to_unit_interval(i as i64, npoints);
        let cosh_exact = cosh_fn(x);
        let factor_b_exact = factor_b_fn(x);
        let product_exact = cosh_exact * factor_b_exact;

        // Library call: read one value back from each factor QTT.
        let cosh_qtt = cosh_qtci.evaluate(&[i as i64])?;
        let factor_b_qtt = factor_b_qtci.evaluate(&[i as i64])?;
        let site_values = global_index_to_quantics_sites(i, bits);
        // Library call: read the pointwise product back from the TreeTN.
        let product_raw = evaluate_tree_point(product_raw_tn, product_site_indices, &site_values)?;
        let product_compressed =
            evaluate_tree_point(product_compressed_tn, product_site_indices, &site_values)?;

        samples.push(SamplePoint {
            index: i,
            x,
            cosh_exact,
            cosh_qtt,
            factor_b_exact,
            factor_b_qtt,
            product_exact,
            product_raw,
            product_compressed,
            abs_error_raw: (product_exact - product_raw).abs(),
            abs_error_compressed: (product_exact - product_compressed).abs(),
        });
    }

    Ok(samples)
}

/// Collect the bond-dimension profile for the two factors and their product.
///
/// The factor QTTs are simple tensor trains, so we use `link_dims()` there.
/// The product lives as TreeTN, so we use `TreeTN::link_dims()` for those rows.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_bond_profile(
    cosh_tt: &SimpleTensorTrain<f64>,
    factor_b_tt: &SimpleTensorTrain<f64>,
    product_raw_tn: &TreeTN<TensorDynLen, usize>,
    product_compressed_tn: &TreeTN<TensorDynLen, usize>,
) -> Result<Vec<BondProfileRow>, Box<dyn Error>> {
    let cosh_bonds = cosh_tt.link_dims();
    let factor_b_bonds = factor_b_tt.link_dims();
    let raw_bonds = product_raw_tn.link_dims();
    let compressed_bonds = product_compressed_tn.link_dims();

    let len = [
        cosh_bonds.len(),
        factor_b_bonds.len(),
        raw_bonds.len(),
        compressed_bonds.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);

    let mut rows = Vec::with_capacity(len);
    for i in 0..len {
        rows.push(BondProfileRow {
            bond_index: i + 1,
            cosh: *cosh_bonds.get(i).unwrap_or(&1),
            factor_b: *factor_b_bonds.get(i).unwrap_or(&1),
            product_raw: *raw_bonds.get(i).unwrap_or(&1),
            product_compressed: *compressed_bonds.get(i).unwrap_or(&1),
        });
    }

    Ok(rows)
}

/// Write the sample table to CSV.
///
/// Output-only helper: the Tensor4all APIs have already done the numerical work
/// by the time this function is called.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(path: &Path, samples: &[SamplePoint]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "index,x,cosh_exact,cosh_qtt,factor_b_exact,factor_b_qtt,product_exact,product_raw,product_compressed,abs_error_raw,abs_error_compressed"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.index,
            sample.x,
            sample.cosh_exact,
            sample.cosh_qtt,
            sample.factor_b_exact,
            sample.factor_b_qtt,
            sample.product_exact,
            sample.product_raw,
            sample.product_compressed,
            sample.abs_error_raw,
            sample.abs_error_compressed
        )?;
    }

    println!("wrote {}", path.display());
    Ok(())
}

/// Write the bond-dimension profile to CSV.
///
/// Another output-only helper so the binary stays focused on the tensor logic.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(path: &Path, rows: &[BondProfileRow]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "bond_index,cosh,factor_b,product_raw,product_compressed")?;
    for row in rows {
        writeln!(
            w,
            "{},{},{},{},{}",
            row.bond_index, row.cosh, row.factor_b, row.product_raw, row.product_compressed
        )?;
    }

    println!("wrote {}", path.display());
    Ok(())
}
