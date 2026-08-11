//! Shared setup for the difference-kernel MPO tutorial.
//!
//! The helper builds a one-dimensional kernel QTT, converts it to the
//! `Complex64` representation expected by `difference_kernel_mpo`, samples the
//! resulting MPO as a dense matrix, and writes the CSV files used by plotting.

use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use num_complex::Complex64;
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, QtciOptions, QuanticsTensorCI2, UnfoldingScheme,
};
use tensor4all_quanticstransform::{difference_kernel_mpo, BoundaryCondition};
use tensor4all_simplett::{
    types::tensor3_zeros, AbstractTensorTrain, SimpleTensorTrain, Tensor3Ops,
};

/// Configuration for the difference-kernel tutorial.
#[derive(Debug, Clone, Copy)]
pub struct DifferenceKernelTutorialConfig {
    pub bits: usize,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub maxiter: usize,
}

/// Default configuration used by the tutorial binary.
pub const DEFAULT_DIFFERENCE_KERNEL_CONFIG: DifferenceKernelTutorialConfig =
    DifferenceKernelTutorialConfig {
        bits: 6,
        tolerance: 1e-12,
        max_bond_dim: 64,
        maxiter: 20,
    };

/// One row in the dense matrix sample table.
#[derive(Debug, Clone)]
pub struct DifferenceKernelSamplePoint {
    pub x_index: usize,
    pub xprime_index: usize,
    pub x: usize,
    pub xprime: usize,
    pub difference: usize,
    pub source_exact: f64,
    pub source_qtt: f64,
    pub source_abs_error: f64,
    pub kernel_exact: f64,
    pub kernel_mpo: f64,
    pub abs_error: f64,
}

/// One row in the bond-dimension profile table.
#[derive(Debug, Clone)]
pub struct DifferenceKernelBondDimRow {
    pub bond_index: usize,
    pub kernel_qtt_bond_dim: Option<usize>,
    pub difference_kernel_mpo_bond_dim: Option<usize>,
}

/// Result of building the source kernel QTT.
pub type DifferenceKernelQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);

/// Number of grid points in the one-dimensional quantics grid.
pub fn point_count(bits: usize) -> usize {
    1usize << bits
}

/// Smooth periodic kernel used in the tutorial.
pub fn kernel_value(z: usize, n: usize) -> f64 {
    let theta = 2.0 * PI * z as f64 / n as f64;
    (2.0 * (theta.cos() - 1.0)).exp()
}

/// Periodic difference index for `A[x, x'] = f(x - x')`.
pub fn periodic_difference(x: usize, xprime: usize, n: usize) -> usize {
    (x + n - xprime) % n
}

/// Build the one-dimensional kernel QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_kernel_qtt(
    config: &DifferenceKernelTutorialConfig,
) -> Result<DifferenceKernelQttOutput, Box<dyn Error>> {
    let n = point_count(config.bits);
    let sizes = [n];
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Interleaved)
        .with_verbosity(0);

    let callback = move |idx: &[i64]| -> f64 {
        let z = (idx[0] - 1) as usize;
        kernel_value(z, n)
    };

    let initial_pivots = vec![
        vec![1],
        vec![(n / 4).max(1) as i64],
        vec![(n / 2).max(1) as i64],
        vec![n as i64],
    ];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        callback,
        Some(initial_pivots),
        options,
    )?)
}

/// Convert a real QTT to the complex-valued QTT expected by quantics transforms.
/// # Errors
///
/// Returns an error when the operation fails (a shape mismatch or backend
/// /// failure).
///
pub fn real_qtt_to_complex(
    tt: &SimpleTensorTrain<f64>,
) -> Result<SimpleTensorTrain<Complex64>, Box<dyn Error>> {
    let mut tensors = Vec::with_capacity(tt.len());
    for site in 0..tt.len() {
        let core = tt.site_tensor(site);
        let mut complex_core = tensor3_zeros(core.left_dim(), core.site_dim(), core.right_dim());
        for left in 0..core.left_dim() {
            for local in 0..core.site_dim() {
                for right in 0..core.right_dim() {
                    complex_core.set3(
                        left,
                        local,
                        right,
                        Complex64::new(*core.get3(left, local, right), 0.0),
                    );
                }
            }
        }
        tensors.push(complex_core);
    }

    Ok(SimpleTensorTrain::new(tensors)?)
}

/// Build `A[x, x'] = f((x - x') mod 2^R)` as a periodic difference-kernel MPO.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_periodic_difference_kernel_mpo(
    kernel_tt: &SimpleTensorTrain<f64>,
) -> Result<SimpleTensorTrain<Complex64>, Box<dyn Error>> {
    let complex_kernel = real_qtt_to_complex(kernel_tt)?;
    Ok(difference_kernel_mpo(
        &complex_kernel,
        BoundaryCondition::Periodic,
    )?)
}

/// Convert a zero-based integer grid point into quantics bits.
pub fn integer_to_quantics_sites(value: usize, bits: usize) -> Vec<usize> {
    (0..bits).rev().map(|bit| (value >> bit) & 1).collect()
}

/// Evaluate the fused-site MPO at a matrix entry `(x, x')`.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn evaluate_difference_kernel_mpo(
    mpo: &SimpleTensorTrain<Complex64>,
    x: usize,
    xprime: usize,
    bits: usize,
) -> Result<f64, Box<dyn Error>> {
    let x_bits = integer_to_quantics_sites(x, bits);
    let xprime_bits = integer_to_quantics_sites(xprime, bits);
    let sites: Vec<usize> = x_bits
        .iter()
        .zip(xprime_bits.iter())
        .map(|(&x_bit, &xprime_bit)| x_bit * 2 + xprime_bit)
        .collect();
    Ok(mpo.evaluate(&sites)?.re)
}

/// Collect dense matrix samples against the analytic periodic reference.
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples(
    kernel: &QuanticsTensorCI2<f64>,
    mpo: &SimpleTensorTrain<Complex64>,
    config: &DifferenceKernelTutorialConfig,
) -> Result<Vec<DifferenceKernelSamplePoint>, Box<dyn Error>> {
    let n = point_count(config.bits);
    let mut samples = Vec::with_capacity(n * n);

    for x in 0..n {
        for xprime in 0..n {
            let difference = periodic_difference(x, xprime, n);
            let source_exact = kernel_value(difference, n);
            let source_qtt = kernel.evaluate(&[(difference + 1) as i64])?;
            let kernel_exact = source_exact;
            let kernel_mpo = evaluate_difference_kernel_mpo(mpo, x, xprime, config.bits)?;

            samples.push(DifferenceKernelSamplePoint {
                x_index: x + 1,
                xprime_index: xprime + 1,
                x,
                xprime,
                difference,
                source_exact,
                source_qtt,
                source_abs_error: (source_exact - source_qtt).abs(),
                kernel_exact,
                kernel_mpo,
                abs_error: (kernel_exact - kernel_mpo).abs(),
            });
        }
    }

    Ok(samples)
}

/// Pair kernel-QTT and difference-kernel-MPO bond dimensions.
pub fn collect_bond_dims(
    kernel_qtt: &[usize],
    difference_kernel_mpo: &[usize],
) -> Vec<DifferenceKernelBondDimRow> {
    let row_count = kernel_qtt.len().max(difference_kernel_mpo.len());
    (0..row_count)
        .map(|i| DifferenceKernelBondDimRow {
            bond_index: i + 1,
            kernel_qtt_bond_dim: kernel_qtt.get(i).copied(),
            difference_kernel_mpo_bond_dim: difference_kernel_mpo.get(i).copied(),
        })
        .collect()
}

fn write_optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Write dense matrix samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(
    path: &Path,
    samples: &[DifferenceKernelSamplePoint],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "x_index,xprime_index,x,xprime,difference,source_exact,source_qtt,source_abs_error,kernel_exact,kernel_mpo,abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.x_index,
            sample.xprime_index,
            sample.x,
            sample.xprime,
            sample.difference,
            sample.source_exact,
            sample.source_qtt,
            sample.source_abs_error,
            sample.kernel_exact,
            sample.kernel_mpo,
            sample.abs_error
        )?;
    }

    Ok(())
}

/// Write bond dimensions to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(
    path: &Path,
    rows: &[DifferenceKernelBondDimRow],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "bond_index,kernel_qtt_bond_dim,difference_kernel_mpo_bond_dim"
    )?;
    for row in rows {
        writeln!(
            w,
            "{},{},{}",
            row.bond_index,
            write_optional_usize(row.kernel_qtt_bond_dim),
            write_optional_usize(row.difference_kernel_mpo_bond_dim)
        )?;
    }

    Ok(())
}

/// Print a compact summary to the terminal used by the binary.
pub fn print_summary(
    kernel: &QuanticsTensorCI2<f64>,
    mpo: &SimpleTensorTrain<Complex64>,
    ranks: &[usize],
    errors: &[f64],
    samples: &[DifferenceKernelSamplePoint],
    config: &DifferenceKernelTutorialConfig,
) {
    let max_source_error = samples
        .iter()
        .map(|sample| sample.source_abs_error)
        .fold(0.0_f64, f64::max);
    let max_mpo_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);

    println!("Difference-kernel MPO tutorial");
    println!("bits = {}", config.bits);
    println!("grid points = {}", point_count(config.bits));
    println!("kernel QTT rank = {}", kernel.rank());
    println!("kernel QTT link_dims = {:?}", kernel.link_dims());
    println!("difference-kernel MPO link_dims = {:?}", mpo.link_dims());
    println!("rank history length = {}", ranks.len());
    println!("error history length = {}", errors.len());
    println!("max kernel QTT abs error = {:.3e}", max_source_error);
    println!("max MPO abs error = {:.3e}", max_mpo_error);
}
