//! Shared setup for the affine transformation tutorial.
//!
//! This helper keeps affine-operator construction, fused quantics grid conversion,
//! dense sampling, CSV writing, and summary printing out of the tutorial binary.

use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use num_complex::Complex64;
use tensor4all_core::index::{DynId, Index, TagSet};
use tensor4all_core::TensorDynLen;
use tensor4all_quanticstci::{
    quanticscrossinterpolate_discrete, InherentDiscreteGrid, QtciOptions, QuanticsTensorCI2,
    UnfoldingScheme,
};
use tensor4all_quanticstransform::{affine_operator, AffineParams, BoundaryCondition};
use tensor4all_treetn::{
    apply_linear_operator, tensor_train_to_treetn, ApplyOptions, LinearOperator, TreeTN,
};

/// Quantics site index type used by the affine helper.
pub type SiteIndex = Index<DynId, TagSet>;

/// Configuration for the affine transformation tutorial.
#[derive(Debug, Clone, Copy)]
pub struct AffineTutorialConfig {
    pub bits: usize,
    pub tolerance: f64,
    pub max_bond_dim: usize,
    pub maxiter: usize,
}

/// Default configuration used throughout the affine tutorial.
pub const DEFAULT_AFFINE_CONFIG: AffineTutorialConfig = AffineTutorialConfig {
    bits: 6,
    tolerance: 1e-12,
    max_bond_dim: 64,
    maxiter: 20,
};

/// Boundary-condition mode for the transformed function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineBoundaryMode {
    Periodic,
    AntiPeriodic,
    Open,
}

/// One sample row in the exported dense table.
#[derive(Debug, Clone)]
pub struct AffineSamplePoint {
    pub x_index: usize,
    pub y_index: usize,
    pub x: usize,
    pub y: usize,
    pub source_u_periodic: usize,
    pub source_v: usize,
    pub source_exact: f64,
    pub periodic_exact: f64,
    pub periodic_qtt: f64,
    pub periodic_abs_error: f64,
    pub antiperiodic_exact: f64,
    pub antiperiodic_qtt: f64,
    pub antiperiodic_abs_error: f64,
    pub open_exact: f64,
    pub open_qtt: f64,
    pub open_abs_error: f64,
}

pub type AffineBondDimRow = (
    usize,
    Option<usize>,
    Option<usize>,
    Option<usize>,
    Option<usize>,
);
pub type AffineOperatorBondDimRow = (usize, Option<usize>, Option<usize>, Option<usize>);

/// Result of applying the affine operator to a source QTT.
pub type AffineTransformOutput = (TreeTN<TensorDynLen, usize>, Vec<SiteIndex>);
/// Result of building the source QTT.
pub type AffineQttOutput = (QuanticsTensorCI2<f64>, Vec<usize>, Vec<f64>);

/// Number of grid points per direction.
pub fn point_count(bits: usize) -> usize {
    1usize << bits
}

/// Source function used in the tutorial.
pub fn source_function(u: usize, v: usize, n: usize) -> f64 {
    let u = u as f64;
    let v = v as f64;
    let n = n as f64;
    (2.0 * PI * u / n).sin()
        + 0.5 * (2.0 * PI * v / n).cos()
        + 0.25 * (2.0 * PI * (u + 2.0 * v) / n).sin()
}

/// Analytic reference for the affine pullback.
pub fn transformed_reference(x: usize, y: usize, bits: usize, mode: AffineBoundaryMode) -> f64 {
    let n = point_count(bits);
    match mode {
        AffineBoundaryMode::Periodic => source_function((x + y) % n, y, n),
        AffineBoundaryMode::AntiPeriodic => {
            let sign = if x + y >= n { -1.0 } else { 1.0 };
            sign * source_function((x + y) % n, y, n)
        }
        AffineBoundaryMode::Open => {
            if x + y >= n {
                0.0
            } else {
                source_function(x + y, y, n)
            }
        }
    }
}

/// Build the source QTT for the periodic analytic function.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_source_qtt(config: &AffineTutorialConfig) -> Result<AffineQttOutput, Box<dyn Error>> {
    let n = point_count(config.bits);
    let sizes = vec![n, n];
    let options = QtciOptions::default()
        .with_tolerance(config.tolerance)
        .with_max_bond_dim(config.max_bond_dim)
        .with_maxiter(config.maxiter)
        .with_nrandominitpivot(0)
        .with_unfoldingscheme(UnfoldingScheme::Fused)
        .with_verbosity(0);

    let callback = move |grid_1based: &[i64]| -> f64 {
        let u = (grid_1based[0] - 1) as usize;
        let v = (grid_1based[1] - 1) as usize;
        source_function(u, v, n)
    };

    let initial_pivots = vec![
        vec![1, 1],
        vec![(n / 2) as i64, (n / 2) as i64],
        vec![n as i64, n as i64],
    ];

    Ok(quanticscrossinterpolate_discrete(
        &sizes,
        callback,
        Some(initial_pivots),
        options,
    )?)
}

fn affine_params() -> Result<AffineParams, Box<dyn Error>> {
    Ok(AffineParams::from_integers(
        vec![1, 0, 1, 1],
        vec![0, 0],
        2,
        2,
    )?)
}

fn boundary_conditions(mode: AffineBoundaryMode) -> Vec<BoundaryCondition> {
    match mode {
        AffineBoundaryMode::Periodic => vec![BoundaryCondition::Periodic; 2],
        AffineBoundaryMode::AntiPeriodic => {
            vec![BoundaryCondition::AntiPeriodic, BoundaryCondition::Periodic]
        }
        AffineBoundaryMode::Open => vec![BoundaryCondition::Open; 2],
    }
}

/// Build the affine operator and transpose it to obtain the passive pullback.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn build_affine_operator(
    config: &AffineTutorialConfig,
    mode: AffineBoundaryMode,
) -> Result<LinearOperator<TensorDynLen, usize>, Box<dyn Error>> {
    let params = affine_params()?;
    Ok(affine_operator(config.bits, &params, &boundary_conditions(mode))?.transpose())
}

/// Evaluate a TreeTN at one set of site values and return the complex scalar.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn evaluate_tree_point(
    tn: &TreeTN<TensorDynLen, usize>,
    site_indices: &[SiteIndex],
    site_values: &[usize],
) -> Result<Complex64, Box<dyn Error>> {
    let value = tn.evaluate_point(site_indices, site_values)?;
    Ok(Complex64::new(value.real(), value.imag()))
}

/// Apply the affine operator to the source QTT.
/// # Errors
///
/// Returns an error when the construction or evaluation fails (a shape
/// /// mismatch or backend failure).
///
pub fn apply_affine_operator(
    source: &QuanticsTensorCI2<f64>,
    operator: &LinearOperator<TensorDynLen, usize>,
) -> Result<AffineTransformOutput, Box<dyn Error>> {
    let tt = source.tensor_train();
    let (state, _site_indices) = tensor_train_to_treetn(&tt)?;

    let mut aligned_operator = operator.clone();
    aligned_operator.align_to_state(&state)?;

    // use tensor4all-rs to carry out MPO-MPS contraction
    let transformed = apply_linear_operator(&aligned_operator, &state, ApplyOptions::naive())?;
    let output_site_indices = tensor4all_core::TensorIndex::external_indices(&transformed);
    Ok((transformed, output_site_indices))
}

/// Collect dense transformed samples against the analytic reference.
#[allow(clippy::too_many_arguments)]
/// # Errors
///
/// Returns an error when the collection fails (a shape mismatch or
/// /// backend failure).
///
pub fn collect_samples(
    periodic: &TreeTN<TensorDynLen, usize>,
    periodic_site_indices: &[SiteIndex],
    antiperiodic: &TreeTN<TensorDynLen, usize>,
    antiperiodic_site_indices: &[SiteIndex],
    open: &TreeTN<TensorDynLen, usize>,
    open_site_indices: &[SiteIndex],
    evaluation_grid: &InherentDiscreteGrid,
    config: &AffineTutorialConfig,
) -> Result<Vec<AffineSamplePoint>, Box<dyn Error>> {
    let n = point_count(config.bits);
    let mut samples = Vec::with_capacity(n * n);

    for x in 0..n {
        for y in 0..n {
            let sites: Vec<usize> = evaluation_grid
                .grididx_to_quantics(&[(x + 1) as i64, (y + 1) as i64])?
                .into_iter()
                .map(|site| (site - 1) as usize)
                .collect();
            let periodic_qtt = evaluate_tree_point(periodic, periodic_site_indices, &sites)?.re;
            let antiperiodic_qtt =
                evaluate_tree_point(antiperiodic, antiperiodic_site_indices, &sites)?.re;
            let open_qtt = evaluate_tree_point(open, open_site_indices, &sites)?.re;
            let source_exact = source_function(x, y, n);
            let periodic_exact =
                transformed_reference(x, y, config.bits, AffineBoundaryMode::Periodic);
            let antiperiodic_exact =
                transformed_reference(x, y, config.bits, AffineBoundaryMode::AntiPeriodic);
            let open_exact = transformed_reference(x, y, config.bits, AffineBoundaryMode::Open);

            samples.push(AffineSamplePoint {
                x_index: x + 1,
                y_index: y + 1,
                x,
                y,
                source_u_periodic: (x + y) % n,
                source_v: y,
                source_exact,
                periodic_exact,
                periodic_qtt,
                periodic_abs_error: (periodic_exact - periodic_qtt).abs(),
                antiperiodic_exact,
                antiperiodic_qtt,
                antiperiodic_abs_error: (antiperiodic_exact - antiperiodic_qtt).abs(),
                open_exact,
                open_qtt,
                open_abs_error: (open_exact - open_qtt).abs(),
            });
        }
    }

    Ok(samples)
}

/// Pair input and transformed bond dimensions for the tutorial CSV output.
pub fn collect_bond_dims(
    input: &[usize],
    periodic: &[usize],
    antiperiodic: &[usize],
    open: &[usize],
) -> Vec<AffineBondDimRow> {
    let row_count = input
        .len()
        .max(periodic.len())
        .max(antiperiodic.len())
        .max(open.len());
    (0..row_count)
        .map(|i| {
            (
                i + 1,
                input.get(i).copied(),
                periodic.get(i).copied(),
                antiperiodic.get(i).copied(),
                open.get(i).copied(),
            )
        })
        .collect()
}

/// Pair affine operator bond dimensions for the tutorial CSV output.
pub fn collect_operator_bond_dims(
    periodic: &[usize],
    antiperiodic: &[usize],
    open: &[usize],
) -> Vec<AffineOperatorBondDimRow> {
    let row_count = periodic.len().max(antiperiodic.len()).max(open.len());
    (0..row_count)
        .map(|i| {
            (
                i + 1,
                periodic.get(i).copied(),
                antiperiodic.get(i).copied(),
                open.get(i).copied(),
            )
        })
        .collect()
}

fn write_optional_usize(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Write dense samples to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_samples_csv(path: &Path, samples: &[AffineSamplePoint]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "x_index,y_index,x,y,source_u_periodic,source_v,source_exact,periodic_exact,periodic_qtt,periodic_abs_error,antiperiodic_exact,antiperiodic_qtt,antiperiodic_abs_error,open_exact,open_qtt,open_abs_error"
    )?;
    for sample in samples {
        writeln!(
            w,
            "{},{},{},{},{},{},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16},{:.16}",
            sample.x_index,
            sample.y_index,
            sample.x,
            sample.y,
            sample.source_u_periodic,
            sample.source_v,
            sample.source_exact,
            sample.periodic_exact,
            sample.periodic_qtt,
            sample.periodic_abs_error,
            sample.antiperiodic_exact,
            sample.antiperiodic_qtt,
            sample.antiperiodic_abs_error,
            sample.open_exact,
            sample.open_qtt,
            sample.open_abs_error
        )?;
    }

    Ok(())
}

/// Write bond-dimension rows to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_bond_dims_csv(path: &Path, rows: &[AffineBondDimRow]) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "bond_index,input_bond_dim,periodic_transformed_bond_dim,antiperiodic_transformed_bond_dim,open_transformed_bond_dim"
    )?;
    for (index, input, periodic, antiperiodic, open) in rows {
        writeln!(
            w,
            "{},{},{},{},{}",
            index,
            write_optional_usize(*input),
            write_optional_usize(*periodic),
            write_optional_usize(*antiperiodic),
            write_optional_usize(*open)
        )?;
    }

    Ok(())
}

/// Write affine operator bond dimensions to CSV.
/// # Errors
///
/// Returns an error when the CSV output cannot be written (an I/O
/// /// failure).
///
pub fn write_operator_bond_dims_csv(
    path: &Path,
    rows: &[AffineOperatorBondDimRow],
) -> Result<(), Box<dyn Error>> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    writeln!(
        w,
        "bond_index,periodic_operator_bond_dim,antiperiodic_operator_bond_dim,open_operator_bond_dim"
    )?;
    for (index, periodic, antiperiodic, open) in rows {
        writeln!(
            w,
            "{},{},{},{}",
            index,
            write_optional_usize(*periodic),
            write_optional_usize(*antiperiodic),
            write_optional_usize(*open)
        )?;
    }

    Ok(())
}

/// Print a compact summary to the terminal used by the binary.
pub fn print_summary(
    source: &QuanticsTensorCI2<f64>,
    periodic: &TreeTN<TensorDynLen, usize>,
    antiperiodic: &TreeTN<TensorDynLen, usize>,
    open: &TreeTN<TensorDynLen, usize>,
    samples: &[AffineSamplePoint],
    config: &AffineTutorialConfig,
) {
    let max_periodic_error = samples
        .iter()
        .map(|sample| sample.periodic_abs_error)
        .fold(0.0_f64, f64::max);
    let max_open_error = samples
        .iter()
        .map(|sample| sample.open_abs_error)
        .fold(0.0_f64, f64::max);
    let max_antiperiodic_error = samples
        .iter()
        .map(|sample| sample.antiperiodic_abs_error)
        .fold(0.0_f64, f64::max);

    println!("Affine transformation tutorial");
    println!("bits = {}", config.bits);
    println!("grid points per direction = {}", point_count(config.bits));
    println!("source rank = {}", source.rank());
    println!(
        "periodic transformed node count = {}",
        periodic.node_count()
    );
    println!(
        "anti-periodic transformed node count = {}",
        antiperiodic.node_count()
    );
    println!("open transformed node count = {}", open.node_count());
    println!("max periodic abs error = {:.3e}", max_periodic_error);
    println!(
        "max anti-periodic abs error = {:.3e}",
        max_antiperiodic_error
    );
    println!("max open abs error = {:.3e}", max_open_error);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch_csv(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time is after the Unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}-{nonce}.csv"))
    }

    #[test]
    fn transformed_reference_applies_boundary_modes() {
        let bits = 3;
        let n = point_count(bits);

        let periodic = transformed_reference(n - 1, 2, bits, AffineBoundaryMode::Periodic);
        assert!((periodic - source_function(1, 2, n)).abs() < 1e-12);

        let open_outside = transformed_reference(n - 1, 2, bits, AffineBoundaryMode::Open);
        assert_eq!(open_outside, 0.0);

        let open_inside = transformed_reference(1, 2, bits, AffineBoundaryMode::Open);
        assert!((open_inside - source_function(3, 2, n)).abs() < 1e-12);
    }

    #[test]
    fn bond_dim_rows_pad_missing_profiles() {
        assert_eq!(
            collect_bond_dims(&[2, 3], &[5], &[6], &[7, 11, 13]),
            vec![
                (1, Some(2), Some(5), Some(6), Some(7)),
                (2, Some(3), None, None, Some(11)),
                (3, None, None, None, Some(13)),
            ]
        );

        assert_eq!(
            collect_operator_bond_dims(&[2], &[4], &[3, 5]),
            vec![(1, Some(2), Some(4), Some(3)), (2, None, None, Some(5))]
        );
    }

    #[test]
    fn csv_writers_keep_headers_and_empty_optional_cells() -> Result<(), Box<dyn Error>> {
        let samples_path = scratch_csv("affine-samples");
        write_samples_csv(
            &samples_path,
            &[AffineSamplePoint {
                x_index: 1,
                y_index: 2,
                x: 0,
                y: 1,
                source_u_periodic: 1,
                source_v: 1,
                source_exact: 0.5,
                periodic_exact: 0.25,
                periodic_qtt: 0.125,
                periodic_abs_error: 0.125,
                antiperiodic_exact: -0.25,
                antiperiodic_qtt: -0.125,
                antiperiodic_abs_error: 0.125,
                open_exact: 0.0,
                open_qtt: 0.0,
                open_abs_error: 0.0,
            }],
        )?;
        let sample_csv = fs::read_to_string(&samples_path)?;
        assert!(sample_csv.starts_with(
            "x_index,y_index,x,y,source_u_periodic,source_v,source_exact,periodic_exact,periodic_qtt,periodic_abs_error,antiperiodic_exact,antiperiodic_qtt,antiperiodic_abs_error,open_exact,open_qtt,open_abs_error\n"
        ));
        assert!(sample_csv.contains(",0.5000000000000000,0.2500000000000000,"));

        let bond_dims_path = scratch_csv("affine-bonds");
        write_bond_dims_csv(&bond_dims_path, &[(1, Some(2), None, Some(3), Some(4))])?;
        let bond_csv = fs::read_to_string(&bond_dims_path)?;
        assert_eq!(
            bond_csv,
            "bond_index,input_bond_dim,periodic_transformed_bond_dim,antiperiodic_transformed_bond_dim,open_transformed_bond_dim\n1,2,,3,4\n"
        );

        let operator_dims_path = scratch_csv("affine-operator-bonds");
        write_operator_bond_dims_csv(&operator_dims_path, &[(1, None, Some(6), Some(8))])?;
        let operator_csv = fs::read_to_string(&operator_dims_path)?;
        assert_eq!(
            operator_csv,
            "bond_index,periodic_operator_bond_dim,antiperiodic_operator_bond_dim,open_operator_bond_dim\n1,,6,8\n"
        );

        Ok(())
    }
}
