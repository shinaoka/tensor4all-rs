use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use tensor4all_simplett::AbstractTensorTrain;
use tensor4all_tutorial_code::{
    output_paths, qtt_affine_common, qtt_difference_kernel_common, qtt_elementwise_product_utils,
    qtt_fourier_common, qtt_function_utils, qtt_integral_sweep_utils, qtt_partial_fourier2d_common,
    qtt_r_sweep_utils,
};

fn scratch_dir(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX_EPOCH")
        .as_nanos();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("verification")
        .join(format!("{name}-{nonce}-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn read_lines(path: &Path) -> Vec<String> {
    fs::read_to_string(path)
        .expect("read generated csv")
        .lines()
        .map(|line| line.to_owned())
        .collect()
}

fn assert_header(path: &Path, expected: &str) {
    let lines = read_lines(path);
    assert_eq!(lines.first().map(String::as_str), Some(expected));
    assert_eq!(lines.len(), 2);
}

#[test]
fn output_paths_support_default_and_override() {
    let manifest_dir = Path::new("/tmp/project-root");

    let default_dir = output_paths::resolve_data_dir(manifest_dir, None);
    assert_eq!(default_dir, manifest_dir.join("docs").join("data"));

    let override_dir = output_paths::resolve_data_dir(manifest_dir, Some("/tmp/check/data"));
    assert_eq!(override_dir, PathBuf::from("/tmp/check/data"));
}

#[test]
fn quantics_index_helpers_keep_expected_order() {
    assert_eq!(
        qtt_fourier_common::global_index_to_quantics_sites(5, 3),
        vec![1, 0, 0]
    );
    assert_eq!(
        qtt_partial_fourier2d_common::x_site_node_mapping(3),
        vec![(0, 0), (1, 2), (2, 4)]
    );
    assert_eq!(
        qtt_partial_fourier2d_common::interleaved_site_values(2, 1, 2),
        vec![1, 0, 0, 1]
    );
    assert!((qtt_function_utils::discrete_index_to_unit_interval(4, 8) - 0.375).abs() < 1e-12);
    assert!(
        (qtt_elementwise_product_utils::discrete_index_to_unit_interval(4, 8) - 0.375).abs()
            < 1e-12
    );
}

#[test]
fn csv_writers_keep_stable_headers() -> Result<(), Box<dyn Error>> {
    let scratch = scratch_dir("csv-contracts");

    let function_samples = scratch.join("function_samples.csv");
    qtt_function_utils::write_samples_csv(
        &function_samples,
        &[qtt_function_utils::SamplePoint {
            index: 1,
            x: 0.0,
            exact: 1.0,
            qtt: 1.0,
            abs_error: 0.0,
        }],
    )?;
    assert_header(&function_samples, "index,x,exact,qtt,abs_error");

    let fourier_samples = scratch.join("fourier_samples.csv");
    qtt_fourier_common::write_samples_csv(
        &fourier_samples,
        &[qtt_fourier_common::SamplePoint {
            index: 1,
            x: 0.0,
            k: 0.0,
            analytic_re: 1.0,
            analytic_im: 0.0,
            qtt_re: 1.0,
            qtt_im: 0.0,
            abs_error: 0.0,
        }],
    )?;
    assert_header(
        &fourier_samples,
        "index,x,k,analytic_re,analytic_im,qtt_re,qtt_im,abs_error",
    );

    let partial_samples = scratch.join("partial_fourier_samples.csv");
    qtt_partial_fourier2d_common::write_samples_csv(
        &partial_samples,
        &[qtt_partial_fourier2d_common::PartialFourier2dSamplePoint {
            k_index: 1,
            t_index: 1,
            source_x_index: 1,
            k: 0.0,
            t: 0.0,
            analytic_re: 1.0,
            analytic_im: 0.0,
            qtt_re: 1.0,
            qtt_im: 0.0,
            abs_error: 0.0,
        }],
    )?;
    assert_header(
        &partial_samples,
        "k_index,t_index,source_x_index,k,t,analytic_re,analytic_im,qtt_re,qtt_im,abs_error",
    );

    let affine_samples = scratch.join("affine_samples.csv");
    qtt_affine_common::write_samples_csv(
        &affine_samples,
        &[qtt_affine_common::AffineSamplePoint {
            x_index: 1,
            y_index: 1,
            x: 0,
            y: 0,
            source_u_periodic: 0,
            source_v: 0,
            source_exact: 1.0,
            periodic_exact: 1.0,
            periodic_qtt: 1.0,
            periodic_abs_error: 0.0,
            antiperiodic_exact: 1.0,
            antiperiodic_qtt: 1.0,
            antiperiodic_abs_error: 0.0,
            open_exact: 1.0,
            open_qtt: 1.0,
            open_abs_error: 0.0,
        }],
    )?;
    assert_header(
        &affine_samples,
        "x_index,y_index,x,y,source_u_periodic,source_v,source_exact,periodic_exact,periodic_qtt,periodic_abs_error,antiperiodic_exact,antiperiodic_qtt,antiperiodic_abs_error,open_exact,open_qtt,open_abs_error",
    );

    let sweep = scratch.join("integral_sweep.csv");
    qtt_integral_sweep_utils::write_sweep_csv(
        &sweep,
        &[qtt_integral_sweep_utils::IntegralSweepRow {
            r: 3,
            npoints: 8,
            integral: 1.0,
            exact_integral: 1.0,
            abs_error: 0.0,
            rank: 2,
        }],
    )?;
    assert_header(&sweep, "r,npoints,integral,exact_integral,abs_error,rank");

    let r_sweep = scratch.join("r_sweep.csv");
    qtt_r_sweep_utils::write_stats_csv(
        &r_sweep,
        &[qtt_r_sweep_utils::SweepStats {
            r: 3,
            npoints: 8,
            build_time_sec: 0.25,
            mean_abs_error: 0.0,
            max_abs_error: 0.0,
            rank: 2,
        }],
    )?;
    assert_header(
        &r_sweep,
        "r,npoints,build_time_sec,mean_abs_error,max_abs_error,rank",
    );

    Ok(())
}

#[test]
fn qtt_difference_kernel_demo_builds_periodic_mpo_and_exports_csv() -> Result<(), Box<dyn Error>> {
    let config = qtt_difference_kernel_common::DifferenceKernelTutorialConfig {
        bits: 3,
        tolerance: 1e-12,
        max_bond_dim: 32,
        maxiter: 20,
    };

    let (kernel, ranks, errors) = qtt_difference_kernel_common::build_kernel_qtt(&config)?;
    let kernel_tt = kernel.tensor_train();
    let mpo = qtt_difference_kernel_common::build_periodic_difference_kernel_mpo(&kernel_tt)?;
    let samples = qtt_difference_kernel_common::collect_samples(&kernel, &mpo, &config)?;
    let bond_dims =
        qtt_difference_kernel_common::collect_bond_dims(&kernel.link_dims(), &mpo.link_dims());

    assert!(!ranks.is_empty());
    assert!(!errors.is_empty());
    assert!(kernel.rank() > 0);
    assert_eq!(
        samples.len(),
        qtt_difference_kernel_common::point_count(config.bits).pow(2)
    );
    assert!(!bond_dims.is_empty());

    let max_source_error = samples
        .iter()
        .map(|sample| sample.source_abs_error)
        .fold(0.0_f64, f64::max);
    let max_mpo_error = samples
        .iter()
        .map(|sample| sample.abs_error)
        .fold(0.0_f64, f64::max);
    assert!(
        max_source_error < 1e-9,
        "unexpectedly large kernel QTT error: {max_source_error}"
    );
    assert!(
        max_mpo_error < 1e-9,
        "unexpectedly large difference-kernel MPO error: {max_mpo_error}"
    );

    let n = qtt_difference_kernel_common::point_count(config.bits);
    let wrapped = samples
        .iter()
        .find(|sample| sample.x == 0 && sample.xprime == 1)
        .expect("wrapped sample exists");
    assert_eq!(wrapped.difference, n - 1);
    assert!(
        (wrapped.kernel_exact - qtt_difference_kernel_common::kernel_value(n - 1, n)).abs() < 1e-12
    );

    let scratch = scratch_dir("qtt-difference-kernel");
    let samples_path = scratch.join("samples.csv");
    let bond_dims_path = scratch.join("bond_dims.csv");
    qtt_difference_kernel_common::write_samples_csv(&samples_path, &samples)?;
    qtt_difference_kernel_common::write_bond_dims_csv(&bond_dims_path, &bond_dims)?;

    assert_eq!(
        read_lines(&samples_path).first().map(String::as_str),
        Some("x_index,xprime_index,x,xprime,difference,source_exact,source_qtt,source_abs_error,kernel_exact,kernel_mpo,abs_error")
    );
    assert_eq!(
        read_lines(&bond_dims_path).first().map(String::as_str),
        Some("bond_index,kernel_qtt_bond_dim,difference_kernel_mpo_bond_dim")
    );

    Ok(())
}
