// Adaptive PartitionedTT patching benchmark.
//
// Run:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-partitionedtt --example benchmark_patching --release
//
// Optional args:
//   --x N
//   --y N
//   --components N
//   --max-bond-dim N
//   --rtol FLOAT
//   --repeats N

use std::hint::black_box;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_partitionedtt::{
    add_with_patching, PartitionedTT, PatchSplitStrategy, PatchingOptions, SubDomainTT,
    TensorTrain,
};

#[derive(Debug, Clone)]
struct Options {
    x_dim: usize,
    y_dim: usize,
    components: usize,
    max_bond_dim: usize,
    rtol: f64,
    repeats: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            x_dim: 24,
            y_dim: 16,
            components: 10,
            max_bond_dim: 3,
            rtol: 1.0e-8,
            repeats: 3,
        }
    }
}

#[derive(Debug, Clone)]
struct BenchmarkInput {
    tt: TensorTrain,
    x_index: DynIndex,
    y_index: DynIndex,
    reference: Vec<f64>,
    reference_norm: f64,
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    strategy_name: &'static str,
    elapsed: Duration,
    patches: usize,
    total_parameters: usize,
    max_patch_bond_dim: usize,
    relative_error: f64,
}

fn main() -> Result<()> {
    let options = parse_args()?;
    let input = make_anisotropic_gaussian_input(&options)?;

    println!("tensor4all PartitionedTT adaptive patching benchmark");
    println!(
        "config,x_dim={},y_dim={},components={},max_bond_dim={},rtol={:.3e},repeats={}",
        options.x_dim,
        options.y_dim,
        options.components,
        options.max_bond_dim,
        options.rtol,
        options.repeats
    );
    println!("strategy,patches,total_parameters,max_patch_bond_dim,relative_error,time_ms");

    let sequential_options = patching_options(
        &options,
        PatchSplitStrategy::Sequential,
        &[input.x_index.clone(), input.y_index.clone()],
    );
    let exact_gain_options = patching_options(
        &options,
        PatchSplitStrategy::ExactParameterGain,
        &[input.x_index.clone(), input.y_index.clone()],
    );

    run_case("sequential_xy", &input, &sequential_options, options.repeats)?;
    run_case("exact_gain", &input, &exact_gain_options, options.repeats)?;

    Ok(())
}

fn parse_args() -> Result<Options> {
    let mut options = Options::default();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: benchmark_patching [--x N] [--y N] [--components N] \
                     [--max-bond-dim N] [--rtol FLOAT] [--repeats N]"
                );
                std::process::exit(0);
            }
            "--x" => {
                i += 1;
                options.x_dim = parse_arg(&args, i, "--x")?;
            }
            "--y" => {
                i += 1;
                options.y_dim = parse_arg(&args, i, "--y")?;
            }
            "--components" => {
                i += 1;
                options.components = parse_arg(&args, i, "--components")?;
            }
            "--max-bond-dim" => {
                i += 1;
                options.max_bond_dim = parse_arg(&args, i, "--max-bond-dim")?;
            }
            "--rtol" => {
                i += 1;
                options.rtol = parse_arg(&args, i, "--rtol")?;
            }
            "--repeats" => {
                i += 1;
                options.repeats = parse_arg(&args, i, "--repeats")?;
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    if options.x_dim == 0 || options.y_dim == 0 || options.components == 0 {
        bail!("grid dimensions and component count must be positive");
    }
    if options.max_bond_dim == 0 {
        bail!("max-bond-dim must be positive");
    }
    if !options.rtol.is_finite() || options.rtol < 0.0 {
        bail!("rtol must be finite and non-negative");
    }
    if options.repeats == 0 {
        bail!("repeats must be positive");
    }

    Ok(options)
}

fn parse_arg<T>(args: &[String], index: usize, name: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    args.get(index)
        .with_context(|| format!("missing value for {name}"))?
        .parse::<T>()
        .with_context(|| format!("invalid value for {name}"))
}

fn patching_options(
    options: &Options,
    split_strategy: PatchSplitStrategy,
    patch_order: &[DynIndex],
) -> PatchingOptions {
    PatchingOptions {
        rtol: options.rtol,
        max_bond_dim: Some(options.max_bond_dim),
        patch_order: patch_order.to_vec(),
        split_strategy,
    }
}

fn run_case(
    strategy_name: &'static str,
    input: &BenchmarkInput,
    options: &PatchingOptions,
    repeats: usize,
) -> Result<()> {
    let mut best: Option<BenchmarkResult> = None;
    for _ in 0..repeats {
        let started = Instant::now();
        let patched =
            add_with_patching(vec![SubDomainTT::from_tt(input.tt.clone())], options)?;
        let elapsed = started.elapsed();
        let result = summarize_result(strategy_name, input, black_box(&patched), elapsed)?;
        if best
            .as_ref()
            .is_none_or(|current| result.elapsed < current.elapsed)
        {
            best = Some(result);
        }
    }

    let result = best.expect("repeats is validated as positive");
    println!(
        "{},{},{},{},{:.6e},{:.6}",
        result.strategy_name,
        result.patches,
        result.total_parameters,
        result.max_patch_bond_dim,
        result.relative_error,
        result.elapsed.as_secs_f64() * 1.0e3
    );

    Ok(())
}

fn summarize_result(
    strategy_name: &'static str,
    input: &BenchmarkInput,
    partitioned: &PartitionedTT,
    elapsed: Duration,
) -> Result<BenchmarkResult> {
    let total_parameters = partitioned_parameter_count(partitioned)?;
    let max_patch_bond_dim = partitioned
        .values()
        .map(|subdomain| subdomain.max_bond_dim())
        .max()
        .unwrap_or(0);
    let relative_error = dense_relative_error(input, partitioned)?;
    Ok(BenchmarkResult {
        strategy_name,
        elapsed,
        patches: partitioned.len(),
        total_parameters,
        max_patch_bond_dim,
        relative_error,
    })
}

fn make_anisotropic_gaussian_input(options: &Options) -> Result<BenchmarkInput> {
    let x_index = DynIndex::new_dyn(options.x_dim);
    let component_index = DynIndex::new_dyn(options.components);
    let y_index = DynIndex::new_dyn(options.y_dim);
    let mut left = vec![0.0; options.x_dim * options.components];
    let mut right = vec![0.0; options.components * options.y_dim];

    for component in 0..options.components {
        let center_x = deterministic_unit(component, 17) * (options.x_dim.saturating_sub(1) as f64);
        let center_y = deterministic_unit(component, 29) * (options.y_dim.saturating_sub(1) as f64);
        let narrow = 0.035 + 0.035 * deterministic_unit(component, 43);
        let wide = 0.16 + 0.16 * deterministic_unit(component, 61);
        let sigma_x = if component % 2 == 0 {
            narrow * options.x_dim as f64
        } else {
            wide * options.x_dim as f64
        };
        let sigma_y = if component % 2 == 0 {
            wide * options.y_dim as f64
        } else {
            narrow * options.y_dim as f64
        };
        let amplitude = 0.6 + deterministic_unit(component, 73);

        for x in 0..options.x_dim {
            left[x + options.x_dim * component] = amplitude * gaussian(x as f64, center_x, sigma_x);
        }
        for y in 0..options.y_dim {
            right[component + options.components * y] = gaussian(y as f64, center_y, sigma_y);
        }
    }

    let t0 = TensorDynLen::from_dense(vec![x_index.clone(), component_index.clone()], left)?;
    let t1 = TensorDynLen::from_dense(vec![component_index, y_index.clone()], right)?;
    let tt = TensorTrain::new(vec![t0, t1])?;
    let dense = tt.to_dense()?.to_vec::<f64>()?;
    let reference_norm = dense.iter().map(|value| value * value).sum::<f64>().sqrt();

    Ok(BenchmarkInput {
        tt,
        x_index,
        y_index,
        reference: dense,
        reference_norm,
    })
}

fn deterministic_unit(component: usize, salt: usize) -> f64 {
    let value = (component as u64)
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add((salt as u64).wrapping_mul(1_442_695_040_888_963_407));
    ((value >> 11) as f64) / ((1u64 << 53) as f64)
}

fn gaussian(x: f64, center: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1.0e-12);
    let z = (x - center) / sigma;
    (-0.5 * z * z).exp()
}

fn dense_relative_error(input: &BenchmarkInput, partitioned: &PartitionedTT) -> Result<f64> {
    let dense = partitioned.to_tensor_train()?.to_dense()?.to_vec::<f64>()?;
    if dense.len() != input.reference.len() {
        bail!(
            "dense result length mismatch: got {}, expected {}",
            dense.len(),
            input.reference.len()
        );
    }
    let diff_norm = dense
        .iter()
        .zip(&input.reference)
        .map(|(actual, expected)| {
            let diff = actual - expected;
            diff * diff
        })
        .sum::<f64>()
        .sqrt();
    if input.reference_norm == 0.0 {
        Ok(diff_norm)
    } else {
        Ok(diff_norm / input.reference_norm)
    }
}

fn partitioned_parameter_count(partitioned: &PartitionedTT) -> Result<usize> {
    partitioned.values().try_fold(0usize, |total, subdomain| {
        let patch_count = tensor_train_parameter_count(subdomain.data())?;
        total
            .checked_add(patch_count)
            .context("partitioned parameter count overflowed usize")
    })
}

fn tensor_train_parameter_count(tt: &TensorTrain) -> Result<usize> {
    tt.tensors().into_iter().try_fold(0usize, |total, tensor| {
        let tensor_count = tensor.dims().into_iter().try_fold(1usize, |acc, dim| {
            acc.checked_mul(dim)
                .context("tensor parameter count overflowed usize")
        })?;
        total
            .checked_add(tensor_count)
            .context("tensor train parameter count overflowed usize")
    })
}
