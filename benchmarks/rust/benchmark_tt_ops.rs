// Benchmark TensorTrain-level operations against ITensorMPS.jl.
//
// Run:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-itensorlike --example benchmark_tt_ops --release
//
// Optional args:
//   --L N
//   --d N
//   --zipup-L N
//   --chis 4,8,16,32
//   --warm-up-time SECONDS
//   --measurement-time SECONDS
//   --min-samples N
//   --inner-only

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use num_complex::Complex64;
use tensor4all_core::{
    contract, contract_pair, print_and_reset_pairwise_contract_profile,
    reset_pairwise_contract_profile, AnyScalar, DynIndex, SvdTruncationPolicy,
    TensorContractionLike, TensorDynLen,
};
use tensor4all_itensorlike::{CanonicalForm, ContractOptions, TensorTrain};
use tenferro::{DotGeneralConfig, Tensor, TypedTensor};
use tenferro_ad::{EagerRuntime, EagerTensor};
use tenferro_cpu::CpuBackend;

#[derive(Debug, Clone)]
struct Options {
    length: usize,
    phys_dim: usize,
    zipup_length: usize,
    chis: Vec<usize>,
    warmup_seconds: f64,
    measurement_seconds: f64,
    min_samples: usize,
    skip_zipup: bool,
    inner_only: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            length: 32,
            phys_dim: 2,
            zipup_length: 10,
            chis: vec![4, 8, 16, 32, 64],
            warmup_seconds: 1.0,
            measurement_seconds: 2.0,
            min_samples: 10,
            skip_zipup: false,
            inner_only: false,
        }
    }
}

fn parse_args() -> Result<Options> {
    let mut opts = Options::default();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: benchmark_tt_ops [--L N] [--d N] [--zipup-L N] \
                     [--chis LIST] [--warm-up-time SEC] [--measurement-time SEC] \
                     [--min-samples N] [--no-zipup] [--inner-only]"
                );
                std::process::exit(0);
            }
            "--L" => {
                i += 1;
                opts.length = parse_arg(&args, i, "--L")?;
            }
            "--d" => {
                i += 1;
                opts.phys_dim = parse_arg(&args, i, "--d")?;
            }
            "--zipup-L" => {
                i += 1;
                opts.zipup_length = parse_arg(&args, i, "--zipup-L")?;
            }
            "--chis" => {
                i += 1;
                let value = args
                    .get(i)
                    .with_context(|| "missing value for --chis".to_string())?;
                opts.chis = value
                    .split(',')
                    .map(|part| part.parse::<usize>())
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--warm-up-time" => {
                i += 1;
                opts.warmup_seconds = parse_arg(&args, i, "--warm-up-time")?;
            }
            "--measurement-time" => {
                i += 1;
                opts.measurement_seconds = parse_arg(&args, i, "--measurement-time")?;
            }
            "--min-samples" => {
                i += 1;
                opts.min_samples = parse_arg(&args, i, "--min-samples")?;
            }
            "--no-zipup" => {
                opts.skip_zipup = true;
            }
            "--inner-only" => {
                opts.inner_only = true;
                opts.skip_zipup = true;
            }
            other => bail!("unknown argument: {other}"),
        }
        i += 1;
    }

    if opts.length == 0 || opts.zipup_length == 0 || opts.phys_dim == 0 {
        bail!("lengths and physical dimension must be positive");
    }
    if opts.chis.is_empty() || opts.chis.contains(&0) {
        bail!("all bond dimensions must be positive");
    }
    if opts.warmup_seconds < 0.0 || opts.measurement_seconds < 0.0 {
        bail!("timing windows must be nonnegative");
    }
    if opts.min_samples == 0 {
        bail!("min-samples must be positive");
    }

    Ok(opts)
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

fn deterministic_value(idx: usize, seed: usize) -> Complex64 {
    let real = ((idx * 17 + seed * 13 + 3) % 97) as f64 / 97.0 - 0.5;
    let imag = ((idx * 29 + seed * 7 + 5) % 89) as f64 / 89.0 - 0.5;
    Complex64::new(real, imag)
}

fn deterministic_tensor(indices: Vec<DynIndex>, seed: usize) -> Result<TensorDynLen> {
    let len = indices.iter().map(|index| index.size()).product::<usize>();
    let data = (0..len)
        .map(|idx| deterministic_value(idx, seed))
        .collect::<Vec<_>>();
    TensorDynLen::from_dense(indices, data).map_err(anyhow::Error::from)
}

fn deterministic_native_tensor(shape: Vec<usize>, seed: usize) -> Tensor {
    let len = shape.iter().product::<usize>();
    let data = (0..len)
        .map(|idx| deterministic_value(idx, seed))
        .collect::<Vec<_>>();
    Tensor::C64(TypedTensor::from_vec_col_major(shape, data))
}

fn make_sites(length: usize, phys_dim: usize) -> Vec<DynIndex> {
    (0..length)
        .map(|_| DynIndex::new_dyn(phys_dim))
        .collect()
}

fn make_mps(sites: &[DynIndex], chi: usize, seed_offset: usize) -> Result<TensorTrain> {
    let length = sites.len();
    let links = (0..length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(chi))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(length);

    for site in 0..length {
        let mut indices = Vec::new();
        if site > 0 {
            indices.push(links[site - 1].clone());
        }
        indices.push(sites[site].clone());
        if site + 1 < length {
            indices.push(links[site].clone());
        }
        tensors.push(deterministic_tensor(indices, seed_offset + site + 1)?);
    }

    Ok(TensorTrain::new(tensors)?)
}

fn make_native_mps_t4a_shapes(
    length: usize,
    phys_dim: usize,
    chi: usize,
    seed_offset: usize,
) -> Vec<Tensor> {
    (0..length)
        .map(|site| {
            let mut shape = Vec::new();
            if site > 0 {
                shape.push(chi);
            }
            shape.push(phys_dim);
            if site + 1 < length {
                shape.push(chi);
            }
            deterministic_native_tensor(shape, seed_offset + site + 1)
        })
        .collect()
}

fn eager_mps_tensors(ctx: &Arc<EagerRuntime>, tensors: Vec<Tensor>) -> Vec<EagerTensor> {
    tensors
        .into_iter()
        .map(|tensor| EagerTensor::from_tensor_in(tensor, Arc::clone(ctx)))
        .collect()
}

#[derive(Debug, Clone)]
struct RawEagerInnerConfigs {
    first_site: DotGeneralConfig,
    env_bra: DotGeneralConfig,
    tmp_ket: DotGeneralConfig,
}

impl RawEagerInnerConfigs {
    fn new() -> Self {
        Self {
            first_site: DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
            env_bra: DotGeneralConfig {
                lhs_contracting_dims: vec![0],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
            tmp_ket: DotGeneralConfig {
                lhs_contracting_dims: vec![0, 1],
                rhs_contracting_dims: vec![0, 1],
                lhs_batch_dims: vec![],
                rhs_batch_dims: vec![],
            },
        }
    }
}

fn maybe_snapshot_output(tensor: &EagerTensor, snapshot_outputs: bool) {
    if snapshot_outputs {
        black_box(tensor.data().clone());
    }
}

fn raw_eager_inner_t4a_shapes(
    bra: &[EagerTensor],
    ket: &[EagerTensor],
    configs: &RawEagerInnerConfigs,
    snapshot_outputs: bool,
) -> Result<Complex64> {
    if bra.len() != ket.len() {
        bail!(
            "raw eager inputs must have the same length: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok(Complex64::new(0.0, 0.0));
    }

    let mut env = bra[0]
        .dot_general_with_conj(&ket[0], &configs.first_site, true, false)
        .context("failed raw eager first-site contraction")?;
    maybe_snapshot_output(&env, snapshot_outputs);

    for site in 1..bra.len() {
        env = env
            .dot_general_with_conj(&bra[site], &configs.env_bra, false, true)
            .with_context(|| format!("failed raw eager env-bra contraction at site {site}"))?;
        maybe_snapshot_output(&env, snapshot_outputs);

        env = env
            .dot_general_with_conj(&ket[site], &configs.tmp_ket, false, false)
            .with_context(|| format!("failed raw eager tmp-ket contraction at site {site}"))?;
        maybe_snapshot_output(&env, snapshot_outputs);
    }

    let scalar = env
        .data()
        .as_slice::<Complex64>()
        .context("raw eager inner output is not Complex64")?
        .first()
        .copied()
        .context("raw eager inner output is empty")?;
    Ok(scalar)
}

fn make_mpo(
    input_sites: &[DynIndex],
    output_sites: &[DynIndex],
    chi: usize,
    seed_offset: usize,
) -> Result<TensorTrain> {
    if input_sites.len() != output_sites.len() {
        bail!("input/output site lengths must match");
    }

    let length = input_sites.len();
    let links = (0..length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(chi))
        .collect::<Vec<_>>();
    let mut tensors = Vec::with_capacity(length);

    for site in 0..length {
        let mut indices = Vec::new();
        if site > 0 {
            indices.push(links[site - 1].clone());
        }
        indices.push(input_sites[site].clone());
        indices.push(output_sites[site].clone());
        if site + 1 < length {
            indices.push(links[site].clone());
        }
        tensors.push(deterministic_tensor(indices, seed_offset + site + 1)?);
    }

    Ok(TensorTrain::new(tensors)?)
}

fn run_for_seconds<T, F>(
    warmup_seconds: f64,
    measurement_seconds: f64,
    min_samples: usize,
    mut f: F,
) -> Result<(T, Vec<f64>)>
where
    F: FnMut() -> Result<T>,
{
    let mut sink = f()?;

    let warmup_start = Instant::now();
    while warmup_start.elapsed().as_secs_f64() < warmup_seconds {
        sink = f()?;
        black_box(&sink);
    }

    let mut times_ms = Vec::new();
    let measurement_start = Instant::now();
    while measurement_start.elapsed().as_secs_f64() < measurement_seconds
        || times_ms.len() < min_samples
    {
        let start = Instant::now();
        sink = f()?;
        black_box(&sink);
        times_ms.push(start.elapsed().as_secs_f64() * 1.0e3);
    }

    Ok((sink, times_ms))
}

fn stats_ms(times: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let min = sorted[0];
    let max = *sorted.last().expect("non-empty timings");
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let median = if sorted.len().is_multiple_of(2) {
        let hi = sorted.len() / 2;
        0.5 * (sorted[hi - 1] + sorted[hi])
    } else {
        sorted[sorted.len() / 2]
    };
    (min, median, mean, max)
}

fn print_result(case: &str, params: &str, times: &[f64], value: &str, max_bond: usize) {
    let (min, median, mean, max) = stats_ms(times);
    println!(
        "{case},{params},{},{min:.6},{median:.6},{mean:.6},{max:.6},{max_bond},{value}",
        times.len()
    );
}

fn inner_sitewise_pair_no_sim(bra: &TensorTrain, ket: &TensorTrain) -> Result<AnyScalar> {
    if bra.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok(AnyScalar::new_real(0.0));
    }

    let a0_conj = bra.tensor(0)?.conj();
    let mut env = contract_pair(&a0_conj, ket.tensor(0)?)
        .context("failed to contract leftmost site tensors")?;

    for site in 1..bra.len() {
        let ai_conj = bra.tensor(site)?.conj();
        env = contract_pair(&env, &ai_conj)
            .with_context(|| format!("failed to contract environment with site {site}"))?;
        env = contract_pair(&env, ket.tensor(site)?)
            .with_context(|| format!("failed to contract ket tensor at site {site}"))?;
    }

    env.sum().map_err(anyhow::Error::from)
}

fn preconjugate_sites(bra: &TensorTrain) -> Result<Vec<TensorDynLen>> {
    (0..bra.len())
        .map(|site| Ok(bra.tensor(site)?.conj()))
        .collect()
}

fn inner_sitewise_pair_preconj_no_sim(
    bra_conj: &[TensorDynLen],
    ket: &TensorTrain,
) -> Result<AnyScalar> {
    if bra_conj.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra_conj.len(),
            ket.len()
        );
    }
    if bra_conj.is_empty() {
        return Ok(AnyScalar::new_real(0.0));
    }

    let mut env = contract_pair(&bra_conj[0], ket.tensor(0)?)
        .context("failed to contract leftmost site tensors")?;

    for (site, bra_site) in bra_conj.iter().enumerate().skip(1) {
        env = contract_pair(&env, bra_site)
            .with_context(|| format!("failed to contract environment with site {site}"))?;
        env = contract_pair(&env, ket.tensor(site)?)
            .with_context(|| format!("failed to contract ket tensor at site {site}"))?;
    }

    env.sum().map_err(anyhow::Error::from)
}

fn inner_sitewise_nary_no_sim(bra: &TensorTrain, ket: &TensorTrain) -> Result<AnyScalar> {
    if bra.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok(AnyScalar::new_real(0.0));
    }

    let a0_conj = bra.tensor(0)?.conj();
    let mut env = contract_pair(&a0_conj, ket.tensor(0)?)
        .context("failed to contract leftmost site tensors")?;

    for site in 1..bra.len() {
        let ai_conj = bra.tensor(site)?.conj();
        env = contract(&[&env, &ai_conj, ket.tensor(site)?])
            .with_context(|| format!("failed to contract three-tensor environment at site {site}"))?;
    }

    env.sum().map_err(anyhow::Error::from)
}

fn inner_sitewise_binary_contract_no_sim(bra: &TensorTrain, ket: &TensorTrain) -> Result<AnyScalar> {
    if bra.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok(AnyScalar::new_real(0.0));
    }

    let a0_conj = bra.tensor(0)?.conj();
    let mut env =
        contract(&[&a0_conj, ket.tensor(0)?]).context("failed to contract leftmost site tensors")?;

    for site in 1..bra.len() {
        let ai_conj = bra.tensor(site)?.conj();
        env = contract(&[&env, &ai_conj])
            .with_context(|| format!("failed to contract environment with site {site}"))?;
        env = contract(&[&env, ket.tensor(site)?])
            .with_context(|| format!("failed to contract ket tensor at site {site}"))?;
    }

    env.sum().map_err(anyhow::Error::from)
}

#[derive(Debug, Default)]
struct InnerBreakdown {
    conj_ms: f64,
    first_contract_ms: f64,
    env_contract_ms: f64,
    ket_contract_ms: f64,
    nary_contract_ms: f64,
    sum_ms: f64,
}

fn inner_sitewise_pair_breakdown_no_sim(
    bra: &TensorTrain,
    ket: &TensorTrain,
) -> Result<(AnyScalar, InnerBreakdown)> {
    if bra.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok((AnyScalar::new_real(0.0), InnerBreakdown::default()));
    }

    let mut breakdown = InnerBreakdown::default();

    let started = Instant::now();
    let a0_conj = bra.tensor(0)?.conj();
    breakdown.conj_ms += started.elapsed().as_secs_f64() * 1.0e3;

    let started = Instant::now();
    let mut env = contract_pair(&a0_conj, ket.tensor(0)?)
        .context("failed to contract leftmost site tensors")?;
    breakdown.first_contract_ms += started.elapsed().as_secs_f64() * 1.0e3;

    for site in 1..bra.len() {
        let started = Instant::now();
        let ai_conj = bra.tensor(site)?.conj();
        breakdown.conj_ms += started.elapsed().as_secs_f64() * 1.0e3;

        let started = Instant::now();
        env = contract_pair(&env, &ai_conj)
            .with_context(|| format!("failed to contract environment with site {site}"))?;
        breakdown.env_contract_ms += started.elapsed().as_secs_f64() * 1.0e3;

        let started = Instant::now();
        env = contract_pair(&env, ket.tensor(site)?)
            .with_context(|| format!("failed to contract ket tensor at site {site}"))?;
        breakdown.ket_contract_ms += started.elapsed().as_secs_f64() * 1.0e3;
    }

    let started = Instant::now();
    let value = env.sum()?;
    breakdown.sum_ms += started.elapsed().as_secs_f64() * 1.0e3;
    Ok((value, breakdown))
}

fn inner_sitewise_nary_breakdown_no_sim(
    bra: &TensorTrain,
    ket: &TensorTrain,
) -> Result<(AnyScalar, InnerBreakdown)> {
    if bra.len() != ket.len() {
        bail!(
            "Tensor trains must have the same length for inner product: {} vs {}",
            bra.len(),
            ket.len()
        );
    }
    if bra.is_empty() {
        return Ok((AnyScalar::new_real(0.0), InnerBreakdown::default()));
    }

    let mut breakdown = InnerBreakdown::default();

    let started = Instant::now();
    let a0_conj = bra.tensor(0)?.conj();
    breakdown.conj_ms += started.elapsed().as_secs_f64() * 1.0e3;

    let started = Instant::now();
    let mut env = contract_pair(&a0_conj, ket.tensor(0)?)
        .context("failed to contract leftmost site tensors")?;
    breakdown.first_contract_ms += started.elapsed().as_secs_f64() * 1.0e3;

    for site in 1..bra.len() {
        let started = Instant::now();
        let ai_conj = bra.tensor(site)?.conj();
        breakdown.conj_ms += started.elapsed().as_secs_f64() * 1.0e3;

        let started = Instant::now();
        env = contract(&[&env, &ai_conj, ket.tensor(site)?])
            .with_context(|| format!("failed to contract three-tensor environment at site {site}"))?;
        breakdown.nary_contract_ms += started.elapsed().as_secs_f64() * 1.0e3;
    }

    let started = Instant::now();
    let value = env.sum()?;
    breakdown.sum_ms += started.elapsed().as_secs_f64() * 1.0e3;
    Ok((value, breakdown))
}

fn print_inner_breakdown(kind: &str, params: &str, value: AnyScalar, breakdown: &InnerBreakdown) {
    eprintln!(
        "inner_breakdown,{kind},{params},conj_ms={:.6},first_contract_ms={:.6},env_contract_ms={:.6},ket_contract_ms={:.6},nary_contract_ms={:.6},sum_ms={:.6},value={}",
        breakdown.conj_ms,
        breakdown.first_contract_ms,
        breakdown.env_contract_ms,
        breakdown.ket_contract_ms,
        breakdown.nary_contract_ms,
        breakdown.sum_ms,
        format_scalar(value),
    );
}

fn conj_sites(tt: &TensorTrain) -> Result<usize> {
    let mut total_elements = 0usize;
    for site in 0..tt.len() {
        let tensor = tt.tensor(site)?.conj();
        total_elements += tensor.dims().iter().product::<usize>();
        black_box(&tensor);
    }
    Ok(total_elements)
}

fn format_scalar(value: impl Into<tensor4all_core::AnyScalar>) -> String {
    let value = value.into();
    if let Some(z) = value.as_c64() {
        format!("{:.12e}+{:.12e}im", z.re, z.im)
    } else {
        format!("{:.12e}", value.real())
    }
}

fn main() -> Result<()> {
    let opts = parse_args()?;

    println!("tensor4all TensorTrain ops benchmark");
    println!("  L:                {}", opts.length);
    println!("  d:                {}", opts.phys_dim);
    println!("  zipup L:          {}", opts.zipup_length);
    println!(
        "  chis:             {}",
        opts.chis
            .iter()
            .map(|chi| chi.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("  warm-up time:     {}", opts.warmup_seconds);
    println!("  measurement time: {}", opts.measurement_seconds);
    println!("  min samples:      {}", opts.min_samples);
    println!("  skip zipup:       {}", opts.skip_zipup);
    println!("  inner only:       {}", opts.inner_only);
    println!();
    println!("case,params,samples,min_ms,median_ms,mean_ms,max_ms,max_bond,value");

    for &chi in &opts.chis {
        let sites = make_sites(opts.length, opts.phys_dim);
        let bra = make_mps(&sites, chi, 0)?;
        let ket = make_mps(&sites, chi, opts.length)?;
        let bra_conj = preconjugate_sites(&bra)?;
        let raw_ctx = EagerRuntime::with_cpu_backend(CpuBackend::with_threads(1));
        let raw_bra = eager_mps_tensors(
            &raw_ctx,
            make_native_mps_t4a_shapes(opts.length, opts.phys_dim, chi, 0),
        );
        let raw_ket = eager_mps_tensors(
            &raw_ctx,
            make_native_mps_t4a_shapes(opts.length, opts.phys_dim, chi, opts.length),
        );
        let raw_configs = RawEagerInnerConfigs::new();
        let mps_params = format!("L_{}_chi_{}_d_{}", opts.length, chi, opts.phys_dim);

        reset_pairwise_contract_profile();
        let profiled_inner_value = bra.inner(&ket)?;
        black_box(profiled_inner_value);
        eprintln!("pairwise_profile,current_inner,{mps_params}");
        print_and_reset_pairwise_contract_profile();

        let (pair_profile_value, pair_breakdown) =
            inner_sitewise_pair_breakdown_no_sim(&bra, &ket)?;
        print_inner_breakdown(
            "sitewise_pair_no_sim_once",
            &mps_params,
            pair_profile_value,
            &pair_breakdown,
        );
        let (nary_profile_value, nary_breakdown) =
            inner_sitewise_nary_breakdown_no_sim(&bra, &ket)?;
        print_inner_breakdown(
            "sitewise_nary_no_sim_once",
            &mps_params,
            nary_profile_value,
            &nary_breakdown,
        );

        let (inner_value, inner_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || Ok(bra.inner(black_box(&ket))?),
        )?;
        print_result(
            "tensor4all_inner_mps",
            &mps_params,
            &inner_times,
            &format_scalar(inner_value),
            0,
        );

        let (raw_eager_value, raw_eager_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || {
                raw_eager_inner_t4a_shapes(
                    black_box(&raw_bra),
                    black_box(&raw_ket),
                    black_box(&raw_configs),
                    false,
                )
            },
        )?;
        print_result(
            "tenferro_raw_eager_inner_t4a_shapes",
            &mps_params,
            &raw_eager_times,
            &format_scalar(raw_eager_value),
            0,
        );

        let (raw_eager_snapshot_value, raw_eager_snapshot_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || {
                raw_eager_inner_t4a_shapes(
                    black_box(&raw_bra),
                    black_box(&raw_ket),
                    black_box(&raw_configs),
                    true,
                )
            },
        )?;
        print_result(
            "tenferro_raw_eager_inner_t4a_shapes_snapshot_outputs",
            &mps_params,
            &raw_eager_snapshot_times,
            &format_scalar(raw_eager_snapshot_value),
            0,
        );

        let (preconj_pair_value, preconj_pair_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || inner_sitewise_pair_preconj_no_sim(black_box(&bra_conj), black_box(&ket)),
        )?;
        print_result(
            "tensor4all_inner_mps_sitewise_pair_preconj_no_sim",
            &mps_params,
            &preconj_pair_times,
            &format_scalar(preconj_pair_value),
            0,
        );

        if opts.inner_only {
            continue;
        }

        let (sitewise_pair_value, sitewise_pair_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || inner_sitewise_pair_no_sim(black_box(&bra), black_box(&ket)),
        )?;
        print_result(
            "tensor4all_inner_mps_sitewise_pair_no_sim",
            &mps_params,
            &sitewise_pair_times,
            &format_scalar(sitewise_pair_value),
            0,
        );

        let (sitewise_nary_value, sitewise_nary_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || inner_sitewise_nary_no_sim(black_box(&bra), black_box(&ket)),
        )?;
        print_result(
            "tensor4all_inner_mps_sitewise_nary_no_sim",
            &mps_params,
            &sitewise_nary_times,
            &format_scalar(sitewise_nary_value),
            0,
        );

        let (binary_contract_value, binary_contract_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || inner_sitewise_binary_contract_no_sim(black_box(&bra), black_box(&ket)),
        )?;
        print_result(
            "tensor4all_inner_mps_sitewise_binary_contract_no_sim",
            &mps_params,
            &binary_contract_times,
            &format_scalar(binary_contract_value),
            0,
        );

        let (conj_tt, conj_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || Ok(black_box(&bra).conj()),
        )?;
        print_result(
            "tensor4all_conj_mps",
            &mps_params,
            &conj_times,
            "ok",
            conj_tt.maxbonddim(),
        );

        let (conj_elements, conj_sites_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || conj_sites(black_box(&bra)),
        )?;
        print_result(
            "tensor4all_conj_sites_mps",
            &mps_params,
            &conj_sites_times,
            &conj_elements.to_string(),
            0,
        );

        let (sum, directsum_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || Ok(bra.add(black_box(&ket))?),
        )?;
        let sum_norm = sum.inner(&sum)?;
        print_result(
            "tensor4all_directsum_mps",
            &mps_params,
            &directsum_times,
            &format_scalar(sum_norm),
            sum.maxbonddim(),
        );

        if opts.skip_zipup {
            continue;
        }

        let zipup_sites_in = make_sites(opts.zipup_length, opts.phys_dim);
        let zipup_sites_mid = make_sites(opts.zipup_length, opts.phys_dim);
        let zipup_sites_out = make_sites(opts.zipup_length, opts.phys_dim);
        let mut mpo_a = make_mpo(&zipup_sites_in, &zipup_sites_mid, chi, 2 * opts.length)?;
        let mut mpo_b = make_mpo(
            &zipup_sites_mid,
            &zipup_sites_out,
            chi,
            2 * opts.length + opts.zipup_length,
        )?;
        mpo_a.orthogonalize_with(0, CanonicalForm::Unitary)?;
        mpo_b.orthogonalize_with(0, CanonicalForm::Unitary)?;

        let zipup_options = ContractOptions::zipup()
            .with_svd_policy(SvdTruncationPolicy::new(0.0))
            .with_max_rank(chi);
        let zipup_params = format!(
            "L_{}_chi_{}_d_{}_maxdim_{}",
            opts.zipup_length, chi, opts.phys_dim, chi
        );
        let (zipup_result, zipup_times) = run_for_seconds(
            opts.warmup_seconds,
            opts.measurement_seconds,
            opts.min_samples,
            || Ok(mpo_a.contract(black_box(&mpo_b), black_box(&zipup_options))?),
        )?;
        let zipup_norm = zipup_result.inner(&zipup_result)?;
        print_result(
            "tensor4all_zipup_mpo_prepared",
            &zipup_params,
            &zipup_times,
            &format_scalar(zipup_norm),
            zipup_result.maxbonddim(),
        );
    }

    Ok(())
}
