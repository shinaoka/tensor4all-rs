use std::env;
use std::error::Error;
use std::time::{Duration, Instant};

use tensor4all_core::{CudaExecutionContext, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

#[derive(Debug, Clone, Copy)]
struct Config {
    chain_length: usize,
    bond_dim: usize,
    iterations: usize,
}

fn parse_positive(name: &str, value: &str) -> Result<usize, Box<dyn Error>> {
    let value = value.parse::<usize>()?;
    if value == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(value)
}

fn env_setting(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match env::var(name) {
        Ok(value) => parse_positive(name, &value),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(error.into()),
    }
}

fn config() -> Result<Config, Box<dyn Error>> {
    let mut config = Config {
        chain_length: env_setting("CUDA_TREE_CHAIN_LENGTH", 4)?,
        bond_dim: env_setting("CUDA_TREE_BOND_DIM", 3)?,
        iterations: env_setting("CUDA_TREE_ITERATIONS", 10)?,
    };
    let args: Vec<_> = env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--chain-length" => config.chain_length = parse_positive(flag, value)?,
            "--bond-dim" => config.bond_dim = parse_positive(flag, value)?,
            "--iterations" => config.iterations = parse_positive(flag, value)?,
            _ => return Err(format!("unknown argument {flag}").into()),
        }
        index += 2;
    }
    Ok(config)
}

fn dense(indices: Vec<DynIndex>, seed: usize) -> Result<IdxTensor, Box<dyn Error>> {
    let elements = indices
        .iter()
        .map(IndexLike::dim)
        .try_fold(1usize, |size, dim| size.checked_mul(dim))
        .ok_or("tensor element count overflow")?;
    let values = (0..elements)
        .map(|offset| ((seed + offset) % 19) as f64 / 19.0 + 0.25)
        .collect();
    Ok(IdxTensor::from_dense(indices, values)?)
}

fn build_chain(config: Config) -> Result<TreeTN<IdxTensor, usize>, Box<dyn Error>> {
    let sites: Vec<_> = (0..config.chain_length)
        .map(|_| DynIndex::new_dyn(2))
        .collect();
    let bonds: Vec<_> = (0..config.chain_length.saturating_sub(1))
        .map(|_| DynIndex::new_dyn(config.bond_dim))
        .collect();
    let mut tensors = Vec::with_capacity(config.chain_length);
    for node in 0..config.chain_length {
        let mut indices = Vec::with_capacity(3);
        if let Some(left) = node.checked_sub(1).and_then(|index| bonds.get(index)) {
            indices.push(left.clone());
        }
        indices.push(
            sites
                .get(node)
                .cloned()
                .ok_or("site index missing while building chain")?,
        );
        if let Some(right) = bonds.get(node) {
            indices.push(right.clone());
        }
        tensors.push(dense(indices, node * 31 + 1)?);
    }
    Ok(TreeTN::from_tensors(
        tensors,
        (0..config.chain_length).collect(),
    )?)
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1.0e3
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = config()?;

    let started = Instant::now();
    let host_tree = build_chain(config)?;
    let host_setup = started.elapsed();

    let started = Instant::now();
    let context = CudaExecutionContext::new()?;
    let cuda_context_setup = started.elapsed();

    let started = Instant::now();
    let resident_tree = host_tree.upload_cuda(&context)?;
    let upload = started.elapsed();

    let started = Instant::now();
    let warmup = resident_tree.contract_to_tensor_cuda(&context)?;
    context.synchronize()?;
    std::hint::black_box(&warmup);
    let cuda_warmup = started.elapsed();

    let started = Instant::now();
    let mut resident_result = None;
    for _ in 0..config.iterations {
        let result = resident_tree.contract_to_tensor_cuda(&context)?;
        context.synchronize()?;
        std::hint::black_box(&result);
        resident_result = Some(result);
    }
    let cuda_steady = started.elapsed();
    let resident_result = resident_result.ok_or("no CUDA steady-state result")?;
    resident_result.validate_cuda_residency(&context)?;
    assert!(
        resident_result.to_vec::<f64>().is_err(),
        "resident CUDA output unexpectedly allowed host extraction"
    );

    let started = Instant::now();
    let downloaded = resident_result.download(&context)?;
    let download = started.elapsed();

    let started = Instant::now();
    let mut cpu_result = None;
    for _ in 0..config.iterations {
        let result = host_tree.contract_to_tensor()?;
        std::hint::black_box(&result);
        cpu_result = Some(result);
    }
    let cpu_steady = started.elapsed();
    let cpu_result = cpu_result.ok_or("no CPU steady-state result")?;

    assert_eq!(downloaded.indices(), cpu_result.indices());
    let residual = downloaded.sub(&cpu_result)?.maxabs()?;
    assert!(
        residual <= 1.0e-10,
        "CUDA/CPU residual {residual} exceeds 1e-10"
    );

    println!(
        "config chain_length={} bond_dim={} iterations={}",
        config.chain_length, config.bond_dim, config.iterations
    );
    println!(
        "device name={:?} visible_cuda_ordinal={} runtime=cuda-eager",
        context.device_name(),
        context.ordinal()
    );
    println!("residual_max_abs={residual:.6e}");
    println!("host_setup_ms={:.6}", milliseconds(host_setup));
    println!(
        "cuda_context_setup_ms={:.6}",
        milliseconds(cuda_context_setup)
    );
    println!("upload_ms={:.6}", milliseconds(upload));
    println!("cuda_warmup_ms={:.6}", milliseconds(cuda_warmup));
    println!(
        "cuda_steady_sync_ms_total={:.6} cuda_steady_sync_ms_per_iter={:.6}",
        milliseconds(cuda_steady),
        milliseconds(cuda_steady) / config.iterations as f64
    );
    println!("download_ms={:.6}", milliseconds(download));
    println!(
        "cpu_steady_ms_total={:.6} cpu_steady_ms_per_iter={:.6}",
        milliseconds(cpu_steady),
        milliseconds(cpu_steady) / config.iterations as f64
    );

    Ok(())
}
