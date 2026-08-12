// Benchmark non-AD IdxTensor vector-space operations.
//
// Run:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-core --example benchmark_tensor_ops --release
//
// Optional args:
//   cargo run -p tensor4all-core --example benchmark_tensor_ops --release -- <repeats> <dim1> <dim2> ...
//
// Example matching a two-site local tensor with small bonds:
//   RAYON_NUM_THREADS=1 cargo run -p tensor4all-core --example benchmark_tensor_ops --release -- 20000 6 2 2 6

use std::hint::black_box;
use std::time::Instant;

use anyhow::{bail, Result};
use num_complex::Complex64;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensor4all_core::{AnyScalar, DynIndex, TensorContractionLike, IdxTensor};

fn parse_args() -> Result<(usize, Vec<usize>)> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let repeats = args
        .first()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(20_000);
    if repeats == 0 {
        bail!("repeats must be greater than zero");
    }

    let dims = if args.len() > 1 {
        args[1..]
            .iter()
            .map(|value| value.parse::<usize>())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        vec![6, 2, 2, 6]
    };
    if dims.is_empty() || dims.contains(&0) {
        bail!("all dimensions must be positive");
    }
    Ok((repeats, dims))
}

fn make_indices(dims: &[usize]) -> Vec<DynIndex> {
    dims.iter().map(|&dim| DynIndex::new_dyn(dim)).collect()
}

fn elapsed_seconds<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let started = Instant::now();
    let result = f();
    (started.elapsed().as_secs_f64(), result)
}

fn main() -> Result<()> {
    let (repeats, dims) = parse_args()?;
    let element_count = dims.iter().product::<usize>();
    let indices = make_indices(&dims);
    let mut rng = StdRng::seed_from_u64(0x5EED_1234);
    let a = IdxTensor::random::<Complex64, _>(&mut rng, indices.clone())?;
    let b = IdxTensor::random::<Complex64, _>(&mut rng, indices)?;
    let alpha = AnyScalar::new_complex(0.7, -0.2);
    let beta = AnyScalar::new_complex(-0.3, 0.4);

    // Warm up caches and allocator paths.
    for _ in 0..32 {
        black_box(a.inner_product(&b)?);
        black_box(a.norm()?);
        black_box(a.axpby(alpha.clone(), &b, beta.clone())?);
        black_box(a.conj().contract_pair(&b)?.sum()?);
    }

    println!("=== IdxTensor non-AD tensor ops benchmark ===");
    println!(
        "dims={dims:?} elements={element_count} repeats={repeats} dtype=Complex64"
    );

    let (inner_seconds, inner_checksum) = elapsed_seconds(|| -> Result<Complex64> {
        let mut checksum = Complex64::new(0.0, 0.0);
        for _ in 0..repeats {
            let value: Complex64 = black_box(a.inner_product(black_box(&b))?).into();
            checksum += value;
        }
        Ok(checksum)
    });
    let inner_checksum = inner_checksum?;
    println!(
        "inner_seconds = {:.6} per_call_us = {:.3} checksum = {:.6e}+{:.6e}im",
        inner_seconds,
        inner_seconds * 1.0e6 / repeats as f64,
        inner_checksum.re,
        inner_checksum.im,
    );

    let (norm_seconds, norm_checksum) = elapsed_seconds(|| -> Result<f64> {
        let mut checksum = 0.0;
        for _ in 0..repeats {
            checksum += black_box(a.norm()?);
        }
        Ok(checksum)
    });
    let norm_checksum = norm_checksum?;
    println!(
        "norm_seconds = {:.6} per_call_us = {:.3} checksum = {:.6e}",
        norm_seconds,
        norm_seconds * 1.0e6 / repeats as f64,
        norm_checksum,
    );

    let (axpby_seconds, axpby_checksum) = elapsed_seconds(|| -> Result<f64> {
        let mut checksum = 0.0;
        for _ in 0..repeats {
            let out = black_box(a.axpby(alpha.clone(), black_box(&b), beta.clone())?);
            checksum += black_box(out.maxabs()?);
        }
        Ok(checksum)
    });
    println!(
        "axpby_seconds = {:.6} per_call_us = {:.3} checksum = {:.6e}",
        axpby_seconds,
        axpby_seconds * 1.0e6 / repeats as f64,
        axpby_checksum?,
    );

    let (conj_contract_seconds, conj_contract_checksum) = elapsed_seconds(|| -> Result<Complex64> {
        let mut checksum = Complex64::new(0.0, 0.0);
        for _ in 0..repeats {
            let value: Complex64 = black_box(a.conj().contract_pair(black_box(&b))?.sum()?).into();
            checksum += value;
        }
        Ok(checksum)
    });
    let conj_contract_checksum = conj_contract_checksum?;
    println!(
        "conj_contract_sum_seconds = {:.6} per_call_us = {:.3} checksum = {:.6e}+{:.6e}im",
        conj_contract_seconds,
        conj_contract_seconds * 1.0e6 / repeats as f64,
        conj_contract_checksum.re,
        conj_contract_checksum.im,
    );

    Ok(())
}
