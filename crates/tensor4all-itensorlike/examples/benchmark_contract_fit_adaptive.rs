//! Benchmark: adaptive low-rank FIT contraction vs zip-up vs zip-up-init FIT.
//!
//! Compare three ways to contract two MPOs `A` and `B` in the regime
//! `χ_A ≈ χ_B ≫ χ_C(τ)` (moderately large input bonds, compressible product):
//!
//! 1. Zip-up contraction (exact, untruncated product).
//! 2. FIT with the zip-up initializer (`FitInitializer::ZipUp`), the previous
//!    default that materializes the χ_A·χ_B product during initialization.
//! 3. FIT with the low-rank random initializer (`FitInitializer::LowRankRandom`,
//!    the new default), `max_bond_dim = None`, tolerance-driven adaptive growth.
//!
//! `A` is a generic random bond-`χ` MPO. `B` is the identity operator padded
//! to physical bond `χ` (rank-1 content), so `A·B = A` needs an exact product
//! rank equal to `A`'s bond while the naïve product bond would be `χ²`. The
//! fit tolerance prunes `A`'s singular tail, landing at `χ_C(τ) < χ`.
//!
//! Run:
//! ```sh
//! RAYON_NUM_THREADS=1 BLAS_NUM_THREADS=1 cargo run -p tensor4all-itensorlike \
//!   --example benchmark_contract_fit_adaptive --release -- 12 32 2 3
//! ```
//! Arguments: `length` `chi` `phys` `nsweeps`.

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use anyhow::Result;
use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_itensorlike::{ContractOptions, FitInitializer, TensorTrain};

fn random_mpo(
    length: usize,
    input: &[DynIndex],
    output: &[DynIndex],
    links: &[DynIndex],
    rng: &mut StdRng,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    for i in 0..length {
        let mut indices = vec![input[i].clone(), output[i].clone()];
        if i > 0 {
            indices.push(links[i - 1].clone());
        }
        if i < length - 1 {
            indices.push(links[i].clone());
        }
        tensors.push(IdxTensor::random::<f64, _>(rng, indices).unwrap());
    }
    TensorTrain::new(tensors).unwrap()
}

fn identity_mpo_with_bond(
    length: usize,
    shared: &[DynIndex],
    output: &[DynIndex],
    bond_dim: usize,
) -> TensorTrain {
    let mut tensors = Vec::with_capacity(length);
    let mut prev: Option<DynIndex> = None;
    for i in 0..length {
        let mut indices: Vec<DynIndex> = Vec::new();
        if let Some(p) = prev.take() {
            indices.push(p);
        }
        indices.push(shared[i].clone());
        indices.push(output[i].clone());
        if i < length - 1 {
            let b = DynIndex::new_dyn(bond_dim);
            indices.push(b.clone());
            prev = Some(b);
        }
        let dim = shared[i].dim();
        let has_l = i > 0;
        let has_r = i < length - 1;
        let mut dims = vec![bond_dim, dim, dim, bond_dim];
        if !has_l {
            dims.remove(0);
        }
        if !has_r {
            dims.pop();
        }
        let total: usize = dims.iter().product();
        let mut data = vec![0.0_f64; total];
        // Only (left=0, output=j, shared=j, right=0) entries are nonzero.
        // Column-major flat indexing: strides grow left-to-right.
        let out_stride = if has_r { bond_dim } else { 1 };
        let sh_stride = out_stride * dim;
        let _ = has_l;
        for j in 0..dim {
            let acc = j * sh_stride + j * out_stride;
            data[acc] = 1.0;
        }
        tensors.push(IdxTensor::from_dense(indices, data).unwrap());
    }
    TensorTrain::new(tensors).unwrap()
}

fn rel_error(a: &TensorTrain, b: &TensorTrain) -> f64 {
    let diff = a.axpby(1.0.into(), b, (-1.0).into()).unwrap();
    diff.norm().unwrap() / b.norm().unwrap()
}

fn timed<F: FnOnce() -> T, T>(f: F) -> (T, std::time::Duration) {
    let start = Instant::now();
    let out = f();
    (out, start.elapsed())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let length: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(12);
    let chi: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(32);
    let phys: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(2);
    let nsweeps: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);
    let tol: f64 = 1e-8;

    let s_input: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("si={}", i + 1)).unwrap())
        .collect();
    let s_shared: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("sc={}", i + 1)).unwrap())
        .collect();
    let s_output: Vec<DynIndex> = (0..length)
        .map(|i| DynIndex::new_dyn_with_tag(phys, &format!("so={}", i + 1)).unwrap())
        .collect();
    let links_a: Vec<DynIndex> = (0..length - 1).map(|_| DynIndex::new_dyn(chi)).collect();

    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let a = random_mpo(length, &s_input, &s_shared, &links_a, &mut rng);
    let b = identity_mpo_with_bond(length, &s_shared, &s_output, chi);

    println!("=== adaptive fit contraction benchmark ===");
    println!("length={length} chi_a=chi_b={chi} phys={phys} nsweeps={nsweeps} tol={tol:e}");
    println!(
        "A max bond = {}, B max bond = {}",
        a.max_bond_dim(),
        b.max_bond_dim()
    );

    // Reference: exact product (zipup).
    let (exact, dt) = timed(|| a.contract(&b, &ContractOptions::zipup()).unwrap());
    println!(
        "zipup exact:            {dt:8.3?}  maxbd={} (chi^2={})",
        exact.max_bond_dim(),
        chi * chi
    );

    // FIT with zip-up initializer + tolerance (previous default path).
    let zipup_fit_opts = ContractOptions::fit()
        .with_initializer(FitInitializer::ZipUp)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(tol))
        .with_nsweeps(nsweeps);
    let (fit_zipup, dt_zipup_init) = timed(|| a.contract(&b, &zipup_fit_opts).unwrap());
    let err_zp = rel_error(&fit_zipup, &exact);
    println!(
        "fit + zipup init:       {dt_zipup_init:8.3?}  maxbd={}  rel_err={err_zp:.3e}",
        fit_zipup.max_bond_dim()
    );

    // FIT with low-rank random init, no max_bond_dim, tolerance-driven growth.
    let lowrank_opts = ContractOptions::fit()
        .with_initializer(FitInitializer::LowRankRandom {
            bond_dim: 1,
            seed: Some(7),
        })
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(tol))
        .with_nsweeps(nsweeps);
    let (fit_lr, dt_lr) = timed(|| a.contract(&b, &lowrank_opts).unwrap());
    let err_lr = rel_error(&fit_lr, &exact);
    println!(
        "fit + low-rank init:    {dt_lr:8.3?}  maxbd={} (grew from 1)  rel_err={err_lr:.3e}",
        fit_lr.max_bond_dim()
    );

    println!(
        "\nlow-rank init:  every bond starts at 1 (no chi^2 = {} intermediate)",
        chi * chi
    );
    println!(
        "zipup init:     builds product-bond intermediates up to chi_a*chi_b = {}",
        chi * chi
    );
    Ok(())
}
