//! Benchmark for main branch (tenferro backend).
//! Run: cargo test --release -p tensor4all-itensorlike --test bench_basic_ops -- --nocapture

use std::time::Instant;
use tensor4all_core::{DynIndex, IndexLike, TensorDynLen, TensorIndex};
use tensor4all_itensorlike::{ContractOptions, TensorTrain, TruncateOptions};

fn make_random_mps(n_sites: usize, d: usize, bond_dim: usize, seed: u64) -> TensorTrain {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut tensors = Vec::with_capacity(n_sites);
    let mut prev_bond: Option<DynIndex> = None;
    for i in 0..n_sites {
        let s = DynIndex::new_dyn_with_tag(d, &format!("s={}", i + 1)).unwrap();
        let br = if i < n_sites - 1 { bond_dim } else { 1 };
        let mut indices = Vec::new();
        if let Some(ref b) = prev_bond {
            indices.push(b.clone());
        }
        indices.push(s);
        let next_bond = if i < n_sites - 1 {
            let b = DynIndex::new_dyn(br);
            indices.push(b.clone());
            Some(b)
        } else {
            None
        };
        let size: usize = indices.iter().map(|idx| idx.dim()).product();
        let data: Vec<f64> = (0..size).map(|_| rng.random::<f64>() - 0.5).collect();
        tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
        prev_bond = next_bond;
    }
    TensorTrain::new(tensors).unwrap()
}

fn make_random_mpo_pair(n_sites: usize, d: usize, bond_dim: usize) -> (TensorTrain, TensorTrain) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let shared: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(d, &format!("mid={}", i + 1)).unwrap())
        .collect();
    let make_mpo =
        |rng: &mut rand::rngs::StdRng, tag: &str, shared: &[DynIndex], shared_first: bool| {
            let mut tensors = Vec::with_capacity(n_sites);
            let mut prev_bond: Option<DynIndex> = None;
            for (i, shared_idx) in shared.iter().enumerate().take(n_sites) {
                let phys = DynIndex::new_dyn_with_tag(d, &format!("{}={}", tag, i + 1)).unwrap();
                let br = if i < n_sites - 1 { bond_dim } else { 1 };
                let mut indices = Vec::new();
                if let Some(ref b) = prev_bond {
                    indices.push(b.clone());
                }
                if shared_first {
                    indices.push(shared_idx.clone());
                    indices.push(phys);
                } else {
                    indices.push(phys);
                    indices.push(shared_idx.clone());
                }
                let next_bond = if i < n_sites - 1 {
                    let b = DynIndex::new_dyn(br);
                    indices.push(b.clone());
                    Some(b)
                } else {
                    None
                };
                let size: usize = indices.iter().map(|idx| idx.dim()).product();
                let data: Vec<f64> = (0..size).map(|_| rng.random::<f64>() - 0.5).collect();
                tensors.push(TensorDynLen::from_dense(indices, data).unwrap());
                prev_bond = next_bond;
            }
            TensorTrain::new(tensors).unwrap()
        };
    let a = make_mpo(&mut rng, "r", &shared, false);
    let b = make_mpo(&mut rng, "c", &shared, true);
    (a, b)
}

fn time_it<F: FnOnce() -> R, R>(label: &str, f: F) -> R {
    let start = Instant::now();
    let result = f();
    eprintln!("  {label}: {:.3}s", start.elapsed().as_secs_f64());
    result
}

#[test]
#[ignore = "benchmark-only timing test"]
fn bench_norm() {
    eprintln!("\n=== norm benchmark (main) ===");
    for &n in &[20, 45, 90] {
        let mps = make_random_mps(n, 2, 16, 123);
        time_it(&format!("norm({n} sites, bd=16)"), || {
            let _ = mps.norm().unwrap();
        });
    }
}

#[test]
#[ignore = "benchmark-only timing test"]
fn bench_inner() {
    eprintln!("\n=== inner benchmark (main) ===");
    for &n in &[20, 45, 90] {
        let a = make_random_mps(n, 2, 16, 123);
        let b = make_random_mps(n, 2, 16, 456);
        let a_sites: Vec<DynIndex> = a.site_indices().into_iter().flatten().collect();
        let b_sites: Vec<DynIndex> = b.site_indices().into_iter().flatten().collect();
        let b = b.replace_indices(&b_sites, &a_sites).unwrap();
        time_it(&format!("inner({n} sites, bd=16)"), || {
            let _ = a.inner(&b).unwrap();
        });
    }
}

#[test]
#[ignore = "benchmark-only timing test"]
fn bench_truncate() {
    eprintln!("\n=== truncate benchmark (main) ===");
    for &n in &[20, 45, 90] {
        let mut mps = make_random_mps(n, 2, 16, 123);
        let opts = TruncateOptions::svd()
            .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
            .with_max_rank(8);
        time_it(&format!("truncate({n} sites, bd=16→8)"), || {
            mps.truncate(&opts).unwrap();
        });
    }
}

#[test]
#[ignore = "benchmark-only timing test"]
fn bench_contract_zipup() {
    eprintln!("\n=== contract zipup benchmark (main) ===");
    for &n in &[10, 20, 45] {
        let (a, b) = make_random_mpo_pair(n, 2, 8);
        let opts = ContractOptions::zipup()
            .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
            .with_max_rank(16);
        time_it(&format!("zipup({n} sites, bd=8)"), || {
            let _ = a.contract(&b, &opts).unwrap();
        });
    }
}

#[test]
#[ignore = "benchmark-only timing test"]
fn bench_contract_fit() {
    eprintln!("\n=== contract fit benchmark (main) ===");
    for &n in &[10, 20, 45] {
        let (a, b) = make_random_mpo_pair(n, 2, 8);
        let opts = ContractOptions::fit()
            .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(1e-6))
            .with_max_rank(16);
        time_it(&format!("fit({n} sites, bd=8)"), || {
            let _ = a.contract(&b, &opts).unwrap();
        });
    }
}
