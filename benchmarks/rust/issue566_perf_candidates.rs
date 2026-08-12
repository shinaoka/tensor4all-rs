// End-to-end measurement harness for issue #566 performance candidates.
//
// Measures the share of each candidate in a representative end-to-end
// workload. See
// `docs/superpowers/ledgers/2026-08-11-issue-566-pr4-ledger.md` for results.
//
// Candidates:
//   1. `TreeTN::evaluate` rebuilds the evaluator per call
//   2. (C API cached evaluator — measured in the capi crate)
//   3. (global pivot search — measured in the tensorci crate)
//   4. SimpleTT per-site allocation during evaluation
//   5. `contract_profile_enabled()` re-reads the env var per call

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_treetn::tensor_train_to_treetn;

/// Build a random chain TT: n_sites sites, local dim `local_dim`, bond `bond_dim`.
fn random_chain_tt(n_sites: usize, local_dim: usize, bond_dim: usize, seed: u64) -> SimpleTensorTrain<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tensors = Vec::with_capacity(n_sites);
    for site in 0..n_sites {
        let left = if site == 0 { 1 } else { bond_dim };
        let right = if site + 1 == n_sites { 1 } else { bond_dim };
        let data: Vec<f64> = (0..left * local_dim * right)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        tensors.push(tensor4all_simplett::tensor3_from_data(data, left, local_dim, right).unwrap());
    }
    SimpleTensorTrain::new(tensors).unwrap()
}

/// TCI-style sampling points: n_points multi-indices over n_sites sites.
fn random_points(n_sites: usize, local_dim: usize, n_points: usize, seed: u64) -> Vec<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_points)
        .map(|_| (0..n_sites).map(|_| rng.random_range(0..local_dim)).collect())
        .collect()
}

fn main() {
    // Hardware/backend configuration (recorded for reproducibility).
    if let Ok(model) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in model.lines().take(1) {
            println!("hw: {line}");
        }
    }
    println!("hw: logical CPUs: {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
    println!("hw: backend: default CPU/faer (no T4A_TT_BACKEND override)");
    println!("hw: T4A_PROFILE_CONTRACT set: {}", std::env::var("T4A_PROFILE_CONTRACT").is_ok());

    let n_sites = 32usize;
    let local_dim = 2usize;
    let bond_dim = 16usize;
    let n_points = 200usize;
    let n_reps = 20usize;

    let tt = random_chain_tt(n_sites, local_dim, bond_dim, 7);
    let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();
    let points = random_points(n_sites, local_dim, n_points, 11);
    let values = points
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect::<Vec<usize>>();
    let shape = [n_sites, n_points];
    let values_ref = ColMajorArrayRef::new(&values, &shape).unwrap();

    // --- Candidate 1: TreeTN::evaluate rebuilds the evaluator per call ---
    let t0 = Instant::now();
    for _ in 0..n_reps {
        std::hint::black_box(tree.evaluate(&site_indices, values_ref).unwrap());
    }
    let t_evaluate = t0.elapsed();

    let evaluator = tensor4all_treetn::TreeTNEvaluator::new(&tree, &site_indices).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_reps {
        std::hint::black_box(evaluator.evaluate_batched(values_ref).unwrap());
    }
    let t_evaluator_reused = t0.elapsed();

    println!(
        "candidate1 TreeTN::evaluate (rebuild per call): {t_evaluate:?} | reused evaluator: {t_evaluator_reused:?} | rebuild share: {:.1}%",
        100.0 * (t_evaluate.as_secs_f64() - t_evaluator_reused.as_secs_f64()) / t_evaluate.as_secs_f64()
    );

    // --- Candidate 1b: per-point calls (TCI sampling style) ---
    // Each single-point call to TreeTN::evaluate rebuilds the evaluator.
    let single_shape = [n_sites, 1];
    let n_single_reps = 4000usize;
    let mut pts = random_points(n_sites, local_dim, n_single_reps, 13);
    pts.truncate(n_single_reps);
    let single_values = pts
        .iter()
        .flat_map(|p| p.iter().copied())
        .collect::<Vec<usize>>();
    let t0 = Instant::now();
    for i in 0..n_single_reps {
        let values_ref = ColMajorArrayRef::new(&single_values[i * n_sites..(i + 1) * n_sites], &single_shape).unwrap();
        std::hint::black_box(tree.evaluate(&site_indices, values_ref).unwrap());
    }
    let t_per_point_evaluate = t0.elapsed();

    let t0 = Instant::now();
    for i in 0..n_single_reps {
        let values_ref = ColMajorArrayRef::new(&single_values[i * n_sites..(i + 1) * n_sites], &single_shape).unwrap();
        std::hint::black_box(evaluator.evaluate_batched(values_ref).unwrap());
    }
    let t_per_point_reused = t0.elapsed();
    println!(
        "candidate1b per-point TreeTN::evaluate: {t_per_point_evaluate:?} | reused evaluator: {t_per_point_reused:?} | rebuild share: {:.1}%",
        100.0 * (t_per_point_evaluate.as_secs_f64() - t_per_point_reused.as_secs_f64()) / t_per_point_evaluate.as_secs_f64()
    );

    // --- Candidate 4: SimpleTT per-site evaluation ---
    let t0 = Instant::now();
    let mut sink = 0.0;
    for rep in 0..n_reps {
        for point in &points {
            let v = tt.evaluate(point).unwrap();
            sink += v;
        }
        let _ = rep;
    }
    let t_simplett_eval = t0.elapsed();
    println!("candidate4 SimpleTT evaluate ({} x {} points): {t_simplett_eval:?} (sink={sink:.2})", n_reps, n_points);

    // --- Candidate 5: contract_profile_enabled env lookup cost ---
    let t0 = Instant::now();
    for _ in 0..1_000_000 {
        std::hint::black_box(std::env::var("T4A_PROFILE_CONTRACT").is_ok());
    }
    let t_env = t0.elapsed();
    println!(
        "candidate5 env::var(T4A_PROFILE_CONTRACT) x 1_000_000: {t_env:?} ({:.1} ns/call)",
        t_env.as_secs_f64() * 1e9 / 1_000_000.0
    );

    // contract() cost for comparison: contract two 4x4 matrices a few times
        let i = tensor4all_core::DynIndex::new_dyn(4);
    let j = tensor4all_core::DynIndex::new_dyn(4);
    let k = tensor4all_core::DynIndex::new_dyn(4);
    let a = tensor4all_core::IdxTensor::from_dense(
        vec![i.clone(), j.clone()],
        (0..16).map(|x| x as f64).collect(),
    )
    .unwrap();
    let b = tensor4all_core::IdxTensor::from_dense(
        vec![j, k],
        (0..16).map(|x| (x as f64) * 0.5).collect(),
    )
    .unwrap();
    let t0 = Instant::now();
    for _ in 0..100_000 {
        std::hint::black_box(tensor4all_core::contract(&[&a, &b]).unwrap());
    }
    let t_contract = t0.elapsed();
    // contract() calls contract_profile_enabled() twice per call; each env::var
    // costs ~77 ns measured above.
    let per_contract_env_ns = 2.0 * t_env.as_secs_f64() * 1e9 / 1_000_000.0;
    let per_contract_ns = t_contract.as_secs_f64() * 1e9 / 100_000.0;
    println!(
        "candidate5 contract(4x4, 4x4): {per_contract_ns:.0} ns/call — 2x env::var is {per_contract_env_ns:.0} ns/call = {:.2}% of a contract call",
        100.0 * per_contract_env_ns / per_contract_ns
    );
}
