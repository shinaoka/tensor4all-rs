// End-to-end measurement for issue #566 performance candidates 2 and 3.
//
// Candidate 2: C API cached-evaluator reconstruction per FFI call
// Candidate 3: whole-TT evaluation in global pivot search (floating_zone)
//
// Results recorded in docs/superpowers/ledgers/2026-08-11-issue-566-pr4-ledger.md

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_simplett::{AbstractTensorTrain, SimpleTensorTrain};
use tensor4all_tensorci::globalsearch::floating_zone;
use tensor4all_treetn::{tensor_train_to_treetn, CachedEvaluatorOptions, TreeTNCachedEvaluator};

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

fn main() {
    if let Ok(model) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in model.lines().take(1) {
            println!("hw: {line}");
        }
    }
    println!("hw: logical CPUs: {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
    println!("hw: backend: default CPU/faer (no T4A_TT_BACKEND override)");
    println!("hw: T4A_PROFILE_CONTRACT set: {}", std::env::var("T4A_PROFILE_CONTRACT").is_ok());

    let n_sites = std::env::var("N_SITES").map(|v| v.parse().unwrap()).unwrap_or(16usize);
    let local_dim = std::env::var("LOCAL_DIM").map(|v| v.parse().unwrap()).unwrap_or(2usize);
    let bond_dim = std::env::var("BOND_DIM").map(|v| v.parse().unwrap()).unwrap_or(8usize);

    let tt = random_chain_tt(n_sites, local_dim, bond_dim, 21);
    let (tree, site_indices) = tensor_train_to_treetn(&tt).unwrap();

    // --- Candidate 3: floating_zone re-evaluates the whole TT per candidate ---
    // Count tt.evaluate calls inside floating_zone via the f closure, then
    // measure the raw cost of that many tt.evaluate calls.
    let call_counter = std::cell::Cell::new(0usize);
    let f_counted = |idx: &Vec<usize>| -> f64 {
        call_counter.set(call_counter.get() + 1);
        (0..idx.len()).map(|i| (idx[i] as f64) * 0.1).sum::<f64>()
    };
    let t0 = Instant::now();
    let (pivot, err) = floating_zone(&tt, &f_counted, &vec![local_dim; n_sites], None, 1.0e-12);
    let t_zone = t0.elapsed();
    std::hint::black_box((&pivot, err));
    let n_evals = call_counter.get();
    println!(
        "candidate3 floating_zone({n_sites} sites, dim {local_dim}): {t_zone:?} n_tt_evaluates={n_evals} pivot={pivot:?} err={err:.3e}"
    );

    let mut rng = StdRng::seed_from_u64(99);
    let mut pivot_v = vec![0usize; n_sites];
    let t0 = Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..n_evals {
        let ipos = rng.random_range(0..n_sites);
        pivot_v[ipos] = rng.random_range(0..local_dim);
        acc += tt.evaluate(&pivot_v).unwrap_or(0.0);
    }
    let t_tt_eval = t0.elapsed();
    println!(
        "candidate3 tt.evaluate x {n_evals}: {t_tt_eval:?} — {:.1}% of floating_zone time (acc={acc:.2})",
        100.0 * t_tt_eval.as_secs_f64() / t_zone.as_secs_f64()
    );

    // Batching alternative: a batched floating_zone that evaluates each
    // site's candidates as one TTCache batch, then picks the best. Compare
    // against the original sequential floating_zone on the same target.
    fn floating_zone_batched<F>(
        tt: &SimpleTensorTrain<f64>,
        f: &F,
        local_dims: &[usize],
        early_stop_tol: f64,
    ) -> (Vec<usize>, f64)
    where
        F: Fn(&Vec<usize>) -> f64,
    {
        let n = local_dims.len();
        let mut pivot = vec![0usize; n];
        let mut cache = tensor4all_simplett::TTCache::new(tt);
        let mut max_error = f64::MAX;
        let max_sweeps = n * 10;
        for _ in 0..max_sweeps {
            let mut any_improved = false;
            for ipos in 0..n {
                let mut best_local_error = 0.0f64;
                let mut best_idx = pivot[ipos];
                let batch: Vec<Vec<usize>> = (0..local_dims[ipos])
                    .map(|v| {
                        let mut p = pivot.clone();
                        p[ipos] = v;
                        p
                    })
                    .collect();
                let vals = cache.evaluate_many(&batch, None).unwrap();
                for (v, tt_val) in vals.iter().enumerate() {
                    let mut p = pivot.clone();
                    p[ipos] = v;
                    let diff = f(&p) - *tt_val;
                    let error = diff.abs();
                    if error > best_local_error {
                        best_local_error = error;
                        best_idx = v;
                    }
                }
                if best_idx != pivot[ipos] {
                    pivot[ipos] = best_idx;
                    any_improved = true;
                }
                max_error = max_error.min(best_local_error);
            }
            if !any_improved || max_error > early_stop_tol {
                break;
            }
        }
        (pivot, max_error)
    }

    let f2 = |idx: &Vec<usize>| -> f64 {
        (0..idx.len()).map(|i| (idx[i] as f64) * 0.1).sum::<f64>()
    };
    let t0 = Instant::now();
    let (pivot_b, err_b) = floating_zone_batched(&tt, &f2, &vec![local_dim; n_sites], 1.0e-12);
    let t_batched = t0.elapsed();
    std::hint::black_box((pivot_b, err_b));
    println!(
        "candidate3 batched floating_zone (TTCache, {local_dim}-wide batches): {t_batched:?} — vs sequential {t_zone:?} ({:.2}x)",
        t_zone.as_secs_f64() / t_batched.as_secs_f64()
    );

    // --- Candidate 3b / 4b: TTCache batch evaluation vs per-point evaluate ---
    // TTCache reuses left/right environments; per-point evaluate reallocates.
    let n_points = 1000usize;
    let mut rng4 = StdRng::seed_from_u64(1234);
    let batch: Vec<Vec<usize>> = (0..n_points)
        .map(|_| (0..n_sites).map(|_| rng4.random_range(0..local_dim)).collect())
        .collect();
    let t0 = Instant::now();
    for _point in &batch {
        tt.evaluate(_point).unwrap_or(0.0);
    }
    let t_per_point = t0.elapsed();

    let mut cache = tensor4all_simplett::TTCache::new(&tt);
    let t0 = Instant::now();
    let results = cache.evaluate_many(&batch, None).unwrap();
    let t_cache = t0.elapsed();
    let _sum: f64 = results.iter().sum();
    println!(
        "candidate4 per-point evaluate x {n_points}: {t_per_point:?} | TTCache evaluate_many: {t_cache:?} — cache is {:.1}x faster",
        t_per_point.as_secs_f64() / t_cache.as_secs_f64()
    );

    // --- Candidate 2: TreeTNCachedEvaluator reconstruction per call ---
    // Simulate the capi handle pattern: rebuild the cached evaluator on every
    // FFI call, vs constructing it once and reusing it.
    let n_points = 32usize;
    let points: Vec<usize> = (0..n_sites * n_points)
        .map(|i| (i / n_sites + i % n_sites) % local_dim)
        .collect();
    let shape = [n_sites, n_points];
    let values = ColMajorArrayRef::new(&points, &shape).unwrap();
    let n_calls = 2000usize;

    let t0 = Instant::now();
    for _ in 0..n_calls {
        let mut cached = TreeTNCachedEvaluator::new(
            &tree,
            &site_indices,
            CachedEvaluatorOptions::<usize>::default(),
        )
        .unwrap();
        cached.evaluate_batched(values).unwrap();
    }
    let t_rebuild = t0.elapsed();

    // C API persists the center after the first call and passes it into
    // subsequent reconstructions; measure that realistic rebuild cost.
    let mut warm = TreeTNCachedEvaluator::new(
        &tree,
        &site_indices,
        CachedEvaluatorOptions::<usize>::default(),
    )
    .unwrap();
    warm.evaluate_batched(values).unwrap();
    let center = *warm.center().unwrap();
    let t0 = Instant::now();
    for _ in 0..n_calls {
        let mut cached = TreeTNCachedEvaluator::new(
            &tree,
            &site_indices,
            CachedEvaluatorOptions { center: Some(center), ..Default::default() },
        )
        .unwrap();
        cached.evaluate_batched(values).unwrap();
    }
    let t_rebuild_with_center = t0.elapsed();

    let mut cached = TreeTNCachedEvaluator::new(
        &tree,
        &site_indices,
        CachedEvaluatorOptions::<usize>::default(),
    )
    .unwrap();
    let t0 = Instant::now();
    for _ in 0..n_calls {
        cached.evaluate_batched(values).unwrap();
    }
    let t_reused = t0.elapsed();
    println!(
        "candidate2 cached-evaluator rebuild per call: {t_rebuild:?} | with persisted center: {t_rebuild_with_center:?} | reused: {t_reused:?} | rebuild share (default): {:.1}%, (center persisted): {:.1}%",
        100.0 * (t_rebuild.as_secs_f64() - t_reused.as_secs_f64()) / t_rebuild.as_secs_f64(),
        100.0 * (t_rebuild_with_center.as_secs_f64() - t_reused.as_secs_f64()) / t_rebuild_with_center.as_secs_f64()
    );
}
