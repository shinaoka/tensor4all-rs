//! Benchmark: linsolve with MPO for x (unknown) and b (RHS).
//!
//! Solve for MPO x:
//!   (a0*I + a1*A) * x = b
//! where A(x) = A0 * x (MPO×MPO contraction, operator composition).
//!
//! This corresponds to the Julia benchmark `benchmarks/julia/benchmark_linsolve_mpo.jl`.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example benchmark_linsolve_mpo --release
//!
//! Optional args:
//!   cargo run -p tensor4all-treetn --example benchmark_linsolve_mpo --release -- <N> <bond_dim>

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tensor4all_core::{
    index::{DynId, Index},
    DynIndex, IndexLike, TensorContractionLike, TensorDynLen, TensorIndex,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalForm,
    CanonicalizationOptions, IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan,
    SquareLinsolveUpdater, TreeTN, TruncationOptions,
};

fn make_node_name(i: usize) -> String {
    format!("site{i}")
}

/// Create a DynIndex with a deterministic ID from the seeded RNG.
fn unique_dyn_index(used: &mut HashSet<DynId>, dim: usize, rng: &mut StdRng) -> DynIndex {
    loop {
        let id = DynId(rng.random());
        if used.insert(id) {
            return Index::new(id, dim);
        }
    }
}

fn mpo_node_indices(
    n: usize,
    i: usize,
    bonds: &[DynIndex],
    s_out: &[DynIndex],
    s_in: &[DynIndex],
) -> Vec<DynIndex> {
    if n == 1 {
        vec![s_out[i].clone(), s_in[i].clone()]
    } else if i == 0 {
        vec![s_out[i].clone(), s_in[i].clone(), bonds[i].clone()]
    } else if i + 1 == n {
        vec![bonds[i - 1].clone(), s_out[i].clone(), s_in[i].clone()]
    } else {
        vec![
            bonds[i - 1].clone(),
            s_out[i].clone(),
            s_in[i].clone(),
            bonds[i].clone(),
        ]
    }
}

/// Create an N-site identity-like MPO for x_true.
///
/// Structure matches `test_linsolve_mpo_identity.rs`:
/// - x has indices [external, s_out_tmp, s_in_tmp, bonds]
/// - external = true_site_indices (the contracted index with operator)
/// - s_out_tmp, s_in_tmp = x's own internal MPO indices (remain after contraction)
/// - The tensor data has identity structure: δ(s_out_tmp, s_in_tmp) for each external value
///
/// When operator A acts on x:
/// - x's external contracts with A's s_in (via input_mapping)
/// - Result has A's s_out (mapped back to external) + x's s_out_tmp, s_in_tmp
fn create_identity_mpo_state(
    n: usize,
    phys_dim: usize,
    bond_dim: usize,
    true_site_indices: &[DynIndex], // external indices (contracted with operator)
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // MPO bonds
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim, rng))
        .collect();

    // x's own internal MPO indices (NOT the operator's)
    let s_out_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_in_tmp: Vec<_> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);

        // Base tensor: [external, s_out_tmp, s_in_tmp] with identity structure
        // I[ext, s_out, s_in] = δ(s_out, s_in) for each ext value
        let mut base_data = vec![0.0_f64; phys_dim * phys_dim * phys_dim];
        for ext_val in 0..phys_dim {
            for k in 0..phys_dim {
                // Identity on (s_out, s_in): δ(s_out, s_in)
                // Index order: [external, s_out, s_in]
                let idx = ext_val * phys_dim * phys_dim + k * phys_dim + k;
                base_data[idx] = 1.0;
            }
        }
        let base = TensorDynLen::from_dense(
            vec![
                true_site_indices[i].clone(), // external (contracted with operator)
                s_out_tmp[i].clone(),
                s_in_tmp[i].clone(),
            ],
            base_data,
        )
        .unwrap();

        // Add bond indices via outer product if needed
        let t = if n == 1 {
            base
        } else {
            let mut bond_inds = Vec::new();
            if i > 0 {
                bond_inds.push(bonds[i - 1].clone());
            }
            if i < n - 1 {
                bond_inds.push(bonds[i].clone());
            }
            if bond_inds.is_empty() {
                base
            } else {
                let bond_size: usize = bond_inds.iter().map(|b| b.dim()).product();
                let ones = TensorDynLen::from_dense(bond_inds, vec![1.0_f64; bond_size]).unwrap();
                TensorDynLen::outer_product(&base, &ones)?
            }
        };

        let node = mpo.add_tensor(node_name.clone(), t).unwrap();
        nodes.push(node);
    }

    for i in 0..n.saturating_sub(1) {
        mpo.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok(mpo)
}

/// Create an N-site random MPO for the **operator** A only.
/// Tensors have [s_out, s_in] (+ bonds) with no separate "external" index.
/// All index IDs are generated deterministically from the seeded RNG.
#[allow(clippy::type_complexity)]
fn create_random_mpo_operator(
    n: usize,
    phys_dim: usize,
    bond_dim: usize,
    true_site_indices: &[DynIndex],
    used_ids: &mut HashSet<DynId>,
    rng: &mut StdRng,
) -> anyhow::Result<(
    TreeTN<TensorDynLen, String>,
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
)> {
    anyhow::ensure!(true_site_indices.len() == n, "site index count mismatch");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // MPO bonds (deterministic IDs from rng)
    let bonds: Vec<_> = (0..n.saturating_sub(1))
        .map(|_| unique_dyn_index(used_ids, bond_dim, rng))
        .collect();

    // Internal indices (deterministic IDs from rng)
    let s_in_tmp: Vec<DynIndex> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();
    let s_out_tmp: Vec<DynIndex> = (0..n)
        .map(|_| unique_dyn_index(used_ids, phys_dim, rng))
        .collect();

    let mut input_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();
    let mut output_mapping: HashMap<String, IndexMapping<DynIndex>> = HashMap::new();

    let mut nodes = Vec::with_capacity(n);
    for i in 0..n {
        let node_name = make_node_name(i);

        let indices = mpo_node_indices(n, i, &bonds, &s_out_tmp, &s_in_tmp);
        let t = TensorDynLen::random::<f64, _>(rng, indices)?;

        let node = mpo.add_tensor(node_name.clone(), t).unwrap();
        nodes.push(node);

        input_mapping.insert(
            node_name.clone(),
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            node_name,
            IndexMapping {
                true_index: true_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    for i in 0..n.saturating_sub(1) {
        mpo.connect(nodes[i], &bonds[i], nodes[i + 1], &bonds[i])
            .unwrap();
    }

    Ok((mpo, input_mapping, output_mapping))
}

fn summarize_times(label: &str, times: &[Duration]) {
    let secs: Vec<f64> = times.iter().map(|d| d.as_secs_f64()).collect();
    let mean = secs.iter().sum::<f64>() / secs.len() as f64;
    let min = secs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = secs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var = secs
        .iter()
        .map(|t| {
            let d = t - mean;
            d * d
        })
        .sum::<f64>()
        / secs.len() as f64;
    let std = var.sqrt();

    println!("=== {label} ===");
    println!("Average: {:.3} ms", mean * 1000.0);
    println!("Min:     {:.3} ms", min * 1000.0);
    println!("Max:     {:.3} ms", max * 1000.0);
    println!("Stddev:  {:.3} ms", std * 1000.0);
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let n_sites: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let bond_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    let phys_dim = 2usize;
    let max_bond_dim = bond_dim;

    let seed = 1234_u64;

    // Match Julia benchmark defaults
    let n_sweeps = 5usize;
    let cutoff = 1e-8_f64;
    let rtol = cutoff.sqrt();

    // Coefficients: (a0*I + a1*A) * x = b
    let a0 = 0.0_f64;
    let a1 = 1.0_f64;

    let gmres_tol = 1e-6_f64;
    let gmres_max_restarts = 50usize;
    let gmres_restart_dim = 30usize;

    let n_runs = 10usize;

    println!("=== linsolve Benchmark (MPO unknown; Rust/tensor4all-treetn) ===");
    println!("Problem: (a0*I + a1*A) * x = b, with x::MPO");
    println!("A(x) = A0 * x (MPO×MPO contraction)");
    println!("N = {n_sites}");
    println!("phys_dim = {phys_dim}");
    println!("bond_dim = {bond_dim}");
    println!("max_bond_dim = {max_bond_dim}");
    println!("n_sweeps = {n_sweeps}");
    println!("cutoff = {cutoff}");
    println!("rtol = sqrt(cutoff) = {rtol:.6}");
    println!("GMRES: tol={gmres_tol}, maxiter={gmres_max_restarts}, gmres_restart_dim={gmres_restart_dim}");
    println!("coefficients: a0={a0}, a1={a1}");
    println!("seed = {seed} (used for random operator MPO A0)");
    println!("n_runs = {n_runs} (excluding warmup)");
    println!();

    let mut used_ids = HashSet::<DynId>::new();
    let mut rng_a = StdRng::seed_from_u64(seed);
    let mut rng_x = StdRng::seed_from_u64(seed + 1);

    // True site indices (shared between x_true and b)
    // Use rng_a for deterministic ID generation
    let true_site_indices: Vec<_> = (0..n_sites)
        .map(|_| unique_dyn_index(&mut used_ids, phys_dim, &mut rng_a))
        .collect();

    // Random operator A0
    let (operator_a, a_input_mapping, a_output_mapping) = create_random_mpo_operator(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices,
        &mut used_ids,
        &mut rng_a,
    )?;

    // x_true (the true solution): identity-like MPO
    // Structure: [external, s_out_tmp, s_in_tmp, bonds]
    // Same as test_linsolve_mpo_identity.rs
    let x_true = create_identity_mpo_state(
        n_sites,
        phys_dim,
        bond_dim,
        &true_site_indices,
        &mut used_ids,
        &mut rng_x,
    )?;

    // Compute b = A * x_true
    let apply_opts = ApplyOptions::zipup()
        .with_max_bond_dim(max_bond_dim)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol));
    let linop_for_rhs = LinearOperator::new(
        operator_a.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
    );
    let rhs = apply_linear_operator(&linop_for_rhs, &x_true, apply_opts)?;

    let center = make_node_name(n_sites / 2);

    let truncation = TruncationOptions::default()
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol))
        .with_max_bond_dim(max_bond_dim);

    let options = LinsolveOptions::default()
        .with_nfullsweeps(n_sweeps)
        .with_truncation(truncation)
        .with_gmres_tol(gmres_tol)
        .with_gmres_max_restarts(gmres_max_restarts)
        .with_gmres_restart_dim(gmres_restart_dim)
        .with_coefficients(a0, a1);

    let compute_rel_residual = |x: &TreeTN<TensorDynLen, String>| -> anyhow::Result<f64> {
        let linop = LinearOperator::new(
            operator_a.clone(),
            a_input_mapping.clone(),
            a_output_mapping.clone(),
        );
        let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;

        let ax_full = ax.contract_to_tensor()?;
        let x_full = x.contract_to_tensor()?;
        let b_full = rhs.contract_to_tensor()?;

        // Align indices using b_full as reference (like test_linsolve_mpo_identity.rs)
        let ref_order: Vec<DynIndex> = b_full.external_indices();

        let order_for = |tensor: &TensorDynLen| -> anyhow::Result<Vec<DynIndex>> {
            let inds: Vec<DynIndex> = tensor.external_indices();
            let by_id: HashMap<DynId, DynIndex> =
                inds.into_iter().map(|i: DynIndex| (*i.id(), i)).collect();
            let mut out = Vec::with_capacity(ref_order.len());
            for r in ref_order.iter() {
                let id: DynId = *r.id();
                let idx = by_id
                    .get(&id)
                    .ok_or_else(|| anyhow::anyhow!("residual: index {:?} not found in tensor", id))?
                    .clone();
                out.push(idx);
            }
            Ok(out)
        };

        let order_x = order_for(&x_full)?;
        let order_ax = order_for(&ax_full)?;
        let x_aligned = x_full.permuteinds(&order_x)?;
        let ax_aligned = ax_full.permuteinds(&order_ax)?;

        let ax_vec = ax_aligned.to_vec::<f64>()?;
        let x_vec = x_aligned.to_vec::<f64>()?;
        let b_vec = b_full.to_vec::<f64>()?;

        anyhow::ensure!(ax_vec.len() == b_vec.len(), "vector length mismatch");
        anyhow::ensure!(x_vec.len() == b_vec.len(), "vector length mismatch");

        let mut r2 = 0.0_f64;
        let mut b2 = 0.0_f64;
        for ((ax_i, x_i), b_i) in ax_vec.iter().zip(x_vec.iter()).zip(b_vec.iter()) {
            let opx_i = a0 * x_i + a1 * ax_i;
            let r_i = opx_i - b_i;
            r2 += r_i * r_i;
            b2 += b_i * b_i;
        }
        Ok(if b2 > 0.0 {
            (r2 / b2).sqrt()
        } else {
            r2.sqrt()
        })
    };

    println!("Warmup run (excluded from stats)...");
    let warmup_start = Instant::now();

    // Initial guess x0 = rhs (same as Julia benchmark)
    let init = rhs.clone();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x = init.canonicalize([center.clone()], canon_opts)?;

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        operator_a.clone(),
        a_input_mapping.clone(),
        a_output_mapping.clone(),
        rhs.clone(),
        options.clone(),
    );

    let r0 = compute_rel_residual(&x)?;
    for _ in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
    }
    let r1 = compute_rel_residual(&x)?;

    println!(
        "Warmup: {:.3} ms",
        warmup_start.elapsed().as_secs_f64() * 1000.0
    );
    println!("Residual (rel): {:.3e} -> {:.3e}", r0, r1);
    println!();

    println!("Measured runs...");
    let mut times = Vec::with_capacity(n_runs);
    let mut x_last: Option<TreeTN<TensorDynLen, String>> = None;

    for run in 1..=n_runs {
        let start = Instant::now();

        let init = rhs.clone();
        let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        let mut x = init.canonicalize([center.clone()], canon_opts)?;

        let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

        let mut updater = SquareLinsolveUpdater::with_index_mappings(
            operator_a.clone(),
            a_input_mapping.clone(),
            a_output_mapping.clone(),
            rhs.clone(),
            options.clone(),
        );

        for _ in 1..=n_sweeps {
            apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        }

        let dur = start.elapsed();
        times.push(dur);

        // Compute residual for each run to verify consistency
        let r = compute_rel_residual(&x)?;
        x_last = Some(x);
        println!(
            "  Run {run}: {:.3} ms, residual: {:.6e}",
            dur.as_secs_f64() * 1000.0,
            r
        );
    }

    println!();
    summarize_times("Results", &times);

    // Compute residual for last run
    if let Some(x) = x_last {
        println!();
        println!("=== Residual (best-effort) ===");
        let r_final = compute_rel_residual(&x)?;
        println!("||r||/||b|| = {:.6e}", r_final);
    }

    Ok(())
}
