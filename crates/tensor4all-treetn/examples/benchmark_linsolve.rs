//! Benchmark: linsolve (MPO/MPS linear solve) using tensor4all-treetn.
//!
//! This benchmark is based on the proven implementation pattern in
//! `examples/test_linsolve_identity_residual_n3.rs`.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example benchmark_linsolve --release
//!
//! Optional args:
//!   cargo run -p tensor4all-treetn --example benchmark_linsolve --release -- <N> <bond_dim>

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;
use tensor4all_core::{
    index::DynId, DynIndex, IdxTensor, IndexLike, TensorContractionLike, TensorIndex,
};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, random_treetn, ApplyOptions, CanonicalForm,
    CanonicalizationOptions, IndexMapping, LinearOperator, LinkSpace, LinsolveOptions,
    LocalUpdateSweepPlan, SiteIndexNetwork, SquareLinsolveUpdater, TreeTN, TruncationOptions,
};

/// Create an N-site all-ones MPS on a simple chain topology.
/// Returns (mps, site_indices).
fn create_n_site_ones_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<IdxTensor, String>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mps = TreeTN::<IdxTensor, String>::new();

    // Physical indices with tags (for readable debug output)
    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}")).unwrap())
        .collect();

    // Bond indices (n_sites - 1 bonds)
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| DynIndex::new_dyn_with_tag(bond_dim, &format!("bond{i}")).unwrap())
        .collect();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let indices = if i == 0 {
            vec![site_indices[i].clone(), bond_indices[i].clone()]
        } else if i == n_sites - 1 {
            vec![bond_indices[i - 1].clone(), site_indices[i].clone()]
        } else {
            vec![
                bond_indices[i - 1].clone(),
                site_indices[i].clone(),
                bond_indices[i].clone(),
            ]
        };

        let nelem: usize = indices.iter().map(|idx| idx.dim).product();
        let tensor = IdxTensor::from_dense(indices, vec![1.0_f64; nelem]).unwrap();
        mps.add_tensor(name, tensor).unwrap();
    }

    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mps.node_index(&name_i).unwrap();
        let nj = mps.node_index(&name_j).unwrap();
        mps.connect(ni, bond, nj, bond).unwrap();
    }

    (mps, site_indices)
}

/// Create an N-site random MPO (chain) with internal indices.
/// Returns (mpo, s_in_tmp, s_out_tmp).
fn create_random_chain_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
    mpo_bond_dim: usize,
    seed: u64,
) -> (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    let mut net: SiteIndexNetwork<String, DynIndex> = SiteIndexNetwork::new();
    for i in 0..n_sites {
        let name = format!("site{i}");
        net.add_node(
            name,
            [s_in_tmp[i].clone(), s_out_tmp[i].clone()]
                .into_iter()
                .collect::<HashSet<_>>(),
        )
        .unwrap();
    }
    for i in 0..n_sites - 1 {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        net.add_edge(&name_i, &name_j).unwrap();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mpo = random_treetn::<f64, _, _>(&mut rng, &net, LinkSpace::uniform(mpo_bond_dim)).unwrap();
    (mpo, s_in_tmp, s_out_tmp)
}

fn create_n_site_index_mappings(
    state_site_indices: &[DynIndex],
    s_in_tmp: &[DynIndex],
    s_out_tmp: &[DynIndex],
) -> (
    HashMap<String, IndexMapping<DynIndex>>,
    HashMap<String, IndexMapping<DynIndex>>,
) {
    let n_sites = state_site_indices.len();
    assert_eq!(s_in_tmp.len(), n_sites);
    assert_eq!(s_out_tmp.len(), n_sites);

    let mut input_mapping = HashMap::new();
    let mut output_mapping = HashMap::new();

    for i in 0..n_sites {
        let site = format!("site{i}");
        input_mapping.insert(
            site.clone(),
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_in_tmp[i].clone(),
            },
        );
        output_mapping.insert(
            site,
            IndexMapping {
                true_index: state_site_indices[i].clone(),
                internal_index: s_out_tmp[i].clone(),
            },
        );
    }

    (input_mapping, output_mapping)
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

    anyhow::ensure!(n_sites >= 2, "This benchmark currently requires N>=2");

    let phys_dim = 2usize;
    let max_bond_dim = bond_dim;

    let seed = 1234_u64;

    // Match Julia benchmark defaults as much as possible.
    let n_sweeps = 5usize;
    let cutoff = 1e-8_f64;
    let rtol = cutoff.sqrt();

    // Match Julia run: use operator A (a0=0, a1=1) so (a0*I + a1*A) * x = b
    let a0 = 0.0_f64;
    let a1 = 1.0_f64;

    let gmres_tol = 1e-6_f64;
    let gmres_max_restarts = 20usize;
    let gmres_restart_dim = 30usize;

    let n_runs = 10usize;

    println!("=== linsolve Benchmark (Rust/tensor4all-treetn) ===");
    println!("N = {n_sites}");
    println!("phys_dim = {phys_dim}");
    println!("bond_dim = {bond_dim}");
    println!("n_sweeps = {n_sweeps}");
    println!("max_bond_dim = {max_bond_dim}");
    println!("cutoff = {cutoff}");
    println!("rtol = sqrt(cutoff) = {rtol:.6}");
    println!("GMRES: tol={gmres_tol}, maxiter={gmres_max_restarts}, gmres_restart_dim={gmres_restart_dim}");
    println!("coefficients: a0={a0}, a1={a1}");
    println!("seed = {seed} (used for random MPO)");
    println!("n_runs = {n_runs} (excluding warmup)");
    println!();

    // x_true: all-ones MPS. Define b = A*x_true, and initialize x with RHS b.
    let (x_true, site_indices) = create_n_site_ones_mps(n_sites, phys_dim, bond_dim);

    let (mpo, s_in_tmp, s_out_tmp) =
        create_random_chain_mpo_with_internal_indices(n_sites, phys_dim, bond_dim, seed);
    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    let apply_opts = ApplyOptions::zipup()
        .with_max_bond_dim(max_bond_dim)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol));
    let linop_for_rhs =
        LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());
    let rhs = apply_linear_operator(&linop_for_rhs, &x_true, apply_opts)?;

    let center = "site0".to_string();

    // Use an isometric (unitary) canonical form (QR-based) to match standard MPS
    // algorithms more closely.
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

    let compute_rel_residual = |x: &TreeTN<IdxTensor, String>| -> anyhow::Result<f64> {
        let linop = LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());
        let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;

        let ax_full = ax.contract_to_tensor()?;
        let x_full = x.contract_to_tensor()?;
        let b_full = rhs.contract_to_tensor()?;

        // Align tensor indices to b's order before converting to vectors
        let ref_order = b_full.external_indices();
        let order_for = |tensor: &IdxTensor| -> anyhow::Result<Vec<DynIndex>> {
            let inds = tensor.external_indices();
            let by_id: HashMap<DynId, DynIndex> = inds.into_iter().map(|i| (*i.id(), i)).collect();
            let mut out = Vec::with_capacity(ref_order.len());
            for r in &ref_order {
                let id = *r.id();
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

    let init = rhs.clone();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x = init.canonicalize([center.clone()], canon_opts)?;

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        mpo.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
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
    let mut x_last: Option<TreeTN<IdxTensor, String>> = None;

    for run in 1..=n_runs {
        let start = Instant::now();

        let init = rhs.clone();
        let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
        let mut x = init.canonicalize([center.clone()], canon_opts)?;
        let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
            .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

        let mut updater = SquareLinsolveUpdater::with_index_mappings(
            mpo.clone(),
            input_mapping.clone(),
            output_mapping.clone(),
            rhs.clone(),
            options.clone(),
        );

        for _ in 1..=n_sweeps {
            apply_local_update_sweep(&mut x, &plan, &mut updater)?;
        }

        let dur = start.elapsed();
        times.push(dur);
        x_last = Some(x);
        println!("  Run {run}: {:.3} ms", dur.as_secs_f64() * 1000.0);
    }

    println!();
    summarize_times("Results", &times);

    // Compute residual for last run (best-effort)
    if let Some(x) = x_last {
        println!();
        println!("=== Residual (best-effort) ===");
        let r_final = compute_rel_residual(&x)?;
        println!("||r||/||b|| = {:.6e}", r_final);
    }

    Ok(())
}
