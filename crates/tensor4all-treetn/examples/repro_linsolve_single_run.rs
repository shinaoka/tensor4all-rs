//! Repro test: solve a single MPO/MPS linear system once.
//!
//! Purpose
//! - Reproduce the previously observed SVD-related failure at N=10 (when A is identity).
//! - Keep all parameters fixed except `A` and `N`.
//!
//! Solve:
//!   (a0 * I + a1 * A) * x = b
//!
//! Conventions:
//! - Define x_true once
//! - Define b = A * x_true
//! - Initialize x0 = b
//! - Run a fixed number of sweeps once
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example repro_linsolve_single_run --release -- <N> <bond_dim> <A> <rhs> <a0> <a1>
//!
//! Args:
//! - N: number of sites (>=2)
//! - bond_dim: initial bond dimension for the state and random MPO bond dimension
//! - A: "identity" or "random"
//! - rhs: "ones" (b = all-ones MPS) or "ax" (b = A*x_true)
//! - a0, a1: coefficients for (a0*I + a1*A)

use std::collections::{HashMap, HashSet};

use anyhow::Context;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tensor4all_core::{DynIndex, TensorDynLen};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, random_treetn, ApplyOptions, CanonicalForm,
    CanonicalizationOptions, IndexMapping, LinearOperator, LinkSpace, LinsolveOptions,
    LocalUpdateSweepPlan, SiteIndexNetwork, SquareLinsolveUpdater, TreeTN, TruncationOptions,
};

fn create_n_site_ones_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}")).unwrap())
        .collect();

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
        let tensor = TensorDynLen::from_dense(indices, vec![1.0_f64; nelem]).unwrap();
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

fn create_identity_chain_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    let bond_indices: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let mut data = vec![0.0; phys_dim * phys_dim];
        for j in 0..phys_dim {
            data[j * phys_dim + j] = 1.0;
        }

        let tensor = if i == 0 {
            TensorDynLen::from_dense(
                vec![
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else if i == n_sites - 1 {
            TensorDynLen::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            TensorDynLen::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        };
        mpo.add_tensor(name, tensor).unwrap();
    }

    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mpo.node_index(&name_i).unwrap();
        let nj = mpo.node_index(&name_j).unwrap();
        mpo.connect(ni, bond, nj, bond).unwrap();
    }

    (mpo, s_in_tmp, s_out_tmp)
}

fn create_random_chain_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
    mpo_bond_dim: usize,
    seed: u64,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
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

fn compute_rel_residual(
    x: &TreeTN<TensorDynLen, String>,
    linop: &LinearOperator<TensorDynLen, String>,
    rhs: &TreeTN<TensorDynLen, String>,
    a0: f64,
    a1: f64,
) -> anyhow::Result<f64> {
    let ax = apply_linear_operator(linop, x, ApplyOptions::default())?;

    let ax_full = ax.contract_to_tensor()?;
    let x_full = x.contract_to_tensor()?;
    let b_full = rhs.contract_to_tensor()?;

    let ax_vec = ax_full.to_vec::<f64>()?;
    let x_vec = x_full.to_vec::<f64>()?;
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
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let n_sites: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let bond_dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let a_kind: String = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "identity".to_string());
    let rhs_kind: String = args.get(4).cloned().unwrap_or_else(|| "ones".to_string());
    let a0: f64 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let a1: f64 = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0.0);

    anyhow::ensure!(n_sites >= 2, "This example requires N>=2");
    anyhow::ensure!(
        a_kind == "identity" || a_kind == "random",
        "A must be identity|random"
    );
    anyhow::ensure!(
        rhs_kind == "ones" || rhs_kind == "ax",
        "rhs must be ones|ax"
    );

    // Fixed parameters (keep constant except N and A).
    let phys_dim = 2usize;
    let max_bond_dim = bond_dim;

    let seed = 1234_u64;

    let n_sweeps = 5usize;
    let cutoff = 1e-8_f64;
    let rtol = cutoff.sqrt();

    let gmres_tol = 1e-6_f64;
    let gmres_max_restarts = 20usize;
    let gmres_restart_dim = 30usize;

    println!("=== repro_linsolve_single_run ===");
    println!("N = {n_sites}");
    println!("phys_dim = {phys_dim}");
    println!("bond_dim = {bond_dim}");
    println!("A = {a_kind}");
    println!("rhs = {rhs_kind}");
    println!("n_sweeps = {n_sweeps}");
    println!("max_bond_dim = {max_bond_dim}");
    println!("cutoff = {cutoff}");
    println!("rtol = sqrt(cutoff) = {rtol:.6}");
    println!("GMRES: tol={gmres_tol}, maxiter={gmres_max_restarts}, gmres_restart_dim={gmres_restart_dim}");
    println!("coefficients: a0={a0}, a1={a1}"); // This line remains unchanged
    println!();

    let (x_true, site_indices) = create_n_site_ones_mps(n_sites, phys_dim, bond_dim);

    let (mpo, s_in_tmp, s_out_tmp) = if a_kind == "identity" {
        create_identity_chain_mpo_with_internal_indices(n_sites, phys_dim)
    } else {
        create_random_chain_mpo_with_internal_indices(n_sites, phys_dim, bond_dim, seed)
    };

    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    let apply_opts = ApplyOptions::zipup()
        .with_max_bond_dim(max_bond_dim)
        .with_svd_policy(tensor4all_core::SvdTruncationPolicy::new(rtol));
    let linop = LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());

    let rhs = if rhs_kind == "ones" {
        x_true.clone()
    } else {
        apply_linear_operator(&linop, &x_true, apply_opts)
            .with_context(|| "failed to build rhs b = A*x_true")?
    };

    // x0 = b
    let center = "site0".to_string();
    let canon_opts = CanonicalizationOptions::default().with_form(CanonicalForm::Unitary);
    let mut x = rhs
        .clone()
        .canonicalize([center.clone()], canon_opts)
        .with_context(|| "failed to canonicalize initial x0=b")?;

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

    let plan = LocalUpdateSweepPlan::from_treetn(&x, &center, 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        mpo,
        input_mapping,
        output_mapping,
        rhs.clone(),
        options,
    );

    let r0 = compute_rel_residual(&x, &linop, &rhs, a0, a1)?;
    println!("Residual (rel) before: {r0:.6e}");

    for sweep in 1..=n_sweeps {
        apply_local_update_sweep(&mut x, &plan, &mut updater)
            .with_context(|| format!("apply_local_update_sweep failed at sweep {sweep}"))?;
    }

    let r1 = compute_rel_residual(&x, &linop, &rhs, a0, a1)?;
    println!("Residual (rel) after:  {r1:.6e}");

    Ok(())
}
