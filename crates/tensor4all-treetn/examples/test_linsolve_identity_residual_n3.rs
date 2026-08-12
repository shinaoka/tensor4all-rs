//! Test: linsolve identity residual (N=3).
//!
//! Fixed conditions:
//! - N = 3
//! - A = identity (diagonal MPO with all diag values = 1.0) with internal indices + index mappings
//! - a0=0, a1=1 (equation: A x = b)
//!
//! Test cases:
//! 1. init=rhs
//! 2. init=random
//!
//! This example runs 20 sweeps and prints the relative residual before and after:
//!   ||r||_2 / ||b||_2, where r = A x - b.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_identity_residual_n3 --release

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{DynIndex, IdxTensor};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

/// Create an N-site MPS chain with identity-like structure.
/// Returns (mps, site_indices, bond_indices).
fn create_n_site_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>) {
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

    // Create tensors for each site (index order matches tests/linsolve.rs)
    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            // First site: [s0, b01] - identity-like
            let mut data = vec![0.0; phys_dim * bond_dim];
            for j in 0..phys_dim.min(bond_dim) {
                data[j * bond_dim + j] = 1.0;
            }
            IdxTensor::from_dense(vec![site_indices[i].clone(), bond_indices[i].clone()], data)
                .unwrap()
        } else if i == n_sites - 1 {
            // Last site: [b_{n-2,n-1}, s_{n-1}] - identity-like
            let mut data = vec![0.0; bond_dim * phys_dim];
            for j in 0..phys_dim.min(bond_dim) {
                data[j * phys_dim + j] = 1.0;
            }
            IdxTensor::from_dense(
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
                data,
            )
            .unwrap()
        } else {
            // Middle sites: [b_{i-1,i}, s_i, b_{i,i+1}] - identity-like
            let mut data = vec![0.0; bond_dim * phys_dim * bond_dim];
            for j in 0..phys_dim.min(bond_dim) {
                let idx = j * phys_dim * bond_dim + j * bond_dim + j;
                data[idx] = 1.0;
            }
            IdxTensor::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    site_indices[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        };
        mps.add_tensor(name, tensor).unwrap();
    }

    // Connect adjacent sites
    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mps.node_index(&name_i).unwrap();
        let nj = mps.node_index(&name_j).unwrap();
        mps.connect(ni, bond, nj, bond).unwrap();
    }

    (mps, site_indices, bond_indices)
}

fn create_random_mps_with_same_sites(
    n_sites: usize,
    bond_dim: usize,
    site_indices: &[DynIndex],
    seed: u64,
) -> anyhow::Result<TreeTN<IdxTensor, String>> {
    anyhow::ensure!(site_indices.len() == n_sites, "site index count mismatch");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|_| DynIndex::new_dyn(bond_dim))
        .collect();

    let mut mps = TreeTN::<IdxTensor, String>::new();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            IdxTensor::random::<f64, _>(
                &mut rng,
                vec![site_indices[i].clone(), bond_indices[i].clone()],
            )
        } else if i == n_sites - 1 {
            IdxTensor::random::<f64, _>(
                &mut rng,
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
            )
        } else {
            IdxTensor::random::<f64, _>(
                &mut rng,
                vec![
                    bond_indices[i - 1].clone(),
                    site_indices[i].clone(),
                    bond_indices[i].clone(),
                ],
            )
        }?;
        mps.add_tensor(name, tensor).unwrap();
    }

    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mps.node_index(&name_i).unwrap();
        let nj = mps.node_index(&name_j).unwrap();
        mps.connect(ni, bond, nj, bond).unwrap();
    }

    Ok(mps)
}

/// Create an N-site diagonal (identity) MPO with internal indices.
/// Returns (mpo, s_in_tmp, s_out_tmp).
fn create_n_site_mpo_with_internal_indices(
    diag_values: &[f64],
    phys_dim: usize,
) -> (TreeTN<IdxTensor, String>, Vec<DynIndex>, Vec<DynIndex>) {
    let n_sites = diag_values.len();
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mpo = TreeTN::<IdxTensor, String>::new();

    // Internal indices (independent IDs)
    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Bond indices (dim 1 for diagonal operator)
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let mut data = vec![0.0; phys_dim * phys_dim];
        for j in 0..phys_dim {
            data[j * phys_dim + j] = diag_values[i];
        }

        let tensor = if i == 0 {
            IdxTensor::from_dense(
                vec![
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                    bond_indices[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else if i == n_sites - 1 {
            IdxTensor::from_dense(
                vec![
                    bond_indices[i - 1].clone(),
                    s_out_tmp[i].clone(),
                    s_in_tmp[i].clone(),
                ],
                data,
            )
            .unwrap()
        } else {
            IdxTensor::from_dense(
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

    // Connect adjacent sites
    for (i, bond) in bond_indices.iter().enumerate() {
        let name_i = format!("site{i}");
        let name_j = format!("site{}", i + 1);
        let ni = mpo.node_index(&name_i).unwrap();
        let nj = mpo.node_index(&name_j).unwrap();
        mpo.connect(ni, bond, nj, bond).unwrap();
    }

    (mpo, s_in_tmp, s_out_tmp)
}

/// Create N-site index mappings from MPO and state site indices.
/// Returns (input_mapping, output_mapping).
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

fn run_test_case(init_mode: &str, bond_dim: usize) -> anyhow::Result<()> {
    let a0 = 0.0_f64;
    let a1 = 1.0_f64;
    let n_sites = 3usize;
    let phys_dim = 2usize;

    // RHS
    let (rhs, site_indices, _rhs_bond_indices) = create_n_site_mps(n_sites, phys_dim, bond_dim);

    // A = I (diagonal MPO with ones)
    let diag_values: Vec<f64> = vec![1.0; n_sites];
    let (mpo, s_in_tmp, s_out_tmp) =
        create_n_site_mpo_with_internal_indices(&diag_values, phys_dim);
    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    // init selection
    let init = match init_mode {
        "rhs" => rhs.clone(),
        "random" => create_random_mps_with_same_sites(n_sites, bond_dim, &site_indices, 0)?,
        other => anyhow::bail!("unknown init_mode {other:?} (expected rhs|random)"),
    };
    let mut x = init.canonicalize(["site0".to_string()], CanonicalizationOptions::default())?;

    // Setup linsolve options and updater
    let options = LinsolveOptions::default()
        .with_nfullsweeps(20)
        .with_gmres_tol(1e-10)
        .with_max_bond_dim(4)
        .with_coefficients(a0, a1);

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        mpo.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
        rhs.clone(),
        options,
    );

    // Helper: compute relative residual ||(a0*I + a1*A) x - b|| / ||b|| in full space.
    let compute_rel_residual = |x: &TreeTN<IdxTensor, String>| -> anyhow::Result<f64> {
        let linop = LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());
        let ax = apply_linear_operator(&linop, x, ApplyOptions::default())?;

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
    };

    // Print initial residual
    let initial_residual = compute_rel_residual(&x)?;
    println!(
        "a0={a0}, a1={a1}, init={init_mode}: initial ||r||_2 / ||b||_2 = {:.3e}",
        initial_residual
    );

    // Run 20 sweeps
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &"site0".to_string(), 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    for _sweep in 1..=20 {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
    }

    // Print final residual
    let final_residual = compute_rel_residual(&x)?;
    println!(
        "a0={a0}, a1={a1}, init={init_mode}: final ||r||_2 / ||b||_2 = {:.3e}",
        final_residual
    );

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let bond_dim = 1usize;

    println!("=== Test cases (a0=0, a1=1) ===");
    println!();

    // Test case 1: init=rhs
    println!("Test 1: init=rhs");
    run_test_case("rhs", bond_dim)?;
    println!();

    // Test case 2: init=random
    println!("Test 2: init=random");
    run_test_case("random", bond_dim)?;
    println!();

    Ok(())
}
