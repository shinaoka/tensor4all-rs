//! Test: linsolve with a0=1, a1=0 (identity operator issue).
//!
//! This test focuses on investigating issues with a0 ≠ 0 cases, specifically a0=1, a1=0.
//! This case corresponds to solving I x = b, where I is the identity operator.
//!
//! Test setup:
//! - N = 6, 7
//! - A = Pauli-X operator (non-diagonal, bit-flip operator) with internal indices + index mappings
//! - Exact solution: x_exact = |000...0⟩ + |111...1⟩ (simple two-state superposition, bond_dim=2)
//! - RHS: b = (a0*I + a1*A) * x_exact = 2 * x_exact (since a0=2, a1=0)
//! - Initial guess: random (to test convergence from arbitrary starting point)
//! - Test: Solve (a0*I + a1*A) x = b and verify convergence to x_exact
//!
//! Test case:
//! - a0=1, a1=0: I x = b (I is non-singular, should work but has convergence issues at N=7)
//!
//! This test is isolated from other test cases to investigate the specific issue
//! where a0 ≠ 0 cases fail or have poor convergence, particularly at N=7.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_identity_operator_issue --release

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
use tensor4all_treetn::{
    apply_linear_operator, apply_local_update_sweep, ApplyOptions, CanonicalizationOptions,
    IndexMapping, LinearOperator, LinsolveOptions, LocalUpdateSweepPlan, SquareLinsolveUpdater,
    TreeTN,
};

/// Create an N-site MPS representing the all-ones vector (uniform superposition).
/// This creates |000...0⟩ + |000...1⟩ + ... + |111...1⟩ (all basis states with equal amplitude).
/// For N sites with phys_dim=2, bond_dim=1 is sufficient (each site contributes equally).
/// Returns (mps, site_indices, bond_indices).
fn create_all_ones_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");
    assert!(bond_dim >= 1, "bond_dim must be at least 1");

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    // Physical indices with tags
    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}")).unwrap())
        .collect();

    // Bond indices
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| DynIndex::new_dyn_with_tag(bond_dim, &format!("bond{i}")).unwrap())
        .collect();

    // Create tensors that represent uniform superposition (all-ones vector)
    // For bond_dim=1, each site contributes equally to all physical states
    // This creates |000⟩ + |001⟩ + ... + |111⟩ with equal amplitude

    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            // First site: [s0, b01]
            // Each physical state connects to the single bond index with equal amplitude
            let mut data = vec![0.0; phys_dim * bond_dim];
            for s in 0..phys_dim {
                for b in 0..bond_dim {
                    let idx = s * bond_dim + b;
                    // Equal amplitude for all physical states
                    data[idx] = 1.0 / (phys_dim as f64).sqrt();
                }
            }
            TensorDynLen::from_dense(vec![site_indices[i].clone(), bond_indices[i].clone()], data)
                .unwrap()
        } else if i == n_sites - 1 {
            // Last site: [b_{n-2,n-1}, s_{n-1}]
            let mut data = vec![0.0; bond_dim * phys_dim];
            for b in 0..bond_dim {
                for s in 0..phys_dim {
                    let idx = b * phys_dim + s;
                    // Equal amplitude for all physical states
                    data[idx] = 1.0 / (phys_dim as f64).sqrt();
                }
            }
            TensorDynLen::from_dense(
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
                data,
            )
            .unwrap()
        } else {
            // Middle sites: [b_{i-1,i}, s_i, b_{i,i+1}]
            let mut data = vec![0.0; bond_dim * phys_dim * bond_dim];
            for b_in in 0..bond_dim {
                for s in 0..phys_dim {
                    for b_out in 0..bond_dim {
                        let idx = b_in * phys_dim * bond_dim + s * bond_dim + b_out;
                        // Pass through bond index and distribute equally across physical states
                        if b_in == b_out {
                            data[idx] = 1.0 / (phys_dim as f64).sqrt();
                        }
                    }
                }
            }
            TensorDynLen::from_dense(
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

/// Create an N-site MPS representing |000...0⟩ + |111...1⟩ (first and last basis states only).
/// This is a simple superposition of two product states.
/// Requires bond_dim >= 2.
/// Returns (mps, site_indices, bond_indices).
fn create_simple_two_state_mps(
    n_sites: usize,
    phys_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");
    assert!(phys_dim >= 2, "phys_dim must be at least 2");

    // This state requires bond_dim=2: one bond index for |000...0⟩, one for |111...1⟩
    let bond_dim = 2usize;

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    // Physical indices with tags
    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}")).unwrap())
        .collect();

    // Bond indices
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| DynIndex::new_dyn_with_tag(bond_dim, &format!("bond{i}")).unwrap())
        .collect();

    // Create tensors for |000...0⟩ + |111...1⟩
    // Bond index 0 corresponds to |000...0⟩, bond index 1 corresponds to |111...1⟩
    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            // First site: [s0, b01]
            // s0=0 connects to bond 0 (|000...0⟩), s0=1 connects to bond 1 (|111...1⟩)
            let mut data = vec![0.0; phys_dim * bond_dim];
            data[0] = 1.0 / 2.0_f64.sqrt(); // |0⟩ -> bond 0
            data[bond_dim + 1] = 1.0 / 2.0_f64.sqrt(); // |1⟩ -> bond 1
            TensorDynLen::from_dense(vec![site_indices[i].clone(), bond_indices[i].clone()], data)
                .unwrap()
        } else if i == n_sites - 1 {
            // Last site: [b_{n-2,n-1}, s_{n-1}]
            // bond 0 connects to s=0 (|000...0⟩), bond 1 connects to s=1 (|111...1⟩)
            let mut data = vec![0.0; bond_dim * phys_dim];
            data[0] = 1.0 / 2.0_f64.sqrt(); // bond 0 -> |0⟩
            data[phys_dim + 1] = 1.0 / 2.0_f64.sqrt(); // bond 1 -> |1⟩
            TensorDynLen::from_dense(
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
                data,
            )
            .unwrap()
        } else {
            // Middle sites: [b_{i-1,i}, s_i, b_{i,i+1}]
            // Pass through bond index: bond 0 -> s=0 -> bond 0, bond 1 -> s=1 -> bond 1
            let mut data = vec![0.0; bond_dim * phys_dim * bond_dim];
            data[0] = 1.0 / 2.0_f64.sqrt(); // bond 0 -> |0⟩ -> bond 0
            data[phys_dim * bond_dim + bond_dim + 1] = 1.0 / 2.0_f64.sqrt(); // bond 1 -> |1⟩ -> bond 1
            TensorDynLen::from_dense(
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
    site_indices: &[DynIndex],
    init_bond_dim: usize,
    seed: u64,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    anyhow::ensure!(site_indices.len() == n_sites, "site index count mismatch");

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| DynIndex::new_dyn_with_tag(init_bond_dim, &format!("init_bond{i}")).unwrap())
        .collect();

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            TensorDynLen::random::<f64, _>(
                &mut rng,
                vec![site_indices[i].clone(), bond_indices[i].clone()],
            )
        } else if i == n_sites - 1 {
            TensorDynLen::random::<f64, _>(
                &mut rng,
                vec![bond_indices[i - 1].clone(), site_indices[i].clone()],
            )
        } else {
            TensorDynLen::random::<f64, _>(
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

/// Scale a TreeTN by a scalar factor.
/// For MPS, we only scale one tensor (the last one) to avoid scaling by scalar^n_sites.
fn scale_treetn(
    treetn: &TreeTN<TensorDynLen, String>,
    scalar: f64,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    let mut scaled = TreeTN::<TensorDynLen, String>::new();
    let node_names: Vec<String> = treetn.node_names().into_iter().collect();

    // For MPS, scale only the last tensor to scale the entire vector by scalar
    // (scaling all tensors would scale by scalar^n_sites)
    let last_idx = node_names.len() - 1;
    for (i, node_name) in node_names.iter().enumerate() {
        let node_idx = treetn.node_index(node_name).unwrap();
        let tensor = treetn.tensor(node_idx).unwrap();
        let scaled_tensor = if i == last_idx {
            // Scale only the last tensor
            tensor.scale(AnyScalar::new_real(scalar))?
        } else {
            // Copy other tensors without scaling
            tensor.clone()
        };
        scaled.add_tensor(node_name.clone(), scaled_tensor)?;
    }

    // Copy connections (each edge only once)
    for (node_a, node_b) in treetn.site_index_network().edges() {
        let edge = treetn.edge_between(&node_a, &node_b).unwrap();
        let bond = treetn.bond_index(edge).unwrap();
        let node_a_idx = scaled.node_index(&node_a).unwrap();
        let node_b_idx = scaled.node_index(&node_b).unwrap();
        scaled.connect(node_a_idx, bond, node_b_idx, bond)?;
    }

    Ok(scaled)
}

/// Create an N-site Pauli-X MPO (bit-flip operator) with internal indices.
///
/// ## Pauli-X Operator
///
/// The Pauli-X operator is a fundamental quantum gate that flips qubits:
///
/// - **Matrix representation**: X = [[0, 1], [1, 0]]
/// - **Action**: X|0⟩ = |1⟩, X|1⟩ = |0⟩
/// - **Properties**: X² = I (self-inverse), X† = X (Hermitian), X†X = I (unitary)
///
/// For N sites, this creates the tensor product: X_0 ⊗ X_1 ⊗ ... ⊗ X_{N-1}
/// which flips all bits simultaneously.
///
/// Returns (mpo, s_in_tmp, s_out_tmp).
fn create_n_site_pauli_x_mpo_with_internal_indices(
    n_sites: usize,
    phys_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");
    assert_eq!(phys_dim, 2, "Pauli-X requires phys_dim=2");

    let mut mpo = TreeTN::<TensorDynLen, String>::new();

    // Internal indices (independent IDs)
    let s_in_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();
    let s_out_tmp: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(phys_dim)).collect();

    // Bond indices (dim 1 for Pauli-X operator)
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(1)).collect();

    // Pauli X matrix: [[0, 1], [1, 0]]
    // As a tensor [out, in]: X[0,0]=0, X[0,1]=1, X[1,0]=1, X[1,1]=0
    let pauli_x = [0.0, 1.0, 1.0, 0.0];

    for i in 0..n_sites {
        let name = format!("site{i}");
        let mut data = vec![0.0; phys_dim * phys_dim];
        for out_idx in 0..phys_dim {
            for in_idx in 0..phys_dim {
                data[out_idx * phys_dim + in_idx] = pauli_x[out_idx * phys_dim + in_idx];
            }
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

/// Print bond dimensions of a TreeTN MPS.
fn print_bond_dims(mps: &TreeTN<TensorDynLen, String>, label: &str) {
    let edges: Vec<_> = mps.site_index_network().edges().collect();
    if edges.is_empty() {
        println!("{label}: no bonds");
        return;
    }
    let mut dims = Vec::new();
    for (node_a, node_b) in edges {
        if let Some(edge) = mps.edge_between(&node_a, &node_b) {
            if let Some(bond) = mps.bond_index(edge) {
                dims.push(bond.dim);
            }
        }
    }
    println!("{label}: bond_dims = {:?}", dims);
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

fn run_test_case(
    n_sites: usize,
    a0: f64,
    a1: f64,
    use_simple_exact: bool,
    max_bond_dim: usize,
    init_mode: &str, // "random" or "rhs"
) -> anyhow::Result<(f64, f64, f64, f64)> {
    let phys_dim = 2usize;

    // Always verbose for detailed debugging
    let verbose = true;

    if verbose {
        println!("=== Setting up test with known exact solution ===");
    }
    let (x_exact, site_indices, _) = if use_simple_exact {
        if verbose {
            println!(
                "Exact solution: x_exact = |000...0⟩ + |111...1⟩ (simple two-state superposition)"
            );
        }
        create_simple_two_state_mps(n_sites, phys_dim)
    } else {
        if verbose {
            println!("Exact solution: x_exact = all-ones vector (uniform superposition)");
        }
        create_all_ones_mps(n_sites, phys_dim, 1)
    };
    if verbose {
        println!("RHS: b = (a0*I + a1*A) * x_exact");
        println!();
        println!("Exact solution (x_exact) created");
        print_bond_dims(&x_exact, "x_exact bond dimensions");
    }

    // Create Pauli-X operator
    let (mpo, s_in_tmp, s_out_tmp) =
        create_n_site_pauli_x_mpo_with_internal_indices(n_sites, phys_dim);
    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    // Compute RHS: b = (a0*I + a1*A) * x_exact
    // This ensures that x_exact is the exact solution of (a0*I + a1*A) x = b
    if verbose {
        println!("Computing RHS: b = (a0*I + a1*A) * x_exact...");
    }

    // For |000...0⟩ + |111...1⟩, X * x_exact = x_exact (since it's an eigenvector with eigenvalue 1)
    // Therefore: (a0*I + a1*X) * x_exact = a0 * x_exact + a1 * X * x_exact = a0 * x_exact + a1 * x_exact = (a0 + a1) * x_exact
    // So b = (a0 + a1) * x_exact
    // For a0=1, a1=0: b = x_exact
    let rhs = scale_treetn(&x_exact, a0 + a1)?;

    // Verify that b = (a0*I + a1*X) * x_exact (only in verbose mode)
    if verbose && a1 != 0.0 {
        println!("Verifying b computation...");
        let linop = LinearOperator::new(mpo.clone(), input_mapping.clone(), output_mapping.clone());
        let x_exact_x = apply_linear_operator(&linop, &x_exact, ApplyOptions::default())?;

        let x_exact_full = x_exact.contract_to_tensor()?;
        let x_exact_x_full = x_exact_x.contract_to_tensor()?;
        let x_exact_vec = x_exact_full.to_vec::<f64>()?;
        let x_exact_x_vec = x_exact_x_full.to_vec::<f64>()?;
        anyhow::ensure!(
            x_exact_vec.len() == x_exact_x_vec.len(),
            "vector length mismatch"
        );

        let mut diff2 = 0.0_f64;
        let mut norm2 = 0.0_f64;
        for (x_i, x_x_i) in x_exact_vec.iter().zip(x_exact_x_vec.iter()) {
            let diff = x_i - x_x_i;
            diff2 += diff * diff;
            norm2 += x_i * x_i;
        }
        let rel_diff = if norm2 > 0.0 {
            (diff2 / norm2).sqrt()
        } else {
            diff2.sqrt()
        };
        println!(
            "  ||X * x_exact - x_exact|| / ||x_exact|| = {:.3e}",
            rel_diff
        );
        if rel_diff > 1e-10 {
            println!("  WARNING: X * x_exact != x_exact (unexpected!)");
        } else {
            println!("  OK: X * x_exact = x_exact (as expected)");
        }
    }

    // Get the actual bond dimensions of b to match init
    let rhs_bond_dims: Vec<usize> = {
        let edges: Vec<_> = rhs.site_index_network().edges().collect();
        let mut dims = Vec::new();
        for (node_a, node_b) in edges {
            if let Some(edge) = rhs.edge_between(&node_a, &node_b) {
                if let Some(bond) = rhs.bond_index(edge) {
                    dims.push(bond.dim);
                }
            }
        }
        dims
    };

    // Use the bond dimension from b for initial guess to avoid dimension mismatch
    // Note: RHS bond dimension is fixed, but x's bond dimension can grow during linsolve (up to max_bond_dim)
    let init_bond_dim = if !rhs_bond_dims.is_empty() {
        rhs_bond_dims[0] // Use first bond dimension (they should all be the same)
    } else {
        1 // Fallback to minimal bond dimension
    };

    if verbose {
        print_bond_dims(
            &rhs,
            "b (RHS) bond dimensions (computed from (a0*I + a1*A) * x_exact)",
        );
        println!(
            "b (RHS) vector dimension: 2^{} = {} (not printed)",
            n_sites,
            1 << n_sites
        );
        println!("Using init bond_dim={init_bond_dim} to match b (RHS) bond dimensions");
        println!(
            "Note: x's bond dimension can grow during linsolve (up to max_bond_dim={max_bond_dim})"
        );
        println!();
        println!("Creating initial guess...");
        println!("  Using init_mode={init_mode}");
    }

    let init = match init_mode {
        "rhs" => {
            if verbose {
                println!("  Initializing with RHS (b)");
            }
            rhs.clone()
        }
        "perturbed" => {
            // Use random MPS with bond_dim=4 as initial guess
            if verbose {
                println!("  Using random initial guess (bond_dim=4)");
            }
            create_random_mps_with_same_sites(n_sites, &site_indices, 4, 42)?
        }
        _ => anyhow::bail!("Unknown init_mode: {init_mode}"),
    };
    if verbose {
        print_bond_dims(&init, &format!("init ({init_mode}) bond dimensions"));
    }

    let mut x = init.canonicalize(["site0".to_string()], CanonicalizationOptions::default())?;

    // Setup linsolve options and updater
    // max_bond_dim is passed as parameter to allow bond dimension growth during sweeps
    // Adjust GMRES parameters for better convergence
    let options = LinsolveOptions::default()
        .with_nfullsweeps(10)
        .with_gmres_tol(1e-8) // Slightly relaxed from 1e-10
        .with_gmres_max_restarts(200) // Increased from default 100
        .with_gmres_restart_dim(50) // Increased from default 30
        .with_max_bond_dim(max_bond_dim)
        .with_coefficients(a0, a1)
        .with_convergence_tol(1e-6); // Early termination if residual < 1e-6

    if verbose {
        println!("Linsolve options: max_bond_dim={max_bond_dim}, nfullsweeps=10, gmres_tol=1e-8, gmres_max_restarts=200, gmres_restart_dim=50, convergence_tol=1e-6");
    }

    let mut updater = SquareLinsolveUpdater::with_index_mappings(
        mpo.clone(),
        input_mapping.clone(),
        output_mapping.clone(),
        rhs.clone(),
        options,
    );

    // Helper: compute relative residual ||(a0*I + a1*A) x - b|| / ||b|| in full space.
    let compute_rel_residual = |x: &TreeTN<TensorDynLen, String>| -> anyhow::Result<f64> {
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

    // Helper: compute error relative to exact solution ||x - x_exact|| / ||x_exact||
    let compute_exact_error = |x: &TreeTN<TensorDynLen, String>| -> anyhow::Result<f64> {
        let x_full = x.contract_to_tensor()?;
        let x_exact_full = x_exact.contract_to_tensor()?;
        let x_vec = x_full.to_vec::<f64>()?;
        let x_exact_vec = x_exact_full.to_vec::<f64>()?;
        anyhow::ensure!(x_vec.len() == x_exact_vec.len(), "vector length mismatch");

        let mut diff2 = 0.0_f64;
        let mut exact2 = 0.0_f64;
        for (x_i, x_exact_i) in x_vec.iter().zip(x_exact_vec.iter()) {
            let diff = x_i - x_exact_i;
            diff2 += diff * diff;
            exact2 += x_exact_i * x_exact_i;
        }
        Ok(if exact2 > 0.0 {
            (diff2 / exact2).sqrt()
        } else {
            diff2.sqrt()
        })
    };

    // Print initial residual and error
    let initial_residual = compute_rel_residual(&x)?;
    let initial_error = compute_exact_error(&x)?;
    if verbose {
        println!(
            "a0={a0}, a1={a1}, init={init_mode}: initial ||r||_2 / ||b||_2 = {:.3e}",
            initial_residual
        );
        println!(
            "a0={a0}, a1={a1}, init={init_mode}: initial ||x - x_exact||_2 / ||x_exact||_2 = {:.3e}",
            initial_error
        );
    }

    // Run sweeps with detailed output
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &"site0".to_string(), 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    if verbose {
        println!("Starting sweeps...");
    }
    let mut last_residual = initial_residual;
    let mut last_error = initial_error;

    // Adjust number of sweeps based on N (kept minimal for fast tests)
    let n_sweeps = if n_sites >= 10 {
        20
    } else if n_sites >= 7 {
        10
    } else {
        5
    };
    if verbose {
        println!("Running {n_sweeps} sweeps (N={n_sites})");
    }

    for sweep in 1..=n_sweeps {
        if verbose && (sweep % 10 == 0 || sweep <= 5) {
            println!("  Sweep {sweep}/{n_sweeps}...");
        }
        let sweep_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            apply_local_update_sweep(&mut x, &plan, &mut updater)
        }));

        match sweep_result {
            Ok(Ok(())) => {
                // Compute residual and error to track convergence (only print in verbose mode)
                match compute_rel_residual(&x) {
                    Ok(residual) => {
                        match compute_exact_error(&x) {
                            Ok(error) => {
                                if verbose {
                                    print_bond_dims(
                                        &x,
                                        &format!("  x bond dimensions (after sweep {sweep})"),
                                    );
                                    println!(
                                        "    Residual: {:.3e}, Error: {:.3e}",
                                        residual, error
                                    );
                                }
                                // Check if we're making progress
                                if residual < last_residual || error < last_error {
                                    last_residual = residual;
                                    last_error = error;
                                } else if verbose {
                                    println!("    Warning: Residual/error not decreasing");
                                }
                            }
                            Err(e) => {
                                if verbose {
                                    println!("    Warning: Could not compute exact error: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        if verbose {
                            println!("    Warning: Could not compute residual: {}", e);
                        }
                    }
                }
            }
            Ok(Err(e)) => {
                println!("  Error during sweep {sweep}: {}", e);
                println!("  Error type: {:?}", e);

                // Check if it's a GMRES singular matrix error
                let error_str = format!("{}", e);
                if error_str.contains("Near-singular") || error_str.contains("GMRES") {
                    println!("  This is a GMRES numerical issue. Trying to continue with current solution...");
                    // Try to compute current residual/error before returning
                    if let Ok(residual) = compute_rel_residual(&x) {
                        if let Ok(error) = compute_exact_error(&x) {
                            println!("  Current residual: {:.3e}, error: {:.3e}", residual, error);
                            // If we're close enough, consider it a success
                            if residual < 1e-6 && error < 1e-6 {
                                println!(
                                    "  Solution is close enough despite GMRES error. Continuing..."
                                );
                                continue;
                            }
                        }
                    }
                }

                println!("  Current x bond dimensions:");
                print_bond_dims(&x, "  x bond dimensions (before error)");
                return Err(anyhow::Error::new(e));
            }
            Err(_) => {
                println!("  Panic during sweep {sweep} (likely dimension mismatch)");
                println!("  Current x bond dimensions:");
                print_bond_dims(&x, "  x bond dimensions (before panic)");
                anyhow::bail!("Panic during sweep {sweep}");
            }
        }
    }

    // Print final residual, error, and solution bond dimensions
    let final_residual = compute_rel_residual(&x)?;
    let final_error = compute_exact_error(&x)?;
    if verbose {
        print_bond_dims(&x, "solution (x) bond dimensions (after all sweeps)");
        println!(
            "a0={a0}, a1={a1}, init={init_mode}: final ||r||_2 / ||b||_2 = {:.3e}",
            final_residual
        );
        println!(
            "a0={a0}, a1={a1}, init={init_mode}: final ||x - x_exact||_2 / ||x_exact||_2 = {:.3e}",
            final_error
        );
    }

    Ok((initial_residual, initial_error, final_residual, final_error))
}

fn main() -> anyhow::Result<()> {
    // max_bond_dim for linsolve (allows bond dimension to grow during sweeps)
    let max_bond_dim = 30usize;

    // Test N=6 and N=7 to compare convergence behavior
    let test_n_values = vec![6, 7];

    println!("=== Test: Investigating a0=1, a1=0 (identity operator) issue ===");
    println!("Testing N=6, 7 with init=random and init=rhs");
    println!("Equation: (a0*I + a1*A) x = b, where A = X (Pauli-X, bit-flip operator)");
    println!("  - For a0=1, a1=0: I x = b (identity operator)");
    println!("  - Exact solution: x_exact = |000...0⟩ + |111...1⟩ (bond_dim=2)");
    println!("  - RHS: b = x_exact");
    println!("  - Initial guess: random or rhs (bond_dim matching RHS)");
    println!("  - max_bond_dim for linsolve: {max_bond_dim}");
    println!();
    println!("Metrics explanation:");
    println!("  - Residual: ||(a0*I + a1*A) x - b|| / ||b|| (how well the equation is satisfied)");
    println!("  - Error: ||x - x_exact|| / ||x_exact|| (distance from the exact solution)");
    println!();

    // Test case: a0=1, a1=0 only
    let test_cases = vec![(1.0, 0.0, "a0=1, a1=0 (I x = b)")];

    // Test both init modes
    let init_modes = vec!["perturbed", "rhs"];

    for &n_sites in &test_n_values {
        println!("========================================");
        println!("=== Testing N={n_sites} ===");
        println!(
            "Full vector space dimension: 2^{n_sites} = {}",
            1 << n_sites
        );
        println!();

        for (a0, a1, case_desc) in &test_cases {
            for &init_mode in &init_modes {
                println!("--- Test case: {case_desc}, init={init_mode} ---");

                let result = run_test_case(n_sites, *a0, *a1, true, max_bond_dim, init_mode);

                match result {
                    Ok((initial_residual, initial_error, final_residual, final_error)) => {
                        println!("  N={n_sites}, {case_desc}, init={init_mode}: SUCCESS");
                        println!(
                            "    Initial residual: {:.3e} (||(a0*I + a1*A) x_init - b|| / ||b||)",
                            initial_residual
                        );
                        println!(
                            "    Initial error: {:.3e} (||x_init - x_exact|| / ||x_exact||)",
                            initial_error
                        );
                        println!(
                            "    Final residual: {:.3e} (||(a0*I + a1*A) x_final - b|| / ||b||)",
                            final_residual
                        );
                        println!(
                            "    Final error: {:.3e} (||x_final - x_exact|| / ||x_exact||)",
                            final_error
                        );

                        // Consider error > 0.1 as failure
                        if final_error > 0.1 {
                            println!("    WARNING: Error > 0.1, convergence may be poor");
                        }
                    }
                    Err(e) => {
                        println!("  N={n_sites}, {case_desc}, init={init_mode}: FAILED");
                        println!("    Error: {}", e);
                    }
                }

                println!();
            }
        }
    }

    println!("========================================");
    println!("Testing complete!");

    Ok(())
}
