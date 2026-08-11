//! Test: linsolve with general coefficients (a0≠0, a1≠0) for N=3.
//!
//! Fixed conditions:
//! - N = 3
//! - A = Pauli-X operator (non-diagonal, bit-flip operator) with internal indices + index mappings
//!
//! ## Pauli-X Operator
//!
//! The Pauli-X operator (also known as the bit-flip operator) is a fundamental quantum gate:
//!
//! - **Single-site Pauli-X**: X = [[0, 1], [1, 0]]
//!   - X|0⟩ = |1⟩ (flips 0 to 1)
//!   - X|1⟩ = |0⟩ (flips 1 to 0)
//!   - X is its own inverse: X² = I
//!
//! - **Multi-site Pauli-X**: For N sites, X_0 ⊗ X_1 ⊗ ... ⊗ X_{N-1}
//!   - Flips all bits simultaneously
//!   - For 3 sites: |000⟩ → |111⟩, |001⟩ → |110⟩, |010⟩ → |101⟩, etc.
//!   - The 8×8 matrix representation is an anti-diagonal matrix (all 1s on the anti-diagonal)
//!
//! - **Properties**:
//!   - Non-diagonal (has off-diagonal elements)
//!   - Hermitian: X† = X
//!   - Unitary: X†X = I
//!   - Eigenvalues: +1 and -1
//!
//! Test cases:
//! - Various coefficient combinations (a0, a1) for equation: (a0*I + a1*A) x = b
//!   - a0=1, a1=1: (I + X) x = b
//!   - a0=2, a1=-1: (2I - X) x = b
//!   - a0=1, a1=2: (I + 2X) x = b
//!
//! This example runs 20 sweeps and prints the relative residual before and after:
//!   ||r||_2 / ||b||_2, where r = (a0*I + a1*A) x - b.
//!
//! Run:
//!   cargo run -p tensor4all-treetn --example test_linsolve_general_coefficients_n3 --release

use std::collections::HashMap;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use tensor4all_core::{AnyScalar, DynIndex, TensorDynLen};
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
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");

    let mut mps = TreeTN::<TensorDynLen, String>::new();

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
            TensorDynLen::from_dense(vec![site_indices[i].clone(), bond_indices[i].clone()], data)
                .unwrap()
        } else if i == n_sites - 1 {
            // Last site: [b_{n-2,n-1}, s_{n-1}] - identity-like
            let mut data = vec![0.0; bond_dim * phys_dim];
            for j in 0..phys_dim.min(bond_dim) {
                data[j * phys_dim + j] = 1.0;
            }
            TensorDynLen::from_dense(
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

/// Create an N-site MPS with a more complex structure that uses higher bond dimensions.
/// This creates a vector that is a superposition of multiple basis states,
/// requiring higher bond dimensions to represent.
/// For N=3, this creates a vector like: |000⟩ + |001⟩ + |010⟩ + |011⟩ + |100⟩ + |101⟩ + |110⟩ + |111⟩
/// (approximately, normalized)
/// Returns (mps, site_indices, bond_indices).
fn create_complex_n_site_mps(
    n_sites: usize,
    phys_dim: usize,
    bond_dim: usize,
) -> (TreeTN<TensorDynLen, String>, Vec<DynIndex>, Vec<DynIndex>) {
    assert!(n_sites >= 2, "Need at least 2 sites");
    assert!(bond_dim >= 2, "bond_dim must be at least 2");

    let mut mps = TreeTN::<TensorDynLen, String>::new();

    // Physical indices with tags
    let site_indices: Vec<DynIndex> = (0..n_sites)
        .map(|i| DynIndex::new_dyn_with_tag(phys_dim, &format!("site{i}")).unwrap())
        .collect();

    // Bond indices
    let bond_indices: Vec<DynIndex> = (0..n_sites - 1)
        .map(|i| DynIndex::new_dyn_with_tag(bond_dim, &format!("bond{i}")).unwrap())
        .collect();

    // Create tensors that represent a superposition of multiple basis states
    // We'll create a pattern where each bond index corresponds to a different combination
    // of basis states, allowing for a richer superposition

    for i in 0..n_sites {
        let name = format!("site{i}");
        let tensor = if i == 0 {
            // First site: [s0, b01]
            // Create a tensor that distributes amplitude across multiple bond indices
            let mut data = vec![0.0; phys_dim * bond_dim];
            for s in 0..phys_dim {
                // Distribute amplitude across bond indices
                // Each physical state connects to multiple bond states
                for b in 0..bond_dim {
                    let idx = s * bond_dim + b;
                    // Create a pattern: each physical state connects to different bond states
                    // This allows for superposition of multiple basis states
                    data[idx] = 1.0 / (bond_dim as f64).sqrt();
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
                    // Connect each bond state to each physical state
                    data[idx] = 1.0 / (bond_dim as f64).sqrt();
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
                        // Create a pattern that allows flow through different bond states
                        // This creates entanglement between sites
                        if b_in == b_out {
                            // Allow same bond index to pass through
                            data[idx] = 1.0 / (bond_dim as f64).sqrt();
                        } else if (b_in + 1) % bond_dim == b_out {
                            // Also allow transitions between adjacent bond indices
                            data[idx] = 0.5 / (bond_dim as f64).sqrt();
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
fn scale_treetn(
    treetn: &TreeTN<TensorDynLen, String>,
    scalar: f64,
) -> anyhow::Result<TreeTN<TensorDynLen, String>> {
    let mut scaled = TreeTN::<TensorDynLen, String>::new();
    let node_names: Vec<String> = treetn.node_names().into_iter().collect();

    // Scale all tensors
    for node_name in &node_names {
        let node_idx = treetn.node_index(node_name).unwrap();
        let tensor = treetn.tensor(node_idx).unwrap();
        let scaled_tensor = tensor.scale(AnyScalar::new_real(scalar))?;
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
    a0: f64,
    a1: f64,
    init_mode: &str,
    bond_dim: usize,
    use_complex_rhs: bool,
) -> anyhow::Result<()> {
    let n_sites = 3usize;
    let phys_dim = 2usize;

    // RHS: use complex structure for larger bond dimensions
    let (rhs, site_indices, _rhs_bond_indices) = if use_complex_rhs {
        create_complex_n_site_mps(n_sites, phys_dim, bond_dim)
    } else {
        create_n_site_mps(n_sites, phys_dim, bond_dim)
    };
    print_bond_dims(
        &rhs,
        &format!("b (RHS) bond dimensions (requested bond_dim={bond_dim})"),
    );

    // Print the actual vector representation of b
    let b_full = rhs.contract_to_tensor()?;
    let b_vec = b_full.to_vec::<f64>()?;
    println!("b (RHS) vector: {:?}", b_vec);

    // A = Pauli-X operator (non-diagonal, bit-flip operator)
    // X = [[0, 1], [1, 0]] on each site, combined: X_0 ⊗ X_1 ⊗ X_2
    let (mpo, s_in_tmp, s_out_tmp) =
        create_n_site_pauli_x_mpo_with_internal_indices(n_sites, phys_dim);
    let (input_mapping, output_mapping) =
        create_n_site_index_mappings(&site_indices, &s_in_tmp, &s_out_tmp);

    // init selection
    // For random initialization, use the same bond dimension as RHS
    let init = match init_mode {
        "rhs" => rhs.clone(),
        "rhs/2" => scale_treetn(&rhs, 0.5)?,
        "random" => create_random_mps_with_same_sites(n_sites, &site_indices, bond_dim, 0)?,
        other => anyhow::bail!("unknown init_mode {other:?} (expected rhs|rhs/2|random)"),
    };

    // Print init information
    println!("init mode: {init_mode}");
    print_bond_dims(&init, "init bond dimensions");

    let mut x = init.canonicalize(["site0".to_string()], CanonicalizationOptions::default())?;

    // Setup linsolve options and updater
    let options = LinsolveOptions::default()
        .with_nfullsweeps(10)
        .with_gmres_tol(1e-10)
        .with_max_bond_dim(50)
        .with_coefficients(a0, a1);

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

    // Print initial residual
    let initial_residual = compute_rel_residual(&x)?;
    println!(
        "a0={a0}, a1={a1}, init={init_mode}: initial ||r||_2 / ||b||_2 = {:.3e}",
        initial_residual
    );

    // Run 20 sweeps
    let plan = LocalUpdateSweepPlan::from_treetn(&x, &"site0".to_string(), 2)
        .ok_or_else(|| anyhow::anyhow!("Failed to create 2-site sweep plan"))?;

    for _sweep in 1..=5 {
        apply_local_update_sweep(&mut x, &plan, &mut updater)?;
    }

    // Print final residual and solution bond dimensions
    let final_residual = compute_rel_residual(&x)?;
    print_bond_dims(&x, "solution (x) bond dimensions (after sweeps)");
    println!(
        "a0={a0}, a1={a1}, init={init_mode}: final ||r||_2 / ||b||_2 = {:.3e}",
        final_residual
    );

    Ok(())
}

/// Print the 8x8 matrix representation of the Pauli-X operator X_0 ⊗ X_1 ⊗ X_2
///
/// ## Pauli-X Operator (Bit-Flip Operator)
///
/// The Pauli-X operator is a fundamental quantum gate that flips qubits:
///
/// **Single-site Pauli-X**:
///   - Matrix: X = [[0, 1], [1, 0]]
///   - Action: X|0⟩ = |1⟩, X|1⟩ = |0⟩
///   - Properties: X² = I (self-inverse), X† = X (Hermitian), X†X = I (unitary)
///
/// **3-site Pauli-X** (X_0 ⊗ X_1 ⊗ X_2):
///   - Flips all bits simultaneously
///   - Mapping: |000⟩ → |111⟩, |001⟩ → |110⟩, |010⟩ → |101⟩, |011⟩ → |100⟩,
///     |100⟩ → |011⟩, |101⟩ → |010⟩, |110⟩ → |001⟩, |111⟩ → |000⟩
///   - The 8×8 matrix is an anti-diagonal matrix (all 1s on the anti-diagonal, zeros elsewhere)
///
fn print_pauli_x_matrix() {
    let n_sites = 3usize;
    let dim = 1 << n_sites; // 2^3 = 8

    // Initialize matrix with zeros
    let mut matrix = vec![vec![0.0; dim]; dim];

    // Pauli-X flips all bits: |i⟩ → |~i⟩ where ~i is bitwise NOT
    #[allow(clippy::needless_range_loop)]
    for i in 0..dim {
        let flipped = dim - 1 - i; // Bitwise NOT for n_sites bits
        matrix[flipped][i] = 1.0;
    }

    println!("8x8 matrix representation of X_0 ⊗ X_1 ⊗ X_2:");
    println!("Basis order: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩");
    println!("(Each Pauli-X flips a bit: X|0⟩ = |1⟩, X|1⟩ = |0⟩)");
    println!();
    print!("      ");
    for j in 0..dim {
        print!("|{:03b}⟩  ", j);
    }
    println!();
    for (i, row) in matrix.iter().enumerate().take(dim) {
        print!("|{:03b}⟩ ", i);
        for val in row.iter().take(dim) {
            print!("{:4.0}  ", val);
        }
        println!();
    }
    println!();
}

fn main() -> anyhow::Result<()> {
    let bond_dim = 2usize;

    println!("=== Matrix representation of Pauli-X operator ===");
    println!();
    println!("Pauli-X Operator (Bit-Flip Operator):");
    println!("  - Single-site: X = [[0, 1], [1, 0]]");
    println!("  - Action: X|0⟩ = |1⟩, X|1⟩ = |0⟩");
    println!("  - Properties: X² = I (self-inverse), X† = X (Hermitian), X†X = I (unitary)");
    println!("  - 3-site: X_0 ⊗ X_1 ⊗ X_2 flips all bits simultaneously");
    println!("  - The matrix below shows the 8×8 representation (anti-diagonal matrix)");
    println!();
    print_pauli_x_matrix();
    println!();

    println!("=== Test cases with general coefficients (a0≠0, a1≠0) ===");
    println!("Equation: (a0*I + a1*A) x = b, where A = X (Pauli-X, bit-flip operator)");
    println!("A = X_0 ⊗ X_1 ⊗ X_2, where X = [[0, 1], [1, 0]]");
    println!();

    // Test case 1: a0=1, a1=1 -> (I + X) x = b
    // Note: X^2 = I, so (I + X) is invertible for most cases
    println!("=== Test case 1: a0=1, a1=1 (equation: (I + X) x = b) ===");
    println!();
    println!("Test 1.1: init=rhs");
    run_test_case(1.0, 1.0, "rhs", bond_dim, false)?;
    println!();
    println!("Test 1.2: init=random");
    run_test_case(1.0, 1.0, "random", bond_dim, false)?;
    println!();

    // Test case 2: a0=2, a1=-1 -> (2I - X) x = b
    // This should be well-conditioned
    println!("=== Test case 2: a0=2, a1=-1 (equation: (2I - X) x = b) ===");
    println!();
    println!("Test 2.1: init=rhs");
    run_test_case(2.0, -1.0, "rhs", bond_dim, false)?;
    println!();
    println!("Test 2.2: init=random");
    run_test_case(2.0, -1.0, "random", bond_dim, false)?;
    println!();

    // Test case 3: a0=1, a1=2 -> (I + 2X) x = b
    println!("=== Test case 3: a0=1, a1=2 (equation: (I + 2X) x = b) ===");
    println!();
    println!("Test 3.1: init=rhs");
    run_test_case(1.0, 2.0, "rhs", bond_dim, false)?;
    println!();
    println!("Test 3.2: init=random");
    run_test_case(1.0, 2.0, "random", bond_dim, false)?;
    println!();

    // Test case 4: Larger bond dimension (bond_dim=10) with complex RHS
    let bond_dim_large = 10usize;
    println!("=== Test case 4: a0=1, a1=1 with larger bond_dim={bond_dim_large} ===");
    println!("This tests the solver with a higher bond dimension RHS");
    println!("Using a complex RHS vector (superposition of multiple basis states)");
    println!();
    println!("Test 4.1: init=rhs, bond_dim={bond_dim_large}");
    run_test_case(1.0, 1.0, "rhs", bond_dim_large, true)?;
    println!();

    Ok(())
}
